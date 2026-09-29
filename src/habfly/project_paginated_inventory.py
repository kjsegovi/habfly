"""Offline consistency checks for a separately versioned recorded page chain.

This is NOT a pager, a live inventory producer, or an accepted journal/scoring
receipt. Captures cannot prove that a control was exposed, that navigation really
happened, or that the same live attempt produced every file. Even valid output
therefore keeps live_pagination_verified and collection_count_verified false.

Plan v1 pins page.json receipts for a forward pass, a complete reverse revisit,
and a fresh first-page assessment anchor. Each page receipt pins both existing
probe captures AND their manifests. No existing single-page contract is relaxed.

Plan fields: schema_version=1, mode=PLAN_MODE, forward/revisit lists of
{receipt: history-relative page.json path, sha256: file hash}, assessment_anchor
with the same reference shape, and expected_stars. Each page.json has only
schema_version=1, mode=PAGE_MODE, section='stellar', viewing={start,end,total},
rows=[{name,row_accessibility_sha256,source_sha256}], and source_sha256 containing
exactly the before/after observation.json and manifest.json file hashes.
Timestamp order is checked, not taken as proof of actual browser transitions.
"""

import hashlib
import json
import re
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json, rows_hash
from .browser_numeric import screen_identity
from .browser_project_inventory import NAME, ROW, TOTAL, VIEWING
from .browser_project_navigation import project_view
from .browser_stellar import SIMULATION_URL, _atoms

PLAN_MODE = "recorded_collected_stellar_page_chain"
PAGE_MODE = "recorded_collected_stellar_page"
MODE = "offline_paginated_inventory_consistency"
CAPTURE_FILES = {
    f"{section}/{name}.json" for section in ("before", "after") for name in ("observation", "manifest")
}


class PaginatedInventoryError(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise PaginatedInventoryError("paginated_inventory_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def _checksum(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


class _Sources:
    def __init__(self, history):
        self.root = Path(history).resolve(strict=True)
        _require(self.root.is_dir(), "owned_history_required")
        self.hashes = {}
        self.bytes = 0

    def path(self, value):
        _require(isinstance(value, (str, Path)), "invalid_path")
        candidate = Path(value)
        _require(".." not in candidate.parts, "path_traversal")
        candidate = candidate if candidate.is_absolute() else self.root / candidate
        _require(candidate.resolve().is_relative_to(self.root), "source_outside_history")
        _require(
            not any(p.is_symlink() for p in (candidate, *candidate.parents) if p != self.root),
            "symlink_source",
        )
        return candidate.resolve()

    def read(self, path, expected=None):
        path = self.path(path)
        _require(path.is_file() and path.stat().st_size <= 2_000_000, "missing_or_oversized_source")
        with path.open("rb") as stream:
            raw = stream.read(2_000_001)
        _require(len(raw) <= 2_000_000, "missing_or_oversized_source")
        name, digest = str(path.relative_to(self.root)), _sha(raw)
        if expected is not None:
            _require(_checksum(expected) and digest == expected, "source_hash_mismatch")
        _require(name not in self.hashes or self.hashes[name] == digest, "source_changed")
        if name not in self.hashes:
            self.bytes += len(raw)
        _require(self.bytes <= 40_000_000, "source_budget_exceeded")
        self.hashes[name] = digest
        return raw

    def json(self, path, expected=None):
        try:
            return json.loads(
                self.read(path, expected),
                object_pairs_hook=_object,
                parse_constant=lambda _: (_ for _ in ()).throw(
                    PaginatedInventoryError("paginated_inventory_nonfinite_json")
                ),
            )
        except (UnicodeError, json.JSONDecodeError):
            raise PaginatedInventoryError("paginated_inventory_invalid_json") from None

    def unchanged(self):
        for path, expected in list(self.hashes.items()):
            self.read(path, expected)


def _relative(value):
    _require(
        isinstance(value, str)
        and value
        and not Path(value).is_absolute()
        and ".." not in Path(value).parts
        and str(Path(value)) == value,
        "invalid_relative_source",
    )
    return value


def _counts(text):
    views, totals = VIEWING.findall(text), TOTAL.findall(text)
    _require(len(views) == len(totals) == 1, "ambiguous_page_counts")
    start, end, total = map(int, views[0])
    _require(1 <= start <= end <= total <= 30 and int(totals[0]) == total, "invalid_page_range")
    return {"start": start, "end": end, "total": total}


def _timestamp(value):
    try:
        _require(isinstance(value, str), "capture_timestamp_required")
        result = datetime.fromisoformat(value)
        _require(result.tzinfo is not None, "capture_timezone_required")
        return result.timestamp()
    except (ValueError, OverflowError):
        raise PaginatedInventoryError("paginated_inventory_invalid_capture_timestamp") from None


def _page_capture(report):
    _require(
        isinstance(report, dict)
        and type(report.get("schema_version")) is int
        and report["schema_version"] == 1
        and report.get("mode") == "read_only_browser_preflight"
        and report.get("ignored_frame_urls") == []
        and report.get("allow_submission") is False
        and type(report.get("actions_executed")) is int
        and report["actions_executed"] == 0,
        "unsupported_capture",
    )
    _require(
        project_view(report) == {"surface": "list", "section": "stellar", "star": None},
        "stellar_list_required",
    )
    simulation = next(f for f in report["frames"] if f["url"] == SIMULATION_URL)
    texts = [v for k, v in _atoms(simulation["accessibility"]) if k == "text" and isinstance(v, str)]
    counts = _counts(" ".join(texts))
    _require(_counts(simulation["text"]) == counts, "text_and_accessibility_counts_differ")
    rows = []
    for text in texts:
        normalized = " ".join(text.split())
        match = ROW.fullmatch(normalized)
        if match:
            rows.append({"name": match[1], "row_accessibility_sha256": _sha(normalized.encode())})
    _require(
        len(rows) == counts["end"] - counts["start"] + 1
        and len({r["name"].casefold() for r in rows}) == len(rows),
        "duplicate_or_missing_page_rows",
    )
    # These are the two observed unnamed footer controls, NOT a direction or a
    # locator. Only their enabled state may vary between recorded pages.
    controls = []
    pager_count = 0
    for control in simulation["controls"]:
        item = dict(control)
        if item.get("role") == "button" and item.get("accessibility") in {"- button", "- button [disabled]"}:
            _require(
                type(item.get("enabled")) is bool
                and item["enabled"] == (item["accessibility"] == "- button"),
                "inconsistent_unnamed_control",
            )
            pager_count += 1
            item["accessibility"], item["enabled"] = "- button", None
        controls.append(item)
    _require(pager_count == 2, "observed_footer_control_pair_required")
    status = []
    for pattern in (
        r"\bFUNDING\s*\$([0-9]+)\b",
        r"\bDATA QUALITY\s+([0-9]+(?:\.[0-9]+)?)%",
        r"\bSCAVENGER HUNT\s+([0-9]+)/8\b",
    ):
        values = re.findall(pattern, " ".join(simulation["text"].upper().split()))
        _require(len(values) == 1, "ambiguous_project_status")
        status.append(values[0])
    boundary = {
        "outer_url": report.get("outer_url"),
        "outer_controls": report.get("outer_controls"),
        "frames": [f for f in report["frames"] if f["url"] != SIMULATION_URL],
        "simulation_id": simulation["id"],
        "simulation_controls": controls,
        "status": status,
    }
    return counts, rows, boundary, rows_hash(simulation["text"])


def _load_page(book, reference):
    _require(
        isinstance(reference, dict)
        and set(reference) == {"receipt", "sha256"}
        and _checksum(reference["sha256"]),
        "invalid_page_reference",
    )
    path = book.path(_relative(reference["receipt"]))
    _require(path.name == "page.json", "page_receipt_name_required")
    data = book.json(path, reference["sha256"])
    _require(
        isinstance(data, dict)
        and set(data) == {"schema_version", "mode", "section", "viewing", "rows", "source_sha256"}
        and type(data["schema_version"]) is int
        and data["schema_version"] == 1
        and data["mode"] == PAGE_MODE
        and data["section"] == "stellar",
        "unsupported_page_receipt",
    )
    hashes = data["source_sha256"]
    _require(
        isinstance(hashes, dict)
        and set(hashes) == CAPTURE_FILES
        and all(_checksum(v) for v in hashes.values()),
        "exact_capture_hashes_required",
    )
    captures, parsed = {}, {}
    for section in ("before", "after"):
        manifest = book.json(path.parent / section / "manifest.json", hashes[section + "/manifest.json"])
        report = book.json(path.parent / section / "observation.json", hashes[section + "/observation.json"])
        _require(
            isinstance(manifest, dict)
            and type(manifest.get("schema_version")) is int
            and manifest["schema_version"] == 1
            and manifest.get("mode") == "read_only_browser_preflight"
            and type(manifest.get("actions_executed")) is int
            and manifest["actions_executed"] == 0
            and manifest.get("browser_acceptance_passed") is False
            and manifest.get("observation_sha256") == hashes[section + "/observation.json"],
            "capture_manifest_mismatch",
        )
        captures[section], parsed[section] = report, _page_capture(report)
    before, after = captures["before"], captures["after"]
    _require(
        screen_identity(before) == screen_identity(after) and parsed["before"] == parsed["after"],
        "page_changed_during_capture",
    )
    begin, end = _timestamp(before.get("captured_at")), _timestamp(after.get("captured_at"))
    _require(begin < end, "fresh_ordered_capture_pair_required")
    counts, rows, boundary, anchor = parsed["after"]
    _require(
        isinstance(data["viewing"], dict)
        and set(data["viewing"]) == {"start", "end", "total"}
        and all(type(v) is int for v in data["viewing"].values())
        and data["viewing"] == counts,
        "receipt_range_mismatch",
    )
    expected_rows = [{**r, "source_sha256": hashes["after/observation.json"]} for r in rows]
    _require(data["rows"] == expected_rows, "receipt_rows_mismatch")
    return {
        "receipt": reference["receipt"],
        "receipt_sha256": reference["sha256"],
        "viewing": counts,
        "rows": expected_rows,
        "screen_sha256": screen_identity(after),
        "visible_rows_sha256": anchor,
        "before_time": begin,
        "after_time": end,
        "boundary": boundary,
        "after_capture_sha256": hashes["after/observation.json"],
    }


def _validate(book, plan_path, plan_sha256):
    _require(_checksum(plan_sha256), "plan_hash_required")
    plan_path = book.path(plan_path)
    plan = book.json(plan_path, plan_sha256)
    _require(
        isinstance(plan, dict)
        and set(plan)
        == {"schema_version", "mode", "forward", "revisit", "assessment_anchor", "expected_stars"}
        and type(plan["schema_version"]) is int
        and plan["schema_version"] == 1
        and plan["mode"] == PLAN_MODE,
        "unsupported_plan",
    )
    _require(
        isinstance(plan["forward"], list)
        and 1 <= len(plan["forward"]) <= 30
        and isinstance(plan["revisit"], list)
        and len(plan["revisit"]) == len(plan["forward"]),
        "complete_forward_and_reverse_pass_required",
    )
    references = [*plan["forward"], *plan["revisit"], plan["assessment_anchor"]]
    visits = [_load_page(book, ref) for ref in references]
    _require(len({p["receipt"] for p in visits}) == len(visits), "reused_page_receipt")
    _require(
        all(a["after_time"] < b["before_time"] for a, b in pairwise(visits)), "capture_chronology_mismatch"
    )
    _require(all(p["boundary"] == visits[0]["boundary"] for p in visits), "project_boundary_changed")
    count = len(plan["forward"])
    forward, revisit, anchor = visits[:count], visits[count:-1], visits[-1]
    total, next_start, rows = forward[0]["viewing"]["total"], 1, []
    for page in forward:
        viewing = page["viewing"]
        _require(
            viewing["total"] == total and viewing["start"] == next_start, "noncontiguous_or_changed_total"
        )
        next_start = viewing["end"] + 1
        rows.extend(page["rows"])
    _require(next_start == total + 1 and len(rows) == total, "incomplete_collection_coverage")
    _require(len({r["name"].casefold() for r in rows}) == total, "duplicate_collection_name")
    for source, repeated in zip(reversed(forward), revisit):
        _require(
            source["viewing"] == repeated["viewing"]
            and source["screen_sha256"] == repeated["screen_sha256"]
            and source["visible_rows_sha256"] == repeated["visible_rows_sha256"],
            "revisit_changed_or_reordered",
        )
    _require(
        anchor["viewing"] == forward[0]["viewing"]
        and anchor["screen_sha256"] == forward[0]["screen_sha256"]
        and anchor["visible_rows_sha256"] == forward[0]["visible_rows_sha256"],
        "assessment_anchor_changed",
    )
    expected = plan["expected_stars"]
    _require(
        isinstance(expected, list)
        and len(expected) == total
        and all(isinstance(n, str) and re.fullmatch(NAME, n) and n == " ".join(n.split()) for n in expected)
        and len({n.casefold() for n in expected}) == total
        and {n.casefold() for n in expected} == {r["name"].casefold() for r in rows},
        "expected_names_mismatch",
    )
    collection = {
        "domain": "recorded_stellar_collection_v1",
        "total": total,
        "rows": [
            {
                "ordinal": i + 1,
                "name": r["name"].casefold(),
                "row_accessibility_sha256": r["row_accessibility_sha256"],
            }
            for i, r in enumerate(rows)
        ],
    }
    public_visit = lambda p: {
        k: v for k, v in p.items() if k not in {"boundary", "before_time", "after_time"}
    }
    book.unchanged()
    return {
        "schema_version": 1,
        "mode": MODE,
        "authority": "recorded_capture_consistency_only",
        "recorded_page_chain_validated": True,
        "total_collected_in_recorded_chain": total,
        "page_count": count,
        "forward": [public_visit(p) for p in forward],
        "revisit": [public_visit(p) for p in revisit],
        "assessment_anchor": public_visit(anchor),
        "whole_collection_sha256": _digest(collection),
        "visible_assessment_anchor_sha256": anchor["visible_rows_sha256"],
        "collection_digest_domain": collection["domain"],
        "rows": collection["rows"],
        "plan_path": str(plan_path.relative_to(book.root)),
        "plan_sha256": plan_sha256,
        "source_sha256": dict(book.hashes),
        "source_bytes": book.bytes,
        "capture_timestamp_order_validated": True,
        "reverse_revisit_validated": True,
        "recorded_complete_coverage_validated": True,
        "live_pagination_verified": False,
        "pager_controls_grounded": False,
        "pager_transition_verified": False,
        "same_live_attempt_verified": False,
        "visible_exposure_verified": False,
        "collection_count_verified": False,
        "complete_visible_list_verified": False,
        "assessment_panel_verified": False,
        "journal_import_enabled": False,
        "scoring_input_enabled": False,
        "browser_actions": 0,
        "journal_writes": 0,
        "scientific_verified": False,
        "training_label": False,
        "task_completed": False,
        "project_completed": False,
        "submitted": False,
        "automatic_retry": False,
        "pending_gate": "ground_enabled_pager_and_validate_real_page_transitions_before_live_use",
    }


def validate_paginated_inventory(run_history, plan_path, plan_sha256):
    """Read-only validation. Paths in the pinned plan are history-relative."""
    try:
        return _validate(_Sources(run_history), plan_path, plan_sha256)
    except PaginatedInventoryError:
        raise
    except (OSError, KeyError, TypeError, ValueError, BrowserSafetyStop):
        raise PaginatedInventoryError("paginated_inventory_invalid_source_structure") from None


def merge_paginated_inventory(run_history, output, *, plan_path, plan_sha256):
    """Write only validated.json in a new owned output; never confirmed.json.

    The deliberately distinct name/mode cannot be mistaken for a currently
    accepted single-page inventory by the existing journal/scoring importers.
    """
    try:
        book = _Sources(run_history)
        result = _validate(book, plan_path, plan_sha256)
        directory = book.path(output)
        _require(directory != book.root and not directory.exists(), "new_owned_output_required")
        _require(
            not any(book.path(p).is_relative_to(directory) for p in book.hashes), "output_contains_source"
        )
        book.unchanged()
        directory.mkdir(parents=True, exist_ok=False)
        book.unchanged()
        persist_json(directory / "validated.json", result)
        return result
    except PaginatedInventoryError:
        raise
    except (OSError, KeyError, TypeError, ValueError, BrowserSafetyStop):
        raise PaginatedInventoryError("paginated_inventory_invalid_source_structure") from None
