"""Bounded visible stellar-list pagination, separate from legacy inventories.

The native 950x600 layout and two 30px footer buttons were observed in probes
332--335: right advances 1--10 to 11--11; left returns to the original rows.
Their horizontal positions vary with the footer text. No SVG/path/event-handler
data, hidden catalog, sort mutation or application state is read.

This certifies observed traversal/revisit consistency, NOT an atomic server
snapshot or analyzed work. The offline page-chain validator remains an explicitly
non-authoritative consistency layer; its flags are never promoted or modified.
"""

import hashlib
import json
import math
import os
import re
import time

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_collected import row_icon_exposed
from .browser_collected_revisit import parse_stellar_row
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_project_inventory import NAME, ROW, VISIBLE_TEXT, _outer_exposed
from .browser_project_submission_preflight import _EXPOSED
from .browser_stellar import SIMULATION_URL, _atoms
from .contracts import RuntimeEvent
from .project_paginated_inventory import (
    PAGE_MODE,
    PLAN_MODE,
    _counts,
    _digest,
    _page_capture,
    _Sources,
    validate_paginated_inventory,
)

MODE = "live_paginated_collected_stellar_inventory"
AUTHORITY = "guarded_native_pager_visible_readback"
NATIVE_MODE = "exposed_stellar_page_pair"
TRANSITION_MODE = "guarded_stellar_pager_transition"
GEOMETRY = "observed_950x600_footer_pair_v1"
ZERO = (
    "answer_writes",
    "save_clicks",
    "deletion_clicks",
    "assessment_clicks",
    "submission_clicks",
    "collection_clicks",
    "scrolls",
    "journal_writes",
)
FALSE = (
    "atomic_server_snapshot_verified",
    "hidden_catalog_read",
    "learned_perception",
    "scientific_verified",
    "training_label",
    "task_completed",
    "project_completed",
    "cross_session_persistence_verified",
    "automatic_retry",
    "config_mutated",
    "journal_import_performed",
    "assessment_panel_verified",
)


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("paginated_inventory_live_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _names(values):
    _require(
        isinstance(values, (list, tuple))
        and 1 <= len(values) <= 30
        and all(isinstance(n, str) and re.fullmatch(NAME, n) and n == " ".join(n.split()) for n in values)
        and len({n.casefold() for n in values}) == len(values),
        "exact_unique_expected_stars_required",
    )
    return list(values)


def paired_footer_autosave(report):
    """Recognize probe340's exact paired footer notice without changing evidence."""
    frames = [f for f in report.get("frames", []) if f.get("url") == SIMULATION_URL]
    if len(frames) != 1:
        return False
    text, ax = frames[0].get("text", ""), frames[0].get("accessibility", "")
    if not isinstance(text, str) or not isinstance(ax, str):
        return False
    return bool(
        text.count("Data saved") == ax.count("Data saved") == 1
        and re.search(
            r"\nData saved(?=\n[ \n]*VIEWING\n[0-9]+-[0-9]+ OF [0-9]+"
            r"\n[ \n]*TOTAL COLLECTED\n[0-9]+\nSave$)",
            text,
        )
        and re.search(
            r"\n- text: Data saved(?=\n- button(?: \[disabled\])?"
            r"\n- button(?: \[disabled\])?\n- text: viewing [0-9]+-[0-9]+ of [0-9]+"
            r' total collected [0-9]+\n- button "Save"$)',
            ax,
        )
    )


def read_settled_inventory(page, config, *, check, deadline, clock=time.monotonic, inspector=None):
    """Wait read-only for a known list notice, retaining strict raw comparisons.

    The caller supplies its absolute deadline and complete context/source guard.
    Three seconds is a sub-budget, not an extension or browser-action retry.
    """
    inspect = inspector or inspect_page
    check()
    until = min(deadline, clock() + 3)
    report = inspect(page, config)
    while paired_footer_autosave(report):
        check()
        _require(clock() < until, "autosave_notice_did_not_settle")
        page.wait_for_timeout(min(100, max(1, int((until - clock()) * 1000))))
        check()
        report = inspect(page, config)
    check()
    return report


def _parse(report):
    counts, hashed, boundary, row_sha = _page_capture(report)
    _require(
        counts["start"] in {1, 11, 21} and counts["end"] == min(counts["start"] + 9, counts["total"]),
        "unsupported_page_size",
    )
    frame = next(f for f in report["frames"] if f["url"] == SIMULATION_URL)
    rows, other = [], []
    for kind, value in _atoms(frame["accessibility"]):
        if kind not in {"text", "subscript", "superscript"} or not isinstance(value, str):
            continue
        text = " ".join(value.split())
        match = ROW.fullmatch(text) if kind == "text" else None
        if match:
            parse_stellar_row(text, match[1])
            rows.append({"name": match[1], "text": text, "row_accessibility_sha256": _sha(text.encode())})
        elif not (kind == "text" and re.search(r"\bviewing\b", text, re.IGNORECASE)):
            other.append((kind, text))
    _require(
        [{k: r[k] for k in ("name", "row_accessibility_sha256")} for r in rows] == hashed,
        "row_parser_mismatch",
    )
    # Preserve the public non-row AX/header context (including assessment details)
    # in addition to the existing unchanged controls, funding and outside frames.
    boundary["non_row_accessibility"] = other
    return counts, rows, boundary, row_sha


def _geometry(pair, counts):
    _require(isinstance(pair, list) and len(pair) == 2, "exact_footer_pair_required")
    for index, control in enumerate(pair):
        _require(
            set(control) == {"index", "target_id", "box", "enabled", "native_button", "exposed"}
            and control["index"] == index
            and type(control["index"]) is int
            and isinstance(control["target_id"], str)
            and re.fullmatch(r"simulation-0:c[0-9]+", control["target_id"])
            and control["native_button"] is True
            and control["exposed"] is True
            and type(control["enabled"]) is bool,
            "unsupported_footer_control",
        )
        box = control["box"]
        _require(
            isinstance(box, dict)
            and set(box) == {"x", "y", "width", "height"}
            and all(type(v) in {int, float} and math.isfinite(v) for v in box.values())
            and abs(box["width"] - 30) <= 0.1
            and abs(box["height"] - 30) <= 0.1
            and abs(box["y"] - 555) <= 0.1
            and 300 <= box["x"] <= 390,
            "unsupported_footer_geometry",
        )
    a, b = [p["box"] for p in pair]
    _require(
        pair[0]["target_id"] != pair[1]["target_id"]
        and abs(b["x"] - a["x"] - 34.453125) <= 0.15
        and abs(a["y"] - b["y"]) <= 0.1
        and [p["enabled"] for p in pair] == [counts["start"] > 1, counts["end"] < counts["total"]],
        "footer_direction_or_enabled_state_changed",
    )


def _pager(frame, report, counts):
    viewport = frame.locator("body").evaluate("()=>({width:innerWidth,height:innerHeight})")
    _require(viewport == {"width": 950, "height": 600}, "unsupported_frame_geometry")
    recorded = next(f for f in report["frames"] if f["url"] == SIMULATION_URL)
    captured = [
        c
        for c in recorded["controls"]
        if c["role"] == "button" and c["accessibility"] in {"- button", "- button [disabled]"}
    ]
    candidates = []
    for control in frame.get_by_role("button").all():
        if control.is_visible() and control.aria_snapshot() in {"- button", "- button [disabled]"}:
            box = control.evaluate(_EXPOSED, "button")
            _require(isinstance(box, dict) and _outer_exposed(frame, [box]), "pager_occluded")
            candidates.append((box, control))
    _require(len(captured) == len(candidates) == 2, "exact_footer_pair_required")
    candidates.sort(key=lambda item: item[0]["x"])
    pair, handles = [], []
    for index, ((box, control), observed) in enumerate(zip(candidates, captured)):
        _require(
            control.aria_snapshot() == observed["accessibility"]
            and control.is_enabled() is observed["enabled"],
            "pager_capture_mismatch",
        )
        pair.append(
            {
                "index": index,
                "target_id": observed["id"],
                "box": box,
                "enabled": control.is_enabled(),
                "native_button": True,
                "exposed": True,
            }
        )
        handles.append(control.element_handle(timeout=2000))
    _geometry(pair, counts)
    return pair, handles


def _visible_page(frame, counts, rows):
    """Partial-page counterpart; the original full-page validator is untouched."""
    nodes = frame.locator("body").evaluate(VISIBLE_TEXT)
    _require(_counts(" ".join(n["text"] for n in nodes)) == counts, "footer_unexposed_or_changed")
    _require(_outer_exposed(frame, [b for n in nodes for b in n["boxes"]]), "outer_frame_occluded")
    visible, handles = [], []
    for row in rows:
        name = row["name"]
        _require(
            sum(n["text"].casefold() == name.casefold() for n in nodes) == 1,
            "row_name_unexposed_or_ambiguous",
        )
        names = [
            e
            for e in frame.get_by_text(re.compile("^" + re.escape(name) + "$", re.IGNORECASE)).all()
            if e.is_visible()
        ]
        _require(len(names) == 1, "ambiguous_row_locator")
        label = names[0]
        data, container = label.locator("../.."), label.locator("../../..")
        _require(
            " ".join(data.inner_text().split()).casefold() == row["text"].casefold(),
            "row_text_or_structure_changed",
        )
        nb, rb = label.bounding_box(), container.bounding_box()
        eyes = []
        for icon in container.get_by_role("img").all():
            b = icon.bounding_box() if icon.is_visible() else None
            if (
                b
                and nb
                and rb
                and icon.aria_snapshot() == "- img"
                and b["width"] == 22
                and b["height"] == 25
                and rb["x"] <= b["x"] < b["x"] + 22 <= nb["x"]
                and rb["y"] <= b["y"] < b["y"] + 25 <= rb["y"] + rb["height"]
                and row_icon_exposed(icon)
            ):
                eyes.append(icon)
        _require(len(eyes) == 1, "row_eye_structure_unverified")
        visible.append({**row, "visible_name_box": nb})
        handles.append(label.element_handle(timeout=2000))
    return visible, handles


def parse_inventory_page(report):
    """Strict public page mapping for an already authorized live-paged consumer."""
    counts, rows, _, _ = _parse(report)
    return counts, [{"name": r["name"], "text": r["text"]} for r in rows]


def visible_inventory_page(frame, counts, rows):
    """Expose only this page's actual rows; does not certify the whole collection."""
    _require(
        isinstance(rows, list) and len(rows) == counts["end"] - counts["start"] + 1,
        "visible_page_row_count_mismatch",
    )
    enriched = []
    for row in rows:
        match = ROW.fullmatch(row["text"])
        _require(match and match[1] == row["name"], "visible_page_row_invalid")
        parse_stellar_row(row["text"], row["name"])
        enriched.append(
            {"name": row["name"], "text": row["text"], "row_accessibility_sha256": _sha(row["text"].encode())}
        )
    return _visible_page(frame, counts, enriched)


def _same_handles(left, right):
    return len(left) == len(right) and all(
        a.evaluate("(a,b)=>a.isConnected&&a===b", b) for a, b in zip(left, right)
    )


def _relative(book, path):
    return str(book.path(path).relative_to(book.root))


def _reference(book, path):
    return {"path": _relative(book, path), "sha256": _sha(book.read(path))}


def _load_ref(book, reference):
    _require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "invalid_reference")
    path = book.path(reference["path"])
    return path, book.json(path, reference["sha256"])


def _scope(value):
    _require(
        isinstance(value, dict)
        and type(value.get("schema_version")) is int
        and value["schema_version"] == 1
        and value.get("mode") == MODE
        and value.get("authority") == AUTHORITY
        and value.get("pager_geometry") == GEOMETRY
        and value.get("page_size") == 10
        and value.get("max_pages") == 3
        and value.get("max_total") == 30,
        "unsupported_scope",
    )
    _names(value.get("expected_stars"))
    for key, lower, upper in (("max_seconds", 1, 600), ("max_clicks", 0, 4), ("max_advances", 1, 8)):
        number = value.get(key)
        _require(
            type(number) in ({int, float} if key == "max_seconds" else {int})
            and math.isfinite(number)
            and lower <= number <= upper,
            "invalid_fixed_budget",
        )
    _require(
        all(type(value.get(k)) is int and value[k] == 0 for k in ZERO)
        and all(value.get(k) is False for k in FALSE)
        and isinstance(value.get("config_sha256"), str)
        and re.fullmatch(r"[a-f0-9]{64}", value["config_sha256"]),
        "unsupported_scope_flags",
    )
    _require(
        set(value)
        == {
            "schema_version",
            "mode",
            "authority",
            "pager_geometry",
            "page_size",
            "max_pages",
            "max_total",
            "expected_stars",
            "max_seconds",
            "max_clicks",
            "max_advances",
            "config_sha256",
            *ZERO,
            *FALSE,
        },
        "unexpected_scope_fields",
    )


def _native_page(book, reference, recorded):
    path, native = _load_ref(book, reference)
    _require(
        path.name == "native.json"
        and native.get("schema_version") == 1
        and native.get("mode") == NATIVE_MODE
        and native.get("geometry") == GEOMETRY
        and native.get("stable_native_handles_verified") is True
        and native.get("visible_rows_exposed") is True,
        "unsupported_native_page",
    )
    _require(
        set(native)
        == {
            "schema_version",
            "mode",
            "geometry",
            "page",
            "rows",
            "pager",
            "stable_native_handles_verified",
            "visible_rows_exposed",
        },
        "native_page_fields",
    )
    page_path, page = _load_ref(book, native["page"])
    _require(
        page_path == path.parent / "page.json"
        and str(page_path.relative_to(book.root)) == recorded["receipt"]
        and native["page"]["sha256"] == recorded["receipt_sha256"],
        "page_source_mismatch",
    )
    after = book.json(path.parent / "after/observation.json", page["source_sha256"]["after/observation.json"])
    counts, rows, boundary, _ = _parse(after)
    expected = []
    _require(isinstance(native["rows"], list) and len(native["rows"]) == len(rows), "native_rows_missing")
    for row, declaration in zip(rows, native["rows"]):
        box = declaration.get("visible_name_box")
        _require(
            isinstance(box, dict)
            and set(box) == {"x", "y", "width", "height"}
            and all(type(v) in {int, float} and math.isfinite(v) for v in box.values())
            and box["width"] > 0
            and box["height"] > 0,
            "invalid_row_geometry",
        )
        expected.append(
            {**row, "visible_name_box": box, "source_sha256": page["source_sha256"]["after/observation.json"]}
        )
    _require(native["rows"] == expected, "native_rows_changed")
    _geometry(native["pager"], counts)
    return native, after, boundary


def _validated(book, directory, declaration):
    """Recompute exact new-mode proof; never treat offline flags as live proof."""
    scope = book.json(directory / "scope.json")
    _scope(scope)
    plan_path, plan = _load_ref(book, declaration["plan"])
    _require(plan_path == directory / "plan.json", "plan_outside_output")
    consistency = validate_paginated_inventory(book.root, plan_path, declaration["plan"]["sha256"])
    stored_consistency = book.json(directory / "consistency.json")
    _require(
        stored_consistency == consistency and consistency["live_pagination_verified"] is False,
        "offline_consistency_changed",
    )
    for name, expected in consistency["source_sha256"].items():
        book.read(name, expected)
    visits = [*consistency["forward"], *consistency["revisit"], consistency["assessment_anchor"]]
    page_count = consistency["page_count"]
    _require(
        page_count <= 3
        and plan["expected_stars"] == scope["expected_stars"]
        and isinstance(declaration["native_pages"], list)
        and len(declaration["native_pages"]) == len(visits),
        "incomplete_native_chain",
    )
    native, captures, boundaries = [], [], []
    for reference, recorded in zip(declaration["native_pages"], visits):
        n, c, b = _native_page(book, reference, recorded)
        _require(book.path(reference["path"]).is_relative_to(directory), "native_page_outside_output")
        native.append(n)
        captures.append(c)
        boundaries.append(b)
    _require(all(b == boundaries[0] for b in boundaries), "non_row_boundary_changed")
    transitions = declaration["transitions"]
    _require(
        isinstance(transitions, list)
        and len(transitions) == 2 * (page_count - 1)
        and len(transitions) <= scope["max_clicks"],
        "incomplete_transition_chain",
    )
    pairs = [(i, i + 1, "next") for i in range(page_count - 1)]
    pairs += [(page_count + i, page_count + i + 1, "previous") for i in range(page_count - 1)]
    for reference, (before_index, after_index, direction) in zip(transitions, pairs):
        path, receipt = _load_ref(book, reference)
        _require(
            path.name == "confirmed.json" and path.parent.parent == directory / "transitions",
            "transition_outside_output",
        )
        reserved = book.json(path.parent / "reserved.json")
        _require(
            receipt
            == {
                **reserved,
                "click_returned": True,
                "navigation_clicks": 1,
                "after_native_page": declaration["native_pages"][after_index],
            },
            "transition_receipt_changed",
        )
        target = native[before_index]["pager"][1 if direction == "next" else 0]
        expected = {
            "schema_version": 1,
            "mode": TRANSITION_MODE,
            "direction": direction,
            "from": visits[before_index]["viewing"],
            "to": visits[after_index]["viewing"],
            "target": target,
            "before_native_page": declaration["native_pages"][before_index],
            "max_clicks": 1,
            "automatic_retry": False,
            "native_binding_verified": True,
            **dict.fromkeys(ZERO, 0),
        }
        _require(reserved == expected and target["enabled"] is True, "invalid_transition_reservation")
        for part in ("before", "pre-click"):
            manifest = book.json(path.parent / part / "manifest.json")
            capture = book.json(path.parent / part / "observation.json", manifest["observation_sha256"])
            _require(
                screen_identity(capture) == screen_identity(captures[before_index]),
                "transition_before_changed",
            )
    rows = []
    for index, value in enumerate(native[:page_count]):
        for row in value["rows"]:
            rows.append({**row, "ordinal": len(rows) + 1, "page": declaration["native_pages"][index]})
    anchor_path = book.path(declaration["native_pages"][-1]["path"]).parent / "after"
    flags = {
        "live_pagination_verified": True,
        "pager_controls_grounded": True,
        "pager_transition_verified": True,
        "same_live_attempt_verified": True,
        "visible_exposure_verified": True,
        "collection_count_verified": True,
        "complete_collection_traversal_verified": True,
        "reverse_revisit_verified": True,
        "fresh_first_page_anchor_verified": True,
        "expected_stars_verified": True,
        # This field deliberately does NOT impersonate the legacy full-page receipt.
        "complete_visible_list_verified": False,
    }
    result = {
        **scope,
        **flags,
        "section": "stellar",
        "total_collected": len(rows),
        "rows": rows,
        "page_count": page_count,
        "viewing": consistency["assessment_anchor"]["viewing"],
        "whole_collection_sha256": consistency["whole_collection_sha256"],
        "collection_digest_domain": consistency["collection_digest_domain"],
        "visible_assessment_anchor_sha256": consistency["visible_assessment_anchor_sha256"],
        "anchor_capture": _relative(book, anchor_path),
        "anchor_capture_sha256": consistency["assessment_anchor"]["after_capture_sha256"],
        "plan": declaration["plan"],
        "native_pages": declaration["native_pages"],
        "transitions": transitions,
        "navigation_clicks": len(transitions),
        "navigation_dispatch_attempts": len(transitions),
        "browser_actions": len(transitions),
        "advances": 2 * page_count + 1,
        "status": "completed",
        "finished": True,
        "phase": "inventory_verified",
        "failure_reason": None,
        "validated_directories": sorted({_relative(book, book.path(p).parent) for p in book.hashes}),
    }
    for name in result["validated_directories"]:
        _require(
            not any((book.path(name) / f).exists() for f in ("stopped.json", "invalidated.json")),
            "failed_source_directory",
        )
    _require(result["advances"] <= scope["max_advances"], "advance_budget_changed")
    return result, captures[-1]


def load_live_paginated_inventory(run_history, directory, expected_sha256=None):
    """Return (new receipt, full parsed rows, actual first-page anchor capture).

    Only this exact live mode is supported. Owned source files, native receipts,
    transitions, immutable hashes and the independent consistency layer are all
    recomputed; there is no acceptance of a caller-edited offline result.
    """
    book = _Sources(run_history)
    directory = book.path(directory)
    _require(not any((directory / n).exists() for n in ("stopped.json", "invalidated.json")), "failed_output")
    receipt = book.json(directory / "confirmed.json", expected_sha256)
    _require(receipt.get("mode") == MODE, "unsupported_receipt_mode")
    # Avoid self-reference in the exact artifact map.
    book.hashes.pop(_relative(book, directory / "confirmed.json"))
    expected, anchor = _validated(book, directory, receipt)
    events = book.read(directory / "events.jsonl")
    lines = [RuntimeEvent.model_validate_json(line) for line in events.splitlines()]
    _require(
        lines
        and [e.sequence for e in lines] == list(range(len(lines)))
        and all(e.version == 1 for e in lines)
        and lines[-1].event == "episode_summary"
        and lines[-1].payload == expected,
        "event_stream_changed",
    )
    expected.update(events_sha256=_sha(events), source_sha256=dict(book.hashes))
    _require(receipt == expected, "receipt_or_sources_changed")
    book.unchanged()
    return receipt, [{"name": r["name"], "text": r["text"]} for r in receipt["rows"]], anchor


class PaginatedInventorySteps:
    """One native pager click at most per advance; no retries or browser ownership."""

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        expected_stars,
        max_seconds=240,
        max_clicks=4,
        max_advances=8,
        emit=lambda _: None,
        cancelled=lambda: False,
        _clock=time.monotonic,
    ):
        self.book = _Sources(run_history)
        self.output = self.book.path(output)
        _require(self.output != self.book.root and not self.output.exists(), "new_owned_output_required")
        self.page, self.config = page, config.model_copy(deep=True)
        self._config = self.config.model_dump(mode="json")
        rules = [r for r in self.config.frames if r.url == SIMULATION_URL]
        _require(
            len(rules) == 1 and rules[0].count == 1 and rules[0].name == "simulation",
            "unsupported_simulation_boundary",
        )
        rules[0].required_text = []
        self._config = self.config.model_dump(mode="json")
        self.scope = {
            "schema_version": 1,
            "mode": MODE,
            "authority": AUTHORITY,
            "pager_geometry": GEOMETRY,
            "page_size": 10,
            "max_pages": 3,
            "max_total": 30,
            "expected_stars": _names(expected_stars),
            "max_seconds": max_seconds,
            "max_clicks": max_clicks,
            "max_advances": max_advances,
            "config_sha256": _digest(self._config),
            **dict.fromkeys(ZERO, 0),
            **dict.fromkeys(FALSE, False),
        }
        _scope(self.scope)
        _require(callable(emit) and callable(cancelled) and callable(_clock), "invalid_callback")
        self._callback, self._cancelled, self._clock = emit, cancelled, _clock
        self._started = _clock()
        self.phase, self.status, self.failure, self.report = "initial_capture", "paused", None, None
        self.advances = self.clicks = self.attempts = self._sequence = 0
        self._busy = self._finalizing = self._forward_failed = self._aborted = False
        self._frames = self._frame = self._boundary = self._previous = None
        self._dialogs, self._popups, self._listeners = [], [], False
        self.forward, self.revisit, self.native, self.transitions = [], [], [], []
        self._anchor = None
        self.output.mkdir(parents=True, exist_ok=False)
        persist_json(self.output / "scope.json", self.scope)
        self.book.read(self.output / "scope.json")
        self._stream = (self.output / "events.jsonl").open("x", encoding="utf-8")

    @property
    def finished(self):
        return self.status in {"completed", "stopped", "aborted"}

    def _check(self):
        _require(not self.finished and not self._aborted and not self._cancelled(), "cancelled_or_stopped")
        _require(self._clock() - self._started < self.scope["max_seconds"], "time_limit")
        _require(
            self.config.model_dump(mode="json") == self._config
            and self.book.json(self.output / "scope.json") == self.scope,
            "scope_changed",
        )
        self.book.unchanged()

    def _context(self):
        self._check()
        if not self._listeners:
            self._frames = self.page.frames.copy()
            self._dialog = lambda _: self._dialogs.append(True)
            self._popup = lambda _: self._popups.append(True)
            self.page.on("dialog", self._dialog)
            self.page.context.on("page", self._popup)
            self._listeners = True
        self.page.wait_for_timeout(0)
        _require(
            not self._dialogs
            and not self._popups
            and self.page.frames == self._frames
            and len(self.page.context.pages) == 1
            and self.config.allows(self.page.url),
            "context_changed",
        )

    def _settled_report(self):
        """Read-only settling, max3 seconds within the original component deadline.

        Never strip a message, retry a click, or accept changed rows/context.
        All subsequent native and capture comparisons still use raw evidence.
        """
        return read_settled_inventory(
            self.page,
            self.config,
            check=self._context,
            deadline=self._started + self.scope["max_seconds"],
            clock=self._clock,
        )

    def _read(self):
        report = self._settled_report()
        counts, rows, boundary, row_sha = _parse(report)
        frames = [f for f in self.page.frames if f.url == SIMULATION_URL]
        _require(
            len(frames) == 1
            and frames[0].parent_frame == self.page.main_frame
            and (self._frame is None or self._frame == frames[0]),
            "frame_changed",
        )
        self._frame = frames[0]
        pair, pager_handles = _pager(self._frame, report, counts)
        exposed, row_handles = _visible_page(self._frame, counts, rows)
        if self._boundary is None:
            self._boundary = boundary
        _require(
            boundary == self._boundary and counts["total"] == len(self.scope["expected_stars"]),
            "collection_or_boundary_changed",
        )
        self._context()
        return {
            "report": report,
            "counts": counts,
            "rows": exposed,
            "pager": pair,
            "handles": pager_handles + row_handles,
            "row_sha": row_sha,
        }

    def _capture(self, report, path):
        save_probe(report, path)
        for filename in ("observation.json", "manifest.json"):
            self.book.read(path / filename)

    def _visit(self, label, expected=None):
        first = self._read()
        if expected is not None:
            _require(first["counts"] == expected, "unexpected_page_transition")
        directory = self.output / label
        self._capture(first["report"], directory / "before")
        last = self._read()
        _require(
            screen_identity(first["report"]) == screen_identity(last["report"])
            and first["rows"] == last["rows"]
            and first["pager"] == last["pager"]
            and _same_handles(first["handles"], last["handles"]),
            "changed_during_page_readback",
        )
        self._capture(last["report"], directory / "after")
        hashes = {
            f"{part}/{name}.json": _sha(self.book.read(directory / part / f"{name}.json"))
            for part in ("before", "after")
            for name in ("observation", "manifest")
        }
        recorded = {
            "schema_version": 1,
            "mode": PAGE_MODE,
            "section": "stellar",
            "viewing": last["counts"],
            "rows": [
                {
                    "name": r["name"],
                    "row_accessibility_sha256": r["row_accessibility_sha256"],
                    "source_sha256": hashes["after/observation.json"],
                }
                for r in last["rows"]
            ],
            "source_sha256": hashes,
        }
        persist_json(directory / "page.json", recorded)
        native = {
            "schema_version": 1,
            "mode": NATIVE_MODE,
            "geometry": GEOMETRY,
            "page": _reference(self.book, directory / "page.json"),
            "rows": [{**r, "source_sha256": hashes["after/observation.json"]} for r in last["rows"]],
            "pager": last["pager"],
            "stable_native_handles_verified": True,
            "visible_rows_exposed": True,
        }
        persist_json(directory / "native.json", native)
        reference = _reference(self.book, directory / "native.json")
        self.native.append(reference)
        self._previous = last["report"]
        self._check()
        return {
            "receipt": _relative(self.book, directory / "page.json"),
            "sha256": native["page"]["sha256"],
        }, reference

    def _transition(self, direction, label):
        self._check()
        _require(self.attempts < self.scope["max_clicks"], "click_limit")
        first = self._read()
        _require(screen_identity(first["report"]) == screen_identity(self._previous), "current_page_changed")
        counts = first["counts"]
        index = 1 if direction == "next" else 0
        start = counts["start"] + (10 if direction == "next" else -10)
        expected = {"start": start, "end": min(start + 9, counts["total"]), "total": counts["total"]}
        _require(
            first["pager"][index]["enabled"] is True and 1 <= start <= counts["total"],
            "direction_not_available",
        )
        directory = self.output / "transitions" / f"{self.attempts + 1:02d}"
        self._capture(first["report"], directory / "before")
        self._emit(
            "action_proposed",
            {
                "kind": "CLICK",
                "target": "pager_" + direction,
                "from": counts,
                "to": expected,
                "action_source": "deterministic_visible_navigation",
            },
        )
        self._check()
        current = self._read()
        _require(
            screen_identity(first["report"]) == screen_identity(current["report"])
            and first["pager"] == current["pager"]
            and _same_handles(first["handles"], current["handles"]),
            "native_binding_changed",
        )
        self._capture(current["report"], directory / "pre-click")
        reservation = {
            "schema_version": 1,
            "mode": TRANSITION_MODE,
            "direction": direction,
            "from": counts,
            "to": expected,
            "target": current["pager"][index],
            "before_native_page": self.native[-1],
            "max_clicks": 1,
            "automatic_retry": False,
            "native_binding_verified": True,
            **dict.fromkeys(ZERO, 0),
        }
        persist_json(directory / "reserved.json", reservation)
        self.book.read(directory / "reserved.json")
        self.attempts += 1  # Crash-conservative; an uncertain transition is never retried.
        self._context()
        # Final native exposure/identity check after durable reservation; no event callback follows it.
        pair, handles = _pager(self._frame, current["report"], counts)
        _require(
            pair == current["pager"] and _same_handles(first["handles"][:2], handles),
            "native_binding_changed",
        )
        self._check()
        timeout = max(1, min(3000, int((self.scope["max_seconds"] - (self._clock() - self._started)) * 1000)))
        handles[index].click(timeout=timeout)
        self.clicks += 1
        self._check()
        recorded, native = self._visit(label, expected)
        receipt = {**reservation, "click_returned": True, "navigation_clicks": 1, "after_native_page": native}
        persist_json(directory / "confirmed.json", receipt)
        self.transitions.append(_reference(self.book, directory / "confirmed.json"))
        self._emit(
            "action_result",
            {
                "target": "pager_" + direction,
                "navigation_verified": True,
                "from": counts,
                "to": expected,
                "task_completed": False,
            },
        )
        self._check()
        return recorded

    def _emit(self, kind, payload):
        event = RuntimeEvent(event=kind, sequence=self._sequence, run_id=self.output.name, payload=payload)
        self._sequence += 1
        raw = event.model_dump_json()
        self._stream.write(raw + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        if not self._forward_failed:
            try:
                self._callback(json.loads(raw))
            except Exception:  # noqa: BLE001 - callback text may contain private data
                self._forward_failed = True
                raise BrowserSafetyStop("paginated_inventory_live_event_forwarding_failed") from None

    def state(self):
        return {
            "mode": MODE,
            "phase": self.phase,
            "status": self.status,
            "finished": self.finished,
            "advances": self.advances,
            "navigation_clicks": self.clicks,
            "navigation_dispatch_attempts": self.attempts,
            "navigation_outcome_uncertain": self.attempts > len(self.transitions),
            "failure_reason": self.failure,
            "forward_pages": len(self.forward),
            "revisited_pages": len(self.revisit),
            "inventory_dir": _relative(self.book, self.output) if self.status == "completed" else None,
            "inventory_sha256": _sha((self.output / "confirmed.json").read_bytes())
            if self.status == "completed"
            else None,
            **dict.fromkeys(ZERO, 0),
            **dict.fromkeys(FALSE, False),
        }

    def _cleanup(self):
        if self._listeners:
            self.page.remove_listener("dialog", self._dialog)
            self.page.context.remove_listener("page", self._popup)
            self._listeners = False

    def _stop(self, reason, aborted=False):
        if self.finished or self._finalizing:
            return
        self._finalizing = True
        self.status, self.phase, self.failure = ("aborted" if aborted else "stopped"), "stopped", reason
        try:
            self._cleanup()
            self.report = self.state()
            self._emit("episode_summary", self.report)
        except Exception:  # noqa: BLE001 - cleanup must retain the stopped reservation evidence
            self._forward_failed = True
        finally:
            self._stream.close()
            persist_json(self.output / "stopped.json", self.state())
            self._finalizing = False

    def _complete(self):
        plan = {
            "schema_version": 1,
            "mode": PLAN_MODE,
            "forward": self.forward,
            "revisit": self.revisit,
            "assessment_anchor": self._anchor,
            "expected_stars": self.scope["expected_stars"],
        }
        persist_json(self.output / "plan.json", plan)
        plan_ref = _reference(self.book, self.output / "plan.json")
        consistency = validate_paginated_inventory(
            self.book.root, self.output / "plan.json", plan_ref["sha256"]
        )
        persist_json(self.output / "consistency.json", consistency)
        declaration = {"plan": plan_ref, "native_pages": self.native, "transitions": self.transitions}
        evidence = _Sources(self.book.root)
        result, _ = _validated(evidence, self.output, declaration)
        self._check()
        self._emit("episode_summary", result)
        self._check()
        self._cleanup()
        self._stream.close()
        raw_events = evidence.read(self.output / "events.jsonl")
        result.update(events_sha256=_sha(raw_events), source_sha256=dict(evidence.hashes))
        evidence.unchanged()
        persist_json(self.output / "confirmed.json", result)
        self.report = result
        self.phase, self.status = "inventory_verified", "completed"

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy:
            self._aborted = True
            raise BrowserSafetyStop("paginated_inventory_live_reentrant_advance")
        self._busy = True
        try:
            self._check()
            _require(self.advances < self.scope["max_advances"], "advance_limit")
            self.advances += 1
            self._emit("state", self.state())
            self._check()
            if self.phase == "initial_capture":
                total = len(self.scope["expected_stars"])
                record, _ = self._visit("forward-01", {"start": 1, "end": min(10, total), "total": total})
                self.forward.append(record)
                self.phase = "forward_next" if total > 10 else "revisit_last"
            elif self.phase == "forward_next":
                self.forward.append(self._transition("next", f"forward-{len(self.forward) + 1:02d}"))
                if len(self.forward) * 10 >= len(self.scope["expected_stars"]):
                    self.phase = "revisit_last"
            elif self.phase == "revisit_last":
                record, _ = self._visit(f"revisit-{len(self.forward):02d}")
                self.revisit.append(record)
                self.phase = "reverse_previous" if len(self.forward) > 1 else "anchor_capture"
            elif self.phase == "reverse_previous":
                self.revisit.append(
                    self._transition("previous", f"revisit-{len(self.forward) - len(self.revisit):02d}")
                )
                if len(self.revisit) == len(self.forward):
                    self.phase = "anchor_capture"
            elif self.phase == "anchor_capture":
                self._anchor, _ = self._visit("anchor-01")
                self._complete()
            else:
                raise BrowserSafetyStop("paginated_inventory_live_unknown_phase")
        except (KeyboardInterrupt, SystemExit):
            self._aborted = True
            self._stop("operator_aborted", aborted=True)
            raise
        except Exception as exc:  # noqa: BLE001 - driver errors are redacted, never retried
            reason = str(exc)
            if not isinstance(exc, BrowserSafetyStop) or not re.fullmatch(r"[a-z0-9_]{1,180}", reason):
                reason = "paginated_inventory_live_operation_failed"
            self._stop(reason, aborted=self._aborted or self._cancelled())
        finally:
            self._busy = False
        return self.state()

    def pause(self):
        if not self.finished:
            self.status = "paused"
        return self.state()

    def resume(self):
        if not self.finished:
            self.status = "running"
        return self.state()

    def tick(self):
        return self.advance() if self.status == "running" else self.state()

    def abort(self):
        self._aborted = True
        if not self._busy:
            self._stop("operator_aborted", aborted=True)
        return self.state()

    def close(self):
        return self.abort() if not self.finished else self.state()
