"""Strict inventory evidence dispatch, never a browser or a synthetic anchor.

Legacy single-page receipts retain their original validator and binding shape.
The separate live pager validator alone may authorize a paginated receipt;
offline consistency reports are deliberately not accepted as inventory proof.
"""

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .browser_assessment_actions import rows_hash
from .browser_stellar import SIMULATION_URL
from .project_progress import ProgressError

LEGACY_MODE = "read_only_collected_star_inventory"
LIVE_MODE = "live_paginated_collected_stellar_inventory"


def _require(condition, reason):
    if not condition:
        raise ProgressError("project_inventory_source_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _checksum(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _whole(rows):
    """Same ordered-row identity domain as the recorded page-chain validator."""
    collection = {
        "domain": "recorded_stellar_collection_v1",
        "total": len(rows),
        "rows": [
            {
                "ordinal": i + 1,
                "name": row["name"].casefold(),
                "row_accessibility_sha256": _sha(row["text"].encode()),
            }
            for i, row in enumerate(rows)
        ],
    }
    return _sha(json.dumps(collection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


@dataclass(frozen=True)
class InventoryEvidence:
    receipt: dict
    rows: list[dict]
    anchor: dict
    anchor_dir: Path
    whole_collection_sha256: str
    kind: str
    anchor_evidence: dict

    def binding_fields(self):
        """Only additive live provenance; never override a project header version."""
        if self.kind == "legacy_single_page":
            return {}
        _require(self.kind == "live_paginated", "unsupported_kind")
        return {
            "inventory_evidence_version": 1,
            "inventory_kind": "live_paginated",
            "whole_collection_sha256": self.whole_collection_sha256,
            "inventory_anchor": deepcopy(self.anchor_evidence),
        }


def _anchor(book, directory, capture, *, visible_rows_sha256=None):
    directory = book.path(directory)
    _require(book.capture(directory) == capture, "anchor_capture_mismatch")
    return {
        "capture_dir": str(directory.relative_to(book.history)),
        "observation_sha256": _sha(book.read(directory / "observation.json")),
        "manifest_sha256": _sha(book.read(directory / "manifest.json")),
        # The legacy inventory validator does not require the assessment-text
        # hash domain. Do not add a new gate to previously valid old captures.
        "visible_rows_sha256": visible_rows_sha256,
    }


def _load_live(run_history, directory, expected_sha256):
    # Lazy import keeps legacy evidence usable when the optional producer is not
    # installed. A missing validator never falls back to offline-only receipts.
    try:
        from .browser_project_paginated_inventory_steps import load_live_paginated_inventory
    except ImportError:
        raise ProgressError("project_inventory_source_live_validator_unavailable") from None
    return load_live_paginated_inventory(run_history, directory, expected_sha256=expected_sha256)


def load_inventory_source(book, directory, *, expected_sha256=None):
    """Return actual captures and adopt every immutable source into caller book."""
    from .project_evidence import _inventory

    directory = book.clean(book.path(directory))
    raw = book.read(directory / "confirmed.json")
    digest = _sha(raw)
    _require(
        expected_sha256 is None or _checksum(expected_sha256) and digest == expected_sha256,
        "receipt_hash_mismatch",
    )
    receipt = json.loads(raw)
    _require(isinstance(receipt, dict), "unsupported_inventory")
    if receipt.get("mode") == LEGACY_MODE:
        receipt, rows = _inventory(book, directory)
        anchor_dir = directory / "after"
        capture = book.capture(anchor_dir)
        return InventoryEvidence(
            receipt,
            rows,
            capture,
            anchor_dir,
            _whole(rows),
            "legacy_single_page",
            _anchor(book, anchor_dir, capture),
        )
    _require(receipt.get("mode") == LIVE_MODE, "unsupported_inventory")
    return _live_evidence(book, directory, receipt, digest)


def _live_evidence(book, directory, original, digest):
    """The producer's exact validated anchor contract is required, not guessed."""
    receipt, rows, anchor = _load_live(book.history, directory, digest)
    _require(receipt == original, "validator_receipt_mismatch")
    _require(
        receipt.get("mode") == LIVE_MODE
        and receipt.get("authority") == "guarded_native_pager_visible_readback"
        and receipt.get("collection_count_verified") is True
        and receipt.get("live_pagination_verified") is True
        and receipt.get("complete_collection_traversal_verified") is True
        and receipt.get("complete_visible_list_verified") is False,
        "unverified_live_inventory",
    )
    _require(
        isinstance(rows, list)
        and 1 <= len(rows) <= 30
        and all(
            isinstance(r, dict)
            and set(r) == {"name", "text"}
            and all(isinstance(r[k], str) and r[k] for k in ("name", "text"))
            for r in rows
        )
        and len({row["name"].casefold() for row in rows}) == len(rows)
        and type(receipt.get("total_collected")) is int
        and receipt["total_collected"] == len(rows)
        and isinstance(receipt.get("rows"), list)
        and all(isinstance(r, dict) and isinstance(r.get("name"), str) for r in receipt["rows"])
        and [r["name"] for r in receipt["rows"]] == [r["name"] for r in rows]
        and receipt.get("collection_digest_domain") == "recorded_stellar_collection_v1"
        and receipt.get("whole_collection_sha256") == _whole(rows),
        "live_rows_or_whole_digest_mismatch",
    )
    sources = receipt.get("source_sha256")
    from .project_evidence import _hashes

    _hashes(book, sources)
    directories = receipt.get("validated_directories")
    _require(
        isinstance(directories, list)
        and directories
        and all(isinstance(name, str) for name in directories)
        and len(directories) == len(set(directories)),
        "missing_live_directory_audit",
    )
    for name in directories:
        _require(
            isinstance(name, str)
            and name
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and str(Path(name)) == name,
            "invalid_live_directory",
        )
        book.clean(book.history / name)
    _require(
        {str(Path(name).parent) for name in sources} <= set(directories), "incomplete_live_directory_audit"
    )
    location = receipt.get("anchor_capture")
    _require(
        isinstance(location, str)
        and location
        and not Path(location).is_absolute()
        and ".." not in Path(location).parts
        and str(Path(location)) == location,
        "invalid_live_anchor",
    )
    anchor_dir = book.path(book.history / location)
    _require(anchor_dir.is_relative_to(directory), "anchor_outside_inventory")
    frames = [f for f in anchor["frames"] if f["url"] == SIMULATION_URL]
    _require(len(frames) == 1, "ambiguous_anchor_frame")
    visible = rows_hash(frames[0]["text"])
    proof = _anchor(book, anchor_dir, anchor, visible_rows_sha256=visible)
    _require(
        proof["observation_sha256"]
        == receipt.get("anchor_capture_sha256")
        == sources.get(location + "/observation.json")
        and proof["manifest_sha256"] == sources.get(location + "/manifest.json")
        and visible == receipt.get("visible_assessment_anchor_sha256"),
        "live_anchor_source_mismatch",
    )
    book.unchanged()
    return InventoryEvidence(
        receipt, rows, anchor, anchor_dir, receipt["whole_collection_sha256"], "live_paginated", proof
    )
