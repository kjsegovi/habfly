"""Offline stellar transport sources, independent of any planet/Save branch.

The core bundle and consumed files match the legacy No-workflow stellar prefix.
Actual class and visible applicability are retained; absent non-main fields are
never interpreted as zero or recovered from hidden state. This reader proves
historical transport, not current paint, scientific correctness or completion.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_full_stellar import UNITS
from .browser_no_planet_workflow import _events, _Evidence, _hash_provenance, _stellar, _values_match
from .browser_numeric import committed_display
from .browser_star_preflight import validate_star_class_source
from .browser_stellar import CLASSES, StellarMappingError

MODE = "verified_stellar_source_readback_v1"


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("stellar_sources_" + reason)


def _typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _stream(evidence, directory):
    manifest, events = _events(evidence, directory)
    _require(
        all(
            _typed_equal(events[-1].payload.get(key), value)
            for key, value in manifest.items()
            if key != "events_sha256"
        ),
        "manifest_summary_type_mismatch",
    )
    return manifest, events


def _load(evidence, *, numeric_dir, color_dir, class_dir):
    directories = [evidence.clean(Path(p)) for p in (numeric_dir, color_dir, class_dir)]
    numeric_dir, color_dir, class_dir = directories
    numeric, events = _stream(evidence, numeric_dir)
    _hash_provenance(numeric["provenance"], ("checkpoint_sha256", "graph_hash", "knowledge_pack_hash"))
    initial = _stellar(evidence.capture(numeric_dir / "capture"))
    star = initial["star_name"]
    selected = numeric["provenance"]["selected_class"]
    prefix = numeric["provenance"]["lifetime_prefix"]
    required = set(UNITS) if selected == "main_sequence" else {"distance", "luminosity", "temperature"}
    _require(
        selected in CLASSES
        and numeric.get("full_stellar_numeric_transport_verified") is True
        and numeric.get("outcome") == "full_stellar_numeric_transport_verified"
        and numeric.get("learned_policy") is True
        and numeric.get("checkpoint_unchanged") is True
        and type(numeric.get("optimizer_updates")) is int
        and numeric["optimizer_updates"] == 0
        and type(numeric.get("write_attempts")) is int
        and numeric["write_attempts"] == len(required)
        and isinstance(numeric.get("verified_fields"), list)
        and len(numeric["verified_fields"]) == len(required)
        and set(numeric["verified_fields"]) == required
        and set(numeric.get("numeric_readbacks", {})) == required,
        "incomplete_learned_numeric_receipt",
    )
    _require(
        (selected == "main_sequence" and prefix in {"ka", "Ma", "Ga", "Ta"})
        or (selected != "main_sequence" and prefix is None),
        "invalid_lifetime_unit",
    )
    _require(
        any(
            e.event == "action_proposed" and e.payload.get("action_source") == "frozen_lifetime_checkpoint"
            for e in events
        ),
        "missing_learned_numeric_proposals",
    )
    copies = [
        e.payload
        for e in events
        if e.event == "action_result" and e.payload.get("numeric_copy_verified") is True
    ]
    _require(len(copies) == len(required), "incomplete_numeric_copy_events")
    for name, receipt in numeric["numeric_readbacks"].items():
        unit = prefix if name == "lifetime" else UNITS[name]
        formatting = committed_display(receipt["exact_copied"], receipt["display_value"])
        _require(
            receipt["unit"] == unit
            and receipt.get("exact_input_verified") is True
            and receipt.get("commit_key") == "Tab"
            and all(_typed_equal(receipt.get(k), v) for k, v in formatting.items()),
            "invalid_numeric_copy_receipt",
        )
        matching = [p for p in copies if _typed_equal(p.get("numeric_readback"), receipt)]
        _require(len(matching) == 1, "numeric_copy_event_mismatch")
        copy = matching[0]
        field = copy["observation"]["values"]["browser_field_map"][name]
        _require(
            copy["observation"]["values"]["star_name"].casefold() == star.casefold()
            and copy["action"]["kind"] == "TYPE"
            and copy["action"]["value"] == receipt["exact_copied"]
            and field["current_value"] == receipt["display_value"]
            and field["unit"] == unit,
            "numeric_copy_readback_mismatch",
        )
    color, color_events = _stream(evidence, color_dir)
    _hash_provenance(color["provenance"], ("color_checkpoint_sha256", "color_reference_hash"))
    _require(
        any(
            e.event == "action_proposed" and e.payload.get("action_source") == "checkpoint"
            for e in color_events
        ),
        "missing_learned_color_proposal",
    )
    _require(
        color.get("color_transport_verified") is True
        and color.get("outcome") == "color_transport_verified"
        and type(color.get("write_attempts")) is int
        and color["write_attempts"] == 1
        and color["receipt"].get("readback_verified") is True
        and color["provenance"].get("color_gate_passed") is True
        and type(color["provenance"].get("optimizer_updates")) is int
        and color["provenance"]["optimizer_updates"] == 0,
        "incomplete_learned_color_receipt",
    )
    confirmed_colors = [
        e.payload
        for e in color_events
        if e.event == "action_result" and e.payload.get("color_transport_verified") is True
    ]
    _require(
        len(confirmed_colors) == 1 and _typed_equal(confirmed_colors[0]["receipt"], color["receipt"]),
        "color_event_mismatch",
    )
    bundle = {
        "star": star,
        "class": selected,
        "prefix": prefix,
        "readbacks": numeric["numeric_readbacks"],
        "color": color["receipt"]["selected_color"],
        "measurements": initial["observation"]["values"]["measurements"],
    }
    _values_match(_stellar(evidence.capture(color_dir / "capture")), bundle, color=False)
    _values_match(
        {
            "star_name": confirmed_colors[0]["observation"]["values"]["star_name"],
            "observation": confirmed_colors[0]["observation"],
        },
        bundle,
    )
    class_source = validate_star_class_source(evidence.history, class_dir, star, selected)
    _require(class_source["measurements"] == bundle["measurements"], "class_measurements_changed")
    for name, expected_hash in class_source["source_hashes"].items():
        path = evidence.history / name
        directories.append(evidence.clean(path.parent))
        _require(hashlib.sha256(evidence.read(path)).hexdigest() == expected_hash, "class_source_changed")
    bundle["class_rendering_sha256"] = class_source["class_rendering_sha256"]
    consumed = set(class_source["source_hashes"])
    for directory in (numeric_dir, color_dir):
        consumed.update(
            str((directory / name).relative_to(evidence.history))
            for name in ("manifest.json", "events.jsonl", "capture/manifest.json", "capture/observation.json")
        )
    return deepcopy(
        {
            "schema_version": 1,
            "mode": MODE,
            "bundle": bundle,
            "numeric_provenance": numeric["provenance"],
            "color_provenance": color["provenance"],
            "class_provenance": class_source,
            "source_sha256": {name: evidence.hashes[name] for name in sorted(consumed)},
            "validated_directories": sorted({str(p.relative_to(evidence.history)) for p in directories}),
            "browser_actions": 0,
            "current_class_paint_verified": False,
            "scientific_verified": False,
            "task_completed": False,
        }
    )


def load_stellar_sources(evidence, *, numeric_dir, color_dir, class_dir):
    """Adopt exact source leaves into an existing owned `_Evidence`-style book.

    The caller remains responsible for its book's final `unchanged()` check.
    Returned source hashes cover only this stellar prefix, not earlier book reads.
    """
    try:
        return _load(evidence, numeric_dir=numeric_dir, color_dir=color_dir, class_dir=class_dir)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, StellarMappingError) as exc:
        if isinstance(exc, BrowserSafetyStop):
            raise
        raise BrowserSafetyStop("stellar_sources_malformed_evidence") from None


def read_stellar_sources(run_history, *, numeric_dir, color_dir, class_dir, expected_star=None):
    """Read-only convenience wrapper; no browser, model, credentials or writes."""
    _require(
        expected_star is None or (isinstance(expected_star, str) and bool(expected_star.strip())),
        "invalid_expected_star",
    )
    try:
        evidence = _Evidence(run_history)
        result = load_stellar_sources(
            evidence, numeric_dir=numeric_dir, color_dir=color_dir, class_dir=class_dir
        )
        _require(
            expected_star is None or result["bundle"]["star"].casefold() == expected_star.casefold(),
            "star_mismatch",
        )
        evidence.unchanged()
        return result
    except (OSError, json.JSONDecodeError) as exc:
        raise BrowserSafetyStop("stellar_sources_malformed_evidence") from exc
