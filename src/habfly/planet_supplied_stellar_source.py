"""Offline, non-main public M/R provenance; no browser or write authority.

The initial guarded navigation links a historical supplied class to the first
Planet readouts. The existing presence/raw-copy validators preserve those exact
readouts through their own native transitions. Matching the navigation readouts
to the later presence capture is NOT continuous class-paint verification, an
atomic server snapshot, or proof of physical correctness/default origin.
"""

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_no_planet_workflow import _Evidence, _planet, _stellar, _values_match
from .browser_numeric import screen_identity
from .browser_planet_numeric import planet_projection
from .browser_positive_planet_workflow import _copy_chain, _RasterEvidence
from .browser_project_navigation import (
    _outside,
    _required,
    navigation_status_projection,
    project_view,
)
from .browser_raster_planet_evidence import (
    RAW,
    _action_source,
    _flags_for,
    _fresh_evidence,
    _presence,
    _reload,
    _reservation,
)
from .browser_stellar import StellarMappingError
from .browser_stellar_sources import load_stellar_sources
from .planet_supplied_inputs import SCOPE, adapter_manifest, load_supplied_planet_pack

MODE = "supplied_planet_stellar_inputs_v1"
CLASSES = {"white_dwarf", "red_giant", "supergiant"}
FLAGS = {
    "browser_actions": 0,
    "answer_writes": 0,
    "native_browser_enabled": False,
    "write_authorized": False,
    "current_class_paint_verified": False,
    "continuous_class_identity_verified": False,
    "quantity_origin_or_default_behavior_verified": False,
    "scientific_verified": False,
    "learned_measurements": False,
    "training_label": False,
    "task_completed": False,
    "project_completed": False,
}


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("supplied_stellar_source_" + reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _equal(left, right):
    return _canonical(left) == _canonical(right)


def _hash(value):
    _require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "invalid_source_hash")
    return value


def _star(value):
    _require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", value),
        "invalid_star",
    )
    return value.casefold()


def _inputs(mapping, star):
    _require(_star(mapping["star_name"]) == _star(star), "star_changed")
    values = mapping["observation"]["values"]["stellar_inputs"]
    _require(set(values) == {"stellar_mass", "stellar_radius"}, "missing_supplied_inputs")
    for name, unit in (("stellar_mass", "Msun"), ("stellar_radius", "Rsun")):
        item = values[name]
        value, text = item["value"], item["display_text"]
        _require(
            type(value) in {int, float}
            and math.isfinite(value)
            and value > 0
            and isinstance(text, str)
            and 0 < len(text) <= 80
            and math.isfinite(float(text))
            and float(text) == value
            and item["unit"] == unit
            and isinstance(item.get("source"), str),
            "invalid_supplied_input",
        )
    return deepcopy(values)


def _navigation(book, directory, bundle):
    directory = book.clean(directory)
    intent, receipt = book.json(directory / "reserved.json"), book.json(directory / "confirmed.json")
    before, preclick, after = [book.capture(directory / n) for n in ("before", "pre-click", "after")]
    start, end = project_view(before), project_view(after)
    _require(
        start["surface"] == end["surface"] == "detail"
        and start["section"] == "stellar"
        and end["section"] == "planet"
        and _star(start["star"]) == _star(end["star"]) == _star(bundle["star"]),
        "navigation_star_or_destination",
    )
    _values_match(_stellar(before), bundle)
    _values_match(_stellar(preclick), bundle)
    target = intent["target_evidence"]
    box = target["box"]
    _require(
        set(target) == {"number", "box", "accessibility"}
        and type(target["number"]) is int
        and target["number"] == 2
        and isinstance(target["accessibility"], str)
        and 0 < len(target["accessibility"]) <= 2000
        and set(box) == {"x", "y", "width", "height"}
        and all(type(v) in {int, float} and math.isfinite(v) for v in box.values())
        and box["width"] > 0
        and box["height"] > 0,
        "invalid_navigation_target",
    )
    expected = {
        "mode": "bounded_project_navigation",
        "destination": "planet",
        "from": start,
        "expected_star": intent["expected_star"],
        "target_evidence": target,
        "max_clicks": 1,
        "answer_writes": 0,
        "collection_clicks": 0,
        "save_clicks": 0,
        "assessment_clicks": 0,
        "submission_clicks": 0,
        "automatic_retry": False,
        "task_completed": False,
        "action_source": "deterministic_navigation",
    }
    _require(
        _star(intent["expected_star"]) == _star(bundle["star"])
        and _equal(intent, expected)
        and _equal(
            receipt,
            {
                **expected,
                "to": end,
                "navigation_clicks": 1,
                "destination_verified": True,
                "same_star_verified": True,
                "suggested_required_text": _required(after, end),
                "config_mutated": False,
            },
        )
        and _equal(project_view(preclick), start)
        and screen_identity(navigation_status_projection(before, start))
        == screen_identity(navigation_status_projection(preclick, start))
        and _equal(_outside(before), _outside(after)),
        "invalid_navigation_chain",
    )
    mapping = _planet(after)
    fields = mapping["observation"]["values"]["browser_field_map"]
    _require(
        mapping["observation"]["values"]["has_planet"] is None
        and set(fields) == {"observation_days", *RAW}
        and all(item["current_value"] == "" and item["enabled"] is True for item in fields.values()),
        "navigation_planet_not_fresh",
    )
    return _inputs(mapping, bundle["star"])


def _raw_sources(book, directory):
    """Reuse native/source validators, without the old main-only derived proof.

    This is the raw prefix of the positive-workflow contract. It intentionally
    never accepts an arbitrary caller-supplied list of hashes as a copy proof.
    """
    directory = book.clean(directory)
    owner = _RasterEvidence(book)
    raw, intent = book.json(directory / "report.json"), book.json(directory / "reserved.json")
    presence, initial, _ = _presence(owner, book.history / raw["presence_path"], raw["presence_sha256"])
    evidence, star = presence["evidence"], presence["star"]
    flags = _flags_for(evidence)
    mode = flags["provenance"]
    _require(
        all(_equal(raw.get(k), v) for k, v in flags.items())
        and type(raw.get("schema_version")) is int
        and raw["schema_version"] == 1
        and raw.get("mode") == mode
        and raw.get("stage") == "inputs"
        and _star(raw.get("star")) == _star(star)
        and _equal(raw.get("evidence"), evidence)
        and raw.get("output") == owner.relative(directory)
        and type(raw.get("maximum_numeric_writes")) is int
        and raw["maximum_numeric_writes"] == 3
        and raw.get("destinations") == list(RAW)
        and raw.get("action_source") == _action_source(evidence),
        "invalid_raw_provenance",
    )
    _require(
        _equal(book.json(_reservation(owner, star, "inputs")), intent)
        and _equal(
            raw,
            {
                **intent,
                "raw_measurement_transport_verified": True,
                "numeric_writes": 3,
                "verified_fields": raw["verified_fields"],
                "precopy": raw["precopy"],
                "native_events": raw["native_events"],
                "saved": False,
                "assessed": False,
                "submitted": False,
            },
        ),
        "invalid_raw_reservation",
    )
    _reload(owner, evidence)
    _require(isinstance(raw["precopy"], list) and len(raw["precopy"]) == 3, "invalid_precopy_count")
    for i, name in enumerate(("fresh", "precopy-1", "precopy-2", "precopy-3")):
        recorded = raw["fresh"] if i == 0 else raw["precopy"][i - 1]
        if i:
            _require(_equal(book.json(directory / name / "evidence.json"), recorded), "precopy_changed")
        _fresh_evidence(owner, directory / name, recorded, evidence)
    receipts, last = _copy_chain(
        book,
        directory / "native-copies",
        set(RAW),
        source="reference_diagnostic",
        initial=initial,
        expected=evidence["measurements"],
    )
    _require(_equal(receipts, raw["verified_fields"]), "raw_readbacks_changed")
    events = []
    for name in RAW:
        receipt = receipts[name]
        events.extend(
            {"kind": kind, "payload": payload, "provenance": mode}
            for kind, payload in (
                (
                    "action_proposed",
                    {
                        k: v
                        for k, v in receipt.items()
                        if k not in {"display", "readback_verified", "correctness_verified"}
                    },
                ),
                ("action_result", receipt),
            )
        )
    _require(
        _equal(raw["native_events"], events)
        and _equal([book.json(p) for p in sorted(directory.glob("native-event-*.json"))], events),
        "raw_native_events_changed",
    )
    return presence, initial, last, mode


def _build(
    book,
    *,
    numeric_dir,
    color_dir,
    class_dir,
    navigation_dir,
    raw_dir,
    current_capture_dir,
    current_capture_sha256,
    expected_star,
    selected_class,
    expected_pack_hash,
    expected_adapter_sha256,
):
    _star(expected_star)
    _require(isinstance(selected_class, str) and selected_class in CLASSES, "unsupported_actual_class")
    pack, adapter = load_supplied_planet_pack(), adapter_manifest()
    _require(pack.checksum == _hash(expected_pack_hash), "pack_changed")
    _require(adapter["sha256"] == _hash(expected_adapter_sha256), "adapter_changed")
    stellar = load_stellar_sources(book, numeric_dir=numeric_dir, color_dir=color_dir, class_dir=class_dir)
    bundle = stellar["bundle"]
    _require(
        _star(bundle["star"]) == _star(expected_star)
        and bundle["class"] == selected_class
        and bundle["prefix"] is None
        and set(bundle["readbacks"]) == {"distance", "luminosity", "temperature"},
        "stellar_source_class_or_star_changed",
    )
    initial_inputs = _navigation(book, navigation_dir, bundle)
    presence, initial, last, mode = _raw_sources(book, raw_dir)
    _require(
        _star(presence["star"]) == _star(expected_star)
        and _equal(initial_inputs, _inputs(_planet(initial), expected_star))
        and _equal(initial_inputs, _inputs(_planet(last), expected_star)),
        "initial_supplied_inputs_changed",
    )
    current_capture_dir = book.clean(current_capture_dir)
    current = book.capture(current_capture_dir)
    capture_key = str((current_capture_dir / "observation.json").relative_to(book.history))
    _require(book.hashes[capture_key] == _hash(current_capture_sha256), "current_capture_changed")
    mapping = _planet(current)
    _require(
        _equal(initial_inputs, _inputs(mapping, expected_star))
        and _equal(planet_projection(current, mapping), planet_projection(last, _planet(last))),
        "current_planet_projection_changed",
    )
    # Close per-star sources, not append-only shared claim registries. A later
    # star may add its own reservation, but the exact claims consumed here and
    # registry failure markers remain pinned/rechecked by the evidence book.
    directories = set(book.clean_directories) | {book.path(book.history / p).parent for p in book.hashes}
    shared = {book.history} | {
        book.history / f"planet-raster-{stage}-reservations" for stage in ("presence", "inputs")
    }
    trees = {}
    for directory in directories:
        book.clean(directory)
        if directory in shared:
            continue
        tree = book.tree(directory)
        book.closed_trees[directory] = tree
        trees[str(directory.relative_to(book.history))] = [list(item) for item in tree]
    book.unchanged()
    source_args = {
        name: str(book.path(path).relative_to(book.history))
        for name, path in {
            "numeric_dir": numeric_dir,
            "color_dir": color_dir,
            "class_dir": class_dir,
            "navigation_dir": navigation_dir,
            "raw_dir": raw_dir,
            "current_capture_dir": current_capture_dir,
        }.items()
    }
    return {
        "schema_version": 1,
        "mode": MODE,
        "scope": SCOPE,
        **FLAGS,
        "star": bundle["star"],
        "actual_class": selected_class,
        "inputs": initial_inputs,
        "stellar_source": stellar,
        "measurement_mode": mode,
        "current_capture_sha256": current_capture_sha256,
        "initial_navigation_capture_sha256": book.hashes[
            str((book.path(navigation_dir) / "after/observation.json").relative_to(book.history))
        ],
        "pack_sha256": pack.checksum,
        "adapter": adapter,
        "assumptions": list(pack.assumptions),
        "source_arguments": source_args,
        "source_sha256": dict(sorted(book.hashes.items())),
        "validated_directories": sorted(str(p.relative_to(book.history)) for p in book.clean_directories),
        "directory_trees": dict(sorted(trees.items())),
        "limitations": [
            "Visible reconstruction readouts; their physical correctness and default origin are unverified.",
            "Class is historical supplied provenance, not a current Planet-tab class-paint measurement.",
            "Initial and later visible star/M/R match; no claim of continuous hidden state or atomic snapshot.",
            "Offline source receipt only: not learned stellar M/R, browser authorization, or task completion.",
        ],
    }


def build_supplied_planet_stellar_inputs(run_history, **kwargs):
    """Validate owned immutable sources and return a receipt; write nothing.

    Required kwargs are the six source directories listed in `_build`, current
    observation SHA, expected star/actual non-main class, and installed pack and
    adapter SHA. Relative source directories resolve under run_history.
    """
    try:
        book = _Evidence(run_history)
        arguments = dict(kwargs)
        for name in (
            "numeric_dir",
            "color_dir",
            "class_dir",
            "navigation_dir",
            "raw_dir",
            "current_capture_dir",
        ):
            path = Path(arguments[name])
            arguments[name] = path if path.is_absolute() else book.history / path
        return _build(book, **arguments)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, StellarMappingError):
        raise BrowserSafetyStop("supplied_stellar_source_malformed_evidence") from None


def load_supplied_planet_stellar_inputs(
    run_history,
    receipt_path,
    *,
    expected_sha256,
    expected_star,
    selected_class,
    expected_pack_hash,
    expected_adapter_sha256,
):
    """Reload all sources and compare exact typed receipt content; no replay writes."""
    try:
        book = _Evidence(run_history)
        path = Path(receipt_path)
        path = book.path(path if path.is_absolute() else book.history / path)
        raw = book.read(path)
        _require(hashlib.sha256(raw).hexdigest() == _hash(expected_sha256), "receipt_hash_changed")
        saved = json.loads(raw)
        _require(
            set(saved["source_arguments"])
            == {
                "numeric_dir",
                "color_dir",
                "class_dir",
                "navigation_dir",
                "raw_dir",
                "current_capture_dir",
            },
            "invalid_source_arguments",
        )
        rebuilt = build_supplied_planet_stellar_inputs(
            run_history,
            **saved["source_arguments"],
            current_capture_sha256=saved["current_capture_sha256"],
            expected_star=expected_star,
            selected_class=selected_class,
            expected_pack_hash=expected_pack_hash,
            expected_adapter_sha256=expected_adapter_sha256,
        )
        _require(_equal(saved, rebuilt), "receipt_or_sources_changed")
        book.unchanged()
        return rebuilt
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError):
        raise BrowserSafetyStop("supplied_stellar_source_malformed_receipt") from None


def matches_mapping(receipt, mapping):
    """Compare supplied readouts to a caller-guarded current native mapping.

    This does not reload provenance or read a browser; use the strict loader
    first. It neither verifies class paint on Planet nor authorizes any copy.
    """
    try:
        return (
            type(receipt["schema_version"]) is int
            and receipt["schema_version"] == 1
            and receipt["mode"] == MODE
            and receipt["scope"] == SCOPE
            and receipt["actual_class"] in CLASSES
            and all(_equal(receipt.get(key), value) for key, value in FLAGS.items())
            and _equal(receipt["inputs"], _inputs(mapping, receipt["star"]))
        )
    except (BrowserSafetyStop, ValueError, TypeError, KeyError, AttributeError):
        return False
