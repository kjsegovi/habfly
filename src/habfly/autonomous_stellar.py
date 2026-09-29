"""Offline reference-assisted stellar decisions, never learned classification.

This separate opt-in policy interprets the existing, unchanged diagnostic H-R
pack. Its v2 nearest-region fallback includes explicitly labelled log-axis
extrapolation: an uncalibrated best guess, not an exact course boundary or
grading label. It cannot authorize or perform browser writes.
The caller must revalidate the returned source hashes before using the payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_star_preflight import _blank_mapping, _Evidence
from .browser_stellar import StellarMappingError, map_stellar_capture
from .knowledge import LocalCalculator, load_knowledge_pack
from .lifetime_prefix import PREFIX_YEARS, lifetime_to_prefix
from .stellar_reference import (
    CLASSES,
    _region,
    classify_hr_reference,
    load_hr_reference,
    validate_hr_reference_source,
)

_HASH = re.compile(r"[0-9a-f]{64}")
_STAR = re.compile(r"[A-Za-z][A-Za-z0-9 '-]{0,79}")
_POLICY = {
    "version": 2,
    "id": "nearest_diagram_region_v2",
    "interior_rule": "unchanged_diagnostic_unique_interior",
    "fallback_rule": "unique_nearest_closed_polygon_at_true_log_projected_coordinates",
    "outside_plot_rule": "extrapolate_visible_log_axes_without_clamping_or_extending_polygons",
    "region_distance": "zero_inside_otherwise_euclidean_distance_to_digitized_edges",
    "numerical_tie_tolerance_pixels": 1e-9,
    "maximum_fallback_distance_pixels": None,
    "boundary_authority": "analyst_chosen_not_course_boundaries",
    "distance_calibrated": False,
    "extrapolation_calibrated": False,
    "limitations": [
        "Nearest digitized-region distance is a heuristic, not a probability or physical uncertainty.",
        "Outside-plot points extrapolate the visible log axes; no class region or course boundary is extended.",
        "Extrapolated choices are unsupported by visible diagram coverage and are not scientific verification.",
    ],
    "lifetime_prefix_rule": "largest_supported_factor_not_exceeding_years_else_smallest",
}
_CLAIMS = ("class-selection-reserved.json", "lifetime-prefix-reserved.json")


class AutonomousStellarError(ValueError):
    """A bounded reference decision was unavailable; no browser action occurred."""


def _require(condition, reason):
    if not condition:
        raise AutonomousStellarError("autonomous_stellar_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _valid_hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _finite_number(value):
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def _unreserved(fresh):
    _require(
        not any((fresh / name).exists() or (fresh / name).is_symlink() for name in _CLAIMS),
        "class_setup_already_reserved",
    )


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate_source_key")
            result[key] = value
        return result

    def reject(_):
        raise AutonomousStellarError("autonomous_stellar_nonfinite_source_json")

    result = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    _require(isinstance(result, dict), "invalid_source_object")
    return result


def _fresh_source(history, directory, receipt_sha, capture_sha, expected_star):
    evidence = _Evidence(history)
    fresh = evidence.clean(directory)
    _unreserved(fresh)
    raw_receipt = evidence.read(fresh / "confirmed.json")
    _require(_sha(raw_receipt) == receipt_sha, "fresh_receipt_hash_mismatch")
    receipt = _json(raw_receipt)
    stellar = evidence.clean(fresh / "stellar")
    manifest = _json(evidence.read(stellar / "manifest.json"))
    raw_capture = evidence.read(stellar / "observation.json")
    source_sha = _sha(raw_capture)
    _require(manifest.get("observation_sha256") == source_sha, "capture_manifest_hash_mismatch")
    _require(capture_sha is None or capture_sha == source_sha, "capture_expected_hash_mismatch")
    report = _json(raw_capture)
    _require(report.get("ignored_frame_urls") == [], "unknown_capture_frame")
    mapping = map_stellar_capture(
        report, capture_sha256=source_sha, allow_color_selection=True, allow_main_sequence_fields=True
    )
    star = mapping["star_name"]
    _require(
        isinstance(star, str)
        and _STAR.fullmatch(star)
        and isinstance(receipt.get("star"), str)
        and receipt["star"].casefold() == star.casefold()
        and (expected_star is None or expected_star.casefold() == star.casefold()),
        "star_mismatch",
    )
    _require(
        receipt.get("fresh_blank_numeric_answers_verified") is True
        and receipt.get("class_selection_verified") is False
        and type(receipt.get("answer_writes")) is int
        and receipt["answer_writes"] == 0
        and receipt.get("action_source") == "deterministic_navigation"
        and (receipt.get("painted_stellar_class") is None or receipt.get("painted_stellar_class") in CLASSES)
        and receipt.get("task_completed", False) is False,
        "unsupported_fresh_receipt",
    )
    values = mapping["observation"]["values"]
    measurements = values["measurements"]
    _blank_mapping(mapping, star, measurements, conditional=False)
    _require(
        all(field["enabled"] is True for field in values["browser_field_map"].values()), "disabled_fields"
    )
    return evidence, fresh, mapping, source_sha


def _calculate(calculator, operation, bindings, *, conditional=False):
    result = (
        calculator.execute(operation, bindings, "main_sequence")
        if conditional
        else calculator.execute_unclassified_common(operation, bindings)
    )
    _require(
        result.ok is True and _finite_number(result.value) and result.value > 0,
        "calculation_failed_" + operation,
    )
    return {"value": result.value, "unit": result.unit}


def _geometry(pack, diagnostic):
    point = diagnostic.get("diagram_pixel_coordinates")
    _require(
        isinstance(point, list) and len(point) == 2 and all(_finite_number(v) for v in point),
        "invalid_diagram_coordinates",
    )
    left, top, right, bottom = pack["axes"]["plot_bounds"]
    # Only distances to the rectangle use excess coordinates; the point passed
    # to every polygon remains the diagnostic's unmodified log projection.
    offsets = [max(left - point[0], 0.0, point[0] - right), max(top - point[1], 0.0, point[1] - bottom)]
    outside_distance = math.hypot(*offsets)
    _require(_finite_number(outside_distance), "invalid_diagram_distance")
    outside = outside_distance > 0
    _require(
        (diagnostic.get("reason") == "outside_reference_diagram") is outside
        and (not outside or diagnostic.get("status") == "abstained"),
        "reference_diagram_status_mismatch",
    )
    regions = {name: _region(point, polygon) for name, polygon in pack["regions"].items()}
    _require(
        all(
            type(inside) is bool and _finite_number(distance) and distance >= 0
            for inside, distance in regions.values()
        ),
        "invalid_region_distance",
    )
    inside = [name for name, (contained, _) in regions.items() if contained]
    _require(len(inside) <= 1, "overlapping_reference_regions")
    ranked = sorted((0.0 if contained else distance, name) for name, (contained, distance) in regions.items())
    nearest, runner_up = ranked[:2]
    gap = runner_up[0] - nearest[0]
    heuristic = diagnostic["status"] != "reference_candidate"
    if heuristic:
        _require(
            diagnostic.get("reason")
            in {
                "near_approximate_region_boundary",
                "outside_digitized_interiors",
                "outside_reference_diagram",
            },
            "unsupported_reference_abstention",
        )
        _require(gap > _POLICY["numerical_tie_tolerance_pixels"], "ambiguous_nearest_regions")
        selected = nearest[1]
    else:
        selected = diagnostic["selected_class"]
        _require(inside == [selected], "diagnostic_region_mismatch")
    return selected, {
        "diagram_pixel_coordinates": point,
        "outside_plot": outside,
        "extrapolated_axis_coordinates": outside,
        "outside_plot_distance_pixels": outside_distance,
        "outside_axis_distance_pixels": {"temperature": offsets[0], "luminosity": offsets[1]},
        "diagnostic_status": diagnostic["status"],
        "diagnostic_reason": diagnostic["reason"],
        "candidate_interiors": inside,
        "fallback_applied": heuristic,
        "nearest_region": nearest[1],
        "nearest_distance_pixels": nearest[0],
        "runner_up_region": runner_up[1],
        "runner_up_distance_pixels": runner_up[0],
        "distance_gap_pixels": gap,
        "regions": {
            name: {
                "contains_point": contained,
                "boundary_distance_pixels": distance,
                "region_distance_pixels": 0.0 if contained else distance,
            }
            for name, (contained, distance) in regions.items()
        },
        "distance_calibrated": False,
        "extrapolation_calibrated": False,
        "physical_uncertainty": None,
    }


def _prefix(years):
    _require(
        PREFIX_YEARS == {"ka": 1000, "Ma": 1000000, "Ga": 1000000000, "Ta": 1000000000000}
        and all(type(value) is int for value in PREFIX_YEARS.values()),
        "unsupported_prefix_constants",
    )
    quantity = Decimal(str(years))
    ordered = sorted(PREFIX_YEARS, key=PREFIX_YEARS.get)
    prefix = next((p for p in reversed(ordered) if quantity >= PREFIX_YEARS[p]), ordered[0])
    return prefix, lifetime_to_prefix(years, prefix)


def decide_stellar_reference(
    run_history,
    fresh_star_dir,
    *,
    fresh_star_sha256,
    source_root,
    expected_star=None,
    capture_sha256=None,
):
    """Return a source-bound explicit setup payload; perform no actions or writes.

    ``fresh_star_sha256`` pins confirmed.json. Because older fresh receipts do
    not themselves hash their capture, callers should additionally supply their
    existing ``capture_sha256`` pin. Returned ``source_sha256`` covers all three
    consumed owned files regardless; revalidate it before any later UI action.
    A result describes the captured star, not proof the current page is fresh.
    """
    _require(_valid_hash(fresh_star_sha256), "invalid_fresh_receipt_hash")
    _require(capture_sha256 is None or _valid_hash(capture_sha256), "invalid_capture_hash")
    _require(
        expected_star is None or (isinstance(expected_star, str) and _STAR.fullmatch(expected_star)),
        "invalid_expected_star",
    )
    try:
        history = Path(run_history).absolute()
        _require(history.is_dir() and history == history.resolve(), "invalid_history")
        directory = Path(fresh_star_dir)
        if not directory.is_absolute():
            directory = history / directory
        evidence, fresh, mapping, source_sha = _fresh_source(
            history, directory, fresh_star_sha256, capture_sha256, expected_star
        )
        pack, pack_sha = load_hr_reference()
        source_validation = validate_hr_reference_source(source_root)
        _require(source_validation["pack_sha256"] == pack_sha, "hr_pack_changed")
        knowledge = load_knowledge_pack()
        calculator = LocalCalculator(knowledge)
        golden = calculator.verify()
        inputs = deepcopy(mapping["observation"]["values"]["measurements"])
        known = {
            name: {"value": inputs["browser_" + name]["value"], "unit": inputs["browser_" + name]["unit"]}
            for name in ("parallax", "flux", "wavelength")
        }
        calculated = {
            "distance": _calculate(calculator, "distance", {"parallax": known["parallax"]}),
            "temperature": _calculate(calculator, "temperature", {"wavelength": known["wavelength"]}),
        }
        calculated["luminosity"] = _calculate(
            calculator, "luminosity", {"flux": known["flux"], "distance": calculated["distance"]}
        )
        diagnostic = classify_hr_reference(
            luminosity=calculated["luminosity"]["value"],
            luminosity_unit=calculated["luminosity"]["unit"],
            temperature=calculated["temperature"]["value"],
            temperature_unit=calculated["temperature"]["unit"],
            source_root=source_root,
        )
        _require(
            diagnostic["pack_sha256"] == pack_sha and diagnostic["source_image_verified"] is True,
            "hr_pack_changed",
        )
        selected, geometry = _geometry(pack, diagnostic)
        prefix, lifetime_quantity = None, None
        if selected == "main_sequence":
            calculated["mass"] = _calculate(
                calculator, "mass", {"luminosity": calculated["luminosity"]}, conditional=True
            )
            calculated["lifetime"] = _calculate(
                calculator, "lifetime", {"mass": calculated["mass"]}, conditional=True
            )
            _require(calculated["lifetime"]["unit"] == "yr", "unsupported_lifetime_unit")
            prefix, lifetime_quantity = _prefix(calculated["lifetime"]["value"])
        route = (
            "the uncalibrated nearest fixed digitized region at extrapolated log-axis coordinates outside the plot"
            if geometry["outside_plot"]
            else "the uncalibrated nearest digitized region for this diagram gap or boundary"
            if geometry["fallback_applied"]
            else "the unique interior in the coarse digitized visible H-R diagram"
        )
        rationale = (
            f"Visible parallax, flux and wavelength yield {calculated['luminosity']['value']:.8g} Lsun "
            f"and {calculated['temperature']['value']:.8g} K using the pinned local reference tool. "
            f"Select {selected.replace('_', ' ')} from {route}. "
            "This is an approximate reference-assisted decision, not a learned or scientifically verified class "
            "and not an exact course boundary or training label. "
            + (
                f"Select {prefix} using the tool-derived lifetime and existing exact unit conversion."
                if prefix is not None
                else "Main-sequence-only lifetime units are not applicable."
            )
        )
        evidence.unchanged()
        _unreserved(fresh)
        _require(load_hr_reference()[1] == pack_sha, "hr_pack_changed")
        _require(validate_hr_reference_source(source_root) == source_validation, "reference_source_changed")
        _require(load_knowledge_pack().checksum == knowledge.checksum, "knowledge_pack_changed")
        policy = {**deepcopy(_POLICY), "prefix_years": dict(PREFIX_YEARS)}
        return {
            "schema_version": 1,
            "decision_kind": "stellar_class",
            "status": "reference_decision",
            "mode": "approximate_reference_assisted_stellar_v2",
            "star": mapping["star_name"],
            "payload": {
                "selected_class": selected,
                "reference_rationale": rationale,
                "lifetime_prefix": prefix,
            },
            "inputs": inputs,
            "calculations": calculated,
            "lifetime_quantity_in_selected_prefix": lifetime_quantity,
            "geometry": geometry,
            "policy": policy,
            "policy_sha256": _sha(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()),
            "fresh_star_dir": str(fresh.relative_to(history)),
            "fresh_star_sha256": fresh_star_sha256,
            "capture_sha256": source_sha,
            "capture_expected_hash_supplied": capture_sha256 is not None,
            "source_sha256": dict(evidence.hashes),
            "provenance": {
                "hr_pack_sha256": pack_sha,
                "hr_source": deepcopy(pack["source"]),
                "hr_source_image_verified": True,
                "hr_pack_diagnostic_only_unchanged": True,
                "knowledge_pack_sha256": knowledge.checksum,
                "arithmetic_golden_validation": golden,
            },
            "approximate": True,
            "reference_assisted": True,
            "classification_learned": False,
            "learned_perception": False,
            "scientific_verified": False,
            "training_label": False,
            "course_correctness_verified": False,
            "write_authorized": False,
            "current_browser_state_verified": False,
            "task_completed": False,
            "project_completed": False,
            "browser_actions": 0,
        }
    except AutonomousStellarError:
        raise
    except (BrowserSafetyStop, StellarMappingError):
        raise AutonomousStellarError("autonomous_stellar_invalid_fresh_source") from None
    except (ValueError, OSError, TypeError, KeyError, OverflowError, RecursionError, RuntimeError):
        # Do not echo filesystem paths, source text, driver data or arbitrary JSON.
        raise AutonomousStellarError("autonomous_stellar_invalid_reference_or_source") from None
