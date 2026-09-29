"""Approximate fixed-window reference measurements, not daily chart semantics.

Only immutable student-visible raster/axis evidence is consumed. A repeating
pixel feature can suggest an observed spacing, not rule out missed, aliased or
subpixel events. The separate daily-tooltip validators remain unchanged. Nothing
in this module selects a planet answer, drives a browser or supplies a training
label; a caller must explicitly choose whether to use a reference estimate.
"""

import hashlib
import json
import math
import re
from itertools import pairwise
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_observation_progress import trace_progress
from .planet_window_dip import _axis, analyze_window_dip, detector_manifest

METHOD = {
    "version": "fixed_5000_day_approximate_raster_measurements_v1",
    "visible_time_axis_days": [0, 5000],
    "minimum_supported_components": 3,
    "minimum_clear_columns_between_components": 3,
    "center_uncertainty_pixels": 1,
    "deepest_trace_uncertainty_pixels": 1,
    "maximum_component_depth_spread_pixels": 2,
    "period": "common arithmetic progression of observed centers within +/-1px each; no inserted cycles",
    "depth": "deepest original supported pixel center minus exposed 100% tick, divided by linear flux scale",
    "pixel_coordinates": "centers at inclusive index+0.5; component center is midpoint of supported bounds",
    "uncertainty": "deterministic pixel bounds conditional on supplied linear axes, not a statistical confidence interval",
    "scope": "User-approved approximate reference fallback for the fixed5000day view; possible course-score loss",
    "limitations": [
        "scientific correctness is unverified",
        "raster endpoint is not complete daily coverage",
        "missing or subpixel events are not ruled out",
        "a regular visible spacing may be an alias of the physical period",
        "deepest painted trace pixel is not a resolved transit minimum",
        "pixel bounds do not cover unmodeled systematic chart or axis errors",
        "not learned perception and not a training label",
    ],
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def measurement_manifest():
    value = {**json.loads(json.dumps(METHOD)), "positive_detector_sha256": detector_manifest()["sha256"]}
    return {**value, "sha256": _digest(value)}


def _base():
    return {
        "schema_version": 1,
        "method": measurement_manifest(),
        "status": "measurement_error",
        "error": None,
        "source": "immutable_fixed_5000_day_chart_pixels_and_visible_axes",
        "provenance": "approximate_reference_measurement_not_learned",
        "approximate": True,
        "chart_sha256": None,
        "axis_sha256": None,
        "positive_evidence_sha256": None,
        "examined_day_interval": [0, 5000],
        "period_days": None,
        "brightness_drop_percent": None,
        "planet_decision": None,
        "absence_proven": False,
        "scientific_verified": False,
        "learned_perception": False,
        "training_label": False,
        "task_completed": False,
        "observation_completed": False,
        "daily_coverage_verified": False,
        "missing_events_ruled_out": False,
        "aliasing_ruled_out": False,
        "browser_actions": 0,
        "answer_writes": 0,
    }


def _error(report, code, stage):
    # Never return partial numeric answers for a rejected measurement set.
    return {
        **report,
        "status": "measurement_error",
        "error": {"code": code, "stage": stage},
        "period_days": None,
        "brightness_drop_percent": None,
    }


def _hash_valid(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _spacing(centers):
    """Feasible slope/intercept bounds for centers with +/-1px uncertainty.

    Every observed component gets one consecutive index: gaps are not repaired
    with guessed events or harmonics. Pairwise constraints also require a single
    phase across the whole window, not merely similar adjacent differences.
    """
    uncertainty = METHOD["center_uncertainty_pixels"]
    lower, upper = 0.0, math.inf
    for i, first in enumerate(centers):
        for j in range(i + 1, len(centers)):
            delta = centers[j] - first
            lower = max(lower, (delta - 2 * uncertainty) / (j - i))
            upper = min(upper, (delta + 2 * uncertainty) / (j - i))
    if lower <= 0 or not math.isfinite(upper) or lower > upper + 1e-12:
        raise ValueError("inconsistent_observed_component_spacing")
    spacing = (lower + upper) / 2
    phase_low = max(center - uncertainty - i * spacing for i, center in enumerate(centers))
    phase_high = min(center + uncertainty - i * spacing for i, center in enumerate(centers))
    if phase_low > phase_high + 1e-12:
        raise ValueError("inconsistent_observed_component_spacing")
    phase = (phase_low + phase_high) / 2
    residuals = [center - (phase + i * spacing) for i, center in enumerate(centers)]
    return spacing, lower, upper, phase, residuals


def estimate_planet_window_measurements(
    png, time_labels, flux_labels, *, expected_chart_sha256, expected_axis_sha256
):
    """Replay hash-pinned pixels into explicitly approximate reference values.

    Both hashes are mandatory. Axis identity uses canonical JSON of
    {"time": time_labels, "flux": flux_labels}, matching the positive detector.
    The file loader below obtains these inputs from a hash-pinned public capture
    report. Insufficient support yields structured errors, never fabricated zeros.
    """
    result = _base()
    if not isinstance(png, bytes) or not 0 < len(png) <= 1024 * 1024:
        return _error(result, "invalid_chart_bytes", "identity")
    result["chart_sha256"] = hashlib.sha256(png).hexdigest()
    try:
        result["axis_sha256"] = _digest({"time": time_labels, "flux": flux_labels})
    except (ValueError, TypeError, OverflowError):
        return _error(result, "invalid_axis_identity", "identity")
    if not _hash_valid(expected_chart_sha256) or result["chart_sha256"] != expected_chart_sha256:
        return _error(result, "chart_hash_mismatch", "identity")
    if not _hash_valid(expected_axis_sha256) or result["axis_sha256"] != expected_axis_sha256:
        return _error(result, "axis_hash_mismatch", "identity")
    positive = analyze_window_dip(png, time_labels, flux_labels)
    result["positive_evidence_sha256"] = _digest(positive)
    result["positive_evidence"] = positive
    if positive["status"] != "dip_observed":
        return _error(result, positive["reason"], "positive_geometry")
    try:
        times, time_scale, time_origin = _axis(time_labels, "center_x")
        fluxes, flux_scale, _ = _axis(flux_labels, "center_y")
        if times[0][0] != 0 or times[-1][0] != 5000:
            return _error(result, "full_0_to_5000_axis_required", "axes")
        readiness = trace_progress(png, time_labels, requested_days=5000)
    except (BrowserSafetyStop, ValueError, TypeError, KeyError, ZeroDivisionError):
        return _error(result, "invalid_or_unsupported_axis", "axes")
    result["raster_endpoint_verified"] = readiness["endpoint_visible"]
    if not readiness["endpoint_visible"]:
        return _error(result, "full_window_raster_endpoint_required", "coverage")
    bounds = positive["support_pixel_bounds"]
    if len(bounds) < METHOD["minimum_supported_components"]:
        return _error(result, "at_least_three_supported_dips_required", "measurements")
    if any(
        right[0] - left[2] - 1 < METHOD["minimum_clear_columns_between_components"]
        for left, right in pairwise(bounds)
    ):
        return _error(result, "dip_components_not_well_separated", "measurements")
    bottoms = [bound[3] + 0.5 for bound in bounds]
    if max(bottoms) - min(bottoms) > METHOD["maximum_component_depth_spread_pixels"]:
        return _error(result, "inconsistent_observed_component_depths", "measurements")
    centers = [(bound[0] + bound[2]) / 2 + 0.5 for bound in bounds]
    try:
        spacing, lower, upper, phase, residuals = _spacing(centers)
    except ValueError as exc:
        return _error(result, str(exc), "measurements")
    baseline = fluxes[-1][1]
    deepest = max(bottoms)
    depth_pixels = deepest - baseline
    depth_error = METHOD["deepest_trace_uncertainty_pixels"]
    drop = depth_pixels / -flux_scale
    drop_low, drop_high = (
        (depth_pixels - depth_error) / -flux_scale,
        (depth_pixels + depth_error) / -flux_scale,
    )
    if not 0 < drop_low <= drop <= drop_high < 100:
        return _error(result, "depth_interval_outside_valid_flux_domain", "measurements")
    result.update(
        status="approximate_reference_measurements",
        supported_dip_components=len(bounds),
        support_pixel_bounds=bounds,
        component_centers_pixels=centers,
        component_deepest_pixels=bottoms,
        baseline_pixel_y=baseline,
        deepest_pixel_y=deepest,
        time_pixels_per_day=time_scale,
        time_axis_origin_pixel=time_origin,
        flux_pixels_per_percent=abs(flux_scale),
        period_days={
            "value": spacing / time_scale,
            "lower": lower / time_scale,
            "upper": upper / time_scale,
            "unit": "days",
            "center_uncertainty_pixels": METHOD["center_uncertainty_pixels"],
            "observed_spacing_pixels": spacing,
            "fitted_phase_pixel": phase,
            "maximum_center_residual_pixels": max(map(abs, residuals)),
            "interpretation": "observed raster recurrence only; physical period may be aliased",
        },
        brightness_drop_percent={
            "value": drop,
            "lower": drop_low,
            "upper": drop_high,
            "unit": "percent",
            "deepest_trace_uncertainty_pixels": depth_error,
            "pixel_uncertainty_percent": depth_error / abs(flux_scale),
            "interpretation": "deepest supported painted pixel, not an exact transit minimum",
        },
    )
    return result


def load_planet_window_measurements(report_path, *, expected_report_sha256):
    """Read a pinned public progress report and its sibling chart.png offline.

    The capture report binds both visible axis arrays and the PNG hash. Its
    asserted readiness is not trusted: positive geometry and endpoint checks
    are independently replayed. Files and existing report contents are untouched.
    """
    result = _base()
    try:
        path = Path(report_path)
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if not _hash_valid(expected_report_sha256) or actual != expected_report_sha256:
            return _error(result, "source_report_hash_mismatch", "identity")
        report = json.loads(raw)
        if (
            not isinstance(report, dict)
            or report.get("method") != "rendered_blue_trace_frontier_v1"
            or report.get("source") != "chart_crop_pixels_and_visible_axis_labels"
            or type(report.get("requested_days")) is not int
            or report["requested_days"] != 5000
            or any(
                type(report.get(key)) is not int or report[key] != 0
                for key in ("browser_actions", "answer_writes")
            )
            or not isinstance(report.get("star"), str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", report["star"])
        ):
            return _error(result, "unsupported_source_capture_report", "identity")
        time_labels, flux_labels = report["time_axis_labels"], report["flux_axis_labels"]
        result = estimate_planet_window_measurements(
            (path.parent / "chart.png").read_bytes(),
            time_labels,
            flux_labels,
            expected_chart_sha256=report["chart_sha256"],
            expected_axis_sha256=_digest({"time": time_labels, "flux": flux_labels}),
        )
        return {**result, "star": report["star"], "source_report_sha256": actual}
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return _error(result, "unreadable_source_capture_report", "identity")
