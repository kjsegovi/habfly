"""Explicit demo shortcut: ignore one supported feature, never prove absence.

This separately versioned policy retains the old baseline-band decisions. Its
only additional No assumption is one narrow, baseline-connected original-blue
feature in a complete 0..5000 overview. It neither measures an orbit/depth nor
turns arbitrary visual uncertainty into a No answer. Native exposure and source
freshness remain the caller's responsibility; this module only replays pixels.
"""

import io
import json
import math

import numpy as np
from PIL import Image

from .browser import BrowserSafetyStop
from .browser_observation_progress import trace_mask
from .planet_window_baseline_band import analyze_planet_window as band_analyze
from .planet_window_baseline_band import analyze_recorded_planet_window as prior_analyze
from .planet_window_baseline_band import policy_manifest as band_manifest
from .planet_window_baseline_band import recorded_policy_manifest as prior_manifest
from .planet_window_dip import _axis, _components, analyze_window_dip, detector_manifest
from .planet_window_policy import _digest

POLICY = {
    **band_manifest(),
    "version": "user_approved_5000_day_single_event_no_planet_v1",
    "base_policy_sha256": band_manifest()["sha256"],
    "scope": "User-approved demo assumption: retain baseline-band No, or ignore exactly one supported visible feature in a complete 0..5000 overview; possible real planet intentionally missed.",
    "single_event": {
        "positive_support_detector_sha256": detector_manifest()["sha256"],
        "axes": "Exactly exposed 0..5000 time range and existing linear 100-percent baseline; no extended observation",
        "coverage": "Original-blue baseline starts at requested first column, is contiguous to endpoint, with at most one dark neutral final border column",
        "feature": "Exactly one 4-connected original-blue component below existing1.5px baseline band, at least3rows and at most4columns, connected to baseline with3baseline columns each side",
        "other_hints": "Every non-neutral pixel below baseline+0.75 must lie in that same component's columns; no skipped secondary hint",
        "palette": "Unchanged positive detector validates original blue plus independently explained antialias; all remaining plot pixels must be dark neutral(max64)",
        "clipping": "No original trace at bottom row or above baseline tolerance; no clipped/edge component",
        "interpretation": "One visible candidate feature is deliberately ignored, not identified as noise or scientifically disproven",
        "evidence": "Original PNG/axis hashes, pixel bounds and support counts only; no period/depth or numerical answer",
    },
}
POLICY.pop("sha256")


def policy_manifest():
    return {**json.loads(json.dumps(POLICY)), "sha256": _digest(POLICY)}


def recorded_policy_manifest(policy=None):
    """Missing historical policy means frozen v2, never this new shortcut."""
    if policy is not None:
        known = policy_manifest()
        try:
            if policy == known and _digest(policy) == _digest(known):
                return known
        except (TypeError, ValueError):
            pass
    return prior_manifest(policy)


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _single_event_evidence(png, time_labels, flux_labels, base):
    _require(base.get("endpoint_visible") is True, "incomplete_rendered_window")
    times, slope, origin = _axis(time_labels, "center_x")
    fluxes, _, _ = _axis(flux_labels, "center_y")
    _require(times[0][0] == 0 and times[-1][0] == 5000, "requires_exact_5000_axis")
    # This exact failure means the unchanged positive detector passed its axis,
    # palette, overlay, clipping, baseline, connectivity and component-shape
    # checks, but found fewer than two components. Zero is rejected below.
    positive = analyze_window_dip(png, time_labels, flux_labels)
    _require(
        positive["reason"] == "insufficient_independent_dip_support",
        "unsupported_single_feature_geometry_or_palette",
    )
    left, right = math.ceil(origin - 0.5), math.floor(origin + 5000 * slope - 0.5)
    original = trace_mask(png)[20:152, left : right + 1]
    with Image.open(io.BytesIO(png)) as picture:
        pixels = np.asarray(picture.convert("RGB"), dtype=int)[20:152, left : right + 1]
    rows, baseline = np.arange(20, 152)[:, None] + 0.5, fluxes[-1][1]
    baseline_columns = (original & (np.abs(rows - baseline) <= 1.5)).any(axis=0)
    indices = np.flatnonzero(baseline_columns)
    _require(
        len(indices) >= 64
        and indices[0] == 0
        and baseline_columns[: indices[-1] + 1].all()
        and len(baseline_columns) - len(indices) <= 1,
        "incomplete_original_baseline",
    )
    if not baseline_columns[-1]:
        _require(
            pixels[:, -1].max() <= 64 and np.ptp(pixels[:, -1], axis=1).max() == 0,
            "unsupported_final_border",
        )
    components = _components(original & (rows > baseline + 1.5))
    _require(len(components) == 1, "requires_exactly_one_original_feature")
    ys, xs = zip(*components[0], strict=True)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    unknown = (np.ptp(pixels, axis=2) > 8) & ~original
    _require(
        not np.any((pixels.max(axis=2) > 64) & ~(original | unknown)),
        "bright_non_trace_plot_pixels",
    )
    extra_columns = (np.ptp(pixels, axis=2) > 0) & (rows > baseline + 0.75)
    extra_columns[:, x0 : x1 + 1] = False
    _require(not extra_columns.any(), "additional_colored_hint")
    return {
        "original_blue_feature_bounds": [int(x0 + left), int(y0 + 20), int(x1 + left), int(y1 + 20)],
        "original_blue_feature_pixels": len(components[0]),
        "baseline_y": baseline,
        "requested_columns": [left, right],
        "supported_baseline_columns": [left, left + int(indices[-1])],
        "baseline_columns": int(baseline_columns.sum()),
        "explained_antialias_pixels": positive["explained_antialias_pixels"],
        "positive_support_analysis_sha256": _digest(positive),
        "complete_original_baseline": True,
        "single_baseline_connected_feature": True,
        "additional_colored_hints": 0,
        "image_modified": False,
        "native_tooltip_confirmed": False,
        "physical_event_count_verified": False,
    }


def analyze_planet_window(png, time_labels, flux_labels, *, requested_days=5000):
    base = band_analyze(png, time_labels, flux_labels, requested_days=requested_days)
    report = {**base, "policy": policy_manifest(), "base_analysis_sha256": _digest(base)}
    if base.get("planet_decision") == "No":
        return report
    try:
        _require(type(requested_days) is int and requested_days == 5000, "requires_5000_days")
        evidence = _single_event_evidence(png, time_labels, flux_labels, base)
    except (
        BrowserSafetyStop,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        ZeroDivisionError,
        OSError,
    ) as error:
        # All detailed reason strings are local fixed guards, not driver text.
        reason = str(error) if type(error) is ValueError else "invalid_single_event_evidence"
        return {**report, "single_event_rejection": reason}
    return {
        **report,
        "status": "assume_no_planet",
        "reason": "user_approved_single_event_shortcut",
        "planet_decision": "No",
        "observation_limit_days": 5000,
        "visible_candidate_events": 1,
        "possible_planet_ignored": True,
        "approximation": "user_approved_single_event_no_planet_shortcut",
        "single_event_evidence": evidence,
        "absence_proven": False,
        "scientific_verified": False,
        "learned_perception": False,
        "training_label": False,
        "task_completed": False,
        "answer_writes": 0,
    }


def analyze_recorded_planet_window(png, time_labels, flux_labels, *, requested_days=5000, policy=None):
    known = recorded_policy_manifest(policy)
    if known == policy_manifest():
        return analyze_planet_window(png, time_labels, flux_labels, requested_days=requested_days)
    return prior_analyze(png, time_labels, flux_labels, requested_days=requested_days, policy=known)
