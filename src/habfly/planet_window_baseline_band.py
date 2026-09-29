"""Opt-in coarse-window No assumption, not proof that a planet is absent.

The user's 5,000-day overview rule explicitly permits missed subpixel features.
Supported blue shades inside the existing baseline tolerance therefore need not
trigger period measurement. Deeper/unsupported features remain unresolved. No
pixels are repainted and no measurement, training label or positive claim is made.
"""

import io
import json
import math
from fractions import Fraction

import numpy as np
from PIL import Image

from .browser import BrowserSafetyStop
from .browser_observation_progress import trace_mask
from .planet_window_baseline_edge import analyze_planet_window as edge_analyze
from .planet_window_baseline_edge import policy_manifest as edge_manifest
from .planet_window_policy import _axis, _digest
from .planet_window_policy import analyze_planet_window as frozen_analyze
from .planet_window_policy import policy_manifest as frozen_manifest

POLICY = {
    **frozen_manifest(),
    "version": "user_approved_5000_day_baseline_band_no_dip_v1",
    "base_policy_sha256": edge_manifest()["sha256"],
    "compatibility": {
        "entry": "baseline-edge unknown_plot_palette only; original PNG and axes unchanged",
        "coverage": "contiguous original-blue trace from requested left to endpoint; at most one dark neutral final border column; no internal gaps or cursor allowances",
        "baseline": "all original-blue pixels within existing 1.5px tolerance",
        "extra_pixels": "only below baseline within existing 1.5px tolerance, immediately supported by original-blue pixel above",
        "colors": "each RGB group independently feasible as blue (80,132,154) with neutral 0..64, alpha 3/50..1/2, closed half-channel rounding bins",
        "context": "three pixels below each extra pixel dark neutral: maximum 64, channel spread <=8",
        "overlays": "all remaining non-trace/non-extra plot pixels RGB maximum <=64",
        "interpretation": "subpixel features inside the baseline band are not resolved by this coarse overview; assumed No under the user-approved demo rule, not a claim that features are noise",
        "evidence": "original hashes, support rows, RGB groups, feasible alpha intervals; image never modified",
    },
}
POLICY.pop("sha256")


def policy_manifest():
    return {**json.loads(json.dumps(POLICY)), "sha256": _digest(POLICY)}


def recorded_policy_manifest(policy=None):
    """Historical absence still means frozen v2, never the newest policy."""
    if policy is None:
        return frozen_manifest()
    try:
        for known in (frozen_manifest(), edge_manifest(), policy_manifest()):
            if policy == known and _digest(policy) == _digest(known):
                return known
    except (ValueError, TypeError):
        pass
    raise BrowserSafetyStop("unsupported_planet_window_policy")


def _blend_interval(rgb):
    # Eliminate the shared gray term in the three closed rounding intervals.
    # This proves color-family membership only, not that a pixel is an artifact.
    lower, upper = [(0, 0)], [(-128, 128)]
    for value, blue in zip(rgb, (80, 132, 154), strict=True):
        lower.append((-2 * blue, 2 * int(value) - 1))
        upper.append((-2 * blue, 2 * int(value) + 1))
    start, end = Fraction(3, 50), Fraction(1, 2)
    for left_slope, left_intercept in lower:
        for right_slope, right_intercept in upper:
            coefficient = left_slope - right_slope
            bound = right_intercept - left_intercept
            if coefficient == 0:
                if bound < 0:
                    return None
            elif coefficient > 0:
                end = min(end, Fraction(bound, coefficient))
            else:
                start = max(start, Fraction(bound, coefficient))
    return None if start > end else (start, end)


def _band_evidence(png, times, fluxes, base):
    _, slope, origin = _axis(times, "center_x")
    flux, _, _ = _axis(fluxes, "center_y")
    baseline = flux[-1][1]
    left, right = math.ceil(origin - 0.5), math.floor(origin + 5000 * slope - 0.5)
    with Image.open(io.BytesIO(png)) as picture:
        pixels = np.asarray(picture.convert("RGB"), dtype=int)
    original = trace_mask(png)
    mask = original[20:152, left : right + 1]
    plot = pixels[20:152, left : right + 1]
    columns = mask.any(axis=0)
    indices = np.flatnonzero(columns)
    if (
        not base.get("endpoint_visible")
        or not len(indices)
        or indices[0] != 0
        or not columns[: indices[-1] + 1].all()
        or len(columns) - len(indices) > 1
    ):
        raise ValueError("original_trace_not_contiguous_and_complete")
    if not columns[-1] and (plot[:, -1].max() > 64 or np.ptp(plot[:, -1], axis=1).max() > 8):
        raise ValueError("unsupported_final_border_column")
    rows = np.arange(20, 152)[:, None] + 0.5
    if np.any(mask & (np.abs(rows - baseline) > 1.5)):
        raise ValueError("original_trace_outside_baseline_band")
    unknown = (np.ptp(plot, axis=2) > 8) & ~mask
    if np.any((plot.max(axis=2) > 64) & ~(mask | unknown)):
        raise ValueError("bright_non_trace_overlay")
    ys, xs = np.nonzero(unknown)
    if not len(xs):
        raise ValueError("missing_supported_band_pixels")
    absolute_y, absolute_x = ys + 20, xs + left
    if np.any((absolute_y + 0.5 <= baseline) | (absolute_y + 0.5 > baseline + 1.5)):
        raise ValueError("extra_pixels_outside_below_baseline_band")
    if not original[absolute_y - 1, absolute_x].all():
        raise ValueError("extra_pixels_missing_original_blue_support")
    for offset in (1, 2, 3):
        context = pixels[absolute_y + offset, absolute_x]
        if np.any(context.max(axis=1) > 64) or np.any(np.ptp(context, axis=1) > 8):
            raise ValueError("unsupported_below_band_context")
    colors, counts = np.unique(plot[unknown], axis=0, return_counts=True)
    groups = []
    for color, count in zip(colors, counts, strict=True):
        interval = _blend_interval(color)
        if interval is None:
            raise ValueError("unsupported_band_color")
        groups.append(
            {
                "rgb": color.tolist(),
                "pixels": int(count),
                "alpha_interval": [str(value) for value in interval],
            }
        )
    return {
        "baseline_y": baseline,
        "baseline_tolerance_pixels": 1.5,
        "requested_columns": [left, right],
        "supported_columns": [left, left + int(indices[-1])],
        "extra_pixel_rows": sorted(set(absolute_y.tolist())),
        "support_rows": sorted(set((absolute_y - 1).tolist())),
        "unknown_pixels": int(unknown.sum()),
        "groups": groups,
        "coverage_from_original_blue_only": True,
        "subpixel_features_not_resolved": True,
        "image_modified": False,
    }


def analyze_planet_window(png, time_labels, flux_labels, *, requested_days=5000):
    base = edge_analyze(png, time_labels, flux_labels, requested_days=requested_days)
    report = {**base, "policy": policy_manifest(), "base_analysis_sha256": _digest(base)}
    if base["reason"] != "unknown_plot_palette":
        return report
    try:
        evidence = _band_evidence(png, time_labels, flux_labels, base)
    except (ValueError, TypeError, KeyError, OSError, BrowserSafetyStop):
        return {**report, "baseline_band_rejection": "unsupported_baseline_band"}
    return {
        **report,
        "baseline_band_evidence": evidence,
        "status": "assume_no_planet",
        "reason": "user_approved_window_no_visible_dip",
        "planet_decision": "No",
        "observation_limit_days": 5000,
        "white_cursor_gap_columns": 0,
    }


def analyze_recorded_planet_window(png, time_labels, flux_labels, *, requested_days=5000, policy=None):
    known = recorded_policy_manifest(policy)
    if known == frozen_manifest():
        analyzer = frozen_analyze
    elif known == edge_manifest():
        analyzer = edge_analyze
    else:
        analyzer = analyze_planet_window
    return analyzer(png, time_labels, flux_labels, requested_days=requested_days)
