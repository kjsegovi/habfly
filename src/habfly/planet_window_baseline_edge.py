"""Opt-in raster compatibility for one continuous, supported baseline edge.

The frozen v2 detector is never changed or fed repainted pixels. Its ordinary
outcomes remain authoritative. Only its unknown-palette outcome may additionally
pass the stricter original-pixel checks here. This remains the user's approximate
5,000-day No assumption, not scientific absence or a training label.
"""

import io
import json
import math
from fractions import Fraction

import numpy as np
from PIL import Image

from .browser import BrowserSafetyStop
from .browser_observation_progress import trace_mask
from .planet_window_policy import _axis, _digest
from .planet_window_policy import analyze_planet_window as frozen_analyze
from .planet_window_policy import policy_manifest as frozen_manifest

POLICY = {
    **frozen_manifest(),
    "version": "user_approved_5000_day_baseline_edge_no_dip_v1",
    "base_policy_sha256": frozen_manifest()["sha256"],
    "compatibility": {
        "entry": "frozen unknown_plot_palette only; original PNG and axes unchanged",
        "coverage": "original-blue trace starts at requested left edge and is contiguous to endpoint; at most one dark neutral final border column within frozen endpoint tolerance; no internal gaps or cursor allowances",
        "edge": "exactly one full unknown row below baseline within existing 1.5px tolerance",
        "support": "uniform original-blue row immediately above every edge pixel",
        "other_pixels": "all original-blue pixels within baseline tolerance; no other unknown chromatic pixels",
        "overlays": "all remaining non-trace/non-edge pixels in requested plot must have RGB maximum <=64",
        "grid": "edge variants only at exposed time-tick columns with three equal neutral pixels below",
        "background": "uniform off-grid three-row neutral <=64 and uniform exposed same-row x262..279 neutral <=64",
        "rounding": "one shared alpha across all RGB colors, independent neutral per group, closed half-channel bins",
        "blue": [80, 132, 154],
        "alpha_interval": ["3/50", "1/4"],
        "neutral_envelopes": "off-grid uses below-row/margin interval; grid uses its exact visible below-row gray",
        "evidence": "record edge/support rows, RGB groups, neutral samples, feasible alpha and original hashes",
    },
}
POLICY.pop("sha256")


def policy_manifest():
    # Round-trip creates a fresh nested object just like the original manifest.
    return {**json.loads(json.dumps(POLICY)), "sha256": _digest(POLICY)}


def recorded_policy_manifest(policy=None):
    """Missing historical identity means v2; partial/unknown identities reject."""
    if policy is None:
        return frozen_manifest()
    try:
        for known in (frozen_manifest(), policy_manifest()):
            if policy == known and _digest(policy) == _digest(known):
                return known
    except (ValueError, TypeError):
        pass
    raise BrowserSafetyStop("unsupported_planet_window_policy")


def _alpha_interval(rgb, low, high):
    """Exact common RGB blend feasibility; b = 2*(1-alpha)*gray."""
    lower = [(-2 * low, 2 * low)]
    upper = [(-2 * high, 2 * high)]
    for observed, direction in zip(rgb, (80, 132, 154), strict=True):
        lower.append((-2 * direction, 2 * observed - 1))
        upper.append((-2 * direction, 2 * observed + 1))
    start, end = Fraction(3, 50), Fraction(1, 4)
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


def _neutral(pixels):
    if not pixels.size or np.any(pixels != pixels.reshape(-1, 3)[0]):
        raise ValueError("nonuniform_neutral_sample")
    rgb = pixels.reshape(-1, 3)[0]
    if rgb.min() != rgb.max() or rgb[0] > 64:
        raise ValueError("unsupported_neutral_sample")
    return int(rgb[0])


def _edge_evidence(png, times, fluxes, base):
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
    covered_right = left + int(indices[-1])
    if not columns[-1] and (plot[:, -1].max() > 64 or np.ptp(plot[:, -1], axis=1).max() > 8):
        raise ValueError("unsupported_final_border_column")
    positions = np.arange(20, 152)[:, None] + 0.5
    if np.any(mask & (np.abs(positions - baseline) > 1.5)) or mask[-1].any():
        raise ValueError("original_trace_outside_baseline_band")
    unknown = (np.ptp(plot, axis=2) > 8) & ~mask
    if np.any((plot.max(axis=2) > 64) & ~(mask | unknown)):
        raise ValueError("bright_non_trace_overlay")
    row_indices = np.flatnonzero(unknown.any(axis=1))
    if len(row_indices) != 1 or not np.array_equal(unknown[row_indices[0]], columns):
        raise ValueError("unknown_pixels_not_one_full_edge_row")
    row = int(row_indices[0]) + 20
    if not 0 < row + 0.5 - baseline <= 1.5 or row <= 20:
        raise ValueError("edge_outside_below_baseline_tolerance")
    requested_right, right = right, covered_right
    plot = plot[:, : len(indices)]
    if not original[row - 1, left : right + 1].all():
        raise ValueError("edge_missing_original_blue_support")
    support = pixels[row - 1, left : right + 1]
    if np.any(support != support[0]):
        raise ValueError("nonuniform_original_blue_support")
    # The margin is outside the plotted area and never counts as trace coverage.
    margin = _neutral(pixels[row, 262:280])
    tick_columns = {round(float(tick["center_x"]) - 0.5) for tick in times}
    grid = np.array([column in tick_columns for column in range(left, right + 1)])
    if not grid.any() or int((~grid).sum()) < 64:
        raise ValueError("insufficient_separate_grid_and_background_samples")
    off_gray = _neutral(pixels[row + 1 : row + 4, left : right + 1][:, ~grid])
    off_color = plot[row - 20, ~grid]
    if np.any(off_color != off_color[0]):
        raise ValueError("nonuniform_off_grid_edge")
    grid_colors = plot[row - 20, grid]
    if np.any(grid_colors != grid_colors[0]):
        raise ValueError("nonuniform_grid_edge")
    grid_gray = _neutral(pixels[row + 1 : row + 4, left : right + 1][:, grid])
    if grid_gray <= max(off_gray, margin):
        raise ValueError("grid_not_supported_by_visible_neutral_contrast")
    groups = [
        {
            "kind": "off_grid",
            "rgb": off_color[0].tolist(),
            "gray_interval": sorted([off_gray, margin]),
            "columns": int((~grid).sum()),
        },
        {
            "kind": "grid",
            "rgb": grid_colors[0].tolist(),
            "gray_interval": [grid_gray, grid_gray],
            "columns": int(grid.sum()),
        },
    ]
    start, end = Fraction(3, 50), Fraction(1, 4)
    for group in groups:
        interval = _alpha_interval(group["rgb"], *group["gray_interval"])
        if interval is None:
            raise ValueError("incompatible_edge_quantization")
        start, end = max(start, interval[0]), min(end, interval[1])
    if start > end:
        raise ValueError("edge_colors_have_no_shared_alpha")
    return {
        "edge_row": row,
        "support_row": row - 1,
        "baseline_y": baseline,
        "requested_columns": [left, requested_right],
        "supported_columns": [left, right],
        "unknown_pixels": int(unknown.sum()),
        "support_rgb": support[0].tolist(),
        "groups": groups,
        "neutral_samples": {
            "margin_columns": [262, 279],
            "margin_gray": margin,
            "below_rows": [row + 1, row + 3],
            "off_grid_gray": off_gray,
            "grid_gray": grid_gray,
        },
        "shared_alpha_interval": [str(start), str(end)],
        "coverage_from_original_blue_only": True,
        "image_modified": False,
    }


def analyze_planet_window(png, time_labels, flux_labels, *, requested_days=5000):
    base = frozen_analyze(png, time_labels, flux_labels, requested_days=requested_days)
    report = {**base, "policy": policy_manifest(), "base_analysis_sha256": _digest(base)}
    if base["reason"] != "unknown_plot_palette":
        return report
    try:
        evidence = _edge_evidence(png, time_labels, flux_labels, base)
    except (ValueError, TypeError, KeyError, OSError, BrowserSafetyStop):
        return {**report, "baseline_edge_rejection": "unsupported_continuous_baseline_edge"}
    return {
        **report,
        "baseline_edge_evidence": evidence,
        "status": "assume_no_planet",
        "reason": "user_approved_window_no_visible_dip",
        "planet_decision": "No",
        "observation_limit_days": 5000,
        "white_cursor_gap_columns": 0,
    }


def analyze_recorded_planet_window(png, time_labels, flux_labels, *, requested_days=5000, policy=None):
    known = recorded_policy_manifest(policy)
    analyzer = frozen_analyze if known == frozen_manifest() else analyze_planet_window
    return analyzer(png, time_labels, flux_labels, requested_days=requested_days)
