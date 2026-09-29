"""Offline raster evidence for an explicitly pragmatic, user-approved policy.

No dip in the first 5,000 plotted days permits a reference No hypothesis, not
proof of absence. Long-period, shallow and subpixel events may be missed. This
module never drives a browser, supplies measurements or generates training labels.
"""

import hashlib
import io
import json
import math

import numpy as np
from PIL import Image

from .browser import BrowserSafetyStop
from .browser_observation_progress import trace_mask, trace_progress

POLICY = {
    "version": "user_approved_5000_day_raster_no_dip_v2",
    "requested_days": 5000,
    "visible_axis_maxima": [5000, 10000],
    "crop_sizes": [[280, 195], [280, 196]],
    "plot_x": [30.5, 260.5], "plot_y": [20, 151],
    "baseline_percent": 100, "baseline_tolerance_pixels": 1.5,
    "flux_axis": "At least 3 exposed linear ticks in 0..100; visible 100 at y19..22; no visible zero required",
    "flux_tick_pixel_bounds": [19, 152],
    "axis_fit_tolerance_pixels": 1, "maximum_white_cursor_gap_pixels": 8,
    "maximum_internal_gaps": 1,
    "pixel_coordinates": "centers at index+0.5; only centers within requested interval",
    "cursor": "neutral RGB>=220 within 2.5 pixels of baseline",
    "cursor_edge": "blue-white linear blend within RGB3, baseline band and next to white cursor",
    "minimum_trace_start_columns": 8, "start_tolerance_pixels": 1,
    "dip_support": "two adjacent blue pixels below baseline tolerance",
    "unknown_palette": "non-blue channel spread greater than 8 blocks inference",
    "minimum_trace_columns": 64, "minimum_endpoint_columns": 6,
    "endpoint_band_pixels": 9, "endpoint_tolerance_pixels": 1.5,
    "provenance": "reference_prediction", "user_approved": True,
    "scope": "First 5000 days only; no visible raster dip permits assumed No despite possible score loss.",
    "limitations": ["not proven absence", "long-period planets may be missed",
                    "shallow or subpixel dips may be missed", "pixel sensitivity depends on visible zoom",
                    "not learned perception"],
    "palette": "trace_mask: r10..110,g/r1.35..1.95,b/r1.65..2.35,b>=g+5",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def policy_manifest():
    """A fresh JSON-compatible identity; no mutable shared report metadata."""
    return {**json.loads(json.dumps(POLICY)), "sha256": _digest(POLICY)}


def _axis(labels, coordinate):
    if not isinstance(labels, list) or not 3 <= len(labels) <= 64:
        raise ValueError("invalid_axis_labels")
    if any(type(row[key]) not in {str, int, float} for row in labels for key in ("value", coordinate)):
        raise ValueError("invalid_axis_labels")
    points = sorted((float(row["value"]), float(row[coordinate])) for row in labels)
    if (not all(math.isfinite(v) for point in points for v in point)
            or len({point[0] for point in points}) != len(points)):
        raise ValueError("invalid_axis_labels")
    slope = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
    origin = points[0][1] - points[0][0] * slope
    if any(abs(origin + value * slope - position) > 1 for value, position in points):
        raise ValueError("nonlinear_axis")
    return points, slope, origin


def analyze_planet_window(png, time_labels, flux_labels, *, requested_days=5000):
    """Return replayable evidence from PNG bytes and fully exposed axis labels.

    Unknown geometry/palette, unexplained gaps and clipping fail closed. The
    only tolerated internal gap is <=8 columns covered by a white hover cursor.
    No answer is written, and no depth, period, zero or grading label is created.
    """
    report = {
        "status": "insufficient_visual_evidence", "reason": "invalid_evidence",
        "policy": policy_manifest(), "provenance": "reference_prediction",
        "requested_days": requested_days if type(requested_days) is int else None,
        "chart_sha256": hashlib.sha256(png).hexdigest(),
        "axis_sha256": None, "planet_decision": None, "absence_proven": False,
        "scientific_verified": False, "learned_perception": False,
        "training_label": False, "task_completed": False, "answer_writes": 0,
    }
    if type(requested_days) is not int or requested_days != 5000:
        return {**report, "reason": "policy_requires_exactly_5000_days"}
    try:
        times, slope, origin = _axis(time_labels, "center_x")
        fluxes, flux_slope, _ = _axis(flux_labels, "center_y")
        baseline = fluxes[-1][1]
        if (times[0][0] != 0 or times[-1][0] not in {5000, 10000} or slope <= 0
                or not 29 <= origin <= 32 or not 258 <= times[-1][1] <= 262
                or flux_slope >= 0 or fluxes[0][0] < 0 or fluxes[-1][0] != 100
                or any(not 19 <= position <= 152 for _, position in fluxes)
                or not 19 <= baseline <= 22):
            raise ValueError("unsupported_axis_range")
        report["axis_sha256"] = _digest({"time": time_labels, "flux": flux_labels})
        readiness = trace_progress(png, time_labels, requested_days=requested_days)
        mask = trace_mask(png)
        with Image.open(io.BytesIO(png)) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=int)
        if np.any(rgba[:, :, 3] != 255):
            raise ValueError("transparent_chart_crop")
    except (BrowserSafetyStop, ValueError, TypeError, KeyError, ZeroDivisionError, OSError):
        return {**report, "reason": "invalid_or_unsupported_chart_or_axes"}
    left = math.ceil(origin - 0.5)
    right = math.floor(origin + requested_days * slope - 0.5)
    mask = mask[20:152, left:right + 1]
    pixels = rgba[20:152, left:right + 1, :3]
    rows = np.arange(20, 152)[:, None] + 0.5
    baseline_band = np.abs(rows - baseline) <= 2.5
    white = ((pixels.min(axis=2) >= 220) & (np.ptp(pixels, axis=2) <= 10)
             & baseline_band).any(axis=0)
    # A rendered white hover circle has colored antialiasing where it meets
    # the blue baseline. Count only the still-visible blue component, not the
    # white occluded interior, and never accept such colors away from a cursor.
    alpha = (pixels[:, :, 0] - 80) / 175
    blend = np.array([80, 132, 154]) + alpha[:, :, None] * np.array([175, 123, 101])
    near_white = np.convolve(white.astype(int), [1, 1, 1], mode="same") > 0
    mask |= ((alpha > 0) & (alpha < 1) & (np.abs(pixels - blend).max(axis=2) <= 3)
             & baseline_band & near_white)
    columns = mask.any(axis=0)
    indices = np.flatnonzero(columns)
    report.update({key: readiness[key] for key in (
        "trace_columns", "endpoint_trace_columns", "rightmost_trace_pixel", "endpoint_visible")})
    report["examined_day_interval"] = [0, 5000]
    report["requested_pixel_columns"], report["covered_pixel_columns"] = len(columns), len(indices)
    # Chromatic pixels outside the established trace palette are not guessed.
    if np.any((np.ptp(pixels, axis=2) > 8) & ~mask):
        return {**report, "reason": "unknown_plot_palette"}
    if len(indices) < 8 or indices[0] > 1:
        return {**report, "reason": "missing_trace_start_or_insufficient_trace"}
    if np.any(mask[-1]) or np.any(mask & (rows < baseline - 1.5)):
        return {**report, "reason": "trace_clipped_or_baseline_inconsistent"}
    dips = mask & (rows > baseline + 1.5)
    if np.any(dips):
        supported = np.any(dips[1:] & dips[:-1]) or np.any(dips[:, 1:] & dips[:, :-1])
        return {**report, "status": "dip_observed" if supported else "insufficient_visual_evidence",
                "reason": "visible_blue_dip" if supported else "isolated_below_baseline_pixel"}
    missing = np.flatnonzero(~columns[:indices[-1] + 1])
    runs = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)
    gaps = [run for run in runs if len(run) and not (len(run) == 1 and run[0] == 0)]
    if len(gaps) > 1 or any(len(run) > 8 or not white[run].all() for run in gaps):
        return {**report, "reason": "unexplained_trace_gap"}
    report["white_cursor_gap_columns"] = sum(len(run) for run in gaps)
    if not readiness["endpoint_visible"]:
        return {**report, "status": "still_collecting", "reason": "requested_endpoint_not_reached"}
    return {**report, "status": "assume_no_planet", "reason": "user_approved_window_no_visible_dip",
            "planet_decision": "No", "observation_limit_days": 5000}
