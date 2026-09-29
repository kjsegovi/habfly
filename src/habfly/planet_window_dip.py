"""Positive-only supplemental raster evidence; never a planet answer or absence.

The frozen negative-window policy is intentionally not changed. This separate
detector can explain blue antialias edges on visibly established neutral chart
background/grid pixels. Only the original trace_mask supplies dip support;
explained antialias pixels never manufacture a component or fill missing data.
"""

import hashlib
import io
import json
import math
from collections import Counter

import numpy as np
from PIL import Image

from .browser import BrowserSafetyStop
from .browser_observation_progress import trace_mask

DETECTOR = {
    "version": "positive_only_rendered_window_dip_v1",
    "requested_days": 5000,
    "visible_axis_maxima": [5000, 10000],
    "crop_sizes": [[280, 195], [280, 196]],
    "plot_y": [20, 151],
    "baseline_percent": 100,
    "baseline_tolerance_pixels": 1.5,
    "trace_palette": "unchanged trace_mask: r10..110,g/r1.35..1.95,b/r1.65..2.35,b>=g+5",
    "antialias_blue": [80, 132, 154],
    "antialias_max_channel_residual": 1.5,
    "antialias_alpha": [0.06, 1],
    "antialias_support": "8-neighbor original blue pixel; never counted as dip support",
    "neutral_background": "equal RGB channels in 0..64, exposed within one pixel",
    "grid_background": "same row/column within1.5px of exposed axis tick, at least32 equal neutral pixels",
    "minimum_baseline_columns": 64,
    "baseline": "unbroken original-blue baseline from first to last visible column; starts within1px",
    "minimum_components": 2,
    "minimum_component_rows": 3,
    "maximum_component_columns": 4,
    "component_connection": "4-neighbor original-blue path to baseline; three baseline columns each side",
    "unknown_palette": "non-blue channel spread>8 blocks unless independently explained antialias",
    "bright_neutral_overlay": "equal/near-neutral RGB>=180 within plot blocks inference",
    "scope": "Positive follow-up evidence only in first5000days; no measured period/depth or planet answer",
    "limitations": [
        "not scientific verification",
        "not learned perception",
        "narrow multi-component raster support only",
        "single, broad, clipped, shallow or subpixel dips may remain unresolved",
        "requires independently guarded student-visible chart capture and axes",
    ],
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def detector_manifest():
    return {**json.loads(json.dumps(DETECTOR)), "sha256": _digest(DETECTOR)}


def _axis(labels, coordinate):
    if not isinstance(labels, list) or not 3 <= len(labels) <= 64:
        raise ValueError("invalid_axis_labels")
    if any(type(row[key]) not in {str, int, float} for row in labels for key in ("value", coordinate)):
        raise ValueError("invalid_axis_labels")
    points = sorted((float(row["value"]), float(row[coordinate])) for row in labels)
    if not all(math.isfinite(v) for row in points for v in row) or len({p[0] for p in points}) != len(points):
        raise ValueError("invalid_axis_labels")
    slope = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
    origin = points[0][1] - points[0][0] * slope
    if any(abs(origin + value * slope - position) > 1 for value, position in points):
        raise ValueError("nonlinear_axis")
    return points, slope, origin


def _neutral(pixel):
    return int(pixel[0]) if pixel[0] == pixel[1] == pixel[2] and pixel[0] <= 64 else None


def _grid_neutrals(pixels):
    counts = Counter(value for pixel in pixels if (value := _neutral(pixel)) is not None)
    return {value for value, count in counts.items() if count >= 32}


def _explained_edge(pixels, original, x, y, left, times, fluxes):
    height, width = original.shape
    y0, y1, x0, x1 = max(0, y - 1), min(height, y + 2), max(0, x - 1), min(width, x + 2)
    if not original[y0:y1, x0:x1].any():
        return False
    backgrounds = {
        value for pixel in pixels[y0:y1, x0:x1].reshape(-1, 3) if (value := _neutral(pixel)) is not None
    }
    # A grid line can be covered for several rows by a real dip. Only exposed
    # ticks plus a substantial run of visible same-row/column neutral pixels
    # permit its background blend; no SVG path or covered value is inspected.
    if any(abs(position - (left + x + 0.5)) <= 1.5 for _, position in times):
        backgrounds |= _grid_neutrals(pixels[:, x])
    if any(abs(position - (20 + y + 0.5)) <= 1.5 for _, position in fluxes):
        backgrounds |= _grid_neutrals(pixels[y])
    blue, value = np.array(DETECTOR["antialias_blue"], dtype=float), pixels[y, x].astype(float)
    for background in backgrounds:
        direction = blue - background
        alpha = float(np.dot(value - background, direction) / np.dot(direction, direction))
        residual = np.abs(value - (background + alpha * direction)).max()
        if 0.06 <= alpha < 1 and residual <= 1.5:
            return True
    return False


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = remaining.pop()
        component, pending = {first}, [first]
        while pending:
            y, x = pending.pop()
            for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    pending.append(point)
        result.append(component)
    return result


def analyze_window_dip(png, time_labels, flux_labels):
    """Replay a bounded positive-only detector over already guarded visible data.

    The return value never says Yes/No, measures an orbit, or certifies a task.
    Antialias recognition only explains extra colors; existing blue pixels must
    independently support at least two narrow baseline-connected features.
    """
    report = {
        "status": "insufficient_visual_evidence",
        "reason": "invalid_evidence",
        "detector": detector_manifest(),
        "source": "student_visible_chart_crop_and_exposed_axis_labels",
        "provenance": "reference_positive_followup_not_learned",
        "chart_sha256": hashlib.sha256(png).hexdigest() if isinstance(png, bytes) else None,
        "axis_sha256": None,
        "requested_days": 5000,
        "examined_day_interval": [0, 5000],
        "planet_decision": None,
        "planet_presence": None,
        "absence_proven": False,
        "scientific_verified": False,
        "learned_perception": False,
        "training_label": False,
        "task_completed": False,
        "observation_completed": False,
        "browser_actions": 0,
        "answer_writes": 0,
        "supported_dip_components": 0,
        "support_pixel_bounds": [],
    }
    try:
        if not isinstance(png, bytes):
            raise TypeError("invalid_image")
        times, slope, origin = _axis(time_labels, "center_x")
        fluxes, flux_slope, _ = _axis(flux_labels, "center_y")
        baseline = fluxes[-1][1]
        if (
            times[0][0] != 0
            or times[-1][0] not in {5000, 10000}
            or slope <= 0
            or not 29 <= origin <= 32
            or not 258 <= times[-1][1] <= 262
            or flux_slope >= 0
            or fluxes[0][0] < 0
            or fluxes[-1][0] != 100
            or any(not 19 <= position <= 152 for _, position in fluxes)
            or not 19 <= baseline <= 22
        ):
            raise ValueError("unsupported_axes")
        report["axis_sha256"] = _digest({"time": time_labels, "flux": flux_labels})
        original = trace_mask(png)
        with Image.open(io.BytesIO(png)) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=int)
        if np.any(rgba[:, :, 3] != 255):
            raise ValueError("transparent_crop")
    except (BrowserSafetyStop, ValueError, TypeError, KeyError, ZeroDivisionError, OSError):
        return {**report, "reason": "invalid_or_unsupported_chart_or_axes"}
    left, right = math.ceil(origin - 0.5), math.floor(origin + 5000 * slope - 0.5)
    original = original[20:152, left : right + 1]
    pixels = rgba[20:152, left : right + 1, :3]
    rows = np.arange(20, 152)[:, None] + 0.5
    if np.any((pixels.min(axis=2) >= 180) & (np.ptp(pixels, axis=2) <= 10)):
        return {**report, "reason": "bright_neutral_plot_overlay"}
    unknown = (np.ptp(pixels, axis=2) > 8) & ~original
    candidates = np.argwhere(unknown)
    report["antialias_candidates"] = len(candidates)
    explained = sum(
        _explained_edge(pixels, original, int(x), int(y), left, times, fluxes) for y, x in candidates
    )
    report["explained_antialias_pixels"] = explained
    if explained != len(candidates):
        return {**report, "reason": "unknown_plot_palette"}
    if np.any(original[-1]) or np.any(original & (rows < baseline - 1.5)):
        return {**report, "reason": "trace_clipped_or_baseline_inconsistent"}
    baseline_mask = original & (np.abs(rows - baseline) <= 1.5)
    baseline_columns = baseline_mask.any(axis=0)
    indices = np.flatnonzero(baseline_columns)
    if len(indices) < 64 or indices[0] > 1 or not baseline_columns[indices[0] : indices[-1] + 1].all():
        return {**report, "reason": "insufficient_unbroken_baseline"}
    connected = set()
    for component in _components(original):
        if any(baseline_mask[y, x] for y, x in component):
            connected |= component
    below = original & (rows > baseline + 1.5)
    bounds = []
    for component in _components(below):
        ys, xs = zip(*component)
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if (
            y1 - y0 + 1 < 3
            or x1 - x0 + 1 > 4
            or not component.issubset(connected)
            or x0 < 3
            or x1 + 3 >= original.shape[1]
            or not baseline_columns[x0 - 3 : x0].all()
            or not baseline_columns[x1 + 1 : x1 + 4].all()
        ):
            return {**report, "reason": "unsupported_below_baseline_component"}
        bounds.append([int(x0 + left), int(y0 + 20), int(x1 + left), int(y1 + 20)])
    if len(bounds) < 2:
        return {**report, "reason": "insufficient_independent_dip_support"}
    return {
        **report,
        "status": "dip_observed",
        "reason": "multiple_original_blue_baseline_connected_features",
        "supported_dip_components": len(bounds),
        "support_pixel_bounds": sorted(bounds),
    }
