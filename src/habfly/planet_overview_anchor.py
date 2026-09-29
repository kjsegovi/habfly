"""Exact plot-only identity for a restored, visible 5,000-day overview.

The only omitted paint is the axis/gutter outside x31:261,y20:152. Full source
PNG/report hashes must still be retained by callers. An absent zero tick is
allowed only with a visible 500,1000 start proving one extrapolated interval.
No pixel tolerance, physical period, depth, or answer authority is supplied.
"""

import hashlib
import io
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from PIL import Image

from .browser import BrowserSafetyStop
from .browser_observation_progress import trace_progress

METHOD = "exact_restored_overview_plot_rgb_affine_axis_v1"
PLOT = (31, 20, 261, 152)


def _require(value, reason):
    if not value:
        raise BrowserSafetyStop("overview_anchor_" + reason)


def _number(value):
    _require(type(value) in {str, int, float}, "invalid_axis_value")
    result = Decimal(str(value))
    _require(result.is_finite(), "nonfinite_axis_value")
    return Fraction(result)


def _ratio(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def overview_anchor(png, time_labels, flux_labels):
    """Return a deterministic signature; exact equality is the only match."""
    try:
        _require(isinstance(png, bytes) and 0 < len(png) <= 1024 * 1024, "invalid_png")
        with Image.open(io.BytesIO(png)) as image:
            _require(image.size in {(280, 195), (280, 196)}, "unsupported_geometry")
            _require(image.convert("RGBA").getextrema()[3] == (255, 255), "transparent_plot")
            geometry = list(image.size)
            pixels = image.convert("RGB").crop(PLOT).tobytes()
        _require(isinstance(time_labels, list) and 3 <= len(time_labels) <= 64, "insufficient_time_labels")
        points = []
        for row in time_labels:
            _require(isinstance(row, dict) and set(row) == {"value", "center_x"}, "invalid_time_label")
            points.append((_number(row["value"]), _number(row["center_x"])))
        points.sort()
        _require(len({day for day, _ in points}) == len(points), "duplicate_time_labels")
        _require(points[0][0] in {0, 500} and points[-1][0] == 5000, "full_window_required")
        if points[0][0] == 500:
            _require(points[1][0] == 1000, "zero_extrapolation_exceeds_one_interval")
        slope = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
        origin = points[0][1] - points[0][0] * slope
        _require(
            slope > 0
            and 29 <= origin <= 32
            and 259 <= origin + 5000 * slope <= 262
            and all(origin + day * slope == pixel for day, pixel in points),
            "nonexact_or_unsupported_affine_axis",
        )
        _require(isinstance(flux_labels, list) and 3 <= len(flux_labels) <= 64, "insufficient_flux_labels")
        values = []
        for row in flux_labels:
            _require(isinstance(row, dict) and set(row) == {"value", "center_y"}, "invalid_flux_label")
            value, pixel = _number(row["value"]), _number(row["center_y"])
            _require(0 <= value <= 100 and 0 <= pixel < geometry[1], "invalid_flux_domain")
            values.append(value)
        _require(len(set(values)) == len(values) and max(values) == 100, "invalid_flux_labels")
        progress = trace_progress(png, time_labels, requested_days=5000)
        _require(progress["endpoint_visible"] is True, "endpoint_unverified")
        return {
            "method": METHOD,
            "source": "exact_plot_rgb_not_full_png_identity",
            "geometry": geometry,
            "plot_rectangle": list(PLOT),
            "plot_rgb_sha256": hashlib.sha256(pixels).hexdigest(),
            "time_affine": {"origin_pixel": _ratio(origin), "pixels_per_day": _ratio(slope)},
            # Retain every nonzero tick exactly: only the separately grounded
            # zero-glyph omission is ignored, never arbitrary missing labels.
            "nonzero_time_labels": [deepcopy(row) for row in time_labels if _number(row["value"]) != 0],
            "flux_axis_labels": deepcopy(flux_labels),
            "endpoint_visible": True,
            "pixel_tolerance": 0,
            "scientific_verified": False,
            "answer_authorized": False,
        }
    except BrowserSafetyStop:
        raise
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError, InvalidOperation):
        raise BrowserSafetyStop("overview_anchor_invalid_evidence") from None
