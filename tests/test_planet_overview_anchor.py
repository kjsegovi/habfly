"""Exact painted-plot identity; no native page or physical inference."""

import io
from copy import deepcopy

import pytest
from PIL import Image
from test_browser_shallow_transit_probe import png

from habfly.browser import BrowserSafetyStop
from habfly.planet_overview_anchor import overview_anchor


def labels():
    return (
        [{"value": str(day), "center_x": 30.5 + day * 0.046} for day in range(0, 5001, 500)],
        [{"value": str(value), "center_y": 20.5 + (100 - value) * 1.3} for value in range(10, 101, 10)],
    )


def change_pixel(data, position):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.putpixel(position, (1, 2, 3))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_only_grounded_zero_glyph_and_left_border_paint_can_differ():
    times, flux = labels()
    original = png()
    changed = change_pixel(change_pixel(original, (30, 21)), (27, 165))
    first = overview_anchor(original, times, flux)
    assert changed != original
    assert overview_anchor(changed, times[1:], flux) == first
    assert first["time_affine"] == {
        "origin_pixel": {"numerator": 61, "denominator": 2},
        "pixels_per_day": {"numerator": 23, "denominator": 500},
    }
    assert first["source"] == "exact_plot_rgb_not_full_png_identity"
    assert first["pixel_tolerance"] == 0 and not first["scientific_verified"]


@pytest.mark.parametrize("pixel", [(31, 20), (40, 30), (260, 151)])
def test_one_changed_plot_pixel_changes_identity_without_tolerance(pixel):
    times, flux = labels()
    assert overview_anchor(png(), times, flux) != overview_anchor(change_pixel(png(), pixel), times, flux)


@pytest.mark.parametrize("change", ["flux", "geometry", "missing_nonzero", "numeric_type"])
def test_remaining_capture_fingerprint_is_exact(change):
    times, flux = labels()
    original = overview_anchor(png(), times, flux)
    data = png()
    if change == "flux":
        flux[0]["center_y"] += 0.000001
    elif change == "geometry":
        data = png(size=(280, 196))
    elif change == "missing_nonzero":
        times.pop(3)
    else:
        times[1]["value"] = 500
    assert overview_anchor(data, times, flux) != original


@pytest.mark.parametrize(
    "change",
    [
        "first250",
        "first1000",
        "missing1000",
        "last4500",
        "duplicate",
        "nonaffine",
        "nonfinite",
        "short",
        "incomplete",
        "flux_missing100",
    ],
)
def test_unverified_overview_never_has_a_comparable_fingerprint(change):
    times, flux = labels()
    data = png()
    if change == "first250":
        times = times[1:]
        times[0] = {"value": "250", "center_x": 42}
    elif change == "first1000":
        times = times[2:]
    elif change == "missing1000":
        times = [times[1], *times[3:]]
    elif change == "last4500":
        times.pop()
    elif change == "duplicate":
        times.append(deepcopy(times[1]))
    elif change == "nonaffine":
        times[2]["center_x"] += 0.000001
    elif change == "nonfinite":
        times[2]["center_x"] = float("nan")
    elif change == "short":
        times = [times[0], times[-1]]
    elif change == "flux_missing100":
        flux.pop()
    else:
        image = Image.open(io.BytesIO(data))
        for x in range(230, 280):
            for y in range(195):
                image.putpixel((x, y), (0, 0, 0))
        output = io.BytesIO()
        image.save(output, format="PNG")
        data = output.getvalue()
    with pytest.raises(BrowserSafetyStop):
        overview_anchor(data, times, flux)
