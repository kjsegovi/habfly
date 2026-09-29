"""Synthetic raster fixtures validate policy safety, never astronomy labels."""

import hashlib
import io
import json
import socket

import pytest
from PIL import Image, ImageDraw

from habfly.planet_window_policy import analyze_planet_window, policy_manifest

BLUE = (80, 132, 154)


def labels(maximum=10000):
    time = [{"value": str(day), "center_x": 30.5 + day * 230 / maximum}
            for day in range(0, maximum + 1, maximum // 10)]
    flux = [{"value": str(value), "center_y": 151.5 - value * 1.31}
            for value in range(10, 101, 10)]
    return time, flux


def crop(*, end=260, color=BLUE, gap=None, cursor=True, dip=None, stray=None,
         size=(280, 195), plot=True, antialias=False):
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(20, 152, 13):
        draw.line((30, y, 260, y), fill=(60, 60, 60))
    for x in range(30, 261, 23):
        draw.line((x, 20, x, 151), fill=(60, 60, 60))
    draw.rectangle((230, 0, 260, 17), fill=BLUE)
    draw.line((230, 167, 260, 167), fill=BLUE)
    if plot:
        draw.line((30, 20, end, 20), fill=(40, 66, 77) if antialias else color)
    if dip is not None:
        x, bottom = dip
        draw.line((x, 20, x, bottom), fill=BLUE)
    if gap is not None:
        left, right = gap
        draw.rectangle((left, 18, right, 23), fill=(255, 255, 255) if cursor else (0, 0, 0))
    if stray is not None:
        draw.point(stray, fill=BLUE)
    result = io.BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


def analyze(png=None, *, maximum=10000, requested_days=5000):
    time, flux = labels(maximum)
    return analyze_planet_window(crop() if png is None else png, time, flux, requested_days=requested_days)


@pytest.mark.parametrize("maximum,size,antialias", [(5000, (280, 195), False),
                                                   (10000, (280, 196), True)])
def test_flat_complete_window_is_only_an_explicit_reference_assumption(maximum, size, antialias):
    report = analyze(crop(size=size, antialias=antialias), maximum=maximum)
    assert report["status"] == "assume_no_planet" and report["planet_decision"] == "No"
    assert report["provenance"] == "reference_prediction" and report["observation_limit_days"] == 5000
    for flag in ["absence_proven", "scientific_verified", "learned_perception", "training_label", "task_completed"]:
        assert report[flag] is False
    assert report["answer_writes"] == 0 and report["examined_day_interval"] == [0, 5000]
    assert "period" not in report and "depth" not in report


def test_supported_dip_is_evidence_not_an_automatic_yes_or_measurement():
    report = analyze(crop(end=110, dip=(70, 30)))
    assert report["status"] == "dip_observed"
    assert report["planet_decision"] is None and not report["endpoint_visible"]
    assert not report["scientific_verified"]


def test_only_first_5000_days_are_examined_on_10000_day_axis():
    assert analyze(crop(dip=(146, 40)))["status"] == "assume_no_planet"
    assert analyze(crop(dip=(145, 40)))["status"] == "dip_observed"


def zoomed_flux_labels(minimum):
    return [{"value": str(minimum + (100 - minimum) * step / 4),
             "center_y": 150.5 - step * 32.5} for step in range(5)]


@pytest.mark.parametrize("minimum", [98, 99.99])
def test_visible_zoom_needs_no_zero_label_and_uses_same_pixel_threshold(minimum):
    time = labels()[0]
    flux = zoomed_flux_labels(minimum)
    assert analyze_planet_window(crop(), time, flux)["status"] == "assume_no_planet"
    assert analyze_planet_window(crop(dip=(80, 24)), time, flux)["status"] == "dip_observed"
    assert analyze_planet_window(crop(dip=(80, 21)), time, flux)["status"] == "assume_no_planet"
    assert "pixel sensitivity depends on visible zoom" in policy_manifest()["limitations"]


@pytest.mark.parametrize("mode", ["offset", "nonlinear", "outside_plot", "missing_100", "negative"])
def test_zoomed_axis_requires_visible_top_baseline_and_linear_exposed_ticks(mode):
    time = labels()[0]
    flux = zoomed_flux_labels(99.99)
    if mode == "offset":
        flux = [{**row, "center_y": row["center_y"] + 4} for row in flux]
    elif mode == "nonlinear":
        flux[2]["center_y"] += 3
    elif mode == "outside_plot":
        flux = [{**row, "center_y": 20.5 + (row["center_y"] - 20.5) * 1.5} for row in flux]
    elif mode == "missing_100":
        flux = flux[:-1]
    else:
        flux = zoomed_flux_labels(-1)
    result = analyze_planet_window(crop(), time, flux)
    assert result["status"] == "insufficient_visual_evidence" and result["planet_decision"] is None


def test_partial_empty_gray_grid_and_stray_pixels_never_become_no():
    assert analyze(crop(end=110))["status"] == "still_collecting"
    for png in [crop(plot=False), crop(color=(60, 60, 60)), crop(end=30, stray=(145, 20)),
                crop(end=110, stray=(145, 20)), crop(stray=(90, 35))]:
        report = analyze(png)
        assert report["status"] == "insufficient_visual_evidence"
        assert report["planet_decision"] is None


@pytest.mark.parametrize("gap,cursor,expected", [((80, 87), True, "assume_no_planet"),
                                                ((80, 88), True, "insufficient_visual_evidence"),
                                                ((80, 81), False, "insufficient_visual_evidence")])
def test_only_bounded_white_hover_cursor_gaps_are_allowed(gap, cursor, expected):
    assert analyze(crop(gap=gap, cursor=cursor))["status"] == expected


def test_multiple_separate_occlusions_are_not_near_continuous_coverage():
    image = Image.open(io.BytesIO(crop(gap=(80, 82)))).convert("RGB")
    ImageDraw.Draw(image).rectangle((100, 18, 102, 23), fill=(255, 255, 255))
    png = io.BytesIO()
    image.save(png, format="PNG")
    assert analyze(png.getvalue())["reason"] == "unexplained_trace_gap"


def test_unknown_palette_clipped_trace_and_shallow_accepted_resolution():
    assert analyze(crop(color=(255, 80, 20)))["reason"] == "unknown_plot_palette"
    assert analyze(crop(dip=(80, 151)))["reason"] == "trace_clipped_or_baseline_inconsistent"
    # One-pixel vertical excursions are below the disclosed raster threshold.
    assert analyze(crop(dip=(80, 21)))["status"] == "assume_no_planet"
    assert analyze(crop(dip=(80, 22)))["status"] == "insufficient_visual_evidence"


def test_hover_circle_blends_require_nearby_white_cursor_and_known_colors():
    image = Image.open(io.BytesIO(crop(gap=(69, 75)))).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.point((68, 20), fill=(118, 157, 174))
    draw.point((76, 20), fill=(212, 225, 230))
    png = io.BytesIO()
    image.save(png, format="PNG")
    assert analyze(png.getvalue())["status"] == "assume_no_planet"
    draw.point((76, 20), fill=(212, 205, 230))
    png = io.BytesIO()
    image.save(png, format="PNG")
    assert analyze(png.getvalue())["reason"] == "unknown_plot_palette"
    image = Image.open(io.BytesIO(crop())).convert("RGB")
    ImageDraw.Draw(image).point((76, 20), fill=(212, 225, 230))
    png = io.BytesIO()
    image.save(png, format="PNG")
    assert analyze(png.getvalue())["reason"] == "unknown_plot_palette"


@pytest.mark.parametrize("bad", ["nonlinear_time", "nonlinear_flux", "short", "nan", "duplicate",
                                "negative", "unknown_max", "shifted", "bool", "geometry", "png"])
def test_invalid_axes_and_crops_fail_closed(bad):
    time, flux = labels()
    png = crop()
    if bad == "nonlinear_time":
        time[4]["center_x"] += 5
    elif bad == "nonlinear_flux":
        flux[4]["center_y"] += 5
    elif bad == "short":
        time = time[:2]
    elif bad == "nan":
        flux[0]["value"] = "nan"
    elif bad == "duplicate":
        time[2] = time[1]
    elif bad == "negative":
        time[0]["value"] = "-100"
    elif bad == "unknown_max":
        time = labels(20000)[0]
    elif bad == "shifted":
        flux = [{**row, "center_y": row["center_y"] + 5} for row in flux]
    elif bad == "bool":
        time[0]["value"] = False
    elif bad == "geometry":
        png = crop(size=(300, 200))
    else:
        png = b"not a PNG"
    report = analyze_planet_window(png, time, flux)
    assert report["status"] == "insufficient_visual_evidence" and report["planet_decision"] is None


@pytest.mark.parametrize("days", [0, 4999, 10000, True, 5000.0])
def test_unapproved_window_is_not_silently_replaced(days):
    assert analyze(requested_days=days)["reason"] == "policy_requires_exactly_5000_days"


def test_offline_replay_is_deterministic_and_hashes_all_input_evidence(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: pytest.fail("Network is forbidden"))
    png = crop()
    report = analyze(png)
    assert report == analyze(png)
    assert report["chart_sha256"] == hashlib.sha256(png).hexdigest()
    manifest = policy_manifest()
    digest = manifest.pop("sha256")
    assert digest == hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
    assert report["policy"]["sha256"] == digest
    time, flux = labels()
    flux[0]["center_y"] += 0.1
    assert analyze_planet_window(png, time, flux)["axis_sha256"] != report["axis_sha256"]
    manifest["limitations"].clear()
    assert policy_manifest()["limitations"]
