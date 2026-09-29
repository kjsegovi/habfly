"""Baseline rendering compatibility is an assumption, never a science label."""

import hashlib
import io
import json
import socket
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from test_planet_window_policy import crop, labels

from habfly.browser import BrowserSafetyStop
from habfly.planet_window_baseline_edge import (
    _alpha_interval,
    analyze_planet_window,
    analyze_recorded_planet_window,
    policy_manifest,
    recorded_policy_manifest,
)
from habfly.planet_window_dip import detector_manifest
from habfly.planet_window_policy import analyze_planet_window as frozen
from habfly.planet_window_policy import policy_manifest as frozen_manifest


def encode(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def edge_crop(*, maximum=5000, off_color=(29, 36, 39), grid_color=(55, 61, 64)):
    picture = Image.new("RGB", (280, 196))
    draw = ImageDraw.Draw(picture)
    for x in range(30, 261, 23):
        draw.line((x, 20, x, 151), fill=(51,) * 3)
    draw.line((262, 21, 279, 21), fill=(26,) * 3)
    for x in range(30, 260):
        picture.putpixel((x, 20), (80, 131, 153))
        picture.putpixel((x, 21), grid_color if (x - 30) % 23 == 0 else off_color)
    # Visible final grid border is not silently promoted to trace coverage.
    picture.putpixel((260, 20), (53, 56, 58))
    return picture


def analyze(picture=None, *, maximum=5000):
    times, fluxes = labels(maximum)
    return analyze_planet_window(encode(edge_crop() if picture is None else picture), times, fluxes)


def test_complete_edge_explained_only_with_original_trace_and_visible_backgrounds(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("No network"))
    png = encode(edge_crop())
    times, fluxes = labels(5000)
    assert frozen(png, times, fluxes)["reason"] == "unknown_plot_palette"
    result = analyze()
    assert result == analyze()
    assert result["status"] == "assume_no_planet" and result["planet_decision"] == "No"
    assert result["chart_sha256"] == hashlib.sha256(png).hexdigest()
    assert result["policy"] == policy_manifest()
    assert result["baseline_edge_evidence"]["supported_columns"] == [30, 259]
    assert result["baseline_edge_evidence"]["shared_alpha_interval"] == ["9/74", "7/54"]
    assert result["white_cursor_gap_columns"] == 0
    for key in (
        "absence_proven",
        "scientific_verified",
        "learned_perception",
        "training_label",
        "task_completed",
    ):
        assert result[key] is False
    assert result["answer_writes"] == 0


@pytest.mark.parametrize(
    "kind",
    [
        "support_gap",
        "support_color",
        "endpoint_gap",
        "start_gap",
        "cursor_gap",
        "second_row",
        "blue_below",
        "unknown_below",
        "off_outlier",
        "off_grid_variant",
        "margin_outlier",
        "margin_non_neutral",
        "margin_bright",
        "off_below",
        "grid_below",
        "transparent",
        "white_overlay",
        "gray_overlay",
        "bright_final_border",
        "incompatible_rgb",
        "disjoint_alpha",
    ],
)
def test_changed_raster_evidence_is_never_repaired(kind):
    picture = edge_crop()
    draw = ImageDraw.Draw(picture)
    if kind == "support_gap":
        picture.putpixel((100, 20), (0, 0, 0))
    elif kind == "support_color":
        picture.putpixel((100, 20), (40, 66, 77))
    elif kind == "endpoint_gap":
        draw.rectangle((255, 20, 259, 21), fill=(0, 0, 0))
    elif kind == "start_gap":
        draw.rectangle((30, 20, 30, 21), fill=(0, 0, 0))
    elif kind == "cursor_gap":
        draw.rectangle((100, 19, 102, 22), fill=(255,) * 3)
    elif kind == "second_row":
        picture.putpixel((100, 22), (29, 36, 39))
    elif kind == "blue_below":
        picture.putpixel((100, 80), (80, 132, 154))
    elif kind == "unknown_below":
        picture.putpixel((100, 80), (29, 36, 39))
    elif kind == "off_outlier":
        picture.putpixel((100, 21), (29, 36, 38))
    elif kind == "off_grid_variant":
        picture.putpixel((100, 21), (55, 61, 64))
    elif kind == "margin_outlier":
        picture.putpixel((270, 21), (0, 0, 0))
    elif kind == "margin_non_neutral":
        draw.line((262, 21, 279, 21), fill=(25, 26, 26))
    elif kind == "margin_bright":
        draw.line((262, 21, 279, 21), fill=(100,) * 3)
    elif kind == "off_below":
        picture.putpixel((100, 24), (1, 1, 1))
    elif kind == "grid_below":
        picture.putpixel((53, 24), (50, 51, 51))
    elif kind == "transparent":
        picture = picture.convert("RGBA")
        picture.putpixel((100, 80), (0, 0, 0, 254))
    elif kind in {"white_overlay", "gray_overlay"}:
        draw.rectangle((100, 65, 140, 90), fill=(255 if kind == "white_overlay" else 100,) * 3)
    elif kind == "bright_final_border":
        draw.line((260, 20, 260, 151), fill=(255,) * 3)
    elif kind == "incompatible_rgb":
        picture = edge_crop(off_color=(29, 36, 42))
    elif kind == "disjoint_alpha":
        picture = edge_crop(off_color=(24, 40, 46))
    result = analyze(picture)
    assert result["status"] != "assume_no_planet"
    assert result["planet_decision"] is None


def test_one_shared_alpha_required_even_when_each_rgb_has_separate_fit():
    # Independent witnesses: off-grid=.25*blue+.75*gray5; grid=.127*blue+.873*gray51.
    first = _alpha_interval((24, 37, 42), 0, 26)
    second = _alpha_interval((55, 61, 64), 51, 51)
    assert first and second and first[0] > second[1]
    assert analyze(edge_crop(off_color=(24, 37, 42)))["planet_decision"] is None


def test_independent_known_shared_blend_witness():
    alpha = Fraction(127, 1000)
    for rgb, gray in (((29, 36, 39), Fraction(109, 5)), ((55, 61, 64), Fraction(51))):
        calculated = [alpha * blue + (1 - alpha) * gray for blue in (80, 132, 154)]
        assert all(
            abs(value - expected) <= Fraction(1, 2) for value, expected in zip(calculated, rgb, strict=True)
        )


def test_first_5000_only_on_10000_axis_but_never_ignore_an_in_window_dip():
    picture = edge_crop()
    ImageDraw.Draw(picture).line((180, 20, 180, 35), fill=(80, 132, 154))
    assert analyze(picture, maximum=10000)["status"] == "assume_no_planet"
    picture.putpixel((100, 35), (80, 132, 154))
    assert analyze(picture, maximum=10000)["planet_decision"] is None


@pytest.mark.parametrize("kind", ["nonlinear_time", "nonlinear_flux", "baseline", "geometry", "days"])
def test_original_axis_geometry_and_window_restrictions_still_apply(kind):
    picture = edge_crop()
    times, fluxes = labels(5000)
    days = 5000
    if kind == "nonlinear_time":
        times[4]["center_x"] += 5
    elif kind == "nonlinear_flux":
        fluxes[4]["center_y"] += 5
    elif kind == "baseline":
        fluxes = [{**item, "center_y": item["center_y"] - 1} for item in fluxes]
    elif kind == "geometry":
        picture = picture.resize((281, 196))
    else:
        days = 10000
    assert (
        analyze_planet_window(encode(picture), times, fluxes, requested_days=days)["planet_decision"] is None
    )


def test_legacy_dispatch_and_non_unknown_outcomes_are_exact():
    times, fluxes = labels(5000)
    for png in (crop(), crop(dip=(70, 40)), crop(end=110), b"bad"):
        base = frozen(png, times, fluxes)
        assert analyze_recorded_planet_window(png, times, fluxes) == base
        assert analyze_recorded_planet_window(png, times, fluxes, policy=frozen_manifest()) == base
        new = analyze_planet_window(png, times, fluxes)
        assert {k: v for k, v in new.items() if k not in {"policy", "base_analysis_sha256"}} == {
            k: v for k, v in base.items() if k != "policy"
        }
    assert recorded_policy_manifest() == frozen_manifest()
    png = encode(edge_crop())
    assert analyze_recorded_planet_window(png, times, fluxes)["planet_decision"] is None
    assert (
        analyze_recorded_planet_window(png, times, fluxes, policy=policy_manifest())["planet_decision"]
        == "No"
    )


@pytest.mark.parametrize("kind", ["missing", "hash", "version", "bool", "float", "unknown", "nan"])
def test_partial_mixed_or_changed_policy_identity_rejects(kind):
    policy = policy_manifest()
    if kind == "missing":
        policy.pop("sha256")
    elif kind == "hash":
        policy["sha256"] = frozen_manifest()["sha256"]
    elif kind == "version":
        policy["version"] = frozen_manifest()["version"]
    elif kind == "bool":
        policy["user_approved"] = 1
    elif kind == "float":
        policy["requested_days"] = 5000.0
    elif kind == "unknown":
        policy = "latest"
    else:
        policy["requested_days"] = float("nan")
    with pytest.raises(BrowserSafetyStop, match="unsupported_planet_window_policy"):
        recorded_policy_manifest(policy)


def test_frozen_hashes_and_fresh_manifest_unchanged():
    assert frozen_manifest()["sha256"] == "1e376eb83c896c6828aa9335ca98820e95ffbf0328ae4a6f33ce53118e16e826"
    assert detector_manifest()["sha256"] == "fbd28b7dacbb2239ec9a41de8aaff6a5e5dadb2882586d5fab48c5da3df480e2"
    manifest = policy_manifest()
    manifest["compatibility"].clear()
    assert policy_manifest()["compatibility"]


def test_preserved_real_ferahir_bytes_are_not_rewritten_or_old_decision_upgraded():
    directory = Path(
        "experiments/browser-project-supplied-three-star/5e1bbaded51a438c87e28f9960936996/campaign/stars/003/owner/star/shallow/initial/progress"
    )
    if not directory.is_dir():
        pytest.skip("Optional historical local raster; synthetic fixtures always run")
    raw = (directory / "chart.png").read_bytes()
    report = json.loads((directory / "report.json").read_text())
    assert (
        hashlib.sha256(raw).hexdigest() == "7c61e470b3ea1ebdd4a87d0d76b0af7837cc98473069e302976956c22c751c3d"
    )
    args = (raw, report["time_axis_labels"], report["flux_axis_labels"])
    assert frozen(*args)["reason"] == "unknown_plot_palette"
    assert analyze_planet_window(*args)["status"] == "assume_no_planet"
    assert (directory / "chart.png").read_bytes() == raw
