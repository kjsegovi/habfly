"""Conservative positive-only synthetic pixels and immutable saved chart replay."""

import hashlib
import io
import json
import socket
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from test_planet_window_policy import BLUE, crop, labels, zoomed_flux_labels

from habfly.browser_observation_progress import trace_mask
from habfly.planet_window_dip import analyze_window_dip, detector_manifest
from habfly.planet_window_policy import analyze_planet_window, policy_manifest


def png(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def positive_image(*, end=260, dips=(80, 120), bottom=30, size=(280, 195), blends=False):
    image = Image.open(io.BytesIO(crop(end=end, size=size))).convert("RGB")
    draw = ImageDraw.Draw(image)
    if blends:
        # Visible grid backgrounds reproduce the measured mechanisms, not a
        # lookup table of Dulat's unknown pixel values.
        draw.line((30, 21, 260, 21), fill=(26, 26, 26))
        draw.line((99, 20, 99, 151), fill=(51, 51, 51))
        draw.line((30, 20, end, 20), fill=BLUE)
    for x in dips:
        draw.line((x, 20, x, bottom), fill=BLUE)
    if blends:
        draw.point((79, 21), fill=(39, 52, 58))
        draw.point((79, 26), fill=(10, 17, 19))
        # Grid line at the visible1500-day x99.5 tick; background is exposed
        # below the feature, not guessed from hidden chart values.
        for y in range(22, bottom + 1):
            draw.point((99, y), fill=(62, 81, 90))
    return image


def analyze(image=None, *, maximum=5000, flux=None):
    time, normal_flux = labels(maximum)
    return analyze_window_dip(
        png(positive_image() if image is None else image), time, normal_flux if flux is None else flux
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Positive pixel replay must never contact a network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_positive_is_a_followup_not_a_planet_answer_or_measured_orbit():
    result = analyze()
    assert result["status"] == "dip_observed" and result["supported_dip_components"] == 2
    assert result["support_pixel_bounds"] == [[80, 22, 80, 30], [120, 22, 120, 30]]
    assert result["planet_decision"] is result["planet_presence"] is None
    for flag in (
        "absence_proven",
        "scientific_verified",
        "learned_perception",
        "training_label",
        "task_completed",
        "observation_completed",
    ):
        assert result[flag] is False
    assert result["answer_writes"] == result["browser_actions"] == 0
    assert not any(key in result for key in ("period", "depth", "planet_radius", "answer"))


def test_black_and_visible_grid_antialias_explanations_do_not_expand_support_mask():
    image = positive_image(dips=(80, 98), blends=True)
    raw = png(image)
    original = trace_mask(raw)
    before = original.copy()
    result = analyze(image)
    assert result["status"] == "dip_observed"
    assert result["antialias_candidates"] == result["explained_antialias_pixels"] == 11
    assert result["support_pixel_bounds"] == [[80, 22, 80, 30], [98, 22, 98, 30]]
    assert not original[26, 79] and not original[26, 99]
    assert np.array_equal(before, trace_mask(raw))
    time, flux = labels(5000)
    assert analyze_planet_window(raw, time, flux)["reason"] == "unknown_plot_palette"


@pytest.mark.parametrize(
    "color,point",
    [
        ((255, 80, 20), (79, 26)),
        ((62, 81, 90), (79, 26)),
        ((10, 17, 19), (160, 80)),
        ((80, 150, 154), (79, 26)),
    ],
)
def test_unknown_controls_or_unanchored_blends_block_even_other_good_dips(color, point):
    image = positive_image()
    ImageDraw.Draw(image).point(point, fill=color)
    assert analyze(image)["reason"] == "unknown_plot_palette"


@pytest.mark.parametrize(
    "kind", ["single", "isolated", "horizontal", "detached", "wide", "clipped", "shallow"]
)
def test_artifacts_and_weak_components_never_become_positive(kind):
    if kind == "single":
        image = positive_image(dips=(80,))
    elif kind == "shallow":
        image = positive_image(bottom=23)
    else:
        image = positive_image()
        draw = ImageDraw.Draw(image)
        if kind == "isolated":
            draw.point((160, 40), fill=BLUE)
        elif kind == "horizontal":
            draw.line((160, 40, 170, 40), fill=BLUE)
        elif kind == "detached":
            draw.line((160, 35, 160, 44), fill=BLUE)
        elif kind == "wide":
            draw.rectangle((155, 20, 165, 30), fill=BLUE)
        else:
            draw.line((160, 20, 160, 151), fill=BLUE)
    result = analyze(image)
    assert result["status"] == "insufficient_visual_evidence" and result["planet_decision"] is None


@pytest.mark.parametrize("kind", ["white", "gray", "black_gap", "blue_above", "transparent"])
def test_occlusions_gaps_and_baseline_mismatch_remain_uncertain(kind):
    image = positive_image().convert("RGBA")
    draw = ImageDraw.Draw(image)
    if kind in {"white", "gray"}:
        draw.rectangle((160, 40, 170, 50), fill=(255, 255, 255) if kind == "white" else (190, 190, 190))
    elif kind == "black_gap":
        draw.rectangle((160, 20, 170, 22), fill=(0, 0, 0))
    elif kind == "blue_above":
        # Shift100 beyond the approved baseline band, leaving the line at20.
        flux = [{**row, "center_y": row["center_y"] + 1.6} for row in labels(5000)[1]]
        assert analyze(image, flux=flux)["status"] == "insufficient_visual_evidence"
        return
    else:
        draw.point((0, 0), fill=(0, 0, 0, 0))
    assert analyze(image)["status"] == "insufficient_visual_evidence"


@pytest.mark.parametrize("kind", ["flat", "empty", "gray", "alias_only", "short_baseline"])
def test_no_original_supported_blue_dips_never_means_no(kind):
    if kind == "flat":
        image = positive_image(dips=())
    elif kind == "empty":
        image = Image.open(io.BytesIO(crop(plot=False)))
    elif kind == "gray":
        image = positive_image().convert("L").convert("RGB")
    elif kind == "short_baseline":
        image = positive_image(end=85, dips=(50, 70))
    else:
        image = positive_image(dips=())
        draw = ImageDraw.Draw(image)
        for x in (80, 120):
            draw.line((x, 21, x, 30), fill=(10, 17, 19))
    result = analyze(image)
    assert result["status"] == "insufficient_visual_evidence" and result["planet_decision"] is None


def test_partial_observation_and_zoom_are_not_turned_into_completion():
    image = positive_image(end=140, dips=(80, 120), size=(280, 196))
    result = analyze(image, flux=zoomed_flux_labels(99.99))
    assert result["status"] == "dip_observed" and not result["observation_completed"]


def test_only_first_5000_days_can_support_handoff_on_legacy_axis():
    result = analyze(positive_image(dips=(180, 220)), maximum=10000)
    assert result["status"] == "insufficient_visual_evidence"
    assert analyze(maximum=10000)["status"] == "dip_observed"


@pytest.mark.parametrize(
    "bad",
    [
        "nonlinear",
        "duplicate",
        "short",
        "nan",
        "bool",
        "negative_flux",
        "missing_100",
        "offset",
        "geometry",
        "png",
        "not_bytes",
    ],
)
def test_invalid_visible_geometry_and_evidence_fail_closed(bad):
    time, flux = labels(5000)
    raw = png(positive_image())
    if bad == "nonlinear":
        time[2]["center_x"] += 3
    elif bad == "duplicate":
        time[2] = time[1]
    elif bad == "short":
        time = time[:2]
    elif bad == "nan":
        time[1]["value"] = "nan"
    elif bad == "bool":
        time[1]["value"] = True
    elif bad == "negative_flux":
        flux = zoomed_flux_labels(-1)
    elif bad == "missing_100":
        flux = flux[:-1]
    elif bad == "offset":
        time = [{**row, "center_x": row["center_x"] + 3} for row in time]
    elif bad == "geometry":
        raw = png(Image.new("RGB", (300, 200)))
    elif bad == "png":
        raw = b"not a png"
    else:
        raw = None
    assert analyze_window_dip(raw, time, flux)["status"] == "insufficient_visual_evidence"


def test_deterministic_replay_and_frozen_negative_policy_identity():
    raw = png(positive_image())
    time, flux = labels(5000)
    result = analyze_window_dip(raw, time, flux)
    assert result == analyze_window_dip(raw, time, flux)
    assert result["chart_sha256"] == hashlib.sha256(raw).hexdigest()
    manifest = detector_manifest()
    checksum = manifest.pop("sha256")
    assert (
        checksum
        == hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    manifest["limitations"].clear()
    assert detector_manifest()["limitations"]
    assert policy_manifest()["sha256"] == "1e376eb83c896c6828aa9335ca98820e95ffbf0328ae4a6f33ce53118e16e826"
    assert analyze_planet_window(crop(), *labels())["planet_decision"] == "No"


def test_saved_dulat90_replay_without_mutation_when_available():
    directory = (
        Path(__file__).resolve().parents[1]
        / "experiments/full-stellar-probe/20260926-005/planet-window-90/progress-000"
    )
    if not directory.is_dir():
        pytest.skip("Local saved Dulat crop is not distributed")
    chart, source = (directory / "chart.png").read_bytes(), (directory / "report.json").read_bytes()
    reference = json.loads(source)
    result = analyze_window_dip(chart, reference["time_axis_labels"], reference["flux_axis_labels"])
    assert result["chart_sha256"] == "8400661c54c4586d75096d415fc8bb6632eccbf2a000b319df1a4bc38adf2033"
    assert result["status"] == "dip_observed" and result["supported_dip_components"] == 25
    assert result["antialias_candidates"] == result["explained_antialias_pixels"] == 60
    assert (
        analyze_planet_window(chart, reference["time_axis_labels"], reference["flux_axis_labels"])["reason"]
        == "unknown_plot_palette"
    )
    assert (directory / "chart.png").read_bytes() == chart and (
        directory / "report.json"
    ).read_bytes() == source
