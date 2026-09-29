"""Coarse 5000-day No assumptions, not physical absence or native acceptance.

Synthetic PNGs exercise strict original-pixel guards. Optional saved examples
only read immutable student-visible crops; they never resume their old owners.
"""

import hashlib
import io
import json
import socket
from copy import deepcopy
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
from random import Random

import pytest
from PIL import Image, ImageDraw
from test_planet_window_policy import BLUE, crop, labels

from habfly.browser_observation_progress import trace_mask
from habfly.planet_window_baseline_band import _blend_interval, analyze_planet_window, policy_manifest
from habfly.planet_window_baseline_edge import analyze_planet_window as edge
from habfly.planet_window_baseline_edge import policy_manifest as edge_manifest
from habfly.planet_window_dip import detector_manifest
from habfly.planet_window_policy import analyze_planet_window as frozen
from habfly.planet_window_policy import policy_manifest as frozen_manifest


def encode(picture):
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def band_crop(*, colors=((29, 36, 38), (33, 46, 51)), end=260):
    picture = Image.open(io.BytesIO(crop(end=end, size=(280, 196)))).convert("RGB")
    for x, color in zip(range(100, 100 + len(colors)), colors, strict=True):
        picture.putpixel((x, 21), color)
    return picture


def analyze(picture=None, *, times=None, fluxes=None, requested_days=5000):
    default_times, default_fluxes = labels(5000)
    return analyze_planet_window(
        encode(band_crop() if picture is None else picture),
        default_times if times is None else times,
        default_fluxes if fluxes is None else fluxes,
        requested_days=requested_days,
    )


def assert_no_authority(result):
    assert result["provenance"] == "reference_prediction"
    assert result["answer_writes"] == 0
    for flag in (
        "absence_proven",
        "scientific_verified",
        "learned_perception",
        "training_label",
        "task_completed",
    ):
        assert result[flag] is False
    assert not any(
        key in result for key in ("period", "depth", "planet_mass", "planet_radius", "training_answer")
    )


def test_supported_faint_band_only_changes_explicit_approximate_decision(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("No network allowed"))
    png = encode(band_crop())
    original = bytes(png)
    times, fluxes = labels(5000)
    axes = deepcopy((times, fluxes))
    baseline = frozen(png, times, fluxes)
    before_mask = trace_mask(png).copy()
    assert baseline["reason"] == "unknown_plot_palette" and baseline["planet_decision"] is None
    prior_edge = edge(png, times, fluxes)
    assert prior_edge["planet_decision"] is None
    result = analyze_planet_window(png, times, fluxes)
    assert result["status"] == "assume_no_planet" and result["planet_decision"] == "No"
    assert result["observation_limit_days"] == 5000
    assert result["examined_day_interval"] == [0, 5000]
    assert result["policy"] == policy_manifest()
    assert result["policy"]["version"] == "user_approved_5000_day_baseline_band_no_dip_v1"
    assert result["chart_sha256"] == hashlib.sha256(png).hexdigest()
    assert result["base_analysis_sha256"] == digest(prior_edge)
    assert result["white_cursor_gap_columns"] == 0
    assert result["baseline_band_evidence"]
    evidence = result["baseline_band_evidence"]
    assert evidence["extra_pixel_rows"] == [21] and evidence["support_rows"] == [20]
    assert evidence["unknown_pixels"] == 2
    assert evidence["coverage_from_original_blue_only"] is True
    assert evidence["subpixel_features_not_resolved"] is True
    assert evidence["image_modified"] is False
    assert sum(group["pixels"] for group in evidence["groups"]) == 2
    assert {tuple(group["rgb"]) for group in evidence["groups"]} == {(29, 36, 38), (33, 46, 51)}
    assert all(
        Fraction(3, 50)
        <= Fraction(group["alpha_interval"][0])
        <= Fraction(group["alpha_interval"][1])
        <= Fraction(1, 2)
        for group in evidence["groups"]
    )
    assert result == analyze_planet_window(png, times, fluxes)
    assert frozen(png, times, fluxes) == baseline
    assert png == original and (times, fluxes) == axes
    assert (trace_mask(png) == before_mask).all()
    assert_no_authority(result)


@pytest.mark.parametrize("color", [(29, 36, 38), (33, 46, 51), (41, 57, 64)])
def test_saved_public_rgb_examples_have_supported_original_blue(color):
    picture = band_crop(colors=(color,))
    raw = encode(picture)
    assert trace_mask(raw)[20, 100] and not trace_mask(raw)[21, 100]
    result = analyze(picture)
    assert result["planet_decision"] == "No"
    assert_no_authority(result)


@pytest.mark.parametrize(
    "kind",
    [
        "support_gap",
        "first_column_gap",
        "internal_gap",
        "end_gap",
        "all_trace_missing",
        "cursor_gap",
        "blue_deeper",
        "faint_deeper",
        "unknown_deeper",
        "chained_support",
        "blue_above",
        "unknown_above",
        "transparent_plot",
        "transparent_outside_plot",
        "white_overlay",
        "gray_overlay",
        "bright_below",
        "chromatic_below",
        "bright_final_border",
        "unknown_final_border",
        "wrong_size",
        "baseline_missing",
    ],
)
def test_unsupported_geometry_coverage_or_overlay_cannot_authorize_no(kind):
    picture = band_crop()
    draw = ImageDraw.Draw(picture)
    if kind == "support_gap":
        picture.putpixel((100, 20), (0, 0, 0))
    elif kind == "first_column_gap":
        draw.rectangle((30, 20, 30, 23), fill=(0, 0, 0))
    elif kind == "internal_gap":
        draw.rectangle((120, 20, 120, 23), fill=(0, 0, 0))
    elif kind == "end_gap":
        draw.rectangle((254, 20, 260, 23), fill=(0, 0, 0))
    elif kind == "all_trace_missing":
        draw.line((30, 20, 260, 20), fill=(0, 0, 0))
    elif kind == "cursor_gap":
        draw.rectangle((120, 19, 122, 23), fill=(255, 255, 255))
    elif kind == "blue_deeper":
        draw.line((140, 22, 140, 24), fill=BLUE)
    elif kind == "faint_deeper":
        picture.putpixel((140, 22), (29, 36, 38))
    elif kind == "unknown_deeper":
        picture.putpixel((140, 80), (33, 46, 51))
    elif kind == "chained_support":
        picture.putpixel((100, 22), (29, 36, 38))
    elif kind == "blue_above":
        picture.putpixel((140, 20), BLUE)
        # Shift every original column out of the allowed baseline band using axes below.
        return assert_bad_baseline(picture)
    elif kind == "unknown_above":
        picture.putpixel((140, 20), (29, 36, 38))
    elif kind.startswith("transparent"):
        picture = picture.convert("RGBA")
        picture.putpixel((140, 80) if kind == "transparent_plot" else (2, 2), (0, 0, 0, 254))
    elif kind in {"white_overlay", "gray_overlay"}:
        draw.rectangle((130, 65, 150, 85), fill=(255 if kind == "white_overlay" else 100,) * 3)
    elif kind == "bright_below":
        picture.putpixel((100, 24), (65, 65, 65))
    elif kind == "chromatic_below":
        picture.putpixel((100, 24), (10, 20, 30))
    elif kind == "bright_final_border":
        draw.line((260, 20, 260, 151), fill=(255, 255, 255))
    elif kind == "unknown_final_border":
        draw.line((260, 20, 260, 151), fill=(33, 46, 51))
    elif kind == "wrong_size":
        picture = picture.resize((281, 196))
    else:
        draw.line((30, 20, 260, 20), fill=(40, 40, 40))
    result = analyze(picture)
    assert result["planet_decision"] is None and result["status"] != "assume_no_planet"
    assert_no_authority(result)


def assert_bad_baseline(picture):
    times, fluxes = labels(5000)
    # The 100% reference remains in the legacy19..22 range; y20.5 trace is1.6px above it.
    fluxes = [{**row, "center_y": row["center_y"] + 1.6} for row in fluxes]
    assert analyze(picture, times=times, fluxes=fluxes)["planet_decision"] is None


@pytest.mark.parametrize(
    "color", [(64, 0, 64), (64, 0, 0), (0, 64, 0), (0, 64, 64), (0, 0, 64), (45, 88, 105), (0, 5, 9)]
)
def test_non_blue_or_unsupported_blend_is_not_repaired(color):
    picture = band_crop(colors=(color,))
    assert analyze(picture)["planet_decision"] is None


@pytest.mark.parametrize(
    "kind",
    [
        "no_times",
        "few_times",
        "duplicate_times",
        "nonlinear_times",
        "shifted_times",
        "nan_times",
        "bool_time",
        "missing_origin",
        "no_flux",
        "few_flux",
        "duplicate_flux",
        "nonlinear_flux",
        "wrong_baseline",
        "nan_flux",
        "bool_flux",
        "beyond_5000",
        "float_days",
        "bool_days",
    ],
)
def test_axes_and_exact_observation_cap_remain_required(kind):
    times, fluxes = labels(5000)
    days = 5000
    if kind == "no_times":
        times = []
    elif kind == "few_times":
        times = times[:2]
    elif kind == "duplicate_times":
        times[1] = dict(times[0])
    elif kind == "nonlinear_times":
        times[4]["center_x"] += 5
    elif kind == "shifted_times":
        times = [{**row, "center_x": row["center_x"] + 5} for row in times]
    elif kind == "nan_times":
        times[3]["center_x"] = float("nan")
    elif kind == "bool_time":
        times[0]["value"] = False
    elif kind == "missing_origin":
        times = times[1:]
    elif kind == "no_flux":
        fluxes = []
    elif kind == "few_flux":
        fluxes = fluxes[-2:]
    elif kind == "duplicate_flux":
        fluxes[1] = dict(fluxes[0])
    elif kind == "nonlinear_flux":
        fluxes[4]["center_y"] += 5
    elif kind == "wrong_baseline":
        fluxes[-1]["value"] = "99"
    elif kind == "nan_flux":
        fluxes[3]["center_y"] = float("nan")
    elif kind == "bool_flux":
        fluxes[-1]["value"] = True
    else:
        days = {"beyond_5000": 10000, "float_days": 5000.0, "bool_days": True}[kind]
    result = analyze(times=times, fluxes=fluxes, requested_days=days)
    assert result["planet_decision"] is None
    assert_no_authority(result)


@pytest.mark.parametrize("end", [40, 110, 200, 251])
def test_incomplete_original_trace_never_becomes_complete(end):
    result = analyze(band_crop(end=end))
    assert result["planet_decision"] is None


def test_one_dark_final_border_allowed_but_never_internal_gap():
    picture = band_crop(end=259)
    picture.putpixel((260, 20), (53, 56, 58))
    assert analyze(picture)["planet_decision"] == "No"
    picture.putpixel((100, 20), (53, 56, 58))
    assert analyze(picture)["planet_decision"] is None


def test_first_5000_only_on_10000_axis_never_ignores_an_in_window_feature():
    picture = band_crop()
    times, fluxes = labels(10000)
    ImageDraw.Draw(picture).line((180, 20, 180, 40), fill=BLUE)
    assert analyze(picture, times=times, fluxes=fluxes)["planet_decision"] == "No"
    picture.putpixel((140, 23), BLUE)
    assert analyze(picture, times=times, fluxes=fluxes)["planet_decision"] is None


def polygon_oracle(rgb):
    """Independent exact 2D vertex feasibility, not production bound elimination.

    Let q=(1-alpha)*gray. Each (a,b,c) means a*alpha+b*q<=c.
    The compact polygon includes alpha and visible neutral-domain constraints.
    """
    half = Fraction(1, 2)
    limits = [(-1, 0, -Fraction(3, 50)), (1, 0, half), (0, -1, 0), (64, 1, 64)]
    for channel, blue in zip(rgb, (80, 132, 154), strict=True):
        limits.extend([(blue, 1, Fraction(channel) + half), (-blue, -1, -Fraction(channel) + half)])
    feasible = []
    for (a, b, c), (d, e, f) in combinations(limits, 2):
        determinant = a * e - b * d
        if not determinant:
            continue
        alpha = Fraction(c * e - b * f, determinant)
        q = Fraction(a * f - c * d, determinant)
        if all(x * alpha + y * q <= z for x, y, z in limits):
            feasible.append(alpha)
    return (min(feasible), max(feasible)) if feasible else None


def test_blend_interval_matches_independent_rational_polygon_oracle():
    rng = Random(20260928)
    colors = {(29, 36, 38), (33, 46, 51), (41, 57, 64), (0, 64, 64), (45, 88, 105)}
    for alpha, gray in product(
        (Fraction(3, 50), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2)), (0, 26, 51, 64)
    ):
        colors.add(tuple(round(alpha * blue + (1 - alpha) * gray) for blue in (80, 132, 154)))
    colors.update(tuple(rng.randrange(111) for _ in range(3)) for _ in range(64))
    assert len(colors) == 85
    for rgb in sorted(colors):
        assert _blend_interval(rgb) == polygon_oracle(rgb), rgb


def test_manifest_is_complete_separate_detached_and_hash_deterministic():
    first = policy_manifest()
    assert first["sha256"] == digest({key: value for key, value in first.items() if key != "sha256"})
    assert first["sha256"] not in {
        frozen_manifest()["sha256"],
        edge_manifest()["sha256"],
        detector_manifest()["sha256"],
    }
    previous = deepcopy(first)
    first["limitations"].clear()
    assert policy_manifest() == previous
    assert frozen_manifest()["sha256"] == "1e376eb83c896c6828aa9335ca98820e95ffbf0328ae4a6f33ce53118e16e826"
    assert edge_manifest()["sha256"] == "d3013611379fab5aab8905f956d62cf8f19c75510602f91ed24908f0f158f470"
    assert detector_manifest()["sha256"] == "fbd28b7dacbb2239ec9a41de8aaff6a5e5dadb2882586d5fab48c5da3df480e2"


@pytest.mark.parametrize("raw", [b"bad", b"", None])
def test_malformed_png_fails_closed(raw):
    times, fluxes = labels(5000)
    if raw is None:
        # Matches the original typed byte-input contract; callers must not treat an exception as No.
        with pytest.raises(TypeError):
            analyze_planet_window(raw, times, fluxes)
    else:
        assert analyze_planet_window(raw, times, fluxes)["planet_decision"] is None


@pytest.mark.parametrize("raw", [crop(), crop(dip=(70, 40)), crop(end=110), b"bad"])
def test_other_frozen_outcomes_remain_exact_except_new_policy_identity(raw):
    times, fluxes = labels(5000)
    old = frozen(raw, times, fluxes)
    result = analyze_planet_window(raw, times, fluxes)
    assert result["base_analysis_sha256"] == digest(edge(raw, times, fluxes))
    assert {key: result[key] for key in old if key != "policy"} == {
        key: value for key, value in old.items() if key != "policy"
    }
    assert "baseline_band_evidence" not in result


ROOT = Path(__file__).resolve().parents[1]
SAVED = (
    (
        "experiments/browser-project-two-event-three-star/1c1e53a561e74d3c9fcc89e369e50987/campaign/stars/002/owner/star/window/progress-013",
        "20c85ad0e800754bfa58ced35a79200b4b8385ae713d38bef5497a1c0c24f8e8",
        "fc4a8a373e341e5ac07fd7501c1640c741fabb95ed25bbcad55cb09c04a30db7",
    ),
    (
        "experiments/browser-project-supplied-three-star/736e75273b6c424d970dc3c17a95dbc2/campaign/stars/001/owner/star/shallow/initial/progress",
        "2e9c16bf4488e3cb587ba33b2f3ab3d6349c061e960e3976f48f4c357ea74cc5",
        "c20feb595199555c81484f499d4fbc9096ec963eb3693d57d3dd85bc394eaec8",
    ),
)


@pytest.mark.parametrize("directory,png_sha,report_sha", SAVED)
def test_optional_original_public_crops_keep_exact_old_report_and_only_new_assumption(
    directory, png_sha, report_sha
):
    path = ROOT / directory
    if not path.is_dir():
        pytest.skip("Saved public crop is optional; mandatory synthetic regressions cover the contract")
    png, raw = (path / "chart.png").read_bytes(), (path / "report.json").read_bytes()
    assert hashlib.sha256(png).hexdigest() == png_sha
    assert hashlib.sha256(raw).hexdigest() == report_sha
    report = json.loads(raw)
    times, fluxes = report["time_axis_labels"], report["flux_axis_labels"]
    old, prior_edge = frozen(png, times, fluxes), edge(png, times, fluxes)
    old_bytes = json.dumps(old, sort_keys=True)
    result = analyze_planet_window(png, times, fluxes)
    assert old["reason"] == "unknown_plot_palette" and old["planet_decision"] is None
    assert prior_edge["planet_decision"] is None
    assert result["planet_decision"] == "No" and result["baseline_band_evidence"]
    assert result["base_analysis_sha256"] == digest(prior_edge)
    assert_no_authority(result)
    assert json.dumps(frozen(png, times, fluxes), sort_keys=True) == old_bytes
    assert (path / "chart.png").read_bytes() == png and (path / "report.json").read_bytes() == raw
