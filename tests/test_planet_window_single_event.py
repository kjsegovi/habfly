"""Explicit single-feature shortcut only; no browser or physical-absence claim."""

import hashlib
import io
import json
import socket
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from test_planet_window_baseline_band import band_crop, encode
from test_shallow_deep_single_hint_stop import PNG_SHA, SAVED, deep_single_png
from test_shallow_two_hint_stop import axes

from habfly import planet_window_baseline_band as band
from habfly import planet_window_baseline_edge as edge
from habfly import planet_window_dip as positive
from habfly import planet_window_policy as frozen
from habfly import planet_window_single_event as subject
from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_mask


def assert_not_science(result):
    assert result["answer_writes"] == 0
    for key in (
        "absence_proven",
        "scientific_verified",
        "learned_perception",
        "training_label",
        "task_completed",
    ):
        assert result[key] is False
    assert not {"period", "depth", "mass", "radius", "training_answer"} & result.keys()


def test_explicit_shortcut_ignores_one_supported_feature_not_proven_absence(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("Offline analyzer"))
    png, (times, flux) = deep_single_png(), axes()
    originals = (bytes(png), deepcopy((times, flux)), trace_mask(png).copy())
    base = band.analyze_planet_window(png, times, flux)
    assert base["planet_decision"] is None
    report = subject.analyze_planet_window(png, times, flux)
    assert report["status"] == "assume_no_planet" and report["planet_decision"] == "No"
    assert report["reason"] == "user_approved_single_event_shortcut"
    assert report["possible_planet_ignored"] is True and report["visible_candidate_events"] == 1
    assert report["approximation"] == "user_approved_single_event_no_planet_shortcut"
    assert report["policy"] == subject.policy_manifest()
    assert report["chart_sha256"] == hashlib.sha256(png).hexdigest()
    assert report["axis_sha256"] == base["axis_sha256"]
    assert report["base_analysis_sha256"] == subject._digest(base)
    evidence = report["single_event_evidence"]
    assert evidence["original_blue_feature_bounds"] == [243, 22, 244, 28]
    assert evidence["original_blue_feature_pixels"] == 14
    assert evidence["baseline_columns"] == 230
    assert evidence["requested_columns"] == [30, 260]
    assert evidence["supported_baseline_columns"] == [30, 259]
    assert evidence["complete_original_baseline"] is True
    assert evidence["explained_antialias_pixels"] == 1
    assert evidence["additional_colored_hints"] == 0
    assert evidence["native_tooltip_confirmed"] is evidence["physical_event_count_verified"] is False
    assert evidence["image_modified"] is False
    assert_not_science(report)
    assert report == subject.analyze_planet_window(png, times, flux)
    assert png == originals[0] and (times, flux) == originals[1]
    assert (trace_mask(png) == originals[2]).all()
    assert band.analyze_planet_window(png, times, flux) == base


@pytest.mark.parametrize(
    "mutation",
    [
        "second_deep_feature",
        "second_shallow_hint",
        "second_faint_hint",
        "unknown_color",
        "bright_overlay",
        "midgray_overlay",
        "transparent",
        "wrong_size",
        "first_gap",
        "internal_gap",
        "incomplete_end",
        "detached_feature",
        "clipped_feature",
        "wide_feature",
        "edge_feature",
        "unknown_final_border",
        "bright_final_border",
    ],
)
def test_unsupported_pixels_never_become_single_event_no(mutation):
    picture = Image.open(io.BytesIO(deep_single_png())).convert("RGB")
    draw = ImageDraw.Draw(picture)
    if mutation == "second_deep_feature":
        draw.rectangle((100, 21, 101, 28), fill=(50, 82, 96))
    elif mutation == "second_shallow_hint":
        picture.putpixel((100, 21), (50, 82, 96))
    elif mutation == "second_faint_hint":
        picture.putpixel((100, 22), (5, 8, 10))
    elif mutation == "unknown_color":
        picture.putpixel((244, 21), (80, 0, 80))
    elif mutation in {"bright_overlay", "midgray_overlay"}:
        draw.rectangle((100, 60, 120, 70), fill=(100 if mutation == "midgray_overlay" else 255,) * 3)
    elif mutation == "transparent":
        picture = picture.convert("RGBA")
        picture.putpixel((1, 1), (0, 0, 0, 254))
    elif mutation == "wrong_size":
        picture = picture.resize((281, 196))
    elif mutation in {"first_gap", "internal_gap", "incomplete_end"}:
        x0, x1 = {"first_gap": (30, 30), "internal_gap": (100, 100), "incomplete_end": (254, 260)}[mutation]
        draw.rectangle((x0, 20, x1, 23), fill=(0, 0, 0))
    elif mutation == "detached_feature":
        draw.line((243, 21, 244, 21), fill=(0, 0, 0))
    elif mutation == "clipped_feature":
        draw.rectangle((243, 21, 244, 151), fill=(50, 82, 96))
    elif mutation in {"wide_feature", "edge_feature"}:
        x0, x1 = (100, 104) if mutation == "wide_feature" else (31, 32)
        draw.rectangle((x0, 21, x1, 28), fill=(50, 82, 96))
    else:
        draw.line(
            (260, 20, 260, 151), fill=(10, 20, 30) if mutation == "unknown_final_border" else (100, 100, 100)
        )
    result = subject.analyze_planet_window(encode(picture), *axes())
    assert result["planet_decision"] is None
    assert result["status"] != "assume_no_planet"
    assert "single_event_evidence" not in result
    assert "possible_planet_ignored" not in result
    assert_not_science(result)


@pytest.mark.parametrize(
    "mutation",
    ["missing_time", "nonlinear_time", "bool_time", "nan_flux", "bad_baseline", "few_flux", "long_window"],
)
def test_axes_and_completed_5000_scope_are_not_inferred(mutation):
    times, flux = axes()
    png = deep_single_png()
    if mutation == "missing_time":
        times = times[1:]
    elif mutation == "nonlinear_time":
        times[3]["center_x"] += 5
    elif mutation == "bool_time":
        times[0]["value"] = False
    elif mutation == "nan_flux":
        flux[0]["center_y"] = float("nan")
    elif mutation == "bad_baseline":
        flux[-1]["value"] = "99"
    elif mutation == "few_flux":
        flux = flux[-2:]
    else:
        # Keep the feature inside the first5000 of a10000axis. It must not use
        # the new shortcut, which only accepts the complete0..5000 overview.
        times = [{**row, "value": str(int(row["value"]) * 2)} for row in times]
        picture = Image.open(io.BytesIO(png)).convert("RGB")
        ImageDraw.Draw(picture).rectangle((100, 21, 101, 28), fill=(50, 82, 96))
        png = encode(picture)
    result = subject.analyze_planet_window(png, times, flux)
    assert result["planet_decision"] is None
    assert_not_science(result)


@pytest.mark.parametrize("days", [True, 5000.0, 0, 10000, "5000"])
def test_exact_requested_limit_required(days):
    result = subject.analyze_planet_window(deep_single_png(), *axes(), requested_days=days)
    assert result["planet_decision"] is None


def test_existing_band_no_is_retained_without_claiming_ignored_feature():
    png, (times, flux) = encode(band_crop()), axes()
    base = band.analyze_planet_window(png, times, flux)
    result = subject.analyze_planet_window(png, times, flux)
    assert base["planet_decision"] == result["planet_decision"] == "No"
    assert base["reason"] == result["reason"] == "user_approved_window_no_visible_dip"
    assert (
        not {"possible_planet_ignored", "single_event_evidence", "visible_candidate_events"} & result.keys()
    )
    assert_not_science(result)


@pytest.mark.parametrize("module", [frozen, edge, band])
def test_historical_dispatch_and_results_remain_exact(module):
    raw, (times, flux) = deep_single_png(), axes()
    policy = module.policy_manifest()
    assert subject.recorded_policy_manifest(policy) == policy
    assert subject.analyze_recorded_planet_window(
        raw, times, flux, policy=policy
    ) == module.analyze_planet_window(raw, times, flux)
    assert subject.recorded_policy_manifest() == frozen.policy_manifest()
    assert subject.analyze_recorded_planet_window(raw, times, flux) == frozen.analyze_planet_window(
        raw, times, flux
    )


@pytest.mark.parametrize("mutation", ["version", "sha", "bool_alias", "extra", "missing", "mixed"])
def test_unknown_or_mixed_full_manifest_rejected(mutation):
    policy = subject.policy_manifest()
    if mutation == "version":
        policy["version"] = "unsupported"
    elif mutation == "sha":
        policy["sha256"] = "0" * 64
    elif mutation == "bool_alias":
        policy["user_approved"] = 1
    elif mutation == "extra":
        policy["extra"] = False
    elif mutation == "missing":
        del policy["single_event"]
    else:
        policy["base_policy_sha256"] = edge.policy_manifest()["sha256"]
    with pytest.raises(BrowserSafetyStop, match="unsupported_planet_window_policy"):
        subject.recorded_policy_manifest(policy)
    with pytest.raises(BrowserSafetyStop, match="unsupported_planet_window_policy"):
        subject.analyze_recorded_planet_window(deep_single_png(), *axes(), policy=policy)


def test_new_manifest_detached_and_explicit_dispatch_matches_direct():
    policy = subject.policy_manifest()
    result = subject.analyze_recorded_planet_window(deep_single_png(), *axes(), policy=policy)
    assert result == subject.analyze_planet_window(deep_single_png(), *axes())
    policy["single_event"]["feature"] = "changed"
    assert policy != subject.policy_manifest()


def test_frozen_sources_are_unchanged():
    expected = {
        frozen: "47db07078c9bb73cca0de4c5a378e9fb8835690e52d7826db1280533b15131d0",
        edge: "4622dd3d15b8c61a48ee1d6629fad72a190ccc3eebbca1cbca90a817838879b5",
        band: "f1b1a8b0cb042fb64e16e9d563af186d5bcd4b2a07441e5fbce0dddfe0638460",
        positive: "b1f2dc44ea848a92964b9bfc699edde40d610cac63c4d8d56ea7f43660a3e79c",
    }
    for module, expected_sha in expected.items():
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected_sha


def test_optional_exact_jarlarth_crop_only_changes_new_optin_decision():
    if not SAVED.is_dir():
        pytest.skip("Optional immutable public Jarlarth capture not distributed")
    original = {path.name: path.read_bytes() for path in SAVED.iterdir() if path.is_file()}
    raw, report = original["chart.png"], json.loads(original["report.json"])
    assert hashlib.sha256(raw).hexdigest() == PNG_SHA
    assert report["star"] == "JARLARTH"
    result = subject.analyze_planet_window(raw, report["time_axis_labels"], report["flux_axis_labels"])
    assert result["reason"] == "user_approved_single_event_shortcut"
    assert result["single_event_evidence"]["original_blue_feature_bounds"] == [243, 22, 244, 28]
    assert (
        subject.analyze_recorded_planet_window(
            raw, report["time_axis_labels"], report["flux_axis_labels"], policy=band.policy_manifest()
        )["planet_decision"]
        is None
    )
    assert {path.name: path.read_bytes() for path in SAVED.iterdir() if path.is_file()} == original
