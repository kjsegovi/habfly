"""Quantized public-pixel hints only; no browser, answers, model or new period."""

import hashlib
import io
import json
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image
from test_browser_shallow_transit_probe import flux, low_intensity_png, png, sample
from test_planet_tooltip_reference import _rewrite_hint_metadata, fixture, load

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_mask
from habfly.browser_shallow_transit_probe import (
    FIRST_THREE_HINT_POLICY,
    FIRST_THREE_LOW_INTENSITY_HINT_POLICY,
    FIRST_THREE_QUANTIZED_HINT_POLICY,
    _quantized_gray_feasible,
    candidate_columns,
    matches_overview_hint_metadata,
    overview_hint_metadata,
    summarize_samples,
)
from habfly.planet_tooltip_reference import measurement_manifest
from habfly.planet_window_dip import detector_manifest
from habfly.planet_window_policy import policy_manifest

V3 = FIRST_THREE_QUANTIZED_HINT_POLICY
V2 = FIRST_THREE_LOW_INTENSITY_HINT_POLICY


def quantized_png(colors=((5, 8, 10), (10, 17, 19), (53, 56, 58)), *, support=True, neutral=True):
    picture = Image.open(io.BytesIO(png(baseline=20, features=()))).convert("RGB")
    for x, color, gray in zip((50, 100, 150), colors, (0, 0, 51), strict=True):
        for y in range(20, 23):
            for column in range(x - 1, x + 2):
                picture.putpixel((column, y), (gray,) * 3 if neutral else (3, 2, 1))
        picture.putpixel((x, 20), (80, 132, 154) if support else (26, 26, 26))
        picture.putpixel((x, 21), color)
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize(
    "rgb,alpha,gray",
    [
        ((5, 8, 10), Fraction(63, 1000), 0),
        ((10, 17, 19), Fraction(126, 1000), 0),
        ((53, 56, 58), Fraction(65, 1000), 51),
    ],
)
def test_independent_common_blend_witnesses_within_exact_half_channel_bins(rgb, alpha, gray):
    # Known witnesses are independent of the interval-elimination implementation.
    predicted = [alpha * channel + (1 - alpha) * gray for channel in (80, 132, 154)]
    assert all(
        abs(observed - value) <= Fraction(1, 2) for observed, value in zip(rgb, predicted, strict=True)
    )
    assert _quantized_gray_feasible(rgb, gray, gray)


@pytest.mark.parametrize("rgb,gray", [((5, 8, 11), 0), ((5, 10, 10), 0), ((53, 56, 60), 51)])
def test_independent_channel_fits_do_not_substitute_for_one_shared_alpha(rgb, gray):
    for value, direction in zip(rgb, (80, 132, 154), strict=True):
        low = max(Fraction(3, 50), Fraction(2 * value - 1 - 2 * gray, 2 * (direction - gray)))
        high = min(Fraction(1, 4), Fraction(2 * value + 1 - 2 * gray, 2 * (direction - gray)))
        assert low <= high  # each channel alone has a fit; the common RGB tuple does not
    assert not _quantized_gray_feasible(rgb, gray, gray)
    assert not _quantized_gray_feasible(rgb, 0, 51)


def test_v3_handles_only_context_rounding_without_reclassifying_frozen_trace():
    raw = quantized_png()
    assert candidate_columns(raw, flux(20.45), hint_policy=V3) == [[50], [100], [150]]
    with pytest.raises(BrowserSafetyStop, match="unsupported_low_intensity_hint_context"):
        candidate_columns(raw, flux(20.45), hint_policy=V2)
    assert not trace_mask(raw)[21, [50, 100, 150]].any()
    assert candidate_columns(raw, flux(20.45)) == []


@pytest.mark.parametrize("options", [{"support": False}, {"neutral": False}])
def test_missing_original_support_or_observed_gray_still_blocks(options):
    with pytest.raises(BrowserSafetyStop, match="unsupported_low_intensity_hint_context"):
        candidate_columns(quantized_png(**options), flux(20.45), hint_policy=V3)


@pytest.mark.parametrize("pixel", [(5, 8, 11), (5, 10, 10)])
def test_incompatible_first_candidate_is_not_dropped_for_later_valid_hints(pixel):
    raw = low_intensity_png(pixels={40: pixel, 70: (29, 36, 38), 100: (29, 36, 38), 130: (29, 36, 38)})
    with pytest.raises(BrowserSafetyStop, match="unsupported_low_intensity_hint_context"):
        candidate_columns(raw, flux(20.45), hint_policy=V3)


def test_v3_does_not_promote_pixels_outside_unchanged_raw_candidate_eligibility():
    raw = low_intensity_png(pixels={40: (53, 56, 60), 70: (29, 36, 38), 100: (29, 36, 38), 130: (29, 36, 38)})
    assert candidate_columns(raw, flux(20.45), hint_policy=V3) == [[70], [100], [130]]
    assert candidate_columns(raw, flux(20.45), hint_policy=V2) == [[70], [100], [130]]


@pytest.mark.parametrize("first", [[31], list(range(40, 47))])
def test_invalid_first_group_is_never_split_or_skipped(first):
    raw = low_intensity_png(pixels={x: (5, 8, 10) for x in first + [70, 100, 130]})
    with pytest.raises(BrowserSafetyStop, match="broad_or_edge_feature"):
        candidate_columns(raw, flux(20.45), hint_policy=V3)


def test_v3_cannot_relax_focused_palette_or_native_bracket_requirements():
    with pytest.raises(BrowserSafetyStop, match="invalid_overview_hint_policy"):
        candidate_columns(png(baseline=80), flux(80), overview=False, hint_policy=V3)
    for samples in ([sample(1), sample(2), sample(3)], [sample(1, "99"), sample(2, "98"), sample(3)]):
        result = summarize_samples(samples)
        assert result["status"] == "unresolved" and result["planet_decision"] is None
        assert result["answer_authorized"] is False


def test_historical_recipes_and_frozen_detector_measurement_hashes_unchanged():
    assert overview_hint_metadata(None) == {}
    assert overview_hint_metadata(FIRST_THREE_HINT_POLICY) == {
        "overview_hint_policy": FIRST_THREE_HINT_POLICY
    }
    assert (
        overview_hint_metadata(V2)["overview_hint_recipe"]["sha256"]
        == "76a1576ab9b3cc5b2ef6b150f59f978342e0dd6686eda5252419bf5ed15799bd"
    )
    assert (
        overview_hint_metadata(V3)["overview_hint_recipe"]["sha256"]
        == "e22aa71a3e41e0813704c869de30c23ed667969fd764bb29d00956e6a75d2582"
    )
    assert policy_manifest()["sha256"] == "1e376eb83c896c6828aa9335ca98820e95ffbf0328ae4a6f33ce53118e16e826"
    assert detector_manifest()["sha256"] == "fbd28b7dacbb2239ec9a41de8aaff6a5e5dadb2882586d5fab48c5da3df480e2"
    assert (
        measurement_manifest()["sha256"] == "50e38151704221593d91feb949c2a40927bcb455cb5874f769333c62171f3416"
    )
    assert candidate_columns(low_intensity_png(), flux(20.45), hint_policy=V2) == [[57], [86], [115]]
    assert candidate_columns(low_intensity_png(), flux(20.45), hint_policy=V3) == [[57], [86], [115]]


@pytest.mark.parametrize(
    "key,value",
    [
        ("answer_authorized", 0),
        ("maximum_channel", 64.0),
        ("maximum_channel_residual", True),
        ("sha256", "0" * 64),
    ],
)
def test_v3_metadata_has_strict_types_and_hash(key, value):
    metadata = overview_hint_metadata(V3)
    assert matches_overview_hint_metadata(metadata, V3)
    metadata["overview_hint_recipe"][key] = value
    assert not matches_overview_hint_metadata(metadata, V3)


def test_saved_kanivermyr_exact_crop_axis_and_ordered_search_groups():
    root = (
        Path(__file__).resolve().parents[1]
        / "experiments/browser-project-supplied-three-star/7d6840f4339d49ba8fe8324c9fde5e74/campaign/stars/002/owner/star/shallow/initial/progress"
    )
    if not root.exists():
        pytest.skip("Optional immutable public Kanivermyr capture unavailable")
    raw = (root / "chart.png").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest() == "24292999f3348821a9c03a7d76e0f150fb00db1fbf7bd2390d60ed53b0c07206"
    )
    report = json.loads((root / "report.json").read_bytes())
    assert report["star"] == "KANIVERMYR" and report["endpoint_visible"] is True
    assert candidate_columns(raw, report["flux_axis_labels"], hint_policy=V3) == [[44, 45], [60, 61], [76]]
    with pytest.raises(BrowserSafetyStop, match="unsupported_low_intensity_hint_context"):
        candidate_columns(raw, report["flux_axis_labels"], hint_policy=V2)
    assert candidate_columns(raw, report["flux_axis_labels"], hint_policy=FIRST_THREE_HINT_POLICY) == [
        [45],
        [60],
        [76],
    ]


@pytest.fixture
def v3_bundle(tmp_path):
    return tmp_path, [
        fixture(
            tmp_path, f"v3-{i}", 100 * (i + 1), linked=True, overview_hint_index=i, overview_hint_policy=V3
        )
        for i in range(3)
    ]


def test_recorded_v3_rebuilds_exact_recipe_without_promoting_authority(v3_bundle):
    result = load(v3_bundle)
    assert result["period_days"]["value"] == "100"
    assert result["overview_hint_positions_used"] is False and result["answer_authorized"] is False
    assert result["brightness_drop_percent"]["physical_bounds"] is None
    metadata = overview_hint_metadata(V3)
    assert all(
        all(feature[key] == value for key, value in metadata.items()) for feature in result["features"]
    )
    assert result == load(v3_bundle)


@pytest.mark.parametrize("policy", [None, FIRST_THREE_HINT_POLICY, V2])
def test_mixed_historical_and_v3_bundle_never_retags_old_proof(v3_bundle, policy):
    root, specs = v3_bundle
    specs[2] = fixture(
        root,
        "historical-third",
        300,
        linked=True,
        overview_hint_index=None if policy is None else 2,
        overview_hint_policy=policy,
    )
    with pytest.raises(BrowserSafetyStop, match="mixed_or_nonconsecutive_overview_hints"):
        load(v3_bundle)


def test_rehashed_v3_recipe_mutation_fails_strict_reader(v3_bundle):
    recipe = overview_hint_metadata(V3)["overview_hint_recipe"]
    recipe["maximum_channel_residual"] = True
    _rewrite_hint_metadata(v3_bundle, 0, overview_hint_recipe=recipe)
    with pytest.raises(BrowserSafetyStop, match="unsupported_overview_hint_recipe"):
        load(v3_bundle)
