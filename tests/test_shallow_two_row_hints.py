"""V4 search-context tests only: no browser, scientific answer or model run."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_browser_shallow_transit_probe import flux, png, sample
from test_planet_tooltip_reference import _rewrite_hint_metadata, fixture, load

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_mask
from habfly.browser_shallow_transit_probe import (
    FIRST_THREE_HINT_POLICY as V1,
)
from habfly.browser_shallow_transit_probe import (
    FIRST_THREE_LOW_INTENSITY_HINT_POLICY as V2,
)
from habfly.browser_shallow_transit_probe import (
    FIRST_THREE_QUANTIZED_HINT_POLICY as V3,
)
from habfly.browser_shallow_transit_probe import (
    FIRST_THREE_TWO_ROW_HINT_POLICY as V4,
)
from habfly.browser_shallow_transit_probe import (
    _low_intensity_hints,
    _quantized_hint_context,
    _two_row_hint_context,
    candidate_columns,
    matches_overview_hint_metadata,
    overview_hint_metadata,
    summarize_samples,
)
from habfly.planet_tooltip_reference import measurement_manifest
from habfly.planet_window_baseline_edge import analyze_planet_window as baseline_no
from habfly.planet_window_baseline_edge import policy_manifest as baseline_manifest
from habfly.planet_window_dip import detector_manifest
from habfly.planet_window_policy import analyze_planet_window as frozen_no
from habfly.planet_window_policy import policy_manifest


def two_row_png(*, columns=(50, 100, 150), change=None):
    picture = Image.open(io.BytesIO(png(baseline=20, features=()))).convert("RGB")
    for x in range(31, 261):
        picture.putpixel((x, 21), (26, 26, 26))
    for x in columns:
        picture.putpixel((x, 21), (32, 39, 42))
        picture.putpixel((x, 22), (10, 17, 19))
    if change:
        change(picture, columns[0])
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    return stream.getvalue()


def context_inputs():
    colors = np.zeros((7, 5, 3), dtype=float)
    colors[2, 1] = colors[3, 1] = (26, 26, 26)
    colors[2, 2] = (80, 132, 154)
    colors[3, 2] = (32, 39, 42)
    colors[4, 2] = colors[5, 2] = (10, 17, 19)
    original = np.zeros((7, 5), dtype=bool)
    original[2, 2] = True
    candidates = np.zeros_like(original)
    candidates[3:6, 2] = True
    return colors, original, candidates


def test_one_immutable_v3_parent_only_and_no_input_or_frozen_mask_mutation():
    colors, original, candidates = context_inputs()
    copies = tuple(value.copy() for value in (colors, original, candidates))
    direct = _quantized_hint_context(colors, original, candidates)
    assert direct[3:6, 2].tolist() == [True, False, False]
    context = _two_row_hint_context(colors, original, candidates)
    # Deliberately test three raw rows independent of the production 2.5px band.
    assert context[3:6, 2].tolist() == [True, True, False]
    assert np.array_equal(direct, _quantized_hint_context(colors, original, candidates))
    for original_value, saved in zip((colors, original, candidates), copies, strict=True):
        assert np.array_equal(original_value, saved)


@pytest.mark.parametrize(
    "change",
    [
        "invalid_parent",
        "parent_not_raw",
        "diagonal_parent",
        "missing_original",
        "original_three_rows_up",
        "invalid_child",
        "unsupported_child_gray",
        "missing_child_gray",
    ],
)
def test_extension_requires_exact_parent_original_and_independent_current_context(change):
    colors, original, candidates = context_inputs()
    if change == "invalid_parent":
        colors[3, 2] = (55, 61, 64)
    elif change == "parent_not_raw":
        candidates[3, 2] = False
    elif change == "diagonal_parent":
        colors[3, 1] = colors[3, 2]
        original[2, 1] = True
        candidates[3, 1] = True
        candidates[3, 2] = False
    elif change == "missing_original":
        original[2, 2] = False
    elif change == "original_three_rows_up":
        original[2, 2] = False
        original[1, 2] = True
    elif change == "invalid_child":
        colors[4, 2] = (5, 8, 11)
    elif change == "unsupported_child_gray":
        colors[4, 2] = (55, 61, 64)
    elif change == "missing_child_gray":
        colors[3:6, 1:4] = (1, 2, 3)
        colors[3, 2] = (32, 39, 42)
        colors[4, 2] = (10, 17, 19)
        assert _quantized_hint_context(colors, original, candidates)[3, 2]
    assert not _two_row_hint_context(colors, original, candidates)[4, 2]


def test_v4_keeps_raw_candidates_grouping_and_frozen_trace_unchanged():
    raw = two_row_png(columns=(45, 46, 65, 66, 84, 85, 104, 105))
    colors = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=float)
    original = trace_mask(raw)
    candidates, _ = _low_intensity_hints(colors, original, 20.45)
    assert np.flatnonzero(candidates.any(axis=0)).tolist() == [45, 46, 65, 66, 84, 85, 104, 105]
    assert not original[21:23].any()
    assert candidate_columns(raw, flux(20.45), hint_policy=V4) == [[45, 46], [65, 66], [84, 85]]
    for historical in (V2, V3):
        with pytest.raises(BrowserSafetyStop, match="unsupported_low_intensity_hint_context"):
            candidate_columns(raw, flux(20.45), hint_policy=historical)


@pytest.mark.parametrize("pixel", [(5, 8, 11), (55, 61, 64)])
def test_invalid_first_raw_hint_is_never_dropped_for_a_later_valid_group(pixel):
    raw = two_row_png(columns=(40, 70, 100, 130), change=lambda picture, x: picture.putpixel((x, 22), pixel))
    with pytest.raises(BrowserSafetyStop, match="unsupported_low_intensity_hint_context"):
        candidate_columns(raw, flux(20.45), hint_policy=V4)


@pytest.mark.parametrize("first", [(31,), tuple(range(40, 47))])
def test_invalid_first_group_is_not_split_or_skipped(first):
    with pytest.raises(BrowserSafetyStop, match="broad_or_edge_feature"):
        candidate_columns(two_row_png(columns=first + (70, 100, 130)), flux(20.45), hint_policy=V4)


def test_v4_cannot_relax_focused_palette_or_native_bracketed_confirmation():
    with pytest.raises(BrowserSafetyStop, match="invalid_overview_hint_policy"):
        candidate_columns(png(baseline=80), flux(80), overview=False, hint_policy=V4)
    for samples in ([sample(1), sample(2), sample(3)], [sample(1, "99"), sample(2, "98"), sample(3)]):
        result = summarize_samples(samples)
        assert result["status"] == "unresolved"
        assert result["planet_decision"] is None and result["answer_authorized"] is False


def test_unchanged_historical_recipes_detectors_and_measurement_identity():
    assert overview_hint_metadata(None) == {}
    assert overview_hint_metadata(V1) == {"overview_hint_policy": V1}
    assert overview_hint_metadata(V2)["overview_hint_recipe"]["sha256"] == (
        "76a1576ab9b3cc5b2ef6b150f59f978342e0dd6686eda5252419bf5ed15799bd"
    )
    assert overview_hint_metadata(V3)["overview_hint_recipe"]["sha256"] == (
        "e22aa71a3e41e0813704c869de30c23ed667969fd764bb29d00956e6a75d2582"
    )
    assert overview_hint_metadata(V4)["overview_hint_recipe"]["sha256"] == (
        "ea6d5a1b9114fff504b7fa537b63db941b0fb6cc61bae250aa4e1a8ebb9c2db8"
    )
    assert policy_manifest()["sha256"] == "1e376eb83c896c6828aa9335ca98820e95ffbf0328ae4a6f33ce53118e16e826"
    assert baseline_manifest()["sha256"] == "d3013611379fab5aab8905f956d62cf8f19c75510602f91ed24908f0f158f470"
    assert detector_manifest()["sha256"] == "fbd28b7dacbb2239ec9a41de8aaff6a5e5dadb2882586d5fab48c5da3df480e2"
    assert (
        measurement_manifest()["sha256"] == "50e38151704221593d91feb949c2a40927bcb455cb5874f769333c62171f3416"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("maximum_support_edges", True),
        ("maximum_support_edges", 2.0),
        ("answer_authorized", 0),
        ("sha256", "0" * 64),
        ("version", V3),
    ],
)
def test_v4_recipe_types_hash_and_policy_are_exact(key, value):
    metadata = overview_hint_metadata(V4)
    assert matches_overview_hint_metadata(metadata, V4)
    metadata["overview_hint_recipe"][key] = value
    assert not matches_overview_hint_metadata(metadata, V4)


def test_saved_belebron_rejects_no_and_yields_only_ordered_v4_search_hints():
    root = Path(__file__).resolve().parents[1] / (
        "experiments/browser-project-supplied-three-star/a11d7a17ba064f9791e1827a38f9830d/"
        "campaign/stars/003/owner/star/shallow/initial/progress"
    )
    if not root.exists():
        pytest.skip("Optional immutable public Belebron capture unavailable")
    raw = (root / "chart.png").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest() == "e36d8c12feb0b706a3f5417dc769c266c3307ec7a308f9227c367111ed69eae2"
    )
    report = json.loads((root / "report.json").read_bytes())
    assert report["star"] == "BELEBRON" and report["endpoint_visible"] is True
    labels = report["flux_axis_labels"]
    assert candidate_columns(raw, labels, hint_policy=V4) == [[45, 46], [65, 66], [84, 85]]
    for policy in (V2, V3):
        with pytest.raises(BrowserSafetyStop, match="unsupported_low_intensity_hint_context"):
            candidate_columns(raw, labels, hint_policy=policy)
    colors = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=float)
    original = trace_mask(raw)
    candidates, _ = _low_intensity_hints(colors, original, 20.450004577636705)
    direct = _quantized_hint_context(colors, original, candidates)
    extended = _two_row_hint_context(colors, original, candidates)
    assert np.argwhere(extended & ~direct).tolist() == [[22, 66], [22, 222]]
    assert not extended[21, 145] and not extended[22, 145]
    for analyzer in (frozen_no, baseline_no):
        result = analyzer(raw, report["time_axis_labels"], labels, requested_days=5000)
        assert result["status"] == "insufficient_visual_evidence"
        assert result["planet_decision"] is None


@pytest.fixture
def v4_bundle(tmp_path):
    return tmp_path, [
        fixture(
            tmp_path, f"v4-{i}", 100 * (i + 1), linked=True, overview_hint_index=i, overview_hint_policy=V4
        )
        for i in range(3)
    ]


def test_recorded_v4_rebuilds_exact_recipe_and_never_promotes_search_authority(v4_bundle):
    result = load(v4_bundle)
    assert result["period_days"]["value"] == "100"
    assert result["overview_hint_positions_used"] is False and result["answer_authorized"] is False
    assert result["brightness_drop_percent"]["physical_bounds"] is None
    metadata = overview_hint_metadata(V4)
    assert all(
        all(feature[key] == value for key, value in metadata.items()) for feature in result["features"]
    )
    assert result == load(v4_bundle)


@pytest.mark.parametrize("policy", [None, V1, V2, V3])
def test_mixed_historical_v4_bundle_cannot_retag_old_evidence(v4_bundle, policy):
    root, specs = v4_bundle
    specs[2] = fixture(
        root,
        "historical-third",
        300,
        linked=True,
        overview_hint_index=None if policy is None else 2,
        overview_hint_policy=policy,
    )
    with pytest.raises(BrowserSafetyStop, match="mixed_or_nonconsecutive_overview_hints"):
        load(v4_bundle)


def test_rehashed_v4_recipe_mutation_rejected_by_strict_reader(v4_bundle):
    recipe = overview_hint_metadata(V4)["overview_hint_recipe"]
    recipe["maximum_support_edges"] = True
    _rewrite_hint_metadata(v4_bundle, 0, overview_hint_recipe=recipe)
    with pytest.raises(BrowserSafetyStop, match="unsupported_overview_hint_recipe"):
        load(v4_bundle)
