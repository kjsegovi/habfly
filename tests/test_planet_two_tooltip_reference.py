"""Pure source-chain tests; synthetic native event files, not browser acceptance."""

import hashlib
import io
import json
from copy import deepcopy
from decimal import localcontext

import pytest
from PIL import Image, ImageDraw
from test_planet_tooltip_reference import fixture, save

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_progress
from habfly.browser_shallow_transit_probe import (
    EXACT_TWO_HINT_POLICY,
    FIRST_THREE_HINT_POLICY,
    FIRST_THREE_TWO_ROW_HINT_POLICY,
    overview_hint_metadata,
)
from habfly.planet_tooltip_reference import (
    DIAGNOSTIC_FILES,
    MODE,
    TWO_MODE,
    load_tooltip_reference,
    measurement_manifest,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def repin(history, spec):
    directory = history / spec["directory"]
    spec["files"] = {path.name: sha(path.read_bytes()) for path in directory.iterdir()}


def rewrite(history, spec, name, value):
    directory = history / spec["directory"]
    save(directory / name, value)
    if name == "report.json":
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        events[-1]["payload"] = value
        (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    repin(history, spec)


def two_bundle(history, *, centers=(1000, 3000), values=None, columns=(76, 168)):
    """Reusable genuine-reader seam: every file is synthetic, every validator real.

    The exposed tooltip samples and exact overview hint/source/event pins are
    authored fixture data. This helper does not assert actual native execution.
    """
    picture = Image.new("RGB", (280, 196), "black")
    draw = ImageDraw.Draw(picture)
    draw.line((30, 20, 260, 20), fill=(80, 132, 154))
    for column in columns:
        picture.putpixel((column, 21), (55, 83, 96))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    png = buffer.getvalue()
    specs = []
    for index, center in enumerate(centers):
        spec = fixture(history, f"two-{index}", center, linked=True, values=values)
        directory = history / spec["directory"]
        scope, report = read(directory / "scope.json"), read(directory / "report.json")
        source = history / scope["source_dir"]
        original = read(source / "report.json")
        original.update(trace_progress(png, original["time_axis_labels"], requested_days=5000))
        original["chart_sha256"] = sha(png)
        (source / "chart.png").write_bytes(png)
        hashes = {"chart.png": sha(png), "report.json": save(source / "report.json", original)}
        before = read(directory / "before.json")
        before["chart_sha256"] = sha(png)
        (directory / "before.png").write_bytes(png)
        save(directory / "before.json", before)
        column = columns[index]
        interval = [(column - 0.5 - 30.5) * (5000 / 230), (column + 1.5 - 30.5) * (5000 / 230)]
        metadata = overview_hint_metadata(EXACT_TWO_HINT_POLICY)
        scope.update(
            **metadata,
            source_hashes=hashes,
            candidate_index=index,
            candidate_columns=[column],
            source_hint_day_interval=interval,
        )
        report.update(**metadata, source_hashes=hashes, source_hint_day_interval=interval)
        rewrite(history, spec, "scope.json", scope)
        rewrite(history, spec, "report.json", report)
        specs.append(spec)
    return specs


@pytest.fixture
def bundle(tmp_path):
    return tmp_path, two_bundle(tmp_path)


def load(bundle, **kwargs):
    root, specs = bundle
    return load_tooltip_reference(root, specs, expected_star="EXAMPLE", mode=TWO_MODE, **kwargs)


def test_exact_two_golden_is_one_assumed_spacing_and_not_recurrence(bundle):
    result = load(bundle)
    assert result["mode"] == TWO_MODE
    assert result["period_days"]["value"] == "2000"
    assert result["period_days"]["exact_representative"] == {"numerator": 2000, "denominator": 1}
    assert result["period_days"]["compatibility_interval"] == {
        "lower": "1998",
        "upper": "2002",
        "endpoints": "open",
    }
    assert result["period_days"]["fitted_bracket_points"] == ["1000", "3000"]
    assert result["period_days"]["estimate_kind"] == "single_spacing"
    assert "ASSUMED consecutive" in result["period_days"]["interpretation"]
    assert result["confirmed_feature_count"] == 2
    assert result["observed_interval_count"] == 1
    assert type(result["consistency_redundancy"]) is int and result["consistency_redundancy"] == 0
    assert result["consecutive_events_assumed"] is True and result["single_spacing_compatible"] is True
    assert result["brightness_drop_percent"]["value"] == "0.762"
    assert result["brightness_drop_percent"]["physical_bounds"] is None
    for name in (
        "recurrence_confirmed",
        "observed_recurrence_compatible",
        "physical_period_verified",
        "period_evidence_verified",
        "minimum_depth_verified",
        "scientific_verified",
        "learned_perception",
        "training_label",
        "task_completed",
        "answer_authorized",
        "missing_events_ruled_out",
        "aliasing_ruled_out",
        "daily_coverage_verified",
        "absence_proven",
        "overview_hint_positions_used",
    ):
        # No new physical-period flag is fabricated if it is absent historically.
        assert result.get(name, False) is False
    assert result["observation_limit_days"] == 5000 and result["browser_actions"] == 0
    assert [feature["overview_candidate_index"] for feature in result["features"]] == [0, 1]
    assert {feature["sensor_recipe"] for feature in result["features"]} == {"daily_focus_v2"}
    assert result == load(bundle)


def test_asymmetric_bracket_widths_and_low_decimal_context_do_not_change_exact_fit(tmp_path):
    specs = two_bundle(tmp_path, values=["100", "99.9", "99.8", "100"])
    with localcontext() as context:
        context.prec = 6
        result = load((tmp_path, specs))
        assert context.prec == 6
    assert result["period_days"]["compatibility_interval"] == {
        "lower": "1997",
        "upper": "2003",
        "endpoints": "open",
    }
    assert result["period_days"]["fitted_bracket_points"] == ["999.5", "2999.5"]
    assert result["brightness_drop_percent"]["value"] == "0.2"


def test_hint_coordinates_never_enter_the_single_spacing_estimate(tmp_path):
    specs = two_bundle(tmp_path, centers=(1003, 2997))
    result = load((tmp_path, specs))
    assert result["period_days"]["value"] == "1994"
    assert result["period_days"]["compatibility_interval"] == {
        "lower": "1992",
        "upper": "1996",
        "endpoints": "open",
    }


def test_method_is_separate_complete_and_detached(bundle):
    old, new = measurement_manifest(), measurement_manifest(TWO_MODE)
    assert old["sha256"] == "50e38151704221593d91feb949c2a40927bcb455cb5874f769333c62171f3416"
    assert old["diagnostics"] == [3, 8] and new["diagnostics"] == [2, 2]
    assert old["sha256"] != new["sha256"]
    assert list(new["sensor_recipes"]) == ["daily_focus_v2"]
    new["limitations"].clear()
    new["sensor_recipes"].clear()
    assert measurement_manifest(TWO_MODE)["limitations"]
    assert measurement_manifest()["sensor_recipes"] == old["sensor_recipes"]
    result = load(bundle)
    checksum = result.pop("receipt_sha256")
    assert checksum == sha(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    assert set(bundle[1][0]["files"]) == DIAGNOSTIC_FILES
    for path, checksum in result["source_sha256"].items():
        assert sha((bundle[0] / path).read_bytes()) == checksum


@pytest.mark.parametrize("mode", [None, True, [], {}, "unknown", 1])
def test_unknown_modes_never_fall_back(bundle, mode):
    with pytest.raises(BrowserSafetyStop, match="unsupported_measurement_mode"):
        load_tooltip_reference(bundle[0], bundle[1], expected_star="EXAMPLE", mode=mode)
    with pytest.raises(BrowserSafetyStop, match="unsupported_measurement_mode"):
        measurement_manifest(mode)


@pytest.mark.parametrize("count", [0, 1, 3, 8])
def test_exact_two_count_not_minimum_two(bundle, count):
    specs = (bundle[1] * 4)[:count]
    with pytest.raises(BrowserSafetyStop, match="diagnostic_count"):
        load((bundle[0], specs))


def test_old_loader_never_accepts_two_even_with_new_policy(bundle):
    with pytest.raises(BrowserSafetyStop, match="diagnostic_count"):
        load_tooltip_reference(bundle[0], bundle[1], expected_star="EXAMPLE")


@pytest.mark.parametrize("recipe", [None, "daily_focus_v2"])
def test_new_mode_cannot_relabel_old_unselected_bundles(tmp_path, recipe):
    specs = [fixture(tmp_path, f"d-{i}", 1000 * (i + 1), recipe=recipe, linked=True) for i in range(2)]
    with pytest.raises(BrowserSafetyStop, match="two_mode_requires_exact_two_daily_hints"):
        load((tmp_path, specs))


def test_new_mode_cannot_relabel_first_three_policy(tmp_path):
    specs = [fixture(tmp_path, f"d-{i}", 100 * (i + 1), linked=True, overview_hint_index=i) for i in range(2)]
    with pytest.raises(BrowserSafetyStop, match="two_mode_requires_exact_two_daily_hints"):
        load((tmp_path, specs))


def test_new_and_legacy_hint_sources_cannot_mix_in_either_mode(bundle):
    root, specs = bundle
    old = fixture(root, "old-third", 4000, linked=True)
    with pytest.raises(BrowserSafetyStop, match="two_mode_requires_exact_two_daily_hints"):
        load((root, [specs[0], old]))
    with pytest.raises(BrowserSafetyStop, match="mixed_or_nonconsecutive_overview_hints"):
        load_tooltip_reference(root, [*specs, old], expected_star="EXAMPLE")


def test_two_bundle_requires_daily_recipe_and_cannot_mix_historical_zoom_recipe(bundle):
    root, specs = bundle
    old = fixture(root, "old-zoom", 3000, recipe=None, linked=True)
    with pytest.raises(BrowserSafetyStop, match="mixed_sensor_recipes"):
        load((root, [specs[0], old]))


@pytest.mark.parametrize("columns", [(76,), (76, 168, 210)])
def test_source_pixels_must_have_exactly_two_groups_not_truncated_or_invented(bundle, columns):
    root, specs = bundle
    directory = root / specs[0]["directory"]
    scope = read(directory / "scope.json")
    source = root / scope["source_dir"]
    picture = Image.new("RGB", (280, 196), "black")
    ImageDraw.Draw(picture).line((30, 20, 260, 20), fill=(80, 132, 154))
    for column in columns:
        picture.putpixel((column, 21), (55, 83, 96))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    png = buffer.getvalue()
    original = read(source / "report.json")
    original.update(trace_progress(png, original["time_axis_labels"], requested_days=5000))
    original["chart_sha256"] = sha(png)
    (source / "chart.png").write_bytes(png)
    hashes = {"chart.png": sha(png), "report.json": save(source / "report.json", original)}
    for name in ("scope.json", "report.json"):
        item = read(directory / name)
        item["source_hashes"] = hashes
        rewrite(root, specs[0], name, item)
    with pytest.raises(BrowserSafetyStop, match="exact_two_overview_hints_required"):
        load(bundle)


def test_duplicate_and_foreign_diagnostics_reject(bundle):
    with pytest.raises(BrowserSafetyStop, match="duplicate_diagnostic"):
        load((bundle[0], [bundle[1][0]] * 2))
    with pytest.raises(BrowserSafetyStop, match="mixed_method_or_star"):
        load_tooltip_reference(bundle[0], bundle[1], expected_star="OTHER", mode=TWO_MODE)


@pytest.mark.parametrize("index", [True, 0.0, 2, -1])
def test_invalid_candidate_index_rejects_even_with_rehashed_scope(bundle, index):
    root, specs = bundle
    value = read(root / specs[0]["directory"] / "scope.json")
    value["candidate_index"] = index
    rewrite(root, specs[0], "scope.json", value)
    with pytest.raises(BrowserSafetyStop, match="invalid_overview_hint_index"):
        load(bundle)


@pytest.mark.parametrize(
    "key,value", [("answer_authorized", 0), ("maximum_support_edges", 2.0), ("sha256", "0" * 64)]
)
def test_recipe_relabeling_and_bool_numeric_aliases_reject(bundle, key, value):
    root, specs = bundle
    for name in ("scope.json", "report.json"):
        item = read(root / specs[0]["directory"] / name)
        item["overview_hint_recipe"][key] = value
        rewrite(root, specs[0], name, item)
    with pytest.raises(BrowserSafetyStop, match="unsupported_overview_hint_recipe"):
        load(bundle)


@pytest.mark.parametrize("name", ["before.png", "readable.png", "events.jsonl", "scope.json"])
def test_unpinned_source_mutation_is_rejected(bundle, name):
    path = bundle[0] / bundle[1][0]["directory"] / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(BrowserSafetyStop):
        load(bundle)


@pytest.mark.parametrize("change", ["missing_clear", "wrong_day", "missing_result", "invented_summary"])
def test_native_recipe_and_confirmed_samples_remain_mandatory(bundle, change):
    root, specs = bundle
    directory = root / specs[0]["directory"]
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    if change == "missing_clear":
        del events[13:15]
    elif change == "wrong_day":
        events[15]["payload"]["requested_day"] += 1
    elif change == "missing_result":
        del events[16]
    else:
        events[-1]["payload"]["samples"][1]["brightness_percent"] = "98"
    for index, event in enumerate(events):
        event["sequence"] = index
    (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    repin(root, specs[0])
    with pytest.raises(BrowserSafetyStop):
        load(bundle)


@pytest.mark.parametrize(
    "values", [["100", "100", "100"], ["99", "99", "100"], ["100", "99", "100", "99", "100"]]
)
def test_flat_unbracketed_and_multiple_declines_never_supply_spacing(tmp_path, values):
    specs = two_bundle(tmp_path, values=values)
    with pytest.raises(BrowserSafetyStop, match="unbracketed_decline|multiple_declines_in_bracket"):
        load((tmp_path, specs))


def test_reversed_descriptor_order_is_deterministic_and_inputs_unchanged(bundle):
    specs = deepcopy(bundle[1])
    before = deepcopy(specs)
    assert load((bundle[0], list(reversed(specs)))) == load(bundle)
    assert specs == before


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("legacy", "2d1d8fceab30141eb922c32cfac5e4588a34baa31f6463b9736172428c3495ca"),
        ("daily", "30372b4e2c85127320e13168904766a712b4ab555604ed000bd4696c0f0bc167"),
        ("first_three", "4160dbb0e8c82f7d4874f59f78d00e605897f7e2a3cbd47b27cce294a4e91987"),
        ("v4", "9485ac9a20f6931def768797a8e7cb0427c8bc4ef070c0bdea2b69d5d8ca2ee4"),
    ],
)
def test_legacy_canonical_receipts_exactly_match_prechange_hashes(tmp_path, variant, expected):
    kwargs = {"recipe": None} if variant == "legacy" else {}
    if variant in {"first_three", "v4"}:
        kwargs = {
            "linked": True,
            "overview_hint_policy": FIRST_THREE_HINT_POLICY
            if variant == "first_three"
            else FIRST_THREE_TWO_ROW_HINT_POLICY,
        }
    specs = [
        fixture(
            tmp_path,
            f"diagnostic-{i}",
            100 * (i + 1),
            **kwargs,
            **({"overview_hint_index": i} if variant in {"first_three", "v4"} else {}),
        )
        for i in range(3)
    ]
    implicit = load_tooltip_reference(tmp_path, specs, expected_star="EXAMPLE")
    explicit = load_tooltip_reference(tmp_path, specs, expected_star="EXAMPLE", mode=MODE)
    assert implicit == explicit and implicit["receipt_sha256"] == expected
    assert "recurrence_confirmed" not in implicit and "consistency_redundancy" not in implicit
