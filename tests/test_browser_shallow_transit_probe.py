"""Offline shallow-feature diagnostic contracts; no browser or training."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from habfly.browser import BrowserSafetyStop
from habfly.browser_shallow_transit_probe import (
    FIRST_THREE_HINT_POLICY,
    FIRST_THREE_LOW_INTENSITY_HINT_POLICY,
    candidate_columns,
    matches_overview_hint_metadata,
    overview_hint_metadata,
    probe_shallow_feature,
    summarize_samples,
)
from habfly.planet_charts import FluxSample


def png(*, baseline=21, features=((148, 149, 22),), size=(280, 195)):
    picture = Image.new("RGB", size, "black")
    for x in range(31, 261):
        picture.putpixel((x, baseline), (80, 132, 154))
    for left, right, bottom in features:
        for x in range(left, right + 1):
            for y in range(baseline, bottom + 1):
                picture.putpixel((x, y), (80, 132, 154))
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    return stream.getvalue()


def flux(baseline=20.95):
    return [{"value": str(value), "center_y": baseline + (100 - value) * 2.6} for value in (90, 95, 100)]


def sample(day, value="100"):
    return FluxSample(day=day, brightness_percent=value, source="visible_hover_tooltip")


def test_shallow_hint_is_not_a_dip_or_no_decision():
    assert candidate_columns(png(), flux()) == [[148, 149]]
    assert candidate_columns(png(features=()), flux()) == []


def test_multiple_independent_hints_and_fractional_axis():
    data = png(features=((40, 41, 22), (148, 149, 22), (257, 257, 22)))
    assert candidate_columns(data, flux(20.95)) == [[40, 41], [148, 149], [257]]


def test_single_row_in_full_flux_range_is_only_a_probe_hint():
    data = png(baseline=20, features=((83, 83, 21), (137, 137, 21), (245, 245, 21)))
    assert candidate_columns(data, flux(20.45)) == [[83], [137], [245]]
    assert candidate_columns(png(baseline=20, features=()), flux(20.45)) == []


def test_grid_blend_is_an_untrusted_search_hint_not_positive_trace_support():
    from habfly.browser_observation_progress import trace_mask

    image = Image.open(io.BytesIO(png(baseline=20, features=()))).convert("RGB")
    image.putpixel((191, 21), (62, 81, 89))
    out = io.BytesIO()
    image.save(out, format="PNG")
    assert not trace_mask(out.getvalue())[21, 191]
    assert candidate_columns(out.getvalue(), flux(20.45)) == [[191]]


def low_intensity_png(*, pixels=None, support=True, gray=26):
    picture = Image.open(io.BytesIO(png(baseline=20, features=()))).convert("RGB")
    for x in range(31, 261):
        picture.putpixel((x, 21), (gray,) * 3)
    pixels = (
        pixels
        if pixels is not None
        else {
            57: (29, 36, 38),
            86: (27, 31, 32),
            115: (29, 36, 38),
            144: (29, 36, 38),
            173: (29, 36, 38),
            202: (31, 41, 45),
            231: (31, 41, 45),
        }
    )
    for x, color in pixels.items():
        picture.putpixel((x, 21), color)
        if not support:
            picture.putpixel((x, 20), (gray,) * 3)
    raw = io.BytesIO()
    picture.save(raw, format="PNG")
    return raw.getvalue()


def test_v2_faint_hints_preserve_weak_earlier_feature_without_changing_legacy():
    raw = low_intensity_png()
    assert candidate_columns(raw, flux(20.45)) == [[202], [231]]
    with pytest.raises(BrowserSafetyStop, match="three_overview_hints_required"):
        candidate_columns(raw, flux(20.45), hint_policy=FIRST_THREE_HINT_POLICY)
    assert candidate_columns(raw, flux(20.45), hint_policy=FIRST_THREE_LOW_INTENSITY_HINT_POLICY) == [
        [57],
        [86],
        [115],
    ]
    from habfly.browser_observation_progress import trace_mask

    assert not trace_mask(raw)[21, 57:232].any()


@pytest.mark.parametrize("color", [(26, 26, 26), (38, 36, 29), (29, 38, 36), (27, 31, 40), (80, 84, 85)])
def test_v2_does_not_turn_neutral_or_unmatched_color_into_hints(color):
    raw = low_intensity_png(pixels={57: color, 86: color, 115: color})
    with pytest.raises(BrowserSafetyStop, match="three_overview_hints_required"):
        candidate_columns(raw, flux(20.45), hint_policy=FIRST_THREE_LOW_INTENSITY_HINT_POLICY)


@pytest.mark.parametrize("options", [{"support": False}, {"gray": 0}])
def test_v2_requires_original_blue_and_observed_neutral_background(options):
    raw = low_intensity_png(pixels={x: (29, 36, 38) for x in (57, 86, 115)}, **options)
    with pytest.raises(BrowserSafetyStop, match="unsupported_low_intensity_hint_context"):
        candidate_columns(raw, flux(20.45), hint_policy=FIRST_THREE_LOW_INTENSITY_HINT_POLICY)


@pytest.mark.parametrize("first", [[31], list(range(40, 47))])
def test_v2_malformed_earliest_hint_is_not_skipped(first):
    raw = low_intensity_png(pixels={x: (29, 36, 38) for x in first + [70, 100, 130]})
    with pytest.raises(BrowserSafetyStop, match="broad_or_edge_feature"):
        candidate_columns(raw, flux(20.45), hint_policy=FIRST_THREE_LOW_INTENSITY_HINT_POLICY)


def test_v2_cannot_relax_focused_detector_or_sample_requirements():
    with pytest.raises(BrowserSafetyStop, match="invalid_overview_hint_policy"):
        candidate_columns(
            png(baseline=80), flux(80), overview=False, hint_policy=FIRST_THREE_LOW_INTENSITY_HINT_POLICY
        )
    assert summarize_samples([sample(1), sample(2), sample(3)])["status"] == "unresolved"


@pytest.mark.parametrize(
    "key,value",
    [
        ("answer_authorized", 0),
        ("scientific_verified", 0.0),
        ("maximum_channel", 64.0),
        ("maximum_channel_residual", True),
        ("sha256", "0" * 64),
    ],
)
def test_v2_metadata_requires_exact_types_and_canonical_recipe_hash(key, value):
    metadata = overview_hint_metadata(FIRST_THREE_LOW_INTENSITY_HINT_POLICY)
    assert matches_overview_hint_metadata(metadata, FIRST_THREE_LOW_INTENSITY_HINT_POLICY)
    metadata["overview_hint_recipe"][key] = value
    assert not matches_overview_hint_metadata(metadata, FIRST_THREE_LOW_INTENSITY_HINT_POLICY)
    assert overview_hint_metadata(FIRST_THREE_HINT_POLICY) == {
        "overview_hint_policy": FIRST_THREE_HINT_POLICY
    }
    assert overview_hint_metadata(None) == {}


def test_saved_kanzain_pixels_only_expand_search_hints():
    base = (
        Path(__file__).resolve().parents[1]
        / "experiments/browser-project-autonomous-three-star/e3e4a09caf7b479aac074ab1f7bf688b/campaign/stars/001/owner/star/shallow/initial/progress"
    )
    if not base.exists():
        pytest.skip("Optional immutable saved public crop unavailable")
    raw = (base / "chart.png").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest() == "1cf0a39fad897ba8c54697f4be74024aa782925c3f2e6eda8f0299b8aff61591"
    )
    report = json.loads((base / "report.json").read_bytes())
    labels = report["flux_axis_labels"]
    assert candidate_columns(raw, labels) == [[202], [231]]
    assert candidate_columns(raw, labels, hint_policy=FIRST_THREE_LOW_INTENSITY_HINT_POLICY) == [
        [57],
        [86],
        [115],
    ]
    from habfly.planet_window_dip import analyze_window_dip
    from habfly.planet_window_policy import analyze_planet_window

    negative = analyze_planet_window(raw, report["time_axis_labels"], labels)
    positive = analyze_window_dip(raw, report["time_axis_labels"], labels)
    assert negative["status"] == "insufficient_visual_evidence" and negative["planet_decision"] is None
    assert positive["status"] == "insufficient_visual_evidence" and positive["supported_dip_components"] == 0


def many_features():
    return tuple(
        (32 + round(index * 4.8), 32 + round(index * 4.8) + index % 2, 22) for index in range(47)
    ) + ((258, 259, 22),)


def test_explicit_first_three_policy_accepts_dense_hints_without_repairing_later_edge():
    features = many_features()
    assert len(features) == 48 and features[-1][:2] == (258, 259)
    expected = [list(range(left, right + 1)) for left, right, _ in features[:3]]
    assert candidate_columns(png(features=features), flux(), hint_policy=FIRST_THREE_HINT_POLICY) == expected
    with pytest.raises(BrowserSafetyStop, match="too_many_candidate_groups"):
        candidate_columns(png(features=features), flux())


@pytest.mark.parametrize(
    "features",
    [
        ((31, 31, 22), (50, 50, 22), (80, 80, 22), (110, 110, 22)),
        ((40, 46, 22), (60, 60, 22), (80, 80, 22), (110, 110, 22)),
        ((40, 40, 22), (60, 66, 22), (80, 80, 22), (110, 110, 22)),
        ((40, 40, 22), (60, 60, 22), (80, 86, 22), (110, 110, 22)),
        ((40, 40, 22), (60, 60, 22), (258, 259, 22)),
    ],
)
def test_first_three_policy_rejects_bad_selected_groups_instead_of_skipping(features):
    with pytest.raises(BrowserSafetyStop, match="broad_or_edge_feature"):
        candidate_columns(png(features=features), flux(), hint_policy=FIRST_THREE_HINT_POLICY)


def test_first_three_policy_does_not_validate_or_classify_unexamined_later_groups():
    features = ((40, 40, 22), (60, 60, 22), (80, 80, 22), (110, 130, 22))
    assert candidate_columns(png(features=features), flux(), hint_policy=FIRST_THREE_HINT_POLICY) == [
        [40],
        [60],
        [80],
    ]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_first_three_policy_requires_three_actual_groups(count):
    with pytest.raises(BrowserSafetyStop, match="three_overview_hints_required"):
        candidate_columns(png(features=many_features()[:count]), flux(), hint_policy=FIRST_THREE_HINT_POLICY)


@pytest.mark.parametrize("policy", ["", "unknown", True, 0, [], {}])
def test_unknown_hint_policy_fails_before_reading_pixels(policy):
    with pytest.raises(BrowserSafetyStop, match="invalid_overview_hint_policy"):
        candidate_columns(b"not an image", [], hint_policy=policy)


def test_overview_policy_cannot_relax_focused_feature_validation():
    with pytest.raises(BrowserSafetyStop, match="invalid_overview_hint_policy"):
        candidate_columns(png(baseline=80), flux(80), overview=False, hint_policy=FIRST_THREE_HINT_POLICY)


@pytest.mark.parametrize("features", [((35, 44, 23),), ((31, 31, 23),), ((259, 259, 23),)])
def test_broad_and_edge_features_are_not_automatically_repaired(features):
    # x259 is kept long enough for the explicit edge rule to reject it.
    with pytest.raises(BrowserSafetyStop):
        candidate_columns(png(features=features), flux())


def test_focus_uses_current_visible_flux_baseline_not_overview_coordinates():
    data = png(baseline=80, features=((88, 89, 85),))
    assert candidate_columns(data, flux(80), overview=False) == [[88, 89]]
    with pytest.raises(BrowserSafetyStop):
        candidate_columns(data, flux(80))


def test_bracketed_readings_are_not_exact_depth_period_or_answer_authority():
    result = summarize_samples([sample(2335), sample(2338, "99.238"), sample(2342)])
    assert result["status"] == "visible_bracketed_dip"
    assert result["minimum_sampled_brightness_percent"] == "99.238"
    assert not result["daily_coverage_verified"]
    for key in (
        "period_evidence_verified",
        "minimum_depth_verified",
        "task_completed",
        "answer_authorized",
        "training_label",
    ):
        assert result[key] is False
    assert result["planet_decision"] is None and result["answer_writes"] == 0


@pytest.mark.parametrize("values", [("100", "100", "100"), ("99", "98", "100"), ("100", "98", "99")])
def test_unresolved_readings_never_imply_no(values):
    result = summarize_samples([sample(day, value) for day, value in zip((1, 2, 3), values)])
    assert result["status"] == "unresolved" and result["planet_decision"] is None


@pytest.mark.parametrize("days", [(1, 1, 2), (3, 2, 1), (4999, 5000, 5001)])
def test_duplicate_stale_or_outside_window_days_fail(days):
    with pytest.raises(BrowserSafetyStop):
        summarize_samples([sample(day) for day in days])


def source(tmp_path):
    path = tmp_path / "overview"
    path.mkdir()
    data = png()
    (path / "chart.png").write_bytes(data)
    report = {
        "star": "EXAMPLE",
        "requested_days": 5000,
        "browser_actions": 0,
        "answer_writes": 0,
        "endpoint_visible": True,
        "chart_sha256": hashlib.sha256(data).hexdigest(),
        "time_axis_labels": [{"value": str(v), "center_x": 30.5 + v * 0.046} for v in (0, 2500, 5000)],
        "flux_axis_labels": flux(),
    }
    raw = json.dumps(report).encode()
    (path / "report.json").write_bytes(raw)
    return path, report, hashlib.sha256(raw).hexdigest()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import habfly.browser_shallow_transit_probe as module

    path, report, checksum = source(tmp_path)
    state = SimpleNamespace(
        calls=[],
        phase="overview",
        fail=None,
        changes=None,
        star="EXAMPLE",
        time_offset=230,
        overview_png=png(),
        samples=[sample(2567), sample(2568), sample(2569, "99.238"), sample(2570)],
    )

    class Session:
        def __init__(self, page, config, emit, **options):
            state.calls.append(("construct", options))
            self.actions, self.star = 0, state.star
            self.handle = SimpleNamespace(evaluate=lambda _: True)
            self.chart = SimpleNamespace(screenshot=self.screenshot)

        def _guard(self):
            if state.fail == "guard":
                raise BrowserSafetyStop("chart_context_or_answers_changed")

        def screenshot(self, **options):
            from habfly.presentation_capture import EVIDENCE_SCREENSHOT_STYLE

            assert options == {"style": EVIDENCE_SCREENSHOT_STYLE}
            return (
                state.overview_png
                if state.phase == "overview"
                else png(baseline=80, features=((88, 89, 85),))
            )

        def time_axis_labels(self):
            return (
                report["time_axis_labels"]
                if state.phase == "overview"
                else [
                    {"value": str(v + state.time_offset), "center_x": 79.5 + (v - 2334) * 2.08}
                    for v in (2310, 2340, 2370)
                ]
            )

        def flux_axis_labels(self):
            return flux() if state.phase == "overview" else flux(80)

        def zoom(self, x, y, delta):
            state.calls.append(("zoom", delta))
            self.actions += 1
            state.phase = "focused"
            if state.changes:
                state.changes()

        def hover_day(self, x, y, requested_day):
            state.calls.append(("hover", x))
            self.actions += 1
            return state.samples.pop(0)

        def clear_pointer(self):
            state.calls.append(("clear_pointer",))
            self.actions += 1

        def close(self):
            state.calls.append(("close",))

    monkeypatch.setattr(module, "FluxChartSession", Session)
    monkeypatch.setattr(module, "center_baseline", lambda session: 80)
    state.options = {
        "run_history": tmp_path,
        "source_dir": path,
        "source_report_sha256": checksum,
        "candidate_index": 0,
    }
    state.output = tmp_path / "diagnostic"
    return state


def test_recipe_and_caps_with_injected_transport(rig):
    result = probe_shallow_feature(None, None, rig.output, **rig.options)
    assert result["status"] == "visible_bracketed_dip" and result["browser_actions"] == 11
    assert ("clear_pointer",) in rig.calls
    assert [call[1] for call in rig.calls if call[0] == "zoom"] == [-500, -500, -500, -500, -500, -250]
    assert rig.calls[0][1] == {"max_actions": 16, "max_seconds": 180, "verify_day_axis": True}
    assert rig.calls[-1] == ("close",)
    assert not result["answer_authorized"]
    assert result["source_hint_linked"]
    assert "overview_hint_policy" not in result
    assert "overview_hint_policy" not in json.loads((rig.output / "scope.json").read_bytes())
    assert "overview_hint_policy" not in (rig.output / "events.jsonl").read_text()


def test_first_three_wrapper_pins_policy_in_scope_report_and_episode_without_changing_recipe(rig):
    rig.overview_png = png(features=many_features())
    directory = rig.options["source_dir"]
    (directory / "chart.png").write_bytes(rig.overview_png)
    report = json.loads((directory / "report.json").read_bytes())
    report["chart_sha256"] = hashlib.sha256(rig.overview_png).hexdigest()
    raw = json.dumps(report).encode()
    (directory / "report.json").write_bytes(raw)
    rig.options.update(
        source_report_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_index=2,
        overview_hint_policy=FIRST_THREE_HINT_POLICY,
    )
    result = probe_shallow_feature(None, None, rig.output, **rig.options)
    scope = json.loads((rig.output / "scope.json").read_bytes())
    episode = json.loads((rig.output / "events.jsonl").read_text().splitlines()[-1])
    assert scope["candidate_columns"] == [42] and scope["candidate_index"] == 2
    assert scope["overview_hint_policy"] == result["overview_hint_policy"] == FIRST_THREE_HINT_POLICY
    assert episode["event"] == "episode_summary" and episode["payload"] == result
    assert json.loads((rig.output / "report.json").read_bytes()) == result
    assert result["browser_actions"] == 11 and not result["answer_authorized"]
    assert [call[1] for call in rig.calls if call[0] == "zoom"] == [-500, -500, -500, -500, -500, -250]
    assert rig.calls[0][1] == {"max_actions": 16, "max_seconds": 180, "verify_day_axis": True}


@pytest.mark.parametrize("index", [3, 7, -1, True])
def test_first_three_probe_cannot_address_a_later_or_invalid_candidate(rig, index):
    rig.options.update(candidate_index=index, overview_hint_policy=FIRST_THREE_HINT_POLICY)
    with pytest.raises(BrowserSafetyStop, match="invalid_candidate_index"):
        probe_shallow_feature(None, None, rig.output, **rig.options)
    assert not rig.calls and not rig.output.exists()


@pytest.mark.parametrize("policy", ["", "unknown", True, 0, [], {}])
def test_invalid_probe_policy_never_constructs_browser(rig, policy):
    rig.options["overview_hint_policy"] = policy
    with pytest.raises(BrowserSafetyStop, match="invalid_overview_hint_policy"):
        probe_shallow_feature(None, None, rig.output, **rig.options)
    assert not rig.calls and not rig.output.exists()


def test_visible_dip_at_another_date_is_not_linked_to_original_hint(rig):
    rig.time_offset = 0
    rig.samples = [sample(2337), sample(2338), sample(2339, "99.238"), sample(2340)]
    result = probe_shallow_feature(None, None, rig.output, **rig.options)
    assert result["status"] == "visible_decline_not_source_linked"
    assert result["sampled_decline_verified"] and not result["source_hint_linked"]
    assert result["source_hint_day_interval"][0] > result["sampled_day_interval"][1]
    assert not result["answer_authorized"] and not result["period_evidence_verified"]


@pytest.mark.parametrize("problem", ["star", "guard", "source_change"])
def test_changes_stop_without_continuation_or_success_report(rig, problem):
    if problem == "star":
        rig.star = "OTHER"
    elif problem == "guard":
        rig.fail = "guard"
    else:
        rig.changes = lambda: (Path(rig.options["source_dir"]) / "chart.png").write_bytes(b"changed")
    with pytest.raises(BrowserSafetyStop):
        probe_shallow_feature(None, None, rig.output, **rig.options)
    assert (rig.output / "stopped.json").exists() and not (rig.output / "report.json").exists()
    assert not any(call[0] == "hover" for call in rig.calls)


def test_invalid_source_hash_never_constructs_browser(rig):
    rig.options["source_report_sha256"] = "0" * 64
    with pytest.raises(BrowserSafetyStop):
        probe_shallow_feature(None, None, rig.output, **rig.options)
    assert not rig.calls and not rig.output.exists()


def test_cooperative_sensor_close_before_first_zoom_never_dispatches(rig):
    from habfly.browser_shallow_transit_probe import iterate_shallow_feature

    steps = iterate_shallow_feature(None, None, rig.output, **rig.options)
    assert next(steps)["stage"] == "source_verified"
    steps.close()
    assert not any(call[0] in {"zoom", "hover", "clear_pointer"} for call in rig.calls)
    assert json.loads((rig.output / "stopped.json").read_text())["reason"] == "operator_aborted"
    assert not (rig.output / "report.json").exists()


def test_daily_probe_rejects_nearby_but_wrong_integer_day(rig):
    rig.samples[0] = sample(2566)
    with pytest.raises(BrowserSafetyStop, match="requested_day_changed"):
        probe_shallow_feature(None, None, rig.output, **rig.options)
    assert not (rig.output / "report.json").exists()


@pytest.mark.parametrize("value", [None, "", True, 0, "a" * 63, "A" * 64, "x" * 64])
def test_report_hash_cannot_be_omitted_or_unpinned(rig, value):
    rig.options["source_report_sha256"] = value
    with pytest.raises(BrowserSafetyStop, match="explicit_report_hash_required"):
        probe_shallow_feature(None, None, rig.output, **rig.options)
    assert not rig.calls and not rig.output.exists()


@pytest.mark.parametrize("value", ["missing", None, "", True, 0, "a" * 63, "A" * 64, "x" * 64])
def test_chart_hash_cannot_be_omitted_or_unpinned(rig, value):
    path = Path(rig.options["source_dir"]) / "report.json"
    report = json.loads(path.read_bytes())
    if value == "missing":
        del report["chart_sha256"]
    else:
        report["chart_sha256"] = value
    raw = json.dumps(report).encode()
    path.write_bytes(raw)
    rig.options["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(BrowserSafetyStop, match="explicit_chart_hash_required"):
        probe_shallow_feature(None, None, rig.output, **rig.options)
    assert not rig.calls and not rig.output.exists()


def test_overlapping_output_does_not_create_a_directory(rig):
    output = Path(rig.options["source_dir"]) / "new-output"
    with pytest.raises(BrowserSafetyStop, match="overlapping_output"):
        probe_shallow_feature(None, None, output, **rig.options)
    assert not rig.calls and not output.exists()
