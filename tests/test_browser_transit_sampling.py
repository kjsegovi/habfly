"""Sampler fixtures are explicit synthetic visible-tool responses, not course data."""

import hashlib
import json

import pytest

from habfly.browser import BrowserSafetyStop
from habfly.browser_transit_sampling import (
    baseline_position,
    center_baseline,
    record_transit_sampling,
    recover_timed_out_scan,
    sample_transits,
    validated_scan_samples,
)
from habfly.planet_charts import FluxSample


class Chart:
    def bounding_box(self):
        return {"width": 280, "height": 195}


class Session:
    def __init__(self, scale=4, *, flat=False, period=20):
        self.scale, self.offset, self.flat = scale, 1000, flat
        self.period = period
        self.chart, self.actions = Chart(), []
        self.baseline = 21

    def flux_axis_labels(self):
        return [{"value": str(v), "center_y": self.baseline + 100 - v} for v in (80, 90, 100)]

    def hover(self, x, y):
        self.actions.append("hover")
        day = int(self.offset + x * 280 * self.scale)
        return FluxSample(
            day=day, brightness_percent="99.99" if not self.flat and day % self.period == 0 else "100"
        )

    def zoom(self, x, y, delta):
        assert delta in {-500, -250, -125, 125, 250, 500}
        assert (x, y) == (40 / 280, 80 / 195)
        self.actions.append("zoom")
        self.scale *= 2 ** (delta / 500)

    def pan(self, start, end):
        self.actions.append("pan")
        self.offset += (start[0] - end[0]) * 280 * self.scale
        self.baseline += (end[1] - start[1]) * 195


def test_zoom_calibration_and_three_complete_transits_require_continuous_days():
    session = Session()
    seen = []
    report = sample_transits(session, on_sample=seen.append)
    assert report["zooms"] == 2
    assert report["coverage_complete"] and len(report["complete_transits"]) >= 3
    assert report["period_days"] == "20" and report["brightness_drop_percent"] == "0.01"
    assert report["has_planet_answer"] is None
    assert not report["learned_chart_perception"] and not report["task_completed"]
    assert len(seen) == len(report["samples"])
    assert set(session.actions) == {"hover", "zoom", "pan"}


def test_flat_coverage_is_inconclusive_and_bounded():
    session = Session(scale=1, flat=True)
    report = sample_transits(session, max_windows=2)
    assert report["windows"] == 2 and report["status"] == "no_transit_in_observed_interval"
    assert report["period_days"] is None and report["has_planet_answer"] is None
    assert len(session.actions) == 1 + 2 + 55 * 2 + 1  # Baseline centering plus one horizontal pan.


@pytest.mark.parametrize("initial_scale", [0.68, 1.35])
def test_calibration_avoids_duplicate_day_oversampling_with_existing_budget(initial_scale):
    session = Session(scale=initial_scale, flat=True)
    report = sample_transits(session, max_windows=1, max_zooms=3)
    assert 42 <= report["unique_days"] <= 55
    assert report["coverage_complete"] and report["zooms"] <= 3
    assert report["has_planet_answer"] is None


def test_gaps_and_failed_resolution_are_not_repaired():
    session = Session()
    with pytest.raises(BrowserSafetyStop, match="resolution_not_reached"):
        sample_transits(session, max_zooms=0)
    session = Session(scale=1)
    original = session.hover

    def gap(x, y):
        sample = original(x, y)
        if sample.day >= 1050:
            sample.day += 1
        return sample

    session.hover = gap
    with pytest.raises(BrowserSafetyStop, match="coverage_gap"):
        sample_transits(session)
    for budget in (0, True, 33):
        with pytest.raises(ValueError, match="window budget"):
            sample_transits(Session(), max_windows=budget)


def test_axis_interpolation_rejects_missing_nonlinear_and_far_extrapolation():
    labels = [{"value": str(v), "center_y": 120 - v} for v in (80, 90, 100)]
    assert baseline_position(labels) == 20
    with pytest.raises(BrowserSafetyStop, match="insufficient"):
        baseline_position(labels[:2])
    labels[1]["center_y"] += 5
    with pytest.raises(BrowserSafetyStop, match="nonlinear"):
        baseline_position(labels)
    with pytest.raises(BrowserSafetyStop, match="outside_supported"):
        baseline_position([{"value": str(v), "center_y": 120 - v} for v in (10, 20, 30)])


def test_readable_baseline_needs_no_pan_but_clipped_position_still_rejects():
    session = Session()
    session.baseline = 30.45
    assert center_baseline(session) == pytest.approx(30.45)
    assert not session.actions
    session.baseline = 21
    session.pan = lambda *args: None
    with pytest.raises(BrowserSafetyStop, match="readability_unverified"):
        center_baseline(session)


def test_unzoomed_pan_clamp_uses_existing_zoom_budget_before_sampling():
    session = Session(scale=2)
    pan, zoom = session.pan, session.zoom
    session.pan = lambda *a: pan(*a) if session.scale < 2 else None

    def anchor_zoom(x, y, delta):
        # Bootstrap zoom anchors on the still-clipped visible baseline.
        assert y == 21 / 195
        zoom(x, 80 / 195, delta)

    session.zoom = anchor_zoom
    report = sample_transits(session, max_zooms=1)
    assert report["zooms"] == 1 and report["status"] == "periodic_transits_observed"
    blocked = Session(scale=2)
    blocked.pan = lambda *a: None
    with pytest.raises(BrowserSafetyStop, match="readability_unverified"):
        sample_transits(blocked, max_zooms=0)


def test_separate_batches_can_continue_same_star_continuous_daily_evidence():
    session = Session(scale=1, period=40)
    session.baseline = 80
    first = sample_transits(session, max_windows=1)
    first.update(star="FIXTURE", period_evidence_verified=False)
    prior = validated_scan_samples(first, star="FIXTURE")
    second = sample_transits(session, max_windows=2, prior_samples=prior)
    assert second["status"] == "periodic_transits_observed"
    assert second["period_days"] == "40"
    assert second["inherited_samples"] == 55
    assert second["new_samples"] == 110
    assert second["coverage_complete"]
    with pytest.raises(BrowserSafetyStop, match="incompatible"):
        validated_scan_samples(first, star="OTHER")
    first["last_day"] += 1
    with pytest.raises(BrowserSafetyStop, match="recalculation"):
        validated_scan_samples(first, star="FIXTURE")


def test_separate_batches_never_join_a_coverage_gap_or_conflicting_values():
    session = Session(scale=1, flat=True)
    first = sample_transits(session, max_windows=1)
    first.update(star="FIXTURE", period_evidence_verified=False)
    prior = validated_scan_samples(first, star="FIXTURE")
    session.offset += 500
    with pytest.raises(BrowserSafetyStop, match="coverage_gap"):
        sample_transits(session, max_windows=1, prior_samples=prior)
    first["samples"].append({**first["samples"][0], "brightness_percent": "90"})
    with pytest.raises(BrowserSafetyStop, match="conflicting"):
        validated_scan_samples(first, star="FIXTURE")


def test_recording_persists_immutable_window_checkpoints_for_bounded_continuation(monkeypatch, tmp_path):
    class RecordedSession(Session):
        def __init__(self, *args, **kwargs):
            super().__init__(scale=1, period=40)
            self.star = "FIXTURE"
            self.closed = False
            self.chart.screenshot = lambda **kwargs: None

        def close(self):
            self.closed = True

    monkeypatch.setattr("habfly.browser_planet_chart.FluxChartSession", RecordedSession)
    report = record_transit_sampling(None, None, tmp_path / "first")
    checkpoint = tmp_path / "first/window-01.json"
    earlier = json.loads(checkpoint.read_text())
    assert not earlier["period_evidence_verified"]
    assert len(validated_scan_samples(earlier, star="FIXTURE")) == 55
    assert report["period_evidence_verified"]
    from habfly.runtime import read_trace

    events = read_trace(tmp_path / "first/events.jsonl")
    assert events[0].payload["policy"] == "scripted_reference_sensor"
    assert events[-1].event == "episode_summary" and not events[-1].payload["task_completed"]
    resumed = record_transit_sampling(None, None, tmp_path / "second", prior_report=checkpoint)
    assert (tmp_path / "second/prior-report.json").read_bytes() == checkpoint.read_bytes()
    assert resumed["inherited_samples"] == 55 and resumed["prior_report_sha256"]
    assert resumed["period_evidence_verified"]
    assert json.loads(checkpoint.read_text()) == earlier
    (checkpoint.parent / "invalidated.json").write_text('{"allow_continuation":false}')
    with pytest.raises(BrowserSafetyStop, match="invalidated_transit_scan_evidence"):
        record_transit_sampling(None, None, tmp_path / "invalidated", prior_report=checkpoint)


@pytest.mark.parametrize("mutation", [None, "star", "sample", "stop", "duplicate"])
def test_budget_recovery_requires_confirmed_read_only_same_star_journal(tmp_path, mutation):
    rows = [FluxSample(day=d, brightness_percent="100").model_dump() for d in range(10)]
    events = [
        {"event": "observation", "payload": {"chart": {"source": "visible_tooltips", "star": "FIXTURE"}}}
    ]
    for index, row in enumerate(rows):
        events.extend(
            [
                {"event": "action_proposed", "payload": {"sequence": index, "kind": "HOVER"}},
                {
                    "event": "action_result",
                    "payload": {"sequence": index, "chart_sample": row.copy(), "task_completed": False},
                },
            ]
        )
    scope = {
        "star": "FIXTURE",
        "sampling_method": "scripted_bounded_visible_tooltips",
        "answer_writes": False,
        "learned_chart_perception": False,
    }
    stopped = {"reason": "chart_time_limit", "exception_type": "BrowserSafetyStop"}
    if mutation == "star":
        scope["star"] = "OTHER"
    if mutation == "sample":
        rows[3]["brightness_percent"] = "99"
    if mutation == "stop":
        stopped["reason"] = "missing_or_ambiguous_visible_tooltip"
    if mutation == "duplicate":
        events.append(events[-1])
    for name, value in (("scope.json", scope), ("stopped.json", stopped)):
        (tmp_path / name).write_text(json.dumps(value))
    for name, value in (("samples.jsonl", rows), ("events.jsonl", events)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in value))
    if mutation:
        with pytest.raises(BrowserSafetyStop):
            recover_timed_out_scan(tmp_path)
        assert not (tmp_path / "recovered-report.json").exists()
    else:
        report = recover_timed_out_scan(tmp_path)
        assert len(validated_scan_samples(report, star="FIXTURE")) == 10
        assert not report["automatic_browser_actions"] and not report["period_evidence_verified"]
        with pytest.raises(FileExistsError):
            recover_timed_out_scan(tmp_path)


@pytest.mark.parametrize("mutation", [None, "missing", "hash", "count", "axis", "gap", "invalidated"])
def test_timeout_continuation_retains_hash_checked_prior_evidence(tmp_path, mutation):
    from habfly.planet_charts import analyze_flux_samples

    axis = "each_hover_against_visible_tick_glyphs"
    prior_rows = [FluxSample(day=d, brightness_percent="100").model_dump() for d in range(10)]
    prior = {
        **analyze_flux_samples(prior_rows),
        "star": "FIXTURE",
        "sampling_method": "scripted_bounded_visible_tooltips",
        "has_planet_answer": None,
        "learned_chart_perception": False,
        "task_completed": False,
        "period_evidence_verified": False,
        "time_axis_validation": "legacy" if mutation == "axis" else axis,
        "samples": prior_rows,
    }
    raw = json.dumps(prior).encode()
    (tmp_path / "prior-report.json").write_bytes(raw)
    scope = {
        "star": "FIXTURE",
        "sampling_method": "scripted_bounded_visible_tooltips",
        "answer_writes": False,
        "learned_chart_perception": False,
        "time_axis_validation": axis,
        "prior_samples": 11 if mutation == "count" else 10,
        "prior_report_sha256": "wrong" if mutation == "hash" else hashlib.sha256(raw).hexdigest(),
    }
    rows = [
        FluxSample(day=d, brightness_percent="100").model_dump()
        for d in range(11 if mutation == "gap" else 9, 20)
    ]
    events = [
        {"event": "observation", "payload": {"chart": {"source": "visible_tooltips", "star": "FIXTURE"}}}
    ]
    for index, row in enumerate(rows):
        events += [
            {"event": "action_proposed", "payload": {"kind": "HOVER", "sequence": index}},
            {
                "event": "action_result",
                "payload": {"chart_sample": row, "sequence": index, "task_completed": False},
            },
        ]
    (tmp_path / "scope.json").write_text(json.dumps(scope))
    (tmp_path / "stopped.json").write_text(
        json.dumps({"reason": "chart_time_limit", "exception_type": "BrowserSafetyStop"})
    )
    (tmp_path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    if mutation == "missing":
        (tmp_path / "prior-report.json").rename(tmp_path / "retained-original.json")
    if mutation == "invalidated":
        (tmp_path / "invalidated.json").write_text("{}")
    if mutation:
        with pytest.raises(BrowserSafetyStop):
            recover_timed_out_scan(tmp_path)
        assert not (tmp_path / "recovered-report.json").exists()
        if mutation == "missing":
            recovered = recover_timed_out_scan(tmp_path, prior_report=tmp_path / "retained-original.json")
            assert recovered["unique_days"] == 20 and recovered["inherited_samples"] == 10
    else:
        report = recover_timed_out_scan(tmp_path)
        assert report["unique_days"] == 20 and report["coverage_complete"]
        assert report["inherited_samples"] == 10 and report["new_samples"] == 11
        assert report["source_sha256"]["prior-report.json"] == hashlib.sha256(raw).hexdigest()
        assert report["time_axis_validation"] == axis
        assert not report["automatic_browser_actions"] and not report["period_evidence_verified"]
