"""Cooperative scheduling cannot restart, extend or hallucinate a window."""

import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from test_planet_window_policy import crop, labels

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_progress
from habfly.browser_planet_window import PlanetWindowSession, _partial_palette_wait
from habfly.browser_stellar import SIMULATION_URL


@pytest.fixture
def scheduled(tmp_path, monkeypatch):
    import habfly.browser_classification as classes
    import habfly.browser_planet_window as module

    clock = [100.0]
    images, calls, events = [crop(end=130), crop()], [], []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def start(page, config, output, **kwargs):
        calls.append(("start", kwargs["days"]))
        return {"star": "Fixture", "observation_completed": False}

    def capture(page, config, output, **kwargs):
        calls.append(("capture", kwargs["requested_days"]))
        output.mkdir()
        png = images.pop(0)
        time, flux = labels(5000)
        report = {
            **trace_progress(png, time, requested_days=5000),
            "star": "FIXTURE",
            "time_axis_labels": time,
            "flux_axis_labels": flux,
            "chart_sha256": hashlib.sha256(png).hexdigest(),
            "browser_actions": 0,
            "answer_writes": 0,
        }
        (output / "chart.png").write_bytes(png)
        (output / "report.json").write_text(json.dumps(report))
        return report

    def choose(page, config, output, **kwargs):
        calls.append(("choice", "No"))
        assert hashlib.sha256(kwargs["evidence_path"].read_bytes()).hexdigest() == kwargs["evidence_sha256"]
        output.mkdir()
        receipt = {"readback_verified": True, "value": "No", "task_completed": False}
        (output / "confirmed.json").write_text(json.dumps(receipt))
        return receipt

    monkeypatch.setattr(module, "start_planet_observation", start)
    monkeypatch.setattr(module, "capture_observation_progress", capture)
    monkeypatch.setattr(module, "select_no_planet_from_window", choose)
    monkeypatch.setattr(classes, "read_planet_class_choices", lambda _: ({"selected": None}, {}))
    page = SimpleNamespace(frames=[SimpleNamespace(url=SIMULATION_URL)])
    return SimpleNamespace(
        clock=clock,
        images=images,
        calls=calls,
        events=events,
        page=page,
        root=tmp_path,
        emit=lambda *item: events.append(item),
    )


def make(fixture, **kwargs):
    return PlanetWindowSession(
        fixture.page, object(), fixture.root / "window", run_history=fixture.root, emit=fixture.emit, **kwargs
    )


def test_default_reports_assumption_without_native_write(scheduled):
    session = make(scheduled)
    assert session.advance()["phase"] == "observing"
    for _ in range(10):
        session.advance()
    assert scheduled.calls == [("start", 5000)]
    scheduled.clock[0] += 5
    assert session.advance()["phase"] == "observing"
    scheduled.clock[0] += 5
    assert session.advance()["phase"] == "assumed_no_ready"
    assert session.finished and not session.state()["task_completed"]
    session.advance()
    assert scheduled.calls == [("start", 5000), ("capture", 5000), ("capture", 5000)]


def test_opt_in_no_occupies_separate_cancellable_step(scheduled):
    scheduled.images[:] = [crop()]
    session = make(scheduled, select_no=True)
    session.advance()
    scheduled.clock[0] += 5
    assert session.advance()["phase"] == "ready_to_select_no"
    assert not any(c[0] == "choice" for c in scheduled.calls)
    assert session.advance()["phase"] == "no_selected"
    assert session.state()["decision_readback_verified"]
    session.advance()
    assert sum(c[0] == "choice" for c in scheduled.calls) == 1
    assert not json.loads((session.output / "report.json").read_text())["task_completed"]


@pytest.mark.parametrize("stage", ["before_start", "observing", "before_choice"])
def test_abort_never_advances_or_compensates(scheduled, stage):
    scheduled.images[:] = [crop()]
    session = make(scheduled, select_no=True)
    if stage != "before_start":
        session.advance()
    if stage == "before_choice":
        scheduled.clock[0] += 5
        session.advance()
    before = list(scheduled.calls)
    session.abort()
    assert session.advance()["phase"] == "aborted"
    session.abort()
    assert scheduled.calls == before


def test_dip_hands_off_without_yes_or_numerical_answer(scheduled):
    scheduled.images[:] = [crop(dip=(70, 40))]
    session = make(scheduled, select_no=True)
    session.advance()
    scheduled.clock[0] += 5
    assert session.advance()["phase"] == "dip_observed"
    assert not session.receipt and len(scheduled.calls) == 2


@pytest.mark.parametrize("reason", ["time", "poll", "palette"])
def test_unresolved_windows_never_become_no_or_restart(scheduled, reason):
    if reason == "palette":
        scheduled.images[:] = [crop(color=(180, 30, 50))]
    session = make(scheduled, select_no=True, max_seconds=30, max_polls=1)
    session.advance()
    scheduled.clock[0] += 31 if reason == "time" else 5
    session.advance()
    if reason == "poll":
        scheduled.clock[0] += 5
        session.advance()
    assert session.finished and session.phase == "stopped"
    assert not session.receipt and not session.state()["task_completed"]
    before = list(scheduled.calls)
    session.advance()
    assert scheduled.calls == before


@pytest.mark.parametrize("error", [BrowserSafetyStop("fixture_stop"), RuntimeError("SECRET URL")])
def test_failed_start_is_redacted_and_never_repeated(scheduled, monkeypatch, error):
    import habfly.browser_planet_window as module

    def failed(*args, **kwargs):
        scheduled.calls.append(("failed", 5000))
        raise error

    monkeypatch.setattr(module, "start_planet_observation", failed)
    session = make(scheduled)
    session.advance()
    session.advance()
    assert len(scheduled.calls) == 1 and session.finished
    assert "SECRET" not in json.dumps(scheduled.events)


@pytest.mark.parametrize(
    "options",
    [
        {"select_no": 1},
        {"max_seconds": 1000},
        {"max_seconds": float("nan")},
        {"poll_interval": 0},
        {"max_polls": True},
        {"max_polls": 121},
    ],
)
def test_invalid_budgets_reject_before_artifacts(scheduled, options):
    with pytest.raises(ValueError):
        make(scheduled, **options)
    assert not (scheduled.root / "window").exists()


@pytest.mark.parametrize("stop_event", ["state", "observation", "action_proposed"])
def test_callback_abort_never_dispatches_following_action(scheduled, stop_event):
    scheduled.images[:] = [crop()]
    session = make(scheduled, select_no=True)
    session.emit = lambda event, _: session.abort() if event == stop_event else None
    for _ in range(5):
        scheduled.clock[0] += 5
        session.advance()
    assert session.phase == "aborted"
    assert not any(c[0] == "choice" for c in scheduled.calls)
    if stop_event == "state":
        assert scheduled.calls == []
    assert json.loads((session.output / "report.json").read_text())["phase"] == "aborted"


def test_supplemental_positive_evidence_never_calls_no_selector(scheduled, monkeypatch):
    import habfly.browser_planet_window as module

    scheduled.images[:] = [crop()]
    original = module.analyze_planet_window

    def inconclusive(*args, **kwargs):
        return {
            **original(*args, **kwargs),
            "status": "insufficient_visual_evidence",
            "reason": "unknown_plot_palette",
        }

    monkeypatch.setattr(module, "analyze_planet_window", inconclusive)
    monkeypatch.setattr(
        module,
        "analyze_window_dip",
        lambda *_: {
            "status": "dip_observed",
            "reason": "supported_reference_dips",
            "planet_decision": None,
            "scientific_verified": False,
            "training_label": False,
        },
    )
    session = make(scheduled, select_no=True)
    session.advance()
    scheduled.clock[0] += 5
    session.advance()
    assert session.phase == "dip_observed" and session.finished
    assert session.analysis["negative_policy_reason"] == "unknown_plot_palette"
    assert not session.state()["task_completed"]
    assert not session.state()["waiting_for_clean_partial_chart"]
    assert not any(c[0] == "choice" for c in scheduled.calls)
    assert (session.output / "progress-000/positive-dip-analysis.json").exists()


def partial_edge_crop():
    # Rendered Avan frontier: one (32,39,42) pixel beside the established blue
    # line. It remains unknown to the unchanged decision policy, not ignored.
    image = Image.open(io.BytesIO(crop(end=161))).convert("RGB")
    image.putpixel((162, 20), (32, 39, 42))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_partial_palette_wait_requires_later_complete_clean_capture(scheduled):
    scheduled.images[:] = [partial_edge_crop(), crop()]
    session = make(scheduled, select_no=True)
    original_policy = session.scope["policy"]
    session.advance()
    scheduled.clock[0] += 5
    state = session.advance()
    assert state["phase"] == "observing" and state["waiting_for_clean_partial_chart"]
    assert session.analysis["reason"] == "unknown_plot_palette"
    assert session.analysis["planet_decision"] is None and session.receipt is None
    assert session.scope["policy"] == original_policy
    assert json.loads((session.output / "progress-000/analysis.json").read_text()) == session.analysis
    # Waiting is idle until due and never repeats Play or writes a decision.
    for _ in range(100):
        session.advance()
    assert scheduled.calls == [("start", 5000), ("capture", 5000)]
    scheduled.clock[0] += 5
    state = session.advance()
    assert state["phase"] == "ready_to_select_no" and not state["waiting_for_clean_partial_chart"]
    assert session.receipt is None
    session.advance()
    assert scheduled.calls == [("start", 5000), ("capture", 5000), ("capture", 5000), ("choice", "No")]


@pytest.mark.parametrize("stop", ["deadline", "polls", "abort"])
def test_unresolved_partial_palette_obeys_original_bounds(scheduled, stop):
    scheduled.images[:] = [partial_edge_crop()]
    session = make(scheduled, select_no=True, max_seconds=30, max_polls=1)
    session.advance()
    scheduled.clock[0] += 5
    session.advance()
    assert session.waiting_for_clean_partial_chart
    if stop == "abort":
        session.abort()
    else:
        scheduled.clock[0] += 31 if stop == "deadline" else 5
        session.advance()
    assert session.finished and session.receipt is None
    assert scheduled.calls == [("start", 5000), ("capture", 5000)]
    assert session.scope["max_seconds"] == 30 and session.scope["max_polls"] == 1
    assert not session.state()["task_completed"]


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "unknown"},
        {"requested_days": 10000},
        {"requested_days": 5000.0},
        {"status": "endpoint_visible"},
        {"endpoint_visible": True},
        {"endpoint_visible": None},
        {"observation_completed": True},
        {"trace_columns": 7},
        {"trace_columns": True},
        {"approximate_rendered_day": 0},
        {"approximate_rendered_day": 5000},
        {"approximate_rendered_day": float("nan")},
        {"approximate_rendered_day": True},
    ],
)
def test_palette_wait_is_not_permission_for_unverified_progress(changes):
    progress = trace_progress(partial_edge_crop(), labels(5000)[0], requested_days=5000)
    analysis = {"status": "insufficient_visual_evidence", "reason": "unknown_plot_palette"}
    assert _partial_palette_wait(analysis, progress)
    assert not _partial_palette_wait(analysis, {**progress, **changes})


def test_unknown_palette_at_complete_endpoint_still_stops(scheduled):
    image = Image.open(io.BytesIO(crop())).convert("RGB")
    image.putpixel((162, 20), (32, 39, 42))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    scheduled.images[:] = [stream.getvalue()]
    session = make(scheduled, select_no=True)
    session.advance()
    scheduled.clock[0] += 5
    session.advance()
    assert session.phase == "stopped" and session.analysis["reason"] == "unknown_plot_palette"
    assert not session.waiting_for_clean_partial_chart and not session.receipt


def test_explicit_shallow_handoff_keeps_frozen_failure_and_never_selects_no(scheduled):
    image = Image.open(io.BytesIO(crop())).convert("RGB")
    image.putpixel((162, 20), (32, 39, 42))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    scheduled.images[:] = [stream.getvalue()]
    session = make(scheduled, select_no=True, allow_shallow_reference=True)
    session.advance()
    scheduled.clock[0] += 5
    session.advance()
    assert session.phase == "shallow_reference_required" and session.finished
    assert session.analysis["reason"] == "unknown_plot_palette"
    assert not session.receipt and session.failure is None
    assert session.scope["shallow_reference_enabled"]
    assert not session.state()["task_completed"]
    assert not any(call[0] == "choice" for call in scheduled.calls)
    before = list(scheduled.calls)
    session.advance()
    assert scheduled.calls == before
