"""Pure scheduler regressions: a partial positive crop is not a final window.

The native start/capture/choice seams are the existing explicitly injected
fixture; the frozen raster analyzers still process actual synthetic PNG bytes.
No browser, model, replay artifact or production policy is changed here.
"""

import hashlib
import json

import pytest
import test_browser_planet_window as fixtures
from test_browser_planet_window import make
from test_planet_window_dip import png as encode_png
from test_planet_window_dip import positive_image
from test_planet_window_policy import crop

RULE = "complete_rendered_window_before_positive_handoff_v1"
scheduled = fixtures.scheduled


def poll(rig, session):
    rig.clock[0] += 5
    return session.advance()


def snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def start_partial(rig, *, partial_png=None, **options):
    rig.images[:] = [partial_png if partial_png is not None else crop(end=130, dip=(70, 40))]
    session = make(rig, select_no=True, **options)
    session.advance()
    state = poll(rig, session)
    return session, state


def test_positive_scheduler_rule_is_explicit_in_scope_and_state(scheduled):
    session = make(scheduled)
    assert session.scope["scheduling_rule"] == RULE
    assert json.loads((session.output / "scope.json").read_bytes())["scheduling_rule"] == RULE
    assert session.state()["scheduling_rule"] == RULE
    assert session.state()["prior_dip_observed"] is False
    assert session.state()["waiting_for_positive_endpoint"] is False


def test_early_dip_waits_without_rewriting_analysis_or_restart(scheduled):
    session, state = start_partial(scheduled)
    assert state["phase"] == "observing" and not session.finished
    assert state["prior_dip_observed"] is True
    assert state["waiting_for_positive_endpoint"] is True
    assert state["window_status"] == "dip_observed"
    directory = session.output / "progress-000"
    saved = snapshot(directory)
    assert json.loads(saved["analysis.json"]) == session.analysis
    assert session.analysis["status"] == "dip_observed"
    progress = json.loads(saved["report.json"])
    assert progress["endpoint_visible"] is False and progress["observation_completed"] is False
    for _ in range(1000):
        assert session.ready_to_advance() is False
        assert session.advance()["waiting_for_positive_endpoint"] is True
    assert snapshot(directory) == saved
    assert scheduled.calls == [("start", 5000), ("capture", 5000)]
    assert not session.receipt and not state["task_completed"]


def test_endpoint_dip_hands_off_latest_report_not_first_positive(scheduled):
    session, state = start_partial(scheduled)
    assert state["phase"] == "observing"
    prior = snapshot(session.output / "progress-000")
    scheduled.images[:] = [crop(end=210, dip=(70, 40)), crop(dip=(70, 40))]
    assert poll(scheduled, session)["phase"] == "observing"
    middle = snapshot(session.output / "progress-001")
    state = poll(scheduled, session)
    assert state["phase"] == "dip_observed" and session.finished
    assert state["prior_dip_observed"] is True
    assert state["waiting_for_positive_endpoint"] is False
    latest = session.output / "progress-002/report.json"
    assert session.progress_path == latest
    assert json.loads(latest.read_bytes())["endpoint_visible"] is True
    # The producer deliberately never asserts physical observation completion.
    assert json.loads(latest.read_bytes())["observation_completed"] is False
    assert session.report["progress_path"] == str(latest.relative_to(scheduled.root))
    assert session.report["progress_sha256"] == hashlib.sha256(latest.read_bytes()).hexdigest()
    assert session.report["scheduling_rule"] == RULE
    assert snapshot(session.output / "progress-000") == prior
    assert snapshot(session.output / "progress-001") == middle
    assert not session.receipt and not state["task_completed"]
    calls = list(scheduled.calls)
    session.advance()
    assert scheduled.calls == calls
    assert calls == [("start", 5000)] + [("capture", 5000)] * 3


def test_prior_dip_clean_partial_then_clean_endpoint_is_conflict_not_no(scheduled):
    session, _ = start_partial(scheduled)
    previous = snapshot(session.output / "progress-000")
    scheduled.images[:] = [crop(end=210), crop()]
    partial = poll(scheduled, session)
    assert partial["phase"] == "observing"
    assert partial["window_status"] == "still_collecting"
    assert partial["prior_dip_observed"] is True
    assert partial["waiting_for_positive_endpoint"] is True
    state = poll(scheduled, session)
    assert state["phase"] == "stopped" and session.finished
    assert state["failure_reason"] == "planet_window_prior_dip_conflict"
    # Keep even the conflicting current per-capture analysis truthful.
    assert session.analysis["status"] == "assume_no_planet"
    assert json.loads((session.output / "progress-002/analysis.json").read_bytes()) == session.analysis
    assert snapshot(session.output / "progress-000") == previous
    assert not session.receipt and not state["task_completed"]
    assert not any(call[0] == "choice" for call in scheduled.calls)
    before = list(scheduled.calls)
    session.advance()
    assert scheduled.calls == before


def test_clean_partial_between_dips_does_not_erase_latch(scheduled):
    session, _ = start_partial(scheduled)
    scheduled.images[:] = [crop(end=210), crop(dip=(70, 40))]
    assert poll(scheduled, session)["waiting_for_positive_endpoint"] is True
    state = poll(scheduled, session)
    assert state["phase"] == "dip_observed" and state["prior_dip_observed"] is True
    assert session.progress_path == session.output / "progress-002/report.json"
    assert not session.receipt


def test_supplemental_detector_positive_also_requires_current_endpoint(scheduled, monkeypatch):
    import habfly.browser_planet_window as module

    original = module.analyze_planet_window

    def unresolved(*args, **kwargs):
        return {
            **original(*args, **kwargs),
            "status": "insufficient_visual_evidence",
            "reason": "unknown_plot_palette",
        }

    monkeypatch.setattr(module, "analyze_planet_window", unresolved)
    session, state = start_partial(scheduled, partial_png=encode_png(positive_image(end=130)))
    assert state["phase"] == "observing" and state["prior_dip_observed"] is True
    directory = session.output / "progress-000"
    saved = snapshot(directory)
    assert json.loads(saved["positive-dip-analysis.json"])["status"] == "dip_observed"
    assert session.analysis["negative_policy_reason"] == "unknown_plot_palette"
    scheduled.images[:] = [encode_png(positive_image())]
    state = poll(scheduled, session)
    assert state["phase"] == "dip_observed"
    assert session.progress_path == session.output / "progress-001/report.json"
    assert snapshot(directory) == saved
    assert not session.receipt


@pytest.mark.parametrize(
    "leaf", ["chart.png", "report.json", "analysis.json", "positive-dip-analysis.json", "policy.json"]
)
@pytest.mark.parametrize("change", ["changed", "missing", "symlink"])
def test_prior_dip_pins_are_checked_even_on_idle_ticks(scheduled, monkeypatch, leaf, change):
    import habfly.browser_planet_window as module

    if leaf == "positive-dip-analysis.json":
        original = module.analyze_planet_window
        monkeypatch.setattr(
            module,
            "analyze_planet_window",
            lambda *a, **k: {
                **original(*a, **k),
                "status": "insufficient_visual_evidence",
                "reason": "unknown_plot_palette",
            },
        )
    session, _ = start_partial(
        scheduled,
        allow_baseline_edge_reference=leaf == "policy.json",
        partial_png=encode_png(positive_image(end=130)) if leaf == "positive-dip-analysis.json" else None,
    )
    target = session.output / "progress-000" / leaf
    original_bytes = target.read_bytes()
    if change == "changed":
        target.write_bytes(original_bytes + b"\n")
    elif change == "missing":
        target.unlink()
    else:
        other = scheduled.root / "replacement"
        other.write_bytes(original_bytes)
        target.unlink()
        target.symlink_to(other)
    before = list(scheduled.calls)
    # Still before next_poll: no additional native read should be needed.
    state = session.advance()
    assert state["phase"] == "stopped"
    assert state["failure_reason"] == "planet_window_prior_dip_source_changed"
    assert scheduled.calls == before and not session.receipt
    session.advance()
    assert scheduled.calls == before


def test_prior_dip_endpoint_unresolved_cannot_fall_back_to_shallow_handoff(scheduled):
    session, _ = start_partial(scheduled, allow_shallow_reference=True)
    scheduled.images[:] = [crop(stray=(90, 70))]
    state = poll(scheduled, session)
    assert state["phase"] == "stopped"
    assert state["failure_reason"] == "planet_window_prior_dip_conflict"
    assert session.analysis["status"] == "insufficient_visual_evidence"
    assert not session.receipt and not state["task_completed"]
    assert not any(call[0] == "choice" for call in scheduled.calls)


def test_prior_source_rechecked_after_endpoint_observation_callback(scheduled):
    session, _ = start_partial(scheduled)
    prior = session.output / "progress-000/analysis.json"
    scheduled.images[:] = [crop(dip=(70, 40))]

    def mutate(kind, _):
        if kind == "observation":
            prior.write_bytes(prior.read_bytes() + b"\n")

    session.emit = mutate
    state = poll(scheduled, session)
    assert state["phase"] == "stopped"
    assert state["failure_reason"] == "planet_window_prior_dip_source_changed"
    assert not session.receipt and not state["task_completed"]


@pytest.mark.parametrize("boundary", ["deadline", "polls", "abort", "paused_past_deadline"])
def test_positive_wait_preserves_original_fixed_bounds(scheduled, boundary):
    session, state = start_partial(scheduled, max_seconds=30, max_polls=1)
    assert state["phase"] == "observing"
    deadline, next_poll = session.deadline, session.next_poll
    before = list(scheduled.calls)
    if boundary == "abort":
        session.abort()
        reason, phase = "operator_aborted", "aborted"
    elif boundary == "polls":
        scheduled.clock[0] = next_poll
        reason, phase = "planet_window_poll_limit", "stopped"
    else:
        # Parent pause means no advance calls; wall-time remains consumed.
        scheduled.clock[0] = deadline
        reason, phase = "planet_window_time_limit", "stopped"
    state = session.advance()
    assert state["phase"] == phase and state["failure_reason"] == reason
    assert session.deadline == deadline and session.scope["max_seconds"] == 30
    assert session.scope["max_polls"] == 1 and session.polls == 1
    assert not state["waiting_for_positive_endpoint"]
    assert scheduled.calls == before and not session.receipt
    session.advance()
    assert scheduled.calls == before


def test_callback_abort_after_partial_observation_prevents_next_capture(scheduled):
    scheduled.images[:] = [crop(end=130, dip=(70, 40)), crop(dip=(70, 40))]
    session = make(scheduled, select_no=True)
    session.emit = lambda kind, _: session.abort() if kind == "observation" else None
    session.advance()
    state = poll(scheduled, session)
    assert state["phase"] == "aborted" and session.finished
    before = list(scheduled.calls)
    scheduled.clock[0] += 10
    session.advance()
    assert scheduled.calls == before == [("start", 5000), ("capture", 5000)]
    assert not session.receipt


@pytest.mark.parametrize("dip,phase", [(None, "ready_to_select_no"), ((70, 40), "dip_observed")])
def test_endpoint_first_poll_retains_existing_outcomes(scheduled, dip, phase):
    scheduled.images[:] = [crop(dip=dip)]
    session = make(scheduled, select_no=True)
    session.advance()
    state = poll(scheduled, session)
    assert state["phase"] == phase
    assert state["waiting_for_positive_endpoint"] is False
    assert scheduled.calls == [("start", 5000), ("capture", 5000)]
    assert not session.receipt and not state["task_completed"]


@pytest.mark.parametrize("boundary", ["idle", "before_play_callback"])
def test_scheduling_rule_drift_stops_before_next_capture_or_play(scheduled, boundary):
    if boundary == "idle":
        session, _ = start_partial(scheduled)
        session.scope["scheduling_rule"] = "unchecked_partial_handoff"
    else:
        session = make(scheduled)

        def mutate(kind, _):
            if kind == "state":
                session.scope["scheduling_rule"] = "unchecked_partial_handoff"

        session.emit = mutate
    before = list(scheduled.calls)
    state = session.advance()
    assert state["phase"] == "stopped"
    assert state["failure_reason"] == "planet_window_scheduling_rule_changed"
    assert scheduled.calls == before


@pytest.mark.parametrize("change", ["option", "scope", "scope_file"])
def test_fixed_scope_or_no_option_cannot_drift_while_waiting(scheduled, change):
    session, _ = start_partial(scheduled)
    if change == "option":
        session.select_no = False
    elif change == "scope":
        session.scope["max_polls"] += 1
    else:
        path = session.output / "scope.json"
        value = json.loads(path.read_bytes())
        value["scheduling_rule"] = "unchecked_partial_handoff"
        path.write_text(json.dumps(value))
    before = list(scheduled.calls)
    state = session.advance()
    assert state["phase"] == "stopped" and not state["task_completed"]
    assert scheduled.calls == before


@pytest.mark.parametrize("key,value", [("endpoint_visible", 1), ("observation_completed", 0)])
def test_recomputed_readiness_rejects_boolean_numeric_aliases(scheduled, monkeypatch, key, value):
    import habfly.browser_planet_window as module

    capture = module.capture_observation_progress

    def aliased(page, config, output, **options):
        report = capture(page, config, output, **options)
        report[key] = value
        (output / "report.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(module, "capture_observation_progress", aliased)
    scheduled.images[:] = [crop(dip=(70, 40))]
    session = make(scheduled)
    session.advance()
    state = poll(scheduled, session)
    assert state["phase"] == "stopped"
    assert state["failure_reason"] == "planet_window_readiness_changed"
    assert not session.receipt and not state["task_completed"]


@pytest.mark.parametrize("change", ["added_file", "added_directory", "parent_symlink"])
def test_prior_capture_directory_membership_and_parent_identity_are_closed(scheduled, change):
    session, _ = start_partial(scheduled)
    directory = session.output / "progress-000"
    if change == "added_file":
        (directory / "unexpected.json").write_text("{}")
    elif change == "added_directory":
        (directory / "unexpected").mkdir()
    else:
        moved = session.output / "moved-capture"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
    before = list(scheduled.calls)
    state = session.advance()
    assert state["phase"] == "stopped"
    assert state["failure_reason"] == "planet_window_prior_dip_source_changed"
    assert scheduled.calls == before


def test_no_selection_backstop_rejects_prior_dip_even_if_phase_corrupted(scheduled):
    session, _ = start_partial(scheduled)
    session.phase = "ready_to_select_no"
    before = list(scheduled.calls)
    state = session.advance()
    assert state["phase"] == "stopped"
    assert state["failure_reason"] == "planet_window_prior_dip_conflict"
    assert scheduled.calls == before and not session.receipt


def test_capture_that_exhausts_original_deadline_cannot_handoff_positive(scheduled, monkeypatch):
    import habfly.browser_planet_window as module

    capture = module.capture_observation_progress

    def expensive(page, config, output, **options):
        report = capture(page, config, output, **options)
        scheduled.clock[0] += 30
        return report

    monkeypatch.setattr(module, "capture_observation_progress", expensive)
    scheduled.images[:] = [crop(dip=(70, 40))]
    session = make(scheduled, max_seconds=30)
    deadline = session.deadline
    session.advance()
    state = poll(scheduled, session)
    assert scheduled.clock[0] > deadline
    assert state["phase"] == "stopped"
    assert state["failure_reason"] == "planet_window_time_limit"
    assert session.deadline == deadline and not session.receipt
