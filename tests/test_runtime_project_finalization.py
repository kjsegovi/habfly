"""Runtime ownership/scheduling seams only; no browser or invented live receipts."""

import io
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_project_compact_wire import envelope, independent_reconstruction, wrap
from test_runtime_browser_project import options

import habfly.browser_project_finalize_steps as finalizer_module
import habfly.runtime as runtime_module
from habfly.project_wire import compact_project_event
from habfly.runtime import RunOptions, Runtime, parse_run_options


@pytest.fixture
def rig(tmp_path, monkeypatch):
    calls, constructors = [], []
    runtime = Runtime(io.StringIO())
    runtime.options = parse_run_options(
        options(
            tmp_path,
            stars=30,
            project_campaign=True,
            project_campaign_max_seconds=10800,
            project_allow_scoring=True,
            project_allow_submission=True,
            project_compact_wire=True,
        )
    )
    runtime.run_id = "fresh-attempt"
    runtime.trace_path = runtime.options.artifact_dir / "fresh-attempt.jsonl"
    runtime._finalization_authorization = runtime._authorization()
    runtime.status = "paused"
    output = runtime.options.artifact_dir / runtime.run_id
    output.mkdir(parents=True)
    report = {
        "status": "handoff",
        "phase": "awaiting_assessment",
        "finished": True,
        "project_campaign": True,
        "target_stars": 30,
        "target_workflows_verified": True,
        "task_completed": False,
        "project_completed": False,
        "failure_reason": None,
        "project_progress": {"collected": 30, "verified": 30},
    }
    (output / "report.json").write_text(json.dumps(report))
    bridge = SimpleNamespace(
        finished=True,
        status="handoff",
        phase="awaiting_assessment",
        report=report,
        output=output,
        page=object(),
        config=object(),
        journal=object(),
        closed=False,
        state=lambda: deepcopy(report),
        advance_if_due=lambda: calls.append("campaign_tick"),
        command=lambda command: calls.append(("campaign_command", command["command"])),
    )

    def close_bridge():
        calls.append("browser_close")
        bridge.closed = True

    bridge.close = close_bridge
    runtime.env = bridge
    value = SimpleNamespace(
        runtime=runtime,
        calls=calls,
        constructors=constructors,
        bridge=bridge,
        output=output,
        factory_hook=None,
        start_hook=None,
        advance_hook=None,
        bad_completion=False,
        close_failure=False,
    )

    class Finalizer:
        def __init__(self, page, config, output, **kw):
            calls.append("construct")
            constructors.append(self)
            assert page is bridge.page and config is bridge.config
            assert output == bridge.output / "finalization"
            assert kw["run_history"] == kw["campaign_runtime_dir"] == bridge.output
            assert kw["journal"] is bridge.journal
            assert kw["allow_score_transfer"] is True
            self.kw, self.output = kw, output
            self.status, self.phase, self.finished, self.actions = "idle", "not_started", False, 0
            self.sequence = 0
            if value.factory_hook:
                value.factory_hook()

        def state(self):
            return {
                "mode": "bounded_post_campaign_finalization",
                "status": self.status,
                "phase": self.phase,
                "finished": self.finished,
                "task_completed": value.bad_completion,
                "project_completed": value.bad_completion,
                "submitted": False,
                "allow_score_transfer": True,
                "allow_submission": self.kw["allow_submission"],
                "scoring_max_seconds": 600,
                "submission_max_seconds": 180,
                "scoring_max_advances": 46,
                "submission_max_advances": 4,
                "project_progress": {"collected": 30, "verified": 30},
                "finalization_component": {
                    "phase": self.phase,
                    "submit_write_may_have_occurred": self.actions > 0,
                },
                "submission_outcome": "unknown" if self.finished and self.actions else "not_dispatched",
            }

        def emit(self):
            self.kw["emit"](
                {
                    "version": 1,
                    "event": "state",
                    "run_id": "finalizer",
                    "sequence": self.sequence,
                    "payload": self.state(),
                }
            )
            self.sequence += 1

        def start(self, *, paused):
            calls.append(("start_finalizer", paused))
            self.phase, self.status = "scoring_initializing", "paused" if paused else "running"
            if value.start_hook:
                value.start_hook()
            self.emit()

        def step(self):
            assert self.status == "paused"
            self.advance()

        def tick(self):
            if self.status == "running":
                self.advance()

        def advance(self):
            if self.finished:
                return
            if value.advance_hook:
                value.advance_hook()
            if self.kw["cancelled"]():
                return self.abort()
            self.actions += 1
            calls.append("child_advance")
            self.phase = "scoring_active"
            if self.actions == 3:
                self.finished, self.status, self.phase = True, "stopped", "unknown_pending"
            self.emit()

        def pause(self):
            if not self.finished:
                self.status = "paused"

        def resume(self):
            if not self.finished:
                self.status = "running"

        def abort(self):
            calls.append("finalizer_abort")
            if not self.finished:
                self.finished, self.status = True, "aborted"
                self.phase = "unknown_pending" if self.actions else "aborted"

        def close(self):
            calls.append("finalizer_close")
            self.abort()
            if value.close_failure:
                raise ValueError("private cleanup text")

    monkeypatch.setattr(finalizer_module, "BrowserProjectFinalizeSteps", Finalizer)
    yield value
    value.close_failure = False
    runtime.close()


def command(rig, name, payload=None):
    rig.runtime.command({"command": name, "payload": payload or {}})


def queued(rig, *, running=False):
    rig.runtime.status = "running" if running else "paused"
    if running:
        rig.runtime.advance_if_due()
    else:
        command(rig, "step")
    assert rig.runtime._finalization_pending
    return rig.runtime


def initialized(rig):
    runtime = queued(rig)
    command(rig, "step")
    assert runtime.finalization is rig.constructors[0]
    return runtime


def test_default_legacy_no_scoring_or_constructor(rig):
    runtime = rig.runtime
    runtime.options.project_allow_scoring = runtime.options.project_allow_submission = False
    runtime._finalization_authorization = runtime._authorization()
    for _ in range(20):
        runtime.advance_if_due()
    assert not rig.constructors and not runtime._owns_finalization()
    assert runtime.status == "handoff"


def test_separate_paused_stage_preserves_browser_report_and_budgets(rig):
    before = (rig.output / "report.json").read_bytes()
    runtime = queued(rig)
    assert not rig.constructors and runtime.status == "paused"
    for _ in range(10):
        runtime.advance_if_due()
    assert not rig.constructors
    command(rig, "step")
    assert len(rig.constructors) == 1 and runtime.finalization.actions == 0
    assert runtime.finalization.status == "paused"
    command(rig, "step")
    assert runtime.finalization.actions == 1
    assert (rig.output / "report.json").read_bytes() == before
    state = runtime._finalization_state()
    assert state["scoring_max_seconds"] == 600 and state["submission_max_seconds"] == 180
    assert state["campaign_budget_extended"] is False
    assert "browser_close" not in rig.calls


def test_running_handoff_constructs_on_next_tick_without_first_child_step(rig):
    runtime = queued(rig, running=True)
    assert runtime.status == "running" and not rig.constructors
    runtime.advance_if_due()
    assert len(rig.constructors) == 1 and runtime.finalization.actions == 0
    assert runtime.finalization.status == "running"
    runtime.last_tick = 0
    runtime.advance_if_due()
    assert runtime.finalization.actions == 1
    assert rig.calls.count("campaign_tick") == 1


@pytest.mark.parametrize("when", ("pending", "constructor", "start", "advance"))
def test_abort_before_boundary_prevents_any_child_actions(rig, when):
    runtime = queued(rig)
    if when == "pending":
        command(rig, "abort")
    else:
        setattr(
            rig,
            {"constructor": "factory_hook", "start": "start_hook", "advance": "advance_hook"}[when],
            lambda: command(rig, "abort"),
        )
        command(rig, "step")
        if when == "advance":
            command(rig, "step")
    for _ in range(5):
        runtime.advance_if_due()
        command(rig, "step")
    assert not rig.constructors or runtime.finalization.actions == 0
    assert runtime.status == "aborted"
    assert rig.calls.count("construct") <= 1


def test_pausing_pending_in_publish_callback_preserves_unadvanced_start(rig, monkeypatch):
    runtime, original = rig.runtime, rig.runtime.emit
    paused = False

    def intercept(kind, payload):
        nonlocal paused
        result = original(kind, payload)
        if payload.get("phase") == "finalization_pending" and not paused:
            paused = True
            command(rig, "pause")
        return result

    monkeypatch.setattr(runtime, "emit", intercept)
    queued(rig, running=True)
    assert runtime.status == "paused"
    runtime.advance_if_due()
    assert not rig.constructors
    command(rig, "step")
    assert runtime.finalization.actions == 0 and runtime.finalization.status == "paused"


def test_aborting_pending_in_publish_callback_never_constructs(rig, monkeypatch):
    runtime, original = rig.runtime, rig.runtime.emit

    def intercept(kind, payload):
        result = original(kind, payload)
        if payload.get("phase") == "finalization_pending":
            command(rig, "abort")
        return result

    monkeypatch.setattr(runtime, "emit", intercept)
    runtime.status = "running"
    runtime.advance_if_due()
    assert runtime.status == "aborted" and not runtime._finalization_pending
    command(rig, "step")
    assert not rig.constructors


def test_abort_during_campaign_terminal_callback_blocks_automatic_continuation(rig):
    rig.bridge.advance_if_due = lambda: command(rig, "abort")
    rig.runtime.status = "running"
    rig.runtime.advance_if_due()
    assert not rig.constructors and not rig.runtime._finalization_pending
    assert rig.runtime.status == "aborted"
    for _ in range(3):
        rig.runtime.advance_if_due()
    assert not rig.constructors


def test_failed_constructor_is_sticky_no_retry_and_sanitized(rig):
    runtime = queued(rig)

    def fail():
        raise ValueError("http://localhost?password=private-value")

    rig.factory_hook = fail
    command(rig, "step")
    for _ in range(10):
        runtime.advance_if_due()
        command(rig, "step")
    assert len(rig.constructors) == 1 and runtime.status == "stopped"
    assert "private-value" not in runtime.output.getvalue()


@pytest.mark.parametrize(
    "field,value",
    (
        ("project_allow_scoring", False),
        ("project_allow_submission", False),
        ("project_allow_submission", 1),
        ("stars", 2),
    ),
)
def test_authorization_mutation_fails_closed_even_if_reverted(rig, field, value):
    runtime = queued(rig)
    prior = getattr(runtime.options, field)
    setattr(runtime.options, field, value)
    with pytest.raises(ValueError, match="browser_project_invalid_command"):
        command(rig, "step")
    setattr(runtime.options, field, prior)
    with pytest.raises(ValueError, match="browser_project_invalid_command"):
        command(rig, "step")
    assert not rig.constructors
    assert runtime._finalization_authorization_failed
    command(rig, "abort")
    assert runtime.status == "aborted"


def test_unsupported_step_payload_has_no_side_effects(rig):
    runtime = queued(rig)
    before = list(rig.calls)
    with pytest.raises(ValueError, match="browser_project_invalid_command"):
        command(rig, "step", {"allow_submission": True})
    assert rig.calls == before and runtime._finalization_pending


def test_unknown_pending_never_becomes_completed_or_restarts(rig):
    runtime = initialized(rig)
    for _ in range(3):
        command(rig, "step")
    assert runtime.finalization.phase == "unknown_pending" and runtime.status == "stopped"
    for _ in range(5):
        command(rig, "step")
        runtime.advance_if_due()
    assert runtime.finalization.actions == 3 and len(rig.constructors) == 1
    state = runtime._finalization_state()
    assert state["task_completed"] is False and state["project_completed"] is False
    assert state["submission_outcome"] == "unknown"


def test_true_child_completion_is_rejected_not_promoted(rig):
    runtime = initialized(rig)
    rig.bad_completion = True
    command(rig, "step")
    assert runtime.status == "stopped" and runtime.finalization.actions == 0
    last = json.loads(runtime.output.getvalue().splitlines()[-1])["payload"]
    assert last["task_completed"] is False and last["project_completed"] is False


def test_close_releases_child_before_browser_even_on_child_failure(rig):
    runtime = initialized(rig)
    rig.close_failure = True
    with pytest.raises(ValueError, match="project_finalization_cleanup_failed"):
        runtime.close()
    assert rig.calls.index("finalizer_close") < rig.calls.index("browser_close")
    assert runtime.env is None and runtime.finalization is None


def test_finalizer_stream_bindings_and_synthetic_state_do_not_claim_missing_record(rig):
    runtime = initialized(rig)
    events = [json.loads(line) for line in runtime.output.getvalue().splitlines()]
    assert any(e["payload"]["project_wire"]["source_trace"] == "finalization/events.jsonl" for e in events)
    queued_event = next(e for e in events if e["payload"].get("phase") == "finalization_pending")
    assert queued_event["payload"]["project_wire"]["source_trace"] is None
    assert (
        queued_event["payload"]["project_wire"]["source_origin"]
        == "runtime_generated_no_raw_component_record"
    )
    assert all(e["payload"]["project_completed"] is False for e in events)
    assert all(e["payload"]["current_star"] is None for e in events)


@pytest.mark.parametrize(
    "changes",
    (
        {"project_allow_scoring": True, "stars": 2},
        {"project_allow_scoring": True, "project_campaign": False},
        {"project_allow_scoring": False, "project_allow_submission": True},
        {"project_allow_scoring": "true"},
        {"project_allow_submission": 1},
    ),
)
def test_finalization_options_reject_unsupported_scope_offline(tmp_path, changes):
    value = options(tmp_path, stars=30, project_campaign=True, project_campaign_max_seconds=10800)
    value.update(changes)
    with pytest.raises(ValueError):
        parse_run_options(value)
    assert not (tmp_path / "runs").exists()


def test_flags_are_legacy_false_and_never_nonproject():
    assert RunOptions().project_allow_scoring is False
    assert RunOptions().project_allow_submission is False
    with pytest.raises(ValueError, match="requires_thirty"):
        parse_run_options({"project_allow_scoring": True})


@pytest.mark.parametrize(
    "identity",
    (
        "project.assessment",
        "project.panel.data_quality",
        "project.panel.scavenger_hunt",
        "project.inventory.data_quality",
        "project.inventory.scavenger_hunt",
        "project.inventory.score_transfer",
    ),
)
def test_finalization_native_events_preserve_relay_lineage(identity):
    source = envelope("action_proposed", {"kind": "CLICK", "target": "exact-assessment-control"})
    event, payload = wrap(source, ("project.finalization.scoring", identity))
    result = compact_project_event(event, payload, source_trace="finalization/events.jsonl")
    assert independent_reconstruction(result) == source


def test_compact_finalizer_preserves_uncertainty_and_fixed_budgets():
    payload = {
        "mode": "bounded_post_campaign_finalization",
        "phase": "unknown_pending",
        "status": "stopped",
        "task_completed": False,
        "project_completed": False,
        "scoring_max_seconds": 600,
        "submission_max_seconds": 180,
        "finalization_component": {
            "phase": "unknown_pending",
            "pending_canonical_action": {"kind": "submit", "revision": 90},
            "submit_write_may_have_occurred": True,
            "submit_click_returned": True,
            "readiness_readback_verified": True,
            "submission_outcome": "unknown",
        },
    }
    output = compact_project_event("state", payload, source_trace="finalization/events.jsonl")
    for key, value in payload.items():
        assert output[key] == value


def test_direct_finalizer_data_keeps_known_full_envelope(rig):
    runtime = initialized(rig)
    native = envelope("error", {"type": "SubmissionStop", "message": "unknown_pending"})
    result = runtime._finalization_event(native)
    assert independent_reconstruction(result.payload) == native
    assert result.payload["project_wire"]["source_trace"] == "finalization/events.jsonl"


@pytest.mark.parametrize(
    "header,trace",
    (
        ({"event": "error", "sequence": True}, "finalization/events.jsonl"),
        ({"event": "observation", "sequence": 0}, "finalization/events.jsonl"),
        ({"event": "error", "payload": {}, "sequence": 0}, "finalization/events.jsonl"),
        ({"event": "error", "sequence": 0}, None),
    ),
)
def test_native_source_header_cannot_forge_event_or_raw_record(header, trace):
    with pytest.raises(ValueError, match="project_compact_wire_invalid_event"):
        compact_project_event(
            "error", {"message": "unknown_pending"}, source_header=header, source_trace=trace
        )


def test_output_callback_failure_never_permits_next_child_write(rig, monkeypatch):
    runtime = queued(rig)
    original = runtime.emit

    def fail(kind, payload):
        if payload.get("mode") == "bounded_post_campaign_finalization":
            raise RuntimeError("secret-value http://localhost?session=private")
        return original(kind, payload)

    monkeypatch.setattr(runtime, "emit", fail)
    with pytest.raises(ValueError, match="browser_project_invalid_command"):
        command(rig, "step")
    assert runtime._finalization_cancelled and runtime.finalization.actions == 0
    monkeypatch.setattr(runtime, "emit", original)
    for _ in range(3):
        runtime.advance_if_due()
        command(rig, "step")
    assert runtime.finalization.actions == 0 and len(rig.constructors) == 1
    assert "secret-value" not in runtime.output.getvalue()


def test_unreadable_child_state_and_failed_abort_remain_stopped_and_redacted(rig, monkeypatch):
    runtime = initialized(rig)

    def unreadable():
        raise ValueError("private source path")

    monkeypatch.setattr(runtime.finalization, "state", unreadable)
    monkeypatch.setattr(runtime.finalization, "abort", unreadable)
    command(rig, "step")
    state = json.loads(runtime.output.getvalue().splitlines()[-1])["payload"]
    assert state["status"] == "stopped" and state["project_progress"] is None
    assert state["source_state_unavailable"] is True
    assert state["task_completed"] is False and state["project_completed"] is False
    assert runtime.finalization.actions == 0
    assert "private source" not in runtime.output.getvalue()
    monkeypatch.undo()  # Restore cleanup seam before fixture closes the child.


def test_protocol_error_pauses_finalizer_not_retired_campaign(rig, monkeypatch):
    runtime = initialized(rig)
    command(rig, "resume")
    assert runtime.finalization.status == "running"
    # Retired campaign intentionally has no pause(): the finalizer alone owns
    # scheduler controls. serve() still closes the same original browser last.
    monkeypatch.setattr(runtime_module, "Runtime", lambda _output: runtime)
    runtime_module.serve(io.StringIO('{"command":"step","payload":{"unsupported":true}}\n'))
    states = [
        json.loads(line)["payload"]
        for line in runtime.output.getvalue().splitlines()
        if json.loads(line)["event"] == "state"
    ]
    assert any(
        p.get("mode") == "bounded_post_campaign_finalization" and p.get("status") == "paused" for p in states
    )
    assert "child_advance" not in rig.calls


def test_revoked_authorization_does_not_generate_repeated_automatic_failures(rig):
    runtime = initialized(rig)
    command(rig, "resume")
    runtime.options.project_allow_submission = False
    with pytest.raises(ValueError, match="authorization_changed"):
        runtime.advance_if_due()
    before = len(runtime.output.getvalue())
    for _ in range(20):
        runtime.advance_if_due()
    assert len(runtime.output.getvalue()) == before
    assert runtime.status == "stopped" and runtime.finalization.actions == 0
