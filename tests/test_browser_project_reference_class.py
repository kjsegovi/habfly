"""Explicit decision → cooperative setup → owner handoff; injected drivers only."""

import json
from copy import deepcopy

import pytest
from test_browser_project_runtime import create, setup_ready, sha, write
from test_browser_project_runtime import rig as bridge_rig  # noqa: F401
from test_runtime_browser_project import awaiting_class, command, integrated, start  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.contracts import RuntimeEvent
from habfly.runtime import RunOptions, read_trace

RATIONALE = "The supplied visible course reference supports this explicit class choice; transport this reference decision without treating it as learned classification or a scientific proof."


@pytest.fixture
def fake_class_factory():
    class ExplicitClassFixture:
        def __init__(
            self,
            page,
            config,
            output,
            *,
            run_history,
            fresh_star,
            selected_class,
            reference_rationale,
            lifetime_prefix,
            emit,
            **limits,
        ):
            assert fresh_star == run_history / "initial-star"
            assert selected_class in {"main_sequence", "red_giant", "supergiant", "white_dwarf"}
            assert len(reference_rationale) >= 40
            self.output, self.history, self.emit = output, run_history, emit
            self.selected, self.prefix = selected_class, lifetime_prefix
            self.status, self.phase, self.finished, self.report = "ready", "class_pending", False, None
            self.advances, self.writes, self.closed, self.sequence = 0, [], False, 0
            self.limits = limits
            self.source_hashes = {}
            output.mkdir()
            write(
                output / "decision.json", {"selected_class": selected_class, "rationale": reference_rationale}
            )
            self.source_hashes[str((output / "decision.json").relative_to(run_history))] = sha(
                output / "decision.json"
            )
            self.stream = (output / "events.jsonl").open("x")

        def state(self):
            return {
                "star": "ALPHA",
                "status": self.status,
                "phase": self.phase,
                "finished": self.finished,
                "setup_verified": self.status == "completed",
                "task_completed": False,
                "scientific_verified": False,
                "training_label": False,
                "classification_learned": False,
                "selected_class": self.selected,
                "lifetime_prefix": self.prefix,
                "source_hashes": deepcopy(self.source_hashes),
                "advances": self.advances,
            }

        def event(self, kind, payload):
            item = RuntimeEvent(event=kind, payload=payload, sequence=self.sequence, run_id="class-fixture")
            self.sequence += 1
            self.stream.write(item.model_dump_json() + "\n")
            self.stream.flush()
            self.emit(item.model_dump(mode="json"))

        def advance(self):
            if self.finished:
                return self.state()
            self.advances += 1
            prefix = self.phase == "prefix_pending"
            self.event(
                "action_proposed",
                {
                    "kind": "SELECT",
                    "target": "lifetime_prefix" if prefix else "stellar_class",
                    "value": self.prefix if prefix else self.selected,
                },
            )
            if self.finished:
                return self.state()
            self.writes.append("prefix" if prefix else "class")
            write(self.output / ("prefix" if prefix else "class") / "confirmed.json", {"star": "ALPHA"})
            source = self.output / ("prefix" if prefix else "class") / "confirmed.json"
            self.source_hashes[str(source.relative_to(self.history))] = sha(source)
            self.event("action_result", {"readback_verified": True, "task_completed": False})
            if self.finished:
                return self.state()
            if not prefix and self.prefix:
                self.phase = "prefix_pending"
                self.event("state", self.state())
            else:
                self.status, self.phase, self.finished = "completed", "verified", True
                self.report = self.state()
                write(self.output / "report.json", self.report)
                self.event("episode_summary", {**self.report, "completed": False})
            return self.state()

        def abort(self):
            if not self.finished:
                self.status, self.phase, self.finished = "aborted", "aborted", True
                self.report = self.state()
                write(self.output / "report.json", self.report)
            return self.state()

        def close(self):
            self.closed = True
            self.abort()
            if not self.stream.closed:
                self.stream.close()

    return ExplicitClassFixture


@pytest.fixture
def rig(bridge_rig, fake_class_factory):  # noqa: F811
    bridge_rig.factory = fake_class_factory
    return bridge_rig


def ready(rig, **kwargs):
    bridge = setup_ready(rig, _class_factory=rig.factory, **kwargs)
    assert bridge.phase == "awaiting_class_source"
    return bridge


def reference(bridge, selected="main_sequence", prefix="Ga"):
    bridge.command(
        {
            "command": "step",
            "payload": {
                "reference_class": {
                    "selected_class": selected,
                    "reference_rationale": RATIONALE,
                    "lifetime_prefix": prefix,
                }
            },
        }
    )


def test_explicit_class_constructor_has_no_actions_and_each_step_has_one_write(rig):
    bridge = ready(rig)
    calls = len(rig.calls)
    reference(bridge)
    assert bridge.phase == "setting_reference_class" and bridge.status == "paused"
    assert len(rig.calls) == calls and not bridge.class_setup.writes
    bridge.advance_if_due()
    assert not bridge.class_setup.writes
    bridge.step()
    assert bridge.phase == "setting_lifetime_prefix" and bridge.class_setup.writes == ["class"]
    assert bridge.project.phase == "awaiting_class_source"
    bridge.step()
    assert bridge.phase == "class_setup_handoff" and bridge.class_setup.writes == ["class", "prefix"]
    assert bridge.project.phase == "awaiting_class_source"
    bridge.step()
    assert bridge.phase == "ready" and bridge.status == "paused"
    assert len([call for call in rig.calls if call[0] == "class_handoff"]) == 1
    assert not any(call[0] == "model_step" for call in rig.calls)
    bridge.close()


@pytest.mark.parametrize("selected", ["red_giant", "supergiant", "white_dwarf"])
def test_non_main_explicit_class_never_gets_a_prefix(rig, selected):
    bridge = ready(rig)
    reference(bridge, selected, None)
    bridge.step()
    assert bridge.phase == "class_setup_handoff" and bridge.class_setup.writes == ["class"]
    bridge.step()
    assert bridge.phase == "ready"
    bridge.close()


def test_resumed_class_setup_does_not_resume_owner_until_receipt_handoff(rig):
    bridge = ready(rig)
    reference(bridge)
    bridge.resume()
    assert bridge.status == "running" and bridge.project.status == "paused"
    for _ in range(3):
        bridge.tick()
    assert bridge.phase == "ready" and bridge.project.status == bridge.status == "running"
    assert not any(call[0] == "model_step" for call in rig.calls)
    bridge.tick()
    assert bridge.status == "handoff" and not bridge.report["task_completed"]
    bridge.close()


@pytest.mark.parametrize(
    "phase", ["setting_reference_class", "setting_lifetime_prefix", "class_setup_handoff"]
)
def test_abort_closes_class_setup_and_prevents_later_handoff(rig, phase):
    bridge = ready(rig)
    reference(bridge)
    while bridge.phase != phase:
        bridge.step()
    writes = len(bridge.class_setup.writes)
    bridge.abort()
    bridge.step()
    assert bridge.class_setup.closed and len(bridge.class_setup.writes) == writes
    assert bridge.status == "aborted" and not any(call[0] == "class_handoff" for call in rig.calls)
    bridge.close()


def test_component_summary_cannot_replace_owner_lifecycle_or_complete_task(rig):
    bridge = ready(rig)
    reference(bridge, "white_dwarf", None)
    bridge.step()
    assert bridge.phase == "class_setup_handoff" and bridge._project_state["phase"] == "awaiting_class_source"
    events = read_trace(bridge.trace_path)
    assert not any(e.event == "episode_summary" for e in events)
    summary = next(e for e in events if "component_summary" in e.payload)
    assert summary.payload["component"] == "browser.reference_class"
    assert not summary.payload["task_completed"] and not summary.payload["project_completed"]
    child = read_trace(bridge.class_setup.output / "events.jsonl")
    assert [
        e.payload["component_event"]
        for e in events
        if e.payload.get("component") == "browser.reference_class"
    ] == [e.model_dump(mode="json") for e in child]
    assert RATIONALE not in bridge.trace_path.read_text()
    bridge.close()


@pytest.mark.parametrize("change", ["initial", "report", "decision", "class", "stopped"])
def test_changed_setup_evidence_blocks_receipt_handoff(rig, change):
    bridge = ready(rig)
    reference(bridge, "white_dwarf", None)
    bridge.step()
    path = {
        "initial": bridge.output / "initial-star/stellar/observation.json",
        "report": bridge.class_setup.output / "report.json",
        "decision": bridge.class_setup.output / "decision.json",
        "class": bridge.class_setup.output / "class/confirmed.json",
        "stopped": bridge.class_setup.output / "stopped.json",
    }[change]
    path.write_text("{}")
    bridge.step()
    assert bridge.status == "stopped" and not any(c[0] == "class_handoff" for c in rig.calls)
    bridge.close()


@pytest.mark.parametrize("abort", [True, False])
def test_callback_abort_or_changed_initial_source_blocks_pending_class_write(rig, abort):
    holder = {}

    def callback(kind, payload):
        if kind == "action_proposed" and payload.get("component") == "browser.reference_class":
            if abort:
                holder["bridge"].abort()
            else:
                (holder["bridge"].output / "initial-star/stellar/observation.json").write_text("{}")

    bridge = holder["bridge"] = ready(rig, emit=callback)
    reference(bridge)
    bridge.step()
    assert bridge.finished and not bridge.class_setup.writes
    bridge.close()


def test_unverified_prefix_never_enters_owner(rig):
    bridge = ready(rig)
    reference(bridge)
    bridge.step()
    bridge.class_setup.abort()
    bridge.step()
    assert bridge.status == "stopped" and bridge.failure == "project_runtime_class_setup_unverified"
    assert not any(c[0] == "class_handoff" for c in rig.calls)
    bridge.close()


def stop_fixture_class(child, monkeypatch, reason, *, mutate=lambda report: None, emit=True):
    """Inject a persisted stopped child at the real outer handoff boundary."""
    original_state = child.state

    def state():
        report = {**original_state(), "failure_reason": reason, "event_forwarding_failed": False}
        mutate(report)
        return report

    def advance():
        child.status, child.phase, child.finished = "stopped", "stopped", True
        child.report = state()
        write(child.output / "report.json", child.report)
        write(child.output / "stopped.json", child.report)
        if emit:
            child.event("episode_summary", {**child.report, "completed": False})

    monkeypatch.setattr(child, "state", state)
    monkeypatch.setattr(child, "advance", advance)


@pytest.mark.parametrize("prefix_started", [False, True])
def test_stopped_class_report_preserves_cause_and_never_hands_off_or_retries(
    rig, monkeypatch, prefix_started
):
    bridge = ready(rig)
    reference(bridge)
    if prefix_started:
        bridge.step()
    stop_fixture_class(bridge.class_setup, monkeypatch, "unsupported_class_circle_rendering")
    writes = list(bridge.class_setup.writes)
    bridge.step()
    assert bridge.status == "stopped" and bridge.failure == "unsupported_class_circle_rendering"
    assert bridge.report["event_forwarding_failed"] is False
    assert bridge.report["task_completed"] is bridge.report["project_completed"] is False
    assert bridge._class_setup_report is None
    assert not (bridge.output / "reference-class-handoff.json").exists()
    stopped = json.loads((bridge.output / "report.json").read_bytes())
    assert stopped == bridge.report
    before = len(rig.calls)
    bridge.step()
    bridge.tick()
    bridge.advance_if_due()
    with pytest.raises(BrowserSafetyStop, match="explicit_handoff_required"):
        bridge.resume()
    assert len(rig.calls) == before and bridge.class_setup.writes == writes
    assert not any(c[0] in {"class_handoff", "model_step"} for c in rig.calls)
    assert bridge.failure == "unsupported_class_circle_rendering"
    bridge.close()


@pytest.mark.parametrize(
    "change",
    [
        {"star": "OTHER"},
        {"selected_class": "red_giant"},
        {"lifetime_prefix": "Ma"},
        {"task_completed": True},
        {"setup_verified": True},
        {"finished": False},
    ],
)
def test_unbound_or_inconsistent_stopped_class_cannot_supply_outer_cause(rig, monkeypatch, change):
    bridge = ready(rig)
    reference(bridge)
    stop_fixture_class(
        bridge.class_setup,
        monkeypatch,
        "unsupported_class_circle_rendering",
        mutate=lambda report: report.update(change),
    )
    bridge.step()
    assert bridge.status == "stopped" and bridge.failure == "project_runtime_class_setup_unverified"
    assert not any(c[0] == "class_handoff" for c in rig.calls)
    bridge.close()


def test_changed_persisted_stopped_report_cannot_supply_outer_cause(rig, monkeypatch):
    bridge = ready(rig)
    reference(bridge)
    child = bridge.class_setup
    stop_fixture_class(child, monkeypatch, "unsupported_class_circle_rendering", emit=False)
    advance = child.advance

    def changed_report():
        advance()
        write(child.output / "report.json", {**child.report, "star": "OTHER"})

    monkeypatch.setattr(child, "advance", changed_report)
    bridge.step()
    assert bridge.failure == "project_runtime_class_setup_unverified"
    bridge.close()


def test_real_callback_failure_remains_distinct_from_reported_class_stop(rig, monkeypatch):
    def callback(kind, payload):
        if kind == "state" and payload.get("component") == "browser.reference_class":
            raise RuntimeError("private callback details must not be exposed")

    bridge = ready(rig, emit=callback)
    reference(bridge)
    stop_fixture_class(bridge.class_setup, monkeypatch, "unsupported_class_circle_rendering")
    bridge.step()
    assert bridge.failure == "project_runtime_event_forwarding_failed"
    assert bridge.report["event_forwarding_failed"] is True
    assert not bridge.class_setup.writes and not any(c[0] == "class_handoff" for c in rig.calls)
    assert "private callback" not in bridge.trace_path.read_text()
    bridge.close()


@pytest.mark.parametrize(
    "reason",
    ["private session https://example.invalid/token", "private\ntext", {"private": "value"}, None],
)
def test_class_report_cause_and_display_are_sanitized_without_rewriting_source(rig, monkeypatch, reason):
    bridge = ready(rig)
    reference(bridge)
    child = bridge.class_setup
    # The real child sanitizes its event payloads. Inject a malformed report
    # without an event to exercise the additional outer report boundary only.
    stop_fixture_class(child, monkeypatch, reason, emit=False)
    bridge.step()
    assert bridge.failure == "project_runtime_operation_failed"
    assert bridge.state()["class_setup"]["failure_reason"] in {None, "project_runtime_operation_failed"}
    assert json.loads((child.output / "report.json").read_bytes())["failure_reason"] == reason
    assert "private" not in bridge.trace_path.read_text()
    assert "private" not in (bridge.output / "report.json").read_text()
    bridge.close()


def test_class_cleanup_failure_does_not_replace_first_cause(rig, monkeypatch):
    bridge = ready(rig)
    reference(bridge)
    child = bridge.class_setup
    stop_fixture_class(child, monkeypatch, "unsupported_class_circle_rendering")
    original_close = child.close

    def failed_close():
        original_close()
        raise RuntimeError("private cleanup details")

    monkeypatch.setattr(child, "close", failed_close)
    bridge.step()
    assert bridge.failure == "unsupported_class_circle_rendering"
    assert bridge.report["cleanup_failed"] is True
    assert bridge.report["event_forwarding_failed"] is False
    assert "private cleanup" not in bridge.trace_path.read_text()
    bridge.close()


def test_abort_during_stopped_class_summary_remains_operator_abort(rig, monkeypatch):
    holder = {}

    def callback(kind, payload):
        if kind == "state" and payload.get("component") == "browser.reference_class":
            holder["bridge"].abort()

    bridge = holder["bridge"] = ready(rig, emit=callback)
    reference(bridge)
    stop_fixture_class(bridge.class_setup, monkeypatch, "unsupported_class_circle_rendering")
    bridge.step()
    assert bridge.status == "aborted" and bridge.failure == "operator_aborted"
    assert not bridge.class_setup.writes and not any(c[0] == "class_handoff" for c in rig.calls)
    bridge.close()


def test_owner_inventory_phases_are_cooperative_and_automatic_flag_is_explicit(rig):
    received = []

    def owner(*args, **kwargs):
        received.append(kwargs["automatic_inventory"])
        return rig.steps(*args, **kwargs)

    bridge = ready(rig, steps=owner)
    assert received == [True] and bridge.scope["automatic_inventory"] is True
    for phase in ("inventory_initializing", "inventory_active", "inventory_import"):
        bridge.phase = bridge.project.phase = phase
        bridge.project.step = lambda: None
        bridge.step()
        assert bridge.status == "paused" and bridge.phase == phase
    bridge.close()


def test_v1_runtime_routes_reference_payload_without_inventing_fields(integrated, fake_class_factory):  # noqa: F811
    bridge = awaiting_class(integrated)
    bridge._class_factory = fake_class_factory
    command(
        integrated,
        "step",
        reference_class={
            "selected_class": "main_sequence",
            "reference_rationale": RATIONALE,
            "lifetime_prefix": "Ga",
        },
    )
    assert integrated.runtime.status == "paused" and bridge.phase == "setting_reference_class"
    for _ in range(3):
        command(integrated, "step")
    assert bridge.phase == "ready" and integrated.runtime.status == "paused"
    assert len(bridge.class_setup.writes) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"reference_class": "secret-session"},
        {"reference_class": {"selected_class": "main_sequence"}},
        {
            "reference_class": {
                "selected_class": "main_sequence",
                "reference_rationale": RATIONALE,
                "lifetime_prefix": "Ga",
                "credentials": "secret-session",
            }
        },
    ],
)
def test_malformed_reference_command_has_no_page_action_or_secret_echo(integrated, payload):  # noqa: F811
    bridge = awaiting_class(integrated)
    before = len(integrated.calls)
    with pytest.raises(ValueError) as error:
        command(integrated, "step", **payload)
    assert "secret-session" not in str(error.value)
    assert len(integrated.calls) == before and bridge.class_setup is None


def test_fixed_class_limits_are_explicit_and_not_mutated_after_start(rig):
    bridge = ready(rig, class_limits={"max_advances": 3, "max_seconds": 240})
    reference(bridge)
    assert bridge.scope["class_setup_limits"] == {"max_advances": 3, "max_seconds": 240}
    assert bridge.class_setup.limits["max_seconds"] == 240
    assert bridge.class_setup.limits["max_advances"] == 3
    bridge.close()


@pytest.mark.parametrize("limits", [{"max_advances": 5}, {"max_seconds": 601}, {"unknown": 1}])
def test_bad_class_limits_reject_before_browser_start(rig, limits):
    with pytest.raises(BrowserSafetyStop, match="invalid_class_limits"):
        create(rig, class_limits=limits)
    assert not rig.calls


def test_runtime_forwards_fixed_class_limits(integrated):  # noqa: F811
    bridge = start(integrated, project_class_max_advances=3, project_class_max_seconds=240)
    assert bridge.class_limits == {"max_advances": 3, "max_seconds": 240}
    assert not integrated.calls


@pytest.mark.parametrize(
    "change",
    [
        {"project_class_max_advances": 0},
        {"project_class_max_advances": 5},
        {"project_class_max_advances": True},
        {"project_class_max_seconds": 4},
        {"project_class_max_seconds": 601},
        {"project_class_max_seconds": float("inf")},
        {"project_class_max_seconds": True},
    ],
)
def test_runtime_class_budget_schema_is_fixed_and_finite(change):
    with pytest.raises(ValueError):
        RunOptions(**change)
