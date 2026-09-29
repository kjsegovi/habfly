"""Cooperative scheduling tests; fake policies never represent learned scores."""
# ruff: noqa: F811

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_browser_full_stellar import full_page, select  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401

import habfly.browser_stellar_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.contracts import Action, Observation, StepResult


@pytest.fixture
def rig(tmp_path, monkeypatch):
    flags = {"end": 3, "success": True, "failure": None, "exception": None, "startup": None, "act_hook": None}
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen fixture checkpoint")
    experiment = tmp_path / "experiment"
    (experiment / "training").mkdir(parents=True)
    color_checkpoint = experiment / "training/checkpoint.pt"
    color_checkpoint.write_bytes(b"frozen color fixture checkpoint")
    fake_page = SimpleNamespace(remove_listener=lambda *_: None)
    policies = []

    class Policy:
        def __init__(self):
            self.inputs, self.activities = [], 0
            policies.append(self)

        def act(self, observation, state):
            assert not torch.is_grad_enabled()
            self.inputs.append(state)
            if flags["act_hook"]:
                flags["act_hook"]()
            return (
                Action(kind="WAIT", action_confidence=0.99, target_confidence=0.99, calibrated=True),
                torch.tensor(float(len(self.inputs))),
                {"activity_pathway": "fixture_path", "activity": 1},
            )

        def neural_activity(self, state):
            assert not torch.is_grad_enabled()
            self.activities += 1
            return {"top_neurons": [], "fixture_state": state.item()}

    class Session:
        def __init__(self, page, config, journal, *args, **kwargs):
            self.journal, self.attempts, self.verified, self.stopped = journal, 0, [], False
            self._dialog = lambda _: None

        def start(self):
            if flags["startup"]:
                raise flags["startup"]
            self.journal.emit("state", {"native_session_ready": True})

    class Environment:
        def __init__(self, session, *args):
            self.session, self.steps = session, 0
            self.completed = self.transport_verified = False

        def observe(self):
            return Observation(instruction="fixture", progress={"task_completed": False})

        def step(self, action, **kwargs):
            assert (
                action.action_confidence is None
                and action.target_confidence is None
                and not action.calibrated
            )
            self.session.journal.emit("state", {"native_write_boundary": self.steps})
            if flags["exception"]:
                raise flags["exception"]
            self.session.attempts += 1
            self.steps += 1
            terminal = self.steps >= flags["end"]
            if terminal and flags["success"]:
                self.transport_verified = self.completed = True
                self.session.verified = [
                    "distance",
                    "luminosity",
                    "temperature",
                    "mass",
                    "radius",
                    "lifetime",
                ]
                self.session.journal.receipt = {"selected_color": "UV", "readback_verified": True}
            return StepResult(
                observation=self.observe(),
                steps=self.steps,
                terminated=terminal,
                failure_reason=flags["failure"],
            )

    monkeypatch.setattr(module, "FullStellarSession", Session)
    monkeypatch.setattr(module, "FullStellarColorSession", Session)
    monkeypatch.setattr(module, "FullStellarToolEnv", Environment)
    monkeypatch.setattr(module, "BrowserColorEnv", Environment)
    monkeypatch.setattr(
        module,
        "load_full_stellar_policy",
        lambda *_: (
            Policy(),
            {
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "graph_hash": "graph",
                "knowledge_pack_hash": "pack",
                "optimizer_updates": 0,
            },
        ),
    )
    monkeypatch.setattr(
        module, "require_browser_color_gate", lambda *_: (Policy(), SimpleNamespace(checksum="reference"))
    )

    def create(kind, name="run", emit=lambda _: None):
        common = {
            "selected_class": "main_sequence",
            "lifetime_prefix": "Ga",
            "graph_path": tmp_path / "graph",
            "emit": emit,
        }
        if kind == "numeric":
            return module.FullStellarNumericSteps(
                fake_page,
                None,
                tmp_path / name,
                dataset=tmp_path / "dataset",
                checkpoint=checkpoint,
                **common,
            )
        return module.FullStellarColorSteps(fake_page, None, tmp_path / name, experiment=experiment, **common)

    return SimpleNamespace(
        flags=flags,
        create=create,
        checkpoint=checkpoint,
        color_checkpoint=color_checkpoint,
        policies=policies,
        root=tmp_path,
    )


@pytest.mark.parametrize("kind", ["numeric", "color"])
def test_one_decision_per_advance_pause_boundary_and_exact_forwarding(rig, kind):
    events = []
    component = rig.create(kind, emit=events.append)
    assert component.steps == 0 and component.session.attempts == 0
    assert component.state()["status"] == "ready" and rig.policies[0].inputs == []
    before = component.state()
    for _ in range(4):
        assert component.state() == before
    with torch.enable_grad():
        component.advance()
        assert torch.is_grad_enabled()
        assert component.steps == 1 and component.session.attempts == 1 and not component.finished
        for _ in range(2):
            component.advance()
            assert torch.is_grad_enabled()
    assert component.finished and component.state()["transport_verified"]
    assert component.report["outcome"] == (
        "full_stellar_numeric_transport_verified" if kind == "numeric" else "color_transport_verified"
    )
    assert not component.report["task_completed"]
    assert len(rig.policies[0].inputs) == 3 and rig.policies[0].inputs[0] is None
    assert rig.policies[0].inputs[1].item() == 1 and component._hidden is None
    actual = [json.loads(line) for line in (rig.root / "run/events.jsonl").read_text().splitlines()]
    assert events == actual
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert len({event["run_id"] for event in events}) == 1
    assert sum(event["event"] == "neural_activity" for event in events) == 3
    assert sum(event["event"] == "episode_summary" for event in events) == 1
    assert json.loads((rig.root / "run/manifest.json").read_text()) == component.report
    assert (
        component.report["events_sha256"]
        == hashlib.sha256((rig.root / "run/events.jsonl").read_bytes()).hexdigest()
    )
    snapshot = (rig.root / "run/events.jsonl").read_bytes()
    component.advance()
    component.abort()
    component.close()
    assert component.session.attempts == 3 and (rig.root / "run/events.jsonl").read_bytes() == snapshot
    assert rig.policies[0].activities == (3 if kind == "numeric" else 0)


@pytest.mark.parametrize("kind", ["numeric", "color"])
@pytest.mark.parametrize("at", [0, 1])
def test_abort_and_close_are_terminal_and_never_resume(rig, kind, at):
    component = rig.create(kind)
    for _ in range(at):
        component.advance()
    report = component.abort()
    assert report["outcome"] == "operator_aborted" and not component.state()["transport_verified"]
    assert component.finished and component.session.stopped and component._hidden is None
    assert component.close() is report
    component.advance()
    assert len(rig.policies[0].inputs) == at and component.session.attempts == at


@pytest.mark.parametrize("kind", ["numeric", "color"])
@pytest.mark.parametrize("when", ["between", "during_act"])
def test_changed_checkpoint_stops_before_the_next_native_write(rig, kind, when):
    component = rig.create(kind)
    path = rig.checkpoint if kind == "numeric" else rig.color_checkpoint
    if when == "between":
        component.advance()
        path.write_bytes(b"changed")
        expected = 1
    else:
        rig.flags["act_hook"] = lambda: path.write_bytes(b"changed")
        expected = 0
    component.advance()
    assert component.finished and not component.state()["checkpoint_unchanged"]
    assert component.report["outcome"] == (
        "checkpoint_changed" if kind == "numeric" else "color_checkpoint_changed"
    )
    assert component.session.attempts == expected


@pytest.mark.parametrize("kind", ["numeric", "color"])
@pytest.mark.parametrize("phase", ["startup", "exception"])
def test_unknown_exception_details_are_never_persisted(rig, kind, phase):
    rig.flags[phase] = RuntimeError("private password and URL")
    component = rig.create(kind)
    if phase != "startup":
        component.advance()
    assert component.finished and not component.state()["transport_verified"]
    assert "private password" not in (rig.root / "run/events.jsonl").read_text()
    assert "private password" not in (rig.root / "run/manifest.json").read_text()
    assert component.session.attempts == 0


@pytest.mark.parametrize("kind", ["numeric", "color"])
def test_termination_is_not_success_and_step_budget_is_exact(rig, kind):
    rig.flags["success"] = False
    rig.flags["end"] = 1
    stopped = rig.create(kind, "stopped")
    stopped.advance()
    assert stopped.finished and stopped.report["outcome"] == "policy_stopped"
    assert not stopped.state()["transport_verified"]
    rig.flags["end"] = 1000
    limited = rig.create(kind, "limited")
    for _ in range(limited._limit + 2):
        limited.advance()
    assert limited.steps == limited._limit
    assert limited.report["outcome"] == ("policy_step_limit" if kind == "numeric" else "color_step_limit")


@pytest.mark.parametrize("kind", ["numeric", "color"])
@pytest.mark.parametrize("event", ["action_proposed", "native_write_boundary"])
def test_callback_abort_stops_even_at_native_write_boundary(rig, kind, event):
    holder = {}

    def emit(message):
        if message["event"] == event or event in message["payload"]:
            holder["component"].abort()

    component = rig.create(kind, emit=emit)
    holder["component"] = component
    component.advance()
    assert component.finished and component.report["outcome"] == "operator_aborted"
    assert component.session.attempts == 0 and component._hidden is None
    component.advance()
    assert component.session.attempts == 0


@pytest.mark.parametrize("kind", ["numeric", "color"])
@pytest.mark.parametrize("event", ["hello", "action_proposed", "native_write_boundary"])
def test_callback_failure_is_redacted_and_stops_before_writes(rig, kind, event):
    def emit(message):
        if message["event"] == event or event in message["payload"]:
            raise RuntimeError("private callback detail")

    component = rig.create(kind, emit=emit)
    component.advance()
    assert component.finished and component.report["outcome"] == "stellar_event_forwarding_failed"
    assert component.session is None or component.session.attempts == 0
    assert "private callback" not in (rig.root / "run/events.jsonl").read_text()


def test_hidden_state_is_reset_between_numeric_and_color(rig):
    numeric = rig.create("numeric", "numeric")
    numeric.advance()
    numeric.abort()
    color = rig.create("color", "color")
    color.advance()
    color.close()
    assert len(rig.policies) == 2 and all(policy.inputs[0] is None for policy in rig.policies)
    assert numeric._hidden is color._hidden is None


@pytest.mark.parametrize("kind", ["numeric", "color"])
def test_abort_inside_policy_act_cannot_restore_hidden_state_or_write(rig, kind):
    component = rig.create(kind)
    rig.flags["act_hook"] = component.abort
    component.advance()
    assert component.finished and component._hidden is None
    assert component.session.attempts == 0
    assert component.report["outcome"] == "operator_aborted"
    assert not component.state()["transport_verified"]


@pytest.mark.parametrize("kind", ["numeric", "color"])
def test_callback_failure_while_forwarding_error_still_finishes(rig, kind):
    rig.flags["exception"] = RuntimeError("private browser failure")

    def emit(message):
        if message["event"] == "error":
            raise RuntimeError("private error callback")

    component = rig.create(kind, emit=emit)
    component.advance()
    assert component.finished and component.journal.stream.closed
    assert component.report["outcome"] == component._failed_outcome
    assert component.state()["event_forwarding_failed"]
    assert component.session.stopped and component.session.attempts == 0
    assert "private" not in (rig.root / "run/events.jsonl").read_text()
    component.advance()
    assert len(rig.policies[0].inputs) == 1


@pytest.mark.parametrize("kind", ["numeric", "color"])
def test_loader_failure_is_redacted_before_a_browser_session(rig, monkeypatch, kind):
    def denied(*args):
        raise RuntimeError("private checkpoint path")

    monkeypatch.setattr(
        module, "load_full_stellar_policy" if kind == "numeric" else "require_browser_color_gate", denied
    )
    with pytest.raises(BrowserSafetyStop, match="checkpoint_loading_failed") as caught:
        rig.create(kind)
    assert "private checkpoint" not in str(caught.value)
    assert not (rig.root / "run").exists()


@pytest.mark.parametrize("kind", ["numeric", "color"])
def test_interrupt_during_inference_is_durable_and_restores_grad_mode(rig, kind):
    component = rig.create(kind)

    def interrupt():
        raise KeyboardInterrupt

    rig.flags["act_hook"] = interrupt
    with torch.enable_grad():
        with pytest.raises(KeyboardInterrupt):
            component.advance()
        assert torch.is_grad_enabled()
    assert component.finished and component.report["outcome"] == "operator_aborted"
    assert component.session.attempts == 0 and component.journal.stream.closed
    component.advance()
    assert len(rig.policies[0].inputs) == 1


@pytest.mark.parametrize("kind", ["numeric", "color"])
def test_artifact_finalization_failure_is_not_reported_as_success(rig, monkeypatch, kind):
    rig.flags["end"] = 1
    component = rig.create(kind)

    def failed_report():
        raise OSError("private artifact path")

    monkeypatch.setattr(component, "_final_report", failed_report)
    component.advance()
    assert component.finished and component.session.stopped and component.journal.stream.closed
    assert not component.state()["transport_verified"]
    assert component.report["outcome"] == "stellar_report_finalization_failed"
    receipt = (rig.root / "run/finalization_failed.json").read_text()
    assert json.loads(receipt) == component.report and "private" not in receipt
    component.advance()
    assert component.session.attempts == 1


@pytest.mark.skipif(
    os.environ.get("HABFLY_TEST_COOPERATIVE_REAL_GRAPH") != "1",
    reason="Opt-in frozen 2000-node manual-fixture inference, not sealed evaluation",
)
def test_real_frozen_components_on_intercepted_manual_fixture(full_page, tmp_path):
    # Same already-published manual fixture/seed used by existing browser
    # transfer checks. No training or sealed final-case evaluator is called.
    select(full_page, "main_sequence", "Ga")
    events = []
    numeric = module.FullStellarNumericSteps(
        full_page,
        config(),
        tmp_path / "numeric",
        selected_class="main_sequence",
        lifetime_prefix="Ga",
        dataset=Path("experiments/lifetime-003"),
        checkpoint=Path("experiments/lifetime-003/training/checkpoint.pt"),
        graph_path=Path("data/processed/graphs-v2/graph-2000"),
        seed=8500000,
        emit=events.append,
    )
    try:
        while not numeric.finished:
            numeric.advance()
        assert numeric.report["full_stellar_numeric_transport_verified"], numeric.report
        assert numeric.report["write_attempts"] == 6 and numeric.report["checkpoint_unchanged"]
        assert numeric.report["optimizer_updates"] == 0 and not numeric.report["task_completed"]
    finally:
        numeric.close()
    color = module.FullStellarColorSteps(
        full_page,
        config(),
        tmp_path / "color",
        selected_class="main_sequence",
        lifetime_prefix="Ga",
        experiment=Path("experiments/color-pilot-006"),
        graph_path=Path("data/processed/graphs-v2/graph-2000"),
        emit=events.append,
    )
    try:
        while not color.finished:
            color.advance()
        assert color.report["color_transport_verified"], color.report
        assert color.report["write_attempts"] == 1 and color.report["receipt"]["selected_color"] == "UV"
        assert color.report["provenance"]["optimizer_updates"] == 0 and not color.report["task_completed"]
    finally:
        color.close()
    assert not full_page.get_by_role("checkbox").is_checked()
    assert len({message["run_id"] for message in events}) == 2
