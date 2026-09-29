"""Cooperative fixtures are scheduling evidence, not learned acceptance scores."""
# ruff: noqa: F811

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_numeric import planet_page  # noqa: F401

import habfly.browser_planet_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.contracts import Action, Observation, StepResult
from habfly.environments.planet_calculations import FIELDS, QUANTITY_UNITS, SCOPE, planet_expert
from habfly.planet_knowledge import PlanetCalculator


def seed_sources(root):
    pilot, final = root / "pilot", root / "final"
    (pilot / "training").mkdir(parents=True)
    final.mkdir()
    checkpoint = pilot / "training/checkpoint.pt"
    checkpoint.write_bytes(b"mock checkpoint, not trained")
    metadata = checkpoint.with_suffix(".pt.json")
    metadata.write_text(
        json.dumps(
            {
                "provenance": {
                    "planet_calculations": {
                        "scope": SCOPE,
                        "knowledge_pack_hash": PlanetCalculator().pack.checksum,
                    }
                }
            }
        )
    )
    gate = final / "report.json"
    gate.write_text(
        json.dumps(
            {
                "scope": SCOPE,
                "checkpoint_sha256": module.file_hash(checkpoint),
                "checkpoint_unchanged": True,
                "optimizer_updates": 0,
                "final_test_episodes": 100,
                "learning_gate_passed": True,
                "closed_loop": {
                    "test": {
                        "episodes": 100,
                        "completed": 100,
                        "invalid_actions": 0,
                        "tool_errors": 0,
                        "infrastructure_failures": 0,
                        "api_failures": 0,
                    }
                },
            }
        )
    )
    return pilot, final, checkpoint, metadata, gate


@pytest.fixture
def rig(tmp_path, monkeypatch):
    pilot, final, checkpoint, metadata, gate = seed_sources(tmp_path)
    flags = {
        "end": 4,
        "success": True,
        "failure": None,
        "exception": None,
        "startup": None,
        "act_hook": None,
        "graph_hash": "fixture-graph",
    }
    policies, sessions = [], []
    graph = SimpleNamespace(body_ids=list(range(2000)))

    class Policy:
        hidden_size, observation_encoding = 16, "structured_planet_tool_v1"

        def __init__(self):
            self.inputs = []
            policies.append(self)

        def act(self, observation, state):
            assert not torch.is_grad_enabled()
            self.inputs.append(state)
            if flags["act_hook"]:
                flags["act_hook"]()
            return (
                Action(kind="WAIT", action_confidence=0.9, target_confidence=0.9, calibrated=True),
                torch.tensor(len(self.inputs)),
                {"fixture_activity": len(self.inputs)},
            )

    class Session:
        def __init__(self, page, config, output, emit, *, max_seconds):
            assert max_seconds == 900
            self.attempted, self.verified, self.writes = set(), {}, 0
            self.emit, self.stopped = emit, False
            sessions.append(self)
            if flags["startup"]:
                raise flags["startup"]

        def close(self):
            self.stopped = True

        def copy(self, destination):
            assert not self.stopped
            self.attempted.add(destination)
            intent = {"kind": "TYPE", "destination": destination, "action_source": "checkpoint"}
            self.emit("action_proposed", intent)
            assert not self.stopped
            self.writes += 1
            self.verified[destination] = {**intent, "readback_verified": True}
            self.emit("action_result", self.verified[destination])

    class Environment:
        def __init__(self, session, *, supplied_star_class, seed):
            self.session, self.steps, self.transport_verified = session, 0, False

        def observe(self):
            return Observation(instruction="fixture", progress={"task_completed": False})

        def step(self, action):
            assert action.action_confidence is action.target_confidence is None and not action.calibrated
            if flags["exception"]:
                raise flags["exception"]
            if self.steps < 4:
                self.session.copy(FIELDS[self.steps])
            self.steps += 1
            end = self.steps >= flags["end"]
            self.transport_verified = end and flags["success"]
            return StepResult(
                observation=self.observe(), steps=self.steps, terminated=end, failure_reason=flags["failure"]
            )

    monkeypatch.setattr(
        module, "load_checkpoint", lambda *a, **k: (Policy(), {"graph_hash": "fixture-graph"})
    )
    monkeypatch.setattr(module, "graph_fingerprint", lambda _: flags["graph_hash"])
    monkeypatch.setattr(module, "PlanetNumericSession", Session)
    monkeypatch.setattr(module, "PlanetBrowserToolEnv", Environment)

    def create(name="run", emit=lambda _: None):
        return module.PlanetDerivedSteps(
            None,
            None,
            tmp_path / name,
            pilot=pilot,
            final_evaluation=final,
            graph=graph,
            supplied_star_class="main_sequence",
            emit=emit,
        )

    return SimpleNamespace(
        create=create,
        root=tmp_path,
        flags=flags,
        policies=policies,
        sessions=sessions,
        checkpoint=checkpoint,
        metadata=metadata,
        gate=gate,
        graph=graph,
    )


def events(rig, name="run"):
    return [json.loads(line) for line in (rig.root / name / "events.jsonl").read_text().splitlines()]


def test_single_decisions_pause_and_exact_flushed_events_report(rig):
    received = []

    def callback(event):
        assert events(rig)[-1] == event  # callback occurs after flush, never before
        received.append(event)

    component = rig.create(emit=callback)
    assert not component.finished and component.steps == 0 and not rig.policies[0].inputs
    with torch.enable_grad():
        component.advance()
        assert torch.is_grad_enabled() and component.steps == 1 and component.session.writes == 1
        before = (rig.root / "run/events.jsonl").read_bytes()
        for _ in range(3):
            assert component.state()["status"] == "ready"
        assert (rig.root / "run/events.jsonl").read_bytes() == before
        for _ in range(3):
            component.advance()
            assert torch.is_grad_enabled()
    assert component.finished and component.report["planet_transport_verified"]
    assert component.report["outcome"] == "planet_derived_transport_verified"
    assert component.session.stopped and component._hidden is None
    assert rig.policies[0].inputs[0] is None and rig.policies[0].inputs[1].item() == 1
    assert received == events(rig)
    assert [e["sequence"] for e in received] == list(range(len(received)))
    assert len({e["run_id"] for e in received}) == 1
    assert sum(e["event"] == "neural_activity" for e in received) == 4
    assert sum(e["event"] == "episode_summary" for e in received) == 1
    report = component.report
    assert events(rig)[-1]["payload"] == {k: v for k, v in report.items() if k != "events_sha256"}
    assert report["events_sha256"] == hashlib.sha256((rig.root / "run/events.jsonl").read_bytes()).hexdigest()
    assert json.loads((rig.root / "run/report.json").read_text()) == report
    assert not any(
        report[k]
        for k in (
            "task_completed",
            "browser_acceptance_passed",
            "saved",
            "assessment_performed",
            "submitted",
            "optimizer_updates",
        )
    )
    before = (rig.root / "run/events.jsonl").read_bytes()
    assert component.abort() is report and component.close() is report
    component.advance()
    assert component.session.writes == 4 and (rig.root / "run/events.jsonl").read_bytes() == before


@pytest.mark.parametrize("at", [0, 1])
def test_abort_is_terminal_and_new_component_resets_hidden_state(rig, at):
    component = rig.create()
    for _ in range(at):
        component.advance()
    report = component.close()
    component.advance()
    assert component.finished and component.session.writes == at and component._hidden is None
    assert report["outcome"] == "operator_aborted" and not report["planet_transport_verified"]
    other = rig.create("other")
    other.advance()
    other.abort()
    assert rig.policies[-1].inputs == [None]


def test_abort_inside_policy_act_cannot_restore_hidden_state_or_write(rig):
    component = rig.create()
    rig.flags["act_hook"] = component.abort
    component.advance()
    assert component.finished and component._hidden is None and component.session.writes == 0
    assert component.report["outcome"] == "operator_aborted"


@pytest.mark.parametrize("source", ["checkpoint", "metadata", "gate", "graph", "deleted_metadata"])
@pytest.mark.parametrize("when", ["paused", "act", "native"])
def test_frozen_sources_cannot_change_before_native_copy(rig, source, when):
    def mutate():
        if source == "graph":
            rig.flags["graph_hash"] = "different"
        elif source == "deleted_metadata":
            rig.metadata.unlink()
        else:
            getattr(rig, source).write_bytes(b"changed fixture")

    def callback(event):
        if when == "native" and event["event"] == "action_proposed" and "destination" in event["payload"]:
            mutate()

    component = rig.create(emit=callback)
    if when == "paused":
        mutate()
    elif when == "act":
        rig.flags["act_hook"] = mutate
    component.advance()
    assert component.finished and not component.report["planet_transport_verified"]
    assert not component.report["sources_unchanged"] and component.session.writes == 0
    component.advance()
    assert component.session.writes == 0


@pytest.mark.parametrize("event", ["observation", "action_proposed", "neural_activity", "native"])
@pytest.mark.parametrize("effect", ["abort", "failure", "reentry"])
def test_callback_at_decision_and_native_reservation_cannot_dispatch(rig, event, effect):
    holder = {}

    def callback(message):
        native = message["event"] == "action_proposed" and "destination" in message["payload"]
        if event == "native" and native or event != "native" and message["event"] == event and not native:
            if effect == "abort":
                holder["component"].abort()
            elif effect == "reentry":
                holder["component"].advance()
            else:
                raise RuntimeError("private callback secret")

    component = rig.create(emit=callback)
    holder["component"] = component
    component.advance()
    assert component.finished and component.session.stopped and component.session.writes == 0
    assert not component.report["planet_transport_verified"]
    assert "private callback secret" not in (rig.root / "run/events.jsonl").read_text()
    component.advance()
    assert component.session.writes == 0


@pytest.mark.parametrize("event", ["hello", "state", "error", "episode_summary"])
def test_forwarding_failure_always_finalizes_truthfully(rig, event):
    if event == "error":
        rig.flags["exception"] = RuntimeError("private browser detail")

    def callback(message):
        if message["event"] == event:
            raise RuntimeError("private callback detail")

    component = rig.create(emit=callback)
    while not component.finished:
        component.advance()
    assert not component.report["planet_transport_verified"] and component._stream.closed
    assert events(rig)[-1]["payload"] == {k: v for k, v in component.report.items() if k != "events_sha256"}
    assert "private callback detail" not in (rig.root / "run/events.jsonl").read_text()
    assert "private browser detail" not in (rig.root / "run/events.jsonl").read_text()
    assert component.session is None or component.session.stopped


@pytest.mark.parametrize("phase", ["startup", "exception", "act_hook"])
def test_unknown_exceptions_are_sanitized(rig, phase):
    error = RuntimeError("private path and password")
    if phase == "act_hook":

        def hook():
            raise error

        rig.flags[phase] = hook
    else:
        rig.flags[phase] = error
    component = rig.create()
    component.advance()
    assert component.finished and component.report["outcome"] == "planet_browser_operation_failed"
    assert "private path and password" not in (rig.root / "run/events.jsonl").read_text()
    assert "private path and password" not in (rig.root / "run/report.json").read_text()


def test_termination_and_limits_are_not_success(rig):
    rig.flags.update(end=1, success=False)
    component = rig.create()
    component.advance()
    assert component.report["outcome"] == "policy_stopped"
    rig.flags["end"] = 1000
    limited = rig.create("limited")
    for _ in range(130):
        limited.advance()
    assert limited.steps == 128 and limited.report["outcome"] == "policy_step_limit"
    assert not limited.report["planet_transport_verified"]


def test_keyboard_interrupt_is_durable_and_restores_grad_mode(rig):
    def interrupt():
        raise KeyboardInterrupt

    component = rig.create()
    rig.flags["act_hook"] = interrupt
    with torch.enable_grad(), pytest.raises(KeyboardInterrupt):
        component.advance()
    assert torch.is_grad_enabled() and component.finished and component._stream.closed
    assert component.report["outcome"] == "operator_aborted" and component.session.writes == 0


def test_terminal_callback_interrupt_leaves_durable_failure_and_no_resume(rig):
    rig.flags["end"] = 1

    def emit(message):
        if message["event"] == "episode_summary":
            raise KeyboardInterrupt

    component = rig.create(emit=emit)
    with pytest.raises(KeyboardInterrupt):
        component.advance()
    assert component.finished and component.session.stopped and component._stream.closed
    assert component.report["outcome"] == "operator_aborted"
    assert not component.report["planet_transport_verified"]
    assert json.loads((rig.root / "run/finalization_failed.json").read_text()) == component.report
    component.advance()
    assert component.session.writes == 1


@pytest.mark.parametrize("effect", ["abort", "metadata", "failure"])
def test_callback_after_native_readback_preserves_written_receipt_but_stops(rig, effect):
    holder = {}

    def emit(message):
        if message["event"] == "action_result" and "destination" in message["payload"]:
            if effect == "abort":
                holder["component"].abort()
            elif effect == "metadata":
                rig.metadata.write_bytes(b"changed")
            else:
                raise RuntimeError("private callback failure")

    component = rig.create(emit=emit)
    holder["component"] = component
    component.advance()
    assert component.finished and component.session.writes == 1
    assert len(component.report["verified_fields"]) == 1 and not component.report["planet_transport_verified"]
    component.advance()
    assert component.session.writes == 1


def test_finalization_disk_failure_stops_permanently(rig, monkeypatch):
    component = rig.create()
    rig.flags["end"] = 1

    def fail(*args):
        raise OSError("private path")

    monkeypatch.setattr(module, "write_json", fail)
    component.advance()
    assert component.finished and component.session.stopped and component._stream.closed
    assert component.report["outcome"] == "planet_report_finalization_failed"
    assert not component.report["planet_transport_verified"]
    assert json.loads((rig.root / "run/finalization_failed.json").read_text()) == component.report
    component.advance()
    assert component.session.writes == 1


@pytest.mark.parametrize("change", ["gate", "scope", "size", "encoding", "hidden", "loader"])
def test_invalid_model_gate_stops_before_output_or_browser(rig, monkeypatch, change):
    if change == "gate":
        rig.gate.write_text("{}")
    elif change == "scope":
        content = json.loads(rig.metadata.read_text())
        content["provenance"]["planet_calculations"]["scope"] = "wrong"
        rig.metadata.write_text(json.dumps(content))
    elif change == "size":
        rig.graph.body_ids = [1]
    else:
        original = module.load_checkpoint

        def load(*args, **kwargs):
            if change == "loader":
                raise RuntimeError("private checkpoint detail")
            policy, manifest = original(*args, **kwargs)
            if change == "encoding":
                policy.observation_encoding = "wrong"
            else:
                policy.hidden_size = 20
            return policy, manifest

        monkeypatch.setattr(module, "load_checkpoint", load)
    with pytest.raises(BrowserSafetyStop, match="planet_checkpoint_loading_failed"):
        rig.create()
    assert not (rig.root / "run").exists() and not rig.sessions


def fixture_component(planet_page, tmp_path, monkeypatch, emit=lambda _: None):
    page, frame = planet_page
    pilot, final, *_ = seed_sources(tmp_path)
    for name, text in {"period_days": "108", "line_shift": "0.000000629", "brightness_drop": "0.008"}.items():
        frame.locator("#" + name).fill(text)

    class ReferencePolicy:
        hidden_size, observation_encoding = 16, "structured_planet_tool_v1"

        def act(self, observation, state):
            assert not torch.is_grad_enabled()
            action = planet_expert(observation, PlanetCalculator().pack)
            return action, None, {"fixture": "scripted reference, not learned"}

    monkeypatch.setattr(
        module, "load_checkpoint", lambda *a, **k: (ReferencePolicy(), {"graph_hash": "fixture"})
    )
    monkeypatch.setattr(module, "graph_fingerprint", lambda _: "fixture")
    return module.PlanetDerivedSteps(
        page,
        config(),
        tmp_path / "run",
        pilot=pilot,
        final_evaluation=final,
        graph=SimpleNamespace(body_ids=list(range(2000))),
        supplied_star_class="main_sequence",
        emit=emit,
    )


def test_intercepted_native_fixture_all_four_receipts_and_event_chain(planet_page, tmp_path, monkeypatch):
    component = fixture_component(planet_page, tmp_path, monkeypatch)
    while not component.finished:
        component.advance()
    assert component.report["planet_transport_verified"], component.report
    assert component.steps == 62 and set(component.session.verified) == set(FIELDS)
    journal = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    for name, receipt in component.report["verified_fields"].items():
        assert receipt["unit"] == QUANTITY_UNITS[name] and receipt["action_source"] == "checkpoint"
        assert sum(e["event"] == "action_result" and e["payload"] == receipt for e in journal) == 1
    assert len(list((tmp_path / "run/native-copies").glob("copy-*-confirmed.json"))) == 4
    assert not component.report["task_completed"] and not planet_page[0].get_by_role("checkbox").is_checked()


@pytest.mark.parametrize("effect", ["abort", "failure", "metadata"])
def test_native_fixture_reservation_callback_blocks_actual_fill(planet_page, tmp_path, monkeypatch, effect):
    holder = {}

    def emit(message):
        if message["event"] == "action_proposed" and message["payload"].get("kind") == "TYPE":
            if effect == "abort":
                holder["component"].abort()
            elif effect == "metadata":
                holder["component"].checkpoint.with_suffix(".pt.json").write_bytes(b"changed")
            else:
                raise RuntimeError("private callback value")

    component = fixture_component(planet_page, tmp_path, monkeypatch, emit)
    holder["component"] = component
    while not component.finished:
        component.advance()
    assert len(component.session.attempted) == 1 and not component.session.verified
    assert all(planet_page[1].locator("#" + name).input_value() == "" for name in FIELDS)
    assert len(list((tmp_path / "run/native-copies").glob("copy-*-reserved.json"))) == 1
    assert not list((tmp_path / "run/native-copies").glob("copy-*-confirmed.json"))
    assert not component.report["planet_transport_verified"]


@pytest.mark.skipif(
    not os.environ.get("HABFLY_PLANET_PILOT") or not os.environ.get("HABFLY_PLANET_FINAL"),
    reason="Opt-in frozen real graph/manual fixture only; no final cases opened",
)
def test_opted_real_frozen_checkpoint_intercepted_manual_fixture(planet_page, tmp_path):
    from habfly.data import load_graph

    page, frame = planet_page
    for name, text in {"period_days": "108", "line_shift": "0.000000629", "brightness_drop": "0.008"}.items():
        frame.locator("#" + name).fill(text)
    component = module.PlanetDerivedSteps(
        page,
        config(),
        tmp_path / "learned",
        pilot=Path(os.environ["HABFLY_PLANET_PILOT"]),
        final_evaluation=Path(os.environ["HABFLY_PLANET_FINAL"]),
        graph=load_graph("data/processed/graphs-v2/graph-2000"),
        supplied_star_class="main_sequence",
    )
    try:
        while not component.finished:
            component.advance()
        assert component.report["planet_transport_verified"], component.report
        assert component.steps == 62 and component.report["sources_unchanged"]
        assert not component.report["task_completed"] and component.report["optimizer_updates"] == 0
    finally:
        component.close()
