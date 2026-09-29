"""Cooperative fixture behavior, not learned or live-course acceptance."""
# ruff: noqa: F811

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_browser_habitability_numeric import habitat_page  # noqa: F401
from test_browser_habitability_policy import mapping
from test_browser_numeric import chromium, config, page  # noqa: F401

import habfly.browser_habitability_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_habitability_policy import HabitabilityBrowserToolEnv
from habfly.contracts import Action, Observation, StepResult
from habfly.environments.habitability_calculations import SCOPE, habitability_expert
from habfly.habitability_knowledge import HabitabilityCalculator

NATIVE_SESSION = module._BoundaryCheckedNumericSession


def seed_sources(root):
    pilot, final = root / "pilot", root / "final"
    (pilot / "training").mkdir(parents=True)
    final.mkdir()
    checkpoint = pilot / "training/checkpoint.pt"
    checkpoint.write_bytes(b"fixture only, not trained")
    content = {
        "graph_hash": "fixture-graph",
        "knowledge_pack_hash": HabitabilityCalculator().pack.checksum,
        "splits": {split: {} for split in module.PILOT_COUNTS},
    }
    metadata = checkpoint.with_suffix(".pt.json")
    metadata.write_text(json.dumps({"provenance": {"habitability_calculations": content}}))
    (pilot / "report.json").write_text("{}")
    for split in module.PILOT_COUNTS:
        (pilot / f"private-{split}-cases.json").write_text("[]")
        if split != "gate":
            (pilot / f"expert-{split}").mkdir()
            (pilot / f"expert-{split}/episodes.json").write_text("[]")
    for split in ("train", "development"):
        (pilot / f"learned-{split}").mkdir()
        (pilot / f"learned-{split}/summaries.json").write_text("[]")
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
    return pilot, final, checkpoint, metadata, gate, content


@pytest.fixture
def rig(tmp_path, monkeypatch):
    pilot, final, checkpoint, metadata, gate, content = seed_sources(tmp_path)
    flags = {
        "end": 2,
        "success": True,
        "failure": None,
        "truncated": False,
        "exception": None,
        "startup": None,
        "act_hook": None,
        "copy_hook": None,
        "graph_hash": "fixture-graph",
        "receipt_verified": True,
    }
    policies, sessions = [], []
    graph = SimpleNamespace(body_ids=list(range(2000)))

    class Policy:
        hidden_size, observation_encoding = 16, module.ENCODING

        def __init__(self):
            self.inputs = []
            policies.append(self)

        def eval(self):
            return self

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
        def __init__(self, page, config, output, *, max_seconds, guard, before_write):
            assert max_seconds == 300
            self.stopped = self.attempted = False
            self.writes = 0
            self.mapping = mapping()
            self.guard, self.before_write = guard, before_write
            sessions.append(self)
            if flags["startup"]:
                raise flags["startup"]

        def copy(self, text, unit, *, source):
            assert not self.stopped and not self.attempted
            assert unit == "K" and source == "checkpoint" and text == "300.123456789"
            self.guard()
            self.attempted = True
            self.before_write()
            self.guard()
            assert not self.stopped
            self.writes += 1
            if flags["copy_hook"]:
                flags["copy_hook"]()
            return {
                "destination": "equilibrium_temp",
                "value": text,
                "unit": unit,
                "readback_verified": flags["receipt_verified"],
                "correctness_verified": False,
            }

        def close(self):
            self.stopped = True

    class Environment:
        def __init__(self, mapping, *, supplied_greenhouse_increment, seed):
            self.steps, self.proposals_complete, self.answers = 0, False, {}

        def observe(self):
            return Observation(instruction="fixture", progress={"task_completed": False})

        def step(self, action):
            assert action.action_confidence is action.target_confidence is None and not action.calibrated
            if flags["exception"]:
                raise flags["exception"]
            self.steps += 1
            end = self.steps >= flags["end"]
            if end and flags["success"]:
                self.proposals_complete = True
                self.answers = {"equilibrium_temp": 300.123456789, "surface_temp": 330.123456789}
            return StepResult(
                observation=self.observe(),
                steps=self.steps,
                terminated=end,
                truncated=flags["truncated"],
                failure_reason=flags["failure"],
            )

    monkeypatch.setattr(module, "load_validated_pilot", lambda *a: (Policy(), content, {}))
    monkeypatch.setattr(module, "graph_fingerprint", lambda _: flags["graph_hash"])
    monkeypatch.setattr(module, "_BoundaryCheckedNumericSession", Session)
    monkeypatch.setattr(module, "HabitabilityBrowserToolEnv", Environment)

    def create(name="run", emit=lambda _: None, **kwargs):
        return module.HabitabilityTemperatureSteps(
            kwargs.pop("page", None),
            kwargs.pop("config", None),
            tmp_path / name,
            pilot=pilot,
            final_evaluation=final,
            graph=graph,
            supplied_greenhouse_increment=kwargs.pop("supplied_greenhouse_increment", 30),
            emit=emit,
            **kwargs,
        )

    return SimpleNamespace(
        create=create,
        root=tmp_path,
        flags=flags,
        policies=policies,
        sessions=sessions,
        pilot=pilot,
        final=final,
        checkpoint=checkpoint,
        metadata=metadata,
        gate=gate,
        content=content,
        graph=graph,
    )


def events(rig, name="run"):
    return [json.loads(line) for line in (rig.root / name / "events.jsonl").read_text().splitlines()]


def ready(component):
    while not component.finished and component.state()["phase"] != "ready_to_copy":
        component.advance()


def test_one_decision_per_advance_then_separate_copy_and_exact_flushed_events(rig):
    received = []

    def callback(event):
        assert events(rig)[-1] == event
        received.append(event)

    component = rig.create(emit=callback)
    assert not component.finished and component.steps == 0 and not rig.policies[0].inputs
    with torch.enable_grad():
        component.advance()
        assert torch.is_grad_enabled() and component.steps == 1 and component.session.writes == 0
        saved = (rig.root / "run/events.jsonl").read_bytes()
        assert component.state() == component.state()
        assert (rig.root / "run/events.jsonl").read_bytes() == saved
        component.advance()
        assert component.steps == 2 and component.state()["phase"] == "ready_to_copy"
        assert component.session.writes == 0 and component._hidden is None
        component.advance()
    assert component.finished and component.session.writes == 1 and len(rig.policies[0].inputs) == 2
    report = component.report
    assert (
        report["equilibrium_transport_verified"]
        and report["checkpoint_unchanged"]
        and report["sources_unchanged"]
    )
    assert report["local_proposals"]["surface_temp"] == 330.123456789
    assert report["native_receipt"]["destination"] == "equilibrium_temp"
    assert all(
        report[key] is False
        for key in ("task_completed", "course_acceptance_passed", "saved", "assessed", "submitted")
    )
    assert report["optimizer_updates"] == 0
    assert received == events(rig)
    assert [event["sequence"] for event in received] == list(range(len(received)))
    assert all(event["version"] == 1 for event in received)
    assert report["events_sha256"] == module.file_hash(rig.root / "run/events.jsonl")
    native = [e for e in received if e["payload"].get("native_write_boundary")]
    assert len(native) == 1 and native[0]["payload"]["value"] == "300.123456789"
    assert [e["event"] for e in received].count("neural_activity") == 2
    component.close()
    component.advance()
    assert events(rig) == received and component._hidden is None


@pytest.mark.parametrize("at", [0, 1, 2])
def test_abort_and_close_are_sticky_with_no_hidden_state_or_later_copy(rig, at):
    component = rig.create()
    for _ in range(at):
        component.advance()
    component.abort()
    assert component.finished and component._hidden is None and component.session.stopped
    assert component.report["outcome"] == "operator_aborted"
    before = events(rig)
    component.advance()
    component.close()
    assert events(rig) == before and not component.session.writes
    other = rig.create("other")
    other.advance()
    assert rig.policies[-1].inputs == [None]
    other.close()


def test_abort_inside_inference_cannot_restore_hidden_state(rig):
    component = rig.create()
    rig.flags["act_hook"] = component.abort
    component.advance()
    assert component.finished and component._hidden is None and component.steps == 0
    assert component.session.writes == 0


@pytest.mark.parametrize(
    "source",
    [
        "checkpoint",
        "metadata",
        "gate",
        "pilot_report",
        "pilot_cases",
        "pilot_episodes",
        "pilot_summaries",
        "graph",
        "deleted_metadata",
    ],
)
@pytest.mark.parametrize("when", ["between", "act", "native"])
def test_changed_frozen_sources_prevent_next_copy(rig, source, when):
    def mutate():
        if source == "graph":
            rig.flags["graph_hash"] = "changed"
        elif source == "deleted_metadata":
            rig.metadata.unlink()
        else:
            path = {
                "checkpoint": rig.checkpoint,
                "metadata": rig.metadata,
                "gate": rig.gate,
                "pilot_report": rig.pilot / "report.json",
                "pilot_cases": rig.pilot / "private-train-cases.json",
                "pilot_episodes": rig.pilot / "expert-calibration/episodes.json",
                "pilot_summaries": rig.pilot / "learned-development/summaries.json",
            }[source]
            path.write_text("changed")

    component = rig.create(
        emit=lambda event: (
            mutate() if when == "native" and event["payload"].get("native_write_boundary") else None
        )
    )
    if when == "act":
        rig.flags["act_hook"] = mutate
    else:
        ready(component)
        if when == "between":
            mutate()
    component.advance()
    assert component.finished and not component.report["equilibrium_transport_verified"]
    assert not component.session.writes and not component.report["sources_unchanged"]
    assert component._hidden is None


@pytest.mark.parametrize("event", ["observation", "action_proposed", "neural_activity", "native", "ready"])
@pytest.mark.parametrize("effect", ["abort", "failure", "reentry"])
def test_callback_cancel_failure_or_reentry_blocks_copy(rig, event, effect):
    component = None

    def callback(item):
        selected = item["event"] == event
        selected |= event == "native" and item["payload"].get("native_write_boundary", False)
        selected |= event == "ready" and item["payload"].get("phase") == "ready_to_copy"
        if component and selected:
            if effect == "abort":
                component.abort()
            elif effect == "reentry":
                component.advance()
            else:
                raise ValueError("SECRET callback contents")

    component = rig.create(emit=callback)
    ready(component)
    component.advance()
    assert component.finished and not component.report["equilibrium_transport_verified"]
    assert component.session.writes == 0 and component._hidden is None
    assert "SECRET" not in (rig.root / "run/events.jsonl").read_text()


@pytest.mark.parametrize("event", ["hello", "state", "error", "episode_summary"])
def test_forwarding_failure_has_durable_truthful_report(rig, event):
    def callback(item):
        if item["event"] == event:
            raise RuntimeError("SECRET forwarding failure")

    if event == "error":
        rig.flags["exception"] = RuntimeError("SECRET private URL")
    component = rig.create(emit=callback)
    ready(component)
    component.advance()
    assert component.finished and not component.report["equilibrium_transport_verified"]
    assert "SECRET" not in (rig.root / "run/events.jsonl").read_text()
    assert component._hidden is None


@pytest.mark.parametrize("fault", ["incomplete", "truncated", "failure", "limit", "receipt"])
def test_termination_limits_and_unverified_readback_are_not_success(rig, fault):
    rig.flags.update(
        {
            "success": fault != "incomplete",
            "truncated": fault == "truncated",
            "failure": "fixture_rejected" if fault == "failure" else None,
            "end": 129 if fault == "limit" else 2,
            "receipt_verified": fault != "receipt",
        }
    )
    component = rig.create()
    ready(component)
    component.advance()
    assert component.finished and not component.report["equilibrium_transport_verified"]
    assert not component.report["task_completed"]
    assert component.steps == (128 if fault == "limit" else 1 if fault == "truncated" else 2)
    assert component.session.writes == (1 if fault == "receipt" else 0)


@pytest.mark.parametrize("when", ["act", "native", "summary"])
def test_keyboard_interrupt_persists_abort_and_restores_grad_mode(rig, when):
    def interrupt():
        raise KeyboardInterrupt

    def callback(event):
        if (
            when == "native"
            and event["payload"].get("native_write_boundary")
            or when == "summary"
            and event["event"] == "episode_summary"
        ):
            interrupt()

    component = rig.create(emit=callback)
    if when == "act":
        rig.flags["act_hook"] = interrupt
    else:
        ready(component)
    with torch.enable_grad(), pytest.raises(KeyboardInterrupt):
        component.advance()
    assert torch.is_grad_enabled() and component.finished and component._hidden is None
    assert not component.report["equilibrium_transport_verified"]
    assert component.session.writes == (1 if when == "summary" else 0)


@pytest.mark.parametrize("effect", ["abort", "metadata", "failure"])
def test_post_copy_cancellation_retains_receipt_but_never_reports_success(rig, effect):
    component = None

    def callback(event):
        if event["event"] == "action_result" and "native_copy" in event["payload"]:
            if effect == "abort":
                component.abort()
            elif effect == "metadata":
                rig.metadata.write_text("changed")
            else:
                raise RuntimeError("SECRET")

    component = rig.create(emit=callback)
    ready(component)
    component.advance()
    assert component.finished and component.session.writes == 1
    assert not component.report["equilibrium_transport_verified"]
    assert component.report["native_receipt"]["readback_verified"]


@pytest.mark.parametrize("bad", [True, None, "30", 1, -1, float("inf")])
def test_invalid_supplied_increment_never_starts(rig, bad):
    with pytest.raises(BrowserSafetyStop, match="supplied_greenhouse"):
        rig.create(supplied_greenhouse_increment=bad)
    assert not rig.sessions and not (rig.root / "run").exists()


@pytest.mark.parametrize("fault", ["gate", "split", "loader", "hidden", "encoding", "content"])
def test_bad_loading_gate_is_redacted_before_browser_or_output(rig, monkeypatch, fault):
    if fault == "gate":
        rig.gate.write_text("{}")
    elif fault == "split":
        value = json.loads(rig.metadata.read_text())
        value["provenance"]["habitability_calculations"]["splits"]["test"] = {}
        rig.metadata.write_text(json.dumps(value))
    else:
        loader = module.load_validated_pilot

        def changed(*args):
            if fault == "loader":
                raise ValueError("SECRET private data")
            policy, content, cases = loader(*args)
            if fault == "hidden":
                policy.hidden_size = 32
            elif fault == "encoding":
                policy.observation_encoding = "wrong"
            else:
                content = {**content, "extra": "changed"}
            return policy, content, cases

        monkeypatch.setattr(module, "load_validated_pilot", changed)
    with pytest.raises(BrowserSafetyStop, match="^temperature_checkpoint_loading_failed$"):
        rig.create()
    assert not rig.sessions and not (rig.root / "run").exists()


def test_finalization_disk_failure_is_not_success(rig, monkeypatch):
    component = rig.create()
    ready(component)
    monkeypatch.setattr(module, "write_json", lambda *a: (_ for _ in ()).throw(OSError("SECRET disk")))
    component.advance()
    assert component.finished and component.report["artifact_finalization_failed"]
    assert not component.report["equilibrium_transport_verified"]
    assert "SECRET" not in (rig.root / "run/finalization_failed.json").read_text()


def test_changed_pack_and_terminal_source_mutation_fail_closed(rig, monkeypatch):
    component = rig.create()
    ready(component)
    monkeypatch.setattr(
        module, "HabitabilityCalculator", lambda: SimpleNamespace(pack=SimpleNamespace(checksum="changed"))
    )
    component.advance()
    assert component.finished and not component.report["sources_unchanged"] and not component.session.writes


def test_terminal_source_mutation_corrects_summary_and_keeps_native_receipt(rig):
    def callback(event):
        if event["event"] == "episode_summary":
            rig.metadata.write_text("changed")

    component = rig.create(emit=callback)
    ready(component)
    component.advance()
    assert component.finished and component.session.writes == 1
    assert not component.report["equilibrium_transport_verified"]
    assert component.report["native_receipt"]["readback_verified"]
    assert events(rig)[-1]["payload"]["outcome"] == "temperature_source_changed"


def test_callback_receives_detached_serialized_events(rig):
    def callback(event):
        event["payload"]["callback_only"] = True

    component = rig.create(emit=callback)
    ready(component)
    component.advance()
    assert component.report["equilibrium_transport_verified"]
    assert all("callback_only" not in event["payload"] for event in events(rig))


@pytest.mark.skipif(
    os.environ.get("HABFLY_REAL_GRAPH_TEST") != "1", reason="explicit frozen-model fixture check"
)
def test_real_frozen_checkpoint_public_fixture_offline_without_final_case_reads(tmp_path, monkeypatch):
    import socket

    from habfly.data import load_graph
    from habfly.spreadsheet import SpreadsheetAdapter

    def deny(*args, **kwargs):
        raise AssertionError("offline component contacted an external service")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    final = Path("experiments/habitability-final-001").resolve()
    original_open = Path.open
    opened_final = []

    def bounded_open(path, *args, **kwargs):
        resolved = path.resolve()
        if final in resolved.parents:
            assert resolved == final / "report.json", "final cases must not be read"
            opened_final.append(resolved)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", bounded_open)

    class OfflineCopy:
        def __init__(self, page, config, output, *, guard, before_write, max_seconds):
            self.mapping, self.attempted, self.stopped = mapping(), False, False
            self.guard, self.before_write = guard, before_write
            self.writes = 0

        def copy(self, text, unit, *, source):
            self.guard()
            assert not self.attempted and not self.stopped and unit == "K" and source == "checkpoint"
            self.attempted = True
            self.before_write()
            self.guard()
            self.writes += 1
            return {
                "destination": "equilibrium_temp",
                "value": text,
                "unit": unit,
                "readback_verified": True,
                "fixture_only": True,
            }

        def close(self):
            self.stopped = True

    monkeypatch.setattr(module, "_BoundaryCheckedNumericSession", OfflineCopy)
    component = module.HabitabilityTemperatureSteps(
        None,
        None,
        tmp_path / "run",
        pilot="experiments/habitability-pilot-001",
        final_evaluation=final,
        graph=load_graph("data/processed/graphs-v2/graph-2000"),
        supplied_greenhouse_increment=30,
    )
    before = {key: value.clone() for key, value in component.policy.state_dict().items()}
    ready(component)
    assert component.steps == 28 and not component.session.writes and not component.finished
    component.advance()
    assert component.report["equilibrium_transport_verified"] and component.session.writes == 1
    assert component.report["sources_unchanged"] and component._hidden is None
    assert all(torch.equal(value, component.policy.state_dict()[key]) for key, value in before.items())
    assert opened_final and set(opened_final) == {final / "report.json"}
    assert component.report["task_completed"] is False and component.report["optimizer_updates"] == 0


def native_component(rig, monkeypatch, habitat_page, emit=lambda _: None):
    page, _ = habitat_page
    monkeypatch.setattr(module, "_BoundaryCheckedNumericSession", NATIVE_SESSION)
    monkeypatch.setattr(module, "HabitabilityBrowserToolEnv", HabitabilityBrowserToolEnv)
    component = rig.create(page=page, config=config(), emit=emit)

    def act(observation, state):
        assert not torch.is_grad_enabled()
        return habitability_expert(observation, HabitabilityCalculator().pack), None, {"fixture_only": True}

    component.policy.act = act
    return component


def test_intercepted_native_fixture_has_one_equilibrium_receipt_no_other_writes(
    rig, monkeypatch, habitat_page
):
    page, frame = habitat_page
    component = native_component(rig, monkeypatch, habitat_page)
    ready(component)
    assert component.steps == 28 and frame.locator("#temperature").input_value() == "0"
    assert not (rig.root / "run/native-copy/reserved.json").exists()
    component.advance()
    assert component.report["equilibrium_transport_verified"] and component.steps == 28
    assert frame.locator("#surface").inner_text() == frame.locator("#temperature").input_value()
    assert all(
        frame.locator(selector).input_value() == "" for selector in ("#gases", "#greenhouse", "#phase")
    )
    assert not page.get_by_role("checkbox").is_checked()
    assert (
        json.loads((rig.root / "run/native-copy/confirmed.json").read_text())
        == component.report["native_receipt"]
    )


@pytest.mark.parametrize("effect", ["abort", "metadata", "failure", "page"])
def test_intercepted_native_reservation_callback_blocks_actual_fill(rig, monkeypatch, habitat_page, effect):
    _, frame = habitat_page
    component = None

    def callback(event):
        if event["payload"].get("native_write_boundary"):
            assert (rig.root / "run/native-copy/reserved.json").exists()
            if effect == "abort":
                component.abort()
            elif effect == "metadata":
                rig.metadata.write_text("changed")
            elif effect == "page":
                frame.locator("#phase").select_option(label="Gas")
            else:
                raise ValueError("SECRET")

    component = native_component(rig, monkeypatch, habitat_page, callback)
    ready(component)
    component.advance()
    assert component.finished and not component.report["equilibrium_transport_verified"]
    assert frame.locator("#temperature").input_value() == "0"
    assert not (rig.root / "run/native-copy/confirmed.json").exists()
    assert (rig.root / "run/native-copy/stopped.json").exists()
    assert component._hidden is None
