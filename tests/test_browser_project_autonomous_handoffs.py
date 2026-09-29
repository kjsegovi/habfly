"""Injected post-stellar autonomous scheduling, not browser/science acceptance."""
# ruff: noqa: F401,F811 -- imported pytest fixtures

import json
from copy import deepcopy

import pytest
from test_browser_project_runtime import ready, sha, write
from test_browser_project_runtime import rig as bridge_rig
from test_browser_project_steps import rig
from test_browser_project_steps_positive import positive_rig
from test_browser_project_steps_terrestrial import advance_until, initialized, terrestrial_rig

from habfly import autonomous_planet
from habfly.browser import BrowserSafetyStop

PHASES = ("awaiting_planet_class", "awaiting_gases", "awaiting_habitability")
PAYLOADS = {
    "awaiting_planet_class": {"name": "terrestrial", "rationale": "Fixture-only reference class"},
    "awaiting_gases": {"gases": ["CO2"], "rationale": "Fixture-only rendered spectrum"},
    "awaiting_habitability": {"choice": "not_habitable", "rationale": "Fixture-only phase reference"},
}


@pytest.fixture
def runtime_rig(bridge_rig, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No network, browser launch, training or live evidence")

    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    bridge_rig.options.project_autonomous_decisions = True
    return bridge_rig


def automatic_handoff(rig, monkeypatch, phase):
    class Child(rig.steps):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.decisions, self.native_steps, self.resumes = [], [], 0

        def step(self):
            if self.phase == "ready":
                self.phase, self.status = phase, "paused"
                self.emit("state", self.state())
                return
            self.emit("action_proposed", {"kind": "CLICK", "fixture": self.phase})
            if not self.finished:
                self.native_steps.append(self.phase)

        def choose(self, kind, payload):
            self.decisions.append((kind, deepcopy(payload)))
            self.phase = (
                "selecting_planet_class" if phase == "awaiting_planet_class" else "terrestrial_active"
            )
            self.emit("state", self.state())

        def provide_planet_class(self, **payload):
            self.choose("planet", payload)

        def provide_autonomous_gases(self, **payload):
            assert "supplied_greenhouse_increment" not in payload
            self.choose("gases", payload)

        def provide_habitability(self, **payload):
            self.choose("habitability", payload)

        def resume(self):
            self.resumes += 1
            super().resume()

    bridge = ready(rig, steps=Child, reference_planet_continuation=True, terrestrial_options={})
    bridge.step()
    assert bridge.phase == phase and bridge.status == "paused"
    result = {
        "decision_kind": phase.removeprefix("awaiting_"),
        "status": "reference_decision",
        "payload": deepcopy(PAYLOADS[phase]),
        "source_sha256": {
            "initial-star/stellar/observation.json": sha(
                bridge.output / "initial-star/stellar/observation.json"
            )
        },
        "learned": False,
        "scientific_verified": False,
        "training_label": False,
    }
    references = []

    def decide(**kwargs):
        references.append(deepcopy(kwargs))
        assert kwargs["phase"] == phase and kwargs["expected_star"] == "ALPHA"
        assert kwargs["owner_state"]["star"] == "ALPHA" and kwargs["run_history"] == bridge.output
        return deepcopy(result)

    monkeypatch.setattr(autonomous_planet, "decide_planet_handoff", decide)
    return bridge, result, references


@pytest.mark.parametrize("phase", PHASES)
def test_automatic_prepare_apply_resume_and_native_step_are_separate(runtime_rig, monkeypatch, phase):
    bridge, result, references = automatic_handoff(runtime_rig, monkeypatch, phase)
    bridge.resume()
    assert bridge.status == "running" and bridge.project.status == "paused"
    bridge.tick()
    assert bridge.phase == phase and bridge.autonomous_decision["status"] == "prepared"
    assert not bridge.project.decisions and not bridge.project.native_steps
    assert len(references) == 1
    bridge.tick()
    assert bridge.project.decisions == [
        (phase.removeprefix("awaiting_").replace("planet_class", "planet"), result["payload"])
    ]
    assert len(references) == 2 and not bridge.project.native_steps
    assert bridge.status == bridge.project.status == "running" and bridge.project.resumes == 1
    assert bridge._autonomous_pending is None and bridge.autonomous_decision["status"] == "applied"
    assert not bridge.state()["task_completed"] and not bridge.state()["project_completed"]
    bridge.tick()
    assert len(bridge.project.native_steps) == 1
    saved = json.loads((bridge.output / bridge.autonomous_decision["artifact"]).read_bytes())
    assert saved["browser_actions"] == 0 and saved["decision"] == result
    assert bridge.autonomous_decision["learned"] is False
    bridge.close()


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize(
    "mutation", ["source", "record", "cached_star", "context_star", "reference", "authority"]
)
def test_changed_prepared_context_or_reference_stops_without_choice(
    runtime_rig, monkeypatch, phase, mutation
):
    bridge, result, _ = automatic_handoff(runtime_rig, monkeypatch, phase)
    bridge.step()
    if mutation == "source":
        (bridge.output / "initial-star/stellar/observation.json").write_text("{}")
    elif mutation == "record":
        (bridge.output / bridge._autonomous_pending["path"]).write_text("{}")
    elif mutation == "cached_star":
        bridge._autonomous_pending["record"]["current_star"]["star"] = "FOREIGN"
    elif mutation == "context_star":
        bridge._current_context["star"] = "FOREIGN"
    elif mutation == "reference":
        result["payload"]["rationale"] = "Changed reference"
    else:
        bridge.autonomous_decisions = False
    bridge.step()
    assert bridge.status == "stopped" and not bridge.project.decisions and not bridge.project.native_steps
    assert not bridge.report["task_completed"] and not bridge.report["project_completed"]
    before = list(runtime_rig.calls)
    bridge.step()
    bridge.tick()
    assert runtime_rig.calls == before
    bridge.close()


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("effect", ["pause", "abort", "throw"])
@pytest.mark.parametrize("boundary", ["prepare", "child_state", "child_action"])
def test_callback_pause_abort_and_failure_cannot_run_later_stages(
    runtime_rig, monkeypatch, phase, effect, boundary
):
    bridge, _, _ = automatic_handoff(runtime_rig, monkeypatch, phase)
    original, fired = bridge._callback, []

    def callback(kind, payload):
        original(kind, payload)
        selected = (
            boundary == "prepare"
            and (payload.get("autonomous_decision") or {}).get("status") == "prepared"
            or boundary == "child_state"
            and kind == "state"
            and payload.get("component") == "project.owner"
            and (payload.get("component_state") or {}).get("phase")
            in {"selecting_planet_class", "terrestrial_active"}
            or boundary == "child_action"
            and kind == "action_proposed"
        )
        if not selected or fired:
            return
        fired.append(True)
        if effect == "pause":
            bridge.pause()
        elif effect == "abort":
            bridge.abort()
        else:
            raise RuntimeError("private callback detail")

    bridge._callback = callback
    bridge.resume()
    for _ in range({"prepare": 1, "child_state": 2, "child_action": 3}[boundary]):
        bridge.tick()
    assert fired
    assert bridge.status == {"pause": "paused", "abort": "aborted", "throw": "stopped"}[effect]
    assert not bridge.project.decisions if boundary == "prepare" else len(bridge.project.decisions) == 1
    # Pause is a stage boundary, not an in-call interrupt: only the already
    # entered injected action may finish. Abort/throw prevent even that action.
    assert len(bridge.project.native_steps) == int(boundary == "child_action" and effect == "pause")
    count = len(bridge.project.native_steps)
    bridge.tick()
    bridge.advance_if_due()
    assert len(bridge.project.native_steps) == count
    assert "private callback detail" not in bridge.trace_path.read_text()
    assert not bridge.state()["task_completed"]
    bridge.close()


def test_actual_owner_forwards_autonomous_gases_without_invented_increment(terrestrial_rig, monkeypatch):
    owner = initialized(terrestrial_rig)
    advance_until(owner, "awaiting_gases")
    child = terrestrial_rig.children[0]
    calls = []

    def accept(*, gases, rationale):
        calls.append({"gases": gases, "rationale": rationale})
        child.gas_decision = calls[-1]
        write(child.output / "gas-decision.json", child.gas_decision)
        child.phase = "select_gases"
        child.send("state", child.state())

    monkeypatch.setattr(child, "provide_autonomous_gases", accept, raising=False)
    before = child.advances
    owner.provide_autonomous_gases(gases=["CO2"], rationale="Rendered fixture spectrum")
    assert calls == [{"gases": ["CO2"], "rationale": "Rendered fixture spectrum"}]
    assert owner.phase == "terrestrial_active" and owner.status == "paused"
    assert child.advances == before and child.saves == 0
    assert owner._terrestrial_decisions["gas"]["hashes"]
    assert not owner.state()["task_completed"]
    owner.abort()
