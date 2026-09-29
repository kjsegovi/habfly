"""Scripted protocol coverage is separate from the optional frozen learned runtime."""

import io
import json
import os
import socket
from pathlib import Path

import pytest

from habfly.environments.habitability_calculations import HabitabilityCalculationEnv, habitability_cases
from habfly.habitability_knowledge import HabitabilityCalculator
from habfly.runtime import RunOptions, Runtime, read_trace
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.habitability_session import validate_habitability_options
from habfly.training.planet_sequence import file_hash


def payload(output):
    return {**json.loads(Path("configs/habitability_tui.json").read_text()), "artifact_dir": str(output)}


@pytest.mark.parametrize(
    "change",
    [
        {"policy": "expert"},
        {"environment": "browser"},
        {"calculation_backend": "google_sheets"},
        {"spreadsheet_config": "no.json"},
        {"knowledge_pack": "no.json"},
        {"stars": 2},
        {"habitability_evaluation": None},
        {"checkpoint": "wrong.pt"},
        {"seed": 18000000},
        {"seed": 15000000},
        {"planet_evaluation": "wrong-evaluation"},
    ],
)
def test_temperature_runtime_rejects_wrong_scope(tmp_path, change):
    with pytest.raises(ValueError):
        validate_habitability_options(RunOptions.model_validate({**payload(tmp_path), **change}))


def execute_and_replay(data):
    runtime = Runtime(io.StringIO())
    try:
        runtime.command({"command": "start", "payload": data})
        assert runtime.status == "paused"
        revision = runtime.observation.revision
        runtime.advance_if_due()
        assert runtime.observation.revision == revision
        runtime.command({"command": "step"})
        assert runtime.status == "paused"
        runtime.command({"command": "resume"})
        runtime.command({"command": "pause"})
        for _ in range(127):
            runtime.command({"command": "step"})
            if runtime.status in {"completed", "stopped"}:
                break
        assert runtime.status == "completed"
        runtime.command({"command": "abort"})
        assert runtime.status == "completed"
        source = runtime.trace_path
        events = read_trace(source)
        assert len([e for e in events if e.event == "action_proposed"]) == 28
        assert len([e for e in events if e.event == "neural_activity"]) == 28
        assert all(e.version == 1 for e in events)
        assert any(e.event == "episode_summary" and e.payload["completed"] for e in events)
        assert "expected" not in json.dumps([e.payload for e in events if e.event == "observation"])
    finally:
        runtime.close()
    replay = Runtime(io.StringIO())
    try:
        replay.command({"command": "replay", "payload": {"path": str(source)}})
        while replay.status == "running":
            replay.tick()
        assert replay.status == "completed" and read_trace(source) == events
    finally:
        replay.close()
    return events


def test_temperature_pause_step_resume_abort_and_v1_replay(monkeypatch, tmp_path):
    import habfly.training.habitability_session as module

    env = HabitabilityCalculationEnv(HabitabilityCalculator(), habitability_cases("development", 1))

    class ScriptedFixture:
        def __init__(self):
            self.calibration = {"status": "uncalibrated"}

        def act(self, observation, state):
            return env.expert_action(observation), None, {}

        def neural_activity(self, state):
            return {"populations": {"sensory": 0.0}, "top_neurons": []}

    monkeypatch.setattr(
        module,
        "load_habitability_session",
        lambda _: (ScriptedFixture(), env, {"scope": "scripted_protocol_fixture_not_learned"}),
    )
    execute_and_replay(payload(tmp_path))


@pytest.mark.skipif(
    os.environ.get("HABFLY_REAL_HABITABILITY_RUNTIME") != "1",
    reason="Opt-in frozen real-graph temperature runtime",
)
def test_real_temperature_runtime_is_offline_frozen_and_replayable(monkeypatch, tmp_path):
    def deny(*args, **kwargs):
        raise AssertionError("Temperature runtime tried network or spreadsheet access")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    data = payload(tmp_path / "real")
    before = file_hash(data["checkpoint"])
    events = execute_and_replay(data)
    assert file_hash(data["checkpoint"]) == before
    provenance = events[0].payload["provenance"]
    assert provenance["case_split"] == "recorded_development_reuse"
    assert not provenance["habitability_decision_learned"]
    assert not provenance["fresh_unseen_evaluation"] and provenance["optimizer_updates"] == 0
