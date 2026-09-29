"""Protocol fixtures are scripted; the opt-in real-checkpoint test is separate."""

import io
import json
import os
import socket
from pathlib import Path

import pytest

from habfly.environments.planet_calculations import PlanetCalculationEnv, planet_cases, planet_expert
from habfly.planet_knowledge import PlanetCalculator
from habfly.runtime import RunOptions, Runtime, read_trace
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.planet_sequence import file_hash
from habfly.training.planet_session import validate_planet_options


def payload(tmp_path):
    data = json.loads(Path("configs/planet_tui.json").read_text())
    return {**data, "artifact_dir": str(tmp_path)}


@pytest.mark.parametrize(
    "change",
    [
        {"policy": "expert"},
        {"environment": "browser"},
        {"calculation_backend": "google_sheets"},
        {"spreadsheet_config": "unrelated.json"},
        {"knowledge_pack": "wrong.json"},
        {"stars": 2},
        {"planet_evaluation": None},
        {"checkpoint": "unrelated.pt"},
        {"seed": 13000000},
        {"seed": 10000000},
    ],
)
def test_planet_runtime_rejects_unsupported_scope_before_start(tmp_path, change):
    options = RunOptions.model_validate({**payload(tmp_path), **change})
    with pytest.raises(ValueError):
        validate_planet_options(options)


def test_planet_protocol_pause_step_abort_and_replay_use_existing_v1(monkeypatch, tmp_path):
    import habfly.training.planet_session as module

    env = PlanetCalculationEnv(PlanetCalculator(), planet_cases("development", 1))

    class ScriptedFixture:
        def __init__(self):
            self.calibration = {"status": "uncalibrated"}

        def act(self, observation, state):
            return planet_expert(observation, env.pack), None, {}

        def neural_activity(self, state):
            return {"populations": {"sensory": 0.0}, "top_neurons": []}

    monkeypatch.setattr(
        module,
        "load_planet_session",
        lambda _: (
            ScriptedFixture(),
            env,
            {"scope": "test_fixture_not_learned"},
        ),
    )
    runtime = Runtime(io.StringIO())
    runtime.command({"command": "start", "payload": payload(tmp_path)})
    assert runtime.status == "paused"
    revision = runtime.observation.revision
    runtime.advance_if_due()
    assert runtime.observation.revision == revision
    runtime.command({"command": "step"})
    assert runtime.status == "paused" and runtime.observation.revision != revision
    runtime.command({"command": "resume"})
    runtime.command({"command": "pause"})
    for _ in range(61):
        runtime.command({"command": "step"})
    assert runtime.status == "completed"
    runtime.command({"command": "abort"})
    assert runtime.status == "completed"
    original = runtime.trace_path
    runtime.close()
    events = read_trace(original)
    assert len([e for e in events if e.event == "action_proposed"]) == 62
    assert all(e.version == 1 for e in events)
    summaries = [e for e in events if e.event == "episode_summary"]
    assert len(summaries) == 1 and summaries[0].payload["completed"]
    replay = Runtime(io.StringIO())
    replay.command({"command": "replay", "payload": {"path": str(original)}})
    while replay.status == "running":
        replay.tick()
    assert replay.status == "completed" and read_trace(original) == events
    replay.close()


@pytest.mark.skipif(
    os.environ.get("HABFLY_REAL_PLANET_RUNTIME") != "1", reason="Opt-in frozen real-graph runtime"
)
def test_real_planet_checkpoint_offline_runtime_reload_and_replay(monkeypatch, tmp_path):
    def deny(*args, **kwargs):
        raise AssertionError("Planet runtime tried network or spreadsheet access")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    destination = Path(os.environ.get("HABFLY_PLANET_RUNTIME_OUTPUT", str(tmp_path / "real")))
    assert not destination.exists()
    data = payload(destination)
    before = file_hash(data["checkpoint"])
    runtime = Runtime(io.StringIO())
    try:
        runtime.command({"command": "start", "payload": data})
        assert runtime.status == "paused"
        for _ in range(128):
            runtime.command({"command": "step"})
            if runtime.status in {"completed", "stopped"}:
                break
        assert runtime.status == "completed"
        trace = runtime.trace_path
        events = read_trace(trace)
        actions = [e for e in events if e.event == "action_proposed"]
        assert len(actions) == 62
        assert len([e for e in events if e.event == "neural_activity"]) == 62
        assert events[0].payload["provenance"]["case_split"] == "recorded_development_reuse"
        assert "expected" not in json.dumps([e.payload for e in events if e.event == "observation"])
    finally:
        runtime.close()
    assert file_hash(data["checkpoint"]) == before
    replay = Runtime(io.StringIO())
    replay.command({"command": "replay", "payload": {"path": str(trace)}})
    while replay.status == "running":
        replay.tick()
    assert replay.status == "completed" and read_trace(trace) == events
    replay.close()
    (destination / "report.json").write_text(
        json.dumps(
            {
                "scope": "offline_recorded_development_runtime",
                "seed": data["seed"],
                "steps": len(actions),
                "task_completed": True,
                "browser_acceptance_passed": False,
                "checkpoint_sha256": before,
                "checkpoint_unchanged": True,
                "optimizer_updates": 0,
                "trace": str(trace),
                "offline_replay_verified": True,
                "network_and_sheets_denied": True,
            },
            indent=2,
        )
        + "\n"
    )
