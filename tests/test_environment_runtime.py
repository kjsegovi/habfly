import io
import json

import pytest
from pydantic import ValidationError

from habfly.content import calculate
from habfly.contracts import Action, Observation, RuntimeCommand
from habfly.environments import MiniHabWorlds, expert_action, generate_demonstrations
from habfly.runtime import Runtime, read_trace, serve


@pytest.mark.parametrize("stars", [1, 3, 30])
def test_expert_completes_acceptance_ladder(stars):
    episodes = generate_demonstrations([13], stars=stars)
    result = episodes[0][-1]["result"]
    assert result["terminated"] and not result["truncated"]
    assert result["observation"]["progress"]["analyzed"] == stars
    assert result["observation"]["progress"]["score"] == 1


def test_reset_deterministic_and_values_not_revealed_before_chart_interaction():
    first, second = MiniHabWorlds(), MiniHabWorlds()
    assert first.reset(seed=42) == second.reset(seed=42)
    for _ in range(3):
        observation = first.observe()
        action = expert_action(observation)
        result = first.step(action)
    observation = Observation.model_validate(result[0])
    assert "period" not in observation.values
    assert "brightness_drop" not in observation.values
    assert "tooltip" not in observation.chart
    first.reset(seed=42)
    assert first.collection == {}


def test_stale_and_invalid_targets_rejected_without_mutation():
    env = MiniHabWorlds()
    env.reset(seed=1)
    stale = expert_action(env.observe())
    env.step(Action(kind="WAIT"))
    _, reward, _, _, info = env.step(stale)
    assert info["result"]["failure_reason"] == "stale_observation"
    assert env.screen == "explore" and reward < 0
    _, _, _, _, info = env.step(Action(kind="CLICK", target="fake"))
    assert info["result"]["failure_reason"] == "unavailable_target"


def test_completion_is_independent_of_perfect_score():
    env = MiniHabWorlds()
    env.reset(seed=1)
    while not env.terminated:
        action = expert_action(env.observe())
        if action.kind == "TYPE" and action.value != "300":
            action.value = "0"
        env.step(action)
    assert env.observe().progress["score"] < 1
    assert env.observe().progress["submitted"]


def test_repeated_check_does_not_farm_rewards():
    env = MiniHabWorlds()
    env.reset(seed=0)
    while "calculations correct" not in env.feedback:
        env.step(expert_action(env.observe()))
    check = next(c for c in env.observe().controls if c.label == "Check calculations")
    result = env.step(Action(kind="CLICK", target=check.id))
    assert result[1] < 0
    assert "analyzed_star" not in result[4]["result"]["reward_components"]


def test_truncation_and_finished_step_guard():
    env = MiniHabWorlds(max_steps=1)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(Action(kind="WAIT"))
    assert not terminated and truncated
    assert info["result"]["failure_reason"] == "step_limit"
    with pytest.raises(RuntimeError):
        env.step(Action(kind="WAIT"))


def test_formula_grammar_and_protocol_validation():
    assert calculate("2 * mass ** 2", {"mass": 3}) == 18
    for expression in ["__import__('os')", "mass.__class__", "1e309", "2 ** 200", "unknown"]:
        with pytest.raises(ValueError):
            calculate(expression, {"mass": 2})
    with pytest.raises(ValidationError):
        Action(kind="TYPE", target="answer")
    with pytest.raises(ValidationError):
        RuntimeCommand(version=2, command="start")


def test_runtime_pause_step_abort_and_trace(tmp_path):
    output = io.StringIO()
    runtime = Runtime(output)
    runtime.command({"command": "start", "payload": {"paused": True, "artifact_dir": str(tmp_path)}})
    assert runtime.status == "paused"
    runtime.command({"command": "step"})
    assert runtime.env.steps == 1 and runtime.status == "paused"
    runtime.command({"command": "resume"})
    runtime.tick()
    runtime.command({"command": "pause"})
    saved = tmp_path / "copy" / "trace.jsonl"
    runtime.command({"command": "save_trace", "payload": {"path": str(saved)}})
    runtime.command({"command": "abort"})
    events = read_trace(saved)
    assert any(e.event == "action_proposed" for e in events)
    neural = next(e for e in events if e.event == "neural_activity")
    assert neural.payload["activity_source"] == "untrained_observer"
    assert neural.payload["top_neurons"]
    action = next(e for e in events if e.event == "action_proposed")
    assert action.payload["action_source"] == "scripted_expert"
    assert action.payload["action_confidence"] is None
    assert runtime.env is None
    runtime.command({"command": "replay", "payload": {"path": str(saved)}})
    while runtime.status == "running":
        runtime.tick()
    assert runtime.status == "completed"
    assert runtime.trace_path == saved
    assert json.loads(output.getvalue().splitlines()[-1])["payload"]["replay"]
    runtime.close()


def test_runtime_jsonl_recovers_malformed_messages():
    output = io.StringIO()
    serve(io.StringIO('not-json\n{"command":"pause"}\n'), output)
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert events[0]["event"] == "hello"
    assert sum(e["event"] == "error" for e in events) == 2


def test_quitting_finished_runtime_preserves_original_completion(tmp_path):
    runtime = Runtime(io.StringIO())
    runtime.command({"command": "start", "payload": {"artifact_dir": str(tmp_path)}})
    while runtime.status == "running":
        runtime.tick()
    assert runtime.status == "completed"
    path = runtime.trace_path
    runtime.command({"command": "abort"})
    runtime.command({"command": "abort"})
    assert runtime.status == "completed"
    summaries = [e.payload for e in read_trace(path) if e.event == "episode_summary"]
    assert len(summaries) == 1 and summaries[0]["completed"]


def test_trace_rejects_malformed_and_nonmonotonic_events(tmp_path):
    trace = tmp_path / "bad.jsonl"
    event = {"event": "state", "sequence": 1, "payload": {}}
    trace.write_text(json.dumps(event) + "\n" + json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="Non-monotonic"):
        read_trace(trace)
