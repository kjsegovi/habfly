import copy
import io
import json
import socket

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.environments.habitability_calculations import HabitabilityCalculationEnv, habitability_cases
from habfly.habitability_knowledge import HabitabilityCalculator
from habfly.model import ConnectomePolicy
from habfly.model.habitability_tool_state import OPTION_WIDTH, STATE_WIDTH, habitability_state_features
from habfly.runtime import Runtime, read_trace
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.curriculum import Example
from habfly.training.habitability_calculations import collect_split, content_identity, new_policy, run_smoke
from habfly.training.train import supervised_loss


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Offline temperature tests attempted network or Sheets")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(old)


def fixture():
    graph, calculator = make_demo_graph(16), HabitabilityCalculator()
    cases = {s: habitability_cases(s, 1) for s in ("train", "calibration", "development")}
    env = HabitabilityCalculationEnv(calculator, cases["train"])
    env.reset(seed=cases["train"][0]["seed"])
    return graph, calculator, cases, env


def test_all_expert_steps_have_finite_gradients_and_versioned_widths():
    graph, _, _, env = fixture()
    policy = new_policy(graph)
    assert policy.tool_state_projection.in_features == STATE_WIDTH == 73
    assert policy.option_projection.in_features == OPTION_WIDTH == 10
    gradients = set()
    while not env.completed:
        observation = env.observe()
        action = env.expert_action(observation)
        policy.zero_grad(set_to_none=True)
        loss, output = supervised_loss(policy, [Example(observation, action)])
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(output.state).all()
        assert output.state.shape == (1, 16, 16)
        for name, parameter in policy.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
                if parameter.grad.abs().sum() > 0:
                    gradients.add(name)
        env.step(action)
    assert {
        "cell.weight_hh",
        "tool_state_projection.weight",
        "control_projection.weight",
        "option_projection.weight",
        "measurement_identity.encoding.weight",
    } <= gradients


def test_public_features_propagate_through_edges_and_not_grading_answers():
    graph, _, _, env = fixture()
    policy = new_policy(graph)
    before = env.observe().model_dump(mode="json")
    after = copy.deepcopy(before)
    after["calculation"]["operation"] = "equilibrium_temp"
    hidden = copy.deepcopy(before)
    hidden["progress"].update(expected={"surface_temp": 77}, private_stage="copy", steps=999)
    assert habitability_state_features(before) == habitability_state_features(hidden)
    broken = copy.deepcopy(graph)
    broken.edge_weight[:] = 0
    disconnected = new_policy(broken)
    disconnected.load_state_dict(policy.state_dict())
    with torch.no_grad():
        assert not torch.equal(policy([before]).action_logits, policy([after]).action_logits)
        assert torch.equal(disconnected([before]).action_logits, disconnected([after]).action_logits)


def test_checkpoint_scope_and_encoding_mismatch_rejects_silent_reuse(tmp_path):
    graph, calculator, cases, env = fixture()
    identity = content_identity(calculator, cases, graph)
    policy = new_policy(graph)
    save_checkpoint(tmp_path / "new.pt", policy, stage="test", seed=0, content_pack=identity)
    restored, manifest = load_checkpoint(tmp_path / "new.pt", graph, content_pack=identity)
    assert manifest["provenance"]["habitability_calculations"] == identity
    with torch.no_grad():
        torch.testing.assert_close(
            policy([env.observe()]).state, restored([env.observe()]).state, rtol=0, atol=0
        )
    for key, value in (
        ("task", "planet_calculations"),
        ("knowledge_pack_hash", "changed"),
        ("calculation_backend", "google_sheets"),
    ):
        with pytest.raises(ValueError, match="content-pack mismatch"):
            load_checkpoint(tmp_path / "new.pt", graph, content_pack={**identity, key: value})
    with pytest.raises(ValueError, match="Habitability encoding"):
        ConnectomePolicy(graph, observation_encoding="structured_habitability_tool_v1")
    with pytest.raises(ValueError, match="Habitability encoding"):
        ConnectomePolicy(graph, control_encoding="semantic_habitability_tool_v1")
    with pytest.raises(ValueError, match="real 2,000-node"):
        run_smoke(graph, tmp_path / "not_real")
    assert not (tmp_path / "not_real").exists()


def test_offline_recording_replay_and_content_separation(tmp_path):
    graph, calculator, cases, _ = fixture()
    episodes, summaries = collect_split(calculator, cases["train"], tmp_path / "expert")
    assert summaries[0]["completed"] and len(episodes[0]) == 28
    source = next((tmp_path / "expert").glob("*.events.jsonl"))
    events = read_trace(source)
    assert events[0].payload["task"] == "habitability_calculations"
    output = io.StringIO()
    runtime = Runtime(output)
    runtime.command({"command": "replay", "payload": {"path": str(source)}})
    while runtime.status == "running":
        runtime.tick()
    replayed = [json.loads(line) for line in output.getvalue().splitlines()]
    assert any(e["event"] == "episode_summary" and e["payload"]["completed"] for e in replayed)
    assert json.loads((tmp_path / "expert/episodes.json").read_text()) == episodes
    changed = copy.deepcopy(cases)
    changed["development"][0]["case_id"] = changed["train"][0]["case_id"]
    with pytest.raises(ValueError, match="split overlap"):
        content_identity(calculator, changed, graph)
    runtime.close()


def test_result_identity_does_not_depend_on_numeric_payload_or_option_order():
    graph, _, _, env = fixture()
    policy = new_policy(graph)
    while not env.results:
        env.step(env.expert_action(env.observe()))
    observation = env.observe()
    control = next(c for c in observation.controls if c.label == "Calculated result")
    with torch.no_grad():
        pooled = policy([observation]).pooled
        original = policy.option_scores(observation, control, pooled)
        changed = observation.model_copy(deep=True)
        result_id = next(iter(changed.calculation["results"]))
        changed.calculation["results"]["opaque"] = changed.calculation["results"].pop(result_id)
        changed.calculation["results"]["opaque"]["value"] = 987654321
        other = control.model_copy(deep=True)
        other.options = ["opaque"]
        torch.testing.assert_close(original, policy.option_scores(changed, other, pooled), rtol=0, atol=0)
        operation = next(c for c in observation.controls if c.label == "Calculation")
        expected = policy.option_scores(observation, operation, pooled)
        operation.options.reverse()
        torch.testing.assert_close(expected, policy.option_scores(observation, operation, pooled).flip(0))
