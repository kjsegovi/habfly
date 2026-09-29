"""New planet tensors/provenance cannot silently reuse stellar checkpoints."""

import copy
import io
import json
import socket

import pytest
import torch

from habfly.contracts import Observation, RuntimeEvent
from habfly.data import make_demo_graph
from habfly.environments.planet_calculations import PlanetCalculationEnv, planet_cases
from habfly.model.planet_tool_state import OPTION_WIDTH, STATE_WIDTH, planet_state_features
from habfly.planet_knowledge import PlanetCalculator
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.curriculum import Example
from habfly.training.planet_calculations import (
    collect_split,
    content_identity,
    new_policy,
    run_smoke,
    summarize,
    validate_separation,
)
from habfly.training.train import supervised_loss


@pytest.fixture(autouse=True)
def local_cpu(monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("Unexpected network or Sheets call")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(old_threads)


def fixture():
    graph, calculator = make_demo_graph(16), PlanetCalculator()
    cases = {s: planet_cases(s, 1) for s in ("train", "calibration", "development")}
    env = PlanetCalculationEnv(calculator, cases["train"])
    env.reset(seed=cases["train"][0]["seed"])
    return graph, calculator, cases, env


def test_shapes_gradients_and_every_reference_section_within_budget():
    graph, _, _, env = fixture()
    policy = new_policy(graph)
    assert policy.tool_state_projection.in_features == STATE_WIDTH == 128
    assert policy.option_projection.in_features == OPTION_WIDTH == 22
    gradients = set()
    while not env.completed:
        obs, action = env.observe(), env.expert_action(env.observe())
        policy.zero_grad(set_to_none=True)
        loss, output = supervised_loss(policy, [Example(obs, action)])
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
        "option_projection.weight",
        "control_projection.weight",
        "measurement_identity.encoding.weight",
    } <= gradients


def test_state_features_are_public_and_must_propagate_through_edges():
    graph, _, _, env = fixture()
    policy = new_policy(graph)
    before = env.observe().model_dump(mode="json")
    after = copy.deepcopy(before)
    after["calculation"]["operation"] = "period_years"
    features = planet_state_features(before)
    assert len(features) == STATE_WIDTH
    hidden = copy.deepcopy(before)
    hidden["progress"].update(expected={"planet_mass": 55}, private_stage="copy", steps=999)
    assert planet_state_features(hidden) == features
    broken = copy.deepcopy(graph)
    broken.edge_weight[:] = 0
    disconnected = new_policy(broken)
    disconnected.load_state_dict(policy.state_dict())
    with torch.no_grad():
        assert not torch.equal(policy([before]).action_logits, policy([after]).action_logits)
        assert torch.equal(disconnected([before]).action_logits, disconnected([after]).action_logits)


def test_option_pointer_equivariance_and_numeric_payload_separation():
    graph, _, _, env = fixture()
    policy = new_policy(graph)
    # Include generated intermediate result options as well as measurements.
    while not env.results:
        env.step(env.expert_action(env.observe()))
    obs = env.observe()
    control = next(c for c in obs.controls if c.label == "Calculated result")
    with torch.no_grad():
        pooled = policy([obs]).pooled
        scores = policy.option_scores(obs, control, pooled)
        changed = obs.model_copy(deep=True)
        old_result = next(iter(changed.calculation["results"]))
        changed.calculation["results"]["opaque"] = changed.calculation["results"].pop(old_result)
        changed.calculation["results"]["opaque"]["value"] = 987654321
        other = control.model_copy(deep=True)
        other.options = ["opaque"]
        torch.testing.assert_close(scores, policy.option_scores(changed, other, pooled), rtol=0, atol=0)
        operation = next(c for c in obs.controls if c.label == "Calculation")
        expected = policy.option_scores(obs, operation, pooled)
        operation.options.reverse()
        torch.testing.assert_close(expected, policy.option_scores(obs, operation, pooled).flip(0))


def test_checkpoint_content_and_encoding_contracts(tmp_path):
    from habfly.model import ConnectomePolicy

    graph, calculator, cases, env = fixture()
    identity = content_identity(calculator, cases, graph)
    policy = new_policy(graph)
    save_checkpoint(tmp_path / "new.pt", policy, stage="test", seed=0, content_pack=identity)
    restored, manifest = load_checkpoint(tmp_path / "new.pt", graph, content_pack=identity)
    assert manifest["provenance"]["planet_calculations"] == identity
    assert restored.configuration() == policy.configuration()
    with torch.no_grad():
        torch.testing.assert_close(
            policy([env.observe()]).state, restored([env.observe()]).state, rtol=0, atol=0
        )
    for key, value in (
        ("task", "stellar"),
        ("calculation_backend", "google_sheets"),
        ("knowledge_pack_hash", "changed"),
    ):
        with pytest.raises(ValueError, match="content-pack mismatch"):
            load_checkpoint(tmp_path / "new.pt", graph, content_pack={**identity, key: value})
    old = ConnectomePolicy(graph, hidden_size=16)
    save_checkpoint(tmp_path / "old.pt", old, stage="test", seed=0)
    old_restored, _ = load_checkpoint(tmp_path / "old.pt", graph)
    assert old_restored.configuration() == old.configuration()
    with pytest.raises(ValueError, match="Planet encoding"):
        ConnectomePolicy(graph, observation_encoding="structured_planet_tool_v1")
    with pytest.raises(ValueError, match="Planet encoding"):
        ConnectomePolicy(graph, control_encoding="semantic_planet_tool_v1")
    with pytest.raises(ValueError, match="real 2,000-node"):
        run_smoke(graph, tmp_path / "not_real")


def test_split_separation_recording_metrics_and_offline_replay(tmp_path):
    graph, calculator, cases, _ = fixture()
    identity = content_identity(calculator, cases, graph)
    assert identity["final_test_status"] == "not_generated_or_evaluated"
    overlapping = copy.deepcopy(cases)
    overlapping["development"] = overlapping["train"]
    with pytest.raises(ValueError, match="split overlap"):
        validate_separation(overlapping)
    episodes, rows = collect_split(calculator, cases["train"], tmp_path / "expert")
    report = summarize(rows)
    assert report["completed"] == 1
    for key in (
        "input_binding_accuracy",
        "output_copy_accuracy",
        "unit_selection_accuracy",
        "numeric_answer_accuracy",
        "calculation_selection_accuracy",
    ):
        assert report[key] == 1
    assert all("expected" not in json.dumps(s["observation"]) for s in episodes[0])
    path = next((tmp_path / "expert").glob("*.events.jsonl"))
    events = [RuntimeEvent.model_validate_json(line) for line in path.read_text().splitlines()]
    assert all(event.version == 1 for event in events)
    assert events[0].payload["task"] == "planet_calculations"
    assert events[1].payload["stage"] == "planet_calculations"
    assert events[-1].payload["completed"]
    assert [e.sequence for e in events] == list(range(len(events)))
    recorded = [e.payload for e in events if e.event == "observation"]
    assert recorded[:-1] == [s["observation"] for s in episodes[0]]
    assert not Observation.model_validate(recorded[-1]).progress["course_acceptance_passed"]
    from habfly.runtime import Runtime

    stream = io.StringIO()
    runtime = Runtime(stream)
    runtime.command({"command": "replay", "payload": {"path": str(path)}})
    while runtime.status == "running":
        runtime.tick()
    replayed = [RuntimeEvent.model_validate_json(line) for line in stream.getvalue().splitlines()]
    for kind in ("observation", "action_proposed", "action_result", "episode_summary"):
        original_payloads = [e.payload for e in events if e.event == kind]
        replayed_payloads = [
            {k: v for k, v in e.payload.items() if k != "replay"} for e in replayed if e.event == kind
        ]
        assert original_payloads == replayed_payloads
    runtime.close()
    failed = summarize([{"completed": False, "steps": 128, "tool_errors": 2}])
    assert failed["input_binding_accuracy"] is None and failed["numeric_answer_accuracy"] == 0


def test_planet_recording_names_confidence_scope_and_real_activity_source(tmp_path):
    from habfly.training.stellar import record_episode

    graph, calculator, cases, _ = fixture()
    env = PlanetCalculationEnv(calculator, cases["train"], max_steps=2)
    policy = new_policy(graph)
    policy.calibration = {"status": "uncalibrated", "scope": "test_fixture_not_a_calibration_claim"}
    path = tmp_path / "model.events.jsonl"
    record_episode(env, cases["train"][0]["seed"], policy, event_path=path)
    events = [RuntimeEvent.model_validate_json(line) for line in path.read_text().splitlines()]
    assert all(
        e.payload["calibration_scope"] == "test_fixture_not_a_calibration_claim"
        for e in events
        if e.event == "action_proposed"
    )
    activity = [e.payload for e in events if e.event == "neural_activity"]
    assert activity and all(e["activity_source"] == "checkpoint" and e["top_neurons"] for e in activity)
