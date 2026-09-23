import json
from collections import Counter

import numpy as np
import pytest
import torch

from habfly.contracts import Action, ActionKind, Control, Observation, validate_action
from habfly.data.graphs import make_demo_graph
from habfly.model import CharacterTokenizer, ConnectomePolicy, TopologyFreePolicy
from habfly.model.policy import observation_text
from habfly.training import curriculum_examples, load_checkpoint, run_curriculum, save_checkpoint
from habfly.training.baselines import matched_random_graph
from habfly.training.calibration import fit_temperature
from habfly.training.curriculum import arithmetic_ood_examples
from habfly.training.environment import CurriculumEnvironment
from habfly.training.ppo import train_ppo
from habfly.training.train import train_behavioral_cloning


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(text="Click the star."):
    return Observation(instruction=text, controls=[Control(id="a", label="Star"), Control(id="b", label="Planet")])


def test_character_roundtrip_and_unknown():
    tokenizer = CharacterTokenizer()
    for text in ["x = 3.14", "CLICK Star!", "μ ± 2 °"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.UNK in tokenizer.encode("🪰")
    assert tokenizer.encode("x" * 100, max_length=8)[-1] == tokenizer.EOS
    assert CharacterTokenizer(**tokenizer.config()).fingerprint == tokenizer.fingerprint


def test_fixed_topology_has_finite_deterministic_responsive_outputs():
    torch.manual_seed(2)
    model = ConnectomePolicy(make_demo_graph(8), hidden_size=8, max_answer_length=6).eval()
    obs = observation()
    with torch.no_grad():
        expected = model([obs])
        for _ in range(100):
            actual = model([obs])
            assert torch.isfinite(actual.state).all()
            assert torch.equal(actual.action_logits, expected.action_logits)
        changed = model([observation("Type the planet's distance in meters.")])
        assert not torch.equal(expected.action_logits, changed.action_logits)
    model([obs]).action_logits.sum().backward()
    assert model.sensory_projection.weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad.abs().sum() > 0
    assert "adjacency" not in model.state_dict()
    diagnostics = model.neural_activity(expected.state)
    assert diagnostics["top_neurons"][0]["body_id"] in model.graph.body_ids
    assert diagnostics["populations"]["efferent"] > 0


@pytest.mark.parametrize("size", [2000, 5000, 10000, 30000])
def test_requested_graph_sizes_forward_and_gradients(size):
    model = ConnectomePolicy(make_demo_graph(size), hidden_size=4, max_answer_length=2)
    output = model([observation()])
    assert output.state.shape == (1, size, 4)
    assert torch.isfinite(output.state).all()
    output.action_logits.sum().backward()
    assert torch.isfinite(model.cell.weight_hh.grad).all()


def test_action_legality_revision_and_calibration_status():
    model = ConnectomePolicy(make_demo_graph(8), hidden_size=8)
    obs = Observation(revision=7, controls=[Control(id="no", label="Hidden", enabled=False),
        Control(id="yes", label="Answer", role="combobox", options=["yes", "no"], actions=[ActionKind.SELECT])])
    with torch.no_grad():
        model.action_head.weight.zero_()
        model.action_head.bias.fill_(-100)
        model.action_head.bias[2] = 100
    action, _, diagnostics = model.act(obs)
    validate_action(obs, action)
    assert action.kind == ActionKind.SELECT and action.target == "yes"
    assert action.value in ["yes", "no"]
    assert action.observation_revision == 7
    assert action.calibrated is False and diagnostics["calibration"]["status"] == "uncalibrated"


def test_checkpoints_fail_closed_and_transfer_recalibrates(tmp_path):
    graph = make_demo_graph(8)
    model = ConnectomePolicy(graph, hidden_size=8)
    model.calibration = {"status": "calibrated"}
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, stage="unit_test", seed=7, content_pack={"id": "synthetic"})
    restored, manifest = load_checkpoint(path, graph, content_pack={"id": "synthetic"})
    assert manifest["seed"] == 7
    assert manifest["provenance"]["dependency_lock_hash"]
    assert manifest["provenance"]["code_hashes"]["src/habfly/model/policy.py"]
    assert isinstance(manifest["provenance"]["git_dirty"], bool)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    with pytest.raises(ValueError, match="graph mismatch"):
        load_checkpoint(path, make_demo_graph(9))
    with pytest.raises(ValueError, match="content-pack mismatch"):
        load_checkpoint(path, graph, content_pack={"id": "wrong"})
    transferred, _ = load_checkpoint(path, make_demo_graph(9), allow_graph_transfer=True)
    assert transferred.calibration["status"] == "uncalibrated"
    assert transferred([observation()]).state.shape[1] == 9
    payload = torch.load(path, weights_only=True)
    payload["manifest"]["provenance"]["code_hashes"]["src/habfly/model/policy.py"] = "changed"
    torch.save(payload, tmp_path / "changed.pt")
    uncalibrated, _ = load_checkpoint(tmp_path / "changed.pt", graph)
    assert uncalibrated.calibration["reason"] == "model_source_changed_or_unverified"


def test_arithmetic_operand_and_grounding_composition_splits():
    train, test = curriculum_examples("arithmetic", count=200)
    assert not {e.group for e in train} & {e.group for e in test}
    assert all(max(map(int, e.group.split(":")[1:])) <= 9 for e in train + test)
    assert all(e.group.startswith("ood:") for e in arithmetic_ood_examples())
    train, test = curriculum_examples("ground", count=96)
    assert not {e.group for e in train} & {e.group for e in test}
    assert {e.action.kind for e in train} == {ActionKind.CLICK, ActionKind.TYPE, ActionKind.SELECT}
    for example in train + test:
        validate_action(example.observation, example.action)


def test_matched_baselines_preserve_degrees_signs_and_parameters():
    graph = make_demo_graph(20)
    randomized = matched_random_graph(graph, seed=42)
    sign = np.sign(graph.edge_weight)
    for endpoint in ("edge_src", "edge_dst"):
        assert Counter(zip(getattr(graph, endpoint), sign)) == Counter(zip(getattr(randomized, endpoint), sign))
    model = ConnectomePolicy(graph, hidden_size=8)
    baseline = TopologyFreePolicy(graph, hidden_size=8)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in baseline.parameters())
    assert not torch.equal(baseline([observation("A")]).action_logits, baseline([observation("B")]).action_logits)


def test_temperature_scaling_improves_overconfidence():
    logits = torch.tensor([[12., 0.], [12., 0.], [12., 0.], [12., 0.]])
    labels = torch.tensor([0, 1, 0, 1])
    fit = fit_temperature(logits, labels)
    assert fit["temperature"] > 1
    assert fit["nll"] < torch.nn.functional.cross_entropy(logits, labels)


def test_vision_encoder_is_local_trainable_and_inference_gated():
    model = ConnectomePolicy(make_demo_graph(8), hidden_size=8)
    pixels = torch.rand(1, 1, 64, 64)
    model([observation()], chart_pixels=pixels).action_logits.sum().backward()
    assert model.chart_encoder.layers[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="chart_vision_untrained"):
        model.act(observation(), chart_pixels=pixels)


def test_observation_encoding_excludes_artifact_paths_and_revision():
    obs = observation()
    obs.chart_crop = "/private/chart.png"
    obs.revision = 121237
    assert "/private/chart.png" not in observation_text(obs)
    assert "121237" not in observation_text(obs)
    assert observation_text(obs).startswith('{"instruction"')


def test_policy_is_invariant_to_observation_local_target_ids():
    policy = ConnectomePolicy(make_demo_graph(8), hidden_size=8).eval()
    original = observation()
    renamed = original.model_copy(deep=True)
    renamed.revision = 471
    for index, control in enumerate(renamed.controls):
        control.id = f"471:field-{index}"
    assert observation_text(original) == observation_text(renamed)
    with torch.no_grad():
        first, second = policy([original]), policy([renamed])
    assert torch.equal(first.action_logits, second.action_logits)
    assert torch.equal(first.target_logits, second.target_logits)


def test_curriculum_training_creates_measured_checkpoint(tmp_path):
    report = run_curriculum("recognize", tmp_path, make_demo_graph(8), epochs=1, count=4, hidden_size=8)
    assert report["heldout"]["examples"] == 2
    assert isinstance(report["gate_passed"], bool)
    assert len(report["losses"]) == 1
    assert json.loads((tmp_path / "report.json").read_text())["seed"] == 0
    load_checkpoint(tmp_path / "checkpoint.pt", make_demo_graph(8))
    with pytest.raises(FileExistsError, match="not empty"):
        run_curriculum("recognize", tmp_path, make_demo_graph(8), epochs=1, count=4, hidden_size=8)


def test_composed_curriculum_sequence_uses_common_contract():
    env = CurriculumEnvironment("ground", sequence_length=3)
    first, _ = env.reset(seed=12)
    repeated, _ = env.reset(seed=12)
    assert first == repeated and "Then" in first["instruction"]
    for index in range(3):
        action = env.examples[index].action.model_copy(update={"observation_revision": index})
        obs, reward, done, truncated, info = env.step(action)
        assert reward == 1 and not truncated
        assert info["result"]["observation"] == obs
    assert done


def test_behavioral_cloning_recurrent_smoke_and_overlap_rejection(tmp_path):
    from habfly.environments import generate_demonstrations

    episodes = generate_demonstrations([1])
    validation = generate_demonstrations([10, 11])
    report = train_behavioral_cloning(episodes, make_demo_graph(8), tmp_path / "run",
                                      validation_episodes=validation, epochs=1, hidden_size=8)
    assert report["training_episodes"] == 1 and report["gate_passed"] is False
    assert report["calibration"]["scope"] == "heldout_recurrent_expert_trajectories"
    with pytest.raises(ValueError, match="overlap"):
        train_behavioral_cloning(episodes, make_demo_graph(8), tmp_path / "invalid", validation_episodes=episodes)


class OneStepEnvironment:
    """Known one-click fixture tests PPO mechanics, not HabWorlds competence."""

    def reset(self, seed=None):
        self.observation = Observation(instruction="Finish", controls=[Control(id="finish", label="Finish")])
        return self.observation.model_dump(mode="json"), {}

    def step(self, action):
        completed = action.kind == "CLICK" and action.target == "finish"
        final = Observation(progress={"submitted": completed})
        return final.model_dump(mode="json"), float(completed), True, False, {"result": {}}

    def close(self):
        pass


def test_ppo_enforces_gate_and_freezes_deterministic_argument_support(tmp_path):
    policy = ConnectomePolicy(make_demo_graph(8), hidden_size=8, max_answer_length=4)
    expert = lambda obs: Action(kind=ActionKind.CLICK, target="finish")
    with torch.no_grad():
        policy.action_head.weight.zero_()
        policy.action_head.bias.fill_(-100)
        policy.action_head.bias[8] = 100
    with pytest.raises(ValueError, match="behavioral-cloning gate failed"):
        train_ppo(policy, OneStepEnvironment, tmp_path / "blocked", expert=expert, updates=1, gate_seeds=[1])
    assert not (tmp_path / "blocked").exists()
    with torch.no_grad():
        policy.action_head.bias[8] = -100
        policy.action_head.bias[0] = 100
    original = {key: tensor.clone() for key, tensor in policy.state_dict().items()}
    report = train_ppo(policy, OneStepEnvironment, tmp_path / "ppo", expert=expert, updates=1,
                       episodes_per_update=1, optimization_epochs=1, gate_seeds=[1])
    assert report["gate"]["completion_rate"] == 1
    assert report["heldout"]["completion_rate"] == 1
    for key, tensor in policy.state_dict().items():
        if not key.startswith(("action_head.", "target_query.", "value_head.")):
            assert torch.equal(tensor, original[key]), key
    assert not torch.equal(policy.value_head.weight, original["value_head.weight"])
    assert all(p.requires_grad for p in policy.parameters())
