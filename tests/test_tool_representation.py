"""Tool-state features and learned option selection must not become an expert shortcut."""

from copy import deepcopy

import pytest
import torch

from habfly.contracts import Action, Control, Observation
from habfly.data import make_demo_graph
from habfly.model import ConnectomePolicy
from habfly.training import load_checkpoint, save_checkpoint
from habfly.training.curriculum import Example
from habfly.training.train import supervised_loss


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(previous)


def observation():
    return Observation(
        controls=[
            Control(id="select", label="Measurement or result", actions=["SELECT"], options=["a", "b"])
        ],
        values={
            "required_fields": ["distance"],
            "measurements": {
                "a": {"kind": "flux", "unit": "W/m2", "value": 1.0, "source": "current star"},
                "b": {"kind": "parallax", "unit": "arcsec", "value": 0.1, "source": "reference star"},
            },
        },
        calculation={
            "operation": "distance",
            "parameter": "parallax",
            "source": "a",
            "bindings": {},
            "results": {},
            "pending_result": None,
            "selected_result": "",
            "destination": "",
            "tool_error": None,
        },
    )


def model(graph=None):
    return ConnectomePolicy(
        graph or make_demo_graph(16),
        hidden_size=16,
        observation_encoding="structured_tool_v3",
        selection_mode="option_pointer_v1",
    )


def test_features_are_visible_state_not_time_order_or_expert_correctness():
    policy, obs = model(), observation()
    initial = policy.tool_features(obs)
    altered = obs.model_copy(deep=True)
    altered.revision = 777
    altered.progress = {"steps": 777, "private_stage": "execute", "expected": 99}
    altered.controls.reverse()
    torch.testing.assert_close(initial, policy.tool_features(altered), rtol=0, atol=0)
    # Even a wrong binding is represented faithfully, not repaired or suppressed.
    altered.calculation["bindings"] = {"parallax": "a"}
    bound = policy.tool_features(altered)
    assert initial[3] == 0 and bound[3] == 1
    assert altered.calculation["bindings"] == {"parallax": "a"}
    altered.calculation["bindings"] = {"parallax": "b"}
    torch.testing.assert_close(bound, policy.tool_features(altered))
    assert policy.tool_features(Observation()).count_nonzero() == 0


def test_feature_effect_must_travel_through_connectome():
    graph = make_demo_graph(16)
    connected = model(graph)
    broken_graph = deepcopy(graph)
    broken_graph.edge_weight[:] = 0
    disconnected = model(broken_graph)
    disconnected.load_state_dict(connected.state_dict())
    before, after = observation(), observation()
    after.calculation["bindings"] = {"parallax": "a"}
    with torch.no_grad():
        assert not torch.equal(connected([before]).action_logits, connected([after]).action_logits)
        assert torch.equal(disconnected([before]).action_logits, disconnected([after]).action_logits)


def test_pointer_is_permutation_and_opaque_id_equivariant():
    policy, obs = model(), observation()
    with torch.no_grad():
        pooled = policy([obs]).pooled
        original = policy.option_scores(obs, obs.controls[0], pooled)
        reordered = obs.model_copy(deep=True)
        reordered.controls[0].options.reverse()
        torch.testing.assert_close(
            original, policy.option_scores(reordered, reordered.controls[0], pooled).flip(0)
        )
        renamed = obs.model_copy(deep=True)
        renamed.values["measurements"] = {
            "new_a": obs.values["measurements"]["a"],
            "new_b": obs.values["measurements"]["b"],
        }
        renamed.controls[0].options = ["new_a", "new_b"]
        torch.testing.assert_close(original, policy.option_scores(renamed, renamed.controls[0], pooled))


def test_pointer_supervision_finite_and_uses_actual_expert_option():
    policy, obs = model(), observation()
    example = Example(obs, Action(kind="SELECT", target="select", value="a"))
    loss, _ = supervised_loss(policy, [example])
    loss.backward()
    assert torch.isfinite(loss)
    assert policy.embedding.weight.grad.abs().sum() > 0
    assert policy.value_chars.weight.grad is None
    assert policy.answer_chars.weight.grad is None
    assert torch.isfinite(policy.cell.weight_hh.grad).all()
    with pytest.raises(ValueError, match="visible option"):
        supervised_loss(policy, [Example(obs, Action(kind="SELECT", target="select", value="hidden"))])


def test_modes_round_trip_without_adding_parameters_and_legacy_manifest_loads(tmp_path):
    policy = model()
    legacy = ConnectomePolicy(policy.graph, hidden_size=16)
    assert sum(p.numel() for p in policy.parameters()) == sum(p.numel() for p in legacy.parameters())
    path = tmp_path / "new.pt"
    save_checkpoint(path, policy, stage="test", seed=0)
    restored, _ = load_checkpoint(path, policy.graph)
    assert restored.observation_encoding == "structured_tool_v3"
    assert restored.selection_mode == "option_pointer_v1"
    payload = torch.load(path, weights_only=True)
    payload["manifest"]["model"].pop("observation_encoding")
    payload["manifest"]["model"].pop("selection_mode")
    torch.save(payload, tmp_path / "legacy.pt")
    old, _ = load_checkpoint(tmp_path / "legacy.pt", policy.graph)
    assert old.observation_encoding == "pooled_text_v2" and old.selection_mode == "characters"


def test_semantic_selection_does_not_rank_measurement_digits():
    policy, obs = model(), observation()
    policy.selection_mode = "option_pointer_semantic_v2"
    with torch.no_grad():
        pooled = policy([obs]).pooled
        expected = policy.option_scores(obs, obs.controls[0], pooled)
        changed = obs.model_copy(deep=True)
        changed.values["measurements"]["a"]["value"] = 9.87654321e20
        changed.values["measurements"]["b"]["value"] = 0
        actual = policy.option_scores(changed, changed.controls[0], pooled)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        assert obs.values["measurements"]["b"]["value"] == 0.1
        changed.values["measurements"]["b"]["kind"] = "wavelength"
        assert not torch.equal(expected, policy.option_scores(changed, changed.controls[0], pooled))


def test_semantic_value_only_training_preserves_workflow_and_graph(tmp_path):
    from habfly.training.train import train_behavioral_cloning

    policy, obs = model(), observation()
    policy.selection_mode = "option_pointer_semantic_v2"
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith("value_decoder."))
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    action = Action(kind="SELECT", target="select", value="b")
    episode = [{"observation": obs.model_dump(mode="json"), "action": action.model_dump(mode="json")}]
    validation = deepcopy(episode)
    validation[0]["observation"]["instruction"] = "Separate validation"
    report = train_behavioral_cloning(
        [episode],
        policy.graph,
        tmp_path / "value-only",
        policy=policy,
        validation_episodes=[validation],
        epochs=1,
        sequence_length=10,
    )
    assert report["optimizer_steps"] == 1
    for name, parameter in policy.named_parameters():
        if not name.startswith("value_decoder."):
            assert torch.equal(parameter, before[name])
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in policy.named_parameters()
        if name.startswith("value_decoder.")
    )
    restored, _ = load_checkpoint(tmp_path / "value-only/checkpoint.pt", policy.graph)
    assert restored.selection_mode == "option_pointer_semantic_v2"


@pytest.mark.parametrize("settings", [{"observation_encoding": "secret_expert"}, {"selection_mode": "bad"}])
def test_invalid_representation_rejected(settings):
    with pytest.raises(ValueError, match="Unknown"):
        ConnectomePolicy(make_demo_graph(8), **settings)
