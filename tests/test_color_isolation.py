"""Selected value isolation without replacing biological computation or source choice."""

from copy import deepcopy

import pytest
import torch

from habfly.color_reference import COLOR_CONTROL, ColorReference, load_color_reference
from habfly.data import make_demo_graph
from habfly.environments.color import color_cases
from habfly.model import ConnectomePolicy
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.color import record, refinement_features
from habfly.training.color_isolation import irrelevant_context_variant


@pytest.fixture
def sample():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = ConnectomePolicy(
        make_demo_graph(),
        hidden_size=16,
        observation_encoding="structured_color_v1",
        selection_mode="measurement_result_v3",
        control_encoding="semantic_color_v1",
        color_readout="ordinal_v2",
        color_input="selected_graph_v1",
    )
    reference = load_color_reference()
    episodes = [record(reference, c)[0] for c in color_cases("train", 4, reference)]
    policy.embedding.requires_grad_(False)
    policy.text_encoder.requires_grad_(False)
    policy.color_head.fit_normalization(refinement_features(policy, episodes)[0])
    return policy, episodes


def test_irrelevant_context_order_and_previous_activity_cannot_change_color(sample, monkeypatch):
    policy, episodes = sample
    obs = episodes[0][1].observation.model_dump(mode="json")
    control = next(c for c in obs["controls"] if c["label"] == COLOR_CONTROL)
    variant = irrelevant_context_variant(obs)
    # Selected identity is local and arbitrary, not a hidden numeric signal.
    source = variant["calculation"]["source"]
    variant["values"]["measurements"]["renamed"] = variant["values"]["measurements"].pop(source)
    variant["calculation"]["source"] = "renamed"
    monkeypatch.setattr(ColorReference, "private_label", lambda *a: pytest.fail("Inference used oracle"))
    monkeypatch.setattr(ColorReference, "guard", lambda *a: pytest.fail("Inference used band thresholds"))
    with torch.no_grad():
        initial = policy([obs])
        changed = policy([variant], torch.randn_like(initial.state))
        assert not torch.equal(initial.pooled, changed.pooled)
        first = policy.option_scores(obs, control, initial.pooled)
        second = policy.option_scores(variant, control, changed.pooled)
        assert torch.equal(first, second)
        assert torch.equal(policy.selected_color_activity(obs)[0], policy.selected_color_activity(variant)[0])
        reversed_control = {**control, "options": list(reversed(control["options"]))}
        assert torch.equal(first.flip(0), policy.option_scores(variant, reversed_control, changed.pooled))


def test_selected_value_controls_graph_and_wrong_source_is_not_repaired(sample, monkeypatch):
    policy, episodes = sample
    obs = episodes[0][1].observation.model_dump(mode="json")
    wrong = deepcopy(obs)
    wrong["calculation"]["source"] = next(
        k for k, m in obs["values"]["measurements"].items() if m["source"] == "reference star"
    )
    copied = deepcopy(obs)
    selected = copied["calculation"]["source"]
    copied["values"]["measurements"][selected]["value"] = wrong["values"]["measurements"][
        wrong["calculation"]["source"]
    ]["value"]
    state, pooled = policy.selected_color_activity(obs)
    wrong_state, wrong_pooled = policy.selected_color_activity(wrong)
    assert not torch.equal(state, wrong_state) and not torch.equal(pooled, wrong_pooled)
    assert torch.equal(wrong_pooled, policy.selected_color_activity(copied)[1])
    # Prove the readout consumes propagated effector features, not raw numbers.
    calls = []

    def zero_graph(encoded, recurrent):
        calls.append(recurrent)
        return state * 0, pooled * 0

    monkeypatch.setattr(policy, "propagate", zero_graph)
    assert torch.equal(policy.color_readout_features(obs, pooled), torch.zeros_like(pooled))
    assert calls == [None]


@pytest.mark.parametrize("value", [None, 0, -1, True, "500", float("nan"), float("inf")])
def test_invalid_selected_values_fail_without_fallback(sample, value):
    policy, episodes = sample
    obs = episodes[0][1].observation.model_dump(mode="json")
    obs["values"]["measurements"][obs["calculation"]["source"]]["value"] = value
    with pytest.raises(ValueError, match="selected_measurement_invalid"):
        policy.selected_color_activity(obs)


def test_missing_selection_and_wrong_unit_fail(sample):
    policy, episodes = sample
    obs = episodes[0][0].observation.model_dump(mode="json")
    with pytest.raises(ValueError, match="selected_measurement_invalid"):
        policy.selected_color_activity(obs)
    obs = episodes[0][1].observation.model_dump(mode="json")
    obs["values"]["measurements"][obs["calculation"]["source"]]["unit"] = "K"
    with pytest.raises(ValueError, match="selected_measurement_invalid"):
        policy.selected_color_activity(obs)


def test_workflow_is_unchanged_gradients_finite_and_checkpoint_versioned(sample, tmp_path):
    policy, episodes = sample
    obs = episodes[0][1].observation
    config = policy.configuration()
    config.pop("architecture")
    legacy = ConnectomePolicy(policy.graph, **{**config, "color_input": "workflow_v1"})
    legacy.load_state_dict(policy.state_dict(), strict=True)
    initial = torch.randn(1, len(policy.graph.body_ids), 16)
    before, after = legacy([obs], initial), policy([obs], initial)
    for field in ("state", "pooled", "action_logits", "target_logits", "typed_value_logits"):
        assert torch.equal(getattr(before, field), getattr(after, field))
    _, pooled = policy.selected_color_activity(obs)
    loss = policy.color_head.loss(pooled, torch.tensor([0]))[0]
    loss.backward()
    for layer in (policy.color_projection, policy.sensory_projection, policy.cell, policy.color_head):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, policy, stage="color", seed=0)
    loaded, _ = load_checkpoint(checkpoint, policy.graph)
    assert loaded.color_input == "selected_graph_v1"
    assert torch.equal(loaded.selected_color_activity(obs)[1], pooled)
    assert "color_input" not in legacy.configuration()
    with pytest.raises(ValueError, match="Incompatible color input"):
        ConnectomePolicy(policy.graph, color_input="selected_graph_v1")


def test_action_telemetry_uses_isolated_activity_but_returns_workflow_state(sample, monkeypatch):
    from habfly.model.policy import ACTION_KINDS

    policy, episodes = sample
    obs = episodes[0][1].observation
    original = policy.forward

    def force_color_target(*args, **kwargs):
        output = original(*args, **kwargs)
        output.action_logits.fill_(-100)
        output.action_logits[0, ACTION_KINDS.index("SELECT")] = 100
        output.target_logits.fill_(-100)
        output.target_logits[0, next(i for i, c in enumerate(obs.controls) if c.label == COLOR_CONTROL)] = 100
        return output

    monkeypatch.setattr(policy, "forward", force_color_target)
    with torch.no_grad():
        expected_workflow = original([obs]).state
        selected_state, _ = policy.selected_color_activity(obs)
        action, state, diagnostics = policy.act(obs)
    assert action.kind == "SELECT"
    assert torch.equal(state, expected_workflow)
    assert diagnostics["activity_pathway"] == "selected_measurement_color_graph"
    expected = policy.neural_activity(selected_state)
    assert diagnostics["top_neurons"] == expected["top_neurons"]
    assert diagnostics["populations"] == expected["populations"]
    assert diagnostics["workflow_activity"] == policy.neural_activity(expected_workflow)
