"""Ordered, fully learned readout: finite distributions and versioned compatibility."""

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.model import ConnectomePolicy
from habfly.model.color_head import OrdinalColorHead


def test_ordinal_probabilities_and_gradients_remain_finite_at_extreme_scores():
    torch.manual_seed(0)
    head = OrdinalColorHead(16)
    features = torch.randn(18, 16)
    head.fit_normalization(features)
    assert head.normalization_fitted
    logits = head.ordinal_logits(features)
    assert (head.cutpoints().diff() > 0).all()
    probabilities = head(features).exp()
    assert torch.allclose(probabilities.sum(-1), torch.ones(18), atol=1e-6)
    assert torch.allclose(probabilities[:, 1:-1], logits[:, :-1].sigmoid() - logits[:, 1:].sigmoid())
    loss, nll, ordinal = head.loss(features * 100, torch.arange(18) % 9)
    loss.backward()
    assert all(torch.isfinite(t) for t in (loss, nll, ordinal))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_train_only_normalization_is_frozen_and_constant_features_are_safe():
    head = OrdinalColorHead(16)
    training = torch.ones(4, 16)
    head.fit_normalization(training)
    before = (head.feature_mean.clone(), head.feature_scale.clone())
    assert torch.isfinite(head(training)).all()
    head(torch.zeros(16, 16))  # Evaluation must never refit statistics.
    assert torch.equal(before[0], head.feature_mean) and torch.equal(before[1], head.feature_scale)
    with pytest.raises(ValueError, match="already fitted"):
        head.fit_normalization(training)
    for bad in (torch.ones(1, 16), torch.ones(4, 8), torch.full((4, 16), float("nan"))):
        with pytest.raises(ValueError, match="finite training"):
            OrdinalColorHead(16).fit_normalization(bad)


def test_color_readout_configuration_is_opt_in_and_does_not_change_numeric_model():
    graph = make_demo_graph()
    legacy = ConnectomePolicy(graph, hidden_size=16)
    assert "color_readout" not in legacy.configuration()
    assert not any(name.startswith("color_") for name in legacy.state_dict())
    with pytest.raises(ValueError, match="Incompatible color readout"):
        ConnectomePolicy(graph, color_readout="ordinal_v2")
    with pytest.raises(ValueError, match="Incompatible color readout"):
        ConnectomePolicy(graph, observation_encoding="structured_color_v1", color_readout="unknown")
