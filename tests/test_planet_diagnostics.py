"""Counterfactuals are visible-only, nonmutating and never optimizer updates."""

from copy import deepcopy

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.environments.planet_period import PlanetPeriodEnv, period_cases
from habfly.planet_knowledge import PlanetCalculator
from habfly.training.planet_calculations import new_policy
from habfly.training.planet_diagnostics import counterfactual, decision, diagnose_trajectories
from habfly.training.stellar import record_episode


def test_counterfactuals_preserve_identity_bindings_and_original_data():
    env = PlanetPeriodEnv(PlanetCalculator(), period_cases("train", 1))
    episode, _ = record_episode(env, 10000000)
    observation = episode[-2]["observation"]
    saved = deepcopy(observation)
    numeric = counterfactual(observation, "numeric_payload")
    assert observation == saved
    assert numeric["controls"] == observation["controls"]
    assert numeric["calculation"]["bindings"] == observation["calculation"]["bindings"]
    assert numeric["values"]["measurements"] != observation["values"]["measurements"]
    assert numeric["calculation"]["results"] != observation["calculation"]["results"]
    assert decision(episode[-2]["action"], numeric) == decision(episode[-2]["action"], saved)
    ordered = counterfactual(observation, "control_order")
    assert decision(episode[-2]["action"], ordered) == decision(episode[-2]["action"], saved)
    with pytest.raises(ValueError, match="Unknown"):
        counterfactual(observation, "not_supported")


def test_diagnostic_is_frozen_and_does_not_claim_closed_loop_success():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        policy = new_policy(make_demo_graph(16))
        env = PlanetPeriodEnv(PlanetCalculator(), period_cases("train", 1))
        episode, _ = record_episode(env, 10000000)
        before = deepcopy(episode)
        report = diagnose_trajectories(policy, [episode])
        assert episode == before
        assert report["decisions"] == 10
        assert report["optimizer_updates"] == 0 and report["parameters_unchanged"]
        assert report["scope"].endswith("not_closed_loop")
        assert report["counterfactuals"]["control_order"]["changed_decisions"] == 0
        assert "completed" not in report
        assert all(parameter.grad is None for parameter in policy.parameters())
    finally:
        torch.set_num_threads(old_threads)
