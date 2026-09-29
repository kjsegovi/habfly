"""Public temperature bridge fixtures, never a course-correctness oracle."""

import json
import os
from pathlib import Path

import pytest

from habfly.browser import BrowserSafetyStop
from habfly.browser_habitability_policy import HabitabilityBrowserToolEnv, visible_temperature_inputs
from habfly.contracts import Action
from habfly.environments.habitability_calculations import habitability_expert


def mapping():
    return {
        "observation": {
            "values": {
                "equilibrium_temp": {"value": "0"},
                "greenhouse": None,
                "measurements": {
                    "stellar_luminosity": {"value": 20.44, "unit": "Lsun"},
                    "orbital_radius": {"value": 0.5919, "unit": "au"},
                    "albedo": {"value": 0.05, "unit": "fraction"},
                },
            }
        }
    }


def test_reference_fixture_completes_without_oracle_reward_or_browser_write():
    env = HabitabilityBrowserToolEnv(mapping(), supplied_greenhouse_increment=30)
    assert "expected" not in env.case
    with pytest.raises(RuntimeError, match="no scripted expert"):
        env.expert_action(env.observe())
    for _ in range(28):
        result = env.step(habitability_expert(env.observe(), env.pack))
        assert result.reward == result.cumulative_reward == 0
        assert not result.observation.progress["task_completed"]
    assert result.terminated and result.failure_reason is None and env.proposals_complete
    assert env.answers["surface_temp"] - env.answers["equilibrium_temp"] == 30
    assert env.grades() == (set(), {})
    assert "expected" not in json.dumps(env.observe().model_dump(mode="json"))
    env.reset(seed=20000000)
    assert not env.proposals_complete and not env.observe().progress["temperature_proposals_complete"]
    assert env.answers == {} and env.results == {} and env.bindings == {}


@pytest.mark.parametrize("value", [None, True, "30", -1, 1, float("inf")])
def test_supplied_increment_is_explicit_and_bounded(value):
    with pytest.raises(BrowserSafetyStop, match="supplied_greenhouse"):
        visible_temperature_inputs(mapping(), supplied_greenhouse_increment=value)


@pytest.mark.parametrize(
    "bad", ["missing", "unit", "zero", "albedo_one", "nan", "bool", "prepopulated", "greenhouse"]
)
def test_missing_invalid_or_existing_values_are_not_repaired(bad):
    data = mapping()
    values = data["observation"]["values"]
    if bad == "prepopulated":
        values["equilibrium_temp"]["value"] = "500"
    elif bad == "greenhouse":
        values["greenhouse"] = "Weak (+10)"
    elif bad == "missing":
        del values["measurements"]["stellar_luminosity"]
    elif bad == "unit":
        values["measurements"]["orbital_radius"]["unit"] = "m"
    elif bad == "albedo_one":
        values["measurements"]["albedo"]["value"] = 1
    else:
        values["measurements"]["stellar_luminosity"]["value"] = {
            "zero": 0,
            "nan": float("nan"),
            "bool": True,
        }[bad]
    with pytest.raises(BrowserSafetyStop):
        visible_temperature_inputs(data, supplied_greenhouse_increment=0)


def test_zero_albedo_and_zero_increment_remain_valid():
    data = mapping()
    data["observation"]["values"]["measurements"]["albedo"]["value"] = 0
    assert [row["value"] for row in visible_temperature_inputs(data, supplied_greenhouse_increment=0)][
        -2:
    ] == [0, 0]


def test_incomplete_check_stops_without_choosing_inputs():
    env = HabitabilityBrowserToolEnv(mapping(), supplied_greenhouse_increment=0)
    control = next(c for c in env.observe().controls if c.id.endswith(":check"))
    result = env.step(Action(kind="CLICK", target=control.id))
    assert result.terminated and result.failure_reason == "temperature_proposals_incomplete"
    assert env.answers == {} and not env.proposals_complete


@pytest.mark.skipif(
    os.environ.get("HABFLY_REAL_GRAPH_TEST") != "1", reason="explicit local frozen-model check"
)
def test_frozen_real_policy_uses_visible_fixture_and_keeps_weights(monkeypatch):
    import socket

    import torch

    from habfly.data import load_graph
    from habfly.spreadsheet import SpreadsheetAdapter
    from habfly.training.habitability_evaluation import load_validated_pilot
    from habfly.training.planet_sequence import file_hash

    def deny(*args, **kwargs):
        raise AssertionError("offline temperature bridge contacted external service")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    checkpoint = Path("experiments/habitability-pilot-001/training/checkpoint.pt")
    before = file_hash(checkpoint)
    policy, _, _ = load_validated_pilot(
        checkpoint.parent.parent, load_graph("data/processed/graphs-v2/graph-2000")
    )
    weights = {k: v.clone() for k, v in policy.state_dict().items()}
    env = HabitabilityBrowserToolEnv(mapping(), supplied_greenhouse_increment=30)
    state = None
    with torch.no_grad():
        for steps in range(128):
            action, state, _ = policy.act(env.observe(), state)
            result = env.step(action)
            if result.terminated or result.truncated:
                break
    assert steps + 1 == 28 and env.proposals_complete and result.failure_reason is None
    assert file_hash(checkpoint) == before
    assert all(torch.equal(v, weights[k]) for k, v in policy.state_dict().items())
