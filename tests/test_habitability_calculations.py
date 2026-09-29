import copy
import json

import pytest

from habfly.contracts import Action
from habfly.environments.habitability_calculations import (
    FIELDS,
    TEMPLATES,
    HabitabilityCalculationEnv,
    habitability_cases,
)
from habfly.habitability_knowledge import HabitabilityCalculator


def environment(split="gate", count=1):
    cases = habitability_cases(split, count)
    env = HabitabilityCalculationEnv(HabitabilityCalculator(), cases)
    env.reset(seed=cases[0]["seed"])
    return env


def act(env, key, value=None):
    obs = env.observe()
    target = next(c.id for c in obs.controls if c.id.endswith(":" + key))
    return env.step(Action(kind="SELECT" if value is not None else "CLICK", target=target, value=value))


def test_inverse_fixture_expert_completes_100_without_a_course_oracle():
    env = environment(count=100)
    for seed in env.cases:
        env.reset(seed=seed)
        while not env.terminated and not env.truncated:
            env.step(env.expert_action(env.observe()))
        assert env.completed and env.steps == 28
        assert all(
            env.metrics[k] == 0
            for k in ("invalid_actions", "tool_errors", "api_failures", "infrastructure_failures")
        )
        obs = env.observe()
        assert not obs.progress["course_acceptance_passed"]
        assert not obs.progress["habitability_decision_learned"]
        assert all(env.units[k] == "K" for k in FIELDS)
        for result in env.results.values():
            assert env.answers[result["kind"]] == result["value"]
    with pytest.raises(ValueError, match="no training oracle"):
        env.adapter.reference_answers()


def test_private_grading_does_not_affect_public_observation_or_expert_decisions():
    first = environment()
    other_case = copy.deepcopy(first.case)
    other_case["expected"] = dict.fromkeys(FIELDS, -123456789)
    second = HabitabilityCalculationEnv(HabitabilityCalculator(), [other_case])
    second.reset(seed=other_case["seed"])
    assert first.observe() == second.observe()
    assert first.expert_action(first.observe()) == second.expert_action(second.observe())
    text = json.dumps(first.observe().model_dump(mode="json"))
    assert '"expected"' not in text and '"case_id"' not in text


def test_valid_wrong_source_is_executed_without_repair_and_units_are_not_duplicated():
    env = environment()
    act(env, "operation", "equilibrium_temp")
    for kind in ("stellar_luminosity", "orbital_radius", "albedo"):
        ref = next(
            k
            for k, m in env.case["measurements"].items()
            if m["kind"] == kind and m["source"] == "reference star"
        )
        act(env, "parameter", kind)
        act(env, "source", ref)
        act(env, "bind")
    act(env, "execute")
    assert not env.tool_error and env.metrics["input_correct"] == 0
    assert env.results["r1"]["value"] != pytest.approx(env.case["expected"]["equilibrium_temp"])
    for c in env.observe().controls:
        assert len(c.options) == len(set(c.options))
    env.reset(seed=env.case["seed"])
    assert not env.results and not env.answers and not env.bindings


def test_missing_and_zero_inputs_are_distinct_and_wrong_units_are_errors():
    env = environment()
    act(env, "operation", "surface_temp")
    act(env, "execute")
    assert env.tool_error and not env.results
    calculator = env.adapter
    inputs = {
        "equilibrium_temp": {"value": 250, "unit": "K"},
        "greenhouse_increment": {"value": 0, "unit": "K"},
    }
    assert calculator.execute("surface_temp", inputs, assumptions=calculator.pack.assumptions).value == 250
    inputs["greenhouse_increment"]["unit"] = "fraction"
    assert not calculator.execute("surface_temp", inputs, assumptions=calculator.pack.assumptions).ok


def test_development_splits_and_templates_are_distinct_without_consuming_final_test():
    splits = {name: habitability_cases(name, 4) for name in ("train", "calibration", "development", "gate")}
    assert len({TEMPLATES[name] for name in splits}) == 4
    assert len({c["seed"] for rows in splits.values() for c in rows}) == 16
    assert len({c["case_id"] for rows in splits.values() for c in rows}) == 16
    for quantity in ("stellar_luminosity", "orbital_radius", "albedo"):
        payloads = [
            m["value"]
            for rows in splits.values()
            for c in rows
            for m in c["measurements"].values()
            if m["kind"] == quantity and m["source"] == "current star"
        ]
        assert len(set(payloads)) == 16
    assert habitability_cases("train", 4) == splits["train"]
    for count in (0, 101, True):
        with pytest.raises(ValueError):
            habitability_cases("train", count)
