"""Tool-use simulator gates, explicitly not browser/course acceptance."""

import copy
import json
import socket

import pytest

from habfly.contracts import Action, task_completed
from habfly.environments.planet_calculations import (
    FIELDS,
    SEEDS,
    TEMPLATES,
    PlanetCalculationEnv,
    planet_cases,
)
from habfly.planet_knowledge import PlanetCalculator


def action(env, key, value=None):
    obs = env.observe()
    target = next(c.id for c in obs.controls if c.id.split(":", 1)[1] == key)
    result = env.step(
        Action(
            kind="SELECT" if value is not None else "CLICK",
            target=target,
            value=value,
            observation_revision=obs.revision,
        )
    )
    return result[-1]["result"]


def make_env(split="gate", count=1):
    cases = planet_cases(split, count)
    env = PlanetCalculationEnv(PlanetCalculator(), cases)
    env.reset(seed=cases[0]["seed"])
    return env, cases


def test_expert_completes_100_independent_cases_offline(monkeypatch):
    def no_network(*_args, **_kwargs):
        raise AssertionError("Network access is forbidden")

    monkeypatch.setattr(socket, "socket", no_network)
    env, cases = make_env(count=100)
    for case in cases:
        obs, _ = env.reset(seed=case["seed"])
        while not (env.terminated or env.truncated):
            obs, _, _, _, info = env.step(env.expert_action(obs))
            assert info["result"]["failure_reason"] is None
        assert env.completed and task_completed(env.observe())
        assert env.steps == 62
        assert env.metrics["numeric_fields_correct"] == len(FIELDS)
        assert env.metrics["invalid_actions"] == env.metrics["tool_errors"] == 0
        assert not env.observe().progress["course_acceptance_passed"]


def test_splits_cases_templates_and_reset_are_distinct_and_deterministic():
    splits = {split: planet_cases(split, 4) for split in SEEDS}
    hashes = [case["case_id"] for cases in splits.values() for case in cases]
    seeds = [case["seed"] for cases in splits.values() for case in cases]
    assert len(set(hashes)) == len(hashes) == len(set(seeds))
    assert len(set(TEMPLATES.values())) == len(TEMPLATES)
    assert planet_cases("train", 4) == splits["train"]
    env, cases = make_env("train")
    before = env.observe().model_dump(mode="json")
    action(env, "operation", "period_years")
    assert not env.results and not env.answers  # Selection does not execute.
    env.reset(seed=cases[0]["seed"])
    assert env.observe().model_dump(mode="json") == before
    assert set(env.case["expected"]) == set(FIELDS)
    text = json.dumps(before)
    assert '"expected"' not in text and '"expected_provenance"' not in text
    assert "independent_physics_only_course_pending" in text


def test_wrong_same_unit_source_executes_without_repair_and_pending_result_invalidates():
    env, _ = make_env()
    sources = env.case["measurements"]
    good = next(
        k for k, row in sources.items() if row["kind"] == "period_days" and row["source"] == "current star"
    )
    wrong = next(
        k for k, row in sources.items() if row["kind"] == "period_days" and row["source"] == "reference star"
    )
    action(env, "operation", "period_years")
    action(env, "parameter", "period_days")
    action(env, "source", wrong)
    action(env, "bind")
    action(env, "execute")
    assert env.results["r1"]["value"] == pytest.approx(sources[wrong]["value"] / 365)
    assert env.metrics["input_correct"] == 0
    action(env, "source", good)
    action(env, "bind")
    assert not env.results["r1"]["valid"] and not env.pending_result
    action(env, "execute")
    assert env.results["r2"]["value"] == pytest.approx(sources[good]["value"] / 365)
    env.reset(seed=env.case["seed"])
    assert not env.results and not env.answers


def test_missing_and_incompatible_inputs_are_errors_not_corrected():
    env, _ = make_env()
    action(env, "operation", "period_years")
    action(env, "execute")
    assert env.tool_error == "missing_or_extra_inputs" and not env.results
    wrong = next(k for k, row in env.case["measurements"].items() if row["unit"] == "nm")
    action(env, "parameter", "period_days")
    action(env, "source", wrong)
    action(env, "bind")
    action(env, "execute")
    assert env.tool_error == "incompatible_unit" and not env.results


def test_wrong_units_and_stop_do_not_count_as_completion():
    env, _ = make_env()
    while env.steps < 61:
        env.step(env.expert_action(env.observe()))
    action(env, "unit_planet_density", "nm")
    action(env, "check")
    assert not env.completed and not env.terminated
    env.step(Action(kind="STOP", observation_revision=env.steps))
    assert env.terminated and not task_completed(env.observe())


def test_observation_local_targets_and_128_step_limit():
    env, _ = make_env()
    obs = env.observe()
    old = env.expert_action(obs)
    env.step(old)
    result = env.step(old)[-1]["result"]
    assert result["failure_reason"] and env.metrics["invalid_actions"] == 1
    while not env.truncated:
        env.step(Action(kind="WAIT", observation_revision=env.steps))
    assert env.steps == 128 and not env.completed


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(scope="stellar"),
        lambda c: c.update(star_class="white_dwarf"),
        lambda c: c.update(required=["distance"]),
        lambda c: c.update(expected={}),
    ],
)
def test_incompatible_stellar_or_course_cases_rejected(mutation):
    case = copy.deepcopy(planet_cases("train", 1)[0])
    mutation(case)
    with pytest.raises(ValueError, match="Incompatible"):
        PlanetCalculationEnv(PlanetCalculator(), [case])


def test_existing_planet_course_oracle_stays_disabled():
    with pytest.raises(ValueError, match="no training oracle"):
        PlanetCalculator().reference_answers({})


def test_wrong_reference_result_lineage_is_not_counted_as_correct_binding():
    env, _ = make_env()
    source = next(
        k
        for k, row in env.case["measurements"].items()
        if row["kind"] == "period_days" and row["source"] == "reference star"
    )
    for key, value in (("operation", "period_years"), ("parameter", "period_days"), ("source", source)):
        action(env, key, value)
    action(env, "bind")
    action(env, "execute")
    for key, value in (("operation", "orbital_radius"), ("parameter", "period_years"), ("source", "r1")):
        action(env, key, value)
    action(env, "bind")
    assert env.metrics["input_attempts"] == 2 and env.metrics["input_correct"] == 0
