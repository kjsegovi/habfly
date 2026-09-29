"""Curriculum scopes keep all real decisions and do not inflate full-task scores."""

import pytest

from habfly.environments.planet_calculations import planet_cases
from habfly.environments.planet_curriculum import STAGES, stage_cases, stage_environment
from habfly.environments.planet_period import PlanetPeriodEnv, period_cases
from habfly.planet_knowledge import PlanetCalculator


@pytest.mark.parametrize("stage", STAGES)
def test_all_stages_expert_complete_100_cases_at_declared_lengths(stage):
    spec = STAGES[stage]
    cases = stage_cases(stage, "gate", 100)
    env = stage_environment(stage)(PlanetCalculator(), cases)
    for case in cases:
        env.reset(seed=case["seed"])
        assert len(next(c for c in env.observe().controls if c.label == "Calculation").options) == 6
        while not env.terminated and not env.truncated:
            env.step(env.expert_action(env.observe()))
        assert env.completed and env.steps == spec.steps
        assert env.metrics["invalid_actions"] == env.metrics["tool_errors"] == 0
        assert env.metrics["numeric_fields_correct"] == len(spec.required)
        assert env.metrics["units_correct"] == len(spec.required)
        assert not env.observe().progress["course_acceptance_passed"]


def test_original_full_and_period_cases_and_observations_are_unchanged():
    calculator = PlanetCalculator()
    assert stage_cases("full", "train", 4) == planet_cases("train", 4)
    assert stage_cases("period", "train", 4) == period_cases("train", 4)
    cases = period_cases("train", 1)
    old = PlanetPeriodEnv(calculator, cases)
    new = stage_environment("period")(calculator, cases)
    assert old.reset(seed=cases[0]["seed"])[0] == new.reset(seed=cases[0]["seed"])[0]
    with pytest.raises(ValueError, match="Incompatible"):
        stage_environment("orbit")(calculator, cases)
    for spec in STAGES.values():
        assert len(set(spec.templates.values())) == 5


def test_orbit_expert_reuses_period_result_without_automatic_execution():
    cases = stage_cases("orbit", "train", 1)
    env = stage_environment("orbit")(PlanetCalculator(), cases)
    env.reset(seed=cases[0]["seed"])
    assert env.expert_action(env.observe()).value == "period_years"
    for _ in range(5):
        env.step(env.expert_action(env.observe()))
    assert env.results["r1"]["kind"] == "period_years"
    assert not env.answers
    for _ in range(8):
        env.step(env.expert_action(env.observe()))
    assert env.results["r2"]["kind"] == "orbital_radius"
    assert env.results["r2"]["bindings"]["period_years"] == "r1"


def test_advancement_requires_learned_closed_loop_not_expert_or_teacher_scores():
    from copy import deepcopy

    from habfly.training.planet_period import require_parent_gate

    report = {
        "closed_loop": {
            split: {
                "episodes": n,
                "completed": n,
                "numeric_answer_accuracy": 1.0,
                "invalid_actions": 0,
                "tool_errors": 0,
                "infrastructure_failures": 0,
                "api_failures": 0,
            }
            for split, n in (("train", 4), ("development", 2))
        }
    }
    require_parent_gate(report)
    for key, value in (
        ("completed", 0),
        ("tool_errors", 1),
        ("invalid_actions", 1),
        ("numeric_answer_accuracy", 0.5),
    ):
        changed = deepcopy(report)
        changed["closed_loop"]["development"][key] = value
        with pytest.raises(ValueError, match="closed-loop gate"):
            require_parent_gate(changed)
    with pytest.raises(ValueError, match="closed-loop gate"):
        require_parent_gate({"expert_gate": {"completed": 100}, "teacher_forced": {"accuracy": 1.0}})
