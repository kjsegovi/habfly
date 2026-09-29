"""The first prerequisite is explicitly learned, never auto-executed."""

import socket

import pytest

from habfly.data import make_demo_graph
from habfly.environments.planet_calculations import PlanetCalculationEnv, planet_cases
from habfly.environments.planet_period import SCOPE, TEMPLATES, PlanetPeriodEnv, period_cases
from habfly.planet_knowledge import PlanetCalculator
from habfly.training.planet_calculations import collect_split, content_identity, summarize


def test_expert_period_gate_offline_100_and_parent_case_identity():
    calculator = PlanetCalculator()
    cases = period_cases("gate", 100)
    env = PlanetPeriodEnv(calculator, cases)
    for case, parent in zip(cases, planet_cases("gate", 100)):
        assert case["measurements"] == parent["measurements"]
        assert case["case_id"] != parent["case_id"]
        env.reset(seed=case["seed"])
        assert not env.results and not env.answers
        assert env.expert_action(env.observe()).value == "period_years"
        for _ in range(10):
            env.step(env.expert_action(env.observe()))
        assert env.completed and env.steps == 10
        assert env.metrics["numeric_fields_correct"] == env.metrics["required_fields"] == 1
        assert env.observe().progress["scope"] == SCOPE
        assert not env.observe().progress["course_acceptance_passed"]


def test_recorded_period_scope_units_and_metrics(tmp_path, monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", deny)
    calculator = PlanetCalculator()
    cases = {s: period_cases(s, 1) for s in ("train", "calibration", "development")}
    content = content_identity(
        calculator,
        cases,
        make_demo_graph(16),
        scope=SCOPE,
        required_fields=("period_years",),
        templates=TEMPLATES,
    )
    assert content["required_fields"] == ["period_years"]
    episodes, summaries = collect_split(
        calculator, cases["train"], tmp_path / "recorded", environment=PlanetPeriodEnv, expected_steps=10
    )
    assert len(episodes[0]) == 10
    report = summarize(summaries, required_count=1)
    assert report["completed"] == 1
    assert report["numeric_answer_accuracy"] == report["unit_selection_accuracy"] == 1
    assert all(s["observation"]["instruction"] == TEMPLATES["train"] for s in episodes[0])
    with pytest.raises(ValueError, match="Incompatible"):
        PlanetCalculationEnv(calculator, cases["train"])
    with pytest.raises(ValueError, match="Incompatible"):
        PlanetPeriodEnv(calculator, planet_cases("train", 1))
