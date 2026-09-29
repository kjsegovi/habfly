"""Final temperature cases stay sealed until the fixed pilot passes."""

from copy import deepcopy

import pytest

from habfly.environments.habitability_calculations import SCOPE
from habfly.training.habitability_evaluation import (
    require_pilot_gate,
    reserve_final_set,
    run_final_evaluation,
)
from habfly.training.habitability_sequence import PILOT_BUDGET, PILOT_COUNTS
from habfly.training.stellar import write_json


def passed_pilot():
    content = {"scope": SCOPE, "splits": {s: {"count": n} for s, n in PILOT_COUNTS.items()}}
    return {
        "content": content,
        "profile": "pilot",
        "budget": PILOT_BUDGET,
        "training": {"optimizer_steps": 320, "epochs": 5, "training_episodes": 64, "validation_episodes": 32},
        "checkpoint_reload_verified": True,
        "frozen_text_verified": True,
        "parent_unchanged": True,
        "final_test_episodes": 0,
        "closed_loop": {
            s: {
                "episodes": n,
                "completed": n,
                "completion_rate": 1.0,
                "numeric_answer_accuracy": 1.0,
                "invalid_actions": 0,
                "tool_errors": 0,
                "api_failures": 0,
                "infrastructure_failures": 0,
            }
            for s, n in (("train", 64), ("development", 16))
        },
    }


def test_final_requires_exact_pilot_budget_and_completion_errors():
    report = passed_pilot()
    require_pilot_gate(report, report["content"])
    for key, value in (
        ("completed", 0),
        ("tool_errors", 1),
        ("numeric_answer_accuracy", 0.8),
        ("invalid_actions", 1),
        ("completion_rate", 0.5),
    ):
        changed = deepcopy(report)
        changed["closed_loop"]["development"][key] = value
        with pytest.raises(ValueError):
            require_pilot_gate(changed, changed["content"])
    changed = deepcopy(report)
    changed["budget"]["new_cases"] = 1000
    with pytest.raises(ValueError, match="exact pilot budget"):
        require_pilot_gate(changed, changed["content"])
    changed = deepcopy(report)
    changed["final_test_episodes"] = 100
    with pytest.raises(ValueError, match="no prior final"):
        require_pilot_gate(changed, changed["content"])


def test_reservation_is_once_only_and_distinct_from_planet_suite(tmp_path):
    path = reserve_final_set(tmp_path, "frozen", "first")
    original = path.read_bytes()
    assert path.name == "habitability-supplied-temperature-final-v1.reservation.json"
    with pytest.raises(ValueError, match="already reserved"):
        reserve_final_set(tmp_path, "different", "second")
    assert path.read_bytes() == original


def test_failed_pilot_does_not_generate_final_cases_or_reserve(monkeypatch, tmp_path):
    import habfly.training.habitability_evaluation as module

    def deny(*args, **kwargs):
        raise AssertionError("Failed pilot opened final test cases")

    monkeypatch.setattr(module, "habitability_cases", deny)
    training = tmp_path / "source/training"
    training.mkdir(parents=True)
    (training / "checkpoint.pt").write_bytes(b"fixture")
    report = passed_pilot()
    report["closed_loop"]["development"]["completed"] = 0
    write_json(
        training / "checkpoint.pt.json", {"provenance": {"habitability_calculations": report["content"]}}
    )
    write_json(training.parent / "report.json", report)
    output, registry = tmp_path / "output", tmp_path / "registry"
    with pytest.raises(ValueError, match="learned completion"):
        run_final_evaluation(training.parent, None, output, registry=registry)
    assert not registry.exists() and not output.exists()
