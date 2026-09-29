"""Final cases stay unopened until a verified pilot passes; one reservation."""

from copy import deepcopy

import pytest

from habfly.data import make_demo_graph
from habfly.environments.planet_calculations import SCOPE, planet_cases
from habfly.planet_knowledge import PlanetCalculator
from habfly.training.checkpoints import source_hash
from habfly.training.planet_calculations import collect_split, content_identity
from habfly.training.planet_evaluation import require_pilot_gate, reserve_final_set, verify_parent_artifacts
from habfly.training.planet_period import PILOT_COUNTS, run_curriculum_diagnostic, run_planet_pilot
from habfly.training.stellar import write_json


def passed_pilot():
    content = {"scope": SCOPE, "splits": {s: {"count": n} for s, n in PILOT_COUNTS.items()}}
    return {
        "content": content,
        "profile": "pilot",
        "budget": {"additional_optimizer_updates": 320},
        "training": {"optimizer_steps": 320, "epochs": 5, "training_episodes": 64, "validation_episodes": 32},
        "checkpoint_reload_verified": True,
        "frozen_text_verified": True,
        "parent_unchanged": True,
        "closed_loop": {
            s: {
                "episodes": n,
                "completed": n,
                "completion_rate": 1.0,
                "numeric_answer_accuracy": 1.0,
                "invalid_actions": 0,
                "tool_errors": 0,
                "infrastructure_failures": 0,
                "api_failures": 0,
            }
            for s, n in (("train", 64), ("development", 16))
        },
    }


def test_final_gate_requires_full_pilot_and_actual_closed_loop_scores():
    report = passed_pilot()
    require_pilot_gate(report, report["content"])
    for key, value in (
        ("completed", 0),
        ("completion_rate", 0.5),
        ("tool_errors", 1),
        ("numeric_answer_accuracy", 0.8),
    ):
        changed = deepcopy(report)
        changed["closed_loop"]["development"][key] = value
        with pytest.raises(ValueError, match="learned completion"):
            require_pilot_gate(changed, changed["content"])
    for key, value in (("epochs", 20), ("optimizer_steps", 80), ("training_episodes", 4)):
        changed = deepcopy(report)
        changed["training"][key] = value
        with pytest.raises(ValueError, match="fixed-budget pilot"):
            require_pilot_gate(changed, changed["content"])


def test_final_set_reservation_cannot_be_overwritten_by_new_checkpoint(tmp_path):
    path = reserve_final_set(tmp_path, "first", "one")
    saved = path.read_bytes()
    with pytest.raises(ValueError, match="already reserved"):
        reserve_final_set(tmp_path, "second", "two")
    assert path.read_bytes() == saved


def test_parent_artifacts_validate_cases_and_trajectory_hashes(tmp_path):
    graph, calculator = make_demo_graph(8), PlanetCalculator()
    cases = {s: planet_cases(s, 1) for s in ("train", "calibration", "development")}
    content = content_identity(calculator, cases, graph)
    episodes = {}
    for split, rows in cases.items():
        write_json(tmp_path / f"private-{split}-cases.json", rows)
        episodes[split], _ = collect_split(calculator, rows, tmp_path / f"expert-{split}")
    report = {
        "training": {
            "training_data_hash": source_hash(episodes["train"]),
            "validation_data_hash": source_hash([*episodes["calibration"], *episodes["development"]]),
        }
    }
    assert verify_parent_artifacts(tmp_path, content, report) == cases
    bad = deepcopy(episodes["train"])
    bad[0][0]["action"]["value"] = "wrong"
    write_json(tmp_path / "expert-train/episodes.json", bad)
    with pytest.raises(ValueError, match="trajectory artifact"):
        verify_parent_artifacts(tmp_path, content, report)
    write_json(tmp_path / "private-train-cases.json", [])
    with pytest.raises(ValueError, match="case artifact"):
        verify_parent_artifacts(tmp_path, content, report)


def test_pilot_budget_entrypoint_is_explicit_and_rejects_prerequisite_scope(monkeypatch, tmp_path):
    import habfly.training.planet_period as module

    with pytest.raises(ValueError, match="complete scope"):
        run_curriculum_diagnostic(None, None, tmp_path, stage="mass", pilot=True)
    with pytest.raises(ValueError, match="real 2,000-node"):
        run_planet_pilot(None, make_demo_graph(8), tmp_path)
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(module, "run_curriculum_diagnostic", capture)
    run_planet_pilot("parent", "graph", "output")
    assert seen == {"stage": "full", "progress": None, "pilot": True, "resume_optimizer": True}
    assert PILOT_COUNTS == {"gate": 100, "train": 64, "calibration": 16, "development": 16}
