"""Whole-sequence diagnostics may reuse only the recorded smoke contract."""

import json

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.environments.habitability_calculations import habitability_cases
from habfly.habitability_knowledge import HabitabilityCalculator
from habfly.training.checkpoints import save_checkpoint, source_hash
from habfly.training.habitability_calculations import BUDGET, COUNTS, content_identity, new_policy
from habfly.training.habitability_sequence import (
    FOLLOWUP_BUDGET,
    PILOT_BUDGET,
    PILOT_COUNTS,
    require_diagnostic_gate,
    run_sequence_followup,
    validate_followup_parent,
    validated_smoke,
)
from habfly.training.stellar import write_json


@pytest.fixture
def parent(tmp_path):
    # Hash/loader-only fixture; it is not a learned policy or real-graph run.
    graph, calculator = make_demo_graph(16), HabitabilityCalculator()
    cases = {s: habitability_cases(s, n) for s, n in COUNTS.items()}
    content = content_identity(calculator, cases, graph)
    episodes = {
        s: [[{"fixture_step": i, "seed": c["seed"]} for i in range(28)] for c in cases[s]]
        for s in ("train", "calibration", "development")
    }
    validation = [row for pair in zip(episodes["calibration"], episodes["development"]) for row in pair]
    training = {
        "optimizer_steps": 32,
        "training_data_hash": source_hash(episodes["train"]),
        "validation_data_hash": source_hash(validation),
    }
    for split, rows in cases.items():
        write_json(tmp_path / f"private-{split}-cases.json", rows)
    for split, rows in episodes.items():
        (tmp_path / f"expert-{split}").mkdir()
        write_json(tmp_path / f"expert-{split}/episodes.json", rows)
    write_json(tmp_path / "manifest.json", {"budget": BUDGET, "counts": COUNTS, "content": content})
    write_json(
        tmp_path / "report.json",
        {
            "content": content,
            "training": training,
            "checkpoint_reload_verified": True,
            "final_test_episodes": 0,
        },
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        save_checkpoint(
            tmp_path / "training/checkpoint.pt",
            new_policy(graph),
            stage="test",
            seed=0,
            content_pack=content,
            evaluation=training,
        )
        yield tmp_path, graph, calculator
    finally:
        torch.set_num_threads(previous)


def test_parent_hashes_split_roles_and_exact_update_cap(parent):
    path, graph, calculator = parent
    _, content, _, episodes, validation = validated_smoke(path, graph, calculator)
    assert validation[::2] == episodes["calibration"]
    assert validation[1::2] == episodes["development"]
    assert content["final_test_status"] == "not_generated_or_evaluated"
    assert FOLLOWUP_BUDGET["epochs"] * len(episodes["train"]) == 80
    assert FOLLOWUP_BUDGET["sequence_length"] == 28
    assert FOLLOWUP_BUDGET["new_cases"] == FOLLOWUP_BUDGET["final_test_episodes"] == 0
    with pytest.raises(ValueError, match="real 2,000-node"):
        run_sequence_followup(path, graph, path / "disallowed")
    assert not (path / "disallowed").exists()


@pytest.mark.parametrize(
    "filename,mutate",
    [
        ("manifest.json", lambda d: d["budget"].update(epochs=200)),
        ("report.json", lambda d: d.update(final_test_episodes=100)),
        ("private-train-cases.json", lambda d: d[0]["expected"].update(surface_temp=999)),
        ("expert-train/episodes.json", lambda d: d[0][0].update(fixture_step=999)),
        ("expert-calibration/episodes.json", lambda d: d[0][0].update(fixture_step=999)),
        ("expert-development/episodes.json", lambda d: d.pop()),
    ],
)
def test_modified_parent_rejected(parent, filename, mutate):
    path, graph, calculator = parent
    file = path / filename
    value = json.loads(file.read_text())
    mutate(value)
    write_json(file, value)
    with pytest.raises(ValueError, match="temperature|Temperature"):
        validated_smoke(path, graph, calculator)


@pytest.fixture
def followup(parent):
    path, graph, calculator = parent
    policy, content, _, episodes, validation = validated_smoke(path, graph, calculator)
    directory = path / "followup"
    directory.mkdir()
    training = {
        "optimizer_steps": 80,
        "training_data_hash": source_hash(episodes["train"]),
        "validation_data_hash": source_hash(validation),
    }
    write_json(directory / "manifest.json", {"content": content, "budget": FOLLOWUP_BUDGET})
    write_json(
        directory / "report.json",
        {
            "content": content,
            "budget": FOLLOWUP_BUDGET,
            "training": training,
            "checkpoint_reload_verified": True,
            "final_test_episodes": 0,
        },
    )
    optimizer = torch.optim.AdamW(policy.parameters())
    parameter = next(policy.parameters())
    optimizer.state[parameter] = {
        "step": torch.tensor(80.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }
    save_checkpoint(
        directory / "training/checkpoint.pt",
        policy,
        stage="test",
        seed=0,
        content_pack=content,
        evaluation=training,
        optimizer=optimizer,
    )
    return directory, graph, episodes, validation


def test_refinement_requires_matching_saved_optimizer_and_demonstrations(followup):
    directory, graph, episodes, validation = followup
    policy, optimizer, _ = validate_followup_parent(directory, graph, episodes, validation)
    assert policy.hidden_size == 16
    assert max(int(s["step"]) for s in optimizer["state"].values()) == 80
    altered = json.loads(json.dumps(episodes))
    altered["train"][0][0]["fixture_step"] = -1
    with pytest.raises(ValueError, match="Incompatible temperature refinement"):
        validate_followup_parent(directory, graph, altered, validation)


def test_refinement_rejects_mismatched_moment_step(followup):
    directory, graph, episodes, validation = followup
    path = directory / "training/checkpoint.pt"
    payload = torch.load(path, weights_only=True)
    next(iter(payload["optimizer"]["state"].values()))["step"] = torch.tensor(79.0)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="optimizer step mismatch"):
        validate_followup_parent(directory, graph, episodes, validation)


def test_pilot_has_fixed_budget_and_requires_error_free_diagnostic():
    from habfly.training.planet_calculations import ERRORS

    scores = {
        split: {"episodes": n, "completed": n, "numeric_answer_accuracy": 1.0, **dict.fromkeys(ERRORS, 0)}
        for split, n in (("train", 4), ("development", 2))
    }
    require_diagnostic_gate({"closed_loop": scores})
    assert PILOT_BUDGET["epochs"] * PILOT_COUNTS["train"] == 320
    assert PILOT_BUDGET["final_test_episodes"] == 0
    assert PILOT_COUNTS == {"gate": 100, "train": 64, "calibration": 16, "development": 16}
    for key, value in (("completed", 1), ("numeric_answer_accuracy", 0.5), *[(k, 1) for k in ERRORS]):
        changed = json.loads(json.dumps(scores))
        changed["development"][key] = value
        with pytest.raises(ValueError, match="completed four/two-case"):
            require_diagnostic_gate({"closed_loop": changed})
