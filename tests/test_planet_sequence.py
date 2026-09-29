"""Explicit continuation requires the original case, demonstration and model hashes."""

import copy
import json

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.environments.planet_calculations import planet_cases
from habfly.planet_knowledge import PlanetCalculator
from habfly.training.checkpoints import save_checkpoint, source_hash
from habfly.training.planet_calculations import BUDGET, COUNTS, content_identity, new_policy
from habfly.training.planet_sequence import run_sequence_followup, validated_smoke
from habfly.training.stellar import write_json


@pytest.fixture
def parent(tmp_path):
    # Loader-only fixture, not a learned-success claim or real-graph run.
    graph, calculator = make_demo_graph(16), PlanetCalculator()
    cases = {s: planet_cases(s, n) for s, n in COUNTS.items()}
    content = content_identity(calculator, cases, graph)
    episodes = {
        s: [[{"fixture_step": i, "seed": c["seed"]} for i in range(62)] for c in cases[s]]
        for s in ("train", "calibration", "development")
    }
    validation = [row for pair in zip(episodes["calibration"], episodes["development"]) for row in pair]
    training = {
        "optimizer_steps": 64,
        "training_data_hash": source_hash(episodes["train"]),
        "validation_data_hash": source_hash(validation),
    }
    for s, rows in cases.items():
        write_json(tmp_path / f"private-{s}-cases.json", rows)
    for s, rows in episodes.items():
        (tmp_path / f"expert-{s}").mkdir()
        write_json(tmp_path / f"expert-{s}/episodes.json", rows)
    write_json(tmp_path / "manifest.json", {"budget": BUDGET, "counts": COUNTS, "content": content})
    write_json(
        tmp_path / "report.json",
        {"content": content, "training": training, "checkpoint_reload_verified": True},
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


def test_loader_preserves_split_roles_and_rejects_synthetic_training(parent):
    path, graph, calculator = parent
    _, content, _, episodes, validation = validated_smoke(path, graph, calculator)
    assert validation[::2] == episodes["calibration"]
    assert validation[1::2] == episodes["development"]
    assert content["final_test_status"] == "not_generated_or_evaluated"
    with pytest.raises(ValueError, match="real 2,000-node"):
        run_sequence_followup(path, graph, path / "disallowed")
    assert not (path / "disallowed").exists()


@pytest.mark.parametrize(
    "filename,mutate",
    [
        ("manifest.json", lambda d: d["budget"].update(epochs=200)),
        ("private-train-cases.json", lambda d: d[0]["expected"].update(planet_mass=999)),
        ("expert-train/episodes.json", lambda d: d[0][0].update(fixture_step=999)),
        ("expert-calibration/episodes.json", lambda d: d[0][0].update(fixture_step=999)),
        ("expert-development/episodes.json", lambda d: d.pop()),
    ],
)
def test_modified_parent_budget_cases_or_demonstrations_rejected(parent, filename, mutate):
    path, graph, calculator = parent
    file = path / filename
    data = copy.deepcopy(json.loads(file.read_text()))
    mutate(data)
    write_json(file, data)
    with pytest.raises(ValueError, match="planet|Planet"):
        validated_smoke(path, graph, calculator)
