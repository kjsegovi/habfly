import copy

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy
from habfly.training.source_request import train_source, validate_pairs
from habfly.training.temperature_source import (
    composition_records,
    source_records,
    validate_separation,
    workflow_cases,
)


def test_contrastive_language_dataset_separates_new_holdouts():
    records = {s: source_records(s) for s in ("train", "development")}
    calculator = LocalCalculator(load_knowledge_pack())
    cases = {
        s: workflow_cases(s, 16, calculator)
        for s in ("train", "calibration", "development", "test", "manual")
    }
    validate_separation(cases, records)
    assert len(records["train"]) == 700 and len(records["development"]) == 200
    for rows in records.values():
        validate_pairs(rows)
        sources = [
            next(c["source"] for c in row["candidates"] if c["id"] == row["expected_id"]) for row in rows
        ]
        assert sources.count("current star") == sources.count("reference star")
    assert all(c["seed"] >= 5400000 for c in cases["test"])
    assert len({c["instruction"] for c in cases["test"]}) == 4
    # The failed old wording is now explicitly rehearsal, not passed off as unseen.
    assert any("not those of the" in r["instruction"] for r in records["train"])
    broken = copy.deepcopy(cases)
    broken["test"][0]["instruction"] = records["train"][0]["instruction"]
    with pytest.raises(ValueError, match="overlap"):
        validate_separation(broken, records)


def test_new_source_mode_trains_only_source_head_with_finite_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = ConnectomePolicy(
        make_demo_graph(16),
        hidden_size=16,
        observation_encoding="structured_tool_v4",
        selection_mode="measurement_result_v3",
        control_encoding="semantic_tool_v1",
    )
    model.requires_grad_(False)
    model.measurement_identity.source_request.requires_grad_(True)
    before = {n: p.detach().clone() for n, p in model.state_dict().items()}
    records = source_records("train")[:12]
    _, report = train_source(model, records, torch.randn(3, 16), updates=1, budget_limit=1)
    assert report["optimizer_steps"] == 1 and report["gradient_norms"][0] > 0
    changed = [n for n, p in model.state_dict().items() if not torch.equal(p, before[n])]
    assert changed and all(n.startswith("measurement_identity.source_request.") for n in changed)
    with pytest.raises(ValueError, match="explicit budget"):
        train_source(model, records, torch.randn(3, 16), updates=2, budget_limit=1)


@pytest.mark.parametrize("punctuation", [False, True])
def test_composition_orders_do_not_leak_whole_heldout_instructions(punctuation):
    records = composition_records(punctuation=punctuation)
    assert len(records) == (7488 if punctuation else 6912)
    validate_pairs(records)
    calculator = LocalCalculator(load_knowledge_pack())
    cases = {
        s: workflow_cases(s, 8, calculator) for s in ("train", "calibration", "development", "test", "manual")
    }
    validate_separation(cases, {"train": records, "development": source_records("development")})
    first_word_sources, last_word_sources = set(), set()
    for record in records:
        instruction = record["instruction"]
        expected = next(c["source"] for c in record["candidates"] if c["id"] == record["expected_id"])
        first = (
            "current star"
            if instruction.index("current star") < instruction.index("reference star")
            else "reference star"
        )
        first_word_sources.add(expected == first)
        last_word_sources.add(expected != first)
    assert first_word_sources == last_word_sources == {False, True}
