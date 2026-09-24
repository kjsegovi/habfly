from copy import deepcopy

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.environments.distance_diagnostic import DistanceDiagnosticEnv
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy
from habfly.training.source_aliases import (
    alias_records,
    rehearsal_records,
    validate_separation,
    workflow_cases,
)
from habfly.training.source_curriculum import resume_source_pointer
from habfly.training.source_request import train_source, validate_pairs


def test_alias_pairs_and_workflow_splits_are_separate_and_deterministic():
    calculator = LocalCalculator(load_knowledge_pack())
    records = {s: alias_records(s) for s in ("train", "development", "test")}
    cases = {s: workflow_cases(s, 18, calculator) for s in ("train", "development", "test", "manual")}
    assert records["train"] == alias_records("train")
    validate_separation(records, cases)
    validate_pairs(rehearsal_records())
    for rows in records.values():
        validate_pairs(rows)
        for left, right in zip(rows[::2], rows[1::2]):
            sources = [
                next(c["source"] for c in r["candidates"] if c["id"] == r["expected_id"])
                for r in (left, right)
            ]
            assert sources == ["current star", "reference star"]
            assert len(left["instruction"]) < 1022
    duplicate = deepcopy(cases)
    duplicate["test"][0]["case_id"] = cases["train"][0]["case_id"]
    with pytest.raises(ValueError, match="overlap"):
        validate_separation(records, duplicate)


def test_distance_case_instruction_is_visible_without_private_answers():
    calculator = LocalCalculator(load_knowledge_pack())
    case = workflow_cases("manual", 1, calculator)[0]
    env = DistanceDiagnosticEnv(calculator, [case], max_steps=32)
    observation, _ = env.reset(seed=case["seed"])
    assert observation["instruction"] == case["instruction"]
    assert "expected" not in observation["values"]
    while not env.terminated and not env.truncated:
        env.step(env.expert_action(env.observe()))
    assert env.observe().progress["task_completed"] and env.steps == 10


def test_expanded_budget_is_explicit_and_source_only():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = ConnectomePolicy(
        make_demo_graph(16),
        hidden_size=16,
        observation_encoding="structured_tool_v3",
        selection_mode="measurement_source_v2",
    )
    resume_source_pointer(policy)
    before = {n: v.clone() for n, v in policy.state_dict().items()}
    records, contexts = alias_records("train")[:12], torch.randn(4, 16)
    _, report = train_source(policy, records, contexts, updates=2, budget_limit=1000)
    assert len(report["losses"]) == 2 and all(torch.isfinite(torch.tensor(report["losses"])))
    assert all(
        torch.equal(v, policy.state_dict()[n]) for n, v in before.items() if ".source_request." not in n
    )
    for settings in (
        {"updates": 201},
        {"updates": 1001, "budget_limit": 1000},
        {"updates": 2, "budget_limit": 1001},
    ):
        with pytest.raises(ValueError, match="capped"):
            train_source(policy, records, contexts, **settings)
