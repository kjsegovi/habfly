"""Isolation, visibility, compatibility and bounded learning for identity selection."""

from copy import deepcopy

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.model import ConnectomePolicy
from habfly.model.measurement_identity import measurement_request
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.measurement_identity import (
    TEMPLATES,
    attach_identity_pointer,
    evaluate_identity,
    recognition_records,
    train_identity,
)


@pytest.fixture(autouse=True)
def cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(previous)


def parent():
    return ConnectomePolicy(
        make_demo_graph(16),
        hidden_size=16,
        observation_encoding="structured_tool_v3",
        selection_mode="option_pointer_v1",
    )


def source_observation():
    record = recognition_records("train")[0]
    return {
        "instruction": record["instruction"],
        "values": {"measurements": {c["id"]: c for c in record["candidates"]}},
        "calculation": {"parameter": "x", "reference_card": {"inputs": {"x": record["requirement"]}}},
        "controls": [
            {
                "id": "opaque-target",
                "label": "Measurement or result",
                "role": "combobox",
                "surface": "calculation",
                "actions": ["SELECT"],
                "enabled": True,
                "options": [c["id"] for c in record["candidates"]],
            }
        ],
    }


def test_datasets_separate_repeatable_and_factorial():
    train, development = recognition_records("train"), recognition_records("development")
    assert train == recognition_records("train")
    assert len(train) == 144 and len(development) == 72
    assert not set(TEMPLATES["train"]) & set(TEMPLATES["development"])
    for field in ("id", "value"):
        left = {c[field] for r in train for c in r["candidates"]}
        right = {c[field] for r in development for c in r["candidates"]}
        assert not left & right
    assert not {r["instruction"] for r in train} & {r["instruction"] for r in development}
    for record in train + development:
        correct = next(c for c in record["candidates"] if c["id"] == record["expected_id"])
        assert correct["kind"] == record["requirement"]["quantity"]
        assert correct["unit"] == record["requirement"]["unit"]
        for field in ("kind", "unit", "source"):
            assert any(
                all(c[k] == correct[k] for k in ("kind", "unit", "source") if k != field)
                and c[field] != correct[field]
                for c in record["candidates"]
            )
    with pytest.raises(ValueError):
        recognition_records("test")


def test_values_ids_labels_order_excluded_but_semantics_and_graph_used():
    policy = parent()
    attach_identity_pointer(policy)
    record = recognition_records("train")[0]
    pooled = torch.randn(1, 16)
    original = policy.measurement_identity([record], pooled)[0]
    changed = deepcopy(record)
    changed["expected_id"] = "hidden-label-never-read"
    for candidate in changed["candidates"]:
        candidate.update(value=-123456.0, id="arbitrary")
    torch.testing.assert_close(original, policy.measurement_identity([changed], pooled)[0], rtol=0, atol=0)
    changed["candidates"].reverse()
    torch.testing.assert_close(original, policy.measurement_identity([changed], pooled)[0].flip(0))
    for key, value in (
        ("kind", "different quantity"),
        ("unit", "different unit"),
        ("source", "different star"),
    ):
        mutated = deepcopy(record)
        mutated["candidates"][0][key] = value
        assert not torch.equal(original, policy.measurement_identity([mutated], pooled)[0])
    assert not torch.equal(original, policy.measurement_identity([record], pooled + 1)[0])
    # Finite gradients originate in a learned scorer, not a deterministic matching rule.
    original.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in policy.measurement_identity.parameters()
    )
    assert all(
        p.grad is None for n, p in policy.named_parameters() if not n.startswith("measurement_identity.")
    )


def test_source_routing_and_non_source_outputs_unchanged():
    policy, obs = parent(), source_observation()
    pooled = torch.randn(1, 16)
    non_source = {
        "id": "op",
        "label": "Calculation",
        "role": "combobox",
        "surface": "calculation",
        "options": ["distance", "temperature"],
    }
    before = policy.option_scores(obs, non_source, pooled).detach()
    parent_outputs = policy([obs])
    attach_identity_pointer(policy)
    for field in ("action_logits", "target_logits", "state", "pooled"):
        torch.testing.assert_close(
            getattr(parent_outputs, field), getattr(policy([obs]), field), rtol=0, atol=0
        )
    torch.testing.assert_close(before, policy.option_scores(obs, non_source, pooled), rtol=0, atol=0)
    request = measurement_request(obs, obs["controls"][0])
    torch.testing.assert_close(
        policy.option_scores(obs, obs["controls"][0], pooled),
        policy.measurement_identity([request], pooled)[0],
    )
    assert measurement_request(obs, non_source) is None
    missing = deepcopy(obs)
    missing["calculation"]["parameter"] = ""
    assert measurement_request(missing, missing["controls"][0]) is None
    mixed = deepcopy(obs["controls"][0])
    mixed["options"].append("result-id")
    assert measurement_request(obs, mixed) is None


def test_wrong_choice_is_not_repaired_and_oversized_fields_fail():
    policy = parent()
    attach_identity_pointer(policy)
    record = recognition_records("train")[0]
    correct = next(c for c in record["candidates"] if c["id"] == record["expected_id"])
    record["candidates"].remove(correct)
    record["candidates"].append(correct)
    for parameter in policy.measurement_identity.parameters():
        parameter.data.zero_()
    assert int(policy.measurement_identity([record], torch.zeros(1, 16))[0].argmax()) == 0
    assert record["candidates"][0]["id"] != record["expected_id"]
    record["instruction"] = "x" * policy.tokenizer.max_length
    with pytest.raises(ValueError, match="token budget"):
        policy.measurement_identity([record], torch.zeros(1, 16))


def test_short_training_frozen_parent_and_checkpoint_reload(tmp_path):
    policy = parent()
    old = {n: p.detach().clone() for n, p in policy.state_dict().items()}
    save_checkpoint(tmp_path / "parent.pt", policy, stage="test", seed=0)
    legacy, _ = load_checkpoint(tmp_path / "parent.pt", policy.graph)
    assert not hasattr(legacy, "measurement_identity")
    attach_identity_pointer(policy)
    new = {n: p.detach().clone() for n, p in policy.measurement_identity.state_dict().items()}
    records, contexts = recognition_records("train"), torch.randn(4, 16)
    optimizer, report = train_identity(policy, records, contexts, updates=2)
    assert report["optimizer_steps"] == 2 and report["supervised_decisions"] == 24
    assert all(torch.equal(v, policy.state_dict()[n]) for n, v in old.items())
    assert any(not torch.equal(v, policy.measurement_identity.state_dict()[n]) for n, v in new.items())
    save_checkpoint(
        tmp_path / "new.pt",
        policy,
        stage="measurement_identity",
        seed=0,
        optimizer=optimizer,
        content_pack={"identity": 1},
    )
    restored, _ = load_checkpoint(tmp_path / "new.pt", policy.graph, content_pack={"identity": 1})
    assert restored.selection_mode == "measurement_identity_v1"
    assert evaluate_identity(restored, records[:12], contexts) == evaluate_identity(
        policy, records[:12], contexts
    )
    with pytest.raises(ValueError, match="content-pack mismatch"):
        load_checkpoint(tmp_path / "new.pt", policy.graph, content_pack={"identity": 2})
    with pytest.raises(ValueError, match="capped"):
        train_identity(policy, records, contexts, updates=201)
    policy.embedding.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="Only the new"):
        train_identity(policy, records, contexts, updates=1)
    with pytest.raises(ValueError, match="unmodified"):
        attach_identity_pointer(policy)
