"""Source-only updates may not alter the working quantity/unit or workflow weights."""

from copy import deepcopy

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.model import ConnectomePolicy
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.source_request import (
    TEMPLATES,
    attach_source_pointer,
    paired_accuracy,
    paired_records,
    source_probe,
    train_source,
    validate_pairs,
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
        selection_mode="measurement_identity_v1",
    )


def test_contrastive_data_separate_balanced_and_not_value_shortcuts():
    train, development = paired_records("train"), paired_records("development")
    assert len(train) == 144 and len(development) == 72
    assert train == paired_records("train")
    assert not set(TEMPLATES["train"]) & set(TEMPLATES["development"])
    for records in (train, development):
        validate_pairs(records)
        for left, right in zip(records[::2], records[1::2]):
            assert left["candidates"] == right["candidates"]
            assert left["expected_id"] != right["expected_id"]
            assert left["instruction"] != right["instruction"]
    for key in ("value", "id"):
        assert not {c[key] for r in train for c in r["candidates"]} & {
            c[key] for r in development for c in r["candidates"]
        }
    with pytest.raises(ValueError):
        paired_records("test")
    broken = deepcopy(train[:2])
    broken[1]["candidates"][0]["value"] += 1
    with pytest.raises(ValueError, match="differ only"):
        validate_pairs(broken)
    with pytest.raises(ValueError, match="complete"):
        validate_pairs(train[:1])


def test_source_head_uses_instruction_not_values_ids_or_private_labels():
    model = parent()
    attach_source_pointer(model)
    records = paired_records("train")[:2]
    contexts = torch.randn(1, 16).expand(2, -1)
    pointer = model.measurement_identity.source_request
    scores = pointer(records, contexts)
    assert not torch.equal(scores[0], scores[1])
    changed = deepcopy(records)
    for record in changed:
        record["expected_id"] = "private-label-not-an-input"
        record["requirement"] = {"quantity": "irrelevant", "unit": "irrelevant"}
        for candidate in record["candidates"]:
            candidate.update(value=0, id="renamed", kind="irrelevant", unit="irrelevant")
    for original, altered in zip(scores, pointer(changed, contexts)):
        torch.testing.assert_close(original, altered, rtol=0, atol=0)
    for record in changed:
        record["candidates"].reverse()
    for original, altered in zip(scores, pointer(changed, contexts)):
        torch.testing.assert_close(original, altered.flip(0))
    source_labels = [c["source"] for c in records[0]["candidates"]]
    for source in set(source_labels):
        same_source = scores[0][[s == source for s in source_labels]]
        assert (same_source == same_source[0]).all()
    assert not torch.equal(scores[0], pointer(records, contexts + 1)[0])
    probe = source_probe(model, records, contexts[0])
    assert probe["optimizer_updates"] == 0 and probe["finite_gradient"]
    assert probe["embedding_gradient_norm"] > 0 and probe["max_logit_change"] > 0
    for parameter in pointer.parameters():
        parameter.data.zero_()
    assert pointer(records, contexts)[0].count_nonzero() == 0
    changed[0]["instruction"] = "x" * model.tokenizer.max_length
    with pytest.raises(ValueError, match="token budget"):
        pointer(changed, contexts)


def test_training_and_reload_preserve_parent_and_quantity_unit_scores(tmp_path):
    policy, records, contexts = parent(), paired_records("train"), torch.randn(4, 16)
    frozen = {name: p.detach().clone() for name, p in policy.state_dict().items()}
    pair = records[:2]
    expected = policy.measurement_identity.quantity_unit_scores(pair, contexts[:2])
    obs = {
        "instruction": "Choose a calculation",
        "controls": [
            {
                "id": "random",
                "role": "combobox",
                "label": "Calculation",
                "surface": "calculation",
                "actions": ["SELECT"],
                "options": ["distance", "temperature"],
            }
        ],
    }
    action_logits = policy([obs]).action_logits.detach()
    option_scores = policy.option_scores(obs, obs["controls"][0], contexts[:1]).detach()
    save_checkpoint(tmp_path / "old.pt", policy, stage="test", seed=0)
    old, _ = load_checkpoint(tmp_path / "old.pt", policy.graph)
    assert not hasattr(old.measurement_identity, "source_request")
    attach_source_pointer(policy)
    new = {
        name: p.detach().clone()
        for name, p in policy.measurement_identity.source_request.state_dict().items()
    }
    optimizer, report = train_source(policy, records, contexts, updates=2)
    assert report["optimizer_steps"] == 2 and report["supervised_decisions"] == 24
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in frozen.items())
    assert any(
        not torch.equal(v, policy.measurement_identity.source_request.state_dict()[n]) for n, v in new.items()
    )
    for before, after in zip(expected, policy.measurement_identity.quantity_unit_scores(pair, contexts[:2])):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(action_logits, policy([obs]).action_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        option_scores, policy.option_scores(obs, obs["controls"][0], contexts[:1]), rtol=0, atol=0
    )
    path = tmp_path / "new.pt"
    save_checkpoint(
        path, policy, stage="source_request", seed=0, optimizer=optimizer, content_pack={"source": 1}
    )
    restored, _ = load_checkpoint(path, policy.graph, content_pack={"source": 1})
    assert restored.selection_mode == "measurement_source_v2"
    assert paired_accuracy(policy, pair, contexts) == paired_accuracy(restored, pair, contexts)
    with pytest.raises(ValueError, match="content-pack mismatch"):
        load_checkpoint(path, policy.graph, content_pack={"source": 2})
    with pytest.raises(ValueError, match="capped"):
        train_source(policy, records, contexts, updates=201)
    policy.measurement_identity.embedding.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="Only the new"):
        train_source(policy, records, contexts, updates=1)
    with pytest.raises(ValueError, match="unmodified"):
        attach_source_pointer(policy)
