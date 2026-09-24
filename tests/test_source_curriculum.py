"""Known-word composition must be separate from vocabulary and consumed evaluations."""

from copy import deepcopy

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.model import ConnectomePolicy
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.source_curriculum import (
    HELD_OUT,
    PRIOR_COMBINATIONS,
    curriculum_records,
    resume_source_pointer,
    validate_curriculum,
)
from habfly.training.source_request import train_source


def datasets():
    return {s: curriculum_records(s) for s in ("vocabulary", "train", "development", "synonym_diagnostic")}


@pytest.fixture(autouse=True)
def cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(previous)


def model():
    return ConnectomePolicy(
        make_demo_graph(16),
        hidden_size=16,
        observation_encoding="structured_tool_v3",
        selection_mode="measurement_source_v2",
    )


def test_curriculum_is_deterministic_with_known_words_and_fresh_combinations():
    rows = datasets()
    assert rows == datasets()
    coverage = validate_curriculum(rows)
    assert coverage["split_examples"] == {
        "vocabulary": 144,
        "train": 432,
        "development": 144,
        "synonym_diagnostic": 144,
    }
    assert coverage["composition_unknown_words"] == []
    assert coverage["unseen_diagnostic_words"] == ["exclude", "gather", "omit", "retrieve"]
    assert not coverage["synonym_diagnostic_is_gate"]
    assert not HELD_OUT & PRIOR_COMBINATIONS
    for split in ("train", "development"):
        combinations = {(r["curriculum"]["positive"], r["curriculum"]["negative"]) for r in rows[split]}
        assert len(combinations) == (12 if split == "train" else 4)
        assert {r["curriculum"]["order"] for r in rows[split]} == {"positive_first", "negative_first"}
    with pytest.raises(ValueError, match="Unsupported"):
        curriculum_records("test")
    unknown = deepcopy(rows)
    for record in unknown["development"][:2]:
        record["instruction"] += " Surreptitiously."
    with pytest.raises(ValueError, match="untaught"):
        validate_curriculum(unknown)
    overlap = deepcopy(rows)
    for record in overlap["development"][:2]:
        record["candidates"][0]["id"] = overlap["train"][0]["candidates"][0]["id"]
    with pytest.raises(ValueError, match="overlap"):
        validate_curriculum(overlap)


def test_positive_and_negative_vocabulary_labels_are_correct():
    records = curriculum_records("vocabulary")
    examples = {
        "Use the current star's parallax.": "current star",
        "Ignore the reference star's parallax.": "current star",
        "Not the current star's flux.": "reference star",
        "Disregard the current star's wavelength.": "reference star",
    }
    for text, source in examples.items():
        matching = [r for r in records if r["instruction"] == text]
        assert len(matching) == 3  # Each requested unit, including counterfactual labels.
        for record in matching:
            expected = next(c for c in record["candidates"] if c["id"] == record["expected_id"])
            assert expected["source"] == source
            assert expected["kind"] == record["requirement"]["quantity"]
            assert expected["unit"] == record["requirement"]["unit"]


def test_fixed_schedule_resume_freeze_and_checkpoint(tmp_path):
    policy, records, contexts = model(), datasets(), torch.randn(4, 16)
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    resume_source_pointer(policy)
    assert all(torch.equal(v, policy.state_dict()[n]) for n, v in before.items())
    progress = []
    optimizer, report = train_source(
        policy,
        records["train"],
        contexts,
        updates=3,
        warmup_records=records["vocabulary"],
        warmup_updates=1,
        rehearsal_pairs=1,
        progress_callback=progress.append,
    )
    assert report["optimizer_steps"] == 3 and report["supervised_decisions"] == 36
    assert report["curriculum"] == {
        "warmup_updates": 1,
        "rehearsal_pairs": 1,
        "vocabulary_decisions": 16,
        "composition_decisions": 20,
    }
    assert [p["stage"] for p in progress] == ["vocabulary", "composition", "composition"]
    assert any(
        not torch.equal(v, policy.state_dict()[n]) for n, v in before.items() if ".source_request." in n
    )
    assert all(
        torch.equal(v, policy.state_dict()[n]) for n, v in before.items() if ".source_request." not in n
    )
    path = tmp_path / "curriculum.pt"
    save_checkpoint(
        path, policy, stage="source_vocabulary", seed=0, optimizer=optimizer, content_pack={"curriculum": 1}
    )
    restored, _ = load_checkpoint(path, policy.graph, content_pack={"curriculum": 1})
    pair = records["development"][:2]
    expected = policy.measurement_identity(pair, contexts[:2])
    actual = restored.measurement_identity(pair, contexts[:2])
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    changed = deepcopy(pair)
    for record in changed:
        record["curriculum"] = {"secret_correct_source": "other", "split": "fabricated"}
        record["expected_id"] = "never-used-as-a-feature"
    for left, right in zip(expected, policy.measurement_identity(changed, contexts[:2])):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for settings in (
        {"updates": 201},
        {"updates": 3, "warmup_updates": 1},
        {"updates": 3, "warmup_records": records["vocabulary"], "warmup_updates": 3},
    ):
        with pytest.raises(ValueError):
            train_source(policy, records["train"], contexts, **settings)
    invalid = ConnectomePolicy(policy.graph, hidden_size=16, selection_mode="measurement_identity_v1")
    with pytest.raises(ValueError, match="existing"):
        resume_source_pointer(invalid)
