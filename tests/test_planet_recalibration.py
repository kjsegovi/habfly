"""Calibration receipts do not authorize changed weights, cases or model code."""

import json
from types import SimpleNamespace

import pytest

from habfly.contracts import Action, Control, Observation
from habfly.training.checkpoints import source_hash
from habfly.training.planet_calculations import summarize
from habfly.training.planet_recalibration import (
    apply_refreshed_calibration,
    model_source_hashes,
    semantic_actions,
)
from habfly.training.planet_sequence import file_hash
from habfly.training.stellar import write_json
from habfly.training.train import episodes_to_examples


@pytest.fixture
def receipt(tmp_path):
    # Tiny transport fixture, not a training/evaluation success claim.
    source, output = tmp_path / "source", tmp_path / "refresh"
    source.mkdir()
    output.mkdir()
    observation = Observation(controls=[Control(id="local", label="Copy result")])
    step = {
        "observation": observation.model_dump(mode="json"),
        "action": Action(kind="CLICK", target="local").model_dump(mode="json"),
    }
    episodes = [[step] for _ in range(16)]
    cases = [{"seed": 12000000 + i} for i in range(16)]
    summaries = [{"completed": True, "steps": 62} for _ in range(16)]
    for path, value in (
        (source / "expert-calibration/episodes.json", episodes),
        (source / "learned-development/episodes.json", episodes),
        (source / "private-development-cases.json", cases),
        (output / "development/episodes.json", episodes),
        (output / "development/summaries.json", summaries),
    ):
        path.parent.mkdir(exist_ok=True)
        write_json(path, value)
    checkpoint = source / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable test fixture")
    content = {"task": "planet_calculations"}
    calibration = {
        "status": "calibrated",
        "action": {"temperature": 0.5},
        "target": {"temperature": 0.75},
        "scope": "recorded_planet_calibration_reuse_after_source_change",
        "data_hash": source_hash([e.observation.model_dump() for e in episodes_to_examples(episodes)]),
    }
    report = {
        "version": 1,
        "mode": "frozen_recalibration_recorded_cases",
        "checkpoint_sha256": file_hash(checkpoint),
        "content_hash": source_hash(content),
        "model_sources": model_source_hashes(),
        "calibration_episodes_hash": source_hash(episodes),
        "development_cases_hash": source_hash(cases),
        "original_development_actions_hash": source_hash(semantic_actions(episodes)),
        "development_episodes_sha256": file_hash(output / "development/episodes.json"),
        "development_summaries_sha256": file_hash(output / "development/summaries.json"),
        "development": summarize(summaries),
        "action_temperature": 0.5,
        "target_temperature": 0.75,
        "calibration": calibration,
        "checkpoint_unchanged": True,
        "action_streams_match": True,
        "optimizer_updates": 0,
        "final_test_episodes": 0,
        "fresh_unseen_evaluation": False,
    }
    path = output / "report.json"
    write_json(path, report)
    return path, {"checkpoint": checkpoint, "dataset": source, "content": content}


def test_overlay_applies_measured_confidence_without_touching_checkpoint(receipt):
    path, kwargs = receipt
    before = file_hash(kwargs["checkpoint"])
    policy = SimpleNamespace()
    result = apply_refreshed_calibration(policy, path, **kwargs)
    assert policy.action_temperature == 0.5 and policy.target_temperature == 0.75
    assert result["optimizer_updates"] == 0 and not result["fresh_unseen_evaluation"]
    assert file_hash(kwargs["checkpoint"]) == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_sources", {}),
        ("checkpoint_sha256", "changed"),
        ("content_hash", "changed"),
        ("calibration_episodes_hash", "changed"),
        ("development_cases_hash", "changed"),
        ("action_temperature", 2.0),
        ("target_temperature", float("nan")),
        ("action_temperature", True),
        ("optimizer_updates", 1),
        ("final_test_episodes", 100),
        ("fresh_unseen_evaluation", True),
        ("checkpoint_unchanged", False),
    ],
)
def test_changed_receipt_rejects_before_calibration_mutation(receipt, key, value):
    path, kwargs = receipt
    report = json.loads(path.read_text())
    report[key] = value
    path.write_text(json.dumps(report))
    policy = SimpleNamespace(action_temperature=1.0, target_temperature=1.0)
    with pytest.raises(ValueError, match="provenance mismatch"):
        apply_refreshed_calibration(policy, path, **kwargs)
    assert policy.action_temperature == policy.target_temperature == 1.0


@pytest.mark.parametrize("relative", ["development/episodes.json", "development/summaries.json"])
def test_changed_replayed_evidence_rejected(receipt, relative):
    path, kwargs = receipt
    (path.parent / relative).write_text("[]")
    with pytest.raises(ValueError, match="provenance mismatch"):
        apply_refreshed_calibration(SimpleNamespace(), path, **kwargs)


def test_changed_behavior_rejected_even_with_updated_file_hash(receipt):
    path, kwargs = receipt
    episodes_path = path.parent / "development/episodes.json"
    episodes = json.loads(episodes_path.read_text())
    episodes[0][0]["action"]["kind"] = "WAIT"
    write_json(episodes_path, episodes)
    report = json.loads(path.read_text())
    report["development_episodes_sha256"] = file_hash(episodes_path)
    write_json(path, report)
    with pytest.raises(ValueError, match="development verification failed"):
        apply_refreshed_calibration(SimpleNamespace(), path, **kwargs)
