"""Use synthetic payloads only: never inspect sealed experiment cases in regression tests."""

import copy
import importlib.util
from pathlib import Path

import pytest

from habfly.training.checkpoints import source_hash

spec = importlib.util.spec_from_file_location("distance_final", Path("scripts/evaluate_distance_final.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def payload_and_splits():
    payload = {
        "cases": [
            {
                "case_id": "synthetic",
                "seed": 42,
                "split": "test",
                "required": ["distance"],
                "inputs": {"parallax": 0.1},
            }
        ],
        "episodes": [[{} for _ in range(10)]],
    }
    splits = {s: {"case_ids": [], "seeds": []} for s in ("train", "calibration", "development")}
    splits["test"] = {"sha256": source_hash(payload), "count": 1, "case_ids": ["synthetic"], "seeds": [42]}
    return payload, splits


def test_sealed_payload_validates_without_changing_it():
    payload, splits = payload_and_splits()
    before = copy.deepcopy(payload)
    runner.validate_cases(payload, splits)
    assert payload == before


@pytest.mark.parametrize("change", ["hash", "scope", "steps", "count", "seed", "overlap"])
def test_sealed_payload_rejects_tampering_and_overlap(change):
    payload, splits = payload_and_splits()
    if change == "hash":
        payload["cases"][0]["inputs"]["parallax"] = 0.2
    elif change == "scope":
        payload["cases"][0]["required"] = ["luminosity"]
    elif change == "steps":
        payload["episodes"][0].pop()
    elif change == "count":
        splits["test"]["count"] = 2
    elif change == "seed":
        splits["test"]["seeds"] = [43]
    elif change == "overlap":
        splits["development"]["case_ids"] = ["synthetic"]
    if change != "hash":
        splits["test"]["sha256"] = source_hash(payload)
    with pytest.raises(ValueError):
        runner.validate_cases(payload, splits)
