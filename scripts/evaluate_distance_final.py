"""Frozen evaluation of the approved vocabulary checkpoint on eight distance cases.

Run from the repository root with one fresh output directory. No training,
checkpoint selection, calibration fitting, new cases, or network access.
After this evaluation these cases are consumed, not a reusable unseen gate.
"""

import argparse
import hashlib
import importlib.metadata
import json
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from habfly.data import load_graph
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint, source_hash
from habfly.training.distance_diagnostic import closed_loop, fit_gate, teacher_forced_stages
from habfly.training.stellar import write_json
from habfly.training.train import peak_process_rss_bytes

CHECKPOINT = Path("experiments/distance-vocabulary-001/training/checkpoint.pt")
CHECKPOINT_SHA256 = "4e857c8bc739f77d902403a0e005e9c39446b059750326e3d6dfa3ad242dd32f"
SOURCE = Path("experiments/distance-diagnostic-002")
TEST_SHA256 = "23b1acdc03b309d9e0ce99c844c26ab8823b8136e1afa8f699a3742f6a08352c"
GRAPH = Path("data/processed/graphs-v2/graph-2000")


def deny(*args, **kwargs):
    raise RuntimeError("Frozen evaluation attempted network or spreadsheet access")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cases(payload, splits):
    """Check the sealed payload and separation before handing cases to the environment."""
    expected = splits["test"]
    cases, episodes = payload["cases"], payload["episodes"]
    ids = {case["case_id"] for case in cases}
    seeds = [case["seed"] for case in cases]
    if (
        source_hash(payload) != expected["sha256"]
        or len(cases) != expected["count"]
        or len(episodes) != len(cases)
        or any(len(episode) != 10 for episode in episodes)
        or len(ids) != len(cases)
        or sorted(ids) != expected["case_ids"]
        or seeds != expected["seeds"]
        or len(set(seeds)) != len(seeds)
        or len({c["inputs"]["parallax"] for c in cases}) != len(cases)
        or any(c["split"] != "test" or c["required"] != ["distance"] for c in cases)
    ):
        raise ValueError("Test payload identity, scope or episode mismatch")
    for split in ("train", "calibration", "development"):
        if ids.intersection(splits[split]["case_ids"]) or set(seeds).intersection(splits[split]["seeds"]):
            raise ValueError("Test split overlaps a consumed split")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh evaluation output; existing artifacts are preserved")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    SpreadsheetAdapter.__init__ = deny
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if file_hash(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Approved checkpoint hash mismatch")
    parent = CHECKPOINT.parent.parent
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    parent_report = json.loads((parent / "report.json").read_text())
    original = json.loads((SOURCE / "manifest.json").read_text())
    content = parent_report["content"]
    splits = content["dataset_splits"]
    if (
        not parent_report["training_fit_gate_passed"]
        or content != parent_manifest["content"]
        or Path(parent_manifest["source"]) != SOURCE
        or splits != original["content"]["dataset_splits"]
        or splits["test"]["sha256"] != TEST_SHA256
        or splits["test"]["count"] != 8
        or splits["test"]["seeds"] != list(range(900000, 900008))
        or content["max_steps"] != 32
    ):
        raise ValueError("Approved experiment or test manifest mismatch")
    graph, pack = load_graph(GRAPH), load_knowledge_pack()
    policy, checkpoint_manifest = load_checkpoint(CHECKPOINT, graph, content_pack=content)
    if (
        len(graph.body_ids) != 2000
        or policy.graph_hash != parent_manifest["graph_hash"]
        or pack.checksum != content["knowledge_pack_hash"]
        or policy.configuration() != checkpoint_manifest["model"]
    ):
        raise ValueError("Graph, model or knowledge-pack mismatch")
    provenance = checkpoint_manifest["provenance"]
    code_hashes = {str(p): file_hash(p) for p in sorted(Path("src/habfly").rglob("*.py"))}
    versions = {name: importlib.metadata.version(name) for name in provenance["dependency_versions"]}
    if (
        code_hashes != provenance["code_hashes"]
        or file_hash("uv.lock") != provenance["dependency_lock_hash"]
        or versions != provenance["dependency_versions"]
    ):
        raise ValueError("Source or dependency drift since the approved checkpoint")
    policy.requires_grad_(False)
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    calculator = LocalCalculator(pack)
    verification = calculator.verify()
    args.output.mkdir(parents=True, exist_ok=False)
    # Persist the choice and gate before opening test observations or private answers.
    write_json(
        args.output / "manifest.json",
        {
            "version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "scope": "frozen_eight_case_distance_evaluation",
            "content": content,
            "checkpoint": {"path": str(CHECKPOINT), "sha256": CHECKPOINT_SHA256},
            "dataset": {"path": str(SOURCE / "test.json"), **splits["test"]},
            "graph": {"path": str(GRAPH), "sha256": policy.graph_hash, "nodes": 2000},
            "code_hashes": code_hashes,
            "runner_sha256": file_hash(__file__),
            "dependency_versions": versions,
            "seed": 0,
            "device": "cpu",
            "threads": 1,
            "max_steps_per_episode": 32,
            "optimizer_updates": 0,
            "calibration": policy.calibration,
            "calculator_verification": verification,
            "gate": {
                "min_teacher_forced_exact_accuracy": 0.95,
                "required_completions": 8,
                "max_invalid_actions": 0,
                "max_tool_errors": 0,
                "max_api_failures": 0,
                "max_infrastructure_failures": 0,
            },
            "evaluation_order": ["closed_loop", "teacher_forced_attribution"],
            "test_scope_after_opening": "consumed; never use as a fresh checkpoint-selection gate",
        },
    )
    started = time.perf_counter()
    try:
        write_json(args.output / "status.json", {"stage": "opening_test", "test_consumed": True})
        payload = json.loads((SOURCE / "test.json").read_text())
        validate_cases(payload, splits)
        print("Evaluating fixed checkpoint: eight cases, at most 32 actions each", flush=True)
        with torch.no_grad():
            rollout = closed_loop(policy, calculator, payload["cases"], args.output / "test", 32)
            teacher = teacher_forced_stages(policy, payload["episodes"])
        unchanged = all(torch.equal(before[n], v) for n, v in policy.state_dict().items())
        if not unchanged or file_hash(CHECKPOINT) != CHECKPOINT_SHA256:
            raise RuntimeError("Frozen checkpoint or model state changed during evaluation")
        passed = fit_gate(teacher, rollout)
        infrastructure_failed = bool(rollout["infrastructure_failures"] or rollout["api_failures"])
        report = {
            "scope": "local_tool_assisted_distance_only",
            "test_consumed": True,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "test_sha256": source_hash(payload),
            "outcome": "infrastructure_failure"
            if infrastructure_failed
            else ("distance_diagnostic_passed" if passed else "unseen_generalization_failed"),
            "distance_gate_passed": passed,
            "stellar_acceptance_gate_passed": False,
            "closed_loop": rollout,
            "teacher_forced": teacher,
            "optimizer_updates": 0,
            "model_state_unchanged": unchanged,
            "checkpoint_file_unchanged": True,
            "calibration": policy.calibration,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_process_rss_bytes(),
        }
        write_json(args.output / "report.json", report)
        write_json(
            args.output / "status.json",
            {"stage": "finished", "outcome": report["outcome"], "test_consumed": True},
        )
        print(
            json.dumps(
                {
                    "outcome": report["outcome"],
                    "completed": rollout["completed"],
                    "requested": 8,
                    "report": str(args.output / "report.json"),
                }
            )
        )
    except BaseException as exc:
        write_json(
            args.output / "status.json",
            {"stage": "failed", "test_consumed": True, "error": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
