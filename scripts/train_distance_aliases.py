"""Bounded source-only training or a separately invoked frozen 100-case distance test.

Training uses one declared budget, CPU/one thread/seed 0. Existing biological,
workflow and quantity/unit parameters remain frozen. No browser or Sheets access.
"""

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import torch
from train_measurement_identity import workflow_contexts

from habfly.data import load_graph
from habfly.environments.distance_diagnostic import DistanceDiagnosticEnv
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint, save_checkpoint, source_hash
from habfly.training.distance_diagnostic import closed_loop, teacher_forced_stages
from habfly.training.measurement_identity import evaluate_identity
from habfly.training.source_aliases import (
    VERSION,
    alias_records,
    rehearsal_records,
    validate_separation,
    workflow_cases,
)
from habfly.training.source_curriculum import resume_source_pointer
from habfly.training.source_request import paired_accuracy, train_source
from habfly.training.stellar import record_episode, write_json
from habfly.training.train import peak_process_rss_bytes


def deny(*args, **kwargs):
    raise RuntimeError("Distance alias run attempted network or spreadsheet access")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(rollout, count):
    return (
        rollout["completed"] == count
        and len(rollout["episodes"]) == count
        and all(e["steps"] == 10 for e in rollout["episodes"])
        and not any(
            rollout[k] for k in ("invalid_actions", "tool_errors", "api_failures", "infrastructure_failures")
        )
    )


def evaluate(directory, graph, calculator):
    target = directory / "final"
    if target.exists():
        raise ValueError(
            "Final evaluation already opened; preserve it and use a new holdout for future tuning"
        )
    report = json.loads((directory / "report.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if not report["development_gate_passed"] or report["content"] != manifest["content"]:
        raise ValueError("Development gate or content identity mismatch")
    checkpoint = directory / "training/checkpoint.pt"
    if file_hash(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    policy, _ = load_checkpoint(checkpoint, graph, content_pack=report["content"])
    if calculator.pack.checksum != report["content"]["knowledge_pack_hash"]:
        raise ValueError("Knowledge pack changed")
    policy.requires_grad_(False)
    before = {n: v.detach().clone() for n, v in policy.state_dict().items()}
    target.mkdir()
    write_json(
        target / "manifest.json",
        {
            "checkpoint_sha256": file_hash(checkpoint),
            "optimizer_updates": 0,
            "dataset": report["content"]["workflow_splits"]["test"],
            "gate": {"minimum_completed": 90, "requested": 100, "invalid_actions_and_errors": 0},
            "test_consumed": True,
        },
    )
    rows = json.loads((directory / "test.json").read_text())
    if source_hash(rows) != report["content"]["workflow_splits"]["test"]["sha256"] or len(rows) != 100:
        raise ValueError("Sealed test dataset mismatch")
    started = time.perf_counter()
    rollout = closed_loop(policy, calculator, rows, target / "rollouts", 32)
    unchanged = all(torch.equal(v, policy.state_dict()[n]) for n, v in before.items())
    if not unchanged or file_hash(checkpoint) != report["checkpoint_sha256"]:
        raise RuntimeError("Frozen model changed")
    passed = (
        rollout["completed"] >= 90
        and len(rollout["episodes"]) == 100
        and not any(
            rollout[k] for k in ("invalid_actions", "tool_errors", "api_failures", "infrastructure_failures")
        )
    )
    result = {
        "scope": "local_tool_assisted_distance_only",
        "distance_gate_passed": passed,
        "stellar_acceptance_gate_passed": False,
        "test_consumed": True,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "optimizer_updates": 0,
        "model_state_unchanged": unchanged,
        "closed_loop": rollout,
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_process_rss_bytes(),
        "calibration": policy.calibration,
    }
    write_json(target / "report.json", result)
    print(
        json.dumps(
            {
                "completed": rollout["completed"],
                "requested": 100,
                "distance_gate_passed": passed,
                "final_report": str(target / "report.json"),
            }
        )
    )


def train(args, graph, calculator):
    if args.output.exists():
        raise ValueError("Choose a fresh experiment output; existing artifacts are preserved")
    parent_report = json.loads((args.parent.parent.parent / "report.json").read_text())
    policy, _ = load_checkpoint(args.parent, graph, content_pack=parent_report["content"])
    if (
        len(graph.body_ids) != 2000
        or policy.selection_mode != "measurement_source_v2"
        or policy.hidden_size != 16
        or policy.observation_encoding != "structured_tool_v3"
        or calculator.pack.checksum != parent_report["content"]["knowledge_pack_hash"]
    ):
        raise ValueError("Incompatible workflow/knowledge/graph parent")
    original = Path("experiments/distance-diagnostic-002")
    old = json.loads((original / "train.json").read_text())
    if source_hash(old) != parent_report["content"]["dataset_splits"]["train"]["sha256"]:
        raise ValueError("Original training context hash mismatch")
    contexts = workflow_contexts(policy, old["episodes"])
    records = {s: alias_records(s) for s in ("train", "development")}
    rehearsal = rehearsal_records()
    cases = {
        s: workflow_cases(s, n, calculator)
        for s, n in (("train", 24), ("development", 36), ("test", 100), ("manual", 12))
    }
    validate_separation(records, cases)
    if {r["instruction"] for r in rehearsal} & {r["instruction"] for r in records["development"]}:
        raise ValueError("Rehearsal overlaps development")
    args.output.mkdir(parents=True, exist_ok=False)
    content = {
        **parent_report["content"],
        "diagnostic": VERSION,
        "training_scope": "source_request_only; known aliases and held-out templates",
        "workflow_splits": {
            s: {"count": len(rows), "sha256": source_hash(rows), "seeds": [c["seed"] for c in rows]}
            for s, rows in cases.items()
        },
        "alias_splits": {s: {"count": len(rows), "sha256": source_hash(rows)} for s, rows in records.items()},
        "rehearsal": {
            "scope": "previously_consumed_not_holdout",
            "count": len(rehearsal),
            "sha256": source_hash(rehearsal),
        },
    }
    for split, rows in cases.items():
        write_json(args.output / f"{split}.json", rows)
    for split, rows in records.items():
        write_json(args.output / f"recognition-{split}.json", rows)
    write_json(args.output / "rehearsal.json", rehearsal)
    parent_hash = file_hash(args.parent)
    write_json(
        args.output / "manifest.json",
        {
            "content": content,
            "parent": {"path": str(args.parent), "sha256": parent_hash},
            "graph_hash": policy.graph_hash,
            "graph_nodes": 2000,
            "optimizer_updates": args.updates,
            "batch_size": 12,
            "seed": 0,
            "threads": 1,
            "device": "cpu",
            "runner_sha256": file_hash(__file__),
            "test_status": "sealed_until_separate_evaluation",
            "development_gate": {
                "source_accuracy": 0.98,
                "paired_accuracy": 0.95,
                "quantity_unit_accuracy": 1.0,
                "rehearsal_accuracy": 0.98,
                "completed": 36,
                "maximum_steps": 10,
                "invalid_actions_and_errors": 0,
            },
        },
    )
    started = time.perf_counter()
    resume_source_pointer(policy)
    policy.calibration = {"status": "uncalibrated", "reason": "source_alias_curriculum"}
    inherited = {
        n: v.detach().clone()
        for n, v in policy.state_dict().items()
        if not n.startswith("measurement_identity.source_request.")
    }
    before = evaluate_identity(policy, records["development"], contexts)

    def progress(row):
        write_json(args.output / "status.json", {"stage": "training", **row})
        with (args.output / "training-progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if row["optimizer_steps"] % 100 == 0:
            print(f"Source update {row['optimizer_steps']}/{args.updates}: {row['loss']:.5f}", flush=True)

    optimizer, training = train_source(
        policy,
        records["train"] + rehearsal,
        contexts,
        updates=args.updates,
        budget_limit=1000,
        progress_callback=progress,
    )
    unchanged = all(torch.equal(v, policy.state_dict()[n]) for n, v in inherited.items())
    if not unchanged or file_hash(args.parent) != parent_hash:
        raise RuntimeError("Frozen parent tensors or file changed")
    recognition = {
        s: evaluate_identity(policy, rows, contexts)
        for s, rows in {**records, "rehearsal": rehearsal}.items()
    }
    paired = paired_accuracy(policy, records["development"], contexts)
    checkpoint = args.output / "training/checkpoint.pt"
    save_checkpoint(
        checkpoint, policy, stage="source_aliases", seed=0, optimizer=optimizer, content_pack=content
    )
    restored, _ = load_checkpoint(checkpoint, graph, content_pack=content)
    if any(not torch.equal(v, restored.state_dict()[n]) for n, v in policy.state_dict().items()):
        raise RuntimeError("Checkpoint reload changed parameters")
    print("Evaluating fresh development workflows; final 100 remain sealed", flush=True)
    development = closed_loop(restored, calculator, cases["development"], args.output / "development", 32)
    seen = closed_loop(restored, calculator, old["cases"], args.output / "seen", 32)
    teacher = teacher_forced_stages(restored, old["episodes"])
    # Expert contract validation is independent of the learned policy and never trains it.
    expert_count = 0
    for case in workflow_cases("train", 100, calculator):
        _, summary = record_episode(DistanceDiagnosticEnv(calculator, [case], max_steps=32), case["seed"])
        if summary["completed"] and not summary["invalid_actions"] and not summary["tool_errors"]:
            expert_count += 1
    gates = {
        "source": recognition["development"]["accuracy"]["source"] >= 0.98,
        "paired": paired["accuracy"] >= 0.95,
        "rehearsal": recognition["rehearsal"]["accuracy"]["source"] >= 0.98,
        "quantity_unit": all(r["accuracy"][k] == 1 for r in recognition.values() for k in ("kind", "unit")),
        "workflow": clean(development, 36) and clean(seen, 4),
        "expert": expert_count == 100,
        "frozen_parent": unchanged,
    }
    report = {
        "content": content,
        "training": training,
        "before": before,
        "recognition": recognition,
        "paired": paired,
        "development": development,
        "seen": seen,
        "teacher_forced": teacher,
        "expert_completed": expert_count,
        "gates": gates,
        "development_gate_passed": all(gates.values()),
        "checkpoint_reload_verified": True,
        "frozen_parent_unchanged": unchanged,
        "checkpoint_sha256": file_hash(checkpoint),
        "test": {"status": "not_run"},
        "stellar_acceptance_gate_passed": False,
        "calibration": restored.calibration,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_process_rss_bytes(),
    }
    write_json(args.output / "report.json", report)
    write_json(
        args.output / "status.json", {"stage": "finished", "development_gate_passed": all(gates.values())}
    )
    print(
        json.dumps(
            {
                "gates": gates,
                "source_accuracy": recognition["development"]["accuracy"],
                "completed": development["completed"],
                "report": str(args.output / "report.json"),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if (
        (args.updates is not None and not 1 <= args.updates <= 1000)
        or (args.evaluate and args.updates is not None)
        or (args.evaluate and args.parent)
        or (not args.evaluate and not args.parent)
    ):
        parser.error("Use --parent for capped training, or --evaluate for a frozen final test")
    args.updates = args.updates if args.updates is not None else 1000
    if not args.evaluate and args.output.exists():
        parser.error("Choose a fresh experiment output")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    SpreadsheetAdapter.__init__ = deny
    torch.set_num_threads(1)
    torch.manual_seed(0)
    graph = load_graph("data/processed/graphs-v2/graph-2000")
    calculator = LocalCalculator(load_knowledge_pack())
    calculator.verify()
    try:
        if args.evaluate:
            evaluate(args.output, graph, calculator)
        else:
            train(args, graph, calculator)
    finally:
        calculator.close()


if __name__ == "__main__":
    main()
