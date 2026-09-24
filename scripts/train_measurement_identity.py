"""One bounded offline identity-selector stage; the entire parent workflow is frozen.

Usage: .venv/bin/python scripts/train_measurement_identity.py OUTPUT --parent CHECKPOINT
Fixed budget: 200 updates, batch 12, hidden 16, CPU/one thread, seed 0.
This does not train on additional stellar episodes or open the final-test split.
"""

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import torch

from habfly.data import load_graph
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model.measurement_identity import measurement_request
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint, save_checkpoint, source_hash
from habfly.training.distance_diagnostic import closed_loop, fit_gate, teacher_forced_stages
from habfly.training.measurement_identity import (
    BATCH_SIZE,
    MAX_UPDATES,
    attach_identity_pointer,
    evaluate_identity,
    recognition_records,
    train_identity,
)
from habfly.training.stellar import write_json
from habfly.training.train import peak_process_rss_bytes, prepare_output_directory


def deny(*args, **kwargs):
    raise RuntimeError("Identity experiment attempted network or spreadsheet access")


@torch.no_grad()
def workflow_contexts(policy, episodes):
    """Get source-selection readouts from the parent's four training histories only."""
    contexts = []
    for episode in episodes:
        state = None
        for step in episode:
            output = policy([step["observation"]], state)
            state = output.state
            control = next(c for c in step["observation"]["controls"] if c["id"] == step["action"]["target"])
            if measurement_request(step["observation"], control) is not None:
                contexts.append(output.pooled[0].detach())
    if len(contexts) != 4:
        raise ValueError("Expected exactly four parent training source-selection histories")
    return torch.stack(contexts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("experiments/distance-diagnostic-002"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh identity experiment output; existing artifacts are preserved")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    SpreadsheetAdapter.__init__ = deny
    torch.set_num_threads(1)
    torch.manual_seed(0)
    graph = load_graph("data/processed/graphs-v2/graph-2000")
    pack = load_knowledge_pack()
    manifest = json.loads((args.source / "manifest.json").read_text())
    parent_report = json.loads((args.parent.parent.parent / "report.json").read_text())
    if not parent_report["training_fit_gate_passed"]:
        raise ValueError("Parent must pass the training workflow-fit gate")
    policy, _ = load_checkpoint(args.parent, graph, content_pack=parent_report["content"])
    if (
        type(policy).__name__ != "ConnectomePolicy"
        or policy.hidden_size != 16
        or policy.observation_encoding != "structured_tool_v3"
        or policy.selection_mode != "option_pointer_v1"
        or policy.graph_hash != manifest["graph_hash"]
        or len(graph.body_ids) != 2000
        or pack.checksum != manifest["content"]["knowledge_pack_hash"]
        or parent_report["content"]["dataset_splits"] != manifest["content"]["dataset_splits"]
    ):
        raise ValueError("Incompatible workflow parent, graph, pack or datasets")
    records = {}
    for split in ("train", "development"):
        records[split] = json.loads((args.source / f"{split}.json").read_text())
        if source_hash(records[split]) != manifest["content"]["dataset_splits"][split]["sha256"]:
            raise ValueError("Workflow dataset hash mismatch")
    if len(records["train"]["episodes"]) != 4 or any(len(ep) != 10 for ep in records["train"]["episodes"]):
        raise ValueError("Expected the original four ten-action training cases")
    directory = prepare_output_directory(args.output)
    write_json(directory / "status.json", {"stage": "preparing"})
    try:
        started = time.perf_counter()
        contexts = workflow_contexts(policy, records["train"]["episodes"])
        inherited = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        # Seed only the new selector; never reinitialize the trained workflow.
        torch.manual_seed(0)
        attach_identity_pointer(policy)
        practice = {split: recognition_records(split) for split in ("train", "development")}
        content = {
            **manifest["content"],
            "diagnostic": "measurement-identity-v1",
            "learning_mode": "local_tool_assisted",
            "training_contract": {
                "scope": "new_measurement_identity_parameters_only",
                **policy.configuration(),
            },
            "recognition_splits": {
                split: {"examples": len(rows), "sha256": source_hash(rows)}
                for split, rows in practice.items()
            },
        }
        parent = {"path": str(args.parent), "sha256": hashlib.sha256(args.parent.read_bytes()).hexdigest()}
        for split, rows in practice.items():
            write_json(directory / f"recognition-{split}.json", rows)
        write_json(
            directory / "manifest.json",
            {
                "content": content,
                "parent": parent,
                "source": str(args.source),
                "graph_hash": policy.graph_hash,
                "graph_nodes": 2000,
                "max_optimizer_steps": MAX_UPDATES,
                "batch_size": BATCH_SIZE,
                "seed": 0,
                "device": "cpu",
                "threads": 1,
                "trainable_parameters": [n for n, p in policy.named_parameters() if p.requires_grad],
                "trainable_parameter_count": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "unseen_policy": "not_loaded_or_evaluated",
                "recognition_scope": "syntactic quantity/unit/source matching; includes counterfactual units, not arithmetic cases",
            },
        )
        before = {split: evaluate_identity(policy, rows, contexts) for split, rows in practice.items()}

        def progress(row):
            write_json(directory / "status.json", {"stage": "selector_training", **row})
            with (directory / "training-progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if row["optimizer_steps"] % 25 == 0:
                print(
                    f"Selector update {row['optimizer_steps']}/{MAX_UPDATES}: loss={row['loss']:.5f}",
                    flush=True,
                )

        optimizer, training = train_identity(policy, practice["train"], contexts, progress_callback=progress)
        frozen_unchanged = all(
            torch.equal(value, policy.state_dict()[name]) for name, value in inherited.items()
        )
        if not frozen_unchanged:
            raise RuntimeError("Frozen workflow tensors changed")
        training.update(
            {"seconds": time.perf_counter() - started, "peak_rss_bytes": peak_process_rss_bytes()}
        )
        recognition = {split: evaluate_identity(policy, rows, contexts) for split, rows in practice.items()}
        checkpoint = directory / "training/checkpoint.pt"
        save_checkpoint(
            checkpoint,
            policy,
            stage="measurement_identity",
            seed=0,
            optimizer=optimizer,
            content_pack=content,
        )
        restored, _ = load_checkpoint(checkpoint, graph, content_pack=content)
        if any(
            not torch.equal(value, restored.state_dict()[name]) for name, value in policy.state_dict().items()
        ):
            raise RuntimeError("Checkpoint reload changed weights")
        for split, rows in practice.items():
            if evaluate_identity(restored, rows, contexts) != recognition[split]:
                raise RuntimeError("Checkpoint reload changed recognition predictions")
        write_json(directory / "status.json", {"stage": "workflow_evaluation"})
        print(
            "Selector saved and reloaded. Evaluating the original four train / two development workflows.",
            flush=True,
        )
        teacher = {s: teacher_forced_stages(restored, records[s]["episodes"]) for s in records}
        calculator = LocalCalculator(pack)
        try:
            seen = closed_loop(restored, calculator, records["train"]["cases"], directory / "seen", 32)
            development = closed_loop(
                restored, calculator, records["development"]["cases"], directory / "development", 32
            )
        finally:
            calculator.close()
        report = {
            "content": content,
            "parent": parent,
            "training": training,
            "recognition_before": before,
            "recognition": recognition,
            "teacher_forced": teacher,
            "seen": seen,
            "development": development,
            "frozen_workflow_weights_unchanged": frozen_unchanged,
            "checkpoint_reload_verified": True,
            "training_fit_gate_passed": fit_gate(teacher["train"], seen),
            "stellar_acceptance_gate_passed": False,
            "unseen": {"status": "not_run"},
            "calibration": restored.calibration,
            "total_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_process_rss_bytes(),
        }
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"stage": "complete"})
        print(
            json.dumps(
                {
                    "recognition": recognition["development"]["accuracy"],
                    "seen_completed": seen["completed"],
                    "development_completed": development["completed"],
                }
            ),
            flush=True,
        )
    except Exception as exc:
        write_json(directory / "status.json", {"stage": "failed", "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
