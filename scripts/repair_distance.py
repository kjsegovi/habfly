"""Bounded offline ablations; explicit semantic repair can inherit a frozen workflow.

Usage: .venv/bin/python scripts/repair_distance.py OUTPUT --variant structured
Default budget: four training cases, 20 epochs, CPU/one thread, seed 0,
hidden size 16, 80 sequence updates. The explicit 800-update comparison uses
200 epochs. Unseen cases remain sealed.
"""

import argparse
import hashlib
import json
import socket
from pathlib import Path

import torch

from habfly.data import load_graph
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy, TopologyFreePolicy
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint, source_hash
from habfly.training.distance_diagnostic import closed_loop, fit_gate, teacher_forced_stages
from habfly.training.stellar import write_json
from habfly.training.train import prepare_output_directory, train_behavioral_cloning


def deny(*args, **kwargs):
    raise RuntimeError("Repair experiment attempted network or spreadsheet access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("structured", "structured-options", "topology-free-options", "semantic-options"),
    )
    parser.add_argument("--source", type=Path, default=Path("experiments/distance-diagnostic-002"))
    parser.add_argument("--parent", type=Path, help="Required trained checkpoint for semantic-options only")
    parser.add_argument(
        "--optimizer-updates",
        type=int,
        choices=(80, 800),
        default=80,
        help="800 is the separately authorized biological comparison, not an automatic retry",
    )
    args = parser.parse_args()
    if args.optimizer_updates == 800 and args.variant != "structured-options":
        parser.error("The 800-update comparison is restricted to structured-options on the biological graph")
    if (args.variant == "semantic-options") != bool(args.parent):
        parser.error("semantic-options requires --parent; other variants start fresh and reject --parent")
    epochs = args.optimizer_updates // 4
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    SpreadsheetAdapter.__init__ = deny
    torch.set_num_threads(1)
    torch.manual_seed(0)
    graph = load_graph("data/processed/graphs-v2/graph-2000")
    pack = load_knowledge_pack()
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    records = {}
    for split in ("train", "calibration", "development"):
        records[split] = json.loads((args.source / f"{split}.json").read_text())
        if source_hash(records[split]) != source_manifest["content"]["dataset_splits"][split]["sha256"]:
            raise ValueError("Source dataset hash mismatch")
    if len(records["train"]["episodes"]) != 4 or any(len(ep) != 10 for ep in records["train"]["episodes"]):
        raise ValueError("Repair requires exactly four ten-action episodes")
    policy_class = TopologyFreePolicy if args.variant == "topology-free-options" else ConnectomePolicy
    policy = policy_class(
        graph,
        hidden_size=16,
        observation_encoding="structured_tool_v3",
        selection_mode="characters" if args.variant == "structured" else "option_pointer_v1",
    )
    inherited = {}
    parent_identity = None
    if args.parent:
        parent_report = json.loads((args.parent.parent.parent / "report.json").read_text())
        if not parent_report["training_fit_gate_passed"]:
            raise ValueError("The workflow checkpoint must already pass its training-fit gate")
        policy, _ = load_checkpoint(args.parent, graph, content_pack=parent_report["content"])
        if (
            policy.configuration()["architecture"] != "ConnectomePolicy"
            or policy.hidden_size != 16
            or policy.observation_encoding != "structured_tool_v3"
            or policy.selection_mode != "option_pointer_v1"
            or parent_report["content"]["dataset_splits"] != source_manifest["content"]["dataset_splits"]
        ):
            raise ValueError("Incompatible parent policy or source cases")
        policy.selection_mode = "option_pointer_semantic_v2"
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(name.startswith("value_decoder."))
            if not parameter.requires_grad:
                inherited[name] = parameter.detach().clone()
        parent_identity = {
            "checkpoint": str(args.parent),
            "sha256": hashlib.sha256(args.parent.read_bytes()).hexdigest(),
            "trainable_parameters": [n for n, p in policy.named_parameters() if p.requires_grad],
        }
    if (
        policy.graph_hash != source_manifest["graph_hash"]
        or pack.checksum != source_manifest["content"]["knowledge_pack_hash"]
    ):
        raise ValueError("Source graph/knowledge pack mismatch")
    content = {
        **source_manifest["content"],
        "diagnostic": "distance-repair-v3",
        "training_contract": {"sequence_length": 10, **policy.configuration()},
    }
    directory = prepare_output_directory(args.output)
    write_json(
        directory / "manifest.json",
        {
            "content": content,
            "variant": args.variant,
            "source": str(args.source),
            "graph_hash": policy.graph_hash,
            "graph_nodes": len(graph.body_ids),
            "max_optimizer_steps": args.optimizer_updates,
            "max_supervised_decisions": args.optimizer_updates * 10,
            "seed": 0,
            "epochs": epochs,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parent": parent_identity,
            "threads": 1,
            "device": "cpu",
            "unseen_policy": "not_loaded_or_evaluated; separate authorization after selecting a variant",
        },
    )
    write_json(directory / "status.json", {"stage": "training"})
    print(
        f"Training {args.variant}: fixed {args.optimizer_updates} updates / {args.optimizer_updates * 10} decisions",
        flush=True,
    )

    def progress(row):
        write_json(directory / "status.json", {"stage": "training", **row})
        with (directory / "training-progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if row["epoch"] % 10 == 0:
            print(
                f"Epoch {row['epoch']}/{epochs}: loss={row['loss']:.4f}, updates={row['optimizer_steps']}",
                flush=True,
            )

    try:
        validation = [
            ep
            for pair in zip(records["calibration"]["episodes"], records["development"]["episodes"])
            for ep in pair
        ]
        training = train_behavioral_cloning(
            records["train"]["episodes"],
            graph,
            directory / "training",
            validation_episodes=validation,
            epochs=epochs,
            seed=0,
            hidden_size=16,
            content_pack=content,
            policy=policy,
            sequence_length=10,
            progress_callback=progress,
        )
        unchanged = all(
            torch.equal(value, dict(policy.named_parameters())[name]) for name, value in inherited.items()
        )
        if not unchanged:
            raise RuntimeError("Frozen workflow weights changed during option repair")
        restored, _ = load_checkpoint(directory / "training/checkpoint.pt", graph, content_pack=content)
        print("Evaluating training fit and development; unseen remains sealed", flush=True)
        teacher = {
            s: teacher_forced_stages(restored, records[s]["episodes"]) for s in ("train", "development")
        }
        calculator = LocalCalculator(pack)
        try:
            seen = closed_loop(restored, calculator, records["train"]["cases"], directory / "seen", 32)
            development = closed_loop(
                restored, calculator, records["development"]["cases"], directory / "development", 32
            )
        finally:
            calculator.close()
        report = {
            "variant": args.variant,
            "content": content,
            "training": training,
            "teacher_forced": teacher,
            "seen": seen,
            "development": development,
            "training_fit_gate_passed": fit_gate(teacher["train"], seen),
            "stellar_acceptance_gate_passed": False,
            "unseen": {"status": "not_run"},
            "parent": parent_identity,
            "frozen_workflow_weights_unchanged": unchanged if args.parent else None,
        }
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"stage": "complete"})
        print(
            json.dumps(
                {
                    "train_exact": teacher["train"]["exact_action_accuracy"],
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
