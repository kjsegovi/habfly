"""One capped source-only repair, followed by a separately invoked fresh final test."""

import argparse
import json
import socket
import time
from pathlib import Path

import torch
from train_luminosity import deny, evaluate, expert_episodes, file_hash, read

from habfly.data import load_graph
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model.measurement_identity import measurement_request
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.chained_workflow import workflow_spec
from habfly.training.checkpoints import load_checkpoint, save_checkpoint, source_hash
from habfly.training.luminosity import clean, rollout
from habfly.training.measurement_identity import evaluate_identity
from habfly.training.source_aliases import alias_records, rehearsal_records
from habfly.training.source_request import paired_accuracy, train_source
from habfly.training.stellar import write_json
from habfly.training.temperature_source import (
    VERSION,
    composition_records,
    source_records,
    validate_separation,
    workflow_cases,
)
from habfly.training.train import calibrate_policy, episodes_to_examples, peak_process_rss_bytes


@torch.no_grad()
def training_contexts(policy, episodes):
    contexts = []
    for episode in episodes:
        state = None
        for step in episode:
            output = policy([step["observation"]], state)
            state = output.state
            control = next(c for c in step["observation"]["controls"] if c["id"] == step["action"]["target"])
            if measurement_request(step["observation"], control, include_results=True) is not None:
                contexts.append(output.pooled[0].detach())
    if len(contexts) != 32:
        raise ValueError("Expected four source choices in each of eight training episodes")
    return torch.stack(contexts)


def train(args, graph, calculator):
    if args.output.exists() or not 1 <= args.updates <= 1000:
        raise ValueError("Use fresh output and an explicit source-only cap of at most 1000 updates")
    if args.composition and args.updates <= 100:
        raise ValueError("Composition requires more than the fixed 100-update warm-up")
    if args.punctuation and not args.composition:
        raise ValueError("Punctuation practice requires composition mode")
    workflow = workflow_spec("temperature")
    parent_directory = args.parent.parent.parent
    parent_report = read(parent_directory / "report.json")
    policy, _ = load_checkpoint(args.parent, graph, content_pack=parent_report["content"])
    if (
        len(graph.body_ids) != 2000
        or policy.hidden_size != 16
        or policy.selection_mode != "measurement_result_v3"
        or parent_report["content"]["scope"] != workflow.scope
        or calculator.pack.checksum != parent_report["content"]["knowledge_pack_hash"]
    ):
        raise ValueError("Incompatible temperature parent")
    calculator.verify()
    parent_hash = file_hash(args.parent)
    policy.requires_grad_(False)
    # Use only the parent's recorded training histories, never failed-test trajectories.
    context_report = read(args.context_dataset / "report.json")
    context_cases = read(args.context_dataset / "train.json")
    if source_hash(context_cases) != context_report["content"]["workflow_splits"]["train"]["sha256"]:
        raise ValueError("Recorded workflow context cases changed")
    context_episodes = read(args.context_dataset / "train-demonstrations.json")
    contexts = training_contexts(policy, context_episodes)
    cases = {
        s: workflow_cases(s, n, calculator)
        for s, n in (("train", 8), ("calibration", 16), ("development", 16), ("test", 100), ("manual", 12))
    }
    recognition = {s: source_records(s) for s in ("train", "development")}
    rehearsal = alias_records("train") + rehearsal_records()
    if args.composition:
        rehearsal += recognition["train"]
        recognition["train"] = composition_records(punctuation=args.punctuation)
    validate_separation(cases, recognition)
    if {r["instruction"] for r in rehearsal} & {r["instruction"] for r in recognition["development"]}:
        raise ValueError("Source rehearsal overlaps development")
    content = {
        **calculator.pack.content_identity(),
        "scope": workflow.scope,
        "required_fields": list(workflow.required),
        "max_steps": workflow.max_steps,
        "source_curriculum": VERSION
        + ("-composition" if args.composition else "")
        + ("-punctuation" if args.punctuation else ""),
        "context_dataset": {
            "path": str(args.context_dataset),
            "training_demonstrations_sha256": source_hash(context_episodes),
        },
        "workflow_splits": {
            s: {"count": len(rows), "sha256": source_hash(rows), "seeds": [c["seed"] for c in rows]}
            for s, rows in cases.items()
        },
        "source_splits": {
            s: {"count": len(rows), "sha256": source_hash(rows)} for s, rows in recognition.items()
        },
        "rehearsal": {
            "count": len(rehearsal),
            "sha256": source_hash(rehearsal),
            "status": "previously_consumed",
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in cases.items():
        write_json(args.output / f"{split}.json", rows)
    for split, rows in recognition.items():
        write_json(args.output / f"recognition-{split}.json", rows)
    write_json(args.output / "rehearsal.json", rehearsal)
    write_json(
        args.output / "manifest.json",
        {
            "content": content,
            "parent": {"path": str(args.parent), "sha256": parent_hash},
            "graph_hash": policy.graph_hash,
            "graph_nodes": 2000,
            "graph_edges": len(graph.edge_src),
            "budget": {
                "source_updates": args.updates,
                "workflow_updates": 0,
                "batch": 12,
                "seed": 0,
                "threads": 1,
                "device": "cpu",
            },
            "consumed_wording": "temperature-001 train, development, calibration and failed final-test templates are explicit language rehearsal; no failed-test numeric cases or trajectories train this model",
            "runner_sha256": file_hash(__file__),
            "test_status": "sealed_until_separate_evaluation",
        },
    )
    started = time.perf_counter()
    before = evaluate_identity(policy, recognition["development"], contexts)
    frozen = {
        n: v.detach().clone()
        for n, v in policy.state_dict().items()
        if not n.startswith("measurement_identity.source_request.")
    }
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith("measurement_identity.source_request."))
    policy.calibration = {"status": "uncalibrated", "reason": "source_only_repair"}
    policy.action_temperature = policy.target_temperature = 1.0

    def progress(row):
        write_json(args.output / "status.json", row)
        if row["optimizer_steps"] % 100 == 0:
            print(json.dumps(row), flush=True)

    schedule = (
        {"warmup_records": rehearsal, "warmup_updates": 100, "rehearsal_pairs": 2} if args.composition else {}
    )
    optimizer, training = train_source(
        policy,
        recognition["train"] if args.composition else recognition["train"] + rehearsal,
        contexts,
        updates=args.updates,
        budget_limit=1000,
        progress_callback=progress,
        **schedule,
    )
    unchanged = all(torch.equal(v, policy.state_dict()[n]) for n, v in frozen.items())
    if not unchanged or file_hash(args.parent) != parent_hash:
        raise RuntimeError("Frozen workflow, quantity/unit tensors or parent file changed")
    identity = {
        s: evaluate_identity(policy, rows, contexts)
        for s, rows in {**recognition, "rehearsal": rehearsal}.items()
    }
    paired = paired_accuracy(policy, recognition["development"], contexts)
    policy.requires_grad_(False)
    calibration_episodes = expert_episodes(calculator, cases["calibration"], workflow)
    write_json(args.output / "calibration-demonstrations.json", calibration_episodes)
    calibrate_policy(
        policy, episodes_to_examples(calibration_episodes), episode_lengths=[workflow.steps] * 16
    )
    expert_episodes(calculator, workflow_cases("train", 100, calculator), workflow)
    checkpoint = args.output / "training/checkpoint.pt"
    save_checkpoint(checkpoint, policy, stage=VERSION, seed=0, optimizer=optimizer, content_pack=content)
    restored, _ = load_checkpoint(checkpoint, graph, content_pack=content)
    if any(not torch.equal(v, restored.state_dict()[n]) for n, v in policy.state_dict().items()):
        raise RuntimeError("Reload changed weights")
    development = rollout(
        restored,
        calculator,
        cases["development"],
        args.output / "development",
        environment=workflow.environment,
    )
    seen = rollout(
        restored, calculator, cases["train"], args.output / "seen", environment=workflow.environment
    )
    old_cases = context_cases
    retained = rollout(
        restored, calculator, old_cases, args.output / "retained", environment=workflow.environment
    )
    gate = (
        clean(development, 16, steps=31)
        and clean(seen, 8, steps=31)
        and clean(retained, 8, steps=31)
        and identity["development"]["accuracy"]["exact"] >= 0.98
        and paired["accuracy"] >= 0.95
        and identity["rehearsal"]["accuracy"]["exact"] >= 0.98
    )
    report = {
        "content": content,
        "checkpoint_sha256": file_hash(checkpoint),
        "training": training,
        "before": before,
        "recognition": identity,
        "paired": paired,
        "expert_completed": 100,
        "frozen_parent_unchanged": unchanged,
        "development": development,
        "seen": seen,
        "retained": retained,
        "calibration": restored.calibration,
        "development_gate_passed": gate,
        "final_test_opened": False,
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_process_rss_bytes(),
    }
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "completed": development["completed"],
                "seen": seen["completed"],
                "retained": retained["completed"],
                "source": identity["development"]["accuracy"],
                "paired": paired["accuracy"],
                "rehearsal": identity["rehearsal"]["accuracy"],
                "gate": gate,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parent", type=Path, default=Path("experiments/temperature-001/training/checkpoint.pt")
    )
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--composition", action="store_true")
    parser.add_argument("--punctuation", action="store_true")
    parser.add_argument("--context-dataset", type=Path, default=Path("experiments/temperature-001"))
    args = parser.parse_args()
    args.task = "temperature"
    socket.socket = socket.create_connection = deny
    SpreadsheetAdapter.__init__ = deny
    torch.set_num_threads(1)
    torch.manual_seed(0)
    graph = load_graph("data/processed/graphs-v2/graph-2000")
    calculator = LocalCalculator(load_knowledge_pack())
    (train if args.mode == "train" else evaluate)(args, graph, calculator)


if __name__ == "__main__":
    main()
