"""Explicitly capped, offline distance-to-luminosity experiments on the real graph.

Train and final-test are separate invocations. Existing outputs are never reused.
"""

import argparse
import hashlib
import itertools
import json
import random
import socket
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from habfly.data import load_graph
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.chained_workflow import workflow_spec
from habfly.training.checkpoints import load_checkpoint, save_checkpoint, source_hash
from habfly.training.frozen_text import frozen_text_cache
from habfly.training.luminosity import ERRORS, clean, rollout
from habfly.training.measurement_identity import evaluate_identity, labels_for
from habfly.training.stellar import record_episode, write_json
from habfly.training.train import (
    calibrate_policy,
    episodes_to_examples,
    evaluate_examples,
    peak_process_rss_bytes,
    supervised_loss,
)


def deny(*args, **kwargs):
    raise RuntimeError("Local luminosity experiment attempted network or Sheets access")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def set_workflow_trainable(policy, mode):
    if mode not in {"workflow", "options"}:
        raise ValueError("Unknown workflow training components")
    prefixes = (
        (
            "tool_state_projection.",
            "control_projection.",
            "option_projection.",
            "option_query.",
            "cell.",
            "norm.",
            "sensory_projection.",
            "feature_projection.",
            "action_head.",
            "target_query.",
            "value_head.",
        )
        if mode == "workflow"
        else ("option_projection.", "option_query.")
    )
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    return prefixes


def cycle_loss_summary(losses, cases):
    """Report all branches, not only the last (often easier) episode in a cycle."""
    start = max(0, len(losses) - len(cases))
    pairs = [(cases[i % len(cases)]["star_class"], losses[i]) for i in range(start, len(losses))]
    return {
        "mean_loss_last_cycle": sum(v for _, v in pairs) / len(pairs),
        "mean_loss_by_class_last_cycle": {
            cls: sum(v for c, v in pairs if c == cls) / sum(c == cls for c, _ in pairs)
            for cls in sorted({c for c, _ in pairs})
        },
    }


def recognition_records(seed, *, task="luminosity"):
    rng, records = random.Random(seed), []
    quantities, units = ("parallax", "flux", "wavelength", "distance"), ("arcsec", "W/m2", "nm", "ly")
    if task in {"temperature", "mass", "radius", "lifetime"}:
        quantities, units = (*quantities, "luminosity"), (*units, "Lsun")
    if task in {"mass", "radius", "lifetime"}:
        quantities, units = (*quantities, "temperature"), (*units, "K")
    if task in {"radius", "lifetime"}:
        quantities, units = (*quantities, "mass"), (*units, "Msun")
    if task == "lifetime":
        quantities, units = (*quantities, "radius"), (*units, "Rsun")
    fields = list(
        itertools.product(
            quantities,
            units,
            ("current star", "reference star"),
        )
    )
    for kind, unit, source in fields:
        candidates = [
            {"kind": k, "unit": u, "source": s, "value": rng.random(), "id": f"{rng.getrandbits(96):x}"}
            for k, u, s in fields
        ]
        rng.shuffle(candidates)
        records.append(
            {
                "instruction": f"Use the {source}'s measurements.",
                "requirement": {"quantity": kind, "unit": unit},
                "candidates": candidates,
                "expected_id": next(
                    c["id"] for c in candidates if (c["kind"], c["unit"], c["source"]) == (kind, unit, source)
                ),
            }
        )
    return records


def expert_episodes(calculator, cases, workflow=None):
    workflow = workflow or workflow_spec("luminosity")
    episodes = []
    for case in cases:
        episode, summary = record_episode(
            workflow.environment(calculator, [case], max_steps=workflow.max_steps), case["seed"]
        )
        if (
            not summary["completed"]
            or summary["steps"] != workflow.expected_steps(case)
            or any(summary.get(k, 0) for k in ERRORS)
        ):
            raise ValueError(f"Expert chain failed: {summary}")
        episodes.append(episode)
    return episodes


def train(args, graph, calculator):
    workflow = workflow_spec(args.task)
    if args.output.exists():
        raise ValueError("Use a fresh experiment directory; prior artifacts are preserved")
    if not 1 <= args.updates <= 800:
        raise ValueError("One invocation is capped at 800 full-sequence optimizer updates")
    if not 0 <= args.recognition_updates <= 200:
        raise ValueError("Recognition is capped at 200 optimizer updates")
    if args.train_components == "options" and args.recognition_updates:
        raise ValueError("Option-only refinement must keep recognition frozen")
    parent_content = read(args.parent.parent.parent / "report.json")["content"]
    parent, _ = load_checkpoint(args.parent, graph, content_pack=parent_content)
    if (
        len(graph.body_ids) != 2000
        or parent.hidden_size != 16
        or calculator.pack.checksum != parent_content["knowledge_pack_hash"]
    ):
        raise ValueError("Incompatible graph, parent or pack")
    torch.manual_seed(0)
    config = parent.configuration()
    config.pop("architecture")
    policy = ConnectomePolicy(
        graph,
        tokenizer=parent.tokenizer,
        **{
            **config,
            "observation_encoding": args.observation_encoding,
            "selection_mode": "measurement_result_v3",
            "control_encoding": args.control_encoding,
        },
    )
    if args.observation_encoding in {"structured_tool_v5", "structured_tool_v6"}:
        from habfly.training.task_state import migrate_task_state

        migrate_task_state(parent, policy)
    else:
        incompatible = policy.load_state_dict(parent.state_dict(), strict=False)
        allowed = ("tool_state_projection.", "option_projection.", "option_query.", "control_projection.")
        if incompatible.unexpected_keys or any(
            not name.startswith(allowed) for name in incompatible.missing_keys
        ):
            raise ValueError(f"Unexpected parent migration: {incompatible}")
    calculator.verify()
    cases = {
        s: workflow.cases(s, n, calculator, offset=args.offset)
        for s, n in (("train", 8), ("calibration", 16), ("development", 16), ("test", 100), ("manual", 12))
    }
    for a, b in itertools.combinations(cases, 2):
        for key in ("seed", "case_id", "instruction"):
            if {c[key] for c in cases[a]} & {c[key] for c in cases[b]}:
                raise ValueError("Dataset split overlap")
        for key in ("parallax", "flux", "wavelength"):
            if {c["inputs"][key] for c in cases[a]} & {c["inputs"][key] for c in cases[b]}:
                raise ValueError("Numeric dataset split overlap")
    args.output.mkdir(parents=True, exist_ok=False)
    content = {
        **calculator.pack.content_identity(),
        "scope": workflow.scope,
        "required_fields": list(workflow.required),
        "max_steps": workflow.max_steps,
        "workflow_splits": {
            s: {"count": len(rows), "sha256": source_hash(rows), "seeds": [c["seed"] for c in rows]}
            for s, rows in cases.items()
        },
    }
    if workflow.applicability_metrics:
        content["required_fields_by_class"] = {
            cls: workflow.required_for(cls) for cls in ("main_sequence", "white_dwarf", "giant")
        }
    parent_hash = file_hash(args.parent)
    write_json(
        args.output / "manifest.json",
        {
            "content": content,
            "parent": {"path": str(args.parent), "sha256": parent_hash},
            "graph_hash": policy.graph_hash,
            "graph_nodes": len(graph.body_ids),
            "graph_edges": len(graph.edge_src),
            "observation_encoding": args.observation_encoding,
            "budget": {
                "recognition_updates": args.recognition_updates,
                "sequence_updates": args.updates,
                "training_cases": 8,
                "seed": 0,
                "cpu_threads": 1,
                "ppo": False,
                "train_components": args.train_components,
            },
            "runner_sha256": file_hash(__file__),
            "test_status": "sealed_until_separate_evaluation",
        },
    )
    for split, rows in cases.items():
        write_json(args.output / f"{split}.json", rows)
    started = time.perf_counter()
    expert_episodes(
        calculator, workflow.cases("train", 100, calculator, offset=10000 + args.offset), workflow
    )
    episodes = {
        s: expert_episodes(calculator, cases[s], workflow) for s in ("train", "calibration", "development")
    }
    for split, rows in episodes.items():
        write_json(args.output / f"{split}-demonstrations.json", rows)
    print(
        "Independent pack validation and 100 expert chains passed; extending quantity/unit recognition",
        flush=True,
    )
    policy.requires_grad_(False)
    for name, parameter in policy.named_parameters():
        if name.startswith("measurement_identity.") and not name.startswith(
            "measurement_identity.source_request."
        ):
            parameter.requires_grad_(True)
    records = recognition_records(workflow.recognition_seed + args.offset, task=args.task)
    recognition_development = recognition_records(
        workflow.recognition_seed + 100000 + args.offset, task=args.task
    )
    write_json(args.output / "recognition-train.json", records)
    write_json(args.output / "recognition-development.json", recognition_development)
    contexts = torch.randn(32, 16).clamp(-1, 1)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.003)
    rng, recognition_losses = random.Random(0), []
    for update in range(args.recognition_updates):
        batch = rng.sample(records, 12)
        pooled = contexts[[rng.randrange(len(contexts)) for _ in batch]]
        logits = torch.stack(policy.measurement_identity(batch, pooled))
        loss = F.cross_entropy(logits, labels_for(batch, policy.device))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite recognition loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        recognition_losses.append(float(loss.detach()))
    recognition = evaluate_identity(policy, recognition_development, contexts)
    if not args.recognition_updates and recognition["accuracy"]["exact"] != 1:
        raise ValueError("Skipping recognition requires a perfect frozen recognition check")
    print(f"Recognition: {recognition['accuracy']}", flush=True)
    trainable = set_workflow_trainable(policy, args.train_components)
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if not parameter.requires_grad
    }
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.003)
    examples = [episodes_to_examples([episode]) for episode in episodes["train"]]
    losses = []
    with frozen_text_cache(policy):
        for update in range(args.updates):
            policy.train()
            state, parts = None, []
            for example in examples[update % len(examples)]:
                loss, output = supervised_loss(policy, [example], state)
                parts.append(loss)
                state = output.state
            loss = torch.stack(parts).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite sequence loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if (update + 1) % 20 == 0:
                row = {
                    "update": update + 1,
                    "cap": args.updates,
                    "loss": losses[-1],
                    **cycle_loss_summary(losses, cases["train"]),
                    "seconds": time.perf_counter() - started,
                }
                write_json(args.output / "status.json", row)
                print(json.dumps(row), flush=True)
        policy.eval()
        teacher = {
            s: evaluate_examples(policy, episodes_to_examples(rows), episode_lengths=[len(e) for e in rows])
            for s, rows in episodes.items()
            if s != "calibration"
        }
    # Calibration is separate, does not change greedy decisions or train weights.
    if any(not torch.equal(value, dict(policy.named_parameters())[name]) for name, value in frozen.items()):
        raise RuntimeError("Frozen parameters changed during workflow fitting")
    write_json(
        args.output / "frozen-workflow-components.json",
        {
            "train_components": args.train_components,
            "trainable_prefixes": list(trainable),
            "unchanged_frozen_parameter_names": list(frozen),
            "frozen_parameters_unchanged": True,
        },
    )
    policy.requires_grad_(False)
    calibrate_policy(
        policy,
        episodes_to_examples(episodes["calibration"]),
        episode_lengths=[len(e) for e in episodes["calibration"]],
    )
    checkpoint = args.output / "training/checkpoint.pt"
    save_checkpoint(
        checkpoint,
        policy,
        stage="_".join(workflow.required),
        seed=0,
        optimizer=optimizer,
        content_pack=content,
    )
    restored, _ = load_checkpoint(checkpoint, graph, content_pack=content)
    if any(not torch.equal(v, restored.state_dict()[k]) for k, v in policy.state_dict().items()):
        raise RuntimeError("Reload changed tensors")
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
    if file_hash(args.parent) != parent_hash:
        raise RuntimeError("Parent checkpoint changed")
    report = {
        "content": content,
        "checkpoint_sha256": file_hash(checkpoint),
        "expert_completed": 100,
        "recognition": recognition,
        "recognition_losses": recognition_losses,
        "losses": losses,
        "teacher": teacher,
        "development": development,
        "seen": seen,
        "calibration": restored.calibration,
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_process_rss_bytes(),
        "development_gate_passed": clean(development, 16, steps=workflow.expected_steps)
        and clean(seen, 8, steps=workflow.expected_steps),
        "final_test_opened": False,
    }
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "development_completed": development["completed"],
                "seen_completed": seen["completed"],
                "teacher": teacher,
                "gate": report["development_gate_passed"],
            }
        ),
        flush=True,
    )


def evaluate(args, graph, calculator):
    workflow = workflow_spec(args.task)
    directory = args.output
    report, manifest = read(directory / "report.json"), read(directory / "manifest.json")
    if not report["development_gate_passed"] or report["content"] != manifest["content"]:
        raise ValueError("Development or dataset identity gate failed")
    if report["content"]["scope"] != workflow.scope or report["content"]["required_fields"] != list(
        workflow.required
    ):
        raise ValueError("Evaluation workflow mismatch")
    checkpoint = directory / "training/checkpoint.pt"
    if (
        file_hash(checkpoint) != report["checkpoint_sha256"]
        or calculator.pack.checksum != report["content"]["knowledge_pack_hash"]
    ):
        raise ValueError("Checkpoint/pack changed")
    policy, _ = load_checkpoint(checkpoint, graph, content_pack=report["content"])
    policy.requires_grad_(False)
    rows = read(directory / "test.json")
    if source_hash(rows) != report["content"]["workflow_splits"]["test"]["sha256"] or len(rows) != 100:
        raise ValueError("Sealed dataset changed")
    final = directory / "final"
    if final.exists():
        raise ValueError("Final evaluation already opened; preserve it and use a new holdout")
    test_identity = report["content"]["workflow_splits"]["test"]
    for prior in directory.parent.glob("*/final/manifest.json"):
        previous = read(prior)
        if set(previous.get("dataset", {}).get("seeds", [])) & set(test_identity["seeds"]):
            raise ValueError(f"Test numeric cases already consumed by {prior}; choose new holdout seeds")
    final.mkdir(exist_ok=False)
    write_json(
        final / "manifest.json",
        {
            "checkpoint_sha256": file_hash(checkpoint),
            "test_consumed": True,
            "optimizer_updates": 0,
            "dataset": report["content"]["workflow_splits"]["test"],
        },
    )
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    started = time.perf_counter()
    result = rollout(policy, calculator, rows, final / "rollouts", environment=workflow.environment)
    unchanged = (
        all(torch.equal(v, policy.state_dict()[k]) for k, v in before.items())
        and file_hash(checkpoint) == report["checkpoint_sha256"]
    )
    passed = (
        unchanged
        and result["completed"] >= 90
        and result["chained"] >= 90
        and len(result["episodes"]) == 100
        and not any(result[k] for k in ERRORS)
    )
    if workflow.applicability_metrics:
        passed = (
            passed
            and all(
                group["requested"] > 0
                and group["completed"] / group["requested"] >= 0.9
                and all(group[key] == group["requested"] for key in workflow.applicability_metrics)
                for group in result["by_class"].values()
            )
            and set(result["by_class"]) == {"main_sequence", "white_dwarf", "giant"}
        )
    final_report = {
        "scope": workflow.scope,
        "chain_gate_passed": passed,
        "stellar_acceptance_gate_passed": False,
        "test_consumed": True,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "optimizer_updates": 0,
        "model_state_unchanged": unchanged,
        "closed_loop": result,
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_process_rss_bytes(),
        "calibration": policy.calibration,
    }
    write_json(final / "report.json", final_report)
    print(
        json.dumps({"completed": result["completed"], "chained": result["chained"], "gate": passed}),
        flush=True,
    )


def main(default_task="luminosity"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("luminosity", "temperature", "mass", "radius", "lifetime"), default=default_task
    )
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--recognition-updates", type=int, default=200)
    parser.add_argument(
        "--train-components",
        choices=("workflow", "options"),
        default="workflow",
        help="Options freezes the recurrent core and all other heads; recognition updates must be zero",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--observation-encoding", choices=("structured_tool_v4", "structured_tool_v5", "structured_tool_v6")
    )
    parser.add_argument(
        "--control-encoding",
        choices=("characters", "semantic_tool_v1"),
        default="semantic_tool_v1",
        help="Use characters to reproduce the archived first-run control-label failure",
    )
    args = parser.parse_args()
    args.parent = args.parent or workflow_spec(args.task).parent
    args.observation_encoding = args.observation_encoding or (
        "structured_tool_v6"
        if args.task == "lifetime"
        else "structured_tool_v5"
        if args.task in {"mass", "radius"}
        else "structured_tool_v4"
    )
    socket.socket = socket.create_connection = deny
    SpreadsheetAdapter.__init__ = deny
    torch.set_num_threads(1)
    torch.manual_seed(0)
    graph = load_graph("data/processed/graphs-v2/graph-2000")
    calculator = LocalCalculator(load_knowledge_pack())
    (train if args.mode == "train" else evaluate)(args, graph, calculator)


if __name__ == "__main__":
    main()
