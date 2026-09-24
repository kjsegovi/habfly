"""One capped source-only experiment, with paired requests and frozen parent weights.

Usage: .venv/bin/python scripts/train_source_request.py OUTPUT --parent CHECKPOINT
Fixed budget: 200 updates, six pairs/batch, hidden 16, CPU/one thread, seed 0.
No automatic retry, full-workflow training, calibration or final-test access.
--curriculum vocabulary-v1 continues a source checkpoint with a fixed 50-update
lexical warm-up and 150 composition updates, including lexical rehearsal.
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
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint, save_checkpoint, source_hash
from habfly.training.distance_diagnostic import closed_loop, fit_gate, teacher_forced_stages
from habfly.training.measurement_identity import evaluate_identity
from habfly.training.source_curriculum import (
    VERSION,
    curriculum_records,
    resume_source_pointer,
    validate_curriculum,
)
from habfly.training.source_request import (
    MAX_UPDATES,
    PAIRS_PER_BATCH,
    attach_source_pointer,
    paired_accuracy,
    paired_records,
    source_probe,
    train_source,
)
from habfly.training.stellar import write_json
from habfly.training.train import peak_process_rss_bytes, prepare_output_directory


def deny(*args, **kwargs):
    raise RuntimeError("Source experiment attempted network or spreadsheet access")


@torch.no_grad()
def frozen_scores(policy, records, contexts):
    scores = []
    for offset in range(0, len(records), 12):
        batch = records[offset : offset + 12]
        pooled = contexts[[(offset + i) % len(contexts) for i in range(len(batch))]]
        scores.extend(
            s.detach().clone() for s in policy.measurement_identity.quantity_unit_scores(batch, pooled)
        )
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--curriculum", choices=("vocabulary-v1",))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh source experiment output; existing artifacts are preserved")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    SpreadsheetAdapter.__init__ = deny
    torch.set_num_threads(1)
    torch.manual_seed(0)
    graph, pack = load_graph("data/processed/graphs-v2/graph-2000"), load_knowledge_pack()
    parent_dir = args.parent.parent.parent
    parent_manifest = json.loads((parent_dir / "manifest.json").read_text())
    parent_report = json.loads((parent_dir / "report.json").read_text())
    policy, _ = load_checkpoint(args.parent, graph, content_pack=parent_report["content"])
    if (
        not parent_report["training_fit_gate_passed"]
        or type(policy).__name__ != "ConnectomePolicy"
        or policy.hidden_size != 16
        or policy.observation_encoding != "structured_tool_v3"
        or policy.selection_mode
        != ("measurement_source_v2" if args.curriculum else "measurement_identity_v1")
        or len(graph.body_ids) != 2000
        or policy.graph_hash != parent_manifest["graph_hash"]
        or pack.checksum != parent_report["content"]["knowledge_pack_hash"]
        or parent_manifest["content"] != parent_report["content"]
    ):
        raise ValueError("Incompatible identity parent, graph, pack or dataset identity")
    if args.curriculum and parent_report["content"]["diagnostic"] != "source-request-v1":
        raise ValueError(
            "Vocabulary stage requires the recorded source-request-v1 experiment, not an automatic resume"
        )
    identity_dir = Path(parent_manifest["parent"]["path"]).parent.parent if args.curriculum else parent_dir
    workflow, legacy = {}, {}
    for split in ("train", "development"):
        workflow[split] = json.loads((Path(parent_manifest["source"]) / f"{split}.json").read_text())
        legacy[split] = json.loads((identity_dir / f"recognition-{split}.json").read_text())
        if (
            source_hash(workflow[split]) != parent_report["content"]["dataset_splits"][split]["sha256"]
            or source_hash(legacy[split]) != parent_report["content"]["recognition_splits"][split]["sha256"]
        ):
            raise ValueError("Parent dataset hash mismatch")
    if (
        len(workflow["train"]["episodes"]) != 4
        or len(workflow["development"]["episodes"]) != 2
        or any(len(ep) != 10 for rows in workflow.values() for ep in rows["episodes"])
    ):
        raise ValueError("Expected original four train and two development ten-action workflows")
    directory = prepare_output_directory(args.output)
    write_json(directory / "status.json", {"stage": "preparing"})
    started = time.perf_counter()
    try:
        contexts = workflow_contexts(policy, workflow["train"]["episodes"])
        curriculum = None
        if args.curriculum:
            curriculum = {
                s: curriculum_records(s) for s in ("vocabulary", "train", "development", "synonym_diagnostic")
            }
            coverage = validate_curriculum(curriculum)
            practice = {s: curriculum[s] for s in ("train", "development")}
        else:
            practice = {split: paired_records(split) for split in ("train", "development")}
        datasets = {
            **{f"paired_{s}": r for s, r in practice.items()},
            **{f"legacy_{s}": r for s, r in legacy.items()},
        }
        if curriculum:
            datasets.update({s: curriculum[s] for s in ("vocabulary", "synonym_diagnostic")})
            for split in ("train", "development"):
                rows = json.loads((parent_dir / f"paired-{split}.json").read_text())
                if source_hash(rows) != parent_report["content"]["source_request_splits"][split]["sha256"]:
                    raise ValueError("Previous paired dataset hash mismatch")
                datasets[f"previous_paired_{split}"] = rows
            # Old failed development remains a consumed regression set, never a fresh holdout.
            if any(
                {r["instruction"] for r in practice["development"]} & {r["instruction"] for r in datasets[s]}
                for s in (
                    "legacy_train",
                    "legacy_development",
                    "previous_paired_train",
                    "previous_paired_development",
                )
            ):
                raise ValueError("Fresh composition holdout overlaps prior instructions")
        before = {
            split: evaluate_identity(policy, rows, contexts)
            for split, rows in datasets.items()
            if split != "synonym_diagnostic"
        }
        expected_scores = {split: frozen_scores(policy, rows, contexts) for split, rows in datasets.items()}
        inherited = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
            if not (args.curriculum and name.startswith("measurement_identity.source_request."))
        }
        preflight = {"parent": source_probe(policy, practice["train"][:2], contexts[0])}
        torch.manual_seed(0)
        if args.curriculum:
            resume_source_pointer(policy)
        else:
            attach_source_pointer(policy)
        probe_key = "continued_source" if args.curriculum else "new_source"
        preflight[probe_key] = source_probe(policy, practice["train"][:2], contexts[0])
        if (
            not preflight[probe_key]["finite_gradient"]
            or preflight[probe_key]["embedding_gradient_norm"] <= 0
            or preflight[probe_key]["max_logit_change"] <= 0
        ):
            raise RuntimeError("Source instructions or gradients did not reach the new scorer")
        content = {
            **parent_report["content"],
            "diagnostic": VERSION if args.curriculum else "source-request-v1",
            "training_contract": {
                "scope": "existing_source_request_parameters_only"
                if args.curriculum
                else "new_source_request_parameters_only",
                **policy.configuration(),
            },
            "source_request_splits": {
                s: {"examples": len(rows), "pairs": len(rows) // 2, "sha256": source_hash(rows)}
                for s, rows in practice.items()
            },
        }
        if curriculum:
            content["curriculum"] = {
                **coverage,
                "warmup_updates": 50,
                "composition_updates": 150,
                "rehearsal_pairs": 1,
                "optimizer": "fresh AdamW; inherit source weights without resetting them",
                "additional_splits": {
                    s: {"examples": len(curriculum[s]), "sha256": source_hash(curriculum[s])}
                    for s in ("vocabulary", "synonym_diagnostic")
                },
                "previous_development_scope": "already_consumed_regression_only; original failure remains recorded",
            }
        parent = {"path": str(args.parent), "sha256": hashlib.sha256(args.parent.read_bytes()).hexdigest()}
        for split, rows in practice.items():
            write_json(directory / f"paired-{split}.json", rows)
        if curriculum:
            for split in ("vocabulary", "synonym_diagnostic"):
                write_json(directory / f"{split}.json", curriculum[split])
        write_json(directory / "preflight.json", preflight)
        write_json(
            directory / "manifest.json",
            {
                "content": content,
                "parent": parent,
                "source": parent_manifest["source"],
                "graph_hash": policy.graph_hash,
                "graph_nodes": 2000,
                "seed": 0,
                "device": "cpu",
                "threads": 1,
                "max_optimizer_steps": MAX_UPDATES,
                "pairs_per_batch": PAIRS_PER_BATCH,
                "trainable_parameters": [n for n, p in policy.named_parameters() if p.requires_grad],
                "trainable_parameter_count": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "context_helper_sha256": hashlib.sha256(
                    Path(__file__).with_name("train_measurement_identity.py").read_bytes()
                ).hexdigest(),
                "unseen_policy": "not_loaded_or_evaluated; no automatic promotion",
                "gate": {
                    "development_source_minimum": 0.95,
                    "development_paired_both_correct_minimum": 0.95,
                    "quantity_unit_accuracy": 1.0,
                    "workflow_completed": 6,
                    "workflow_max_steps": 10,
                    "invalid_actions_and_errors": 0,
                    "frozen_parent_tensors_and_quantity_unit_scores_unchanged": True,
                    "vocabulary_source_minimum": 0.95 if args.curriculum else None,
                    "synonym_diagnostic_is_gate": False,
                },
                "recognition_scope": "syntactic field recognition with counterfactual units; not additional stellar episodes",
                "curriculum": content.get("curriculum"),
                "frozen_scope": "all except measurement_identity.source_request.*"
                if args.curriculum
                else "all parent tensors",
            },
        )

        def progress(row):
            write_json(directory / "status.json", {"stage": "source_training", **row})
            with (directory / "training-progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if row["optimizer_steps"] % 25 == 0:
                print(
                    f"Source update {row['optimizer_steps']}/{MAX_UPDATES}: loss={row['loss']:.5f}",
                    flush=True,
                )

        training_started = time.perf_counter()
        schedule = (
            {"warmup_records": curriculum["vocabulary"], "warmup_updates": 50, "rehearsal_pairs": 1}
            if curriculum
            else {}
        )
        optimizer, training = train_source(
            policy, practice["train"], contexts, progress_callback=progress, **schedule
        )
        training["seconds"] = time.perf_counter() - training_started
        unchanged = all(torch.equal(value, policy.state_dict()[name]) for name, value in inherited.items())
        scores_unchanged = all(
            all(
                torch.equal(a, b)
                for a, b in zip(expected_scores[split], frozen_scores(policy, rows, contexts))
            )
            for split, rows in datasets.items()
        )
        if not unchanged or not scores_unchanged:
            raise RuntimeError("Frozen parent tensors or quantity/unit scores changed")
        recognition = {
            s: evaluate_identity(policy, rows, contexts)
            for s, rows in datasets.items()
            if s != "synonym_diagnostic"
        }
        paired = {s: paired_accuracy(policy, rows, contexts) for s, rows in practice.items()}
        checkpoint = directory / "training/checkpoint.pt"
        save_checkpoint(
            checkpoint,
            policy,
            stage="source_vocabulary" if curriculum else "source_request",
            seed=0,
            optimizer=optimizer,
            content_pack=content,
        )
        restored, _ = load_checkpoint(checkpoint, graph, content_pack=content)
        if any(not torch.equal(v, restored.state_dict()[n]) for n, v in policy.state_dict().items()):
            raise RuntimeError("Checkpoint reload changed weights")
        # The unseen-synonym diagnostic runs only after the single checkpoint is
        # fixed and reloaded; it cannot steer updates or checkpoint selection.
        if curriculum:
            recognition["synonym_diagnostic"] = evaluate_identity(
                restored, datasets["synonym_diagnostic"], contexts
            )
        for split, rows in datasets.items():
            if evaluate_identity(restored, rows, contexts) != recognition[split]:
                raise RuntimeError("Checkpoint reload changed predictions")
        write_json(directory / "status.json", {"stage": "workflow_evaluation"})
        print(
            "Source checkpoint reloaded; evaluating the six existing workflows. Final tests stay sealed.",
            flush=True,
        )
        teacher = {s: teacher_forced_stages(restored, rows["episodes"]) for s, rows in workflow.items()}
        calculator = LocalCalculator(pack)
        try:
            seen = closed_loop(restored, calculator, workflow["train"]["cases"], directory / "seen", 32)
            development = closed_loop(
                restored, calculator, workflow["development"]["cases"], directory / "development", 32
            )
        finally:
            calculator.close()
        gates = {
            "development_source": all(
                recognition[s]["accuracy"]["source"] >= 0.95
                for s in ("paired_development", "legacy_development")
            ),
            "paired_switch": paired["development"]["accuracy"] >= 0.95,
            "quantity_unit": all(
                r["accuracy"][k] == 1.0 for r in recognition.values() for k in ("kind", "unit")
            ),
            "workflow": seen["completed"] == 4
            and development["completed"] == 2
            and all(e["steps"] <= 10 for group in (seen, development) for e in group["episodes"])
            and all(
                group[key] == 0
                for group in (seen, development)
                for key in ("invalid_actions", "tool_errors", "infrastructure_failures", "api_failures")
            ),
            "frozen_parent": unchanged and scores_unchanged,
        }
        if curriculum:
            gates["vocabulary_grounding"] = recognition["vocabulary"]["accuracy"]["source"] >= 0.95
        report = {
            "content": content,
            "parent": parent,
            "preflight": preflight,
            "training": training,
            "recognition_before": before,
            "recognition": recognition,
            "paired": paired,
            "teacher_forced": teacher,
            "seen": seen,
            "development": development,
            "frozen_parent_weights_unchanged": unchanged,
            "quantity_unit_scores_unchanged": scores_unchanged,
            "checkpoint_reload_verified": True,
            "gates": gates,
            "source_gate_passed": all(gates.values()),
            "training_fit_gate_passed": fit_gate(teacher["train"], seen),
            "stellar_acceptance_gate_passed": False,
            "unseen": {"status": "not_run"},
            "calibration": restored.calibration,
            "total_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_process_rss_bytes(),
            "gate_scope": "known-vocabulary composition plus preserved workflows"
            if curriculum
            else "source-request-v1",
        }
        if curriculum:
            report["previous_development_regression"] = {
                "scope": "already_consumed; not a new holdout and does not replace the old report",
                "source_accuracy": recognition["previous_paired_development"]["accuracy"]["source"],
                "paired": paired_accuracy(restored, datasets["previous_paired_development"], contexts),
            }
            report["synonym_diagnostic_is_gate"] = False
        write_json(directory / "report.json", report)
        write_json(
            directory / "status.json", {"stage": "complete", "source_gate_passed": all(gates.values())}
        )
        print(
            json.dumps(
                {
                    "development": recognition["paired_development"]["accuracy"],
                    "gates": gates,
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
