"""Explicit whole-episode temperature diagnostic, capped at 80 new updates."""

import itertools
import math
import time
from pathlib import Path

import torch

from habfly.environments.habitability_calculations import habitability_cases
from habfly.habitability_knowledge import HabitabilityCalculator

from .checkpoints import load_checkpoint, source_hash
from .frozen_text import frozen_text_cache
from .habitability_calculations import BUDGET, COUNTS, ENCODING, collect_split, content_identity
from .planet_calculations import ERRORS, summarize
from .planet_diagnostics import diagnose_trajectories
from .planet_sequence import file_hash, read
from .stellar import write_json
from .train import (
    episodes_to_examples,
    evaluate_examples,
    peak_process_rss_bytes,
    prepare_output_directory,
    train_behavioral_cloning,
)

FOLLOWUP_BUDGET = {
    "epochs": 20,
    "additional_optimizer_updates": 80,
    "sequence_length": 28,
    "new_cases": 0,
    "cpu_threads": 1,
    "seed": 0,
    "hidden_size": 16,
    "frozen_parameters": ["embedding", "text_encoder"],
    "optimizer_state": "fresh_AdamW_moments_for_explicit_followup",
    "ppo": False,
    "final_test_episodes": 0,
}
RESUME_BUDGET = {**FOLLOWUP_BUDGET, "optimizer_state": "resume_parent_AdamW_moments_same_stage"}
PILOT_COUNTS = {"gate": 100, "train": 64, "calibration": 16, "development": 16}
PILOT_BUDGET = {
    **RESUME_BUDGET,
    "epochs": 5,
    "additional_optimizer_updates": 320,
    "new_cases": 88,
    "new_training_cases": 60,
}


def require_diagnostic_gate(report, source=None):
    for split, expected in (("train", 4), ("development", 2)):
        row = report.get("closed_loop", {}).get(split, {})
        if (
            row.get("episodes") != expected
            or row.get("completed") != expected
            or row.get("numeric_answer_accuracy") != 1.0
            or any(row.get(key) != 0 for key in ERRORS)
        ):
            raise ValueError("Temperature pilot requires the completed four/two-case diagnostic")
        if (
            source is not None
            and summarize(read(Path(source) / f"learned-{split}/summaries.json"), required_count=2) != row
        ):
            raise ValueError("Temperature diagnostic summary artifact mismatch")


def diagnose_followup(source, graph, output, *, progress=lambda _: None):
    """Frozen counterfactuals on the same smoke train/development demonstrations."""
    source = Path(source)
    manifest = read(source / "manifest.json")
    parent = Path(manifest["source_dataset"])
    _, parent_content, _, episodes, _ = validated_smoke(parent, graph, HabitabilityCalculator())
    content = manifest["content"]
    lineage = content["sequence_diagnostic"]
    if (
        manifest.get("budget") not in (FOLLOWUP_BUDGET, RESUME_BUDGET)
        or {k: v for k, v in content.items() if k != "sequence_diagnostic"} != parent_content
        or lineage.get("dataset_checkpoint_sha256", lineage["parent_checkpoint_sha256"])
        != file_hash(parent / "training/checkpoint.pt")
        or manifest.get("parent_checkpoint") is not None
        and file_hash(manifest["parent_checkpoint"]) != lineage["parent_checkpoint_sha256"]
    ):
        raise ValueError("Temperature diagnostic parent mismatch")
    checkpoint = source / "training/checkpoint.pt"
    checksum = file_hash(checkpoint)
    policy, _ = load_checkpoint(checkpoint, graph, content_pack=content)
    torch.set_num_threads(1)
    directory = prepare_output_directory(output)
    report = {
        "content": content,
        "checkpoint_sha256": checksum,
        "optimizer_updates": 0,
        "final_test_episodes": 0,
        "splits": {},
    }
    for split in ("train", "development"):
        report["splits"][split] = diagnose_trajectories(policy, episodes[split])
        progress({"split": split, **report["splits"][split]})
    if file_hash(checkpoint) != checksum:
        raise ValueError("Frozen temperature diagnostic changed checkpoint")
    write_json(directory / "report.json", report)
    return report


def validated_smoke(source, graph, calculator):
    source = Path(source)
    manifest, report = read(source / "manifest.json"), read(source / "report.json")
    if (
        manifest.get("budget") != BUDGET
        or manifest.get("counts") != COUNTS
        or report.get("content") != manifest.get("content")
        or report.get("checkpoint_reload_verified") is not True
        or report.get("training", {}).get("optimizer_steps") != 32
        or report.get("final_test_episodes") != 0
    ):
        raise ValueError("Incompatible temperature smoke parent")
    cases = {s: read(source / f"private-{s}-cases.json") for s in COUNTS}
    if {s: len(rows) for s, rows in cases.items()} != COUNTS:
        raise ValueError("Temperature smoke case count mismatch")
    content = content_identity(calculator, cases, graph)
    if content != manifest["content"]:
        raise ValueError("Temperature smoke content, graph, pack or split hash mismatch")
    episodes = {
        s: read(source / f"expert-{s}/episodes.json") for s in ("train", "calibration", "development")
    }
    if any(len(episodes[s]) != COUNTS[s] or any(len(ep) != 28 for ep in episodes[s]) for s in episodes):
        raise ValueError("Temperature smoke trajectory length mismatch")
    validation = list(itertools.chain.from_iterable(zip(episodes["calibration"], episodes["development"])))
    if source_hash(episodes["train"]) != report["training"].get("training_data_hash") or source_hash(
        validation
    ) != report["training"].get("validation_data_hash"):
        raise ValueError("Temperature smoke demonstration hash mismatch")
    policy, saved = load_checkpoint(source / "training/checkpoint.pt", graph, content_pack=content)
    if policy.observation_encoding != ENCODING or policy.hidden_size != 16:
        raise ValueError("Temperature smoke model contract mismatch")
    if any(
        saved["evaluation_scores"].get(k) != report["training"].get(k)
        for k in ("training_data_hash", "validation_data_hash", "optimizer_steps")
    ):
        raise ValueError("Temperature smoke checkpoint dataset mismatch")
    return policy, content, cases, episodes, validation


def validate_followup_parent(source, graph, episodes, validation):
    """Same-task refinement cannot silently switch data, encoding or checkpoint."""
    source = Path(source)
    manifest, report = read(source / "manifest.json"), read(source / "report.json")
    if (
        manifest.get("budget") not in (FOLLOWUP_BUDGET, RESUME_BUDGET)
        or report.get("budget") != manifest.get("budget")
        or report.get("content") != manifest.get("content")
        or report.get("checkpoint_reload_verified") is not True
        or report.get("final_test_episodes") != 0
        or report.get("training", {}).get("optimizer_steps") != 80
        or report["training"].get("training_data_hash") != source_hash(episodes["train"])
        or report["training"].get("validation_data_hash") != source_hash(validation)
    ):
        raise ValueError("Incompatible temperature refinement parent")
    checkpoint = source / "training/checkpoint.pt"
    policy, saved = load_checkpoint(checkpoint, graph, content_pack=manifest["content"])
    if (
        policy.observation_encoding != ENCODING
        or policy.hidden_size != 16
        or any(
            saved["evaluation_scores"].get(k) != report["training"].get(k)
            for k in ("training_data_hash", "validation_data_hash", "optimizer_steps")
        )
    ):
        raise ValueError("Temperature refinement checkpoint mismatch")
    optimizer = torch.load(checkpoint, map_location="cpu", weights_only=True).get("optimizer")
    if not optimizer:
        raise ValueError("Temperature refinement requires saved optimizer moments")
    step = max((int(row.get("step", 0)) for row in optimizer["state"].values()), default=0)
    expected_step = report["training"].get("optimizer_initial_step", 0) + 80
    if step != expected_step:
        raise ValueError("Temperature refinement optimizer step mismatch")
    return policy, optimizer, manifest["content"]


def run_sequence_followup(
    source, graph, output, *, progress=lambda _: None, planet_parent=None, resume_from=None, pilot=False
):
    if pilot and resume_from is None:
        raise ValueError("Temperature pilot requires explicit same-task optimizer continuation")
    if planet_parent is not None and resume_from is not None:
        raise ValueError("Choose explicit transfer or same-task refinement, not both")
    if len(graph.body_ids) != 2000 or graph.manifest.get("source_kind") != "biological":
        raise ValueError("Temperature sequence diagnostic requires the real 2,000-node graph")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    started = time.perf_counter()
    calculator = HabitabilityCalculator()
    verification = calculator.verify()
    policy, parent_content, cases, episodes, validation = validated_smoke(source, graph, calculator)
    parent_path = Path(source) / "training/checkpoint.pt"
    dataset_parent_hash = file_hash(parent_path)
    transfer = None
    optimizer_state = None
    inherited = 32
    budget = PILOT_BUDGET if pilot else RESUME_BUDGET if resume_from else FOLLOWUP_BUDGET
    if resume_from is not None:
        policy, optimizer_state, prior_content = validate_followup_parent(
            resume_from, graph, episodes, validation
        )
        if {k: v for k, v in prior_content.items() if k != "sequence_diagnostic"} != parent_content:
            raise ValueError("Temperature refinement changed original case scope")
        prior = prior_content["sequence_diagnostic"]
        if prior.get("dataset_checkpoint_sha256", prior["parent_checkpoint_sha256"]) != dataset_parent_hash:
            raise ValueError("Temperature refinement smoke lineage mismatch")
        inherited = (prior.get("inherited_optimizer_updates") or 0) + 80
        parent_path = Path(resume_from) / "training/checkpoint.pt"
        if pilot:
            require_diagnostic_gate(read(Path(resume_from) / "report.json"), resume_from)
            cases = {s: habitability_cases(s, n) for s, n in PILOT_COUNTS.items()}
            parent_content = content_identity(calculator, cases, graph)
    if planet_parent is not None:
        from .habitability_transfer import transfer_planet_policy

        parent_path = Path(planet_parent)
        metadata = read(parent_path.with_suffix(".pt.json"))
        planet_content = metadata.get("provenance", {}).get("planet_calculations")
        if not planet_content or planet_content.get("calculation_backend") != "local":
            raise ValueError("Temperature transfer requires a declared local planet parent")
        parent, _ = load_checkpoint(parent_path, graph, content_pack=planet_content)
        policy, transfer = transfer_planet_policy(parent)
    parent_hash = file_hash(parent_path)
    content = {
        **parent_content,
        "sequence_diagnostic": {
            "version": 1,
            "parent_checkpoint_sha256": parent_hash,
            "dataset_checkpoint_sha256": dataset_parent_hash,
            "inherited_optimizer_updates": None if transfer else inherited,
            "initialization": "same_task_optimizer_resume"
            if resume_from
            else "explicit_planet_transfer"
            if transfer
            else "temperature_smoke_checkpoint",
            "transfer": transfer,
            "budget": budget,
            "checkpoint_selection": "last_fixed_budget_epoch_no_development_selection",
        },
    }
    directory = prepare_output_directory(output)
    write_json(
        directory / "manifest.json",
        {
            "version": 1,
            "content": content,
            "verification": verification,
            "source_dataset": str(Path(source).resolve()),
            "parent_checkpoint": str(parent_path.resolve()),
            "budget": budget,
        },
    )
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(not name.startswith(("embedding.", "text_encoder.")))
    frozen = {name: p.detach().clone() for name, p in policy.named_parameters() if not p.requires_grad}

    def update(row):
        write_json(
            directory / "status.json",
            {"status": "training", "additional_update_cap": budget["additional_optimizer_updates"], **row},
        )
        progress(row)

    update({"phase": "whole_episode_diagnostic"})
    try:
        if pilot:
            episodes = {}
            for split, rows in cases.items():
                episodes[split], summaries = collect_split(calculator, rows, directory / f"expert-{split}")
                write_json(directory / f"private-{split}-cases.json", rows)
                progress({"phase": "expert", "split": split, "completed": len(summaries)})
            validation = list(
                itertools.chain.from_iterable(
                    zip(episodes["calibration"], episodes["development"], strict=True)
                )
            )
        with frozen_text_cache(policy):
            training = train_behavioral_cloning(
                episodes["train"],
                graph,
                directory / "training",
                policy=policy,
                validation_episodes=validation,
                epochs=budget["epochs"],
                seed=0,
                hidden_size=16,
                sequence_length=28,
                content_pack=content,
                progress_callback=update,
                optimizer_state=optimizer_state,
            )
            teacher = {
                split: evaluate_examples(
                    policy, episodes_to_examples(rows), episode_lengths=[len(ep) for ep in rows]
                )
                for split, rows in episodes.items()
                if split in ("train", "development")
            }
        if training["optimizer_steps"] != budget["additional_optimizer_updates"] or not all(
            map(math.isfinite, training["losses"])
        ):
            raise ValueError("Temperature sequence budget or finite-loss failure")
        if any(not torch.equal(v, dict(policy.named_parameters())[k]) for k, v in frozen.items()):
            raise ValueError("Frozen temperature text parameters changed")
        restored, _ = load_checkpoint(directory / "training/checkpoint.pt", graph, content_pack=content)
        if any(not torch.equal(v, restored.state_dict()[k]) for k, v in policy.state_dict().items()):
            raise ValueError("Temperature sequence checkpoint reload changed weights")
        write_json(directory / "status.json", {"status": "closed_loop_evaluation"})
        evaluations = {}
        for split in ("train", "development"):
            _, rows = collect_split(calculator, cases[split], directory / f"learned-{split}", policy=restored)
            evaluations[split] = summarize(rows, required_count=2)
            progress({"phase": "closed_loop", "split": split, **evaluations[split]})
        if file_hash(parent_path) != parent_hash:
            raise ValueError("Parent temperature checkpoint was changed")
        if file_hash(Path(source) / "training/checkpoint.pt") != dataset_parent_hash:
            raise ValueError("Temperature smoke dataset checkpoint was changed")
        report = {
            "task": "habitability_calculations",
            "profile": "pilot" if pilot else "sequence_diagnostic",
            "calculation_mode": "local_tool_assisted",
            "content": content,
            "budget": budget,
            "training": training,
            "teacher_forced": teacher,
            "closed_loop": evaluations,
            "checkpoint_reload_verified": True,
            "frozen_text_verified": True,
            "parent_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": peak_process_rss_bytes(),
            "code_smoke_passed": True,
            "learning_gate_passed": False,
            "course_acceptance_passed": False,
            "final_test_episodes": 0,
            "gate_note": "Fixed local pilot; final tests remain sealed and gas/phase/habitability classification remains pending."
            if pilot
            else "Recorded four-case diagnostic, not unseen success or gas/phase/habitability classification.",
            **(
                {"expert_gate": {"completed": 100, "episodes": 100, "policy": "scripted_expert"}}
                if pilot
                else {}
            ),
        }
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"status": "complete", "learning_gate_passed": False})
        return report
    except Exception as exc:
        write_json(directory / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise
