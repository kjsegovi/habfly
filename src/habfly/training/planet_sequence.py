"""Explicit 80-update full-sequence follow-up; never an automatic budget increase."""

import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import torch

from habfly.planet_knowledge import PlanetCalculator

from .checkpoints import load_checkpoint, source_hash
from .frozen_text import frozen_text_cache
from .planet_calculations import (
    BUDGET,
    COUNTS,
    ENCODING,
    collect_split,
    content_identity,
    summarize,
)
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
    "sequence_length": 62,
    "new_cases": 0,
    "cpu_threads": 1,
    "seed": 0,
    "hidden_size": 16,
    "frozen_parameters": ["embedding", "text_encoder"],
    "optimizer_state": "fresh_AdamW_moments_for_explicit_followup",
    "ppo": False,
    "final_test_episodes": 0,
}


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validated_smoke(source, graph, calculator):
    source = Path(source)
    manifest, report = read(source / "manifest.json"), read(source / "report.json")
    if (
        manifest.get("budget") != BUDGET
        or manifest.get("counts") != COUNTS
        or report.get("content") != manifest.get("content")
        or not report.get("checkpoint_reload_verified")
        or report.get("training", {}).get("optimizer_steps") != 64
    ):
        raise ValueError("Incompatible planet smoke parent")
    cases = {s: read(source / f"private-{s}-cases.json") for s in COUNTS}
    if {s: len(rows) for s, rows in cases.items()} != COUNTS:
        raise ValueError("Planet smoke case count mismatch")
    content = content_identity(calculator, cases, graph)
    if content != manifest["content"]:
        raise ValueError("Planet smoke content, graph, pack or split hash mismatch")
    episodes = {
        s: read(source / f"expert-{s}/episodes.json") for s in ("train", "calibration", "development")
    }
    if any(len(episodes[s]) != COUNTS[s] or any(len(ep) != 62 for ep in episodes[s]) for s in episodes):
        raise ValueError("Planet smoke trajectory length mismatch")
    validation = list(itertools.chain.from_iterable(zip(episodes["calibration"], episodes["development"])))
    if source_hash(episodes["train"]) != report["training"].get("training_data_hash") or source_hash(
        validation
    ) != report["training"].get("validation_data_hash"):
        raise ValueError("Planet smoke demonstration hash mismatch")
    checkpoint = source / "training/checkpoint.pt"
    policy, saved = load_checkpoint(checkpoint, graph, content_pack=content)
    if policy.observation_encoding != ENCODING or policy.hidden_size != 16:
        raise ValueError("Planet smoke model contract mismatch")
    if saved["evaluation_scores"].get("training_data_hash") != report["training"].get("training_data_hash"):
        raise ValueError("Planet smoke checkpoint dataset mismatch")
    return policy, content, cases, episodes, validation


def run_sequence_followup(source, graph, output, *, progress=None, stellar_parent=None):
    if len(graph.body_ids) != 2000 or graph.manifest.get("source_kind") != "biological":
        raise ValueError("Planet sequence diagnostic requires the real 2,000-node graph")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    started = time.perf_counter()
    calculator = PlanetCalculator()
    verification = calculator.verify()
    policy, parent_content, cases, episodes, validation = validated_smoke(source, graph, calculator)
    parent_path = Path(source) / "training/checkpoint.pt"
    transfer = None
    if stellar_parent is not None:
        from .planet_transfer import transfer_stellar_policy

        parent_path = Path(stellar_parent)
        metadata = read(parent_path.with_suffix(parent_path.suffix + ".json"))
        stellar_content = metadata.get("provenance", {}).get("stellar")
        if not stellar_content or stellar_content.get("calculation_backend") != "local":
            raise ValueError("Planet transfer needs a declared local stellar parent")
        parent, _ = load_checkpoint(parent_path, graph, content_pack=stellar_content)
        policy, transfer = transfer_stellar_policy(parent)
    parent_hash = file_hash(parent_path)
    content = {
        **parent_content,
        "sequence_diagnostic": {
            "version": 1,
            "parent_checkpoint_sha256": parent_hash,
            "inherited_optimizer_updates": None if transfer else 64,
            "initialization": "explicit_stellar_transfer" if transfer else "planet_smoke_checkpoint",
            "transfer": transfer,
            "budget": FOLLOWUP_BUDGET,
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
            "budget": FOLLOWUP_BUDGET,
        },
    )
    write_json(directory / "status.json", {"status": "training", "additional_update_cap": 80})
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(not name.startswith(("embedding.", "text_encoder.")))
    frozen = {name: p.detach().clone() for name, p in policy.named_parameters() if not p.requires_grad}

    def update(row):
        write_json(directory / "status.json", {"status": "training", **row})
        if progress:
            progress(row)

    try:
        with frozen_text_cache(policy):
            training = train_behavioral_cloning(
                episodes["train"],
                graph,
                directory / "training",
                policy=policy,
                validation_episodes=validation,
                epochs=20,
                seed=0,
                hidden_size=16,
                sequence_length=62,
                content_pack=content,
                progress_callback=update,
            )
            teacher = {
                split: evaluate_examples(
                    policy, episodes_to_examples(rows), episode_lengths=[len(ep) for ep in rows]
                )
                for split, rows in episodes.items()
                if split != "calibration"
            }
        if training["optimizer_steps"] != 80 or not all(map(math.isfinite, training["losses"])):
            raise ValueError("Planet sequence budget or finite-loss failure")
        if any(not torch.equal(v, dict(policy.named_parameters())[k]) for k, v in frozen.items()):
            raise ValueError("Frozen planet text parameters changed")
        restored, _ = load_checkpoint(directory / "training/checkpoint.pt", graph, content_pack=content)
        if any(not torch.equal(v, restored.state_dict()[k]) for k, v in policy.state_dict().items()):
            raise ValueError("Planet sequence checkpoint reload changed weights")
        write_json(directory / "status.json", {"status": "closed_loop_evaluation"})
        evaluations = {}
        for split in ("train", "development"):
            _, rows = collect_split(calculator, cases[split], directory / f"learned-{split}", policy=restored)
            evaluations[split] = summarize(rows)
            if progress:
                progress({"phase": "closed_loop", "split": split, **evaluations[split]})
        if file_hash(parent_path) != parent_hash:
            raise ValueError("Parent checkpoint was changed")
        report = {
            "content": content,
            "budget": FOLLOWUP_BUDGET,
            "training": training,
            "teacher_forced": teacher,
            "closed_loop": evaluations,
            "checkpoint_reload_verified": True,
            "frozen_text_verified": True,
            "parent_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": peak_process_rss_bytes(),
            "learning_gate_passed": False,
            "final_test_episodes": 0,
            "course_acceptance_passed": False,
            "gate_note": "Training/development diagnostic only; no unseen-100 or browser completion claim.",
        }
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"status": "complete", "learning_gate_passed": False})
        return report
    except Exception as exc:
        write_json(directory / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise
