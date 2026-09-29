"""Explicitly capped prerequisite diagnostics and a separate full-task pilot."""

import itertools
import math
import time
from pathlib import Path

import torch

from habfly.environments.planet_curriculum import STAGES, stage_cases, stage_environment
from habfly.planet_knowledge import PlanetCalculator

from .checkpoints import load_checkpoint
from .frozen_text import frozen_text_cache
from .planet_calculations import COUNTS, ENCODING, collect_split, content_identity, summarize
from .planet_sequence import file_hash, read
from .stellar import write_json
from .train import peak_process_rss_bytes, prepare_output_directory, train_behavioral_cloning

BUDGET = {
    "epochs": 20,
    "additional_optimizer_updates": 80,
    "sequence_length": 10,
    "new_numeric_cases": 0,
    "hidden_size": 16,
    "cpu_threads": 1,
    "seed": 0,
    "max_episode_actions": 128,
    "ppo": False,
    "final_test_episodes": 0,
    "optimizer_state": "fresh_AdamW_moments_for_prerequisite_scope",
}
PILOT_COUNTS = {"gate": 100, "train": 64, "calibration": 16, "development": 16}


def run_planet_pilot(source, graph, output, *, progress=None):
    """Five epochs on 64 cases; never automatically opens the final test."""
    return run_curriculum_diagnostic(
        source, graph, output, stage="full", progress=progress, pilot=True, resume_optimizer=True
    )


def require_parent_gate(report):
    for split in ("train", "development"):
        scores = report.get("closed_loop", {}).get(split, {})
        if (
            scores.get("episodes") != COUNTS[split]
            or scores.get("completed") != COUNTS[split]
            or any(
                scores.get(k) != 0
                for k in ("invalid_actions", "tool_errors", "infrastructure_failures", "api_failures")
            )
            or scores.get("numeric_answer_accuracy") != 1.0
        ):
            raise ValueError("Preceding planet curriculum stage has not passed its closed-loop gate")


def run_period_diagnostic(source, graph, output, *, progress=None):
    return run_curriculum_diagnostic(source, graph, output, stage="period", progress=progress)


def run_curriculum_diagnostic(
    source, graph, output, *, stage, progress=None, refine=False, resume_optimizer=False, pilot=False
):
    if stage not in STAGES:
        raise ValueError("Unknown planet curriculum stage")
    spec = STAGES[stage]
    if pilot and (stage != "full" or refine):
        raise ValueError("Pilot requires the complete scope, not a prerequisite refinement")
    if resume_optimizer and not (refine or pilot):
        raise ValueError("Optimizer continuation requires the same curriculum stage")
    counts = PILOT_COUNTS if pilot else COUNTS
    budget = {**BUDGET, "sequence_length": spec.steps}
    if pilot:
        budget.update(epochs=5, additional_optimizer_updates=320, new_numeric_cases=88, new_training_cases=60)
    if resume_optimizer:
        budget["optimizer_state"] = "resume_parent_AdamW_moments_same_stage"
    environment = stage_environment(stage)

    if len(graph.body_ids) != 2000 or graph.manifest.get("source_kind") != "biological":
        raise ValueError("Planet experiment requires the real 2,000-node graph")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    calculator, started = PlanetCalculator(), time.perf_counter()
    verification = calculator.verify()
    checkpoint = Path(source) / "training/checkpoint.pt"
    parent_hash = file_hash(checkpoint)
    parent_content = read(checkpoint.with_suffix(".pt.json"))["provenance"]["planet_calculations"]
    expected_parent_scope = spec.scope if refine or pilot else spec.parent_scope
    if (
        parent_content.get("scope") != expected_parent_scope
        or parent_content.get("knowledge_pack_hash") != calculator.pack.checksum
    ):
        raise ValueError("Planet curriculum needs its compatible explicitly preceding parent scope")
    if refine and (
        parent_content.get("prerequisite_diagnostic", {}).get("stage") != stage
        or read(Path(source) / "report.json").get("training", {}).get("optimizer_steps") != 80
    ):
        raise ValueError("Refinement requires a completed same-stage 80-update parent")
    if pilot and parent_content.get("prerequisite_diagnostic", {}).get("stage") != "full":
        raise ValueError("Pilot requires a completed full-scope diagnostic, not another pilot")
    if stage != "period" and not refine:
        require_parent_gate(read(Path(source) / "report.json"))
    if pilot:
        from .planet_evaluation import verify_parent_artifacts

        verify_parent_artifacts(source, parent_content, read(Path(source) / "report.json"))
    policy, _ = load_checkpoint(checkpoint, graph, content_pack=parent_content)
    optimizer_state = None
    if resume_optimizer:
        optimizer_state = torch.load(checkpoint, map_location="cpu", weights_only=True).get("optimizer")
        if not optimizer_state:
            raise ValueError("Parent does not contain the required optimizer state")
    if policy.observation_encoding != ENCODING or policy.hidden_size != 16:
        raise ValueError("Period parent model contract mismatch")
    cases = {s: stage_cases(stage, s, n) for s, n in counts.items()}
    content = content_identity(
        calculator, cases, graph, scope=spec.scope, required_fields=spec.required, templates=spec.templates
    )
    content["pilot" if pilot else "prerequisite_diagnostic"] = {
        "version": 1,
        "stage": stage,
        "refinement": refine,
        "inherited_stage_updates": (
            parent_content["prerequisite_diagnostic"].get("inherited_stage_updates", 0) + 80 if refine else 0
        ),
        "parent_checkpoint_sha256": parent_hash,
        "parent_scope": expected_parent_scope,
        "budget": budget,
    }
    directory = prepare_output_directory(output)
    write_json(
        directory / "manifest.json", {"content": content, "budget": budget, "verification": verification}
    )
    write_json(directory / "status.json", {"status": "collecting"})

    def update(row):
        write_json(directory / "status.json", {"status": "training", **row})
        if progress:
            progress(row)

    try:
        episodes = {}
        for split, rows in cases.items():
            episodes[split], summaries = collect_split(
                calculator,
                rows,
                directory / f"expert-{split}",
                environment=environment,
                expected_steps=spec.steps,
            )
            write_json(directory / f"private-{split}-cases.json", rows)
            if progress:
                progress({"phase": "expert", "split": split, "completed": len(summaries)})
        validation = list(
            itertools.chain.from_iterable(zip(episodes["calibration"], episodes["development"]))
        )
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(not name.startswith(("embedding.", "text_encoder.")))
        frozen = {k: v.detach().clone() for k, v in policy.named_parameters() if not v.requires_grad}
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
                sequence_length=spec.steps,
                content_pack=content,
                progress_callback=update,
                optimizer_state=optimizer_state,
            )
        if training["optimizer_steps"] != budget["additional_optimizer_updates"] or not all(
            map(math.isfinite, training["losses"])
        ):
            raise ValueError("Planet finite-loss or update budget failure")
        if any(not torch.equal(v, dict(policy.named_parameters())[k]) for k, v in frozen.items()):
            raise ValueError("Period frozen text parameters changed")
        restored, _ = load_checkpoint(directory / "training/checkpoint.pt", graph, content_pack=content)
        if any(not torch.equal(v, restored.state_dict()[k]) for k, v in policy.state_dict().items()):
            raise ValueError("Period reload changed weights")
        write_json(directory / "status.json", {"status": "closed_loop_evaluation"})
        evaluation = {}
        for split in ("train", "development"):
            _, rows = collect_split(
                calculator,
                cases[split],
                directory / f"learned-{split}",
                policy=restored,
                environment=environment,
                expected_steps=spec.steps,
            )
            evaluation[split] = summarize(rows, required_count=len(spec.required))
            if progress:
                progress({"phase": "closed_loop", "split": split, **evaluation[split]})
        if file_hash(checkpoint) != parent_hash:
            raise ValueError("Period diagnostic changed its parent")
        report = {
            "content": content,
            "budget": budget,
            "training": training,
            "closed_loop": evaluation,
            "expert_gate": {"completed": 100, "episodes": 100, "policy": "scripted_expert"},
            "checkpoint_reload_verified": True,
            "parent_unchanged": True,
            "frozen_text_verified": True,
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": peak_process_rss_bytes(),
            "learning_gate_passed": False,
            "course_acceptance_passed": False,
            "final_test_episodes": 0,
            "gate_note": "Local development experiment only; no 100-unseen-case learning gate or course acceptance.",
            "profile": "pilot" if pilot else "curriculum_diagnostic",
        }
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"status": "complete", "learning_gate_passed": False})
        return report
    except Exception as exc:
        write_json(directory / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise
