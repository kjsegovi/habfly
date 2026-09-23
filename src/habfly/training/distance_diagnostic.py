"""Bounded memorization and unseen-distance diagnostics, separate from stellar acceptance."""

from __future__ import annotations

import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import Field, model_validator

from habfly.contracts import Contract, Observation
from habfly.data import load_graph
from habfly.environments.distance_diagnostic import DistanceDiagnosticEnv, distance_case
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import graph_fingerprint

from .checkpoints import load_checkpoint, source_hash
from .stellar import record_episode, write_json
from .train import peak_process_rss_bytes, prepare_output_directory, train_behavioral_cloning


class DistanceDiagnosticConfig(Contract):
    name: Literal["distance-diagnostic-v1"] = "distance-diagnostic-v1"
    graph: Path = Path("data/processed/graphs-v2/graph-2000")
    knowledge_pack: Path | None = None
    training_cases: int = Field(default=4, ge=1, le=4)
    calibration_cases: int = Field(default=2, ge=1, le=2)
    development_cases: int = Field(default=2, ge=1, le=2)
    unseen_cases: int = Field(default=8, ge=1, le=8)
    epochs: int = Field(default=20, ge=1, le=20)
    hidden_size: Literal[16] = 16
    device: Literal["cpu"] = "cpu"
    threads: Literal[1] = 1
    seed: Literal[0] = 0
    max_steps: int = Field(default=32, ge=10, le=32)

    @model_validator(mode="after")
    def matching_validation_counts(self):
        if self.calibration_cases != self.development_cases:
            raise ValueError("Calibration and development counts must match")
        return self


def load_distance_config(path=None):
    return (
        DistanceDiagnosticConfig.model_validate(yaml.safe_load(Path(path).read_text()) or {})
        if path
        else DistanceDiagnosticConfig()
    )


STAGES = {
    "operation": "select_calculation",
    "parameter": "select_input",
    "source": "select_source",
    "bind": "bind_input",
    "execute": "execute_calculation",
    "result": "select_result",
    "destination": "select_destination",
    "copy": "copy_result",
    "unit_distance": "select_unit",
    "check": "check_completion",
}
SPLIT_SEEDS = {"train": 600000, "calibration": 700000, "development": 800000, "test": 900000}


def control_key(observation, action):
    target = next((c for c in observation["controls"] if c["id"] == action.get("target")), None)
    return target["id"].split(":", 1)[1] if target else "untargeted"


@torch.no_grad()
def teacher_forced_stages(policy, episodes):
    """Score actual policy decoding, with expert observations but no expert action injection."""
    rows = {
        stage: {
            "attempts": 0,
            "action_correct": 0,
            "target_correct": 0,
            "value_correct": 0,
            "exact_correct": 0,
        }
        for stage in STAGES.values()
    }
    for episode in episodes:
        state = None
        for step in episode:
            expected = step["action"]
            stage = STAGES[control_key(step["observation"], expected)]
            proposed, state, _ = policy.act(Observation.model_validate(step["observation"]), state)
            action_ok = proposed.kind == expected["kind"]
            target_ok = proposed.target == expected["target"]
            value_ok = expected["value"] is None or proposed.value == expected["value"]
            row = rows[stage]
            row["attempts"] += 1
            row["action_correct"] += int(action_ok)
            row["target_correct"] += int(target_ok)
            row["value_correct"] += int(value_ok)
            row["exact_correct"] += int(action_ok and target_ok and value_ok)
    for row in rows.values():
        row["exact_accuracy"] = row["exact_correct"] / row["attempts"] if row["attempts"] else None
    total = sum(row["attempts"] for row in rows.values())
    return {
        "scope": "teacher_forced_expert_observations",
        "examples": total,
        "exact_action_accuracy": sum(row["exact_correct"] for row in rows.values()) / max(total, 1),
        "stages": rows,
    }


def rollout_stages(case, trajectory):
    """Private analysis only: stage labels never enter policy observations or rewards."""
    reached, attempts, correct, action_counts = set(), Counter(), Counter(), Counter()
    expected = case["expected"]["distance"]

    def good_source(ref):
        source = case["measurements"].get(ref, {})
        return source.get("kind") == "parallax" and source.get("source") == "current star"

    def good_result(result):
        return bool(
            result
            and result.get("valid")
            and result["kind"] == "distance"
            and result["unit"] == "ly"
            and math.isclose(result["value"], expected, rel_tol=1e-12)
            and good_source(result["bindings"].get("parallax"))
        )

    for step in trajectory:
        key = control_key(step["observation"], step["action"])
        action_counts[f"{step['action']['kind']}:{key}"] += 1
        if key not in STAGES:
            continue
        stage = STAGES[key]
        attempts[stage] += 1
        if "invalid_action" in step["result"]["reward_components"]:
            continue
        expected_kind = "CLICK" if key in {"bind", "execute", "copy", "check"} else "SELECT"
        if step["action"]["kind"] != expected_kind:
            continue
        obs = step["result"]["observation"]
        tool = obs["calculation"]
        operation = tool.get("last_operation") or {}
        current_result = tool["results"].get(tool["selected_result"])
        value = step["action"].get("value")
        flags = {
            "operation": value == "distance",
            "parameter": tool["operation"] == "distance" and value == "parallax",
            "source": good_source(value),
            "bind": tool["operation"] == "distance"
            and good_source(tool["bindings"].get("parallax"))
            and not tool["tool_error"],
            "execute": operation.get("kind") == "calculate"
            and good_result(tool["results"].get(tool["pending_result"])),
            "result": good_result(current_result),
            "destination": value == "distance",
            "copy": operation.get("kind") == "copy"
            and operation.get("destination") == "distance"
            and good_result(tool["results"].get(operation.get("result_id")))
            and obs["values"]["answers"].get("distance") == expected,
            "unit_distance": value == "ly",
            "check": bool(obs["progress"].get("task_completed")),
        }
        if flags[key]:
            reached.add(stage)
            correct[stage] += 1
    return {
        "stages_reached": [stage for stage in STAGES.values() if stage in reached],
        "first_unreached_stage": next((stage for stage in STAGES.values() if stage not in reached), None),
        "attempts": dict(attempts),
        "correct_attempts": dict(correct),
        "action_counts": dict(action_counts),
    }


def closed_loop(policy, calculator, cases, directory, max_steps):
    directory = prepare_output_directory(directory)
    summaries = []
    for case in cases:
        env = DistanceDiagnosticEnv(calculator, [case], max_steps=max_steps)
        trajectory, summary = record_episode(
            env, case["seed"], policy, event_path=directory / f"{case['seed']}.events.jsonl"
        )
        write_json(directory / f"{case['seed']}.trajectory.json", trajectory)
        summary.update(rollout_stages(case, trajectory))
        summaries.append(summary)
        write_json(directory / "episodes.json", summaries)
        if summary.get("infrastructure_failures") or summary.get("api_failures"):
            break
    errors = {
        key: sum(s.get(key, 0) for s in summaries)
        for key in ("invalid_actions", "tool_errors", "infrastructure_failures", "api_failures")
    }
    report = {
        "scope": "closed_loop_policy_actions",
        "requested_episodes": len(cases),
        "episodes": summaries,
        "completed": sum(s["completed"] for s in summaries),
        **errors,
        "completion_rate": sum(s["completed"] for s in summaries) / len(cases),
        "stage_reach": {
            stage: sum(stage in s["stages_reached"] for s in summaries) for stage in STAGES.values()
        },
        "first_unreached_counts": dict(
            Counter(s["first_unreached_stage"] or "all_reached" for s in summaries)
        ),
    }
    write_json(directory / "report.json", report)
    return report


def fit_gate(teacher, rollout):
    return (
        teacher["exact_action_accuracy"] >= 0.95
        and rollout["completed"] == rollout["requested_episodes"]
        and not any(
            rollout[k] for k in ("invalid_actions", "tool_errors", "infrastructure_failures", "api_failures")
        )
    )


def run_distance_diagnostic(output, settings):
    """One fresh capped BC run. Never resume, increase the budget, or use Sheets/browser."""
    torch.set_num_threads(settings.threads)
    torch.manual_seed(settings.seed)
    graph = load_graph(settings.graph)
    pack = load_knowledge_pack(settings.knowledge_pack)
    calculator = LocalCalculator(pack)
    verification = calculator.verify()
    directory = prepare_output_directory(output)
    started = time.perf_counter()

    def status(stage, **extra):
        print(f"distance diagnostic: {stage}", file=sys.stderr, flush=True)
        write_json(directory / "status.json", {"stage": stage, **extra})

    try:
        status("collecting")
        counts = {
            "train": settings.training_cases,
            "calibration": settings.calibration_cases,
            "development": settings.development_cases,
            "test": settings.unseen_cases,
        }
        cases, episodes, splits = {}, {}, {}
        all_ids, all_parallaxes = set(), set()
        for split, count in counts.items():
            cases[split] = [
                distance_case(SPLIT_SEEDS[split] + index, split, calculator) for index in range(count)
            ]
            # Never use existing stellar development/final-test cases for this diagnostic.
            ids = {c["case_id"] for c in cases[split]}
            parallaxes = {c["inputs"]["parallax"] for c in cases[split]}
            if len(ids) != count or len(parallaxes) != count or ids & all_ids or parallaxes & all_parallaxes:
                raise ValueError("Distance diagnostic split overlap")
            all_ids.update(ids)
            all_parallaxes.update(parallaxes)
            episodes[split] = []
            for case in cases[split]:
                env = DistanceDiagnosticEnv(calculator, [case], max_steps=settings.max_steps)
                episode, summary = record_episode(
                    env, case["seed"], event_path=directory / f"expert-{case['seed']}.events.jsonl"
                )
                if (
                    not summary["completed"]
                    or summary["invalid_actions"]
                    or summary["tool_errors"]
                    or len(episode) != 10
                ):
                    raise ValueError("Distance expert gate failed")
                episodes[split].append(episode)
            payload = {"cases": cases[split], "episodes": episodes[split]}
            write_json(directory / f"{split}.json", payload)
            splits[split] = {
                "file": f"{split}.json",
                "sha256": source_hash(payload),
                "count": count,
                "seeds": [c["seed"] for c in cases[split]],
                "case_ids": sorted(ids),
            }
        content = {
            **pack.content_identity(),
            "diagnostic": settings.name,
            "required_fields": ["distance"],
            "dataset_splits": splits,
            "max_steps": settings.max_steps,
        }
        manifest = {
            "version": 1,
            "content": content,
            "graph_hash": graph_fingerprint(graph),
            "graph_nodes": len(graph.body_ids),
            "settings": settings.model_dump(mode="json"),
            "verification": verification,
            "expert_gate_passed": True,
            "max_optimizer_steps": settings.training_cases * 10 * settings.epochs,
            "unseen_policy_evaluation_requires_fit_gate": True,
        }
        write_json(directory / "manifest.json", manifest)
        validation = [
            episode for pair in zip(episodes["calibration"], episodes["development"]) for episode in pair
        ]
        status("training", max_optimizer_steps=manifest["max_optimizer_steps"], epochs=settings.epochs)
        training = train_behavioral_cloning(
            episodes["train"],
            graph,
            directory / "training",
            validation_episodes=validation,
            epochs=settings.epochs,
            seed=settings.seed,
            hidden_size=settings.hidden_size,
            content_pack=content,
        )
        policy, _ = load_checkpoint(directory / "training/checkpoint.pt", graph, content_pack=content)
        status("evaluating_training_fit")
        teacher = {
            split: teacher_forced_stages(policy, episodes[split]) for split in ("train", "development")
        }
        seen = closed_loop(policy, calculator, cases["train"], directory / "seen", settings.max_steps)
        passed = fit_gate(teacher["train"], seen)
        unseen = {"status": "not_run", "reason": "training_fit_gate_failed"}
        if passed:
            status("evaluating_unseen")
            unseen = {
                "status": "evaluated",
                **closed_loop(policy, calculator, cases["test"], directory / "unseen", settings.max_steps),
            }
            teacher["unseen"] = teacher_forced_stages(policy, episodes["test"])
        if (
            seen["infrastructure_failures"]
            or seen["api_failures"]
            or unseen.get("infrastructure_failures")
            or unseen.get("api_failures")
        ):
            outcome = "infrastructure_failure"
            if unseen["status"] == "not_run":
                unseen["reason"] = "infrastructure_failure"
        elif not passed:
            outcome = (
                "training_imitation_failed"
                if teacher["train"]["exact_action_accuracy"] < 0.95
                else "training_rollout_failed"
            )
        else:
            outcome = (
                "distance_diagnostic_passed"
                if fit_gate(teacher["unseen"], unseen)
                else "unseen_generalization_failed"
            )
        report = {
            "diagnostic": settings.name,
            "calculation_mode": "local_tool_assisted",
            "outcome": outcome,
            "training_fit_gate_passed": passed,
            "distance_gate_passed": outcome == "distance_diagnostic_passed",
            "stellar_acceptance_gate_passed": False,
            "teacher_forced": teacher,
            "seen": seen,
            "unseen": unseen,
            "training": {
                k: training[k] for k in ("epochs", "losses", "elapsed_seconds", "process_peak_rss_bytes")
            },
            "content": content,
            "graph_hash": manifest["graph_hash"],
            "graph_nodes": manifest["graph_nodes"],
            "budget": settings.model_dump(mode="json"),
            "max_optimizer_steps": manifest["max_optimizer_steps"],
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": peak_process_rss_bytes(),
            "note": "Stage reach counts independent observed milestones, not an enforced action order. Failure does not identify a root cause by itself. No retry or budget increase.",
        }
        write_json(directory / "report.json", report)
        status("complete", outcome=outcome)
        return report
    except Exception as exc:
        status("failed", error_type=type(exc).__name__)
        raise
    finally:
        calculator.close()
