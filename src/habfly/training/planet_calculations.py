"""Bounded offline smoke for supplied-measurement planet tool use.

No browser, course grading oracle, old checkpoint migration or final-test cases.
The scripted expert gate and learned closed-loop results are separate artifacts.
"""

import itertools
import json
import math
import time

import torch

from habfly.environments.planet_calculations import (
    FIELDS,
    SCOPE,
    TEMPLATES,
    PlanetCalculationEnv,
    planet_cases,
    planet_expert,
)
from habfly.model import ConnectomePolicy, graph_fingerprint
from habfly.planet_knowledge import PlanetCalculator

from .checkpoints import load_checkpoint, source_hash
from .stellar import record_episode, write_json
from .train import peak_process_rss_bytes, prepare_output_directory, train_behavioral_cloning

ENCODING = "structured_planet_tool_v1"
CONTROL_ENCODING = "semantic_planet_tool_v1"
COUNTS = {"gate": 100, "train": 4, "calibration": 2, "development": 2}
ERRORS = ("invalid_actions", "tool_errors", "infrastructure_failures", "api_failures")
BUDGET = {
    "epochs": 2,
    "hidden_size": 16,
    "cpu_threads": 1,
    "seed": 0,
    "sequence_length": 8,
    "max_episode_actions": 128,
    "optimizer_updates": 64,
    "ppo": False,
    "final_test_episodes": 0,
}


def new_policy(graph):
    return ConnectomePolicy(
        graph,
        hidden_size=16,
        observation_encoding=ENCODING,
        control_encoding=CONTROL_ENCODING,
        selection_mode="measurement_result_v3",
    )


def validate_separation(cases, *, templates=TEMPLATES):
    for left, right in itertools.combinations(cases.values(), 2):
        for key in ("seed", "case_id"):
            if {c[key] for c in left} & {c[key] for c in right}:
                raise ValueError("Planet case split overlap")
        if {templates[c["split"]] for c in left} & {templates[c["split"]] for c in right}:
            raise ValueError("Planet template split overlap")
        for quantity in ("period_days", "line_shift", "brightness_drop", "stellar_mass", "stellar_radius"):

            def numbers(rows, kind=quantity):
                return {m["value"] for c in rows for m in c["measurements"].values() if m["kind"] == kind}

            if numbers(left) & numbers(right):
                raise ValueError("Planet numeric split overlap")


def content_identity(calculator, cases, graph, *, scope=SCOPE, required_fields=FIELDS, templates=TEMPLATES):
    validate_separation(cases, templates=templates)
    return {
        "task": "planet_calculations",
        "scope": scope,
        "calculation_mode": "local_tool_assisted",
        "calculation_backend": "local",
        "knowledge_pack_hash": calculator.pack.checksum,
        "graph_hash": graph_fingerprint(graph),
        "observation_encoding": ENCODING,
        "control_encoding": CONTROL_ENCODING,
        "required_fields": list(required_fields),
        "validation_scope": calculator.pack.validation_scope,
        "course_acceptance_passed": False,
        "expected_provenance": "inverse_generated_physics_not_course_grading",
        "splits": {
            split: {
                "count": len(rows),
                "sha256": source_hash(rows),
                "seeds": [c["seed"] for c in rows],
                "case_ids": [c["case_id"] for c in rows],
                "instruction_hash": source_hash(templates[split]),
            }
            for split, rows in cases.items()
        },
        "final_test_status": "not_generated_or_evaluated",
    }


def collect_split(
    calculator, cases, directory, *, policy=None, environment=PlanetCalculationEnv, expected_steps=62
):
    """Preserve failures and runtime-v1 replay without exposing private answers."""
    directory = prepare_output_directory(directory)
    env = environment(calculator, cases)
    episodes, summaries = [], []
    for case in cases:
        episode, summary = record_episode(
            env, case["seed"], policy, event_path=directory / f"{case['seed']}.events.jsonl"
        )
        # Diagnostic only: compare choices after the episode. This reference
        # action is never fed to, or executed in place of, the learned policy.
        choices = []
        for step in episode:
            action, observation = step["action"], step["observation"]
            target = next((c for c in observation["controls"] if c["id"] == action.get("target")), {})
            if action["kind"] == "SELECT" and target.get("label") == "Calculation":
                expected = planet_expert(observation, calculator.pack)
                choices.append(expected.target == action["target"] and expected.value == action["value"])
        summary.update(operation_choice_attempts=len(choices), operation_choice_correct=sum(choices))
        episodes.append(episode)
        summaries.append(summary)
        write_json(directory / "summaries.json", summaries)
        if policy is None and (
            not summary["completed"]
            or summary["steps"] != expected_steps
            or any(summary.get(k, 0) for k in ERRORS)
        ):
            raise ValueError("Planet scripted expert gate failed; preserved replay is not training data")
    write_json(directory / "episodes.json", episodes)
    return episodes, summaries


def summarize(rows, *, required_count=4):
    def ratio(correct, total):
        denominator = sum(r.get(total, 0) for r in rows)
        return sum(r.get(correct, 0) for r in rows) / denominator if denominator else None

    return {
        "episodes": len(rows),
        "completed": sum(bool(r["completed"]) for r in rows),
        "completion_rate": sum(bool(r["completed"]) for r in rows) / len(rows),
        "mean_steps": sum(r["steps"] for r in rows) / len(rows),
        "mean_reward": sum(r.get("reward", 0) for r in rows) / len(rows),
        "input_binding_accuracy": ratio("input_correct", "input_attempts"),
        "calculation_selection_accuracy": ratio("operation_choice_correct", "operation_choice_attempts"),
        "output_copy_accuracy": ratio("copy_correct", "copy_attempts"),
        "unit_selection_accuracy": ratio("unit_correct", "unit_attempts"),
        "numeric_answer_accuracy": sum(r.get("numeric_fields_correct", 0) for r in rows)
        / (len(rows) * required_count),
        "selection_metric_note": "Attempt-based selection metrics; not sequence or task success.",
        **{key: sum(row.get(key, 0) for row in rows) for key in ERRORS},
    }


def run_smoke(graph, output, *, progress=None):
    if len(graph.body_ids) != 2000 or graph.manifest.get("source_kind") != "biological":
        raise ValueError("Planet smoke requires the real 2,000-node graph")
    started = time.perf_counter()
    directory = prepare_output_directory(output)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    calculator = PlanetCalculator()
    verification = calculator.verify()
    cases = {s: planet_cases(s, n) for s, n in COUNTS.items()}
    content = content_identity(calculator, cases, graph)
    manifest = {
        "version": 1,
        "content": content,
        "budget": BUDGET,
        "counts": COUNTS,
        "verification": verification,
        "graph_nodes": len(graph.body_ids),
        "graph_edges": len(graph.edge_src),
        "parent_checkpoint": None,
        "training_stage": "random_initialization_smoke",
    }
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "status.json", {"status": "collecting"})
    try:
        episodes = {}
        for split, rows in cases.items():
            episodes[split], summaries = collect_split(calculator, rows, directory / f"expert-{split}")
            write_json(directory / f"private-{split}-cases.json", rows)
            if progress:
                progress({"phase": "expert", "split": split, "completed": len(summaries)})
        # Explicit alternation matches BC's calibration[::2]/heldout[1::2]
        # contract: calibration and development never exchange roles.
        validation = list(
            itertools.chain.from_iterable(zip(episodes["calibration"], episodes["development"]))
        )
        write_json(directory / "status.json", {"status": "training"})
        policy = new_policy(graph)
        train = train_behavioral_cloning(
            episodes["train"],
            graph,
            directory / "training",
            validation_episodes=validation,
            epochs=2,
            seed=0,
            hidden_size=16,
            sequence_length=8,
            content_pack=content,
            policy=policy,
            progress_callback=progress,
        )
        if train["optimizer_steps"] != BUDGET["optimizer_updates"] or not all(
            map(math.isfinite, train["losses"])
        ):
            raise ValueError("Planet smoke violated its finite-loss/update budget")
        restored, checkpoint = load_checkpoint(
            directory / "training/checkpoint.pt", graph, content_pack=content
        )
        if not all(torch.equal(v, restored.state_dict()[k]) for k, v in policy.state_dict().items()):
            raise ValueError("Planet checkpoint reload changed parameters")
        write_json(directory / "status.json", {"status": "closed_loop_evaluation"})
        evaluations = {}
        for split in ("train", "development"):
            _, rows = collect_split(calculator, cases[split], directory / f"learned-{split}", policy=restored)
            evaluations[split] = summarize(rows)
            if progress:
                progress({"phase": "learned_closed_loop", "split": split, **evaluations[split]})
        report = {
            "task": "planet_calculations",
            "calculation_mode": "local_tool_assisted",
            "content": content,
            "budget": BUDGET,
            "training": train,
            "closed_loop": evaluations,
            "expert_gate": {"completed": 100, "episodes": 100, "policy": "scripted_expert"},
            "checkpoint_reload_verified": True,
            "checkpoint_content_hash": checkpoint["content_pack_hash"],
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": peak_process_rss_bytes(),
            "code_smoke_passed": True,
            "learning_gate_passed": False,
            "gate_note": "A smoke run cannot meet the separate 100-unseen-case learning gate or course acceptance.",
            "course_acceptance_passed": False,
            "final_test_episodes": 0,
        }
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"status": "complete", "learning_gate_passed": False})
        return report
    except Exception as exc:
        write_json(directory / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise


def print_progress(item):
    print(json.dumps(item), flush=True)
