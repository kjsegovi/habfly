"""Fixed-budget local supplied-temperature smoke, with no browser acceptance claim."""

import itertools
import math
import time

import torch

from habfly.environments.habitability_calculations import (
    FIELDS,
    SCOPE,
    TEMPLATES,
    HabitabilityCalculationEnv,
    habitability_cases,
)
from habfly.habitability_knowledge import HabitabilityCalculator
from habfly.model import ConnectomePolicy, graph_fingerprint

from .checkpoints import load_checkpoint, source_hash
from .planet_calculations import ERRORS, summarize
from .stellar import record_episode, write_json
from .train import peak_process_rss_bytes, prepare_output_directory, train_behavioral_cloning

ENCODING = "structured_habitability_tool_v1"
CONTROL_ENCODING = "semantic_habitability_tool_v1"
COUNTS = {"gate": 100, "train": 4, "calibration": 2, "development": 2}
BUDGET = {
    "epochs": 2,
    "hidden_size": 16,
    "cpu_threads": 1,
    "seed": 0,
    "sequence_length": 8,
    "max_episode_actions": 128,
    "optimizer_updates": 32,
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


def content_identity(calculator, cases, graph):
    for left, right in itertools.combinations(cases.values(), 2):
        for key in ("seed", "case_id"):
            if {c[key] for c in left} & {c[key] for c in right}:
                raise ValueError("Habitability case split overlap")
        if {TEMPLATES[c["split"]] for c in left} & {TEMPLATES[c["split"]] for c in right}:
            raise ValueError("Habitability template split overlap")
        for quantity in ("stellar_luminosity", "orbital_radius", "albedo"):
            numbers = lambda rows, quantity=quantity: {
                m["value"] for c in rows for m in c["measurements"].values() if m["kind"] == quantity
            }
            if numbers(left) & numbers(right):
                raise ValueError("Habitability continuous numeric split overlap")
    return {
        "task": "habitability_calculations",
        "scope": SCOPE,
        "calculation_mode": "local_tool_assisted",
        "calculation_backend": "local",
        "knowledge_pack_hash": calculator.pack.checksum,
        "graph_hash": graph_fingerprint(graph),
        "observation_encoding": ENCODING,
        "control_encoding": CONTROL_ENCODING,
        "required_fields": list(FIELDS),
        "validation_scope": calculator.pack.validation_scope,
        "course_acceptance_passed": False,
        "expected_provenance": "inverse_generated_energy_balance_not_course_grading",
        "categorical_overlap": "The four supplied greenhouse increments are shared labels, not held-out continuous measurements.",
        "splits": {
            split: {
                "count": len(rows),
                "sha256": source_hash(rows),
                "seeds": [c["seed"] for c in rows],
                "case_ids": [c["case_id"] for c in rows],
                "instruction_hash": source_hash(TEMPLATES[split]),
            }
            for split, rows in cases.items()
        },
        "final_test_status": "not_generated_or_evaluated",
    }


def collect_split(calculator, cases, directory, *, policy=None):
    directory = prepare_output_directory(directory)
    env = HabitabilityCalculationEnv(calculator, cases)
    episodes, summaries = [], []
    for case in cases:
        episode, summary = record_episode(
            env, case["seed"], policy, event_path=directory / f"{case['seed']}.events.jsonl"
        )
        choices = []
        for step in episode:
            action, observation = step["action"], step["observation"]
            target = next((c for c in observation["controls"] if c["id"] == action.get("target")), {})
            if action["kind"] == "SELECT" and target.get("label") == "Calculation":
                # Diagnostic after the episode, never fed to the learned policy.
                expected = env.expert_action(observation)
                choices.append(expected.target == action["target"] and expected.value == action["value"])
        summary.update(operation_choice_attempts=len(choices), operation_choice_correct=sum(choices))
        episodes.append(episode)
        summaries.append(summary)
        write_json(directory / "summaries.json", summaries)
        if policy is None and (
            not summary["completed"] or summary["steps"] != 28 or any(summary.get(k, 0) for k in ERRORS)
        ):
            raise ValueError("Supplied-temperature expert gate failed")
    write_json(directory / "episodes.json", episodes)
    return episodes, summaries


def run_smoke(graph, output, *, progress=lambda _: None):
    if len(graph.body_ids) != 2000 or graph.manifest.get("source_kind") != "biological":
        raise ValueError("Habitability smoke requires the real 2,000-node graph")
    started = time.perf_counter()
    directory = prepare_output_directory(output)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    calculator = HabitabilityCalculator()
    verification = calculator.verify()
    cases = {s: habitability_cases(s, n) for s, n in COUNTS.items()}
    content = content_identity(calculator, cases, graph)
    write_json(
        directory / "manifest.json",
        {
            "version": 1,
            "content": content,
            "budget": BUDGET,
            "counts": COUNTS,
            "verification": verification,
            "graph_nodes": len(graph.body_ids),
            "graph_edges": len(graph.edge_src),
            "parent_checkpoint": None,
            "training_stage": "random_initialization_smoke",
        },
    )
    write_json(directory / "status.json", {"status": "collecting"})
    try:
        episodes = {}
        for split, rows in cases.items():
            episodes[split], summaries = collect_split(calculator, rows, directory / f"expert-{split}")
            write_json(directory / f"private-{split}-cases.json", rows)
            progress({"phase": "expert", "split": split, "completed": len(summaries)})
        validation = list(
            itertools.chain.from_iterable(zip(episodes["calibration"], episodes["development"]))
        )
        write_json(directory / "status.json", {"status": "training"})
        policy = new_policy(graph)
        training = train_behavioral_cloning(
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
        if training["optimizer_steps"] != BUDGET["optimizer_updates"] or not all(
            map(math.isfinite, training["losses"])
        ):
            raise ValueError("Habitability smoke violated finite-loss/update budget")
        restored, checkpoint = load_checkpoint(
            directory / "training/checkpoint.pt", graph, content_pack=content
        )
        if not all(torch.equal(v, restored.state_dict()[k]) for k, v in policy.state_dict().items()):
            raise ValueError("Habitability checkpoint reload changed parameters")
        write_json(directory / "status.json", {"status": "closed_loop_evaluation"})
        evaluations = {}
        for split in ("train", "development"):
            _, rows = collect_split(calculator, cases[split], directory / f"learned-{split}", policy=restored)
            evaluations[split] = summarize(rows, required_count=2)
            progress({"phase": "learned_closed_loop", "split": split, **evaluations[split]})
        report = {
            "task": "habitability_calculations",
            "calculation_mode": "local_tool_assisted",
            "content": content,
            "budget": BUDGET,
            "training": training,
            "closed_loop": evaluations,
            "expert_gate": {"completed": 100, "episodes": 100, "policy": "scripted_expert"},
            "checkpoint_reload_verified": True,
            "checkpoint_content_hash": checkpoint["content_pack_hash"],
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": peak_process_rss_bytes(),
            "code_smoke_passed": True,
            "learning_gate_passed": False,
            "course_acceptance_passed": False,
            "final_test_episodes": 0,
            "gate_note": "Supplied temperature practice is not gas, water-phase or habitability classification, and a smoke run is not the unseen learning gate.",
        }
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"status": "complete", "learning_gate_passed": False})
        return report
    except Exception as exc:
        write_json(directory / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise
