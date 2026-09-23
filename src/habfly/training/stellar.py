"""Recorded Sheets demonstrations, offline BC, and live closed-loop evaluation."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

from habfly.contracts import Observation, RuntimeEvent, task_completed
from habfly.environments.stellar import StellarAnalysisEnv, stellar_expert
from habfly.environments.stellar_common import make_case
from habfly.knowledge import KnowledgePack, LocalCalculator
from habfly.spreadsheet import SpreadsheetError, digest

from .train import peak_process_rss_bytes, prepare_output_directory, train_behavioral_cloning

SPLITS = {"train": 0, "calibration": 100000, "development": 200000, "test": 300000, "gate": 400000}


def make_adapter(config):
    if isinstance(config, KnowledgePack):
        return LocalCalculator(config)
    # Late import preserves optional-backend injection and avoids initializing Sheets locally.
    from habfly.spreadsheet import SpreadsheetAdapter

    return SpreadsheetAdapter(config)


def make_environment(adapter, cases):
    if isinstance(adapter, LocalCalculator):
        from habfly.environments.local_stellar import LocalStellarEnv

        return LocalStellarEnv(adapter, cases)
    return StellarAnalysisEnv(adapter, cases)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def record_episode(env, seed, policy=None, *, event_path=None):
    """Stream replay events even when a transport fails part-way through a run."""
    stream = event_path.open("x") if event_path else None
    sequence = 0

    def emit(event, payload):
        nonlocal sequence
        if stream:
            item = RuntimeEvent(event=event, sequence=sequence, run_id=f"stellar-{seed}", payload=payload)
            stream.write(item.model_dump_json() + "\n")
            stream.flush()
        sequence += 1

    trajectory = []
    state = None
    try:
        emit(
            "hello",
            {
                "protocol_version": 1,
                "task": "stellar",
                "calculation_mode": "local_tool_assisted"
                if isinstance(env.adapter, LocalCalculator)
                else "google_sheets",
                "policy": "checkpoint" if policy else "scripted_expert",
            },
        )
        emit(
            "state",
            {
                "stage": "stellar",
                "seed": seed,
                "status": "running",
                "browser_status": "not_connected",
                "calculation_mode": "local_tool_assisted"
                if isinstance(env.adapter, LocalCalculator)
                else "google_sheets",
            },
        )
        obs, _ = env.reset(seed=seed)
        emit("observation", obs)
        while True:
            if policy is None:
                action, diagnostics = (
                    (env.expert_action(obs) if hasattr(env, "expert_action") else stellar_expert(obs)),
                    None,
                )
            else:
                with torch.no_grad():
                    action, state, diagnostics = policy.act(Observation.model_validate(obs), state)
            emit(
                "action_proposed",
                {
                    **action.model_dump(mode="json"),
                    "action_source": "checkpoint" if policy else "scripted_expert",
                },
            )
            if diagnostics:
                emit("neural_activity", diagnostics)
            following, _, done, truncated, info = env.step(action)
            emit("action_result", info["result"])
            emit("observation", following)
            trajectory.append(
                {"observation": obs, "action": action.model_dump(mode="json"), "result": info["result"]}
            )
            obs = following
            if done or truncated:
                summary = {
                    "seed": seed,
                    "completed": task_completed(Observation.model_validate(obs)),
                    "steps": env.steps,
                    "reward": env.total,
                    "truncated": truncated,
                    "failure_reason": info["result"]["failure_reason"],
                    **env.metrics,
                }
                emit("episode_summary", summary)
                return trajectory, summary
    except SpreadsheetError as exc:
        summary = {
            **getattr(env, "metrics", {}),
            "seed": seed,
            "completed": False,
            "api_failures": 1,
            "infrastructure_failures": 1,
            "steps": len(trajectory),
            "failure_reason": str(exc),
            "truncated": True,
        }
        emit("error", {"message": str(exc), "type": "SpreadsheetError"})
        emit("episode_summary", summary)
        return trajectory, summary
    except (OSError, RuntimeError) as exc:
        summary = {
            **getattr(env, "metrics", {}),
            "seed": seed,
            "completed": False,
            "infrastructure_failures": 1,
            "steps": len(trajectory),
            "truncated": True,
            "failure_reason": f"runtime_failure:{type(exc).__name__}",
        }
        emit("error", {"message": summary["failure_reason"], "type": type(exc).__name__})
        emit("episode_summary", summary)
        return trajectory, summary
    finally:
        if stream:
            stream.close()


def collect_demonstrations(config, output, counts, *, seed=0, adapter=None, graph_hash=None):
    if (
        seed < 0
        or seed >= 100000
        or any(n < 1 or seed + n > 100000 for n in counts.values())
        or seed + 100 > 100000
    ):
        raise ValueError("Use seed/count below 100000 to retain disjoint split ranges")
    if set(counts) != set(SPLITS) - {"gate"}:
        raise ValueError("Collect all four separate dataset splits")
    directory = prepare_output_directory(output)
    owned = adapter is None
    adapter = adapter or make_adapter(config)
    started = time.perf_counter()
    reports, files = {}, {}
    identity = config.content_identity()
    write_json(directory / "status.json", {"status": "collecting", "content": identity})
    try:
        verification = adapter.verify()
        # Always pass the 100-case live expert gate before publishing a dataset.
        for split, count in {"gate": 100, **counts}.items():
            cases, episodes, summaries = [], [], []
            for index in range(count):
                case = make_case(SPLITS[split] + seed + index, split)
                case["expected"] = (
                    adapter.reference_answers(case["inputs"], case["star_class"])
                    if isinstance(adapter, LocalCalculator)
                    else adapter.calculate(case["inputs"])
                )
                env = make_environment(adapter, [case])
                episode, summary = record_episode(
                    env, case["seed"], event_path=directory / f"{split}-{index}.events.jsonl"
                )
                summaries.append(summary)
                print(
                    f"stellar {split}: {index + 1}/{count} cases; completed={summary['completed']}",
                    file=sys.stderr,
                )
                # Progress survives interruption, and failures are retained.
                write_json(directory / f"{split}-report.json", summaries)
                if not summary["completed"] or summary.get("invalid_actions"):
                    raise ValueError("stellar_expert_gate_failed")
                cases.append(case)
                episodes.append(episode)
            payload = {"cases": cases, "episodes": episodes}
            path = directory / f"{split}.json"
            write_json(path, payload)
            files[split] = {
                "file": path.name,
                "sha256": digest(payload),
                "count": count,
                "seeds": [c["seed"] for c in cases],
                "case_ids": [c["case_id"] for c in cases],
            }
            reports[split] = {"completed": count, "episodes": count}
        manifest = {
            "version": 1,
            "content": identity,
            "splits": files,
            "expert_gate_passed": True,
            "graph_hash": graph_hash,
            "verification": verification,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "status.json", {"status": "complete", "reports": reports})
        return manifest
    except Exception as exc:
        write_json(
            directory / "status.json",
            {
                "status": "incomplete",
                "failure_reason": str(exc) if isinstance(exc, SpreadsheetError) else type(exc).__name__,
                "completed_splits": list(files),
                "note": "Not usable for training; inspect recorded failure events.",
            },
        )
        raise
    finally:
        if owned:
            adapter.close()


def load_dataset(path):
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("content", {}).get("diagnostic"):
        raise ValueError("Diagnostic datasets cannot be used for full stellar training/evaluation")
    if manifest.get("version") != 1 or not manifest.get("expert_gate_passed"):
        raise ValueError("Dataset has not passed the stellar expert gate")
    mode = manifest["content"].get("calculation_mode")
    if mode not in ("google_sheets", "local_tool_assisted"):
        raise ValueError("Unknown calculation mode")
    data, all_seeds, all_ids = {}, set(), set()
    for split in SPLITS:
        item = manifest["splits"][split]
        if item["file"] != f"{split}.json":
            raise ValueError("Unexpected dataset split path")
        payload = json.loads((directory / item["file"]).read_text())
        if digest(payload) != item["sha256"]:
            raise ValueError("Dataset split hash mismatch")
        seeds = {c["seed"] for c in payload["cases"]}
        ids = {c["case_id"] for c in payload["cases"]}
        if seeds & all_seeds or ids & all_ids or len(seeds) != item["count"] or len(ids) != item["count"]:
            raise ValueError("Dataset splits overlap or contain duplicate cases")
        if len(payload["episodes"]) != item["count"]:
            raise ValueError("Dataset episode count mismatch")
        for case, episode in zip(payload["cases"], payload["episodes"]):
            surface = "calculation" if mode == "local_tool_assisted" else "spreadsheet"
            if any(not step["observation"].get(surface) for step in episode):
                raise ValueError("Demonstration backend mismatch")
            if (
                case["split"] != split
                or not episode
                or not episode[-1]["result"]["observation"]["progress"].get("task_completed")
            ):
                raise ValueError("Invalid or unfinished demonstration")
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in case["expected"].values()):
                raise ValueError("Invalid grading reference")
        all_seeds.update(seeds)
        all_ids.update(ids)
        data[split] = payload
    if manifest["splits"]["gate"]["count"] != 100:
        raise ValueError("The expert gate requires 100 cases")
    return manifest, data


def training_content(manifest):
    return {
        **manifest["content"],
        "dataset_manifest_hash": digest(manifest),
        "dataset_splits": manifest["splits"],
    }


def train_stellar(dataset, graph, output, *, epochs=5, hidden_size=16, seed=0, expected_content=None):
    from habfly.model import graph_fingerprint

    manifest, data = load_dataset(dataset)
    if expected_content is not None and manifest["content"] != expected_content:
        raise ValueError("Dataset backend or knowledge pack mismatch; collect a new dataset")
    if manifest.get("graph_hash") and manifest["graph_hash"] != graph_fingerprint(graph):
        raise ValueError("Dataset profile graph mismatch")
    calibration, development = data["calibration"]["episodes"], data["development"]["episodes"]
    if len(calibration) != len(development):
        raise ValueError("Calibration and development must have equal episode counts")
    # Existing BC partitions validation by alternate whole episodes.
    validation = [episode for pair in zip(calibration, development) for episode in pair]
    return train_behavioral_cloning(
        data["train"]["episodes"],
        graph,
        output,
        validation_episodes=validation,
        epochs=epochs,
        hidden_size=hidden_size,
        seed=seed,
        content_pack=training_content(manifest),
    )


def evaluate_stellar(policy, config, dataset, output, *, split="test", runs=None, seed=None, adapter=None):
    manifest, data = load_dataset(dataset)
    if config.content_identity() != manifest["content"]:
        raise ValueError("Calculation backend and dataset content mismatch")
    if split not in ("development", "test"):
        raise ValueError("Evaluate only development or test split")
    cases = data[split]["cases"]
    if seed is not None:
        cases = [c for c in cases if c["seed"] >= seed]
    if runs is not None:
        if runs < 1 or runs > len(cases):
            raise ValueError("Requested runs exceed collected unseen cases")
        cases = cases[:runs]
    if not cases:
        raise ValueError("No collected cases match requested evaluation")
    directory = prepare_output_directory(output)
    owned = adapter is None
    adapter = adapter or make_adapter(config)
    started = time.perf_counter()
    summaries = []
    try:
        adapter.verify()
        for case in cases:
            env = make_environment(adapter, [case])
            trajectory, summary = record_episode(
                env, case["seed"], policy, event_path=directory / f"{case['seed']}.events.jsonl"
            )
            write_json(directory / f"{case['seed']}.trajectory.json", trajectory)
            summaries.append(summary)
            write_json(directory / "episodes.json", summaries)
            if summary.get("api_failures") or summary.get("infrastructure_failures"):
                break  # Do not continue writing after any transport/safety failure.
    except SpreadsheetError as exc:
        summaries.append(
            {
                "completed": False,
                "api_failures": 1,
                "infrastructure_failures": 1,
                "failure_reason": str(exc),
                "invalid_actions": 0,
            }
        )
    finally:
        if owned:
            adapter.close()
    totals = {
        key: sum(e.get(key, 0) for e in summaries)
        for key in (
            "invalid_actions",
            "api_failures",
            "input_attempts",
            "input_correct",
            "copy_attempts",
            "copy_correct",
            "unit_attempts",
            "unit_correct",
            "numeric_fields_correct",
            "required_fields",
            "units_correct",
            "calculation_attempts",
            "calculation_correct",
            "tool_errors",
            "infrastructure_failures",
        )
    }
    report = {
        "task": "stellar",
        "policy": "learned_local_tool_assisted"
        if isinstance(config, KnowledgePack)
        else "learned_spreadsheet_assisted",
        "calculation_mode": manifest["content"]["calculation_mode"],
        "split": split,
        "requested_episodes": len(cases),
        "episodes": summaries,
        **totals,
        "completion_rate": sum(e["completed"] for e in summaries) / len(cases),
        "calibration": policy.calibration,
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": peak_process_rss_bytes(),
        "content": training_content(manifest),
    }
    for metric in ("input", "copy", "unit", "calculation"):
        report[f"{metric}_accuracy"] = totals[f"{metric}_correct"] / max(1, totals[f"{metric}_attempts"])
    report["numeric_field_accuracy"] = totals["numeric_fields_correct"] / max(1, totals["required_fields"])
    report["final_unit_accuracy"] = totals["units_correct"] / max(1, totals["required_fields"])
    report["gate_passed"] = (
        split == "test"
        and len(cases) >= 100
        and report["completion_rate"] >= 0.9
        and not totals["invalid_actions"]
        and not totals["api_failures"]
        and not totals["infrastructure_failures"]
    )
    write_json(directory / "report.json", report)
    return report
