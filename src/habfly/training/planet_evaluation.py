"""A separately invoked, frozen 100-case test after the bounded planet pilot.

This is supplied-measurement physics-tool evaluation, never course acceptance.
The final set is reserved once in the caller's experiment registry before opening
any of its cases. Retesting the same set is not a fresh unseen evaluation.
"""

import json
import os
import time
from pathlib import Path

import torch

from habfly.environments.planet_calculations import SCOPE, planet_cases
from habfly.model import graph_fingerprint
from habfly.planet_knowledge import PlanetCalculator

from .checkpoints import load_checkpoint, source_hash
from .planet_calculations import (
    CONTROL_ENCODING,
    ENCODING,
    ERRORS,
    collect_split,
    summarize,
    validate_separation,
)
from .planet_period import PILOT_COUNTS
from .planet_sequence import file_hash, read
from .stellar import write_json
from .train import peak_process_rss_bytes, prepare_output_directory


def require_frozen_final_gate(report, checkpoint_hash):
    scores = report.get("closed_loop", {}).get("test", {})
    if (
        report.get("scope") != SCOPE
        or report.get("checkpoint_sha256") != checkpoint_hash
        or not report.get("checkpoint_unchanged")
        or report.get("optimizer_updates") != 0
        or report.get("final_test_episodes") != 100
        or not report.get("learning_gate_passed")
        or scores.get("episodes") != 100
        or scores.get("completed", 0) < 90
        or any(scores.get(key) != 0 for key in ERRORS)
    ):
        raise ValueError("Planet transfer requires the matching frozen 100-case learning gate")


def require_pilot_gate(report, content, *, scope=SCOPE):
    if (
        report.get("profile") != "pilot"
        or content.get("scope") != scope
        or report.get("content") != content
        or report.get("budget", {}).get("additional_optimizer_updates") != 320
        or report.get("training", {}).get("optimizer_steps") != 320
        or report.get("training", {}).get("epochs") != 5
        or report.get("training", {}).get("training_episodes") != 64
        or report.get("training", {}).get("validation_episodes") != 32
        or any(content.get("splits", {}).get(s, {}).get("count") != n for s, n in PILOT_COUNTS.items())
        or not report.get("checkpoint_reload_verified")
        or not report.get("frozen_text_verified")
        or not report.get("parent_unchanged")
    ):
        raise ValueError("Final evaluation requires a verified complete fixed-budget pilot")
    for split in ("train", "development"):
        row = report.get("closed_loop", {}).get(split, {})
        if (
            row.get("episodes") != PILOT_COUNTS[split]
            or row.get("completion_rate", 0) < 0.9
            or row.get("completed", 0) / PILOT_COUNTS[split] != row.get("completion_rate")
            or row.get("numeric_answer_accuracy", 0) < 0.9
            or any(row.get(key) != 0 for key in ERRORS)
        ):
            raise ValueError("Pilot has not passed the learned completion/error gate")


def verify_parent_artifacts(source, content, report, *, split_validator=validate_separation):
    """Check actual saved cases/trajectories, not only reported success counts."""
    source = Path(source)
    cases, episodes = {}, {}
    for split, metadata in content["splits"].items():
        cases[split] = read(source / f"private-{split}-cases.json")
        if len(cases[split]) != metadata["count"] or source_hash(cases[split]) != metadata["sha256"]:
            raise ValueError("Planet parent case artifact mismatch")
        if split != "gate":
            episodes[split] = read(source / f"expert-{split}/episodes.json")
    validation = [
        ep for pair in zip(episodes["calibration"], episodes["development"], strict=True) for ep in pair
    ]
    if (
        source_hash(episodes["train"]) != report["training"]["training_data_hash"]
        or source_hash(validation) != report["training"]["validation_data_hash"]
    ):
        raise ValueError("Planet parent trajectory artifact mismatch")
    split_validator(cases)
    return cases


def reserve_final_set(registry, checkpoint_hash, output):
    registry = Path(registry)
    registry.mkdir(parents=True, exist_ok=True)
    path = registry / "planet-final-test-v1.reservation.json"
    try:
        with path.open("x") as stream:
            json.dump(
                {
                    "checkpoint_sha256": checkpoint_hash,
                    "output": str(output),
                    "seed_start": 13000000,
                    "episodes": 100,
                },
                stream,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError("Planet final set already reserved; do not reuse it as unseen test data") from exc
    return path


def run_final_evaluation(source, graph, output, *, registry, progress=None):
    torch.set_num_threads(1)
    checkpoint = Path(source) / "training/checkpoint.pt"
    content = read(checkpoint.with_suffix(".pt.json"))["provenance"]["planet_calculations"]
    parent_report = read(Path(source) / "report.json")
    require_pilot_gate(parent_report, content)
    calculator = PlanetCalculator()
    calculator.verify()
    if (
        len(graph.body_ids) != 2000
        or graph.manifest.get("source_kind") != "biological"
        or content.get("graph_hash") != graph_fingerprint(graph)
        or content.get("knowledge_pack_hash") != calculator.pack.checksum
        or content.get("calculation_backend") != "local"
        or content.get("observation_encoding") != ENCODING
        or content.get("control_encoding") != CONTROL_ENCODING
    ):
        raise ValueError("Final evaluation graph/pack/backend contract mismatch")
    parent_cases = verify_parent_artifacts(source, content, parent_report)
    policy, _ = load_checkpoint(checkpoint, graph, content_pack=content)
    if policy.hidden_size != 16:
        raise ValueError("Final checkpoint exceeds the agreed hidden-size budget")
    if policy.calibration.get("status") != "calibrated":
        raise ValueError("Final checkpoint confidence is not verified/calibrated")
    checkpoint_hash = file_hash(checkpoint)
    directory = prepare_output_directory(output)
    reservation = reserve_final_set(registry, checkpoint_hash, directory)
    write_json(directory / "status.json", {"status": "reserved", "reservation": str(reservation)})
    started = time.perf_counter()
    try:
        cases = planet_cases("test", 100)
        validate_separation({**parent_cases, "test": cases})
        write_json(directory / "private-test-cases.json", cases)
        write_json(
            directory / "manifest.json",
            {
                "task": "planet_calculations",
                "scope": SCOPE,
                "calculation_mode": "local_tool_assisted",
                "checkpoint_sha256": checkpoint_hash,
                "content": content,
                "test_case_sha256": source_hash(cases),
                "optimizer_updates": 0,
                "calibration_refitted": False,
                "cpu_threads": 1,
                "final_test_episodes": 100,
                "reservation": str(reservation),
            },
        )
        _, rows = collect_split(calculator, cases, directory / "learned-test", policy=policy)
        scores = summarize(rows)
        if file_hash(checkpoint) != checkpoint_hash:
            raise RuntimeError("Frozen evaluation modified source checkpoint")
        passed = scores["completed"] >= 90 and all(scores[k] == 0 for k in ERRORS)
        report = {
            "task": "planet_calculations",
            "scope": SCOPE,
            "calculation_mode": "local_tool_assisted",
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_unchanged": True,
            "optimizer_updates": 0,
            "final_test_episodes": 100,
            "closed_loop": {"test": scores},
            "calibration": policy.calibration,
            "calibration_refitted": False,
            "learning_gate_passed": passed,
            "course_acceptance_passed": False,
            "gate_note": "Supplied-measurement local physics only; no chart sensing, classification, habitability or browser acceptance.",
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": peak_process_rss_bytes(),
        }
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"status": "complete", "learning_gate_passed": passed})
        if progress:
            progress(report)
        return report
    except Exception as exc:
        write_json(
            directory / "status.json",
            {"status": "failed", "error_type": type(exc).__name__, "final_set_remains_reserved": True},
        )
        raise
