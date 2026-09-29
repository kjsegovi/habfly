"""One sealed, frozen 100-case supplied-temperature evaluation, never course grading."""

import time
from pathlib import Path

import torch

from habfly.browser_assessment_actions import persist_json
from habfly.environments.habitability_calculations import SCOPE, habitability_cases
from habfly.habitability_knowledge import HabitabilityCalculator
from habfly.model import graph_fingerprint

from .checkpoints import load_checkpoint, source_hash
from .habitability_calculations import CONTROL_ENCODING, ENCODING, collect_split, content_identity
from .habitability_sequence import PILOT_BUDGET
from .planet_calculations import ERRORS, summarize
from .planet_evaluation import require_pilot_gate as _require_pilot_gate
from .planet_evaluation import verify_parent_artifacts
from .planet_sequence import file_hash, read
from .stellar import write_json
from .train import peak_process_rss_bytes, prepare_output_directory


def require_pilot_gate(report, content):
    _require_pilot_gate(report, content, scope=SCOPE)
    if report.get("budget") != PILOT_BUDGET or report.get("final_test_episodes") != 0:
        raise ValueError("Temperature final test requires its exact pilot budget and no prior final use")


def reserve_final_set(registry, checkpoint_hash, output):
    directory = Path(registry)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "habitability-supplied-temperature-final-v1.reservation.json"
    try:
        persist_json(
            path,
            {
                "checkpoint_sha256": checkpoint_hash,
                "output": str(output),
                "seed_start": 18000000,
                "episodes": 100,
                "scope": SCOPE,
            },
        )
    except FileExistsError as exc:
        raise ValueError("Temperature final set already reserved; cannot reuse as unseen cases") from exc
    return path


def load_validated_pilot(source, graph):
    """Load the exact calibrated pilot and verify its recorded evidence, offline."""
    source = Path(source)
    torch.set_num_threads(1)
    checkpoint = source / "training/checkpoint.pt"
    content = read(checkpoint.with_suffix(".pt.json"))["provenance"]["habitability_calculations"]
    parent = read(source / "report.json")
    require_pilot_gate(parent, content)
    calculator = HabitabilityCalculator()
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
        raise ValueError("Temperature final graph/pack/backend contract mismatch")
    cases = verify_parent_artifacts(
        source, content, parent, split_validator=lambda rows: content_identity(calculator, rows, graph)
    )
    if content_identity(calculator, cases, graph) != {
        k: v for k, v in content.items() if k != "sequence_diagnostic"
    }:
        raise ValueError("Temperature pilot content artifact mismatch")
    for split in ("train", "development"):
        if (
            summarize(read(source / f"learned-{split}/summaries.json"), required_count=2)
            != parent["closed_loop"][split]
        ):
            raise ValueError("Temperature pilot learned summary artifact mismatch")
    policy, saved = load_checkpoint(checkpoint, graph, content_pack=content)
    if policy.hidden_size != 16 or policy.calibration.get("status") != "calibrated":
        raise ValueError("Temperature final requires calibrated hidden-size-16 policy")
    if any(
        saved["evaluation_scores"].get(k) != parent["training"].get(k)
        for k in ("training_data_hash", "validation_data_hash", "optimizer_steps")
    ):
        raise ValueError("Temperature final checkpoint training provenance mismatch")
    return policy, content, cases


def run_final_evaluation(source, graph, output, *, registry, progress=lambda _: None):
    source = Path(source)
    checkpoint = source / "training/checkpoint.pt"
    checksum = file_hash(checkpoint)
    policy, content, cases = load_validated_pilot(source, graph)
    calculator = HabitabilityCalculator()
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    directory = prepare_output_directory(output)
    reservation = reserve_final_set(registry, checksum, directory)
    write_json(directory / "status.json", {"status": "reserved", "reservation": str(reservation)})
    started = time.perf_counter()
    try:
        test = habitability_cases("test", 100)
        content_identity(calculator, {**cases, "test": test}, graph)  # split separation before inference
        write_json(directory / "private-test-cases.json", test)
        write_json(
            directory / "manifest.json",
            {
                "task": "habitability_calculations",
                "scope": SCOPE,
                "content": content,
                "calculation_mode": "local_tool_assisted",
                "checkpoint_sha256": checksum,
                "test_case_sha256": source_hash(test),
                "optimizer_updates": 0,
                "calibration_refitted": False,
                "cpu_threads": 1,
                "final_test_episodes": 100,
                "reservation": str(reservation),
            },
        )
        _, rows = collect_split(calculator, test, directory / "learned-test", policy=policy)
        scores = summarize(rows, required_count=2)
        if file_hash(checkpoint) != checksum or any(
            not torch.equal(v, policy.state_dict()[k]) for k, v in before.items()
        ):
            raise ValueError("Frozen temperature evaluation changed source checkpoint or weights")
        passed = scores["completed"] >= 90 and all(scores[k] == 0 for k in ERRORS)
        report = {
            "task": "habitability_calculations",
            "scope": SCOPE,
            "calculation_mode": "local_tool_assisted",
            "checkpoint_sha256": checksum,
            "checkpoint_unchanged": True,
            "optimizer_updates": 0,
            "final_test_episodes": 100,
            "closed_loop": {"test": scores},
            "calibration": policy.calibration,
            "calibration_refitted": False,
            "learning_gate_passed": passed,
            "course_acceptance_passed": False,
            "gate_note": "Supplied-temperature local physics only; no gas, water-phase or habitability decision learning or browser acceptance.",
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": peak_process_rss_bytes(),
        }
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"status": "complete", "learning_gate_passed": passed})
        progress(report)
        return report
    except Exception as exc:
        write_json(
            directory / "status.json",
            {"status": "failed", "error_type": type(exc).__name__, "final_set_remains_reserved": True},
        )
        raise


def require_frozen_final_gate(report, checksum):
    scores = report.get("closed_loop", {}).get("test", {})
    if (
        report.get("scope") != SCOPE
        or report.get("checkpoint_sha256") != checksum
        or report.get("checkpoint_unchanged") is not True
        or report.get("optimizer_updates") != 0
        or report.get("final_test_episodes") != 100
        or report.get("learning_gate_passed") is not True
        or scores.get("episodes") != 100
        or scores.get("completed", 0) < 90
        or any(scores.get(key) != 0 for key in ERRORS)
    ):
        raise ValueError("Temperature demo requires the matching frozen 100-case learning gate")
