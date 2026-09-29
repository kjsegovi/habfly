"""Explicit frozen-weight recalibration after code changes; no new test cases.

The immutable checkpoint and historical final evaluation remain untouched.
Only recorded calibration trajectories fit temperatures. Recorded development
episodes must reproduce their original semantic action streams exactly.
"""

import hashlib
import math
import time
from pathlib import Path

import torch

from habfly.model import graph_fingerprint
from habfly.planet_knowledge import PlanetCalculator

from .checkpoints import load_checkpoint, source_hash
from .planet_calculations import ERRORS, collect_split, summarize
from .planet_evaluation import require_pilot_gate, verify_parent_artifacts
from .planet_sequence import file_hash, read
from .stellar import write_json
from .train import calibrate_policy, episodes_to_examples, prepare_output_directory


def model_source_hashes():
    root = Path(__file__).resolve().parents[1] / "model"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def semantic_actions(episodes):
    def identity(step):
        a = step["action"]
        control = next((c for c in step["observation"]["controls"] if c["id"] == a.get("target")), {})
        return {"kind": a["kind"], "label": control.get("label"), "value": a.get("value")}

    return [[identity(step) for step in ep] for ep in episodes]


def refresh_planet_calibration(source, graph, output, *, progress=lambda _: None):
    source = Path(source)
    checkpoint = source / "training/checkpoint.pt"
    checksum, code = file_hash(checkpoint), model_source_hashes()
    content = read(checkpoint.with_suffix(".pt.json"))["provenance"]["planet_calculations"]
    parent = read(source / "report.json")
    require_pilot_gate(parent, content)
    cases = verify_parent_artifacts(source, content, parent)
    calculator = PlanetCalculator()
    calculator.verify()
    if (
        len(graph.body_ids) != 2000
        or graph.manifest.get("source_kind") != "biological"
        or graph_fingerprint(graph) != content["graph_hash"]
        or calculator.pack.checksum != content["knowledge_pack_hash"]
    ):
        raise ValueError("Planet recalibration graph or pack mismatch")
    calibration = read(source / "expert-calibration/episodes.json")
    old_actions = semantic_actions(read(source / "learned-development/episodes.json"))
    if len(calibration) != 16 or len(cases["development"]) != 16 or len(old_actions) != 16:
        raise ValueError("Planet recalibration requires the recorded 16/16 splits")
    directory = prepare_output_directory(output)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    started = time.perf_counter()
    policy, _ = load_checkpoint(checkpoint, graph, content_pack=content)
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    write_json(directory / "status.json", {"status": "calibration", "optimizer_updates": 0})
    try:
        calibrate_policy(
            policy, episodes_to_examples(calibration), episode_lengths=[len(ep) for ep in calibration]
        )
        policy.calibration["scope"] = "recorded_planet_calibration_reuse_after_source_change"
        progress({"phase": "recorded_calibration", "episodes": 16, "optimizer_updates": 0})
        episodes, summaries = collect_split(
            calculator, cases["development"], directory / "development", policy=policy
        )
        summary = summarize(summaries)
        if (
            summary["completed"] != 16
            or any(summary[k] for k in ERRORS)
            or semantic_actions(episodes) != old_actions
        ):
            raise ValueError("Changed planet development behavior; cannot refresh demo calibration")
        if (
            file_hash(checkpoint) != checksum
            or model_source_hashes() != code
            or any(not torch.equal(v, policy.state_dict()[k]) for k, v in before.items())
        ):
            raise ValueError("Frozen planet checkpoint or model code changed during recalibration")
        report = {
            "version": 1,
            "mode": "frozen_recalibration_recorded_cases",
            "checkpoint_sha256": checksum,
            "content_hash": source_hash(content),
            "model_sources": code,
            "calibration_episodes_hash": source_hash(calibration),
            "development_cases_hash": source_hash(cases["development"]),
            "original_development_actions_hash": source_hash(old_actions),
            "development_episodes_sha256": file_hash(directory / "development/episodes.json"),
            "development_summaries_sha256": file_hash(directory / "development/summaries.json"),
            "development": summary,
            "action_streams_match": True,
            "action_temperature": policy.action_temperature,
            "target_temperature": policy.target_temperature,
            "calibration": policy.calibration,
            "checkpoint_unchanged": True,
            "optimizer_updates": 0,
            "final_test_episodes": 0,
            "fresh_unseen_evaluation": False,
            "course_acceptance_passed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"status": "complete", "optimizer_updates": 0})
        return report
    except Exception as exc:
        write_json(directory / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise


def apply_refreshed_calibration(policy, report_path, *, checkpoint, dataset, content):
    """Validate the complete local receipt before restoring measured confidence."""
    path, source = Path(report_path), Path(dataset)
    report = read(path)
    calibration_episodes = read(source / "expert-calibration/episodes.json")
    development_cases = read(source / "private-development-cases.json")
    old_actions = semantic_actions(read(source / "learned-development/episodes.json"))
    episodes_path, summaries_path = (
        path.parent / "development/episodes.json",
        path.parent / "development/summaries.json",
    )
    temperatures = (report.get("action_temperature"), report.get("target_temperature"))
    calibration = report.get("calibration", {})
    expected_data = source_hash(
        [e.observation.model_dump() for e in episodes_to_examples(calibration_episodes)]
    )
    if (
        report.get("version") != 1
        or report.get("mode") != "frozen_recalibration_recorded_cases"
        or report.get("checkpoint_sha256") != file_hash(checkpoint)
        or report.get("content_hash") != source_hash(content)
        or report.get("model_sources") != model_source_hashes()
        or report.get("calibration_episodes_hash") != source_hash(calibration_episodes)
        or report.get("development_cases_hash") != source_hash(development_cases)
        or report.get("original_development_actions_hash") != source_hash(old_actions)
        or report.get("development_episodes_sha256") != file_hash(episodes_path)
        or report.get("development_summaries_sha256") != file_hash(summaries_path)
        or any(type(t) not in {float, int} or not math.isfinite(t) or not 0 < t <= 100 for t in temperatures)
        or calibration.get("status") != "calibrated"
        or calibration.get("action", {}).get("temperature") != temperatures[0]
        or calibration.get("target", {}).get("temperature") != temperatures[1]
        or calibration.get("data_hash") != expected_data
        or calibration.get("scope") != "recorded_planet_calibration_reuse_after_source_change"
        or report.get("checkpoint_unchanged") is not True
        or report.get("action_streams_match") is not True
        or report.get("optimizer_updates") != 0
        or report.get("final_test_episodes") != 0
        or report.get("fresh_unseen_evaluation") is not False
    ):
        raise ValueError("Planet calibration refresh provenance mismatch")
    rows = read(summaries_path)
    summary = summarize(rows)
    if (
        len(calibration_episodes) != 16
        or len(development_cases) != 16
        or summary != report["development"]
        or summary["completed"] != 16
        or any(summary[k] for k in ERRORS)
        or semantic_actions(read(episodes_path)) != old_actions
    ):
        raise ValueError("Planet calibration refresh development verification failed")
    policy.action_temperature, policy.target_temperature = temperatures
    policy.calibration = calibration
    return {
        "report_sha256": file_hash(path),
        "mode": report["mode"],
        "recorded_development_completed": 16,
        "fresh_unseen_evaluation": False,
        "optimizer_updates": 0,
    }
