"""Gated, offline manual demonstration on recorded development cases only."""

from copy import deepcopy

import torch

from habfly.data import load_graph
from habfly.environments.planet_calculations import PlanetCalculationEnv
from habfly.model import graph_fingerprint
from habfly.planet_knowledge import PlanetCalculator

from .checkpoints import load_checkpoint
from .planet_calculations import CONTROL_ENCODING, ENCODING
from .planet_evaluation import require_frozen_final_gate, require_pilot_gate, verify_parent_artifacts
from .planet_sequence import file_hash, read


def validate_planet_options(options):
    if (
        options.task != "planet_calculations"
        or options.environment != "simulator"
        or options.policy != "checkpoint"
        or options.backend != "local"
        or options.stars != 1
        or not options.graph
        or not options.checkpoint
        or not options.dataset
        or not options.planet_evaluation
        or options.spreadsheet_config
        or options.knowledge_pack
        or options.content_pack
    ):
        raise ValueError("Planet demo requires a local graph, promoted pilot and matching frozen evaluation")
    if options.checkpoint.resolve() != (options.dataset / "training/checkpoint.pt").resolve():
        raise ValueError("Planet demo checkpoint must belong to its pilot")
    if not 12000000 <= options.seed <= 12000015:
        raise ValueError("Planet manual demos reuse development seeds 12000000 through 12000015 only")


def load_planet_session(options):
    validate_planet_options(options)
    torch.set_num_threads(1)
    checksum = file_hash(options.checkpoint)
    content = read(options.checkpoint.with_suffix(".pt.json"))["provenance"]["planet_calculations"]
    report = read(options.dataset / "report.json")
    require_pilot_gate(report, content)
    require_frozen_final_gate(read(options.planet_evaluation / "report.json"), checksum)
    cases = verify_parent_artifacts(options.dataset, content, report)["development"]
    selected = [case for case in cases if case["seed"] == options.seed]
    if len(selected) != 1:
        raise ValueError("Choose a recorded planet development seed, 12000000 through 12000015")
    graph, calculator = load_graph(options.graph), PlanetCalculator()
    calculator.verify()
    if (
        len(graph.body_ids) != 2000
        or graph.manifest.get("source_kind") != "biological"
        or graph_fingerprint(graph) != content.get("graph_hash")
        or calculator.pack.checksum != content.get("knowledge_pack_hash")
        or content.get("calculation_backend") != "local"
        or content.get("observation_encoding") != ENCODING
        or content.get("control_encoding") != CONTROL_ENCODING
    ):
        raise ValueError("Planet demo graph, pack or encoding mismatch")
    policy, manifest = load_checkpoint(options.checkpoint, graph, content_pack=content)
    refresh = None
    if options.planet_calibration is not None:
        from .planet_recalibration import apply_refreshed_calibration

        refresh = apply_refreshed_calibration(
            policy,
            options.planet_calibration,
            checkpoint=options.checkpoint,
            dataset=options.dataset,
            content=content,
        )
    if policy.hidden_size != 16 or policy.calibration.get("status") != "calibrated":
        raise ValueError("Planet demo requires the calibrated hidden-size-16 checkpoint")
    policy.eval()
    env = PlanetCalculationEnv(calculator, deepcopy(selected))
    provenance = {
        "checkpoint_sha256": checksum,
        "graph_hash": manifest["graph_hash"],
        "knowledge_pack_hash": calculator.pack.checksum,
        "case_split": "recorded_development_reuse",
        "fresh_unseen_evaluation": False,
        "optimizer_updates": 0,
        "course_acceptance_passed": False,
        "scope": content["scope"],
        "calibration_refresh": refresh,
    }
    return policy, env, provenance
