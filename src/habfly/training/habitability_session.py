"""Offline supplied-temperature TUI practice on recorded development cases only."""

from copy import deepcopy

from habfly.data import load_graph
from habfly.environments.habitability_calculations import HabitabilityCalculationEnv
from habfly.habitability_knowledge import HabitabilityCalculator

from .habitability_evaluation import load_validated_pilot, require_frozen_final_gate
from .planet_sequence import file_hash, read


def validate_habitability_options(options):
    if (
        options.task != "habitability_calculations"
        or options.environment != "simulator"
        or options.policy != "checkpoint"
        or options.backend != "local"
        or options.stars != 1
        or not options.graph
        or not options.checkpoint
        or not options.dataset
        or not options.habitability_evaluation
        or options.spreadsheet_config
        or options.knowledge_pack
        or options.content_pack
        or options.planet_evaluation
        or options.planet_calibration
    ):
        raise ValueError("Temperature demo requires its local promoted pilot and frozen evaluation")
    if options.checkpoint.resolve() != (options.dataset / "training/checkpoint.pt").resolve():
        raise ValueError("Temperature demo checkpoint must belong to its pilot")
    if not 17000000 <= options.seed <= 17000015:
        raise ValueError("Temperature demos reuse recorded development seeds 17000000 through 17000015")


def load_habitability_session(options):
    validate_habitability_options(options)
    checksum = file_hash(options.checkpoint)
    require_frozen_final_gate(read(options.habitability_evaluation / "report.json"), checksum)
    policy, content, cases = load_validated_pilot(options.dataset, load_graph(options.graph))
    selected = [case for case in cases["development"] if case["seed"] == options.seed]
    if len(selected) != 1:
        raise ValueError("Choose a recorded temperature development seed")
    policy.eval()
    env = HabitabilityCalculationEnv(HabitabilityCalculator(), deepcopy(selected))
    return (
        policy,
        env,
        {
            "checkpoint_sha256": checksum,
            "graph_hash": content["graph_hash"],
            "knowledge_pack_hash": content["knowledge_pack_hash"],
            "scope": content["scope"],
            "case_split": "recorded_development_reuse",
            "fresh_unseen_evaluation": False,
            "optimizer_updates": 0,
            "course_acceptance_passed": False,
            "gas_identification_learned": False,
            "water_phase_learned": False,
            "habitability_decision_learned": False,
        },
    )
