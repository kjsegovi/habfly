"""Bounded boundary curriculum and independent, immutable regression evidence."""

from habfly.environments.color_boundary import SAMPLING_RECIPE, boundary_curriculum

from .color import GRAPH, file_hash, frozen_workflow_hash, read, validate_splits
from .color_stabilization import SELECTION_RULE, _continue_color, validate_epoch_selection


def train_color_boundaries(source, output, *, graph_path=GRAPH, notify=print):
    return _continue_color(source, output, graph_path=graph_path, notify=notify, boundary=True)


def all_training_cases(directory):
    """Union of current and inherited training cases, not two disjoint splits."""
    original = read(directory / "regression.json")
    mixed = read(directory / "train.json")
    return list({c["case_id"]: c for c in original + mixed}.values())


def validate_boundary_experiment(directory, policy, reference, report):
    content = report["content"]
    config = content["boundary_curriculum"]
    budget = content["budget"]
    cap = budget["train"] * budget["epochs"]
    inherited = config.get("inherited_selected_optimizer_updates", -1)
    selected_epoch = config.get("inherited_selected_epoch", -1)
    if (
        config.get("version") != 1
        or config.get("selection_rule") != SELECTION_RULE
        or config.get("sampling_recipe") != SAMPLING_RECIPE
        or config.get("new_numeric_cases") != budget["train"] // 2
        or config.get("retained_numeric_cases") != budget["train"] // 2
        or config.get("additional_optimizer_update_cap") != cap
        or config.get("inherited_color_optimizer_updates") != 3 * cap
        or not 0 <= selected_epoch <= budget["epochs"]
        or inherited != 2 * cap + selected_epoch * budget["train"]
        or report.get("cumulative_color_optimizer_updates") != 4 * cap
        or config.get("learning_rate") != 0.003
        or config.get("frozen_workflow_sha256") != frozen_workflow_hash(policy)
        or len(report.get("epoch_history", [])) != budget["epochs"] + 1
    ):
        raise ValueError("Color boundary provenance mismatch")
    original_hashes = config["original_dataset_hashes"]
    if (
        file_hash(directory / "regression.json") != config.get("regression_sha256")
        or config["regression_sha256"] != original_hashes["train"]
        or any(content["dataset_hashes"][s] != original_hashes[s] for s in ("calibration", "development"))
    ):
        raise ValueError("Color boundary original/held-out dataset mismatch")
    original = read(directory / "regression.json")
    if len(original) != budget["train"] or read(directory / "train.json") != boundary_curriculum(
        reference, original
    ):
        raise ValueError("Color boundary curriculum does not match the deterministic recipe")
    validate_splits(
        {"train": all_training_cases(directory)}
        | {s: read(directory / f"{s}.json") for s in ("calibration", "development")}
    )
    validate_epoch_selection(
        directory, policy, report, inherited_retained=inherited, splits=("train", "development", "regression")
    )
    baseline = report["epoch_history"][0]["regression"]["completed"]
    regression = report["regression"]
    passed = regression["completed"] >= baseline and not any(
        regression[k] for k in ("invalid_actions", "reference_errors", "infrastructure_failures")
    )
    if (
        baseline != config.get("regression_minimum_completed")
        or baseline != report.get("parent_train_completed")
        or report.get("regression_gate_passed") != passed
        or (report.get("ready_for_final_test") and not passed)
    ):
        raise ValueError("Color original-case regression gate mismatch")
