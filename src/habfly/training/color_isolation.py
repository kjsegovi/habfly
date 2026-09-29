"""Selected-measurement graph input, with unchanged workflow and bounded training."""

from copy import deepcopy

import torch

from habfly.environments.color_boundary import boundary_curriculum

from .color import GRAPH, file_hash, frozen_workflow_hash, read, validate_splits
from .color_boundary import all_training_cases
from .color_stabilization import SELECTION_RULE, _continue_color, validate_epoch_selection


def isolate_color(source, output, *, graph_path=GRAPH, notify=print):
    return _continue_color(source, output, graph_path=graph_path, notify=notify, isolation=True)


def irrelevant_context_variant(observation):
    """Counterfactual: keep selected value fixed, alter only irrelevant context."""
    obs = deepcopy(observation)
    selected = obs["calculation"]["source"]
    for key, measurement in obs["values"]["measurements"].items():
        if key != selected:
            measurement["value"] *= 1.37
    obs["values"]["measurements"] = dict(reversed(list(obs["values"]["measurements"].items())))
    obs["controls"].reverse()
    obs["instruction"] = "Changed irrelevant wording, same selected measurement."
    obs["feedback"] = ["Unrelated previous observation."]
    obs["values"]["answers"] = {"color": "IR"}
    return obs


@torch.no_grad()
def invariance_report(policy, episodes):
    changes, max_delta = 0, 0.0
    for episode in episodes:
        obs = episode[1].observation.model_dump(mode="json")
        zeros = policy.embedding.weight.new_zeros(1, policy.hidden_size)
        first = policy.color_head(policy.color_readout_features(obs, zeros))
        changed = policy.color_head(
            policy.color_readout_features(irrelevant_context_variant(obs), zeros + 100)
        )
        max_delta = max(max_delta, float((first - changed).abs().max()))
        changes += int(first.argmax() != changed.argmax())
    if max_delta != 0:
        raise RuntimeError("Selected-input invariance gate failed")
    return {
        "cases": len(episodes),
        "prediction_changes": changes,
        "max_logit_delta": max_delta,
        "split": "training_only",
        "scope": "irrelevant_values_order_wording_answers_and_workflow_pooled_state",
    }


def validate_isolation(directory, policy, reference, report):
    content = report["content"]
    config = content["selected_input"]
    budget = content["budget"]
    cap = budget["train"] * budget["epochs"]
    inherited = config.get("inherited_selected_optimizer_updates", -1)
    if (
        policy.color_input != "selected_graph_v1"
        or config.get("version") != 1
        or config.get("selection_rule") != SELECTION_RULE
        or config.get("learning_rate") != 0.003
        or config.get("additional_optimizer_update_cap") != cap
        or config.get("inherited_color_optimizer_updates") != 4 * cap
        or report.get("cumulative_color_optimizer_updates") != 5 * cap
        or not 2 * cap <= inherited <= 4 * cap
        or config.get("new_numeric_cases") != 0
        or config.get("optimizer_state") != "reset_moments_for_changed_feature_space"
        or config.get("normalization") != "selected_graph_training_only_refitted_once_then_frozen"
        or config.get("frozen_workflow_sha256") != frozen_workflow_hash(policy)
        or len(report.get("epoch_history", [])) != budget["epochs"] + 1
    ):
        raise ValueError("Color selected-input provenance mismatch")
    if (
        content["dataset_hashes"] != config.get("parent_dataset_hashes")
        or content["demonstrations_sha256"] != config.get("parent_demonstrations_sha256")
        or file_hash(directory / "regression.json") != config.get("regression_sha256")
        or config["regression_sha256"] != content["boundary_curriculum"]["regression_sha256"]
        or read(directory / "train.json")
        != boundary_curriculum(reference, read(directory / "regression.json"))
    ):
        raise ValueError("Color selected-input dataset mismatch")
    validate_splits(
        {"train": all_training_cases(directory)}
        | {s: read(directory / f"{s}.json") for s in ("calibration", "development")}
    )
    validate_epoch_selection(
        directory, policy, report, inherited_retained=inherited, splits=("train", "development", "regression")
    )
    saved = torch.load(directory / "training/checkpoint.pt", map_location="cpu", weights_only=True)
    selected = report["selected_checkpoint_optimizer_updates"]
    states = saved["optimizer"]["state"]
    if (selected and not states) or any(float(s["step"]) != selected for s in states.values()):
        raise ValueError("Color selected-input optimizer mismatch")
    baseline_head = torch.load(
        directory / "epochs/epoch-000/training/checkpoint.pt", map_location="cpu", weights_only=True
    )
    for name in ("feature_mean", "feature_scale", "normalization_fitted"):
        key = f"color_head.{name}"
        if not torch.equal(saved["model"][key], baseline_head["model"][key]):
            raise ValueError("Color selected-input normalization changed")
    regression = report["regression"]
    passed = regression["completed"] >= config["regression_minimum_completed"] and not any(
        regression[k] for k in ("invalid_actions", "reference_errors", "infrastructure_failures")
    )
    if report.get("regression_gate_passed") != passed or (report.get("ready_for_final_test") and not passed):
        raise ValueError("Color selected-input regression gate mismatch")
