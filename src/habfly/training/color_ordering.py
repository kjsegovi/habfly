"""Train-only biological readout initialization; never a wavelength lookup at inference."""

import hashlib
from itertools import pairwise

import torch

from habfly.color_reference import COLOR_CONTROL

from .color import GRAPH

RIDGE = 0.001


def projection_hash(head):
    digest = hashlib.sha256()
    for name, value in sorted(head.score.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def training_wavelengths(episodes):
    """Inputs from demonstrated selections, not expected answers or reference bands."""
    values = []
    for episode in episodes:
        choices = [
            e
            for e in episode
            if next(c for c in e.observation.controls if c.id == e.action.target).label == COLOR_CONTROL
        ]
        if len(choices) != 1:
            raise ValueError("Expected one training color selection")
        obs = choices[0].observation
        row = obs.values["measurements"][obs.calculation["source"]]
        value = row["value"]
        if (
            row["kind"] != "wavelength"
            or row["unit"] != "nm"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise ValueError("Invalid training wavelength")
        values.append(value)
    result = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(result).all():
        raise ValueError("Invalid training wavelength")
    return result


@torch.no_grad()
def ordering_diagnostic(head, features, wavelengths):
    scores = head.score((features - head.feature_mean) / head.feature_scale)[:, 0]
    order = wavelengths.argsort(stable=True)
    diffs = scores[order].diff()
    return {
        "cases": len(features),
        "adjacent_pairs": len(diffs),
        "score_decreases": int((diffs < 0).sum()),
        "minimum_score_delta": float(diffs.min()),
        "scope": "training_only_not_global_monotonicity_proof",
    }


@torch.no_grad()
def initialize_ordered_readout(head, features, labels, wavelengths):
    """One double-precision ridge solve, then label-derived cutpoint initialization.

    Regression predicts a standardized log wavelength FROM biological activity.
    Adjacent training secants also teach its slope, preventing broad wavelength
    ranges from dominating the fit while narrow class transitions fold backward.
    Inference still uses the unchanged ordinal head, with no numeric bypass.
    Hyperparameters are fixed before the run; no search or held-out fitting.
    """
    classes = head.raw_gaps.numel() + 2
    if (
        features.ndim != 2
        or len(features) < 2
        or labels.shape != (len(features),)
        or wavelengths.shape != (len(features),)
        or not head.normalization_fitted
        or not torch.isfinite(features).all()
        or not torch.isfinite(wavelengths).all()
        or (wavelengths <= 0).any()
        or labels.dtype != torch.long
        or (labels < 0).any()
        or (labels >= classes).any()
        or labels.unique().numel() < 2
    ):
        raise ValueError("Invalid ordered-readout training data")
    z = ((features - head.feature_mean) / head.feature_scale).double()
    log_values = wavelengths.double().log()
    if log_values.std(unbiased=False) <= 0:
        raise ValueError("Training wavelengths must vary")
    target = 4 * (log_values - log_values.mean()) / log_values.std(unbiased=False)
    centered = z - z.mean(0)
    order = log_values.argsort(stable=True)
    gaps = log_values[order].diff()
    if (gaps <= 0).any():
        raise ValueError("Training wavelengths must be distinct")
    secants = z[order].diff(dim=0) / gaps[:, None]
    slope_targets = target[order].diff() / gaps
    design = torch.cat((centered, secants))
    targets = torch.cat((target, slope_targets))
    weights = torch.linalg.solve(
        design.T @ design + RIDGE * torch.eye(z.shape[1], dtype=z.dtype, device=z.device),
        design.T @ targets,
    )
    bias = target.mean() - z.mean(0) @ weights
    head.score.weight.copy_(weights.to(head.score.weight).reshape(1, -1))
    head.score.bias.copy_(bias.to(head.score.bias).reshape(1))
    scores = head.score(z.to(features))[:, 0].double()
    diagnostic = ordering_diagnostic(head, features, wavelengths)
    if diagnostic["score_decreases"]:
        raise ValueError("Ordered-readout initialization failed training wavelength ordering")
    present = sorted(labels.unique().tolist())
    centers = {label: scores[labels == label].mean() for label in present}
    spacing = torch.stack([(centers[b] - centers[a]) / (b - a) for a, b in pairwise(present)]).median()
    cuts = []
    for k in range(classes - 1):
        left, right = [c for c in present if c <= k], [c for c in present if c > k]
        if left and right:
            lo, hi = max(left), min(right)
            lower, upper = scores[labels == lo].max(), scores[labels == hi].min()
            if upper <= lower:
                raise ValueError("Training labels conflict with wavelength ordering")
            cuts.append(lower + (upper - lower) * ((k + 0.5 - lo) / (hi - lo)))
        else:
            anchor = present[0] if not left else present[-1]
            cuts.append(centers[anchor] + (k + 0.5 - anchor) * spacing)
    cuts = torch.stack(cuts).to(head.first_cut)
    gaps = cuts.diff() - 1e-4
    if not torch.isfinite(cuts).all() or (gaps <= 0).any():
        raise ValueError("Invalid training-derived ordered cutpoints")
    head.first_cut.copy_(cuts[0])
    head.raw_gaps.copy_(gaps + torch.log(-torch.expm1(-gaps)))
    softness = (cuts.diff().min() / 4).clamp(
        min=torch.exp(torch.tensor(-5.0)), max=torch.exp(torch.tensor(3.0))
    )
    head.log_softness.copy_(softness.log())
    head.score.requires_grad_(False)
    return {
        "version": 2,
        "ridge": RIDGE,
        "closed_form_fits": 1,
        "target": "training_selected_log_wavelength_standardized_times_four",
        "cutpoints": "training_labels_nearest_class_score_gap_midpoints_missing_labels_interpolated",
        "initial_softness": "minimum_cutpoint_spacing_divided_by_four_clamped_to_existing_head_limits",
        "observed_classes": present,
        "fit_examples": len(features),
        "fit_rows": len(design),
        "training_secants": len(secants),
        "secant_weight": 1.0,
        "projection_sha256": projection_hash(head),
        "ordering": diagnostic,
    }


def order_color(source, output, *, graph_path=GRAPH, notify=print):
    from .color_stabilization import _continue_color

    return _continue_color(source, output, graph_path=graph_path, notify=notify, ordering=True)


def validate_ordering(directory, policy, reference, report):
    from habfly.environments.color_boundary import boundary_curriculum

    from .color import file_hash, frozen_workflow_hash, read, validate_splits
    from .color_boundary import all_training_cases
    from .color_stabilization import SELECTION_RULE, validate_epoch_selection

    content = report["content"]
    config = content["readout_ordering"]
    budget = content["budget"]
    cap = budget["train"] * budget["epochs"]
    inherited = config.get("inherited_selected_optimizer_updates", -1)
    init = config.get("head_initialization", {})
    if (
        policy.color_input != "selected_graph_v1"
        or config.get("version") != 1
        or config.get("selection_rule") != SELECTION_RULE
        or config.get("learning_rate") != 0.003
        or config.get("inherited_color_optimizer_updates") != 5 * cap
        or config.get("additional_optimizer_update_cap") != cap
        or report.get("cumulative_color_optimizer_updates") != 6 * cap
        or not 2 * cap <= inherited <= 5 * cap
        or config.get("new_numeric_cases") != 0
        or config.get("closed_form_fits") != 1
        or report.get("closed_form_fits") != 1
        or report.get("closed_form_fit_retained") is not True
        or init.get("closed_form_fits") != 1
        or init.get("version") != 2
        or init.get("ridge") != RIDGE
        or init.get("fit_examples") != budget["train"]
        or init.get("training_secants") != budget["train"] - 1
        or init.get("fit_rows") != 2 * budget["train"] - 1
        or init.get("ordering", {}).get("score_decreases") != 0
        or config.get("optimizer_state") != "reset_for_cutpoints_and_softness_only"
        or config.get("normalization") != "inherited_selected_graph_training_only_not_refitted"
        or set(config.get("trainable_parameters", []))
        != {"color_head.first_cut", "color_head.raw_gaps", "color_head.log_softness"}
        or config.get("frozen_workflow_sha256") != frozen_workflow_hash(policy)
        or config.get("frozen_projection_sha256") != projection_hash(policy.color_head)
        or init.get("projection_sha256") != projection_hash(policy.color_head)
        or len(report.get("epoch_history", [])) != budget["epochs"] + 1
        or report.get("ordering_after", {}).get("score_decreases") != 0
    ):
        raise ValueError("Color readout-ordering provenance mismatch")
    if (
        content["dataset_hashes"] != config.get("parent_dataset_hashes")
        or content["demonstrations_sha256"] != config.get("parent_demonstrations_sha256")
        or file_hash(directory / "regression.json") != config.get("regression_sha256")
        or config["regression_sha256"] != content["boundary_curriculum"]["regression_sha256"]
        or read(directory / "train.json")
        != boundary_curriculum(reference, read(directory / "regression.json"))
    ):
        raise ValueError("Color readout-ordering dataset mismatch")
    validate_splits(
        {"train": all_training_cases(directory)}
        | {s: read(directory / f"{s}.json") for s in ("calibration", "development")}
    )
    validate_epoch_selection(
        directory, policy, report, inherited_retained=inherited, splits=("train", "development", "regression")
    )
    saved = torch.load(directory / "training/checkpoint.pt", map_location="cpu", weights_only=True)
    baseline = torch.load(
        directory / "epochs/epoch-000/training/checkpoint.pt", map_location="cpu", weights_only=True
    )
    for name in ("feature_mean", "feature_scale", "normalization_fitted", "score.weight", "score.bias"):
        key = f"color_head.{name}"
        if not torch.equal(saved["model"][key], baseline["model"][key]):
            raise ValueError("Color ordering frozen projection/normalization mismatch")
    selected = report["selected_checkpoint_optimizer_updates"]
    optimizer = saved["optimizer"]
    if (
        sum(len(g["params"]) for g in optimizer["param_groups"]) != 3
        or (selected and len(optimizer["state"]) != 3)
        or any(float(s["step"]) != selected for s in optimizer["state"].values())
    ):
        raise ValueError("Color ordering optimizer mismatch")
    regression = report["regression"]
    minimum = config["regression_minimum_completed"]
    passed = regression["completed"] >= minimum and not any(
        regression[k] for k in ("invalid_actions", "reference_errors", "infrastructure_failures")
    )
    if (
        minimum < content["selected_input"]["regression_minimum_completed"]
        or report.get("regression_gate_passed") != passed
        or (report.get("ready_for_final_test") and not passed)
    ):
        raise ValueError("Color ordering regression gate mismatch")
