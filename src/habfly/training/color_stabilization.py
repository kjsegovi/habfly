"""One lower-rate continuation, selected on development only; final stays sealed."""

import hashlib
import json
import math
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path

import torch

from habfly.color_reference import validate_color_reference
from habfly.contracts import Action, Observation
from habfly.environments.color import SCOPE, color_cases

from .checkpoints import load_checkpoint, save_checkpoint, source_hash
from .color import (
    GRAPH,
    color_calibration,
    evaluate,
    file_hash,
    finish_color_training,
    frozen_workflow_hash,
    load_color_experiment,
    read,
    record,
    refinement_features,
    validate_splits,
    write,
)
from .curriculum import Example
from .frozen_text import frozen_text_cache

SELECTION_RULE = "fewest_development_errors_then_most_completed_then_lowest_color_nll_earliest_tie"


def head_hash(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.color_head.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def selection_key(epoch):
    """No training/calibration/test metrics influence checkpoint selection."""
    dev = epoch["development"]
    nll = epoch["fixed_weight_losses"]["development"]["color_nll"]
    if not math.isfinite(nll):
        raise ValueError("Nonfinite development selection loss")
    errors = sum(dev[k] for k in ("invalid_actions", "reference_errors", "infrastructure_failures"))
    return -errors, dev["completed"], -nll


def stabilize_color(source, output, *, graph_path=GRAPH, notify=print):
    """Continue a single ordinal refinement, at 0.003, for one unchanged budget."""
    return _continue_color(source, output, graph_path=graph_path, notify=notify)


def _continue_color(
    source, output, *, graph_path=GRAPH, notify=print, boundary=False, isolation=False, ordering=False
):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new color experiment directory")
    policy, reference, parent = load_color_experiment(source, graph_path)
    if ordering and (
        boundary
        or isolation
        or policy.color_input != "selected_graph_v1"
        or not parent["content"].get("selected_input")
        or parent["content"].get("readout_ordering")
    ):
        raise ValueError("Ordering requires one isolated color parent, not an ordering experiment")
    if isolation and (
        boundary or policy.color_input != "workflow_v1" or not parent["content"].get("boundary_curriculum")
    ):
        raise ValueError("Selected-input training requires one legacy boundary parent")
    if boundary and (
        policy.color_readout != "ordinal_v2"
        or not parent["content"].get("stabilization")
        or parent["content"].get("boundary_curriculum")
    ):
        raise ValueError("Boundary training requires one stabilized color parent, not a boundary experiment")
    if (
        not boundary
        and not isolation
        and not ordering
        and (policy.color_readout != "ordinal_v2" or parent["content"].get("stabilization"))
    ):
        raise ValueError("Stabilization requires one ordinal refinement, not a stabilization")
    if (source / "final").exists():
        raise ValueError("Cannot stabilize an experiment whose final test was already opened")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    started = time.monotonic()
    budget = parent["content"]["budget"]
    cap = budget["train"] * budget["epochs"]
    golden = validate_color_reference(reference)
    cases = {s: read(source / f"{s}.json") for s in ("train", "calibration", "development")}
    if any(len(rows) != budget[s] for s, rows in cases.items()):
        raise ValueError("Color dataset counts differ from budget")
    expert = color_cases("expert", 100, reference)
    validate_splits({**cases, "expert": expert})
    if boundary:
        from habfly.environments.color_boundary import SAMPLING_RECIPE, boundary_curriculum

        cases["regression"] = cases["train"]
        cases["train"] = boundary_curriculum(reference, cases["regression"])
        validate_splits({s: rows for s, rows in cases.items() if s != "regression"} | {"expert": expert})
    if isolation or ordering:
        from .color_boundary import all_training_cases

        cases["regression"] = read(source / "regression.json")
        validate_splits(
            {"train": all_training_cases(source), **{s: cases[s] for s in ("calibration", "development")}}
        )
        policy.color_input = "selected_graph_v1"
    for case in expert:
        _, summary = record(reference, case)
        if not summary["completed"] or summary["steps"] != 3:
            raise ValueError("Color expert failed deterministic validation")
    episodes = [
        [
            Example(Observation.model_validate(e["observation"]), Action.model_validate(e["action"]))
            for e in row
        ]
        for row in read(source / "demonstrations.json")
    ]
    if boundary:
        episodes = []
        for case in cases["train"]:
            examples, summary = record(reference, case)
            if not summary["completed"] or summary["steps"] != 3:
                raise ValueError("Boundary curriculum expert failed validation")
            episodes.append(examples)
    if len(episodes) != budget["train"] or any(len(e) != 3 for e in episodes):
        raise ValueError("Color demonstration count differs from budget")
    workflow = frozen_workflow_hash(policy)
    policy.requires_grad_(False)
    features = {
        "train": refinement_features(policy, episodes),
        "development": refinement_features(policy, [record(reference, c)[0] for c in cases["development"]]),
    }
    if boundary or isolation or ordering:
        features["regression"] = refinement_features(
            policy, [record(reference, c)[0] for c in cases["regression"]]
        )
    if isolation:
        # The feature space changed. Refit ONLY on training; retain learned head
        # parameters as a warm start, but discard incompatible optimizer moments.
        policy.color_head.normalization_fitted.fill_(False)
        policy.color_head.fit_normalization(features["train"][0])
    if ordering:
        from .color_ordering import (
            initialize_ordered_readout,
            ordering_diagnostic,
            projection_hash,
            training_wavelengths,
        )

        wavelengths = training_wavelengths(episodes)
        ordering_before = ordering_diagnostic(policy.color_head, features["train"][0], wavelengths)
        initialization = initialize_ordered_readout(policy.color_head, *features["train"], wavelengths)
    if isolation or ordering:
        from .color_isolation import invariance_report

        invariance_before = invariance_report(policy, episodes)
    # Development statistics never fit normalization; freeze it during training.
    normalization = {n: t.clone() for n, t in policy.color_head.named_buffers()}
    policy.color_head.requires_grad_(True)
    if ordering:
        policy.color_head.score.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in policy.color_head.parameters() if p.requires_grad], lr=0.003)
    payload = torch.load(source / "training/checkpoint.pt", map_location="cpu", weights_only=True)
    if not payload.get("optimizer"):
        raise ValueError("Color stabilization requires parent optimizer state")
    if not isolation and not ordering:
        optimizer.load_state_dict(payload["optimizer"])
    inherited_retained = parent.get(
        "selected_checkpoint_cumulative_color_optimizer_updates", parent["cumulative_color_optimizer_updates"]
    )
    expected_steps = inherited_retained - cap if boundary else cap
    if (
        not isolation
        and not ordering
        and (
            any(float(state["step"]) != expected_steps for state in optimizer.state.values())
            or not optimizer.state
        )
    ):
        raise ValueError("Unexpected parent color-head optimizer steps")
    for group in optimizer.param_groups:
        group["lr"] = 0.003
    output.mkdir(parents=True, exist_ok=False)
    for name in [
        *(f"{s}.json" for s in cases if not boundary or s not in {"train", "regression"}),
        *([] if boundary else ["demonstrations.json"]),
    ]:
        shutil.copyfile(source / name, output / name)
    if boundary:
        shutil.copyfile(source / "train.json", output / "regression.json")
        write(output / "train.json", cases["train"])
        write(
            output / "demonstrations.json",
            [
                [
                    {
                        "observation": e.observation.model_dump(mode="json"),
                        "action": e.action.model_dump(mode="json"),
                    }
                    for e in episode
                ]
                for episode in episodes
            ],
        )
    operation = (
        "readout_ordering"
        if ordering
        else "selected_input"
        if isolation
        else "boundary_curriculum"
        if boundary
        else "stabilization"
    )
    content = {
        **parent["content"],
        operation: {
            "version": 1,
            "parent_color_checkpoint_sha256": parent["checkpoint_sha256"],
            "parent_color_content_hash": source_hash(parent["content"]),
            "inherited_color_optimizer_updates": parent["cumulative_color_optimizer_updates"],
            "additional_optimizer_update_cap": cap,
            "learning_rate": 0.003,
            "optimizer_state": "resume_parent_moments_and_steps_change_learning_rate_only",
            "frozen_workflow_sha256": workflow,
            "trainable_parameters": [n for n, p in policy.named_parameters() if p.requires_grad],
            "normalization": "inherited_train_only_not_refitted",
            "selection_rule": SELECTION_RULE,
            "baseline_candidate": True,
            "development_usage": "checkpoint_selection_not_independent_final_evaluation",
            "new_numeric_cases": 0,
        },
    }
    if boundary:
        content[operation].update(
            new_numeric_cases=budget["train"] // 2,
            retained_numeric_cases=budget["train"] // 2,
            sampling_recipe=SAMPLING_RECIPE,
            regression_sha256=file_hash(output / "regression.json"),
            original_dataset_hashes=parent["content"]["dataset_hashes"],
            original_demonstrations_sha256=parent["content"]["demonstrations_sha256"],
            inherited_selected_optimizer_updates=inherited_retained,
            inherited_selected_epoch=parent["selected_epoch"],
            regression_minimum_completed=parent["train"]["completed"],
            boundary_expert_cases_passed=len(episodes),
            optimizer_state="resume_selected_parent_moments_and_steps",
        )
        content["dataset_hashes"] = {
            s: file_hash(output / f"{s}.json") for s in ("train", "calibration", "development")
        }
        content["demonstrations_sha256"] = file_hash(output / "demonstrations.json")
    if isolation:
        content["color_input"] = "selected_graph_v1"
        content[operation].update(
            optimizer_state="reset_moments_for_changed_feature_space",
            head_initialization="parent_parameters_with_refitted_training_normalization",
            normalization="selected_graph_training_only_refitted_once_then_frozen",
            inherited_selected_optimizer_updates=inherited_retained,
            regression_sha256=file_hash(output / "regression.json"),
            regression_minimum_completed=parent["regression"]["completed"],
            parent_dataset_hashes=parent["content"]["dataset_hashes"],
            parent_demonstrations_sha256=parent["content"]["demonstrations_sha256"],
            recurrent_state="fresh_zero_state_per_color_choice_shared_frozen_graph_core",
        )
    if ordering:
        # Preserve the strongest inherited original-case guard, not the failed
        # immediate parent's lower score (45/64 versus the older 58/64).
        minimum = max(
            parent["regression"]["completed"],
            parent["content"]["selected_input"]["regression_minimum_completed"],
        )
        content[operation].update(
            optimizer_state="reset_for_cutpoints_and_softness_only",
            head_initialization=initialization,
            normalization="inherited_selected_graph_training_only_not_refitted",
            frozen_projection_sha256=projection_hash(policy.color_head),
            inherited_selected_optimizer_updates=inherited_retained,
            regression_sha256=file_hash(output / "regression.json"),
            regression_minimum_completed=minimum,
            parent_dataset_hashes=parent["content"]["dataset_hashes"],
            parent_demonstrations_sha256=parent["content"]["demonstrations_sha256"],
            closed_form_fits=1,
        )
    write(output / "dataset-manifest.json", content)
    policy.calibration = {"status": "uncalibrated", "scope": "epoch_candidate_before_final_calibration"}
    history, losses, components, rng = [], [], [], random.Random(0)
    best, best_head, best_optimizer = None, None, None

    def assess(epoch):
        nonlocal best, best_head, best_optimizer
        if frozen_workflow_hash(policy) != workflow or any(
            not torch.equal(normalization[n], t) for n, t in policy.color_head.named_buffers()
        ):
            raise RuntimeError("Frozen workflow or normalization changed")
        if ordering and projection_hash(policy.color_head) != initialization["projection_sha256"]:
            raise RuntimeError("Frozen color ordering projection changed")
        policy.eval()
        directory = output / "epochs" / f"epoch-{epoch:03d}"
        scores = {s: evaluate(policy, reference, cases[s], directory / f"{s}-rollouts") for s in features}
        fixed_losses = {}
        with torch.no_grad():
            for split, (x, y) in features.items():
                values = policy.color_head.loss(x, y)
                if not all(torch.isfinite(v) for v in values):
                    raise RuntimeError("Nonfinite fixed-weight color evaluation loss")
                fixed_losses[split] = dict(zip(("objective", "color_nll", "ordinal_bce"), map(float, values)))
        entry = {
            "epoch": epoch,
            "optimizer_updates": len(losses),
            "fixed_weight_losses": fixed_losses,
            **{s: {k: v for k, v in result.items() if k != "rows"} for s, result in scores.items()},
            "head_sha256": head_hash(policy),
        }
        checkpoint = directory / "training/checkpoint.pt"
        save_checkpoint(
            checkpoint,
            policy,
            stage=SCOPE,
            seed=0,
            optimizer=optimizer,
            content_pack=content,
            evaluation=entry,
        )
        entry["checkpoint_sha256"] = file_hash(checkpoint)
        history.append(entry)
        if best is None or selection_key(entry) > selection_key(best):
            best = entry
            best_head = deepcopy(policy.color_head.state_dict())
            best_optimizer = deepcopy(optimizer.state_dict())
        write(directory / "report.json", entry)
        status = {
            "epoch": epoch,
            "epochs": budget["epochs"],
            "optimizer_updates": len(losses),
            "train_completed": scores["train"]["completed"],
            "development_completed": scores["development"]["completed"],
            "fixed_train_loss": fixed_losses["train"]["objective"],
            "selected_epoch": best["epoch"],
            "elapsed_seconds": time.monotonic() - started,
        }
        write(output / "status.json", status)
        notify(json.dumps(status))

    notify(
        f"Color head at 0.003; cap {cap} additional updates; baseline plus every epoch saved; mode={operation}."
    )
    assess(0)
    x, y = features["train"]
    for epoch in range(1, budget["epochs"] + 1):
        order = list(range(len(x)))
        rng.shuffle(order)
        for index in order:
            loss, nll, ordinal = policy.color_head.loss(x[index : index + 1], y[index : index + 1])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite color stabilization loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.color_head.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            components.append({"color_nll": float(nll.detach()), "ordinal_bce": float(ordinal.detach())})
        assess(epoch)
    policy.color_head.load_state_dict(best_head)
    optimizer.load_state_dict(best_optimizer)
    if file_hash(source / "training/checkpoint.pt") != parent["checkpoint_sha256"]:
        raise RuntimeError("Parent checkpoint changed during stabilization")
    with frozen_text_cache(policy):
        calibration = color_calibration(policy, [record(reference, c)[0] for c in cases["calibration"]])
    regression_report = {}
    if boundary or isolation or ordering:
        regression = evaluate(policy, reference, cases["regression"], output / "regression-rollouts")
        minimum = parent["regression"]["completed"] if isolation else parent["train"]["completed"]
        if ordering:
            minimum = content[operation]["regression_minimum_completed"]
        regression_report = {
            "regression": regression,
            "regression_gate_passed": regression["completed"] >= minimum
            and not any(
                regression[k] for k in ("invalid_actions", "reference_errors", "infrastructure_failures")
            ),
        }
    return finish_color_training(
        output,
        policy,
        policy.graph,
        reference,
        cases,
        content,
        optimizer,
        losses,
        calibration,
        golden,
        started,
        extra_report={
            "loss_components": components,
            "frozen_workflow_verified": True,
            "epoch_history": history,
            "selected_epoch": best["epoch"],
            "selected_checkpoint_optimizer_updates": best["optimizer_updates"],
            "discarded_optimizer_updates": len(losses) - best["optimizer_updates"],
            "cumulative_color_optimizer_updates": parent["cumulative_color_optimizer_updates"] + len(losses),
            "selected_checkpoint_cumulative_color_optimizer_updates": (
                inherited_retained + best["optimizer_updates"]
            ),
            "parent_train_completed": parent["train"]["completed"],
            "parent_development_completed": parent["development"]["completed"],
            **(
                {
                    "invariance_before": invariance_before,
                    "invariance_after": invariance_report(policy, episodes),
                }
                if isolation or ordering
                else {}
            ),
            **(
                {
                    "ordering_before": ordering_before,
                    "ordering_after": ordering_diagnostic(
                        policy.color_head, features["train"][0], wavelengths
                    ),
                    "closed_form_fits": 1,
                    "closed_form_fit_retained": True,
                }
                if ordering
                else {}
            ),
            **regression_report,
        },
    )


def validate_stabilization(directory, policy, report):
    """Verify selection/accounting and saved candidate identities when loading."""
    config = report["content"]["stabilization"]
    budget = report["content"]["budget"]
    cap = budget["train"] * budget["epochs"]
    history = report.get("epoch_history", [])
    if (
        config.get("version") != 1
        or config.get("selection_rule") != SELECTION_RULE
        or config.get("inherited_color_optimizer_updates") != 2 * cap
        or config.get("additional_optimizer_update_cap") != cap
        or config.get("learning_rate") != 0.003
        or config.get("frozen_workflow_sha256") != frozen_workflow_hash(policy)
        or report.get("cumulative_color_optimizer_updates") != 3 * cap
        or len(history) != budget["epochs"] + 1
    ):
        raise ValueError("Color stabilization provenance mismatch")
    validate_epoch_selection(directory, policy, report, inherited_retained=2 * cap)


def validate_epoch_selection(
    directory, policy, report, *, inherited_retained, splits=("train", "development")
):
    budget = report["content"]["budget"]
    cap = budget["train"] * budget["epochs"]
    history = report["epoch_history"]
    for epoch, entry in enumerate(history):
        checkpoint = directory / "epochs" / f"epoch-{epoch:03d}" / "training/checkpoint.pt"
        if (
            entry["epoch"] != epoch
            or entry["optimizer_updates"] != epoch * budget["train"]
            or (entry["checkpoint_sha256"] != file_hash(checkpoint))
        ):
            raise ValueError("Color epoch checkpoint mismatch")
    selected = max(history, key=selection_key)  # max retains earliest exact tie.
    if (
        report.get("selected_epoch") != selected["epoch"]
        or report.get("selected_checkpoint_optimizer_updates") != selected["optimizer_updates"]
        or report.get("selected_checkpoint_cumulative_color_optimizer_updates")
        != inherited_retained + selected["optimizer_updates"]
        or report.get("discarded_optimizer_updates") != cap - selected["optimizer_updates"]
        or head_hash(policy) != selected["head_sha256"]
        or any(report[s][k] != v for s in splits for k, v in selected[s].items())
    ):
        raise ValueError("Color selected checkpoint mismatch")
    archived, _ = load_checkpoint(
        directory / "epochs" / f"epoch-{selected['epoch']:03d}" / "training/checkpoint.pt",
        policy.graph,
        content_pack=report["content"],
    )
    if head_hash(archived) != head_hash(policy) or frozen_workflow_hash(archived) != frozen_workflow_hash(
        policy
    ):
        raise ValueError("Color selected checkpoint weights mismatch")
