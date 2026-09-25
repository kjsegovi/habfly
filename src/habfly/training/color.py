"""Bounded offline color learning. No reference label lookup at policy inference.

Training, calibration, development, expert validation, manual and final numeric
cases/templates are separate. A failed learning gate never becomes promotion.
"""

import hashlib
import itertools
import json
import random
import shutil
import time
from pathlib import Path

import torch

from habfly.color_reference import COLOR_CONTROL, COLOR_LABELS, load_color_reference, validate_color_reference
from habfly.contracts import Action, Observation, RuntimeEvent
from habfly.data import load_graph
from habfly.environments.color import SCOPE, ColorEnv, color_cases
from habfly.model import ConnectomePolicy
from habfly.model.color_head import OrdinalColorHead

from .calibration import fit_temperature
from .checkpoints import load_checkpoint, save_checkpoint, source_hash
from .curriculum import Example
from .frozen_text import frozen_text_cache
from .train import peak_process_rss_bytes, supervised_loss

GRAPH = Path("data/processed/graphs-v2/graph-2000")
PARENT = Path("experiments/temperature-source-003/training/checkpoint.pt")
BUDGETS = {
    "smoke": {"train": 4, "epochs": 2, "calibration": 2, "development": 2},
    "pilot": {"train": 64, "epochs": 5, "calibration": 16, "development": 16},
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_workflow_hash(policy):
    """Exact learned state outside the replaceable color head, including buffers."""
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        if not name.startswith("color_head."):
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_splits(cases):
    for left, right in itertools.combinations(cases.values(), 2):
        for key in ("seed", "case_id", "instruction"):
            if {c[key] for c in left} & {c[key] for c in right}:
                raise ValueError("Color split identity/template overlap")
        if {m["value"] for c in left for m in c["measurements"].values()} & {
            m["value"] for c in right for m in c["measurements"].values()
        }:
            raise ValueError("Color split numeric overlap")


def record(reference, case, policy=None, trace=None):
    env = ColorEnv(reference, [case])
    raw, _ = env.reset(seed=case["seed"])
    observation, state, sequence = Observation.model_validate(raw), None, 0
    examples = []
    stream = Path(trace).open("x") if trace else None  # noqa: SIM115 - optional stream closed in finally

    def emit(kind, payload):
        nonlocal sequence
        if stream:
            stream.write(
                RuntimeEvent(
                    event=kind, payload=payload, sequence=sequence, run_id=case["case_id"]
                ).model_dump_json()
                + "\n"
            )
            stream.flush()
        sequence += 1

    try:
        emit(
            "hello",
            {
                "protocol_version": 1,
                "task": "color",
                "scope": SCOPE,
                "policy": "checkpoint" if policy else "scripted_expert",
                "reference_hash": reference.checksum,
            },
        )
        emit(
            "state",
            {"stage": "color", "status": "running", "seed": case["seed"], "browser_status": "not_connected"},
        )
        emit("observation", observation.model_dump(mode="json"))
        while not (env.terminated or env.truncated):
            if policy is None:
                action = env.expert_action(observation)
            else:
                with torch.no_grad():
                    action, state, activity = policy.act(observation, state)
                emit("neural_activity", {**activity, "activity_source": "checkpoint"})
            examples.append(Example(observation, action))
            emit(
                "action_proposed",
                {
                    **action.model_dump(mode="json"),
                    "action_source": "checkpoint" if policy else "scripted_expert",
                },
            )
            raw, _, _, _, info = env.step(action)
            observation = Observation.model_validate(raw)
            emit("action_result", info["result"])
            emit("observation", raw)
        summary = {
            "completed": env.completed,
            "steps": env.steps,
            "failure_reason": env.error,
            "metrics": env.metrics,
            "selected_source": env.source,
            "selected_color": env.answer,
            "scope": SCOPE,
            "browser_acceptance_passed": False,
        }
        emit("episode_summary", summary)
        emit("state", {"status": "completed" if env.completed else "stopped", "stage": "color"})
        return examples, summary
    finally:
        env.close()
        if stream:
            stream.close()


def evaluate(policy, reference, cases, directory):
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        _, summary = record(reference, case, policy, directory / f"{case['case_id']}.jsonl")
        rows.append({"case_id": case["case_id"], "expected_color": case["expected_color"], **summary})
    completed = sum(r["completed"] for r in rows)
    totals = {key: sum(r["metrics"][key] for r in rows) for key in rows[0]["metrics"]}
    result = {
        "episodes": len(rows),
        "completed": completed,
        "completion_rate": completed / len(rows),
        "input_selection_accuracy": totals["input_correct"] / max(totals["input_attempts"], 1),
        "color_selection_accuracy": totals["color_correct"] / max(totals["color_attempts"], 1),
        "per_color": {
            label: {
                "episodes": sum(r["expected_color"] == label for r in rows),
                "completed": sum(r["completed"] and r["expected_color"] == label for r in rows),
            }
            for label in COLOR_LABELS
        },
        "invalid_actions": totals["invalid_actions"],
        "reference_errors": totals["reference_errors"],
        "infrastructure_failures": 0,
        "scope": SCOPE,
        "rows": rows,
    }
    write(directory / "report.json", result)
    return result


def new_policy(graph, parent_path):
    parent, parent_manifest = load_checkpoint(parent_path, graph)
    if (
        len(graph.body_ids) != 2000
        or parent.hidden_size != 16
        or parent_manifest["model"]["architecture"] != "ConnectomePolicy"
        or parent.observation_encoding != "structured_tool_v4"
        or parent.selection_mode != "measurement_result_v3"
    ):
        raise ValueError("Color requires the reviewed 2000-node, hidden-16 temperature parent")
    parent_content = read(parent_path.parent.parent / "report.json")["content"]
    if (
        parent_content.get("scope") != "local_tool_assisted_distance_luminosity_temperature"
        or source_hash(parent_content) != parent_manifest["content_pack_hash"]
    ):
        raise ValueError("Incompatible color parent task")
    config = parent.configuration()
    config.pop("architecture")
    config.update(observation_encoding="structured_color_v1", control_encoding="semantic_color_v1")
    policy = ConnectomePolicy(graph, tokenizer=parent.tokenizer, **config)
    weights = {
        k: v
        for k, v in parent.state_dict().items()
        if not k.startswith(("tool_state_projection.", "control_projection."))
    }
    result = policy.load_state_dict(weights, strict=False)
    if result.unexpected_keys or any(
        not k.startswith(("color_", "control_projection.")) for k in result.missing_keys
    ):
        raise ValueError("Unexpected color checkpoint migration")
    trainable = (
        "color_",
        "control_projection.",
        "cell.",
        "norm.",
        "sensory_projection.",
        "feature_projection.",
        "action_head.",
        "target_query.",
    )
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith(trainable))
    return policy


def color_calibration(policy, examples):
    logits, labels = [], []
    with torch.no_grad():
        for episode in examples:
            state = None
            for example in episode:
                output = policy([example.observation], state)
                state = output.state
                target = next(c for c in example.observation.controls if c.id == example.action.target)
                if target.label == COLOR_CONTROL:
                    logits.append(policy.color_head(output.pooled)[0])
                    labels.append(COLOR_LABELS.index(example.action.value))
    return {
        **fit_temperature(torch.stack(logits), torch.tensor(labels)),
        "scope": "teacher_forced_color_option_only_not_browser_or_action_confidence",
    }


def train_color(output, *, profile="smoke", graph_path=GRAPH, parent_path=PARENT, notify=print):
    budget = BUDGETS[profile]
    output, parent_path = Path(output), Path(parent_path)
    if output.exists():
        raise FileExistsError("Choose a new color experiment directory")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    reference = load_color_reference()
    golden = validate_color_reference(reference)
    cases = {s: color_cases(s, budget[s], reference) for s in ("train", "calibration", "development")}
    expert_cases = color_cases("expert", 100, reference)
    validate_splits({**cases, "expert": expert_cases})
    for case in expert_cases:
        _, summary = record(reference, case)
        if not summary["completed"] or summary["steps"] != 3:
            raise ValueError("Color expert failed deterministic validation")
    graph = load_graph(graph_path)
    policy = new_policy(graph, parent_path)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    for split, rows in cases.items():
        write(output / f"{split}.json", rows)
    episodes = {s: [record(reference, c)[0] for c in rows] for s, rows in cases.items()}
    demonstrations = [
        [
            {"observation": e.observation.model_dump(mode="json"), "action": e.action.model_dump(mode="json")}
            for e in episode
        ]
        for episode in episodes["train"]
    ]
    write(output / "demonstrations.json", demonstrations)
    content = {
        "task": "color",
        "scope": SCOPE,
        "reference_hash": reference.checksum,
        "graph_hash": policy.graph_hash,
        "parent_checkpoint_sha256": file_hash(parent_path),
        "seed": 0,
        "profile": profile,
        "budget": budget,
        "network_required": False,
        "dataset_hashes": {s: file_hash(output / f"{s}.json") for s in cases},
        "demonstrations_sha256": file_hash(output / "demonstrations.json"),
        "ambiguous_cases": "excluded_and_fail_closed_not_counted_as_success",
        "training_wavelength_domain_nm": [100, 1800],
        "reserved_final_test": {"episodes": 100, "split": "test"},
    }
    write(output / "dataset-manifest.json", content)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.003)
    losses, rng = [], random.Random(0)
    notify(f"Golden reference and 100 expert cases passed; cap {budget['train'] * budget['epochs']} updates.")
    with frozen_text_cache(policy):
        for epoch in range(budget["epochs"]):
            order = list(range(len(episodes["train"])))
            rng.shuffle(order)
            for index in order:
                policy.train()
                state, parts = None, []
                for example in episodes["train"][index]:
                    loss, result = supervised_loss(policy, [example], state)
                    parts.append(loss)
                    state = result.state
                loss = torch.stack(parts).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite color loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            status = {
                "epoch": epoch + 1,
                "epochs": budget["epochs"],
                "optimizer_updates": len(losses),
                "loss": losses[-1],
                "elapsed_seconds": time.monotonic() - started,
            }
            write(output / "status.json", status)
            notify(json.dumps(status))
        policy.eval()
        calibration = color_calibration(policy, episodes["calibration"])
    return finish_color_training(
        output, policy, graph, reference, cases, content, optimizer, losses, calibration, golden, started
    )


def finish_color_training(
    output,
    policy,
    graph,
    reference,
    cases,
    content,
    optimizer,
    losses,
    calibration,
    golden,
    started,
    *,
    extra_report=None,
):
    policy.requires_grad_(False)
    policy.calibration = {
        "status": "uncalibrated",
        "scope": "color_action_and_browser_not_calibrated",
        "color_option": calibration,
    }
    checkpoint = output / "training/checkpoint.pt"
    save_checkpoint(checkpoint, policy, stage=SCOPE, seed=0, optimizer=optimizer, content_pack=content)
    reloaded, _ = load_checkpoint(checkpoint, graph, content_pack=content)
    scores = {
        s: evaluate(reloaded, reference, cases[s], output / f"{s}-rollouts") for s in ("train", "development")
    }
    dev = scores["development"]
    ready = (
        content["profile"] == "pilot"
        and dev["completion_rate"] >= 0.9
        and not (dev["invalid_actions"] or dev["reference_errors"])
    )
    report = {
        "scope": SCOPE,
        "content": content,
        "golden_validation": golden,
        "expert_cases_passed": 100,
        "optimizer_updates": len(losses),
        "losses": losses,
        "checkpoint_reload_verified": True,
        "checkpoint_sha256": file_hash(checkpoint),
        "calibration": calibration,
        "train": scores["train"],
        "development": dev,
        "ready_for_final_test": ready,
        "browser_eligible": False,
        "final_test_status": "not_run",
        "elapsed_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": peak_process_rss_bytes(),
        **(extra_report or {}),
    }
    write(output / "report.json", report)
    return report


def refinement_features(policy, episodes):
    """Cache only the frozen biological readout at each demonstrated color choice."""
    features, labels = [], []
    policy.eval()
    with torch.no_grad(), frozen_text_cache(policy):
        for episode in episodes:
            state, choices = None, 0
            for example in episode:
                result = policy([example.observation], state)
                state = result.state
                target = next(c for c in example.observation.controls if c.id == example.action.target)
                if target.label == COLOR_CONTROL:
                    features.append(result.pooled[0].detach().clone())
                    labels.append(COLOR_LABELS.index(example.action.value))
                    choices += 1
            if choices != 1:
                raise ValueError("Expected one demonstrated color choice per episode")
    return torch.stack(features), torch.tensor(labels)


def refine_color(source, output, *, graph_path=GRAPH, notify=print):
    """One capped head-only comparison on the original splits, with no new cases.

    This adds the profile's update budget to its linear-head parent's budget.
    It cannot recursively refine a refinement or train on final-test artifacts.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new color experiment directory")
    policy, reference, parent_report = load_color_experiment(source, graph_path)
    budget = parent_report["content"]["budget"]
    cap = budget["train"] * budget["epochs"]
    if (
        policy.color_readout != "linear_v1"
        or parent_report["optimizer_updates"] != cap
        or parent_report["content"].get("refinement")
    ):
        raise ValueError("Refinement requires a single bounded linear_v1 color parent")
    if (source / "final").exists():
        raise ValueError("Cannot refine an experiment whose final test was already opened")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    started = time.monotonic()
    golden = validate_color_reference(reference)
    cases = {s: read(source / f"{s}.json") for s in ("train", "calibration", "development")}
    if any(len(rows) != budget[s] for s, rows in cases.items()):
        raise ValueError("Color dataset counts differ from budget")
    expert_cases = color_cases("expert", 100, reference)
    validate_splits({**cases, "expert": expert_cases})
    for case in expert_cases:
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
    if len(episodes) != budget["train"] or any(len(e) != 3 for e in episodes):
        raise ValueError("Color demonstration count differs from budget")
    workflow_hash = frozen_workflow_hash(policy)
    policy.requires_grad_(False)
    features, labels = refinement_features(policy, episodes)
    policy.color_head = OrdinalColorHead(policy.hidden_size, len(COLOR_LABELS))
    policy.color_readout = "ordinal_v2"
    policy.color_head.fit_normalization(features)
    output.mkdir(parents=True, exist_ok=False)
    for name in [*(f"{s}.json" for s in cases), "demonstrations.json"]:
        shutil.copyfile(source / name, output / name)
    refinement = {
        "parent_color_checkpoint_sha256": parent_report["checkpoint_sha256"],
        "parent_color_content_hash": source_hash(parent_report["content"]),
        "inherited_color_optimizer_updates": parent_report["optimizer_updates"],
        "additional_optimizer_update_cap": cap,
        "frozen_workflow_sha256": workflow_hash,
        "trainable_parameters": [n for n, p in policy.named_parameters() if p.requires_grad],
        "feature_source": "frozen_biological_effector_pool_no_numeric_bypass",
        "feature_normalization_split": "train_only",
        "objective": "color_nll_plus_0.25_ordinal_bce_not_sequence_averaged",
        "learning_rate": 0.03,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "new_numeric_cases": 0,
    }
    content = {**parent_report["content"], "color_readout": "ordinal_v2", "refinement": refinement}
    write(output / "dataset-manifest.json", content)
    optimizer = torch.optim.AdamW(policy.color_head.parameters(), lr=refinement["learning_rate"])
    losses, components, rng = [], [], random.Random(0)
    notify(f"Frozen workflow verified; reused cases; cap {cap} additional color-head-only updates.")
    for epoch in range(budget["epochs"]):
        order = list(range(len(features)))
        rng.shuffle(order)
        for index in order:
            loss, nll, ordinal = policy.color_head.loss(
                features[index : index + 1], labels[index : index + 1]
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite color refinement loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.color_head.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            components.append({"color_nll": float(nll.detach()), "ordinal_bce": float(ordinal.detach())})
        status = {
            "epoch": epoch + 1,
            "epochs": budget["epochs"],
            "optimizer_updates": len(losses),
            "loss": losses[-1],
            "elapsed_seconds": time.monotonic() - started,
        }
        write(output / "status.json", status)
        notify(json.dumps(status))
    if frozen_workflow_hash(policy) != workflow_hash:
        raise RuntimeError("Frozen color workflow changed during refinement")
    if file_hash(source / "training/checkpoint.pt") != parent_report["checkpoint_sha256"]:
        raise RuntimeError("Parent color checkpoint changed during refinement")
    calibration_episodes = [record(reference, c)[0] for c in cases["calibration"]]
    with frozen_text_cache(policy):
        calibration = color_calibration(policy, calibration_episodes)
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
            "cumulative_color_optimizer_updates": parent_report["optimizer_updates"] + len(losses),
            "parent_train_completed": parent_report["train"]["completed"],
            "parent_development_completed": parent_report["development"]["completed"],
        },
    )


def load_color_experiment(directory, graph_path=GRAPH):
    directory = Path(directory)
    report, content = read(directory / "report.json"), read(directory / "dataset-manifest.json")
    reference = load_color_reference()
    validate_color_reference(reference)
    if (
        content != report["content"]
        or content.get("task") != "color"
        or content.get("scope") != SCOPE
        or content["reference_hash"] != reference.checksum
        or content.get("profile") not in BUDGETS
        or content["budget"] != BUDGETS[content["profile"]]
        or report["checkpoint_sha256"] != file_hash(directory / "training/checkpoint.pt")
    ):
        raise ValueError("Color experiment/reference/checkpoint mismatch")
    if set(content["dataset_hashes"]) != {"train", "development", "calibration"}:
        raise ValueError("Incomplete color dataset splits")
    for split, checksum in content["dataset_hashes"].items():
        if (
            split not in {"train", "development", "calibration"}
            or file_hash(directory / f"{split}.json") != checksum
        ):
            raise ValueError("Color dataset hash mismatch")
    if file_hash(directory / "demonstrations.json") != content["demonstrations_sha256"]:
        raise ValueError("Color demonstration hash mismatch")
    torch.set_num_threads(1)
    policy, manifest = load_checkpoint(
        directory / "training/checkpoint.pt", load_graph(graph_path), content_pack=content
    )
    if (
        manifest["model"]["observation_encoding"] != "structured_color_v1"
        or manifest["model"]["architecture"] != "ConnectomePolicy"
        or manifest["model"]["control_encoding"] != "semantic_color_v1"
        or manifest["model"]["selection_mode"] != "measurement_result_v3"
        or manifest["graph_size"] != 2000
        or policy.hidden_size != 16
        or policy.graph_hash != content["graph_hash"]
        or policy.color_readout != content.get("color_readout", "linear_v1")
    ):
        raise ValueError("Color model architecture mismatch")
    if policy.color_readout == "ordinal_v2":
        refinement = content.get("refinement", {})
        cap = content["budget"]["train"] * content["budget"]["epochs"]
        if (
            not policy.color_head.normalization_fitted
            or frozen_workflow_hash(policy) != refinement.get("frozen_workflow_sha256")
            or report.get("optimizer_updates") != cap
            or refinement.get("additional_optimizer_update_cap") != cap
            or refinement.get("inherited_color_optimizer_updates") != cap
            or report.get("cumulative_color_optimizer_updates") != 2 * cap
            or not report.get("frozen_workflow_verified")
        ):
            raise ValueError("Color refinement provenance mismatch")
    return policy, reference, report


def development_gate(report):
    dev = report["development"]
    return (
        report["content"]["profile"] == "pilot"
        and dev["episodes"] == 16
        and dev["completed"] >= 15
        and dev["invalid_actions"] == 0
        and dev["reference_errors"] == 0
    )


def test_color(directory, graph_path=GRAPH):
    directory = Path(directory)
    policy, reference, report = load_color_experiment(directory, graph_path)
    if not report["ready_for_final_test"] or not development_gate(report):
        raise ValueError("Color development gate not passed; final test remains sealed")
    cases = color_cases("test", 100, reference)
    validate_splits(
        {s: read(directory / f"{s}.json") for s in ("train", "calibration", "development")} | {"test": cases}
    )
    result = evaluate(policy, reference, cases, directory / "final")
    # Do not promote a classifier that ignores a whole narrow band.
    per_band = all(v["completed"] / v["episodes"] >= 0.8 for v in result["per_color"].values())
    eligible = (
        result["completed"] >= 90
        and per_band
        and not (result["invalid_actions"] or result["reference_errors"])
    )
    result.update(
        browser_eligible=eligible,
        checkpoint_sha256=report["checkpoint_sha256"],
        reference_hash=reference.checksum,
        final_cases_sha256=source_hash(cases),
        per_band_gate_passed=per_band,
        scope=SCOPE,
        browser_acceptance_passed=False,
    )
    write(directory / "final/report.json", result)
    return result


def require_browser_color_gate(directory, graph_path=GRAPH):
    policy, reference, report = load_color_experiment(directory, graph_path)
    final = read(Path(directory) / "final/report.json")
    cases = color_cases("test", 100, reference)
    bands = final.get("per_color", {})
    complete_bands = set(bands) == set(COLOR_LABELS) and all(
        v.get("episodes", 0) > 0 and v.get("completed", 0) / v["episodes"] >= 0.8 for v in bands.values()
    )
    if (
        not report["ready_for_final_test"]
        or not development_gate(report)
        or not final.get("browser_eligible")
        or final.get("checkpoint_sha256") != report["checkpoint_sha256"]
        or final.get("reference_hash") != reference.checksum
        or final.get("scope") != SCOPE
        or final.get("episodes") != 100
        or final.get("completed", 0) < 90
        or not final.get("per_band_gate_passed")
        or not complete_bands
        or final.get("final_cases_sha256") != source_hash(cases)
        or final.get("invalid_actions") != 0
        or final.get("reference_errors") != 0
    ):
        raise ValueError("Color browser gate not passed")
    return policy, reference
