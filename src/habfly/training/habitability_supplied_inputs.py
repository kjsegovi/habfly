"""One explicit offline 4-by-4 temperature check; no optimizer or final cases."""

import hashlib
import json
import socket
import time
from pathlib import Path
from unittest.mock import patch

import torch

from habfly.environments.habitability_calculations import SCOPE as LEGACY_SCOPE
from habfly.habitability_knowledge import DEFAULT_HABITABILITY_PACK
from habfly.habitability_supplied_inputs import (
    CLASSES,
    PACK_PATH,
    SCOPE,
    SuppliedTemperatureCalculator,
    SuppliedTemperatureEnv,
    adapter_manifest,
    inference_view,
    supplied_case,
)
from habfly.model import graph_fingerprint

from .checkpoints import load_checkpoint, source_hash
from .frozen_text import frozen_text_cache
from .stellar import write_json
from .train import prepare_output_directory

BUDGET = {
    "rollouts": 16,
    "numeric_cases": 4,
    "max_episode_actions": 128,
    "wall_seconds": 600,
    "cpu_threads": 1,
    "optimizer_updates": 0,
    "seed": 0,
    "final_test_episodes": 0,
    "ppo": False,
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _denied(*args, **kwargs):
    raise RuntimeError("supplied_temperature_network_or_training_forbidden")


def _identity(action):
    return {key: getattr(action, key) for key in ("kind", "target", "value", "observation_revision")}


def run_supplied_temperature_smoke(pilot, graph, output):
    """Reused development fixtures only, never class-population or native proof.

    Frozen weights are loaded with their original content identity. New adapter
    scope invalidates confidence calibration; old final-test reports/cases are
    neither opened nor promoted. The output directory is exclusive/no-restart.
    """
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    pilot = Path(pilot)
    checkpoint = pilot / "training/checkpoint.pt"
    metadata = checkpoint.with_suffix(".pt.json")
    cases_path = pilot / "private-development-cases.json"
    content = json.loads(metadata.read_bytes())["provenance"]["habitability_calculations"]
    cases = json.loads(cases_path.read_bytes())
    if len(cases) < 4 or source_hash(cases) != content["splits"]["development"]["sha256"]:
        raise ValueError("supplied_temperature_parent_development_mismatch")
    calculator = SuppliedTemperatureCalculator()
    if (
        content.get("knowledge_pack_hash") != calculator.pack.declaration["base_pack_hash"]
        or content.get("scope") != LEGACY_SCOPE
        or len(graph.body_ids) != 2000
        or graph.manifest.get("source_kind") != "biological"
    ):
        raise ValueError("supplied_temperature_parent_contract")
    package = Path(__file__).parents[1]
    paths = [
        checkpoint,
        metadata,
        cases_path,
        DEFAULT_HABITABILITY_PACK,
        PACK_PATH,
        Path(__file__),
        package / "habitability_supplied_inputs.py",
        package / "habitability_knowledge.py",
        package / "knowledge.py",
        package / "environments/habitability_calculations.py",
        package / "environments/local_stellar.py",
        *sorted((package / "model").glob("*.py")),
    ]
    source_pins = {str(path.resolve()): _sha(path) for path in paths}
    policy, manifest = load_checkpoint(checkpoint, graph, content_pack=content)
    if (
        policy.hidden_size != 16
        or policy.observation_encoding != "structured_habitability_tool_v1"
        or policy.control_encoding != "semantic_habitability_tool_v1"
        or policy.selection_mode != "measurement_result_v3"
        or graph_fingerprint(graph) != manifest["graph_hash"]
    ):
        raise ValueError("supplied_temperature_parent_model_contract")
    policy.requires_grad_(False)
    policy.calibration = {"status": "uncalibrated", "scope": SCOPE, "reason": "explicit_new_input_adapter"}
    policy.action_temperature = policy.target_temperature = 1.0
    weights = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    directory = prepare_output_directory(output)
    adapter, verification = adapter_manifest(), calculator.verify()
    rows, env = [], None
    try:
        with (
            patch.object(socket.socket, "connect", _denied),
            patch.object(socket.socket, "connect_ex", _denied),
            patch.object(socket, "create_connection", _denied),
            patch.object(torch.optim.Optimizer, "__init__", _denied),
            patch.object(torch.autograd, "backward", _denied),
            torch.inference_mode(),
            frozen_text_cache(policy),
        ):
            for original in cases[:4]:
                baseline = None
                for actual_class in CLASSES:
                    env = SuppliedTemperatureEnv(calculator, [supplied_case(original, actual_class)])
                    env.reset(seed=original["seed"])
                    hidden, steps = None, []
                    for _ in range(128):
                        if time.monotonic() - started >= 600:
                            raise ValueError("supplied_temperature_smoke_time_limit")
                        observation = env.observe()
                        authoritative = observation.model_dump(mode="json")
                        view = inference_view(
                            observation,
                            expected_adapter_sha256=adapter["sha256"],
                            expected_pack_hash=calculator.pack.checksum,
                        )
                        action, hidden, _ = policy.act(view, hidden)
                        if time.monotonic() - started >= 600:
                            raise ValueError("supplied_temperature_smoke_time_limit")
                        if not torch.isfinite(hidden).all():
                            raise ValueError("supplied_temperature_nonfinite_hidden")
                        action.action_confidence = action.target_confidence = None
                        action.calibrated = False
                        # Comparison happens AFTER policy inference and never
                        # replaces the action executed by the environment.
                        expected = env.expert_action(observation)
                        match = _identity(action) == _identity(expected)
                        _, _, terminated, truncated, info = env.step(action)
                        steps.append(
                            {
                                "observation": authoritative,
                                "inference_view_sha256": source_hash(view.model_dump(mode="json")),
                                "action": action.model_dump(mode="json"),
                                "matches_expert": match,
                                "tool_error": env.tool_error,
                                "failure_reason": info["result"].get("failure_reason"),
                            }
                        )
                        if terminated or truncated:
                            break
                    actions = [_identity(type(action).model_validate(step["action"])) for step in steps]
                    if baseline is None:
                        baseline = actions
                    rows.append(
                        {
                            "case_id": env.case["case_id"],
                            "numeric_case_id": original["case_id"],
                            "actual_star_class": actual_class,
                            "steps": len(steps),
                            "local_task_completed": bool(env.completed),
                            "exact_actions": sum(step["matches_expert"] for step in steps),
                            "same_actions_as_main_fixture": actions == baseline,
                            "metrics": dict(env.metrics),
                        }
                    )
                    write_json(
                        directory / f"rollout-{len(rows):02d}.json", {"summary": rows[-1], "steps": steps}
                    )
                    env.close()
                    env = None
        if len(rows) != 16 or any(
            not torch.equal(value, policy.state_dict()[key]) for key, value in weights.items()
        ):
            raise ValueError("supplied_temperature_budget_or_weights_changed")
        if (
            any(_sha(Path(path)) != digest for path, digest in source_pins.items())
            or graph_fingerprint(graph) != manifest["graph_hash"]
        ):
            raise ValueError("supplied_temperature_source_changed")
        report = {
            "mode": "offline_supplied_temperature_class_invariance_smoke_v1",
            "scope": SCOPE,
            "budget": BUDGET,
            "source_sha256": source_pins,
            "parent_checkpoint_sha256": _sha(checkpoint),
            "parent_content_hash": manifest["content_pack_hash"],
            "graph_hash": manifest["graph_hash"],
            "new_pack_hash": calculator.pack.checksum,
            "adapter": adapter,
            "verification": verification,
            "case_provenance": "First four existing development numeric fixtures under four supplied class labels; not new held-out data or class-population validation",
            "rows": rows,
            "local_completed": sum(row["local_task_completed"] for row in rows),
            "all_class_action_sequences_identical": all(row["same_actions_as_main_fixture"] for row in rows),
            "elapsed_seconds": time.monotonic() - started,
            "parent_unchanged": True,
            "weights_unchanged": True,
            "calibration_verified": False,
            "calibration": dict(policy.calibration),
            "learning_gate_passed": False,
            "scientific_verified": False,
            "course_acceptance_passed": False,
            "native_browser_enabled": False,
            "browser_started": False,
            "network_used": False,
            "optimizer_updates": 0,
            "final_test_episodes": 0,
            "final_test_cases_read": False,
            "habitability_decision_learned": False,
        }
        write_json(directory / "report.json", report)
        return report
    except Exception as exc:
        write_json(
            directory / "stopped.json",
            {
                "scope": SCOPE,
                "error_type": type(exc).__name__,
                "rollouts_completed": len(rows),
                "browser_started": False,
                "optimizer_updates": 0,
                "automatic_retry": False,
                "learning_gate_passed": False,
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
