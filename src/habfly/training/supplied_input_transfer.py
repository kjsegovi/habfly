"""Separately reserved, frozen supplied-input transfer gates; never v1 acceptance.

Preparing/importing this module does not materialize evaluation values. Actual
execution requires an explicit call, claims one immutable task reservation,
then evaluates 100 new inverse-generated cases. No optimizer or browser exists
in this path. Old final cases/reports are never opened.
"""

import hashlib
import json
import math
import os
import random
import socket
import time
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch

from habfly import habitability_supplied_inputs as temperature
from habfly import planet_supplied_inputs as planet
from habfly.contracts import Action, Observation, StepResult, validate_action
from habfly.environments import habitability_calculations as old_temperature
from habfly.environments import planet_calculations as old_planet
from habfly.environments.local_stellar import LocalStellarEnv
from habfly.model import graph_fingerprint

from .checkpoints import load_checkpoint, source_hash
from .frozen_text import frozen_text_cache
from .train import peak_process_rss_bytes

SCOPE = "supplied_input_transfer_gate"
VERSION = 1
# Explicit execution-code contract: these already-installed files were exercised
# by both supplied-input development smokes. This is not a claim that the older
# planet checkpoint was trained with the later habitability-support source.
EXECUTION_MODEL_MAP_SHA256 = "e11cab2a1d350cf399433b18b7a48436b81cda3fbe4d903911a0197f464a6c12"
CLASSES = planet.CLASSES
SEED_START = {"planet": 30_000_000, "temperature": 31_000_000}
TEMPLATES = {
    "planet": "Transfer check: reconstruct the current star's four planetary quantities using the supplied physical assumptions and visible current-star measurements. Select operations and bind their inputs explicitly; reuse results, copy exact answers, and select units. Ignore reference-star distractors.",
    "temperature": "Transfer check: obtain the current terrestrial system's equilibrium and surface temperatures from its supplied luminosity, orbit, albedo and warming increment. Use explicit tool inputs and intermediate results, then copy each temperature with its unit. Ignore reference-star distractors; no habitability judgment is requested.",
}
BUDGET = {
    "episodes": 100,
    "episodes_per_class": 25,
    "cpu_threads": 1,
    "optimizer_updates": 0,
    "max_episode_actions": 128,
    "wall_seconds": 600,
    "seed": 0,
    "ppo": False,
}
FALSE_CLAIMS = (
    "learning_gate_passed",
    "course_acceptance_passed",
    "scientific_verified",
    "native_browser_enabled",
    "browser_started",
    "calibration_verified",
    "calibration_refitted",
    "network_used",
    "legacy_final_test_cases_read",
    "legacy_final_reports_used_as_authority",
    "new_training_performed",
)
PLANET_STRATA = {
    "interpretation": "Supplied UI capability magnitude stress fixtures, not stellar relations or population validation",
    "anchor_count_per_non_main_class": 5,
    "white_dwarf": {"anchor": [1, 0.01], "mass_range": [0.7, 1.3], "radius_range": [0.008, 0.02]},
    "red_giant": {"anchor": [8, 70], "mass_range": [4, 12], "radius_range": [40, 100]},
    "supergiant": {"anchor": [8, 70], "mass_range": [4, 12], "radius_range": [40, 100]},
    "non_main_radius_ratio_range": [0.02, 0.5],
    "domain": "planet_radius=109*stellar_radius*ratio; brightness_drop=100*ratio**2, with no clipping",
}
CAPABILITY_RATIONALE = {
    "experiments/non-main-planet-capability-white-dwarf-002/diagnostic-report.json": "e9dee8fcd1cccda9eabaff94b9b4cc93fcc06a9e604d15a66de2679e51955f1c",
    "experiments/non-main-planet-capability-red-giant-001/diagnostic-report.json": "74a10a7f25cb1687f58fef844b8d664abcaa5b540f1dddb31107d18f68e51d48",
    "experiments/non-main-planet-capability-supergiant-001/diagnostic-report.json": "f418fa6548686087d009833e49467560316fd032103e3429d3f0f47d1ce35cdc",
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError("supplied_transfer_" + reason)


def _json(data):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "duplicate_json_key")
            value[key] = item
        return value

    value = json.loads(data, object_pairs_hook=unique)
    # Reject nonfinite numbers anywhere, including free-form observation data.
    _digest(value)
    return value


def _config(task):
    _require(type(task) is str and task in SEED_START, "unsupported_task")
    if task == "planet":
        return planet, old_planet, planet.SuppliedPlanetCalculator, "planet_calculations"
    return (
        temperature,
        old_temperature,
        temperature.SuppliedTemperatureCalculator,
        "habitability_calculations",
    )


def schedule(task):
    """Public seed/class plan only; contains no measurement or grading values."""
    _config(task)
    return [{"seed": SEED_START[task] + index, "star_class": CLASSES[index % 4]} for index in range(100)]


def _disjoint(task, content):
    _, legacy, _, _ = _config(task)
    seeds = {row["seed"] for row in schedule(task)}
    # Original generators bound every split/offset to a 100,000-seed interval,
    # including their sealed test bands. This proves seed disjointness without
    # opening/generating any old final examples.
    _require(
        all(
            not any(start <= seed < start + 100_000 for seed in seeds)
            for start in (*old_planet.SEEDS.values(), *old_temperature.SEEDS.values())
        ),
        "legacy_seed_overlap",
    )
    splits = content.get("splits")
    _require(
        isinstance(splits, dict) and {"train", "calibration", "development"} <= splits.keys(),
        "parent_split_metadata",
    )
    for split in splits.values():
        _require(
            isinstance(split, dict)
            and isinstance(split.get("seeds"), list)
            and all(type(seed) is int for seed in split["seeds"]),
            "parent_seed_metadata",
        )
        _require(not seeds.intersection(split["seeds"]), "parent_seed_overlap")
    _require(TEMPLATES[task] not in legacy.TEMPLATES.values(), "legacy_instruction_overlap")


def _inverse_case(task, seed, actual_class):
    """Independent inverse arithmetic, not calculator-generated grading answers.

    Stellar class is supplied fixture metadata, not a claim about a realistic
    distribution of measured properties for that population.
    """
    module, legacy, _, _ = _config(task)
    rng = random.Random(seed)
    if task == "planet":
        stellar_mass, stellar_radius = rng.uniform(0.5, 2.5), rng.uniform(0.5, 2.5)
        orbit, mass, radius = rng.uniform(0.1, 5), rng.uniform(0.5, 300), rng.uniform(0.5, 12)
        if actual_class != "main_sequence":
            stratum = PLANET_STRATA[actual_class]
            ordinal = ((seed - SEED_START[task]) // 4) % 25
            if ordinal < PLANET_STRATA["anchor_count_per_non_main_class"]:
                stellar_mass, stellar_radius = stratum["anchor"]
            else:
                stellar_mass = rng.uniform(*stratum["mass_range"])
                stellar_radius = rng.uniform(*stratum["radius_range"])
            radius = 109 * stellar_radius * rng.uniform(*PLANET_STRATA["non_main_radius_ratio_range"])
        inputs = {
            "stellar_mass": stellar_mass,
            "stellar_radius": stellar_radius,
            "period_days": 365 * math.sqrt(orbit**3 / stellar_mass),
            "line_shift": mass / (11.177 * math.sqrt(orbit * stellar_mass)) * 656.3 / 3e8,
            "brightness_drop": 100 * (radius / (109 * stellar_radius)) ** 2,
        }
        expected = {
            "orbital_radius": orbit,
            "planet_mass": mass,
            "planet_radius": radius,
            "planet_density": mass * 5.97e27 / ((4 * math.pi / 3) * (radius * 6.37e8) ** 3),
        }
    else:
        equilibrium, albedo, orbit = rng.uniform(150, 900), rng.uniform(0, 0.85), rng.uniform(0.1, 4)
        # Independent of class: each population receives all four warming
        # increments; the class label is never an answer-selection signal.
        increment = (0, 10, 30, 100)[rng.randrange(4)]
        emission = 4 * math.pi * (orbit * 149597870700) ** 2 * 5.670374419e-8 * equilibrium**4
        inputs = {
            "stellar_luminosity": 4 * emission / ((1 - albedo) * 3.827e26),
            "orbital_radius": orbit,
            "albedo": albedo,
            "greenhouse_increment": increment,
        }
        expected = {"equilibrium_temp": equilibrium, "surface_temp": equilibrium + increment}
    rows = []
    for quantity, value in inputs.items():
        distractor = (
            (10 if value == 0 else 0)
            if quantity == "greenhouse_increment"
            else value * (1.31 if task == "planet" else 1.1)
        )
        for source, numeric in (("current star", value), ("reference star", distractor)):
            rows.append(
                {
                    "kind": quantity,
                    "value": numeric,
                    "unit": legacy.QUANTITY_UNITS[quantity],
                    "source": source,
                }
            )
    rng.shuffle(rows)
    return {
        "seed": seed,
        "split": "supplied_transfer",
        "scope": module.SCOPE,
        "evaluation_scope": SCOPE,
        "evaluation_version": VERSION,
        "star_class": actual_class,
        **({"planet_class": "terrestrial"} if task == "temperature" else {}),
        "case_id": _digest({"task": task, "seed": seed, "star_class": actual_class, "inputs": inputs}),
        "numeric_case_id": _digest(inputs),
        "measurements": {f"m{i}": row for i, row in enumerate(rows)},
        "required": list(legacy.FIELDS),
        "expected": expected,
        "expected_provenance": "inverse_generated_candidate_physics_not_course_grading",
        "population_validation": False,
    }


class _TransferEnv:
    """Validation shared by separately scoped subclasses, not a smoke override."""

    def __init__(self, adapter, case, *, task):
        module, _, calculator_type, _ = _config(task)
        _require(type(adapter) is calculator_type, "calculator_scope")
        _require(
            type(case) is dict and type(case.get("seed")) is int and case.get("star_class") in CLASSES,
            "case_identity",
        )
        _require(
            _digest(case) == _digest(_inverse_case(task, case["seed"], case["star_class"])), "case_changed"
        )
        self._transfer_task = task
        LocalStellarEnv.__init__(self, adapter, [deepcopy(case)], max_steps=128)
        _require(
            self.pack.checksum == module.load_supplied_planet_pack().checksum
            if task == "planet"
            else self.pack.checksum == module.load_supplied_temperature_pack().checksum,
            "pack_changed",
        )

    def task_instruction(self):
        return TEMPLATES[self._transfer_task]


class PlanetTransferEnv(_TransferEnv, planet.SuppliedPlanetEnv):
    pass


class TemperatureTransferEnv(_TransferEnv, temperature.SuppliedTemperatureEnv):
    pass


def _env(task, calculator, case):
    cls = PlanetTransferEnv if task == "planet" else TemperatureTransferEnv
    return cls(calculator, case, task=task)


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reserve(registry, task, output, identity):
    registry = Path(registry)
    _require(not registry.is_symlink(), "symlink_registry")
    registry.mkdir(parents=True, exist_ok=True)
    path = registry / f"supplied-input-transfer-{task}-v1.reserved.json"
    value = {
        "scope": SCOPE,
        "version": VERSION,
        "task": task,
        "budget": BUDGET,
        "schedule": schedule(task),
        "schedule_sha256": _digest(schedule(task)),
        "instruction_sha256": _digest(TEMPLATES[task]),
        "output": str(Path(output).resolve()),
        "identity": identity,
        "automatic_retry": False,
    }
    _write(path, value)
    fd = os.open(registry, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return path, value


def _materialize(task, reservation, path):
    _require(
        _digest(_json(Path(path).read_bytes())) == _digest(reservation)
        and _digest(reservation["schedule"]) == _digest(schedule(task))
        and reservation["scope"] == SCOPE
        and reservation["task"] == task
        and reservation["automatic_retry"] is False,
        "reservation_changed",
    )
    return [_inverse_case(task, row["seed"], row["star_class"]) for row in schedule(task)]


def _source_paths(task, checkpoint, metadata):
    module, legacy, _, _ = _config(task)
    package = Path(__file__).resolve().parents[1]
    paths = [
        checkpoint,
        metadata,
        Path(__file__),
        Path(module.__file__),
        Path(legacy.__file__),
        module.PACK_PATH,
        Path(__file__).parent / "checkpoints.py",
        Path(__file__).parent / "frozen_text.py",
        package / "knowledge.py",
        package / "content.py",
        package / "contracts.py",
        package / "environments/local_stellar.py",
        package / "environments/mini_habworlds.py",
        package / "environments/stellar_common.py",
        package / ("planet_knowledge.py" if task == "planet" else "habitability_knowledge.py"),
        package
        / "packs"
        / ("planet_physical_candidate.json" if task == "planet" else "habitability_physical_candidate.json"),
        *sorted((package / "model").glob("*.py")),
    ]
    if task == "planet":
        repository = package.parents[1]
        for relative in CAPABILITY_RATIONALE:
            path = repository / relative
            paths.append(path)
    return list(dict.fromkeys(str(path.resolve()) for path in paths))


def _sources(task, checkpoint, metadata):
    pins = {path: _sha(path) for path in _source_paths(task, checkpoint, metadata)}
    if task == "planet":
        repository = Path(__file__).resolve().parents[3]
        _require(
            all(
                pins[str((repository / path).resolve())] == digest
                for path, digest in CAPABILITY_RATIONALE.items()
            ),
            "capability_rationale_changed",
        )
    return pins


def _identity(action):
    return {key: getattr(action, key) for key in ("kind", "target", "value", "observation_revision")}


def _parent_model(parent, adapter, sources):
    model = parent.get("model", {})
    _require(
        model.get("architecture") == "ConnectomePolicy"
        and type(model.get("hidden_size")) is int
        and model["hidden_size"] == 16
        and model.get("observation_encoding") == adapter["parent_encoding"]
        and model.get("control_encoding") == adapter["control_encoding"]
        and model.get("selection_mode") == "measurement_result_v3",
        "parent_model_contract",
    )
    saved = {
        key: value
        for key, value in parent.get("provenance", {}).get("code_hashes", {}).items()
        if key.startswith("src/habfly/model/") and key.endswith(".py")
    }
    package = Path(__file__).resolve().parents[1]
    execution = {
        f"src/habfly/model/{path.name}": sources.get(str(path.resolve()))
        for path in (package / "model").glob("*.py")
    }
    _require(_digest(execution) == EXECUTION_MODEL_MAP_SHA256, "execution_model_sources_changed")
    _require(
        saved
        and all(
            type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
            for value in saved.values()
        ),
        "training_model_source_metadata",
    )
    return {
        "version": "fixed_supplied_transfer_execution_sources_v1",
        "execution_map_sha256": EXECUTION_MODEL_MAP_SHA256,
        "training_code_sha256": saved,
        "execution_code_sha256": execution,
        "same_as_training": _digest(saved) == _digest(execution),
        "delta": {
            key: {"training_sha256": saved.get(key), "execution_sha256": execution.get(key)}
            for key in sorted(saved.keys() | execution.keys())
            if saved.get(key) != execution.get(key)
        },
        "historical_code_equivalence_claimed": False,
        "calibration_transferred": False,
    }


def _summary(case, env, steps, exact):
    _, grades = env.grades()
    return {
        "case_id": case["case_id"],
        "seed": case["seed"],
        "actual_star_class": case["star_class"],
        "completed": env.completed,
        "steps": steps,
        "exact_actions": exact,
        "metrics": {
            **{
                key: env.metrics[key] for key in ("invalid_actions", "tool_errors", "infrastructure_failures")
            },
            **grades,
        },
    }


def _scores(rows):
    steps = sum(row["steps"] for row in rows)
    return {
        "episodes": len(rows),
        "completed": sum(row["completed"] for row in rows),
        "steps": steps,
        "exact_actions": sum(row["exact_actions"] for row in rows),
        "exact_action_accuracy": sum(row["exact_actions"] for row in rows) / steps if steps else 0,
        "min_steps": min((row["steps"] for row in rows), default=0),
        "max_steps": max((row["steps"] for row in rows), default=0),
        **{
            key: sum(row["metrics"].get(key, 0) for row in rows)
            for key in (
                "invalid_actions",
                "tool_errors",
                "infrastructure_failures",
                "numeric_fields_correct",
                "units_correct",
            )
        },
    }


def _passed(scores):
    return (
        scores["episodes"] == 100
        and scores["completed"] >= 90
        and all(scores[key] == 0 for key in ("invalid_actions", "tool_errors", "infrastructure_failures"))
    )


def _deny(*args, **kwargs):
    raise RuntimeError("supplied_transfer_network_or_training_forbidden")


def run_supplied_input_transfer(task, pilot, graph, output, *, registry):
    """Explicit one-shot task run; NEVER called by import, runtime or training."""
    started = time.monotonic()
    module, legacy, calculator_type, content_key = _config(task)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    checkpoint = Path(pilot) / "training/checkpoint.pt"
    metadata = checkpoint.with_suffix(".pt.json")
    content = _json(metadata.read_bytes())["provenance"][content_key]
    _disjoint(task, content)
    calculator = calculator_type()
    verification = calculator.verify()
    _require(
        content.get("scope") == legacy.SCOPE
        and content.get("knowledge_pack_hash") == module.BASE_PACK_HASH
        and len(graph.body_ids) == 2000
        and graph.manifest.get("source_kind") == "biological",
        "parent_contract",
    )
    sources = _sources(task, checkpoint, metadata)
    policy, parent = load_checkpoint(checkpoint, graph, content_pack=content)
    adapter = module.adapter_manifest()
    model_source_provenance = _parent_model(parent, adapter, sources)
    _require(
        policy.hidden_size == 16
        and policy.observation_encoding == adapter["parent_encoding"]
        and policy.control_encoding == adapter["control_encoding"]
        and policy.selection_mode == "measurement_result_v3"
        and parent["graph_hash"] == graph_fingerprint(graph),
        "model_contract",
    )
    policy.requires_grad_(False)
    policy.calibration = {"status": "uncalibrated", "scope": SCOPE, "reason": "new_supplied_input_adapter"}
    policy.action_temperature = policy.target_temperature = 1.0
    weights = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    identity = {
        "model_source_provenance": model_source_provenance,
        "parent_checkpoint_path": str(checkpoint.resolve()),
        "parent_metadata_path": str(metadata.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "parent_metadata_sha256": _sha(metadata),
        "parent_content_hash": parent["content_pack_hash"],
        "parent_pack_hash": module.BASE_PACK_HASH,
        "parent_scope": legacy.SCOPE,
        "graph_hash": parent["graph_hash"],
        "pack_hash": calculator.pack.checksum,
        "adapter_sha256": adapter["sha256"],
        "source_sha256": sources,
    }
    directory = Path(output)
    _require(not directory.exists() and not directory.is_symlink(), "output_exists")
    directory.mkdir(parents=True)
    rows, env = [], None
    reservation_path = None
    try:
        reservation_path, reservation = _reserve(registry, task, directory, identity)
        cases = _materialize(task, reservation, reservation_path)
        _require(len({case["numeric_case_id"] for case in cases}) == 100, "duplicate_numeric_cases")
        _write(directory / "private-transfer-cases.json", cases)
        manifest = {
            "scope": SCOPE,
            "version": VERSION,
            "task": task,
            "budget": BUDGET,
            "identity": identity,
            "adapter": adapter,
            "input_scope": module.SCOPE,
            "reservation": str(reservation_path.resolve()),
            "reservation_sha256": _sha(reservation_path),
            "cases_sha256": _sha(directory / "private-transfer-cases.json"),
            "instruction": TEMPLATES[task],
            "instruction_sha256": _digest(TEMPLATES[task]),
            "verification": verification,
            "numeric_strata": PLANET_STRATA if task == "planet" else None,
            "capability_rationale": CAPABILITY_RATIONALE if task == "planet" else {},
            "automatic_retry": False,
        }
        _write(directory / "manifest.json", manifest)
        with ExitStack() as stack:
            for owner, name in (
                (socket.socket, "connect"),
                (socket.socket, "connect_ex"),
                (socket, "create_connection"),
                (torch.optim.Optimizer, "__init__"),
                (torch.autograd, "backward"),
            ):
                stack.enter_context(patch.object(owner, name, _deny))
            stack.enter_context(torch.inference_mode())
            stack.enter_context(frozen_text_cache(policy))
            for index, case in enumerate(cases):
                _require(time.monotonic() - started < 600, "time_limit")
                _require(all(_sha(path) == digest for path, digest in sources.items()), "source_changed")
                env = _env(task, calculator, case)
                env.reset(seed=case["seed"])
                hidden, exact = None, 0
                trace = directory / f"episode-{index:03d}.jsonl"
                with trace.open("x") as stream:
                    for step in range(128):
                        _require(time.monotonic() - started < 600, "time_limit")
                        observation = env.observe()
                        before = observation.model_dump(mode="json")
                        view = module.inference_view(
                            observation,
                            expected_adapter_sha256=adapter["sha256"],
                            expected_pack_hash=calculator.pack.checksum,
                        )
                        action, hidden, _ = policy.act(view, hidden)
                        _require(time.monotonic() - started < 600, "time_limit")
                        _require(torch.isfinite(hidden).all().item(), "nonfinite_hidden")
                        action.action_confidence = action.target_confidence = None
                        action.calibrated = False
                        # Evaluation-only reference, generated after policy inference;
                        # it is never supplied to the policy or used to repair it.
                        reference_action = env.expert_action(observation)
                        matched = _identity(action) == _identity(reference_action)
                        exact += matched
                        _, _, done, truncated, info = env.step(action)
                        line = {
                            "step": step,
                            "observation": before,
                            "inference_view_sha256": _digest(view.model_dump(mode="json")),
                            "action": action.model_dump(mode="json"),
                            "reference_action": reference_action.model_dump(mode="json"),
                            "matches_expert": matched,
                            "result": info["result"],
                            "failure_reason": info["result"].get("failure_reason"),
                            "tool_error": env.tool_error,
                        }
                        stream.write(json.dumps(line, allow_nan=False, sort_keys=True) + "\n")
                        stream.flush()
                        if done or truncated:
                            break
                    os.fsync(stream.fileno())
                row = {
                    **_summary(case, env, step + 1, exact),
                    "trace": trace.name,
                    "trace_sha256": _sha(trace),
                }
                _write(directory / f"episode-{index:03d}-summary.json", row)
                rows.append(row)
                env.close()
                env = None
        _require(time.monotonic() - started < 600, "time_limit")
        _require(
            all(torch.equal(value, policy.state_dict()[key]) for key, value in weights.items()),
            "weights_changed",
        )
        _require(
            all(_sha(path) == digest for path, digest in sources.items())
            and graph_fingerprint(graph) == identity["graph_hash"],
            "source_changed",
        )
        scores = _scores(rows)
        report = {
            "scope": SCOPE,
            "version": VERSION,
            "task": task,
            "input_scope": module.SCOPE,
            "budget": BUDGET,
            "identity": identity,
            "manifest_sha256": _sha(directory / "manifest.json"),
            "rows": rows,
            "scores": scores,
            "per_class": {
                name: _scores([row for row in rows if row["actual_star_class"] == name]) for name in CLASSES
            },
            "transfer_gate_passed": _passed(scores),
            "weights_unchanged": True,
            "sources_unchanged": True,
            "calibration": dict(policy.calibration),
            "elapsed_seconds": time.monotonic() - started,
            "peak_process_rss_bytes": peak_process_rss_bytes(),
            "optimizer_updates": 0,
            "case_interpretation": "Supplied class labels with new synthetic numeric fixtures; not astrophysical-population or course validation",
            **dict.fromkeys(FALSE_CLAIMS, False),
        }
        _write(directory / "report.json", report)
        return report
    except BaseException as exc:
        _write(
            directory / "stopped.json",
            {
                "scope": SCOPE,
                "task": task,
                "error_type": type(exc).__name__,
                "reservation_retained": reservation_path is not None,
                "episodes_completed": len(rows),
                "transfer_gate_passed": False,
                "automatic_retry": False,
                "optimizer_updates": 0,
                **dict.fromkeys(FALSE_CLAIMS, False),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()


def _recorded_case(task, case):
    """Validate saved fixture identity/domains without regenerating answers."""
    module, legacy, _, _ = _config(task)
    _require(
        type(case) is dict and type(case.get("seed")) is int and case.get("star_class") in CLASSES,
        "case_identity",
    )
    _require(
        case.get("scope") == module.SCOPE
        and case.get("evaluation_scope") == SCOPE
        and type(case.get("evaluation_version")) is int
        and case["evaluation_version"] == VERSION
        and case.get("split") == "supplied_transfer"
        and case.get("population_validation") is False
        and case.get("expected_provenance") == "inverse_generated_candidate_physics_not_course_grading"
        and case.get("required") == list(legacy.FIELDS),
        "recorded_case_contract",
    )
    kinds = (
        {"stellar_mass", "stellar_radius", "period_days", "line_shift", "brightness_drop"}
        if task == "planet"
        else {"stellar_luminosity", "orbital_radius", "albedo", "greenhouse_increment"}
    )
    measurements = case.get("measurements")
    _require(
        type(measurements) is dict and set(measurements) == {f"m{i}" for i in range(2 * len(kinds))},
        "recorded_measurements",
    )
    seen, inputs = set(), {}
    for row in measurements.values():
        _require(
            type(row) is dict and set(row) == {"kind", "value", "unit", "source"}, "recorded_measurement"
        )
        kind, value, source = row["kind"], row["value"], row["source"]
        _require(
            kind in kinds
            and source in {"current star", "reference star"}
            and (kind, source) not in seen
            and row["unit"] == legacy.QUANTITY_UNITS[kind]
            and type(value) in {int, float}
            and math.isfinite(value),
            "recorded_measurement_domain",
        )
        seen.add((kind, source))
        _require(
            value >= 0 if kind in {"albedo", "greenhouse_increment"} else value > 0,
            "recorded_measurement_domain",
        )
        if source == "current star":
            inputs[kind] = value
            _require(kind != "brightness_drop" or value <= 100, "recorded_measurement_domain")
            _require(kind != "albedo" or value < 1, "recorded_measurement_domain")
            _require(
                kind != "greenhouse_increment" or value in {0, 10, 30, 100}, "recorded_measurement_domain"
            )
    _require(
        case.get("numeric_case_id") == _digest(inputs)
        and case.get("case_id")
        == _digest({"task": task, "seed": case["seed"], "star_class": case["star_class"], "inputs": inputs}),
        "recorded_case_identity",
    )
    expected = case.get("expected")
    _require(
        type(expected) is dict
        and set(expected) == set(legacy.FIELDS)
        and all(type(v) in {int, float} and math.isfinite(v) and v > 0 for v in expected.values()),
        "recorded_private_grading",
    )
    _require(task != "temperature" or case.get("planet_class") == "terrestrial", "recorded_planet_class")


def _recorded_episode(task, case, events, *, pack_hash, adapter_sha256):
    """Check recorded continuity, grades and evaluator reference comparisons.

    Does not execute an environment/calculator/policy/expert. The recorded
    reference action is evaluator evidence, not an independent expert rerun.
    """
    module, legacy, _, _ = _config(task)
    _recorded_case(task, case)
    _require(type(events) is list and 1 <= len(events) <= 128, "trace_action_limit")
    exact = invalid = errors = 0
    previous = None
    for step, event in enumerate(events):
        _require(
            type(event) is dict and type(event.get("step")) is int and event["step"] == step,
            "trajectory_step",
        )
        before = Observation.model_validate(event["observation"])
        action = Action.model_validate(event["action"])
        reference = Action.model_validate(event["reference_action"])
        result = StepResult.model_validate(event["result"])
        _require(
            all(
                _digest(event[key]) == _digest(value.model_dump(mode="json"))
                for key, value in (
                    ("observation", before),
                    ("action", action),
                    ("reference_action", reference),
                    ("result", result),
                )
            ),
            "trajectory_types",
        )
        _require(
            before.revision == step
            and result.steps == step + 1
            and result.observation.revision == step + 1
            and (previous is None or _digest(event["observation"]) == _digest(previous)),
            "trajectory_continuity",
        )
        for obs in (before, result.observation):
            _require(
                obs.instruction == TEMPLATES[task]
                and obs.values.get("star_class") == case["star_class"]
                and _digest(obs.values.get("measurements")) == _digest(case["measurements"])
                and obs.values.get("required_fields") == case["required"]
                and "expected" not in obs.values
                and "expected" not in obs.calculation
                and obs.progress.get("scope") == module.SCOPE
                and obs.progress.get("native_browser_enabled") is False
                and obs.progress.get("scientific_verified") is False
                and obs.calculation.get("pack_hash") == pack_hash,
                "trajectory_authoritative_scope",
            )
        _require(before.progress.get("task_completed") is False, "trajectory_after_completion")
        if step == 0:
            _require(
                before.values.get("answers") == {}
                and before.values.get("units") == {}
                and before.calculation.get("results") == {},
                "trajectory_initial_state",
            )
        try:
            validate_action(before, action)
        except ValueError:
            invalid += 1
        validate_action(before, reference)
        view = module.inference_view(
            before, expected_adapter_sha256=adapter_sha256, expected_pack_hash=pack_hash
        )
        match = _identity(action) == _identity(reference)
        _require(
            event.get("inference_view_sha256") == _digest(view.model_dump(mode="json"))
            and event.get("matches_expert") is match
            and action.calibrated is False
            and action.action_confidence is None
            and action.target_confidence is None,
            "trajectory_inference",
        )
        exact += match
        error = result.observation.calculation.get("tool_error")
        _require(
            event.get("tool_error") == error and event.get("failure_reason") == result.failure_reason,
            "trajectory_result",
        )
        errors += error is not None
        terminal = result.terminated or result.truncated
        _require(not terminal or step == len(events) - 1, "trajectory_after_terminal")
        _require(
            not result.truncated or (result.steps == 128 and result.failure_reason == "step_limit"),
            "trajectory_truncation",
        )
        _require(result.failure_reason in {None, "policy_stopped", "step_limit"}, "trajectory_failure")
        _require((action.kind == "STOP") == (result.failure_reason == "policy_stopped"), "trajectory_stop")
        previous = result.observation.model_dump(mode="json")
    _require(result.terminated or result.truncated, "unfinished_episode")
    completed = result.observation.progress.get("task_completed")
    _require(type(completed) is bool, "completion_type")
    answers, units = result.observation.values["answers"], result.observation.values["units"]
    numeric = {
        k
        for k in legacy.FIELDS
        if type(answers.get(k)) in {int, float}
        and math.isfinite(answers[k])
        and math.isclose(answers[k], case["expected"][k], rel_tol=1e-6, abs_tol=1e-12)
    }
    unit_correct = {k for k in legacy.FIELDS if units.get(k) == legacy.QUANTITY_UNITS[k]}
    _require(
        not completed
        or (
            len(numeric & unit_correct) == len(legacy.FIELDS)
            and action.kind == "CLICK"
            and action.target == f"{len(events) - 1}:check"
            and result.terminated
            and not result.truncated
            and result.failure_reason is None
        ),
        "false_completion",
    )
    _require(not result.terminated or completed or action.kind == "STOP", "unexplained_termination")
    return {
        "case_id": case["case_id"],
        "seed": case["seed"],
        "actual_star_class": case["star_class"],
        "completed": completed,
        "steps": len(events),
        "exact_actions": exact,
        "metrics": {
            "invalid_actions": invalid,
            "tool_errors": errors,
            "infrastructure_failures": 0,
            "numeric_fields_correct": len(numeric),
            "required_fields": len(legacy.FIELDS),
            "units_correct": len(unit_correct),
        },
    }


def require_supplied_input_transfer_gate(
    directory,
    *,
    task,
    checkpoint_sha256,
    graph_hash,
    pack_hash,
    adapter_sha256,
    artifact_paths=None,
    source_paths=None,
):
    """Rebuild actual trajectory scores offline before authorizing this scope.

    No old final report or smoke can substitute. No policy/expert/environment or
    generator is called. Relocation maps resolve ORIGINAL absolute paths to
    owned byte-identical copies with no fallback reads; original identity stays
    unchanged. Relocated source verification is historical, not current-code
    or currently loaded weight verification.
    """
    module, legacy, calculator_type, content_key = _config(task)
    directory = Path(directory)
    _require(
        artifact_paths is not None
        or (not directory.is_symlink() and not (directory / "stopped.json").exists()),
        "failed_or_symlink_gate",
    )
    directory = directory.resolve()
    artifacts, origins, source_pins, source_origins = {}, {}, {}, {}
    _require((source_paths is None) == (artifact_paths is None), "relocation_maps_required_together")

    def raw(path, *, source=False):
        original = str(Path(path).resolve())
        mapping = source_paths if source else artifact_paths
        pins, used = (source_pins, source_origins) if source else (artifacts, origins)
        if mapping is not None:
            _require(type(mapping) is dict and original in mapping, "relocation_map_missing")
            path = Path(mapping[original])
            _require(path.is_absolute(), "relocation_map_path")
        else:
            path = Path(original)
        _require(
            path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents))
            and 0 < path.stat().st_size <= 64_000_000,
            "missing_or_oversized_gate_artifact",
        )
        data = path.read_bytes()
        actual, digest = str(path.resolve()), hashlib.sha256(data).hexdigest()
        _require(actual not in pins or pins[actual] == digest, "artifact_changed")
        pins[actual], used[original] = digest, actual
        return data

    def read(name):
        return _json(raw(directory / name))

    def artifact_hash(path):
        return hashlib.sha256(raw(path)).hexdigest()

    report, manifest = read("report.json"), read("manifest.json")
    _require(
        report.get("scope") == manifest.get("scope") == SCOPE
        and type(report.get("version")) is int
        and type(manifest.get("version")) is int
        and report["version"] == manifest.get("version") == VERSION
        and report.get("task") == manifest.get("task") == task
        and report.get("manifest_sha256") == artifact_hash(directory / "manifest.json")
        and _digest(report.get("budget")) == _digest(manifest.get("budget")) == _digest(BUDGET),
        "wrong_gate_contract",
    )
    identity = manifest["identity"]
    _require(
        _digest(identity) == _digest(report.get("identity"))
        and all(
            identity.get(key) == value
            for key, value in {
                "checkpoint_sha256": checkpoint_sha256,
                "graph_hash": graph_hash,
                "pack_hash": pack_hash,
                "adapter_sha256": adapter_sha256,
            }.items()
        ),
        "identity_mismatch",
    )
    calculator = calculator_type()
    _require(
        pack_hash == calculator.pack.checksum
        and adapter_sha256 == module.adapter_manifest()["sha256"]
        and _digest(manifest["adapter"]) == _digest(module.adapter_manifest())
        and manifest["instruction"] == TEMPLATES[task]
        and manifest["instruction_sha256"] == _digest(TEMPLATES[task]),
        "adapter_or_instruction_changed",
    )
    sources = identity["source_sha256"]
    checkpoint, metadata = Path(identity["parent_checkpoint_path"]), Path(identity["parent_metadata_path"])
    _require(checkpoint.is_absolute() and metadata == checkpoint.with_suffix(".pt.json"), "parent_paths")
    parent = _json(raw(metadata, source=True))
    content = parent["provenance"][content_key]
    _disjoint(task, content)
    _require(
        content.get("scope") == identity.get("parent_scope") == legacy.SCOPE
        and content.get("knowledge_pack_hash") == identity.get("parent_pack_hash") == module.BASE_PACK_HASH
        and source_hash(content) == identity.get("parent_content_hash") == parent["content_pack_hash"]
        and parent["graph_hash"] == graph_hash
        and parent["graph_size"] == 2000
        and parent["provenance"]["graph_source_kind"] == "biological"
        and hashlib.sha256(raw(checkpoint, source=True)).hexdigest() == checkpoint_sha256
        and hashlib.sha256(raw(metadata, source=True)).hexdigest() == identity.get("parent_metadata_sha256"),
        "parent_identity",
    )
    _require(
        isinstance(sources, dict)
        and set(sources) == set(_source_paths(task, checkpoint, metadata))
        and all(
            hashlib.sha256(raw(path, source=True)).hexdigest() == digest for path, digest in sources.items()
        ),
        "source_changed",
    )
    _require(
        _digest(identity.get("model_source_provenance"))
        == _digest(_parent_model(parent, module.adapter_manifest(), sources)),
        "model_source_provenance_changed",
    )
    if task == "planet":
        repository = Path(__file__).resolve().parents[3]
        _require(
            all(
                sources[str((repository / path).resolve())] == digest
                for path, digest in CAPABILITY_RATIONALE.items()
            ),
            "capability_rationale_changed",
        )
    reservation_path = Path(manifest["reservation"])
    _require(
        reservation_path.is_absolute()
        and reservation_path.name == f"supplied-input-transfer-{task}-v1.reserved.json"
        and artifact_hash(reservation_path) == manifest["reservation_sha256"],
        "reservation_changed",
    )
    reservation = _json(raw(reservation_path))
    _require(
        _digest(reservation)
        == _digest(
            {
                "scope": SCOPE,
                "version": VERSION,
                "task": task,
                "budget": BUDGET,
                "schedule": schedule(task),
                "schedule_sha256": _digest(schedule(task)),
                "instruction_sha256": _digest(TEMPLATES[task]),
                "output": str(directory.resolve()),
                "identity": identity,
                "automatic_retry": False,
            }
        ),
        "reservation_changed",
    )
    cases = read("private-transfer-cases.json")
    _require(
        artifact_hash(directory / "private-transfer-cases.json") == manifest["cases_sha256"]
        and type(cases) is list
        and len(cases) == 100
        and [{"seed": c.get("seed"), "star_class": c.get("star_class")} for c in cases] == schedule(task)
        and len({c.get("numeric_case_id") for c in cases}) == 100,
        "cases_changed",
    )
    _require(
        len(report.get("rows", [])) == 100
        and Counter(case["star_class"] for case in cases) == dict.fromkeys(CLASSES, 25),
        "case_count",
    )
    rows = []
    for index, case in enumerate(cases):
        row = read(f"episode-{index:03d}-summary.json")
        _require(
            row.get("trace") == f"episode-{index:03d}.jsonl"
            and row["trace_sha256"] == artifact_hash(directory / row["trace"]),
            "trace_changed",
        )
        events = [_json(line) for line in raw(directory / row["trace"]).splitlines()]
        expected = {
            **_recorded_episode(task, case, events, pack_hash=pack_hash, adapter_sha256=adapter_sha256),
            "trace": row["trace"],
            "trace_sha256": row["trace_sha256"],
        }
        _require(_digest(row) == _digest(expected), "summary_mismatch")
        rows.append(row)
    scores = _scores(rows)
    _require(
        _digest(report["rows"]) == _digest(rows)
        and _digest(report["scores"]) == _digest(scores)
        and _digest(report["per_class"])
        == _digest(
            {name: _scores([row for row in rows if row["actual_star_class"] == name]) for name in CLASSES}
        )
        and _passed(scores)
        and report.get("transfer_gate_passed") is True
        and report.get("weights_unchanged") is True
        and report.get("sources_unchanged") is True
        and type(report.get("optimizer_updates")) is int
        and report["optimizer_updates"] == 0
        and type(report.get("elapsed_seconds")) in {int, float}
        and 0 < report["elapsed_seconds"] <= 600
        and all(report.get(key) is False for key in FALSE_CLAIMS)
        and report.get("calibration")
        == {"status": "uncalibrated", "scope": SCOPE, "reason": "new_supplied_input_adapter"},
        "gate_failed",
    )
    _require(
        manifest.get("input_scope") == report.get("input_scope") == module.SCOPE
        and manifest.get("automatic_retry") is False
        and _digest(manifest.get("numeric_strata")) == _digest(PLANET_STRATA if task == "planet" else None)
        and _digest(manifest.get("capability_rationale"))
        == _digest(CAPABILITY_RATIONALE if task == "planet" else {}),
        "manifest_scope",
    )
    _require(
        _digest(manifest.get("verification"))
        == _digest(
            {
                "valid": True,
                "operation_golden_cases": 24,
                "pack_hash": pack_hash,
                "scope": module.SCOPE,
                "course_acceptance_passed": False,
                "scientific_verified": False,
                "native_browser_enabled": False,
                "training_oracle_enabled": False,
            }
        ),
        "independent_golden_record",
    )
    _require(
        all(_sha(path) == digest for path, digest in {**source_pins, **artifacts}.items()), "source_changed"
    )
    _require(
        artifact_paths is None
        or (set(artifact_paths) == set(origins) and set(source_paths) == set(source_origins)),
        "unused_relocation_entries",
    )
    return {
        "scope": SCOPE,
        "task": task,
        "report": report,
        "report_sha256": artifact_hash(directory / "report.json"),
        "identity": identity,
        "scores": scores,
        "source_sha256": source_pins,
        "artifact_sha256": artifacts,
        "artifact_origins": origins,
        "source_origins": source_origins,
        "current_sources_verified": source_paths is None,
        "historical_provenance_verified": True,
        "transfer_gate_passed": True,
        "native_browser_enabled": False,
        "course_acceptance_passed": False,
        "scientific_verified": False,
    }
