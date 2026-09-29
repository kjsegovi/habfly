"""Frozen supplied-temperature tool use, with one guarded equilibrium copy.

The caller supplies a warming increment; this stage does not choose gases or
greenhouse strength. Surface temperature is a local proposal, never a writable
browser field. No course grading answers or scripted expert are available.
"""

import math
import random
import time
from pathlib import Path

import torch

from .browser import BrowserSafetyStop
from .browser_habitability_numeric import HabitabilityNumericSession
from .contracts import RuntimeEvent, StepResult, validate_action
from .environments.habitability_calculations import FIELDS, QUANTITY_UNITS, HabitabilityCalculationEnv
from .environments.local_stellar import LocalStellarEnv
from .habitability_knowledge import HabitabilityCalculator
from .training.habitability_evaluation import load_validated_pilot, require_frozen_final_gate
from .training.planet_sequence import file_hash, read
from .training.stellar import write_json
from .training.train import prepare_output_directory


def visible_temperature_inputs(mapping, *, supplied_greenhouse_increment):
    if type(supplied_greenhouse_increment) not in {int, float} or supplied_greenhouse_increment not in {
        0,
        10,
        30,
        100,
    }:
        raise BrowserSafetyStop("unsupported_supplied_greenhouse_increment")
    values = mapping["observation"]["values"]
    if values["equilibrium_temp"]["value"] not in {"", "0"} or values["greenhouse"] is not None:
        raise BrowserSafetyStop("habitability_temperature_stage_already_populated")
    measurements = []
    for kind in ("stellar_luminosity", "orbital_radius", "albedo"):
        source = values["measurements"].get(kind)
        if not isinstance(source, dict) or source.get("unit") != QUANTITY_UNITS[kind]:
            raise BrowserSafetyStop("missing_or_incompatible_temperature_input")
        value = source.get("value")
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or not (0 <= value < 1 if kind == "albedo" else value > 0)
        ):
            raise BrowserSafetyStop("invalid_visible_temperature_input")
        measurements.append({"kind": kind, "value": value, "unit": source["unit"], "source": "current star"})
    measurements.append(
        {
            "kind": "greenhouse_increment",
            "value": supplied_greenhouse_increment,
            "unit": "K",
            "source": "current star",
        }
    )
    return measurements


class HabitabilityBrowserToolEnv(HabitabilityCalculationEnv):
    def __init__(self, mapping, *, supplied_greenhouse_increment, seed=20000000):
        rows = visible_temperature_inputs(
            mapping, supplied_greenhouse_increment=supplied_greenhouse_increment
        )
        random.Random(seed).shuffle(rows)
        self.proposals_complete = False
        case = {
            "seed": seed,
            "split": "browser",
            "star_class": "main_sequence",
            "planet_class": "terrestrial",
            "required": list(FIELDS),
            "measurements": {f"m{i}": row for i, row in enumerate(rows)},
        }
        # Bypass synthetic-case validation: there are no private expected answers.
        LocalStellarEnv.__init__(self, HabitabilityCalculator(), [case], max_steps=128)
        self.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        self.proposals_complete = False
        return super().reset(seed=seed, options=options)

    def task_instruction(self):
        return "Use the current star's visible luminosity, orbit and albedo and the explicitly supplied warming increment to derive equilibrium and surface temperatures. Select calculations, bind measurements, copy the two results and choose units. Do not infer gases, phase or habitability."

    def observe(self):
        observation = super().observe()
        observation.progress.update(
            task="browser_supplied_temperatures",
            task_completed=False,
            temperature_proposals_complete=self.proposals_complete,
            browser_acceptance_passed=False,
        )
        return observation

    def grades(self):
        return set(), {}

    def expert_action(self, observation):
        raise RuntimeError("Browser temperature tool has no scripted expert")

    def step(self, action):
        control = validate_action(self.observe(), action)
        key = control.id.split(":", 1)[1] if control else ""
        _, _, _, _, info = LocalStellarEnv.step(self, action)
        result = StepResult.model_validate(info["result"])
        if key.startswith("unit_") and action.value != "K":
            result.failure_reason = "temperature_unit_mismatch"
        if key == "check":
            self.proposals_complete = (
                set(self.answers) == set(FIELDS)
                and self.units == dict.fromkeys(FIELDS, "K")
                and all(math.isfinite(v) and v > 0 for v in self.answers.values())
            )
            result.terminated = True
            result.failure_reason = None if self.proposals_complete else "temperature_proposals_incomplete"
        if self.tool_error or result.failure_reason:
            result.terminated = True
            result.failure_reason = result.failure_reason or self.tool_error
        self.terminated = result.terminated
        result.observation = self.observe()
        result.reward = result.cumulative_reward = 0
        result.reward_components = {}
        return result


def run_habitability_temperature(
    page,
    config,
    output,
    *,
    pilot,
    final_evaluation,
    graph,
    supplied_greenhouse_increment,
    seed=20000000,
    notify=lambda _: None,
):
    checkpoint = Path(pilot) / "training/checkpoint.pt"
    checksum = file_hash(checkpoint)
    require_frozen_final_gate(read(Path(final_evaluation) / "report.json"), checksum)
    policy, content, _ = load_validated_pilot(pilot, graph)
    policy.eval()
    policy.calibration = {"status": "uncalibrated", "scope": "browser_transfer_not_calibrated"}
    directory = prepare_output_directory(output)
    provenance = {
        "checkpoint_sha256": checksum,
        "graph_hash": content["graph_hash"],
        "knowledge_pack_hash": content["knowledge_pack_hash"],
        "optimizer_updates": 0,
        "supplied_greenhouse_increment": supplied_greenhouse_increment,
        "greenhouse_selection_learned": False,
        "gas_identification_learned": False,
        "water_phase_learned": False,
        "habitability_decision_learned": False,
        "calibration_scope": "browser_transfer_not_calibrated",
        "surface_proposal_is_not_browser_readback": True,
    }
    session, receipt, env, steps = None, None, None, 0
    outcome, started = "not_started", time.monotonic()
    with (directory / "events.jsonl").open("x") as stream:
        sequence = 0

        def emit(event, payload):
            nonlocal sequence
            stream.write(
                RuntimeEvent(
                    event=event, sequence=sequence, run_id=f"temperature-browser-{seed}", payload=payload
                ).model_dump_json()
                + "\n"
            )
            stream.flush()
            sequence += 1

        emit(
            "hello",
            {
                "protocol_version": 1,
                "task": "habitability_calculations",
                "policy": "checkpoint",
                **provenance,
            },
        )
        try:
            session = HabitabilityNumericSession(page, config, directory / "native-copy", max_seconds=300)
            env = HabitabilityBrowserToolEnv(
                session.mapping, supplied_greenhouse_increment=supplied_greenhouse_increment, seed=seed
            )
            emit(
                "state",
                {
                    "stage": "browser_supplied_temperatures",
                    "status": "running",
                    "browser_status": "connected",
                    **provenance,
                },
            )
            state = None
            with torch.no_grad():
                for _ in range(128):
                    observation = env.observe()
                    emit("observation", observation.model_dump(mode="json"))
                    action, state, diagnostics = policy.act(observation, state)
                    action.action_confidence = action.target_confidence = None
                    action.calibrated = False
                    emit("action_proposed", {**action.model_dump(mode="json"), "action_source": "checkpoint"})
                    emit("neural_activity", {**diagnostics, "activity_source": "checkpoint"})
                    result = env.step(action)
                    steps += 1
                    emit("action_result", result.model_dump(mode="json"))
                    notify({"steps": steps, "failure": result.failure_reason})
                    if result.terminated or result.truncated:
                        break
            if not env.proposals_complete or result.failure_reason:
                raise BrowserSafetyStop(result.failure_reason or "temperature_policy_step_limit")
            # Only after all learned choices pass the interface checks. Do not
            # repair choices or write the derived/read-only surface field.
            session.current()
            emit(
                "action_proposed",
                {
                    "kind": "TYPE",
                    "destination": "equilibrium_temp",
                    "value": str(env.answers["equilibrium_temp"]),
                    "action_source": "deterministic_exact_transport_of_checkpoint_result",
                },
            )
            receipt = session.copy(str(env.answers["equilibrium_temp"]), "K", source="checkpoint")
            emit("action_result", {"native_copy": receipt})
            outcome = "equilibrium_transport_verified"
        except Exception as exc:  # noqa: BLE001 - never log browser URLs or credentials
            outcome = (
                str(exc) if isinstance(exc, BrowserSafetyStop) else "temperature_browser_operation_failed"
            )
            emit("error", {"reason": outcome, "exception_type": type(exc).__name__})
        finally:
            if session is not None:
                session.close()
            unchanged = file_hash(checkpoint) == checksum
            report = {
                "scope": "one_equilibrium_copy_with_supplied_warming_local_proposal",
                "provenance": provenance,
                "outcome": outcome if unchanged else "checkpoint_changed",
                "steps": steps,
                "checkpoint_unchanged": unchanged,
                "optimizer_updates": 0,
                "equilibrium_transport_verified": receipt is not None and unchanged,
                "local_proposals": env.answers if env else {},
                "native_receipt": receipt,
                "task_completed": False,
                "course_acceptance_passed": False,
                "saved": False,
                "assessed": False,
                "submitted": False,
                "elapsed_seconds": time.monotonic() - started,
            }
            emit("episode_summary", report)
    report["events_sha256"] = file_hash(directory / "events.jsonl")
    write_json(directory / "report.json", report)
    return report
