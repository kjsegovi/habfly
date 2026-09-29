"""Frozen four-derived-field planet transport; no chart/assessment oracle.

Raw measurements must already be visibly populated by a separate evidence-backed
stage. A supplied stellar class is not a learned classification. Only the
existing guarded native copy session may write; no navigation or account setup.
"""

import math
import random
import re
import time
from copy import deepcopy
from pathlib import Path

import torch

from .browser import BrowserSafetyStop
from .browser_planet_numeric import PlanetNumericSession
from .browser_stellar import NUMBER
from .contracts import RuntimeEvent, StepResult, validate_action
from .environments.local_stellar import LocalStellarEnv
from .environments.planet_calculations import FIELDS, QUANTITY_UNITS, SCOPE, PlanetCalculationEnv
from .planet_knowledge import PlanetCalculator
from .training.checkpoints import load_checkpoint
from .training.planet_evaluation import require_frozen_final_gate as require_final_gate
from .training.planet_sequence import file_hash, read
from .training.stellar import write_json
from .training.train import prepare_output_directory


def visible_measurements(mapping):
    values = mapping["observation"]["values"]
    if values["has_planet"] != "Yes":
        raise BrowserSafetyStop("planet_yes_selection_required")
    fields = values["browser_field_map"]
    if any(fields[k]["current_value"] != "" for k in FIELDS):
        raise BrowserSafetyStop("planet_derived_fields_must_be_empty")
    measurements = []
    for kind in ("period_days", "line_shift", "brightness_drop", "stellar_mass", "stellar_radius"):
        source = fields.get(kind) or values["stellar_inputs"][kind]
        text = source.get("current_value", source.get("display_text"))
        if not isinstance(text, str) or not re.fullmatch(NUMBER, text):
            raise BrowserSafetyStop("missing_visible_planet_measurement")
        value = float(text)
        if not math.isfinite(value) or value <= 0 or kind == "brightness_drop" and value > 100:
            raise BrowserSafetyStop("invalid_visible_planet_measurement")
        if source["unit"] != QUANTITY_UNITS[kind]:
            raise BrowserSafetyStop("incompatible_visible_planet_measurement_unit")
        measurements.append({"kind": kind, "value": value, "unit": source["unit"], "source": "current star"})
    return measurements


class PlanetBrowserToolEnv(PlanetCalculationEnv):
    def __init__(self, session, *, supplied_star_class, seed=15000000):
        if supplied_star_class != "main_sequence":
            raise BrowserSafetyStop("planet_tool_requires_supplied_main_sequence_class")
        self.session = session
        self.transport_verified = False
        self.starting_measurements = visible_measurements(session.mapping)
        rows = deepcopy(self.starting_measurements)
        random.Random(seed).shuffle(rows)
        case = {
            "seed": seed,
            "split": "browser",
            "star_class": supplied_star_class,
            "required": list(FIELDS),
            "measurements": {f"m{i}": row for i, row in enumerate(rows)},
        }
        # Deliberately bypass synthetic-case validation: no private grading
        # answers, fake expected values or course oracle exist in this case.
        LocalStellarEnv.__init__(self, PlanetCalculator(), [case], max_steps=128)
        self.reset(seed=seed)

    def task_instruction(self):
        return "Use current-star visible observations and the stated physics assumptions to derive orbital radius, planet mass, planet radius and density. Bind inputs and intermediate results, copy each result and select its unit."

    def observe(self):
        observation = super().observe()
        observation.progress.update(
            task="browser_planet_derived",
            task_completed=False,
            planet_transport_verified=self.transport_verified,
            browser_acceptance_passed=False,
        )
        return observation

    def grades(self):
        return set(), {}  # Native readback is not scientific or course correctness.

    def expert_action(self, observation):
        raise RuntimeError("Browser planet transport has no scripted expert")

    def copy_result(self, result):
        if self.destination not in FIELDS:
            raise BrowserSafetyStop("unsupported_planet_derived_destination")
        self.session.copy(self.destination, str(result["value"]), result["unit"], source="checkpoint")
        super().copy_result(result)

    def step(self, action):
        control = validate_action(self.observe(), action)
        key = control.id.split(":", 1)[1] if control else ""
        # Calculator selections operate on a pinned, public measurement snapshot.
        # Each native copy performs its own complete pre/post browser validation;
        # no cached snapshot ever authorizes a write. Recheck once more before
        # reporting transport completion. This avoids 62 expensive full-page
        # captures for 58 purely local decisions without weakening write guards.
        if key == "check":
            self.session.current()
        _, _, _, _, info = LocalStellarEnv.step(self, action)
        result = StepResult.model_validate(info["result"])
        if key.startswith("unit_") and action.value != QUANTITY_UNITS[key[5:]]:
            result.failure_reason = "selected_unit_does_not_match_visible_field"
        if key == "check":
            self.transport_verified = set(self.session.verified) == set(FIELDS) and self.units == {
                field: QUANTITY_UNITS[field] for field in FIELDS
            }
            result.terminated = True
            result.failure_reason = None if self.transport_verified else "planet_derived_transport_incomplete"
            self.feedback = (
                "Native readbacks checked; scientific correctness and course completion are unverified."
            )
        if self.tool_error or result.failure_reason:
            result.terminated = True
            result.failure_reason = result.failure_reason or self.tool_error
        self.terminated = result.terminated
        result.observation = self.observe()
        result.reward = result.cumulative_reward = 0
        result.reward_components = {}
        return result


def run_planet_derived(
    page,
    config,
    output,
    *,
    pilot,
    final_evaluation,
    graph,
    supplied_star_class,
    seed=15000000,
    notify=lambda _: None,
    resume_from=None,
):
    checkpoint = Path(pilot) / "training/checkpoint.pt"
    checkpoint_hash = file_hash(checkpoint)
    recovery = None
    if resume_from is not None:
        from .browser_planet_recovery import load_mass_orbit_recovery

        recovery = load_mass_orbit_recovery(resume_from, checkpoint_hash)
    require_final_gate(read(Path(final_evaluation) / "report.json"), checkpoint_hash)
    content = read(checkpoint.with_suffix(".pt.json"))["provenance"]["planet_calculations"]
    if content["scope"] != SCOPE or content["knowledge_pack_hash"] != PlanetCalculator().pack.checksum:
        raise ValueError("Planet browser checkpoint scope/pack mismatch")
    torch.set_num_threads(1)
    policy, manifest = load_checkpoint(checkpoint, graph, content_pack=content)
    if (
        len(graph.body_ids) != 2000
        or policy.hidden_size != 16
        or policy.observation_encoding != "structured_planet_tool_v1"
    ):
        raise ValueError("Planet browser model contract mismatch")
    policy.calibration = {"status": "uncalibrated", "scope": "browser_transfer_not_calibrated"}
    directory = prepare_output_directory(output)
    provenance = {
        "checkpoint_sha256": checkpoint_hash,
        "graph_hash": manifest["graph_hash"],
        "knowledge_pack_hash": content["knowledge_pack_hash"],
        "optimizer_updates": 0,
        "raw_measurements": "separate_visible_readback_stage",
        "classification_source": "supplied_not_learned",
        "supplied_star_class": supplied_star_class,
        "calibration_scope": "browser_transfer_not_calibrated",
        "browser_validation": "full_before_and_after_each_native_copy_and_at_completion",
        "local_tool_observations": "pinned_visible_measurements_not_live_browser_state",
    }
    if recovery:
        provenance["recovery"] = recovery["provenance"]
    session, steps, outcome, verified = None, 0, "not_started", False
    started, sequence = time.monotonic(), 0
    with (directory / "events.jsonl").open("x") as stream:

        def emit(event, payload):
            nonlocal sequence
            stream.write(
                RuntimeEvent(
                    event=event, sequence=sequence, run_id=f"planet-browser-{seed}", payload=payload
                ).model_dump_json()
                + "\n"
            )
            stream.flush()
            sequence += 1

        emit(
            "hello",
            {"protocol_version": 1, "task": "planet_calculations", "policy": "checkpoint", **provenance},
        )
        try:
            session = PlanetNumericSession(
                page, config, directory / "native-copies", emit=emit, max_seconds=900
            )
            if recovery:
                from .browser_planet_recovery import ReconciledPlanetSession

                session = ReconciledPlanetSession(session, recovery, directory, emit)
            env = PlanetBrowserToolEnv(session, supplied_star_class=supplied_star_class, seed=seed)
            emit(
                "state",
                {
                    "stage": "browser_planet_derived",
                    "seed": seed,
                    "status": "running",
                    "browser_status": "connected",
                    "checkpoint": str(checkpoint),
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
                    if recovery:
                        session.require_prefix(action, steps)
                    emit(
                        "action_proposed",
                        {
                            **action.model_dump(mode="json"),
                            "action_source": "checkpoint",
                            "calibration_scope": "browser_transfer_not_calibrated",
                        },
                    )
                    emit("neural_activity", {**diagnostics, "activity_source": "checkpoint"})
                    result = env.step(action)
                    steps += 1
                    emit("action_result", result.model_dump(mode="json"))
                    emit("observation", result.observation.model_dump(mode="json"))
                    notify(
                        {
                            "steps": steps,
                            "verified_fields": list(session.verified),
                            "failure": result.failure_reason,
                        }
                    )
                    if result.terminated or result.truncated:
                        verified = env.transport_verified and result.failure_reason is None
                        outcome = (
                            "planet_derived_transport_verified"
                            if verified
                            else result.failure_reason or "policy_stopped"
                        )
                        break
                else:
                    outcome = "policy_step_limit"
        except Exception as exc:  # noqa: BLE001 - no Playwright URLs/credentials in protocol errors
            outcome = str(exc) if isinstance(exc, BrowserSafetyStop) else "planet_browser_operation_failed"
            emit("error", {"reason": outcome, "exception_type": type(exc).__name__})
        finally:
            if session is not None:
                session.close()
            unchanged = file_hash(checkpoint) == checkpoint_hash
            report = {
                "scope": "four_derived_planet_browser_transport",
                "provenance": provenance,
                "outcome": outcome if unchanged else "checkpoint_changed",
                "steps": steps,
                "checkpoint_unchanged": unchanged,
                "optimizer_updates": 0,
                "planet_transport_verified": verified and unchanged,
                "task_completed": False,
                "browser_acceptance_passed": False,
                "saved": False,
                "assessment_performed": False,
                "submitted": False,
                "write_attempts": sorted(session.attempted) if session else [],
                "verified_fields": session.verified if session else {},
                "elapsed_seconds": time.monotonic() - started,
            }
            emit("episode_summary", report)
    report["events_sha256"] = file_hash(directory / "events.jsonl")
    write_json(directory / "report.json", report)
    return report
