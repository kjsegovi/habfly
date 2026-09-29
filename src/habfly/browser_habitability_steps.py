"""Cooperative frozen temperature proposals and one separate equilibrium copy.

The caller owns the browser and pause scheduling. No gas, greenhouse, water or
habitability choice is made here; surface temperature remains a local proposal.
Callbacks receive the exact flushed version-1 envelope, never an enriched copy.
"""

import json
import time
from copy import deepcopy
from pathlib import Path

import torch

from .browser import BrowserSafetyStop
from .browser_habitability_numeric import HabitabilityNumericSession
from .browser_habitability_policy import HabitabilityBrowserToolEnv
from .contracts import RuntimeEvent
from .habitability_knowledge import HabitabilityCalculator
from .model import graph_fingerprint
from .training.habitability_calculations import ENCODING
from .training.habitability_evaluation import load_validated_pilot, require_frozen_final_gate
from .training.planet_period import PILOT_COUNTS
from .training.planet_sequence import file_hash, read
from .training.stellar import write_json
from .training.train import prepare_output_directory


class _CancelledAdvance(Exception):
    pass


class _BoundaryCheckedNumericSession(HabitabilityNumericSession):
    """Add cancellation to the existing post-reservation identity check.

    The original adapter still owns the reservation, fill, readback and receipts.
    Recheck the screen after the callback as well: forwarding cannot authorize a
    write against a page or model source changed by that callback.
    """

    def __init__(self, *args, guard, before_write, **kwargs):
        self._guard, self._before_write, self._boundary_emitted = guard, before_write, False
        super().__init__(*args, **kwargs)

    def current(self):
        self._guard()
        current = super().current()
        self._guard()
        if self.attempted and not self._boundary_emitted:
            self._boundary_emitted = True
            self._before_write()
            self._guard()
            current = super().current()
            self._guard()
        return current


class HabitabilityTemperatureSteps:
    """At most 128 learned decisions, followed by one cancellable copy step.

    The validated loader reads recorded pilot evidence only. The final evaluation
    directory contributes its gate report, never final cases or another evaluation.
    Aborted/uncertain components cannot resume; starting one is not a recovery API.
    """

    _limit = 128
    _report_scope = "one_equilibrium_copy_with_supplied_warming_local_proposal"

    def __init__(
        self,
        page,
        config,
        output,
        *,
        pilot,
        final_evaluation,
        graph,
        supplied_greenhouse_increment,
        seed=20000000,
        emit=lambda _: None,
    ):
        if not callable(emit):
            raise TypeError("An event callback must be callable")
        if type(supplied_greenhouse_increment) not in {int, float} or supplied_greenhouse_increment not in {
            0,
            10,
            30,
            100,
        }:
            raise BrowserSafetyStop("unsupported_supplied_greenhouse_increment")
        self.page, self.graph, self.output = page, graph, Path(output)
        pilot = Path(pilot)
        self.checkpoint = pilot / "training/checkpoint.pt"
        self._source_paths = {
            "checkpoint_sha256": self.checkpoint,
            "checkpoint_metadata_sha256": self.checkpoint.with_suffix(".pt.json"),
            "pilot_report_sha256": pilot / "report.json",
            "final_gate_report_sha256": Path(final_evaluation) / "report.json",
        }
        for split in PILOT_COUNTS:
            self._source_paths[f"pilot_{split}_cases_sha256"] = pilot / f"private-{split}-cases.json"
            if split != "gate":
                self._source_paths[f"pilot_{split}_episodes_sha256"] = pilot / f"expert-{split}/episodes.json"
        for split in ("train", "development"):
            self._source_paths[f"pilot_{split}_summaries_sha256"] = pilot / f"learned-{split}/summaries.json"
        self._callback, self._seed = emit, seed
        self.steps, self.finished, self.report = 0, False, None
        self.session = self.env = self._hidden = self.receipt = None
        self._outcome, self._verified, self._phase = "not_started", False, "reasoning"
        self._advancing = self._finalizing = self._forward_failed = False
        self._checkpoint_unchanged = self._sources_unchanged = None
        self._started, self._sequence = time.monotonic(), 0
        try:
            self._source_hashes = {name: file_hash(path) for name, path in self._source_paths.items()}
            self._load_model(pilot, graph)
            self._check_sources()
        except Exception:  # noqa: BLE001 - never expose loader paths or private data
            raise BrowserSafetyStop("temperature_checkpoint_loading_failed") from None
        self.policy.calibration = {"status": "uncalibrated", "scope": "browser_transfer_not_calibrated"}
        self.provenance = {
            **self._source_hashes,
            "graph_hash": self._graph_hash,
            "knowledge_pack_hash": self._pack_hash,
            "optimizer_updates": 0,
            "supplied_greenhouse_increment": supplied_greenhouse_increment,
            "greenhouse_selection_learned": False,
            "gas_identification_learned": False,
            "water_phase_learned": False,
            "habitability_decision_learned": False,
            "calibration_scope": "browser_transfer_not_calibrated",
            "surface_proposal_is_not_browser_readback": True,
            **self._additional_provenance(),
        }
        self.output = prepare_output_directory(self.output)
        self._stream = (self.output / "events.jsonl").open("x")
        try:
            self._emit(
                "hello",
                {
                    "protocol_version": 1,
                    "task": "habitability_calculations",
                    "policy": "checkpoint",
                    **self.provenance,
                },
            )
            self._check()
            self.session = _BoundaryCheckedNumericSession(
                page,
                config,
                self.output / "native-copy",
                max_seconds=300,
                guard=self._check,
                before_write=self._native_boundary,
            )
            self.env = self._environment(
                supplied_greenhouse_increment=supplied_greenhouse_increment, seed=seed
            )
            self._emit(
                "state",
                {
                    "stage": "browser_supplied_temperatures",
                    "status": "running",
                    "browser_status": "connected",
                    **self.provenance,
                },
            )
            self._check()
            self._outcome = "ready"
        except (KeyboardInterrupt, SystemExit):
            self._finish("operator_aborted", False)
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize startup errors
            self._error(exc)

    def _load_model(self, pilot, graph):
        """Legacy v1 loader; new input scopes must validate their own gate."""
        content = read(self._source_paths["checkpoint_metadata_sha256"])["provenance"][
            "habitability_calculations"
        ]
        # Do not allow an injected final/test split to reach the pilot loader.
        if set(content["splits"]) != set(PILOT_COUNTS):
            raise ValueError("Unexpected temperature pilot splits")
        require_frozen_final_gate(
            read(self._source_paths["final_gate_report_sha256"]), self._source_hashes["checkpoint_sha256"]
        )
        self.policy, validated_content, _ = load_validated_pilot(pilot, graph)
        if (
            validated_content != content
            or self.policy.hidden_size != 16
            or self.policy.observation_encoding != ENCODING
        ):
            raise ValueError("Temperature checkpoint contract mismatch")
        self._graph_hash, self._pack_hash = content["graph_hash"], content["knowledge_pack_hash"]
        self.policy.eval()

    def _current_pack_hash(self):
        return HabitabilityCalculator().pack.checksum

    def _additional_provenance(self):
        return {}

    def _environment(self, *, supplied_greenhouse_increment, seed):
        return HabitabilityBrowserToolEnv(
            self.session.mapping, supplied_greenhouse_increment=supplied_greenhouse_increment, seed=seed
        )

    def _inference_observation(self, observation):
        return observation

    def _check_sources(self):
        try:
            checkpoint_ok = file_hash(self.checkpoint) == self._source_hashes["checkpoint_sha256"]
        except OSError:
            checkpoint_ok = False
        try:
            other_ok = all(
                file_hash(path) == self._source_hashes[name]
                for name, path in self._source_paths.items()
                if name != "checkpoint_sha256"
            )
            other_ok = (
                other_ok
                and graph_fingerprint(self.graph) == self._graph_hash
                and self._current_pack_hash() == self._pack_hash
            )
        except Exception:  # noqa: BLE001 - corrupted/missing source blocks cached inference
            other_ok = False
        self._checkpoint_unchanged, self._sources_unchanged = checkpoint_ok, checkpoint_ok and other_ok
        if not checkpoint_ok:
            raise BrowserSafetyStop("checkpoint_changed")
        if not other_ok:
            raise BrowserSafetyStop("temperature_source_changed")

    def _check(self):
        if self.finished:
            raise _CancelledAdvance
        if self._forward_failed:
            raise BrowserSafetyStop("temperature_event_forwarding_failed")
        self._check_sources()

    def _emit(self, event, payload):
        if self.finished and not self._finalizing:
            raise _CancelledAdvance
        line = RuntimeEvent(
            event=event, sequence=self._sequence, run_id=f"temperature-browser-{self._seed}", payload=payload
        ).model_dump_json()
        self._stream.write(line + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            try:
                self._callback(json.loads(line))
            except Exception:  # noqa: BLE001 - callback errors may contain credentials
                self._forward_failed = True
                if not self._finalizing:
                    raise BrowserSafetyStop("temperature_event_forwarding_failed") from None
        if self.finished and not self._finalizing:
            raise _CancelledAdvance

    def _native_boundary(self):
        self._check()
        self._emit(
            "action_proposed",
            {
                "kind": "TYPE",
                "destination": "equilibrium_temp",
                "value": str(self.env.answers["equilibrium_temp"]),
                "action_source": "deterministic_exact_transport_of_checkpoint_result",
                "native_write_boundary": True,
            },
        )
        self._check()

    def state(self):
        """Pure status; it neither advances inference nor reads the browser."""
        return {
            "component": "habitability_temperature",
            "phase": self._phase,
            "status": "finished" if self.finished else "ready",
            "finished": self.finished,
            "outcome": self._outcome,
            "steps": self.steps,
            "step_limit": self._limit,
            "write_attempts": int(self.session.attempted) if self.session else 0,
            "temperature_proposals_complete": bool(self.env and self.env.proposals_complete),
            "equilibrium_transport_verified": self._verified,
            "checkpoint_unchanged": self._checkpoint_unchanged,
            "sources_unchanged": self._sources_unchanged,
            "optimizer_updates": 0,
            "task_completed": False,
            "event_forwarding_failed": self._forward_failed,
        }

    def advance(self):
        if self.finished:
            return self.state()
        if self._advancing:
            self._finish("temperature_component_reentrant_advance", False)
            raise BrowserSafetyStop("temperature_component_reentrant_advance")
        self._advancing = True
        try:
            self._check()
            if self._phase == "ready_to_copy":
                self.receipt = self.session.copy(
                    str(self.env.answers["equilibrium_temp"]), "K", source="checkpoint"
                )
                self._emit("action_result", {"native_copy": self.receipt})
                self._check()
                if self.receipt.get("readback_verified") is not True:
                    raise BrowserSafetyStop("temperature_copy_not_verified")
                self._finish("equilibrium_transport_verified", True)
            else:
                observation = self.env.observe()
                self._emit("observation", observation.model_dump(mode="json"))
                self._check()
                with torch.no_grad():
                    action, hidden, diagnostics = self.policy.act(
                        self._inference_observation(observation), self._hidden
                    )
                self._check()
                self._hidden = hidden
                action.action_confidence = action.target_confidence = None
                action.calibrated = False
                self._emit(
                    "action_proposed", {**action.model_dump(mode="json"), "action_source": "checkpoint"}
                )
                self._check()
                self._emit("neural_activity", {**diagnostics, "activity_source": "checkpoint"})
                self._check()
                result = self.env.step(action)
                self.steps += 1
                self._emit("action_result", result.model_dump(mode="json"))
                self._check()
                if result.terminated or result.truncated:
                    if self.env.proposals_complete and result.failure_reason is None and not result.truncated:
                        self._phase, self._outcome = "ready_to_copy", "temperature_proposals_complete"
                        self._hidden = None
                        self._emit("state", self.state())
                        self._check()
                    else:
                        self._finish(result.failure_reason or "temperature_proposals_incomplete", False)
                elif self.steps >= self._limit:
                    self._finish("temperature_policy_step_limit", False)
                else:
                    self._outcome = "in_progress"
        except _CancelledAdvance:
            pass
        except (KeyboardInterrupt, SystemExit):
            self._finish("operator_aborted", False)
            raise
        except Exception as exc:  # noqa: BLE001 - browser errors may contain private URLs
            if not self.finished:
                self._error(exc)
        finally:
            self._advancing = False
            if self.finished:
                self._hidden = None
        return self.state()

    def _error(self, exc):
        reason = str(exc) if isinstance(exc, BrowserSafetyStop) else "temperature_browser_operation_failed"
        try:
            self._emit("error", {"reason": reason, "exception_type": type(exc).__name__})
        except Exception:  # noqa: BLE001 - failure forwarding cannot prevent termination
            self._forward_failed = True
        finally:
            self._finish(reason, False)

    def _report(self):
        return {
            "scope": self._report_scope,
            "provenance": deepcopy(self.provenance),
            "outcome": self._outcome,
            "steps": self.steps,
            "checkpoint_unchanged": self._checkpoint_unchanged,
            "sources_unchanged": self._sources_unchanged,
            "optimizer_updates": 0,
            "equilibrium_transport_verified": self._verified,
            "local_proposals": deepcopy(self.env.answers) if self.env else {},
            "native_receipt": deepcopy(self.receipt),
            "task_completed": False,
            "course_acceptance_passed": False,
            "saved": False,
            "assessed": False,
            "submitted": False,
            "elapsed_seconds": time.monotonic() - self._started,
        }

    def _finish(self, outcome, verified):
        if self.finished:
            return self.report
        self.finished, self._finalizing, self._phase = True, True, "finished"
        self._outcome, self._verified, self._hidden = outcome, verified, None
        try:
            if self.session is not None:
                self.session.stopped = True
                self.session.close()
            try:
                self._check_sources()
            except BrowserSafetyStop as exc:
                self._outcome, self._verified = str(exc), False
            if self._forward_failed and self._verified:
                self._outcome, self._verified = "temperature_event_forwarding_failed", False
            report = self._report()
            self._emit("episode_summary", report)
            try:
                self._check_sources()
                if self._forward_failed and self._verified:
                    raise BrowserSafetyStop("temperature_event_forwarding_failed")
            except BrowserSafetyStop as exc:
                if self._outcome != str(exc) or self._verified:
                    self._outcome, self._verified, self._forward_failed = str(exc), False, True
                    self._emit("error", {"reason": self._outcome, "exception_type": "BrowserSafetyStop"})
                    report = self._report()
                    self._emit("episode_summary", report)
            self._stream.close()
            report["events_sha256"] = file_hash(self.output / "events.jsonl")
            write_json(self.output / "report.json", report)
            self.report = report
        except BaseException as exc:
            interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
            self._outcome, self._verified = (
                "operator_aborted" if interrupted else "temperature_report_finalization_failed",
                False,
            )
            self.report = {**self._report(), "artifact_finalization_failed": True}
            try:
                self._stream.close()
                with (self.output / "finalization_failed.json").open("x") as stream:
                    stream.write(json.dumps(self.report, indent=2) + "\n")
            except OSError:
                pass
            if interrupted:
                raise
        finally:
            self._finalizing = False
        return self.report

    def abort(self):
        """Stop permanently without compensating browser actions or retry."""
        return self._finish("operator_aborted", False)

    def close(self):
        return self.abort()
