"""Cooperative frozen planet-derived inference, never setup or recovery.

One advance performs one learned decision. The caller owns pause scheduling;
native exact-copy guards still own every browser write. Callbacks receive the
actual flushed version-1 journal envelope, without resequencing or enrichment.
"""

import json
import time
from copy import deepcopy
from pathlib import Path

import torch

from .browser import BrowserSafetyStop
from .browser_planet_numeric import PlanetNumericSession
from .browser_planet_policy import PlanetBrowserToolEnv, require_final_gate
from .contracts import RuntimeEvent
from .environments.planet_calculations import SCOPE
from .model import graph_fingerprint
from .planet_knowledge import PlanetCalculator
from .training.checkpoints import load_checkpoint
from .training.planet_sequence import file_hash, read
from .training.stellar import write_json
from .training.train import prepare_output_directory


class _CancelledAdvance(Exception):
    pass


class PlanetDerivedSteps:
    """At most 128 decisions and four derived copies from existing visible inputs.

    No ``resume_from`` option is accepted: partial or uncertain native writes
    are terminal. Loading reads the already completed final-gate *report* only;
    it never opens final cases, trains, or performs another evaluation.
    """

    _limit = 128
    _report_scope = "four_derived_planet_browser_transport"

    def __init__(
        self,
        page,
        config,
        output,
        *,
        pilot,
        final_evaluation,
        graph,
        supplied_star_class,
        seed=15000000,
        emit=lambda _: None,
    ):
        if not callable(emit):
            raise TypeError("An event callback must be callable")
        self.page, self.graph, self.output = page, graph, Path(output)
        self.checkpoint = Path(pilot) / "training/checkpoint.pt"
        self._source_paths = {
            "checkpoint_sha256": self.checkpoint,
            "checkpoint_metadata_sha256": self.checkpoint.with_suffix(".pt.json"),
            "final_gate_report_sha256": Path(final_evaluation) / "report.json",
        }
        self._callback, self._seed = emit, seed
        self.steps, self.finished, self.report = 0, False, None
        self.session = self.env = self._hidden = None
        self._outcome, self._verified = "not_started", False
        self._advancing = self._finalizing = self._forward_failed = False
        self._checkpoint_unchanged = self._sources_unchanged = None
        self._started, self._sequence = time.monotonic(), 0
        try:
            self._source_hashes = {name: file_hash(path) for name, path in self._source_paths.items()}
            self._load_model(graph)
            self._check_sources()
        except Exception:  # noqa: BLE001 - loader paths/details must not reach runtime logs
            raise BrowserSafetyStop("planet_checkpoint_loading_failed") from None
        self.policy.calibration = {"status": "uncalibrated", "scope": "browser_transfer_not_calibrated"}
        self.provenance = {
            **self._source_hashes,
            "graph_hash": self._graph_hash,
            "knowledge_pack_hash": self._pack_hash,
            "optimizer_updates": 0,
            "raw_measurements": "separate_visible_readback_stage",
            "classification_source": "supplied_not_learned",
            "supplied_star_class": supplied_star_class,
            "calibration_scope": "browser_transfer_not_calibrated",
            "browser_validation": "full_before_and_after_each_native_copy_and_at_completion",
            "local_tool_observations": "pinned_visible_measurements_not_live_browser_state",
            **self._additional_provenance(),
        }
        self.output = prepare_output_directory(self.output)
        self._stream = (self.output / "events.jsonl").open("x")
        try:
            self._emit(
                "hello",
                {
                    "protocol_version": 1,
                    "task": "planet_calculations",
                    "policy": "checkpoint",
                    **self.provenance,
                },
            )
            self._check()
            self.session = PlanetNumericSession(
                page, config, self.output / "native-copies", emit=self._native_emit, max_seconds=900
            )
            self.env = self._environment(supplied_star_class=supplied_star_class, seed=seed)
            self._emit(
                "state",
                {
                    "stage": "browser_planet_derived",
                    "seed": seed,
                    "status": "running",
                    "browser_status": "connected",
                    "checkpoint": str(self.checkpoint),
                    **self.provenance,
                },
            )
            self._check()
            self._outcome = "ready"
        except (KeyboardInterrupt, SystemExit):
            self._finish("operator_aborted", False)
            raise
        except Exception as exc:  # noqa: BLE001 - persist only sanitized startup failures
            self._error(exc)

    def _load_model(self, graph):
        """Legacy v1 loader; a new scope must supply its own complete gate."""
        require_final_gate(
            read(self._source_paths["final_gate_report_sha256"]), self._source_hashes["checkpoint_sha256"]
        )
        content = read(self._source_paths["checkpoint_metadata_sha256"])["provenance"]["planet_calculations"]
        if content["scope"] != SCOPE or content["knowledge_pack_hash"] != PlanetCalculator().pack.checksum:
            raise ValueError("Planet checkpoint scope/pack mismatch")
        torch.set_num_threads(1)
        self.policy, manifest = load_checkpoint(self.checkpoint, graph, content_pack=content)
        if (
            len(graph.body_ids) != 2000
            or self.policy.hidden_size != 16
            or self.policy.observation_encoding != "structured_planet_tool_v1"
        ):
            raise ValueError("Planet browser model contract mismatch")
        self._graph_hash = manifest["graph_hash"]
        self._pack_hash = content["knowledge_pack_hash"]

    def _current_pack_hash(self):
        return PlanetCalculator().pack.checksum

    def _additional_provenance(self):
        return {}

    def _environment(self, *, supplied_star_class, seed):
        return PlanetBrowserToolEnv(self.session, supplied_star_class=supplied_star_class, seed=seed)

    def _inference_observation(self, observation):
        # The authoritative observation remains the event payload. New scoped
        # adapters must declare and hash any separate model-only view.
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
        except Exception:  # noqa: BLE001 - corrupted/missing source must stop cached inference
            # Missing/malformed files are not permission to use cached values.
            other_ok = False
        self._checkpoint_unchanged, self._sources_unchanged = checkpoint_ok, checkpoint_ok and other_ok
        if not checkpoint_ok:
            raise BrowserSafetyStop("checkpoint_changed")
        if not other_ok:
            raise BrowserSafetyStop("planet_source_changed")

    def _check(self):
        if self.finished:
            raise _CancelledAdvance
        if self._forward_failed:
            raise BrowserSafetyStop("planet_event_forwarding_failed")
        self._check_sources()

    def _emit(self, event, payload):
        if self.finished and not self._finalizing:
            raise _CancelledAdvance
        line = RuntimeEvent(
            event=event, sequence=self._sequence, run_id=f"planet-browser-{self._seed}", payload=payload
        ).model_dump_json()
        self._stream.write(line + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            try:
                self._callback(json.loads(line))
            except Exception:  # noqa: BLE001 - callback details may contain credentials
                self._forward_failed = True
                if not self._finalizing:
                    raise BrowserSafetyStop("planet_event_forwarding_failed") from None
        if self.finished and not self._finalizing:
            raise _CancelledAdvance

    def _native_emit(self, event, payload):
        # PlanetNumericSession calls this after its durable reservation and
        # before its final identity check/fill. Callback aborts and changed
        # source files therefore cannot authorize the following write.
        self._check()
        self._emit(event, payload)
        self._check()

    def state(self):
        """Pure local status, with no browser/model activity."""
        return {
            "component": "planet_derived",
            "status": "finished" if self.finished else "ready",
            "finished": self.finished,
            "outcome": self._outcome,
            "steps": self.steps,
            "step_limit": self._limit,
            "write_attempts": len(self.session.attempted) if self.session else 0,
            "verified_fields": list(self.session.verified) if self.session else [],
            "transport_verified": self._verified,
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
            self._finish("planet_component_reentrant_advance", False)
            raise BrowserSafetyStop("planet_component_reentrant_advance")
        self._advancing = True
        try:
            self._check()
            observation = self.env.observe()
            self._emit("observation", observation.model_dump(mode="json"))
            self._check()
            with torch.no_grad():
                action, self._hidden, diagnostics = self.policy.act(
                    self._inference_observation(observation), self._hidden
                )
            action.action_confidence = action.target_confidence = None
            action.calibrated = False
            self._emit(
                "action_proposed",
                {
                    **action.model_dump(mode="json"),
                    "action_source": "checkpoint",
                    "calibration_scope": "browser_transfer_not_calibrated",
                },
            )
            self._check()
            self._emit("neural_activity", {**diagnostics, "activity_source": "checkpoint"})
            self._check()
            result = self.env.step(action)
            self.steps += 1
            self._emit("action_result", result.model_dump(mode="json"))
            self._emit("observation", result.observation.model_dump(mode="json"))
            self._check()
            if result.terminated or result.truncated:
                verified = bool(self.env.transport_verified and result.failure_reason is None)
                self._finish(
                    "planet_derived_transport_verified"
                    if verified
                    else result.failure_reason or "policy_stopped",
                    verified,
                )
            elif self.steps >= self._limit:
                self._finish("policy_step_limit", False)
            else:
                self._outcome = "in_progress"
        except _CancelledAdvance:
            pass
        except (KeyboardInterrupt, SystemExit):
            self._finish("operator_aborted", False)
            raise
        except Exception as exc:  # noqa: BLE001 - driver details may contain private URLs
            if not self.finished:
                self._error(exc)
        finally:
            self._advancing = False
            if self.finished:
                self._hidden = None
        return self.state()

    def _error(self, exc):
        reason = str(exc) if isinstance(exc, BrowserSafetyStop) else "planet_browser_operation_failed"
        try:
            self._emit("error", {"reason": reason, "exception_type": type(exc).__name__})
        except Exception:  # noqa: BLE001 - preserve terminal state even if error journaling fails
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
            "planet_transport_verified": self._verified,
            "task_completed": False,
            "browser_acceptance_passed": False,
            "saved": False,
            "assessment_performed": False,
            "submitted": False,
            "write_attempts": sorted(self.session.attempted) if self.session else [],
            "verified_fields": deepcopy(self.session.verified) if self.session else {},
            "elapsed_seconds": time.monotonic() - self._started,
        }

    def _finish(self, outcome, verified):
        if self.finished:
            return self.report
        self.finished, self._finalizing = True, True
        self._outcome, self._verified = outcome, verified
        self._hidden = None
        try:
            if self.session is not None:
                self.session.stopped = True
                self.session.close()
            try:
                self._check_sources()
            except BrowserSafetyStop as exc:
                self._outcome, self._verified = str(exc), False
            if self._forward_failed and self._verified:
                self._outcome, self._verified = "planet_event_forwarding_failed", False
            report = self._report()
            self._emit("episode_summary", report)
            # A callback can fail or modify a source even on the terminal event.
            # Persist a final corrective failure summary; no further callback
            # or browser work runs. Successful journals retain exactly one.
            try:
                self._check_sources()
                if self._forward_failed and self._verified:
                    raise BrowserSafetyStop("planet_event_forwarding_failed")
            except BrowserSafetyStop as exc:
                if self._outcome != str(exc) or self._verified:
                    self._outcome, self._verified = str(exc), False
                    self._forward_failed = True
                    self._emit("error", {"reason": self._outcome, "exception_type": "BrowserSafetyStop"})
                    report = self._report()
                    self._emit("episode_summary", report)
            self._stream.close()
            report["events_sha256"] = file_hash(self.output / "events.jsonl")
            write_json(self.output / "report.json", report)
            self.report = report
        except BaseException as exc:  # includes interruption while forwarding the terminal summary
            interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
            self._outcome, self._verified = (
                "operator_aborted" if interrupted else "planet_report_finalization_failed",
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
        """Permanently stop; dispatched native writes are never rolled back."""
        return self._finish("operator_aborted", False)

    def close(self):
        return self.abort()
