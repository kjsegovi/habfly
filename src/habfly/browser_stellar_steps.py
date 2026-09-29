"""Cooperative, frozen stellar inference: one learned decision per advance.

The caller owns scheduling/pause boundaries and the already classified page.
No setup, class/prefix selection, navigation, Save, assessment, or training runs
here. Native sub-events remain in their original numeric/color journals; the
optional callback receives those exact serialized events after they flush.
"""

import json
import time
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_color import BrowserColorEnv, ColorJournal
from .browser_full_color import FullStellarColorSession
from .browser_full_stellar import FullStellarSession, FullStellarToolEnv, load_full_stellar_policy
from .browser_numeric import NumericJournal
from .browser_stellar import StellarMappingError
from .training.color import file_hash, require_browser_color_gate


class _CancelledAdvance(Exception):
    pass


class _ForwardJournal:
    def __init__(self, *args, emit, **kwargs):
        self._event_callback = emit
        self._forward_offset = 0
        self.forward_failed = False
        self.cancelled = lambda: False
        self.interruptible = lambda: False
        super().__init__(*args, **kwargs)

    def emit(self, event, payload):
        if self.cancelled():
            raise _CancelledAdvance
        super().emit(event, payload)
        # Do not reconstruct/enrich events independently of the journal or
        # reassign its run_id/sequence. Its flushed JSONL is the single source.
        with (self.output / "events.jsonl").open("rb") as stream:
            stream.seek(self._forward_offset)
            pending = stream.read()
        self._forward_offset += len(pending)
        if self.forward_failed:
            return
        for raw in pending.splitlines():
            try:
                self._event_callback(json.loads(raw))
            except Exception:  # noqa: BLE001 - callback/driver details may contain secrets
                self.forward_failed = True
                if self.interruptible():
                    raise BrowserSafetyStop("stellar_event_forwarding_failed") from None
                break
            if self.cancelled():
                raise _CancelledAdvance


class _NumericJournal(_ForwardJournal, NumericJournal):
    pass


class _ColorJournal(_ForwardJournal, ColorJournal):
    pass


class _StellarSteps:
    def _base(self, page, output, checkpoint, emit):
        if not callable(emit):
            raise TypeError("An event callback must be callable")
        self.page, self.output, self.checkpoint = page, Path(output), Path(checkpoint)
        self.steps, self.finished, self.report = 0, False, None
        self.session = self.env = self._hidden = None
        self._outcome, self._verified = "not_started", False
        self._started = time.monotonic()
        self._checkpoint_unchanged = None
        self._finalizing = self._advancing = False

    def _bind_journal(self):
        self.journal.cancelled = lambda: self.finished and not self._finalizing
        self.journal.interruptible = lambda: not self._finalizing

    def _check(self):
        if self.finished:
            raise _CancelledAdvance
        if self.journal.forward_failed:
            raise BrowserSafetyStop("stellar_event_forwarding_failed")
        try:
            unchanged = file_hash(self.checkpoint) == self._checkpoint_hash
        except OSError:
            unchanged = False
        if not unchanged:
            raise BrowserSafetyStop(self._changed_outcome)

    def _error(self, exc):
        reason = (
            str(exc) if isinstance(exc, (BrowserSafetyStop, StellarMappingError)) else self._failed_outcome
        )
        try:
            self.journal.emit(
                "error",
                {
                    "reason": reason,
                    "exception_type": type(exc).__name__,
                    "writes_may_have_occurred": bool(self.session and self.session.attempts),
                },
            )
        except (BrowserSafetyStop, _CancelledAdvance):
            # The error has already flushed when forwarding invokes the
            # consumer. A failing/aborting consumer cannot prevent the local
            # terminal report or leave the native session writable.
            pass
        finally:
            self._finish(reason, False)

    def state(self):
        """A local snapshot; querying state never advances browser or model."""
        return {
            "component": self._component,
            "status": "finished" if self.finished else "ready",
            "finished": self.finished,
            "outcome": self._outcome,
            "steps": self.steps,
            "step_limit": self._limit,
            "write_attempts": self.session.attempts if self.session else 0,
            "verified_fields": list(getattr(self.session, "verified", [])),
            "transport_verified": self._verified,
            "checkpoint_unchanged": self._checkpoint_unchanged,
            "optimizer_updates": 0,
            "task_completed": False,
            "event_forwarding_failed": self.journal.forward_failed,
        }

    def abort(self):
        """Finish permanently. A native write already dispatched is not rolled back."""
        if not self.finished:
            self._finish("operator_aborted", False)
        return self.report

    def close(self):
        return self.abort()

    def advance(self):
        """Perform at most one policy decision, its result, and neural summary."""
        if self.finished:
            return self.state()
        if self._advancing:
            raise BrowserSafetyStop("stellar_component_reentrant_advance")
        import torch

        self._advancing = True
        try:
            self._check()
            observation = self.env.observe()
            self.journal.emit("observation", observation.model_dump(mode="json"))
            self._check()
            # The grad-mode context always exits before returning to a paused
            # runtime. Hidden states are private to this particular component.
            with torch.no_grad():
                action, self._hidden, diagnostics = self.policy.act(observation, self._hidden)
            action.action_confidence = action.target_confidence = None
            action.calibrated = False
            self.journal.emit(
                "action_proposed",
                {
                    "action": action.model_dump(mode="json"),
                    "action_source": self._action_source,
                },
            )
            self._check()
            result = self._step(action)
            self.steps += 1
            self.journal.emit("action_result", result.model_dump(mode="json"))
            self._check()
            with torch.no_grad():
                activity = self._activity(diagnostics)
            self.journal.emit("neural_activity", activity)
            self._check()
            if result.terminated or result.truncated:
                verified = self._successful(result)
                self._finish(
                    self._success_outcome if verified else result.failure_reason or "policy_stopped", verified
                )
            elif self.steps >= self._limit:
                self._finish(self._limit_outcome, False)
            else:
                self._outcome = "in_progress"
        except _CancelledAdvance:
            pass
        except (KeyboardInterrupt, SystemExit):
            self._finish("operator_aborted", False)
            raise
        except Exception as exc:  # noqa: BLE001 - journal only sanitized failure reasons
            if not self.finished:
                self._error(exc)
        finally:
            # An abort inside policy.act may finish before its return assigns
            # the newly produced state. Terminal components retain no hidden
            # state, even when cancellation races that assignment.
            if self.finished:
                self._hidden = None
            self._advancing = False
        return self.state()

    def _finish(self, outcome, verified):
        if self.finished:
            return self.report
        # Mark terminal before emitting summary so a callback's abort cannot
        # reenter finalization or allow a later native action in this advance.
        self.finished = True
        self._finalizing = True
        self._outcome, self._verified = outcome, verified
        if self.session is not None:
            self.session.stopped = True
            self.page.remove_listener("dialog", self.session._dialog)
        try:
            self._checkpoint_unchanged = file_hash(self.checkpoint) == self._checkpoint_hash
        except OSError:
            self._checkpoint_unchanged = False
        if not self._checkpoint_unchanged:
            self._outcome, self._verified = self._changed_outcome, False
        self._hidden = None
        try:
            self.report = self._final_report()
        except Exception:  # noqa: BLE001 - artifact/driver errors may contain private paths
            # A successful native readback is not a successful recorded run
            # when its journal/manifest cannot be finalized. Remain terminal;
            # never retry the browser work to repair artifact storage.
            self._outcome, self._verified = "stellar_report_finalization_failed", False
            self.report = {
                "schema_version": 1,
                "outcome": self._outcome,
                "artifact_finalization_failed": True,
                "transport_verified": False,
                "task_completed": False,
                "optimizer_updates": 0,
                "write_attempts": self.session.attempts if self.session else 0,
            }
            try:
                self.journal.stream.close()
                with (self.output / "finalization_failed.json").open("x") as stream:
                    stream.write(json.dumps(self.report, indent=2) + "\n")
            except OSError:
                # Disk failure can prevent even a failure receipt. state()
                # and report remain truthful and no later action is allowed.
                pass
        finally:
            self._finalizing = False
        return self.report


class FullStellarNumericSteps(_StellarSteps):
    _component = "full_stellar_numeric"
    _limit = 128
    _action_source = "frozen_lifetime_checkpoint"
    _success_outcome = "full_stellar_numeric_transport_verified"
    _failed_outcome = "browser_full_stellar_operation_failed"
    _changed_outcome = "checkpoint_changed"
    _limit_outcome = "policy_step_limit"

    def __init__(
        self,
        page,
        config,
        output,
        *,
        selected_class,
        lifetime_prefix,
        dataset,
        checkpoint,
        graph_path,
        seed=8500000,
        emit=lambda _: None,
    ):
        self._base(page, output, checkpoint, emit)
        try:
            self.policy, self.provenance = load_full_stellar_policy(
                Path(dataset), self.checkpoint, Path(graph_path)
            )
        except Exception:  # noqa: BLE001 - redact loader paths and driver details
            raise BrowserSafetyStop("stellar_checkpoint_loading_failed") from None
        self._checkpoint_hash = self.provenance["checkpoint_sha256"]
        self.provenance.update(
            browser_execution="autonomous",
            scope="six_calculation_browser_transport",
            selected_class=selected_class,
            lifetime_prefix=lifetime_prefix,
            seed=seed,
            browser_validation="full_at_calculation_each_native_copy_and_completion",
            local_tool_observations="pinned_visible_measurements_not_live_browser_state",
        )
        self.journal = _NumericJournal(
            self.output, learned_policy=True, provenance=self.provenance, emit=emit
        )
        self._bind_journal()
        try:
            self._check()
            self.session = FullStellarSession(
                page, config, self.journal, selected_class=selected_class, lifetime_prefix=lifetime_prefix
            )
            self.session.start()
            self.env = FullStellarToolEnv(self.session, seed)
            self._check()
            self._outcome = "ready"
        except (KeyboardInterrupt, SystemExit):
            self._finish("operator_aborted", False)
            raise
        except Exception as exc:  # noqa: BLE001 - persist sanitized startup failures
            self._error(exc)

    def _step(self, action):
        self.env.copy_approved = True
        self.env.copy_authorization = "autonomous_opt_in"
        return self.env.step(action)

    def _activity(self, diagnostics):
        return self.policy.neural_activity(self._hidden)

    def _successful(self, result):
        return bool(self.env.transport_verified and result.failure_reason is None)

    def _final_report(self):
        report = {
            "schema_version": 1,
            "scope": "six_calculation_browser_transport",
            "outcome": self._outcome,
            "full_stellar_numeric_transport_verified": self._verified,
            "learned_policy": True,
            "classification_learned": False,
            "classification_correctness_verified": False,
            "task_completed": False,
            "browser_acceptance_passed": False,
            "saved": False,
            "assessment_performed": False,
            "submitted": False,
            "optimizer_updates": 0,
            "steps": self.steps,
            "write_attempts": self.session.attempts if self.session else 0,
            "verified_fields": list(self.session.verified) if self.session else [],
            "numeric_readbacks": deepcopy(self.journal.numeric_readbacks),
            "provenance": self.provenance,
            "elapsed_seconds": time.monotonic() - self._started,
            "checkpoint_unchanged": self._checkpoint_unchanged,
        }
        self.journal.emit("episode_summary", report)
        self.journal.stream.close()
        report["events_sha256"] = file_hash(self.output / "events.jsonl")
        with (self.output / "manifest.json").open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
        return report


class FullStellarColorSteps(_StellarSteps):
    _component = "full_stellar_color"
    _limit = 8
    _action_source = "frozen_color_checkpoint"
    _success_outcome = "color_transport_verified"
    _failed_outcome = "full_stellar_color_failed"
    _changed_outcome = "color_checkpoint_changed"
    _limit_outcome = "color_step_limit"

    def __init__(
        self,
        page,
        config,
        output,
        *,
        selected_class,
        lifetime_prefix,
        experiment,
        graph_path,
        emit=lambda _: None,
    ):
        self._base(page, output, Path(experiment) / "training/checkpoint.pt", emit)
        try:
            self.policy, self.reference = require_browser_color_gate(Path(experiment), Path(graph_path))
            self._checkpoint_hash = file_hash(self.checkpoint)
        except Exception:  # noqa: BLE001 - redact loader paths and driver details
            raise BrowserSafetyStop("color_checkpoint_loading_failed") from None
        self.policy.calibration = {"status": "uncalibrated", "scope": "browser_transfer_not_calibrated"}
        self.provenance = {
            "color_gate_passed": True,
            "color_reference_hash": self.reference.checksum,
            "color_checkpoint_sha256": self._checkpoint_hash,
            "browser_execution": "autonomous",
            "optimizer_updates": 0,
            "classified_view": True,
        }
        self.journal = _ColorJournal(self.output, self.provenance, emit=emit)
        self._bind_journal()
        try:
            self._check()
            self.session = FullStellarColorSession(
                page,
                config,
                self.journal,
                self.reference,
                selected_class=selected_class,
                lifetime_prefix=lifetime_prefix,
            )
            self.session.start()
            self.env = BrowserColorEnv(self.session)
            self._check()
            self._outcome = "ready"
        except (KeyboardInterrupt, SystemExit):
            self._finish("operator_aborted", False)
            raise
        except Exception as exc:  # noqa: BLE001 - persist sanitized startup failures
            self._error(exc)

    def _step(self, action):
        return self.env.step(action, confirm=lambda _: True, authorization="autonomous_opt_in")

    def _activity(self, diagnostics):
        return (
            diagnostics if diagnostics.get("activity_pathway") else self.policy.neural_activity(self._hidden)
        )

    def _successful(self, result):
        return bool(self.env.completed and self.journal.receipt and not result.failure_reason)

    def _final_report(self):
        return self.journal.finish(self._outcome, self.session.attempts if self.session else 0)
