"""Cooperative reference-only raster input transport, one native copy per step.

The constructor performs no UI actions. Each advance may perform bounded
ordinary spectrum HOVERs while refreshing evidence, then at most one copy.
Cancellation is cooperative at stage boundaries, not a promise to interrupt
an in-flight Playwright call. The fixed native deadline includes paused time.
"""

import json
from copy import deepcopy

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_planet_numeric import PlanetNumericSession, planet_projection
from .browser_raster_planet_evidence import (
    MODE,
    RAW,
    _action_source,
    _blank_current,
    _failure,
    _flags_for,
    _fresh,
    _Owned,
    _presence,
    _reload,
    _require,
    _reservation,
    _reserve,
    _sha,
)
from .contracts import RuntimeEvent


class _CancelledAdvance(Exception):
    pass


class RasterInputSteps:
    """Consume a confirmed Yes receipt; never select presence or derive answers.

    Success report/native artifacts match ``copy_raster_measured_inputs``.
    Extra runtime JSONL is independent evidence, never a replacement for the
    canonical per-star reservation, native capture chain, or raw event files.
    There is no retry, resume-from-artifacts, or correction path.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        presence_path,
        presence_sha256,
        max_seconds=900,
        emit=lambda _: None,
    ):
        _require(type(max_seconds) in {int, float} and 30 <= max_seconds <= 900, "invalid_time_budget")
        if not callable(emit):
            raise TypeError("An event callback must be callable")
        self.owner = _Owned(run_history)
        self.output = self.owner.output(output)
        self.page, self.config = page, config
        self.presence_path, self.presence_sha256 = self.owner.path(presence_path), presence_sha256
        self.max_seconds, self._callback = max_seconds, emit
        self.finished, self.report, self.steps = False, None, 0
        self.session = self.presence = self.evidence = self.intent = None
        self._flags = _flags_for()
        self._reserved = self._advancing = self._finalizing = self._forward_failed = False
        self._reservation_uncertain = False
        self._outcome = "not_started"
        self._events, self._precopy = [], []
        self._sequence = 0
        self._run_id = "raster-inputs-" + _sha(self.owner.relative(self.output).encode())[:16]
        self._stream = (self.output / "events.jsonl").open("x")
        try:
            self.presence, previous, previous_map = _presence(
                self.owner, self.presence_path, self.presence_sha256
            )
            self.evidence, self.star = self.presence["evidence"], self.presence["star"]
            self._flags = _flags_for(self.evidence)
            _require(not _reservation(self.owner, self.star, "inputs").exists(), "inputs_already_reserved")
            self._emit(
                "hello",
                {
                    "protocol_version": 1,
                    "task": "reference_raster_inputs"
                    if self._flags["provenance"] == MODE
                    else "reference_visible_tooltip_inputs",
                    **self._flags,
                    "maximum_numeric_writes": 3,
                    "cancellation": "between_bounded_freshness_stages_and_before_native_fill",
                    "pause_counts_toward_deadline": True,
                },
            )
            self._check()
            self.session = PlanetNumericSession(
                page, config, self.output / "native-copies", self._native_event, max_seconds=max_seconds
            )
            before, mapping, choices, _ = self.session.current()
            _blank_current(
                mapping, choices, self.evidence, presence="Yes", painted=self.presence["painted_class_after"]
            )
            _require(
                planet_projection(before, mapping) == planet_projection(previous, previous_map)
                and choices == self.presence["painted_choices_after"],
                "presence_current_state_changed",
            )
            self._check()
            self._outcome = "ready"
            self._emit("state", self.state())
            self._check()
        except (KeyboardInterrupt, SystemExit):
            self._stop("operator_aborted")
            raise
        except Exception as exc:  # noqa: BLE001 - redact browser and callback details
            self._stop(self._reason(exc))

    @staticmethod
    def _reason(exc):
        return str(exc) if isinstance(exc, BrowserSafetyStop) else "raster_planet_operation_failed"

    def _check(self):
        if self.finished:
            raise _CancelledAdvance
        if self._forward_failed:
            raise BrowserSafetyStop("raster_event_forwarding_failed")
        presence, _, _ = _presence(self.owner, self.presence_path, self.presence_sha256)
        _require(presence == self.presence, "presence_receipt_changed")
        if self._reserved:
            _require(
                self.owner.json(_reservation(self.owner, self.star, "inputs")) == self.intent
                and self.owner.json(self.output / "reserved.json") == self.intent,
                "inputs_reservation_changed",
            )

    def _emit(self, kind, payload):
        if self.finished and not self._finalizing:
            raise _CancelledAdvance
        line = RuntimeEvent(
            event=kind, payload=payload, run_id=self._run_id, sequence=self._sequence
        ).model_dump_json()
        self._stream.write(line + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            try:
                self._callback(json.loads(line))
            except Exception:  # noqa: BLE001 - callbacks may contain private browser details
                self._forward_failed = True
                if not self._finalizing:
                    raise BrowserSafetyStop("raster_event_forwarding_failed") from None
        if self.finished and not self._finalizing:
            raise _CancelledAdvance

    def _native_event(self, kind, payload):
        self._check()
        event = {"kind": kind, "payload": deepcopy(payload), "provenance": self._flags["provenance"]}
        persist_json(self.output / f"native-event-{len(self._events):02d}.json", event)
        self._events.append(event)
        self._emit(kind, payload)
        self._check()

    def state(self):
        return {
            "component": "raster_inputs",
            "status": "finished" if self.finished else "ready",
            "finished": self.finished,
            "outcome": self._outcome,
            "steps": self.steps,
            "step_limit": 3,
            "next_destination": None if self.finished or self.steps == 3 else RAW[self.steps],
            "write_attempts": len(self.session.attempted) if self.session else 0,
            "verified_fields": list(self.session.verified) if self.session else [],
            "transport_verified": self._outcome == "raw_measurement_transport_verified",
            "reservation_created": self._reserved,
            "event_forwarding_failed": self._forward_failed,
            "task_completed": False,
            "provenance": self._flags["provenance"],
            "approximate": True,
            "learned_perception": False,
            "scientific_verified": False,
            "optimizer_updates": 0,
        }

    def advance(self):
        if self.finished:
            return self.state()
        if self._advancing:
            self._stop("raster_component_reentrant_advance")
            raise BrowserSafetyStop("raster_component_reentrant_advance")
        self._advancing = True
        try:
            self._check()
            self.session.current()
            if not self._reserved:
                fresh = _fresh(self.page, self.config, self.owner, self.output / "fresh", self.evidence)
                self._check()
                self.session.current()
                self.intent = {
                    **self._flags,
                    "schema_version": 1,
                    "mode": self._flags["provenance"],
                    "stage": "inputs",
                    "star": self.star,
                    "evidence": self.evidence,
                    "fresh": fresh,
                    "output": self.owner.relative(self.output),
                    "presence_path": self.owner.relative(self.presence_path),
                    "presence_sha256": self.presence_sha256,
                    "maximum_numeric_writes": 3,
                    "max_seconds": self.max_seconds,
                    "destinations": list(RAW),
                    "action_source": _action_source(self.evidence),
                }
                _reserve(self.owner, self.output, self.star, "inputs", self.intent)
                self._reserved = True
                self._emit("state", {**self.state(), "stage": "inputs_reserved"})
                self._check()
            self._emit("state", {**self.state(), "stage": "refreshing_reference_evidence"})
            self._check()
            directory = self.output / f"precopy-{self.steps + 1}"
            current = _fresh(self.page, self.config, self.owner, directory, self.evidence)
            self._check()
            self._precopy.append(current)
            persist_json(directory / "evidence.json", current)
            self.session.current()
            self._emit("state", {**self.state(), "stage": "reference_evidence_checked"})
            self._check()
            spec = self.evidence["measurements"][RAW[self.steps]]
            self.session.copy(RAW[self.steps], spec["value"], spec["unit"], source="reference_diagnostic")
            self.steps += 1
            self._check()
            if self.steps == 3:
                self.session.current()
                _reload(self.owner, self.evidence)
                self._complete()
            else:
                self._outcome = "in_progress"
                self._emit("state", self.state())
                self._check()
        except _CancelledAdvance:
            pass
        except (KeyboardInterrupt, SystemExit):
            self._stop("operator_aborted")
            raise
        except Exception as exc:  # noqa: BLE001 - only normalized stop reasons are journaled
            if not self.finished:
                self._stop(self._reason(exc))
        finally:
            self._advancing = False
        return self.state()

    def _complete(self):
        # Do not publish the success report until callbacks and frozen sources
        # have passed the final stage boundary. No later native action exists.
        report = {
            **self.intent,
            "raw_measurement_transport_verified": True,
            "numeric_writes": 3,
            "verified_fields": deepcopy(self.session.verified),
            "precopy": deepcopy(self._precopy),
            "native_events": deepcopy(self._events),
            "saved": False,
            "assessed": False,
            "submitted": False,
        }
        self._emit("episode_summary", report)
        self._check()
        self.session.current()
        persist_json(self.output / "report.json", report)
        self.session.close()
        persist_json(
            self.output / "runtime-manifest.json",
            {
                "events_sha256": _sha((self.output / "events.jsonl").read_bytes()),
                "report_sha256": _sha((self.output / "report.json").read_bytes()),
                "finished": True,
                "outcome": "raw_measurement_transport_verified",
                "task_completed": False,
            },
        )
        self._stream.close()
        self.report, self.finished = report, True
        self._outcome = "raw_measurement_transport_verified"

    def _stop(self, reason):
        if self.finished:
            return self.report
        self.finished, self._finalizing = True, True
        self._outcome = reason
        if self.intent is not None:
            try:
                self._reserved = self._reserved or (
                    self.owner.json(_reservation(self.owner, self.star, "inputs")) == self.intent
                )
            except (OSError, BrowserSafetyStop, ValueError):
                # Once known to exist, a missing/changed claim remains an
                # uncertain reservation, never permission for another write.
                self._reservation_uncertain = True
        self.report = {
            **self._flags,
            "mode": self._flags["provenance"],
            "stage": "inputs",
            "outcome": reason,
            "raw_measurement_transport_verified": False,
            "steps": self.steps,
            "numeric_writes": len(self.session.verified) if self.session else 0,
            "write_attempts": sorted(self.session.attempted) if self.session else [],
            "verified_fields": deepcopy(self.session.verified) if self.session else {},
            "reservation_created": self._reserved,
            "reservation_uncertain": self._reservation_uncertain,
            "automatic_retry": False,
        }
        try:
            if self.session is not None:
                self.session.stopped = True
                self.session.close()
            _failure(
                self.output,
                BrowserSafetyStop(reason),
                attempted=bool(self.session and self.session.attempted),
                reserved=self._reserved,
                evidence=self.evidence,
            )
            self._emit("error", {"reason": reason})
            self._emit("episode_summary", self.report)
            persist_json(self.output / "cooperative-stopped.json", self.report)
        except Exception:  # noqa: BLE001 - disk/callback failures cannot reopen a native session
            self.report["artifact_finalization_failed"] = True
        finally:
            self._stream.close()
            self._finalizing = False
        return self.report

    def abort(self):
        return self._stop("operator_aborted")

    def close(self):
        return self.abort()
