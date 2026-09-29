"""Cooperative, explicitly selected shallow-tooltip sensor; never an answer.

Each advance executes one bounded generator call or one restoration action.
Probe calls can include the existing pan plus pointer-clear pair. Pause/abort
are honored between those calls, not a promise to interrupt an in-flight call.
The caller owns the page. No failed owner can resume or claim a new output.
"""

import hashlib
import json
import os
import re
import time
from copy import deepcopy

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_observation_progress import PLOT_EXPOSED, capture_observation_progress
from .browser_planet_chart import FluxChartSession
from .browser_raster_planet_evidence import (
    TOOLTIP_MODE,
    _overview_window,
    _Owned,
    _same_overview,
    _same_star,
)
from .browser_shallow_transit_probe import (
    EXACT_TWO_HINT_POLICY,
    FIRST_THREE_TWO_ROW_HINT_POLICY,
    FLAGS,
    RECIPE,
    candidate_columns,
    iterate_shallow_feature,
    matches_overview_hint_metadata,
    overview_hint_metadata,
)
from .contracts import RuntimeEvent
from .planet_tooltip_reference import DIAGNOSTIC_FILES, TWO_MODE, load_tooltip_reference
from .presentation_capture import evidence_screenshot
from .project_events import ProjectEventRelay

MODE = "cooperative_shallow_transit_measurements"
FIRST_THREE_HINT_POLICY = FIRST_THREE_TWO_ROW_HINT_POLICY
MAX_SECONDS, MAX_NATIVE_ACTIONS, MAX_ADVANCES = 900, 64, 80
RESTORE_RECIPE_VERSION = "post_probe_center_0_5_0_4_v2"
RESTORE_POINTER = {"x_fraction": 0.5, "y_fraction": 0.4}


def _require(value, reason):
    if not value:
        raise BrowserSafetyStop("shallow_steps_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _reason(exc):
    code = str(exc)
    return (
        code
        if isinstance(exc, BrowserSafetyStop) and re.fullmatch(r"[a-z][a-z0-9_]{0,150}", code)
        else "shallow_steps_operation_failed"
    )


class ShallowTransitSteps:
    """Offline selection; exact initial capture and one restoration per probe."""

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        star,
        source_report,
        source_report_sha256,
        allow_two_events=False,
        emit=lambda *_: None,
        cancelled=lambda: False,
        _clock=time.monotonic,
        _session_factory=None,
        _probe_factory=None,
        _capture_progress=None,
        _load_reference=None,
    ):
        _require(callable(emit) and callable(cancelled) and callable(_clock), "invalid_callback")
        self.owner = _Owned(run_history)
        self.history = self.owner.history
        self.source_path = self.owner.path(source_report)
        self.source = _overview_window(self.owner, self.source_path, source_report_sha256)
        _require(_same_star(star, self.source["star"]), "source_star_changed")
        _require(type(allow_two_events) is bool, "invalid_two_event_opt_in")
        self.hint_policy, self.probe_count, self.measurement_mode = FIRST_THREE_HINT_POLICY, 3, TOOLTIP_MODE
        if allow_two_events:
            png = self.owner.read(self.source_path.parent / "chart.png", self.source["chart_sha256"])
            try:
                candidate_columns(png, self.source["flux_axis_labels"], hint_policy=FIRST_THREE_HINT_POLICY)
            except BrowserSafetyStop as exc:
                # This decision is made from the original complete overview,
                # before reserving a sensor or making any native action. Never
                # reinterpret a failed third probe as a two-event success.
                if str(exc) != "shallow_probe_three_overview_hints_required":
                    raise
                candidate_columns(png, self.source["flux_axis_labels"], hint_policy=EXACT_TWO_HINT_POLICY)
                self.hint_policy, self.probe_count, self.measurement_mode = EXACT_TWO_HINT_POLICY, 2, TWO_MODE
        self._selection = (self.hint_policy, self.probe_count, self.measurement_mode)
        self.action_limit = 38 if self.probe_count == 2 else MAX_NATIVE_ACTIONS
        self.output = self.owner.path(output)
        _require(
            self.output != self.history
            and not self.output.exists()
            and not self.output.is_relative_to(self.source_path.parent)
            and not self.source_path.parent.is_relative_to(self.output),
            "invalid_output",
        )
        self.page, self.config, self.star = page, config.model_copy(deep=True), star
        self._config = self.config.model_dump(mode="json")
        self._clock, self._callback, self._cancelled = _clock, emit, cancelled
        self._session_factory = _session_factory or FluxChartSession
        self._probe_factory = _probe_factory or iterate_shallow_feature
        self._capture_progress = _capture_progress or capture_observation_progress
        self._load_reference = _load_reference or load_tooltip_reference
        self._started = _clock()
        self.scope = {
            "schema_version": 1,
            "mode": MODE,
            "measurement_mode": self.measurement_mode,
            "star": star,
            "max_seconds": MAX_SECONDS,
            "max_native_actions": self.action_limit,
            "max_advances": MAX_ADVANCES,
            "probe_max_seconds": 180,
            "probe_max_actions": 16,
            "probes": self.probe_count,
            "initial_overview": "fresh_capture_exact_original_png_and_axes_no_actions",
            "restorations": self.probe_count,
            "restore_recipe_version": RESTORE_RECIPE_VERSION,
            "restore_pointer": deepcopy(RESTORE_POINTER),
            "restore_recipe": [1500, 1500, "clear_pointer"],
            "sensor_recipe": RECIPE,
            **overview_hint_metadata(self.hint_policy),
            "candidate_policy": "exact_two_current_column_hints_no_skip_or_fallback"
            if self.probe_count == 2
            else "first_three_current_column_hints_no_skip_or_fallback",
            **({"two_event_reference_enabled": True} if allow_two_events else {}),
            "source": deepcopy(self.source),
            "source_owner_recovery": False,
            "pause_counts_toward_deadline": True,
            "cancellation": "between_bounded_generator_calls_and_before_restore_actions",
            "project_completed": False,
            "automatic_deadline_increase": False,
            "optimizer_updates": 0,
            **FLAGS,
        }
        self._scope = deepcopy(self.scope)
        self.status, self.phase, self.failure = "running", "initializing", None
        self.report = self.measurements = self.window_evidence = self.anchor = None
        self._session = self._generator = self._probe_forward = None
        self._signature = self._probe_dir = None
        self._probe_raw, self._probe_actions, self._probe_confirmed = b"", 0, 0
        self._restore_pending = None
        self._busy = self._emitting = self._closing = self._abort_requested = self._forward_failed = False
        self._cleanup_failed = False
        self.advances = self.native_action_attempts = self.native_actions_confirmed = self._sequence = 0
        self.completed_probes, self.diagnostics = [], []
        self.candidates = None
        self._trees, self._hashes = {}, {}
        self._relay = ProjectEventRelay(self._relayed)
        self.claim = self.owner.path(
            self.history / "shallow-transit-owner-claims" / (_sha(star.casefold().encode()) + ".json")
        )
        _require(not self.claim.exists(), "owner_already_claimed")
        _require(not cancelled(), "cancelled")
        self.claim.parent.mkdir(exist_ok=True)
        intent = {**self.scope, "output": self.owner.relative(self.output)}
        try:
            descriptor = os.open(self.claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise BrowserSafetyStop("shallow_steps_owner_already_claimed") from None
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(intent, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(self.claim.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self.output.mkdir(parents=True, exist_ok=False)
        persist_json(self.output / "reserved.json", intent)
        persist_json(self.output / "scope.json", self.scope)
        self._stream = (self.output / "events.jsonl").open("x")
        self._run_id = "shallow-steps-" + _sha(self.owner.relative(self.output).encode())[:16]
        for path in (
            self.claim,
            self.output / "reserved.json",
            self.output / "scope.json",
            self.source_path,
            self.source_path.parent / "chart.png",
        ):
            self._pin(path)

    @property
    def finished(self):
        return self.status in {"completed", "stopped", "aborted"}

    def _pin(self, path):
        name, checksum = self.owner.relative(path), _sha(self.owner.read(path))
        _require(name not in self._hashes or self._hashes[name] == checksum, "pinned_source_changed")
        self._hashes[name] = checksum

    def _tree(self, directory):
        return sorted(self.owner.relative(path) for path in directory.rglob("*"))

    def _pin_tree(self, directory):
        self.owner.clean(directory)
        paths = self._tree(directory)
        _require(paths and len(paths) <= 100, "invalid_completed_tree")
        for name in paths:
            path = self.history / name
            _require(path.name not in {"stopped.json", "invalidated.json"}, "failed_completed_source")
            if path.is_file():
                self._pin(path)
        self._trees[directory] = paths

    def _check(self, *, native=True):
        _require(not self.finished and not self._abort_requested and not self._cancelled(), "cancelled")
        _require(not self._forward_failed, "event_forwarding_failed")
        _require(
            self.scope == self._scope
            and matches_overview_hint_metadata(self.scope, self.hint_policy)
            and (self.hint_policy, self.probe_count, self.measurement_mode) == self._selection
            and self.action_limit == (38 if self.probe_count == 2 else MAX_NATIVE_ACTIONS)
            and self.config.model_dump(mode="json") == self._config,
            "scope_changed",
        )
        _require(self._clock() - self._started < MAX_SECONDS, "time_limit")
        _require(self.native_action_attempts <= self.action_limit, "native_action_limit")
        _require(
            _overview_window(self.owner, self.source_path, self.source["report_sha256"]) == self.source,
            "original_source_changed",
        )
        for name, checksum in self._hashes.items():
            self.owner.read(self.history / name, checksum)
        for directory, paths in self._trees.items():
            self.owner.clean(directory)
            _require(self._tree(directory) == paths, "completed_tree_changed")
        if native and self._session is not None:
            self._session._guard()
            _require(
                _same_star(self._session.star, self.star) and self._session.signature == self._signature,
                "native_context_changed",
            )

    def _emit(self, kind, payload):
        event = RuntimeEvent(event=kind, payload=payload, run_id=self._run_id, sequence=self._sequence)
        self._stream.write(event.model_dump_json() + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._sequence += 1
        if not self._forward_failed:
            self._emitting = True
            try:
                self._callback(kind, event.model_dump(mode="json")["payload"])
            except Exception:  # noqa: BLE001 - redact callback details; never authorize another action
                self._forward_failed = True
                raise BrowserSafetyStop("shallow_steps_event_forwarding_failed") from None
            finally:
                self._emitting = False

    def _native_event(self, kind, payload):
        self._check()
        if kind == "action_proposed":
            _require(
                payload.get("kind") in {"SCROLL", "HOVER"} and self._restore_pending is None,
                "unexpected_restore_action",
            )
            _require(self.native_action_attempts < self.action_limit, "native_action_limit")
            self.native_action_attempts += 1
            self._restore_pending = payload["sequence"]
        elif kind == "action_result":
            _require(payload.get("sequence") == self._restore_pending, "restore_result_mismatch")
            self.native_actions_confirmed += 1
            self._restore_pending = None
        self._emit(kind, payload)
        self._check()

    def _relayed(self, kind, payload):
        self._check()
        self._emit(kind, {**payload, **self.state()} if kind == "state" else payload)
        self._check()

    def _drain_probe(self):
        path = self._probe_dir / "events.jsonl"
        if not path.exists():
            return
        # A failing generator writes stopped.json before it unwinds. Read its
        # bounded event stream even then so attempted native actions are not
        # erased from accounting. Failed trees are never accepted as sources.
        path = self.owner.path(path)
        _require(path.stat().st_size <= 2_000_000, "oversized_probe_events")
        raw = path.read_bytes()
        _require(
            raw.startswith(self._probe_raw) and (not raw or raw.endswith(b"\n")), "probe_event_stream_changed"
        )
        previous = len(self._probe_raw.splitlines())
        pending = []
        for index, line in enumerate(raw[len(self._probe_raw) :].splitlines(), previous):
            event = RuntimeEvent.model_validate_json(line)
            _require(
                event.sequence == index and event.run_id == self._probe_dir.name,
                "probe_event_identity_changed",
            )
            if event.event == "action_proposed":
                _require(event.payload.get("kind") in {"SCROLL", "DRAG", "HOVER"}, "unsupported_probe_action")
                self._probe_actions += 1
                self.native_action_attempts += 1
                _require(
                    event.payload.get("sequence") == self._probe_actions and self._probe_actions <= 16,
                    "probe_action_limit",
                )
            elif event.event == "action_result":
                self._probe_confirmed += 1
                self.native_actions_confirmed += 1
                _require(
                    event.payload.get("sequence") == self._probe_confirmed <= self._probe_actions,
                    "probe_result_mismatch",
                )
            elif event.event == "episode_summary":
                _require(
                    matches_overview_hint_metadata(event.payload, self.hint_policy),
                    "probe_policy_changed",
                )
            pending.append(event.model_dump(mode="json"))
        self._probe_raw = raw
        # Account for the whole completed bounded call before any user callback
        # can abort during replay. A pan+clear call can contain two proposals.
        self._check()
        for event in pending:
            self._probe_forward(event)
        _require(self._probe_raw == raw, "probe_event_stream_changed")

    def _initial(self):
        self._session = self._session_factory(
            self.page,
            self.config,
            self._native_event,
            max_actions=self.action_limit,
            max_seconds=MAX_SECONDS,
            verify_day_axis=True,
        )
        self._signature = deepcopy(self._session.signature)
        self._check()
        _require(self._session.handle.evaluate(PLOT_EXPOSED), "plot_occluded")
        times, fluxes = self._session.time_axis_labels(), self._session.flux_axis_labels()
        image = evidence_screenshot(self._session.chart)
        self._check()
        _require(
            self._session.handle.evaluate(PLOT_EXPOSED)
            and self._session.time_axis_labels() == times == self.source["time_axis_labels"]
            and self._session.flux_axis_labels() == fluxes == self.source["flux_axis_labels"]
            and _sha(image) == self.source["chart_sha256"],
            "initial_current_source_changed",
        )
        persist_json(
            self.output / "native-context.json",
            {
                "star": self._session.star,
                "student_native_values_sha256": _sha(
                    json.dumps(self._signature, sort_keys=True, allow_nan=False).encode()
                ),
                "hidden_application_state_read": False,
            },
        )
        self._pin(self.output / "native-context.json")
        self.phase = "initial_capture"

    def _capture_restore(self, *, initial=False):
        name = "initial" if initial else f"restore-{len(self.completed_probes):02d}"
        directory = self.output / name / "progress"
        remaining = MAX_SECONDS - (self._clock() - self._started)
        _require(remaining >= 1, "time_limit")
        result = self._capture_progress(
            self.page, self.config, directory, requested_days=5000, max_seconds=min(30, remaining)
        )
        path = directory / "report.json"
        _require(self.owner.json(path) == result, "restored_capture_mismatch")
        restored = _overview_window(self.owner, path, _sha(self.owner.read(path)))
        self._check()
        _require(_same_star(restored["star"], self.star), "restored_star_changed")
        if initial:
            # No restoration action may occur before this independent fresh
            # capture. Never adopt a different, merely plausible first plot.
            _require(
                self.anchor is None
                and not self.completed_probes
                and self.native_action_attempts == 0
                and all(
                    restored[key] == self.source[key]
                    for key in ("chart_sha256", "time_axis_labels", "flux_axis_labels")
                ),
                "initial_current_source_changed",
            )
            _same_overview(self.source, restored)
            self.anchor = restored
            groups = candidate_columns(
                self.owner.read(directory / "chart.png"),
                restored["flux_axis_labels"],
                hint_policy=self.hint_policy,
            )
            _require(
                len(groups) == self.probe_count,
                "two_current_hints_required" if self.probe_count == 2 else "three_current_hints_required",
            )
            self.candidates = deepcopy(groups)
        else:
            _require(self.anchor is not None and self.completed_probes, "missing_initial_anchor")
            _same_overview(self.anchor, restored)
        self.window_evidence = {
            "progress_path": self.owner.relative(path),
            "progress_sha256": restored["report_sha256"],
            "chart_path": self.owner.relative(directory / "chart.png"),
            "chart_sha256": restored["chart_sha256"],
        }
        self._pin_tree(directory.parent)
        self.phase = "validating" if len(self.completed_probes) == self.probe_count else "probe_initializing"

    def _initialize_probe(self):
        _require(self.native_action_attempts + 16 <= self.action_limit, "insufficient_probe_action_budget")
        self._probe_dir = self.output / f"probe-{len(self.completed_probes) + 1:02d}"
        self._probe_raw, self._probe_actions, self._probe_confirmed = b"", 0, 0
        self._probe_forward = self._relay.bind("shallow.probe", star=self.star)
        path = self.history / self.window_evidence["progress_path"]
        self._generator = self._probe_factory(
            self.page,
            self.config,
            self._probe_dir,
            run_history=self.history,
            source_dir=path.parent,
            source_report_sha256=self.window_evidence["progress_sha256"],
            candidate_index=len(self.completed_probes),
            overview_hint_policy=self.hint_policy,
        )
        _require(
            hasattr(self._generator, "__next__") and hasattr(self._generator, "close"),
            "invalid_probe_generator",
        )
        self.phase = "probe_active"

    def _verify_probe_scope(self):
        scope = self.owner.json(self._probe_dir / "scope.json")
        _require(
            matches_overview_hint_metadata(scope, self.hint_policy)
            and type(scope.get("candidate_index")) is int
            and scope["candidate_index"] == len(self.completed_probes)
            and type(scope.get("candidate_columns")) is list
            and all(type(column) is int for column in scope["candidate_columns"])
            and scope.get("candidate_columns") == self.candidates[len(self.completed_probes)]
            and scope.get("source_dir")
            == self.owner.relative((self.history / self.window_evidence["progress_path"]).parent)
            and scope.get("source_hashes")
            == {
                "report.json": self.window_evidence["progress_sha256"],
                "chart.png": self.window_evidence["chart_sha256"],
            },
            "candidate_selection_changed",
        )
        self._pin(self._probe_dir / "scope.json")

    def _advance_probe(self):
        result, finished = None, False
        try:
            yielded = next(self._generator)
        except StopIteration as stopped:
            result, finished = stopped.value, True
        finally:
            self._drain_probe()
        self._verify_probe_scope()
        if not finished:
            _require(
                isinstance(yielded, dict) and yielded.get("browser_actions") == self._probe_actions,
                "probe_action_accounting_changed",
            )
            return
        self._generator = None
        _require(
            isinstance(result, dict)
            and result == self.owner.json(self._probe_dir / "report.json")
            and result.get("sensor_recipe") == RECIPE
            and matches_overview_hint_metadata(result, self.hint_policy)
            and result.get("sampled_decline_verified") is True
            and result.get("source_hint_linked") is True
            and result.get("status") in {"visible_bracketed_dip", "visible_decline_not_source_linked"}
            and result.get("browser_actions") == self._probe_actions == self._probe_confirmed
            and _same_star(result.get("star"), self.star)
            and all(type(result.get(k)) is type(v) and result[k] == v for k, v in FLAGS.items()),
            "probe_not_verified",
        )
        self._pin_tree(self._probe_dir)
        spec = {
            "directory": self.owner.relative(self._probe_dir),
            "files": {
                name: _sha(self.owner.read(self._probe_dir / name)) for name in sorted(DIAGNOSTIC_FILES)
            },
        }
        self.diagnostics.append(spec)
        self.completed_probes.append(
            {
                "directory": spec["directory"],
                "report_sha256": _sha(self.owner.read(self._probe_dir / "report.json")),
                "browser_actions": self._probe_actions,
            }
        )
        self._relay.retire()
        self.phase = "restore_zoom_1"

    def _finish(self):
        _require(
            len(self.diagnostics) == self.probe_count,
            "two_verified_probes_required" if self.probe_count == 2 else "three_verified_probes_required",
        )
        self.measurements = self._load_reference(
            self.history,
            self.diagnostics,
            expected_star=self.star,
            **({"mode": TWO_MODE} if self.probe_count == 2 else {}),
        )
        _require(
            self.measurements.get("mode") == self.measurement_mode
            and self.measurements.get("answer_authorized") is False,
            "invalid_measurement_result",
        )
        for directory in self.measurements["validated_directories"]:
            self.owner.clean(self.history / directory)
        for name, checksum in self.measurements["source_sha256"].items():
            _require(_sha(self.owner.read(self.history / name)) == checksum, "measurement_source_changed")
            self._pin(self.history / name)
        self._check()
        self.phase = "measurements_ready"
        self._emit("episode_summary", {**self.state(), "finished": True, "status": "completed"})
        self._check()
        self._session.close()
        self._session = None
        self.report = {
            **self.state(),
            "status": "completed",
            "finished": True,
            "source_sha256": deepcopy(self._hashes),
            "events_sha256": _sha((self.output / "events.jsonl").read_bytes()),
        }
        persist_json(self.output / "report.json", self.report)
        self.status = "completed"
        self._stream.close()

    def state(self):
        return {
            **deepcopy(self.scope),
            "status": self.status,
            "phase": self.phase,
            "finished": self.finished,
            "failure_reason": self.failure,
            "advances": self.advances,
            "native_action_attempts": self.native_action_attempts,
            "native_actions_confirmed": self.native_actions_confirmed,
            "native_action_outcome_uncertain": self.native_action_attempts != self.native_actions_confirmed,
            "probe_index": len(self.completed_probes),
            "completed_probes": deepcopy(self.completed_probes),
            "window_evidence": deepcopy(self.window_evidence),
            "settled_anchor": deepcopy(self.anchor),
            "tooltip_reference": {
                "expected_star": self.star,
                "diagnostics": deepcopy(self.diagnostics),
                **({"mode": TWO_MODE} if self.probe_count == 2 else {}),
            }
            if len(self.diagnostics) == self.probe_count
            else None,
            "measurements": deepcopy(self.measurements),
            "event_forwarding_failed": self._forward_failed,
            "cleanup_failed": self._cleanup_failed,
        }

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy or self._emitting:
            self._abort_requested = True
            raise BrowserSafetyStop("shallow_steps_reentrant_call")
        self._busy = True
        try:
            self._check()
            _require(self.advances < MAX_ADVANCES, "advance_limit")
            self.advances += 1
            if self.phase == "initializing":
                self._emit("hello", {"protocol_version": 1, **self.state()})
                self._check()
                self._initial()
            elif self.phase == "initial_capture":
                self._capture_restore(initial=True)
            elif self.phase in {"restore_zoom_1", "restore_zoom_2"}:
                self._session.zoom(
                    self._scope["restore_pointer"]["x_fraction"],
                    self._scope["restore_pointer"]["y_fraction"],
                    1500,
                )
                self.phase = "restore_zoom_2" if self.phase == "restore_zoom_1" else "restore_clear"
            elif self.phase == "restore_clear":
                self._session.clear_pointer()
                self.phase = "restore_capture"
            elif self.phase == "restore_capture":
                self._capture_restore()
            elif self.phase == "probe_initializing":
                self._initialize_probe()
            elif self.phase == "probe_active":
                self._advance_probe()
            elif self.phase == "validating":
                self._finish()
            else:
                raise BrowserSafetyStop("shallow_steps_unknown_phase")
            if not self.finished:
                self._check()
                self._emit("state", self.state())
                self._check()
        except (KeyboardInterrupt, SystemExit):
            self._abort_requested = True
            self._stop("operator_aborted")
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize native/source failure; preserve claim
            self._stop("operator_aborted" if self._abort_requested else _reason(exc))
        finally:
            self._busy = False
        return self.state()

    def _stop(self, reason):
        if self.finished or self._closing:
            return
        self._closing = True
        self.failure = reason
        self.status = self.phase = "aborted" if self._abort_requested else "stopped"
        try:
            for child in (self._generator, self._session):
                if child is not None:
                    try:
                        child.close()
                    except Exception:  # noqa: BLE001 - cleanup failure never reopens work
                        self._cleanup_failed = True
            self._generator = self._session = None
            self.measurements = None
            self.report = {**self.state(), "source_sha256": deepcopy(self._hashes)}
            persist_json(self.output / "stopped.json", self.report)
            if not self._forward_failed:
                try:
                    self._emit("error", {"type": "ShallowTransitStop", "message": reason})
                    self._emit("episode_summary", self.report)
                except Exception:  # noqa: BLE001 - no retry after diagnostic callback failure
                    self._forward_failed = True
        finally:
            self._relay.retire()
            self._stream.close()
            self._closing = False

    def pause(self):
        if not self.finished:
            self.status = "paused"
        return self.state()

    def resume(self):
        if not self.finished:
            _require(not self._busy and not self._emitting, "reentrant_resume")
            self._check()
            self.status = "running"
        return self.state()

    def tick(self):
        return self.advance() if self.status == "running" else self.state()

    advance_if_due = tick

    def step(self):
        if not self.finished:
            _require(self.status == "paused", "pause_before_step")
        return self.advance()

    def abort(self):
        if not self.finished:
            self._abort_requested = True
            if not self._busy and not self._emitting:
                self._stop("operator_aborted")
        return self.state()

    def close(self):
        return self.abort()
