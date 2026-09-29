"""Cooperative transport of an explicitly supplied fresh-star class decision.

Each advance selects at most one native class or lifetime prefix. Reference
rationale is pinned, not scientifically evaluated; no classifier, numeric answer
entry, save, grading or retry is enabled. The class subtree retains the legacy
fresh-class evidence format and remains immutable before any prefix operation.
"""

import hashlib
import json
import math
import re
import time
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import (
    ClassCircleRenderingStop,
    ClassScreenMismatch,
    StellarSelectionSession,
    read_class_choices,
)
from .browser_full_stellar import full_stellar_status_projection
from .browser_numeric import digest, screen_identity
from .browser_probe import _visible_frame, inspect_page, save_probe
from .browser_star_preflight import _blank_mapping, _Evidence, validate_star_class_source
from .browser_stellar import CLASSES, LIFETIME_PREFIXES
from .contracts import RuntimeEvent

_OPERATION_TYPES = {
    "BrowserSafetyStop",
    "StellarMappingError",
    "ClassCircleRenderingStop",
    "ClassScreenMismatch",
    "TimeoutError",
    "Error",
    "TargetClosedError",
    "ValueError",
    "TypeError",
    "KeyError",
    "AttributeError",
    "RuntimeError",
    "OSError",
    "FileNotFoundError",
    "PermissionError",
}
_MAPPING_CODES = {
    "invalid_accessibility_snapshot",
    "unsupported_accessibility_yaml",
    "unsupported_accessibility_layout",
    "unsupported_accessibility_text",
    "ambiguous_control_snapshot",
    "invalid_capture_contract",
    "ambiguous_simulation_frame",
    "missing_star_identity",
    "conflicting_star_identity",
    "not_stellar_detail_screen",
    "unsupported_conditional_or_ambiguous_fields",
    "ambiguous_color_control",
    "duplicate_control_id",
    "control_frame_mismatch",
    "ambiguous_lifetime_prefix",
    "unsupported_lifetime_prefix",
    "control_snapshot_mismatch",
    "unmapped_color_label",
    "unsupported_color_options_or_selection",
    "unmapped_numeric_label",
    "duplicate_numeric_field",
    "incomplete_field_map",
    "missing_classification_choices",
}
_READ_SAFETY_CODES = {
    "unexpected_popup",
    "capture_attempt_limit",
    "authentication_required",
    "unexpected_modal",
    "control_budget_exceeded",
    "observation_budget_exceeded",
    "navigation_outside_activity",
    "frame_changed_during_observation",
    "unknown_visible_frame",
    "simulation_frame_changed",
}
_DIAGNOSTIC_FUNCTIONS = {
    "__init__",
    "advance",
    "_check",
    "_class_step",
    "_prefix_step",
    "read",
    "_read_once",
    "current",
    "select_class",
    "select_prefix",
    "_fresh_initial_paint",
    "_read_choice_circles",
    "read_class_choices",
    "map_stellar_capture",
    "_atoms",
    "_control_atom",
    "_visible_measurement",
    "inspect_page",
    "_read_frame",
    "_read_controls",
    "_check_auth_and_modals",
    "_visible_frame",
    "save_probe",
}
_DIAGNOSTIC_MODULES = {
    "browser_class_setup_steps",
    "browser_classification",
    "browser_probe",
    "browser_stellar",
}


def _operation_diagnostic(exc, phase):
    """Bounded code identity only; never exception messages, arguments or locals."""
    kind = type(exc).__name__
    code, detail = None, None
    if kind == "StellarMappingError":
        value = str(exc)
        if value in _MAPPING_CODES:
            code = value
        elif value in {
            f"{name}:{field}"
            for name in ("missing_or_ambiguous_measurement", "invalid_measurement_domain")
            for field in ("parallax", "wavelength", "flux")
        }:
            code, detail = value.split(":")
    elif isinstance(exc, BrowserSafetyStop):
        value = str(exc)
        if value in _READ_SAFETY_CODES:
            code = value
        elif value in {
            f"{name}:{frame}"
            for name in ("frame_count_mismatch", "frame_not_ready")
            for frame in ("simulation", "widgets", "score_widgets")
        }:
            code, detail = value.split(":")
    frames, tb, count = [], exc.__traceback__, 0
    base = Path(__file__).resolve().parent
    while tb is not None and count < 128:
        path, function = Path(tb.tb_frame.f_code.co_filename), tb.tb_frame.f_code.co_name
        if path.parent == base and path.stem in _DIAGNOSTIC_MODULES and function in _DIAGNOSTIC_FUNCTIONS:
            frames.append({"module": path.stem, "function": function, "line": tb.tb_lineno})
        tb, count = tb.tb_next, count + 1
    context = getattr(exc, "_habfly_class_read_context", {})
    context = context if isinstance(context, dict) else {}
    read_phase = context.get("read_phase")
    read_stage = context.get("read_stage")
    flags = (
        "native_class_click_invoked",
        "native_class_click_returned",
        "native_prefix_selection_invoked",
        "native_prefix_selection_returned",
    )
    return {
        "schema_version": 1,
        "exception_type": kind if kind in _OPERATION_TYPES else "OtherException",
        "operation_phase": phase if phase in {"class_pending", "prefix_pending", "verified"} else None,
        "error_code": code,
        "error_detail": detail,
        "repo_frames": frames[-4:],
        "read_phase": read_phase
        if read_phase
        in {
            "initial_read",
            "current_read",
            "pre_selection_read",
            "post_click_read",
            "pre_prefix_read",
            "post_prefix_read",
            "pre_dispatch_native_binding",
        }
        else None,
        "read_stage": read_stage
        if read_stage
        in {
            "inspect_page",
            "frame_validation",
            "map_stellar_capture",
            "read_class_choices",
            "fresh_source_validation",
            "confirming_public_capture",
        }
        else None,
        **{key: context[key] for key in flags if type(context.get(key)) is bool},
        "public_capture_available": context.get("report") is not None,
        "additional_dom_queries": 0,
        "task_completed": False,
        "automatic_retry": False,
    }, context.get("report")


class _Cancelled(Exception):
    pass


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("fresh_class_steps_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _reason(exc):
    message = str(exc)
    return (
        message
        if isinstance(exc, BrowserSafetyStop) and re.fullmatch(r"[a-z][a-z0-9_]{0,150}", message)
        else "fresh_class_steps_operation_failed"
    )


def _stable_screen(report):
    # Retain raw captures/hashes; comparison alone ignores the already observed
    # exact stellar Save footer/busy flag, never an answer or classification.
    return screen_identity(full_stellar_status_projection(report))


class FreshStarClassSteps:
    """Caller owns the page. Constructor does local validation/reservation only.

    emit receives the exact flushed v1 event object, not a reconstructed action.
    Cancellation is checked again after every callback and native pre-dispatch
    re-read. An uncertain operation is terminal, including across output paths.
    max_seconds includes pauses after construction; it is never extended.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        fresh_star,
        selected_class,
        reference_rationale,
        lifetime_prefix=None,
        max_seconds=180,
        max_advances=4,
        emit=lambda _: None,
        _clock=time.monotonic,
    ):
        _require(isinstance(selected_class, str) and selected_class in CLASSES, "invalid_explicit_class")
        _require(
            isinstance(reference_rationale, str)
            and 40 <= len(reference_rationale) <= 2000
            and len(reference_rationale.split()) >= 8
            and not any(ord(c) < 32 for c in reference_rationale),
            "substantive_reference_rationale_required",
        )
        _require(
            (selected_class == "main_sequence" and lifetime_prefix in LIFETIME_PREFIXES)
            or (selected_class != "main_sequence" and lifetime_prefix is None),
            "explicit_applicable_prefix_required",
        )
        _require(type(max_advances) is int and 1 <= max_advances <= 4, "invalid_advance_limit")
        _require(
            type(max_seconds) in {int, float} and math.isfinite(max_seconds) and 5 <= max_seconds <= 600,
            "invalid_time_limit",
        )
        _require(callable(emit) and callable(_clock), "invalid_callback")
        self.history = Path(run_history).absolute()
        _require(self.history.is_dir() and self.history == self.history.resolve(), "invalid_history")
        self.output, self.fresh = self._owned(output), self._owned(fresh_star)
        _require(
            self.output != self.history
            and not self.output.exists()
            and not self.output.is_relative_to(self.fresh)
            and not self.fresh.is_relative_to(self.output),
            "invalid_output",
        )
        self._evidence = _Evidence(self.history)
        self._evidence.clean(self.fresh)
        receipt = self._evidence.json(self.fresh / "confirmed.json")
        source = self._evidence.capture(self.fresh / "stellar")
        self.star = source["star_name"]
        _require(
            isinstance(self.star, str)
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{0,79}", self.star)
            and isinstance(receipt.get("star"), str)
            and receipt["star"].casefold() == self.star.casefold()
            and receipt.get("fresh_blank_numeric_answers_verified") is True
            and receipt.get("class_selection_verified") is False
            and type(receipt.get("answer_writes")) is int
            and receipt["answer_writes"] == 0
            and receipt.get("action_source") == "deterministic_navigation"
            and receipt.get("painted_stellar_class") in {None, *CLASSES},
            "invalid_fresh_receipt",
        )
        self.measurements = deepcopy(source["observation"]["values"]["measurements"])
        _blank_mapping(source, self.star, self.measurements, conditional=False)
        _require(not (self.fresh / "class-selection-reserved.json").exists(), "class_already_reserved")
        self.prefix_claim = self.fresh / "lifetime-prefix-reserved.json"
        _require(not self.prefix_claim.exists(), "prefix_already_reserved")
        self.selected_class, self.lifetime_prefix = selected_class, lifetime_prefix
        self.inherited = receipt["painted_stellar_class"] == selected_class
        self.intermediate = "red_giant" if selected_class == "white_dwarf" else "white_dwarf"
        self.plan = [self.intermediate, selected_class] if self.inherited else [selected_class]
        self.source_hash = self._evidence.hashes[
            str((self.fresh / "stellar/observation.json").relative_to(self.history))
        ]
        self.page, self.config, self._callback, self._clock = page, config.model_copy(deep=True), emit, _clock
        self.max_seconds, self.max_advances = max_seconds, max_advances
        self._started = _clock()
        self.phase, self.status, self.failure = "class_pending", "ready", None
        self.advances = self.class_attempts = self.prefix_attempts = self.class_index = 0
        self._sequence = 0
        self._class_events, self._prefix_events = [], []
        self.session = self.report = self.class_source = self.prefix_receipt = None
        self._busy = self._emitting = self._finalizing = self._forward_failed = False
        self._pins = {}
        self._rendering_failure = None
        self._screen_failure = None
        self._operation_failure = None
        self._class_dir, self._prefix_dir = self.output / "class", self.output / "prefix"
        self._run_id = "fresh-class-" + _sha(str(self.output.relative_to(self.history)).encode())[:16]
        claim_dir = self._owned(self.history / "class-setup-claims")
        claim_dir.mkdir(exist_ok=True)
        claim = claim_dir / (_sha(self.star.casefold().encode()) + ".json")
        _require(not claim.exists(), "star_already_reserved")
        self.decision = {
            "star": self.star,
            "selected_class": selected_class,
            "reference_rationale": reference_rationale,
            "lifetime_prefix": lifetime_prefix,
            "source_capture_sha256": self.source_hash,
            "classification_learned": False,
            "scientific_verified": False,
            "training_label": False,
        }
        persist_json(
            claim,
            {
                "star": self.star,
                "output": str(self.output.relative_to(self.history)),
                "fresh_star": str(self.fresh.relative_to(self.history)),
                "decision_sha256": digest(self.decision),
                "automatic_retry": False,
            },
        )
        self._pin(claim)
        self.output.mkdir(parents=True, exist_ok=False)
        self._class_dir.mkdir()
        persist_json(self.output / "decision.json", self.decision)
        self._pin(self.output / "decision.json")
        self._stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        self.scope = {
            "mode": "explicit_reference_class_setup",
            "star": self.star,
            "max_class_clicks": len(self.plan),
            "max_prefix_selections": int(lifetime_prefix is not None),
            "max_advances": max_advances,
            "max_seconds": max_seconds,
            "automatic_retry": False,
            "numeric_answer_writes": 0,
            "assessment_enabled": False,
            "submission_enabled": False,
            "decision_sha256": digest(self.decision),
            "scientific_verified": False,
            "training_label": False,
            "classification_learned": False,
            "task_completed": False,
        }
        persist_json(self.output / "scope.json", self.scope)
        self._pin(self.output / "scope.json")

    def _owned(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.history / path
        _require(
            path.resolve().is_relative_to(self.history)
            and not any(p.is_symlink() for p in (path, *path.parents)),
            "invalid_owned_path",
        )
        return path.resolve()

    @property
    def finished(self):
        return self.status in {"completed", "stopped", "aborted"}

    def _pin(self, path):
        self._pins[str(self._owned(path).relative_to(self.history))] = _sha(path.read_bytes())

    def _pin_capture(self, directory):
        for path in directory.iterdir():
            if path.is_file():
                self._pin(path)

    def _check(self, *, allow_finished=False):
        if self.finished and not allow_finished:
            raise _Cancelled
        _require(self._clock() - self._started < self.max_seconds, "time_limit")
        self._evidence.unchanged()
        _require(digest(self.decision) == self.scope["decision_sha256"], "decision_changed")
        for name, checksum in self._pins.items():
            _require(_sha(self._owned(name).read_bytes()) == checksum, "pinned_source_changed")
        if self.class_source:
            _require(
                validate_star_class_source(self.history, self._class_dir, self.star, self.selected_class)
                == self.class_source,
                "class_source_changed",
            )

    def _emit(self, kind, payload):
        item = RuntimeEvent(event=kind, payload=payload, sequence=self._sequence, run_id=self._run_id)
        raw = item.model_dump_json()
        self._stream.write(raw + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            self._emitting = True
            try:
                self._callback(json.loads(raw))
            except Exception:  # noqa: BLE001 - callback/driver text can contain private session data
                self._forward_failed = True
                raise BrowserSafetyStop("fresh_class_steps_event_forwarding_failed") from None
            finally:
                self._emitting = False

    def _native_event(self, kind, payload):
        self._check()
        prefix = self.phase == "prefix_pending"
        events, directory = (
            (self._prefix_events, self._prefix_dir) if prefix else (self._class_events, self._class_dir)
        )
        _require(kind in {"action_proposed", "action_result"}, "unexpected_native_event")
        _require(
            kind == ("action_proposed" if len(events) % 2 == 0 else "action_result"), "native_event_order"
        )
        if kind == "action_proposed":
            _require(
                payload.get("kind") == "SELECT"
                and payload.get("target") == ("lifetime_prefix" if prefix else "stellar_class")
                and payload.get("value") == (self.lifetime_prefix if prefix else self.plan[self.class_index])
                and payload.get("action_source")
                == ("explicit_unit_transport" if prefix else "reference_diagnostic"),
                "unexpected_native_action",
            )
        persist_json(directory / f"event-{len(events)}.json", {"event": kind, "payload": payload})
        self._pin(directory / f"event-{len(events)}.json")
        events.append({"event": kind, "payload": deepcopy(payload)})
        self._emit(kind, payload)
        self._check()
        # Legacy adapters bind before emitting. Re-read after callback, so abort,
        # replaced handles, a changed star or changed answers cannot sneak in.
        if kind == "action_proposed":
            try:
                self._native_binding()
            except ClassCircleRenderingStop as exc:
                raise exc.at_read("pre_dispatch_native_binding") from None
            self._check()
            if prefix:
                self.prefix_attempts += 1
            else:
                self.class_attempts += 1

    def _native_binding(self):
        session = self.session
        if not (
            session.fresh_star is not None
            and session.choices["selected"] == "main_sequence"
            and not session.mapping["observation"]["values"]["conditional_fields_visible"]
        ):
            session.current()
            return
        # select_class has already marked its pending write. Its read() rightly
        # refuses to interpret inherited Main paint after that mark. Revalidate
        # the *unchanged* previously proven fresh snapshot and exact bindings,
        # without treating this pre-click paint as a newly confirmed class.
        _require(
            len(self.page.context.pages) == 1
            and session.frame in self.page.frames
            and _visible_frame(session.frame, self.page.main_frame),
            "native_frame_changed",
        )
        report = inspect_page(self.page, self.config)
        _require(
            not report["ignored_frame_urls"] and _stable_screen(report) == _stable_screen(session.report),
            "native_state_changed",
        )
        choices, handles = read_class_choices(session.frame)
        _require(choices == session.choices, "native_paint_changed")
        for name, old in session.handles.items():
            _require(old.evaluate("(a,b) => a.isConnected && a===b", handles[name]), "native_handle_changed")
        _require(
            _stable_screen(report) == _stable_screen(inspect_page(self.page, self.config)),
            "native_state_changed",
        )

    def _class_step(self):
        first = self.class_index == 0
        self.session = StellarSelectionSession(
            self.page, self.config, self._native_event, **({"fresh_star": self.fresh} if first else {})
        )
        if first:
            _require(
                self.session.choices["selected"]
                == (
                    self.selected_class
                    if self.inherited
                    else self._evidence.json(self.fresh / "confirmed.json")["painted_stellar_class"]
                ),
                "initial_paint_changed",
            )
            _blank_mapping(self.session.mapping, self.star, self.measurements, conditional=False)
            save_probe(self.session.report, self._class_dir / "before")
            self._pin_capture(self._class_dir / "before")
            persist_json(
                self._class_dir / "scope.json",
                {
                    "star": self.star,
                    "intended_class": self.selected_class,
                    "fresh_star": str(self.fresh),
                    "source_capture_sha256": self.source_hash,
                    "max_class_clicks": len(self.plan),
                    "numeric_writes": 0,
                    "clears_inherited_paint": self.inherited,
                    "intermediate_is_training_label": False,
                    "learned_classification": False,
                    "scientific_verified": False,
                    "training_label": False,
                    "automatic_retry": False,
                },
            )
            self._pin(self._class_dir / "scope.json")
        else:
            before = json.loads((self._class_dir / "intermediate/observation.json").read_bytes())
            _require(
                _stable_screen(self.session.report) == _stable_screen(before), "intermediate_state_changed"
            )
            _require(
                digest(self.session.choices) == self._class_events[-1]["payload"]["rendering_sha256"],
                "intermediate_paint_changed",
            )
        self._check()
        self.session.select_class(
            self.plan[self.class_index],
            source="reference_diagnostic",
            **(
                {
                    "expected_previous": self.intermediate,
                    "revision_reason": "Set the explicitly supplied intended class after clearing inherited radio paint on verified fresh blank data. The intermediate choice is not a scientific inference or training label.",
                }
                if not first
                else {}
            ),
        )
        self._check()
        if first:
            self._pin(self.fresh / "class-selection-reserved.json")
        self.class_index += 1
        if self.class_index < len(self.plan):
            save_probe(self.session.report, self._class_dir / "intermediate")
            self._pin_capture(self._class_dir / "intermediate")
            return
        after = inspect_page(self.page, self.config)
        _require(_stable_screen(after) == _stable_screen(self.session.report), "class_changed_after_readback")
        _blank_mapping(
            self.session.mapping,
            self.star,
            self.measurements,
            conditional=self.selected_class == "main_sequence",
        )
        save_probe(after, self._class_dir / "after")
        persist_json(
            self._class_dir / "confirmed.json",
            {
                "star": self.star,
                "selected_class": self.selected_class,
                "class_clicks": len(self.plan),
                "numeric_writes": 0,
                "readback_verified": True,
                "correctness_verified": False,
                "learned_classification": False,
                "scientific_verified": False,
                "training_label": False,
                "intermediate_is_training_label": False,
                "action_source": "explicit_fresh_star_class_setup",
                "task_completed": False,
            },
        )
        self.class_source = validate_star_class_source(
            self.history, self._class_dir, self.star, self.selected_class
        )
        self.phase = "prefix_pending" if self.lifetime_prefix is not None else "verified"

    def _prefix_step(self):
        self.session = StellarSelectionSession(self.page, self.config, self._native_event)
        after = json.loads((self._class_dir / "after/observation.json").read_bytes())
        _require(_stable_screen(self.session.report) == _stable_screen(after), "prefix_source_changed")
        _require(
            self.session.choices["selected"] == self.selected_class
            and digest(self.session.choices) == self.class_source["class_rendering_sha256"],
            "prefix_class_paint_changed",
        )
        _blank_mapping(self.session.mapping, self.star, self.measurements, conditional=True)
        self._check()
        _require(not self.prefix_claim.exists(), "prefix_already_reserved")
        self._prefix_dir.mkdir()
        save_probe(self.session.report, self._prefix_dir / "before")
        self._pin_capture(self._prefix_dir / "before")
        reservation = {
            "star": self.star,
            "selected_class": self.selected_class,
            "lifetime_prefix": self.lifetime_prefix,
            "output": str(self._prefix_dir.relative_to(self.history)),
            "source_capture_sha256": self.source_hash,
            "class_source_hashes": self.class_source["source_hashes"],
            "decision_sha256": digest(self.decision),
            "max_selections": 1,
            "automatic_retry": False,
        }
        persist_json(self.prefix_claim, reservation)
        self._pin(self.prefix_claim)
        persist_json(self._prefix_dir / "reserved.json", reservation)
        self._pin(self._prefix_dir / "reserved.json")
        self.prefix_receipt = self.session.select_prefix(self.lifetime_prefix)
        self._check()
        values = self.session.mapping["observation"]["values"]
        lifetime = values["browser_field_map"]["lifetime"]["current_value"]
        _require(
            self.prefix_receipt.get("lifetime_prefix") == self.lifetime_prefix
            and self.prefix_receipt.get("readback_verified") is True
            and self.prefix_receipt.get("task_completed") is False
            and type(self.prefix_receipt.get("blank_lifetime_initialized_to_zero")) is bool
            and self.prefix_receipt["blank_lifetime_initialized_to_zero"] is (lifetime == "0.000")
            and lifetime in {"", "0.000"}
            and values["lifetime_prefix"]["selected"] == self.lifetime_prefix
            and values["measurements"] == self.measurements
            and self.session.mapping["star_name"].casefold() == self.star.casefold()
            and digest(self.session.choices) == self.class_source["class_rendering_sha256"]
            and all(
                field["value_known"] is True and (name == "lifetime" or field["current_value"] == "")
                for name, field in values["browser_field_map"].items()
            ),
            "prefix_readback_not_verified",
        )
        after = inspect_page(self.page, self.config)
        _require(
            _stable_screen(after) == _stable_screen(self.session.report), "prefix_changed_after_readback"
        )
        self._check()
        save_probe(after, self._prefix_dir / "after")
        self._pin_capture(self._prefix_dir / "after")
        persist_json(
            self._prefix_dir / "confirmed.json",
            {
                **self.prefix_receipt,
                "star": self.star,
                "numeric_answer_writes": 0,
                "selection_source": "explicit_unit_transport",
                "scientific_verified": False,
                "training_label": False,
                "automatic_retry": False,
                "class_source_hashes": self.class_source["source_hashes"],
            },
        )
        self._pin(self._prefix_dir / "confirmed.json")
        self.phase = "verified"

    def state(self):
        return {
            **self.scope,
            "component": "fresh_star_class_setup",
            "status": self.status,
            "phase": self.phase,
            "finished": self.finished,
            "failure_reason": self.failure,
            "advances": self.advances,
            "class_clicks_may_have_occurred": self.class_attempts,
            "prefix_selections_may_have_occurred": self.prefix_attempts,
            "setup_verified": self.status == "completed",
            "selected_class": self.selected_class,
            "lifetime_prefix": self.lifetime_prefix,
            "event_forwarding_failed": self._forward_failed,
            "class_source": deepcopy(self.class_source),
            **(
                {"class_operation_failure": deepcopy(self._operation_failure)}
                if self._operation_failure
                else {}
            ),
            **({"class_screen_failure": deepcopy(self._screen_failure)} if self._screen_failure else {}),
            **(
                {"class_rendering_failure": deepcopy(self._rendering_failure)}
                if self._rendering_failure
                else {}
            ),
            "source_hashes": {**self._evidence.hashes, **self._pins},
            "artifact_paths": {
                "class": str(self._class_dir.relative_to(self.history)),
                "prefix": str(self._prefix_dir.relative_to(self.history)) if self.prefix_receipt else None,
            },
        }

    def _finish(self, reason=None, *, aborted=False):
        if self.finished or self._finalizing:
            return
        self._finalizing = True
        self.failure = reason
        self.status = "aborted" if aborted else "stopped" if reason else "completed"
        if reason:
            self.phase = self.status
        if self.session:
            self.session.stopped = True
        try:
            before = self.state()
            try:
                self._emit("episode_summary", {**before, "completed": False})
                if reason is None:
                    self._check(allow_finished=True)
            except Exception as exc:  # noqa: BLE001 - terminal callbacks cannot alter pinned evidence
                self.status, self.phase, self.failure = (
                    "stopped",
                    "stopped",
                    _reason(exc),
                )
            self.report = self.state()
            if self.report != before:
                try:
                    self._emit("state", self.report)
                except Exception:  # noqa: BLE001 - final failure is already durable and cannot enable actions
                    self._forward_failed = True
                    self.report = self.state()
            if self.failure:
                persist_json(self.output / "stopped.json", self.report)
            persist_json(self.output / "report.json", self.report)
        finally:
            self._stream.close()
            self._finalizing = False

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy or self._emitting:
            self._finish("fresh_class_steps_reentrant_call")
            raise BrowserSafetyStop("fresh_class_steps_reentrant_call")
        self._busy = True
        try:
            self._check()
            _require(self.advances < self.max_advances, "advance_limit")
            self.advances += 1
            self._class_step() if self.phase == "class_pending" else self._prefix_step()
            self._check()
            if self.phase == "verified":
                self._finish()
            else:
                self._emit("state", self.state())
                self._check()
        except _Cancelled:
            pass
        except (KeyboardInterrupt, SystemExit):
            self.abort()
            raise
        except Exception as exc:  # noqa: BLE001 - all uncertain actions stay terminal without raw driver text
            self._operation_failure = {"artifact_recording_failed": True}
            try:
                diagnostic, captured = _operation_diagnostic(exc, self.phase)
                self._operation_failure = deepcopy(diagnostic)
                directory = self.output / "class-operation-failure"
                directory.mkdir()
                if captured is not None:
                    save_probe(captured, directory / "capture")
                    self._pin_capture(directory / "capture")
                    diagnostic["capture"] = str((directory / "capture").relative_to(self.history))
                path = directory / "diagnostic.json"
                persist_json(path, diagnostic)
                self._pin(path)
                self._operation_failure.update(
                    path=str(path.relative_to(self.history)), sha256=_sha(path.read_bytes())
                )
            except Exception:  # noqa: BLE001 - diagnostic failure cannot prevent the original stop
                self._operation_failure["artifact_recording_failed"] = True
            if isinstance(exc, ClassScreenMismatch):
                self._screen_failure = deepcopy(exc.diagnostic)
                directory = self.output / "class-screen-failure"
                try:
                    save_probe(exc.before, directory / "before")
                    save_probe(exc.after, directory / "after")
                    self._pin_capture(directory / "before")
                    self._pin_capture(directory / "after")
                    persist_json(
                        directory / "difference.json",
                        {
                            **deepcopy(exc.diagnostic),
                            "star": self.star,
                            "requested_class": self.selected_class,
                            "requested_prefix": self.lifetime_prefix,
                            "before": str((directory / "before").relative_to(self.history)),
                            "after": str((directory / "after").relative_to(self.history)),
                        },
                    )
                    self._pin(directory / "difference.json")
                    self._screen_failure.update(
                        path=str((directory / "difference.json").relative_to(self.history)),
                        sha256=_sha((directory / "difference.json").read_bytes()),
                    )
                except Exception:  # noqa: BLE001 - diagnostic I/O never skips the original terminal cause
                    self._screen_failure["artifact_recording_failed"] = True
            if isinstance(exc, ClassCircleRenderingStop):
                diagnostic = {
                    **deepcopy(exc.diagnostic),
                    "star": self.star,
                    "requested_choice": self.plan[self.class_index]
                    if self.class_index < len(self.plan)
                    else None,
                    "class_step_index": self.class_index,
                    "task_completed": False,
                    "automatic_retry": False,
                    "classification_learned": False,
                }
                path = self.output / "class-rendering-failure.json"
                self._rendering_failure = {
                    "failure_phase": diagnostic.get("failure_phase"),
                    "native_class_click_invoked": diagnostic.get("native_class_click_invoked"),
                    "native_class_click_returned": diagnostic.get("native_class_click_returned"),
                }
                try:
                    persist_json(path, diagnostic)
                    self._pin(path)
                    self._rendering_failure.update(
                        path=str(path.relative_to(self.history)),
                        sha256=_sha(path.read_bytes()),
                    )
                except Exception:  # noqa: BLE001 - diagnostic I/O must never skip the terminal stop
                    self._rendering_failure["artifact_recording_failed"] = True
            self._finish(_reason(exc))
        finally:
            self._busy = False
        return self.state()

    def abort(self):
        self._finish("operator_aborted", aborted=True)
        return self.state()

    def close(self):
        return self.abort()
