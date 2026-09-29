"""Cooperative, read-only collection evidence after a completed detail workflow.

Each advance calls at most one existing guarded navigator or inventory verifier.
The caller owns scheduling and the browser; cancellation/deadlines apply between
bounded calls, not inside synchronous Playwright. Pagination is a separate,
explicit mode; neither mode retries or normalizes an unexpected current page.
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
from .browser_no_planet_workflow import _Evidence
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_project_inventory import NAME, verify_project_inventory
from .browser_project_navigation import navigate_project, navigation_status_projection, project_view
from .browser_project_paginated_inventory_steps import PaginatedInventorySteps
from .browser_stellar import SIMULATION_URL
from .contracts import RuntimeEvent
from .project_events import ProjectEventRelay
from .project_inventory_source import load_inventory_source
from .supplied_browser_modes import POSITIVE_SUPPLIED_MODE, TERRESTRIAL_SUPPLIED_MODE

MODES = {
    "no_planet_visible_workflow_readback": ("stellar", "planet"),
    "positive_planet_visible_workflow_readback": ("stellar", "planet"),
    "terrestrial_visible_workflow_readback": ("stellar", "planet", "habitability"),
    POSITIVE_SUPPLIED_MODE: ("stellar", "planet"),
    TERRESTRIAL_SUPPLIED_MODE: ("stellar", "planet", "habitability"),
}
ZERO = ("answer_writes", "save_clicks", "assessment_clicks", "score_transfer_clicks", "submission_clicks")
FALSE = (
    "hidden_values_inspected",
    "scientific_verified",
    "correctness_verified",
    "browser_acceptance_passed",
    "course_completion_verified",
    "project_completed",
    "submitted",
    "training_label",
    "cross_session_persistence_verified",
)


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("project_inventory_steps_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _names(values):
    _require(
        isinstance(values, (list, tuple))
        and 1 <= len(values) <= 30
        and all(isinstance(v, str) and re.fullmatch(NAME, v) and v == " ".join(v.split()) for v in values),
        "invalid_expected_stars",
    )
    _require(len({v.casefold() for v in values}) == len(values), "duplicate_expected_stars")
    return tuple(values)


def _autosave_admission(book, receipt):
    """Bind the alternative authority to its exact declared readback source.

    This is the same bounded admission boundary as the legacy path, not a
    second full scientific/source interpreter. The canonical importer still
    rebuilds the entire branch after obtaining the collected inventory.
    """
    from .browser_autosave import MODE as AUTOSAVE_MODE
    from .browser_autosave import validate_autosave_receipt

    branch = (
        "no_planet"
        if receipt["mode"] == "no_planet_visible_workflow_readback"
        else "positive"
        if receipt["mode"]
        in {
            "positive_planet_visible_workflow_readback",
            POSITIVE_SUPPLIED_MODE,
        }
        else "terrestrial"
    )
    candidates = []
    for name in receipt["source_sha256"]:
        if Path(name).name != "confirmed.json":
            continue
        source = book.json(book.history / name)
        if isinstance(source, dict) and source.get("mode") == AUTOSAVE_MODE:
            candidates.append((book.history / name, source))
    _require(len(candidates) == 1, "ambiguous_autosave_readback_source")
    path, source = candidates[0]
    book.clean(path.parent)
    validate_autosave_receipt(
        source,
        branch=branch,
        star=receipt["star"],
        output=str(path.parent.relative_to(book.history)),
        extra_keys={"choice_path", "choice_sha256"} if branch == "no_planet" else (),
    )
    _require(
        all(
            receipt["source_sha256"].get(name) == checksum
            for name, checksum in source["source_sha256"].items()
        ),
        "autosave_readback_source_mismatch",
    )


def _workflow(book, directory, checksum, star):
    """Pin every declared immutable source plus each verified current capture.

    This is an evidence precondition for collection navigation, not a substitute
    for the journal importer's full workflow/scientific provenance validation.
    """
    directory = book.clean(directory)
    raw = book.read(directory / "confirmed.json")
    _require(isinstance(checksum, str) and re.fullmatch(r"[a-f0-9]{64}", checksum), "invalid_workflow_hash")
    _require(_sha(raw) == checksum, "workflow_hash_mismatch")
    receipt = json.loads(raw)
    from .browser_autosave import ACKNOWLEDGEMENT_SOURCE, validate_autosave_workflow_flags

    autosave = isinstance(receipt, dict) and receipt.get("save_strategy") == "autosave"
    validate_autosave_workflow_flags(receipt, enabled=autosave)
    _require(
        isinstance(receipt, dict)
        and type(receipt.get("schema_version")) is int
        and receipt["schema_version"] == 1
        and isinstance(receipt.get("mode"), str)
        and receipt.get("mode") in MODES
        and receipt.get("authority") == "visible_workflow_readback"
        and isinstance(receipt.get("star"), str)
        and receipt["star"].casefold() == star.casefold()
        and receipt.get("task_completed") is True
        and receipt.get("save_acknowledgement_verified") is (not autosave)
        and (
            not autosave
            or (
                receipt.get("source_save_click_delivered") is False
                and receipt.get("save_acknowledgement_source") == ACKNOWLEDGEMENT_SOURCE
                and all(
                    receipt.get(key, False) is False
                    for key in (
                        "source_save_resumed_same_intent",
                        "source_save_continuation_stopped_predispatch",
                    )
                )
            )
        )
        and all(type(receipt.get(k)) is int and receipt[k] == 0 for k in ZERO)
        and all(receipt.get(k) is False for k in FALSE),
        "unsupported_completed_workflow",
    )
    hashes = receipt.get("source_sha256")
    _require(isinstance(hashes, dict) and 1 <= len(hashes) <= 2000, "missing_workflow_sources")
    for name, expected in hashes.items():
        _require(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and isinstance(expected, str)
            and re.fullmatch(r"[a-f0-9]{64}", expected),
            "invalid_source_hash",
        )
        _require(_sha(book.read(book.history / name)) == expected, "workflow_source_hash_mismatch")
    if autosave:
        _autosave_admission(book, receipt)
    sections = MODES[receipt["mode"]]
    _require(
        isinstance(receipt.get("current_screen_sha256"), dict)
        and set(receipt["current_screen_sha256"]) == set(sections),
        "missing_current_captures",
    )
    for section in sections:
        capture = book.capture(directory / (section + "-readback") / "verified")
        _require(
            screen_identity(capture) == receipt["current_screen_sha256"][section]
            and project_view(capture) == {"surface": "detail", "section": section, "star": receipt["star"]},
            "workflow_current_capture_mismatch",
        )
    book.unchanged()
    return receipt, capture


class _CancelledAdvance(Exception):
    pass


class ProjectInventorySteps:
    """Navigate to the full stellar list and produce its unchanged child receipt.

    Constructor performs offline evidence checks only. Legacy mode permits three
    advances/two navigation clicks/one verifier. Explicit live pagination permits
    eleven advances including two navigation calls, offline child construction,
    and at most eight separately scheduled pager advances/four page clicks.
    Neither a count nor this component's completion grades or completes a task.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        workflow_dir,
        workflow_sha256,
        expected_star,
        expected_stars,
        max_seconds=180,
        inventory_mode="legacy_single_page",
        emit=lambda _kind, _payload: None,
        cancelled=lambda: False,
    ):
        self.expected_stars = _names(expected_stars)
        _require(
            inventory_mode in {"legacy_single_page", "live_paginated"}
            and (inventory_mode != "live_paginated" or 11 <= len(self.expected_stars) <= 30),
            "invalid_inventory_mode_or_count",
        )
        self.inventory_mode = inventory_mode
        paginated = inventory_mode == "live_paginated"
        _require(
            isinstance(expected_star, str)
            and re.fullmatch(NAME, expected_star)
            and expected_star == " ".join(expected_star.split())
            and expected_star.casefold() in {v.casefold() for v in self.expected_stars},
            "invalid_or_missing_expected_star",
        )
        _require(
            type(max_seconds) in {int, float}
            and math.isfinite(max_seconds)
            and 0 < max_seconds <= (420 if paginated else 180),
            "invalid_time_budget",
        )
        _require(callable(emit) and callable(cancelled), "invalid_callback")
        self._started, self.max_seconds = time.monotonic(), max_seconds
        self.book = _Evidence(run_history)
        source = Path(workflow_dir)
        self.workflow_dir = self.book.path(source if source.is_absolute() else self.book.history / source)
        self.source, self._previous = _workflow(self.book, self.workflow_dir, workflow_sha256, expected_star)
        self.output = self.book.path(output)
        _require(
            self.output != self.book.history and not self.output.is_relative_to(self.workflow_dir),
            "invalid_output",
        )
        self.config = config.model_copy(deep=True)
        rules = [r for r in self.config.frames if r.url == SIMULATION_URL]
        _require(len(rules) == 1 and rules[0].count == 1, "unsupported_simulation_boundary")
        rules[0].required_text = []
        self._config = self.config.model_dump(mode="json")
        self.page, self.star, self._callback, self._cancelled = page, expected_star, emit, cancelled
        self.phase, self.status = "to_list", "ready"
        self.finished, self.report, self.failure = False, None, None
        self.advances = self.navigation_clicks = self.navigation_attempts = self._sequence = 0
        self._busy = self._finalizing = self._forward_failed = False
        self._frames, self._dialogs, self._popups = None, [], []
        self._listeners = False
        self._view = project_view(self._previous)
        self._inventory = self._inventory_sha256 = None
        self._pager = None
        self._pager_state = None
        self._binding_fields = {}
        self._relay = ProjectEventRelay(self._emit)
        self.output.mkdir(parents=True, exist_ok=False)
        self._stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        self._run_id = (
            "project-inventory-" + _sha(str(self.output.relative_to(self.book.history)).encode())[:16]
        )
        self.scope = {
            "mode": "cooperative_live_paginated_project_inventory"
            if paginated
            else "cooperative_visible_project_inventory",
            "star": self.star,
            "expected_stars": list(self.expected_stars),
            "max_advances": 11 if paginated else 3,
            "max_seconds": max_seconds,
            "max_navigation_clicks": 2,
            "max_inventory_calls": 1,
            "workflow_path": str((self.workflow_dir / "confirmed.json").relative_to(self.book.history)),
            "workflow_sha256": workflow_sha256,
            "automatic_retry": False,
            "pagination_supported": paginated,
            "cancellation": "between_bounded_adapter_calls",
            "pause_counts_toward_deadline": True,
            "journal_writes": 0,
            "class_writes": 0,
            "scientific_verified": False,
            "task_completed": False,
            "project_completed": False,
            **{key: 0 for key in ZERO},
        }
        if paginated:
            self.scope.update(
                inventory_mode="live_paginated",
                inventory_evidence_version=1,
                max_paginated_seconds=240,
                max_paginated_advances=8,
                max_pagination_clicks=4,
                required_initial_page=1,
            )
        self._scope = deepcopy(self.scope)
        persist_json(self.output / "scope.json", self.scope)
        self.book.read(self.output / "scope.json")
        try:
            self._emit("hello", {"protocol_version": 1, **self.scope})
            self._check()
            self._emit("state", self.state())
        except _CancelledAdvance:
            pass
        except (KeyboardInterrupt, SystemExit):
            self._stop("operator_aborted", aborted=True)
            raise
        except Exception as exc:  # noqa: BLE001 - no arbitrary callback text
            self._stop(self._reason(exc))

    @staticmethod
    def _reason(exc):
        reason = (
            str(exc) if isinstance(exc, BrowserSafetyStop) else "project_inventory_steps_operation_failed"
        )
        return (
            "project_inventory_steps_unsupported_pagination"
            if reason == "project_inventory_incomplete_visible_list"
            else reason
        )

    def _dialog(self, _dialog):
        self._dialogs.append(True)

    def _popup(self, _page):
        self._popups.append(True)

    def _context(self):
        _require(
            not self._dialogs
            and not self._popups
            and self.page.frames == self._frames
            and len(self.page.context.pages) == 1,
            "context_changed",
        )

    def _check(self):
        if self.finished:
            raise _CancelledAdvance
        if self._cancelled():
            self._stop("operator_aborted", aborted=True)
            raise _CancelledAdvance
        _require(not self._forward_failed, "event_forwarding_failed")
        _require(time.monotonic() - self._started < self.max_seconds, "time_limit")
        _require(self.config.model_dump(mode="json") == self._config, "config_changed")
        _require(
            self.scope == self._scope
            and self.max_seconds == self._scope["max_seconds"]
            and list(self.expected_stars) == self._scope["expected_stars"]
            and self.inventory_mode == self._scope.get("inventory_mode", "legacy_single_page"),
            "scope_changed",
        )
        self.book.unchanged()
        if self._listeners:
            self._context()

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
                self._callback(kind, json.loads(line)["payload"])
            except Exception:  # noqa: BLE001 - driver/callback messages may contain private data
                self._forward_failed = True
                if not self._finalizing:
                    raise BrowserSafetyStop("project_inventory_steps_event_forwarding_failed") from None
        if self.finished and not self._finalizing:
            raise _CancelledAdvance

    def state(self):
        result = {
            **deepcopy(self.scope),
            "phase": self.phase,
            "status": self.status,
            "finished": self.finished,
            "advances": self.advances,
            "navigation_clicks": self.navigation_clicks,
            "navigation_attempts": self.navigation_attempts,
            "navigation_may_have_occurred": self.navigation_attempts > self.navigation_clicks,
            "collection_count_verified": self._inventory is not None and self.status == "completed",
            "inventory_dir": str((self.output / "inventory").relative_to(self.book.history))
            if self._inventory
            else None,
            "inventory_sha256": self._inventory_sha256,
            "failure_reason": self.failure,
            "event_forwarding_failed": self._forward_failed,
        }
        if self.inventory_mode == "live_paginated":
            result.update(paginated_inventory=deepcopy(self._pager_state), **deepcopy(self._binding_fields))
        return result

    def _pager_cancelled(self):
        if self.finished:
            return True
        self._check()
        return False

    def _paged(self):
        if self.phase == "verify_inventory":
            remaining = self.max_seconds - (time.monotonic() - self._started)
            _require(remaining > 0, "time_limit")
            child = PaginatedInventorySteps(
                self.page,
                self.config,
                self.output / "inventory",
                run_history=self.book.history,
                expected_stars=self.expected_stars,
                max_seconds=min(240, remaining),
                max_clicks=4,
                max_advances=8,
                emit=self._relay.bind("project.inventory.pages", star=self.star),
                cancelled=self._pager_cancelled,
            )
            self._pager = child
            if self.finished:
                child.abort()
                return
            self.phase = "paginated_active"
        else:
            _require(self._pager is not None, "missing_paginated_child")
            self._pager.advance()
        self._check()
        child = self._pager.state()
        self._pager_state = deepcopy(child)
        if self._pager.finished:
            if child.get("status") != "completed" and child.get("failure_reason"):
                reason = child["failure_reason"]
                _require(
                    isinstance(reason, str) and re.fullmatch(r"[a-z0-9_]{1,180}", reason),
                    "invalid_child_failure",
                )
                raise BrowserSafetyStop(reason)
            _require(
                child.get("status") == "completed"
                and child.get("phase") == "inventory_verified"
                and child.get("finished") is True
                and child.get("failure_reason") is None
                and child.get("task_completed") is False
                and child.get("project_completed") is False,
                "paginated_child_unverified",
            )
            directory = self.output / "inventory"
            _require(
                child.get("inventory_dir") == str(directory.relative_to(self.book.history))
                and isinstance(child.get("inventory_sha256"), str)
                and re.fullmatch(r"[a-f0-9]{64}", child["inventory_sha256"]),
                "paginated_output_changed",
            )
            source = load_inventory_source(
                self.book, directory, expected_sha256=child.get("inventory_sha256")
            )
            _require(
                source.kind == "live_paginated"
                and source.receipt == self._pager.report
                and {row["name"].casefold() for row in source.rows}
                == {name.casefold() for name in self.expected_stars},
                "paginated_sources_or_names_changed",
            )
            # The producer proves a full sweep/revisit and the actual fresh
            # first-page anchor. Never invent a full-list `after` capture.
            self._same_screen(source.anchor)
            self._binding_fields = source.binding_fields()
            self._inventory = source.receipt
            self._inventory_sha256 = _sha(self.book.read(directory / "confirmed.json"))
            self._relay.retire()
            self._check()
            self._finish("completed", "inventory_verified")
        else:
            self._emit("state", self.state())
            self._check()

    def _same_screen(self, capture):
        _require(
            project_view(capture) == self._view
            and screen_identity(navigation_status_projection(capture, self._view))
            == screen_identity(navigation_status_projection(self._previous, self._view)),
            "current_context_or_answers_changed",
        )

    def _guard(self):
        if not self._listeners:
            self._frames = self.page.frames.copy()
            self.page.on("dialog", self._dialog)
            self.page.context.on("page", self._popup)
            self._listeners = True
        self._context()
        report = inspect_page(self.page, self.config)
        self._check()
        _require(report.get("ignored_frame_urls") == [], "unknown_frame")
        self._same_screen(report)
        guard_dir = self.output / f"guard-{self.advances}"
        save_probe(report, guard_dir)
        self._same_screen(self.book.capture(guard_dir))

    def _navigate(self, destination):
        directory = self.output / ("to-" + destination)
        self.navigation_attempts += 1
        receipt = navigate_project(
            self.page,
            self.config,
            directory,
            destination,
            expected_star=self.star if destination == "list" else None,
        )
        self._check()
        self.book.clean(directory)
        _require(self.book.json(directory / "confirmed.json") == receipt, "navigation_receipt_changed")
        self._same_screen(self.book.capture(directory / "before"))
        self._same_screen(self.book.capture(directory / "pre-click"))
        intent = self.book.json(directory / "reserved.json")
        _require(
            all(receipt.get(key) == value for key, value in intent.items())
            and intent.get("mode") == "bounded_project_navigation"
            and intent.get("max_clicks") == 1
            and intent.get("automatic_retry") is False,
            "navigation_intent_changed",
        )
        after = self.book.capture(directory / "after")
        view = project_view(after)
        _require(
            receipt.get("from") == self._view
            and receipt.get("to") == view
            and receipt.get("destination") == destination
            and receipt.get("destination_verified") is True
            and receipt.get("navigation_clicks") == 1
            and receipt.get("task_completed") is False
            and view["surface"] == "list"
            and view["star"] is None
            and (destination == "list" or view["section"] == "stellar")
            and all(
                type(receipt.get(k)) is int and receipt[k] == 0
                for k in (
                    "answer_writes",
                    "collection_clicks",
                    "save_clicks",
                    "assessment_clicks",
                    "submission_clicks",
                )
            ),
            "navigation_not_verified",
        )
        self.navigation_clicks += 1
        self._previous, self._view = after, view
        self.phase = "verify_inventory" if view["section"] == "stellar" else "to_stellar"

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy or self._finalizing:
            self._stop("project_inventory_steps_reentrant_advance")
            raise BrowserSafetyStop("project_inventory_steps_reentrant_advance")
        self._busy = True
        try:
            self._check()
            _require(self.advances < self.scope["max_advances"], "advance_limit")
            self.advances += 1
            self._emit("state", {**self.state(), "scheduled_call": self.phase})
            self._check()
            if self.phase != "paginated_active":
                self._guard()
            self._check()
            if self.phase in {"to_list", "to_stellar"}:
                self._navigate("list" if self.phase == "to_list" else "stellar")
                self._emit("state", self.state())
                self._check()
            elif self.inventory_mode == "live_paginated" and self.phase in {
                "verify_inventory",
                "paginated_active",
            }:
                self._paged()
            elif self.phase == "verify_inventory":
                directory = self.output / "inventory"
                receipt = verify_project_inventory(
                    self.page, self.config, directory, expected_stars=self.expected_stars
                )
                self._check()
                self.book.clean(directory)
                _require(self.book.json(directory / "confirmed.json") == receipt, "inventory_receipt_changed")
                self._same_screen(self.book.capture(directory / "before"))
                self._same_screen(self.book.capture(directory / "after"))
                _require(
                    receipt.get("collection_count_verified") is True
                    and receipt.get("complete_visible_list_verified") is True
                    and receipt.get("expected_stars_verified") is True
                    and receipt.get("task_completed") is False
                    and receipt.get("project_completed") is False,
                    "inventory_not_verified",
                )
                self._inventory = receipt
                self._inventory_sha256 = _sha(self.book.read(directory / "confirmed.json"))
                self._check()
                self._finish("completed", "inventory_verified")
            else:
                raise BrowserSafetyStop("project_inventory_steps_unsupported_phase")
        except _CancelledAdvance:
            pass
        except (KeyboardInterrupt, SystemExit):
            self._stop("operator_aborted", aborted=True)
            raise
        except Exception as exc:  # noqa: BLE001 - stop; never retry a navigation
            if not self.finished:
                self._stop(self._reason(exc))
        finally:
            self._busy = False
        return self.state()

    def _finish(self, status, phase):
        if self.finished:
            return self.report
        self.status, self.phase, self.finished, self._finalizing = status, phase, True, True
        if status != "completed":
            self._inventory = self._inventory_sha256 = None
        self.report = {**self.state(), "source_sha256": dict(self.book.hashes)}
        persist_json(self.output / ("report.json" if status == "completed" else "stopped.json"), self.report)
        try:
            self._emit("episode_summary", self.report)
            final_error = "project_inventory_steps_event_forwarding_failed" if self._forward_failed else None
            if status == "completed" and final_error is None:
                try:
                    _require(not self._cancelled(), "cancelled_after_completion")
                    _require(time.monotonic() - self._started < self.max_seconds, "time_limit")
                    _require(self.config.model_dump(mode="json") == self._config, "config_changed")
                    _require(self.scope == self._scope, "scope_changed")
                    _require(
                        json.loads((self.output / "report.json").read_bytes()) == self.report,
                        "terminal_report_changed",
                    )
                    self.book.unchanged()
                    if self._listeners:
                        self._context()
                except Exception as exc:  # noqa: BLE001 - preserve evidence; never retry
                    final_error = self._reason(exc)
            if status == "completed" and final_error is not None:
                # Keep the emitted/source artifacts, but never make a failed
                # delivery look like a usable owner handoff.
                self.status, self.phase = "stopped", "stopped"
                self.failure = final_error
                self._inventory = self._inventory_sha256 = None
                self._binding_fields = {}
                self.report = {**self.state(), "source_sha256": dict(self.book.hashes)}
                persist_json(self.output / "invalidated.json", self.report)
                persist_json(self.output / "stopped.json", self.report)
        finally:
            self._finalizing = False
            self._stream.close()
            if self._listeners:
                self.page.remove_listener("dialog", self._dialog)
                self.page.context.remove_listener("page", self._popup)
                self._listeners = False
        return self.report

    def _stop(self, reason, aborted=False):
        self.failure = reason
        result = self._finish("aborted" if aborted else "stopped", "aborted" if aborted else "stopped")
        self._relay.retire()
        if self._pager is not None and not self._pager.finished:
            self._pager.abort()
        return result

    def abort(self):
        return self.report if self.finished else self._stop("operator_aborted", aborted=True)

    def close(self):
        return self.report if self.finished else self._stop("operator_closed", aborted=True)
