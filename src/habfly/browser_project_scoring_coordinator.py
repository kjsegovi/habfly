"""Cooperative panel setup plus canonical assessment/score transfer, never submission."""

import hashlib
import math
import re
import time
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json, rows_hash
from .browser_assessment_panel_steps import MODE as PANEL_MODE
from .browser_assessment_panel_steps import ZERO as PANEL_ZERO
from .browser_assessment_panel_steps import AssessmentPanelSteps, _public_panel
from .browser_no_planet_workflow import _Evidence
from .browser_numeric import screen_identity
from .browser_project_paginated_inventory_steps import PaginatedInventorySteps, parse_inventory_page
from .browser_project_scoring_steps import KINDS, BrowserProjectAssessmentSteps
from .browser_stellar import SIMULATION_URL
from .contracts import RuntimeEvent
from .project_events import ProjectEventRelay
from .project_inventory_source import load_inventory_source

MODE = "cooperative_project_scoring_with_panel_navigation"


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("scoring_coordinator_" + reason)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BrowserProjectScoringCoordinator:
    """Offline constructor; each advance schedules at most one child advance.

    The original scoring child enforces thirty unique verified tasks, a complete
    inventory, canonical reservations, two $100 simulation assessments, two
    acknowledgements and optional one score transfer. Their limits are unchanged.
    There is no partial-list fallback or automatic retry/reconstruction.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        journal,
        run_history,
        inventory_dir,
        assessment_history_root,
        allow_score_transfer=False,
        max_seconds=600,
        panel_max_seconds=60,
        assessment_timeout_seconds=10,
        score_timeout_seconds=15,
        emit=lambda _: None,
        cancelled=lambda: False,
        _clock=time.monotonic,
    ):
        _require(type(allow_score_transfer) is bool, "explicit_transfer_flag_required")
        for value, minimum, maximum in (
            (max_seconds, 1, 600),
            (panel_max_seconds, 0.1, 60),
            (assessment_timeout_seconds, 0.1, 30),
            (score_timeout_seconds, 0.1, 30),
        ):
            _require(
                type(value) in {int, float} and math.isfinite(value) and minimum <= value <= maximum,
                "invalid_fixed_budget",
            )
        _require(callable(emit) and callable(cancelled) and callable(_clock), "invalid_callback")
        self.book = _Evidence(run_history)
        self.output = self.book.path(output)
        _require(self.output != self.book.history and not self.output.exists(), "invalid_output")
        self.page, self.config, self.journal = page, config.model_copy(deep=True), journal
        self._config = self.config.model_dump(mode="json")
        self._callback, self._cancelled, self._clock = emit, cancelled, _clock
        self._started = _clock()
        self._options = (
            allow_score_transfer,
            max_seconds,
            panel_max_seconds,
            assessment_timeout_seconds,
            score_timeout_seconds,
        )
        (
            self.allow_score_transfer,
            self.max_seconds,
            self.panel_max_seconds,
            self.assessment_timeout_seconds,
            self.score_timeout_seconds,
        ) = self._options
        self.finished, self.report, self.failure = False, None, None
        self.phase, self.kind = "panel_initializing", KINDS[0]
        self._busy = self._closing = self._emitting = self._forward_failed = False
        self._cleanup_failed = False
        self._sequence = self.advances = 0
        self.panel = self._panel_state = self._scoring_state = self._scoring_forward = None
        self.inventory = self._inventory_state = self._inventory_forward = None
        self._inventory_kind = self._inventory_boundary = self._inventory_max_seconds = None
        self._completed_inventories = {}
        self._completed_panels, self._completed_trees = {}, {}
        self._relay = ProjectEventRelay(self._receive)
        # Existing constructor performs the complete offline gate and claims the
        # revision canonically before any panel work. Do not duplicate/relax it.
        self.scoring = BrowserProjectAssessmentSteps(
            page,
            self.config,
            self.output / "scoring",
            journal=journal,
            run_history=self.book.history,
            inventory_dir=inventory_dir,
            assessment_history_root=assessment_history_root,
            allow_score_transfer=allow_score_transfer,
            max_seconds=max_seconds,
            max_advances=6,
            assessment_timeout_seconds=assessment_timeout_seconds,
            score_timeout_seconds=score_timeout_seconds,
            emit=self._from_scoring,
            cancelled=self._child_cancelled,
            _clock=_clock,
        )
        self._journal_path = self.scoring.journal_path
        self.inventory_dir = self.scoring.inventory_dir
        self.inventory_sha = _sha(self.inventory_dir / "confirmed.json")
        # The parent must adopt clean-directory guards as well as hashes: the
        # scoring child is idle while a pager child dispatches navigation.
        self.inventory_source = load_inventory_source(
            self.book, self.inventory_dir, expected_sha256=self.inventory_sha
        )
        _require(
            self.inventory_source.rows == self.scoring.inventory_source.rows
            and self.inventory_source.kind == self.scoring.inventory_source.kind
            and self.inventory_source.binding_fields() == self.scoring.inventory_source.binding_fields(),
            "child_inventory_source_mismatch",
        )
        self.paginated = self.inventory_source.kind == "live_paginated"
        self.max_advances = 46 if self.paginated else 16
        self._inventory_binding = deepcopy(self.inventory_source.binding_fields())
        self._scoring_state = self.scoring.state()
        self._copy_child_sources(self._scoring_state)
        self.scope = {
            "schema_version": 1,
            "mode": MODE,
            "data_revision": self.scoring.revision,
            "project_rows_sha256": self.scoring.rows_sha,
            "collected": 30,
            "allow_score_transfer": allow_score_transfer,
            "max_seconds": max_seconds,
            "panel_max_seconds": panel_max_seconds,
            "max_advances": self.max_advances,
            "max_scoring_advances": 6,
            "max_panel_advances_each": 3,
            "max_panel_clicks_each": 2,
            "max_assessment_clicks": 2,
            "max_acknowledgement_clicks": 2,
            "max_score_transfer_clicks": int(allow_score_transfer),
            "budget_simulation_dollars": 200,
            "pause_counts_toward_deadline": True,
            "automatic_retry": False,
            "complete_visible_inventory_required": not self.paginated,
            "pagination_supported": self.paginated,
            "browser_owned_by_caller": True,
            "submission_enabled": False,
            "submitted": False,
            "task_completed": False,
            "project_completed": False,
            "scientific_verified": False,
            "training_label": False,
            "real_money": False,
            **self._inventory_binding,
            **(
                {
                    "max_inventory_sweeps": 2 + int(allow_score_transfer),
                    "max_inventory_advances_each": 8,
                    "max_inventory_clicks_each": 4,
                    "max_inventory_seconds_each": 240,
                    "inventory_seconds_capped_to_parent_remaining": True,
                    "fresh_full_sweep_before_each_charge_and_transfer": True,
                    "atomic_server_snapshot_verified": False,
                }
                if self.paginated
                else {}
            ),
        }
        persist_json(self.output / "scope.json", self.scope)
        self.book.read(self.output / "scope.json")
        self._stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        try:
            self._emit("hello", {"protocol_version": 1, **self.scope})
            self._check()
        except Exception as exc:  # noqa: BLE001 - source/callback errors remain sanitized
            self._stop(exc)

    def _copy_child_sources(self, state):
        sources = state.get("source_sha256")
        _require(isinstance(sources, dict) and 1 <= len(sources) <= 3000, "missing_child_sources")
        for name, expected in sources.items():
            _require(
                isinstance(name, str)
                and not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and isinstance(expected, str)
                and re.fullmatch(r"[a-f0-9]{64}", expected),
                "invalid_child_source",
            )
            _require(
                hashlib.sha256(self.book.read(self.book.history / name)).hexdigest() == expected,
                "child_source_changed",
            )

    def _tree(self, directory):
        files = sorted(directory.rglob("*"))
        _require(len(files) <= 3000, "source_tree_limit")
        result = {}
        for file in files:
            file = self.book.path(file)
            _require(
                file.name
                not in {
                    "stopped.json",
                    "invalidated.json",
                    "finalization_failed.json",
                    "event-forwarding-failed.json",
                }
                and not file.name.endswith("-stopped.json"),
                "failed_source",
            )
            if file.is_file():
                result[str(file.relative_to(self.book.history))] = hashlib.sha256(
                    self.book.read(file)
                ).hexdigest()
        return result

    def _check(self):
        _require(not self.finished and not self._cancelled(), "cancelled_or_stopped")
        _require(self._clock() - self._started < self.max_seconds, "time_limit")
        _require(
            (
                self.allow_score_transfer,
                self.max_seconds,
                self.panel_max_seconds,
                self.assessment_timeout_seconds,
                self.score_timeout_seconds,
            )
            == self._options,
            "budgets_changed",
        )
        _require(self.config.model_dump(mode="json") == self._config, "boundary_changed")
        _require(
            self.paginated is (self.inventory_source.kind == "live_paginated")
            and self.max_advances == (46 if self.paginated else 16)
            and self.inventory_source.binding_fields() == self._inventory_binding
            and self.scoring.inventory_source.binding_fields() == self._inventory_binding
            and self.scoring.paginated is self.paginated,
            "inventory_binding_or_budget_changed",
        )
        state = self.scoring.state()
        _require(
            state.get("max_advances") == 6
            and state.get("max_seconds") == self.max_seconds
            and state.get("allow_score_transfer") is self.allow_score_transfer,
            "child_budgets_changed",
        )
        _require(
            self.book.path(self.journal.path) == self._journal_path
            and _sha(self._journal_path) == state.get("journal_sha256"),
            "canonical_journal_changed",
        )
        self.book.unchanged()
        for directory, expected in self._completed_trees.items():
            _require(self._tree(self.book.history / directory) == expected, "completed_tree_changed")

    def _child_cancelled(self):
        # Children also call this immediately before their own native dispatch.
        # Thus the overall deadline and completed sibling source pins remain in
        # force inside a long child advance, not merely between scheduled calls.
        if self.finished or self._closing:
            return True
        self._check()
        return False

    def _emit(self, kind, payload):
        event = RuntimeEvent(event=kind, sequence=self._sequence, run_id=self.output.name, payload=payload)
        self._stream.write(event.model_dump_json() + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            self._emitting = True
            try:
                self._callback(event.model_dump(mode="json"))
            except Exception:  # noqa: BLE001 - do not expose driver/private callback text
                self._forward_failed = True
                raise BrowserSafetyStop("scoring_coordinator_event_forwarding_failed") from None
            finally:
                self._emitting = False

    def _from_scoring(self, event):
        if self._closing:
            # An aborted child persists its own terminal stream. It may not
            # resurrect a retired generation or interrupt the parent's stop.
            return
        _require(
            self.phase == "scoring_active" and self._scoring_forward is not None, "unscheduled_scoring_event"
        )
        self._scoring_forward(event)

    def _receive(self, kind, payload):
        if not self._closing:
            self._check()
        candidate = payload.get("component_state", payload.get("component_summary"))
        if isinstance(candidate, dict):
            if payload.get("component") == "project.assessment":
                self._scoring_state = deepcopy(candidate)
            elif str(payload.get("component", "")).startswith("project.inventory."):
                self._inventory_state = deepcopy(candidate)
            else:
                self._panel_state = deepcopy(candidate)
        self._emit(kind, {**payload, **self.state()} if kind == "state" else payload)
        if not self._closing:
            self._check()

    def state(self):
        return {
            **self.scope,
            "phase": self.phase,
            "finished": self.finished,
            "failure_reason": self.failure,
            "advances": self.advances,
            "requested_panel": self.kind,
            "panel_component": deepcopy(self._panel_state),
            "scoring_component": deepcopy(self._scoring_state),
            "verified_panels": list(self._completed_panels),
            "score_transfer_verified": self.phase == "score_transferred_not_submitted",
            "scoring_dir": str(self.scoring.output.relative_to(self.book.history)),
            "event_forwarding_failed": self._forward_failed,
            "cleanup_failed": self._cleanup_failed,
            **(
                {
                    "inventory_component": deepcopy(self._inventory_state),
                    "requested_inventory_refresh": self._inventory_kind,
                    "verified_inventory_refreshes": deepcopy(self._completed_inventories),
                }
                if self.paginated
                else {}
            ),
        }

    def _pin_panel(self):
        directory = self.output / ("panel-" + self.kind)
        report = self.book.json(directory / "confirmed.json")
        _require(
            report == self.panel.report
            and type(report.get("schema_version")) is int
            and report["schema_version"] == 1
            and report.get("mode") == PANEL_MODE
            and report.get("kind") == self.kind
            and report.get("phase") == "panel_verified"
            and report.get("finished") is True
            and report.get("panel_verified") is True
            and report.get("failure_reason") is None
            and report.get("inventory_sha256") == self.inventory_sha
            and report.get("inventory_path")
            == str((self.inventory_dir / "confirmed.json").relative_to(self.book.history))
            and report.get("project_rows_sha256") == self.scoring.rows_sha
            and type(report.get("collected")) is int
            and report["collected"] == 30
            and type(report.get("max_advances")) is int
            and report["max_advances"] == 3
            and type(report.get("max_navigation_clicks")) is int
            and report["max_navigation_clicks"] == 2
            and report.get("max_seconds") == self.panel_max_seconds
            and all(report.get(k) == v for k, v in self._inventory_binding.items())
            and all(type(report.get(k)) is int and report[k] == 0 for k in PANEL_ZERO)
            and all(
                report.get(k) is False
                for k in (
                    "assessment_confirmed",
                    "task_completed",
                    "project_completed",
                    "scientific_verified",
                    "automatic_retry",
                )
            ),
            "panel_receipt_unverified",
        )
        _require(
            type(report.get("navigation_clicks")) is int
            and 0 <= report["navigation_clicks"] <= 2
            and type(report.get("navigation_dispatch_attempts")) is int
            and report.get("navigation_dispatch_attempts") == report["navigation_clicks"]
            and type(report.get("advances")) is int
            and 1 <= report["advances"] <= 3,
            "panel_budget_unverified",
        )
        self._copy_child_sources(report)
        captured = self.book.capture(directory / "verified")
        _, panel, counts, rows, _ = _public_panel(captured, inventory_kind=self.inventory_source.kind)
        if self.paginated:
            anchor_counts, anchor_rows = parse_inventory_page(self.inventory_source.anchor)
            _require(
                counts == anchor_counts and rows == anchor_rows and len(rows) == 10,
                "paginated_panel_anchor_changed",
            )
        else:
            _require(len(rows) == 30, "panel_capture_unverified")
        text = next(f["text"] for f in captured["frames"] if f["url"] == SIMULATION_URL)
        _require(
            panel is not None
            and panel.mode == self.kind
            and panel.acknowledgement is None
            and panel.model_dump(mode="json") == report.get("assessment")
            and counts["total"] == 30
            and rows_hash(text) == self.scoring.rows_sha
            and screen_identity(captured) == report.get("current_screen_sha256"),
            "panel_capture_unverified",
        )
        tree = self._tree(directory)
        self._completed_trees[str(directory.relative_to(self.book.history))] = tree
        self._completed_panels[self.kind] = {
            "path": str((directory / "confirmed.json").relative_to(self.book.history)),
            "sha256": _sha(directory / "confirmed.json"),
        }
        self._relay.retire()
        self._check()

    def _from_inventory(self, event):
        if self._closing:
            return
        _require(
            self.phase == "inventory_active" and self._inventory_forward is not None,
            "unscheduled_inventory_event",
        )
        self._inventory_forward(event)

    def _schedule_inventory(self, kind):
        _require(
            self.paginated and kind in {*KINDS, "score_transfer"} and kind not in self._completed_inventories,
            "unexpected_inventory_schedule",
        )
        _require(
            (
                kind in KINDS
                and self.kind == kind
                and kind in self._completed_panels
                and self.scoring.phase == f"awaiting_{kind}_panel"
            )
            or (
                kind == "score_transfer"
                and self.allow_score_transfer
                and self.scoring.phase == "score_transfer_ready"
                and self.scoring.state().get("acknowledgements_verified") == 2
            ),
            "inventory_boundary_not_ready",
        )
        self._relay.retire()
        self._inventory_kind, self.phase = kind, "inventory_initializing"
        self.inventory = self._inventory_state = None
        self._inventory_forward = None

    def _initialize_inventory(self):
        self._check()
        remaining = self.max_seconds - (self._clock() - self._started)
        _require(remaining >= 1, "inventory_deadline_exhausted")
        self._inventory_max_seconds = min(240, remaining)
        kind = self._inventory_kind
        boundary_dir = (
            self.scoring.stage_dir / "ack-001-after"
            if kind == "score_transfer"
            else self.output / ("panel-" + kind) / "verified"
        )
        self._inventory_boundary = self.book.capture(boundary_dir)
        self._inventory_forward = self._relay.bind("project.inventory." + kind)
        self.inventory = PaginatedInventorySteps(
            self.page,
            self.config,
            self.output / ("inventory-" + kind),
            run_history=self.book.history,
            expected_stars=[row["name"] for row in self.inventory_source.rows],
            max_seconds=self._inventory_max_seconds,
            max_advances=8,
            max_clicks=4,
            emit=self._from_inventory,
            cancelled=self._child_cancelled,
            _clock=self._clock,
        )
        _require(not self.inventory.finished, "inventory_constructor_failed")
        self._inventory_state = self.inventory.state()
        self.phase = "inventory_active"
        self._check()

    def _pin_inventory(self):
        self._check()
        kind = self._inventory_kind
        directory = self.output / ("inventory-" + kind)
        digest = _sha(directory / "confirmed.json")
        source = load_inventory_source(self.book, directory, expected_sha256=digest)
        receipt = source.receipt
        _require(
            receipt == self.inventory.report
            and source.kind == "live_paginated"
            and source.rows == self.inventory_source.rows
            and source.whole_collection_sha256 == self.inventory_source.whole_collection_sha256
            and receipt.get("max_seconds") == self._inventory_max_seconds
            and type(receipt.get("max_advances")) is int
            and receipt["max_advances"] == 8
            and type(receipt.get("max_clicks")) is int
            and receipt["max_clicks"] == 4
            and receipt.get("expected_stars") == [r["name"] for r in self.inventory_source.rows]
            and receipt.get("status") == "completed"
            and receipt.get("phase") == "inventory_verified"
            and receipt.get("finished") is True
            and receipt.get("failure_reason") is None,
            "inventory_refresh_unverified",
        )
        first_dir = self.book.path(self.book.history / receipt["native_pages"][0]["path"]).parent
        first = self.book.capture(first_dir / "before")
        _require(
            screen_identity(first) == screen_identity(self._inventory_boundary)
            and screen_identity(source.anchor) == screen_identity(self._inventory_boundary)
            and rows_hash(next(f["text"] for f in source.anchor["frames"] if f["url"] == SIMULATION_URL))
            == self.scoring.rows_sha,
            "inventory_refresh_boundary_changed",
        )
        self._copy_child_sources(receipt)
        tree = self._tree(directory)
        self._completed_trees[str(directory.relative_to(self.book.history))] = tree
        self._completed_inventories[kind] = {
            "inventory_dir": str(directory.relative_to(self.book.history)),
            "inventory_sha256": digest,
            **source.binding_fields(),
        }
        self._relay.retire()
        self._check()

    def _handoff_inventory(self):
        self._check()
        kind = self._inventory_kind
        record = self._completed_inventories[kind]
        self.scoring.provide_inventory_refresh(
            kind,
            inventory_dir=self.book.history / record["inventory_dir"],
            inventory_sha256=record["inventory_sha256"],
        )
        self._scoring_state = self.scoring.state()
        self._copy_child_sources(self._scoring_state)
        self._check()
        if kind == "score_transfer":
            self._scoring_forward = self._relay.bind("project.assessment")
            self.phase = "scoring_active"
        else:
            self.phase = "panel_handoff"

    def _pin_scoring(self):
        path = self.scoring.output / "report.json"
        report, state = self.book.json(path), self.scoring.state()
        expected = (
            "score_transferred_not_submitted"
            if self.allow_score_transfer
            else "assessed_score_transfer_disabled"
        )
        _require(
            report == self.scoring.report
            and all(report.get(k) == v for k, v in state.items())
            and report.get("status") == "completed"
            and report.get("phase") == expected
            and report.get("failure_reason") is None
            and report.get("finished") is True
            and isinstance(report.get("assessment_verified"), dict)
            and set(report["assessment_verified"]) == set(KINDS)
            and all(report["assessment_verified"][kind] is True for kind in KINDS)
            and type(report.get("acknowledgements_verified")) is int
            and report["acknowledgements_verified"] == 2
            and report.get("score_transfer_verified") is self.allow_score_transfer
            and report.get("pending_canonical_action") is None
            and report.get("pending_acknowledgement") is None
            and all(
                report.get(k) is False
                for k in (
                    "event_forwarding_failed",
                    "assessment_outcome_uncertain",
                    "task_completed",
                    "project_completed",
                    "submitted",
                )
            ),
            "scoring_result_unverified",
        )
        _require(
            report.get("events_sha256") == _sha(self.scoring.output / "events.jsonl"),
            "scoring_events_changed",
        )
        self._copy_child_sources(report)
        tree = self._tree(self.scoring.output)
        self._completed_trees[str(self.scoring.output.relative_to(self.book.history))] = tree
        self._scoring_state = state
        self._check()
        self._finish(expected)

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy or self._emitting:
            self._stop(BrowserSafetyStop("scoring_coordinator_reentrant_advance"))
            return self.state()
        self._busy = True
        try:
            self._check()
            _require(self.advances < self.max_advances, "advance_limit")
            self.advances += 1
            if self.phase == "panel_initializing":
                callback = self._relay.bind("project.panel." + self.kind)
                self.panel = AssessmentPanelSteps(
                    self.page,
                    self.config,
                    self.output / ("panel-" + self.kind),
                    run_history=self.book.history,
                    inventory_dir=self.inventory_dir,
                    inventory_sha256=self.inventory_sha,
                    kind=self.kind,
                    max_seconds=self.panel_max_seconds,
                    emit=callback,
                    cancelled=self._child_cancelled,
                )
                _require(not self.panel.finished, "panel_constructor_failed")
                self.phase = "panel_active"
                self._panel_state = self.panel.state()
            elif self.phase == "panel_active":
                self.panel.advance()
                if self.finished:
                    return self.state()
                self._panel_state = self.panel.state()
                if self.panel.finished:
                    self._pin_panel()
                    if self.paginated:
                        self._schedule_inventory(self.kind)
                    else:
                        self.phase = "panel_handoff"
            elif self.phase == "inventory_initializing":
                self._initialize_inventory()
            elif self.phase == "inventory_active":
                self.inventory.advance()
                if self.finished:
                    return self.state()
                self._inventory_state = self.inventory.state()
                if self.inventory.finished:
                    self._pin_inventory()
                    self.phase = "inventory_handoff"
            elif self.phase == "inventory_handoff":
                self._handoff_inventory()
            elif self.phase == "panel_handoff":
                self._check()
                self.scoring.provide_panel(self.kind)  # Scheduling only, no child advance or native action.
                self._scoring_state = self.scoring.state()
                self._scoring_forward = self._relay.bind("project.assessment")
                self.phase = "scoring_active"
            elif self.phase == "scoring_active":
                self.scoring.advance()
                if self.finished:
                    return self.state()
                self._scoring_state = self.scoring.state()
                self._copy_child_sources(self._scoring_state)
                if self.scoring.finished:
                    self._pin_scoring()
                elif (
                    self.paginated
                    and self.scoring.phase == "score_transfer_ready"
                    and "score_transfer" not in self._completed_inventories
                ):
                    self._schedule_inventory("score_transfer")
                elif self.scoring.phase == "awaiting_scavenger_hunt_panel":
                    _require(
                        self.kind == "data_quality" and "scavenger_hunt" not in self._completed_panels,
                        "repeated_panel_handoff",
                    )
                    self._relay.retire()
                    self.kind, self.phase, self.panel, self._panel_state = (
                        "scavenger_hunt",
                        "panel_initializing",
                        None,
                        None,
                    )
            else:
                raise BrowserSafetyStop("scoring_coordinator_unknown_phase")
            if not self.finished:
                self._emit("state", self.state())
                self._check()
        except BaseException as exc:
            self._stop(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            self._busy = False
        return self.state()

    def _finish(self, phase):
        self._check()
        self.phase = phase
        self._emit("episode_summary", {**self.state(), "finished": True})
        self._check()
        report = {
            **self.state(),
            "finished": True,
            "source_sha256": dict(self.book.hashes),
            "journal_sha256": _sha(self._journal_path),
            "events_sha256": _sha(self.output / "events.jsonl"),
        }
        persist_json(self.output / "report.json", report)
        self.report, self.finished = report, True
        self._relay.retire()
        self._stream.close()

    def _stop(self, exc):
        if self.finished or self._closing:
            return
        code = str(exc)
        self.failure = (
            code
            if isinstance(exc, BrowserSafetyStop) and re.fullmatch(r"[a-z][a-z0-9_]{0,140}", code)
            else "scoring_coordinator_operation_failed"
        )
        self.finished, self.phase, self._closing = True, "stopped", True
        try:
            for child in (self.panel, self.inventory, self.scoring):
                if child is not None and not child.finished:
                    try:
                        child.abort()
                    except Exception:  # noqa: BLE001 - keep all original claims; never retry cleanup
                        self._cleanup_failed = True
            self._scoring_state = self.scoring.state()
            if self.panel is not None:
                self._panel_state = self.panel.state()
            if self.inventory is not None:
                self._inventory_state = self.inventory.state()
            try:
                current_journal_sha = _sha(self.book.path(self._journal_path))
            except Exception:  # noqa: BLE001 - a missing/foreign journal is not current evidence
                current_journal_sha = None
            self.report = {
                **self.state(),
                "source_sha256": dict(self.book.hashes),
                "journal_sha256": current_journal_sha,
            }
            persist_json(self.output / "stopped.json", self.report)
            if not self._forward_failed:
                self._emit("error", {"type": "ScoringCoordinatorStop", "message": self.failure})
                self._emit("state", self.state())
        finally:
            self._relay.retire()
            self._stream.close()
            self._closing = False

    def abort(self):
        self._stop(BrowserSafetyStop("scoring_coordinator_operator_aborted"))
        return self.state()

    def close(self):
        return self.abort()
