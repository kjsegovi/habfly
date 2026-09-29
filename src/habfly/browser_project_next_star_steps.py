"""Cooperative transition from a completed owner to one fresh visible star.

No new selector, catalog, classification, science answer, journal append or
browser launch. The existing guarded navigator and one-shot picker own all UI
operations. A fresh capture is not a verified collection count or task receipt.
"""

import hashlib
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

from . import project_evidence, project_positive_evidence, project_terrestrial_evidence
from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_next_star import INITIAL_STARFIELD_IMAGE, NextStarPicker, validate_initial_selection
from .browser_no_planet_workflow import _Evidence
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_project_inventory_steps import _names
from .browser_project_navigation import navigate_project, navigation_status_projection, project_view
from .browser_setup import SetupStop, visible_star_point
from .browser_stellar import CLASSES, SIMULATION_URL, map_stellar_capture
from .contracts import RuntimeEvent
from .project_events import ProjectEventRelay
from .project_inventory_source import load_inventory_source
from .project_progress import ProjectJournal, TaskReceipt
from .supplied_browser_modes import POSITIVE_SUPPLIED_MODE, TERRESTRIAL_SUPPLIED_MODE

MODE = "cooperative_project_next_star_transition"
WORKFLOWS = {
    "no_planet_visible_workflow_readback": ("verified_no_planet", project_evidence),
    "positive_planet_visible_workflow_readback": ("verified_positive_planet", project_positive_evidence),
    "terrestrial_visible_workflow_readback": ("verified_terrestrial", project_terrestrial_evidence),
    POSITIVE_SUPPLIED_MODE: ("verified_positive_planet", project_positive_evidence),
    TERRESTRIAL_SUPPLIED_MODE: ("verified_terrestrial", project_terrestrial_evidence),
}


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("project_next_star_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(book, value):
    path = Path(value)
    return book.path(path if path.is_absolute() else book.history / path)


def _completed_owner(book, journal, directory, expected):
    """Reuse strict import planners offline, requiring an exactly idempotent plan."""
    _require(isinstance(journal, ProjectJournal), "canonical_journal_required")
    journal_path = book.path(journal.path)
    _require(
        journal_path.parent == book.history
        and {book.path(p) for p in book.history.glob("project-progress-*.jsonl")} == {journal_path},
        "ambiguous_canonical_attempt",
    )
    progress = journal.load()
    state = progress.reduce()
    book.read(journal_path)
    _require(journal.load() == progress, "journal_changed")
    _require(
        not state.pending and not any(r.write_kind == "submission" for r in state.reservations),
        "uncertain_or_submitted_attempt",
    )
    _require(
        1 <= len(state.stars) < 30
        and {s.name.casefold() for s in state.stars.values()} == {n.casefold() for n in expected},
        "collection_limit_or_names_mismatch",
    )
    directory = book.clean(_path(book, directory))
    owner = book.json(directory / "report.json")
    star = owner.get("star")
    matches = [
        s for s in state.stars.values() if isinstance(star, str) and s.name.casefold() == star.casefold()
    ]
    _require(len(matches) == 1 and matches[0].task_completed, "previous_task_not_verified")
    previous = matches[0]
    last_task = next(
        (r.payload for r in reversed(progress.records) if isinstance(r.payload, TaskReceipt)), None
    )
    _require(
        last_task == previous.completion
        and (state.active is None or state.active.star_id in {None, previous.id}),
        "previous_owner_is_not_latest_active_task",
    )
    _require(
        owner.get("mode") == "bounded_single_star_project_runtime"
        and owner.get("status") == "completed"
        and owner.get("finished") is True
        and owner.get("task_completed") is True
        and owner.get("project_completed") is False
        and owner.get("failure_reason") is None
        and owner.get("event_forwarding_failed") is False
        and owner.get("project_progress") == state.report(),
        "previous_owner_unresolved_or_stale",
    )
    events = [
        RuntimeEvent.model_validate_json(line) for line in book.read(directory / "events.jsonl").splitlines()
    ]
    _require(
        events
        and events[0].run_id is not None
        and all(
            e.sequence == i and e.run_id == events[0].run_id and e.event != "error"
            for i, e in enumerate(events)
        )
        and events[-1].event == "episode_summary"
        and events[-1].payload == {**owner, "completed": True},
        "previous_owner_event_mismatch",
    )
    binding_path = book.history / (
        "project-evidence-" + _sha((journal_path.name + previous.id).encode()) + ".json"
    )
    binding = book.json(binding_path)
    _require(
        all(binding.get(k) == v for k, v in progress.header().items())
        and binding.get("star_id") == previous.id,
        "previous_binding_identity_changed",
    )
    inventory_dir, workflow_dir = [_path(book, binding[key]) for key in ("inventory_path", "workflow_path")]
    sources = _Evidence(book.history)
    inventory_source = load_inventory_source(sources, inventory_dir)
    inventory, rows = inventory_source.receipt, inventory_source.rows
    inventory_binding = inventory_source.binding_fields()
    _require(
        all(binding.get(key) == value for key, value in inventory_binding.items()),
        "previous_inventory_binding_changed",
    )
    raw = sources.read(workflow_dir / "confirmed.json")
    mode = json.loads(raw).get("mode")
    _require(mode in WORKFLOWS and owner.get("phase") == WORKFLOWS[mode][0], "unsupported_previous_workflow")
    helper = WORKFLOWS[mode][1]
    workflow, directories = helper._workflow(sources, workflow_dir, rows)
    workflow_sha, inventory_sha = _sha(raw), _sha(sources.read(inventory_dir / "confirmed.json"))
    _require(
        workflow_sha == binding.get("workflow_sha256") == previous.completion.source_sha256
        and inventory_sha == binding.get("inventory_sha256")
        and {r["name"].casefold() for r in rows} == {n.casefold() for n in expected},
        "previous_sources_changed",
    )
    planned, star_id, names = helper._plan(
        progress, inventory, workflow, workflow_sha, inventory_sha, directories, sources
    )
    _require(
        planned == progress and star_id == previous.id and not names, "previous_workflow_not_exactly_imported"
    )
    imported = book.json(directory / "workflow-import.json")
    _require(
        imported.get("star_id") == previous.id
        and imported.get("workflow_sha256") == workflow_sha
        and imported.get("inventory_sha256") == inventory_sha
        and all(imported.get(key) == value for key, value in inventory_binding.items())
        and imported.get("progress") == state.report(),
        "previous_owner_import_mismatch",
    )
    for name, checksum in sources.hashes.items():
        _require(_sha(book.read(book.history / name)) == checksum, "previous_source_changed")
    # The strict workflow reader can close a completed readback tree. Retain
    # membership/uncertainty guards, not only its already-consumed leaf hashes,
    # for all later transition checks.
    for source_directory in sources.clean_directories:
        book.clean(source_directory)
    for source_directory, tree in sources.closed_trees.items():
        _require(
            source_directory not in book.closed_trees or book.closed_trees[source_directory] == tree,
            "previous_source_tree_changed",
        )
        book.closed_trees[source_directory] = tree
    for name, checksum in binding.get("validated_artifact_sha256", {}).items():
        _require(_sha(book.read(_path(book, name))) == checksum, "previous_binding_source_changed")
    book.unchanged()
    return state, previous, inventory_source.anchor


def _fresh(book, entry):
    _require(
        isinstance(entry, dict) and set(entry) in ({"path", "sha256"}, {"path", "sha256", "starfield_path"}),
        "invalid_visited_receipt",
    )
    path = _path(book, entry["path"])
    directory = book.clean(path.parent)
    _require(
        path.name == "confirmed.json" and _sha(book.read(path)) == entry["sha256"],
        "fresh_receipt_hash_changed",
    )
    receipt = book.json(path)
    _require(
        receipt.get("stellar_observations_verified") is True
        and receipt.get("fresh_blank_numeric_answers_verified") is True
        and receipt.get("class_selection_verified") is False
        and receipt.get("collection_count_verified") is False
        and receipt.get("task_completed") is False
        and receipt.get("action_source") == "deterministic_navigation"
        and type(receipt.get("answer_writes")) is int
        and receipt["answer_writes"] == 0
        and receipt.get("painted_stellar_class") in {None, *CLASSES}
        and "failed_run" not in receipt,
        "unsupported_fresh_receipt",
    )
    capture = book.capture(directory / "stellar")
    mapping = map_stellar_capture(capture, capture_sha256=screen_identity(capture))
    _require(
        isinstance(receipt.get("star"), str)
        and mapping["star_name"].casefold() == receipt["star"].casefold()
        and all(
            field.get("value_known") is True and field.get("current_value") == ""
            for field in mapping["observation"]["values"]["browser_field_map"].values()
        ),
        "fresh_capture_not_blank",
    )
    point = receipt.get("selected_point")
    _require(
        isinstance(point, dict)
        and set(point) == {"x", "y", "width", "height"}
        and all(type(v) in {int, float} and math.isfinite(v) for v in point.values())
        and 400 <= point["width"] <= 2400
        and 300 <= point["height"] <= 1800
        and 0 <= point["x"] < point["width"]
        and 0 <= point["y"] < point["height"],
        "invalid_visited_point",
    )
    scope = book.json(directory / "scope.json")
    before = book.capture(directory / "before")
    initial = scope.get("mode") == "read_only_initial_setup_handoff"
    if initial:
        _require(
            scope.get("selection_source") == "completed_initial_setup_rendered_starfield"
            and scope.get("credentials_recorded") is False
            and type(scope.get("browser_actions")) is int
            and scope["browser_actions"] == 0
            and scope.get("automatic_retry") is False
            and scope.get("ready_screen_sha256") == screen_identity(before) == screen_identity(capture),
            "initial_starfield_source_required",
        )
        if "selection_schema_version" in scope:
            source = book.path(directory / INITIAL_STARFIELD_IMAGE)
            _require(
                "starfield_path" not in entry or _path(book, entry["starfield_path"]) == source,
                "versioned_initial_starfield_path_mismatch",
            )
        else:
            _require("starfield_path" in entry, "initial_starfield_source_required")
            source = _path(book, entry["starfield_path"])
        png = book.read(source)
        validate_initial_selection(scope, png, point)
        selected = point
    else:
        _require(
            "starfield_path" not in entry
            and scope.get("action_source") == "deterministic_navigation"
            and scope.get("selection_basis") == "rendered_dot_geometry_only_not_scientific_class"
            and type(scope.get("max_star_clicks")) is int
            and scope["max_star_clicks"] == 1
            and type(scope.get("max_view_clicks")) is int
            and scope["max_view_clicks"] == 1
            and type(scope.get("answer_writes")) is int
            and scope["answer_writes"] == 0
            and type(scope.get("max_seconds")) is int
            and scope["max_seconds"] == NextStarPicker.MAX_SECONDS
            and scope.get("task_completed") is False
            and scope.get("collection_count_verified") is False
            and project_view(before) == {"surface": "starfield", "section": None, "star": None},
            "unsupported_picker_scope",
        )
        selected = visible_star_point(
            book.read(directory / "setup-starfield.png"),
            excluded=scope["excluded_points"],
            anchor=scope["starfield_anchor"],
        )
    _require(selected == point, "selected_point_source_changed")
    return receipt, scope, capture


class _Cancelled(Exception):
    pass


class ProjectNextStarSteps:
    """One navigation, separate picker construction/calls, then offline handoff."""

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        journal,
        completed_owner_dir,
        expected_stars,
        visited_receipts,
        anchor=(0.4, 0.55),
        max_seconds=180,
        max_advances=128,
        emit=lambda _kind, _payload: None,
        cancelled=lambda: False,
        _clock=time.monotonic,
    ):
        self.expected = _names(expected_stars)
        _require(
            type(max_seconds) in {int, float}
            and math.isfinite(max_seconds)
            and 1 <= max_seconds <= 180
            and type(max_advances) is int
            and 4 <= max_advances <= 256,
            "invalid_fixed_budget",
        )
        _require(callable(emit) and callable(cancelled) and callable(_clock), "invalid_callback")
        _require(
            isinstance(anchor, (list, tuple))
            and len(anchor) == 2
            and all(type(v) in {int, float} and math.isfinite(v) and 0.15 <= v <= 0.85 for v in anchor),
            "invalid_anchor",
        )
        self.book = _Evidence(run_history)
        self.journal = journal
        self.progress, self.previous, self._previous_capture = _completed_owner(
            self.book, journal, completed_owner_dir, self.expected
        )
        _require(
            isinstance(visited_receipts, (list, tuple)) and len(visited_receipts) == len(self.expected),
            "one_point_per_collected_star_required",
        )
        visited = [_fresh(self.book, entry) for entry in visited_receipts]
        by_name = {receipt["star"].casefold(): receipt["selected_point"] for receipt, _, _ in visited}
        _require(
            len(by_name) == len(visited) and set(by_name) == {name.casefold() for name in self.expected},
            "visited_names_mismatch",
        )
        for receipt, scope, _ in visited:
            if scope.get("mode") == "read_only_initial_setup_handoff":
                continue
            names = scope.get("visited_stars")
            _require(
                isinstance(names, list)
                and all(isinstance(n, str) and n.strip() for n in names)
                and len(names) == len({n.casefold() for n in names})
                and receipt["star"].casefold() not in {n.casefold() for n in names}
                and {n.casefold() for n in names} <= by_name.keys()
                and scope.get("excluded_points") == [by_name[n.casefold()] for n in names],
                "visited_exclusion_source_mismatch",
            )
        points = [by_name[name.casefold()] for name in self.expected]
        _require(
            len({(p["x"], p["y"]) for p in points}) == len(points)
            and len({(p["width"], p["height"]) for p in points}) == 1,
            "duplicate_point_or_changed_viewport",
        )
        self.output = _path(self.book, output)
        _require(self.output != self.book.history and not self.output.exists(), "output_already_exists")
        self.claim_path = (
            self.book.history
            / "project-next-star-reservations"
            / (_sha((journal.path.name + ":" + str(self.progress.revision)).encode()) + ".json")
        )
        _require(not self.claim_path.exists(), "transition_already_reserved")
        self.page, self.config = page, config.model_copy(deep=True)
        rules = [r for r in self.config.frames if r.url == SIMULATION_URL]
        _require(len(rules) == 1 and rules[0].count == 1, "unsupported_boundary")
        rules[0].required_text = []
        self._config = self.config.model_dump(mode="json")
        self.points, self.anchor = deepcopy(points), tuple(anchor)
        self.max_seconds, self.max_advances = max_seconds, max_advances
        self._clock, self._started = _clock, _clock()
        self._callback, self._cancelled = emit, cancelled
        self.picker = self.receipt = self.report = None
        self.phase, self.status, self.failure = "to_starfield", "ready", None
        self.finished = self._busy = self._finalizing = self._forward_failed = False
        self._claimed = self._dot_proposed = self._view_proposed = False
        self._listeners = False
        self._dialogs, self._popups, self._frames = [], [], None
        self.advances = self.navigation_attempts = self.navigation_clicks = self.picker_advances = (
            self._sequence
        ) = 0
        self._relay = ProjectEventRelay(self._receive)
        self._fresh_sha = None
        self.scope = {
            "mode": MODE,
            "previous_star": self.previous.name,
            "project_id": self.progress.project_id,
            "attempt_id": self.progress.attempt_id,
            "data_revision": self.progress.revision,
            "expected_stars": list(self.expected),
            "starfield_anchor": list(self.anchor),
            "visited_points": deepcopy(self.points),
            "prior_unresolved_stars": [s.name for s in self.progress.stars.values() if not s.task_completed],
            "max_seconds": max_seconds,
            "max_advances": max_advances,
            "max_star_clicks": 1,
            "max_view_clicks": 1,
            "max_starfield_navigation_clicks": 1,
            "max_stellar_tab_restore_clicks": 1,
            "picker_max_seconds": NextStarPicker.MAX_SECONDS,
            "selection_source": "existing_rendered_dot_picker_not_scientific_class",
            "read_only_picker_hover_possible": True,
            "pause_counts_toward_deadline": True,
            "cancellation": "between_bounded_calls_and_picker_event_boundaries",
            "automatic_retry": False,
            "answer_writes": 0,
            "class_writes": 0,
            "journal_writes": 0,
            "save_clicks": 0,
            "assessment_clicks": 0,
            "score_transfer_clicks": 0,
            "submission_clicks": 0,
            "collection_count_verified": False,
            "class_selection_verified": False,
            "task_completed": False,
            "project_completed": False,
            "scientific_verified": False,
        }
        self.book.unchanged()
        self.output.mkdir(parents=True, exist_ok=False)
        persist_json(self.output / "scope.json", self.scope)
        self.book.read(self.output / "scope.json")
        self._stream = (self.output / "events.jsonl").open("x")
        self._run_id = "next-star-" + _sha(str(self.output.relative_to(self.book.history)).encode())[:16]
        try:
            self._emit("hello", {"protocol_version": 1, **self.scope})
            self._check()
        except _Cancelled:
            pass
        except Exception as exc:  # noqa: BLE001 - sanitize source/callback failures
            self._stop(self._reason(exc))

    @staticmethod
    def _reason(exc):
        return (
            str(exc)
            if isinstance(exc, (BrowserSafetyStop, SetupStop))
            else "project_next_star_operation_failed"
        )

    def _dialog(self, _):
        self._dialogs.append(True)

    def _popup(self, _):
        self._popups.append(True)

    def _check(self):
        if self.finished:
            raise _Cancelled
        if self._cancelled():
            self._stop("operator_aborted", aborted=True)
            raise _Cancelled
        _require(
            not self._forward_failed and self._clock() - self._started < self.max_seconds,
            "event_failure_or_time_limit",
        )
        self._frozen()

    def _frozen(self):
        _require(
            self.scope == self.book.json(self.output / "scope.json")
            and list(self.expected) == self.scope["expected_stars"]
            and list(self.anchor) == self.scope["starfield_anchor"]
            and self.points == self.scope["visited_points"]
            and self.max_seconds == self.scope["max_seconds"]
            and self.max_advances == self.scope["max_advances"],
            "scope_changed",
        )
        _require(self.config.model_dump(mode="json") == self._config, "config_changed")
        _require(
            {self.book.path(p) for p in self.book.history.glob("project-progress-*.jsonl")}
            == {self.book.path(self.journal.path)},
            "ambiguous_canonical_attempt",
        )
        self.book.unchanged()
        if self._listeners:
            _require(
                not self._dialogs
                and not self._popups
                and self.page.frames == self._frames
                and len(self.page.context.pages) == 1
                and self.config.allows(self.page.url),
                "context_changed",
            )

    def _emit(self, kind, payload):
        event = RuntimeEvent(event=kind, payload=payload, sequence=self._sequence, run_id=self._run_id)
        line = json.dumps(event.model_dump(), allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            try:
                self._callback(kind, json.loads(line)["payload"])
            except Exception:  # noqa: BLE001 - never expose callback details
                self._forward_failed = True
                if not self._finalizing:
                    raise BrowserSafetyStop("project_next_star_event_forwarding_failed") from None

    def _receive(self, kind, payload):
        self._check()
        stage = payload.get("component_state", {}).get("setup_stage")
        self._dot_proposed |= stage == "selecting_visible_star"
        self._view_proposed |= stage == "opening_star_data"
        self._emit(kind, {**payload, **self.state()})
        self._check()

    def state(self):
        return {
            **deepcopy(self.scope),
            "phase": self.phase,
            "status": self.status,
            "finished": self.finished,
            "advances": self.advances,
            "navigation_attempts": self.navigation_attempts,
            "navigation_clicks": self.navigation_clicks,
            "transition_reserved": self._claimed,
            "transition_reservation": str(self.claim_path.relative_to(self.book.history))
            if self._claimed
            else None,
            "picker_advances": self.picker_advances,
            "star_click_may_have_occurred": self._dot_proposed
            or bool(self.picker and self.picker.star_clicked),
            "view_click_may_have_occurred": self._view_proposed
            or bool(self.picker and self.picker.view_clicked),
            "stellar_tab_restore_clicked": bool(self.picker and self.picker.stellar_tab_clicked),
            "failure_reason": self.failure,
            "event_forwarding_failed": self._forward_failed,
            "fresh_star_dir": str((self.output / "picker").relative_to(self.book.history))
            if self.receipt
            else None,
            "fresh_star_sha256": self._fresh_sha,
            "star": self.receipt["star"] if self.receipt else None,
            "fresh_blank_numeric_answers_verified": self.receipt is not None,
            "project_progress": self.progress.report(),
            "source_sha256": deepcopy(self.book.hashes),
        }

    def _same_screen(self, report):
        view = project_view(self._previous_capture)
        _require(
            project_view(report) == view
            and screen_identity(navigation_status_projection(report, view))
            == screen_identity(navigation_status_projection(self._previous_capture, view)),
            "current_screen_changed",
        )

    def _guard(self):
        if not self._listeners:
            self._frames = self.page.frames.copy()
            self.page.on("dialog", self._dialog)
            self.page.context.on("page", self._popup)
            self._listeners = True
        self._check()
        report = inspect_page(self.page, self.config)
        _require(report.get("ignored_frame_urls") == [], "unknown_visible_frame")
        self._same_screen(report)
        self._check()

    def _claim(self):
        self.book.path(self.claim_path)
        self.claim_path.parent.mkdir(exist_ok=True)
        persist_json(
            self.claim_path,
            {
                "mode": MODE,
                "output": str(self.output.relative_to(self.book.history)),
                "journal_sha256": self.book.hashes[
                    str(self.book.path(self.journal.path).relative_to(self.book.history))
                ],
                "data_revision": self.progress.revision,
                "automatic_retry": False,
            },
        )
        fd = os.open(self.claim_path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.book.read(self.claim_path)
        self._claimed = True

    def _navigate(self):
        self._guard()
        self._claim()
        self._check()
        self.navigation_attempts += 1
        directory = self.output / "to-starfield"
        returned = navigate_project(self.page, self.config, directory, "starfield")
        self._check()
        self.book.clean(directory)
        receipt, intent = (
            self.book.json(directory / "confirmed.json"),
            self.book.json(directory / "reserved.json"),
        )
        self._same_screen(self.book.capture(directory / "before"))
        self._same_screen(self.book.capture(directory / "pre-click"))
        after = self.book.capture(directory / "after")
        _require(
            receipt == returned
            and all(receipt.get(k) == v for k, v in intent.items())
            and intent.get("mode") == "bounded_project_navigation"
            and type(intent.get("max_clicks")) is int
            and intent["max_clicks"] == 1
            and intent.get("automatic_retry") is False
            and all(
                type(receipt.get(k)) is int and receipt[k] == 0
                for k in (
                    "answer_writes",
                    "collection_clicks",
                    "save_clicks",
                    "assessment_clicks",
                    "submission_clicks",
                )
            )
            and receipt.get("config_mutated") is False
            and receipt.get("destination") == "starfield"
            and receipt.get("from") == project_view(self._previous_capture)
            and receipt.get("to")
            == project_view(after)
            == {"surface": "starfield", "section": None, "star": None}
            and receipt.get("destination_verified") is True
            and receipt.get("navigation_clicks") == 1
            and receipt.get("task_completed") is False,
            "navigation_not_verified",
        )
        self.navigation_clicks += 1
        self._previous_capture, self.phase = after, "picker_initializing"

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy:
            self._stop("project_next_star_reentrant_advance")
            return self.state()
        self._busy = True
        try:
            self._check()
            _require(self.advances < self.max_advances, "advance_limit")
            self.advances += 1
            if self.phase == "to_starfield":
                self._navigate()
            elif self.phase == "picker_initializing":
                self._guard()
                child = NextStarPicker(
                    self.page,
                    self.config,
                    self.output / "picker",
                    visited_stars=list(self.expected),
                    excluded_points=deepcopy(self.points),
                    anchor=self.anchor,
                    emit=self._relay.bind("next_star.picker"),
                )
                self.picker = child
                if self.finished:
                    child.close()
                    return self.state()
                self._same_screen(child.initial)
                self._check()
                self.phase = "picking"
            elif self.phase == "picking":
                self.picker_advances += 1
                result = self.picker.advance()
                self._check()
                _require(result in {"waiting", "stellar"}, "unknown_picker_result")
                if result == "stellar":
                    self.phase = "validating_receipt"
            elif self.phase == "validating_receipt":
                _require(
                    self.picker.closed is True
                    and self.picker.stage == "stellar_screen_ready"
                    and self.picker.star_clicked is True
                    and self.picker.view_clicked is True,
                    "picker_not_completed",
                )
                path = self.output / "picker/confirmed.json"
                checksum = _sha(self.book.read(path))
                receipt, scope, capture = _fresh(self.book, {"path": str(path), "sha256": checksum})
                _require(
                    receipt == self.picker.receipt
                    and receipt["star"].casefold() not in {n.casefold() for n in self.expected}
                    and scope.get("visited_stars") == list(self.expected)
                    and scope.get("excluded_points") == self.points
                    and scope.get("starfield_anchor") == list(self.anchor),
                    "fresh_handoff_changed",
                )
                current = inspect_page(self.page, self.config)
                _require(
                    current.get("ignored_frame_urls") == []
                    and screen_identity(current) == screen_identity(capture),
                    "fresh_handoff_screen_changed",
                )
                save_probe(current, self.output / "handoff")
                self.book.capture(self.output / "handoff")
                self._check()
                self.receipt, self._fresh_sha = receipt, checksum
                self.phase, self.status = "fresh_star_handoff", "completed"
                self._finish()
            else:
                raise BrowserSafetyStop("project_next_star_unknown_phase")
            if not self.finished:
                self._emit("state", self.state())
                self._check()
        except _Cancelled:
            pass
        except (KeyboardInterrupt, SystemExit):
            self._stop("operator_aborted", aborted=True)
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize browser/source failures
            if not self.finished:
                self._stop(self._reason(exc))
        finally:
            self._busy = False
        return self.state()

    def _finish(self):
        self._finalizing = True
        try:
            self.finished = True
            summary = self.state()
            self._emit("episode_summary", summary)
            if self.status == "completed":
                try:
                    self._frozen()
                    _require(
                        self._clock() - self._started < self.max_seconds,
                        "event_failure_or_time_limit",
                    )
                    if self._cancelled():
                        self._stop("operator_aborted", aborted=True)
                except Exception as exc:  # noqa: BLE001 - no successful mutated-source handoff
                    self._stop(self._reason(exc))
            if self._forward_failed:
                self.status, self.phase, self.failure = (
                    "stopped",
                    "stopped",
                    "project_next_star_event_forwarding_failed",
                )
                self.receipt = self._fresh_sha = None
            if self.state() != summary:
                self._emit("state", self.state())
            self.report = self.state()
            persist_json(
                self.output / ("confirmed.json" if self.status == "completed" else "stopped.json"),
                self.report,
            )
            persist_json(self.output / "report.json", self.report)
        finally:
            self._relay.retire()
            if self.picker:
                self.picker.close()
            if self._listeners:
                self.page.remove_listener("dialog", self._dialog)
                self.page.context.remove_listener("page", self._popup)
            self._stream.close()
            self._finalizing = False

    def _stop(self, reason, *, aborted=False):
        if self.finished and not self._finalizing:
            return
        self.failure, self.status = reason, "aborted" if aborted else "stopped"
        self.phase = self.status
        self.receipt = self._fresh_sha = None
        if not self._finalizing:
            self._finish()

    def abort(self):
        self._stop("operator_aborted", aborted=True)
        return self.state()

    close = abort
