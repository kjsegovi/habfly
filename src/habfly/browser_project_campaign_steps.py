"""Fresh, bounded multi-star scheduling; no browser ownership or reference oracle.

Each scientific workflow remains an unchanged BrowserProjectSteps instance.
Only its strict importer establishes a task receipt. Collection transitions use
the existing one-shot picker. Reaching the requested count is an assessment
handoff, not course/scientific success, score transfer, or submission.
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
from .browser_project_navigation import _required, project_view
from .browser_project_next_star_steps import MODE as TRANSITION_MODE
from .browser_project_next_star_steps import WORKFLOWS, ProjectNextStarSteps, _fresh
from .browser_project_steps import _MODEL_KEYS, _REQUIRED_MODELS, BrowserProjectSteps
from .browser_star_preflight import validate_star_class_source
from .browser_stellar import SIMULATION_URL
from .contracts import RuntimeCommand, RuntimeEvent
from .data.graphs import ARRAY_NAMES
from .habitability_knowledge import DEFAULT_HABITABILITY_PACK
from .knowledge import DEFAULT_PACK
from .model import graph_fingerprint
from .planet_knowledge import DEFAULT_PLANET_PACK
from .project_events import ProjectEventRelay
from .project_inventory_source import load_inventory_source
from .project_progress import Collected, ProjectJournal, StageRecorded, TaskReceipt
from .training.planet_period import PILOT_COUNTS

MODE = "fresh_cooperative_project_campaign"
HANDOFFS = {"awaiting_class_source", "awaiting_planet_class", "awaiting_gases", "awaiting_habitability"}
OWNER_PHASES = {
    "ready",
    "active",
    "inventory_initializing",
    "inventory_active",
    "inventory_import",
    "selecting_planet_class",
    "positive_initializing",
    "positive_active",
    "terrestrial_initializing",
    "terrestrial_active",
}


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("project_campaign_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _reason(exc):
    message = str(exc)
    return (
        message
        if isinstance(exc, BrowserSafetyStop) and re.fullmatch(r"[a-z][a-z0-9_:.-]{0,150}", message)
        else "project_campaign_operation_failed"
    )


def _file_hash(path):
    path = Path(path).absolute()
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_model_source")
    _require(path.is_file() and path.stat().st_size <= 512_000_000, "missing_or_oversized_model_source")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _supplied_gate_status(options, terrestrial):
    for directory in (
        options.get("planet_supplied_evaluation"),
        None if terrestrial is None else terrestrial.get("supplied_evaluation"),
    ):
        if directory is not None:
            directory = Path(directory)
            _require(
                not any(p.is_symlink() for p in (directory, *directory.parents))
                and not (directory / "stopped.json").exists(),
                "supplied_gate_stopped",
            )


def _model_dependencies(options, terrestrial):
    """Exact file dependencies of the existing inference loaders, no tree scan.

    Pilot training records below are hashed only because the existing loader
    verifies their provenance. They are not parsed as decisions or sent to the
    policy. The explicit supplied-input option adds only the exact recorded
    transfer gate proof dependencies; no new evaluation/training is performed.
    """
    dataset, color, checkpoint, graph = [
        Path(options[k]).absolute() for k in ("dataset", "color_experiment", "checkpoint", "graph_path")
    ]
    paths = {
        checkpoint,
        checkpoint.with_suffix(checkpoint.suffix + ".json"),
        DEFAULT_PACK,
        Path(__file__).parent / "packs/stellar_color.json",
    }
    # load_checkpoint checks this narrow package for calibration provenance.
    paths.update((Path(__file__).parent / "model").glob("*.py"))
    if options.get("allow_baseline_edge_reference", False):
        paths.add(Path(__file__).parent / "planet_window_baseline_edge.py")
    paths.update(
        dataset / name for name in ("manifest.json", "report.json", "final/report.json", "manual.json")
    )
    paths.update(
        color / name
        for name in (
            "report.json",
            "dataset-manifest.json",
            "training/checkpoint.pt",
            "training/checkpoint.pt.json",
            "final/report.json",
            "train.json",
            "calibration.json",
            "development.json",
            "demonstrations.json",
        )
    )
    _file_hash(color / "report.json")
    report = json.loads((color / "report.json").read_bytes())
    content = report["content"]
    if any(content.get(key) for key in ("readout_ordering", "selected_input", "boundary_curriculum")):
        paths.add(color / "regression.json")
    if any(
        content.get(key)
        for key in ("readout_ordering", "selected_input", "boundary_curriculum", "stabilization")
    ):
        history = report["epoch_history"]
        _require(isinstance(history, list) and 1 <= len(history) <= 101, "unsupported_color_epoch_history")
        for index, entry in enumerate(history):
            _require(
                type(entry.get("epoch")) is int and entry["epoch"] == index, "unsupported_color_epoch_history"
            )
            paths.add(color / f"epochs/epoch-{index:03d}/training/checkpoint.pt")
    paths.update({graph / "graph_manifest.json", graph / "metadata.json"})
    _file_hash(graph / "graph_manifest.json")
    manifest = json.loads((graph / "graph_manifest.json").read_bytes())
    _require(
        manifest.get("num_nodes") == 2000 and set(manifest["arrays"]) == set(ARRAY_NAMES),
        "unsupported_graph_dependencies",
    )
    for name, metadata in manifest["arrays"].items():
        _require(metadata["file"] == f"{name}.npy", "unsupported_graph_array_path")
        paths.add(graph / metadata["file"])
    if options.get("planet_pilot") is not None:
        pilot = Path(options["planet_pilot"])
        paths.update(
            {
                pilot / "training/checkpoint.pt",
                pilot / "training/checkpoint.pt.json",
                Path(options["planet_final_evaluation"]) / "report.json",
                DEFAULT_PLANET_PACK,
            }
        )
    if terrestrial is not None:
        pilot = Path(terrestrial["pilot"])
        paths.update(
            {
                pilot / "report.json",
                pilot / "training/checkpoint.pt",
                pilot / "training/checkpoint.pt.json",
                Path(terrestrial["final_evaluation"]) / "report.json",
                DEFAULT_HABITABILITY_PACK,
            }
        )
        for split in PILOT_COUNTS:
            paths.add(pilot / f"private-{split}-cases.json")
            if split != "gate":
                paths.add(pilot / f"expert-{split}/episodes.json")
        paths.update(pilot / f"learned-{split}/summaries.json" for split in ("train", "development"))
    extra = {}
    supplied_planet = options.get("planet_supplied_evaluation")
    supplied_temperature = None if terrestrial is None else terrestrial.get("supplied_evaluation")
    if supplied_planet is not None or supplied_temperature is not None:
        _supplied_gate_status(options, terrestrial)
        _require(
            supplied_planet is not None
            and supplied_temperature is not None
            and options.get("planet_pilot") is not None
            and terrestrial is not None,
            "supplied_gate_pair_required",
        )
        from .autonomous_validation import _supplied_transfer_sources
        from .data.graphs import load_graph

        fingerprint = graph_fingerprint(load_graph(graph))
        for task, pilot, directory in (
            ("planet", options["planet_pilot"], supplied_planet),
            ("temperature", terrestrial["pilot"], supplied_temperature),
        ):
            _, pins = _supplied_transfer_sources(task, pilot, directory, fingerprint)
            for path, digest in pins.items():
                _require(path not in extra or extra[path] == digest, "supplied_source_changed")
                _require(_file_hash(path) == digest, "supplied_source_changed")
                extra[path] = digest
        paths.update(
            Path(__file__).parent / name
            for name in (
                "browser_supplied_steps.py",
                "browser_supplied_provenance.py",
                "browser_supplied_tool_envs.py",
                "browser_stellar_sources.py",
                "planet_supplied_stellar_source.py",
                "supplied_browser_modes.py",
                "autonomous_validation.py",
            )
        )
    result = {str(path.absolute()): _file_hash(path) for path in sorted(paths)}
    _require(
        all(path not in result or result[path] == digest for path, digest in extra.items()),
        "supplied_source_changed",
    )
    return {**result, **extra}


def _settings(options, reference, settlement, terrestrial):
    def encode(value):
        if isinstance(value, Path):
            return str(value.absolute())
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        return value

    habitat = (
        None
        if terrestrial is None
        else {
            **{k: encode(v) for k, v in terrestrial.items() if k != "graph"},
            "graph_hash": graph_fingerprint(terrestrial["graph"]),
        }
    )
    value = {
        "model_options": encode(options),
        "reference_planet_continuation": reference,
        "positive_save_settle_seconds": settlement,
        "terrestrial_options": habitat,
    }
    json.dumps(value, allow_nan=False)
    return value


def _verified_owner(book, journal, directory, expected, previous):
    """Revalidate the exact owner/import/workflow chain, including final star 30."""
    progress = journal.load()
    state = progress.reduce()
    _require(not state.pending and not state.reservations and not state.receipts, "unexpected_project_write")
    _require(
        len(state.stars) == len(expected)
        and all(s.task_completed for s in state.stars.values())
        and {s.name.casefold() for s in state.stars.values()} == {n.casefold() for n in expected},
        "incomplete_or_unexpected_collection",
    )
    matches = [s for s in state.stars.values() if s.name.casefold() == previous.casefold()]
    _require(len(matches) == 1, "missing_current_star")
    star = matches[0]
    last = next((r.payload for r in reversed(progress.records) if isinstance(r.payload, TaskReceipt)), None)
    _require(
        last == star.completion and (state.active is None or state.active.star_id in {None, star.id}),
        "current_owner_not_latest_task",
    )
    directory = book.clean(directory)
    owner = book.json(directory / "report.json")
    _require(
        owner.get("mode") == "bounded_single_star_project_runtime"
        and owner.get("status") == "completed"
        and owner.get("finished") is True
        and owner.get("task_completed") is True
        and owner.get("project_completed") is False
        and owner.get("failure_reason") is None
        and owner.get("event_forwarding_failed") is False
        and owner.get("star", "").casefold() == previous.casefold()
        and owner.get("project_progress") == state.report(),
        "unverified_owner_report",
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
        "unverified_owner_events",
    )
    binding = book.json(
        book.history / ("project-evidence-" + _sha((journal.path.name + star.id).encode()) + ".json")
    )
    _require(
        all(binding.get(k) == v for k, v in progress.header().items()) and binding.get("star_id") == star.id,
        "binding_attempt_mismatch",
    )
    inventory_dir, workflow_dir = [
        book.path(book.history / binding[k]) for k in ("inventory_path", "workflow_path")
    ]
    inventory_source = load_inventory_source(book, inventory_dir)
    inventory, rows = inventory_source.receipt, inventory_source.rows
    inventory_binding = inventory_source.binding_fields()
    _require(
        all(binding.get(key) == value for key, value in inventory_binding.items()),
        "inventory_binding_changed",
    )
    workflow_path = workflow_dir / "confirmed.json"
    mode = book.json(workflow_path).get("mode")
    _require(mode in WORKFLOWS and owner.get("phase") == WORKFLOWS[mode][0], "unsupported_workflow")
    helper = WORKFLOWS[mode][1]
    workflow, directories = helper._workflow(book, workflow_dir, rows)
    workflow_sha, inventory_sha = (
        _sha(book.read(workflow_path)),
        _sha(book.read(inventory_dir / "confirmed.json")),
    )
    _require(
        workflow_sha == binding.get("workflow_sha256") == star.completion.source_sha256
        and inventory_sha == binding.get("inventory_sha256"),
        "binding_source_mismatch",
    )
    planned, star_id, names = helper._plan(
        progress, inventory, workflow, workflow_sha, inventory_sha, directories, book
    )
    _require(planned == progress and star_id == star.id and not names, "workflow_not_exactly_imported")
    imported = book.json(directory / "workflow-import.json")
    _require(
        imported.get("star_id") == star.id
        and imported.get("workflow_sha256") == workflow_sha
        and imported.get("inventory_sha256") == inventory_sha
        and all(imported.get(key) == value for key, value in inventory_binding.items())
        and imported.get("progress") == state.report(),
        "owner_import_mismatch",
    )
    for name, checksum in binding.get("validated_artifact_sha256", {}).items():
        _require(_sha(book.read(book.history / name)) == checksum, "binding_artifact_changed")
    book.unchanged()
    return (
        state,
        inventory_source.anchor,
        {
            "owner_dir": str(directory.relative_to(book.history)),
            "workflow_dir": str(workflow_dir.relative_to(book.history)),
            "workflow_sha256": workflow_sha,
            "inventory_dir": str(inventory_dir.relative_to(book.history)),
            "inventory_sha256": inventory_sha,
            **inventory_binding,
        },
    )


class _Cancelled(Exception):
    pass


class BrowserProjectCampaignSteps:
    """One browser remains owned by the caller; no resume of old campaigns.

    Finite whole-campaign deadlines include paused reference handoffs. An
    explicit thirty-star "uncapped" campaign retains all fixed child limits.
    Outer setup may create FreshStarClassSteps at current_star_context(). Its
    resulting source must be explicitly provided; no selection is synthesized.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        journal,
        initial_star_dir,
        initial_star_sha256,
        target_stars,
        max_seconds,
        model_options,
        reference_planet_continuation=False,
        positive_save_settle_seconds=20,
        terrestrial_options=None,
        emit=lambda *_: None,
        interval=0.2,
        cancelled=lambda: False,
        _clock=time.monotonic,
        _owner_factory=None,
        _transition_factory=None,
    ):
        _require(type(target_stars) is int and 2 <= target_stars <= 30, "explicit_multistar_target_required")
        _require(
            (type(max_seconds) in {int, float} and math.isfinite(max_seconds) and 1 <= max_seconds <= 10800)
            or (type(max_seconds) is str and max_seconds == "uncapped" and target_stars == 30),
            "invalid_fixed_campaign_deadline",
        )
        _require(
            type(interval) in {int, float} and math.isfinite(interval) and 0 <= interval <= 10,
            "invalid_interval",
        )
        _require(
            isinstance(model_options, dict)
            and _REQUIRED_MODELS <= model_options.keys()
            and not model_options.keys() - _MODEL_KEYS,
            "invalid_model_options",
        )
        _require(
            type(model_options.get("allow_baseline_edge_reference", False)) is bool,
            "invalid_baseline_edge_reference",
        )
        _require(
            type(reference_planet_continuation) is bool
            and (
                terrestrial_options is None
                or isinstance(terrestrial_options, dict)
                and reference_planet_continuation
            ),
            "invalid_reference_options",
        )
        _require(
            type(positive_save_settle_seconds) in {int, float}
            and math.isfinite(positive_save_settle_seconds)
            and 0.1 <= positive_save_settle_seconds <= 30,
            "invalid_save_budget",
        )
        _require(all(callable(c) for c in (emit, cancelled, _clock)), "invalid_callback")
        self.book = _Evidence(run_history)
        self.history = self.book.history
        self.journal = journal
        _require(isinstance(journal, ProjectJournal), "canonical_journal_required")
        self._sole_journal()
        self._progress = journal.load()
        _require(not self._progress.records, "fresh_empty_attempt_required")
        self._journal_raw = self.book.path(journal.path).read_bytes()
        self.output = self._owned(output)
        _require(self.output != self.history and not self.output.exists(), "output_exists")
        fresh = self._owned(initial_star_dir)
        entry = {"path": str(fresh / "confirmed.json"), "sha256": initial_star_sha256}
        receipt, scope, capture = _fresh(self.book, entry)
        _require(
            scope.get("mode") == "read_only_initial_setup_handoff"
            and scope.get("selection_schema_version") == 1,
            "new_initial_provenance_required",
        )
        self.page, self.config = page, config.model_copy(deep=True)
        self._boundary = self._boundary_settings()
        self._labels_from(capture, {"surface": "detail", "section": "stellar", "star": receipt["star"]})
        self.models = deepcopy(model_options)
        self.reference, self.settlement = reference_planet_continuation, positive_save_settle_seconds
        self.terrestrial = None if terrestrial_options is None else dict(terrestrial_options)
        self._settings = _settings(self.models, self.reference, self.settlement, self.terrestrial)
        self._model_hashes = _model_dependencies(self.models, self.terrestrial)
        _require(
            all(not Path(p).resolve().is_relative_to(self.output) for p in self._model_hashes),
            "model_output_overlap",
        )
        self._owner_factory, self._transition_factory = (
            _owner_factory or BrowserProjectSteps,
            _transition_factory or ProjectNextStarSteps,
        )
        self.target, self.max_seconds, self.interval = target_stars, max_seconds, interval
        self._clock, self._callback, self._cancelled = _clock, emit, cancelled
        self._started = self._last_tick = None
        self._busy = self._emitting = self._closing = self._finalizing = self._forward_failed = False
        self._child_failure = None
        self._child_abort_failed = False
        self._allow_import = False
        self._sequence = self.advances = 0
        self.owner = self.transition = self.report = self._owner_state = self._transition_state = None
        self.status, self.phase, self.failure = "idle", "not_started", None
        self._visited, self._completed = [entry], []
        self._current = self._context(1, receipt["star"], fresh, initial_star_sha256)
        self._relay = ProjectEventRelay(self._receive)
        self._run_id = "project-campaign-" + _sha(str(self.output.relative_to(self.history)).encode())[:16]
        self.scope = {
            "mode": MODE,
            "target_stars": target_stars,
            "max_seconds": max_seconds,
            "owner_limits": {"max_seconds": 1800, "max_advances": 512},
            "transition_limits": {"max_seconds": 180, "max_advances": 128},
            "project_id": journal.identity.project_id,
            "attempt_id": journal.identity.attempt_id,
            "pause_counts_toward_deadline": max_seconds != "uncapped",
            "cancellation": "between_bounded_calls_and_callbacks",
            "selection_source": "rendered_point_receipts_not_scientific_class",
            "reference_decisions": "explicit_source_handoffs_only",
            "automatic_retry": False,
            "automatic_deadline_increase": False,
            "optimizer_updates": 0,
            "browser_owned_by_caller": True,
            "pagination_supported": True,
            "live_paginated_inventory_supported": True,
            "thirty_star_live_launch_ready": False,
            "pending_thirty_star_gate": "live_terrestrial_workflow_and_compact_event_acceptance",
            "assessment_enabled": False,
            "submission_enabled": False,
            "task_completed": False,
            "project_completed": False,
            "scientific_verified": False,
        }
        self._frozen_scope = deepcopy(self.scope)
        self.output.mkdir(parents=True, exist_ok=False)
        self._persist("scope.json", self.scope)
        self._persist(
            "model-dependencies.json",
            {
                "mode": "explicit_loader_dependencies",
                "source_sha256": self._model_hashes,
                "settings": self._settings,
                "training_executed": False,
                "final_case_files_read": False,
            },
        )
        self._persist("star-001.json", self._current)
        self._stream = (self.output / "events.jsonl").open("x")

    def _owned(self, path):
        path = Path(path)
        return self.book.path(path if path.is_absolute() else self.history / path)

    def _sole_journal(self):
        path = self.book.path(self.journal.path)
        _require(
            path.parent == self.history
            and {self.book.path(p) for p in self.history.glob("project-progress-*.jsonl")} == {path},
            "ambiguous_attempt",
        )

    def _context(self, ordinal, star, fresh, checksum):
        base = self.output / f"stars/{ordinal:03d}"
        return {
            "ordinal": ordinal,
            "star": star,
            "fresh_dir": str(fresh.relative_to(self.history)),
            "fresh_sha256": checksum,
            "class_output": str((base / "reference-class").relative_to(self.history)),
            "owner_output": str((base / "owner").relative_to(self.history)),
        }

    def current_star_context(self):
        return deepcopy(self._current)

    def _persist(self, name, value):
        path = self._owned(self.output / name)
        path.parent.mkdir(parents=True, exist_ok=True)
        persist_json(path, value)
        self.book.read(path)

    def _boundary_settings(self):
        value = self.config.model_dump(mode="json")
        rules = [r for r in value["frames"] if r["url"] == SIMULATION_URL]
        _require(len(rules) == 1 and rules[0].get("count", 1) == 1, "unsupported_simulation_boundary")
        rules[0]["required_text"] = []
        return value

    def _labels_from(self, capture, expected):
        _require(
            capture.get("ignored_frame_urls") == [] and project_view(capture) == expected,
            "unexpected_current_view",
        )
        labels = _required(capture, expected)
        next(r for r in self.config.frames if r.url == SIMULATION_URL).required_text = labels
        _require(self._boundary_settings() == self._boundary, "navigation_boundary_changed")
        self._config = self.config.model_dump(mode="json")

    @property
    def finished(self):
        return self.status in {"handoff", "stopped", "aborted"}

    def state(self):
        return {
            **deepcopy(self.scope),
            "status": self.status,
            "phase": self.phase,
            "finished": self.finished,
            "star": self._current["star"],
            "current_star": self.current_star_context(),
            "verified_stars": len(self._completed),
            "target_workflows_verified": len(self._completed) == self.target
            and self.status == "handoff"
            and self.phase == "awaiting_assessment"
            and self.failure is None,
            "advances": self.advances,
            "failure_reason": self.failure,
            "event_forwarding_failed": self._forward_failed,
            "child_abort_failed": self._child_abort_failed,
            "project_owner": deepcopy(self._owner_state),
            **{
                key: deepcopy(self._owner_state[key])
                for key in ("save_outcome", "save_outcome_uncertain")
                if isinstance(self._owner_state, dict) and key in self._owner_state
            },
            "next_star": deepcopy(self._transition_state),
            "reference_measurements": deepcopy((self._owner_state or {}).get("reference_measurements"))
            if not self.failure
            else None,
            "project_progress": self._progress.reduce().report(),
            "journal_sha256": _sha(self._journal_raw),
            "source_sha256": deepcopy(self.book.hashes),
            "model_source_sha256": deepcopy(self._model_hashes),
            "artifact_paths": {
                "campaign_dir": str(self.output.relative_to(self.history)),
                **deepcopy(self._current),
            },
        }

    def _emit(self, kind, payload):
        event = RuntimeEvent(event=kind, sequence=self._sequence, run_id=self._run_id, payload=payload)
        line = json.dumps(event.model_dump(), allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            try:
                self._emitting = True
                self._callback(kind, json.loads(line)["payload"])
            except Exception:  # noqa: BLE001 - no callback details in artifacts
                self._forward_failed = True
                if not self._finalizing:
                    raise BrowserSafetyStop("project_campaign_event_forwarding_failed") from None
            finally:
                self._emitting = False

    def _receive(self, kind, payload):
        if self._closing:
            return
        if kind == "action_proposed" and self._child_failure:
            raise BrowserSafetyStop(self._child_failure)
        self._check(allow_child_error=True)
        if kind == "error" and self._child_failure is None:
            self._child_failure = _reason(BrowserSafetyStop(payload.get("message")))
        child = payload.get("component_state")
        if isinstance(child, dict):
            if payload["component"] == "campaign.owner":
                self._owner_state = deepcopy(child)
            elif payload["component"] == "campaign.next_star":
                self._transition_state = deepcopy(child)
        self._emit(kind, {**payload, **self.state()})
        # An ordinary component stop is data, not a broken event callback.
        # Drain its terminal diagnostics, then stop the enclosing advance.
        self._check(allow_child_error=True)

    def _journal_check(self):
        self._sole_journal()
        raw = self.journal.path.read_bytes()
        progress = self.journal.load()
        _require(self.journal.path.read_bytes() == raw, "concurrent_journal_change")
        if raw == self._journal_raw:
            return
        _require(self._allow_import and raw.startswith(self._journal_raw), "unexpected_journal_change")
        before, after = self._progress.reduce(), progress.reduce()
        _require(
            not after.pending and not after.reservations and not after.receipts, "unexpected_project_write"
        )
        _require(all(after.stars.get(k) == v for k, v in before.stars.items()), "prior_star_revised")
        for record in progress.records[len(self._progress.records) :]:
            item = record.payload
            _require(
                isinstance(item, (Collected, StageRecorded, TaskReceipt))
                and after.stars[item.star_id].name.casefold() == self._current["star"].casefold(),
                "foreign_import_record",
            )

    def _check(self, *, models=False, allow_child_error=False):
        if self.finished:
            raise _Cancelled
        if self._child_failure and not allow_child_error:
            raise BrowserSafetyStop(self._child_failure)
        if self._cancelled():
            self.abort()
            raise _Cancelled
        _require(
            not self._forward_failed and (self._started is None or self._within_campaign_deadline()),
            "event_failure_or_time_limit",
        )
        self._sources_check(models=models)

    def _within_campaign_deadline(self):
        return self.max_seconds == "uncapped" or self._clock() - self._started < self.max_seconds

    def _sources_check(self, *, models=False):
        _require(
            self.scope == self._frozen_scope
            and self.target == self.scope["target_stars"]
            and self.max_seconds == self.scope["max_seconds"]
            and type(self.max_seconds) is type(self.scope["max_seconds"])
            and type(self.max_seconds) is type(self._frozen_scope["max_seconds"])
            and self._current == self.book.json(self.output / f"star-{self._current['ordinal']:03d}.json"),
            "campaign_settings_changed",
        )
        _require(
            self.config.model_dump(mode="json") == self._config
            and self._boundary_settings() == self._boundary,
            "config_changed",
        )
        _require(
            _settings(self.models, self.reference, self.settlement, self.terrestrial) == self._settings,
            "model_options_changed",
        )
        self.book.unchanged()
        self._journal_check()
        if models:
            _supplied_gate_status(self.models, self.terrestrial)
            _require(
                all(_file_hash(Path(path)) == checksum for path, checksum in self._model_hashes.items()),
                "model_dependency_changed",
            )

    def _publish(self):
        if not self.finished:
            self._emit("state", self.state())
            self._check()
        return self.state()

    def _enter(self):
        if self._busy or self._emitting or self._finalizing:
            self._stop("project_campaign_reentrant_call")
            raise _Cancelled
        self._busy = True

    def start(self, *, paused=True):
        _require(self.status == "idle" and type(paused) is bool, "already_started_or_invalid_pause")
        self._started = self._clock()
        self.status, self.phase = "paused" if paused else "running", "initializing_star"
        try:
            self._check(models=True)
            self._emit("hello", {"protocol_version": 1, **self.scope})
            self._publish()
        except _Cancelled:
            pass
        except Exception as exc:  # noqa: BLE001 - sanitize protocol/source errors
            self._stop(_reason(exc))
        return self.state()

    def _construct_owner(self):
        child = self._owner_factory(
            self.page,
            self.config,
            self._owned(self._current["owner_output"]),
            run_history=self.history,
            journal=self.journal,
            star=self._current["star"],
            model_options=self.models,
            automatic_inventory=True,
            reference_planet_continuation=self.reference,
            positive_save_settle_seconds=self.settlement,
            terrestrial_options=self.terrestrial,
            emit=self._relay.bind("campaign.owner", star=self._current["star"]),
            interval=self.interval,
            max_seconds=1800,
            max_advances=512,
            _clock=self._clock,
        )
        self.owner = child
        if self.finished:
            child.abort()
            return
        child.start(paused=True)
        self._sync_owner()

    def _sync_owner(self):
        self._owner_state = deepcopy(self.owner.state())
        if self._child_failure:
            raise BrowserSafetyStop(self._child_failure)
        state = self._owner_state
        _require(state.get("star", "").casefold() == self._current["star"].casefold(), "owner_star_changed")
        if self.owner.finished:
            if state.get("status") == "completed" and state.get("task_completed") is True:
                self.phase = "verifying_star"
            elif "unsupported_pagination" in (state.get("failure_reason") or ""):
                self.status, self.phase, self.failure = (
                    "handoff",
                    "pagination_required",
                    state["failure_reason"],
                )
                self._finish()
            else:
                _require(state.get("task_completed") is False, "invalid_terminal_owner")
                self._stop(
                    "project_campaign_incomplete_star_handoff"
                    if state.get("status") == "handoff"
                    else _reason(
                        BrowserSafetyStop(state.get("failure_reason") or "project_campaign_owner_stopped")
                    )
                )
        else:
            _require(
                state.get("task_completed") is False and state.get("phase") in HANDOFFS | OWNER_PHASES,
                "unknown_owner_phase",
            )
            self.phase = state["phase"]
            if self.phase in HANDOFFS:
                self.status = "paused"

    def _verify_star(self):
        self._allow_import = True
        self._journal_check()
        names = [row["star"] for row in self._completed] + [self._current["star"]]
        state, inventory_capture, sources = _verified_owner(
            self.book, self.journal, self._owned(self._current["owner_output"]), names, self._current["star"]
        )
        _require(
            self.owner.report == self.book.json(self._owned(self._current["owner_output"]) / "report.json"),
            "owner_memory_receipt_mismatch",
        )
        self._progress = self.journal.load()
        self._journal_raw = self.journal.path.read_bytes()
        _require(
            self._progress.reduce() == state and self.journal.load() == self._progress, "concurrent_import"
        )
        completed = {**self._current, **sources, "journal_sha256": _sha(self._journal_raw)}
        self._persist(f"verified-{len(names):03d}.json", completed)
        self._completed.append(completed)
        self._labels_from(inventory_capture, {"surface": "list", "section": "stellar", "star": None})
        self._relay.retire()
        if len(names) == self.target:
            self.status, self.phase = "handoff", "awaiting_assessment"
            self._finish()
        else:
            self.phase = "next_star_initializing"

    def _construct_transition(self):
        _require(
            len(self._completed) == len(self._progress.reduce().stars)
            and all(s.task_completed for s in self._progress.reduce().stars.values()),
            "previous_collection_incomplete",
        )
        index = len(self._completed)
        child = self._transition_factory(
            self.page,
            self.config,
            self.output / f"transitions/{index:03d}-to-{index + 1:03d}",
            run_history=self.history,
            journal=self.journal,
            completed_owner_dir=self._owned(self._current["owner_output"]),
            expected_stars=[row["star"] for row in self._completed],
            visited_receipts=deepcopy(self._visited),
            max_seconds=180,
            max_advances=128,
            emit=self._relay.bind("campaign.next_star"),
            cancelled=lambda: self.finished or self._cancelled(),
            _clock=self._clock,
        )
        self.transition = child
        if self.finished:
            child.abort()
            return
        self._transition_state = deepcopy(child.state())
        self.phase = "next_star_active"

    def _adopt_fresh(self):
        state = self.transition.state()
        directory = self._owned(state["fresh_star_dir"])
        receipt_path = directory / "confirmed.json"
        output = self.book.clean(self.transition.output)
        _require(
            output
            == self.output / f"transitions/{len(self._completed):03d}-to-{len(self._completed) + 1:03d}"
            and directory == output / "picker"
            and state.get("mode") == TRANSITION_MODE
            and state.get("previous_star", "").casefold() == self._current["star"].casefold()
            and state.get("expected_stars") == [row["star"] for row in self._completed]
            and state.get("project_progress") == self._progress.reduce().report()
            and state.get("max_seconds") == 180
            and state.get("max_advances") == 128
            and state.get("failure_reason") is None
            and state.get("event_forwarding_failed") is False,
            "transition_scope_mismatch",
        )
        _require(
            state
            == self.transition.report
            == self.book.json(output / "report.json")
            == self.book.json(output / "confirmed.json")
            and state.get("status") == "completed"
            and state.get("phase") == "fresh_star_handoff"
            and all(
                state.get(k) is False
                for k in (
                    "task_completed",
                    "project_completed",
                    "class_selection_verified",
                    "collection_count_verified",
                )
            ),
            "transition_not_verified",
        )
        for name, checksum in state["source_sha256"].items():
            path = self._owned(name)
            # The transition freezes this revision, not every future revision.
            # Keep it separate from immutable files, which remain pinned forever.
            raw = self.journal.path.read_bytes() if path == self.journal.path else self.book.read(path)
            _require(_sha(raw) == checksum, "transition_source_changed")
        entry = {"path": str(receipt_path), "sha256": state["fresh_star_sha256"]}
        receipt, scope, capture = _fresh(self.book, entry)
        _require(
            scope.get("visited_stars") == [row["star"] for row in self._completed]
            and scope.get("excluded_points")
            == [self.book.json(self._owned(e["path"]))["selected_point"] for e in self._visited],
            "fresh_selection_history_changed",
        )
        _require(
            receipt["star"].casefold() not in {row["star"].casefold() for row in self._completed},
            "revisited_star",
        )
        self._visited.append(entry)
        self._current = self._context(len(self._visited), receipt["star"], directory, entry["sha256"])
        self._persist(f"star-{len(self._visited):03d}.json", self._current)
        self._labels_from(capture, {"surface": "detail", "section": "stellar", "star": receipt["star"]})
        self._relay.retire()
        self.owner = self.transition = self._owner_state = self._transition_state = None
        self.phase = "initializing_star"

    def _advance(self):
        if self.finished:
            return self.state()
        try:
            self._enter()
            self._allow_import = self.phase == "verifying_star"
            self._check(models=True)
            self._last_tick = self._clock()
            self.advances += 1
            if self.phase == "initializing_star":
                self._construct_owner()
            elif self.phase in OWNER_PHASES:
                self._allow_import = self.phase == "inventory_import"
                self.owner.step() if self.status == "paused" else self.owner.tick()
                if not self.finished:
                    self._sync_owner()
            elif self.phase == "verifying_star":
                self._verify_star()
            elif self.phase == "next_star_initializing":
                self._construct_transition()
            elif self.phase == "next_star_active":
                self.transition.advance()
                if not self.finished:
                    self._transition_state = deepcopy(self.transition.state())
                    if self.transition.finished:
                        _require(
                            self._transition_state.get("status") == "completed", "next_star_stopped_no_retry"
                        )
                        self.phase = "adopting_fresh_star"
            elif self.phase == "adopting_fresh_star":
                self._adopt_fresh()
            else:
                raise BrowserSafetyStop("project_campaign_explicit_handoff_required")
            self._publish()
        except _Cancelled:
            pass
        except (KeyboardInterrupt, SystemExit):
            self.abort()
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize source/browser errors
            if not self.finished:
                self._stop(_reason(exc))
        finally:
            self._allow_import = False
            self._busy = False
        return self.state()

    def step(self):
        if self.finished:
            return self.state()
        _require(self.status == "paused", "pause_before_step")
        return self.state() if self.phase in HANDOFFS else self._advance()

    def tick(self):
        return self._advance() if self.status == "running" else self.state()

    def advance_if_due(self):
        return (
            self.tick()
            if self.status == "running"
            and (self._last_tick is None or self._clock() - self._last_tick >= self.interval)
            else self.state()
        )

    def pause(self):
        if self.finished:
            return self.state()
        _require(self.status in {"running", "paused"}, "not_started")
        self.status = "paused"
        if self.owner and not self.owner.finished:
            self.owner.pause()
        return self.state()

    def resume(self):
        _require(
            not self._busy and not self._emitting and self.status == "paused" and self.phase not in HANDOFFS,
            "explicit_handoff_required",
        )
        self._check()
        if self.owner and not self.owner.finished:
            self.owner.resume()
        self.status = "running"
        return self.state()

    def _provide(self, phase, method, values):
        _require(
            self.status == "paused" and self.phase == phase and self.owner is not None,
            "reference_handoff_not_ready",
        )
        try:
            self._enter()
            self._check(models=True)
            if method == "provide_class":
                source = validate_star_class_source(
                    self.history,
                    self._owned(values["class_dir"]),
                    self._current["star"],
                    values["selected_class"],
                )
                _require(
                    self._owned(source["fresh_star"]) == self._owned(self._current["fresh_dir"])
                    and source["source_capture_sha256"]
                    == _sha(
                        self.book.read(self._owned(self._current["fresh_dir"]) / "stellar/observation.json")
                    ),
                    "class_not_bound_to_current_star",
                )
            getattr(self.owner, method)(**values)
            if not self.finished:
                self._sync_owner()
                self._publish()
        except _Cancelled:
            pass
        except Exception as exc:  # noqa: BLE001 - source decisions remain local
            self._stop(_reason(exc))
        finally:
            self._busy = False
        return self.state()

    def provide_class(self, *, class_dir, selected_class, lifetime_prefix):
        return self._provide(
            "awaiting_class_source",
            "provide_class",
            {"class_dir": class_dir, "selected_class": selected_class, "lifetime_prefix": lifetime_prefix},
        )

    def provide_planet_class(self, *, name, rationale):
        return self._provide(
            "awaiting_planet_class", "provide_planet_class", {"name": name, "rationale": rationale}
        )

    def provide_gases(self, *, gases, rationale, supplied_greenhouse_increment):
        return self._provide(
            "awaiting_gases",
            "provide_gases",
            {
                "gases": gases,
                "rationale": rationale,
                "supplied_greenhouse_increment": supplied_greenhouse_increment,
            },
        )

    def provide_habitability(self, *, choice, rationale):
        return self._provide(
            "awaiting_habitability", "provide_habitability", {"choice": choice, "rationale": rationale}
        )

    def provide_autonomous_gases(self, *, gases, rationale):
        return self._provide(
            "awaiting_gases", "provide_autonomous_gases", {"gases": gases, "rationale": rationale}
        )

    def _finish(self):
        if self.report is not None or self._finalizing:
            return
        self._finalizing = True
        try:
            previous = self.state()
            self._emit("episode_summary", previous)
            if self.status == "handoff" and not self.failure:
                try:
                    _require(
                        not self._cancelled() and self._within_campaign_deadline(),
                        "cancelled_or_time_limit_at_handoff",
                    )
                    self._sources_check(models=True)
                except Exception as exc:  # noqa: BLE001 - no completion after changed proof
                    self.status, self.phase, self.failure = "stopped", "stopped", _reason(exc)
            if self._forward_failed:
                self.status, self.phase, self.failure = (
                    "stopped",
                    "stopped",
                    "project_campaign_event_forwarding_failed",
                )
            if self.state() != previous:
                self._emit("state", self.state())
            self.report = self.state()
            persist_json(self.output / "report.json", self.report)
        finally:
            self._relay.retire()
            self._stream.close()
            self._finalizing = False

    def _stop(self, reason, *, aborted=False):
        if self.report is not None or self._closing:
            return
        self.status = self.phase = "aborted" if aborted else "stopped"
        self.failure, self._closing = (
            (
                self._child_failure
                if self._child_failure and not aborted and not self._forward_failed
                else reason
            ),
            True,
        )
        try:
            for child in (self.owner, self.transition):
                if child is not None and not child.finished:
                    try:
                        child.abort()
                    except Exception:  # noqa: BLE001 - no retry after cancellation
                        self._child_abort_failed = True
                        if self._child_failure is None:
                            self.failure = "project_campaign_child_abort_failed"
            self._finish()
        finally:
            self._closing = False

    def abort(self):
        if self.report is None:
            self._stop("operator_aborted", aborted=True)
        return self.state()

    close = abort

    def command(self, message):
        message = RuntimeCommand.model_validate(message)
        command, payload = message.command, message.payload
        if command == "start" and set(payload) <= {"paused"}:
            return self.start(**payload)
        if command == "step":
            if not payload:
                return self.step()
            methods = {
                "class_source": self.provide_class,
                "planet_class": self.provide_planet_class,
                "gases": self.provide_gases,
                "habitability": self.provide_habitability,
            }
            if len(payload) == 1 and next(iter(payload)) in methods:
                key = next(iter(payload))
                return methods[key](**payload[key])
        if not payload and command in {"pause", "resume", "abort"}:
            return getattr(self, command)()
        raise ValueError("Unsupported campaign command")
