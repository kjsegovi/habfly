"""A bounded runtime owner for one already-open, explicitly classified star.

No browser creation, login, star collection, inferred classification, scoring or
submission lives here. Setup belongs to the outer runtime. Constructors perform filesystem
validation only; child construction and each cooperative decision occupy separate
ticks. Pause takes effect between bounded child calls, not midway through a
Playwright call. The browser remains owned by the caller even after close.
"""

import hashlib
import json
import math
import os
import re
import time
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_habitability import GASES
from .browser_no_planet_workflow import _Evidence, _planet
from .browser_planet_classification import CLASSES as PLANET_CLASSES
from .browser_planet_classification import select_planet_class
from .browser_planet_numeric import planet_projection
from .browser_positive_finalize import PositiveFinalizationSteps
from .browser_positive_planet_workflow import _planet_sources, _stellar_sources
from .browser_positive_steps import SUPPLIED_OWNER_MODE
from .browser_project_inventory_steps import ProjectInventorySteps
from .browser_star_preflight import validate_star_class_source
from .browser_star_session import BrowserStarSession
from .browser_stellar import CLASSES
from .browser_terrestrial_steps import MODE as TERRESTRIAL_MODE
from .browser_terrestrial_steps import TerrestrialSteps
from .contracts import RuntimeCommand, RuntimeEvent
from .lifetime_prefix import PREFIX_YEARS
from .model import graph_fingerprint
from .project_events import ProjectEventRelay
from .project_evidence import import_verified_no_planet
from .project_inventory_source import load_inventory_source
from .project_positive_evidence import import_verified_positive_planet
from .project_progress import ProjectJournal
from .project_terrestrial_evidence import import_verified_terrestrial
from .supplied_browser_modes import (
    POSITIVE_SUPPLIED_MODE,
    TERRESTRIAL_SUPPLIED_MODE,
    positive_mode,
    source_options,
    terrestrial_mode,
)
from .training.planet_sequence import file_hash

_MODEL_KEYS = {
    "dataset",
    "checkpoint",
    "color_experiment",
    "graph_path",
    "seed",
    "planet_pilot",
    "planet_final_evaluation",
    "planet_supplied_evaluation",
    "planet_seed",
    "planet_preserve_painted_class",
    "no_planet_save_settle_seconds",
    "shallow_reference",
    "two_event_reference",
    "allow_baseline_edge_reference",
    "allow_baseline_band_reference",
    "allow_single_event_reference",
    "save_strategy",
}
_REQUIRED_MODELS = {"dataset", "checkpoint", "color_experiment", "graph_path"}
_HANDOFFS = {
    "planet_measurement_required",
    "planet_classification_required",
    "planet_calculation_unsupported",
}
_INVENTORY_PHASES = {"inventory_initializing", "inventory_active", "inventory_import"}
_POSITIVE_PHASES = {"selecting_planet_class", "positive_initializing", "positive_active"}
_TERRESTRIAL_PHASES = {"terrestrial_initializing", "terrestrial_active"}
_TERRESTRIAL_HANDOFFS = {"awaiting_gases", "awaiting_habitability"}
_CONTINUATION_PHASES = _POSITIVE_PHASES | _TERRESTRIAL_PHASES | _TERRESTRIAL_HANDOFFS
_DRIVABLE = {"ready", "active"} | _INVENTORY_PHASES | _POSITIVE_PHASES | _TERRESTRIAL_PHASES
_TERRESTRIAL_REQUIRED = {"pilot", "final_evaluation", "graph", "candidates"}
_TERRESTRIAL_DEFAULTS = {
    "seed": 20000000,
    "max_seconds": 1800,
    "timeout_seconds": 20,
    "settle_timeout_seconds": 20,
}
_WORKFLOW_PHASES = {
    "no_planet_visible_workflow_readback": "verified_no_planet",
    "positive_planet_visible_workflow_readback": "verified_positive_planet",
    "terrestrial_visible_workflow_readback": "verified_terrestrial",
    POSITIVE_SUPPLIED_MODE: "verified_positive_planet",
    TERRESTRIAL_SUPPLIED_MODE: "verified_terrestrial",
}


def _require(condition, code):
    if not condition:
        raise BrowserSafetyStop("project_steps_" + code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _reason(exc):
    code = str(exc)
    return (
        code
        if isinstance(exc, BrowserSafetyStop) and re.fullmatch(r"[a-z][a-z0-9_:.-]{0,150}", code)
        else "project_steps_component_failed"
    )


class BrowserProjectSteps:
    """Single active star with explicit class and inventory evidence handoffs.

    ``emit(kind, payload)`` plugs into Runtime.emit. The identical serialized
    payload is first flushed to this owner's v1 JSONL. Original child envelopes
    remain nested under component_event; child summaries never become project
    completion. ``component_factory`` is an offline-test injection seam.

    A verified No workflow defaults to waiting for ``provide_inventory``.
    ``automatic_inventory`` opts into bounded visible-list navigation first.
    Only the existing
    strict workflow importer may establish task completion in ProjectJournal.
    Positive branches stop at an explicit handoff by default. The separate
    ``reference_planet_continuation`` opt-in permits one caller-selected reference
    class and bounded gas/ice finalization, never an inferred or learned class.
    ``terrestrial_options`` separately enables the existing terrestrial owner;
    gas and habitability decisions remain explicit offline handoffs.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        journal,
        star,
        model_options,
        automatic_inventory=False,
        reference_planet_continuation=False,
        positive_save_settle_seconds=20,
        terrestrial_options=None,
        emit=lambda *_: None,
        component_factory=None,
        interval=0.2,
        max_advances=512,
        max_seconds=1800,
        _clock=time.monotonic,
    ):
        _require(isinstance(journal, ProjectJournal), "invalid_journal")
        _require(
            isinstance(star, str)
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{0,79}", star)
            and star == " ".join(star.split()),
            "invalid_star",
        )
        _require(
            isinstance(model_options, dict)
            and _REQUIRED_MODELS <= model_options.keys()
            and not (model_options.keys() - _MODEL_KEYS),
            "invalid_model_options",
        )
        _require(callable(emit) and callable(_clock), "invalid_callback")
        _require(type(model_options.get("shallow_reference", False)) is bool, "invalid_shallow_reference")
        _require(
            type(model_options.get("two_event_reference", False)) is bool
            and (
                not model_options.get("two_event_reference", False)
                or model_options.get("shallow_reference") is True
            ),
            "invalid_two_event_reference",
        )
        _require(
            type(model_options.get("allow_baseline_edge_reference", False)) is bool,
            "invalid_baseline_edge_reference",
        )
        _require(
            type(model_options.get("allow_baseline_band_reference", False)) is bool
            and (
                not model_options.get("allow_baseline_band_reference", False)
                or model_options.get("allow_baseline_edge_reference") is True
            ),
            "invalid_baseline_band_reference",
        )
        _require(
            type(model_options.get("allow_single_event_reference", False)) is bool
            and (
                not model_options.get("allow_single_event_reference", False)
                or model_options.get("allow_baseline_band_reference") is True
            ),
            "invalid_single_event_reference",
        )
        _require(
            type(model_options.get("save_strategy", "explicit")) is str
            and model_options.get("save_strategy", "explicit") in {"explicit", "autosave"},
            "invalid_save_strategy",
        )
        _require(type(automatic_inventory) is bool, "invalid_automatic_inventory")
        _require(type(reference_planet_continuation) is bool, "invalid_planet_continuation")
        _require(
            type(positive_save_settle_seconds) in {int, float}
            and math.isfinite(positive_save_settle_seconds)
            and 0.1 <= positive_save_settle_seconds <= 30,
            "invalid_positive_save_settlement",
        )
        _require(component_factory is None or callable(component_factory), "invalid_factory")
        _require(type(max_advances) is int and 1 <= max_advances <= 2048, "invalid_advance_limit")
        for value, low, high in ((interval, 0, 10), (max_seconds, 1, 3600)):
            _require(
                type(value) in {int, float} and math.isfinite(value) and low <= value <= high,
                "invalid_time_limit",
            )
        self.history = Path(run_history).absolute()
        _require(self.history.is_dir() and self.history == self.history.resolve(), "invalid_history")
        self.journal = journal
        journal_path = self._owned(journal.path)
        _require(journal_path.parent == self.history and journal_path.is_file(), "journal_outside_history")
        self.star = star
        self._progress, self._journal_hash = self._read_journal()
        self.output = self._owned(output)
        _require(self.output != self.history and not self.output.exists(), "output_already_exists")
        self.page, self.config = page, config.model_copy(deep=True)
        self.model_options = deepcopy(model_options)
        self._two_event_enabled = model_options.get("two_event_reference", False)
        self._baseline_edge_enabled = model_options.get("allow_baseline_edge_reference", False)
        self._baseline_edge_source = Path(__file__).with_name("planet_window_baseline_edge.py")
        self._baseline_edge_sha = (
            _sha(self._baseline_edge_source.read_bytes()) if self._baseline_edge_enabled else None
        )
        self._baseline_band_enabled = model_options.get("allow_baseline_band_reference", False)
        self._baseline_band_source = Path(__file__).with_name("planet_window_baseline_band.py")
        self._baseline_band_sha = (
            _sha(self._baseline_band_source.read_bytes()) if self._baseline_band_enabled else None
        )
        self._single_event_enabled = model_options.get("allow_single_event_reference", False)
        self._single_event_source = Path(__file__).with_name("planet_window_single_event.py")
        self._single_event_sha = (
            _sha(self._single_event_source.read_bytes()) if self._single_event_enabled else None
        )
        self._save_strategy = model_options.get("save_strategy", "explicit")
        self._autosave_source = Path(__file__).with_name("browser_autosave.py")
        self._autosave_sha = (
            _sha(self._autosave_source.read_bytes()) if self._save_strategy == "autosave" else None
        )
        self._callback, self._factory = emit, component_factory or BrowserStarSession
        self._clock, self.interval = _clock, interval
        self.max_advances, self.max_seconds = max_advances, max_seconds
        self._started_at, self._last_tick, self.advances = None, None, 0
        self.status, self.phase, self.failure = "idle", "not_started", None
        self.component = self.report = self._class_source = self._class_options = None
        self._child_state, self._workflow = None, None
        self._continuation_save_outcome = None
        self.automatic_inventory = automatic_inventory
        self.inventory_component = self._inventory_state = self._inventory_source = None
        self._inventory_expected = None
        self.reference_planet_continuation = reference_planet_continuation
        if model_options.get("planet_supplied_evaluation") is not None:
            _require(
                reference_planet_continuation
                and terrestrial_options is not None
                and terrestrial_options.get("supplied_evaluation") is not None,
                "supplied_inputs_require_complete_planet_and_terrestrial_options",
            )
        self.positive_save_settle_seconds = positive_save_settle_seconds
        self.positive_component = self._positive_state = None
        self._positive_source = self._planet_class_request = self._planet_class_source = None
        self._positive_final_source = None
        self.terrestrial_component = self._terrestrial_state = self._terrestrial_final_source = None
        self._terrestrial_decisions = {}
        self.terrestrial_options = self._terrestrial_settings = self._terrestrial_model_source = None
        if terrestrial_options is not None:
            _require(reference_planet_continuation, "terrestrial_requires_planet_continuation")
            self._configure_terrestrial(terrestrial_options)
        self._derived_capture = None
        self._claim_path = self._claim_hash = None
        self._busy = self._closing = self._finalizing = self._forward_failed = False
        self._child_failure = None
        self._emitting = False
        self._child_abort_failed = False
        self._sequence = 0
        self._run_id = "project-star-" + _sha(str(self.output.relative_to(self.history)).encode())[:16]
        self._relay = ProjectEventRelay(self._receive)
        self.output.mkdir(parents=True, exist_ok=False)
        self._stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        self.scope = {
            "mode": "bounded_single_star_project_runtime",
            "star": star,
            "project_id": journal.identity.project_id,
            "attempt_id": journal.identity.attempt_id,
            "classification_source": "explicit_source_handoff_not_learned",
            "automatic_inventory": automatic_inventory,
            "reference_planet_continuation": reference_planet_continuation,
            "positive_save_settle_seconds": positive_save_settle_seconds,
            "terrestrial_workflow_enabled": terrestrial_options is not None,
            "terrestrial_settings": deepcopy(self._terrestrial_settings),
            "max_advances": max_advances,
            "max_seconds": max_seconds,
            "cancellation": "between_bounded_component_calls",
            "optimizer_updates": 0,
            "automatic_retry": False,
            "browser_owned_by_caller": True,
            "setup_enabled": False,
            "collection_enabled": False,
            "class_selection_enabled": False,
            "explicit_planet_class_selection_enabled": reference_planet_continuation,
            "assessment_enabled": False,
            "submission_enabled": False,
            "scientific_verified": False,
            **(
                {
                    "save_strategy": "autosave",
                    "autosave_source_sha256": self._autosave_sha,
                    "persistence_verified": False,
                }
                if self._save_strategy == "autosave"
                else {}
            ),
            **({"two_event_reference_enabled": True} if self._two_event_enabled else {}),
            **(
                {
                    "single_event_reference_enabled": True,
                    "single_event_source_sha256": self._single_event_sha,
                }
                if self._single_event_enabled
                else {}
            ),
            **(
                {
                    "baseline_band_reference_enabled": True,
                    "baseline_band_source_sha256": self._baseline_band_sha,
                }
                if self._baseline_band_enabled
                else {}
            ),
            **(
                {
                    "baseline_edge_reference_enabled": True,
                    "baseline_edge_source_sha256": self._baseline_edge_sha,
                }
                if self._baseline_edge_enabled
                else {}
            ),
        }
        persist_json(self.output / "scope.json", self.scope)
        self._baseline_band_scope_sha = (
            _sha((self.output / "scope.json").read_bytes()) if self._baseline_band_enabled else None
        )
        self._single_event_scope_sha = (
            _sha((self.output / "scope.json").read_bytes()) if self._single_event_enabled else None
        )
        self._autosave_scope_sha = (
            _sha((self.output / "scope.json").read_bytes()) if self._save_strategy == "autosave" else None
        )

    def _configure_terrestrial(self, options):
        """Freeze explicit options only; the child owns full model/source validation."""
        _require(
            isinstance(options, dict)
            and _TERRESTRIAL_REQUIRED <= options.keys()
            and not (
                options.keys()
                - _TERRESTRIAL_REQUIRED
                - _TERRESTRIAL_DEFAULTS.keys()
                - {"supplied_evaluation"}
            ),
            "invalid_terrestrial_options",
        )
        supplied = {**_TERRESTRIAL_DEFAULTS, **options}
        candidates = supplied["candidates"]
        _require(
            isinstance(candidates, (list, tuple))
            and 1 <= len(candidates) <= len(GASES)
            and all(isinstance(v, str) for v in candidates)
            and len(set(candidates)) == len(candidates)
            and set(candidates) <= set(GASES),
            "invalid_terrestrial_candidates",
        )
        _require(type(supplied["seed"]) is int, "invalid_terrestrial_seed")
        for key, minimum, maximum in (
            ("max_seconds", 0.001, 1800),
            ("timeout_seconds", 0.1, 30),
            ("settle_timeout_seconds", 0.1, 30),
        ):
            value = supplied[key]
            _require(
                type(value) in {int, float} and math.isfinite(value) and minimum <= value <= maximum,
                "invalid_terrestrial_budget",
            )
        _require(
            all(isinstance(supplied[key], (str, Path)) for key in ("pilot", "final_evaluation")),
            "invalid_terrestrial_model_path",
        )
        supplied["pilot"], supplied["final_evaluation"] = (
            Path(supplied[key]).resolve() for key in ("pilot", "final_evaluation")
        )
        supplied["candidates"] = tuple(candidates)
        if "supplied_evaluation" in supplied:
            _require(
                isinstance(supplied["supplied_evaluation"], (str, Path))
                and self.model_options.get("planet_supplied_evaluation") is not None,
                "invalid_supplied_temperature_evaluation",
            )
            supplied["supplied_evaluation"] = Path(supplied["supplied_evaluation"]).resolve()
            _require(supplied["supplied_evaluation"].is_dir(), "missing_supplied_temperature_evaluation")
        self.terrestrial_options = supplied
        paths = {
            "checkpoint": supplied["pilot"] / "training/checkpoint.pt",
            "metadata": supplied["pilot"] / "training/checkpoint.pt.json",
            "pilot_report": supplied["pilot"] / "report.json",
            "final_gate": supplied["final_evaluation"] / "report.json",
        }
        if "supplied_evaluation" in supplied:
            paths["supplied_gate"] = supplied["supplied_evaluation"] / "report.json"
        self._terrestrial_model_source = {str(path): file_hash(path) for path in paths.values()}
        self._terrestrial_settings = self._terrestrial_snapshot()

    def _terrestrial_snapshot(self):
        options = self.terrestrial_options
        return {
            **{key: options[key] for key in _TERRESTRIAL_DEFAULTS},
            "pilot": str(options["pilot"]),
            "final_evaluation": str(options["final_evaluation"]),
            "candidates": list(options["candidates"]),
            "graph_hash": graph_fingerprint(options["graph"]),
            "model_source_sha256": dict(self._terrestrial_model_source),
            **(
                {"supplied_evaluation": str(options["supplied_evaluation"])}
                if "supplied_evaluation" in options
                else {}
            ),
        }

    def _uses_supplied_inputs(self):
        return (
            self.model_options.get("planet_supplied_evaluation") is not None
            and self._class_options is not None
            and self._class_options["selected_class"] != "main_sequence"
        )

    def _supplied_closures(self, book):
        if not self._uses_supplied_inputs():
            return {}
        return {
            "closed_trees": {
                str(path.relative_to(self.history)): [list(row) for row in tree]
                for path, tree in book.closed_trees.items()
            },
            "clean_directories": sorted(
                str(path.relative_to(self.history)) for path in book.clean_directories
            ),
        }

    def _owned(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.history / path
        _require(path.resolve().is_relative_to(self.history), "path_outside_history")
        _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_path")
        return path.resolve()

    def _read_journal(self):
        _require(
            {self._owned(p) for p in self.history.glob("project-progress-*.jsonl")}
            == {self._owned(self.journal.path)},
            "ambiguous_attempt_journal",
        )
        progress = self.journal.load().reduce()
        _require(
            not progress.pending and progress.receipt("submission") is None, "uncertain_or_submitted_journal"
        )
        raw = self.journal.path.read_bytes()
        # Detect a concurrent append between the reducer read and fingerprint.
        _require(self.journal.load().reduce() == progress, "journal_changed")
        return progress, _sha(raw)

    def _check(self, *, allow_child_error=False):
        _require(
            type(self.model_options.get("save_strategy", "explicit")) is str
            and self.model_options.get("save_strategy", "explicit") == self._save_strategy
            and self.scope.get("save_strategy", "explicit") == self._save_strategy
            and (
                self._save_strategy != "autosave"
                or (
                    self.scope.get("persistence_verified") is False
                    and self.scope.get("autosave_source_sha256") == self._autosave_sha
                    and _sha(self._autosave_source.read_bytes()) == self._autosave_sha
                    and _sha(self._owned(self.output / "scope.json").read_bytes()) == self._autosave_scope_sha
                )
            ),
            "save_strategy_changed",
        )
        _require(
            self.model_options.get("allow_single_event_reference", False) is self._single_event_enabled
            and self.scope.get("single_event_reference_enabled", False) is self._single_event_enabled
            and (
                not self._single_event_enabled
                or (
                    self.model_options.get("allow_baseline_band_reference") is True
                    and self.scope.get("single_event_source_sha256") == self._single_event_sha
                    and _sha(self._single_event_source.read_bytes()) == self._single_event_sha
                    and _sha(self._owned(self.output / "scope.json").read_bytes())
                    == self._single_event_scope_sha
                )
            ),
            "single_event_permission_changed",
        )
        _require(
            self.model_options.get("allow_baseline_band_reference", False) is self._baseline_band_enabled
            and self.scope.get("baseline_band_reference_enabled", False) is self._baseline_band_enabled
            and (
                not self._baseline_band_enabled
                or (
                    self.model_options.get("allow_baseline_edge_reference") is True
                    and self.scope.get("baseline_band_source_sha256") == self._baseline_band_sha
                    and _sha(self._baseline_band_source.read_bytes()) == self._baseline_band_sha
                    and _sha(self._owned(self.output / "scope.json").read_bytes())
                    == self._baseline_band_scope_sha
                )
            ),
            "baseline_band_permission_changed",
        )
        _require(
            self.model_options.get("two_event_reference", False) is self._two_event_enabled
            and self.scope.get("two_event_reference_enabled", False) is self._two_event_enabled
            and (not self._two_event_enabled or self.model_options.get("shallow_reference") is True),
            "two_event_permission_changed",
        )
        _require(
            self.model_options.get("allow_baseline_edge_reference", False) is self._baseline_edge_enabled
            and self.scope.get("baseline_edge_reference_enabled", False) is self._baseline_edge_enabled
            and (
                not self._baseline_edge_enabled
                or (
                    self.scope.get("baseline_edge_source_sha256") == self._baseline_edge_sha
                    and _sha(self._baseline_edge_source.read_bytes()) == self._baseline_edge_sha
                )
            ),
            "baseline_edge_permission_changed",
        )
        if self._child_failure and not allow_child_error:
            raise BrowserSafetyStop(self._child_failure)
        progress, digest = self._read_journal()
        _require(digest == self._journal_hash, "journal_changed")
        if self.terrestrial_options is not None:
            _require(
                self._terrestrial_snapshot() == self._terrestrial_settings, "terrestrial_options_changed"
            )
            _require(
                all(
                    file_hash(Path(path)) == expected
                    for path, expected in self._terrestrial_model_source.items()
                ),
                "terrestrial_model_changed",
            )
        if self._class_source:
            source = validate_star_class_source(
                self.history,
                self._class_options["class_dir"],
                self.star,
                self._class_options["selected_class"],
            )
            _require(source == self._class_source, "class_source_changed")
        if self._workflow:
            _require(
                _sha(self._owned(self._workflow["path"]).read_bytes()) == self._workflow["sha256"],
                "workflow_changed",
            )
        if self._claim_path:
            _require(
                _sha(self._owned(self._claim_path).read_bytes()) == self._claim_hash, "reservation_changed"
            )
        for source in (
            self._positive_source,
            self._planet_class_request,
            self._planet_class_source,
            self._positive_final_source,
            self._terrestrial_final_source,
            *self._terrestrial_decisions.values(),
        ):
            if source:
                for name, expected in source["hashes"].items():
                    _require(_sha(self._owned(name).read_bytes()) == expected, "positive_source_changed")
                for name, expected in source.get("trees", {}).items():
                    _require(self._tree(self._owned(name)) == expected, "positive_source_tree_changed")
                if "closed_trees" in source:
                    book = _Evidence(self.history)
                    for name in source["clean_directories"]:
                        book.clean(self._owned(name))
                    for name, expected in source["closed_trees"].items():
                        _require(
                            book.tree(self._owned(name)) == tuple(tuple(row) for row in expected),
                            "supplied_source_tree_changed",
                        )
        if self._inventory_source:
            for directory in (self.output / "collection", self.output / "collection/inventory"):
                _require(
                    not any((directory / name).exists() for name in ("stopped.json", "invalidated.json")),
                    "inventory_source_invalidated",
                )
            for name, expected in self._inventory_source["hashes"].items():
                _require(_sha(self._owned(name).read_bytes()) == expected, "inventory_source_changed")
        self._progress = progress

    @property
    def finished(self):
        return self.status in {"completed", "handoff", "stopped", "aborted"}

    def state(self):
        child = deepcopy(self._child_state)
        inventory = deepcopy(self._inventory_state)
        artifacts = deepcopy((child or {}).get("artifact_paths", {}))
        if self.inventory_component is not None:
            artifacts["inventory_component_dir"] = str((self.output / "collection").relative_to(self.history))
        if self._inventory_source:
            artifacts["inventory_dir"] = self._inventory_source["inventory_dir"]
        if self._planet_class_source:
            artifacts["planet_class_dir"] = self._planet_class_source["directory"]
        if self.positive_component is not None:
            artifacts["positive_finalization_dir"] = str(
                (self.output / "positive-finalization").relative_to(self.history)
            )
        if self.terrestrial_component is not None:
            artifacts["terrestrial_dir"] = str((self.output / "terrestrial").relative_to(self.history))
        if self._workflow:
            artifacts["workflow_dir"] = str(Path(self._workflow["path"]).parent)
        return {
            **self.scope,
            "status": self.status,
            "phase": self.phase,
            "stage": "browser_project_star",
            "browser_phase": self.phase,
            "browser_status": self.phase,
            "policy_stage": (self._terrestrial_state or {}).get("phase", self.phase)
            if self.phase in _TERRESTRIAL_PHASES | _TERRESTRIAL_HANDOFFS
            else self.phase
            if self.phase in _INVENTORY_PHASES | _POSITIVE_PHASES
            else (child or {}).get("phase"),
            "finished": self.finished,
            "task_completed": self.status == "completed",
            "project_completed": False,
            "failure_reason": self.failure,
            "event_forwarding_failed": self._forward_failed,
            "child_abort_failed": self._child_abort_failed,
            "advances": self.advances,
            "star_component": child,
            **{
                key: deepcopy(child[key])
                for key in ("save_outcome", "save_outcome_uncertain")
                if isinstance(child, dict) and key in child
            },
            **(
                {
                    "save_outcome": deepcopy(self._continuation_save_outcome),
                    "save_outcome_uncertain": self._continuation_save_outcome["save_outcome_uncertain"],
                }
                if self._continuation_save_outcome is not None
                else {}
            ),
            "inventory_component": inventory,
            "positive_component": deepcopy(self._positive_state),
            "terrestrial_component": deepcopy(self._terrestrial_state),
            "artifact_paths": artifacts,
            "reference_measurements": (
                deepcopy((child or {}).get("reference_measurements"))
                if not self.failure and self.phase not in _INVENTORY_PHASES
                else None
            ),
            "project_progress": self._progress.report(),
            "class_handoff": (
                {"required": True, "allowed_classes": sorted(CLASSES), "browser_writes": 0}
                if self.phase == "awaiting_class_source"
                else None
            ),
            "inventory_handoff": deepcopy(self._workflow) if self.phase == "awaiting_inventory" else None,
            "planet_class_handoff": (
                {
                    "required": True,
                    "allowed_classes": list(PLANET_CLASSES),
                    "rationale_required": True,
                    "classification_learned": False,
                    "browser_writes": 0,
                    "terrestrial_workflow_available": self.terrestrial_options is not None,
                }
                if self.phase == "awaiting_planet_class"
                else None
            ),
            "planet_class_decision": (
                {"name": self._planet_class_request["name"], "classification_learned": False}
                if self._planet_class_request
                else None
            ),
            "gas_handoff": (
                {
                    "required": True,
                    "candidates": list(self.terrestrial_options["candidates"]),
                    "rationale_required": True,
                    "supplied_greenhouse_increment_required": True,
                    "gas_identification_learned": False,
                    "browser_writes": 0,
                }
                if self.phase == "awaiting_gases"
                else None
            ),
            "habitability_handoff": (
                {
                    "required": True,
                    "allowed_choices": ["habitable", "not_habitable"],
                    "rationale_required": True,
                    "phase_reference": deepcopy((self._terrestrial_state or {}).get("phase_reference")),
                    "habitability_decision_learned": False,
                    "browser_writes": 0,
                }
                if self.phase == "awaiting_habitability"
                else None
            ),
            "trace_path": str((self.output / "events.jsonl").relative_to(self.history)),
        }

    def _emit(self, event, payload):
        item = RuntimeEvent(event=event, sequence=self._sequence, run_id=self._run_id, payload=payload)
        # Reject non-JSON/nonfinite child values instead of silently converting
        # them while claiming the original event remained exact.
        line = json.dumps(item.model_dump(), separators=(",", ":"), allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            try:
                self._emitting = True
                self._callback(event, json.loads(line)["payload"])
            except Exception:  # noqa: BLE001 - callbacks may contain credentials
                self._forward_failed = True
                persist_json(self.output / "event-forwarding-failed.json", {"failed": True})
                if not self._closing:
                    raise BrowserSafetyStop("project_steps_event_forwarding_failed") from None
            finally:
                self._emitting = False

    def _receive(self, kind, payload):
        _require(not self.finished or self._closing, "already_stopped")
        if kind == "action_proposed" and self._child_failure:
            raise BrowserSafetyStop(self._child_failure)
        if not self._closing:
            if self.phase in _CONTINUATION_PHASES:
                self._continuation_check(allow_child_error=True)
            else:
                self._check(allow_child_error=True)
        # Preserve a component's ordinary causal stop without making its
        # already-delivered error look like an exception in our callback.
        if kind == "error" and self._child_failure is None:
            self._child_failure = _reason(BrowserSafetyStop(payload.get("message")))
        if kind == "state":
            candidate = payload.get("component_state")
            if isinstance(candidate, dict) and "star" in candidate:
                self._validate_child(candidate)
                if payload.get("component") == "project.inventory":
                    self._inventory_state = deepcopy(candidate)
                elif payload.get("component") == "planet.finalization":
                    self._positive_state = deepcopy(candidate)
                elif payload.get("component") == "planet.terrestrial":
                    self._terrestrial_state = deepcopy(candidate)
                else:
                    self._child_state = deepcopy(candidate)
            payload = {**payload, **self.state()}
        self._emit(kind, payload)
        _require(not self.finished or self._closing, "already_stopped")
        if not self._closing:
            if self.phase in _CONTINUATION_PHASES:
                self._continuation_check(allow_child_error=True)
            else:
                self._check(allow_child_error=True)

    def _publish(self):
        try:
            self._emit("state", self.state())
        except Exception as exc:  # noqa: BLE001 - protocol boundary redacts driver errors
            self._stop(_reason(exc))
        return self.state()

    def _not_busy(self):
        if self._busy or self._finalizing or self._emitting:
            self._stop("project_steps_reentrant_advance")
            raise BrowserSafetyStop("project_steps_reentrant_advance")

    def start(self, *, paused=True):
        self._not_busy()
        _require(self.status == "idle" and type(paused) is bool, "already_started_or_invalid_start")
        # A start request never bypasses the class evidence handoff.
        self.status, self.phase = "paused", "awaiting_class_source"
        try:
            self._check()
            existing = [s for s in self._progress.stars.values() if s.name.casefold() == self.star.casefold()]
            _require(not any(s.task_completed for s in existing), "star_already_completed")
            self._emit("hello", {"protocol_version": 1, **self.scope})
        except Exception as exc:  # noqa: BLE001 - source errors never expose paths
            self._stop(_reason(exc))
        return self._publish() if not self.finished else self.state()

    def provide_class(self, *, class_dir, selected_class, lifetime_prefix):
        self._not_busy()
        _require(self.status == "paused" and self.phase == "awaiting_class_source", "class_handoff_not_ready")
        try:
            self._check()
            _require(selected_class in CLASSES, "unsupported_class")
            _require(
                lifetime_prefix in PREFIX_YEARS
                if selected_class == "main_sequence"
                else lifetime_prefix is None,
                "invalid_lifetime_prefix",
            )
            directory = self._owned(class_dir)
            source = validate_star_class_source(self.history, directory, self.star, selected_class)
            self._class_source = deepcopy(source)
            self._class_options = {
                "class_dir": directory,
                "selected_class": selected_class,
                "lifetime_prefix": lifetime_prefix,
            }
            persist_json(
                self.output / "class-handoff.json",
                {
                    "class_dir": str(directory.relative_to(self.history)),
                    "source": source,
                    "selected_class": selected_class,
                    "lifetime_prefix": lifetime_prefix,
                    "browser_writes": 0,
                    "classification_learned": False,
                },
            )
            self.phase = "ready"
        except Exception as exc:  # noqa: BLE001 - source errors never expose paths
            self._stop(_reason(exc))
        return self._publish() if not self.finished else self.state()

    def pause(self):
        if self.finished:
            return self.state()
        _require(self.status in {"running", "paused"}, "not_started")
        self.status = "paused"
        # During a child callback, do not recursively publish another callback.
        return self.state() if self._busy or self._emitting else self._publish()

    def resume(self):
        self._not_busy()
        _require(self.status == "paused" and self.phase in _DRIVABLE, "handoff_or_run_not_ready")
        self.status = "running"
        return self._publish()

    def _claim(self):
        directory = self.history / "project-star-runtime-reservations"
        self._owned(directory)
        directory.mkdir(exist_ok=True)
        root_fd = os.open(self.history, os.O_RDONLY)
        try:
            os.fsync(root_fd)  # make a newly created claim directory durable too
        finally:
            os.close(root_fd)
        key = _sha((self.journal.path.name + ":" + self.star.casefold()).encode())
        path = directory / (key + ".json")
        _require(not path.exists() and not path.is_symlink(), "star_already_reserved")
        claim = {
            "star": self.star,
            "output": str(self.output.relative_to(self.history)),
            "journal_sha256": self._journal_hash,
            "class_source": self._class_source,
            "automatic_retry": False,
        }
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(claim, stream, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        self._claim_path, self._claim_hash = path, _sha(path.read_bytes())

    def _validate_child(self, child):
        _require(
            isinstance(child, dict)
            and isinstance(child.get("star"), str)
            and child["star"].casefold() == self.star.casefold(),
            "component_star_changed",
        )

    def _tree(self, directory):
        """Pin an already completed local component, including additions/stops."""
        directory = self._owned(directory)
        _require(directory.is_dir(), "missing_positive_source")
        paths = sorted(directory.rglob("*"))
        _require(len(paths) <= 3000, "positive_source_limit")
        hashes, total = {}, 0
        for path in paths:
            path = self._owned(path)
            _require(
                path.name
                not in {
                    "stopped.json",
                    "invalidated.json",
                    "event-forwarding-failed.json",
                    "finalization_failed.json",
                    "event_forward_failed.json",
                }
                and not path.name.endswith("-stopped.json"),
                "failed_positive_source",
            )
            if path.is_file():
                size = path.stat().st_size
                total += size
                _require(size <= 32_000_000 and total <= 128_000_000, "positive_source_limit")
                hashes[str(path.relative_to(self.history))] = _sha(path.read_bytes())
        _require(bool(hashes), "missing_positive_source")
        return hashes

    def _pin_positive_handoff(self, child):
        """Freeze successful child artifacts; no class or workflow is inferred."""
        paths = child.get("artifact_paths", {})
        expected = {
            "numeric_dir": self.output / "star/numeric",
            "color_dir": self.output / "star/color",
            "class_dir": self._class_options["class_dir"],
            "positive_dir": self.output / "star/positive",
            "raw_dir": self.output / "star/positive/raw",
            "derived_dir": self.output / "star/positive/derived",
        }
        _require(
            all(key in paths and self._owned(paths[key]) == value for key, value in expected.items()),
            "positive_source_paths_changed",
        )
        book = _Evidence(self.history)
        stellar = _stellar_sources(
            book,
            *(expected[key] for key in ("numeric_dir", "color_dir", "class_dir")),
            **source_options(self._uses_supplied_inputs()),
        )
        _require(stellar["star"].casefold() == self.star.casefold(), "positive_source_star_changed")
        _require(book.json(self.output / "star/report.json") == child, "star_report_changed")
        positive = book.json(expected["positive_dir"] / "report.json")
        _require(
            positive == child.get("component")
            and positive.get("star", "").casefold() == self.star.casefold()
            and positive.get("mode")
            == (
                SUPPLIED_OWNER_MODE
                if self._uses_supplied_inputs()
                else "positive_planet_reference_to_frozen_derived"
            )
            and positive.get("phase") == "planet_classification_required"
            and positive.get("finished") is True
            and positive.get("derived_transport_verified") is True
            and positive.get("task_completed") is False
            and positive.get("failure_reason") is None
            and positive.get("event_forwarding_failed") is False,
            "positive_handoff_unverified",
        )
        trees = {str(path.relative_to(self.history)): self._tree(path) for path in expected.values()}
        for kind in ("presence", "raw", "derived"):
            completed = book.json(expected["positive_dir"] / (kind + "-completed.json"))
            path = (
                expected["positive_dir"] / kind / ("confirmed.json" if kind == "presence" else "report.json")
            )
            tree = self._tree(path.parent)
            _require(
                completed.get("report") == positive.get("child_artifacts", {}).get(kind)
                and completed.get("files") == tree
                and completed["report"]
                == {
                    "path": str(path.relative_to(self.history)),
                    "sha256": _sha(book.read(path)),
                    "files": len(tree),
                },
                "positive_completed_source_changed",
            )
        self._derived_capture = book.capture(expected["derived_dir"] / "native-copies/copy-04-after")
        if self._uses_supplied_inputs():
            from .browser_supplied_provenance import validate_supplied_derived_provenance

            supplied = validate_supplied_derived_provenance(
                book, expected["derived_dir"], book.json(expected["derived_dir"] / "report.json")
            )
            _require(
                supplied["star"].casefold() == self.star.casefold()
                and supplied["supplied_star_class"] == self._class_options["selected_class"],
                "positive_supplied_source_changed",
            )
        _require(
            _planet(self._derived_capture)["star_name"].casefold() == self.star.casefold(),
            "positive_source_star_changed",
        )
        book.unchanged()
        source = {
            "directories": {key: str(path.relative_to(self.history)) for key, path in expected.items()},
            "hashes": dict(book.hashes),
            "trees": trees,
            "task_completed": False,
            "classification_learned": False,
            **self._supplied_closures(book),
        }
        path = self.output / "positive-handoff.json"
        persist_json(path, source)
        source["hashes"][str(path.relative_to(self.history))] = _sha(path.read_bytes())
        self._positive_source = source
        self._check()

    def provide_planet_class(self, name, rationale):
        """Record an explicit reference decision offline; native work needs a step."""
        self._not_busy()
        _require(
            self.status == "paused" and self.phase == "awaiting_planet_class",
            "planet_class_handoff_not_ready",
        )
        try:
            self._check()
            _require(name in PLANET_CLASSES, "unsupported_planet_class")
            _require(
                isinstance(rationale, str) and bool(rationale.strip()) and len(rationale) <= 1000,
                "planet_class_rationale_required",
            )
            request = {
                "name": name,
                "rationale": rationale.strip(),
                "star": self.star,
                "action_source": "reference_diagnostic",
                "classification_learned": False,
                "scientific_verified": False,
                "task_completed": False,
                "browser_writes": 0,
                "inherited_paint_revision_allowed": False,
            }
            path = self.output / "planet-class-handoff.json"
            persist_json(path, request)
            self._planet_class_request = {
                **request,
                "hashes": {str(path.relative_to(self.history)): _sha(path.read_bytes())},
            }
            self.phase = "selecting_planet_class"
        except Exception as exc:  # noqa: BLE001 - source details stay out of protocol errors
            self._stop(_reason(exc))
        return self._publish() if not self.finished else self.state()

    def _continuation_check(self, *, allow_child_error=False):
        _require(not self.finished, "already_stopped")
        self._check(allow_child_error=allow_child_error)
        _require(self._clock() - self._started_at < self.max_seconds, "time_limit")

    def _class_dispatch_guard(self, capture, mapping, choices):
        self._continuation_check()
        _require(
            mapping.get("star_name", "").casefold() == self.star.casefold(),
            "planet_class_current_star_changed",
        )
        _require(
            choices.get("selected") is None
            and planet_projection(capture, mapping)
            == planet_projection(self._derived_capture, _planet(self._derived_capture)),
            "planet_class_current_source_changed",
        )

    def _pin_planet_class(self, receipt):
        directory = self.output / "planet-class"
        book = _Evidence(self.history)
        _require(
            book.json(directory / "confirmed.json") == receipt
            and receipt.get("value") == self._planet_class_request["name"],
            "planet_class_receipt_changed",
        )
        paths = self._positive_source["directories"]
        planet = _planet_sources(
            book,
            *(self._owned(paths[key]) for key in ("raw_dir", "derived_dir")),
            directory,
            **source_options(self._uses_supplied_inputs()),
        )
        _require(
            planet["star"].casefold() == self.star.casefold()
            and planet["planet_class"] == self._planet_class_request["name"],
            "planet_class_source_changed",
        )
        book.unchanged()
        self._planet_class_source = {
            "directory": str(directory.relative_to(self.history)),
            "sha256": _sha((directory / "confirmed.json").read_bytes()),
            "hashes": dict(book.hashes),
            "trees": {str(directory.relative_to(self.history)): self._tree(directory)},
            **self._supplied_closures(book),
        }
        self._check()

    def _workflow_ready(self, path):
        path = self._owned(path)
        receipt = json.loads(path.read_bytes())
        from .browser_autosave import validate_autosave_workflow_flags

        autosave = self._save_strategy == "autosave"
        validate_autosave_workflow_flags(receipt, enabled=autosave)
        if autosave:
            _require(
                receipt.get("save_acknowledgement_verified") is False
                and receipt.get("source_save_click_delivered") is False,
                "autosave_workflow_authority_changed",
            )
        _require(
            type(receipt.get("schema_version")) is int
            and receipt["schema_version"] == 1
            and receipt.get("mode") in _WORKFLOW_PHASES
            and receipt.get("authority") == "visible_workflow_readback"
            and receipt.get("task_completed") is True
            and isinstance(receipt.get("star"), str)
            and receipt["star"].casefold() == self.star.casefold(),
            "unsupported_verified_workflow",
        )
        self._workflow = {
            "path": str(path.relative_to(self.history)),
            "sha256": _sha(path.read_bytes()),
            "mode": receipt["mode"],
        }
        if self.automatic_inventory:
            self.phase = "inventory_initializing"
        else:
            self.status, self.phase = "paused", "awaiting_inventory"
        self._relay.retire()

    def _completion_save_authority(self, report, child, *, terrestrial=False):
        """Parent completion must retain the selected readback authority.

        A visible-only child completion is not an acknowledged Save. The full
        workflow/source validator still runs at the inventory/import boundary.
        Legacy explicit reports retain their historical field requirements.
        """
        from .browser_autosave import validate_autosave_workflow_flags

        autosave = self._save_strategy == "autosave"
        validate_autosave_workflow_flags(report, enabled=autosave)
        if not autosave:
            return report.get("save_acknowledgement_verified") is True
        validate_autosave_workflow_flags(child, enabled=True)
        return (
            report.get("save_acknowledgement_verified") is False
            and child.get("save_acknowledgement_verified") is False
            and child.get("visible_readback_verified") is True
            and type(report.get("save_dispatch_attempts_recorded")) is int
            and report["save_dispatch_attempts_recorded"] == 0
            and (
                terrestrial
                or report.get("save_click_delivery_confirmed") is False
                and report.get("save_may_have_occurred") is False
            )
        )

    def _drive_positive(self):
        self._continuation_check()
        if self.phase == "selecting_planet_class":
            request = self._planet_class_request
            self._emit(
                "action_proposed",
                {
                    "kind": "SELECT",
                    "value": request["name"],
                    "star": self.star,
                    "action_source": "reference_diagnostic",
                    "rationale": request["rationale"],
                    "classification_learned": False,
                    "task_completed": False,
                },
            )
            self._continuation_check()
            receipt = select_planet_class(
                self.page,
                self.config,
                self.output / "planet-class",
                request["name"],
                source="reference_diagnostic",
                before_dispatch=self._class_dispatch_guard,
            )
            self._continuation_check()
            self._pin_planet_class(receipt)
            self._emit(
                "action_result",
                {"receipt": receipt, "classification_learned": False, "task_completed": False},
            )
            self._continuation_check()
            if request["name"] == "terrestrial":
                if self.terrestrial_options is None:
                    self.status, self.phase = "handoff", "terrestrial_workflow_required"
                    self._finish()
                else:
                    self.phase = "terrestrial_initializing"
            else:
                self.phase = "positive_initializing"
            return
        if self.phase == "positive_initializing":
            callback = self._relay.bind("planet.finalization", star=self.star)
            paths = self._positive_source["directories"]
            child = PositiveFinalizationSteps(
                self.page,
                self.config,
                self.output / "positive-finalization",
                run_history=self.history,
                **{
                    key: self._owned(paths[key])
                    for key in ("numeric_dir", "color_dir", "class_dir", "raw_dir", "derived_dir")
                },
                planet_class_dir=self._owned(self._planet_class_source["directory"]),
                planet_class_sha256=self._planet_class_source["sha256"],
                settle_timeout_seconds=self.positive_save_settle_seconds,
                settle_reserved_notice=True,
                **({"save_strategy": "autosave"} if self._save_strategy == "autosave" else {}),
                **source_options(self._uses_supplied_inputs()),
                emit=callback,
            )
            self.positive_component = child
            if self.finished:
                self._closing = True
                try:
                    child.abort()
                finally:
                    self._closing = False
                return
            self.phase = "positive_active"
        else:
            self.positive_component.advance()
        if self.finished:
            return
        child = self.positive_component.state()
        self._validate_child(child)
        self._positive_state = deepcopy(child)
        self._continuation_check()
        if self.positive_component.finished:
            report_path = self.output / "positive-finalization/report.json"
            report = json.loads(self._owned(report_path).read_bytes())
            _require(
                report == self.positive_component.report
                and type(report.get("schema_version")) is int
                and report["schema_version"] == 1
                and report.get("mode") == "cooperative_positive_planet_finalization"
                and report.get("star", "").casefold() == self.star.casefold()
                and report.get("planet_class") == self._planet_class_request["name"]
                and report.get("planet_class_sha256") == self._planet_class_source["sha256"]
                and report.get("outcome") == "positive_workflow_verified"
                and report.get("task_completed") is True
                and self._completion_save_authority(report, child)
                and report.get("project_completed") is False
                and child.get("task_completed") is True,
                "positive_finalization_unverified",
            )
            directory = self._owned(report.get("workflow_dir", ""))
            _require(
                directory == self.output / "positive-finalization/workflow"
                and _sha((directory / "confirmed.json").read_bytes()) == report.get("workflow_sha256"),
                "positive_workflow_changed",
            )
            tree = self._tree(report_path.parent)
            self._positive_final_source = {
                "hashes": tree,
                "trees": {str(report_path.parent.relative_to(self.history)): tree},
            }
            self._workflow_ready(directory / "confirmed.json")
            _require(
                self._workflow["mode"] == positive_mode(self._uses_supplied_inputs()), "workflow_mode_changed"
            )

    def _drive_terrestrial(self):
        self._continuation_check()
        _require(self.terrestrial_options is not None, "terrestrial_not_enabled")
        if self.phase == "terrestrial_initializing":
            paths = self._positive_source["directories"]
            options = dict(self.terrestrial_options)
            supplied_gate = options.pop("supplied_evaluation", None)
            if self._uses_supplied_inputs():
                _require(supplied_gate is not None, "supplied_temperature_gate_required")
                options["final_evaluation"] = supplied_gate
                options.update(source_options(True))
            options["max_seconds"] = min(
                options["max_seconds"], self.max_seconds - (self._clock() - self._started_at)
            )
            callback = self._relay.bind("planet.terrestrial", star=self.star)
            child = TerrestrialSteps(
                self.page,
                self.config,
                self.output / "terrestrial",
                run_history=self.history,
                **{
                    key: self._owned(paths[key])
                    for key in ("numeric_dir", "color_dir", "class_dir", "raw_dir", "derived_dir")
                },
                planet_class_dir=self._owned(self._planet_class_source["directory"]),
                planet_class_sha256=self._planet_class_source["sha256"],
                settle_reserved_notice=True,
                **({"save_strategy": "autosave"} if self._save_strategy == "autosave" else {}),
                emit=callback,
                **options,
            )
            self.terrestrial_component = child
            if self.finished:
                self._closing = True
                try:
                    child.abort()
                finally:
                    self._closing = False
                return
            self.phase = "terrestrial_active"
        else:
            self.terrestrial_component.advance()
        if not self.finished:
            self._sync_terrestrial()

    def _sync_terrestrial(self):
        child = self.terrestrial_component.state()
        self._validate_child(child)
        self._terrestrial_state = deepcopy(child)
        self._continuation_check()
        _require(child.get("planet_class") == "terrestrial", "terrestrial_class_changed")
        if not self.terrestrial_component.finished:
            _require(
                child.get("finished") is False
                and child.get("task_completed") is False
                and child.get("failure_reason") is None,
                "terrestrial_component_unverified",
            )
            if child.get("phase") in _TERRESTRIAL_HANDOFFS:
                self.status, self.phase = "paused", child["phase"]
            return
        report_path = self.output / "terrestrial/report.json"
        report = json.loads(self._owned(report_path).read_bytes())
        _require(
            report == self.terrestrial_component.report
            and all(report.get(key) == value for key, value in child.items())
            and type(report.get("schema_version")) is int
            and report["schema_version"] == 1
            and report.get("mode") == TERRESTRIAL_MODE
            and report.get("outcome") == "verified_terrestrial"
            and report.get("task_completed") is True
            and report.get("failure_reason") is None
            and self._completion_save_authority(report, child, terrestrial=True)
            and type(report.get("save_dispatch_attempts_recorded")) is int
            and report["save_dispatch_attempts_recorded"] == (0 if self._save_strategy == "autosave" else 1)
            and all(
                report.get(key) is False
                for key in (
                    "project_completed",
                    "scientific_verified",
                    "gas_identification_learned",
                    "habitability_decision_learned",
                    "cleanup_failed",
                )
            ),
            "terrestrial_finalization_unverified",
        )
        declared = report.get("source_sha256")
        _require(isinstance(declared, dict) and 1 <= len(declared) <= 3000, "missing_terrestrial_sources")
        hashes = {}
        for name, expected in declared.items():
            _require(
                isinstance(name, str)
                and isinstance(expected, str)
                and re.fullmatch(r"[a-f0-9]{64}", expected)
                and _sha(self._owned(name).read_bytes()) == expected,
                "terrestrial_source_changed",
            )
            hashes[str(self._owned(name).relative_to(self.history))] = expected
        directory = self._owned(report.get("workflow_dir", ""))
        _require(
            directory == self.output / "terrestrial/workflow"
            and _sha((directory / "confirmed.json").read_bytes()) == report.get("workflow_sha256")
            and _sha((report_path.parent / "events.jsonl").read_bytes()) == report.get("events_sha256"),
            "terrestrial_workflow_changed",
        )
        tree = self._tree(report_path.parent)
        self._terrestrial_final_source = {
            "hashes": {**hashes, **tree},
            "trees": {str(report_path.parent.relative_to(self.history)): tree},
        }
        if self._uses_supplied_inputs():
            from .browser_supplied_provenance import validate_supplied_temperature_provenance

            book = _Evidence(self.history)
            temperature = report_path.parent / "temperature"
            validate_supplied_temperature_provenance(
                book, temperature, book.json(temperature / "report.json")
            )
            self._terrestrial_final_source["hashes"].update(book.hashes)
            self._terrestrial_final_source.update(self._supplied_closures(book))
        self._workflow_ready(directory / "confirmed.json")
        _require(
            self._workflow["mode"] == terrestrial_mode(self._uses_supplied_inputs()), "workflow_mode_changed"
        )

    def _provide_terrestrial_decision(self, kind, decision, *, autonomous=False):
        self._not_busy()
        phase = "awaiting_gases" if kind == "gas" else "awaiting_habitability"
        _require(
            self.status == "paused" and self.phase == phase and self.terrestrial_component is not None,
            "terrestrial_decision_handoff_not_ready",
        )
        self._busy = True
        try:
            self._continuation_check()
            method = (
                self.terrestrial_component.provide_autonomous_gases
                if autonomous and kind == "gas"
                else self.terrestrial_component.provide_gases
                if kind == "gas"
                else self.terrestrial_component.provide_habitability
            )
            method(**decision)  # Child validation and decision recording are offline.
            if not self.finished:
                self._sync_terrestrial()
                _require(not self.terrestrial_component.finished, "terrestrial_decision_failed")
                expected_phase = "select_gases" if kind == "gas" else "select_habitability"
                _require(
                    self._terrestrial_state.get("phase") == expected_phase, "terrestrial_decision_failed"
                )
                path = self.output / "terrestrial" / (kind + "-decision.json")
                self._terrestrial_decisions[kind] = {
                    "hashes": {str(path.relative_to(self.history)): _sha(path.read_bytes())}
                }
                self._continuation_check()
                self.phase = "terrestrial_active"
        except Exception as exc:  # noqa: BLE001 - explicit decision errors stay sanitized
            if not self.finished:
                self._stop(_reason(exc))
        finally:
            self._busy = False
        return self._publish() if not self.finished else self.state()

    def provide_gases(self, *, gases, rationale, supplied_greenhouse_increment):
        return self._provide_terrestrial_decision(
            "gas",
            {
                "gases": gases,
                "rationale": rationale,
                "supplied_greenhouse_increment": supplied_greenhouse_increment,
            },
        )

    def provide_habitability(self, *, choice, rationale):
        return self._provide_terrestrial_decision("habitability", {"choice": choice, "rationale": rationale})

    def provide_autonomous_gases(self, *, gases, rationale):
        """Use observed post-selection absorption; never supply a guessed increment."""
        return self._provide_terrestrial_decision(
            "gas", {"gases": gases, "rationale": rationale}, autonomous=True
        )

    def _star_ready_to_advance(self):
        """Only an explicit read-only hint may defer counting a child work step."""
        if self.phase == "active" and self.component is not None:
            ready = getattr(self.component, "ready_to_advance", None)
            if ready is not None:
                _require(callable(ready), "invalid_component_readiness")
                result = ready()
                _require(type(result) is bool, "invalid_component_readiness")
                return result
        return True

    def _drive(self):
        self._not_busy()
        if self.finished:
            return self.state()
        _require(self.phase in _DRIVABLE, "handoff_required")
        self._busy = True
        try:
            self._check()
            now = self._clock()
            if self._started_at is None:
                self._started_at = now
            _require(now - self._started_at < self.max_seconds, "time_limit")
            self._last_tick = now
            # Waiting for a scheduled chart poll is not a learned decision or a
            # browser operation. Source/cancellation and wall-time guards above
            # still run on every tick; the child marks its original deadline due.
            if self._star_ready_to_advance() and not self.finished:
                _require(self.advances < self.max_advances, "advance_limit")
                self.advances += 1
                if self.phase in _INVENTORY_PHASES:
                    self._drive_inventory()
                elif self.phase in _POSITIVE_PHASES:
                    self._drive_positive()
                elif self.phase in _TERRESTRIAL_PHASES:
                    self._drive_terrestrial()
                else:
                    self._drive_star()
        except (KeyboardInterrupt, SystemExit):
            self._stop("operator_aborted", aborted=True)
            raise
        except Exception as exc:  # noqa: BLE001 - protocol boundary redacts driver errors
            if not self.finished:
                self._stop(_reason(exc))
        finally:
            self._busy = False
        return self._publish() if not self.finished else self.state()

    def _drive_star(self):
        if self.component is None:
            self._claim()
            self.phase = "active"
            callback = self._relay.bind("star.workflow", star=self.star)
            child = self._factory(
                self.page,
                self.config,
                self.output / "star",
                run_history=self.history,
                star=self.star,
                **self._class_options,
                **self.model_options,
                emit=callback,
            )
            self.component = child
            if self.finished:
                self._closing = True
                try:
                    child.abort()
                finally:
                    self._closing = False
                return self.state()
        else:
            self.component.advance()
        if self.finished:
            return self.state()
        child = self.component.state()
        self._validate_child(child)
        self._child_state = deepcopy(child)
        self._check()
        if self.component.finished:
            phase = child.get("phase")
            if phase == "verified_no_planet" and child.get("task_completed") is True:
                path = self._owned(child.get("artifact_paths", {}).get("workflow_dir", "")) / "confirmed.json"
                _require(
                    path.parent == self.output / "star/workflow" and path.is_file(),
                    "missing_verified_workflow",
                )
                self._workflow_ready(path)
                _require(
                    self._workflow["mode"] == "no_planet_visible_workflow_readback", "workflow_mode_changed"
                )
            elif (
                self.reference_planet_continuation
                and phase == "planet_classification_required"
                and child.get("task_completed") is False
            ):
                self._pin_positive_handoff(child)
                self.status, self.phase = "paused", "awaiting_planet_class"
                self._relay.retire()
            elif phase in _HANDOFFS and child.get("task_completed") is False:
                self.status, self.phase = "handoff", phase
                self._finish()
            else:
                raise BrowserSafetyStop("project_steps_component_unverified")

    def _drive_inventory(self):
        _require(self.automatic_inventory and self._workflow is not None, "automatic_inventory_not_ready")
        if self.phase == "inventory_initializing":
            names = [star.name for star in self._progress.stars.values()]
            if self.star.casefold() not in {name.casefold() for name in names}:
                names.append(self.star)
            self._inventory_expected = tuple(names)
            callback = self._relay.bind("project.inventory", star=self.star)
            remaining = self.max_seconds - (self._clock() - self._started_at)
            _require(remaining > 0, "time_limit")
            paginated = len(self._inventory_expected) > 10
            child = ProjectInventorySteps(
                self.page,
                self.config,
                self.output / "collection",
                run_history=self.history,
                workflow_dir=self._owned(self._workflow["path"]).parent,
                workflow_sha256=self._workflow["sha256"],
                expected_star=self.star,
                expected_stars=self._inventory_expected,
                max_seconds=min(420 if paginated else 180, remaining),
                **({"inventory_mode": "live_paginated"} if paginated else {}),
                emit=callback,
                cancelled=lambda: self.finished,
            )
            self.inventory_component = child
            if self.finished:
                self._closing = True
                try:
                    child.abort()
                finally:
                    self._closing = False
                return
            self.phase = "inventory_active"
        elif self.phase == "inventory_active":
            self.inventory_component.advance()
        else:
            _require(self._inventory_source is not None, "missing_inventory_source")
            self._import_inventory(
                self._owned(self._inventory_source["inventory_dir"]),
                self._inventory_source["inventory_sha256"],
            )
            return
        if self.finished:
            return
        child = self.inventory_component.state()
        self._validate_child(child)
        self._inventory_state = deepcopy(child)
        self._check()
        _require(self._clock() - self._started_at < self.max_seconds, "time_limit")
        if self.inventory_component.finished:
            if child.get("status") != "completed" and child.get("failure_reason"):
                raise BrowserSafetyStop(_reason(BrowserSafetyStop(child["failure_reason"])))
            _require(
                child.get("status") == "completed"
                and child.get("phase") == "inventory_verified"
                and child.get("collection_count_verified") is True
                and child.get("task_completed") is False
                and child.get("project_completed") is False,
                "inventory_component_unverified",
            )
            self._pin_inventory(child)
            self.phase = "inventory_import"
            self._relay.retire()

    def _pin_inventory(self, child):
        inventory = self._owned(child.get("inventory_dir", ""))
        _require(inventory == self.output / "collection/inventory", "inventory_output_changed")
        report_path = self.output / "collection/report.json"
        report = json.loads(self._owned(report_path).read_bytes())
        _require(
            report == self.inventory_component.report
            and all(report.get(key) == value for key, value in child.items())
            and report.get("expected_stars") == list(self._inventory_expected)
            and report.get("workflow_sha256") == self._workflow["sha256"],
            "inventory_report_changed",
        )
        declared = report.get("source_sha256")
        _require(isinstance(declared, dict) and 1 <= len(declared) <= 3000, "missing_inventory_source_hashes")
        hashes = {}
        for name, expected in declared.items():
            _require(
                isinstance(name, str)
                and not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and isinstance(expected, str)
                and re.fullmatch(r"[a-f0-9]{64}", expected),
                "invalid_inventory_source_hash",
            )
            _require(_sha(self._owned(name).read_bytes()) == expected, "inventory_source_changed")
            hashes[name] = expected
        receipt_path = str((inventory / "confirmed.json").relative_to(self.history))
        _require(
            hashes.get(self._workflow["path"]) == self._workflow["sha256"]
            and hashes.get(receipt_path) == child.get("inventory_sha256")
            and isinstance(child.get("inventory_sha256"), str),
            "inventory_receipt_not_pinned",
        )
        hashes[str(report_path.relative_to(self.history))] = _sha(report_path.read_bytes())
        binding_fields = {}
        if len(self._inventory_expected) > 10:
            _require(
                report.get("mode") == "cooperative_live_paginated_project_inventory"
                and report.get("inventory_mode") == "live_paginated"
                and report.get("max_advances") == 11
                and type(report.get("max_seconds")) in {int, float}
                and 0 < report["max_seconds"] <= 420
                and report.get("max_navigation_clicks") == 2
                and report.get("max_inventory_calls") == 1
                and report.get("max_paginated_seconds") == 240
                and report.get("max_paginated_advances") == 8
                and report.get("max_pagination_clicks") == 4
                and report.get("required_initial_page") == 1
                and report.get("pagination_supported") is True,
                "paginated_inventory_scope_changed",
            )
            book = _Evidence(self.history)
            source = load_inventory_source(book, inventory, expected_sha256=child["inventory_sha256"])
            binding_fields = source.binding_fields()
            _require(
                source.kind == "live_paginated"
                and {row["name"].casefold() for row in source.rows}
                == {name.casefold() for name in self._inventory_expected}
                and all(report.get(key) == value for key, value in binding_fields.items())
                and all(hashes.get(name) == checksum for name, checksum in book.hashes.items()),
                "paginated_inventory_binding_changed",
            )
        handoff = {
            "inventory_dir": str(inventory.relative_to(self.history)),
            "inventory_sha256": child["inventory_sha256"],
            "hashes": hashes,
            "expected_stars": list(self._inventory_expected),
            "automatic_inventory": True,
            "task_completed": False,
            "project_completed": False,
            **binding_fields,
        }
        path = self.output / "inventory-handoff.json"
        persist_json(path, handoff)
        hashes[str(path.relative_to(self.history))] = _sha(path.read_bytes())
        self._inventory_source = deepcopy(handoff)
        self._check()

    def step(self):
        if self.finished:
            return self.state()
        _require(self.status == "paused", "pause_before_step")
        return self._drive()

    def tick(self):
        if self.status != "running":
            return self.state()
        return self._drive()

    def advance_if_due(self):
        if self.status == "running" and (
            self._last_tick is None or self._clock() - self._last_tick >= self.interval
        ):
            return self.tick()
        return self.state()

    def provide_inventory(self, *, inventory_dir, inventory_sha256):
        """Explicit offline import only; never navigate to discover an inventory."""
        self._not_busy()
        _require(
            self.status == "paused" and self.phase == "awaiting_inventory", "inventory_handoff_not_ready"
        )
        try:
            self._import_inventory(self._owned(inventory_dir), inventory_sha256)
        except Exception as exc:  # noqa: BLE001 - importer failures never expose private artifacts
            self._stop(_reason(exc))
        return self.state()

    def _import_inventory(self, inventory, inventory_sha256):
        self._check()
        _require(
            isinstance(inventory_sha256, str)
            and re.fullmatch(r"[a-f0-9]{64}", inventory_sha256)
            and _sha((inventory / "confirmed.json").read_bytes()) == inventory_sha256,
            "inventory_hash_changed",
        )
        workflow = self._owned(self._workflow["path"])
        receipt = json.loads(workflow.read_bytes())
        mode = receipt.get("mode")
        _require(mode == self._workflow["mode"] and mode in _WORKFLOW_PHASES, "workflow_mode_changed")
        importer = {
            "no_planet_visible_workflow_readback": import_verified_no_planet,
            "positive_planet_visible_workflow_readback": import_verified_positive_planet,
            "terrestrial_visible_workflow_readback": import_verified_terrestrial,
            POSITIVE_SUPPLIED_MODE: import_verified_positive_planet,
            TERRESTRIAL_SUPPLIED_MODE: import_verified_terrestrial,
        }[mode]
        result = importer(
            self.journal,
            self.history,
            inventory,
            workflow.parent,
        )
        self._progress, self._journal_hash = self._read_journal()
        matches = [s for s in self._progress.stars.values() if s.name.casefold() == self.star.casefold()]
        _require(len(matches) == 1 and matches[0].task_completed, "import_did_not_complete_star")
        persist_json(self.output / "workflow-import.json", result)
        self.status, self.phase = "completed", _WORKFLOW_PHASES[mode]
        self._finish()

    def _finish(self):
        if self.report is not None or self._finalizing:
            return
        self._finalizing = True
        try:
            report = self.state()
            try:
                self._emit("episode_summary", {**report, "completed": report["task_completed"]})
            except Exception:  # noqa: BLE001 - final callback failure remains terminal
                self.status, self.phase, self.failure = (
                    "stopped",
                    "stopped",
                    "project_steps_event_forwarding_failed",
                )
            if self._forward_failed:
                self.status, self.phase, self.failure = (
                    "stopped",
                    "stopped",
                    "project_steps_event_forwarding_failed",
                )
            current = self.state()
            if current != report:
                # Append the final truthful owner state if delivery/cancellation
                # changed the outcome after the immutable summary was flushed.
                self._emit("state", current)
            report = current
            self.report = report
            persist_json(self.output / "report.json", report)
        finally:
            self._relay.retire()
            self._stream.close()
            self._finalizing = False

    def _record_failed_continuation_save(self):
        """Read bounded failure evidence, not a Save/workflow verifier or retry.

        Child counters alone are not proof. Failed-source readers intentionally
        do not use the successful workflow loader, which rejects stopped files.
        A genuine Save followed by a different workflow failure is not labelled
        an uncertain Save. Missing or malformed failure evidence remains unknown.
        """
        if self._save_strategy == "autosave":
            # This branch never calls the explicit Save actuator. A failed
            # readback is neither a Save attempt nor a persistence receipt.
            return
        terrestrial = self.terrestrial_component is not None
        component = self.terrestrial_component if terrestrial else self.positive_component
        if component is None:
            return
        owner_dir = self.output / ("terrestrial" if terrestrial else "positive-finalization")
        directory = owner_dir / "save"
        if not directory.exists() and not directory.is_symlink():
            return
        summary = {
            "scope": "noncanonical_child_save_diagnostic_v1",
            "star": self.star,
            "save_dir": str(directory.relative_to(self.history)),
            "save_surface": "habitability" if terrestrial else "planet",
            "evidence_status": "unavailable_or_invalid",
            "save_outcome_uncertain": None,
            "reservation_retained": None,
            "dispatch_recorded": None,
            "acknowledgement_recorded": None,
            "final_readback_verified": False,
            "canonical_receipt": False,
            "task_completed": False,
            "project_completed": False,
            "automatic_retry": False,
            "source_sha256": {},
        }
        self._continuation_save_outcome = summary
        hashes = {}

        def reject(*_):
            raise ValueError("invalid_failed_save_diagnostic")

        def same(left, right):
            return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
                right, sort_keys=True, allow_nan=False
            )

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    reject()
                result[key] = value
            return result

        def record(path):
            if not path.resolve().is_relative_to(self.history) or any(
                p.is_symlink() for p in (path, *path.parents)
            ):
                reject()
            if not path.exists():
                return None
            if not path.is_file() or not 0 < path.stat().st_size <= 64_000:
                reject()
            with path.open("rb") as stream:
                raw = stream.read(64_001)
            if len(raw) > 64_000:
                reject()
            value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
            if not isinstance(value, dict):
                reject()
            hashes[str(path.relative_to(self.history))] = _sha(raw)
            return value

        def star_matches(value):
            return isinstance(value, str) and value.casefold() == self.star.casefold()

        try:
            if (directory / "invalidated.json").exists() or Path(component.output) != owner_dir:
                reject()
            stopped, confirmed = [record(directory / name) for name in ("stopped.json", "confirmed.json")]
            if stopped is None and isinstance(confirmed, dict):
                saved = getattr(component, "save_receipt", None)
                if (
                    isinstance(saved, dict)
                    and same(confirmed, saved)
                    and star_matches(confirmed.get("star"))
                    and confirmed.get("save_click_delivered") is True
                    and confirmed.get("data_saved_notice_observed") is True
                    and confirmed.get("answers_unchanged") is True
                    and confirmed.get("task_completed") is False
                ):
                    self._continuation_save_outcome = None
                    return
            if (
                not isinstance(stopped, dict)
                or confirmed is not None
                or type(stopped.get("reservation_created")) is not bool
                or type(stopped.get("save_may_have_occurred")) is not bool
                or stopped.get("automatic_retry") is not False
                or stopped.get("task_completed") is not False
                or terrestrial
                and stopped.get("mode") != "terrestrial_habitability_save"
            ):
                reject()
            intent = record(directory / "reserved.json")
            key = hashlib.sha256(self.star.casefold().encode()).hexdigest() + ".json"
            claim = record(
                self.history
                / ("habitability-save-reservations" if terrestrial else "positive-finalization-reservations")
                / key
            )
            dispatch = record(directory / "dispatch.json")
            ack = record(directory / "acknowledgement.json")
            admission = record(directory / "dispatch-budget-rejected.json")
            if admission is not None and (
                admission.get("estimate_not_guarantee") is not True
                or any(
                    admission.get(k) is not False
                    for k in (
                        "deadline_extended",
                        "save_click_dispatched",
                        "automatic_retry",
                        "task_completed",
                    )
                )
                or ack is not None
                # The final admission check can reject after the durable
                # pre-click marker. Preserve its conservative uncertainty;
                # this diagnostic never establishes whether a click occurred.
                or stopped["save_may_have_occurred"] is not (dispatch is not None)
            ):
                reject()
            if stopped["reservation_created"]:
                if (
                    not isinstance(intent, dict)
                    or not isinstance(claim, dict)
                    or not star_matches(intent.get("star"))
                    or not star_matches(claim.get("star"))
                    or intent.get("kind") != "CLICK"
                    or intent.get("visible_label") != "Save"
                    or type(intent.get("max_save_clicks")) is not int
                    or intent["max_save_clicks"] != 1
                    or intent.get("automatic_retry") is not False
                    or intent.get("task_completed") is not False
                    or claim.get("automatic_retry") is not False
                    or claim.get("task_completed") is not False
                    or type(claim.get("schema_version")) is not int
                    or claim["schema_version"] != 1
                ):
                    reject()
                class_path = self.output / "planet-class/confirmed.json"
                class_receipt = record(class_path)
                expected_class = (
                    "terrestrial" if terrestrial else (self._planet_class_request or {}).get("name")
                )
                if (
                    not isinstance(self._planet_class_source, dict)
                    or not isinstance(class_receipt, dict)
                    or self._planet_class_source.get("directory")
                    != str(class_path.parent.relative_to(self.history))
                    or self._planet_class_source.get("sha256")
                    != hashes[str(class_path.relative_to(self.history))]
                    or not star_matches(class_receipt.get("star"))
                    or class_receipt.get("value") != expected_class
                    or expected_class not in {"gas_giant", "ice_giant", "terrestrial"}
                ):
                    reject()
                if terrestrial:
                    if (
                        not same(intent, claim)
                        or intent.get("mode") != "terrestrial_habitability_save"
                        or intent.get("output") != summary["save_dir"]
                    ):
                        reject()
                    declared = intent.get("source_sha256")
                    pinned = getattr(getattr(component, "book", None), "hashes", {})
                    for name in ("phase", "choice"):
                        source = owner_dir / name / "confirmed.json"
                        relative = str(source.relative_to(self.history))
                        value = record(source)
                        if (
                            intent.get(name + "_dir") != str(source.parent.relative_to(self.history))
                            or not isinstance(value, dict)
                            or not star_matches(value.get("star"))
                            or value.get("readback_verified") is not True
                            or not isinstance(declared, dict)
                            or declared.get(relative) != hashes.get(relative)
                            or pinned.get(relative) != hashes.get(relative)
                            or name == "choice"
                            and value.get("choice") != intent.get("choice")
                        ):
                            reject()
                elif (
                    claim.get("mode") != "cooperative_positive_planet_finalization"
                    or claim.get("output") != str(owner_dir.relative_to(self.history))
                    or claim.get("save_output") != summary["save_dir"]
                    or claim.get("planet_class") != expected_class
                    or claim.get("planet_class_sha256") != self._planet_class_source["sha256"]
                    or not same(claim.get("native_intent"), intent)
                    or type(claim.get("maximum_save_dispatches")) is not int
                    or claim["maximum_save_dispatches"] != 1
                ):
                    reject()
            elif intent is not None or claim is not None or stopped["save_may_have_occurred"]:
                reject()
            if dispatch is not None and not same(
                dispatch, {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1}
            ):
                reject()
            if ack is not None and (
                dispatch is None
                or not same(
                    ack,
                    {
                        "visible_text": "Data saved",
                        "source": "fully_exposed_footer_text",
                        "notice_was_already_present": False,
                    },
                )
            ):
                reject()
            if not stopped["save_may_have_occurred"] and (dispatch is not None or ack is not None):
                reject()
            # A marker alone never claims that the click returned or readback
            # succeeded. A fresh notice still cannot complete this failed task.
            summary.update(
                evidence_status="recorded_stop",
                save_outcome_uncertain=stopped["save_may_have_occurred"],
                reservation_retained=stopped["reservation_created"],
                dispatch_recorded=dispatch is not None,
                acknowledgement_recorded=ack is not None,
                source_sha256=hashes,
            )
        except Exception:  # noqa: BLE001,S110 - preserve the original stop and unknown diagnostic
            pass

    def _stop(self, reason, *, aborted=False):
        if self.report is not None or self._closing:
            return
        self.status, self.phase = ("aborted", "aborted") if aborted else ("stopped", "stopped")
        self.failure, self._closing = (
            (
                self._child_failure
                if self._child_failure and not aborted and not self._forward_failed
                else reason
            ),
            True,
        )
        try:
            for component in (
                self.component,
                self.positive_component,
                self.terrestrial_component,
                self.inventory_component,
            ):
                if component is not None and not component.finished:
                    try:
                        component.abort()
                    except Exception:  # noqa: BLE001 - already terminal; no driver details
                        self._child_abort_failed = True
            try:
                self._record_failed_continuation_save()
            except Exception:  # noqa: BLE001,S110 - diagnostics must not replace the original stop
                pass
            if not self._finalizing:
                if not aborted:
                    self._emit("error", {"type": "BrowserProjectStop", "message": reason})
                self._finish()
        finally:
            self._closing = False

    def abort(self):
        if not self.finished:
            self._stop("operator_aborted", aborted=True)
        return self.state()

    def close(self):
        return self.abort()

    def command(self, message):
        """Existing v1 commands; source handoffs use explicit step payloads."""
        command = RuntimeCommand.model_validate(message)
        payload = command.payload
        if command.command == "start" and set(payload) <= {"paused"}:
            return self.start(**payload)
        if command.command == "step":
            if not payload:
                return self.step()
            if set(payload) == {"class_source"} and isinstance(payload["class_source"], dict):
                return self.provide_class(**payload["class_source"])
            if set(payload) == {"inventory"} and isinstance(payload["inventory"], dict):
                return self.provide_inventory(**payload["inventory"])
            if set(payload) == {"planet_class"} and isinstance(payload["planet_class"], dict):
                return self.provide_planet_class(**payload["planet_class"])
            if set(payload) == {"gases"} and isinstance(payload["gases"], dict):
                return self.provide_gases(**payload["gases"])
            if set(payload) == {"habitability"} and isinstance(payload["habitability"], dict):
                return self.provide_habitability(**payload["habitability"])
        if not payload and command.command in {"pause", "resume", "abort"}:
            return getattr(self, command.command)()
        raise ValueError("Unsupported single-star project command or payload")
