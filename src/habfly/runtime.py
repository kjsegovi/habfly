"""Headless JSONL engine shared by the Rust TUI and recorded demonstrations."""

import json
import queue
import sys
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .contracts import Contract, Observation, RuntimeCommand, RuntimeEvent, StepResult, task_completed

LOCAL_CHECKPOINT_TASKS = {"distance", "luminosity", "temperature", "mass", "radius", "lifetime"}
CALCULATION_TASKS = LOCAL_CHECKPOINT_TASKS | {"stellar", "planet_calculations", "habitability_calculations"}
BROWSER_TASKS = {"browser_numeric", "browser_color", "browser_four_field"}


def _raw_finalizer_replay_identity(item):
    """Recognize a raw owner scope for display only, never a trusted receipt."""
    import re

    payload = item.payload
    digest = payload.get("campaign_handoff_sha256")
    if not (
        payload.get("mode") == "bounded_post_campaign_finalization"
        and payload.get("schema_version") == 1
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        and item.run_id == "project-finalize-" + digest[:16]
        and all(
            type(payload.get(key)) is int
            for key in (
                "schema_version",
                "scoring_max_seconds",
                "submission_max_seconds",
                "scoring_max_advances",
                "submission_max_advances",
            )
        )
        and all(payload.get(key) is False for key in ("task_completed", "project_completed", "submitted"))
        and all(type(payload.get(key)) is bool for key in ("allow_score_transfer", "allow_submission"))
        and (not payload["allow_submission"] or payload["allow_score_transfer"])
        and type(payload.get("data_revision")) is int
        and payload["data_revision"] >= 0
        and all(isinstance(payload.get(key), str) and payload[key] for key in ("project_id", "attempt_id"))
        and payload.get("scoring_max_seconds") == 600
        and payload.get("submission_max_seconds") == 180
        and payload.get("scoring_max_advances") == 46
        and payload.get("submission_max_advances") == 4
    ):
        return None
    return (item.run_id,) + tuple(
        payload[key]
        for key in (
            "project_id",
            "attempt_id",
            "data_revision",
            "campaign_handoff_sha256",
            "allow_score_transfer",
            "allow_submission",
        )
    )


def _raw_finalizer_terminal_summary(item, parent):
    payload = item.payload
    if (
        parent is None
        or _raw_finalizer_replay_identity(item) != parent
        or payload.get("finished") is not True
        or {"component", "component_event", "component_state", "component_summary", "component_hello"}
        & payload.keys()
    ):
        return False
    return (payload.get("status"), payload.get("phase")) in {
        ("handoff", "assessed_score_transfer_disabled"),
        ("handoff", "score_transferred_not_submitted"),
        ("handoff", "unknown_pending"),
        ("stopped", "stopped"),
        ("aborted", "aborted"),
    }


def _campaign_replay_child(payload):
    if {
        "component",
        "component_event",
        "component_state",
        "component_summary",
        "component_hello",
        "ancestry",
    } & payload.keys():
        return True
    wire = payload.get("project_wire")
    return isinstance(wire, dict) and wire.get("ancestry") != []


def _outer_campaign_replay_identity(item):
    """Bind recorded outer display context, never manufacture a task receipt."""
    payload = item.payload
    star, progress = payload.get("current_star"), payload.get("project_progress")
    target, wire = payload.get("target_stars"), payload.get("project_wire")
    if _campaign_replay_child(payload) or not (
        isinstance(item.run_id, str)
        and item.run_id
        and payload.get("mode") == "fresh_browser_campaign_project_runtime"
        and payload.get("task") == payload.get("runtime_task") == "browser_project"
        and payload.get("environment") == "browser"
        and payload.get("project_campaign") is True
        and type(target) is int
        and 2 <= target <= 30
        and type(payload.get("stars")) is int
        and payload["stars"] == target
        and payload.get("task_completed") is False
        and payload.get("project_completed") is False
        and ("submitted" not in payload or payload["submitted"] is False)
        and isinstance(star, dict)
        and isinstance(star.get("star"), str)
        and star["star"]
        and type(star.get("ordinal")) is int
        and 1 <= star["ordinal"] <= target
        and isinstance(progress, dict)
        and all(isinstance(progress.get(key), str) and progress[key] for key in ("project_id", "attempt_id"))
        and type(progress.get("target")) is int
        and progress["target"] == 30
        and progress.get("submitted") is False
        and progress.get("project_completed") is False
        and type(progress.get("revision")) is int
        and progress["revision"] >= 0
    ):
        return None
    if "project_wire" in payload:
        if not (
            isinstance(wire, dict)
            and type(wire.get("version")) is int
            and wire["version"] == 1
            and wire.get("mode") == "compact_project_display"
            and wire.get("ancestry") == []
            and isinstance(wire.get("leaf_header"), dict)
            and wire["leaf_header"].get("event") == item.event
            and wire.get("evidence_receipt") is False
        ):
            return None
    elif payload.get("project_compact_wire") is True:
        return None
    # JSON identity makes bool/int aliases distinct and retains the complete
    # fresh-source context, not just a reusable display name.
    return (
        item.run_id,
        payload["mode"],
        target,
        json.dumps(star, sort_keys=True),
        progress["project_id"],
        progress["attempt_id"],
    ), progress["revision"]


def _outer_campaign_terminal_summary(item, parent):
    current = _outer_campaign_replay_identity(item)
    payload = item.payload
    if (
        parent is None
        or current is None
        or current[0] != parent[0]
        or current[1] < parent[1]
        or payload.get("finished") is not True
        or payload.get("completed") is not False
        or payload.get("browser_phase") != payload.get("phase")
    ):
        return False
    if (payload.get("status"), payload.get("phase")) in {("stopped", "stopped"), ("aborted", "aborted")}:
        return payload.get("target_workflows_verified") is False
    progress = payload["project_progress"]
    return (
        (payload.get("status"), payload.get("phase")) == ("handoff", "awaiting_assessment")
        and payload.get("target_workflows_verified") is True
        and payload.get("failure_reason") is None
        and payload["current_star"]["ordinal"] == payload["target_stars"]
        and all(type(progress.get(key)) is int for key in ("collected", "verified", "unresolved"))
        and progress["collected"] == progress["verified"] == payload["target_stars"]
        and progress["unresolved"] == 0
        and progress.get("uncertain_actions") == []
    )


class ProjectViewport(Contract):
    """Optional fixed CSS-pixel geometry for fresh project browser contexts."""

    width: int = Field(ge=950, le=2400, strict=True)
    height: int = Field(ge=600, le=1800, strict=True)


class RunOptions(Contract):
    seed: int = 0
    stars: int = Field(default=1, ge=1, le=30)
    policy: Literal["expert", "untrained", "checkpoint"] = "expert"
    graph: Path | None = None
    checkpoint: Path | None = None
    content_pack: Path | None = None
    artifact_dir: Path = Path("experiments/runs")
    environment: Literal["simulator", "browser"] = "simulator"
    browser_config: Path | None = None
    browser_setup: Literal["manual", "automatic"] = "manual"
    browser_execution: Literal["supervised", "autonomous"] = "supervised"
    paused: bool = False
    interval: float = Field(default=0.2, ge=0, le=10)
    task: Literal[
        "mini_habworlds",
        "stellar",
        "distance",
        "luminosity",
        "temperature",
        "mass",
        "radius",
        "lifetime",
        "browser_numeric",
        "browser_color",
        "browser_four_field",
        "browser_project",
        "color",
        "planet_calculations",
        "habitability_calculations",
    ] = "mini_habworlds"
    spreadsheet_config: Path | None = None
    calculation_backend: Literal["local", "google_sheets"] | None = None
    knowledge_pack: Path | None = None
    dataset: Path | None = None
    color_checkpoint: Path | None = None
    color_dataset: Path | None = None
    planet_evaluation: Path | None = None
    planet_calibration: Path | None = None
    habitability_evaluation: Path | None = None
    color_experiment: Path | None = None
    browser_planet_pilot: Path | None = None
    browser_planet_final_evaluation: Path | None = None
    browser_planet_supplied_evaluation: Path | None = None
    browser_planet_seed: int = 15000000
    browser_no_planet_save_settle_seconds: float = Field(default=20, ge=0.1, le=30, strict=True)
    browser_positive_save_settle_seconds: float = Field(default=20, ge=0.1, le=30, strict=True)
    project_reference_planet_continuation: bool = Field(default=False, strict=True)
    project_reference_shallow_transits: bool = Field(default=False, strict=True)
    project_two_event_reference: bool = Field(default=False, strict=True)
    project_baseline_edge_reference: bool = Field(default=False, strict=True)
    project_baseline_band_reference: bool = Field(default=False, strict=True)
    project_single_event_reference: bool = Field(default=False, strict=True)
    project_save_strategy: str = Field(default="explicit", strict=True, pattern=r"^(explicit|autosave)$")
    project_autonomous_decisions: bool = Field(default=False, strict=True)
    project_supplied_stellar_inputs: bool = Field(default=False, strict=True)
    project_campaign: bool = Field(default=False, strict=True)
    project_compact_wire: bool = Field(default=False, strict=True)
    project_mascot: bool = Field(default=False, strict=True)
    project_allow_scoring: bool = Field(default=False, strict=True)
    project_allow_submission: bool = Field(default=False, strict=True)
    project_campaign_max_seconds: (
        Annotated[float, Field(gt=0, le=10800, strict=True)] | Literal["uncapped"] | None
    ) = None
    project_viewport: ProjectViewport | None = None
    browser_habitability_pilot: Path | None = None
    browser_habitability_final_evaluation: Path | None = None
    browser_habitability_supplied_evaluation: Path | None = None
    browser_habitability_gas_candidates: (
        list[Literal["CH4", "CO2", "H2O", "H2S", "N2O", "NH3", "O3"]] | None
    ) = None
    browser_habitability_seed: int = Field(default=20000000, strict=True)
    browser_habitability_max_seconds: float = Field(default=1800, gt=0, le=1800, strict=True)
    browser_habitability_save_settle_seconds: float = Field(default=20, ge=0.1, le=30, strict=True)
    project_max_advances: int = Field(default=512, ge=1, le=2048, strict=True)
    project_max_seconds: float = Field(default=1800, ge=1, le=3600)
    project_setup_max_advances: int = Field(default=256, ge=1, le=1024, strict=True)
    project_class_max_advances: int = Field(default=4, ge=1, le=4, strict=True)
    project_class_max_seconds: float = Field(default=180, ge=5, le=600, strict=True)

    @property
    def backend(self):
        return self.calculation_backend or ("google_sheets" if self.spreadsheet_config else "local")


def parse_run_options(payload):
    """Validate project opt-in without echoing private input values in errors.

    Frozen-model provenance is checked by the component loaders before inference;
    this read-only check rejects missing files and incompatible launch settings.
    Existing task validation remains in Runtime.start.
    """
    project = isinstance(payload, dict) and payload.get("task") == "browser_project"
    try:
        options = RunOptions.model_validate(payload)
    except ValueError:
        if project:
            raise ValueError("browser_project_invalid_start_options") from None
        raise
    if not project:
        if options.project_mascot:
            raise ValueError("project_mascot_requires_project_task")
        if options.project_supplied_stellar_inputs or any(
            p is not None
            for p in (
                options.browser_planet_supplied_evaluation,
                options.browser_habitability_supplied_evaluation,
            )
        ):
            raise ValueError("project_supplied_inputs_requires_project_task")
        if options.project_autonomous_decisions:
            raise ValueError("project_autonomous_decisions_requires_project_task")
        if options.project_reference_shallow_transits:
            raise ValueError("project_shallow_reference_requires_project_task")
        if options.project_two_event_reference:
            raise ValueError("project_two_event_reference_requires_project_task")
        if options.project_baseline_edge_reference:
            raise ValueError("project_baseline_edge_reference_requires_project_task")
        if options.project_baseline_band_reference:
            raise ValueError("project_baseline_band_reference_requires_project_task")
        if options.project_single_event_reference:
            raise ValueError("project_single_event_reference_requires_project_task")
        if options.project_save_strategy != "explicit":
            raise ValueError("project_autosave_requires_project_task")
        if options.project_allow_scoring or options.project_allow_submission:
            raise ValueError("project_finalization_requires_thirty_star_campaign")
        if options.project_compact_wire:
            raise ValueError("project_compact_wire_requires_project_task")
        if options.project_viewport is not None:
            raise ValueError("project_viewport_requires_project_task")
        if options.project_campaign or options.project_campaign_max_seconds is not None:
            raise ValueError("browser_project_campaign_requires_project_task")
        return options
    if (options.project_allow_scoring or options.project_allow_submission) and (
        not options.project_campaign
        or options.stars != 30
        or options.project_allow_submission
        and not options.project_allow_scoring
    ):
        raise ValueError("project_finalization_requires_authorized_thirty_star_scoring")
    if options.project_campaign:
        if options.stars not in {2, 3, 30}:
            raise ValueError("browser_project_campaign_requires_two_three_or_thirty_stars")
        if (
            options.project_campaign_max_seconds is None
            or options.project_campaign_max_seconds == "uncapped"
            and options.stars != 30
            or options.project_max_advances != 512
            or options.project_max_seconds != 1800
        ):
            raise ValueError("browser_project_campaign_requires_explicit_fixed_budgets")
    elif options.project_campaign_max_seconds is not None:
        raise ValueError("browser_project_campaign_budget_requires_opt_in")
    if (
        options.environment != "browser"
        or options.policy != "checkpoint"
        or options.browser_setup != "automatic"
        or options.browser_execution != "autonomous"
        or (not options.project_campaign and options.stars != 1)
        or options.backend != "local"
        or any(
            getattr(options, key) is not None
            for key in (
                "spreadsheet_config",
                "content_pack",
                "knowledge_pack",
                "color_checkpoint",
                "color_dataset",
                "planet_evaluation",
                "planet_calibration",
                "habitability_evaluation",
            )
        )
        or (options.browser_planet_pilot is None) != (options.browser_planet_final_evaluation is None)
    ):
        raise ValueError("browser_project_requires_local_automatic_single_star_checkpoint_options")
    terrestrial = (
        options.browser_habitability_pilot,
        options.browser_habitability_final_evaluation,
        options.browser_habitability_gas_candidates,
    )
    supplied_gates = (
        options.browser_planet_supplied_evaluation,
        options.browser_habitability_supplied_evaluation,
    )
    if options.project_supplied_stellar_inputs:
        if (
            not options.project_autonomous_decisions
            or not options.project_reference_planet_continuation
            or options.browser_planet_pilot is None
            or any(item is None for item in (*terrestrial, *supplied_gates))
        ):
            raise ValueError("project_supplied_inputs_requires_complete_autonomous_options")
    elif any(item is not None for item in supplied_gates):
        raise ValueError("project_supplied_gate_paths_require_opt_in")
    if options.project_autonomous_decisions and (
        not options.project_reference_planet_continuation
        or options.browser_planet_pilot is None
        or any(item is None for item in terrestrial)
        or set(options.browser_habitability_gas_candidates or [])
        != {"CH4", "CO2", "H2O", "H2S", "N2O", "NH3", "O3"}
    ):
        raise ValueError("project_autonomous_decisions_requires_complete_planet_options")
    if options.project_reference_shallow_transits and (
        not options.project_reference_planet_continuation or options.browser_planet_pilot is None
    ):
        raise ValueError("project_shallow_reference_requires_explicit_planet_options")
    if options.project_two_event_reference and not options.project_reference_shallow_transits:
        raise ValueError("project_two_event_reference_requires_shallow_reference")
    if options.project_baseline_band_reference and (
        not options.project_baseline_edge_reference
        or not options.project_reference_planet_continuation
        or options.browser_planet_pilot is None
    ):
        raise ValueError("project_baseline_band_reference_requires_edge_and_explicit_planet_options")
    if options.project_single_event_reference and not options.project_baseline_band_reference:
        raise ValueError("project_single_event_reference_requires_baseline_band_reference")
    if any(item is not None for item in terrestrial) and (
        any(item is None for item in terrestrial)
        or not options.project_reference_planet_continuation
        or options.browser_planet_pilot is None
        or not options.browser_habitability_gas_candidates
        or len(set(options.browser_habitability_gas_candidates))
        != len(options.browser_habitability_gas_candidates)
    ):
        raise ValueError("browser_project_requires_explicit_terrestrial_options")
    paths = {
        "graph": "directory",
        "dataset": "directory",
        "checkpoint": "file",
        "color_experiment": "directory",
        "browser_config": "file",
    }
    if options.browser_planet_pilot is not None:
        paths.update(browser_planet_pilot="directory", browser_planet_final_evaluation="directory")
    if options.browser_habitability_pilot is not None:
        paths.update(
            browser_habitability_pilot="directory", browser_habitability_final_evaluation="directory"
        )
    if options.project_supplied_stellar_inputs:
        paths.update(
            browser_planet_supplied_evaluation="directory",
            browser_habitability_supplied_evaluation="directory",
        )
    try:
        for key, kind in paths.items():
            path = getattr(options, key)
            if path is None or not (path.is_dir() if kind == "directory" else path.is_file()):
                raise ValueError("missing")
    except (ValueError, OSError):
        raise ValueError("browser_project_missing_local_model_or_config_path") from None
    try:
        from .browser_policy import browser_config

        browser_config(options)
    except (ValueError, OSError):
        raise ValueError("browser_project_invalid_browser_config") from None
    return options


class Runtime:
    def __init__(self, output=None):
        self.output = output or sys.stdout
        self.sequence = 0
        self.run_id = None
        self.status = "idle"
        self.options = RunOptions()
        self.env = self.policy = self.neural_state = self.observation = None
        self.trace = None
        self.trace_path = None
        self.replay_events = None
        self.replay_context = None
        self._replay_finalizer_parent = None
        self._replay_campaign_parent = None
        self.last_tick = 0.0
        self.spreadsheet_adapter = None
        self.mascot = None
        self.expected_browser_identity = None  # Optional in-process batch constraint, never credentials.
        self.finalization = None
        self._finalization_authorization = None
        self._finalization_pending = self._finalization_attempted = False
        self._finalization_cancelled = self._finalization_busy = self._finalization_emitting = False
        self._finalization_mode = "paused"
        self._finalization_terminal = self._finalization_campaign = None
        self._finalization_authorization_failed = False
        self._finalization_last_phase = None

    def emit(self, event, payload):
        item = RuntimeEvent(event=event, sequence=self.sequence, run_id=self.run_id, payload=payload)
        self.sequence += 1
        line = item.model_dump_json()
        self.output.write(line + "\n")
        self.output.flush()
        if self.trace:
            self.trace.write(line + "\n")
            self.trace.flush()
        if self.mascot is not None and self.replay_context is None and self.options.task != "browser_project":
            self._present_mascot(item)
        return item

    def _present_mascot(self, item):
        try:
            # Display is downstream of the unchanged recorded event. It cannot
            # provide actions or observations and is never installed at login.
            page = None
            frames = ()
            if (
                self.env is not None
                and getattr(self.env, "initial_star", None)
                and not getattr(self.env, "closed", True)
            ):
                candidate = getattr(self.env, "page", None)
                config = getattr(self.env, "config", None)
                if candidate is not None and config is not None and config.allows(candidate.url):
                    page = candidate
                    frames = tuple(rule.url for rule in config.frames)
            self.mascot.consume(item.model_dump(mode="json"), page=page, allowed_frame_urls=frames)
        except Exception:  # noqa: BLE001 - presentation failures must not change task execution
            try:
                self.mascot.close()
            except Exception:  # noqa: BLE001, S110 - private driver text; display cleanup is best-effort
                pass
            self.mascot = None
            print("HabFly mascot display unavailable; task safety controls are unchanged.", file=sys.stderr)

    def state(self):
        if self.replay_context is not None:
            self.emit(
                "state",
                {
                    **self.replay_context,
                    "status": self.status,
                    "recorded_status": self.replay_context.get(
                        "recorded_status", self.replay_context.get("status")
                    ),
                    "replay": True,
                    "replay_finished": self.replay_events is None,
                    "trace_path": str(self.trace_path),
                },
            )
            return
        if self.options.task == "browser_project" and self.env is not None:
            if self._owns_finalization():
                self._publish_finalization()
                return
            self.project_event("state", self.env.state())
            return
        self.emit(
            "state",
            {
                "status": self.status,
                "seed": self.options.seed,
                "stars": self.options.stars,
                "policy": self.options.policy,
                "stage": self.options.task
                if self.options.task in CALCULATION_TASKS or self.options.task == "color"
                else ("synthetic_demo" if self.options.environment == "simulator" else "browser_inference"),
                "graph": str(self.options.graph or "synthetic-32"),
                "checkpoint": str(self.options.checkpoint or "none"),
                "browser_status": "visible" if self.options.environment == "browser" else "not_connected",
                "calculation_backend": self.options.backend
                if self.options.task in CALCULATION_TASKS
                else None,
                "calculation_mode": (
                    "local_tool_assisted" if self.options.backend == "local" else "google_sheets"
                )
                if self.options.task in CALCULATION_TASKS
                else None,
                "trace_path": str(self.trace_path) if self.trace_path else None,
                **(self.env.state() if self.options.task in BROWSER_TASKS and self.env else {}),
                **(
                    {
                        "stage": f"{self.options.browser_execution}_browser_numeric_transfer",
                        "calculation_backend": "local",
                        "calculation_mode": "local_tool_assisted",
                    }
                    if self.options.task == "browser_numeric"
                    else {}
                ),
                **(
                    {
                        "stage": f"{self.options.browser_execution}_browser_color_transfer",
                        "calculation_mode": "learned_peak_wavelength_color_v1",
                    }
                    if self.options.task == "browser_color"
                    else {}
                ),
            },
        )

    def start(self, payload):
        options = parse_run_options(payload)
        if options.task == "planet_calculations":
            from .training.planet_session import validate_planet_options

            validate_planet_options(options)
        if options.task == "habitability_calculations":
            from .training.habitability_session import validate_habitability_options

            validate_habitability_options(options)
        if options.browser_setup != "manual" and options.task not in BROWSER_TASKS | {"browser_project"}:
            raise ValueError("Automatic setup is supported only for the supervised browser diagnostic")
        if options.browser_execution == "autonomous" and (
            options.task not in BROWSER_TASKS | {"browser_project"}
            or options.browser_setup != "automatic"
            or (options.stars != 1 and not (options.task == "browser_project" and options.project_campaign))
        ):
            raise ValueError("Autonomous execution requires automatic browser diagnostic setup and one star")
        if options.policy == "checkpoint" and not options.checkpoint:
            raise ValueError("checkpoint policy requires checkpoint path")
        if options.task == "color" and (
            options.environment != "simulator"
            or options.policy != "checkpoint"
            or not options.dataset
            or not options.graph
            or options.backend != "local"
            or options.spreadsheet_config
            or options.knowledge_pack
        ):
            raise ValueError("Color runtime requires a separate local color checkpoint and dataset")
        if options.environment == "browser" and options.policy != "checkpoint":
            raise ValueError("Browser runs require a trained checkpoint")
        if options.task in CALCULATION_TASKS and options.environment == "browser":
            raise ValueError("Stellar v1 cannot run against the HabWorlds browser")
        if options.task in CALCULATION_TASKS and not options.dataset:
            raise ValueError("Stellar runtime requires dataset")
        if options.task in LOCAL_CHECKPOINT_TASKS and (
            options.policy != "checkpoint"
            or not options.graph
            or options.backend != "local"
            or options.spreadsheet_config
        ):
            raise ValueError(
                "Local calculation demo requires a checkpoint, graph and local-only calculation backend"
            )
        self.close()
        self.options = options
        if options.project_mascot:
            from .browser_mascot import BrowserMascot

            self.mascot = BrowserMascot()
        self._finalization_authorization = (
            self._authorization() if options.task == "browser_project" else None
        )
        self.run_id = uuid.uuid4().hex
        options.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = options.artifact_dir / f"{self.run_id}.jsonl"
        self.trace = self.trace_path.open("x")
        if options.task == "browser_project":
            from .browser_project_runtime import BrowserProjectRuntime

            model_options = {
                "dataset": options.dataset,
                "checkpoint": options.checkpoint,
                "color_experiment": options.color_experiment,
                "graph_path": options.graph,
                "seed": options.seed,
                "no_planet_save_settle_seconds": options.browser_no_planet_save_settle_seconds,
            }
            if options.browser_planet_pilot is not None:
                model_options.update(
                    planet_pilot=options.browser_planet_pilot,
                    planet_final_evaluation=options.browser_planet_final_evaluation,
                    planet_seed=options.browser_planet_seed,
                )
            if options.project_reference_shallow_transits:
                model_options["shallow_reference"] = True
            if options.project_two_event_reference:
                model_options["two_event_reference"] = True
            if options.project_baseline_edge_reference:
                model_options["allow_baseline_edge_reference"] = True
            if options.project_baseline_band_reference:
                model_options["allow_baseline_band_reference"] = True
            if options.project_single_event_reference:
                model_options["allow_single_event_reference"] = True
            if options.project_save_strategy == "autosave":
                model_options["save_strategy"] = "autosave"
            if options.project_supplied_stellar_inputs:
                model_options["planet_supplied_evaluation"] = options.browser_planet_supplied_evaluation
            self.policy = self.neural_state = self.observation = None
            terrestrial_options = None
            if options.browser_habitability_pilot is not None:
                from .data.graphs import load_graph

                terrestrial_options = {
                    "pilot": options.browser_habitability_pilot,
                    "final_evaluation": options.browser_habitability_final_evaluation,
                    "graph": load_graph(options.graph),
                    "candidates": options.browser_habitability_gas_candidates,
                    "seed": options.browser_habitability_seed,
                    "max_seconds": options.browser_habitability_max_seconds,
                    "settle_timeout_seconds": options.browser_habitability_save_settle_seconds,
                }
                if options.project_supplied_stellar_inputs:
                    terrestrial_options["supplied_evaluation"] = (
                        options.browser_habitability_supplied_evaluation
                    )
            self.env = BrowserProjectRuntime(
                options,
                options.artifact_dir / self.run_id,
                model_options=model_options,
                emit=self.project_event,
                project_limits={
                    "max_advances": options.project_max_advances,
                    "max_seconds": options.project_max_seconds,
                },
                setup_max_advances=options.project_setup_max_advances,
                class_limits={
                    "max_advances": options.project_class_max_advances,
                    "max_seconds": options.project_class_max_seconds,
                },
                reference_planet_continuation=options.project_reference_planet_continuation,
                positive_save_settle_seconds=options.browser_positive_save_settle_seconds,
                terrestrial_options=terrestrial_options,
            )
            self.env.start()
            self.status = self.env.status
            return
        if options.task in {"planet_calculations", "habitability_calculations"}:
            from .training.planet_session import load_planet_session

            loader = load_planet_session
            if options.task == "habitability_calculations":
                from .training.habitability_session import load_habitability_session

                loader = load_habitability_session
            self.policy, self.env, provenance = loader(options)
            observation, _ = self.env.reset(seed=options.seed)
            self.observation, self.neural_state = Observation.model_validate(observation), None
            self.status = "paused" if options.paused else "running"
            self.emit(
                "hello",
                {
                    "protocol_version": 1,
                    "policy": "checkpoint",
                    "synthetic": True,
                    "activity_source": "checkpoint",
                    "provenance": provenance,
                    "scope": provenance["scope"],
                    "browser_acceptance_passed": False,
                },
            )
            self.state()
            self.emit("observation", self.observation.model_dump(mode="json"))
            return
        if options.task in BROWSER_TASKS:
            from .browser_policy import BrowserPolicyBridge, load_browser_policy

            bridge_type, loader = BrowserPolicyBridge, load_browser_policy
            if options.task == "browser_color":
                from .browser_color_policy import BrowserColorPolicyBridge, load_browser_color_policy

                bridge_type, loader = BrowserColorPolicyBridge, load_browser_color_policy
            elif options.task == "browser_four_field":
                from .browser_four_field_policy import BrowserFourFieldBridge, load_four_field_policy

                bridge_type, loader = BrowserFourFieldBridge, load_four_field_policy
            self.policy, provenance = loader(options)
            if self.expected_browser_identity is not None and any(
                provenance.get(key) != value for key, value in self.expected_browser_identity.items()
            ):
                from .browser import BrowserSafetyStop

                raise BrowserSafetyStop("batch_provenance_changed")
            self.env = bridge_type(options, options.artifact_dir / self.run_id, provenance)
            self.neural_state = self.observation = None
            self.status = (
                "running" if options.browser_execution == "autonomous" and not options.paused else "paused"
            )
            self.emit(
                "hello",
                {
                    "protocol_version": 1,
                    "policy": "checkpoint",
                    "synthetic": False,
                    "activity_source": "checkpoint",
                    "provenance": provenance,
                    "scope": provenance["evaluation_scope"],
                },
            )
            self.state()
            return
        if options.task == "color":
            from .environments.color import SCOPE, ColorEnv, color_cases
            from .training.color import load_color_experiment

            expected = options.dataset / "training/checkpoint.pt"
            if options.checkpoint.resolve() != expected.resolve():
                raise ValueError("Color runtime checkpoint must belong to its experiment")
            self.policy, reference, report = load_color_experiment(options.dataset, options.graph)
            cases = color_cases("manual", 100, reference)
            if options.seed not in {c["seed"] for c in cases}:
                raise ValueError("Choose a color manual seed from 10000000 through 10000099")
            self.env = ColorEnv(reference, cases)
            observation, _ = self.env.reset(seed=options.seed)
            self.observation, self.neural_state = Observation.model_validate(observation), None
            self.status = "paused" if options.paused else "running"
            self.emit(
                "hello",
                {
                    "protocol_version": 1,
                    "policy": "checkpoint",
                    "synthetic": True,
                    "activity_source": "checkpoint",
                    "scope": SCOPE,
                    "reference_hash": reference.checksum,
                    "ready_for_final_test": report["ready_for_final_test"],
                    "browser_acceptance_passed": False,
                    "experimental_local_checkpoint": True,
                },
            )
            self.state()
            self.emit("observation", self.observation.model_dump(mode="json"))
            return
        import torch

        from .content import load_content_pack
        from .data import load_graph, make_demo_graph
        from .environments import MiniHabWorlds
        from .model import ConnectomePolicy

        torch.set_num_threads(1)
        torch.manual_seed(options.seed)
        self.pack = load_content_pack(options.content_pack)
        graph = load_graph(options.graph) if options.graph else make_demo_graph()
        identity = self.pack.model_dump(mode="json")
        if options.task == "stellar":
            from .training.stellar import load_dataset, training_content

            stellar_manifest, stellar_data = load_dataset(options.dataset)
            identity = training_content(stellar_manifest)
        elif options.task in LOCAL_CHECKPOINT_TASKS:
            from .knowledge import load_knowledge_pack
            from .training.distance_session import load_distance_session
            from .training.luminosity_session import load_chain_session

            config = load_knowledge_pack(options.knowledge_pack)
            if options.task == "distance":
                identity, distance_case = load_distance_session(
                    options.dataset, options.checkpoint, config, options.seed
                )
            else:
                identity, distance_case = load_chain_session(
                    options.dataset, options.checkpoint, config, options.seed, task=options.task
                )
        if options.policy == "checkpoint":
            from .training.checkpoints import load_checkpoint

            self.policy, _manifest = load_checkpoint(options.checkpoint, graph, content_pack=identity)
        else:
            self.policy = ConnectomePolicy(graph, hidden_size=8)
        self.policy.eval()
        if options.task == "stellar":
            from .config import calculation_config
            from .training.stellar import make_adapter, make_environment

            config = calculation_config(options)
            if config.content_identity() != stellar_manifest["content"]:
                raise ValueError("Calculation backend and dataset mismatch")
            cases = stellar_data["test"]["cases"]
            if options.seed not in {c["seed"] for c in cases}:
                raise ValueError("Choose a recorded test seed, normally starting at 300000")
            self.spreadsheet_adapter = make_adapter(config)
            try:
                self.spreadsheet_adapter.verify()
                self.env = make_environment(self.spreadsheet_adapter, cases)
                observation, _ = self.env.reset(seed=options.seed)
                self.observation = Observation.model_validate(observation)
            except Exception:
                self.close()
                raise
        elif options.task in LOCAL_CHECKPOINT_TASKS:
            from .environments.distance_diagnostic import DistanceDiagnosticEnv
            from .knowledge import LocalCalculator
            from .training.chained_workflow import workflow_spec

            self.spreadsheet_adapter = LocalCalculator(config)
            self.spreadsheet_adapter.verify()
            workflow = workflow_spec(options.task) if options.task != "distance" else None
            environment = workflow.environment if workflow else DistanceDiagnosticEnv
            self.env = environment(
                self.spreadsheet_adapter,
                [distance_case],
                max_steps=workflow.max_steps if workflow else 32,
            )
            observation, _ = self.env.reset(seed=options.seed)
            self.observation = Observation.model_validate(observation)
        elif options.environment == "browser":
            from .browser import BrowserConfig, TorusBrowser

            if not options.browser_config:
                raise ValueError("browser_config is required")
            self.env = TorusBrowser(BrowserConfig.model_validate_json(options.browser_config.read_text()))
            self.observation = self.env.start()
        else:
            self.env = MiniHabWorlds(stars=options.stars, pack=self.pack)
            observation, _ = self.env.reset(seed=options.seed)
            self.observation = Observation.model_validate(observation)
        self.neural_state = None
        self.status = "paused" if options.paused else "running"
        self.emit(
            "hello",
            {
                "protocol_version": 1,
                "policy": options.policy,
                "content_pack": config.content_identity()
                if options.task in CALCULATION_TASKS
                else self.pack.id,
                "synthetic": self.pack.synthetic,
                "activity_source": "untrained_observer" if options.policy == "expert" else options.policy,
            },
        )
        self.state()
        self.emit("observation", self.observation.model_dump(mode="json"))

    def _authorization(self):
        return (
            self.options.task,
            self.options.stars,
            self.options.project_campaign,
            self.options.project_allow_scoring,
            self.options.project_allow_submission,
            self.options.project_autonomous_decisions,
            self.options.project_supplied_stellar_inputs,
            self.options.browser_planet_supplied_evaluation,
            self.options.browser_habitability_supplied_evaluation,
            self.options.project_baseline_edge_reference,
            self.options.project_baseline_band_reference,
            self.options.project_single_event_reference,
            self.options.project_save_strategy,
            self.options.project_two_event_reference,
            self.options.project_campaign_max_seconds,
        )

    def _check_project_authorization(self):
        if (
            self._finalization_authorization_failed
            or self._finalization_authorization is not None
            and (
                self._authorization() != self._finalization_authorization
                or type(self.options.project_allow_scoring) is not bool
                or type(self.options.project_allow_submission) is not bool
                or type(self.options.project_autonomous_decisions) is not bool
                or type(self.options.project_supplied_stellar_inputs) is not bool
                or type(self.options.project_baseline_edge_reference) is not bool
                or type(self.options.project_baseline_band_reference) is not bool
                or type(self.options.project_single_event_reference) is not bool
                or type(self.options.project_save_strategy) is not str
                or type(self.options.project_two_event_reference) is not bool
                or type(self.options.project_campaign_max_seconds)
                is not type(self._finalization_authorization[-1])
            )
        ):
            self._finalization_authorization_failed = self._finalization_cancelled = True
            if self._owns_finalization():
                # No callback or child method here: this guard also runs inside
                # native pre-dispatch callbacks. Cancellation is sticky and the
                # surrounding scheduled call performs cleanup at its boundary.
                self._finalization_pending = False
                self._finalization_terminal = {
                    "status": "stopped",
                    "phase": "stopped",
                    "finished": True,
                    "failure_reason": "project_finalization_authorization_changed",
                    "task_completed": False,
                    "project_completed": False,
                    "submitted": False,
                    "submission_verified": False,
                }
                self.status = "stopped"
            raise ValueError("project_finalization_authorization_changed")

    def _owns_finalization(self):
        return (
            self._finalization_pending
            or self._finalization_attempted
            or self._finalization_terminal is not None
        )

    def _queue_finalization(self, *, paused):
        self._check_project_authorization()
        if self._owns_finalization() or not self.options.project_allow_scoring:
            return
        if self._finalization_cancelled:
            self._fail_finalization("operator_aborted", aborted=True)
            self._publish_finalization()
            return
        if not (
            self.env.finished and self.env.status == "handoff" and self.env.phase == "awaiting_assessment"
        ):
            return
        report = self.env.report
        if not (
            isinstance(report, dict)
            and report.get("target_workflows_verified") is True
            and report.get("target_stars") == 30
            and report.get("project_campaign") is True
            and report.get("task_completed") is False
            and report.get("project_completed") is False
            and report.get("failure_reason") is None
        ):
            raise ValueError("project_finalization_campaign_handoff_unverified")
        # The separate owner's offline constructor revalidates the complete
        # campaign/source/journal proof. This queue is not a success receipt.
        self._finalization_campaign = deepcopy(report)
        self._finalization_mode = "paused" if paused else "running"
        self._finalization_pending = True
        self.status = self._finalization_mode
        self._publish_finalization()

    def _finalization_state(self):
        if self.finalization is not None:
            try:
                state = deepcopy(self.finalization.state())
            except Exception:  # Failed source reads cannot become completion evidence.
                if self._finalization_terminal is None:
                    raise
                state = {
                    "mode": "post_campaign_finalization",
                    "phase": self._finalization_last_phase,
                    "project_progress": None,
                    "source_state_unavailable": True,
                    "task_completed": False,
                    "project_completed": False,
                    "score_checkpoint_completed": False,
                    "reported_score": None,
                }
            if state.get("task_completed") is not False or state.get("project_completed") is not False:
                if self._finalization_terminal is None:
                    raise ValueError("project_finalization_invalid_completion_claim")
                state = {
                    "mode": "post_campaign_finalization",
                    "project_progress": None,
                    "score_checkpoint_completed": False,
                    "reported_score": None,
                }
            else:
                self._finalization_last_phase = state.get("phase")
        else:
            state = {
                "mode": "post_campaign_finalization",
                "status": self.status,
                "phase": "finalization_pending",
                "finished": False,
                "task_completed": False,
                "project_completed": False,
                "submitted": False,
                "submission_verified": False,
                "project_progress": deepcopy((self._finalization_campaign or {}).get("project_progress")),
                "failure_reason": None,
            }
        state.update(
            assessment_enabled=bool(self._finalization_authorization and self._finalization_authorization[3]),
            submission_enabled=bool(self._finalization_authorization and self._finalization_authorization[4]),
            scoring_max_seconds=600,
            submission_max_seconds=180,
            max_scoring_advances=46,
            max_submission_advances=4,
            campaign_budget_extended=False,
        )
        if self._finalization_terminal:
            # Keep an existing unknown canonical attempt visible after abort or
            # runtime/callback failure. Never turn a pending write into success.
            pending = state.get("phase") == "unknown_pending"
            state.update(self._finalization_terminal)
            if "score_checkpoint_completed" in state or "reported_score" in state:
                state.update(score_checkpoint_completed=False, reported_score=None)
            if pending:
                state["phase"] = "unknown_pending"
        return state

    def _publish_finalization(self, *, _reconcile=True):
        if self._finalization_emitting:
            return
        state = self._finalization_state()
        self._finalization_emitting = True
        try:
            result = self.project_event("state", state, source_trace=None, finalization=True)
        except Exception:  # noqa: BLE001 - failed display callback must not permit a later write
            self._fail_finalization("project_finalization_event_forwarding_failed")
            raise ValueError("project_finalization_event_forwarding_failed") from None
        finally:
            self._finalization_emitting = False
        if (
            self._finalization_cancelled
            and self._finalization_terminal is None
            and not self._finalization_busy
        ):
            self._fail_finalization("operator_aborted", aborted=True)
        if _reconcile and not self._finalization_busy and state != self._finalization_state():
            self._publish_finalization(_reconcile=False)
        return result

    def _finalization_event(self, item):
        self._check_project_authorization()
        if self._finalization_emitting:
            self._finalization_cancelled = True
            raise ValueError("project_finalization_reentrant_event")
        event = RuntimeEvent.model_validate(item)
        if event.event in {"hello", "state", "episode_summary"} and (
            event.payload.get("task_completed") is not False
            or event.payload.get("project_completed") is not False
        ):
            self._finalization_cancelled = True
            raise ValueError("project_finalization_invalid_completion_claim")
        self._finalization_emitting = True
        try:
            return self.project_event(
                event.event,
                event.payload,
                source_trace="finalization/events.jsonl",
                source_header={key: value for key, value in item.items() if key != "payload"},
                finalization=True,
            )
        finally:
            self._finalization_emitting = False

    def _finalization_is_cancelled(self):
        self._check_project_authorization()
        return self._finalization_cancelled

    def _make_finalization(self):
        from .browser_project_finalize_steps import BrowserProjectFinalizeSteps

        return BrowserProjectFinalizeSteps(
            self.env.page,
            self.env.config,
            self.env.output / "finalization",
            run_history=self.env.output,
            journal=self.env.journal,
            campaign_runtime_dir=self.env.output,
            allow_score_transfer=True,
            allow_submission=self._finalization_authorization[4],
            emit=self._finalization_event,
            cancelled=self._finalization_is_cancelled,
        )

    def _fail_finalization(self, reason, *, aborted=False):
        self._finalization_cancelled, self._finalization_pending = True, False
        self._finalization_terminal = {
            "status": "aborted" if aborted else "stopped",
            "phase": "aborted" if aborted else "stopped",
            "finished": True,
            "failure_reason": reason,
            "task_completed": False,
            "project_completed": False,
            "submitted": False,
            "submission_verified": False,
        }
        self.status = self._finalization_terminal["status"]
        if self.finalization is not None and not self._finalization_busy and not self._finalization_emitting:
            self._abort_finalizer()

    def _abort_finalizer(self):
        try:
            self.finalization.abort()
        except Exception:  # noqa: BLE001 - sticky runtime stop remains even if child diagnostics fail
            if self._finalization_terminal is None:
                self._finalization_terminal = {
                    "phase": "stopped",
                    "status": "stopped",
                    "finished": True,
                    "task_completed": False,
                    "project_completed": False,
                    "submitted": False,
                    "submission_verified": False,
                }
            self._finalization_terminal["failure_reason"] = "project_finalization_abort_failed"
            self.status = self._finalization_terminal["status"]

    def _advance_finalization(self):
        if self._finalization_terminal is not None:
            return
        if self._finalization_busy or self._finalization_emitting:
            self._finalization_cancelled = True
            raise ValueError("project_finalization_reentrant_call")
        self._finalization_busy = True
        try:
            self._check_project_authorization()
            self._finalization_state()
            if self._finalization_cancelled:
                self._fail_finalization("operator_aborted", aborted=True)
            elif self._finalization_pending:
                # Sticky BEFORE construction: a failure or exclusive claim may
                # never trigger another constructor on an automatic tick.
                self._finalization_attempted, self._finalization_pending = True, False
                self.finalization = self._make_finalization()
                self._check_project_authorization()
                if not self._finalization_cancelled:
                    self.finalization.start(paused=self._finalization_mode == "paused")
            elif self.finalization is not None and not self.finalization.finished:
                self.finalization.step() if self.status == "paused" else self.finalization.tick()
            self._check_project_authorization()
            self._finalization_state()
        except (KeyboardInterrupt, SystemExit):
            self._fail_finalization("operator_aborted", aborted=True)
            raise
        except Exception:  # noqa: BLE001 - driver/source/credential errors remain private
            self._fail_finalization("project_finalization_runtime_failed")
        finally:
            self._finalization_busy = False
        if self.finalization is not None:
            if self._finalization_cancelled:
                self._abort_finalizer()
            elif self._finalization_mode == "paused" and not self.finalization.finished:
                self.finalization.pause()
            if self._finalization_terminal is None:
                self.status = self.finalization.status
        self._publish_finalization()

    def _finalization_command(self, command):
        if command.payload:
            raise ValueError("project_finalization_payload_not_supported")
        if command.command not in {"pause", "resume", "step", "abort"}:
            raise ValueError("project_finalization_command_not_supported")
        if command.command != "abort":
            self._check_project_authorization()
        if command.command == "abort":
            self._finalization_cancelled = True
            if self._finalization_busy or self._finalization_emitting:
                self.status = "aborted"
                return
            self._fail_finalization("operator_aborted", aborted=True)
        elif self._finalization_terminal or self.finalization is not None and self.finalization.finished:
            return
        elif command.command == "pause":
            self._finalization_mode = self.status = "paused"
            if (
                self.finalization is not None
                and not self._finalization_busy
                and not self._finalization_emitting
            ):
                self.finalization.pause()
        elif command.command == "resume":
            if self.status != "paused" or self._finalization_busy or self._finalization_emitting:
                raise ValueError("project_finalization_pause_before_resume")
            self._finalization_mode = self.status = "running"
            if self.finalization is not None:
                self.finalization.resume()
        else:
            if self.status != "paused":
                raise ValueError("project_finalization_pause_before_step")
            self._advance_finalization()
            return
        if not self._finalization_busy and not self._finalization_emitting:
            self._publish_finalization()

    def project_event(
        self, kind, payload, *, source_trace="events.jsonl", source_header=None, finalization=False
    ):
        """Assign outer v1 sequence numbers, with optional compact display data.

        Lifecycle metadata belongs to this runtime. The bridge's separate exact
        journal is unchanged and its relative trace path is not the outer trace.
        """
        presentation_payload = payload
        if self.options.project_compact_wire:
            from .project_wire import compact_project_event

            payload = compact_project_event(
                kind, payload, source_trace=source_trace, source_header=source_header
            )
        if kind in {"hello", "state", "episode_summary"}:
            if kind in {"state", "episode_summary"} and "status" in payload:
                self.status = payload["status"]
            payload = {
                **payload,
                "task": "browser_project",
                "runtime_task": "browser_project",
                "environment": "browser",
                "seed": self.options.seed,
                "stars": self.options.stars,
                "project_campaign": self.options.project_campaign,
                **({"project_compact_wire": True} if self.options.project_compact_wire else {}),
                "policy": "checkpoint",
                "graph": str(self.options.graph),
                "checkpoint": str(self.options.checkpoint),
                "project_trace_path": str(self.options.artifact_dir / self.run_id / "events.jsonl"),
                "artifact_root": str(self.options.artifact_dir / self.run_id),
                "trace_path": str(self.trace_path),
            }
            if finalization:
                # Runtime-owned display context is not part of the hashed raw
                # finalizer event. Never reuse the retired campaign's live state
                # or propagate the final task's star/reference as current work.
                payload.update(
                    current_star=None,
                    initial_star=None,
                    project_owner=None,
                    campaign=None,
                    class_setup=None,
                    reference_measurements=None,
                    policy_stage="project_finalization",
                    stage="browser_project_finalization",
                    browser_phase=payload.get("phase"),
                    browser_status="closed" if self.env is None or self.env.closed else "visible",
                    assessment_enabled=bool(
                        self._finalization_authorization and self._finalization_authorization[3]
                    ),
                    submission_enabled=bool(
                        self._finalization_authorization and self._finalization_authorization[4]
                    ),
                    campaign_budget_extended=False,
                    finalization_trace_path=str(
                        self.options.artifact_dir / self.run_id / "finalization/events.jsonl"
                    )
                    if self.finalization is not None
                    else None,
                )
        item = self.emit(kind, payload)
        if self.mascot is not None and self.replay_context is None:
            # Keep the full component/star identity for presentation context,
            # even when the wire chooses its existing compact representation.
            self._present_mascot(item.model_copy(update={"payload": presentation_payload}))
        return item

    def tick(self):
        if self.replay_events is not None:
            try:
                item = next(self.replay_events)
            except StopIteration:
                self.replay_events = None
                self.status = "completed"
                self.state()
                return
            payload = {**item.payload, "replay": True}
            if item.event in {"hello", "state"}:
                # Keep the latest observed context for pause/resume and EOF.
                # Playback exhaustion is not a recorded task-success receipt.
                self.replay_context.update(item.payload)
                self._replay_finalizer_parent = _raw_finalizer_replay_identity(item)
                if not _campaign_replay_child(item.payload):
                    self._replay_campaign_parent = _outer_campaign_replay_identity(item)
            elif item.event == "episode_summary" and (
                _raw_finalizer_terminal_summary(item, self._replay_finalizer_parent)
                or _outer_campaign_terminal_summary(item, self._replay_campaign_parent)
            ):
                # This recognized owner's last record is its terminal summary. Do not
                # retain the preceding active phase at EOF, and never merge an
                # arbitrary child's summary into parent completion. The fixed
                # owner contracts above permit only false success claims.
                if _outer_campaign_terminal_summary(item, self._replay_campaign_parent):
                    for key in (
                        "component",
                        "component_event",
                        "component_state",
                        "component_summary",
                        "component_hello",
                        "ancestry",
                    ):
                        self.replay_context.pop(key, None)
                self.replay_context.update(item.payload)
                self.replay_context.pop("recorded_status", None)
            if item.event == "state":
                payload["recorded_status"] = payload.get("status")
                payload["status"] = self.status
            self.emit(item.event, payload)
            return
        if self.env is None or self.status not in {"running", "paused"}:
            raise ValueError("No active run")
        if self.options.task == "browser_project":
            self._check_project_authorization()
            if self._owns_finalization():
                self._advance_finalization()
                return
            paused = self.status == "paused"
            self.env.step() if self.status == "paused" else self.env.tick()
            self.status = self.env.status
            self._queue_finalization(paused=paused)
            return
        if self.options.task in BROWSER_TASKS:
            automatic = self.options.browser_execution == "autonomous" and self.status == "running"
            payload = {"approve_copy": True} if automatic and self.env.phase == "awaiting_copy" else {}
            if automatic and self.env.phase == "awaiting_color":
                payload = {"approve_color": True}
            self.browser_step(payload, automatic=automatic)
            self.last_tick = time.monotonic()
            return
        import torch

        with torch.no_grad():
            proposed, self.neural_state, _diagnostics = self.policy.act(self.observation, self.neural_state)
        if self.options.policy == "expert":
            if self.options.task == "stellar":
                from .environments.stellar import stellar_expert

                proposed = (
                    self.env.expert_action(self.observation)
                    if hasattr(self.env, "expert_action")
                    else stellar_expert(self.observation)
                )
            else:
                from .environments import expert_action

                proposed = expert_action(self.observation, self.pack)
        proposed.observation_revision = self.observation.revision
        self.emit(
            "action_proposed",
            {
                **proposed.model_dump(mode="json"),
                "action_source": "scripted_expert"
                if self.options.policy == "expert"
                else self.options.policy,
                "calibration_scope": self.policy.calibration.get("scope", "uncalibrated")
                if self.options.policy != "expert"
                else "not_applicable_to_expert",
                "calibration": self.policy.calibration if self.options.policy != "expert" else {},
            },
        )
        activity = self.policy.neural_activity(self.neural_state)
        if self.options.task == "color" and _diagnostics.get("activity_pathway"):
            activity = _diagnostics
        if "groups" not in activity:
            activity["groups"] = {
                name: {"mean": value} for name, value in activity.get("populations", {}).items()
            }
        activity["activity_source"] = (
            "untrained_observer" if self.options.policy == "expert" else self.options.policy
        )
        self.emit("neural_activity", activity)
        if self.options.environment == "browser":
            result = self.env.step(proposed)
        else:
            _, _, _, _, info = self.env.step(proposed)
            result = StepResult.model_validate(info["result"])
        self.observation = result.observation
        self.emit("action_result", result.model_dump(mode="json"))
        self.emit("observation", self.observation.model_dump(mode="json"))
        if result.terminated or result.truncated:
            completed = task_completed(self.observation)
            self.status = "completed" if completed else "stopped"
            self.emit(
                "episode_summary",
                {
                    "completed": completed,
                    "steps": result.steps,
                    "reward": result.cumulative_reward,
                    "progress": self.observation.progress,
                    "failure_reason": result.failure_reason,
                    "policy": self.options.policy,
                },
            )
            self.state()
            self.env.close()
        self.last_tick = time.monotonic()

    def browser_step(self, payload, *, automatic=False):
        """One decision or copy per tick; explicit autonomous opt-in never changes scope."""
        bridge = self.env
        if automatic and (self.options.browser_execution != "autonomous" or self.status != "running"):
            raise ValueError("Automatic browser steps require a running autonomous run")
        if not bridge or (not automatic and self.status != "paused"):
            raise ValueError("No paused browser diagnostic")
        if bridge.phase == "setting_up":
            raise ValueError("Scripted setup is in progress; wait for ready or press a to abort")
        ready = payload == {"browser_ready": True}
        combined = self.options.task == "browser_four_field"
        color = self.options.task == "browser_color" or (combined and bridge.active_stage == "color")
        handoff = combined and bridge.phase == "awaiting_handoff" and not payload
        approve = payload == {"approve_color" if color else "approve_copy": True}
        if payload and not (ready or approve):
            raise ValueError("Unknown browser step payload")
        if ready and bridge.phase != "awaiting_ready":
            raise ValueError("Browser already captured")
        if approve and bridge.phase != ("awaiting_color" if color else "awaiting_copy"):
            raise ValueError("No pending color selection" if color else "No pending copy")
        if not payload and bridge.phase != "ready" and not handoff:
            raise ValueError("Use b to capture the ready browser, or y to confirm its pending write")
        try:
            if handoff:
                self.observation = bridge.handoff()
                self.neural_state = None
                self.emit("observation", self.observation.model_dump(mode="json"))
                self.state()
                return
            if ready:
                self.observation = bridge.ready()
                self.emit("observation", self.observation.model_dump(mode="json"))
                self.state()
                return
            if approve:
                result = bridge.approve(automatic=automatic)
            else:
                import torch

                if combined and self.policy.select_stage(bridge.active_stage):
                    self.neural_state = None
                with torch.no_grad():
                    proposed, self.neural_state, diagnostics = self.policy.act(
                        self.observation, self.neural_state
                    )
                proposed.observation_revision = self.observation.revision
                # Local calibration cannot establish browser probabilities.
                proposed.action_confidence = proposed.target_confidence = None
                proposed.calibrated = False
                self.emit(
                    "action_proposed",
                    {
                        **proposed.model_dump(mode="json"),
                        "action_source": "checkpoint",
                        "calibration_scope": "browser_transfer_not_calibrated",
                        **({"policy_stage": bridge.active_stage} if combined else {}),
                    },
                )
                activity = self.policy.neural_activity(self.neural_state)
                if color and diagnostics.get("activity_pathway"):
                    activity = diagnostics
                    activity["action_probability"] = activity["target_probability"] = None
                if "groups" not in activity:
                    activity["groups"] = {
                        name: {"mean": value} for name, value in activity.get("populations", {}).items()
                    }
                activity["activity_source"] = "checkpoint"
                if combined:
                    activity["policy_stage"] = bridge.active_stage
                self.emit("neural_activity", activity)
                result = bridge.propose(proposed)
            if result is not None:
                self.observation = result.observation
                self.emit("action_result", result.model_dump(mode="json"))
                self.emit("observation", self.observation.model_dump(mode="json"))
                if result.terminated or result.truncated:
                    self.status = "stopped"  # This is not a completed HabWorlds star.
                    bridge.close(keep_browser_open=color or combined)
                    self.emit(
                        "episode_summary",
                        {
                            **bridge.summary,
                            "completed": False,
                            "steps": result.steps,
                            "transport_artifacts": str(bridge.journal.output),
                        },
                    )
            self.state()
        except Exception as exc:  # noqa: BLE001 - fail closed without leaking browser URLs
            # Playwright exceptions may contain session URLs. Never serialize them.
            import re

            from .browser import BrowserSafetyStop
            from .browser_stellar import StellarMappingError
            from .color_reference import ColorReferenceError

            code = (
                str(exc)
                if isinstance(exc, (BrowserSafetyStop, StellarMappingError, ColorReferenceError))
                else "browser_operation_failed"
            )
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,100}", code):
                code = "browser_operation_failed"
            bridge.outcome = code
            self.status = "stopped"
            bridge.close(keep_browser_open=color or combined)
            self.emit("error", {"type": "BrowserDiagnosticStop", "message": code})
            self.emit("episode_summary", {**bridge.summary, "completed": False, "failure_reason": code})
            self.state()

    def setup_tick(self):
        bridge = self.env
        active = self.status == ("running" if self.options.browser_execution == "autonomous" else "paused")
        if not bridge or not active or bridge.phase != "setting_up":
            return
        from .browser_setup import SetupStop

        try:
            prior_stage = bridge.setup.stage
            observation = bridge.advance_setup()
            if observation is not None:
                self.observation = observation
                self.emit("observation", observation.model_dump(mode="json"))
                self.last_tick = time.monotonic()
            if observation is not None or bridge.setup.stage != prior_stage:
                self.state()
        except SetupStop as exc:
            bridge.outcome = str(exc)
            self.status = "stopped"
            bridge.close(keep_browser_open=True)
            self.emit("error", {"type": "BrowserSetupStop", "message": str(exc)})
            self.emit("episode_summary", {**bridge.summary, "completed": False, "failure_reason": str(exc)})
            self.state()

    def advance_if_due(self):
        """Shared scheduler for JSONL/TUI and bounded reliability runs.

        Commands are handled before this method. Copy proposals and actual
        copies occupy separate ticks so pause/abort can cancel a pending write.
        """
        if self.replay_context is None and self.options.task == "browser_project" and self.env:
            if self._owns_finalization() and self._finalization_terminal is not None:
                return
            self._check_project_authorization()
            if self._owns_finalization():
                if self.status == "running" and time.monotonic() - self.last_tick >= self.options.interval:
                    self.last_tick = time.monotonic()
                    self._advance_finalization()
                return
            paused = self.status != "running"
            self.env.advance_if_due()
            self.status = self.env.status
            self._queue_finalization(paused=paused)
        elif (
            self.replay_events is None
            and self.options.task in BROWSER_TASKS
            and self.env
            and self.env.phase == "setting_up"
        ):
            self.setup_tick()
        elif self.status == "running" and time.monotonic() - self.last_tick >= self.options.interval:
            self.tick()

    def command(self, message):
        try:
            command = RuntimeCommand.model_validate(message)
        except ValueError:
            if self.options.task == "browser_project" or (
                isinstance(message, dict)
                and isinstance(message.get("payload"), dict)
                and message["payload"].get("task") == "browser_project"
            ):
                raise ValueError("browser_project_invalid_command") from None
            raise
        if (
            self.replay_context is None
            and self.options.task == "browser_project"
            and self.env
            and command.command in {"pause", "resume", "step", "abort"}
        ):
            try:
                if command.command != "abort":
                    self._check_project_authorization()
                if self._owns_finalization():
                    self._finalization_command(command)
                    return
                paused = self.status != "running"
                if command.command == "abort":
                    self._finalization_cancelled = True
                self.env.command(command.model_dump(mode="json"))
                self.status = self.env.status
                if command.command != "abort":
                    self._queue_finalization(paused=paused)
            except Exception as exc:  # noqa: BLE001 - never forward path, URL or credential values
                import re

                from .browser import BrowserSafetyStop

                code = str(exc) if isinstance(exc, BrowserSafetyStop) else "browser_project_invalid_command"
                if not re.fullmatch(r"[a-z][a-z0-9_]{0,150}", code):
                    code = "browser_project_command_failed"
                raise ValueError(code) from None
            finally:
                if not self._owns_finalization():
                    self.status = self.env.status
            return
        if command.command == "start":
            try:
                self.start(command.payload)
            except Exception:
                self.status = "stopped"
                self.close()
                raise
        elif command.command == "replay":
            source = Path(command.payload["path"])
            # Validate the complete log before showing any replay events.
            items = read_trace(source)
            self.close()
            self.run_id = f"replay-{uuid.uuid4().hex}"
            self.trace_path = source
            self.replay_context = dict(next((item.payload for item in items if item.event == "state"), {}))
            self.replay_events = iter(items)
            self.status = "running"
            self.state()
        elif command.command == "save_trace":
            if not self.trace_path:
                raise ValueError("No trace available")
            if self.trace:
                self.trace.flush()
            path = command.payload.get("path")
            if path and Path(path).resolve() != self.trace_path.resolve():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with Path(path).open("x") as destination:
                    destination.write(self.trace_path.read_text())
            self.emit("state", {"status": self.status, "trace_path": str(path or self.trace_path)})
        elif command.command == "abort":
            if self.status not in {"running", "paused"}:
                # Quitting a completed demo must not replace its truthful success
                # summary with a second, operator-aborted failure.
                self.state()
                self.close()
                return
            self.status = "aborted"
            self.emit("episode_summary", {"completed": False, "failure_reason": "operator_aborted"})
            self.state()
            self.close()
        elif command.command in {"pause", "resume"}:
            if self.status not in {"running", "paused"}:
                raise ValueError("No active run")
            if (
                self.options.task in BROWSER_TASKS
                and self.options.browser_execution != "autonomous"
                and self.replay_events is None
                and command.command == "resume"
            ):
                raise ValueError("Browser diagnostic is single-step only: n steps; y confirms one write")
            self.status = "paused" if command.command == "pause" else "running"
            self.state()
        elif command.command == "step":
            if (
                self.options.task in {"browser_color", "browser_four_field"}
                and self.replay_events is None
                and self.env is not None
                and not command.payload
                and (self.env.phase not in {"ready", "awaiting_handoff"} or self.status != "paused")
            ):
                # Extra n presses are guidance, not failed policy decisions. Never
                # consume a pending approval or advance a completed/automatic run.
                self.state()
                return
            if self.status != "paused":
                if (
                    self.options.task in BROWSER_TASKS
                    and self.env is not None
                    and self.env.phase == "finished"
                    and self.replay_events is None
                ):
                    raise ValueError(self.env.state()["browser_guidance"])
                raise ValueError("Pause before single-stepping")
            if self.options.task in BROWSER_TASKS and self.replay_events is None:
                self.browser_step(command.payload)
            else:
                self.tick()

    def close(self):
        if self.mascot is not None:
            try:
                self.mascot.close()
            except Exception:  # noqa: BLE001 - display cleanup must not prevent actual browser cleanup
                print("HabFly mascot cleanup unavailable; closing the owned browser.", file=sys.stderr)
            finally:
                self.mascot = None
        finalization_failed = False
        try:
            if self.finalization is not None:
                self.finalization.close()
        except Exception:  # noqa: BLE001 - cleanup failures must not expose browser/source text
            finalization_failed = True
        finally:
            try:
                if self.env:
                    self.env.close()
            finally:
                self.env = self.finalization = None
                if self.spreadsheet_adapter:
                    self.spreadsheet_adapter.close()
                    self.spreadsheet_adapter = None
                if self.trace:
                    self.trace.close()
                    self.trace = None
                self.replay_events = None
                self.replay_context = None
                self._replay_finalizer_parent = None
                self._replay_campaign_parent = None
                self._finalization_authorization = None
                self._finalization_pending = self._finalization_attempted = False
                self._finalization_cancelled = self._finalization_busy = self._finalization_emitting = False
                self._finalization_terminal = self._finalization_campaign = None
                self._finalization_mode = "paused"
                self._finalization_authorization_failed = False
                self._finalization_last_phase = None
        if finalization_failed:
            raise ValueError("project_finalization_cleanup_failed") from None


def read_trace(path):
    events = []
    previous = -1
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            item = RuntimeEvent.model_validate_json(line)
            if item.sequence <= previous:
                raise ValueError(f"Non-monotonic event sequence at line {line_number}")
            previous = item.sequence
            events.append(item)
    return events


def serve(input_stream=None, output=None, *, start_options=None):
    input_stream = input_stream or sys.stdin
    runtime = Runtime(output)
    messages = queue.Queue(maxsize=64)

    def read_commands():
        for line in input_stream:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=read_commands, daemon=True).start()
    runtime.emit("hello", {"protocol_version": 1, "status": "idle"})
    initial = (
        {"version": 1, "command": "start", "payload": start_options} if start_options is not None else None
    )
    try:
        while True:
            if initial is not None:
                line = ""
            else:
                try:
                    line = messages.get(timeout=0.05)
                except queue.Empty:
                    line = ""
            if line is None:
                break
            project_request = False
            try:
                if initial is not None:
                    message, initial = initial, None
                    project_request = isinstance(message["payload"], dict) and (
                        message["payload"].get("task") == "browser_project"
                    )
                    runtime.command(message)
                elif line:
                    message = json.loads(line)
                    project_request = (
                        isinstance(message, dict)
                        and isinstance(message.get("payload"), dict)
                        and message["payload"].get("task") == "browser_project"
                    )
                    runtime.command(message)
                else:
                    runtime.advance_if_due()
            except Exception as exc:  # noqa: BLE001 - protocol boundary must survive malformed commands
                project = runtime.options.task == "browser_project" or project_request
                runtime.emit(
                    "error",
                    {
                        "message": "browser_project_protocol_error" if project else str(exc),
                        "type": "BrowserProjectProtocolError" if project else type(exc).__name__,
                    },
                )
                if runtime.status == "running":
                    if project and runtime.env is not None and runtime.replay_context is None:
                        if runtime._owns_finalization():
                            runtime._finalization_mode = "paused"
                            if runtime.finalization is not None and not runtime.finalization.finished:
                                runtime.finalization.pause()
                        else:
                            runtime.env.pause()
                    runtime.status = "paused"
                    runtime.state()
    finally:
        runtime.close()
