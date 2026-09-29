"""Fresh visible-browser lifecycle around bounded one-star or campaign owners.

Construction is read-only and never consumes credentials. Launch, navigation,
setup transitions, initial capture and controller creation are separate scheduled
calls. Legacy classification requires an explicit receipt or supplied reference.
Opt-in autonomous reference decisions are derived from pinned public evidence in
separate prepare/apply stages; they are not learned classification or receipts.
Assessment and submission remain separately authorized owners.
"""

import hashlib
import inspect
import json
import math
import os
import re
import time
import uuid
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_class_setup_steps import FreshStarClassSteps
from .browser_next_star import INITIAL_STARFIELD_IMAGE, capture_initial_setup_star, validate_initial_selection
from .browser_no_planet_workflow import _Evidence
from .browser_planet_observation import start_planet_observation
from .browser_policy import browser_config
from .browser_project_campaign_steps import BrowserProjectCampaignSteps
from .browser_project_next_star_steps import _fresh
from .browser_project_steps import _MODEL_KEYS, _REQUIRED_MODELS, BrowserProjectSteps
from .browser_setup import BrowserSetup, SetupStop, consume_credentials
from .browser_star_preflight import validate_star_class_source
from .browser_stellar import SIMULATION_URL
from .contracts import Contract, RuntimeCommand, RuntimeEvent
from .project_events import ProjectEventRelay
from .project_progress import ProjectJournal

_STAGES = {
    "returning_to_configured_preview",
    "refreshing_authenticated_preview",
    "post_login_refresh_verified",
    "waiting_for_simulation",
    "waiting_for_preview_content",
    "cookie_notice_closed",
    "waiting_for_login_ui",
    "waiting_for_login_result",
    "signing_in",
    "opening_stellar_tab",
    "opening_star_data",
    "waiting_for_starfield",
    "waiting_for_starfield_render",
    "waiting_for_stable_starfield",
    "selecting_visible_star",
    "waiting_for_stellar_controls",
    "waiting_for_stellar_frames",
    "waiting_for_stellar_measurements",
    "stellar_screen_ready",
    "verifying_stellar_screen",
    *(f"intro_screen_{i}" for i in range(1, 5)),
}
_PIXELS = {
    "setup-starfield.png",
    "setup-starfield-loading.png",
    "setup-starfield-rejected.png",
    "setup-star-link.png",
}
_CLASS_PHASES = {"setting_reference_class", "setting_lifetime_prefix", "class_setup_handoff"}
_OWNER_PHASES = {
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
_CAMPAIGN_PHASES = {
    "initializing_star",
    "verifying_star",
    "next_star_initializing",
    "next_star_active",
    "adopting_fresh_star",
}
_HANDOFF_PHASES = {
    "awaiting_class_source",
    "awaiting_inventory",
    "awaiting_planet_class",
    "awaiting_gases",
    "awaiting_habitability",
}


def _require(condition, code):
    if not condition:
        raise BrowserSafetyStop("project_runtime_" + code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _reason(exc):
    text = str(exc)
    return (
        text
        if isinstance(exc, (SetupStop, BrowserSafetyStop)) and re.fullmatch(r"[a-z][a-z0-9_:.-]{0,150}", text)
        else "project_runtime_operation_failed"
    )


def _driver():
    from playwright.sync_api import sync_playwright

    return sync_playwright().start()


class BrowserProjectRuntime:
    """Own the browser lifecycle and relay additive version-1 Runtime events.

    options supplies environment='browser', browser_setup='automatic', stars=1
    or an explicit bounded two-/three-/30-star campaign opt-in,
    interval, paused and browser_config (unless config is explicitly supplied).
    Frozen stellar/planet paths remain explicit in model_options. No credentials
    or session URL enter any scope, event, receipt or exception message.

    Terminal runs keep Chromium visible for inspection. close() ends this owned
    context/browser/driver; abort() alone prevents further actions but keeps it
    open. Paused wall time counts against BrowserSetup's existing fixed deadline.
    """

    def __init__(
        self,
        options,
        output,
        *,
        model_options,
        emit=lambda *_: None,
        config=None,
        credentials=None,
        project_limits=None,
        class_limits=None,
        reference_planet_continuation=False,
        positive_save_settle_seconds=20,
        terrestrial_options=None,
        setup_max_advances=256,
        _driver_factory=None,
        _setup_factory=None,
        _capture_initial=None,
        _steps_factory=None,
        _campaign_factory=None,
        _class_factory=None,
        _clock=time.monotonic,
    ):
        campaign = getattr(options, "project_campaign", False)
        campaign_seconds = getattr(options, "project_campaign_max_seconds", None)
        _require(type(campaign) is bool, "invalid_campaign_opt_in")
        _require(
            getattr(options, "environment", None) == "browser"
            and getattr(options, "browser_setup", None) == "automatic"
            and type(getattr(options, "stars", None)) is int
            and (
                campaign
                and getattr(options, "task", None) == "browser_project"
                and options.stars in {2, 3, 30}
                or not campaign
                and options.stars == 1
            ),
            "requires_fresh_automatic_single_star",
        )
        _require(
            (not campaign and campaign_seconds is None)
            or (
                campaign
                and (
                    type(campaign_seconds) in {int, float}
                    and math.isfinite(campaign_seconds)
                    and 1 <= campaign_seconds <= 10800
                    or type(campaign_seconds) is str
                    and campaign_seconds == "uncapped"
                    and options.stars == 30
                )
            ),
            "explicit_campaign_budget_required",
        )
        interval = getattr(options, "interval", 0.2)
        _require(
            type(interval) in {int, float} and math.isfinite(interval) and 0 <= interval <= 10,
            "invalid_interval",
        )
        _require(type(getattr(options, "paused", True)) is bool, "invalid_pause_option")
        viewport = getattr(options, "project_viewport", None)
        if isinstance(viewport, Contract):
            viewport = viewport.model_dump(mode="json")
        _require(
            viewport is None
            or (
                getattr(options, "task", None) == "browser_project"
                and isinstance(viewport, dict)
                and set(viewport) == {"width", "height"}
                and type(viewport["width"]) is int
                and type(viewport["height"]) is int
                and 950 <= viewport["width"] <= 2400
                and 600 <= viewport["height"] <= 1800
            ),
            "invalid_project_viewport",
        )
        _require(type(reference_planet_continuation) is bool, "invalid_planet_continuation")
        autonomous = getattr(options, "project_autonomous_decisions", False)
        _require(type(autonomous) is bool, "invalid_autonomous_decisions")
        supplied = getattr(options, "project_supplied_stellar_inputs", False)
        _require(type(supplied) is bool, "invalid_supplied_inputs")
        supplied_paths = (
            getattr(options, "browser_planet_supplied_evaluation", None),
            getattr(options, "browser_habitability_supplied_evaluation", None),
        )
        if supplied:
            _require(
                getattr(options, "task", None) == "browser_project"
                and autonomous
                and reference_planet_continuation
                and isinstance(model_options, dict)
                and model_options.get("planet_pilot") is not None
                and model_options.get("planet_final_evaluation") is not None
                and isinstance(terrestrial_options, dict)
                and {"pilot", "final_evaluation", "graph", "candidates"} <= terrestrial_options.keys()
                and all(isinstance(p, (str, Path)) and Path(p).is_dir() for p in supplied_paths)
                and model_options.get("planet_supplied_evaluation") == supplied_paths[0]
                and terrestrial_options.get("supplied_evaluation") == supplied_paths[1],
                "supplied_inputs_require_matching_complete_options",
            )
        else:
            _require(
                all(p is None for p in supplied_paths)
                and (not isinstance(model_options, dict) or "planet_supplied_evaluation" not in model_options)
                and (
                    not isinstance(terrestrial_options, dict)
                    or "supplied_evaluation" not in terrestrial_options
                ),
                "supplied_paths_require_opt_in",
            )
        _require(
            not autonomous or reference_planet_continuation and terrestrial_options is not None,
            "autonomous_decisions_require_planet_and_terrestrial_tools",
        )
        _require(
            terrestrial_options is None
            or (isinstance(terrestrial_options, dict) and reference_planet_continuation),
            "invalid_terrestrial_options",
        )
        _require(
            type(positive_save_settle_seconds) in {int, float}
            and math.isfinite(positive_save_settle_seconds)
            and 0.1 <= positive_save_settle_seconds <= 30,
            "invalid_positive_save_budget",
        )
        _require(
            isinstance(model_options, dict)
            and _REQUIRED_MODELS <= model_options.keys()
            and not (model_options.keys() - _MODEL_KEYS),
            "invalid_model_options",
        )
        _require(
            type(model_options.get("allow_baseline_edge_reference", False)) is bool
            and model_options.get("allow_baseline_edge_reference", False)
            is getattr(options, "project_baseline_edge_reference", False),
            "invalid_baseline_edge_reference",
        )
        _require(
            type(model_options.get("two_event_reference", False)) is bool
            and model_options.get("two_event_reference", False)
            is getattr(options, "project_two_event_reference", False)
            and (
                not model_options.get("two_event_reference", False)
                or model_options.get("shallow_reference") is True
                and getattr(options, "project_reference_shallow_transits", False) is True
                and reference_planet_continuation
            ),
            "invalid_two_event_reference",
        )
        _require(
            type(model_options.get("allow_baseline_band_reference", False)) is bool
            and model_options.get("allow_baseline_band_reference", False)
            is getattr(options, "project_baseline_band_reference", False)
            and (
                not model_options.get("allow_baseline_band_reference", False)
                or model_options.get("allow_baseline_edge_reference") is True
                and reference_planet_continuation
                and model_options.get("planet_pilot") is not None
                and model_options.get("planet_final_evaluation") is not None
            ),
            "invalid_baseline_band_reference",
        )
        _require(
            credentials is None
            or (
                isinstance(credentials, tuple)
                and len(credentials) == 2
                and all(isinstance(v, str) and 0 < len(v) <= 4096 for v in credentials)
            ),
            "invalid_credentials",
        )
        _require(
            type(model_options.get("save_strategy", "explicit")) is str
            and model_options.get("save_strategy", "explicit") in {"explicit", "autosave"}
            and type(getattr(options, "project_save_strategy", "explicit")) is str
            and model_options.get("save_strategy", "explicit")
            == getattr(options, "project_save_strategy", "explicit"),
            "invalid_save_strategy",
        )
        _require(
            type(model_options.get("allow_single_event_reference", False)) is bool
            and model_options.get("allow_single_event_reference", False)
            is getattr(options, "project_single_event_reference", False)
            and (
                not model_options.get("allow_single_event_reference", False)
                or model_options.get("allow_baseline_band_reference") is True
            ),
            "invalid_single_event_reference",
        )
        _require(callable(emit) and callable(_clock), "invalid_callback")
        _require(type(setup_max_advances) is int and 1 <= setup_max_advances <= 1024, "invalid_setup_limit")
        limits = {"max_advances": 512, "max_seconds": 1800, **(project_limits or {})}
        _require(
            set(limits) == {"max_advances", "max_seconds"}
            and type(limits["max_advances"]) is int
            and 1 <= limits["max_advances"] <= 2048
            and type(limits["max_seconds"]) in {int, float}
            and math.isfinite(limits["max_seconds"])
            and 1 <= limits["max_seconds"] <= 3600,
            "invalid_project_limits",
        )
        _require(
            not campaign or limits == {"max_advances": 512, "max_seconds": 1800},
            "campaign_owner_limits_must_remain_fixed",
        )
        class_limits = {"max_advances": 4, "max_seconds": 180, **(class_limits or {})}
        _require(
            set(class_limits) == {"max_advances", "max_seconds"}
            and type(class_limits["max_advances"]) is int
            and 1 <= class_limits["max_advances"] <= 4
            and type(class_limits["max_seconds"]) in {int, float}
            and math.isfinite(class_limits["max_seconds"])
            and 5 <= class_limits["max_seconds"] <= 600,
            "invalid_class_limits",
        )
        self.output = Path(output).absolute()
        _require(
            not self.output.exists() and not any(p.is_symlink() for p in (self.output, *self.output.parents)),
            "output_exists_or_symlink",
        )
        try:
            self.config = (config or browser_config(options)).model_copy(deep=True)
        except Exception:  # noqa: BLE001 - configuration failures can contain session URLs
            raise BrowserSafetyStop("project_runtime_invalid_browser_config") from None
        self._callback, self._clock, self.interval = emit, _clock, interval
        self._default_paused = options.paused
        self.project_viewport = deepcopy(viewport)
        self._viewport_expected = deepcopy(viewport)
        self.project_campaign, self.target_stars, self.campaign_max_seconds = (
            campaign,
            options.stars,
            campaign_seconds,
        )
        self._campaign_options = options
        self._campaign_identity = (campaign, options.stars, campaign_seconds)
        self.model_options, self.project_limits = deepcopy(model_options), limits
        self._two_event_enabled = model_options.get("two_event_reference", False)
        self._two_event_options = options
        self._baseline_edge_enabled = model_options.get("allow_baseline_edge_reference", False)
        self._baseline_edge_options = options
        self._baseline_edge_source = Path(__file__).with_name("planet_window_baseline_edge.py")
        self._baseline_edge_sha = (
            _sha(self._baseline_edge_source.read_bytes()) if self._baseline_edge_enabled else None
        )
        self._baseline_band_enabled = model_options.get("allow_baseline_band_reference", False)
        self._baseline_band_options = options
        self._baseline_band_source = Path(__file__).with_name("planet_window_baseline_band.py")
        self._baseline_band_sha = (
            _sha(self._baseline_band_source.read_bytes()) if self._baseline_band_enabled else None
        )
        self._baseline_band_planet_sources = (
            model_options.get("planet_pilot"),
            model_options.get("planet_final_evaluation"),
        )
        self._single_event_enabled = model_options.get("allow_single_event_reference", False)
        self._single_event_options = options
        self._single_event_source = Path(__file__).with_name("planet_window_single_event.py")
        self._single_event_sha = (
            _sha(self._single_event_source.read_bytes()) if self._single_event_enabled else None
        )
        self._single_event_policy = self._single_event_policy_bytes() if self._single_event_enabled else None
        self._save_strategy = model_options.get("save_strategy", "explicit")
        self._save_strategy_options = options
        self._autosave_source = Path(__file__).with_name("browser_autosave.py")
        self._autosave_sha = (
            _sha(self._autosave_source.read_bytes()) if self._save_strategy == "autosave" else None
        )
        self._autosave_manifest = (
            self._autosave_manifest_bytes() if self._save_strategy == "autosave" else None
        )
        self.class_limits = class_limits
        self.reference_planet_continuation = reference_planet_continuation
        self.autonomous_decisions = self._autonomous_authorized = autonomous
        self.supplied_stellar_inputs = supplied
        self._supplied_settings = (supplied, *supplied_paths)
        self.autonomous_decision = self._autonomous_pending = None
        self._autonomous_hashes = {}
        self._autonomous_attempts = set()
        self.positive_save_settle_seconds = positive_save_settle_seconds
        self.terrestrial_options = deepcopy(terrestrial_options)
        self._credentials, self.credentials_consumed = credentials, False
        self._driver_factory, self._setup_factory = _driver_factory or _driver, _setup_factory or BrowserSetup
        self._capture_initial = _capture_initial or capture_initial_setup_star
        self._steps_factory = _steps_factory or BrowserProjectSteps
        self._campaign_factory = _campaign_factory or BrowserProjectCampaignSteps
        self._class_factory = _class_factory or FreshStarClassSteps
        self.driver = self.browser = self.context = self.page = self.setup = self.project = None
        self.class_setup = None
        self._class_setup_report = self._class_decision = None
        self._class_setup_hashes = {}
        self._class_setup_directories = set()
        self._current_context = self._fresh_book = None
        self._current_history = []
        self._campaign_started = None
        self._terminal_campaign_hashes = None
        self.journal = self.report = self.initial_star = self._project_state = None
        self.status, self.phase, self.failure = "idle", "not_started", None
        self._busy = self._emitting = self._closing = self._finalizing = self._forward_failed = False
        self._cleanup_failed = False
        self.closed = False
        self._close_failures = []
        self._stream, self._sequence, self._last_tick = None, 0, None
        self._initial_hashes, self.setup_advances = {}, 0
        self._run_id = "browser-project-" + uuid.uuid4().hex
        self._relay = ProjectEventRelay(self._receive)
        self._class_relay = ProjectEventRelay(self._receive)
        self._setup_callback = self._project_callback = None
        self.scope = {
            "mode": "fresh_browser_campaign_project_runtime"
            if campaign
            else "fresh_browser_single_star_project_runtime",
            "project_campaign": campaign,
            "target_stars": options.stars,
            "campaign_max_seconds": campaign_seconds,
            "browser": "visible_chromium",
            "fresh_context": True,
            **(
                {"project_viewport": deepcopy(viewport), "viewport_policy": "fixed_context_css_pixels"}
                if viewport is not None
                else {}
            ),
            "credentials_recorded": False,
            "authentication_capture_enabled": False,
            "setup_max_advances": setup_max_advances,
            "setup_max_seconds": BrowserSetup.MAX_SECONDS,
            "observation_start_max_seconds": inspect.signature(start_planet_observation)
            .parameters["max_seconds"]
            .default,
            "automatic_deadline_increase": False,
            "no_planet_save_settle_seconds": model_options.get("no_planet_save_settle_seconds", 20),
            "project_limits": limits,
            "class_setup_limits": class_limits,
            "classification_source": "automatic_visible_reference_heuristic"
            if autonomous
            else "explicit_owned_class_receipt_or_reference_decision",
            "autonomous_decisions_enabled": autonomous,
            **({"supplied_stellar_inputs_enabled": True} if supplied else {}),
            **({"two_event_reference_enabled": True} if self._two_event_enabled else {}),
            **(
                {
                    "save_strategy": "autosave",
                    "persistence_verified": False,
                    "autosave_source_sha256": self._autosave_sha,
                    "autosave_manifest": json.loads(self._autosave_manifest),
                    "autosave_manifest_sha256": _sha(self._autosave_manifest),
                }
                if self._save_strategy == "autosave"
                else {}
            ),
            **(
                {
                    "single_event_reference_enabled": True,
                    "single_event_source_sha256": self._single_event_sha,
                    "single_event_policy": json.loads(self._single_event_policy),
                    "single_event_policy_sha256": _sha(self._single_event_policy),
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
            "decision_source": "local_reference_tools_not_learned" if autonomous else "explicit_reference",
            "automatic_inventory": True,
            "reference_planet_continuation": reference_planet_continuation,
            "positive_save_settle_seconds": positive_save_settle_seconds,
            "automatic_planet_classification": autonomous,
            "terrestrial_continuation": terrestrial_options is not None,
            "automatic_gas_identification": autonomous,
            "automatic_habitability_decision": autonomous,
            "automatic_classification": autonomous,
            "experimental_hr_matcher_enabled": autonomous,
            "assessment_enabled": False,
            "submission_enabled": False,
            "optimizer_updates": 0,
            "automatic_retry": False,
            "cancellation": "between_bounded_scheduled_calls",
        }

    @property
    def finished(self):
        return self.status in {"completed", "handoff", "stopped", "aborted"}

    @property
    def initial_star_dir(self):
        return self.output / "initial-star" if self.initial_star else None

    @property
    def trace_path(self):
        return self.output / "events.jsonl"

    def state(self):
        campaign = deepcopy(self._project_state) if self.project_campaign else None
        project = (
            deepcopy((campaign or {}).get("project_owner"))
            if self.project_campaign
            else deepcopy(self._project_state)
        )
        class_setup = deepcopy(self.class_setup.state()) if self.class_setup else None
        if isinstance(class_setup, dict) and class_setup.get("failure_reason") is not None:
            class_setup["failure_reason"] = _reason(BrowserSafetyStop(class_setup["failure_reason"]))
        return {
            **self.scope,
            "status": self.status,
            "phase": self.phase,
            "stage": "browser_project_class_setup"
            if self.phase in _CLASS_PHASES
            else "browser_project_star"
            if self.project
            else "browser_project_setup",
            "browser_phase": self.phase,
            "browser_status": "close_failed"
            if self.closed and self._close_failures
            else "closed"
            if self.closed
            else "visible"
            if self.page is not None
            else "not_launched",
            "finished": self.finished,
            "failure_reason": self.failure,
            "task_completed": not self.project_campaign and self.status == "completed",
            "project_completed": False,
            "campaign": campaign,
            "current_star": deepcopy(self._current_context),
            "target_workflows_verified": bool(
                self.project_campaign
                and self.status == "handoff"
                and self.phase == "awaiting_assessment"
                and not self.failure
                and (campaign or {}).get("target_workflows_verified") is True
            ),
            "event_forwarding_failed": self._forward_failed,
            "cleanup_failed": self._cleanup_failed,
            "credentials_consumed": self.credentials_consumed,
            "setup_stage": self.setup.stage if self.setup else None,
            "setup_advances": self.setup_advances,
            "initial_star": deepcopy(self.initial_star),
            "project_owner": project,
            **{
                key: deepcopy(project[key])
                for key in ("save_outcome", "save_outcome_uncertain")
                if isinstance(project, dict) and key in project
            },
            "class_setup": class_setup,
            "policy_stage": "reference_class_setup"
            if self.phase in _CLASS_PHASES
            else (project or {}).get("policy_stage"),
            "artifact_paths": {
                "setup": "setup",
                "initial_star": "initial-star" if self.initial_star else None,
                "reference_class_setup": self._current_context["class_output"] if self.class_setup else None,
                **(project or {}).get("artifact_paths", {}),
                **({"campaign_dir": "campaign"} if campaign else {}),
            },
            "reference_measurements": deepcopy((project or {}).get("reference_measurements"))
            if not self.failure
            else None,
            **(
                {"autonomous_decision": deepcopy(self.autonomous_decision)}
                if self.autonomous_decisions
                else {}
            ),
            "project_progress": deepcopy((campaign or project or {}).get("project_progress")),
            "trace_path": "events.jsonl" if self._stream is not None else None,
        }

    def _emit(self, kind, payload):
        item = RuntimeEvent(event=kind, payload=payload, run_id=self._run_id, sequence=self._sequence)
        line = json.dumps(item.model_dump(), separators=(",", ":"), allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            try:
                self._emitting = True
                self._callback(kind, json.loads(line)["payload"])
            except Exception:  # noqa: BLE001 - callback text can contain credentials
                self._forward_failed = True
                if not self._closing:
                    raise BrowserSafetyStop("project_runtime_event_forwarding_failed") from None
            finally:
                self._emitting = False

    def _receive(self, kind, payload):
        _require(not self.finished or self._closing, "already_stopped")
        if not self._closing:
            self._check_viewport()
        reference = payload.get("component") == "browser.reference_class"
        if reference and not self._closing:
            self._check_current()
        if kind == "state":
            if payload.get("component") in {"project.owner", "project.campaign"}:
                candidate = payload.get("component_state")
                if isinstance(candidate, dict) and "phase" in candidate:
                    if self.project_campaign:
                        self._adopt_campaign_context(candidate)
                    self._project_state = deepcopy(candidate)
            payload = {**payload, **self.state()}
        self._emit(kind, payload)
        _require(not self.finished or self._closing, "already_stopped")
        if not self._closing:
            self._check_viewport()
        if reference and not self._closing:
            self._check_current()

    def _setup_event(self, kind, payload):
        _require(kind == "state" and isinstance(payload, dict), "unsafe_setup_event")
        _require(payload.get("action_source") == "deterministic_setup", "unsafe_setup_source")
        if "setup_stage" in payload:
            _require(
                set(payload) == {"setup_stage", "action_source"} and payload["setup_stage"] in _STAGES,
                "unsafe_setup_stage",
            )
        else:
            _require(
                set(payload) == {"setup_evidence", "action_source", "starfield_sha256", "visible_point"}
                and payload["setup_evidence"] in _PIXELS
                and isinstance(payload["starfield_sha256"], str)
                and re.fullmatch(r"[a-f0-9]{64}", payload["starfield_sha256"]),
                "unsafe_setup_pixels",
            )
            point = payload["visible_point"]
            _require(
                point is None
                or (
                    isinstance(point, dict)
                    and set(point) == {"x", "y", "width", "height"}
                    and all(type(v) in {int, float} and math.isfinite(v) and v >= 0 for v in point.values())
                ),
                "unsafe_setup_geometry",
            )
        self._setup_callback(kind, payload)

    def _publish(self):
        if self.finished:
            return self.state()
        try:
            self._emit("state", self.state())
        except Exception as exc:  # noqa: BLE001 - sanitized protocol boundary
            self._stop(_reason(exc))
        return self.state()

    def _not_busy(self):
        if self._busy or self._emitting or self._finalizing:
            self._stop("project_runtime_reentrant_call")
            raise BrowserSafetyStop("project_runtime_reentrant_call")

    def start(self, *, paused=None):
        self._not_busy()
        _require(self.status == "idle", "already_started")
        paused = self._default_paused if paused is None else paused
        _require(type(paused) is bool, "invalid_pause_option")
        self.output.mkdir(parents=True, exist_ok=False)
        self._stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        self.journal = ProjectJournal(self.output, project_id="habworlds", attempt_id=self._run_id).create()
        persist_json(self.output / "scope.json", self.scope)
        self.status, self.phase = "paused" if paused else "running", "launch_pending"
        try:
            self._emit("hello", {"protocol_version": 1, **self.scope})
        except Exception as exc:  # noqa: BLE001 - sanitized protocol boundary
            self._stop(_reason(exc))
        return self._publish()

    def _launch(self):
        credentials = None
        try:
            credentials = self._credentials if self._credentials is not None else consume_credentials()
            self._credentials = None
            self.credentials_consumed = True
            # Never inherit login or Playwright debug values into subprocesses.
            for key in ("HABFLY_LOGIN_EMAIL", "HABFLY_LOGIN_PASSWORD", "DEBUG", "PWDEBUG", "DEBUG_FILE"):
                os.environ.pop(key, None)
            self.driver = self._driver_factory()
            self.browser = self.driver.chromium.launch(headless=False, timeout=30000)
            self.context = (
                self.browser.new_context(viewport=deepcopy(self.project_viewport))
                if self.project_viewport is not None
                else self.browser.new_context()
            )
            self.page = self.context.new_page()
            self._check_viewport()
            directory = self.output / "setup"
            directory.mkdir()
            self._setup_callback = self._relay.bind("browser.setup")
            self.setup = self._setup_factory(
                self.page,
                self.config,
                credentials,
                emit=self._setup_event,
                output=directory,
            )
            self.phase = "opening_preview"
        finally:
            credentials = self._credentials = None

    def _owned(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.output / path
        _require(
            path.resolve().is_relative_to(self.output)
            and not any(p.is_symlink() for p in (path, *path.parents)),
            "invalid_owned_path",
        )
        return path.resolve()

    def _pin_initial(self, receipt):
        directory = self.output / "initial-star"
        saved = json.loads((directory / "confirmed.json").read_bytes())
        _require(
            receipt == saved
            and isinstance(saved.get("star"), str)
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{0,79}", saved["star"])
            and saved.get("fresh_blank_numeric_answers_verified") is True
            and saved.get("stellar_observations_verified") is True
            and saved.get("class_selection_verified") is False
            and saved.get("action_source") == "deterministic_navigation"
            and saved.get("collection_count_verified") is False
            and type(saved.get("answer_writes")) is int
            and saved["answer_writes"] == 0
            and saved.get("task_completed") is False,
            "invalid_initial_star_receipt",
        )
        for path in directory.rglob("*"):
            path = self._owned(path)
            if path.is_file():
                _require(path.stat().st_size <= 32_000_000, "oversized_initial_evidence")
                self._initial_hashes[str(path.relative_to(self.output))] = _sha(path.read_bytes())
        _require("initial-star/stellar/observation.json" in self._initial_hashes, "missing_initial_capture")
        scope = json.loads(self._owned(directory / "scope.json").read_bytes())
        source = self._owned(self.output / "setup" / INITIAL_STARFIELD_IMAGE)
        _require(source.is_file() and source.stat().st_size <= 32_000_000, "missing_initial_starfield")
        png = source.read_bytes()
        validate_initial_selection(scope, png, saved.get("selected_point"))
        if "selection_schema_version" in scope:
            captured = self._owned(directory / INITIAL_STARFIELD_IMAGE)
            _require(
                self._initial_hashes.get(str(captured.relative_to(self.output))) == _sha(png),
                "initial_starfield_copy_changed",
            )
        self._initial_hashes[str(source.relative_to(self.output))] = _sha(png)
        self.initial_star = deepcopy(saved)
        if not self.project_campaign:
            self._current_context = {
                "ordinal": 1,
                "star": saved["star"],
                "fresh_dir": "initial-star",
                "fresh_sha256": self._initial_hashes["initial-star/confirmed.json"],
                "class_output": "reference-class",
                "owner_output": "active-star",
            }
        self._check_initial()
        persist_json(
            self.output / "initial-handoff.json", {"receipt": saved, "source_hashes": self._initial_hashes}
        )

    @staticmethod
    def _autosave_manifest_bytes():
        from .browser_autosave import autosave_manifest

        return json.dumps(
            autosave_manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()

    @staticmethod
    def _single_event_policy_bytes():
        from .planet_window_single_event import policy_manifest

        return json.dumps(policy_manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def _check_viewport(self):
        """Never silently resize after a capture or weaken outer-exposure guards."""

        _require(
            type(self.model_options.get("save_strategy", "explicit")) is str
            and self.model_options.get("save_strategy", "explicit") == self._save_strategy
            and type(getattr(self._save_strategy_options, "project_save_strategy", "explicit")) is str
            and getattr(self._save_strategy_options, "project_save_strategy", "explicit")
            == self._save_strategy
            and type(self.scope.get("save_strategy", "explicit")) is str
            and self.scope.get("save_strategy", "explicit") == self._save_strategy
            and (
                self._save_strategy != "autosave"
                or (
                    self.scope.get("persistence_verified") is False
                    and self.scope.get("autosave_source_sha256") == self._autosave_sha
                    and _sha(self._autosave_source.read_bytes()) == self._autosave_sha
                    and self.scope.get("autosave_manifest_sha256") == _sha(self._autosave_manifest)
                    and json.dumps(
                        self.scope.get("autosave_manifest"),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                    == self._autosave_manifest
                    and self._autosave_manifest_bytes() == self._autosave_manifest
                )
            ),
            "save_strategy_changed",
        )
        _require(
            self.model_options.get("allow_single_event_reference", False) is self._single_event_enabled
            and getattr(self._single_event_options, "project_single_event_reference", False)
            is self._single_event_enabled
            and self.scope.get("single_event_reference_enabled", False) is self._single_event_enabled
            and (
                not self._single_event_enabled
                or (
                    self.model_options.get("allow_baseline_band_reference") is True
                    and getattr(self._single_event_options, "project_baseline_band_reference", False) is True
                    and self.scope.get("single_event_source_sha256") == self._single_event_sha
                    and _sha(self._single_event_source.read_bytes()) == self._single_event_sha
                    and self.scope.get("single_event_policy_sha256") == _sha(self._single_event_policy)
                    and json.dumps(
                        self.scope.get("single_event_policy"),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                    == self._single_event_policy
                    and self._single_event_policy_bytes() == self._single_event_policy
                )
            ),
            "single_event_permission_changed",
        )
        _require(
            self.model_options.get("allow_baseline_band_reference", False) is self._baseline_band_enabled
            and getattr(self._baseline_band_options, "project_baseline_band_reference", False)
            is self._baseline_band_enabled
            and self.scope.get("baseline_band_reference_enabled", False) is self._baseline_band_enabled
            and (
                not self._baseline_band_enabled
                or (
                    self.model_options.get("allow_baseline_edge_reference") is True
                    and getattr(self._baseline_band_options, "project_baseline_edge_reference", False) is True
                    and self.reference_planet_continuation is True
                    and getattr(self._baseline_band_options, "project_reference_planet_continuation", False)
                    is True
                    and (
                        self.model_options.get("planet_pilot"),
                        self.model_options.get("planet_final_evaluation"),
                    )
                    == self._baseline_band_planet_sources
                    and self.scope.get("baseline_band_source_sha256") == self._baseline_band_sha
                    and _sha(self._baseline_band_source.read_bytes()) == self._baseline_band_sha
                )
            ),
            "baseline_band_permission_changed",
        )
        _require(
            self.model_options.get("two_event_reference", False) is self._two_event_enabled
            and getattr(self._two_event_options, "project_two_event_reference", False)
            is self._two_event_enabled
            and self.scope.get("two_event_reference_enabled", False) is self._two_event_enabled
            and (
                not self._two_event_enabled
                or self.model_options.get("shallow_reference") is True
                and getattr(self._two_event_options, "project_reference_shallow_transits", False) is True
            ),
            "two_event_permission_changed",
        )
        _require(
            self.model_options.get("allow_baseline_edge_reference", False) is self._baseline_edge_enabled
            and getattr(self._baseline_edge_options, "project_baseline_edge_reference", False)
            is self._baseline_edge_enabled
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

        _require(
            type(self.supplied_stellar_inputs) is bool
            and self.supplied_stellar_inputs is self._supplied_settings[0]
            and (
                self.scope.get("supplied_stellar_inputs_enabled") is True
                and self.model_options.get("planet_supplied_evaluation") == self._supplied_settings[1]
                and isinstance(self.terrestrial_options, dict)
                and self.terrestrial_options.get("supplied_evaluation") == self._supplied_settings[2]
                if self.supplied_stellar_inputs
                else "supplied_stellar_inputs_enabled" not in self.scope
                and "planet_supplied_evaluation" not in self.model_options
                and (
                    self.terrestrial_options is None or "supplied_evaluation" not in self.terrestrial_options
                )
            ),
            "supplied_input_settings_changed",
        )

        def same(value):
            return (
                value is None
                if self._viewport_expected is None
                else isinstance(value, dict)
                and set(value) == {"width", "height"}
                and all(type(v) is int for v in value.values())
                and value == self._viewport_expected
            )

        _require(
            same(self.project_viewport) and same(self.scope.get("project_viewport")),
            "viewport_settings_changed",
        )
        if self._viewport_expected is not None:
            _require(
                self.scope.get("viewport_policy") == "fixed_context_css_pixels", "viewport_settings_changed"
            )
            if self.page is not None:
                _require(same(self.page.viewport_size), "viewport_changed")

    def _check_initial(self):
        self._check_viewport()
        expected_campaign, expected_target, expected_seconds = self._campaign_identity

        def same_seconds(value):
            return type(value) is type(expected_seconds) and value == expected_seconds

        _require(
            self.project_campaign is self.scope["project_campaign"] is expected_campaign
            and getattr(self._campaign_options, "project_campaign", False) is expected_campaign
            and type(self.target_stars) is type(self.scope["target_stars"]) is int
            and type(self._campaign_options.stars) is int
            and self.target_stars
            == self.scope["target_stars"]
            == self._campaign_options.stars
            == expected_target
            and same_seconds(self.campaign_max_seconds)
            and same_seconds(self.scope["campaign_max_seconds"])
            and same_seconds(getattr(self._campaign_options, "project_campaign_max_seconds", None)),
            "campaign_settings_changed",
        )
        _require(
            self.autonomous_decisions is self._autonomous_authorized
            and self.scope["autonomous_decisions_enabled"] is self._autonomous_authorized,
            "autonomous_decision_authority_changed",
        )
        for name, checksum in self._autonomous_hashes.items():
            path = self._owned(name)
            _require(
                path.is_file() and _sha(path.read_bytes()) == checksum, "autonomous_decision_source_changed"
            )
        if (
            self.project_campaign
            and self.campaign_max_seconds != "uncapped"
            and self._campaign_started is not None
            and not self._closing
        ):
            _require(
                self._clock() - self._campaign_started < self.campaign_max_seconds, "campaign_time_limit"
            )
        for name, digest in self._initial_hashes.items():
            path = self._owned(name)
            _require(path.is_file() and _sha(path.read_bytes()) == digest, "initial_source_changed")
        _require(
            not any(
                (self.output / "initial-star" / name).exists()
                for name in ("stopped.json", "invalidated.json")
            ),
            "initial_source_stopped",
        )
        if self._fresh_book is not None:
            self._fresh_book.unchanged()

    def _check_current(self):
        self._check_initial()
        self._check_class_setup()
        _require(self._current_context is not None, "current_star_not_ready")
        if self.project_campaign:
            _require(
                self.project is not None and self.project.current_star_context() == self._current_context,
                "current_star_context_changed",
            )

    def _adopt_campaign_context(self, state):
        """Only adopt after the campaign's separate, verified fresh-source step."""
        self._check_initial()
        context = state.get("current_star")
        _require(
            isinstance(context, dict)
            and set(context)
            == {"ordinal", "star", "fresh_dir", "fresh_sha256", "class_output", "owner_output"}
            and self.project is not None
            and context == self.project.current_star_context()
            and isinstance(context.get("star"), str)
            and type(context.get("ordinal")) is int
            and 1 <= context["ordinal"] <= self.target_stars
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{0,79}", context["star"])
            and state.get("star", "").casefold() == context["star"].casefold()
            and state.get("task_completed") is False
            and state.get("project_completed") is False
            and (
                state.get("project_owner") is None
                or isinstance(state["project_owner"], dict)
                and state["project_owner"].get("star", "").casefold() == context["star"].casefold()
            ),
            "invalid_campaign_current_star",
        )
        if context == self._current_context:
            return
        previous = self._current_context
        ordinal = context["ordinal"]
        _require(
            type(ordinal) is int
            and ordinal == (previous["ordinal"] + 1 if previous else 1)
            and ordinal <= self.target_stars
            and state.get("phase") == "initializing_star"
            and (previous is None or self.phase == "adopting_fresh_star")
            and context["star"].casefold() not in {c["star"].casefold() for c in self._current_history},
            "unverified_campaign_star_transition",
        )
        base = f"campaign/stars/{ordinal:03d}"
        _require(
            context["class_output"] == base + "/reference-class"
            and context["owner_output"] == base + "/owner"
            and context["fresh_dir"]
            == (
                "initial-star"
                if ordinal == 1
                else f"campaign/transitions/{ordinal - 1:03d}-to-{ordinal:03d}/picker"
            )
            and isinstance(context["fresh_sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", context["fresh_sha256"]),
            "invalid_campaign_context_paths",
        )
        if self._fresh_book is None:
            self._fresh_book = _Evidence(self.output)
        _require(
            self._fresh_book.json(self.output / f"campaign/star-{ordinal:03d}.json") == context,
            "campaign_context_receipt_changed",
        )
        receipt, _, _ = _fresh(
            self._fresh_book,
            {
                "path": str(self._owned(context["fresh_dir"]) / "confirmed.json"),
                "sha256": context["fresh_sha256"],
            },
        )
        _require(receipt["star"].casefold() == context["star"].casefold(), "current_fresh_star_mismatch")
        if ordinal == 1:
            _require(
                context["fresh_sha256"] == self._initial_hashes["initial-star/confirmed.json"]
                and context["star"].casefold() == self.initial_star["star"].casefold(),
                "campaign_initial_star_mismatch",
            )
        # Retain previous class/source hashes, but no stale class decision UI or
        # callback is allowed to describe the newly adopted star.
        self._check_class_setup()
        self._class_relay.retire()
        self.class_setup = self._class_setup_report = self._class_decision = None
        self._current_context = deepcopy(context)
        self.autonomous_decision = self._autonomous_pending = None
        self._current_history.append(deepcopy(context))
        if previous is not None:
            self.phase = "initializing_star"

    def _class_directory(self):
        self._check_current()
        return self._owned(self._current_context["class_output"])

    def _current_class_config(self):
        if not self.project_campaign:
            return self.config
        current = self.project.config.model_copy(deep=True)
        before, after = self.config.model_dump(mode="json"), current.model_dump(mode="json")
        # Campaign navigation refreshes only the simulation's public required
        # labels from the verified new capture; origin/path/frame guards persist.
        for settings in (before, after):
            for frame in settings["frames"]:
                if frame["url"] == SIMULATION_URL:
                    frame["required_text"] = []
        _require(before == after, "campaign_class_boundary_changed")
        return current

    def _check_class_setup(self):
        for directory in self._class_setup_directories:
            _require(
                not any((directory / n).exists() for n in ("stopped.json", "invalidated.json")),
                "class_setup_invalidated",
            )
        for name, checksum in self._class_setup_hashes.items():
            path = self._owned(name)
            _require(path.is_file() and _sha(path.read_bytes()) == checksum, "class_setup_source_changed")

    def _capture_class_setup(self):
        report = deepcopy(self.class_setup.report)
        directory = self._class_directory()
        _require(
            isinstance(report, dict)
            and report == self.class_setup.state()
            and report.get("task_completed") is False
            and report.get("scientific_verified") is False
            and report.get("training_label") is False
            and report.get("classification_learned") is False
            and isinstance(report.get("star"), str)
            and report["star"].casefold() == self._current_context["star"].casefold()
            and report.get("selected_class") == self._class_decision["selected_class"]
            and report.get("lifetime_prefix") == self._class_decision["lifetime_prefix"]
            and json.loads((directory / "report.json").read_bytes()) == report,
            "class_setup_unverified",
        )
        if report.get("status") == "stopped":
            _require(
                report.get("finished") is True and report.get("setup_verified") is False,
                "class_setup_unverified",
            )
            # A failed, same-star persisted child report is not a successful
            # handoff, but its sanitized cause must survive the outer stop.
            raise BrowserSafetyStop(_reason(BrowserSafetyStop(report.get("failure_reason"))))
        _require(
            report.get("setup_verified") is True and report.get("status") == "completed",
            "class_setup_unverified",
        )
        _require(
            isinstance(report.get("source_hashes"), dict) and bool(report["source_hashes"]),
            "class_setup_missing_sources",
        )
        for name, checksum in report["source_hashes"].items():
            _require(
                isinstance(name, str)
                and isinstance(checksum, str)
                and re.fullmatch(r"[a-f0-9]{64}", checksum),
                "invalid_class_setup_source",
            )
            path = self._owned(name)
            _require(
                path.is_file() and path.stat().st_size <= 32_000_000 and _sha(path.read_bytes()) == checksum,
                "class_setup_source_changed",
            )
            self._class_setup_hashes[str(path.relative_to(self.output))] = checksum
        self._class_setup_hashes[str((directory / "report.json").relative_to(self.output))] = _sha(
            (directory / "report.json").read_bytes()
        )
        self._class_setup_directories.add(directory)
        self._class_setup_report = report
        self._check_class_setup()
        persist_json(
            directory.parent / "reference-class-handoff.json",
            {
                "selected_class": self._class_decision["selected_class"],
                "lifetime_prefix": self._class_decision["lifetime_prefix"],
                "class_dir": str((directory / "class").relative_to(self.output)),
                "source_hashes": self._class_setup_hashes,
                "setup_verified": True,
                "task_completed": False,
            },
        )
        self._class_relay.retire()
        self.phase = "class_setup_handoff"

    def _sync_project(self):
        if self.project_campaign:
            return self._sync_campaign()
        self._project_state = deepcopy(self.project.state())
        _require(
            self._project_state.get("star", "").casefold() == self.initial_star["star"].casefold(),
            "project_star_changed",
        )
        self.phase = self._project_state["phase"]
        if self.project.finished:
            _require(
                self._project_state["status"] in {"completed", "handoff", "stopped", "aborted"},
                "invalid_project_terminal_state",
            )
            completed = self._project_state["status"] == "completed"
            _require(
                self._project_state.get("task_completed") is completed, "invalid_project_completion_claim"
            )
            if completed:
                stars = [
                    star
                    for star in self.journal.load().reduce().stars.values()
                    if star.name.casefold() == self.initial_star["star"].casefold()
                ]
                _require(len(stars) == 1 and stars[0].task_completed, "missing_canonical_task_receipt")
            self.status = self._project_state["status"]
            failure = self._project_state.get("failure_reason")
            self.failure = _reason(BrowserSafetyStop(failure)) if failure else None
            self._finish()
        elif self.phase in _HANDOFF_PHASES:
            if not self.autonomous_decisions:
                self.status = "paused"

    def _sync_campaign(self):
        state = deepcopy(self.project.state())
        self._adopt_campaign_context(state)
        owner = state.get("project_owner")
        _require(
            state.get("task_completed") is False
            and state.get("project_completed") is False
            and state.get("target_stars") == self.target_stars
            and state.get("max_seconds") == self.campaign_max_seconds
            and type(state.get("max_seconds")) is type(self.campaign_max_seconds)
            and state.get("owner_limits") == {"max_seconds": 1800, "max_advances": 512}
            and (
                owner is None
                or isinstance(owner, dict)
                and owner.get("star", "").casefold() == self._current_context["star"].casefold()
            ),
            "invalid_campaign_state",
        )
        self._project_state = state
        self.phase = state["phase"]
        if self.project.finished:
            _require(
                state.get("status") in {"handoff", "stopped", "aborted"},
                "invalid_campaign_terminal_state",
            )
            if state.get("target_workflows_verified") is True:
                progress = self.journal.load().reduce()
                _require(
                    state["status"] == "handoff"
                    and self.phase == "awaiting_assessment"
                    and state.get("failure_reason") is None
                    and state == self.project.report
                    and json.loads(self._owned("campaign/report.json").read_bytes()) == state
                    and state.get("project_progress") == progress.report()
                    and len(progress.stars) == self.target_stars
                    and all(s.task_completed for s in progress.stars.values())
                    and {s.name.casefold() for s in progress.stars.values()}
                    == {c["star"].casefold() for c in self._current_history}
                    and not progress.pending
                    and not progress.receipts
                    and not progress.reservations,
                    "unverified_campaign_target_handoff",
                )
                sources = state.get("source_sha256")
                _require(isinstance(sources, dict) and bool(sources), "campaign_terminal_sources_missing")
                self._terminal_campaign_hashes = {
                    **deepcopy(sources),
                    str(self.journal.path.relative_to(self.output)): _sha(self.journal.path.read_bytes()),
                    "campaign/report.json": _sha(self._owned("campaign/report.json").read_bytes()),
                }
                self._check_terminal_campaign()
            else:
                _require(self.phase != "awaiting_assessment", "unverified_campaign_assessment_handoff")
            self.status = state["status"]
            failure = state.get("failure_reason")
            self.failure = _reason(BrowserSafetyStop(failure)) if failure else None
            self._finish()
        elif self.phase in _HANDOFF_PHASES:
            if not self.autonomous_decisions:
                self.status = "paused"
        else:
            _require(self.phase in _OWNER_PHASES | _CAMPAIGN_PHASES, "unknown_campaign_phase")

    def _check_terminal_campaign(self):
        if self._terminal_campaign_hashes is None:
            return
        self._check_current()
        _require(self.project.state() == self._project_state, "campaign_terminal_state_changed")
        for name, checksum in self._terminal_campaign_hashes.items():
            _require(
                isinstance(name, str)
                and not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and isinstance(checksum, str)
                and re.fullmatch(r"[a-f0-9]{64}", checksum),
                "invalid_campaign_terminal_source",
            )
            path = self._owned(name)
            _require(
                path.is_file() and path.stat().st_size <= 32_000_000 and _sha(path.read_bytes()) == checksum,
                "campaign_terminal_source_changed",
            )

    def _make_autonomous_decision(self):
        """Local evidence interpretation only; no browser calls or learned claims."""
        from .autonomous_planet import decide_planet_handoff
        from .autonomous_stellar import decide_stellar_reference

        context = self._current_context
        if self.phase == "awaiting_class_source":
            capture = str(Path(context["fresh_dir"]) / "stellar/observation.json")
            return decide_stellar_reference(
                self.output,
                self._owned(context["fresh_dir"]),
                fresh_star_sha256=context["fresh_sha256"],
                source_root=Path.cwd(),
                expected_star=context["star"],
                capture_sha256=(self._fresh_book.hashes if self.project_campaign else self._initial_hashes)[
                    capture
                ],
            )
        _require(
            self.phase in {"awaiting_planet_class", "awaiting_gases", "awaiting_habitability"},
            "autonomous_unsupported_handoff",
        )
        owner = (
            (self._project_state or {}).get("project_owner") if self.project_campaign else self._project_state
        )
        return decide_planet_handoff(
            phase=self.phase,
            run_history=self.output,
            owner_state=deepcopy(owner),
            expected_star=context["star"],
        )

    def _advance_autonomous_decision(self):
        """Prepare and apply in separate scheduled stages, before any native action.

        No classifier score is a receipt or learned confidence. An unsupported
        source stops the run; it never falls back to a supplied grading answer.
        """
        self._check_current()
        phase, context = self.phase, deepcopy(self._current_context)
        key = (context["ordinal"], phase)
        _require(key not in self._autonomous_attempts, "autonomous_decision_already_attempted")
        if self._autonomous_pending is None:
            try:
                decision = self._make_autonomous_decision()
            except Exception as exc:  # noqa: BLE001 - only allowlisted local error codes may escape
                code = str(exc)
                if re.fullmatch(r"autonomous_(?:stellar|planet|gas|habitability)_[a-z0-9_]{1,100}", code):
                    raise BrowserSafetyStop(code) from None
                raise BrowserSafetyStop("project_runtime_autonomous_decision_failed") from None
            _require(
                isinstance(decision, dict)
                and (
                    isinstance(decision.get("payload"), dict)
                    or decision.get("status") == "abstained"
                    and decision.get("payload") is None
                ),
                "invalid_autonomous_decision",
            )
            sources = decision.get("source_sha256")
            _require(isinstance(sources, dict) and 1 <= len(sources) <= 256, "autonomous_sources_missing")
            for name, checksum in sources.items():
                _require(
                    isinstance(name, str)
                    and not Path(name).is_absolute()
                    and isinstance(checksum, str)
                    and re.fullmatch(r"[a-f0-9]{64}", checksum),
                    "invalid_autonomous_source",
                )
                path = self._owned(name)
                _require(
                    path.is_file()
                    and path.stat().st_size <= 32_000_000
                    and _sha(path.read_bytes()) == checksum,
                    "autonomous_source_changed",
                )
            self._check_current()
            record = {"current_star": context, "phase": phase, "decision": decision, "browser_actions": 0}
            path = self._owned(f"autonomous-decisions/star-{context['ordinal']:03d}/{phase}.json")
            _require(not path.exists(), "autonomous_decision_already_recorded")
            path.parent.mkdir(parents=True, exist_ok=True)
            persist_json(path, record)
            self._autonomous_hashes.update(sources)
            self._autonomous_hashes[str(path.relative_to(self.output))] = _sha(path.read_bytes())
            self._autonomous_pending = {
                "path": str(path.relative_to(self.output)),
                "record": deepcopy(record),
            }
            self.autonomous_decision = {
                "star": context["star"],
                "ordinal": context["ordinal"],
                "phase": phase,
                "status": "abstained" if decision.get("status") == "abstained" else "prepared",
                "decision_kind": decision.get("decision_kind"),
                "payload": deepcopy(decision["payload"]),
                "source": "local_reference_tools",
                "learned": False,
                "scientific_verified": False,
                "training_label": False,
                "artifact": str(path.relative_to(self.output)),
            }
            if decision.get("status") == "abstained":
                reason = decision.get("reason")
                self.autonomous_decision["reason"] = (
                    reason
                    if isinstance(reason, str) and re.fullmatch(r"[a-z0-9_]{1,100}", reason)
                    else "unsupported_reference_evidence"
                )
                raise BrowserSafetyStop("project_runtime_autonomous_reference_abstained")
            return
        pending = self._autonomous_pending
        record = pending["record"]
        _require(
            record["current_star"] == context
            and record["phase"] == phase
            and json.loads(self._owned(pending["path"]).read_bytes()) == record,
            "autonomous_pending_context_changed",
        )
        # Reinterpret the same frozen public evidence, never repaired/new inputs.
        _require(self._make_autonomous_decision() == record["decision"], "autonomous_reference_changed")
        self._check_current()
        self._autonomous_attempts.add(key)  # Any later failure is terminal, not a retry.
        payload = record["decision"]["payload"]
        self.autonomous_decision["status"] = "applied"
        if phase == "awaiting_class_source":
            _require(self.class_setup is None, "reference_class_handoff_not_ready")
            self._install_reference_class(**payload)
        else:
            _require(self.project is not None, "autonomous_owner_missing")
            method = {
                "awaiting_planet_class": "provide_planet_class",
                "awaiting_gases": "provide_autonomous_gases",
                "awaiting_habitability": "provide_habitability",
            }[phase]
            getattr(self.project, method)(**payload)
            if not self.finished:
                self._sync_project()
                _require(self.phase != phase, "autonomous_decision_not_accepted")
                if self.status == "running" and not self.finished:
                    self.project.resume()
                    if not self.finished:
                        self._sync_project()
        if not self.finished:
            self._check_current()
            self._autonomous_pending = None

    def _advance(self):
        self._not_busy()
        if self.finished:
            return self.state()
        self._busy = True
        try:
            _require(self.autonomous_decisions is self._autonomous_authorized, "autonomous_authority_changed")
            self._check_viewport()
            self._last_tick = self._clock()
            if self.phase in _HANDOFF_PHASES and self.autonomous_decisions:
                self._advance_autonomous_decision()
            elif self.phase == "launch_pending":
                self._launch()
            elif self.phase == "opening_preview":
                self.page.goto(self.config.url, wait_until="domcontentloaded", timeout=30000)
                self.phase = "setting_up"
            elif self.phase == "setting_up":
                _require(self.setup_advances < self.scope["setup_max_advances"], "setup_advance_limit")
                self.setup_advances += 1
                if self.setup.advance() == "stellar":
                    self.phase = "capturing_initial_star"
            elif self.phase == "capturing_initial_star":
                receipt = self._capture_initial(self.setup, self.output / "initial-star")
                if self.finished:
                    return self.state()
                self._pin_initial(receipt)
                self._relay.retire()
                self.phase = "initializing_project"
            elif self.phase == "initializing_project":
                self._check_initial()
                self._project_callback = self._relay.bind(
                    "project.campaign" if self.project_campaign else "project.owner",
                    star=None if self.project_campaign else self.initial_star["star"],
                )
                factory = self._campaign_factory if self.project_campaign else self._steps_factory
                child_options = (
                    {
                        "initial_star_dir": self.output / "initial-star",
                        "initial_star_sha256": self._initial_hashes["initial-star/confirmed.json"],
                        "target_stars": self.target_stars,
                        "max_seconds": self.campaign_max_seconds,
                        "cancelled": lambda: self.finished,
                        "_clock": self._clock,
                    }
                    if self.project_campaign
                    else {
                        "star": self.initial_star["star"],
                        "automatic_inventory": True,
                        **self.project_limits,
                    }
                )
                child = factory(
                    self.page,
                    self.config,
                    self.output / ("campaign" if self.project_campaign else "active-star"),
                    run_history=self.output,
                    journal=self.journal,
                    model_options=self.model_options,
                    emit=self._project_callback,
                    interval=self.interval,
                    reference_planet_continuation=self.reference_planet_continuation,
                    positive_save_settle_seconds=self.positive_save_settle_seconds,
                    terrestrial_options=self.terrestrial_options,
                    **child_options,
                )
                self.project = child
                if self.finished:
                    child.abort()
                    return self.state()
                if self.project_campaign:
                    self._campaign_started = self._clock()
                child.start(paused=self.status == "paused" if self.project_campaign else True)
                if not self.finished:
                    self._sync_project()
            elif self.phase in {"setting_reference_class", "setting_lifetime_prefix"}:
                self._check_current()
                self.class_setup.advance()
                if not self.finished:
                    self._check_current()
                    if self.class_setup.finished:
                        self._capture_class_setup()
                    else:
                        self.phase = (
                            "setting_lifetime_prefix"
                            if self.class_setup.state().get("phase") == "prefix_pending"
                            else "setting_reference_class"
                        )
            elif self.phase == "class_setup_handoff":
                self._check_current()
                self._check_class_setup()
                self._accept_class_source(
                    class_dir=self._class_directory() / "class",
                    selected_class=self._class_decision["selected_class"],
                    lifetime_prefix=self._class_decision["lifetime_prefix"],
                )
                if not self.finished and self.status == "running":
                    self.project.resume()
                    if not self.finished:
                        self._sync_project()
            elif self.project:
                _require(
                    self.phase in _OWNER_PHASES | (_CAMPAIGN_PHASES if self.project_campaign else set()),
                    "explicit_handoff_required",
                )
                self._check_current()
                self._check_class_setup()
                self.project.step() if self.status == "paused" else self.project.tick()
                if not self.finished:
                    self._sync_project()
            else:
                raise BrowserSafetyStop("project_runtime_invalid_phase")
        except (KeyboardInterrupt, SystemExit):
            self.abort()
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize errors, never widen browser guards
            if not self.finished:
                launching = self.phase == "launch_pending"
                self._stop(_reason(exc))
                if launching:
                    self._release_browser()
        finally:
            self._busy = False
        return self._publish()

    def step(self):
        if self.finished:
            return self.state()
        _require(self.status == "paused", "pause_before_step")
        if self.autonomous_decisions is not self._autonomous_authorized:
            self._stop("project_runtime_autonomous_authority_changed")
            return self.state()
        if self.phase in _HANDOFF_PHASES and not self.autonomous_decisions:
            return self._publish()  # guidance only; never guess or consume a handoff
        return self._advance()

    def tick(self):
        return self._advance() if self.status == "running" else self.state()

    def advance_if_due(self):
        if self.status == "running" and (
            self._last_tick is None or self._clock() - self._last_tick >= self.interval
        ):
            return self.tick()
        return self.state()

    def pause(self):
        if self.finished:
            return self.state()
        _require(self.status in {"running", "paused"}, "not_started")
        self.status = "paused"
        if self.project:
            self.project.pause()
        return self.state() if self._busy or self._emitting else self._publish()

    def resume(self):
        self._not_busy()
        _require(
            self.status == "paused" and (self.phase not in _HANDOFF_PHASES or self.autonomous_decisions),
            "explicit_handoff_required",
        )
        if self.project and self.phase not in _CLASS_PHASES | _HANDOFF_PHASES:
            self.project.resume()
            if self.finished:
                return self.state()
        self.status = "running"
        return self._publish()

    def provide_class(self, *, class_dir, selected_class, lifetime_prefix):
        self._not_busy()
        _require(
            self.project is not None and self.phase == "awaiting_class_source" and self.status == "paused",
            "class_handoff_not_ready",
        )
        try:
            self._accept_class_source(
                class_dir=class_dir, selected_class=selected_class, lifetime_prefix=lifetime_prefix
            )
        except Exception as exc:  # noqa: BLE001 - source paths and malformed values remain private
            self._stop(_reason(exc))
        return self._publish()

    def _accept_class_source(self, *, class_dir, selected_class, lifetime_prefix):
        self._check_current()
        directory = self._owned(class_dir)
        context = self._current_context
        source = validate_star_class_source(self.output, directory, context["star"], selected_class)
        fresh = self._owned(context["fresh_dir"])
        capture_name = str((fresh / "stellar/observation.json").relative_to(self.output))
        expected_capture_sha = (
            self._fresh_book.hashes[capture_name]
            if self.project_campaign
            else self._initial_hashes[capture_name]
        )
        _require(
            self._owned(source["fresh_star"]) == fresh
            and source["source_capture_sha256"] == expected_capture_sha,
            "class_not_bound_to_current_star" if self.project_campaign else "class_not_bound_to_initial_star",
        )
        self._check_current()
        self.project.provide_class(
            class_dir=directory, selected_class=selected_class, lifetime_prefix=lifetime_prefix
        )
        if not self.finished:
            self._check_class_setup()
            self._sync_project()

    def provide_reference_class(self, *, selected_class, reference_rationale, lifetime_prefix):
        """Create local setup only; the next advance owns the first native action."""
        self._not_busy()
        _require(
            self.project is not None
            and self.phase == "awaiting_class_source"
            and self.status == "paused"
            and self.class_setup is None,
            "reference_class_handoff_not_ready",
        )
        try:
            self._install_reference_class(
                selected_class=selected_class,
                reference_rationale=reference_rationale,
                lifetime_prefix=lifetime_prefix,
            )
        except Exception as exc:  # noqa: BLE001 - decision text and source paths remain private
            self._stop(_reason(exc))
        return self._publish()

    def _install_reference_class(self, *, selected_class, reference_rationale, lifetime_prefix):
        self._check_current()
        context = self._current_context
        callback = self._class_relay.bind("browser.reference_class", star=context["star"])
        child = self._class_factory(
            self.page,
            self._current_class_config(),
            self._class_directory(),
            run_history=self.output,
            fresh_star=self._owned(context["fresh_dir"]),
            selected_class=selected_class,
            reference_rationale=reference_rationale,
            lifetime_prefix=lifetime_prefix,
            emit=callback,
            _clock=self._clock,
            **self.class_limits,
        )
        self.class_setup = child
        if self.finished:
            child.abort()
            return
        self._class_decision = {"selected_class": selected_class, "lifetime_prefix": lifetime_prefix}
        self.phase = "setting_reference_class"
        self._check_current()

    def provide_planet_class(self, *, name, rationale):
        """Persist an explicit reference choice; native dispatch remains a later step."""
        self._not_busy()
        _require(
            self.reference_planet_continuation
            and self.project is not None
            and self.phase == "awaiting_planet_class"
            and self.status == "paused",
            "planet_class_handoff_not_ready",
        )
        try:
            self._check_current()
            self._check_class_setup()
            self.project.provide_planet_class(name=name, rationale=rationale)
            if not self.finished:
                self._check_current()
                self._check_class_setup()
                self._sync_project()
        except Exception as exc:  # noqa: BLE001 - decision details remain private
            self._stop(_reason(exc))
        return self._publish()

    def _provide_terrestrial(self, phase, method, payload):
        self._not_busy()
        _require(
            self.terrestrial_options is not None
            and self.project is not None
            and self.phase == phase
            and self.status == "paused",
            "terrestrial_handoff_not_ready",
        )
        try:
            self._check_current()
            self._check_class_setup()
            getattr(self.project, method)(**payload)
            if not self.finished:
                self._check_current()
                self._check_class_setup()
                self._sync_project()
        except Exception as exc:  # noqa: BLE001 - reference text stays out of error messages
            self._stop(_reason(exc))
        return self._publish()

    def provide_gases(self, *, gases, rationale, supplied_greenhouse_increment):
        return self._provide_terrestrial(
            "awaiting_gases",
            "provide_gases",
            {
                "gases": gases,
                "rationale": rationale,
                "supplied_greenhouse_increment": supplied_greenhouse_increment,
            },
        )

    def provide_habitability(self, *, choice, rationale):
        return self._provide_terrestrial(
            "awaiting_habitability",
            "provide_habitability",
            {
                "choice": choice,
                "rationale": rationale,
            },
        )

    def provide_inventory(self, *, inventory_dir, inventory_sha256):
        self._not_busy()
        _require(
            self.project is not None and self.phase == "awaiting_inventory" and self.status == "paused",
            "inventory_handoff_not_ready",
        )
        try:
            self._check_initial()
            self.project.provide_inventory(
                inventory_dir=self._owned(inventory_dir), inventory_sha256=inventory_sha256
            )
            if not self.finished:
                self._sync_project()
        except Exception as exc:  # noqa: BLE001 - importer paths remain private
            self._stop(_reason(exc))
        return self._publish()

    def _stop(self, reason, *, aborted=False):
        if self.report is not None or self._closing:
            return
        self.status = self.phase = "aborted" if aborted else "stopped"
        self.failure, self._closing, self._credentials = reason, True, None
        try:
            for child in (self.setup, self.class_setup, self.project):
                if child is not None:
                    try:
                        child.close()
                    except Exception:  # noqa: BLE001 - already terminal; no retry or driver text
                        self._cleanup_failed = True
            if self._stream is not None and not self._finalizing:
                if not aborted:
                    self._emit("error", {"type": "BrowserProjectRuntimeStop", "message": self.failure})
                self._finish()
        finally:
            self._closing = False

    def _finish(self):
        if self.report is not None or self._finalizing:
            return
        self._finalizing = True
        try:
            report = self.state()
            try:
                self._emit("episode_summary", {**report, "completed": report["task_completed"]})
            except Exception:  # noqa: BLE001 - terminal callback failure must not claim success
                self._forward_failed = True
            if not self._forward_failed:
                try:
                    if self.status in {"completed", "handoff"}:
                        self._check_viewport()
                    self._check_terminal_campaign()
                except Exception:  # noqa: BLE001 - changed proof revokes the outer target claim
                    self.status = self.phase = "stopped"
                    self.failure = "project_runtime_terminal_campaign_validation_failed"
            if self._forward_failed:
                self.status = self.phase = "stopped"
                self.failure = "project_runtime_event_forwarding_failed"
            final = self.state()
            # Consumers use state for live status/counters; the summary alone
            # must not leave a clean terminal handoff looking like active work.
            try:
                self._emit("state", final)
            except Exception:  # noqa: BLE001 - the new publication must remain fail-closed
                self._forward_failed = True
            if not self._forward_failed:
                try:
                    if self.status in {"completed", "handoff"}:
                        self._check_viewport()
                    self._check_terminal_campaign()
                except Exception:  # noqa: BLE001 - recheck proof after the new state callback too
                    self.status = self.phase = "stopped"
                    self.failure = "project_runtime_terminal_campaign_validation_failed"
            if self._forward_failed:
                self.status = self.phase = "stopped"
                self.failure = "project_runtime_event_forwarding_failed"
            corrected = self.state()
            if corrected != final:
                try:
                    self._emit("state", corrected)
                except Exception:  # noqa: BLE001 - persist the terminal forwarding failure
                    self._forward_failed = True
                if self._forward_failed:
                    self.status = self.phase = "stopped"
                    self.failure = "project_runtime_event_forwarding_failed"
                    stopped = self.state()
                    if stopped != corrected:
                        self._emit("state", stopped)  # forwarding is now disabled
                    corrected = stopped
            final = corrected
            self.report = final
            persist_json(self.output / "report.json", final)
        finally:
            self._relay.retire()
            self._class_relay.retire()
            self._stream.close()
            self._finalizing = False

    def abort(self):
        if not self.finished:
            self._stop("operator_aborted", aborted=True)
        return self.state()

    def _release_browser(self):
        failures = []
        for name, method in (("context", "close"), ("browser", "close"), ("driver", "stop")):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception:  # noqa: BLE001 - record only which cleanup failed
                    failures.append(name)
                finally:
                    setattr(self, name, None)
        self.closed = True
        self._close_failures = failures
        if self.output.exists():
            persist_json(
                self.output / "browser-close.json", {"closed": not failures, "failed_resources": failures}
            )

    def close(self):
        self.abort()
        if not self.closed:
            self._release_browser()
        return self.state()

    def command(self, message):
        command = RuntimeCommand.model_validate(message)
        payload = command.payload
        if command.command == "start" and set(payload) <= {"paused"}:
            return self.start(**payload)
        if command.command == "step":
            if not payload:
                return self.step()
            _require(not self.autonomous_decisions, "autonomous_mode_rejects_supplied_decisions")
            if set(payload) == {"class_source"}:
                return self.provide_class(**payload["class_source"])
            if set(payload) == {"reference_class"}:
                return self.provide_reference_class(**payload["reference_class"])
            if set(payload) == {"planet_class"}:
                return self.provide_planet_class(**payload["planet_class"])
            if set(payload) == {"gases"}:
                return self.provide_gases(**payload["gases"])
            if set(payload) == {"habitability"}:
                return self.provide_habitability(**payload["habitability"])
            if set(payload) == {"inventory"}:
                return self.provide_inventory(**payload["inventory"])
        if not payload and command.command in {"pause", "resume", "abort"}:
            return getattr(self, command.command)()
        raise ValueError("Unsupported project runtime command or payload")
