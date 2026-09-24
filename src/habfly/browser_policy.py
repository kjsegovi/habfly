"""Supervised transfer of the frozen three-field policy, never course completion.

Only the already-tested NumericSession can touch native browser controls. The
policy gets the same local tool controls it learned; no expert or grading case
is used. Copies require human approval or explicit bounded autonomous opt-in.
"""

import hashlib
import json
import os
import re

from .browser import BrowserSafetyStop
from .browser_numeric import NumericJournal, NumericSession
from .browser_probe import BrowserProbeConfig
from .browser_stellar import StellarMappingError
from .contracts import Action, StepResult, validate_action
from .environments.local_stellar import LocalStellarEnv
from .environments.temperature import TEMPLATES
from .knowledge import load_knowledge_pack

FIELDS = ("distance", "luminosity", "temperature")
UNITS = dict(zip(FIELDS, ("ly", "Lsun", "K")))


def load_browser_policy(options):
    """Reuse the promoted checkpoint's identity, not a fake browser training identity."""
    import torch

    from .data import load_graph
    from .training.checkpoints import load_checkpoint
    from .training.luminosity_session import load_chain_session

    if (
        options.environment != "browser"
        or options.policy != "checkpoint"
        or options.backend != "local"
        or (not options.paused and options.browser_execution != "autonomous")
        or options.stars != 1
        or options.spreadsheet_config
        or options.knowledge_pack  # NumericSession uses the reviewed default pack.
        or not all((options.graph, options.checkpoint, options.dataset, options.browser_config))
    ):
        raise ValueError("Browser numeric requires the paused local checkpoint profile")
    pack = load_knowledge_pack()
    # Validate promotion artifacts, but never use the returned manual case as an observation.
    identity, _ = load_chain_session(
        options.dataset, options.checkpoint, pack, options.seed, task="temperature"
    )
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    graph = load_graph(options.graph)
    policy, manifest = load_checkpoint(options.checkpoint, graph, content_pack=identity)
    if len(graph.body_ids) != 2000 or manifest["model"]["observation_encoding"] != "structured_tool_v4":
        raise ValueError("Browser numeric requires the promoted 2000-node three-field v4 policy")
    policy.calibration = {"status": "uncalibrated", "scope": "browser_transfer_not_calibrated"}
    provenance = {
        "checkpoint_sha256": hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
        "graph_hash": manifest["graph_hash"],
        "knowledge_pack_hash": pack.checksum,
        "trained_scope": identity["scope"],
        "evaluation_scope": f"{options.browser_execution}_browser_numeric_transfer",
        "browser_execution": options.browser_execution,
        "optimizer_updates": 0,
        "calibration_scope": "browser_transfer_not_calibrated",
    }
    return policy, provenance


def browser_config(options):
    try:
        config = BrowserProbeConfig.model_validate_json(options.browser_config.read_text())
        url = os.environ.get("HABFLY_PREVIEW_URL")
        if url:
            candidate = BrowserProbeConfig.model_validate({**config.model_dump(), "url": url})
            # Override the session query only, never broaden the configured origin/path.
            from .browser_probe import public_url

            if public_url(candidate.url) != public_url(config.url):
                raise ValueError("Activity mismatch")
            config = candidate
        return config
    except ValueError:
        # Pydantic's input values can contain the private session query.
        raise ValueError("Preview URL/config must match the configured activity origin and path") from None


class BrowserToolEnv(LocalStellarEnv):
    """Shared tool state, without synthetic private answers or simulated browser writes."""

    def __init__(self, session, seed):
        self.session = session
        self.native_results = {}
        self.copy_approved = False
        self.copy_authorization = "human_confirmation"
        self.transport_verified = False
        values = session.mapping["observation"]["values"]
        case = {
            "seed": seed,
            "split": "test",
            "star_class": None,
            "required": list(FIELDS),
            "measurements": values["measurements"],
        }
        super().__init__(session.calculator, [case], max_steps=64)
        self.reset(seed=seed)

    def observe(self):
        observation = super().observe()
        observation.instruction = TEMPLATES["manual"][0]
        observation.values["browser_field_map"] = self.session.mapping["observation"]["values"][
            "browser_field_map"
        ]
        observation.values["numeric_readbacks"] = dict(self.session.journal.numeric_readbacks)
        observation.progress.update(
            task="browser_numeric",
            task_completed=False,
            browser_acceptance_passed=False,
            numeric_transport_passed=self.transport_verified,
            completed_fields=len(self.session.verified),
        )
        return observation

    def calculate(self, bound):
        # Translate only identities; retain all selected inputs, including wrong bindings.
        selections = {key: self.native_results.get(ref, ref) for key, ref in self.bindings.items()}
        native_id, result = self.session.calculate(self.operation, selections)
        self.native_results[f"r{len(self.results) + 1}"] = native_id
        return result

    def copy_result(self, result):
        if not self.copy_approved:
            raise BrowserSafetyStop("human_copy_approval_required")
        self.copy_approved = False  # One-use approval, consumed even if the transport fails.
        self.session.copy(
            self.native_results[self.result],
            self.destination,
            confirm=lambda _: True,
            authorization=self.copy_authorization,
        )
        super().copy_result(result)

    def grades(self):
        # No course grading data, expected answers, or success rewards in this adapter.
        return set(), {}

    def step(self, action):
        self.session._current()
        control = validate_action(self.observe(), action)
        key = control.id.split(":", 1)[1] if control else ""
        _, _, _, _, info = super().step(action)
        result = StepResult.model_validate(info["result"])
        if key.startswith("unit_") and action.value != UNITS.get(key[5:]):
            result.failure_reason = "selected_unit_does_not_match_visible_field"
        if key == "check":
            self.transport_verified = set(self.session.verified) == set(FIELDS) and self.units == UNITS
            result.terminated = True
            result.failure_reason = None if self.transport_verified else "numeric_transport_incomplete"
            self.feedback = (
                "Three numeric copies verified. Star not saved, graded, or submitted."
                if self.transport_verified
                else "Numeric transport incomplete; no course assessment executed."
            )
        if self.tool_error or result.failure_reason:
            result.terminated = True
            result.failure_reason = result.failure_reason or self.tool_error
        self.terminated = result.terminated
        result.observation = self.observe()
        result.reward = result.cumulative_reward = 0
        result.reward_components = {}
        return result


class BrowserPolicyBridge:
    """Owns one fresh Chromium context and an explicit ready/propose/approve lifecycle."""

    def __init__(self, options, output, provenance, *, page=None, config=None):
        self.options = options
        self.config = config or browser_config(options)
        self.journal = NumericJournal(
            output,
            learned_policy=True,
            provenance={**provenance, "browser_execution": options.browser_execution},
        )
        self.phase = "awaiting_ready"
        self.pending = None
        self.tool = self.session = None
        self.driver = self.browser = self.context = None
        self.page = page
        self.outcome = "operator_aborted"
        self.summary = None
        self.last_tool_state = None
        self.unchanged_steps = 0
        self.setup = None
        self.browser_held = False
        credentials = None
        try:
            from .browser_setup import BrowserSetup, SetupStop, consume_credentials

            if options.browser_setup == "automatic":
                # Consume before starting the Playwright driver/browser subprocesses.
                credentials = consume_credentials()
                for key in ("DEBUG", "PWDEBUG", "DEBUG_FILE"):
                    os.environ.pop(key, None)
            if page is None:
                from playwright.sync_api import sync_playwright

                self.driver = sync_playwright().start()
                self.browser = self.driver.chromium.launch(headless=False)
                self.context = self.browser.new_context()
                self.page = self.context.new_page()
            if options.browser_setup == "automatic":
                self.setup = BrowserSetup(
                    self.page,
                    self.config,
                    credentials,
                    emit=self.journal.emit,
                    output=self.journal.output,
                )
                self.phase = "setting_up"
                self.journal.provenance["setup_mode"] = "deterministic_login_intro_visible_star"
            if page is None:
                self.page.goto(self.config.url, wait_until="domcontentloaded", timeout=30000)
        except KeyboardInterrupt:
            self.outcome = "operator_aborted"
            self.close()
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize driver failures before the protocol boundary
            self.outcome = str(exc) if isinstance(exc, SetupStop) else "browser_launch_failed"
            self.close()
            raise BrowserSafetyStop(self.outcome) from None
        finally:
            credentials = None

    def state(self):
        autonomous = self.options.browser_execution == "autonomous"
        return {
            "browser_execution": self.options.browser_execution,
            "browser_phase": self.phase,
            "browser_status": self.phase,
            "browser_guidance": {
                "awaiting_ready": "Choose one star manually in Chromium, open Stellar, close help. b = ready.",
                "setting_up": f"Scripted setup: {self.setup.stage if self.setup else 'starting'}. "
                + (
                    "Autonomous three-field run follows."
                    if autonomous
                    else "a aborts; policy remains paused."
                ),
                "ready": "AUTONOMOUS: three fields only; copies are enabled while running."
                if autonomous
                else "n = one learned tool action; no automatic browser writes.",
                "awaiting_copy": "AUTONOMOUS: pending exact copy + Tab on next running tick. Pause or abort to stop it."
                if autonomous
                else "Review the pending copy. y = approve this number + Tab; a = abort.",
                "finished": (
                    "PASS: three numeric copies verified. Star not saved, assessed or submitted."
                    if self.outcome == "numeric_transport_verified"
                    else (
                        f"STOP: {self.outcome}. Chromium kept open for inspection; no more actions. q closes it."
                        if self.browser_held
                        else f"STOP: {self.outcome}. No retry or rollback; inspect the recorded evidence."
                    )
                ),
            }[self.phase],
            "pending_browser_copy": self.pending_copy(),
            "calibration_scope": "browser_transfer_not_calibrated",
            "allow_submission": False,
            "setup_stage": self.setup.stage if self.setup else None,
            "browser_held_open": self.browser_held,
        }

    def advance_setup(self):
        from .browser_setup import SetupStop

        try:
            if self.setup.advance() == "stellar":
                self.phase = "awaiting_ready"
                return self.ready()  # Capture only; never execute a model action automatically.
        except SetupStop:
            raise
        except (BrowserSafetyStop, StellarMappingError) as exc:
            # Preserve known local reason codes, never driver text or session URLs.
            code = str(exc)
            fixed = re.fullmatch(r"[a-z][a-z0-9_]{0,100}", code)
            frame_code = next(
                (
                    prefix
                    for prefix in ("frame_count_mismatch:", "frame_not_ready:")
                    if code.startswith(prefix)
                ),
                None,
            )
            if fixed or (
                frame_code and code[len(frame_code) :] in {rule.name for rule in self.config.frames}
            ):
                raise SetupStop(f"setup_capture_{code}") from None
            raise SetupStop("setup_stellar_capture_failed") from None
        except Exception:  # noqa: BLE001 - sanitize both setup and strict capture failures
            raise SetupStop("setup_stellar_capture_failed") from None
        return None

    def ready(self):
        if self.phase != "awaiting_ready":
            raise ValueError("Browser already captured; n steps or y approves a pending copy")
        self.session = NumericSession(self.page, self.config, self.journal)
        self.session.start()
        self.tool = BrowserToolEnv(self.session, self.options.seed)
        self.phase = "ready"
        return self.tool.observe()

    def propose(self, action):
        if self.phase != "ready":
            raise ValueError("Use b to capture or y to approve the pending browser copy")
        self.session._current()
        control = validate_action(self.tool.observe(), action)
        if control and control.id.split(":", 1)[1] == "copy":
            self.pending = action.model_copy(deep=True)
            self.phase = "awaiting_copy"
            return None
        return self.step(action)

    def pending_copy(self):
        if self.pending is None:
            return None
        result = self.tool.results[self.tool.result]
        destination = self.tool.destination
        fields = self.session.mapping["observation"]["values"]["browser_field_map"]
        return {
            "result_id": self.tool.result,
            "destination": destination,
            "exact_value": repr(result["value"]),
            "unit": result["unit"],
            "current_value": fields[destination]["current_value"],
            "commit_key": "Tab",
        }

    def approve(self, *, automatic=False):
        if automatic and self.options.browser_execution != "autonomous":
            raise ValueError("Autonomous copy was not enabled for this run")
        if self.phase != "awaiting_copy" or self.pending is None:
            raise ValueError("No pending browser copy to approve")
        action, self.pending = self.pending, None
        self.phase = "ready"
        self.tool.copy_approved = True
        self.tool.copy_authorization = "autonomous_opt_in" if automatic else "human_confirmation"
        self.journal.emit("state", {"copy_authorization": self.tool.copy_authorization})
        try:
            return self.step(action)
        finally:
            self.tool.copy_approved = False
            self.tool.copy_authorization = "human_confirmation"

    def step(self, action):
        result = self.tool.step(Action.model_validate(action))
        signature = json.dumps(
            {
                "values": result.observation.values,
                "calculation": {
                    key: value
                    for key, value in result.observation.calculation.items()
                    if key not in {"last_operation", "tool_error"}
                },
            },
            sort_keys=True,
        )
        self.unchanged_steps = self.unchanged_steps + 1 if signature == self.last_tool_state else 0
        self.last_tool_state = signature
        if self.unchanged_steps >= 3:
            result.terminated = True
            result.failure_reason = "repeated_unchanged_tool_actions"
        if result.terminated or result.truncated:
            self.phase = "finished"
            self.outcome = (
                "numeric_transport_verified"
                if self.tool.transport_verified
                else result.failure_reason or "policy_stopped"
            )
        return result

    def close(self, *, keep_browser_open=False):
        if self.summary is None:
            self.phase = "finished"
            self.pending = None
            if self.setup:
                self.setup.close()
            if self.session:
                self.session.stopped = True
                self.page.remove_listener("dialog", self.session._dialog)
            self.summary = self.journal.finish(
                outcome=self.outcome,
                attempts=self.session.attempts if self.session else 0,
                verified=list(self.session.verified) if self.session else [],
                pack_hash=load_knowledge_pack().checksum,
            )
        self.browser_held = keep_browser_open and self.page is not None and not self.page.is_closed()
        if self.browser_held:
            return
        try:
            if self.context:
                self.context.close()
                self.context = None
            if self.browser:
                self.browser.close()
                self.browser = None
        finally:
            if self.driver:
                self.driver.stop()
                self.driver = None
