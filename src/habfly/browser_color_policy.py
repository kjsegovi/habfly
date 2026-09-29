"""Single-write transfer of the separately gated learned color policy."""

import hashlib
import os

from .browser import BrowserSafetyStop
from .browser_color import BrowserColorEnv, ColorJournal, ColorSession
from .browser_numeric import digest
from .browser_policy import BrowserPolicyBridge
from .color_reference import load_color_reference
from .contracts import Action, validate_action


def validate_options(options):
    if (
        options.task != "browser_color"
        or options.environment != "browser"
        or options.policy != "checkpoint"
        or options.backend != "local"
        or (options.browser_execution == "supervised" and not options.paused)
        or (options.browser_execution == "autonomous" and options.browser_setup != "automatic")
        or options.stars != 1
        or options.spreadsheet_config
        or options.knowledge_pack
        or options.content_pack
        or not all((options.graph, options.checkpoint, options.dataset, options.browser_config))
    ):
        raise ValueError(
            "Browser color requires one local checkpoint star, paused supervision or automatic autonomous setup"
        )
    if options.checkpoint.resolve() != (options.dataset / "training/checkpoint.pt").resolve():
        raise ValueError("Browser color checkpoint must belong to its gated experiment")


def load_browser_color_policy(options):
    from .training.color import require_browser_color_gate

    validate_options(options)
    policy, reference = require_browser_color_gate(options.dataset, options.graph)
    policy.calibration = {"status": "uncalibrated", "scope": "browser_transfer_not_calibrated"}
    return policy, {
        "checkpoint_sha256": hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
        "graph_hash": policy.graph_hash,
        "color_reference_hash": reference.checksum,
        "color_gate_passed": True,
        "final_report_sha256": hashlib.sha256(
            (options.dataset / "final/report.json").read_bytes()
        ).hexdigest(),
        "evaluation_scope": f"{options.browser_execution}_browser_color_transfer",
        "browser_execution": options.browser_execution,
        "optimizer_updates": 0,
        "calibration_scope": "browser_transfer_not_calibrated",
    }


class BrowserColorPolicyBridge(BrowserPolicyBridge):
    """Reuse scripted setup and Chromium ownership, never numeric write capability."""

    def __init__(self, options, output, provenance, *, page=None, config=None, setup_complete=False):
        validate_options(options)
        self.reference = load_color_reference()
        if (
            provenance.get("color_gate_passed") is not True
            or provenance.get("color_reference_hash") != self.reference.checksum
        ):
            raise BrowserSafetyStop("color_learning_gate_required")
        # Manual setup must not forward unused secrets or browser debug logging
        # into the Playwright driver. Automatic setup consumes credentials below.
        for key in ("DEBUG", "PWDEBUG", "DEBUG_FILE"):
            os.environ.pop(key, None)
        if options.browser_setup == "manual":
            for key in ("HABFLY_LOGIN_EMAIL", "HABFLY_LOGIN_PASSWORD"):
                os.environ.pop(key, None)
        self.pending_observation = None
        super().__init__(options, output, provenance, page=page, config=config, setup_complete=setup_complete)

    def _make_journal(self, output, provenance):
        return ColorJournal(output, provenance)

    def _finish_journal(self):
        return self.journal.finish(self.outcome, self.session.attempts if self.session else 0)

    def state(self):
        autonomous = self.options.browser_execution == "autonomous"
        guidance = {
            "setting_up": "Scripted login/introduction/star setup; one autonomous color selection follows. a aborts."
            if autonomous
            else "Scripted login/introduction/star setup. Learned color policy stays paused; a aborts.",
            "awaiting_ready": "Open one fresh star's Stellar tab, close help. b captures; no color selected yet.",
            "ready": "AUTONOMOUS: one color selection only while running; pause or abort to stop."
            if autonomous
            else "n = one learned color decision. Native color selection always needs y approval.",
            "awaiting_color": "AUTONOMOUS: next running tick selects the proposed color once; pause or abort to stop."
            if autonomous
            else "Review the measurement and proposed color. y selects this color once; a aborts.",
            "finished": (
                "Color readback verified; correctness/course acceptance not verified. No Save/score/submit."
                if self.outcome == "color_transport_verified"
                else f"STOP: {self.outcome}. No retries or rollback."
            )
            + (" Chromium held open; q closes it." if self.browser_held else ""),
        }
        return {
            "browser_task": "color",
            "browser_execution": self.options.browser_execution,
            "browser_phase": self.phase,
            "browser_status": self.phase,
            "browser_guidance": guidance[self.phase],
            "pending_browser_color": self.pending_color(),
            "pending_browser_copy": None,
            "calibration_scope": "browser_transfer_not_calibrated",
            "allow_submission": False,
            "setup_stage": self.setup.stage if self.setup else None,
            "browser_held_open": self.browser_held,
        }

    def ready(self):
        if self.phase != "awaiting_ready":
            raise ValueError("Browser already captured")
        self.session = ColorSession(self.page, self.config, self.journal, self.reference)
        self.session.start()
        self.tool = BrowserColorEnv(self.session, self.options.seed)
        self.phase = "ready"
        return self.tool.observe()

    def propose(self, action):
        if self.phase != "ready":
            raise ValueError("Capture the ready browser or approve the pending color first")
        self.session._current_color()
        observation = self.tool.observe()
        control = validate_action(observation, action)
        if control and control.id.split(":", 1)[1] == "color":
            self.pending = action.model_copy(deep=True)
            self.pending_observation = digest(observation.model_dump(mode="json"))
            self.phase = "awaiting_color"
            return None
        return self.step(action)

    def pending_color(self):
        if self.pending is None:
            return None
        return {
            "source_id": self.tool.source,
            "measurement": dict(self.tool.case["measurements"][self.tool.source]),
            "selected_color": self.pending.value,
            "destination": "Peak wavelength color",
            "current_value": self.session.mapping["observation"]["values"]["color"]["selected"],
        }

    def approve(self, *, automatic=False):
        if automatic and self.options.browser_execution != "autonomous":
            raise ValueError("Color selection requires explicit human approval")
        if self.phase != "awaiting_color" or self.pending is None:
            raise ValueError("No pending color selection")
        action, self.pending = self.pending, None  # Consume before any validation/write.
        self.phase = "ready"
        if digest(self.tool.observe().model_dump(mode="json")) != self.pending_observation:
            raise BrowserSafetyStop("color_proposal_changed")
        self.pending_observation = None
        authorization = "autonomous_opt_in" if automatic else "human_confirmation"
        self.journal.emit("state", {"color_authorization": authorization})
        return self.step(action, approved=True, authorization=authorization)

    def step(self, action, *, approved=False, authorization="human_confirmation"):
        result = self.tool.step(
            Action.model_validate(action), confirm=lambda _: approved, authorization=authorization
        )
        verified = self.tool.completed and self.session.journal.receipt is not None
        return self._track_step(result, verified, "color_transport_verified")
