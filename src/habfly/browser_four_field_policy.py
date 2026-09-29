"""Scripted coordination of two frozen policies; no Save, grading or submission.

Numeric and color transports retain their own limits, journals and guards.
The only transition is a read-only, identity-checked handoff on the same page.
"""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_color import ColorSession
from .browser_color_policy import BrowserColorPolicyBridge, load_browser_color_policy
from .browser_numeric import screen_identity
from .browser_policy import BrowserPolicyBridge, load_browser_policy
from .contracts import RuntimeEvent


def component_options(options, stage):
    updates = {"task": "browser_numeric"}
    if stage == "color":
        updates.update(
            task="browser_color", dataset=options.color_dataset, checkpoint=options.color_checkpoint, seed=0
        )
    return options.model_copy(update=updates)


def validate_options(options):
    if (
        options.task != "browser_four_field"
        or options.environment != "browser"
        or options.policy != "checkpoint"
        or (options.browser_execution == "supervised" and not options.paused)
        or (options.browser_execution == "autonomous" and options.browser_setup != "automatic")
        or options.stars != 1
        or options.backend != "local"
        or options.knowledge_pack
        or options.content_pack
        or options.spreadsheet_config
        or not all(
            (
                options.graph,
                options.checkpoint,
                options.dataset,
                options.color_checkpoint,
                options.color_dataset,
                options.browser_config,
            )
        )
    ):
        raise ValueError(
            "Four-field transfer requires both frozen policies, one star, and paused supervision or automatic autonomous setup"
        )


class FourFieldPolicy:
    """Routing is scripted. Every measurement, calculation, copy and color choice is learned."""

    def __init__(self, numeric, color):
        self.models = {"numeric": numeric, "color": color}
        self.active_stage = "numeric"
        self.calibration = {"status": "uncalibrated", "scope": "browser_transfer_not_calibrated"}

    def select_stage(self, stage):
        if stage not in self.models or (self.active_stage == "color" and stage != "color"):
            raise BrowserSafetyStop("invalid_policy_handoff")
        changed = stage != self.active_stage
        self.active_stage = stage
        return changed

    def act(self, observation, state=None):
        return self.models[self.active_stage].act(observation, state)

    def neural_activity(self, state):
        return self.models[self.active_stage].neural_activity(state)


def load_four_field_policy(options):
    validate_options(options)
    numeric, numeric_provenance = load_browser_policy(component_options(options, "numeric"))
    color, color_provenance = load_browser_color_policy(component_options(options, "color"))
    if numeric_provenance["graph_hash"] != color_provenance["graph_hash"]:
        raise ValueError("Four-field policies must use the same validated graph")
    return FourFieldPolicy(numeric, color), {
        "evaluation_scope": f"{options.browser_execution}_browser_four_field_transfer",
        "coordinator": "scripted_numeric_then_color_v1",
        "graph_hash": numeric_provenance["graph_hash"],
        "numeric_policy": numeric_provenance,
        "color_policy": color_provenance,
        "numeric_seed": options.seed,
        "color_seed": 0,
        "browser_execution": options.browser_execution,
        "optimizer_updates": 0,
        "calibration_scope": "browser_transfer_not_calibrated",
    }


class FourFieldJournal:
    def __init__(self, output, provenance):
        self.output, self.provenance = Path(output), provenance
        self.output.mkdir(parents=True, exist_ok=False)
        self.stream = (self.output / "events.jsonl").open("x")
        self.sequence, self.run_id = 0, uuid.uuid4().hex
        self.emit(
            "hello",
            {
                "protocol_version": 1,
                "task": "browser_four_field",
                "provenance": provenance,
                "allow_submission": False,
            },
        )

    def emit(self, event, payload):
        item = RuntimeEvent(
            event=event,
            sequence=self.sequence,
            run_id=self.run_id,
            payload={**payload, "mode": "four_field_transport"},
        )
        self.stream.write(item.model_dump_json() + "\n")
        self.stream.flush()
        self.sequence += 1

    def finish(self, outcome, numeric, color, handoff):
        numeric_passed = bool(numeric and numeric.get("numeric_transport_passed"))
        color_passed = bool(color and color.get("color_transport_verified"))
        attempts = sum(item.get("write_attempts", 0) for item in (numeric or {}, color or {}))
        passed = (
            outcome == "four_field_transport_verified"
            and numeric_passed
            and color_passed
            and handoff is not None
            and attempts == 4
        )
        components = {}
        for stage, result in (("numeric", numeric), ("color", color)):
            if result:
                path = self.output / stage / "manifest.json"
                components[stage] = {
                    "manifest": str(path.relative_to(self.output)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
        report = {
            "schema_version": 1,
            "scope": "four_field_transport_only",
            "outcome": outcome,
            "four_field_transport_verified": passed,
            "numeric_transport_passed": numeric_passed,
            "color_transport_verified": color_passed,
            "write_attempts": attempts,
            "numeric_readbacks": (numeric or {}).get("numeric_readbacks", {}),
            "color_receipt": (color or {}).get("receipt"),
            "handoff": handoff,
            "components": components,
            "provenance": self.provenance,
            "task_completed": False,
            "browser_acceptance_passed": False,
            "allow_submission": False,
        }
        self.emit("episode_summary", report)
        self.stream.close()
        report["events_sha256"] = hashlib.sha256((self.output / "events.jsonl").read_bytes()).hexdigest()
        (self.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


class BrowserFourFieldBridge(BrowserPolicyBridge):
    MAX_SECONDS = 1200

    def __init__(self, options, output, provenance, *, page=None, config=None):
        validate_options(options)
        self.active_stage = "numeric"
        self.children = {}
        self.child = None
        self.numeric_steps = 0
        self.handoff_receipt = None
        self.initial_color_handle = None
        self.started = time.monotonic()
        for key in ("DEBUG", "PWDEBUG", "DEBUG_FILE"):
            os.environ.pop(key, None)
        if options.browser_setup == "manual":
            for key in ("HABFLY_LOGIN_EMAIL", "HABFLY_LOGIN_PASSWORD"):
                os.environ.pop(key, None)
        super().__init__(options, output, provenance, page=page, config=config)

    def _make_journal(self, output, provenance):
        return FourFieldJournal(output, provenance)

    def _stop_session(self):
        # Sessions/listeners belong to the component bridges, not this owner of
        # their shared page. Each child releases its listener exactly once below.
        pass

    def _finish_journal(self):
        summaries = {}
        for stage, child in self.children.items():
            if child.summary is None and child.phase != "finished":
                child.outcome = self.outcome
            child.close()
            summaries[stage] = child.summary
        return self.journal.finish(
            self.outcome, summaries.get("numeric"), summaries.get("color"), self.handoff_receipt
        )

    def _guard(self):
        if time.monotonic() - self.started > self.MAX_SECONDS:
            raise BrowserSafetyStop("four_field_time_limit")

    def _create_child(self, stage):
        settings = component_options(self.options, stage)
        bridge_type = BrowserPolicyBridge if stage == "numeric" else BrowserColorPolicyBridge
        child = bridge_type(
            settings,
            self.journal.output / stage,
            self.journal.provenance[f"{stage}_policy"],
            page=self.page,
            config=self.config,
            setup_complete=True,  # Parent owns login/star selection and this same page.
        )
        self.children[stage] = child
        return child

    def ready(self):
        self._guard()
        if self.phase != "awaiting_ready":
            raise ValueError("Browser already captured")
        self.child = self._create_child("numeric")
        observation = self.child.ready()
        self.session, self.tool = self.child.session, self.child.tool
        self.initial_color_handle = ColorSession._resolve_color(self.session.frame, self.session.mapping)
        values = self.session.mapping["observation"]["values"]
        if any(field["current_value"] not in ("", None) for field in values["browser_field_map"].values()):
            raise BrowserSafetyStop("four_field_requires_blank_numeric_fields")
        self.phase = "ready"
        return observation

    def state(self):
        state = self.child.state() if self.child else super().state()
        guidance = state["browser_guidance"]
        if self.phase == "setting_up":
            guidance = f"Scripted setup: {self.setup.stage if self.setup else 'starting'}. " + (
                "Three numeric copies and one color selection follow automatically; a aborts."
                if self.options.browser_execution == "autonomous"
                else "Four-field policy remains paused; a aborts."
            )
        elif self.phase == "awaiting_handoff":
            guidance = (
                "Three numeric readbacks verified. Next running tick rechecks this star and switches policy; pause or abort to stop."
                if self.options.browser_execution == "autonomous"
                else "Three numeric readbacks verified. n rechecks this star and switches to the color policy; a aborts."
            )
        elif self.phase == "finished":
            guidance = (
                "Four readbacks verified. Star not saved, graded or submitted."
                if self.outcome == "four_field_transport_verified"
                else f"STOP: {self.outcome}. No retry or rollback."
            ) + (" Chromium held open; q closes it." if self.browser_held else "")
        return {
            **state,
            "browser_task": "four_field",
            "browser_phase": self.phase,
            "browser_status": self.phase,
            "browser_guidance": guidance,
            "browser_held_open": self.browser_held,
            "workflow": "numeric_then_color",
            "policy_stage": self.active_stage,
            "seed": self.options.seed if self.active_stage == "numeric" else 0,
            "calculation_backend": "local",
            "calculation_mode": "local_tool_assisted"
            if self.active_stage == "numeric"
            else "learned_peak_wavelength_color_v1",
            "stage": f"{self.options.browser_execution}_browser_four_field:{self.active_stage}",
            "checkpoint": str(
                self.options.color_checkpoint if self.active_stage == "color" else self.options.checkpoint
            ),
            "numeric_checkpoint": str(self.options.checkpoint),
            "color_checkpoint": str(self.options.color_checkpoint),
            "pending_browser_copy": self.child.pending_copy()
            if self.child and self.active_stage == "numeric"
            else None,
            "pending_browser_color": self.child.pending_color()
            if self.child and self.active_stage == "color"
            else None,
        }

    def pending_copy(self):
        return None  # Used only by base state before any component exists.

    def _accept_result(self, result):
        self.phase = self.child.phase
        if result is None:
            return None
        if self.active_stage == "color":
            result.steps += self.numeric_steps
        if result.terminated or result.truncated:
            if self.active_stage == "numeric" and self.child.outcome == "numeric_transport_verified":
                self.numeric_steps = result.steps
                self.phase = "awaiting_handoff"
                result.terminated = result.truncated = False
                self.journal.emit(
                    "state", {"workflow_stage": "awaiting_handoff", "numeric_steps": self.numeric_steps}
                )
            else:
                self.outcome = (
                    "four_field_transport_verified"
                    if self.active_stage == "color" and self.child.outcome == "color_transport_verified"
                    else self.child.outcome
                )
        return result

    def propose(self, action):
        self._guard()
        if self.phase != "ready":
            raise ValueError("No learned decision available in this workflow phase")
        return self._accept_result(self.child.propose(action))

    def approve(self, *, automatic=False):
        self._guard()
        if automatic and self.options.browser_execution != "autonomous":
            raise ValueError("Four-field transfer requires human approval of each write")
        return self._accept_result(self.child.approve(automatic=automatic))

    def handoff(self):
        self._guard()
        if self.phase != "awaiting_handoff" or self.active_stage != "numeric":
            raise ValueError("No numeric-to-color handoff is pending")
        numeric = self.children["numeric"]
        numeric.session._current()  # Revalidate across the human pause before opening another adapter.
        color = self._create_child("color")
        observation = color.ready()
        numeric.session._require_same_screen(
            numeric.session.report, color.session.report, "handoff_screen_changed"
        )
        if numeric.session.frame != color.session.frame:
            raise BrowserSafetyStop("handoff_frame_changed")
        if not self.initial_color_handle.evaluate(
            "(old, current) => old.isConnected && old === current", color.session.color_handle
        ):
            raise BrowserSafetyStop("handoff_color_control_replaced")
        for name, handle in numeric.session.handles.items():
            if not handle.evaluate(
                "(old, current) => old.isConnected && old === current", color.session.handles[name]
            ):
                raise BrowserSafetyStop("handoff_control_replaced")
        numeric.session._current()  # Reject a race during the new adapter's binding pass.
        self.handoff_receipt = {
            "star_name": numeric.session.mapping["star_name"],
            "same_star_and_native_controls": True,
            "numeric_screen_sha256": screen_identity(numeric.session.report),
            "color_screen_sha256": screen_identity(color.session.report),
            "numeric_readbacks_preserved": True,
            "recurrent_state_reset": True,
        }
        numeric.close()  # This child does not own the shared browser/context.
        self.child, self.active_stage, self.phase = color, "color", "ready"
        self.session, self.tool = color.session, color.tool
        self.journal.emit("state", {"workflow_stage": "color", "handoff": self.handoff_receipt})
        return observation
