"""Isolated, gated native color transport. Existing numeric runs never call this.

No classifier label oracle, class selection, Save, assessment or submission.
The launch factory requires the separate final color gate. This adapter is
fixture-validated infrastructure, not a claim of live color acceptance.
"""

import copy
import hashlib
import json
import uuid
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_numeric import (
    VISIBLE_PREFIX,
    NumericSession,
    comparable_screen,
    digest,
    screen_changes,
)
from .browser_probe import inspect_page
from .browser_stellar import COLOR_OPTIONS, SIMULATION_URL, _atoms
from .color_reference import COLOR_CONTROL
from .contracts import Action, Control, Observation, RuntimeEvent, StepResult, validate_action
from .environments.color import TEMPLATES, ColorEnv


class ColorJournal:
    def __init__(self, output, provenance):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.stream = (self.output / "events.jsonl").open("x")
        self.sequence, self.run_id = 0, uuid.uuid4().hex
        self.provenance, self.receipt, self.summary = provenance, None, None
        self.screen_diagnostics = []
        self.emit(
            "hello",
            {
                "protocol_version": 1,
                "task": "browser_color",
                "provenance": provenance,
                "allow_submission": False,
            },
        )

    def emit(self, event, payload):
        item = RuntimeEvent(
            event=event,
            sequence=self.sequence,
            run_id=self.run_id,
            payload={**payload, "mode": "color_transport"},
        )
        self.stream.write(item.model_dump_json() + "\n")
        self.stream.flush()
        self.sequence += 1

    def record_screen_change(self, reason, before, after):
        evidence = {"reason": reason, "before": before, "after": after, **screen_changes(before, after)}
        name = f"screen-change-{len(self.screen_diagnostics) + 1:03d}.json"
        raw = (json.dumps(evidence, indent=2) + "\n").encode()
        (self.output / name).write_bytes(raw)
        self.screen_diagnostics.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest()})
        self.emit("state", {"safety_stop": reason, "evidence": self.screen_diagnostics[-1]})

    def finish(self, outcome, attempts):
        if self.summary is not None:
            return self.summary
        report = {
            "schema_version": 1,
            "scope": "color_transport_only",
            "outcome": outcome,
            "color_transport_verified": outcome == "color_transport_verified" and self.receipt is not None,
            "receipt": self.receipt,
            "write_attempts": attempts,
            "provenance": self.provenance,
            "task_completed": False,
            "browser_acceptance_passed": False,
            "allow_submission": False,
        }
        self.emit("episode_summary", report)
        self.stream.close()
        report["events_sha256"] = hashlib.sha256((self.output / "events.jsonl").read_bytes()).hexdigest()
        (self.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        self.summary = report
        return report


def color_projection(report):
    """Exclude only the single native selected color; compare all other evidence."""
    projected = comparable_screen(report)
    frame = next(f for f in projected["frames"] if f["url"] == SIMULATION_URL)
    controls = [c for c in frame["controls"] if c["role"] == "combobox"]
    if len(controls) != 1:
        raise BrowserSafetyStop("ambiguous_color_control")
    controls[0]["value"] = "<selected-color>"

    def normalized(snapshot):
        return [
            (
                key,
                [v.removesuffix(" [selected]") for v in value]
                if key.split(" ", 1)[0] == "combobox" and isinstance(value, list)
                else value,
            )
            for key, value in _atoms(snapshot)
        ]

    controls[0]["accessibility"] = normalized(controls[0]["accessibility"])
    frame["accessibility"] = normalized(frame["accessibility"])
    return projected


class ColorSession(NumericSession):
    """One explicit color selection. Inherited numeric mutation methods are disabled."""

    ALLOW_COLOR_SELECTION = True
    MAX_SECONDS = 120
    MAX_WRITES = 1

    def __init__(self, page, config, journal, reference):
        super().__init__(page, config, journal)
        self.reference = reference
        self.color_handle = None

    def observation(self):
        observation = Observation.model_validate(copy.deepcopy(self.mapping["observation"]))
        color = observation.values["color"]
        observation.revision = self.revision
        observation.controls = [
            Control(
                id=f"color:{self.revision}",
                label=COLOR_CONTROL,
                role="combobox",
                options=color["options"],
                value=color["selected"] or "",
                actions=["SELECT"],
                enabled=not self.attempts,
            )
        ]
        observation.calculation = {
            "reference_card": self.reference.card(),
            "pack_hash": self.reference.checksum,
        }
        observation.progress.update(
            task="browser_color", task_completed=False, browser_acceptance_passed=False
        )
        return observation

    def start(self):
        super().start()
        self.color_handle = self._resolve_color(self.frame, self.mapping)
        self._current_color()
        if self.mapping["observation"]["values"]["color"]["selected"] is not None:
            raise BrowserSafetyStop("preexisting_color_selection")
        measurement = self.mapping["observation"]["values"]["measurements"]["browser_wavelength"]
        self.reference.guard(measurement["value"], measurement["unit"])
        if not 100 <= measurement["value"] <= 1800:
            raise BrowserSafetyStop("color_outside_training_domain")

    def _resolve_color(self, frame, mapping):
        choices = [item for item in frame.get_by_role("combobox").all() if item.is_visible()]
        if len(choices) != 1 or not choices[0].is_enabled():
            raise BrowserSafetyStop("ambiguous_or_disabled_color_control")
        handle = choices[0].element_handle(timeout=2000)
        if not handle or not handle.evaluate("e => e.tagName === 'SELECT' && e.isConnected && !e.multiple"):
            raise BrowserSafetyStop("unsupported_native_color_control")
        prefix = handle.evaluate(VISIBLE_PREFIX)
        if prefix is None or "".join(prefix.casefold().split()) not in {"peakλcolor", "peakwavelengthcolor"}:
            raise BrowserSafetyStop("unverified_visible_color_label")
        visible = handle.evaluate("""e => Array.from(e.options)
            .filter(o => !o.hidden && getComputedStyle(o).display !== 'none')
            .map(o => ({label: o.label, selected: o.selected, disabled: o.disabled}))""")
        if [o["label"] for o in visible] != COLOR_OPTIONS or any(o["disabled"] for o in visible):
            raise BrowserSafetyStop("color_option_inventory_changed")
        selected = [o["label"] for o in visible if o["selected"]]
        if (
            len(selected) > 1
            or (selected[0] if selected else None) != mapping["observation"]["values"]["color"]["selected"]
        ):
            raise BrowserSafetyStop("color_selection_readback_mismatch")
        return handle

    def _current_color(self):
        report, mapping, _handles, frame = self._current()
        color = self._resolve_color(frame, mapping)
        if self.color_handle is not None and not self.color_handle.evaluate(
            "(old, current) => old.isConnected && old === current", color
        ):
            raise BrowserSafetyStop("color_control_replaced")
        self._require_same_screen(
            report, inspect_page(self.page, self.config), "screen_changed_during_color_binding"
        )
        return color

    def select_color(self, label, *, confirm):
        if (
            self.journal.provenance.get("color_gate_passed") is not True
            or self.journal.provenance.get("color_reference_hash") != self.reference.checksum
        ):
            raise BrowserSafetyStop("color_learning_gate_required")
        if self.attempts >= 1:
            raise BrowserSafetyStop("color_write_limit")
        if label not in COLOR_OPTIONS:
            raise BrowserSafetyStop("unknown_color_option")
        handle = self._current_color()
        action = Action(
            kind="SELECT", target=f"color:{self.revision}", value=label, observation_revision=self.revision
        )
        self.journal.emit(
            "action_proposed", {**action.model_dump(mode="json"), "action_source": "checkpoint"}
        )
        if not confirm(label):
            return False
        # Never reuse a handle or page identity across the approval pause unchecked.
        handle = self._current_color()
        self.attempts += 1
        self.journal.emit("state", {"write_attempt": self.attempts, "destination": "color"})
        try:
            handle.select_option(label=label, timeout=3000)
            for _ in range(2):
                report, mapping, handles, frame = self._read()
                current = self._resolve_color(frame, mapping)
                if not handle.evaluate("(old, current) => old.isConnected && old === current", current):
                    raise BrowserSafetyStop("color_control_replaced")
                if mapping["observation"]["values"]["color"]["selected"] != label:
                    raise BrowserSafetyStop("color_selection_readback_mismatch")
                if digest(color_projection(self.report)) != digest(color_projection(report)):
                    self.journal.record_screen_change("unrelated_state_changed_by_color", self.report, report)
                    raise BrowserSafetyStop("unrelated_state_changed_by_color")
                self._guard()
            self.report, self.mapping, self.handles, self.frame = report, mapping, handles, frame
            self.revision += 1
            self.journal.receipt = {
                "selected_color": label,
                "readback_verified": True,
                "reference_hash": self.reference.checksum,
                "correctness_verified": False,
            }
            self.journal.emit(
                "action_result",
                {
                    "action": action.model_dump(mode="json"),
                    "color_transport_verified": True,
                    "receipt": self.journal.receipt,
                    "observation": self.observation().model_dump(mode="json"),
                },
            )
            return True
        except Exception:
            # A change may already have happened. No retry or automatic rollback.
            self.stopped = True
            raise

    def calculate(self, *args, **kwargs):
        raise BrowserSafetyStop("numeric_actions_disabled_in_color_session")

    def copy(self, *args, **kwargs):
        raise BrowserSafetyStop("numeric_actions_disabled_in_color_session")

    def close(self, outcome="operator_aborted"):
        self.stopped = True
        self.page.remove_listener("dialog", self._dialog)
        return self.journal.finish(outcome, self.attempts)


class BrowserColorEnv(ColorEnv):
    """The learner's same source/color/check interface, with no private grading case."""

    def __init__(self, session, seed=0):
        self.session = session
        values = session.mapping["observation"]["values"]
        case = {
            "seed": seed,
            "instruction": TEMPLATES["manual"][0],
            "measurements": values["measurements"],
            "reference_hash": session.reference.checksum,
            "expected_source": None,
            "expected_color": None,
        }
        super().__init__(session.reference, [case])
        self.reset(seed=seed)

    def observe(self):
        observation = super().observe()
        observation.progress.update(
            task="browser_color",
            task_completed=False,
            color_transport_verified=bool(self.session.journal.receipt),
            browser_acceptance_passed=False,
        )
        observation.values["color_readback"] = self.session.journal.receipt
        return observation

    def completed_selection(self):
        return bool(self.session.journal.receipt)

    def step(self, action, *, confirm):
        self.session._current_color()
        action = Action.model_validate(action)
        control = validate_action(self.observe(), action)
        key = control.id.split(":", 1)[1] if control else ""
        if key == "source" and self.session.attempts:
            raise BrowserSafetyStop("color_input_changed_after_commit")
        if key == "color":
            row = self.case["measurements"][self.source]
            if row["kind"] != "wavelength":
                raise BrowserSafetyStop("incompatible_measurement_kind")
            self.reference.guard(row["value"], row["unit"])
            if not self.session.select_color(action.value, confirm=confirm):
                raise BrowserSafetyStop("color_approval_declined")
        _, _, _, _, info = super().step(action)
        result = StepResult.model_validate(info["result"])
        result.reward = result.cumulative_reward = 0
        result.reward_components = {}
        result.observation = self.observe()
        return result

    def close(self):
        outcome = (
            "color_transport_verified"
            if self.completed and self.session.journal.receipt
            else "operator_aborted"
        )
        return self.session.close(outcome)


def open_gated_color_session(page, config, output, experiment, *, graph_path=None):
    """Requires final learned-color evidence before creating any writable session.

    The caller owns a fresh ready page and must still provide per-selection
    confirmation. Automatic chaining into the numeric batch remains disabled.
    """
    from .training.color import GRAPH, file_hash, require_browser_color_gate

    policy, reference = require_browser_color_gate(experiment, graph_path or GRAPH)
    journal = ColorJournal(
        output,
        {
            "color_gate_passed": True,
            "color_checkpoint_sha256": file_hash(Path(experiment) / "training/checkpoint.pt"),
            "color_reference_hash": reference.checksum,
        },
    )
    session = ColorSession(page, config, journal, reference)
    try:
        session.start()
    except Exception:
        session.close("color_setup_stopped")
        raise
    return policy, BrowserColorEnv(session)
