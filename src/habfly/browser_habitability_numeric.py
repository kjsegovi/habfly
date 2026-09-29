"""One at-most-once equilibrium-temperature copy, not a habitability policy."""

import re
import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import read_habitability_choices
from .browser_habitability import map_habitability_capture
from .browser_numeric import VISIBLE_PREFIX, committed_display, comparable_screen, screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_setup import rendered_control
from .browser_stellar import NUMBER, SIMULATION_URL, _atoms, _control_atom


def equilibrium_projection(report, mapping):
    """Preserve visible context, except the explicit copy and dependent readout.

    The separate readback gate verifies both temperature numbers. Absorption
    may format 0 as 0.000, but must remain numerically identical. All other
    values, choices, units, text, chart semantics and outside frames stay fixed.
    """
    result = comparable_screen(report)
    frame = next(f for f in result["frames"] if f["url"] == SIMULATION_URL)

    def normalized(text):
        text = " ".join(text.split())
        text = re.sub(
            rf"(Absorption % )({NUMBER})%",
            lambda m: m[1] + str(Decimal(m[2]).normalize()) + "%",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"(Surface Temp \(K\))(?: ({NUMBER}))?(?= Water Phase)",
            r"\1 <dependent>",
            text,
            flags=re.IGNORECASE,
        )
        return text.removesuffix(" Data saved Save") + (" Save" if text.endswith(" Data saved Save") else "")

    text_atoms, other_atoms = [], []
    for key, value in _atoms(frame["accessibility"]):
        if key == "text":
            text_atoms.append(value)
        else:
            if key.split(" ", 1)[0] in {"textbox", "spinbutton"}:
                value = None
            if key == 'button "Save" [disabled]':
                key = 'button "Save"'
            other_atoms.append((key, value))
    frame["accessibility"] = (normalized(" ".join(text_atoms)).removesuffix(" Data saved"), other_atoms)
    frame["text"] = normalized(frame["text"])
    for control in frame["controls"]:
        if control["role"] in {"textbox", "spinbutton"}:
            control["value"] = None
            control["accessibility"] = (_control_atom(control["accessibility"])[0], None)
        if control["role"] == "button" and control["accessibility"] in {
            '- button "Save"',
            '- button "Save" [disabled]',
        }:
            control["enabled"] = True
            control["accessibility"] = '- button "Save"'
    values = deepcopy(mapping["observation"]["values"])
    values["equilibrium_temp"]["value"] = None
    values["readouts"]["surface_temp"] = None
    values["readouts"]["absorption"]["display_text"] = None
    return result, values


class HabitabilityNumericSession:
    def __init__(self, page, config, output, *, max_seconds=120):
        if type(max_seconds) not in {int, float} or not 1 <= max_seconds <= 300:
            raise ValueError("Habitability copy deadline must be bounded")
        self.page, self.config, self.output = page, config, Path(output)
        self.frames = page.frames.copy()
        self.frame = None
        self.stopped = self.attempted = self.unexpected_dialog = False
        self.deadline = time.monotonic() + max_seconds
        self.report, self.mapping, self.choices, self.handle = self.read()
        values = self.mapping["observation"]["values"]
        if values["equilibrium_temp"]["value"] not in {"", "0"} or values["greenhouse"] is not None:
            raise BrowserSafetyStop("habitability_temperature_stage_already_populated")
        self.output.mkdir(parents=True, exist_ok=False)
        save_probe(self.report, self.output / "initial")
        persist_json(
            self.output / "scope.json",
            {
                "scope": "one_equilibrium_copy_before_greenhouse_selection",
                "star": self.mapping["star_name"],
                "max_writes": 1,
                "max_seconds": max_seconds,
                "task_completed": False,
                "selection_source": "caller_supplied",
                "automatic_retry": False,
            },
        )
        page.on("dialog", self._dialog)

    def _dialog(self, dialog):
        self.unexpected_dialog = True
        dialog.dismiss()

    def close(self):
        self.stopped = True
        self.page.remove_listener("dialog", self._dialog)

    def read(self):
        if self.stopped or time.monotonic() >= self.deadline:
            raise BrowserSafetyStop("habitability_copy_stopped_or_timed_out")
        self.page.wait_for_timeout(0)
        if self.unexpected_dialog or self.page.frames != self.frames or len(self.page.context.pages) != 1:
            raise BrowserSafetyStop("habitability_copy_context_changed")
        report = inspect_page(self.page, self.config)
        if report["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_visible_frame")
        frames = [f for f in self.page.frames if f.url == SIMULATION_URL]
        if len(frames) != 1 or self.frame is not None and frames[0] != self.frame:
            raise BrowserSafetyStop("simulation_frame_changed")
        self.frame = frames[0]
        mapping = map_habitability_capture(report, capture_sha256=screen_identity(report))
        choices, _ = read_habitability_choices(self.frame)
        controls = [
            c
            for role in ("textbox", "spinbutton")
            for c in self.frame.get_by_role(role).all()
            if c.is_visible()
        ]
        if (
            len(controls) != 1
            or not controls[0].is_enabled()
            or not rendered_control(controls[0])
            or not controls[0].evaluate(VISIBLE_PREFIX).strip().casefold().endswith("equilibrium temp (k)")
            or controls[0].input_value() != mapping["observation"]["values"]["equilibrium_temp"]["value"]
        ):
            raise BrowserSafetyStop("unverified_equilibrium_target")
        return report, mapping, choices, controls[0].element_handle(timeout=2000)

    def current(self):
        try:
            report, mapping, choices, handle = self.read()
            values = mapping["observation"]["values"]
            previous = self.mapping["observation"]["values"]
            if (
                equilibrium_projection(report, mapping) != equilibrium_projection(self.report, self.mapping)
                or values["equilibrium_temp"] != previous["equilibrium_temp"]
                or values["readouts"]["surface_temp"] != previous["readouts"]["surface_temp"]
                or choices != self.choices
                or not self.handle.evaluate("(a,b)=>a.isConnected&&a===b", handle)
            ):
                raise BrowserSafetyStop("stale_habitability_observation")
            return report, mapping, choices, handle
        except BaseException:
            self.stopped = True
            raise

    def copy(self, text, unit, *, source):
        if self.attempted:
            raise BrowserSafetyStop("habitability_copy_already_attempted")
        if unit != "K" or source not in {"reference_diagnostic", "checkpoint"}:
            raise BrowserSafetyStop("invalid_habitability_copy_request")
        committed_display(text, text)
        before, mapping, choices, handle = self.current()
        self.attempted = True
        try:
            intent = {
                "kind": "TYPE",
                "destination": "equilibrium_temp",
                "value": text,
                "unit": unit,
                "action_source": source,
                "target": mapping["observation"]["values"]["equilibrium_temp"]["capture_target_id"],
                "task_completed": False,
            }
            persist_json(self.output / "reserved.json", intent)
            self.current()
            handle.fill(text, timeout=3000)
            if handle.input_value() != text:
                raise BrowserSafetyStop("equilibrium_exact_fill_mismatch")
            handle.press("Tab", timeout=3000)
            after, newer, new_choices, new_handle = self.read()
            save_probe(after, self.output / "after")
            if (
                equilibrium_projection(before, mapping) != equilibrium_projection(after, newer)
                or choices != new_choices
                or not handle.evaluate("(a,b)=>a.isConnected&&a===b", new_handle)
            ):
                raise BrowserSafetyStop("unexpected_equilibrium_copy_side_effect")
            values = newer["observation"]["values"]
            visible = values["equilibrium_temp"]["value"]
            receipt = {
                **intent,
                "display": committed_display(text, visible),
                "surface_display": committed_display(
                    visible, values["readouts"]["surface_temp"]["display_text"]
                ),
                "readback_verified": True,
                "correctness_verified": False,
            }
            persist_json(self.output / "confirmed.json", receipt)
            return receipt
        except BaseException as exc:
            self.stopped = True
            persist_json(
                self.output / "stopped.json",
                {
                    "reason": str(exc)
                    if isinstance(exc, BrowserSafetyStop)
                    else "equilibrium_copy_uncertain",
                    "retry_allowed": False,
                    "write_may_have_occurred": True,
                    "task_completed": False,
                },
            )
            raise
