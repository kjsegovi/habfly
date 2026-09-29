"""Bounded exact-copy transport for explicitly selected planet answers.

No calculation, measurement selection, classification, scoring, or submission.
The caller must select Has Planet separately. Each destination is written once;
uncertain writes stop permanently and are never rolled back or retried.
"""

import re
import time
from decimal import Decimal
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import read_planet_class_choices
from .browser_numeric import (
    VISIBLE_PREFIX,
    committed_display,
    comparable_screen,
    screen_identity,
)
from .browser_planet import PLANET_FIELDS, map_planet_capture
from .browser_probe import _visible_frame, inspect_page, save_probe
from .browser_setup import rendered_control
from .browser_stellar import NUMBER, SIMULATION_URL, _atoms, _control_atom

ANSWER_UNITS = {name: unit for name, unit in PLANET_FIELDS.values() if name != "observation_days"}


def planet_projection(report, mapping, destination=None):
    """Ignore only known transient tooltips/footer and this write's exact value.

    This is a comparison policy, not a source of chart measurements. Raw before
    and after captures are retained. Unknown feedback and all other controls,
    values, units, chart axes, and external frames remain comparable.
    """
    value = comparable_screen(report)
    frame = next(f for f in value["frames"] if f["url"] == SIMULATION_URL)
    target = (
        mapping["observation"]["values"]["browser_field_map"][destination]["capture_target_id"]
        if destination
        else None
    )
    atoms = _atoms(frame["accessibility"])
    target_atom = None
    target_ordinal = None
    numeric = [c for c in frame["controls"] if c["role"] in {"textbox", "spinbutton"}]
    if target:
        target_ordinal = next(i for i, c in enumerate(numeric) if c["id"] == target)
        target_atom = _control_atom(numeric[target_ordinal]["accessibility"])
        numeric[target_ordinal]["value"] = None
        numeric[target_ordinal]["accessibility"] = (target_atom[0], None)
    for c in frame["controls"]:
        if c["role"] == "button" and c["accessibility"] in {
            '- button "Save"',
            '- button "Save" [disabled]',
        }:
            c["enabled"] = True
            c["accessibility"] = '- button "Save"'
    position = 0
    projected = []
    for key, text in atoms:
        if key.split(" ", 1)[0] in {"textbox", "spinbutton"}:
            if position == target_ordinal:
                text = None
            position += 1
        if key == 'button "Save" [disabled]':
            key = 'button "Save"'
        if key == "img" and isinstance(text, str) and text.startswith("Normalized Flux Days Observed "):
            # This same transient tooltip is also appended to the SVG's AX
            # name. Keep every axis glyph and only strip its exact suffix.
            text = re.sub(rf" Brightness: {NUMBER}% , Day: [0-9]+$", "", text)
        if key == "text" and isinstance(text, str):
            # Observed spectrum has three static labels and an optional hover
            # prefix. Remove only the extra prefix, never the static scale.
            text = re.sub(rf"^(?:{NUMBER})nm (?=656\.3nm .* observe for$)", "", text)
            if destination in {"orbital_radius", "planet_mass"}:
                text = re.sub(rf"(ORBIT \(years\) )({NUMBER})(?= |$)", r"\1<derived>", text)
            # Footer is only accepted immediately before the final Save.
            if (
                text.endswith(" Data saved")
                and len(atoms) >= 2
                and (key, text) == atoms[-2]
                and atoms[-1][0].startswith('button "Save"')
            ):
                text = text.removesuffix(" Data saved")
        projected.append((key, text))
    frame["accessibility"] = projected
    text = frame["text"]
    text = re.sub(rf"(?m)^(?:{NUMBER})nm\n(?=656\.3nm\n)", "", text)
    # A chart tooltip can disappear when focus moves to a native answer. Its
    # data is never consumed here; the guarded chart transport owns sampling.
    text = re.sub(rf"(?m)^Brightness: {NUMBER}% , Day: [0-9]+\n?", "", text)
    if destination in {"orbital_radius", "planet_mass"}:
        text = re.sub(rf"(ORBIT \(years\)[ \n])({NUMBER})(?=\n|$)", r"\1<derived>", text)
    if text.endswith("\nData saved\nSave"):
        text = text.removesuffix("\nData saved\nSave") + "\nSave"
    frame["text"] = text
    return value


def validate_orbit_readout_change(before, after, destination):
    """A displayed reconstruction readout is not an answer or measurement.

    The course updates ORBIT for both a and planet mass. Accept only one
    finite numeric readout in both captures. Positive added mass cannot
    increase a positive period at fixed a; no exact course formula is assumed.
    Other destinations do not receive this exception.
    """
    if destination not in {"orbital_radius", "planet_mass"}:
        return
    numbers = []
    for report in (before, after):
        text = next(f["text"] for f in report["frames"] if f["url"] == SIMULATION_URL)
        values = re.findall(rf"ORBIT \(years\)\s+({NUMBER})(?=\s|$)", text)
        if len(values) != 1:
            raise BrowserSafetyStop("invalid_dependent_orbit_readout")
        value = Decimal(values[0])
        if not value.is_finite() or value < 0 or value and not -324 <= value.adjusted() <= 308:
            raise BrowserSafetyStop("invalid_dependent_orbit_readout")
        numbers.append(value)
    old, new = numbers
    if old != new and (new <= 0 or destination == "planet_mass" and old > 0 and new > old):
        raise BrowserSafetyStop("unexpected_dependent_orbit_direction")


class PlanetNumericSession:
    """Seven at-most-once answer copies. Already-populated fields are protected."""

    def __init__(
        self,
        page,
        config,
        output,
        emit=lambda *_: None,
        *,
        max_seconds=300,
        _allow_unset_planet=False,
        _allow_no_planet=False,
        _pin_controls=False,
    ):
        if type(max_seconds) not in {int, float} or not 1 <= max_seconds <= 1800:
            raise ValueError("Planet copy deadline must be bounded")
        if type(_pin_controls) is not bool:
            raise ValueError("Pinned control reads require a boolean")
        self.page, self.config, self.output, self.emit = page, config, Path(output), emit
        self._pin_controls = _pin_controls
        self.longest_current_seconds = 0.0
        self.stopped, self.frame, self.attempted, self.verified = False, None, set(), {}
        self.deadline = time.monotonic() + max_seconds
        self.frames = self.page.frames.copy()
        self.unexpected_dialog = False
        self.report, self.mapping, self.choices, self.handles = self.read()
        selected = self.mapping["observation"]["values"]["has_planet"]
        if (
            selected != "Yes"
            and not (_allow_unset_planet and selected is None)
            and not (_allow_no_planet is True and selected == "No")
        ):
            raise BrowserSafetyStop("planet_yes_selection_required")
        self.output.mkdir(parents=True, exist_ok=False)
        save_probe(self.report, self.output / "initial")
        persist_json(
            self.output / "scope.json",
            {
                "scope": "planet_exact_copy_transport",
                "max_writes": 7,
                "max_seconds": max_seconds,
                "star": self.mapping["star_name"],
                "task_completed": False,
                "scientific_choices": "caller_supplied",
                "retries": False,
            },
        )
        page.on("dialog", self._dialog)

    def _dialog(self, dialog):
        self.unexpected_dialog = True
        dialog.dismiss()  # Never accept confirmations or log their text.

    def close(self):
        self.stopped = True
        self.page.remove_listener("dialog", self._dialog)

    def read(self):
        if self.stopped:
            raise BrowserSafetyStop("planet_copy_session_stopped")
        if time.monotonic() >= self.deadline:
            raise BrowserSafetyStop("planet_copy_time_limit")
        self.page.wait_for_timeout(0)
        if self.unexpected_dialog:
            raise BrowserSafetyStop("unexpected_browser_dialog")
        if self.page.frames != self.frames:
            raise BrowserSafetyStop("planet_copy_frame_set_changed")
        if len(self.page.context.pages) != 1:
            raise BrowserSafetyStop("unexpected_popup")
        report = inspect_page(
            self.page, self.config, **({"pin_controls": True} if self._pin_controls else {})
        )
        if report["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_visible_frame")
        frames = [
            f for f in self.page.frames if f.url == SIMULATION_URL and _visible_frame(f, self.page.main_frame)
        ]
        if len(frames) != 1 or self.frame is not None and frames[0] != self.frame:
            raise BrowserSafetyStop("simulation_frame_changed")
        self.frame = frames[0]
        mapping = map_planet_capture(report, capture_sha256=screen_identity(report))
        choices, _ = read_planet_class_choices(self.frame)
        handles = {}
        for role in ("textbox", "spinbutton"):
            for control in self.frame.get_by_role(role).all():
                if not control.is_visible():
                    continue
                label = "".join(control.evaluate(VISIBLE_PREFIX).casefold().split())
                field = PLANET_FIELDS.get(label)
                if field is None or field[0] == "observation_days":
                    continue
                name, unit = field
                spec = mapping["observation"]["values"]["browser_field_map"].get(name)
                if (
                    name in handles
                    or not spec
                    or spec["unit"] != unit
                    or control.input_value() != spec["current_value"]
                    or not control.is_enabled()
                    or not rendered_control(control)
                ):
                    raise BrowserSafetyStop("unverified_planet_target")
                handles[name] = control.element_handle(timeout=2000)
        expected = set(mapping["observation"]["values"]["browser_field_map"]) - {"observation_days"}
        if set(handles) != expected:
            raise BrowserSafetyStop("ambiguous_planet_target_binding")
        return report, mapping, choices, handles

    def current(self):
        started = time.monotonic()
        try:
            report, mapping, choices, handles = self.read()
            if (
                planet_projection(report, mapping) != planet_projection(self.report, self.mapping)
                or choices != self.choices
            ):
                raise BrowserSafetyStop("stale_planet_observation")
            for name, old in self.handles.items():
                if not old.evaluate("(a,b)=>a.isConnected&&a===b", handles[name]):
                    raise BrowserSafetyStop("planet_target_replaced")
            return report, mapping, choices, handles
        except BaseException:
            self.stopped = True
            raise
        finally:
            self.longest_current_seconds = max(self.longest_current_seconds, time.monotonic() - started)

    def copy(self, destination, text, unit, *, source):
        if self.mapping["observation"]["values"]["has_planet"] != "Yes":
            raise BrowserSafetyStop("planet_yes_selection_required")
        if (
            destination not in ANSWER_UNITS
            or unit != ANSWER_UNITS[destination]
            or source not in {"reference_diagnostic", "checkpoint"}
        ):
            raise BrowserSafetyStop("invalid_planet_copy_request")
        # Use the exact caller spelling. Never choose an answer or fix a value.
        committed_display(text, text)
        if destination == "brightness_drop" and Decimal(text) > 100:
            raise BrowserSafetyStop("invalid_planet_brightness_drop")
        before, mapping, choices, handles = self.current()
        field = mapping["observation"]["values"]["browser_field_map"][destination]
        if destination in self.attempted or field["current_value"] != "":
            raise BrowserSafetyStop("planet_destination_already_populated")
        prefix = f"copy-{len(self.attempted) + 1:02d}"
        self.attempted.add(destination)
        try:
            save_probe(before, self.output / f"{prefix}-before")
            intent = {
                "kind": "TYPE",
                "target": field["capture_target_id"],
                "destination": destination,
                "value": text,
                "unit": unit,
                "action_source": source,
                "task_completed": False,
            }
            persist_json(self.output / f"{prefix}-reserved.json", intent)
            self.emit("action_proposed", intent)
            self.current()  # Includes post-reservation target identity check.
            handle = handles[destination]
            handle.fill(text, timeout=3000)
            if handle.input_value() != text:
                raise BrowserSafetyStop("planet_exact_fill_readback_mismatch")
            handle.press("Tab", timeout=3000)
            after, newer, new_choices, new_handles = self.read()
            save_probe(after, self.output / f"{prefix}-after")
            validate_orbit_readout_change(before, after, destination)
            if (
                planet_projection(before, mapping, destination)
                != planet_projection(after, newer, destination)
                or choices != new_choices
                or not handle.evaluate("(a,b)=>a.isConnected&&a===b", new_handles[destination])
            ):
                raise BrowserSafetyStop("unexpected_planet_copy_side_effect")
            visible = newer["observation"]["values"]["browser_field_map"][destination]["current_value"]
            receipt = {
                **intent,
                "display": committed_display(text, visible),
                "readback_verified": True,
                "correctness_verified": False,
            }
            persist_json(self.output / f"{prefix}-confirmed.json", receipt)
            self.report, self.mapping, self.choices, self.handles = after, newer, new_choices, new_handles
            self.verified[destination] = receipt
            self.emit("action_result", receipt)
            return receipt
        except BaseException as exc:
            self.stopped = True
            persist_json(
                self.output / f"{prefix}-stopped.json",
                {
                    "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "planet_copy_uncertain",
                    "retry_allowed": False,
                    "write_may_have_occurred": True,
                    "task_completed": False,
                },
            )
            raise
