"""Reference-assisted terrestrial menu transport, never a learned classifier.

One explicit greenhouse or water-phase selection. No gas inference, temperature
copy, Save, assessment, score update or submission. Changed/uncertain state
stops; opening a new output directory is not an automatic retry mechanism.
"""

import hashlib
import json
import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import read_habitability_choices
from .browser_habitability import load_habitability_capture, map_habitability_capture
from .browser_habitability_numeric import equilibrium_projection
from .browser_numeric import committed_display, screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_setup import rendered_control
from .browser_stellar import SIMULATION_URL, _control_atom
from .habitability_knowledge import HabitabilityCalculator


def menu_projection(report, mapping, field):
    """Normalize only the selected menu and its documented temperature readout."""
    result, values = equilibrium_projection(report, mapping)
    original = mapping["observation"]["values"]
    # Unlike a numeric copy, these actions must preserve the exact input.
    values["equilibrium_temp"] = deepcopy(original["equilibrium_temp"])
    if field != "greenhouse":
        values["readouts"]["surface_temp"] = deepcopy(original["readouts"]["surface_temp"])
    values[field] = None
    frame = next(f for f in result["frames"] if f["url"] == SIMULATION_URL)
    target_id = next(c["id"] for c in mapping["observation"]["controls"] if c["label"] == field)
    target = next(c for c in frame["controls"] if c["id"] == target_id)
    atom = _control_atom(target["accessibility"])
    neutral = (atom[0], [v.removesuffix(" [selected]") for v in atom[1]])
    text, atoms = frame["accessibility"]
    if atoms.count(atom) != 1:
        raise BrowserSafetyStop("ambiguous_habitability_menu_projection")
    frame["accessibility"] = (text, [neutral if a == atom else a for a in atoms])
    target["accessibility"], target["value"] = neutral, None
    return result, values


class HabitabilityMenuSession:
    """Bound one menu action to its current visible star and native control."""

    def __init__(self, page, config, output, *, max_seconds=60, _pin_controls=False):
        if type(max_seconds) not in {int, float} or not 1 <= max_seconds <= 120:
            raise ValueError("Bounded menu deadline required")
        if type(_pin_controls) is not bool:
            raise ValueError("Pinned control reads require a boolean")
        self.page, self.config, self.output = page, config, Path(output)
        self._pin_controls = _pin_controls
        self.frames = page.frames.copy()
        self.deadline = time.monotonic() + max_seconds
        self.unexpected_dialog = self.attempted = self.stopped = False
        self.frame = None
        self.report, self.mapping, self.choices, self.handles = self.read()
        self.output.mkdir(parents=True, exist_ok=False)
        save_probe(self.report, self.output / "initial")
        page.on("dialog", self._dialog)

    def _dialog(self, dialog):
        self.unexpected_dialog = True
        dialog.dismiss()

    def close(self):
        self.stopped = True
        self.page.remove_listener("dialog", self._dialog)

    def read(self):
        if self.stopped or time.monotonic() >= self.deadline:
            raise BrowserSafetyStop("habitability_menu_stopped_or_timed_out")
        self.page.wait_for_timeout(0)
        if self.unexpected_dialog or self.page.frames != self.frames or len(self.page.context.pages) != 1:
            raise BrowserSafetyStop("habitability_menu_context_changed")
        report = inspect_page(
            self.page, self.config, **({"pin_controls": True} if self._pin_controls else {})
        )
        if report["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_visible_frame")
        frames = [f for f in self.frames if f.url == SIMULATION_URL]
        if len(frames) != 1 or self.frame is not None and self.frame != frames[0]:
            raise BrowserSafetyStop("habitability_menu_frame_changed")
        self.frame = frames[0]
        mapping = map_habitability_capture(report, capture_sha256=screen_identity(report))
        choices, _ = read_habitability_choices(self.frame)
        menus = [e for e in self.frame.get_by_role("combobox").all() if e.is_visible()]
        if len(menus) != 3:
            raise BrowserSafetyStop("habitability_menu_inventory_changed")
        handles = {}
        for field, control in zip(("greenhouse", "water_phase"), menus[1:], strict=True):
            if not control.is_enabled() or not rendered_control(control):
                raise BrowserSafetyStop("habitability_menu_unavailable")
            handles[field] = control.element_handle(timeout=1000)
        return report, mapping, choices, handles

    def current(self):
        current = self.read()
        report, mapping, choices, handles = current
        if (
            mapping["observation"]["values"] != self.mapping["observation"]["values"]
            or equilibrium_projection(report, mapping) != equilibrium_projection(self.report, self.mapping)
            or choices != self.choices
            or any(not h.evaluate("(a,b)=>a.isConnected&&a===b", handles[k]) for k, h in self.handles.items())
        ):
            self.stopped = True
            raise BrowserSafetyStop("stale_habitability_menu_observation")
        return current

    def _select(self, field, label, evidence):
        if self.attempted:
            raise BrowserSafetyStop("habitability_menu_already_attempted")
        before, mapping, choices, handles = self.current()
        values = mapping["observation"]["values"]
        if values[field] is not None:
            raise BrowserSafetyStop("habitability_menu_already_populated")
        committed_display(values["equilibrium_temp"]["value"], values["equilibrium_temp"]["value"])
        intent = {
            "star": mapping["star_name"],
            "kind": "SELECT",
            "field": field,
            "label": label,
            "action_source": "reference_diagnostic",
            "evidence": evidence,
            "max_menu_writes": 1,
            "numeric_writes": 0,
            "task_completed": False,
            "correctness_verified": False,
            "automatic_retry": False,
        }
        self.attempted = True
        dispatched = False
        try:
            persist_json(self.output / "reserved.json", intent)
            self.current()
            for path, digest in evidence.get("source_sha256", {}).items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                    raise BrowserSafetyStop("habitability_menu_evidence_changed")
            dispatched = True
            handles[field].select_option(label=label, timeout=3000)
            after, newer, selected, new_handles = self.read()
            save_probe(after, self.output / "after")
            actual = newer["observation"]["values"]
            if (
                actual[field] != label
                or choices != selected
                or menu_projection(before, mapping, field) != menu_projection(after, newer, field)
                or any(
                    not h.evaluate("(a,b)=>a.isConnected&&a===b", new_handles[k]) for k, h in handles.items()
                )
            ):
                raise BrowserSafetyStop("unexpected_habitability_menu_side_effect")
            if field == "greenhouse":
                expected = Decimal(values["equilibrium_temp"]["value"]) + evidence["increment_kelvin"]
                committed_display(str(expected), actual["readouts"]["surface_temp"]["display_text"])
            receipt = {
                **intent,
                "readback_verified": True,
                "surface_temp": actual["readouts"]["surface_temp"],
            }
            persist_json(self.output / "confirmed.json", receipt)
            return receipt
        except BaseException as exc:
            self.stopped = True
            persist_json(
                self.output / "stopped.json",
                {
                    "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "habitability_menu_failed",
                    "write_may_have_occurred": dispatched,
                    "automatic_retry": False,
                    "task_completed": False,
                },
            )
            raise

    def greenhouse_reference(self):
        """Published absorption-band lookup, explicitly not a learned decision."""
        self.current()
        text = self.mapping["observation"]["values"]["readouts"]["absorption"]["display_text"]
        calculator = HabitabilityCalculator()
        reference = calculator.greenhouse_reference(text)
        labels = {"weak": "Weak (+10)", "moderate": "Moderate (+30)", "strong": "Strong (+100)"}
        if not reference["ok"] or reference["name"] not in labels:
            raise BrowserSafetyStop("no_explicit_supported_greenhouse_option")
        return self._select(
            "greenhouse",
            labels[reference["name"]],
            {
                **reference,
                "knowledge_pack_sha256": calculator.pack.checksum,
                "gas_identification_verified": False,
            },
        )

    def water_phase_from_chamber(self, directory):
        """Only a same-star, same-condition confirmed ordinary chamber query."""
        self.current()
        directory = Path(directory).resolve()
        names = (
            "confirmed.json",
            "reserved.json",
            "task-before/observation.json",
            "task-before/manifest.json",
        )
        raw = {name: (directory / name).read_bytes() for name in names}
        result, intent = json.loads(raw["confirmed.json"]), json.loads(raw["reserved.json"])
        before = load_habitability_capture(directory / "task-before")
        values = self.mapping["observation"]["values"]
        icons = result.get("icons", [])
        visible = [
            r.get("phase") for r in icons if r.get("fully_exposed") and r.get("paint", {}).get("opacity") == 1
        ]
        if (
            result.get("conditions_verified") is not True
            or result.get("source") != "visible_chamber_indicator"
            or result.get("phase") not in {"solid", "liquid", "gas"}
            or result.get("task_completed") is not False
            or set(intent)
            != {
                "pressure",
                "pressure_unit",
                "temperature",
                "temperature_unit",
                "action_source",
                "task_answer_write",
            }
            or intent.get("task_answer_write") is not False
            or intent.get("action_source") not in {"reference_diagnostic", "checkpoint"}
            or any(result.get(k) != v for k, v in intent.items())
            or len(icons) != 3
            or {r.get("phase") for r in icons} != {"solid", "liquid", "gas"}
            or visible != [result.get("phase")]
            or any(
                r.get("paint", {}).get("opacity") != 0 for r in icons if r.get("phase") != result.get("phase")
            )
            or (result.get("pressure_unit"), result.get("temperature_unit")) != ("atm", "K")
            or before["star_name"] != self.mapping["star_name"]
            or Decimal(result["pressure"]) != Decimal(values["measurements"]["pressure"]["display_text"])
            or Decimal(result["temperature"]) != Decimal(values["readouts"]["surface_temp"]["display_text"])
            or any(
                Decimal(result["visible_readback"][u]) != Decimal(result[k])
                for u, k in (("atm", "pressure"), ("K", "temperature"))
            )
        ):
            raise BrowserSafetyStop("incompatible_chamber_phase_evidence")
        return self._select(
            "water_phase",
            result["phase"].title(),
            {
                "source": "confirmed_visible_chamber_indicator",
                "phase": result["phase"],
                "pressure": result["pressure"],
                "temperature": result["temperature"],
                "source_sha256": {
                    str(directory / name): hashlib.sha256(value).hexdigest() for name, value in raw.items()
                },
            },
        )
