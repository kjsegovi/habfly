"""One scripted Yes selection supported by same-star measured transit evidence.

Not a learned classification policy. Absence is deliberately unsupported: a
finite flat interval must never become a guessed No. No numeric answers, class,
Save, scoring, or submission are performed here.
"""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_numeric import screen_identity
from .browser_planet import load_planet_capture
from .browser_planet_evidence import measurement_evidence
from .browser_planet_numeric import PlanetNumericSession, planet_projection
from .browser_probe import save_probe
from .browser_setup import rendered_control
from .browser_stellar import SIMULATION_URL, _control_atom

DERIVED = {"orbital_radius", "planet_mass", "planet_radius", "planet_density"}


def preserved_paint_transition(before, after):
    if before == after:
        return True
    previous = before["selected"]
    if previous is None or after["selected"] is not None:
        return False
    unselected = [row for key, row in before["rendering"].items() if key != previous]
    if len(unselected) != 2 or unselected[0] != unselected[1]:
        return False
    expected = deepcopy(before)
    expected["selected"] = None
    expected["rendering"][previous] = deepcopy(unselected[0])
    return after == expected


def presence_projection(report, mapping):
    """Remove only the explicitly expected blank conditional answer panel."""
    result = planet_projection(report, mapping)
    frame = next(f for f in result["frames"] if f["url"] == SIMULATION_URL)
    fields = mapping["observation"]["values"]["browser_field_map"]
    remove = {spec["capture_target_id"] for k, spec in fields.items() if k in DERIVED}
    frame["controls"] = [c for c in frame["controls"] if c["id"] not in remove]
    for i, control in enumerate(frame["controls"]):
        control["id"] = f"{frame['id']}:presence:{i}"
        if control["role"] == "combobox":
            key, options = _control_atom(control["accessibility"])
            control["value"] = None
            control["accessibility"] = (key, [s.removesuffix(" [selected]") for s in options])
    atoms = frame["accessibility"]
    if remove:
        start = atoms.index(("text", "orbital radius (au)"))
        last = max(i for i, (k, _) in enumerate(atoms) if k.split(" ", 1)[0] in {"textbox", "spinbutton"})
        atoms[start : last + 1] = []
    frame["accessibility"] = [
        (k, [s.removesuffix(" [selected]") for s in v]) if k == "combobox" else (k, v) for k, v in atoms
    ]
    text = " ".join(frame["text"].split())
    if remove:
        text, count = re.subn(
            r"orbital radius \(au\)\s*mass \(M\s*E\)\s*radius \(R\s*E\)\s*density \(g/cm\s*3\) ",
            "",
            text,
            flags=re.IGNORECASE,
        )
        if count != 1:
            raise BrowserSafetyStop("unsupported_planet_conditional_panel")
    frame["text"] = text
    values = deepcopy(mapping["observation"]["values"])
    values["has_planet"] = None
    values["browser_field_map"] = {
        k: {a: b for a, b in spec.items() if a != "capture_target_id"}
        for k, spec in fields.items()
        if k not in DERIVED
    }
    return result, values


def select_detected_planet(page, config, output, *, spectrum_path, transit_path, preserve_painted_class=None):
    # The live app can retain a prior star's painted radio while the current
    # star has no presence answer. An explicit expected paint may be preserved,
    # but it is never interpreted as a committed or correct classification.
    if preserve_painted_class not in {None, "gas_giant", "ice_giant", "terrestrial"}:
        raise BrowserSafetyStop("invalid_preserved_planet_paint")
    raw = {
        name: Path(path).read_bytes()
        for name, path in (("spectrum", spectrum_path), ("transit", transit_path))
    }
    evidence = measurement_evidence(json.loads(raw["spectrum"]), json.loads(raw["transit"]))
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    session = None
    attempted = False
    try:
        session = PlanetNumericSession(
            page, config, directory / "read-guard", max_seconds=60, _allow_unset_planet=True
        )
        before, mapping, choices, _ = session.current()
        values = mapping["observation"]["values"]
        if mapping["star_name"].upper() != evidence["star"]:
            raise BrowserSafetyStop("planet_evidence_star_changed")
        if values["has_planet"] is not None or choices["selected"] != preserve_painted_class:
            raise BrowserSafetyStop("planet_presence_or_class_already_selected")
        if any(
            spec["current_value"] != ""
            for k, spec in values["browser_field_map"].items()
            if k != "observation_days"
        ):
            raise BrowserSafetyStop("planet_measurements_already_populated")
        control = session.frame.get_by_role("combobox")
        if (
            control.count() != 1
            or not rendered_control(control)
            or not control.is_enabled()
            or control.input_value() != ""
        ):
            raise BrowserSafetyStop("unverified_planet_presence_control")
        if control.locator("option").all_text_contents() != ["", "Yes", "No"]:
            raise BrowserSafetyStop("unsupported_planet_presence_options")
        handle = control.element_handle(timeout=2000)
        intent = {
            **evidence,
            "kind": "SELECT",
            "value": "Yes",
            "numeric_writes": 0,
            "class_writes": 0,
            "preserved_painted_class": preserve_painted_class,
            "class_selection_verified": False,
            "painted_choices_before": choices,
            "source_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
            "automatic_retry": False,
        }
        persist_json(directory / "reserved.json", intent)
        session.current()
        if not handle.evaluate("(a,b)=>a.isConnected&&a===b", control.element_handle(timeout=2000)):
            raise BrowserSafetyStop("planet_presence_control_replaced")
        attempted = True
        handle.select_option(label="Yes", timeout=3000)
        after, newer, selected, _ = session.read()
        save_probe(after, directory / "after")
        fields = newer["observation"]["values"]["browser_field_map"]
        if (
            newer["observation"]["values"]["has_planet"] != "Yes"
            or not preserved_paint_transition(choices, selected)
            or not DERIVED.issubset(fields)
            or any(fields[k]["current_value"] != "" for k in DERIVED)
            or presence_projection(before, mapping) != presence_projection(after, newer)
            or not handle.evaluate("(a,b)=>a.isConnected&&a===b", control.element_handle(timeout=2000))
        ):
            raise BrowserSafetyStop("unexpected_planet_presence_side_effect")
        receipt = {
            **intent,
            "readback_verified": True,
            "correctness_verified": False,
            "painted_class_after": selected["selected"],
            "inherited_paint_cleared": choices["selected"] is not None and selected["selected"] is None,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "planet_presence_operation_failed",
                "write_may_have_occurred": attempted,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        raise
    finally:
        if session is not None:
            session.close()


def reconcile_presence_readback(page, config, source, output, *, spectrum_path, transit_path):
    """Reconcile current Yes after a stopped readback; never retry the native selection."""
    source, directory = Path(source), Path(output)
    intent = json.loads((source / "reserved.json").read_text())
    failure = json.loads((source / "stopped.json").read_text())
    raw = {
        name: Path(path).read_bytes()
        for name, path in (("spectrum", spectrum_path), ("transit", transit_path))
    }
    evidence = measurement_evidence(json.loads(raw["spectrum"]), json.loads(raw["transit"]))
    if (
        failure.get("reason") != "unexpected_planet_presence_side_effect"
        or failure.get("write_may_have_occurred") is not True
        or intent.get("value") != "Yes"
        or intent.get("kind") != "SELECT"
        or intent.get("numeric_writes") != 0
        or intent.get("class_writes") != 0
        or intent.get("preserved_painted_class") not in {"gas_giant", "ice_giant", "terrestrial"}
        or intent.get("source_sha256") != {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}
        or any(intent.get(k) != value for k, value in evidence.items())
    ):
        raise BrowserSafetyStop("unsupported_presence_reconciliation")
    before_map = load_planet_capture(source / "read-guard/initial")
    after_map = load_planet_capture(source / "after")
    before = json.loads((source / "read-guard/initial/observation.json").read_text())
    after = json.loads((source / "after/observation.json").read_text())
    if (
        before_map["star_name"].upper() != evidence["star"]
        or after_map["star_name"] != before_map["star_name"]
        or before_map["observation"]["values"]["has_planet"] is not None
        or after_map["observation"]["values"]["has_planet"] != "Yes"
        or presence_projection(before, before_map) != presence_projection(after, after_map)
    ):
        raise BrowserSafetyStop("presence_reconciliation_observation_changed")
    fields = after_map["observation"]["values"]["browser_field_map"]
    if not DERIVED.issubset(fields) or any(
        row["current_value"] != "" for k, row in fields.items() if k != "observation_days"
    ):
        raise BrowserSafetyStop("presence_reconciliation_answers_changed")
    directory.mkdir(parents=True, exist_ok=False)
    session = None
    try:
        session = PlanetNumericSession(page, config, directory / "read-guard", max_seconds=60)
        current, mapping, choices, _ = session.current()
        if (
            planet_projection(current, mapping) != planet_projection(after, after_map)
            or choices["selected"] is not None
        ):
            raise BrowserSafetyStop("presence_reconciliation_live_state_changed")
        save_probe(current, directory / "capture")
        session.current()
        receipt = {
            **evidence,
            "value": "Yes",
            "readback_reconciled": True,
            "original_failure_preserved": True,
            "browser_actions": 0,
            "class_writes": 0,
            "numeric_writes": 0,
            "class_selection_verified": False,
            "automatic_retry": False,
            "painted_class_after": None,
            "correctness_verified": False,
            "capture_sha256": screen_identity(current),
            "source_sha256": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ("reserved.json", "stopped.json", "after/observation.json")
            },
        }
        persist_json(directory / "reconciled.json", receipt)
        return receipt
    finally:
        if session is not None:
            session.close()
