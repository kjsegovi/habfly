"""One explicit habitability reference choice after a confirmed phase selection.

This is transport, not a learned habitability classifier. Liquid water is a
necessary reference condition, not proof of a complete scientific assessment.
No other answer, Save, scoring or submission action is available here.
"""

import hashlib
import json
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import read_habitability_choices
from .browser_habitability import load_habitability_capture
from .browser_habitability_actions import HabitabilityMenuSession
from .browser_habitability_numeric import equilibrium_projection
from .browser_probe import save_probe


def confirmed_phase_reference(directory, mapping):
    """Bind an existing phase receipt and its hashed chamber evidence to now."""
    directory = Path(directory).resolve()
    raw = {
        name: (directory / name).read_bytes()
        for name in ("reserved.json", "confirmed.json", "after/observation.json", "after/manifest.json")
    }
    intent, receipt = json.loads(raw["reserved.json"]), json.loads(raw["confirmed.json"])
    prior = load_habitability_capture(directory / "after")
    values = mapping["observation"]["values"]
    evidence = receipt.get("evidence", {})
    hashes = evidence.get("source_sha256")
    if (
        (directory / "stopped.json").exists()
        or receipt
        != {**intent, "readback_verified": True, "surface_temp": values["readouts"]["surface_temp"]}
        or receipt.get("field") != "water_phase"
        or receipt.get("action_source") != "reference_diagnostic"
        or receipt.get("correctness_verified") is not False
        or receipt.get("task_completed") is not False
        or receipt.get("star") != mapping["star_name"]
        or prior["observation"]["values"] != values
        or receipt.get("label") != values["water_phase"]
        or values["water_phase"] not in {"Solid", "Liquid", "Gas"}
        or evidence.get("source") != "confirmed_visible_chamber_indicator"
        or evidence.get("phase") != values["water_phase"].lower()
        or not isinstance(hashes, dict)
        or len(hashes) != 4
        or {Path(p).name for p in hashes}
        != {"confirmed.json", "reserved.json", "observation.json", "manifest.json"}
        or any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in hashes.items())
    ):
        raise BrowserSafetyStop("incompatible_habitability_phase_receipt")
    return {
        "phase": values["water_phase"],
        "pressure": evidence["pressure"],
        "temperature": evidence["temperature"],
        "source_sha256": {
            **hashes,
            **{str(directory / name): hashlib.sha256(data).hexdigest() for name, data in raw.items()},
        },
    }


def select_habitability_reference(page, config, output, *, phase_record, name, rationale):
    if (
        name not in {"habitable", "not_habitable"}
        or not isinstance(rationale, str)
        or not 40 <= len(rationale.strip()) <= 2000
    ):
        raise ValueError("An explicit reference choice and bounded scientific rationale are required")
    directory = Path(output)
    session, dispatched = None, False
    try:
        session = HabitabilityMenuSession(page, config, directory)
        before, mapping, choices, _ = session.current()
        evidence = confirmed_phase_reference(phase_record, mapping)
        if name == "habitable" and evidence["phase"] != "Liquid":
            raise BrowserSafetyStop("habitability_reference_requires_liquid_water")
        visible, handles = read_habitability_choices(session.frame)
        if visible != choices:
            raise BrowserSafetyStop("habitability_choice_changed_during_binding")
        intent = {
            "star": mapping["star_name"],
            "kind": "SELECT",
            "choice": name,
            "previous_paint": choices["selected"],
            "action_source": "reference_diagnostic",
            "rationale": rationale.strip(),
            "evidence": evidence,
            "learned_habitability_decision": False,
            "correctness_verified": False,
            "task_completed": False,
            "numeric_writes": 0,
            "max_choice_clicks": 1,
            "automatic_retry": False,
        }
        persist_json(directory / "reserved.json", intent)
        session.current()
        if confirmed_phase_reference(phase_record, mapping) != evidence:
            raise BrowserSafetyStop("habitability_phase_evidence_changed")
        latest, rebound = read_habitability_choices(session.frame)
        if latest != visible or any(
            not h.evaluate("(a,b)=>a.isConnected&&a===b", rebound[k]) for k, h in handles.items()
        ):
            raise BrowserSafetyStop("habitability_choice_control_replaced")
        dispatched = True
        handles[name].click(timeout=3000)
        after, newer, selected, _ = session.read()
        save_probe(after, directory / "after")
        _, after_handles = read_habitability_choices(session.frame)
        if (
            selected["selected"] != name
            or mapping["observation"]["values"] != newer["observation"]["values"]
            or equilibrium_projection(before, mapping) != equilibrium_projection(after, newer)
            or any(
                not h.evaluate("(a,b)=>a.isConnected&&a===b", after_handles[k]) for k, h in handles.items()
            )
        ):
            raise BrowserSafetyStop("unexpected_habitability_choice_side_effect")
        receipt = {**intent, "readback_verified": True}
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        if session is not None:
            persist_json(
                directory / "stopped.json",
                {
                    "reason": str(exc)
                    if isinstance(exc, BrowserSafetyStop)
                    else "habitability_choice_failed",
                    "write_may_have_occurred": dispatched,
                    "automatic_retry": False,
                    "task_completed": False,
                },
            )
        raise
    finally:
        if session is not None:
            session.close()
