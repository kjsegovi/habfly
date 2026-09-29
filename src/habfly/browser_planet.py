"""Offline mapping of captured public planet fields; no action execution.

Tooltip visibility cannot be established from a saved AX dump. Chart measurements
must come from the separately guarded chart transport, never this field mapper.
"""

import hashlib
import json
import math
import re
from pathlib import Path

from .browser_stellar import NUMBER, SIMULATION_URL, StellarMappingError, _atoms, _control_atom, _role
from .contracts import Control, Observation

PLANET_FIELDS = {
    "observefor": ("observation_days", "day"),
    "dopplershift(nm)": ("line_shift", "nm"),
    "brightnessdrop(%)": ("brightness_drop", "%"),
    "brightnessdropperiod(days)": ("period_days", "day"),
    "orbitalradius(au)": ("orbital_radius", "au"),
    "mass(me)": ("planet_mass", "MEarth"),
    "radius(re)": ("planet_radius", "REarth"),
    "density(g/cm3)": ("planet_density", "g/cm3"),
}
BASE_FIELDS = {"observation_days", "line_shift", "brightness_drop", "period_days"}


def load_planet_capture(directory: Path):
    raw = (directory / "observation.json").read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    manifest = json.loads((directory / "manifest.json").read_text())
    if checksum != manifest.get("observation_sha256"):
        raise StellarMappingError("capture_hash_mismatch")
    return map_planet_capture(json.loads(raw), capture_sha256=checksum)


def map_planet_capture(report, *, capture_sha256):
    fail = StellarMappingError
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("mode") != "read_only_browser_preflight"
        or report.get("allow_submission") is not False
        or report.get("actions_executed") != 0
        or report.get("ignored_frame_urls")
        or not isinstance(capture_sha256, str)
        or not re.fullmatch(r"[a-f0-9]{64}", capture_sha256)
    ):
        raise fail("invalid_planet_capture_contract")
    inventory = report.get("frames")
    if not isinstance(inventory, list) or not all(isinstance(f, dict) for f in inventory):
        raise fail("invalid_planet_frame_inventory")
    frames = [f for f in inventory if f.get("url") == SIMULATION_URL]
    if len(frames) != 1:
        raise fail("ambiguous_simulation_frame")
    frame = frames[0]
    if (
        not all(isinstance(frame.get(k), str) and frame[k] for k in ("id", "text", "accessibility"))
        or not isinstance(frame.get("controls"), list)
        or not all(
            isinstance(c, dict)
            and all(isinstance(c.get(k), str) and c[k] for k in ("id", "role", "accessibility"))
            and type(c.get("enabled")) is bool
            for c in frame["controls"]
        )
    ):
        raise fail("invalid_planet_frame_contract")
    atoms = _atoms(frame["accessibility"])
    texts = [v for k, v in atoms if k == "text"]
    if (
        not texts
        or not isinstance(texts[0], str)
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", texts[0])
    ):
        raise fail("missing_planet_star_identity")
    star = texts[0]
    if not frame["text"].splitlines() or frame["text"].splitlines()[0].strip().casefold() != star.casefold():
        raise fail("conflicting_planet_star_identity")
    text = " ".join(v for k, v in atoms if k in {"text", "subscript", "superscript"} and isinstance(v, str))
    if any(
        len(re.findall(r"\b" + part + r"\b", text, re.IGNORECASE)) != 1
        for part in ("observations", "spectrum", "your reconstruction", "has planet")
    ):
        raise fail("not_planet_detail_screen")
    controls = frame["controls"]
    ids = [c["id"] for c in controls]
    if len(ids) != len(set(ids)) or any(not ident.startswith(frame["id"] + ":") for ident in ids):
        raise fail("invalid_planet_control_ids")
    numeric = [c for c in controls if c["role"] in {"textbox", "spinbutton"}]
    combos = [c for c in controls if c["role"] == "combobox"]
    if len(numeric) not in {4, 8} or len(combos) != 1:
        raise fail("unsupported_planet_fields")
    context, fields, position, selected = [], {}, 0, None
    menu_seen = False
    for atom_index, (key, value) in enumerate(atoms):
        role = _role(key)
        if key in {"text", "subscript", "superscript"}:
            if not isinstance(value, str):
                raise fail("unsupported_planet_label")
            context.append(value)
            continue
        if role not in {"textbox", "spinbutton", "combobox"}:
            context = []
            continue
        preceding = re.split(r"\byour reconstruction\b", " ".join(context), flags=re.IGNORECASE)[-1]
        label = re.sub(r"\s+", "", preceding.casefold())
        if role == "combobox":
            if (
                menu_seen
                or label != "hasplanet?"
                or _control_atom(combos[0]["accessibility"]) != (key, value)
            ):
                raise fail("ambiguous_planet_menu")
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise fail("unsupported_planet_options")
            normalized = [v.removesuffix(" [selected]") for v in value]
            choices = [v.removesuffix(" [selected]") for v in value if v.endswith(" [selected]")]
            if normalized != ['option "Yes"', 'option "No"'] or len(choices) > 1:
                raise fail("unsupported_planet_options")
            selected = choices[0].removeprefix('option "').removesuffix('"') if choices else None
            if combos[0].get("value") != (selected or ""):
                raise fail("planet_selection_readback_disagrees")
            menu_seen = True
        else:
            if position == 0 and label.endswith("observefor"):
                if atom_index + 1 >= len(atoms) or atoms[atom_index + 1] != ("text", "days"):
                    raise fail("unsupported_observation_duration_unit")
                label = "observefor"
                # Transient spectrum hover text is not part of this input's
                # label. The unmodified capture retains all raw evidence.
                preceding = "observe for"
            if label not in PLANET_FIELDS or position >= len(numeric):
                raise fail("unmapped_planet_field")
            name, unit = PLANET_FIELDS[label]
            control = numeric[position]
            if (
                name in fields
                or control["role"] != role
                or _control_atom(control["accessibility"]) != (key, value)
            ):
                raise fail("ambiguous_planet_field")
            if not isinstance(control.get("value"), str):
                raise fail("unknown_planet_field_value")
            if value is not None and str(value) != control["value"]:
                raise fail("planet_field_readback_disagrees")
            fields[name] = {
                "capture_target_id": control["id"],
                "unit": unit,
                "label_evidence": preceding.strip(),
                "current_value": control["value"],
                "enabled": control["enabled"],
                "binding_status": "capture_only_not_live_verified",
            }
            position += 1
        context = []
    required = BASE_FIELDS | (
        {"orbital_radius", "planet_mass", "planet_radius", "planet_density"} if selected == "Yes" else set()
    )
    if not menu_seen or set(fields) != required:
        raise fail("incomplete_planet_field_map")
    if any(
        len(re.findall(r"\b" + choice + r"\b", text, re.IGNORECASE)) != 1
        for choice in ("gas giant", "ice giant", "terrestrial")
    ):
        raise fail("missing_planet_class_choices")
    stellar = {}
    for name, label, unit in [("stellar_mass", "M", "Msun"), ("stellar_radius", "R", "Rsun")]:
        quantity = "MASS" if name == "stellar_mass" else "RADIUS"
        matches = re.findall(
            rf"STAR\s+{quantity}\s*\(\s*{label}s\s*\)\s*({NUMBER})(?=\s|$)", text, re.IGNORECASE
        )
        if len(matches) != 1 or not math.isfinite(float(matches[0])) or float(matches[0]) < 0:
            raise fail("missing_or_invalid_stellar_planet_input")
        stellar[name] = {
            "value": float(matches[0]),
            "display_text": matches[0],
            "unit": unit,
            "source": "visible reconstruction readback; scientific validation separate",
        }
    observation = Observation(
        instruction=f"Inspect {star}'s planet observations and reconstruction. Choose measurements and tools explicitly.",
        controls=[
            Control(
                id=f["capture_target_id"],
                label=name,
                role="textbox",
                actions=[],
                enabled=f["enabled"],
                value=f["current_value"],
            )
            for name, f in fields.items()
        ],
        values={
            "star_name": star,
            "browser_field_map": fields,
            "has_planet": selected,
            "stellar_inputs": stellar,
            "planet_class": None,
            "classification_choices": ["gas_giant", "ice_giant", "terrestrial"],
        },
        chart={
            "sampling_complete": False,
            "measurements_verified": False,
            "reason": "Axis extent and AX tooltip text are not visibility/completion evidence",
        },
        progress={
            "task": "browser_planet_mapping",
            "task_completed": False,
            "submitted": False,
            "mapping_ready": True,
            "policy_ready": False,
        },
    )
    return {
        "schema_version": 1,
        "mode": "offline_planet_field_mapping",
        "capture_sha256": capture_sha256,
        "star_name": star,
        "observation": observation.model_dump(mode="json"),
        "actions_executed": 0,
        "browser_acceptance_passed": False,
    }
