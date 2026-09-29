"""Read-only mapping of public terrestrial observations and reconstruction.

The spectrum's appearance is not interpreted here. Default Not Habitable paint
is not a policy answer, a blank gas menu is not evidence of gas absence, and
calculated readouts are not claims of course correctness.
"""

import hashlib
import json
import math
import re
from pathlib import Path

from .browser_stellar import NUMBER, SIMULATION_URL, StellarMappingError, _atoms, _control_atom
from .contracts import Control, Observation

GASES = ("CH4", "CO2", "H2O", "H2S", "N2O", "NH3", "O3")
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _frame(report, checksum):
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("mode") != "read_only_browser_preflight"
        or report.get("allow_submission") is not False
        or report.get("actions_executed") != 0
        or report.get("ignored_frame_urls")
        or not isinstance(checksum, str)
        or not re.fullmatch(r"[a-f0-9]{64}", checksum)
        or not isinstance(report.get("frames"), list)
        or any(not isinstance(f, dict) for f in report["frames"])
    ):
        raise StellarMappingError("invalid_habitability_capture")
    frames = [f for f in report["frames"] if f.get("url") == SIMULATION_URL]
    if len(frames) != 1:
        raise StellarMappingError("ambiguous_simulation_frame")
    f = frames[0]
    if (
        any(not isinstance(f.get(k), str) for k in ("text", "accessibility", "id"))
        or not isinstance(f.get("controls"), list)
        or any(
            not isinstance(c, dict)
            or any(not isinstance(c.get(k), str) for k in ("id", "role", "accessibility"))
            or type(c.get("enabled")) is not bool
            for c in f["controls"]
        )
    ):
        raise StellarMappingError("invalid_habitability_frame")
    ids = [c["id"] for c in f["controls"]]
    if len(set(ids)) != len(ids) or any(not i.startswith(f["id"] + ":") for i in ids):
        raise StellarMappingError("invalid_habitability_control_ids")
    return f


def load_habitability_capture(directory):
    raw = (Path(directory) / "observation.json").read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest.get("observation_sha256") != checksum:
        raise StellarMappingError("capture_hash_mismatch")
    return map_habitability_capture(json.loads(raw), capture_sha256=checksum)


def _menu(control, *, options=None, native_labels=None):
    key, values = _control_atom(control["accessibility"])
    if key != "combobox" or not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise StellarMappingError("unsupported_habitability_menu")
    normalized = [v.removesuffix(" [selected]") for v in values]
    selected = [v.removesuffix(" [selected]") for v in values if v.endswith(" [selected]")]
    if options is not None and normalized != ["option", *[f'option "{v}"' for v in options]]:
        raise StellarMappingError("changed_habitability_menu_options")
    if len(selected) != 1 or not re.fullmatch(r'option(?: "[^"]*")?', selected[0]):
        raise StellarMappingError("ambiguous_habitability_menu_selection")
    label = "" if selected[0] == "option" else selected[0][8:-1]
    expected_native = native_labels.get(label, label) if native_labels else label
    if control.get("value") != expected_native:
        raise StellarMappingError("habitability_menu_readback_disagrees")
    return label


def map_habitability_capture(report, *, capture_sha256):
    f = _frame(report, capture_sha256)
    atoms = _atoms(f["accessibility"])
    texts = [v for k, v in atoms if k == "text"]
    if not texts or not all(isinstance(v, str) for v in texts):
        raise StellarMappingError("invalid_habitability_text")
    star = texts[0]
    if (
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", star)
        or not f["text"].splitlines()
        or f["text"].splitlines()[0].casefold() != star.casefold()
    ):
        raise StellarMappingError("conflicting_habitability_star_identity")
    text = " ".join(texts)
    for label in (
        "Observations",
        "Modeled Albedo",
        "Modeled Pressure (atm)",
        "Your Reconstruction",
        "Equilibrium Temp (K)",
        "Trace Gases Present",
        "Greenhouse Effect",
        "Surface Temp (K)",
        "Water Phase",
    ):
        if text.casefold().count(label.casefold()) != 1:
            raise StellarMappingError("not_supported_habitability_screen")
    measurements = {}
    for name, label, unit in (
        ("albedo", "Modeled Albedo", "fraction"),
        ("pressure", "Modeled Pressure (atm)", "atm"),
        ("stellar_luminosity", "STAR LUMINOSITY (Ls)", "Lsun"),
        ("orbital_radius", "ORBIT RADIUS (AU)", "au"),
    ):
        matches = re.findall(re.escape(label) + rf"\s+({NUMBER})(?=\s|$)", text, re.IGNORECASE)
        if len(matches) != 1 or not math.isfinite(float(matches[0])) or float(matches[0]) < 0:
            raise StellarMappingError("invalid_habitability_measurement")
        if name == "albedo" and float(matches[0]) > 1:
            raise StellarMappingError("invalid_habitability_albedo")
        measurements[name] = {
            "value": float(matches[0]),
            "display_text": matches[0],
            "unit": unit,
            "source": "visible supplied/modelled observation; not grading reference",
        }
    numeric = [c for c in f["controls"] if c["role"] in {"textbox", "spinbutton"}]
    combos = [c for c in f["controls"] if c["role"] == "combobox"]
    if len(numeric) != 1 or len(combos) != 3:
        raise StellarMappingError("unsupported_habitability_fields")
    control = numeric[0]
    atom = _control_atom(control["accessibility"])
    matching = [i for i, item in enumerate(atoms) if item == atom]
    if (
        len(matching) != 1
        or matching[0] == 0
        or atoms[matching[0] - 1][0] != "text"
        or not atoms[matching[0] - 1][1].endswith("Equilibrium Temp (K)")
        or not isinstance(control.get("value"), str)
        or atom[1] is not None
        and str(atom[1]) != control["value"]
    ):
        raise StellarMappingError("ambiguous_equilibrium_field")
    # Identify the menus from their observed text adjacency, not index alone.
    for c, label in zip(combos, ("Trace Gases Present", "Greenhouse Effect", "Water Phase"), strict=True):
        item = _control_atom(c["accessibility"])
        indices = [i for i, a in enumerate(atoms) if a == item]
        if not any(i > 0 and atoms[i - 1][0] == "text" and atoms[i - 1][1].endswith(label) for i in indices):
            raise StellarMappingError("ambiguous_habitability_menu_label")
    gas_label = _menu(combos[0])
    gas_names = [v.strip().translate(SUBSCRIPTS) for v in gas_label.split(",") if v.strip()]
    if len(set(gas_names)) != len(gas_names) or not set(gas_names) <= set(GASES):
        raise StellarMappingError("unsupported_trace_gas_label")
    checkboxes = [c for c in f["controls"] if c["role"] == "checkbox"]
    if checkboxes:
        checked, names = [], []
        for c in checkboxes:
            atom = _control_atom(c["accessibility"])
            match = re.fullmatch(r'checkbox "([A-Z0-9]+)"( \[checked\])?', atom[0])
            if not match or atom not in atoms or atom[1] is not None:
                raise StellarMappingError("unsupported_gas_checkbox")
            names.append(match[1])
            if match[2]:
                checked.append(match[1])
        if set(names) != set(GASES) or len(names) != 7 or set(checked) != set(gas_names):
            raise StellarMappingError("gas_selection_readback_disagrees")
    greenhouse = _menu(
        combos[1],
        options=["Weak (+10)", "Moderate (+30)", "Strong (+100)"],
        native_labels={"Weak (+10)": "Weak", "Moderate (+30)": "Moderate", "Strong (+100)": "Strong"},
    )
    phase = _menu(combos[2], options=["Solid", "Liquid", "Gas"])
    readouts = {}
    for key, pattern, unit in [
        ("absorption", rf"Absorption % ({NUMBER})% Greenhouse Effect", "%"),
        ("surface_temp", rf"Surface Temp \(K\)(?: ({NUMBER}))? Water Phase", "K"),
    ]:
        matches = re.findall(pattern, text)
        if len(matches) != 1:
            raise StellarMappingError("ambiguous_habitability_readout")
        value = matches[0]
        if value and (
            not math.isfinite(float(value)) or float(value) < 0 or key == "absorption" and float(value) > 100
        ):
            raise StellarMappingError("invalid_habitability_readout")
        readouts[key] = {
            "value": float(value) if value else None,
            "display_text": value or None,
            "unit": unit,
            "source": "visible reconstruction result, not a correct answer",
        }
    obs = Observation(
        instruction=f"Inspect {star}'s terrestrial spectrum and temperature reconstruction. Choose evidence and tools explicitly.",
        controls=[
            Control(
                id=c["id"],
                label=label,
                role=c["role"],
                actions=[],
                enabled=c["enabled"],
                value=c.get("value"),
            )
            for c, label in zip(
                [control, *combos],
                ("equilibrium_temp", "trace_gases", "greenhouse", "water_phase"),
                strict=True,
            )
        ],
        values={
            "star_name": star,
            "measurements": measurements,
            "equilibrium_temp": {"value": control["value"], "unit": "K", "capture_target_id": control["id"]},
            "selected_gas_label": gas_label,
            "selected_gases": gas_names,
            "gas_absence_verified": False,
            "greenhouse": greenhouse or None,
            "water_phase": phase or None,
            "readouts": readouts,
            "habitable": None,
            "habitability_choice_source": "painted_choice_not_read_by_offline_mapper",
        },
        chart={"spectrum_interpreted": False, "gas_identification_verified": False},
        progress={
            "task": "browser_habitability_mapping",
            "mapping_ready": True,
            "policy_ready": False,
            "task_completed": False,
            "submitted": False,
        },
    )
    return {
        "schema_version": 1,
        "mode": "offline_habitability_field_mapping",
        "capture_sha256": capture_sha256,
        "star_name": star,
        "observation": obs.model_dump(mode="json"),
        "actions_executed": 0,
        "browser_acceptance_passed": False,
    }
