"""Offline interpretation of the observed HabWorlds v1.5.2 stellar screen.

Maps rendered accessibility evidence, not hidden application data. Capture-local
control identities are NOT live locators. No browser, model, or calculator runs
here, and no class or grading answers are supplied on behalf of a policy.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import yaml

from .contracts import Control, Observation
from .knowledge import CalculationResult

SIMULATION_URL = "https://sim.argos.education/habworlds-star-project/1.5.2/index.html"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FIELDS = {
    "distance(ly)": ("distance", "ly"),
    "luminosity(ls)": ("luminosity", "Lsun"),
    "temperature(k)": ("temperature", "K"),
}
COLOR_OPTIONS = ["IR", "Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Violet", "UV"]
CLASSES = ["main_sequence", "red_giant", "supergiant", "white_dwarf"]


class StellarMappingError(ValueError):
    """A missing or ambiguous visible fact must not become a guessed mapping."""


def _atoms(snapshot):
    """Accept only the flat AX shape actually observed; reject unsupported layouts."""
    if not isinstance(snapshot, str) or len(snapshot) > 100000:
        raise StellarMappingError("invalid_accessibility_snapshot")
    # Aliases/tags/complex YAML are unnecessary for Playwright's flat snapshot.
    try:
        nodes = list(yaml.parse(snapshot))
        if any(
            isinstance(node, yaml.events.AliasEvent)
            or getattr(node, "anchor", None)
            or getattr(node, "tag", None)
            for node in nodes
        ):
            raise StellarMappingError("unsupported_accessibility_yaml")
        parsed = yaml.safe_load(snapshot)
    except (yaml.YAMLError, RecursionError):
        raise StellarMappingError("invalid_accessibility_snapshot") from None
    if not isinstance(parsed, list) or len(parsed) > 1000:
        raise StellarMappingError("unsupported_accessibility_layout")
    atoms = []
    for item in parsed:
        if isinstance(item, str):
            atoms.append((item, None))
        elif isinstance(item, dict) and len(item) == 1:
            key, value = next(iter(item.items()))
            if not isinstance(key, str):
                raise StellarMappingError("unsupported_accessibility_layout")
            atoms.append((key, value))
        else:
            raise StellarMappingError("unsupported_accessibility_layout")
    return atoms


def _role(key):
    return key.split(" ", 1)[0]


def _text(atoms):
    result = []
    for key, value in atoms:
        if key in {"text", "subscript"}:
            if not isinstance(value, str):
                raise StellarMappingError("unsupported_accessibility_text")
            result.append(value)
    return " ".join(result)


def _control_atom(snapshot):
    atoms = _atoms(snapshot)
    if len(atoms) != 1:
        raise StellarMappingError("ambiguous_control_snapshot")
    return atoms[0]


def _measurements(text):
    patterns = {
        "parallax": rf'parallax\s*\(\s*(?:"|″|arcsec(?:onds)?)\s*\)\s*({NUMBER})(?=\s|$)',
        "wavelength": rf"peak wavelength\s*\(\s*nm\s*\)\s*({NUMBER})(?=\s|$)",
        "flux": rf"\bflux\s*({NUMBER})(?=\s|$)",
    }
    units = {"parallax": "arcsec", "wavelength": "nm", "flux": "W/m2"}
    result = {}
    for kind, pattern in patterns.items():
        label = "peak wavelength" if kind == "wavelength" else kind
        if len(re.findall(rf"\b{label}\b", text, flags=re.IGNORECASE)) != 1:
            raise StellarMappingError(f"missing_or_ambiguous_measurement:{kind}")
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if len(matches) != 1:
            raise StellarMappingError(f"missing_or_ambiguous_measurement:{kind}")
        value = float(matches[0])
        if not math.isfinite(value) or value <= 0:
            raise StellarMappingError(f"invalid_measurement_domain:{kind}")
        result[f"browser_{kind}"] = {
            "kind": kind,
            "value": value,
            "display_text": matches[0],
            "unit": units[kind],
            "source": "current star",
            "unit_source": "docs/habworlds-preview-inspection.md#observed-lesson-content"
            if kind == "flux"
            else "visible measurement label",
        }
    return result


def map_stellar_capture(report: dict, *, capture_sha256: str, allow_color_selection=False) -> dict:
    if (
        report.get("schema_version") != 1
        or report.get("mode") != "read_only_browser_preflight"
        or report.get("actions_executed") != 0
        or report.get("allow_submission") is not False
        or not re.fullmatch(r"[0-9a-f]{64}", capture_sha256)
    ):
        raise StellarMappingError("invalid_capture_contract")
    matches = [frame for frame in report["frames"] if frame.get("url") == SIMULATION_URL]
    if len(matches) != 1:
        raise StellarMappingError("ambiguous_simulation_frame")
    frame = matches[0]
    atoms = _atoms(frame["accessibility"])
    texts = [value for key, value in atoms if key == "text"]
    if (
        not texts
        or not isinstance(texts[0], str)
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", texts[0])
    ):
        raise StellarMappingError("missing_star_identity")
    star = texts[0]
    visible_heading = frame["text"].splitlines()[0].strip()
    if visible_heading.casefold() != star.casefold():
        raise StellarMappingError("conflicting_star_identity")
    all_text = _text(atoms)
    sections = re.split(r"\byour reconstruction\b", all_text, flags=re.IGNORECASE)
    if len(sections) != 2 or not re.search(r"\bobservations\b", sections[0], re.IGNORECASE):
        raise StellarMappingError("not_stellar_detail_screen")
    measurements = _measurements(sections[0])
    numeric_controls = [c for c in frame["controls"] if c["role"] in {"textbox", "spinbutton"}]
    if len(numeric_controls) != 3:
        raise StellarMappingError("unsupported_conditional_or_ambiguous_fields")
    color_controls = [c for c in frame["controls"] if c["role"] == "combobox"]
    if len(color_controls) != 1:
        raise StellarMappingError("ambiguous_color_control")
    ids = [c["id"] for c in frame["controls"]]
    if len(ids) != len(set(ids)):
        raise StellarMappingError("duplicate_control_id")
    if any(not ident.startswith(frame["id"] + ":") for ident in ids):
        raise StellarMappingError("control_frame_mismatch")
    context, fields, colors, position = [], {}, None, 0
    for key, value in atoms:
        role = _role(key)
        if key in {"text", "subscript"}:
            context.append(value)
            continue
        if role not in {"textbox", "spinbutton", "combobox"}:
            context = []
            continue
        preceding = re.split(r"\byour reconstruction\b", " ".join(context), flags=re.IGNORECASE)[-1]
        label = re.sub(r"\s+", "", preceding.casefold())
        if role == "combobox":
            if label not in {"peakλcolor", "peakwavelengthcolor"} or colors is not None:
                raise StellarMappingError("unmapped_color_label")
            expected_options = [f'option "{name}"' for name in COLOR_OPTIONS]
            selected = []
            normalized_options = []
            if isinstance(value, list):
                for option in value:
                    if not isinstance(option, str):
                        raise StellarMappingError("unsupported_color_options_or_selection")
                    if option.endswith(" [selected]"):
                        selected.append(option.removesuffix(" [selected]"))
                    normalized_options.append(option.removesuffix(" [selected]"))
            if (
                normalized_options != expected_options
                or len(selected) > 1
                or (selected and not allow_color_selection)
            ):
                raise StellarMappingError("unsupported_color_options_or_selection")
            control = color_controls[0]
            if _control_atom(control["accessibility"]) != (key, value):
                raise StellarMappingError("control_snapshot_mismatch")
            colors = {
                "capture_target_id": control["id"],
                "options": list(COLOR_OPTIONS),
                "selected": COLOR_OPTIONS[expected_options.index(selected[0])] if selected else None,
            }
        else:
            if label not in FIELDS or position >= len(numeric_controls):
                raise StellarMappingError("unmapped_numeric_label")
            name, unit = FIELDS[label]
            if name in fields:
                raise StellarMappingError("duplicate_numeric_field")
            control = numeric_controls[position]
            if control["role"] != role or _control_atom(control["accessibility"]) != (key, value):
                raise StellarMappingError("control_snapshot_mismatch")
            # A textbox's accessible name "0" is not evidence of its value.
            fields[name] = {
                "capture_target_id": control["id"],
                "unit": unit,
                "label_evidence": preceding.strip(),
                "enabled": control["enabled"],
                "current_value": control.get("value"),
                "value_known": control.get("value") is not None,
                "binding_status": "capture_order_and_visible_label_not_live_verified",
            }
            position += 1
        context = []
    if set(fields) != {"distance", "luminosity", "temperature"} or colors is None:
        raise StellarMappingError("incomplete_field_map")
    # The four class words are choices, not a selected class. Do not invent one.
    for label in ("main sequence", "red giant", "supergiant", "white dwarf"):
        if label not in sections[1].casefold():
            raise StellarMappingError("missing_classification_choices")
    observation = Observation(
        instruction=f"Inspect {star}'s visible stellar fields. Classification is not supplied.",
        controls=[
            Control(
                id=f["capture_target_id"],
                label=name,
                role="textbox",
                actions=[],
                enabled=f["enabled"],
                value=f["current_value"] or "",
            )
            for name, f in fields.items()
        ],
        values={
            "star_name": star,
            "star_class": None,
            "measurements": measurements,
            "visible_numeric_fields": list(fields),
            "classification_choices": CLASSES,
            "browser_field_map": fields,
            "color": colors,
        },
        progress={
            "task": "browser_stellar_mapping",
            "task_completed": False,
            "submitted": False,
            "policy_ready": False,
            "mapping_ready": True,
        },
    )
    return {
        "schema_version": 1,
        "mode": "offline_stellar_field_mapping",
        "capture_sha256": capture_sha256,
        "setup_mode": report.get("setup_mode", "unspecified_legacy_capture"),
        "simulation_frame": frame["id"],
        "star_name": star,
        "observation": observation.model_dump(mode="json"),
        "actions_executed": 0,
        "browser_acceptance_passed": False,
        "pending_gates": [
            "live field identity and value readback",
            "color selection",
            "four-way classification",
            "conditional main-sequence fields and lifetime prefixes",
            "completion and persistence",
        ],
    }


def plan_numeric_copy(mapping, result: CalculationResult, destination: str, *, capture_sha256: str):
    """Check a caller-chosen result/destination; never choose or repair them.

    Returns a NON-EXECUTABLE intent, not an Action or live selector. Native field
    formatting and numeric readback still need verification before any write.
    """
    if mapping["capture_sha256"] != capture_sha256:
        raise StellarMappingError("stale_capture")
    fields = mapping["observation"]["values"]["browser_field_map"]
    if destination not in fields:
        raise StellarMappingError("unmapped_destination")
    field = fields[destination]
    if not field["enabled"]:
        raise StellarMappingError("disabled_destination")
    if not result.ok or result.value is None or not math.isfinite(result.value) or result.value <= 0:
        raise StellarMappingError("invalid_tool_result")
    if result.unit != field["unit"]:
        raise StellarMappingError("incompatible_destination_unit")
    return {
        "mode": "offline_copy_intent",
        "executable": False,
        "capture_sha256": capture_sha256,
        "capture_target_id": field["capture_target_id"],
        "destination": destination,
        "operation_id": result.operation_id,
        "value": repr(result.value),
        "unit": result.unit,
    }


def load_and_map_capture(directory: Path):
    raw = (directory / "observation.json").read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    manifest = json.loads((directory / "manifest.json").read_text())
    if checksum != manifest.get("observation_sha256"):
        raise StellarMappingError("capture_hash_mismatch")
    return map_stellar_capture(json.loads(raw), capture_sha256=checksum)


def save_mapping(mapping, directory: Path):
    encoded = (json.dumps(mapping, indent=2, ensure_ascii=False) + "\n").encode()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "mapping.json").write_bytes(encoded)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "offline_stellar_field_mapping",
                "capture_sha256": mapping["capture_sha256"],
                "mapping_sha256": hashlib.sha256(encoded).hexdigest(),
                "actions_executed": 0,
                "browser_acceptance_passed": False,
            },
            indent=2,
        )
        + "\n"
    )
