"""Student-visible capture mapping, offline; no policy answers or live selectors."""

import copy
import hashlib
import json
import socket

import pytest
from typer.testing import CliRunner

from habfly.browser_probe import save_probe
from habfly.browser_stellar import (
    SIMULATION_URL,
    StellarMappingError,
    load_and_map_capture,
    map_stellar_capture,
    plan_numeric_copy,
)
from habfly.cli import app
from habfly.contracts import Action, Observation
from habfly.knowledge import CalculationResult, LocalCalculator

HASH = "a" * 64
# Minimal transcription of student-visible AX from the 2026-09-24 Chromium capture.
# No raw HTML, hidden state, URL queries, auth data, or grading references.
SNAPSHOT = """- img
- text: Althinagon
- text: Observations parallax (") 0.045 peak wavelength (nm) 370 Flux 5.68E-10 Spectrum metallicity 0.21 Your Reconstruction distance (ly)
- textbox "0"
- text: luminosity (L
- subscript: s
- text: )
- textbox "0"
- text: temperature (K)
- textbox "0"
- text: peak λ color
- combobox:
  - option "IR"
  - option "Red"
  - option "Orange"
  - option "Yellow"
  - option "Green"
  - option "Cyan"
  - option "Blue"
  - option "Violet"
  - option "UV"
- text: mass, radius and lifetime are only relevant for main sequence stars main sequence red giant supergiant white dwarf 1 Rs
- button "Save"
"""


def capture():
    colors = SNAPSHOT[SNAPSHOT.index("- combobox:") : SNAPSHOT.index("- text: mass,")].rstrip()
    controls = [
        {
            "id": f"simulation-0:c{i + 3}",
            "role": "textbox",
            "enabled": True,
            "accessibility": '- textbox "0"',
            "actions": [],
            "protected": False,
        }
        for i in range(3)
    ]
    controls.append(
        {
            "id": "simulation-0:c6",
            "role": "combobox",
            "enabled": True,
            "accessibility": colors,
            "actions": [],
            "protected": False,
        }
    )
    return {
        "schema_version": 1,
        "mode": "read_only_browser_preflight",
        "actions_executed": 0,
        "allow_submission": False,
        "frames": [
            {
                "id": "simulation-0",
                "url": SIMULATION_URL,
                "text": "ALTHINAGON\nOBSERVATIONS",
                "accessibility": SNAPSHOT,
                "controls": controls,
            }
        ],
    }


def test_maps_real_capture_shape_without_treating_placeholder_as_value():
    report = capture()
    original = copy.deepcopy(report)
    mapped = map_stellar_capture(report, capture_sha256=HASH)
    observation = Observation.model_validate(mapped["observation"])
    values = observation.values
    assert values["star_name"] == "Althinagon"
    assert values["star_class"] is None
    assert values["measurements"]["browser_parallax"]["value"] == 0.045
    assert values["measurements"]["browser_flux"]["value"] == 5.68e-10
    assert values["measurements"]["browser_flux"]["display_text"] == "5.68E-10"
    assert values["measurements"]["browser_wavelength"]["unit"] == "nm"
    for name, ident in (("distance", "c3"), ("luminosity", "c4"), ("temperature", "c5")):
        field = values["browser_field_map"][name]
        assert field["capture_target_id"] == f"simulation-0:{ident}"
        assert field["current_value"] is None and not field["value_known"]
    assert not observation.progress["policy_ready"]
    assert not observation.progress["task_completed"]
    assert all(c.actions == [] for c in observation.controls)
    assert "expected" not in values and "answers" not in values
    assert report == original


@pytest.mark.parametrize("value", ["", "0", "1.234e-9"])
def test_explicit_visible_input_values_are_not_accessible_names(value):
    report = capture()
    report["frames"][0]["controls"][0]["value"] = value
    mapped = map_stellar_capture(report, capture_sha256=HASH)
    field = mapped["observation"]["values"]["browser_field_map"]["distance"]
    assert field["value_known"] and field["current_value"] == value


@pytest.mark.parametrize("replacement", ["0", "-1", "1E999", "NaN", "", "1e+bad"])
def test_missing_zero_negative_or_malformed_measurements_reject(replacement):
    report = capture()
    report["frames"][0]["accessibility"] = SNAPSHOT.replace("0.045", replacement)
    with pytest.raises(StellarMappingError):
        map_stellar_capture(report, capture_sha256=HASH)


@pytest.mark.parametrize(
    "before,after",
    [
        ('parallax (")', "parallax (degrees)"),
        ("wavelength (nm)", "wavelength (m)"),
        ("distance (ly)", "distance (pc)"),
        ("temperature (K)", "temperature (C)"),
        ("luminosity (L", "distance (L"),
        ("Your Reconstruction", "Planet Reconstruction"),
        ("Althinagon", "Another Star"),
        ("Flux 5.68E-10", "Flux 5.68E-10 Flux 2E-10"),
    ],
)
def test_ambiguous_identity_units_or_labels_fail_closed(before, after):
    report = capture()
    report["frames"][0]["accessibility"] = SNAPSHOT.replace(before, after)
    with pytest.raises(StellarMappingError):
        map_stellar_capture(report, capture_sha256=HASH)


def test_field_association_follows_visible_labels_not_fixed_field_order():
    report = capture()
    report["frames"][0]["accessibility"] = (
        SNAPSHOT.replace("distance (ly)", "TEMP")
        .replace("temperature (K)", "distance (ly)")
        .replace("TEMP", "temperature (K)")
    )
    mapped = map_stellar_capture(report, capture_sha256=HASH)
    fields = mapped["observation"]["values"]["browser_field_map"]
    assert fields["temperature"]["capture_target_id"] == "simulation-0:c3"
    assert fields["distance"]["capture_target_id"] == "simulation-0:c5"


def test_extra_fields_frames_and_duplicate_ids_fail_closed():
    for mutation in ("frame", "field", "id"):
        report = capture()
        frame = report["frames"][0]
        if mutation == "frame":
            report["frames"].append(copy.deepcopy(frame))
        elif mutation == "field":
            frame["controls"].append(copy.deepcopy(frame["controls"][0]))
        else:
            frame["controls"][1]["id"] = frame["controls"][0]["id"]
        with pytest.raises(StellarMappingError):
            map_stellar_capture(report, capture_sha256=HASH)


def test_control_snapshots_must_agree_and_belong_to_the_simulation():
    report = capture()
    report["frames"][0]["controls"][0]["id"] = "score_widgets-0:c0"
    with pytest.raises(StellarMappingError, match="control_frame_mismatch"):
        map_stellar_capture(report, capture_sha256=HASH)
    report = capture()
    report["frames"][0]["controls"][0]["accessibility"] = '- textbox "0": 42'
    with pytest.raises(StellarMappingError, match="control_snapshot_mismatch"):
        map_stellar_capture(report, capture_sha256=HASH)


@pytest.mark.parametrize(
    "snapshot",
    ["- text: !!python/object:unsafe {}", "- &node text\n- *node", "- group:\n  - text: Althinagon"],
)
def test_unsupported_or_executable_yaml_is_rejected(snapshot):
    report = capture()
    report["frames"][0]["accessibility"] = snapshot
    with pytest.raises(StellarMappingError):
        map_stellar_capture(report, capture_sha256=HASH)


def test_caller_selected_numeric_copy_is_exact_but_not_executable():
    mapped = map_stellar_capture(capture(), capture_sha256=HASH)
    # Deliberately NOT an expert answer. Wrong numeric results must not be repaired.
    chosen = CalculationResult(ok=True, operation_id="distance", value=1.2345678901234567, unit="ly")
    intent = plan_numeric_copy(mapped, chosen, "distance", capture_sha256=HASH)
    assert float(intent["value"]) == chosen.value
    assert not intent["executable"]
    with pytest.raises(ValueError):
        Action.model_validate(intent)
    with pytest.raises(StellarMappingError, match="incompatible_destination_unit"):
        plan_numeric_copy(mapped, chosen, "temperature", capture_sha256=HASH)
    with pytest.raises(StellarMappingError, match="unmapped_destination"):
        plan_numeric_copy(mapped, chosen, "mass", capture_sha256=HASH)
    with pytest.raises(StellarMappingError, match="stale_capture"):
        plan_numeric_copy(mapped, chosen, "distance", capture_sha256="b" * 64)


def test_invalid_tool_result_and_disabled_destination_are_rejected():
    mapped = map_stellar_capture(capture(), capture_sha256=HASH)
    bad = CalculationResult(ok=False, operation_id="distance", error="missing_input")
    with pytest.raises(StellarMappingError, match="invalid_tool_result"):
        plan_numeric_copy(mapped, bad, "distance", capture_sha256=HASH)
    mapped["observation"]["values"]["browser_field_map"]["distance"]["enabled"] = False
    with pytest.raises(StellarMappingError, match="disabled_destination"):
        plan_numeric_copy(mapped, bad, "distance", capture_sha256=HASH)


def test_mapping_cli_is_offline_no_calculator_and_preserves_original(tmp_path, monkeypatch):
    source, output = tmp_path / "capture", tmp_path / "mapping"
    save_probe(capture(), source)
    before = (source / "observation.json").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Mapping must not use network or compute answers")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(LocalCalculator, "execute", forbidden)
    result = CliRunner().invoke(app, ["browser", "map-stellar", str(source), str(output)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["capture_sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["mapping_sha256"] == hashlib.sha256((output / "mapping.json").read_bytes()).hexdigest()
    assert (source / "observation.json").read_bytes() == before
    assert CliRunner().invoke(app, ["browser", "map-stellar", str(source), str(output)]).exit_code != 0
    (source / "observation.json").write_bytes(before + b" ")
    with pytest.raises(StellarMappingError, match="capture_hash_mismatch"):
        load_and_map_capture(source)
