import copy
import hashlib
import json

import pytest
import yaml
from typer.testing import CliRunner

from habfly.browser_habitability import GASES, load_habitability_capture, map_habitability_capture
from habfly.browser_stellar import SIMULATION_URL, StellarMappingError
from habfly.cli import app


def capture(gas="", opened=False, equilibrium="0", greenhouse="", phase=""):
    atoms = [
        {"text": "Jyremis"},
        {"text": "Observations Modeled Albedo 0.05 Modeled Pressure (atm) 9 Transit Spectrum"},
        {"img": "Flux WAVELENGTH (μm) 0 2 4 6"},
        {
            "text": "Modelled Transit Spectrum (8808K) Actual Transit Spectrum Your Reconstruction Equilibrium Temp (K)"
        },
    ]
    controls = []

    def control(atom, role, value=None):
        atoms.append(atom)
        c = {
            "id": f"simulation-0:c{len(controls)}",
            "role": role,
            "enabled": True,
            "accessibility": yaml.safe_dump([atom], allow_unicode=True, sort_keys=False),
        }
        if value is not None:
            c["value"] = value
        controls.append(c)

    def menu(options, selected):
        option = lambda label: "option" + (f' "{label}"' if label else "")
        control(
            {"combobox": [option(v) + (" [selected]" if v == selected else "") for v in options]},
            "combobox",
            selected,
        )

    control({'textbox "0"': equilibrium} if equilibrium != "0" else 'textbox "0"', "textbox", equilibrium)
    atoms.append({"text": "Trace Gases Present"})
    menu([gas], gas)
    if opened:
        for name in GASES:
            control(f'checkbox "{name}"' + (" [checked]" if name == "O3" and gas == "O₃" else ""), "checkbox")
    atoms.append({"text": f"Absorption % {'6.882' if gas else '0.000'}% Greenhouse Effect"})
    menu(["", "Weak (+10)", "Moderate (+30)", "Strong (+100)"], greenhouse)
    controls[-1]["value"] = greenhouse.split(" ", 1)[0]
    atoms.append({"text": f"Surface Temp (K) {'769.4 ' if greenhouse else ''}Water Phase"})
    menu(["", "Solid", "Liquid", "Gas"], phase)
    atoms.append({"text": "Not Habitable Habitable STAR LUMINOSITY (Ls) 20.44 ORBIT RADIUS (AU) 0.5919"})
    return {
        "schema_version": 1,
        "mode": "read_only_browser_preflight",
        "allow_submission": False,
        "actions_executed": 0,
        "ignored_frame_urls": [],
        "frames": [
            {
                "id": "simulation-0",
                "url": SIMULATION_URL,
                "text": "JYREMIS\nOBSERVATIONS",
                "accessibility": yaml.safe_dump(atoms, allow_unicode=True, sort_keys=False),
                "controls": controls,
            }
        ],
    }


@pytest.mark.parametrize("gas,opened", [("", False), ("", True), ("O₃", False), ("O₃", True)])
def test_mapping_no_inferred_gas_absence_or_habitability(gas, opened):
    result = map_habitability_capture(capture(gas, opened), capture_sha256="a" * 64)
    values = result["observation"]["values"]
    assert values["measurements"]["albedo"]["value"] == 0.05
    assert values["equilibrium_temp"]["value"] == "0"
    assert values["selected_gases"] == (["O3"] if gas else [])
    assert values["habitable"] is None and not values["gas_absence_verified"]
    assert values["readouts"]["surface_temp"]["value"] is None
    assert not result["observation"]["progress"]["task_completed"]
    assert result["actions_executed"] == 0


def test_visible_selections_and_readouts_are_not_correct_answers():
    values = map_habitability_capture(
        capture("O₃", True, "759.4", "Weak (+10)", "Gas"), capture_sha256="a" * 64
    )["observation"]["values"]
    assert values["greenhouse"] == "Weak (+10)" and values["water_phase"] == "Gas"
    assert values["readouts"]["surface_temp"]["value"] == 769.4
    assert values["readouts"]["absorption"]["value"] == 6.882


@pytest.mark.parametrize(
    "mutation",
    [
        "unit",
        "star",
        "numeric",
        "native_menu",
        "ids",
        "gas",
        "gas_checked",
        "partial_menu",
        "unknown_frame",
        "extra_numeric",
        "albedo",
        "phase_options",
    ],
)
def test_ambiguity_and_contract_changes_fail_closed(mutation):
    report = capture("O₃", True, "759.4")
    frame = report["frames"][0]
    if mutation == "unit":
        frame["accessibility"] = frame["accessibility"].replace("Pressure (atm)", "Pressure (bar)")
    elif mutation == "star":
        frame["text"] = "OTHERSTAR"
    elif mutation == "numeric":
        frame["controls"][0]["value"] = "99"
    elif mutation == "native_menu":
        frame["controls"][1]["value"] = "CO₂"
    elif mutation == "ids":
        frame["controls"][1]["id"] = frame["controls"][0]["id"]
    elif mutation == "gas":
        frame["accessibility"] = frame["accessibility"].replace("O₃", "Unknown")
        frame["controls"][1]["accessibility"] = frame["controls"][1]["accessibility"].replace("O₃", "Unknown")
        frame["controls"][1]["value"] = "Unknown"
    elif mutation == "gas_checked":
        frame["controls"][8]["accessibility"] = frame["controls"][8]["accessibility"].replace(
            " [checked]", ""
        )
    elif mutation == "partial_menu":
        frame["controls"].pop(2)
    elif mutation == "unknown_frame":
        report["ignored_frame_urls"] = ["https://unknown.invalid/"]
    elif mutation == "extra_numeric":
        frame["controls"].append({**frame["controls"][0], "id": "simulation-0:extra"})
    elif mutation == "albedo":
        frame["accessibility"] = frame["accessibility"].replace("Albedo 0.05", "Albedo 5")
    else:
        for c in frame["controls"]:
            c["accessibility"] = c["accessibility"].replace('option "Gas"', 'option "Vapor"')
        frame["accessibility"] = frame["accessibility"].replace('option "Gas"', 'option "Vapor"')
    with pytest.raises(StellarMappingError):
        map_habitability_capture(report, capture_sha256="a" * 64)


@pytest.mark.parametrize("bad", [None, [], {}, {"frames": [None]}])
def test_malformed_outer_payload(bad):
    with pytest.raises(StellarMappingError):
        map_habitability_capture(bad, capture_sha256="a" * 64)


def test_hash_verified_offline_cli_and_no_action(tmp_path):
    raw = json.dumps(capture()).encode()
    (tmp_path / "observation.json").write_bytes(raw)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"observation_sha256": hashlib.sha256(raw).hexdigest()})
    )
    assert load_habitability_capture(tmp_path)["actions_executed"] == 0
    result = CliRunner().invoke(app, ["browser", "map-habitability", str(tmp_path)])
    assert result.exit_code == 0 and json.loads(result.output)["actions_executed"] == 0
    (tmp_path / "observation.json").write_bytes(raw + b" ")
    with pytest.raises(StellarMappingError, match="capture_hash_mismatch"):
        load_habitability_capture(tmp_path)


def test_same_unit_distractor_measurement_not_repaired():
    original = capture()
    changed = copy.deepcopy(original)
    changed["frames"][0]["accessibility"] = changed["frames"][0]["accessibility"].replace("20.44", "30.44")
    result = map_habitability_capture(changed, capture_sha256="b" * 64)
    assert result["observation"]["values"]["measurements"]["stellar_luminosity"]["value"] == 30.44
