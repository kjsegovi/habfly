import copy
import hashlib
import json

import pytest
import yaml

from habfly.browser_planet import load_planet_capture, map_planet_capture
from habfly.browser_stellar import SIMULATION_URL, StellarMappingError


def capture(selected=None):
    atoms = [
        {"text": "Jyremis"},
        {"text": "Observations Spectrum 656.3nm 656.299997nm 656.300003nm observe for"},
    ]
    controls = []

    def field(label, value=""):
        if label:
            atoms.append({"text": label})
        atom = {'textbox "0"': value} if value else 'textbox "0"'
        atoms.append(atom)
        controls.append(
            {
                "id": f"simulation-0:c{len(controls)}",
                "role": "textbox",
                "accessibility": yaml.safe_dump([atom]),
                "enabled": True,
                "value": value,
            }
        )

    field(None, "10000")
    atoms.extend([{"text": "days"}, 'button "Play"'])
    field("doppler shift (nm)")
    field("brightness drop (%)")
    field("brightness drop period (days)")
    atoms.append({"text": "Your Reconstruction has planet?"})
    options = [f'option "{v}"' + (" [selected]" if v == selected else "") for v in ("Yes", "No")]
    combo = {"combobox": options}
    atoms.append(combo)
    controls.append(
        {
            "id": f"simulation-0:c{len(controls)}",
            "role": "combobox",
            "accessibility": yaml.safe_dump([combo]),
            "enabled": True,
            "value": selected or "",
        }
    )
    if selected == "Yes":
        field("orbital radius (au)")
        atoms.extend([{"text": "mass (M"}, {"subscript": "E"}, {"text": ")"}])
        field(None)
        atoms.extend([{"text": "radius (R"}, {"subscript": "E"}, {"text": ")"}])
        field(None)
        atoms.extend([{"text": "density (g/cm"}, {"superscript": "3"}, {"text": ")"}])
        field(None)
    atoms.append(
        {
            "text": "gas giant ice giant terrestrial STAR MASS (Ms) 2.368 STAR RADIUS (Rs) 1.961 ORBIT (years) 0.000"
        }
    )
    return {
        "schema_version": 1,
        "mode": "read_only_browser_preflight",
        "actions_executed": 0,
        "allow_submission": False,
        "ignored_frame_urls": [],
        "frames": [
            {
                "id": "simulation-0",
                "url": SIMULATION_URL,
                "text": "JYREMIS\nOBSERVATIONS",
                "accessibility": yaml.safe_dump(atoms, sort_keys=False),
                "controls": controls,
            }
        ],
    }


@pytest.mark.parametrize("selected", [None, "Yes", "No"])
def test_public_fields_and_no_inferred_chart_or_class(selected):
    report = map_planet_capture(capture(selected), capture_sha256="a" * 64)
    obs = report["observation"]
    assert len(obs["values"]["browser_field_map"]) == (8 if selected == "Yes" else 4)
    assert obs["values"]["has_planet"] == selected
    assert obs["values"]["planet_class"] is None
    assert obs["values"]["stellar_inputs"]["stellar_mass"]["value"] == 2.368
    assert not obs["chart"]["measurements_verified"]
    assert not obs["progress"]["task_completed"]
    assert all(not c["actions"] for c in obs["controls"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(frames=None),
        lambda r: r.update(frames=[None]),
        lambda r: r["frames"][0].pop("text"),
        lambda r: r["frames"][0].update(accessibility=None),
        lambda r: r["frames"][0].update(id=4),
        lambda r: r["frames"][0].update(controls=None),
        lambda r: r["frames"][0].update(controls=[None]),
        lambda r: r["frames"][0]["controls"][0].pop("id"),
        lambda r: r["frames"][0]["controls"][0].update(id=[]),
        lambda r: r["frames"][0]["controls"][0].update(enabled="true"),
    ],
)
def test_malformed_capture_is_a_structured_mapping_failure(mutation):
    report = capture()
    mutation(report)
    with pytest.raises(StellarMappingError, match="invalid_planet_"):
        map_planet_capture(report, capture_sha256="a" * 64)


@pytest.mark.parametrize(
    "mutation",
    [
        "identity",
        "duplicate_id",
        "frame_id",
        "duplicate_frame",
        "unknown_frame",
        "hidden_values",
        "selection",
        "unit",
        "ambiguous_class",
        "incomplete",
        "native_disagrees",
    ],
)
def test_malformed_mapping_rejected(mutation):
    r = capture("Yes")
    f = r["frames"][0]
    if mutation == "identity":
        f["text"] = "OTHERSTAR"
    elif mutation == "duplicate_id":
        f["controls"][1]["id"] = f["controls"][0]["id"]
    elif mutation == "frame_id":
        f["controls"][0]["id"] = "other:c0"
    elif mutation == "duplicate_frame":
        r["frames"].append(copy.deepcopy(f))
    elif mutation == "unknown_frame":
        r["ignored_frame_urls"] = ["https://unknown.invalid/"]
    elif mutation == "hidden_values":
        f["controls"][1]["value"] = None
    elif mutation == "selection":
        next(c for c in f["controls"] if c["role"] == "combobox")["value"] = "No"
    elif mutation == "unit":
        f["accessibility"] = f["accessibility"].replace("orbital radius (au)", "orbital radius (km)")
    elif mutation == "ambiguous_class":
        f["accessibility"] = f["accessibility"].replace("gas giant", "gas giant gas giant")
    elif mutation == "native_disagrees":
        f["controls"][0]["value"] = "5"
    else:
        f["controls"].pop()
    with pytest.raises(StellarMappingError):
        map_planet_capture(r, capture_sha256="a" * 64)


@pytest.mark.parametrize("report", [None, {}, {"schema_version": 2}])
def test_bad_capture_contract(report):
    with pytest.raises(StellarMappingError):
        map_planet_capture(report, capture_sha256="a" * 64)


def test_hash_verified_loader_and_cli(tmp_path):
    from typer.testing import CliRunner

    from habfly.cli import app

    raw = json.dumps(capture("Yes")).encode()
    (tmp_path / "observation.json").write_bytes(raw)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"observation_sha256": hashlib.sha256(raw).hexdigest()})
    )
    assert load_planet_capture(tmp_path)["star_name"] == "Jyremis"
    result = CliRunner().invoke(app, ["browser", "map-planet", str(tmp_path)])
    assert result.exit_code == 0
    assert json.loads(result.output)["actions_executed"] == 0
    (tmp_path / "observation.json").write_bytes(raw + b" ")
    with pytest.raises(StellarMappingError, match="capture_hash_mismatch"):
        load_planet_capture(tmp_path)


def test_native_zero_stays_distinct_from_missing():
    report = capture("Yes")
    # Native input values are authoritative for blank-vs-zero. The accessible
    # placeholder alone is not a measurement.
    report["frames"][0]["controls"][1]["value"] = "0"
    mapped = map_planet_capture(report, capture_sha256="a" * 64)
    assert mapped["observation"]["values"]["browser_field_map"]["line_shift"]["current_value"] == "0"


def test_transient_spectrum_tooltip_is_not_duration_label():
    before = capture("Yes")
    after = copy.deepcopy(before)
    after["frames"][0]["accessibility"] = after["frames"][0]["accessibility"].replace(
        "656.3nm", "656.300000629nm 656.3nm", 1
    )
    a = map_planet_capture(before, capture_sha256="a" * 64)
    b = map_planet_capture(after, capture_sha256="b" * 64)
    assert a["observation"]["values"] == b["observation"]["values"]
    after["frames"][0]["accessibility"] = after["frames"][0]["accessibility"].replace(
        "text: days", "text: years"
    )
    with pytest.raises(StellarMappingError, match="duration_unit"):
        map_planet_capture(after, capture_sha256="b" * 64)
