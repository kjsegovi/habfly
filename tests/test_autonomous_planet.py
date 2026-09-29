"""Offline synthetic public receipts; no browser, grading labels or live claims."""

import hashlib
import io
import json
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw
from test_browser_habitability import capture as habitat_capture
from test_browser_planet import capture as planet_capture

import habfly.autonomous_planet as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_habitability import GASES
from habfly.browser_habitability_save import _mapping
from habfly.browser_planet import map_planet_capture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def capture(directory, value):
    sha = write(directory / "observation.json", value)
    write(directory / "manifest.json", {"observation_sha256": sha})
    return sha


def populated_planet(mass="1", radius="1", density="5.514"):
    report = planet_capture("Yes")
    frame = report["frames"][0]
    mapping = map_planet_capture(report, capture_sha256="a" * 64)
    fields = mapping["observation"]["values"]["browser_field_map"]
    desired = {"planet_mass": mass, "planet_radius": radius, "planet_density": density}
    values = {fields[name]["capture_target_id"]: value for name, value in desired.items()}
    atoms = yaml.safe_load(frame["accessibility"])
    controls = iter(frame["controls"])
    for index, atom in enumerate(atoms):
        key = atom if isinstance(atom, str) else next(iter(atom))
        if key.startswith("textbox") or key == "combobox":
            control = next(controls)
            if control["id"] in values:
                replacement = {key: values[control["id"]]}
                atoms[index] = replacement
                control["value"] = values[control["id"]]
                control["accessibility"] = yaml.safe_dump([replacement])
    frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)
    return report


@pytest.mark.parametrize("prototype", module.REFERENCE_PACK["prototypes"])
def test_exact_prototypes_return_declared_class_not_course_truth(tmp_path, prototype):
    _, kind, mass, diameter, density = prototype
    report = populated_planet(
        str(Decimal(mass) / Decimal("5.97")), str(Decimal(diameter) / 12756), str(Decimal(density) / 1000)
    )
    directory = tmp_path / "capture"
    sha = capture(directory, report)
    decision = module.decide_planet_class(tmp_path, directory, sha, expected_star="JYREMIS")
    assert decision["payload"]["name"] == kind
    assert decision["payload"]["rationale"].startswith("Approximate NASA")
    assert decision == module.decide_planet_class(tmp_path, directory, sha, expected_star="JYREMIS")
    assert all(
        not decision[k]
        for k in (
            "learned",
            "scientific_verified",
            "correctness_verified",
            "course_thresholds_verified",
            "training_label",
            "confidence_calibrated",
            "task_completed",
            "project_completed",
        )
    )
    assert all(not Path(path).is_absolute() for path in decision["source_sha256"])
    assert decision["reference_pack_sha256"] == module._sha(module._canonical(module.reference_pack()))
    assert "lifetime" not in decision["evidence"]["measurements"]


@pytest.mark.parametrize("value", ["", "0", "-1", "nan", "Infinity", "1e999", "abc", "1" * 65])
def test_planet_invalid_numbers_never_get_repaired(tmp_path, value):
    directory = tmp_path / "capture"
    sha = capture(directory, populated_planet(mass=value))
    with pytest.raises((BrowserSafetyStop, ValueError)):
        module.decide_planet_class(tmp_path, directory, sha, expected_star="Jyremis")


@pytest.mark.parametrize("failure", ["hash", "star", "manifest", "no", "stopped"])
def test_planet_source_binding_rejects_wrong_or_uncertain_source(tmp_path, failure):
    directory = tmp_path / "capture"
    sha = capture(directory, planet_capture("No") if failure == "no" else populated_planet())
    if failure == "manifest":
        write(directory / "manifest.json", {"observation_sha256": "0" * 64})
    if failure == "stopped":
        write(directory / "stopped.json", {})
    with pytest.raises(BrowserSafetyStop):
        module.decide_planet_class(
            tmp_path,
            directory,
            "0" * 64 if failure == "hash" else sha,
            expected_star="Other" if failure == "star" else "Jyremis",
        )


def spectra(selected=(1, 4), *, duplicate=False):
    output = {}
    for gas in ("baseline", *GASES):
        image = Image.new("RGB", (320, 200))
        draw = ImageDraw.Draw(image)
        for y in (30, 50, 70, 90, 110, 130):
            draw.line((10, y, 319, y), fill=(40, 40, 40))
        actual = [
            (x, 64 if any(40 + 35 * i <= x < 48 + 35 * i for i in selected) else 44) for x in range(10, 320)
        ]
        draw.line(actual, fill=(77, 127, 148))
        index = GASES.index(gas) if gas != "baseline" else -1
        if duplicate and index == 2:
            index = 1
        model = [
            (x, 64 if index >= 0 and 40 + 35 * index <= x < 48 + 35 * index else 44) for x in range(10, 320)
        ]
        draw.line(model, fill=(164, 164, 164))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        output[gas] = stream.getvalue()
    return output


def test_actual_pixels_choose_supported_combination_not_gas_order():
    crops = spectra()
    result = module.match_visible_gas_crops(crops)
    assert result["gases"] == ["CO2", "N2O"]
    assert result["rms_pixels"] == 0
    assert result["observed_dip_coverage"] == 1
    assert not result["physical_gas_identity_verified"]
    assert result == module.match_visible_gas_crops(dict(reversed(list(crops.items()))))


def gas_capture(gases, absorption=None):
    text = ", ".join(gases)
    report = habitat_capture(text)
    frame = report["frames"][0]
    original = "6.882" if gases else "0.000"
    absorption = absorption if absorption is not None else original
    frame["accessibility"] = frame["accessibility"].replace(f"{original}%", f"{absorption}%")
    frame["text"] = f"JYREMIS\nTRACE GASES PRESENT {text} ABSORPTION % {absorption}% Save"
    return report


def gas_writes(directory, writes, absorption=None):
    selected = []
    capture(directory / "initial", gas_capture(selected))
    capture(directory / "menu-opened", gas_capture(selected))
    write(
        directory / "scope.json",
        {
            "star": "Jyremis",
            "max_checkbox_writes": len(writes),
            "selection_source": "explicit_reference_comparison_not_learned",
            "hidden_spectrum_data_read": False,
            "numeric_writes": 0,
            "assessment_clicks": 0,
            "task_completed": False,
        },
    )
    for index, (gas, checked) in enumerate(writes, 1):
        intent = {
            "gas": gas,
            "checked": checked,
            "prior": selected,
            "star": "Jyremis",
            "action_source": "reference_diagnostic",
            "automatic_retry": False,
        }
        write(directory / f"write-{index:02d}-reserved.json", intent)
        selected = sorted((set(selected) | {gas}) if checked else (set(selected) - {gas}))
        report = gas_capture(selected, absorption if selected and index == len(writes) else None)
        capture(directory / f"write-{index:02d}-after", report)
        readout = _mapping(report)["observation"]["values"]["readouts"]["absorption"]
        write(
            directory / f"write-{index:02d}-confirmed.json",
            {
                **intent,
                "selected": selected,
                "absorption": readout,
                "readback_verified": True,
                "correctness_verified": False,
                "task_completed": False,
            },
        )
    return readout


def gas_sources(root, absorption="50"):
    directory = root / "gas-comparison"
    gas_writes(directory, [(g, v) for g in GASES for v in (True, False)])
    hashes = {name: write(directory / (name + ".png"), raw) for name, raw in spectra().items()}
    comparison = {
        "star": "Jyremis",
        "candidates": list(GASES),
        "checkbox_writes": 14,
        "baseline_restored": True,
        "gas_selection_inferred": False,
        "learned_gas_identification": False,
        "task_completed": False,
        "chart_sha256": hashes,
    }
    comparison_sha = write(directory / "report.json", comparison)
    selection = root / "gas-selection"
    gases = ["CO2", "N2O"]
    readout = gas_writes(selection, [(g, True) for g in gases], absorption)
    intent = {
        "star": "Jyremis",
        "gases": gases,
        "rationale": "Synthetic visible comparison, not scientific identity.",
        "action_source": "explicit_visual_reference_not_learned",
        "comparison_sha256": comparison_sha,
        "chart_sha256": hashes,
        "correctness_verified": False,
        "automatic_retry": False,
        "task_completed": False,
    }
    write(selection / "selection.json", intent)
    selection_sha = write(
        selection / "report.json",
        {**intent, "readback_verified": True, "checkbox_writes": len(gases), "absorption": readout},
    )
    write(selection / "combined.png", spectra()["baseline"])
    return {
        "run_history": root,
        "comparison_dir": directory,
        "comparison_sha256": comparison_sha,
        "selection_dir": selection,
        "selection_sha256": selection_sha,
        "expected_star": "JYREMIS",
    }


def test_full_public_gas_receipt_chain_uses_pixels_not_checked_labels(tmp_path):
    sources = gas_sources(tmp_path)
    options = {k: v for k, v in sources.items() if not k.startswith("selection_")}
    result = module.decide_gases(**options)
    assert result["status"] == "decided" and result["payload"]["gases"] == ["CO2", "N2O"]
    assert set(result["payload"]) == {"gases", "rationale"}
    assert result["evidence"]["greenhouse_increment_requires_selected_combination_readback"]
    assert len(result["source_sha256"]) > 50
    assert result == module.decide_gases(**options)


@pytest.mark.parametrize(
    "absorption,increment", [("0.5", 10), ("39.99", 10), ("40", 30), ("59.99", 30), ("60", 100), ("100", 100)]
)
def test_greenhouse_uses_actual_combined_visible_absorption(tmp_path, absorption, increment):
    sources = gas_sources(tmp_path, absorption)
    result = module.decide_greenhouse(**sources)
    assert result["payload"] == {"supplied_greenhouse_increment": increment}
    assert result["evidence"]["absorption_percent"] == absorption
    assert not result["scientific_verified"]


@pytest.mark.parametrize("absorption", ["0", "0.49", "0.495", "39.995", "59.995"])
def test_published_greenhouse_gaps_or_missing_native_none_are_not_repaired(tmp_path, absorption):
    result = module.decide_greenhouse(**gas_sources(tmp_path, absorption))
    assert result["status"] == "abstained" and result["payload"] is None


@pytest.mark.parametrize(
    "failure",
    ["crop", "comparison_hash", "selection_hash", "comparison_stopped", "write_receipt", "prior_selected"],
)
def test_gas_source_changes_fail_without_using_the_known_selection(tmp_path, failure):
    options = gas_sources(tmp_path)
    if failure == "crop":
        write(tmp_path / "gas-comparison/CO2.png", spectra()["CH4"])
    elif failure == "comparison_hash":
        options["comparison_sha256"] = "0" * 64
    elif failure == "selection_hash":
        options["selection_sha256"] = "0" * 64
    elif failure == "comparison_stopped":
        write(tmp_path / "gas-comparison/write-01-stopped.json", {})
    elif failure == "write_receipt":
        write(tmp_path / "gas-comparison/write-01-confirmed.json", {})
    else:
        capture(tmp_path / "gas-comparison/initial", gas_capture(["O3"]))
    with pytest.raises(BrowserSafetyStop):
        module.decide_greenhouse(**options)


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "missing",
        "flat",
        "palette",
        "alignment",
        "incomplete",
        "transparent",
        "clipped",
        "target_changed",
    ],
)
def test_uncertain_gas_pixels_abstain_without_guessing(failure):
    crops = spectra(duplicate=failure == "duplicate")
    if failure == "missing":
        crops.pop("O3")
    elif failure == "flat":
        crops = spectra(selected=())
    elif failure in {"palette", "alignment", "incomplete", "transparent", "clipped", "target_changed"}:
        image = Image.open(io.BytesIO(crops["CO2"])).convert("RGB")
        draw = ImageDraw.Draw(image)
        if failure == "palette":
            draw.point((50, 60), fill=(255, 0, 0))
        elif failure == "alignment":
            draw.point((20, 10), fill=(255, 255, 255))
        elif failure == "clipped":
            draw.line((100, 80, 100, 130), fill=(164, 164, 164))
        elif failure == "target_changed":
            draw.line((30, 80, 32, 80), fill=(77, 127, 148))
        elif failure == "transparent":
            image = image.convert("RGBA")
            image.putalpha(0)
        else:
            draw.rectangle((80, 40, 160, 68), fill=(0, 0, 0))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        crops["CO2"] = stream.getvalue()
    with pytest.raises(BrowserSafetyStop):
        module.match_visible_gas_crops(crops)


def phase_sources(root, phase="Liquid"):
    """Synthetic public receipt chain, consumed by production validators."""
    planet = root / "planet-class"
    intent = {
        "star": "Jyremis",
        "kind": "SELECT",
        "value": "terrestrial",
        "action_source": "reference_diagnostic",
        "max_class_writes": 1,
        "numeric_writes": 0,
        "correctness_verified": False,
        "task_completed": False,
    }
    write(planet / "reserved.json", intent)
    class_sha = write(planet / "confirmed.json", {**intent, "readback_verified": True})
    capture(planet / "after", populated_planet())
    chamber = root / "chamber"
    before = habitat_capture("O₃", False, "759.4", "Weak (+10)")
    after = habitat_capture("O₃", False, "759.4", "Weak (+10)", phase)
    capture(chamber / "task-before", before)
    capture(chamber / "task-after", before)
    query = {
        "pressure": "9",
        "pressure_unit": "atm",
        "temperature": "769.4",
        "temperature_unit": "K",
        "action_source": "reference_diagnostic",
        "task_answer_write": False,
    }
    write(chamber / "reserved.json", query)
    result = {
        **query,
        "conditions_verified": True,
        "task_completed": False,
        "source": "visible_chamber_indicator",
        "phase": phase.lower(),
        "visible_readback": {"atm": "9", "K": "769.4"},
        "icons": [
            {"phase": p, "fully_exposed": True, "paint": {"opacity": int(p == phase.lower())}}
            for p in ("solid", "liquid", "gas")
        ],
    }
    write(chamber / "confirmed.json", result)
    write(chamber / "observed.json", result)
    write(chamber / "close-reserved.json", {"retry_allowed": False})
    write(chamber / "chamber.png", b"\x89PNG\r\n\x1a\nfixture_not_scientific_evidence")
    phase_dir = root / "phase"
    capture(phase_dir / "initial", before)
    capture(phase_dir / "after", after)
    source_hashes = {
        str(chamber / name): module._sha((chamber / name).read_bytes())
        for name in (
            "confirmed.json",
            "reserved.json",
            "task-before/observation.json",
            "task-before/manifest.json",
        )
    }
    intent = {
        "star": "Jyremis",
        "kind": "SELECT",
        "field": "water_phase",
        "label": phase,
        "action_source": "reference_diagnostic",
        "evidence": {
            "source": "confirmed_visible_chamber_indicator",
            "phase": phase.lower(),
            "pressure": "9",
            "temperature": "769.4",
            "source_sha256": source_hashes,
        },
        "max_menu_writes": 1,
        "numeric_writes": 0,
        "task_completed": False,
        "correctness_verified": False,
        "automatic_retry": False,
    }
    write(phase_dir / "reserved.json", intent)
    phase_sha = write(
        phase_dir / "confirmed.json",
        {
            **intent,
            "readback_verified": True,
            "surface_temp": _mapping(after)["observation"]["values"]["readouts"]["surface_temp"],
        },
    )
    return {
        "run_history": root,
        "phase_dir": phase_dir,
        "phase_sha256": phase_sha,
        "planet_class_dir": planet,
        "planet_class_sha256": class_sha,
        "expected_star": "JYREMIS",
    }


@pytest.mark.parametrize(
    "phase,choice", [("Liquid", "habitable"), ("Solid", "not_habitable"), ("Gas", "not_habitable")]
)
def test_course_phase_rule_is_source_bound_without_added_lifetime_test(tmp_path, phase, choice):
    options = phase_sources(tmp_path, phase)
    result = module.decide_habitability(**options)
    assert result["payload"]["choice"] == choice
    assert result["evidence"]["lifetime_used"] is False
    assert result["evidence"]["water_vapor_used"] is False
    assert result["reference_pack"]["habitability_source"] == "https://kb.inspark.education/habworlds-project"
    assert result == module.decide_habitability(**options)
    assert not result["scientific_verified"] and not result["task_completed"]


@pytest.mark.parametrize(
    "failure",
    [
        "hash",
        "star",
        "source",
        "uncertain",
        "observed",
        "close",
        "class",
        "pressure",
        "hidden_phase",
        "wrong_readback",
    ],
)
def test_phase_or_class_tampering_cannot_authorize_choice(tmp_path, failure):
    options = phase_sources(tmp_path)
    if failure == "hash":
        options["phase_sha256"] = "0" * 64
    elif failure == "star":
        options["expected_star"] = "Another"
    elif failure == "source":
        write(tmp_path / "chamber/confirmed.json", {})
    elif failure == "uncertain":
        write(tmp_path / "chamber/cleanup-stopped.json", {})
    elif failure in {"observed", "close"}:
        write(
            tmp_path / "chamber" / ("observed.json" if failure == "observed" else "close-reserved.json"), {}
        )
    elif failure == "class":
        receipt = json.loads((tmp_path / "planet-class/confirmed.json").read_bytes())
        receipt["value"] = "ice_giant"
        options["planet_class_sha256"] = write(tmp_path / "planet-class/confirmed.json", receipt)
    else:
        path = tmp_path / "chamber/confirmed.json"
        value = json.loads(path.read_bytes())
        if failure == "pressure":
            value["pressure"] = "8"
        elif failure == "hidden_phase":
            value["icons"][1]["fully_exposed"] = False
        else:
            value["visible_readback"]["K"] = "300"
        write(path, value)
    with pytest.raises(BrowserSafetyStop):
        module.decide_habitability(**options)


def test_decision_wrapper_only_consumes_persisted_explicit_derived_descriptor(tmp_path):
    derived, positive = tmp_path / "positive/derived", tmp_path / "positive"
    capture(derived / "native-copies/copy-04-after", populated_planet())
    report_sha = write(derived / "report.json", {"fixture": "verified upstream calculation"})
    files = {
        str(path.relative_to(tmp_path)): module._sha(path.read_bytes())
        for path in derived.rglob("*")
        if path.is_file()
    }
    write(
        positive / "derived-completed.json",
        {
            "report": {"path": "positive/derived/report.json", "sha256": report_sha, "files": len(files)},
            "files": files,
        },
    )
    owner = {
        "star": "Jyremis",
        "phase": "awaiting_planet_class",
        "finished": False,
        "failure_reason": None,
        "artifact_paths": {"positive_dir": "positive", "derived_dir": "positive/derived"},
    }
    options = {
        "phase": "awaiting_planet_class",
        "run_history": tmp_path,
        "owner_state": owner,
        "expected_star": "JYREMIS",
    }
    result = module.decide_planet_handoff(**options)
    assert result["decision_kind"] == "planet_class" and result["payload"]["name"] == "terrestrial"
    assert "positive/derived-completed.json" in result["source_sha256"]
    assert result == module.decide_planet_handoff(**options)
    write(derived / "report.json", {"changed": True})
    with pytest.raises(BrowserSafetyStop):
        module.decide_planet_handoff(**options)


def test_gas_wrapper_uses_exact_current_stage_descriptor(tmp_path):
    root = tmp_path / "terrestrial"
    source = gas_sources(root)
    owner = {
        "star": "Jyremis",
        "phase": "awaiting_gases",
        "finished": False,
        "failure_reason": None,
        "artifact_paths": {"terrestrial_dir": "terrestrial"},
        "terrestrial_component": {
            "star": "Jyremis",
            "phase": "awaiting_gases",
            "finished": False,
            "failure_reason": None,
            "decision_sources": {
                "gas_comparison": {
                    "directory": "terrestrial/gas-comparison",
                    "report_sha256": source["comparison_sha256"],
                }
            },
        },
    }
    options = {
        "phase": "awaiting_gases",
        "run_history": tmp_path,
        "owner_state": owner,
        "expected_star": "JYREMIS",
    }
    result = module.decide_planet_handoff(**options)
    assert result["payload"]["gases"] == ["CO2", "N2O"]
    assert all(name.startswith("terrestrial/") for name in result["source_sha256"])
    assert result == module.decide_planet_handoff(**options)
    owner["terrestrial_component"]["decision_sources"]["gas_comparison"]["directory"] = "unrelated"
    with pytest.raises(BrowserSafetyStop, match="source_path_changed"):
        module.decide_planet_handoff(**options)


def test_habitability_wrapper_uses_exact_pinned_phase_and_class(tmp_path):
    root = tmp_path / "terrestrial"
    sources = phase_sources(root)
    owner = {
        "star": "Jyremis",
        "phase": "awaiting_habitability",
        "finished": False,
        "failure_reason": None,
        "artifact_paths": {"terrestrial_dir": "terrestrial", "planet_class_dir": "terrestrial/planet-class"},
        "terrestrial_component": {
            "star": "Jyremis",
            "phase": "awaiting_habitability",
            "finished": False,
            "failure_reason": None,
            "decision_sources": {
                "phase": {"directory": "terrestrial/phase", "confirmed_sha256": sources["phase_sha256"]},
                "planet_class": {
                    "directory": "terrestrial/planet-class",
                    "confirmed_sha256": sources["planet_class_sha256"],
                },
            },
        },
    }
    options = {
        "phase": "awaiting_habitability",
        "run_history": tmp_path,
        "owner_state": owner,
        "expected_star": "JYREMIS",
    }
    result = module.decide_planet_handoff(**options)
    assert result["payload"]["choice"] == "habitable"
    assert all(name.startswith("terrestrial/") for name in result["source_sha256"])
    owner["terrestrial_component"]["decision_sources"]["phase"]["confirmed_sha256"] = "0" * 64
    with pytest.raises(BrowserSafetyStop, match="source_hash_mismatch"):
        module.decide_planet_handoff(**options)


def test_gas_visual_abstention_contains_source_pins_not_a_fallback_answer(tmp_path):
    options = gas_sources(tmp_path)
    path = tmp_path / "gas-comparison/report.json"
    report = json.loads(path.read_bytes())
    for name, raw in spectra(duplicate=True).items():
        report["chart_sha256"][name] = write(tmp_path / "gas-comparison" / (name + ".png"), raw)
    options["comparison_sha256"] = write(path, report)
    result = module.decide_gases(**{k: v for k, v in options.items() if not k.startswith("selection_")})
    assert result["status"] == "abstained" and result["payload"] is None
    assert result["reason"] == "autonomous_planet_ambiguous_gas_pixel_match"
    assert result["source_sha256"]["gas-comparison/report.json"] == options["comparison_sha256"]


def test_changed_source_during_decision_never_returns_stale_payload(tmp_path, monkeypatch):
    directory = tmp_path / "capture"
    sha = capture(directory, populated_planet())
    real = module.map_planet_capture

    def mapping(report, **kwargs):
        result = real(report, **kwargs)
        write(directory / "observation.json", {"changed": True})
        return result

    monkeypatch.setattr(module, "map_planet_capture", mapping)
    with pytest.raises(BrowserSafetyStop, match="evidence_changed"):
        module.decide_planet_class(tmp_path, directory, sha, expected_star="Jyremis")


@pytest.mark.parametrize(
    "change",
    [
        {"phase": "selecting_planet_class"},
        {"finished": True},
        {"failure_reason": "stopped"},
        {"star": "Other"},
    ],
)
def test_wrapper_stale_owner_never_dispatches(tmp_path, change):
    state = {
        "star": "Jyremis",
        "phase": "awaiting_planet_class",
        "finished": False,
        "failure_reason": None,
    } | change
    with pytest.raises(BrowserSafetyStop):
        module.decide_planet_handoff(
            phase="awaiting_planet_class", run_history=tmp_path, owner_state=state, expected_star="Jyremis"
        )


def test_outside_source_and_symlink_are_rejected(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    outside = tmp_path / "outside"
    sha = capture(outside, populated_planet())
    with pytest.raises(BrowserSafetyStop):
        module.decide_planet_class(history, outside, sha, expected_star="Jyremis")
    (history / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(BrowserSafetyStop):
        module.decide_planet_class(history, history / "linked", sha, expected_star="Jyremis")
