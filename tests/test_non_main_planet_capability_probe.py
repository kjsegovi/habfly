"""Disposable diagnostic helper checks; native leaf gate is explicitly opt-in.

The native tests cover Yes transport only, not real non-main course capability.
The entire helper and actual class-conditioned UI remain a root-run diagnostic.
"""

# ruff: noqa: F811

import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet import capture
from test_browser_planet_numeric import planet_page  # noqa: F401
from test_browser_planet_presence import presence_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet import map_planet_capture
from habfly.browser_probe import save_probe
from habfly.browser_setup import SetupStop
from habfly.browser_stellar import StellarMappingError

SOURCE = Path(__file__).resolve().parents[1] / "scripts/diagnose_non_main_planet_capability.py"
spec = importlib.util.spec_from_file_location("non_main_capability_probe", SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def mapped(presence=None):
    report = capture(presence)
    frame = report["frames"][0]
    first = frame["controls"][0]
    first["value"], first["accessibility"] = "", yaml.safe_dump(['textbox "0"'])
    atoms = yaml.safe_load(frame["accessibility"])
    atoms[2] = 'textbox "0"'
    frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)
    return map_planet_capture(report, capture_sha256="a" * 64)


@pytest.mark.parametrize("presence", [None, "Yes"])
def test_public_field_summary_preserves_absent_blank_units_and_readonly_inputs(presence):
    value = mapped(presence)
    probe.blank_planet(value, "JYREMIS", presence)
    result = probe.field_summary(value)
    assert result["has_planet"] == presence
    assert set(result["visible_planet_field_map"]) == probe.BASE_FIELDS | (
        probe.DERIVED if presence else set()
    )
    assert all(f["current_value"] == "" for f in result["visible_planet_field_map"].values())
    assert result["stellar_inputs"]["stellar_mass"]["display_text"] == "2.368"
    assert result["stellar_inputs"]["stellar_radius"]["display_text"] == "1.961"
    assert result["stellar_inputs"]["stellar_mass"]["unit"] == "Msun"
    assert result["stellar_inputs"]["stellar_radius"]["unit"] == "Rsun"
    assert not result["quantities_independently_validated"]
    assert not result["quantity_origin_or_default_behavior_verified"]
    result["stellar_inputs"]["stellar_mass"]["display_text"] = "changed"
    assert value["observation"]["values"]["stellar_inputs"]["stellar_mass"]["display_text"] == "2.368"


@pytest.mark.parametrize("change", ["unit", "zero", "unknown", "missing", "disabled", "star", "presence"])
def test_wrong_blank_or_unit_contract_never_becomes_capability(change):
    value = mapped("Yes")
    values = value["observation"]["values"]
    field = values["browser_field_map"]["planet_mass"]
    if change == "unit":
        field["unit"] = "Msun"
    elif change == "zero":
        field["current_value"] = "0"
    elif change == "unknown":
        values["browser_field_map"]["unknown"] = deepcopy(field)
    elif change == "missing":
        del values["browser_field_map"]["planet_density"]
    elif change == "disabled":
        field["enabled"] = False
    elif change == "star":
        value["star_name"] = "Other"
    else:
        values["has_planet"] = "No"
    with pytest.raises(probe.DiagnosticStop):
        probe.blank_planet(value, "JYREMIS", "Yes")


@pytest.mark.parametrize("selected", ["main_sequence", "", "WHITE_DWARF", None, True])
def test_unsupported_class_fails_before_page_access(tmp_path, selected):
    with pytest.raises(probe.DiagnosticStop, match="diagnostic_unsupported_class"):
        probe.CapabilityDiagnostic(None, None, tmp_path, selected, tmp_path)


@pytest.mark.parametrize("error", [RuntimeError, SetupStop, BrowserSafetyStop, probe.DiagnosticStop])
def test_private_driver_or_credential_string_is_not_a_failure_reason(error):
    secret = "https://private.invalid/?token=secret password=fixture-private"
    assert probe.safe_failure(error(secret)) == "diagnostic_operation_failed"


def test_scope_cannot_claim_a_planet_training_or_workflow_completion():
    assert probe.FLAGS["diagnostic_only"] is True
    for key in (
        "detected_planet",
        "reference_classification",
        "scientific_verified",
        "learned_policy",
        "training_label",
        "workflow_verified",
        "canonical_receipt",
        "task_completed",
        "project_completed",
        "submitted",
        "automatic_retry",
    ):
        assert probe.FLAGS[key] is False
    assert all(
        probe.FLAGS[name] == 0
        for name in (
            "derived_numeric_writes",
            "stellar_numeric_writes",
            "planet_raw_numeric_writes",
            "save_clicks",
            "assessment_clicks",
            "score_transfer_clicks",
            "submission_clicks",
        )
    )


def test_counting_includes_explicit_class_and_yes_as_answer_write_attempts():
    owner = probe.CapabilityDiagnostic.__new__(probe.CapabilityDiagnostic)
    owner.class_sessions = [
        SimpleNamespace(_class_click_invoked=True, _class_click_returned=True),
        SimpleNamespace(_class_click_invoked=True, _class_click_returned=False),
    ]
    owner.report = {"yes_selection_attempted": True}
    owner._counts()
    assert owner.report == {
        "yes_selection_attempted": True,
        "class_click_attempts": 2,
        "class_clicks_returned": 1,
        "diagnostic_answer_write_attempts": 3,
    }


@pytest.mark.parametrize("closed_driver", [False, True])
def test_repeated_close_is_latched_and_preserves_sanitized_cleanup_evidence(closed_driver):
    calls = []

    def invoke(name):
        calls.append(name)
        if closed_driver:
            raise RuntimeError("private credential https://secret.invalid")

    page = SimpleNamespace(
        frames=[],
        on=lambda *_: None,
        remove_listener=lambda *_: invoke("dialog"),
        context=SimpleNamespace(on=lambda *_: None, remove_listener=lambda *_: invoke("popup")),
    )
    owner = probe.CapabilityDiagnostic(page, None, Path("."), "white_dwarf", Path("."))
    owner.session = SimpleNamespace(close=lambda: invoke("planet"))
    owner.class_sessions = [SimpleNamespace(stopped=False)]
    owner.report["failure_reason"] = "diagnostic_original_failure"
    owner.close()
    owner.close()
    assert calls == ["planet", "dialog", "popup"]
    assert owner.closed and owner.class_sessions[0].stopped
    assert owner.report["failure_reason"] == "diagnostic_original_failure"
    assert len(owner.report.get("cleanup_failures", [])) == (3 if closed_driver else 0)
    assert "secret" not in json.dumps(owner.report)


def test_outer_cleanup_attempts_browser_after_other_failures_without_masking_original():
    calls = []

    def fail(name):
        calls.append(name)
        raise RuntimeError("private credential https://secret.invalid")

    report = {"failure_reason": "diagnostic_original_failure", "task_completed": False}
    owner = SimpleNamespace(close=lambda: fail("probe"), report={})
    probe.cleanup(
        report,
        setup=SimpleNamespace(close=lambda: fail("setup")),
        probe=owner,
        browser=SimpleNamespace(close=lambda: fail("browser")),
        browser_closed=False,
    )
    assert calls == ["setup", "probe", "browser"]
    assert report["failure_reason"] == "diagnostic_original_failure"
    assert not report["browser_closed"] and not report["cleanup_completed"]
    assert report["cleanup_failures"] == [
        "diagnostic_setup_cleanup_failed",
        "diagnostic_probe_cleanup_failed",
        "diagnostic_browser_cleanup_failed",
    ]
    assert "secret" not in json.dumps(report)


def test_already_closed_browser_does_not_invalidate_successful_cleanup():
    report = {}
    probe.cleanup(report, setup=None, probe=None, browser=None, browser_closed=True)
    assert report == {"browser_closed": True, "cleanup_completed": True}


@pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="root owns live browser; explicit native idle gate required",
)
@pytest.mark.parametrize("mutation", [None, "raw_answer", "same_text_replacement", "wrong_unit"])
def test_actual_presence_leaf_preserves_native_guards(presence_page, tmp_path, mutation):
    page, frame = presence_page
    frame.get_by_role("textbox").first.fill("")  # Explicit synthetic fresh-star fixture setup.
    output = tmp_path / "diagnostic"
    output.mkdir()
    save_probe(capture(), output / "stellar-after")  # Fixture provenance stub, not a real class receipt.
    owner = probe.CapabilityDiagnostic(page, config(), output, "white_dwarf", output / "unused-fresh")
    owner.star = "JYREMIS"
    if mutation:
        code = {
            "raw_answer": "document.getElementById('line_shift').value='1'",
            "same_text_replacement": "e.replaceWith(e.cloneNode(true))",
            "wrong_unit": "document.getElementById('planet_mass').previousElementSibling.innerHTML='mass (Ms)'",
        }[mutation]
        frame.get_by_role("combobox").evaluate(
            "(e,code)=>{const original=e.onchange;e.onchange=()=>{original();new Function('e',code)(e)}}",
            code,
        )
    try:
        if mutation:
            with pytest.raises((probe.DiagnosticStop, BrowserSafetyStop, StellarMappingError)):
                owner.select_yes()
            assert not owner.report["yes_readback_verified"]
        else:
            owner.select_yes()
            assert owner.report["yes_readback_verified"]
            assert set(owner.report["after"]["visible_planet_field_map"]) == probe.BASE_FIELDS | probe.DERIVED
            assert owner.report["after"]["selected_class"] == "white_dwarf"
        owner._counts()
        assert owner.report["diagnostic_answer_write_attempts"] == 1
        assert owner.report["yes_selection_returned"] is True
        assert frame.get_by_role("combobox").input_value() == "Yes"
        assert not (output / "confirmed.json").exists()
        assert not (output / "diagnostic-report.json").exists()  # Outer owner alone writes final diagnostic.
        assert json.loads((output / "diagnostic-yes-intent.json").read_bytes())["detected_planet"] is False
    finally:
        owner.close()
