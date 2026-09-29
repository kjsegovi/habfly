"""Intercepted fixtures only: bounded reference menus, never a grading oracle."""
# ruff: noqa: F811

import json

import pytest
from test_browser_habitability_numeric import habitat_page  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_habitability_actions import HabitabilityMenuSession
from habfly.browser_probe import inspect_page, save_probe


@pytest.fixture
def menu_page(habitat_page):
    page, frame = habitat_page
    frame.locator("#temperature").fill("759.4")
    frame.locator("#temperature").press("Tab")
    frame.locator("#absorption").evaluate("e=>e.textContent='46.88'")
    frame.locator("#greenhouse").evaluate("""e=>e.onchange=()=>{
      window.menuWrites=(window.menuWrites||0)+1;
      document.querySelector('#surface').textContent=String(Number(document.querySelector('#temperature').value)+({Weak:10,Moderate:30,Strong:100}[e.value]||0));
    }""")
    frame.locator("#phase").evaluate("e=>e.onchange=()=>{window.menuWrites=(window.menuWrites||0)+1}")
    return page, frame


def chamber_record(page, tmp_path, phase="gas"):
    directory = tmp_path / "chamber"
    save_probe(inspect_page(page, config()), directory / "task-before")
    intent = {
        "pressure": "9",
        "pressure_unit": "atm",
        "temperature": "759.4",
        "temperature_unit": "K",
        "action_source": "reference_diagnostic",
        "task_answer_write": False,
    }
    result = {
        **intent,
        "phase": phase,
        "conditions_verified": True,
        "source": "visible_chamber_indicator",
        "visible_readback": {"atm": "9", "K": "759.4"},
        "task_completed": False,
        "icons": [
            {"phase": name, "paint": {"opacity": int(name == phase)}, "fully_exposed": True}
            for name in ("solid", "liquid", "gas")
        ],
    }
    (directory / "reserved.json").write_text(json.dumps(intent))
    (directory / "confirmed.json").write_text(json.dumps(result))
    return directory


def test_greenhouse_one_reference_selection_preserves_numbers(menu_page, tmp_path):
    page, frame = menu_page
    session = HabitabilityMenuSession(page, config(), tmp_path / "menu")
    try:
        receipt = session.greenhouse_reference()
        assert receipt["readback_verified"] and not receipt["correctness_verified"]
        assert receipt["action_source"] == "reference_diagnostic"
        assert receipt["evidence"]["increment_kelvin"] == 30
        assert frame.locator("#greenhouse").input_value() == "Moderate"
        assert frame.locator("#temperature").input_value() == "759.4"
        assert frame.locator("#surface").inner_text() == "789.4"
        assert frame.evaluate("window.menuWrites") == 1
        assert not page.get_by_role("checkbox").is_checked()
        with pytest.raises(BrowserSafetyStop):
            session.greenhouse_reference()
        assert frame.evaluate("window.menuWrites") == 1
    finally:
        session.close()


@pytest.mark.parametrize("absorption", ["0", "0.495", "39.995", "59.995"])
def test_none_and_published_band_gaps_are_not_repaired(menu_page, tmp_path, absorption):
    page, frame = menu_page
    frame.locator("#absorption").evaluate("(e,v)=>e.textContent=v", absorption)
    session = HabitabilityMenuSession(page, config(), tmp_path / "menu")
    try:
        with pytest.raises(BrowserSafetyStop, match="no_explicit_supported"):
            session.greenhouse_reference()
        assert not session.attempted and frame.locator("#greenhouse").input_value() == ""
    finally:
        session.close()


@pytest.mark.parametrize(
    "mutation", ["answer", "outer", "replace", "feedback", "auth", "modal", "absorption"]
)
def test_stale_context_stops_before_dispatch(menu_page, tmp_path, mutation):
    page, frame = menu_page
    session = HabitabilityMenuSession(page, config(), tmp_path / "menu")
    if mutation == "outer":
        page.get_by_role("checkbox").check()
    else:
        frame.locator("body").evaluate(
            "(e,code)=>{new Function(code)()}",
            {
                "answer": "document.querySelector('#temperature').value='250'",
                "replace": "let e=document.querySelector('#greenhouse');e.replaceWith(e.cloneNode(true))",
                "feedback": "document.body.append('Unexpected feedback')",
                "auth": "document.body.insertAdjacentHTML('beforeend','<input type=password>')",
                "modal": "document.body.insertAdjacentHTML('beforeend','<div role=dialog>Stop</div>')",
                "absorption": "document.querySelector('#absorption').textContent='10'",
            }[mutation],
        )
    try:
        with pytest.raises(BrowserSafetyStop):
            session.greenhouse_reference()
        assert not session.attempted and frame.locator("#greenhouse").input_value() == ""
    finally:
        session.close()


@pytest.mark.parametrize(
    "mutation", ["surface", "temperature", "phase", "absorption", "feedback", "replace", "dialog"]
)
def test_uncertain_side_effect_is_reserved_never_retried(menu_page, tmp_path, mutation):
    page, frame = menu_page
    code = {
        "surface": "document.querySelector('#surface').textContent='999'",
        "temperature": "document.querySelector('#temperature').value='250'",
        "phase": "document.querySelector('#phase').value='Liquid'",
        "absorption": "document.querySelector('#absorption').textContent='5'",
        "feedback": "document.body.append('Unexpected feedback')",
        "replace": "e.replaceWith(e.cloneNode(true))",
        "dialog": "confirm('private fixture text')",
    }[mutation]
    frame.locator("#greenhouse").evaluate(
        "(e,code)=>{let prior=e.onchange;e.onchange=()=>{prior();new Function('e',code)(e)}}", code
    )
    session = HabitabilityMenuSession(page, config(), tmp_path / "menu")
    try:
        with pytest.raises(BrowserSafetyStop):
            session.greenhouse_reference()
        assert session.attempted and session.stopped
        stopped = json.loads((tmp_path / "menu/stopped.json").read_text())
        assert stopped["write_may_have_occurred"] and not stopped["automatic_retry"]
        assert "private fixture text" not in json.dumps(stopped)
        with pytest.raises(BrowserSafetyStop):
            session.greenhouse_reference()
        assert frame.evaluate("window.menuWrites") == 1
    finally:
        session.close()


@pytest.mark.parametrize("phase", ["solid", "liquid", "gas"])
def test_water_phase_uses_confirmed_same_condition_chamber(menu_page, tmp_path, phase):
    page, frame = menu_page
    record = chamber_record(page, tmp_path, phase)
    session = HabitabilityMenuSession(page, config(), tmp_path / "menu")
    try:
        receipt = session.water_phase_from_chamber(record)
        assert receipt["readback_verified"] and receipt["label"] == phase.title()
        assert len(receipt["evidence"]["source_sha256"]) == 4
        assert frame.locator("#phase").input_value() == phase.title()
        assert frame.locator("#surface").inner_text() == "759.4"
        assert frame.evaluate("window.menuWrites") == 1
    finally:
        session.close()


@pytest.mark.parametrize(
    "mutation", ["temperature", "pressure", "star", "unconfirmed", "icon", "readback", "missing_intent"]
)
def test_incompatible_phase_evidence_cannot_select(menu_page, tmp_path, mutation):
    page, frame = menu_page
    record = chamber_record(page, tmp_path)
    path = record / "confirmed.json"
    data = json.loads(path.read_text())
    if mutation == "star":
        frame.get_by_text("JYREMIS", exact=True).evaluate("e=>e.textContent='ISSATEST'")
    elif mutation == "missing_intent":
        (record / "reserved.json").write_text("{}")
    elif mutation == "temperature":
        data["temperature"] = "760"
    elif mutation == "pressure":
        data["pressure"] = "8"
    elif mutation == "unconfirmed":
        data["conditions_verified"] = False
    elif mutation == "icon":
        data["icons"][0]["paint"]["opacity"] = 1
    else:
        data["visible_readback"]["K"] = "800"
    path.write_text(json.dumps(data))
    session = HabitabilityMenuSession(page, config(), tmp_path / "menu")
    try:
        with pytest.raises(BrowserSafetyStop, match="incompatible_chamber"):
            session.water_phase_from_chamber(record)
        assert not session.attempted and frame.locator("#phase").input_value() == ""
    finally:
        session.close()
