"""Intercepted transport tests; fixture phase values are not course labels."""
# ruff: noqa: F811

import json

import pytest
from test_browser_habitability_actions import chamber_record, menu_page  # noqa: F401
from test_browser_habitability_numeric import habitat_page  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_habitability_actions import HabitabilityMenuSession
from habfly.browser_habitability_choice import select_habitability_reference

RATIONALE = "Explicit reference: evaluate the confirmed water phase at the displayed pressure and surface temperature; not a learned or course-verified decision."


def setup_choice(page, frame, tmp_path, phase):
    chamber = chamber_record(page, tmp_path, phase)
    record = tmp_path / "phase"
    session = HabitabilityMenuSession(page, config(), record)
    try:
        session.water_phase_from_chamber(chamber)
    finally:
        session.close()
    frame.locator("body").evaluate("""e=>{
      const style=document.createElement('style');style.textContent=`
        .choice label.picked{border:1px solid rgb(255,255,255)}
        .choice label.picked::after{opacity:1;background:rgb(255,255,255)}`;
      document.head.append(style);
      const labels=[...document.querySelectorAll('.choice label')];
      labels[0].classList.add('picked');
      labels.forEach(label=>label.onclick=()=>{
        window.choiceWrites=(window.choiceWrites||0)+1;
        labels.forEach(e=>e.classList.remove('picked'));label.classList.add('picked');
      });
    }""")
    return record


@pytest.mark.parametrize(
    "phase,name",
    [
        ("gas", "not_habitable"),
        ("solid", "not_habitable"),
        ("liquid", "habitable"),
        ("liquid", "not_habitable"),
    ],
)
def test_explicit_choice_and_default_confirmation(menu_page, tmp_path, phase, name):
    page, frame = menu_page
    record = setup_choice(page, frame, tmp_path, phase)
    output = tmp_path / "choice"
    result = select_habitability_reference(
        page, config(), output, phase_record=record, name=name, rationale=RATIONALE
    )
    assert result["choice"] == name and result["readback_verified"]
    assert not result["learned_habitability_decision"] and not result["task_completed"]
    assert not result["correctness_verified"] and result["numeric_writes"] == 0
    assert frame.evaluate("window.choiceWrites") == 1
    assert frame.locator("#temperature").input_value() == "759.4"
    assert frame.locator("#phase").input_value() == phase.title()
    with pytest.raises(FileExistsError):
        select_habitability_reference(
            page, config(), output, phase_record=record, name=name, rationale=RATIONALE
        )
    assert frame.evaluate("window.choiceWrites") == 1


@pytest.mark.parametrize(
    "mutation",
    ["phase", "temperature", "pressure", "star", "source", "unconfirmed", "stopped", "habitable_gas"],
)
def test_incompatible_or_stale_evidence_never_clicks(menu_page, tmp_path, mutation):
    page, frame = menu_page
    record = setup_choice(page, frame, tmp_path, "gas")
    if mutation == "phase":
        frame.locator("#phase").select_option(label="Solid")
    elif mutation == "temperature":
        frame.locator("#surface").evaluate("e=>e.textContent='700'")
    elif mutation == "pressure":
        frame.get_by_text("Observations Modeled Albedo 0.05 Modeled Pressure (atm) 9", exact=True).evaluate(
            "e=>e.textContent=e.textContent.replace('atm) 9','atm) 8')"
        )
    elif mutation == "star":
        frame.get_by_text("JYREMIS", exact=True).evaluate("e=>e.textContent='ANOTHER'")
    elif mutation == "source":
        (tmp_path / "chamber/confirmed.json").write_text("{}")
    elif mutation == "stopped":
        (record / "stopped.json").write_text("{}")
    elif mutation == "unconfirmed":
        receipt = json.loads((record / "confirmed.json").read_text())
        receipt["readback_verified"] = False
        (record / "confirmed.json").write_text(json.dumps(receipt))
    with pytest.raises(BrowserSafetyStop):
        select_habitability_reference(
            page,
            config(),
            tmp_path / "choice",
            phase_record=record,
            name="habitable" if mutation == "habitable_gas" else "not_habitable",
            rationale=RATIONALE,
        )
    assert frame.evaluate("window.choiceWrites||0") == 0


@pytest.mark.parametrize(
    "mutation",
    ["temperature", "phase", "absorption", "feedback", "outer", "replace", "dialog", "wrong_choice"],
)
def test_uncertain_dispatch_is_retained_not_retried(menu_page, tmp_path, mutation):
    page, frame = menu_page
    record = setup_choice(page, frame, tmp_path, "liquid")
    code = {
        "temperature": "document.querySelector('#temperature').value='300'",
        "phase": "document.querySelector('#phase').value='Solid'",
        "absorption": "document.querySelector('#absorption').textContent='10'",
        "feedback": "document.body.append('Unexpected feedback')",
        "outer": "document.body.insertAdjacentHTML('beforeend','<input type=password>')",
        "replace": "e.replaceWith(e.cloneNode(true))",
        "dialog": "confirm('private fixture text')",
        "wrong_choice": "e.classList.remove('picked')",
    }[mutation]
    frame.locator(".choice label").nth(1).evaluate(
        "(e,code)=>{const before=e.onclick;e.onclick=()=>{before();new Function('e',code)(e)}}", code
    )
    output = tmp_path / "choice"
    with pytest.raises(BrowserSafetyStop):
        select_habitability_reference(
            page, config(), output, phase_record=record, name="habitable", rationale=RATIONALE
        )
    assert frame.evaluate("window.choiceWrites") == 1
    stopped = json.loads((output / "stopped.json").read_text())
    assert stopped["write_may_have_occurred"] and not stopped["automatic_retry"]
    assert "private fixture text" not in json.dumps(stopped)
    assert not (output / "confirmed.json").exists()
