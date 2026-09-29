"""Intercepted native temperature transport; no course or scientific oracle."""
# ruff: noqa: F811

import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_habitability_numeric import HabitabilityNumericSession
from habfly.browser_stellar import SIMULATION_URL


@pytest.fixture
def habitat_page(page):
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("iframe").first.evaluate("e=>{e.style.width='1000px';e.style.height='700px'}")
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.locator("body").evaluate("""e=>e.innerHTML=`
      <div>JYREMIS</div><img alt='1'><div>Observations Modeled Albedo 0.05 Modeled Pressure (atm) 9</div>
      <svg role='img' width='280' height='30'><text y='20'>Flux WAVELENGTH (μm) 0 2 4 6</text></svg>
      <div>Your Reconstruction Equilibrium Temp (K)</div><input id='temperature' placeholder='0' value='0'>
      <div>Trace Gases Present</div><select id='gases'><option selected></option></select>
      <span>Absorption % </span><span id='absorption'>0</span><span>% Greenhouse Effect</span>
      <select id='greenhouse'><option selected></option><option value='Weak'>Weak (+10)</option><option value='Moderate'>Moderate (+30)</option><option value='Strong'>Strong (+100)</option></select>
      <span>Surface Temp (K) </span><span id='surface'></span><span> Water Phase</span>
      <select id='phase'><option selected></option><option>Solid</option><option>Liquid</option><option>Gas</option></select>
      <div class='choice'><label></label><div>Not Habitable</div></div><div class='choice'><label></label><div>Habitable</div></div>
      <div>STAR LUMINOSITY (Ls) 20.44 ORBIT RADIUS (AU) 0.5919</div><button>Save</button>
      <style>.choice{display:inline-block;width:120px;text-align:center}.choice label{display:block;position:relative;width:26px;height:26px;border:1px solid rgb(0,200,220);margin:auto}.choice label::after{content:"";position:absolute;display:block;width:14px;height:14px;opacity:0;background:transparent;left:6px;top:6px}.choice>div{margin-top:3px;font-size:10px}</style>
    `""")
    frame.locator("#temperature").evaluate("""e=>e.onblur=()=>{
        e.value=Number(e.value).toPrecision(4);
        document.querySelector('#surface').textContent=e.value;
        document.querySelector('#absorption').textContent='0.000';
    }""")
    return page, frame


def test_one_exact_copy_controlled_rounding_and_dependent_readout(habitat_page, tmp_path):
    page, frame = habitat_page
    session = HabitabilityNumericSession(page, config(), tmp_path / "copy")
    try:
        receipt = session.copy("759.431827", "K", source="reference_diagnostic")
        assert receipt["readback_verified"] and not receipt["correctness_verified"]
        assert receipt["display"]["display_value"] == "759.4"
        assert receipt["surface_display"]["format"] == "exact"
        assert not receipt["task_completed"] and not page.get_by_role("checkbox").is_checked()
        assert frame.locator("#greenhouse").input_value() == ""
        with pytest.raises(BrowserSafetyStop, match="already_attempted"):
            session.copy("759.431827", "K", source="reference_diagnostic")
    finally:
        session.close()


@pytest.mark.parametrize("text,unit", [("0", "K"), ("nan", "K"), ("10+20", "K"), ("-1", "K"), ("300", "C")])
def test_bad_temperature_request_cannot_write(habitat_page, tmp_path, text, unit):
    page, frame = habitat_page
    session = HabitabilityNumericSession(page, config(), tmp_path / "copy")
    try:
        with pytest.raises(BrowserSafetyStop):
            session.copy(text, unit, source="checkpoint")
        assert not session.attempted and frame.locator("#temperature").input_value() == "0"
    finally:
        session.close()


@pytest.mark.parametrize("mutation", ["outer", "answer", "replace", "feedback", "modal", "auth"])
def test_changed_context_stops_before_temperature_write(habitat_page, tmp_path, mutation):
    page, frame = habitat_page
    session = HabitabilityNumericSession(page, config(), tmp_path / "copy")
    try:
        if mutation == "outer":
            page.get_by_role("checkbox").check()
        elif mutation == "answer":
            frame.locator("#phase").select_option(label="Liquid")
        elif mutation == "replace":
            frame.locator("#temperature").evaluate("e=>e.replaceWith(e.cloneNode(true))")
        else:
            frame.locator("body").evaluate(
                "(e,html)=>e.insertAdjacentHTML('beforeend',html)",
                {
                    "feedback": "<div>Unexpected feedback</div>",
                    "modal": "<div role=dialog>Stop</div>",
                    "auth": "<input type=password>",
                }[mutation],
            )
        with pytest.raises(BrowserSafetyStop):
            session.copy("300", "K", source="checkpoint")
        assert session.stopped and not session.attempted
        assert frame.locator("#temperature").input_value() == "0"
    finally:
        session.close()


@pytest.mark.parametrize(
    "mutation", ["surface", "rounding", "phase", "absorption", "feedback", "dialog", "replace"]
)
def test_uncertain_copy_is_reserved_and_never_retried(habitat_page, tmp_path, mutation):
    page, frame = habitat_page
    session = HabitabilityNumericSession(page, config(), tmp_path / "copy")
    code = {
        "surface": "document.querySelector('#surface').textContent='999'",
        "rounding": "e.value='999';document.querySelector('#surface').textContent='999'",
        "phase": "document.querySelector('#phase').value='Gas'",
        "absorption": "document.querySelector('#absorption').textContent='4'",
        "feedback": "document.body.append('Unexpected feedback')",
        "dialog": "confirm('private confirmation')",
        "replace": "e.replaceWith(e.cloneNode(true))",
    }[mutation]
    frame.locator("#temperature").evaluate(
        "(e,code)=>{const prior=e.onblur;e.onblur=()=>{prior();new Function('e',code)(e)}}", code
    )
    try:
        with pytest.raises(BrowserSafetyStop):
            session.copy("759.431827", "K", source="checkpoint")
        assert session.stopped and session.attempted
        assert (tmp_path / "copy/reserved.json").exists()
        assert not (tmp_path / "copy/confirmed.json").exists()
        failure = (tmp_path / "copy/stopped.json").read_text()
        assert not json.loads(failure)["retry_allowed"] and "private confirmation" not in failure
        with pytest.raises(BrowserSafetyStop):
            session.copy("300", "K", source="checkpoint")
    finally:
        session.close()


def test_preexisting_work_is_not_overwritten(habitat_page, tmp_path):
    page, frame = habitat_page
    frame.locator("#temperature").fill("250")
    with pytest.raises(BrowserSafetyStop, match="already_populated"):
        HabitabilityNumericSession(page, config(), tmp_path / "copy")
    assert frame.locator("#temperature").input_value() == "250"


def test_known_save_busy_state_is_not_an_answer_change(habitat_page, tmp_path):
    page, frame = habitat_page
    session = HabitabilityNumericSession(page, config(), tmp_path / "copy")
    frame.get_by_role("button", name="Save", exact=True).evaluate("e=>e.disabled=true")
    try:
        receipt = session.copy("300", "K", source="checkpoint")
        assert receipt["readback_verified"]
    finally:
        session.close()
