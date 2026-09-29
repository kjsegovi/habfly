"""Intercepted Chromium fixtures: no live website or scientific answer oracle."""
# ruff: noqa: F811

import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_numeric import ANSWER_UNITS, PlanetNumericSession
from habfly.browser_stellar import SIMULATION_URL


def planet_html():
    fields = {
        "line_shift": "doppler shift (nm)",
        "brightness_drop": "brightness drop (%)",
        "period_days": "brightness drop period (days)",
        "orbital_radius": "orbital radius (au)",
        "planet_mass": "mass (M<sub>E</sub>)",
        "planet_radius": "radius (R<sub>E</sub>)",
        "planet_density": "density (g/cm<sup>3</sup>)",
    }
    answer = lambda key: f'<span>{fields[key]}</span><input id="{key}" placeholder="0">'
    choices = "".join(
        f'<div class="choice"><label></label><div>{n}</div></div>'
        for n in ("gas giant", "ice giant", "terrestrial")
    )
    return (
        '<div>JYREMIS</div><img alt="1"><div>Observations Spectrum</div><button>+</button><button>-</button>'
        '<span>656.3nm 656.299997nm 656.300003nm observe for</span><input placeholder="0" value="10000">'
        '<span>days</span><button>Play</button><svg role="img" width="280" height="30"><text y="20">Normalized Flux Days Observed 1 2 3</text></svg><button>Zoom</button>'
        + "".join(answer(k) for k in ("line_shift", "brightness_drop", "period_days"))
        + "<div>Your Reconstruction has planet?</div><select><option selected>Yes</option><option>No</option></select>"
        + "".join(answer(k) for k in ("orbital_radius", "planet_mass", "planet_radius", "planet_density"))
        + choices
        + '<div id="derived">STAR MASS (Ms) 2.368 STAR RADIUS (Rs) 1.961 ORBIT (years) 0.000</div><button>Save</button>'
        + '<style>.choice{display:inline-block;width:120px;text-align:center}.choice label{display:block;position:relative;width:26px;height:26px;border:1px solid rgb(0,200,220);margin:auto}.choice label::after{content:"";position:absolute;display:block;width:14px;height:14px;opacity:0;background:transparent;left:6px;top:6px}.choice>div{margin-top:3px;font-size:10px}</style>'
    )


@pytest.fixture
def planet_page(page):
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("iframe").first.evaluate("e=>{e.style.width='1200px';e.style.height='800px'}")
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.locator("body").evaluate("(e,html)=>e.innerHTML=html", planet_html())
    return page, frame


def test_all_seven_exact_copies_and_offline_receipts(planet_page, tmp_path):
    page, frame = planet_page
    events = []
    session = PlanetNumericSession(page, config(), tmp_path / "copy", lambda *e: events.append(e))
    for name, unit in ANSWER_UNITS.items():
        result = session.copy(name, "0.123456789", unit, source="checkpoint")
        assert result["display"]["format"] == "exact"
        assert result["readback_verified"] and not result["correctness_verified"]
        assert not result["task_completed"]
        assert frame.locator("#" + name).input_value() == "0.123456789"
    assert len(session.verified) == 7 and len(events) == 14
    for path in (tmp_path / "copy").glob("*-confirmed.json"):
        assert json.loads(path.read_text())["readback_verified"]
    assert not page.get_by_role("checkbox").is_checked()
    with pytest.raises(BrowserSafetyStop, match="already_populated"):
        session.copy("line_shift", "5", "nm", source="checkpoint")


@pytest.mark.parametrize("kind", ["unit", "zero", "nonfinite", "code", "depth", "duration"])
def test_bad_requests_do_not_write(planet_page, tmp_path, kind):
    page, frame = planet_page
    session = PlanetNumericSession(page, config(), tmp_path / "copy")
    requests = {
        "unit": ("planet_mass", "1", "Msun"),
        "zero": ("line_shift", "0", "nm"),
        "nonfinite": ("line_shift", "nan", "nm"),
        "code": ("line_shift", "1+2", "nm"),
        "depth": ("brightness_drop", "101", "%"),
        "duration": ("observation_days", "100", "day"),
    }
    with pytest.raises(BrowserSafetyStop):
        session.copy(*requests[kind], source="reference_diagnostic")
    assert not session.attempted
    assert frame.locator("#line_shift").input_value() == ""


@pytest.mark.parametrize("kind", ["answer", "external", "star", "replacement", "feedback", "modal", "auth"])
def test_stale_or_unsafe_context_rejected_before_writing(planet_page, tmp_path, kind):
    page, frame = planet_page
    session = PlanetNumericSession(page, config(), tmp_path / "copy")
    if kind == "external":
        page.get_by_role("checkbox").check()
    elif kind == "answer":
        frame.locator("#period_days").fill("5")
    elif kind == "replacement":
        frame.locator("#line_shift").evaluate("e=>e.replaceWith(e.cloneNode(true))")
    elif kind == "star":
        frame.locator("div").first.evaluate("e=>e.innerText='OTHERSTAR'")
    else:
        html = {
            "feedback": "<div>unexpected failure</div>",
            "modal": "<div role=dialog>stop</div>",
            "auth": "<input type=password>",
        }[kind]
        frame.locator("body").evaluate("(e,html)=>e.insertAdjacentHTML('beforeend',html)", html)
    with pytest.raises(BrowserSafetyStop):
        session.copy("line_shift", "0.001", "nm", source="checkpoint")
    assert session.stopped and not session.attempted
    assert frame.locator("#line_shift").input_value() == ""


@pytest.mark.parametrize("kind", ["other_answer", "feedback", "rounded_wrong", "replacement"])
def test_uncertain_write_stops_and_retains_reservation(planet_page, tmp_path, kind):
    page, frame = planet_page
    session = PlanetNumericSession(page, config(), tmp_path / "copy")
    body = {
        "other_answer": "document.querySelector('#period_days').value='5'",
        "feedback": "document.body.append('unexpected failure')",
        "rounded_wrong": "e.value='8'",
        "replacement": "e.replaceWith(e.cloneNode(true))",
    }[kind]
    frame.locator("#line_shift").evaluate('(e,body)=>e.onblur=()=>{new Function("e",body)(e)}', body)
    with pytest.raises(BrowserSafetyStop):
        session.copy("line_shift", "0.00123456", "nm", source="checkpoint")
    assert session.stopped and session.attempted == {"line_shift"} and not session.verified
    assert (tmp_path / "copy/copy-01-reserved.json").exists()
    assert (tmp_path / "copy/copy-01-stopped.json").exists()
    with pytest.raises(BrowserSafetyStop, match="stopped"):
        session.copy("line_shift", "0.00123456", "nm", source="checkpoint")


def test_known_rounding_and_derived_orbit_are_not_corrections(planet_page, tmp_path):
    page, frame = planet_page
    session = PlanetNumericSession(page, config(), tmp_path / "copy")
    frame.locator("#orbital_radius").evaluate("""e=>e.onblur=()=>{
        e.value='0.5919';document.querySelector('#derived').innerText='STAR MASS (Ms) 2.368 STAR RADIUS (Rs) 1.961 ORBIT (years) 0.2959';
    }""")
    result = session.copy("orbital_radius", "0.5918538722607202", "au", source="reference_diagnostic")
    assert result["display"]["format"] == "decimal_rounding"
    assert result["display"]["display_value"] == "0.5919"
    assert not result["correctness_verified"]


def test_copy_deadline_is_enforced_without_a_write(planet_page, tmp_path):
    page, frame = planet_page
    session = PlanetNumericSession(page, config(), tmp_path / "copy")
    session.deadline = 0
    with pytest.raises(BrowserSafetyStop, match="time_limit"):
        session.copy("line_shift", "0.001", "nm", source="checkpoint")
    assert session.stopped and frame.locator("#line_shift").input_value() == ""


@pytest.mark.parametrize("orbit,valid", [("5.551", True), ("5.553", False), ("0", False), ("-1", False)])
def test_planet_mass_can_reduce_only_dependent_positive_orbit(planet_page, tmp_path, orbit, valid):
    page, frame = planet_page
    frame.locator("#derived").evaluate("e=>e.innerText=e.innerText.replace('0.000','5.552')")
    session = PlanetNumericSession(page, config(), tmp_path / "copy")
    frame.locator("#planet_mass").evaluate(
        """(e,orbit)=>e.onblur=()=>{
        e.value='57.35';document.querySelector('#derived').innerText=
          'STAR MASS (Ms) 2.368 STAR RADIUS (Rs) 1.961 ORBIT (years) '+orbit;
    }""",
        orbit,
    )
    if valid:
        receipt = session.copy("planet_mass", "57.354917176709556", "MEarth", source="checkpoint")
        assert receipt["readback_verified"] and not receipt["correctness_verified"]
    else:
        with pytest.raises(BrowserSafetyStop, match="dependent_orbit"):
            session.copy("planet_mass", "57.354917176709556", "MEarth", source="checkpoint")
        assert not session.verified and session.stopped
    session.close()


def test_same_url_external_frame_replacement_stops(planet_page, tmp_path):
    page, frame = planet_page
    session = PlanetNumericSession(page, config(), tmp_path / "copy")
    page.locator("iframe").nth(1).evaluate("e=>e.replaceWith(e.cloneNode(true))")
    with pytest.raises(BrowserSafetyStop, match="frame_set"):
        session.copy("line_shift", "0.001", "nm", source="checkpoint")
    assert session.stopped and frame.locator("#line_shift").input_value() == ""


def test_javascript_confirmation_is_dismissed_and_uncertain_copy_never_retried(planet_page, tmp_path):
    page, frame = planet_page
    session = PlanetNumericSession(page, config(), tmp_path / "copy")
    frame.locator("#line_shift").evaluate("e=>e.onblur=()=>confirm('private confirmation')")
    with pytest.raises(BrowserSafetyStop, match="unexpected_browser_dialog"):
        session.copy("line_shift", "0.001", "nm", source="checkpoint")
    assert session.stopped and session.attempted == {"line_shift"} and not session.verified
    assert "private confirmation" not in (tmp_path / "copy/copy-01-stopped.json").read_text()
    session.close()
