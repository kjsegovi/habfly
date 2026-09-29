"""Reference hypotheses stay distinct from learned decisions and proven absence."""
# ruff: noqa: F811

import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_numeric import planet_page  # noqa: F401
from test_browser_planet_presence import presence_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_absence import select_no_planet_hypothesis
from habfly.planet_charts import FluxSample, analyze_flux_samples


@pytest.fixture
def absence_page(presence_page):
    page, frame = presence_page
    frame.locator("body").evaluate("""e=>{
        const span=[...e.querySelectorAll('span')].find(s=>s.textContent.startsWith('656.3nm'));
        span.outerHTML=`<div style='position:relative;width:285px;height:60px'>
            <div style='position:relative;width:285px;height:40px;background:rgb(193,58,44)'>
                <div id='centerline' style='position:absolute;left:140px;width:2px;height:38px;background:black'></div>
            </div><div>656.3nm</div></div><span>656.299997nm 656.300003nm observe for</span>`;
        const menu=document.querySelector('select');
        menu.onchange=()=>{window.presenceWrites=(window.presenceWrites||0)+1;menu.nextElementSibling.hidden=menu.value!=='Yes'};
    }""")
    return page, frame


def scan(tmp_path, *, count=1000, dip=False, star="JYREMIS"):
    samples = [
        FluxSample(day=i, brightness_percent="99" if dip and i == 500 else "100") for i in range(count)
    ]
    report = {
        **analyze_flux_samples(samples),
        "star": star,
        "samples": [s.model_dump() for s in samples],
        "sampling_method": "scripted_bounded_visible_tooltips",
        "learned_chart_perception": False,
        "task_completed": False,
        "period_evidence_verified": False,
        "time_axis_validation": "each_hover_against_visible_tick_glyphs",
    }
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(report))
    return path


RATIONALE = "Explicit fixture hypothesis from limited flat readings; not a proof, trained prediction or validated label."


def test_reference_no_has_no_zero_numeric_answers_or_completion_claim(absence_page, tmp_path):
    page, frame = absence_page
    result = select_no_planet_hypothesis(
        page, config(), tmp_path / "run", transit_path=scan(tmp_path), rationale=RATIONALE
    )
    assert result["readback_verified"] and result["value"] == "No"
    assert not result["absence_proven"] and not result["correctness_verified"]
    assert not result["training_label"] and not result["task_completed"]
    assert result["observed_unique_days"] == 1000 and result["observed_day_max"] == 999
    assert not result["spectrum"]["zero_amplitude_measured"]
    assert frame.evaluate("window.presenceWrites") == 1
    assert frame.locator("#line_shift").input_value() == ""
    assert not page.get_by_role("checkbox").is_checked()
    with pytest.raises(BrowserSafetyStop):
        select_no_planet_hypothesis(
            page, config(), tmp_path / "again", transit_path=tmp_path / "scan.json", rationale=RATIONALE
        )
    assert frame.evaluate("window.presenceWrites") == 1


@pytest.mark.parametrize("bad", ["dip", "short", "other_star", "answer", "markers", "covered", "invalidated"])
def test_incompatible_evidence_never_writes(absence_page, tmp_path, bad):
    page, frame = absence_page
    source = scan(
        tmp_path,
        count=999 if bad == "short" else 1000,
        dip=bad == "dip",
        star="OTHER" if bad == "other_star" else "JYREMIS",
    )
    if bad == "answer":
        frame.locator("#period_days").fill("5")
    elif bad == "markers":
        frame.locator("#centerline").evaluate(
            "e=>{const marker=e.cloneNode();marker.style.width='1px';marker.style.left='30px';e.after(marker)}"
        )
    elif bad == "covered":
        frame.locator("#centerline").evaluate(
            "e=>e.parentElement.insertAdjacentHTML('beforeend','<div style=\"position:absolute;inset:0;background:red\"></div>')"
        )
    elif bad == "invalidated":
        (tmp_path / "invalidated.json").write_text("{}")
    with pytest.raises(BrowserSafetyStop):
        select_no_planet_hypothesis(
            page, config(), tmp_path / "run", transit_path=source, rationale=RATIONALE
        )
    assert not frame.evaluate("!!window.presenceWrites")
    assert frame.get_by_role("combobox").input_value() == ""


@pytest.mark.parametrize("mutation", ["answer", "feedback", "replacement", "dialog"])
def test_uncertain_no_is_not_rolled_back_or_retried(absence_page, tmp_path, mutation):
    page, frame = absence_page
    code = {
        "answer": "document.querySelector('#period_days').value='3'",
        "feedback": "document.body.append('unexpected feedback')",
        "replacement": "e.replaceWith(e.cloneNode(true))",
        "dialog": "confirm('private dialog content')",
    }[mutation]
    frame.get_by_role("combobox").evaluate(
        "(e,code)=>{const old=e.onchange;e.onchange=()=>{old();new Function('e',code)(e)}}", code
    )
    with pytest.raises(BrowserSafetyStop):
        select_no_planet_hypothesis(
            page, config(), tmp_path / "run", transit_path=scan(tmp_path), rationale=RATIONALE
        )
    failure = (tmp_path / "run/stopped.json").read_text()
    assert json.loads(failure)["write_may_have_occurred"] and "private dialog" not in failure
    assert frame.evaluate("window.presenceWrites") == 1
    assert not (tmp_path / "run/confirmed.json").exists()
