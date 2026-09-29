"""Revisit fixtures use intercepted local pages, never the live activity."""
# ruff: noqa: F811

import json

import pytest
from test_browser_full_stellar import full_html
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_collected_revisit import (
    collected_status_projection,
    parse_stellar_row,
    reopen_collected_stellar,
)
from habfly.browser_stellar import SIMULATION_URL

HEADER = """FUNDING TOTAL COLLECTED ANALYZED DATA
Parallax (") Peak λ (nm) Flux (W/m2) Distance (ly) Luminosity (Ls) Peak λ color
Temp. (K) Classification Mass (Ms) Radius (Rs) Lifetime (years)"""
ROW = "Althinagon 0.045 370 5.68E-10 72.4 8.67 UV 7830 Main Sequence 1.71 1.6 2610 Ma"


@pytest.mark.parametrize("notice_in", ["text", "accessibility", "both"])
def test_only_exact_transient_list_footer_is_ignored(notice_in):
    from copy import deepcopy

    report = {
        "frames": [
            {
                "url": SIMULATION_URL,
                "text": "UNCHANGED ROW\n \n \nVIEWING\n1-5 OF 5\n \nTOTAL COLLECTED\n5\nSave",
                "accessibility": '- text: UNCHANGED ROW\n- img\n- img\n- button [disabled]\n- button [disabled]\n- text: viewing 1-5 of 5 total collected 5\n- button "Save"',
                "controls": [{"role": "button", "accessibility": '- button "Save"', "enabled": True}],
            }
        ]
    }
    before = deepcopy(report)
    if notice_in in {"text", "both"}:
        report["frames"][0]["text"] = report["frames"][0]["text"].replace("ROW\n", "ROW\nData saved\n")
    if notice_in in {"accessibility", "both"}:
        report["frames"][0]["accessibility"] = report["frames"][0]["accessibility"].replace(
            "- button [disabled]", "- text: Data saved\n- button [disabled]", 1
        )
    assert collected_status_projection(report) == before
    assert report != before
    changed = deepcopy(report)
    changed["frames"][0]["text"] = changed["frames"][0]["text"].replace("UNCHANGED ROW", "CHANGED ROW")
    assert collected_status_projection(changed) != before
    changed = deepcopy(before)
    changed["frames"][0]["text"] = "Data saved\n" + changed["frames"][0]["text"]
    assert collected_status_projection(changed) != before


def test_foreign_frame_footer_is_not_suppressed():
    report = {
        "frames": [
            {
                "url": "https://other.invalid",
                "text": "Data saved",
                "accessibility": "- text: Data saved",
                "controls": [],
            }
        ]
    }
    assert collected_status_projection(report) == report


def test_row_parser_preserves_actual_answers_without_scientific_correction():
    evidence = parse_stellar_row(ROW, "Althinagon")
    assert evidence["fields"]["luminosity"] == "8.67"
    assert evidence["color"] == "UV" and evidence["lifetime_prefix"] == "Ma"
    assert evidence["classification"] == "main_sequence"


@pytest.mark.parametrize(
    "bad",
    [
        ROW.replace("8.67", "NaN"),
        ROW.replace("0.045", "0"),
        ROW.replace("1.71", "-1"),
        ROW.replace(" Ma", ""),
        ROW.replace("Ma", "years"),
        ROW.replace("Main Sequence", "Red Giant"),
        ROW + " Unexpected",
        ROW.replace("7830", "0 Infinity"),
    ],
)
def test_invalid_row_contract(bad):
    with pytest.raises(BrowserSafetyStop):
        parse_stellar_row(bad, "Althinagon")


def test_non_main_and_blank_rows():
    for kind in ("White Dwarf", "Red Giant", "Supergiant", "-"):
        row = f"White Star 0.045 370 5.68E-10 72.4 8.67 UV 7830 {kind} - - -"
        assert len(parse_stellar_row(row, "White Star")["fields"]) == 3
    blank = parse_stellar_row("Test 1 2 3 - - - - - - - -", "Test")
    assert blank["fields"] == {"distance": "", "luminosity": "", "temperature": ""}


@pytest.fixture
def revisit_page(page):
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("iframe").first.evaluate("e=>e.style.cssText='width:950px;height:600px;border:0'")
    frame = page.frame(url=SIMULATION_URL)
    frame.set_content(full_html())
    frame.evaluate("choose(document.querySelector('.choice label'),true)")
    values = {
        "distance": "72.4",
        "luminosity": "8.67",
        "temperature": "7830",
        "mass": "1.71",
        "radius": "1.6",
        "lifetime": "2610",
    }
    for name, value in values.items():
        frame.locator(f"#{name}").evaluate("(e,v)=>{e.value=v;e.setAttribute('value',v)}", value)
    frame.locator("select").evaluate_all(
        "es=>{for(const e of es){for(const o of e.options){o.removeAttribute('selected')}const label=e.id==='prefix'?'Ma':'UV';[...e.options].find(o=>o.text===label).setAttribute('selected','')}}"
    )
    html = frame.locator("body").inner_html()

    def row(name, numbers):
        return (
            '<div style="display:flex;height:33px;width:900px">'
            '<svg role="img" style="display:block;width:22px;height:25px;flex:none" class="eye"><rect width="22" height="25"/></svg>'
            f'<div style="display:flex"><div><div>{name}</div></div><div> {numbers}</div></div>'
            '<span role="img" style="display:block;width:22px;height:25px">Delete</span></div>'
        )

    frame.set_content(
        f"<div>{HEADER}</div>"
        + row("Previous", "1 2 3 - - - - - - - -")
        + row("Althinagon", ROW.split(" ", 1)[1])
    )
    frame.locator(".eye").nth(0).evaluate("e=>e.onclick=()=>{window.wrongEye=true}")
    frame.locator(".eye").nth(1).evaluate(
        "(e,html)=>e.onclick=()=>{window.eyeClicks=(window.eyeClicks||0)+1;document.body.innerHTML=html}",
        html,
    )
    return page, frame, html


def test_reopen_populated_star_has_one_navigation_and_zero_writes(revisit_page, tmp_path):
    page, frame, _ = revisit_page
    result = reopen_collected_stellar(page, config(), tmp_path / "open", star="ALTHINAGON")
    assert result["row_detail_values_verified"] and not result["classification_correctness_verified"]
    assert result["navigation_clicks"] == 1 and result["answer_writes"] == 0
    assert not result["task_completed"] and frame.evaluate("window.eyeClicks") == 1
    assert not frame.evaluate("!!window.wrongEye")
    assert frame.locator("#luminosity").input_value() == "8.67"
    assert frame.locator("#prefix").input_value() == "Ma"
    assert not page.get_by_role("checkbox").is_checked()


@pytest.mark.parametrize(
    "mutation", ["answer", "star", "measurement", "color", "prefix", "class", "outer", "dialog"]
)
def test_changed_detail_is_not_repaired_or_retried(revisit_page, tmp_path, mutation):
    page, frame, html = revisit_page
    changes = {
        "answer": "document.querySelector('#luminosity').value='8.68'",
        "star": "document.body.firstElementChild.textContent='Otherstar'",
        "measurement": "document.body.innerHTML=document.body.innerHTML.replace('0.045','0.046')",
        "color": "document.querySelector('select').value='IR'",
        "prefix": "document.querySelector('#prefix').value='Ga'",
        "class": "document.querySelector('.selected').classList.remove('selected')",
        "outer": "parent.postMessage('fixture-outer-change','*')",
        "dialog": "confirm('private message')",
    }
    page.evaluate(
        "addEventListener('message',e=>{if(e.data==='fixture-outer-change')document.querySelector('input').checked=true})"
    )
    frame.locator(".eye").nth(1).evaluate(
        "(e,arg)=>e.onclick=()=>{window.eyeClicks=(window.eyeClicks||0)+1;document.body.innerHTML=arg.html;new Function(arg.code)()}",
        {"html": html, "code": changes[mutation]},
    )
    with pytest.raises(BrowserSafetyStop):
        reopen_collected_stellar(page, config(), tmp_path / "open", star="Althinagon")
    failure = (tmp_path / "open/stopped.json").read_text()
    assert json.loads(failure)["click_may_have_occurred"] and "private message" not in failure
    assert frame.evaluate("window.eyeClicks") == 1
    assert not (tmp_path / "open/confirmed.json").exists()


@pytest.mark.parametrize("mutation", ["duplicate", "obscured", "header", "duplicate_eye", "bad_number"])
def test_uncertain_row_is_not_opened(revisit_page, tmp_path, mutation):
    page, frame, _ = revisit_page
    code = {
        "duplicate": "document.body.insertAdjacentHTML('beforeend','<div>Althinagon</div>')",
        "obscured": "document.body.insertAdjacentHTML('beforeend','<div style=\"position:fixed;inset:0;z-index:10\">Overlay</div>')",
        "header": "document.body.firstElementChild.textContent=document.body.firstElementChild.textContent.replace('Distance (ly)','Distance (pc)')",
        "duplicate_eye": "const eye=document.querySelectorAll('.eye')[1];eye.after(eye.cloneNode(true))",
        "bad_number": "document.body.innerHTML=document.body.innerHTML.replace('1.71','NaN')",
    }[mutation]
    frame.evaluate(code)
    with pytest.raises(BrowserSafetyStop):
        reopen_collected_stellar(page, config(), tmp_path / "open", star="Althinagon")
    assert not frame.evaluate("!!window.eyeClicks") and not (tmp_path / "open/reserved.json").exists()
