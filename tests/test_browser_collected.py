"""Collected-row navigation fixtures; every browser request is intercepted."""
# ruff: noqa: F811

import json

import pytest
from test_browser_full_stellar import full_html
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_collected import open_blank_collected_star
from habfly.browser_stellar import SIMULATION_URL


@pytest.fixture
def collected(page):
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("iframe").first.evaluate("e=>e.style.cssText='width:950px;height:600px;border:0'")
    frame = page.frame(url=SIMULATION_URL)
    row = lambda name: (
        '<div style="display:flex;height:33px;width:900px">'
        '<svg role="img" style="display:block;width:22px;height:25px;flex:none" class="eye"><rect width="22" height="25"/></svg>'
        '<div style="display:flex"><div><div>' + name + "</div></div>"
        "<div> 0.045 370 5.68E-10 - - - - - - - -</div></div>"
        '<span role="img" style="display:block;width:22px;height:25px">Delete</span></div>'
    )
    frame.set_content(
        "<div>FUNDING TOTAL COLLECTED ANALYZED DATA</div>" + row("Previous") + row("Althinagon")
    )
    frame.locator(".eye").nth(1).evaluate("(e,html)=>e.onclick=()=>document.body.innerHTML=html", full_html())
    return page, frame


def test_opens_only_named_blank_row_without_answer_write(collected, tmp_path):
    page, _ = collected
    receipt = open_blank_collected_star(page, config(), tmp_path / "open", star="ALTHINAGON")
    assert receipt["star"] == "Althinagon" and receipt["navigation_clicks"] == 1
    assert receipt["answer_writes"] == 0 and receipt["painted_stellar_class"] is None
    assert not receipt["task_completed"] and not page.get_by_role("checkbox").is_checked()


@pytest.mark.parametrize(
    "bad", ["duplicate", "filled", "obscured", "wrong_measurement", "wrong_star", "outer_change"]
)
def test_navigation_rejects_ambiguous_or_changed_evidence(collected, tmp_path, bad):
    page, frame = collected
    if bad == "duplicate":
        frame.locator("body").evaluate("e=>e.insertAdjacentHTML('beforeend','<div>Althinagon</div>')")
    elif bad == "filled":
        frame.get_by_text("0.045 370 5.68E-10 - - - - - - - -", exact=True).nth(1).evaluate(
            "e=>e.innerText='0.045 370 5.68E-10 77 - - - - - - -'"
        )
    elif bad == "obscured":
        frame.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<div style=\"position:fixed;inset:0;z-index:10\">Overlay</div>')"
        )
    elif bad in {"wrong_measurement", "wrong_star"}:
        html = (
            full_html().replace("0.045", "0.046")
            if bad == "wrong_measurement"
            else full_html().replace("Althinagon", "Previous")
        )
        frame.locator(".eye").nth(1).evaluate("(e,html)=>e.onclick=()=>document.body.innerHTML=html", html)
    else:
        page.evaluate(
            "addEventListener('message',e=>{if(e.data==='fixture-navigation')document.querySelector('input').checked=true})"
        )
        frame.locator(".eye").nth(1).evaluate(
            "(e,html)=>e.onclick=()=>{document.body.innerHTML=html;parent.postMessage('fixture-navigation','*')}",
            full_html(),
        )
    with pytest.raises(BrowserSafetyStop):
        open_blank_collected_star(page, config(), tmp_path / "open", star="Althinagon")
    assert not (tmp_path / "open/confirmed.json").exists()
    if bad in {"wrong_measurement", "wrong_star", "outer_change"}:
        assert json.loads((tmp_path / "open/stopped.json").read_text())["click_may_have_occurred"]
    else:
        assert not (tmp_path / "open/reserved.json").exists()


def test_retained_paint_recorded_as_unverified(collected, tmp_path):
    page, frame = collected
    frame.locator(".eye").nth(1).evaluate(
        "(e,html)=>e.onclick=()=>document.body.innerHTML=html",
        full_html().replace("<label onclick=", '<label class="selected" onclick=', 1),
    )
    receipt = open_blank_collected_star(page, config(), tmp_path / "open", star="Althinagon")
    assert receipt["painted_stellar_class"] == "main_sequence" and not receipt["class_selection_verified"]
