"""Disposable gas UI; injected spectra/readouts are fixture data, not science."""
# ruff: noqa: F811

import json

import pytest
from test_browser_habitability_numeric import habitat_page  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_gas_controls import (
    GasComparisonSession,
    compare_visible_gas_candidates,
    select_reference_gases,
)
from habfly.browser_habitability import GASES


@pytest.fixture
def gas_page(habitat_page):
    page, frame = habitat_page
    frame.locator("#gases").evaluate("""e=>{
        const p=document.createElement('div');p.style='position:relative;width:180px;height:28px';
        e.before(p);p.append(e);e.style='width:180px;height:28px';
        const overlay=document.createElement('div');overlay.style='position:absolute;inset:0';p.append(overlay);
        const choices=document.createElement('span');choices.hidden=true;p.after(choices);
        overlay.onclick=()=>{choices.hidden=false};
        for(const name of ['CH4','CO2','H2O','H2S','N2O','NH3','O3']){
            const label=document.createElement('label');label.style='margin-right:8px';
            const c=document.createElement('input');c.type='checkbox';c.setAttribute('aria-label',name);
            label.append(c,document.createTextNode(name));choices.append(label);
            c.onchange=()=>{
                window.gasWrites=(window.gasWrites||0)+1;
                const checked=[...choices.querySelectorAll('input')].filter(c=>c.checked).map(c=>c.getAttribute('aria-label'));
                e.replaceChildren(new Option(checked.join(', '),checked.join(', '),true,true));
                document.querySelector('#absorption').textContent=String(checked.length*10);
            };
        }
        const image=document.querySelector('svg').cloneNode(true);
        image.style='position:absolute;left:550px;top:20px';document.body.append(image);
    }""")
    return page, frame


def test_candidate_comparison_restores_all_gases_without_choosing_answer(gas_page, tmp_path):
    page, frame = gas_page
    result = compare_visible_gas_candidates(page, config(), tmp_path / "run", candidates=list(GASES))
    assert result["baseline_restored"] and result["checkbox_writes"] == 14
    assert not result["gas_selection_inferred"] and not result["task_completed"]
    assert frame.evaluate("window.gasWrites") == 14
    assert frame.locator("#gases").input_value() == ""
    assert frame.locator("#temperature").input_value() == "0"
    assert frame.locator("#greenhouse").input_value() == ""
    assert not page.get_by_role("checkbox").is_checked()
    assert len(list((tmp_path / "run").glob("*.png"))) == 8
    assert len(list((tmp_path / "run").glob("write-*-confirmed.json"))) == 14


def test_wrong_explicit_gas_is_not_repaired_or_assessed(gas_page, tmp_path):
    page, frame = gas_page
    session = GasComparisonSession(page, config(), tmp_path / "run", max_writes=1)
    try:
        session.open_menu()
        receipt = session.toggle("NH3", True)
        assert receipt["selected"] == ["NH3"] and not receipt["correctness_verified"]
        assert frame.get_by_role("checkbox", name="NH3", exact=True).is_checked()
        assert receipt["absorption"]["value"] == 10
        with pytest.raises(BrowserSafetyStop, match="budget"):
            session.toggle("NH3", False)
        assert frame.evaluate("window.gasWrites") == 1
    finally:
        session.close()


@pytest.mark.parametrize(
    "mutation", ["temperature", "greenhouse", "feedback", "gas_feedback", "outer", "replace"]
)
def test_stale_context_cannot_toggle(gas_page, tmp_path, mutation):
    page, frame = gas_page
    session = GasComparisonSession(page, config(), tmp_path / "run")
    session.open_menu()
    if mutation == "outer":
        page.get_by_role("checkbox").check()
    else:
        frame.locator("body").evaluate(
            "(e,code)=>new Function(code)()",
            {
                "temperature": "document.querySelector('#temperature').value='100'",
                "greenhouse": "document.querySelector('#greenhouse').value='Weak'",
                "feedback": "document.body.append('Unexpected message')",
                "gas_feedback": "document.querySelector('#gases').parentElement.after(document.createTextNode('Unexpected message'))",
                "replace": "const e=document.querySelector('input[type=checkbox]');e.replaceWith(e.cloneNode(true))",
            }[mutation],
        )
    try:
        with pytest.raises(BrowserSafetyStop):
            session.toggle("CH4", True)
        assert not frame.get_by_role("checkbox", name="CH4", exact=True).is_checked()
        assert session.writes == 0
    finally:
        session.close()


@pytest.mark.parametrize("mutation", ["temperature", "extra_gas", "feedback", "dialog", "replace"])
def test_failed_write_is_preserved_and_never_automatically_restored(gas_page, tmp_path, mutation):
    page, frame = gas_page
    session = GasComparisonSession(page, config(), tmp_path / "run")
    session.open_menu()
    target = frame.get_by_role("checkbox", name="CH4", exact=True)
    target.evaluate(
        "(e,code)=>{let prior=e.onchange;e.onchange=()=>{prior();new Function('e',code)(e)}}",
        {
            "temperature": "document.querySelector('#temperature').value='100'",
            "extra_gas": "document.querySelector('input[aria-label=CO2]').checked=true",
            "feedback": "document.body.append('Unexpected message')",
            "dialog": "confirm('private fixture prompt')",
            "replace": "e.replaceWith(e.cloneNode(true))",
        }[mutation],
    )
    try:
        with pytest.raises((BrowserSafetyStop, ValueError)):
            session.toggle("CH4", True)
        assert session.stopped and session.writes == 1
        assert frame.evaluate("window.gasWrites") == 1
        failure = json.loads((tmp_path / "run/write-01-stopped.json").read_text())
        assert failure["write_may_have_occurred"] and not failure["automatic_retry"]
        assert "private fixture" not in json.dumps(failure)
        with pytest.raises((BrowserSafetyStop, ValueError)):
            session.toggle("CH4", False)
        assert frame.evaluate("window.gasWrites") == 1
    finally:
        session.close()


@pytest.mark.parametrize("bad", [[], ["CO2", "CO2"], ["O2"], "CO2"])
def test_invalid_comparison_request_never_opens_browser(tmp_path, bad):
    with pytest.raises(ValueError):
        compare_visible_gas_candidates(None, None, tmp_path / "run", candidates=bad)
    assert not (tmp_path / "run").exists()


def test_explicit_combination_preserves_reference_not_learned_status(gas_page, tmp_path):
    page, frame = gas_page
    comparison = tmp_path / "comparison"
    compare_visible_gas_candidates(page, config(), comparison, candidates=["CO2", "N2O"])
    result = select_reference_gases(
        page,
        config(),
        tmp_path / "selected",
        comparison=comparison,
        gases=["CO2", "N2O"],
        rationale="Explicit fixture visual judgment; not a scientific training label.",
    )
    assert result["readback_verified"] and not result["correctness_verified"]
    assert result["checkbox_writes"] == 2 and result["gases"] == ["CO2", "N2O"]
    assert frame.evaluate("window.gasWrites") == 6
    assert frame.locator("#temperature").input_value() == "0"
    assert not result["task_completed"] and not page.get_by_role("checkbox").is_checked()


@pytest.mark.parametrize(
    "mutation", ["crop", "capture", "star", "conditions", "not_compared", "wrong_report"]
)
def test_changed_comparison_never_authorizes_selection(gas_page, tmp_path, mutation):
    page, frame = gas_page
    comparison = tmp_path / "comparison"
    compare_visible_gas_candidates(page, config(), comparison, candidates=["CO2"])
    gases = ["CO2"]
    if mutation == "crop":
        (comparison / "CO2.png").write_bytes(b"changed fixture")
    elif mutation == "capture":
        (comparison / "initial/observation.json").write_text("{}")
    elif mutation == "conditions":
        frame.locator("#temperature").fill("123")
    elif mutation == "not_compared":
        gases = ["O3"]
    else:
        path = comparison / "report.json"
        report = json.loads(path.read_text())
        report["star" if mutation == "star" else "baseline_restored"] = (
            "Other" if mutation == "star" else False
        )
        path.write_text(json.dumps(report))
    with pytest.raises(BrowserSafetyStop):
        select_reference_gases(
            page,
            config(),
            tmp_path / "selected",
            comparison=comparison,
            gases=gases,
            rationale="Explicit fixture selection; not a scientific training label.",
        )
    assert frame.evaluate("window.gasWrites") == 2
    assert not frame.get_by_role("checkbox", name="CO2", exact=True).is_checked()
