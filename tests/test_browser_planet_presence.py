"""One evidence-backed Yes; never an automatic absence answer or overwrite."""
# ruff: noqa: F811

import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_evidence import evidence
from test_browser_planet_numeric import planet_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_numeric import PlanetNumericSession
from habfly.browser_planet_presence import reconcile_presence_readback, select_detected_planet


@pytest.fixture
def presence_page(planet_page):
    page, frame = planet_page
    frame.locator("body").evaluate("""e=>{
        const combo=document.querySelector('select');
        combo.insertAdjacentHTML('afterbegin','<option value="" disabled hidden></option>');combo.value='';
        const panel=document.createElement('div');panel.hidden=true;combo.after(panel);
        for(const name of ['orbital_radius','planet_mass','planet_radius','planet_density']){
            const input=document.getElementById(name);panel.append(input.previousElementSibling,input);
        }
        combo.onchange=()=>panel.hidden=false;
    }""")
    return page, frame


def sources(tmp_path, star="JYREMIS"):
    spectrum, transit = evidence()
    spectrum["events"][0]["payload"]["chart"]["star"] = star
    transit["star"] = star
    for name, value in (("spectrum", spectrum), ("transit", transit)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    return {"spectrum_path": tmp_path / "spectrum.json", "transit_path": tmp_path / "transit.json"}


def test_supported_yes_does_not_fill_answers_or_choose_class(presence_page, tmp_path):
    page, frame = presence_page
    receipt = select_detected_planet(page, config(), tmp_path / "run", **sources(tmp_path))
    assert receipt["readback_verified"] and not receipt["correctness_verified"]
    assert receipt["value"] == "Yes" and receipt["numeric_writes"] == 0
    assert not receipt["task_completed"] and not receipt["automatic_retry"]
    assert frame.get_by_role("combobox").input_value() == "Yes"
    assert frame.locator("#orbital_radius").input_value() == ""
    assert not page.get_by_role("checkbox").is_checked()
    with pytest.raises(BrowserSafetyStop, match="already_selected"):
        select_detected_planet(page, config(), tmp_path / "repeat", **sources(tmp_path))


@pytest.mark.parametrize("kind", ["star", "raw_answer", "selected_no", "modal"])
def test_stale_or_existing_work_is_never_replaced(presence_page, tmp_path, kind):
    page, frame = presence_page
    if kind == "raw_answer":
        frame.locator("#line_shift").fill("1")
    elif kind == "selected_no":
        frame.get_by_role("combobox").evaluate("e=>e.value='No'")
    elif kind == "modal":
        frame.locator("body").evaluate("e=>e.insertAdjacentHTML('beforeend','<div role=dialog>stop</div>')")
    with pytest.raises(BrowserSafetyStop):
        select_detected_planet(
            page, config(), tmp_path / "run", **sources(tmp_path, "OTHER" if kind == "star" else "JYREMIS")
        )
    assert not json.loads((tmp_path / "run/stopped.json").read_text())["write_may_have_occurred"]
    assert frame.get_by_role("combobox").input_value() == ("No" if kind == "selected_no" else "")


@pytest.mark.parametrize("mutation", ["answer", "external", "axis", "dialog", "replacement"])
def test_uncertain_selection_never_retries_or_rolls_back(presence_page, tmp_path, mutation):
    page, frame = presence_page
    effect = {
        "answer": "document.getElementById('line_shift').value='10'",
        "external": "document.body.append('Unexpected message')",
        "axis": "document.querySelector('svg text').textContent='Normalized Flux Days Observed 7 8 9'",
        "dialog": "confirm('private confirmation')",
        "replacement": "e.replaceWith(e.cloneNode(true))",
    }[mutation]
    frame.get_by_role("combobox").evaluate(
        "(e,code)=>{const old=e.onchange;e.onchange=()=>{old();new Function('e',code)(e)}}", effect
    )
    with pytest.raises(BrowserSafetyStop):
        select_detected_planet(page, config(), tmp_path / "run", **sources(tmp_path))
    assert not (tmp_path / "run/confirmed.json").exists()
    raw = (tmp_path / "run/stopped.json").read_text()
    assert json.loads(raw)["write_may_have_occurred"] and "private confirmation" not in raw


def test_unset_planet_read_guard_cannot_be_used_to_copy(presence_page, tmp_path):
    page, frame = presence_page
    session = PlanetNumericSession(page, config(), tmp_path / "read", _allow_unset_planet=True)
    try:
        with pytest.raises(BrowserSafetyStop, match="yes_selection_required"):
            session.copy("line_shift", "1", "nm", source="checkpoint")
        assert not session.attempted and frame.locator("#line_shift").input_value() == ""
    finally:
        session.close()


def test_explicit_inherited_paint_is_preserved_not_claimed_as_classification(presence_page, tmp_path):
    from habfly.browser_classification import read_planet_class_choices

    page, frame = presence_page
    frame.locator("body").evaluate("""e=>{
        const style=document.createElement('style');
        style.textContent='.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}';
        e.append(style);document.querySelectorAll('.choice')[1].classList.add('selected');
    }""")
    before = read_planet_class_choices(frame)[0]
    assert before["selected"] == "ice_giant"
    for name, paint in (("default", None), ("wrong", "gas_giant")):
        with pytest.raises(BrowserSafetyStop, match="already_selected"):
            select_detected_planet(
                page, config(), tmp_path / name, preserve_painted_class=paint, **sources(tmp_path)
            )
        assert frame.get_by_role("combobox").input_value() == ""
    result = select_detected_planet(
        page, config(), tmp_path / "preserved", preserve_painted_class="ice_giant", **sources(tmp_path)
    )
    assert result["readback_verified"] and result["preserved_painted_class"] == "ice_giant"
    assert result["class_writes"] == 0 and not result["class_selection_verified"]
    assert read_planet_class_choices(frame)[0] == before


@pytest.mark.parametrize("legacy_stop", [False, True])
def test_inherited_paint_may_clear_but_reconciliation_never_repeats_yes(
    presence_page, tmp_path, monkeypatch, legacy_stop
):
    from playwright.sync_api import Locator

    import habfly.browser_planet_presence as module

    page, frame = presence_page
    frame.locator("body").evaluate("""e=>{
        const style=document.createElement('style');
        style.textContent='.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}';
        e.append(style);document.querySelectorAll('.choice')[1].classList.add('selected');
        const combo=document.querySelector('select'),old=combo.onchange;
        combo.onchange=()=>{old();document.querySelectorAll('.choice').forEach(c=>c.classList.remove('selected'))};
    }""")
    paths = sources(tmp_path)
    run = tmp_path / "first"
    if not legacy_stop:
        receipt = select_detected_planet(page, config(), run, preserve_painted_class="ice_giant", **paths)
        assert receipt["readback_verified"] and receipt["inherited_paint_cleared"]
        assert receipt["painted_class_after"] is None and not receipt["class_selection_verified"]
        return
    monkeypatch.setattr(module, "preserved_paint_transition", lambda *_: False)
    with pytest.raises(BrowserSafetyStop, match="unexpected_planet_presence_side_effect"):
        select_detected_planet(page, config(), run, preserve_painted_class="ice_giant", **paths)
    original = (run / "stopped.json").read_bytes()

    def deny(*args, **kwargs):
        raise AssertionError("Read-only reconciliation attempted an action")

    for name in ("click", "select_option", "fill", "press"):
        monkeypatch.setattr(Locator, name, deny)
    receipt = reconcile_presence_readback(page, config(), run, tmp_path / "reconciled", **paths)
    assert receipt["readback_reconciled"] and receipt["browser_actions"] == 0
    assert receipt["class_writes"] == 0 and not receipt["class_selection_verified"]
    assert (run / "stopped.json").read_bytes() == original
    assert frame.get_by_role("combobox").input_value() == "Yes"
