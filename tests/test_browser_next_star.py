"""Disposable starfield UI fixtures; no external catalog or navigation."""
# ruff: noqa: F811

import json

import pytest
from test_browser_full_stellar import full_html
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_setup import canvas_html

from habfly.browser import BrowserSafetyStop
from habfly.browser_next_star import NextStarPicker, capture_initial_setup_star, open_next_star
from habfly.browser_stellar import SIMULATION_URL


@pytest.fixture
def starfield(page):
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("iframe").first.evaluate(
        "e=>{e.style.cssText='width:950px;height:600px;border:0;display:block'}"
    )
    frame = page.frame(url=SIMULATION_URL)
    # Detail HTML is embedded in the fixture's script string; exclude its own
    # script tag (classification handlers are unused during navigation).
    frame.set_content(canvas_html(semantic=True, detail=full_html().split("<script>")[0]))
    return page, frame


def test_next_star_records_blank_observations_without_answer_or_score_writes(starfield, tmp_path):
    page, frame = starfield
    receipt = open_next_star(
        page,
        config(),
        tmp_path / "next",
        visited_stars=["Previousstar"],
        excluded_points=[{"x": 600, "y": 300, "width": 950, "height": 600}],
    )
    assert receipt["star"] == "Althinagon" and receipt["fresh_blank_numeric_answers_verified"]
    assert receipt["answer_writes"] == 0 and not receipt["task_completed"]
    assert not page.get_by_role("checkbox").is_checked()
    assert all(x.input_value() == "" for x in frame.get_by_role("textbox").all() if x.is_visible())
    assert (tmp_path / "next/confirmed.json").exists()


@pytest.fixture
def completed_setup(starfield, tmp_path):
    from habfly.browser_setup import BrowserSetup

    page, frame = starfield
    source = tmp_path / "setup"
    source.mkdir()
    setup = BrowserSetup(page, config(), ("", ""), output=source)
    for _ in range(80):
        if setup.advance() == "stellar":
            break
        page.wait_for_timeout(100)
    assert setup.stage == "stellar_screen_ready" and setup.closed
    return setup, frame


def test_completed_initial_setup_has_fresh_class_handoff(completed_setup, tmp_path):
    from habfly.browser_classification import StellarSelectionSession

    setup, frame = completed_setup
    prior = frame.locator("body").inner_text()
    output = tmp_path / "initial"
    receipt = capture_initial_setup_star(setup, output)
    assert receipt["star"] == "Althinagon"
    assert receipt["fresh_blank_numeric_answers_verified"] and receipt["answer_writes"] == 0
    assert frame.locator("body").inner_text() == prior
    source = StellarSelectionSession(setup.page, setup.config, lambda *_: None, fresh_star=output)
    assert source.mapping["star_name"] == receipt["star"]
    assert not (output / "class-selection-reserved.json").exists()


@pytest.mark.parametrize("mutation", ["not_finished", "answer", "point", "paint"])
def test_changed_initial_setup_is_not_a_fresh_handoff(completed_setup, tmp_path, mutation):
    setup, frame = completed_setup
    if mutation == "not_finished":
        setup.stage = "waiting_for_simulation"
    elif mutation == "answer":
        frame.get_by_role("textbox").first.fill("5")
    elif mutation == "point":
        setup.star_point = {**setup.star_point, "x": setup.star_point["x"] + 1}
    else:
        frame.locator("label").first.evaluate("e=>e.style.border='1px solid red'")
    with pytest.raises(BrowserSafetyStop):
        capture_initial_setup_star(setup, tmp_path / "initial")
    assert not (tmp_path / "initial/confirmed.json").exists()


def test_reopening_visited_star_is_not_fresh_and_never_edits(starfield, tmp_path):
    page, _ = starfield
    with pytest.raises(BrowserSafetyStop, match="not_fresh"):
        open_next_star(
            page,
            config(),
            tmp_path / "next",
            visited_stars=["Althinagon"],
            excluded_points=[{"x": 600, "y": 300, "width": 950, "height": 600}],
        )
    failure = json.loads((tmp_path / "next/stopped.json").read_text())
    assert failure["view_click_may_have_occurred"] and not failure["automatic_retry"]
    assert not (tmp_path / "next/confirmed.json").exists()


@pytest.mark.parametrize("bad", ["details", "existing_tooltip", "outer_modal", "auth"])
def test_picker_rejects_unsupported_entry_without_clicks(starfield, tmp_path, bad):
    page, frame = starfield
    if bad == "details":
        frame.set_content(full_html())
    elif bad == "existing_tooltip":
        frame.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<button>VIEW STAR DATA</button>')"
        )
    elif bad == "outer_modal":
        page.locator("body").evaluate("e=>e.insertAdjacentHTML('beforeend','<div role=dialog>Modal</div>')")
    else:
        page.locator("body").evaluate("e=>e.insertAdjacentHTML('beforeend','<input type=password>')")
    with pytest.raises(BrowserSafetyStop):
        open_next_star(page, config(), tmp_path / "next", visited_stars=[], excluded_points=[])
    assert not json.loads((tmp_path / "next/stopped.json").read_text())["star_click_may_have_occurred"]


def test_next_star_safety_rechecks_outer_controls_between_steps(starfield, tmp_path):
    page, _ = starfield
    picker = NextStarPicker(page, config(), tmp_path / "next", visited_stars=[], excluded_points=[])
    page.get_by_role("checkbox").check()
    with pytest.raises(BrowserSafetyStop, match="external_controls_changed"):
        picker.advance()
    assert picker.closed and not picker.star_clicked


def test_new_star_never_accepts_login_destination(starfield, tmp_path):
    page, _ = starfield
    picker = NextStarPicker(page, config(), tmp_path / "next", visited_stars=[], excluded_points=[])
    assert not picker.allowed("http://localhost/authors/log_in")
    assert picker.allowed(config().url)
    picker.close()


def test_sky_region_is_explicit_and_journaled_not_a_class_selector(starfield, tmp_path):
    page, _ = starfield
    picker = NextStarPicker(
        page, config(), tmp_path / "next", visited_stars=[], excluded_points=[], anchor=(0.75, 0.4)
    )
    try:
        assert picker.starfield_anchor == (0.75, 0.4)
        scope = json.loads((tmp_path / "next/scope.json").read_text())
        assert scope["starfield_anchor"] == [0.75, 0.4]
        assert scope["selection_basis"] == "rendered_dot_geometry_only_not_scientific_class"
    finally:
        picker.close()
    with pytest.raises(ValueError):
        NextStarPicker(page, config(), tmp_path / "bad", visited_stars=[], excluded_points=[], anchor=(0, 1))
    assert not (tmp_path / "bad").exists()


def unset_planet_with_unnamed_tab(frame):
    from test_browser_planet_numeric import planet_html

    frame.set_content(planet_html())
    frame.locator("body").evaluate("""e=>{
        const combo=document.querySelector('select');
        combo.insertAdjacentHTML('afterbegin','<option value="" disabled hidden></option>');combo.value='';
        const panel=document.createElement('div');panel.hidden=true;combo.after(panel);
        for(const name of ['orbital_radius','planet_mass','planet_radius','planet_density']){
            const input=document.getElementById(name);panel.append(input.previousElementSibling,input);
        }
        document.querySelector('img').outerHTML='<span role="img" id="stellar-tab">1</span>';
    }""")
    frame.locator("#stellar-tab").evaluate(
        "(e,html)=>e.onclick=()=>document.body.innerHTML=html",
        full_html().split("<script>")[0].replace("Althinagon", "Jyremis"),
    )


def test_retained_planet_tab_uses_visible_number_not_missing_accessible_name(starfield, tmp_path):
    from habfly.browser_setup import numbered_tab

    page, frame = starfield
    picker = NextStarPicker(page, config(), tmp_path / "next", visited_stars=[], excluded_points=[])
    unset_planet_with_unnamed_tab(frame)
    assert frame.get_by_role("img", name="1", exact=True).count() == 0
    assert numbered_tab(frame, 1).aria_snapshot() == '- img: "1"'
    picker.star_clicked = picker.view_clicked = True
    picker.star_point = {"x": 202, "y": 218, "width": 950, "height": 600}
    try:
        for _ in range(5):
            if picker.advance() == "stellar":
                break
        else:
            pytest.fail("Retained-tab navigation did not complete")
        assert picker.receipt["star"] == "Jyremis" and picker.stellar_tab_clicked
    finally:
        picker.close()


def test_opened_star_reconciliation_only_clicks_tab_and_preserves_failed_source(starfield, tmp_path):
    from PIL import Image, ImageDraw

    from habfly.browser_assessment_actions import persist_json
    from habfly.browser_next_star import reconcile_opened_star
    from habfly.browser_probe import inspect_page, save_probe

    page, frame = starfield
    failed = tmp_path / "failed"
    failed.mkdir()
    save_probe(inspect_page(page, config()), failed / "before")
    persist_json(
        failed / "scope.json",
        {
            "visited_stars": ["Previousstar"],
            "excluded_points": [{"x": 600, "y": 300, "width": 950, "height": 600}],
        },
    )
    stopped = {
        "reason": "setup_ambiguous_or_unavailable_control",
        "star_click_may_have_occurred": True,
        "view_click_may_have_occurred": True,
        "automatic_retry": False,
        "task_completed": False,
    }
    persist_json(failed / "stopped.json", stopped)
    pixels = Image.new("RGB", (950, 600), "#081020")
    ImageDraw.Draw(pixels).rectangle((200, 220, 205, 225), fill="white")
    pixels.save(failed / "setup-starfield.png")
    unset_planet_with_unnamed_tab(frame)
    # The real course keeps old class paint on a new star's blank planet panel.
    # Navigation must record that ambiguity, not repair or trust the answer.
    frame.locator(".choice label").nth(1).evaluate(
        "e=>{e.style.borderColor='white';e.insertAdjacentHTML('beforeend','<style>.choice:nth-child(n) label[style]::after{opacity:1;background:white}</style>')}"
    )
    receipt = reconcile_opened_star(page, config(), failed, tmp_path / "reconciled")
    assert (
        receipt["star"] == "Jyremis" and receipt["star_click_retries"] == receipt["view_click_retries"] == 0
    )
    assert receipt["original_failure_preserved"] and receipt["answer_writes"] == 0
    assert receipt["planet_paint_before_tab"] == "ice_giant" and not receipt["class_selection_verified"]
    assert json.loads((failed / "stopped.json").read_text()) == stopped
    assert not page.get_by_role("checkbox").is_checked()
