"""Network-intercepted visible class and six-field transport fixtures."""

from pathlib import Path

import pytest
from test_browser_numeric import chromium, config, page, stellar_html  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_classification import StellarSelectionSession, read_class_choices
from habfly.browser_full_stellar import FullStellarSession, FullStellarToolEnv, load_full_stellar_policy
from habfly.browser_numeric import NumericJournal, screen_identity
from habfly.browser_probe import inspect_page
from habfly.browser_stellar import SIMULATION_URL, StellarMappingError, map_stellar_capture
from habfly.lifetime_prefix import lifetime_to_prefix


def full_html():
    labels = ("main sequence", "red giant", "supergiant", "white dwarf")
    choices = "".join(
        f'<div class="choice"><label onclick="choose(this,{str(i == 0).lower()})"></label><div>{name}</div></div>'
        for i, name in enumerate(labels)
    )
    conditional = (
        '<div id="conditional" style="display:none"><span>mass (M<sub>S</sub>)</span><input placeholder="0" id="mass">'
        '<span>radius (R<sub>S</sub>)</span><input placeholder="0" id="radius">'
        '<span>lifetime (years)</span><input placeholder="0.00000" id="lifetime">'
        '<select id="prefix"><option selected></option><option>ka</option><option>Ma</option><option>Ga</option><option>Ta</option></select></div>'
    )
    return (
        stellar_html().replace(
            "mass, radius and lifetime are only relevant for main sequence stars main sequence red giant supergiant white dwarf",
            '<div id="warning">mass, radius and lifetime are only relevant for main sequence stars</div>'
            + conditional
            + choices,
        )
        + """<style>
    .choice {display:inline-block;width:120px;text-align:center}
    .choice label {display:block;position:relative;width:26px;height:26px;border:1px solid rgb(0,200,220);margin:auto}
    .choice label::after {content:"";position:absolute;display:block;width:14px;height:14px;opacity:0;background:transparent;left:6px;top:6px}
    .choice label.selected {border-color:white}
    .choice label.selected::after {opacity:1;background:white}
    .choice > div {margin-top:3px;font-size:10px}
    </style><script>
    function choose(e, main) {
      document.querySelectorAll('.choice label').forEach(x=>x.classList.remove('selected'));
      e.classList.add('selected');
      document.querySelector('#conditional').style.display=main?'block':'none';
      document.querySelector('#warning').style.display=main?'none':'block';
    }
    </script>"""
    )


@pytest.fixture
def full_page(page):  # noqa: F811
    page.locator("iframe").first.evaluate("e=>{e.style.width='950px';e.style.height='600px'}")
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.locator("body").evaluate("(e,html)=>e.innerHTML=html", full_html())
    # Scripts inserted with innerHTML are intentionally inert; install fixture handler.
    frame.evaluate("""() => window.choose=(e,main)=>{
        document.querySelectorAll('.choice label').forEach(x=>x.classList.remove('selected'));
        e.classList.add('selected');
        document.querySelector('#conditional').style.display=main?'block':'none';
        document.querySelector('#warning').style.display=main?'none':'block';
    }""")
    return page


def select(browser_page, name="main_sequence", prefix="Ga"):
    events = []
    selection = StellarSelectionSession(browser_page, config(), lambda *event: events.append(event))
    receipt = selection.select_class(name, source="reference_diagnostic")
    if name == "main_sequence" and prefix:
        selection.select_prefix(prefix)
    return selection, receipt, events


@pytest.mark.parametrize("name", ["main_sequence", "red_giant", "supergiant", "white_dwarf"])
def test_visible_class_selection_and_conditional_fields(full_page, name):
    selection, receipt, events = select(full_page, name)
    assert receipt["readback_verified"] and not receipt["correctness_verified"]
    assert not receipt["task_completed"]
    assert selection.choices["selected"] == name
    fields = selection.mapping["observation"]["values"]["browser_field_map"]
    assert len(fields) == (6 if name == "main_sequence" else 3)
    assert [e[0] for e in events] == ["action_proposed", "action_result"] * (
        2 if name == "main_sequence" else 1
    )
    with pytest.raises(BrowserSafetyStop, match="class_write_limit"):
        selection.select_class(name, source="checkpoint")


def test_original_mapping_remains_three_field_only(full_page):
    select(full_page)
    report = inspect_page(full_page, config())
    with pytest.raises(StellarMappingError, match="unsupported_conditional"):
        map_stellar_capture(report, capture_sha256=screen_identity(report))
    mapping = map_stellar_capture(
        report, capture_sha256=screen_identity(report), allow_main_sequence_fields=True
    )
    values = mapping["observation"]["values"]
    assert values["lifetime_prefix"]["selected"] == "Ga"
    assert values["browser_field_map"]["lifetime"]["unit"] == "Ga"
    assert values["star_class"] is None  # Field visibility alone is not a class readback.


@pytest.mark.parametrize("change", ["duplicate", "overlay", "style", "geometry"])
def test_class_ambiguity_fails_without_writing(full_page, change):
    frame = next(f for f in full_page.frames if f.url == SIMULATION_URL)
    if change == "duplicate":
        frame.locator(".choice").first.evaluate("e=>e.after(e.cloneNode(true))")
    elif change == "overlay":
        frame.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<div style=\"position:fixed;inset:0;z-index:999;background:black\"></div>')"
        )
    elif change == "style":
        frame.locator(".choice label").first.evaluate("e=>e.style.borderColor='red'")
    else:
        frame.locator(".choice label").first.evaluate("e=>e.style.marginLeft='0'")
    events = []
    with pytest.raises(BrowserSafetyStop):
        StellarSelectionSession(full_page, config(), lambda *e: events.append(e))
    assert events == []


def test_hidden_radios_are_not_read(full_page):
    frame = next(f for f in full_page.frames if f.url == SIMULATION_URL)
    frame.locator("body").evaluate(
        "e=>e.insertAdjacentHTML('beforeend','<input type=radio checked hidden value=supergiant>')"
    )
    choices, _ = read_class_choices(frame)
    assert choices["selected"] is None


@pytest.mark.parametrize("chosen", ["main_sequence", "red_giant", "white_dwarf"])
def test_fresh_star_evidence_allows_one_explicit_class_despite_retained_paint(full_page, tmp_path, chosen):
    from habfly.browser_assessment_actions import persist_json
    from habfly.browser_probe import save_probe

    frame = full_page.frame(url=SIMULATION_URL)
    # New star's numbers are blank and conditional panel absent, but old class
    # paint remains. It is not treated as a scientific answer for the new star.
    frame.locator(".choice label").first.evaluate("e=>e.classList.add('selected')")
    with pytest.raises(BrowserSafetyStop, match="conditional_fields_disagree"):
        StellarSelectionSession(full_page, config(), lambda *_: None)
    source = tmp_path / "fresh"
    source.mkdir()
    save_probe(inspect_page(full_page, config()), source / "stellar")
    persist_json(
        source / "confirmed.json",
        {
            "star": "Althinagon",
            "painted_stellar_class": "main_sequence",
            "fresh_blank_numeric_answers_verified": True,
            "class_selection_verified": False,
            "answer_writes": 0,
            "action_source": "deterministic_navigation",
        },
    )
    session = StellarSelectionSession(full_page, config(), lambda *_: None, fresh_star=source)
    receipt = session.select_class(chosen, source="reference_diagnostic")
    assert receipt["readback_verified"] and receipt["previous_unconfirmed_paint"] == "main_sequence"
    assert not receipt["correctness_verified"]
    assert frame.locator("#distance").input_value() == ""
    assert (source / "class-selection-reserved.json").exists()
    with pytest.raises(BrowserSafetyStop, match="invalid_fresh_star_class_evidence"):
        StellarSelectionSession(full_page, config(), lambda *_: None, fresh_star=source)


def test_fresh_class_proof_cannot_authorize_modified_answers_or_a_different_star(full_page, tmp_path):
    from habfly.browser_assessment_actions import persist_json
    from habfly.browser_probe import save_probe

    frame = full_page.frame(url=SIMULATION_URL)
    source = tmp_path / "fresh"
    source.mkdir()
    save_probe(inspect_page(full_page, config()), source / "stellar")
    persist_json(
        source / "confirmed.json",
        {
            "star": "Althinagon",
            "painted_stellar_class": None,
            "fresh_blank_numeric_answers_verified": True,
            "class_selection_verified": False,
            "answer_writes": 0,
            "action_source": "deterministic_navigation",
        },
    )
    frame.locator("#distance").fill("123")
    with pytest.raises(BrowserSafetyStop, match="fresh_star_class_evidence_changed"):
        StellarSelectionSession(full_page, config(), lambda *_: None, fresh_star=source)
    assert not (source / "class-selection-reserved.json").exists()


def test_planet_circles_are_read_without_inferring_class(full_page):
    from habfly.browser_classification import read_planet_class_choices

    frame = next(f for f in full_page.frames if f.url == SIMULATION_URL)
    labels = frame.locator(".choice > div")
    for index, name in enumerate(("gas giant", "ice giant", "terrestrial")):
        labels.nth(index).evaluate("(e,name)=>e.innerText=name", name)
    frame.locator(".choice").nth(3).evaluate("e=>e.remove()")
    choices, handles = read_planet_class_choices(frame)
    assert choices["selected"] is None
    handles["terrestrial"].click()
    choices, _ = read_planet_class_choices(frame)
    assert choices["selected"] == "terrestrial"


def test_habitability_circle_paint_does_not_establish_who_selected_it(full_page):
    from habfly.browser_classification import read_habitability_choices

    frame = next(f for f in full_page.frames if f.url == SIMULATION_URL)
    for index, name in enumerate(("\n  Not Habitable\n  ", "\n  Habitable\n  ")):
        frame.locator(".choice > div").nth(index).evaluate("(e,name)=>e.innerText=name", name)
    for _ in range(2):
        frame.locator(".choice").last.evaluate("e=>e.remove()")
    choices, handles = read_habitability_choices(frame)
    assert choices["selected"] is None
    handles["not_habitable"].click()
    choices, _ = read_habitability_choices(frame)
    assert choices["selected"] == "not_habitable"


@pytest.mark.parametrize("mutation", ["outer", "measurement", "answer", "button", "feedback"])
def test_unrelated_class_side_effect_stops_without_retry(full_page, mutation):
    frame = next(f for f in full_page.frames if f.url == SIMULATION_URL)
    selection = StellarSelectionSession(full_page, config(), lambda *_: None)
    snippets = {
        "measurement": "document.body.innerHTML=document.body.innerHTML.replace('0.045','0.046')",
        "answer": "document.querySelector('#distance').value='123'",
        "button": "document.querySelector('button').disabled=true",
        "feedback": "document.body.append('unexpected feedback')",
    }
    # Cross-origin fixture mutation is performed by the test harness, not by
    # the adapter or the page under test.
    if mutation == "outer":
        full_page.on("console", lambda _: full_page.get_by_role("checkbox").check())
        snippet = "console.log('fixture change')"
    else:
        snippet = snippets[mutation]
    frame.evaluate(
        "(body)=>{const original=window.choose;window.choose=(e,m)=>{original(e,m);eval(body)}}", snippet
    )
    with pytest.raises(BrowserSafetyStop):
        selection.select_class("main_sequence", source="reference_diagnostic")
    assert selection.stopped
    with pytest.raises(BrowserSafetyStop, match="stopped"):
        selection.select_class("main_sequence", source="reference_diagnostic")


def test_prefix_zero_initialization_is_not_an_answer(full_page):
    selection, _, _ = select(full_page, prefix=None)
    frame = next(f for f in full_page.frames if f.url == SIMULATION_URL)
    frame.locator("#prefix").evaluate("e=>e.onchange=()=>{document.querySelector('#lifetime').value='0.000'}")
    receipt = selection.select_prefix("Ga")
    assert receipt["blank_lifetime_initialized_to_zero"]
    assert not receipt["task_completed"]
    assert (
        selection.mapping["observation"]["values"]["browser_field_map"]["lifetime"]["current_value"]
        == "0.000"
    )
    with pytest.raises(BrowserSafetyStop, match="prefix_write_limit"):
        selection.select_prefix("Ta")


@pytest.mark.parametrize("field,value", [("mass", "0.000"), ("lifetime", "1"), ("lifetime", "0.1")])
def test_prefix_never_allows_arbitrary_answer_mutations(full_page, field, value):
    selection, _, _ = select(full_page, prefix=None)
    frame = next(f for f in full_page.frames if f.url == SIMULATION_URL)
    frame.locator("#prefix").evaluate(
        "(e,p)=>e.onchange=()=>{document.getElementById(p[0]).value=p[1]}", [field, value]
    )
    with pytest.raises(BrowserSafetyStop, match="numeric_data_changed_by_prefix"):
        selection.select_prefix("Ga")
    assert selection.stopped


def test_save_busy_normalization_is_narrow(full_page):
    from copy import deepcopy

    from habfly.browser_full_stellar import full_stellar_status_projection

    select(full_page)
    report = inspect_page(full_page, config())
    changed = deepcopy(report)
    frame = next(f for f in changed["frames"] if f["url"] == SIMULATION_URL)
    control = next(c for c in frame["controls"] if c["accessibility"] == '- button "Save"')
    control["accessibility"] += " [disabled]"
    control["enabled"] = False
    frame["accessibility"] = frame["accessibility"].replace('- button "Save"', '- button "Save" [disabled]')
    assert full_stellar_status_projection(report) == full_stellar_status_projection(changed)
    control["accessibility"] = '- button "Delete" [disabled]'
    assert full_stellar_status_projection(report) != full_stellar_status_projection(changed)


@pytest.mark.parametrize("channel", ["text", "accessibility", "both"])
def test_six_field_autosave_race_normalizes_each_exact_footer(full_page, channel):
    from copy import deepcopy

    from habfly.browser_full_stellar import full_stellar_status_projection

    select(full_page)
    report = inspect_page(full_page, config())
    changed = deepcopy(report)
    frame = next(f for f in changed["frames"] if f["url"] == SIMULATION_URL)
    if channel in {"text", "both"}:
        frame["text"] = frame["text"].removesuffix("1 RsSave") + "\n1 Rs\nData saved\nSave"
        # Fixture inline text differs from live footer line wrapping.
        original = next(f for f in report["frames"] if f["url"] == SIMULATION_URL)
        original["text"] = original["text"].removesuffix("1 RsSave") + "\n1 Rs\nSave"
    if channel in {"accessibility", "both"}:
        frame["accessibility"] = frame["accessibility"].replace(
            '1 Rs\n- button "Save"', '1 Rs Data saved\n- button "Save"'
        )
    assert screen_identity(full_stellar_status_projection(report)) == screen_identity(
        full_stellar_status_projection(changed)
    )
    frame["text"] = "Data saved elsewhere\n" + frame["text"]
    assert screen_identity(full_stellar_status_projection(report)) != screen_identity(
        full_stellar_status_projection(changed)
    )


def test_wrong_selection_is_not_corrected(full_page):
    selection, _, _ = select(full_page, "white_dwarf")
    assert selection.choices["selected"] == "white_dwarf"
    assert len(selection.mapping["observation"]["values"]["browser_field_map"]) == 3


def test_selection_rejects_changed_measurement_and_preexisting_answer(full_page):
    frame = next(f for f in full_page.frames if f.url == SIMULATION_URL)
    selection = StellarSelectionSession(full_page, config(), lambda *_: None)
    frame.locator("body").evaluate("e=>e.innerHTML=e.innerHTML.replace('0.045','0.046')")
    with pytest.raises(BrowserSafetyStop, match="stale_class_observation"):
        selection.select_class("main_sequence", source="reference_diagnostic")


def test_wrong_binding_is_rejected_without_numeric_write(full_page, tmp_path):
    select(full_page, prefix="Ta")
    journal = NumericJournal(tmp_path / "run")
    session = FullStellarSession(
        full_page, config(), journal, selected_class="main_sequence", lifetime_prefix="Ta"
    )
    try:
        session.start()
        with pytest.raises(StellarMappingError):
            session.calculate("lifetime", {"mass": "browser_parallax"})
        assert session.attempts == 0
    finally:
        journal.stream.close()


@pytest.mark.parametrize("mutation", ["measurement", "modal", "navigation", "replaced_target"])
def test_pinned_stellar_choices_cannot_authorize_stale_native_copy(full_page, tmp_path, mutation):
    select(full_page)
    journal = NumericJournal(tmp_path / "pinned", provenance={"browser_execution": "autonomous"})
    session = FullStellarSession(
        full_page, config(), journal, selected_class="main_sequence", lifetime_prefix="Ga"
    )
    frame = full_page.frame(url=SIMULATION_URL)
    try:
        session.start()
        env = FullStellarToolEnv(session, 8500000)
        calls = []
        original = session._current

        def counted():
            calls.append(True)
            return original()

        session._current = counted
        for _ in range(20):
            action = env.expert_action(env.observe())
            if action.target.endswith(":copy"):
                break
            env.step(action)
        else:
            pytest.fail("No copy proposed")
        assert len(calls) == 1  # Only calculation, not every local selection.
        if mutation == "measurement":
            frame.locator("body").evaluate("e=>e.innerHTML=e.innerHTML.replace('0.045','0.046')")
        elif mutation == "modal":
            frame.locator("body").evaluate(
                "e=>{const d=document.createElement('dialog');e.append(d);d.showModal()}"
            )
        elif mutation == "navigation":
            full_page.route("https://outside.invalid/**", lambda route: route.fulfill(body="Outside fixture"))
            full_page.goto("https://outside.invalid/escaped")
        else:
            frame.locator("#distance").evaluate("e=>e.replaceWith(e.cloneNode(true))")
        env.copy_approved = True
        env.copy_authorization = "autonomous_opt_in"
        with pytest.raises(BrowserSafetyStop):
            env.step(action)
        assert session.attempts == 0 and not session.verified
    finally:
        full_page.remove_listener("dialog", session._dialog)
        journal.stream.close()


def test_stellar_final_check_rechecks_live_browser(full_page, tmp_path):
    select(full_page, "white_dwarf")
    journal = NumericJournal(tmp_path / "complete", provenance={"browser_execution": "autonomous"})
    session = FullStellarSession(full_page, config(), journal, selected_class="white_dwarf")
    try:
        session.start()
        env = FullStellarToolEnv(session, 8500000)
        for _ in range(128):
            action = env.expert_action(env.observe())
            if action.target.endswith(":check"):
                break
            env.copy_approved = True
            env.copy_authorization = "autonomous_opt_in"
            env.step(action)
        else:
            pytest.fail("No final check proposed")
        assert len(session.verified) == 3
        full_page.frame(url=SIMULATION_URL).locator("#distance").fill("77")
        with pytest.raises(BrowserSafetyStop, match="stale_numeric_observation"):
            env.step(action)
        assert not env.transport_verified
    finally:
        full_page.remove_listener("dialog", session._dialog)
        journal.stream.close()


def test_conditional_calculations_and_copy(full_page, tmp_path):
    select(full_page, prefix="Ta")
    journal = NumericJournal(tmp_path / "run")
    session = FullStellarSession(
        full_page, config(), journal, selected_class="main_sequence", lifetime_prefix="Ta"
    )
    try:
        session.start()
        steps = [
            ("distance", {"parallax": "browser_parallax"}),
            ("luminosity", {"flux": "browser_flux", "distance": "result:1"}),
            ("temperature", {"wavelength": "browser_wavelength"}),
            ("mass", {"luminosity": "result:2"}),
            ("radius", {"luminosity": "result:2", "temperature": "result:3"}),
            ("lifetime", {"mass": "result:4"}),
        ]
        for destination, bindings in steps:
            ident, result = session.calculate(destination, bindings)
            assert session.copy(ident, destination, confirm=lambda _: True)
            receipt = journal.numeric_readbacks[destination]
            assert receipt["exact_copied"] == (
                lifetime_to_prefix(result.value, "Ta") if destination == "lifetime" else repr(result.value)
            )
            assert receipt["unit"] == ("Ta" if destination == "lifetime" else result.unit)
            if destination == "lifetime":
                assert receipt["calculation_unit"] == "yr"
                assert receipt["calculation_value"] == repr(result.value)
        assert session.verified == [s[0] for s in steps]
        frame = next(f for f in full_page.frames if f.url == SIMULATION_URL)
        frame.locator("#prefix").select_option(label="Ga")
        with pytest.raises(BrowserSafetyStop, match="lifetime_prefix_changed"):
            session._current()
    finally:
        journal.stream.close()


@pytest.mark.parametrize("selected_class", ["main_sequence", "red_giant", "supergiant", "white_dwarf"])
def test_frozen_real_graph_six_field_policy(full_page, tmp_path, selected_class):
    import torch

    select(full_page, selected_class)
    policy, provenance = load_full_stellar_policy(
        Path("experiments/lifetime-003"),
        Path("experiments/lifetime-003/training/checkpoint.pt"),
        Path("data/processed/graphs-v2/graph-2000"),
    )
    journal = NumericJournal(
        tmp_path / "learned",
        learned_policy=True,
        provenance={**provenance, "browser_execution": "autonomous"},
    )
    session = FullStellarSession(
        full_page,
        config(),
        journal,
        selected_class=selected_class,
        lifetime_prefix="Ga" if selected_class == "main_sequence" else None,
    )
    try:
        session.start()
        env = FullStellarToolEnv(session, seed=8500000)
        state = None
        with torch.no_grad():
            for _ in range(128):
                action, state, _ = policy.act(env.observe(), state)
                env.copy_approved = True
                env.copy_authorization = "autonomous_opt_in"
                result = env.step(action)
                if result.terminated or result.truncated:
                    break
        assert env.transport_verified, (result.failure_reason, env.steps)
        assert len(session.verified) == (6 if selected_class == "main_sequence" else 3)
        assert not result.observation.progress["task_completed"]
        assert provenance["optimizer_updates"] == 0
    finally:
        journal.stream.close()


@pytest.mark.parametrize("selected_class", ["main_sequence", "red_giant", "supergiant", "white_dwarf"])
def test_frozen_color_policy_on_six_field_view(full_page, tmp_path, selected_class):
    from habfly.browser_full_color import run_full_stellar_color

    select(full_page, selected_class)
    report = run_full_stellar_color(
        full_page,
        config(),
        tmp_path / "color",
        selected_class=selected_class,
        lifetime_prefix="Ga" if selected_class == "main_sequence" else None,
        experiment=Path("experiments/color-pilot-006"),
        graph_path=Path("data/processed/graphs-v2/graph-2000"),
    )
    assert report["color_transport_verified"], report
    assert report["write_attempts"] == 1
    assert report["receipt"]["selected_color"] == "UV"
    assert not report["task_completed"]
