"""Local intercepted browser fixtures; no live chart, spectrum, or answer oracle."""
# ruff: noqa: F811

import hashlib
import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_numeric import planet_html

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import capture_observation_progress
from habfly.browser_planet_window_choice import select_no_planet_from_window
from habfly.browser_stellar import SIMULATION_URL


@pytest.fixture
def window_page(page):
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("iframe").first.evaluate(
        "e=>{e.style='position:absolute;left:0;top:200px;width:1200px;height:850px'}"
    )
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.locator("body").evaluate(
        "(e,h)=>e.innerHTML=h", planet_html().replace('value="10000"', 'value="5000"')
    )
    frame.evaluate("""()=>{
      const combo=document.querySelector('select');
      combo.insertAdjacentHTML('afterbegin','<option value="" disabled hidden></option>');combo.value='';
      const panel=document.createElement('div');panel.hidden=true;combo.after(panel);
      for(const name of ['orbital_radius','planet_mass','planet_radius','planet_density']){
        const input=document.getElementById(name);panel.append(input.previousElementSibling,input);
      }
      window.fixtureSelections=0;combo.onchange=()=>{window.fixtureSelections++;panel.hidden=combo.value!=='Yes'};
      const chart=document.querySelector('svg');
      chart.outerHTML=`<svg id='chart' role='img' width='280' height='195' style='display:block;background:black'>
        <title>Normalized Flux Days Observed</title><rect width='280' height='195' fill='black'/>
        <line id='trace' x1='30.5' x2='260.5' y1='20.5' y2='20.5' stroke='rgb(80,132,154)' stroke-width='1'/>
        <text class='time' x='30.5' y='165' text-anchor='middle' font-size='4' fill='white'>0</text>
        <text class='time' x='145.5' y='165' text-anchor='middle' font-size='4' fill='white'>2500</text>
        <text class='time' x='260.5' y='165' text-anchor='middle' font-size='4' fill='white'>5000</text>
        <text class='flux' x='25' y='151' text-anchor='end' font-size='4' fill='white'>0</text>
        <text class='flux' x='25' y='124.9' text-anchor='end' font-size='4' fill='white'>20</text>
        <text class='flux' x='25' y='85.75' text-anchor='end' font-size='4' fill='white'>50</text>
        <text class='flux' x='25' y='20.5' text-anchor='end' font-size='4' fill='white'>100</text>
      </svg>`;
      const top=document.querySelector('#chart').getBoundingClientRect().top;
      for(const text of document.querySelectorAll('#chart text')){
        const desired=Number(text.getAttribute('y')),b=text.getBoundingClientRect();
        text.setAttribute('y',desired+desired-(b.top+b.height/2-top));
      }
    }""")
    return page, frame


def source(window_page, tmp_path):
    page, _ = window_page
    directory = tmp_path / "source"
    capture_observation_progress(page, config(), directory, requested_days=5000)
    report = directory / "report.json"
    return {"evidence_path": report, "evidence_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}


def choose(window_page, tmp_path, evidence, **kwargs):
    page, _ = window_page
    return select_no_planet_from_window(
        page, config(), tmp_path / "choice", run_history=tmp_path, **evidence, **kwargs
    )


def test_current_flat_window_selects_only_reference_no(window_page, tmp_path):
    page, frame = window_page
    receipt = choose(window_page, tmp_path, source(window_page, tmp_path))
    assert frame.evaluate("window.fixtureSelections") == 1
    assert frame.get_by_role("combobox").input_value() == "No"
    assert receipt["readback_verified"] and receipt["value"] == "No"
    assert receipt["examined_day_interval"] == [0, 5000]
    assert receipt["policy"]["user_approved"] and receipt["provenance"] == "reference_prediction"
    for name in (
        "absence_proven",
        "scientific_verified",
        "correctness_verified",
        "training_label",
        "learned_perception",
        "task_completed",
        "automatic_retry",
    ):
        assert receipt[name] is False
    for name in (
        "numeric_writes",
        "class_writes",
        "na_writes",
        "assessment_clicks",
        "save_clicks",
        "submission_clicks",
    ):
        assert receipt[name] == 0
    assert frame.locator("#line_shift").input_value() == ""
    assert frame.locator("#brightness_drop").input_value() == ""
    assert frame.locator("#period_days").input_value() == ""
    assert not page.get_by_role("checkbox").is_checked()
    assert (tmp_path / "choice/preselect-progress/chart.png").exists()
    assert len(list((tmp_path / "planet-window-choice-reservations").glob("*.json"))) == 1


@pytest.mark.parametrize("kind", ["report", "png", "status", "invalidated", "hash"])
def test_mutated_or_invalid_saved_evidence_never_selects(window_page, tmp_path, kind):
    _, frame = window_page
    evidence = source(window_page, tmp_path)
    report = evidence["evidence_path"]
    if kind == "report":
        report.write_text(report.read_text() + " ")
    elif kind == "png":
        (report.parent / "chart.png").write_bytes(b"invalid")
    elif kind == "status":
        data = json.loads(report.read_text())
        data["endpoint_visible"] = False
        report.write_text(json.dumps(data))
        evidence["evidence_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
    elif kind == "invalidated":
        (report.parent / "invalidated.json").write_text("{}")
    else:
        evidence["evidence_sha256"] = "bad"
    with pytest.raises(BrowserSafetyStop):
        choose(window_page, tmp_path, evidence)
    assert frame.evaluate("window.fixtureSelections") == 0
    assert not json.loads((tmp_path / "choice/stopped.json").read_text())["write_may_have_occurred"]


@pytest.mark.parametrize(
    "kind",
    [
        "star",
        "answer",
        "presence",
        "duration",
        "modal",
        "auth",
        "dip",
        "partial",
        "axis",
        "occluded",
    ],
)
def test_stale_source_is_rechecked_against_current_visible_state(window_page, tmp_path, kind):
    _, frame = window_page
    evidence = source(window_page, tmp_path)
    if kind == "star":
        frame.locator("div").first.evaluate("e=>e.innerText='OTHERSTAR'")
    elif kind == "answer":
        frame.locator("#period_days").fill("3")
    elif kind == "presence":
        frame.get_by_role("combobox").evaluate("e=>e.value='No'")
    elif kind == "duration":
        frame.get_by_role("textbox").first.fill("4000")
    elif kind == "dip":
        frame.locator("#trace").evaluate("e=>e.setAttribute('y2','50')")
    elif kind == "partial":
        frame.locator("#trace").evaluate("e=>e.setAttribute('x2','130')")
    elif kind == "axis":
        frame.locator(".time").nth(1).evaluate("e=>e.textContent='2400'")
    elif kind == "occluded":
        frame.locator("#chart").evaluate(
            "e=>{const r=e.getBoundingClientRect(),d=document.createElement('div');d.style=`position:fixed;left:${r.left+40}px;top:${r.top+20}px;width:60px;height:20px;background:red`;document.body.append(d)}"
        )
    else:
        frame.locator("body").evaluate(
            "(e,h)=>e.insertAdjacentHTML('beforeend',h)",
            "<div role=dialog>stop</div>" if kind == "modal" else "<input type=password>",
        )
    with pytest.raises(BrowserSafetyStop):
        choose(window_page, tmp_path, evidence)
    assert frame.evaluate("window.fixtureSelections") == 0


@pytest.mark.parametrize("change", ["late_dip", "report", "replacement", "external", "frame"])
def test_change_after_reservation_blocks_select_and_future_output(window_page, tmp_path, monkeypatch, change):
    import habfly.browser_planet_window_choice as module

    page, frame = window_page
    evidence = source(window_page, tmp_path)
    original = module.persist_json

    def persist(path, value):
        original(path, value)
        if path == tmp_path / "choice/reserved.json":
            if change == "late_dip":
                frame.locator("#trace").evaluate("e=>e.setAttribute('y2','50')")
            elif change == "report":
                evidence["evidence_path"].write_text("{}")
            elif change == "replacement":
                frame.get_by_role("combobox").evaluate("e=>e.replaceWith(e.cloneNode(true))")
            elif change == "frame":
                page.locator("iframe").nth(1).evaluate("e=>e.replaceWith(e.cloneNode(true))")
            else:
                page.get_by_role("checkbox").check()

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        choose(window_page, tmp_path, evidence)
    stopped = json.loads((tmp_path / "choice/stopped.json").read_text())
    assert stopped["reservation_created"] and not stopped["write_may_have_occurred"]
    assert frame.evaluate("window.fixtureSelections") == 0
    if change != "report":
        with pytest.raises(BrowserSafetyStop, match="already_reserved"):
            select_no_planet_from_window(page, config(), tmp_path / "again", run_history=tmp_path, **evidence)


@pytest.mark.parametrize("effect", ["answer", "feedback", "class", "replacement", "dialog"])
def test_uncertain_native_selection_stops_without_rollback_or_retry(window_page, tmp_path, effect):
    page, frame = window_page
    evidence = source(window_page, tmp_path)
    script = {
        "answer": "document.querySelector('#line_shift').value='0'",
        "feedback": "document.body.append('unexpected feedback')",
        "class": "document.querySelector('.choice label').style.borderColor='red'",
        "replacement": "e.replaceWith(e.cloneNode(true))",
        "dialog": "confirm('private confirmation')",
    }[effect]
    frame.get_by_role("combobox").evaluate(
        "(e,code)=>{const old=e.onchange;e.onchange=()=>{old();new Function('e',code)(e)}}", script
    )
    with pytest.raises(BrowserSafetyStop):
        choose(window_page, tmp_path, evidence)
    assert frame.evaluate("window.fixtureSelections") == 1
    raw = (tmp_path / "choice/stopped.json").read_text()
    assert json.loads(raw)["write_may_have_occurred"] and "private confirmation" not in raw
    assert not (tmp_path / "choice/confirmed.json").exists()
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        select_no_planet_from_window(page, config(), tmp_path / "again", run_history=tmp_path, **evidence)


def test_legacy_axis_uses_first_5000_without_rewriting_duration(window_page, tmp_path):
    _, frame = window_page
    frame.get_by_role("textbox").first.fill("10000")
    frame.locator(".time").nth(1).evaluate("e=>e.textContent='5000'")
    frame.locator(".time").nth(2).evaluate("e=>e.textContent='10000'")
    receipt = choose(window_page, tmp_path, source(window_page, tmp_path))
    assert receipt["examined_day_interval"] == [0, 5000]
    assert frame.get_by_role("textbox").first.input_value() == "10000"


def test_inherited_class_paint_may_clear_without_classification_claim(window_page, tmp_path):
    _, frame = window_page
    frame.locator("body").evaluate("""e=>{
      e.insertAdjacentHTML('beforeend','<style>.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}</style>');
      document.querySelectorAll('.choice')[1].classList.add('selected');
      const c=document.querySelector('select'),old=c.onchange;
      c.onchange=()=>{old();document.querySelectorAll('.choice').forEach(e=>e.classList.remove('selected'))};
    }""")
    receipt = choose(window_page, tmp_path, source(window_page, tmp_path), preserve_painted_class="ice_giant")
    assert receipt["inherited_paint_cleared"] and not receipt["class_selection_verified"]
    assert receipt["class_writes"] == 0 and receipt["painted_class_after"] is None
