"""Intercepted baseline-edge policy transport, never a live science/learning test.

The native SVG deliberately paints the independently tested baseline-AA RGB
pattern; it does not assert that Chromium generated those blends naturally.
No, Save, screenshot/axis extraction and list inventory use real production
adapters. Existing synthetic stellar numeric/color streams and a fixture-only
two-tab screen switch supply prerequisites; all strict source readers/importers
remain unpatched. Every browser request is fulfilled locally or aborted by the
imported page fixture.
"""
# ruff: noqa: F811

import hashlib
import json
import os

import pytest
from test_browser_no_planet_workflow import navigation_seam, run, snapshot, stellar_sources
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_project_inventory import row

import habfly.browser_planet_window_choice as choice_module
from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.browser_no_planet_save import save_no_planet_work
from habfly.browser_observation_progress import capture_observation_progress
from habfly.browser_project_inventory import verify_project_inventory
from habfly.browser_project_navigation import LIST_LABELS
from habfly.planet_window_baseline_edge import policy_manifest
from habfly.planet_window_policy import analyze_planet_window as frozen_analyze
from habfly.project_evidence import import_verified_no_planet
from habfly.project_progress import ProjectJournal

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Explicit opt-in required for isolated, fully intercepted Chromium",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


@pytest.fixture
def edge_page(window_page):
    page, frame = window_page
    frame.evaluate("""()=>{
      const chart=document.querySelector('#chart');
      chart.querySelector('#trace').remove();
      chart.querySelectorAll('.time').forEach(e=>e.remove());
      const ns='http://www.w3.org/2000/svg';
      const group=document.createElementNS(ns,'g');group.id='edge-pattern';
      chart.querySelector('rect').after(group);
      function rect(x,y,w,h,rgb){
        const e=document.createElementNS(ns,'rect');
        for(const [k,v] of Object.entries({x,y,width:w,height:h,fill:`rgb(${rgb})`}))e.setAttribute(k,v);
        group.append(e);
      }
      for(let x=30;x<=260;x+=23)rect(x,20,1,132,'51,51,51');
      rect(262,21,18,1,'26,26,26');
      for(let x=30;x<260;x++){
        rect(x,20,1,1,'80,131,153');
        rect(x,21,1,1,(x-30)%23===0?'55,61,64':'29,36,39');
      }
      rect(260,20,1,1,'53,56,58');
      const top=chart.getBoundingClientRect().top;
      for(let day=0;day<=5000;day+=500){
        const text=document.createElementNS(ns,'text');
        for(const [k,v] of Object.entries({class:'time',x:30.5+day*.046,y:165,
            'text-anchor':'middle','font-size':4,fill:'white'}))text.setAttribute(k,v);
        text.textContent=day;chart.append(text);
        const box=text.getBoundingClientRect();text.setAttribute('y',330-(box.top+box.height/2-top));
      }
      const save=[...document.querySelectorAll('button')].find(e=>e.textContent==='Save');
      const notice=document.createElement('div');notice.id='save-notice';
      const footer=document.createElement('div');save.before(footer);footer.append(notice,save);
      window.fixtureSaves=0;
      save.onclick=()=>{window.fixtureSaves++;notice.textContent='Data saved'};
    }""")
    return page, frame


def source(fixture, history):
    directory = history / "source"
    capture_observation_progress(fixture[0], config(), directory, requested_days=5000)
    persist_json(directory / "policy.json", policy_manifest())
    return {"evidence_path": directory / "report.json", "evidence_sha256": sha(directory / "report.json")}


def select(fixture, history, evidence, *, output="choice"):
    return choice_module.select_no_planet_from_window(
        fixture[0],
        config(),
        history / output,
        run_history=history,
        policy=policy_manifest(),
        **evidence,
    )


def test_native_baseline_edge_no_save_workflow_and_strict_collection_import(edge_page, tmp_path, monkeypatch):
    page, frame = edge_page
    evidence = source(edge_page, tmp_path)
    original = read(evidence["evidence_path"])
    png = (tmp_path / "source/chart.png").read_bytes()
    assert (
        frozen_analyze(png, original["time_axis_labels"], original["flux_axis_labels"])["reason"]
        == "unknown_plot_palette"
    )
    choice = select(edge_page, tmp_path, evidence)
    assert frame.evaluate("window.fixtureSelections") == 1
    assert frame.get_by_role("combobox").input_value() == "No"
    assert choice["policy"] == policy_manifest()
    source_pins = {}
    for label, directory in (
        ("saved_evidence", tmp_path / "source"),
        ("fresh_evidence", tmp_path / "choice/fresh-progress"),
        ("preselect_evidence", tmp_path / "choice/preselect-progress"),
    ):
        current = choice[label]
        assert read(directory / "policy.json") == choice["policy"]
        assert sha(directory / "policy.json") == current["policy_sha256"]
        assert sha(directory / "chart.png") == current["chart_sha256"]
        assert current["analysis"]["status"] == "assume_no_planet"
        assert current["analysis"]["baseline_edge_evidence"]["supported_columns"] == [30, 259]
        for key in (
            "absence_proven",
            "scientific_verified",
            "learned_perception",
            "training_label",
            "task_completed",
        ):
            assert current["analysis"][key] is False
        for name in ("policy.json", "report.json", "chart.png"):
            source_pins[str((directory / name).relative_to(tmp_path))] = sha(directory / name)
    assert choice["numeric_writes"] == choice["save_clicks"] == choice["submission_clicks"] == 0
    saved = save_no_planet_work(
        page,
        config(),
        tmp_path / "save",
        run_history=tmp_path,
        choice_path=tmp_path / "choice/confirmed.json",
        choice_sha256=sha(tmp_path / "choice/confirmed.json"),
    )
    assert frame.evaluate("window.fixtureSaves") == 1
    assert saved["data_saved_notice_observed"] and saved["answers_unchanged"]
    assert saved["task_completed"] is False and saved["correctness_verified"] is False
    for name in ("line_shift", "brightness_drop", "period_days"):
        assert frame.locator("#" + name).input_value() == ""
    for name in ("orbital_radius", "planet_mass", "planet_radius", "planet_density"):
        assert not frame.locator("#" + name).is_visible()
    planet = snapshot(frame)
    stellar = stellar_sources(page, frame, tmp_path)
    calls = navigation_seam(monkeypatch, frame, {"stellar": stellar, "planet": planet})
    workflow = run(page, tmp_path)
    assert calls == ["stellar", "planet"]
    assert workflow["task_completed"] and workflow["authority"] == "visible_workflow_readback"
    assert workflow["planet"]["outcome"] == "no_planet"
    assert workflow["habitability"]["outcome"] == "not_applicable"
    assert workflow["habitability"]["branch_applicability_verified"]
    assert not workflow["habitability"]["transport_verified"]
    assert not workflow["scientific_verified"] and not workflow["project_completed"]
    assert all(workflow["source_sha256"][path] == expected for path, expected in source_pins.items())

    # A native visible-list capture, not a caller-created inventory receipt.
    frame.locator("body").evaluate(
        "(e,html)=>e.innerHTML=html",
        "<div>Funding $50000 Data Quality 0% Scavenger Hunt 0/8 Observations Analyzed Data Star "
        + " ".join(LIST_LABELS["stellar"])
        + "</div>"
        + row(workflow["star"], "0.045 370 5.68E-10 1 1 UV 1 Main Sequence 1 1 1 Ga")
        + "<div>viewing 1-1 of 1 total collected 1</div>",
    )
    inventory = verify_project_inventory(
        page, config(), tmp_path / "inventory", expected_stars=[workflow["star"]]
    )
    assert inventory["collection_count_verified"] and inventory["browser_actions"] == 0
    journal = ProjectJournal(
        tmp_path, project_id="intercepted-baseline-edge", attempt_id="fixture-only"
    ).create()
    imported = import_verified_no_planet(journal, tmp_path, tmp_path / "inventory", tmp_path / "workflow")
    assert imported["progress"]["verified"] == 1
    assert not imported["progress"]["project_completed"]
    state = journal.load().reduce()
    star = state.stars[imported["star_id"]]
    assert star.planet.policy_label == policy_manifest()["version"]
    assert star.planet.decision.provenance == "reference_prediction"
    assert not star.planet.decision.scientific_verified
    assert not state.reservations and not state.receipts
    before = journal.path.read_bytes()
    assert import_verified_no_planet(journal, tmp_path, tmp_path / "inventory", tmp_path / "workflow")[
        "idempotent"
    ]
    assert journal.path.read_bytes() == before
    assert all(sha(tmp_path / path) == expected for path, expected in source_pins.items())
    assert not page.get_by_role("checkbox").is_checked()


@pytest.mark.parametrize("change", ["below_band_dip", "overlay"])
def test_native_late_change_blocks_no_and_preserves_reservation(edge_page, tmp_path, monkeypatch, change):
    _, frame = edge_page
    evidence = source(edge_page, tmp_path)
    original = choice_module.persist_json

    def persist(path, value):
        original(path, value)
        if path == tmp_path / "choice/reserved.json":
            if change == "below_band_dip":
                frame.locator("#chart").evaluate("""chart=>{
                  const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
                  for(const [k,v] of Object.entries({x:100,y:22,width:2,height:8,fill:'rgb(80,132,154)'}))rect.setAttribute(k,v);
                  chart.append(rect);
                }""")
            else:
                frame.locator("#chart").evaluate("""chart=>{
                  const box=chart.getBoundingClientRect(),e=document.createElement('div');
                  e.id='late-overlay';e.style=`position:fixed;left:${box.x+80}px;top:${box.y+25}px;width:100px;height:60px;background:white;z-index:9999`;
                  document.body.append(e);
                }""")

    monkeypatch.setattr(choice_module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        select(edge_page, tmp_path, evidence)
    stopped = read(tmp_path / "choice/stopped.json")
    assert stopped["reservation_created"] and stopped["write_may_have_occurred"] is False
    assert not (tmp_path / "choice/confirmed.json").exists()
    assert frame.evaluate("window.fixtureSelections") == frame.evaluate("window.fixtureSaves") == 0
    assert frame.get_by_role("combobox").input_value() == ""
    reservations = list((tmp_path / "planet-window-choice-reservations").glob("*.json"))
    assert len(reservations) == 1
    claim = reservations[0].read_bytes()
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        select(edge_page, tmp_path, evidence, output="different-output")
    assert reservations[0].read_bytes() == claim
    assert frame.evaluate("window.fixtureSelections") == frame.evaluate("window.fixtureSaves") == 0
