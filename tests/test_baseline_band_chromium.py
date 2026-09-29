"""Fully intercepted visible control/crop transport, not live HabWorlds evidence.

The fixture deliberately paints the faint RGB samples, not a natural transit.
No detector, policy, source reader, save receipt or importer is patched.
"""
# ruff: noqa: F401,F811

import os

import pytest
from test_browser_no_planet_workflow import navigation_seam, run, snapshot, stellar_sources
from test_browser_numeric import chromium, config, page
from test_browser_planet_window_baseline_edge_chromium import edge_page, read, sha
from test_browser_planet_window_choice import window_page

import habfly.browser_planet_window_choice as choice_module
from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.browser_no_planet_save import save_no_planet_work
from habfly.browser_observation_progress import capture_observation_progress
from habfly.browser_window_replay import load_planet_window_capture
from habfly.planet_window_baseline_band import policy_manifest
from habfly.planet_window_baseline_edge import analyze_planet_window as old_analyze

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Explicit opt-in for isolated, fully intercepted Chromium",
)


@pytest.fixture
def band_page(edge_page):
    page, frame = edge_page
    frame.locator("#edge-pattern").evaluate("""group=>{
      for(const e of [...group.children])if(e.getAttribute('y')==='21')e.remove();
      function rect(x,y,w,h,rgb){
        const e=document.createElementNS('http://www.w3.org/2000/svg','rect');
        for(const [k,v] of Object.entries({x,y,width:w,height:h,fill:`rgb(${rgb})`}))e.setAttribute(k,v);
        group.append(e);
      }
      rect(30,21,230,1,'26,26,26');
      rect(241,21,1,1,'29,36,38');rect(242,21,1,1,'33,46,51');
    }""")
    return page, frame


def source(fixture, history):
    directory = history / "source"
    capture_observation_progress(fixture[0], config(), directory, requested_days=5000)
    persist_json(directory / "policy.json", policy_manifest())
    return {"evidence_path": directory / "report.json", "evidence_sha256": sha(directory / "report.json")}


def select(fixture, history, evidence):
    return choice_module.select_no_planet_from_window(
        fixture[0],
        config(),
        history / "choice",
        run_history=history,
        policy=policy_manifest(),
        **evidence,
    )


def test_native_band_no_save_workflow_and_replay(band_page, tmp_path, monkeypatch):
    page, frame = band_page
    evidence = source(band_page, tmp_path)
    report = read(evidence["evidence_path"])
    png = (tmp_path / "source/chart.png").read_bytes()
    assert old_analyze(png, report["time_axis_labels"], report["flux_axis_labels"])["planet_decision"] is None
    choice = select(band_page, tmp_path, evidence)
    assert frame.evaluate("window.fixtureSelections") == 1
    assert frame.get_by_role("combobox").input_value() == "No"
    for label in ("saved_evidence", "fresh_evidence", "preselect_evidence"):
        analysis = choice[label]["analysis"]
        assert analysis["policy"] == policy_manifest()
        assert analysis["baseline_band_evidence"]["subpixel_features_not_resolved"]
        assert not analysis["scientific_verified"] and not analysis["training_label"]
    saved = save_no_planet_work(
        page,
        config(),
        tmp_path / "save",
        run_history=tmp_path,
        choice_path=tmp_path / "choice/confirmed.json",
        choice_sha256=sha(tmp_path / "choice/confirmed.json"),
    )
    assert saved["data_saved_notice_observed"] and saved["answers_unchanged"]
    assert frame.evaluate("window.fixtureSaves") == 1
    for name in ("line_shift", "brightness_drop", "period_days"):
        assert frame.locator("#" + name).input_value() == ""
    planet = snapshot(frame)
    stellar = stellar_sources(page, frame, tmp_path)
    navigation_seam(monkeypatch, frame, {"stellar": stellar, "planet": planet})
    workflow = run(page, tmp_path)
    assert workflow["task_completed"] and workflow["planet"]["outcome"] == "no_planet"
    assert not workflow["scientific_verified"] and not workflow["project_completed"]
    replay = load_planet_window_capture(tmp_path / "source")
    assert replay["policy"] == policy_manifest()
    assert not replay["write_authorized"] and not replay["task_completed"]


@pytest.mark.parametrize("color", ["80,132,154", "29,36,38"])
def test_native_deeper_feature_added_after_source_blocks_no(band_page, tmp_path, color):
    _, frame = band_page
    evidence = source(band_page, tmp_path)
    frame.locator("#edge-pattern").evaluate(
        """(group,rgb)=>{
      const e=document.createElementNS('http://www.w3.org/2000/svg','rect');
      for(const [k,v] of Object.entries({x:100,y:22,width:2,height:8,fill:`rgb(${rgb})`}))e.setAttribute(k,v);
      group.append(e);
    }""",
        color,
    )
    with pytest.raises(BrowserSafetyStop):
        select(band_page, tmp_path, evidence)
    assert frame.evaluate("window.fixtureSelections") == 0
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not (tmp_path / "choice/confirmed.json").exists()
