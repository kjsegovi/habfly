"""Intercepted demo-shortcut transport, never physical or learned correctness.

SVG pixels and stellar prerequisites are deliberate fixtures. Native No/Save,
captures, source-chain verification and canonical import use production code.
All browser requests are intercepted; no HabWorlds attempt is started.
"""
# ruff: noqa: F401,F811

import os

import pytest
from test_baseline_band_chromium import band_page
from test_browser_no_planet_workflow import navigation_seam, run, snapshot, stellar_sources
from test_browser_numeric import chromium, config, page
from test_browser_planet_window_baseline_edge_chromium import edge_page, read, sha
from test_browser_planet_window_choice import window_page
from test_browser_project_inventory import row

import habfly.browser_planet_window_choice as choice_module
from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.browser_no_planet_save import save_no_planet_work
from habfly.browser_observation_progress import capture_observation_progress
from habfly.browser_project_inventory import verify_project_inventory
from habfly.browser_project_navigation import LIST_LABELS
from habfly.browser_window_replay import load_planet_window_capture
from habfly.planet_window_baseline_band import analyze_planet_window as original_analyze
from habfly.planet_window_single_event import policy_manifest
from habfly.project_evidence import import_verified_no_planet
from habfly.project_progress import ProjectJournal

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Explicit opt-in for isolated, fully intercepted Chromium",
)


def paint_feature(frame, x):
    frame.locator("#edge-pattern").evaluate(
        """(group,x)=>{
          function rect(x,y,w,h,rgb){
            const e=document.createElementNS('http://www.w3.org/2000/svg','rect');
            for(const [k,v] of Object.entries({x,y,width:w,height:h,fill:`rgb(${rgb})`}))
              e.setAttribute(k,v);
            group.append(e);
          }
          rect(x,21,1,1,'60,92,105');rect(x+1,21,1,1,'46,65,74');
          rect(x,22,1,6,'50,82,96');rect(x+1,22,1,6,'30,49,58');
          rect(x,28,1,1,'45,74,86');rect(x+1,28,1,1,'25,41,48');
        }""",
        x,
    )


@pytest.fixture
def single_page(band_page):
    page, frame = band_page
    frame.locator("#edge-pattern").evaluate(
        """group=>{
          for(const e of [...group.children])
            if(e.getAttribute('y')==='21' && e.getAttribute('width')==='1'
               && ['241','242'].includes(e.getAttribute('x')))e.remove();
          // Unlike the baseline-edge fixture's tolerated tinted border, this
          // shortcut requires a genuinely neutral final border column.
          for(const e of [...group.children])
            if(e.getAttribute('x')==='260' && e.getAttribute('y')==='20'
               && e.getAttribute('height')==='1')e.setAttribute('fill','rgb(51,51,51)');
        }"""
    )
    paint_feature(frame, 243)
    return page, frame


def capture_source(page, settings, root):
    directory = root / "source"
    capture_observation_progress(page, settings, directory, requested_days=5000)
    persist_json(directory / "policy.json", policy_manifest())
    return {"evidence_path": directory / "report.json", "evidence_sha256": sha(directory / "report.json")}


def select(page, settings, root, evidence):
    return choice_module.select_no_planet_from_window(
        page, settings, root / "choice", run_history=root, policy=policy_manifest(), **evidence
    )


@pytest.mark.parametrize("pinned", [False, True])
def test_native_single_event_shortcut_save_workflow_import_and_offline_replay(
    single_page, tmp_path, monkeypatch, pinned
):
    page, frame = single_page
    settings = config().model_copy(update={"pinned_control_capture": pinned})
    evidence = capture_source(page, settings, tmp_path)
    original = read(evidence["evidence_path"])
    png = (tmp_path / "source/chart.png").read_bytes()
    assert (
        original_analyze(png, original["time_axis_labels"], original["flux_axis_labels"])["planet_decision"]
        is None
    )
    choice = select(page, settings, tmp_path, evidence)
    assert frame.evaluate("window.fixtureSelections") == 1
    assert frame.get_by_role("combobox").input_value() == "No"
    for name in ("saved_evidence", "fresh_evidence", "preselect_evidence"):
        analysis = choice[name]["analysis"]
        assert analysis["policy"] == policy_manifest()
        assert analysis["reason"] == "user_approved_single_event_shortcut"
        assert analysis["possible_planet_ignored"] is True
        assert analysis["visible_candidate_events"] == 1
        assert all(
            analysis[k] is False
            for k in (
                "absence_proven",
                "scientific_verified",
                "learned_perception",
                "training_label",
                "task_completed",
            )
        )
    saved = save_no_planet_work(
        page,
        settings,
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
    assert workflow["planet"]["policy_label"] == policy_manifest()["version"]
    assert not workflow["scientific_verified"] and not workflow["project_completed"]
    frame.locator("body").evaluate(
        "(e,html)=>e.innerHTML=html",
        "<div>Funding $50000 Data Quality 0% Scavenger Hunt 0/8 Observations Analyzed Data Star "
        + " ".join(LIST_LABELS["stellar"])
        + "</div>"
        + row(workflow["star"], "0.045 370 5.68E-10 1 1 UV 1 Main Sequence 1 1 1 Ga")
        + "<div>viewing 1-1 of 1 total collected 1</div>",
    )
    inventory = verify_project_inventory(
        page, settings, tmp_path / "inventory", expected_stars=[workflow["star"]]
    )
    assert inventory["collection_count_verified"]
    journal = ProjectJournal(tmp_path, project_id="single-event-fixture", attempt_id="fixture-only").create()
    imported = import_verified_no_planet(journal, tmp_path, tmp_path / "inventory", tmp_path / "workflow")
    assert imported["progress"]["verified"] == 1
    assert not imported["progress"]["project_completed"]
    planet_record = journal.load().reduce().stars[imported["star_id"]].planet
    assert planet_record.policy_label == policy_manifest()["version"]
    assert planet_record.decision.scientific_verified is False
    replay = load_planet_window_capture(tmp_path / "source")
    assert replay["policy"] == policy_manifest()
    assert replay["possible_planet_ignored"] is True
    assert not replay["write_authorized"] and not replay["task_completed"]
    assert (tmp_path / "source/chart.png").read_bytes() == png


def test_second_feature_appearing_after_source_blocks_shortcut(single_page, tmp_path):
    page, frame = single_page
    evidence = capture_source(page, config(), tmp_path)
    paint_feature(frame, 150)
    with pytest.raises(BrowserSafetyStop):
        select(page, config(), tmp_path, evidence)
    assert frame.evaluate("window.fixtureSelections") == 0
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not (tmp_path / "choice/confirmed.json").exists()
