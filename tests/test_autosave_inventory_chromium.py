"""Native autosave workflow -> actual inventory owner -> strict journal import.

All network traffic is fulfilled/aborted by the imported local browser fixture.
Stellar numeric/color event sources remain the existing declared test seam (no
model). Real chart choice, readback, source loaders, native List/tab clicks,
inventory verifier, owner scheduling, and importer are NOT mocked.
"""
# ruff: noqa: F401,F811

import os

import pytest
from test_autosave_chromium import forbid_save, readback
from test_baseline_band_chromium import band_page
from test_browser_no_planet_workflow import navigation_seam, run, snapshot, stellar_sources
from test_browser_numeric import chromium, config, page
from test_browser_planet_window_baseline_edge_chromium import edge_page, read, sha
from test_browser_planet_window_choice import window_page
from test_browser_project_inventory import row
from test_single_event_chromium import capture_source, select, single_page

import habfly.browser_no_planet_save as save_module
from habfly.browser_autosave import ACKNOWLEDGEMENT_SOURCE
from habfly.browser_project_inventory_steps import ProjectInventorySteps
from habfly.browser_project_navigation import LIST_LABELS
from habfly.project_evidence import import_verified_no_planet
from habfly.project_progress import ProjectJournal

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Explicit opt-in for isolated, fully intercepted Chromium",
)


def navigable(html):
    # Test document scaffolding models the already grounded two 40px header
    # icons and numbered tabs. It is installed BEFORE all source captures.
    header = """<style>body{margin:0}.fixture-header{position:absolute;top:0;width:40px;height:40px}</style>
    <svg role="img" class="fixture-header" style="left:0"><rect width="40" height="40" fill="gray"/></svg>
    <svg role="img" class="fixture-header" style="left:50px" onclick="fixtureInventoryNavigate('list')"><rect width="40" height="40" fill="gray"/></svg>"""
    tabs = (
        '<nav style="display:flex;gap:10px">'
        + "".join(
            f'<span role="img" onclick="fixtureInventoryNavigate(\'{name}\')" style="display:inline-flex;width:24px;height:24px;align-items:center;justify-content:center;background:lightblue">{i}</span>'
            for i, name in enumerate(("stellar", "planet", "habitability"), 1)
        )
        + "</nav>"
    )
    return (
        header
        + '<div style="padding-top:50px">'
        + html.replace('<img alt="1">', "")
        .replace("<button>Save</button>", "")
        .replace("</div>", "</div>" + tabs, 1)
        + "</div>"
    )


@pytest.mark.parametrize("pinned", [False, True])
def test_actual_inventory_owner_accepts_autosave_navigates_and_imports(
    single_page, tmp_path, monkeypatch, pinned
):
    import test_browser_no_planet_workflow as fixture_sources

    page, frame = single_page
    settings = config().model_copy(update={"pinned_control_capture": pinned})
    frame.locator("body").evaluate("(e,h)=>e.innerHTML=h", navigable(snapshot(frame)))
    # Snapshotting strips the fixture-only select callback; restore its normal
    # conditional panel behavior before producing any scientific source.
    frame.evaluate("""()=>{
      const combo=document.querySelector('select'),panel=combo.nextElementSibling;
      combo.onchange=()=>{window.fixtureSelections++;panel.hidden=combo.value!=='Yes'};
      window.fixtureInventoryClicks=0;window.fixtureInventorySection='planet';
      window.fixtureInventoryNavigate=destination=>{
        window.fixtureInventoryClicks++;
        const section=destination==='list'?window.fixtureInventorySection:destination;
        window.fixtureInventorySection=section;
        document.body.innerHTML=window.fixtureInventoryLists[section];
      };
    }""")
    old_full_html = fixture_sources.full_html
    monkeypatch.setattr(fixture_sources, "full_html", lambda: navigable(old_full_html()))
    choice = select(page, settings, tmp_path, capture_source(page, settings, tmp_path))
    monkeypatch.setattr(save_module, "save_no_planet_work", forbid_save)
    monkeypatch.setattr(save_module, "_notice", forbid_save)
    receipt = readback(page, settings, tmp_path)
    planet = snapshot(frame)
    stellar = stellar_sources(page, frame, tmp_path)
    navigation_seam(monkeypatch, frame, {"stellar": stellar, "planet": planet})
    workflow = run(page, tmp_path)
    assert workflow["task_completed"] is True
    assert workflow["save_acknowledgement_verified"] is False
    assert workflow["save_acknowledgement_source"] == ACKNOWLEDGEMENT_SOURCE
    # Fixture navigation callbacks only supply the destination UI; production
    # owner/navigation still find, bind, guard, click and verify real controls.
    lists = {
        kind: navigable(
            "<div>Funding $50000 Data Quality 0% Scavenger Hunt 0/8 Observations Analyzed Data Star "
            + " ".join(LIST_LABELS[kind])
            + "</div>"
            + row(
                workflow["star"],
                "0.045 370 5.68E-10 1 1 UV 1 Main Sequence 1 1 1 Ga"
                if kind == "stellar"
                else "- - - No - - - - -",
            )
            + "<div>viewing 1-1 of 1 total collected 1</div>"
        )
        for kind in ("planet", "stellar")
    }
    frame.evaluate("value=>window.fixtureInventoryLists=value", lists)
    owner = ProjectInventorySteps(
        page,
        settings,
        tmp_path / "inventory-owner",
        run_history=tmp_path,
        workflow_dir=tmp_path / "workflow",
        workflow_sha256=sha(tmp_path / "workflow/confirmed.json"),
        expected_star=choice["star"],
        expected_stars=[choice["star"]],
    )
    assert frame.evaluate("window.fixtureInventoryClicks") == 0
    for phase in ("to_stellar", "verify_inventory", "inventory_verified"):
        owner.advance()
        assert owner.phase == phase, owner.failure
    assert owner.finished and owner.status == "completed"
    assert owner.advances == 3 and owner.navigation_clicks == owner.navigation_attempts == 2
    assert frame.evaluate("window.fixtureInventoryClicks") == 2
    assert owner.report["task_completed"] is owner.report["project_completed"] is False
    assert receipt["save_click_delivered"] is receipt["persistence_verified"] is False
    journal = ProjectJournal(tmp_path, project_id="autosave-inventory-fixture", attempt_id="fixture").create()
    imported = import_verified_no_planet(
        journal, tmp_path, tmp_path / "inventory-owner/inventory", tmp_path / "workflow"
    )
    assert imported["progress"]["verified"] == imported["progress"]["collected"] == 1
    assert imported["progress"]["project_completed"] is False
    assert not journal.load().reduce().reservations
    assert not (tmp_path / "no-planet-save-reservations").exists()
    assert not (tmp_path / "save/dispatch.json").exists()
    assert not (tmp_path / "save/acknowledgement.json").exists()
