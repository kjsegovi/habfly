"""Intercepted visible-state integration, not proof that the site persisted work.

No live site or learned policy runs. The shortcut and stellar inputs are fixture
prerequisites; native readback, source verification, workflow and import are real.
"""
# ruff: noqa: F401,F811

import hashlib
import os

import pytest
from test_baseline_band_chromium import band_page
from test_browser_no_planet_workflow import load, navigation_seam, run, snapshot, stellar_sources
from test_browser_numeric import chromium, config, page
from test_browser_planet_window_baseline_edge_chromium import edge_page, read, sha
from test_browser_planet_window_choice import window_page
from test_browser_project_inventory import row
from test_single_event_chromium import capture_source, select, single_page

import habfly.browser_no_planet_save as save_module
from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.browser_autosave import ACKNOWLEDGEMENT_SOURCE, MODE
from habfly.browser_no_planet_readback import readback_no_planet_work
from habfly.browser_project_inventory import verify_project_inventory
from habfly.browser_project_navigation import LIST_LABELS
from habfly.project_evidence import import_verified_no_planet
from habfly.project_progress import ProjectJournal

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Explicit opt-in for isolated, fully intercepted Chromium",
)


def readback(page, settings, root):
    return readback_no_planet_work(
        page,
        settings,
        root / "save",
        run_history=root,
        choice_path=root / "choice/confirmed.json",
        choice_sha256=sha(root / "choice/confirmed.json"),
    )


def forbid_save(*args, **kwargs):
    raise AssertionError("Autosave readback must not inspect or actuate explicit Save")


@pytest.mark.parametrize("pinned", [False, True])
def test_no_save_control_needed_for_visible_workflow_import_and_offline_reload(
    single_page, tmp_path, monkeypatch, pinned
):
    page, frame = single_page
    settings = config().model_copy(update={"pinned_control_capture": pinned})
    # The current visible screen has no Save control at all. Existing banners
    # and timers are not prerequisites or persistence evidence.
    frame.get_by_role("button", name="Save", exact=True).evaluate_all("es=>es.forEach(e=>e.remove())")
    choice = select(page, settings, tmp_path, capture_source(page, settings, tmp_path))
    monkeypatch.setattr(save_module, "save_no_planet_work", forbid_save)
    monkeypatch.setattr(save_module, "_notice", forbid_save)
    receipt = readback(page, settings, tmp_path)
    assert receipt["mode"] == MODE
    assert receipt["visible_readback_verified"] is True
    assert receipt["browser_actions"] == 0
    for field in (
        "save_click_delivered",
        "save_acknowledgement_verified",
        "persistence_verified",
        "cross_session_persistence_verified",
        "scientific_verified",
        "task_completed",
    ):
        assert receipt[field] is False
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not (tmp_path / "no-planet-save-reservations").exists()
    assert not any((tmp_path / "save").rglob("*dispatch*"))
    assert not (tmp_path / "save/acknowledgement.json").exists()
    assert receipt["star"] == choice["star"]

    planet = snapshot(frame)
    stellar = stellar_sources(page, frame, tmp_path)
    navigation_seam(monkeypatch, frame, {"stellar": stellar, "planet": planet})
    workflow = run(page, tmp_path)
    assert workflow["task_completed"] is True
    assert workflow["save_strategy"] == "autosave"
    assert workflow["save_authority"] == "visible_readback_only"
    assert workflow["save_acknowledgement_source"] == ACKNOWLEDGEMENT_SOURCE
    assert workflow["save_acknowledgement_verified"] is False
    assert workflow["source_save_click_delivered"] is False
    assert workflow["persistence_verified"] is False
    assert workflow["scientific_verified"] is False
    frame.locator("body").evaluate(
        "(e,html)=>e.innerHTML=html",
        "<div>Funding $50000 Data Quality 0% Scavenger Hunt 0/8 Observations Analyzed Data Star "
        + " ".join(LIST_LABELS["stellar"])
        + "</div>"
        + row(workflow["star"], "0.045 370 5.68E-10 1 1 UV 1 Main Sequence 1 1 1 Ga")
        + "<div>viewing 1-1 of 1 total collected 1</div>",
    )
    verify_project_inventory(page, settings, tmp_path / "inventory", expected_stars=[workflow["star"]])
    journal = ProjectJournal(tmp_path, project_id="autosave-fixture", attempt_id="fixture-only").create()
    imported = import_verified_no_planet(journal, tmp_path, tmp_path / "inventory", tmp_path / "workflow")
    assert imported["progress"]["verified"] == 1
    assert imported["progress"]["project_completed"] is False
    # An entirely offline source rebuild preserves the authority distinction.
    bundle = load(tmp_path)
    assert bundle["save_strategy"] == "autosave"
    assert bundle["save_click_delivered"] is False
    assert bundle["acknowledgement_source"] == ACKNOWLEDGEMENT_SOURCE


@pytest.mark.parametrize("change", ["answer", "choice", "prior_save"])
def test_autosave_does_not_ignore_changed_answers_or_reinterpret_a_prior_save(single_page, tmp_path, change):
    page, frame = single_page
    choice = select(page, config(), tmp_path, capture_source(page, config(), tmp_path))
    if change == "answer":
        frame.locator("#line_shift").fill("12")
    elif change == "choice":
        frame.get_by_role("combobox").select_option("Yes")
    else:
        key = hashlib.sha256(choice["star"].casefold().encode()).hexdigest()
        (tmp_path / "no-planet-save-reservations").mkdir()
        persist_json(tmp_path / "no-planet-save-reservations" / f"{key}.json", {"fixture": True})
    with pytest.raises(BrowserSafetyStop):
        readback(page, config(), tmp_path)
    assert not (tmp_path / "save/confirmed.json").exists()
    assert frame.evaluate("window.fixtureSaves") == 0
