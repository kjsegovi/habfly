"""Intercepted non-main terrestrial transport/import, not learned acceptance.

The WD native prefix reuses the gas fixture's construction up to its finalizer,
but selects the actual terrestrial control and stops before any Planet Save.
Only private gate replay is a production-validator seam. Model event values and
water-chamber indicators are explicit synthetic fixture evidence, never real
weights, private evaluation cases, course labels or HabWorlds traffic.
All native copying, public-source reconstruction, three-tab navigation, final
Habitability Save and the new-mode journal importer use production code.
"""

# ruff: noqa: F811 - imported pytest fixtures

import hashlib
import os

import pytest
import test_supplied_browser_workflows_chromium as prefix_fixture
from test_browser_gas_controls import gas_page
from test_browser_habitability_numeric import habitat_page
from test_browser_habitability_save import prepare_phase_choice_sources
from test_browser_no_planet_workflow import snapshot, write
from test_browser_numeric import config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_raster_planet_evidence import raster_page, read  # noqa: F401
from test_browser_tooltip_reference_chromium import chromium  # noqa: F401 - launch-site idle gate
from test_project_evidence import inventory

import habfly.browser_supplied_provenance as provenance
import habfly.habitability_supplied_inputs as temperature_pack
from habfly.browser import BrowserSafetyStop
from habfly.browser_gas_controls import compare_visible_gas_candidates, select_reference_gases
from habfly.browser_habitability_actions import HabitabilityMenuSession
from habfly.browser_habitability_numeric import HabitabilityNumericSession
from habfly.browser_habitability_save import save_habitability_work
from habfly.browser_terrestrial_workflow import verify_terrestrial_workflow
from habfly.contracts import RuntimeEvent
from habfly.project_progress import ProjectJournal
from habfly.project_terrestrial_evidence import import_verified_terrestrial
from habfly.supplied_browser_modes import TEMPERATURE_SUPPLIED_MODE, TERRESTRIAL_SUPPLIED_MODE

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must authorize an intercepted Chromium slot",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_prefix(raster_page, history, monkeypatch):
    """Reuse fixture construction, not a forged or relaxed production reader."""
    ready = {}

    class PrefixReady(Exception):
        pass

    actual_select = prefix_fixture.select_planet_class

    def terrestrial(*args, **kwargs):
        assert args[3] == "gas_giant"  # fixture's sole branch parameter
        return actual_select(*args[:3], "terrestrial", **kwargs)

    def stop_before_finalization(*args, **kwargs):
        ready.update(kwargs)
        ready["planet_gate"] = provenance.load_archived_supplied_input_transfer_gate
        raise PrefixReady

    with monkeypatch.context() as fixture_patch:
        fixture_patch.setattr(prefix_fixture, "select_planet_class", terrestrial)
        fixture_patch.setattr(prefix_fixture, "PositiveFinalizationSteps", stop_before_finalization)
        with pytest.raises(PrefixReady):
            prefix_fixture.test_white_dwarf_visible_inputs_native_save_and_new_mode_import(
                raster_page, history, fixture_patch
            )
    assert ready["supplied_inputs"] is True
    assert read(history / "planet-class/confirmed.json")["value"] == "terrestrial"
    assert not (history / "finalize").exists()  # no gas finalization or prior Save
    return ready


def temperature_stream(directory, native, supplied_link, gate_link):
    identity = {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_metadata_sha256": "d" * 64,
        "graph_hash": "b" * 64,
        "knowledge_pack_hash": temperature_pack.load_supplied_temperature_pack().checksum,
        "supplied_input_adapter_sha256": temperature_pack.adapter_manifest()["sha256"],
        "original_knowledge_pack_hash": temperature_pack.BASE_PACK_HASH,
        "original_scope": temperature_pack.LEGACY_SCOPE,
        "original_checkpoint_content_hash": "e" * 64,
        "supplied_star_class": "white_dwarf",
        "classification_source": "supplied_not_learned",
        "supplied_stellar_inputs": supplied_link,
        "supplied_input_transfer_gate": gate_link,
        "optimizer_updates": 0,
        "supplied_greenhouse_increment": 10,
        "greenhouse_selection_learned": False,
        "gas_identification_learned": False,
        "water_phase_learned": False,
        "habitability_decision_learned": False,
        "calibration_scope": "browser_transfer_not_calibrated",
        "surface_proposal_is_not_browser_readback": True,
        "synthetic_fixture_only": True,
    }
    report = {
        "scope": TEMPERATURE_SUPPLIED_MODE,
        "provenance": identity,
        "outcome": "equilibrium_transport_verified",
        "steps": 1,
        "checkpoint_unchanged": True,
        "sources_unchanged": True,
        "optimizer_updates": 0,
        "equilibrium_transport_verified": True,
        "local_proposals": {"equilibrium_temp": 759.4, "surface_temp": 769.4},
        "native_receipt": native,
        "task_completed": False,
        "course_acceptance_passed": False,
        "saved": False,
        "assessed": False,
        "submitted": False,
    }
    events = [
        ("hello", {"task": "habitability_calculations", "policy": "checkpoint", **identity}),
        ("action_proposed", {"kind": "STOP", "action_source": "checkpoint", "synthetic_fixture_only": True}),
        ("neural_activity", {"activity_source": "checkpoint", "synthetic_fixture_only": True}),
        (
            "action_proposed",
            {
                "kind": "TYPE",
                "destination": "equilibrium_temp",
                "value": "759.4",
                "action_source": "deterministic_exact_transport_of_checkpoint_result",
            },
        ),
        ("action_result", {"native_copy": native}),
        ("episode_summary", report),
    ]
    raw = "".join(
        RuntimeEvent(
            event=kind, sequence=i, run_id="synthetic-wd-temperature", payload=data
        ).model_dump_json()
        + "\n"
        for i, (kind, data) in enumerate(events)
    )
    (directory / "events.jsonl").write_text(raw)
    write(directory / "report.json", {**report, "events_sha256": hashlib.sha256(raw.encode()).hexdigest()})


def test_white_dwarf_temperature_habitability_save_and_strict_import(raster_page, tmp_path, monkeypatch):
    page, frame = raster_page
    prefix = native_prefix(raster_page, tmp_path, monkeypatch)
    assert frame.evaluate("window.fixtureSaves") == 0
    planet_html = snapshot(frame)
    tabs = frame.locator("nav").evaluate("e=>e.outerHTML")
    habitat_page.__wrapped__(page)
    gas_page.__wrapped__((page, frame))
    frame.locator("svg").last.evaluate("e=>document.querySelector('button').before(e)")
    frame.get_by_text("STAR LUMINOSITY (Ls) 20.44 ORBIT RADIUS (AU) 0.5919", exact=True).evaluate(
        "e=>e.textContent='STAR LUMINOSITY (Ls) 1 ORBIT RADIUS (AU) 1'"
    )
    frame.locator("img[alt='1']").evaluate("(e,h)=>e.outerHTML=h", tabs)
    frame.locator("#temperature").evaluate("""e=>e.onblur=()=>{
      e.value=Number(e.value).toPrecision(4);document.querySelector('#surface').textContent=e.value;
    }""")
    frame.locator("#greenhouse").evaluate("""e=>e.onchange=()=>{
      const extra={Weak:10,Moderate:30,Strong:100}[e.value];
      document.querySelector('#surface').textContent=String(Number(document.querySelector('#temperature').value)+extra);
    }""")
    frame.get_by_role("button", name="Save", exact=True).evaluate("""e=>{
      const footer=document.createElement('div'),notice=document.createElement('div');
      notice.id='final-save-notice';e.before(footer);footer.append(notice,e);
      e.onclick=()=>{window.finalSaveClicks=(window.finalSaveClicks||0)+1;notice.textContent='Data saved'};
    }""")
    # Extend the intercepted app's actual native tab navigation. No production
    # navigator is patched. Planet's snapshot retains native class/raw values.
    frame.evaluate(
        """html=>{
      window.fixtureScreens.planet=html;window.fixtureSection='habitability';
      window.fixtureInstall();
    }""",
        planet_html,
    )

    gate_path = tmp_path / "fixture-temperature-gate.json"
    write(
        gate_path,
        {
            "synthetic_fixture_only": True,
            "identity": {
                "parent_metadata_sha256": "d" * 64,
                "parent_content_hash": "e" * 64,
                "parent_pack_hash": temperature_pack.BASE_PACK_HASH,
                "parent_scope": temperature_pack.LEGACY_SCOPE,
            },
        },
    )
    gate_link = {"path": gate_path.name, "sha256": sha(gate_path)}
    gate_calls = []

    def gate(book, link, **kwargs):
        gate_calls.append(kwargs["task"])
        if kwargs["task"] == "planet":
            return prefix["planet_gate"](book, link, **kwargs)
        assert kwargs["task"] == "temperature" and link == gate_link
        assert kwargs["checkpoint_sha256"] == "a" * 64 and kwargs["graph_hash"] == "b" * 64
        assert kwargs["pack_hash"] == temperature_pack.load_supplied_temperature_pack().checksum
        assert kwargs["adapter_sha256"] == temperature_pack.adapter_manifest()["sha256"]
        return book.json(gate_path)

    monkeypatch.setattr(provenance, "load_archived_supplied_input_transfer_gate", gate)
    compare_visible_gas_candidates(page, config(), tmp_path / "gas-comparison", candidates=["CO2"])
    select_reference_gases(
        page,
        config(),
        tmp_path / "gas-selection",
        comparison=tmp_path / "gas-comparison",
        gases=["CO2"],
        rationale="Synthetic fixture gas reference, not learned or scientifically verified.",
    )
    temperature = tmp_path / "temperature"
    session = HabitabilityNumericSession(page, config(), temperature / "native-copy")
    try:
        native = session.copy("759.4", "K", source="checkpoint")
    finally:
        session.close()
    supplied_link = {"path": "supplied/receipt.json", "sha256": sha(tmp_path / "supplied/receipt.json")}
    temperature_stream(temperature, native, supplied_link, gate_link)
    session = HabitabilityMenuSession(page, config(), tmp_path / "greenhouse")
    try:
        session.greenhouse_reference()
    finally:
        session.close()
    phase_dir, choice_dir = prepare_phase_choice_sources(
        page, frame, tmp_path / "habitability", phase="gas", choice="not_habitable"
    )
    saved = save_habitability_work(
        page,
        config(),
        tmp_path / "final-save",
        run_history=tmp_path,
        phase_dir=phase_dir,
        choice_dir=choice_dir,
    )
    assert saved["save_click_delivered"] and frame.evaluate("window.finalSaveClicks") == 1
    for checkbox in frame.get_by_role("checkbox").all():
        checkbox.evaluate("e=>e.toggleAttribute('checked',e.checked)")

    directories = {
        name + "_dir": tmp_path / path
        for name, path in (
            ("numeric", "numeric"),
            ("color", "color"),
            ("class", "class"),
            ("raw", "copy"),
            ("derived", "derived"),
            ("planet_class", "planet-class"),
            ("gas_comparison", "gas-comparison"),
            ("gas_selection", "gas-selection"),
            ("temperature", "temperature"),
            ("greenhouse", "greenhouse"),
            ("phase", "habitability/phase"),
            ("choice", "habitability/choice"),
            ("save", "final-save"),
        )
    }
    workflow = tmp_path / "terrestrial-workflow"
    receipt = verify_terrestrial_workflow(
        page, config(), workflow, run_history=tmp_path, supplied_inputs=True, **directories
    )
    assert receipt["mode"] == TERRESTRIAL_SUPPLIED_MODE and receipt["task_completed"]
    assert receipt["classification"] == "white_dwarf" and receipt["planet"]["value"] == "terrestrial"
    assert receipt["habitability"]["outcome"] == "not_habitable"
    assert receipt["temperature_provenance"]["supplied_stellar_inputs"] == supplied_link
    assert receipt["derived_provenance"]["supplied_stellar_inputs"] == supplied_link
    assert set(receipt["current_screen_sha256"]) == {"stellar", "planet", "habitability"}
    inv = inventory(tmp_path, ["JYREMIS"], "wd-terrestrial-native", classes={"JYREMIS": "white_dwarf"})
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="synthetic-wd-terrestrial").create()
    result = import_verified_terrestrial(journal, tmp_path, inv, workflow)
    assert result["progress"]["verified"] == 1 and not result["progress"]["project_completed"]
    assert result["evidence_scopes"]["temperature_calculations"]["scope"] == TEMPERATURE_SUPPLIED_MODE
    assert result["evidence_scopes"]["supplied_stellar_inputs"]["actual_class"] == "white_dwarf"
    assert not result["evidence_scopes"]["supplied_stellar_inputs"]["learned_stellar_mass_radius"]
    assert set(gate_calls) == {"planet", "temperature"}
    before = journal.path.read_bytes()
    assert import_verified_terrestrial(journal, tmp_path, inv, workflow)["idempotent"]
    assert journal.path.read_bytes() == before
    assert frame.evaluate("window.finalSaveClicks") == 1 and frame.evaluate("window.fixtureSaves") == 0
    assert not page.get_by_role("checkbox").is_checked()
    assert all(
        receipt[key] is False
        for key in ("scientific_verified", "training_label", "submitted", "project_completed")
    )

    # Mutation rejection reuses the same native chain without another browser
    # action or journal append; the original fixture bytes are restored after.
    path = temperature / "report.json"
    raw = path.read_bytes()
    changed = read(path)
    changed["provenance"]["supplied_star_class"] = "red_giant"
    write(path, changed)
    try:
        with pytest.raises((BrowserSafetyStop, ValueError)):
            import_verified_terrestrial(journal, tmp_path, inv, workflow)
        assert journal.path.read_bytes() == before
    finally:
        path.write_bytes(raw)
