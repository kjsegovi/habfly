"""One intercepted WD/gas transport fixture, not learned/course acceptance.

The sole production validator seam is the private transfer-gate archive loader.
All public Stellar/navigation/raster/raw/M-R/derived/class/Save/workflow/import
evidence is produced by actual guarded adapters and rebuilt by real readers.
Model event streams deliberately carry synthetic-fixture provenance; no weights,
private evaluation cases, or real HabWorlds sessions are opened.
"""
# ruff: noqa: F811

import hashlib
import os
from copy import deepcopy

import pytest
from test_browser_no_planet_workflow import snapshot, stellar_sources, write
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_positive_planet_workflow import derived_stream
from test_browser_raster_planet_evidence import copy, raster_page, select, sources  # noqa: F401
from test_project_evidence import inventory

import habfly.browser_supplied_provenance as provenance
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_planet_classification import select_planet_class
from habfly.browser_planet_numeric import ANSWER_UNITS, PlanetNumericSession
from habfly.browser_positive_finalize import PositiveFinalizationSteps
from habfly.browser_positive_planet_workflow import DERIVED
from habfly.browser_project_navigation import navigate_project
from habfly.browser_stellar_sources import load_stellar_sources
from habfly.planet_supplied_inputs import (
    BASE_PACK_HASH,
    LEGACY_SCOPE,
    adapter_manifest,
    load_supplied_planet_pack,
)
from habfly.planet_supplied_stellar_source import build_supplied_planet_stellar_inputs
from habfly.project_positive_evidence import import_verified_positive_planet
from habfly.project_progress import ProjectJournal
from habfly.supplied_browser_modes import DERIVED_SUPPLIED_MODE, POSITIVE_SUPPLIED_MODE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Requires sole live owner's explicit idle confirmation",
)
def test_white_dwarf_visible_inputs_native_save_and_new_mode_import(raster_page, tmp_path, monkeypatch):
    page, frame = raster_page
    frame.locator("#derived").evaluate(
        "e=>e.textContent='STAR MASS (Ms) 1 STAR RADIUS (Rs) 0.01 ORBIT (years) 0.000'"
    )
    # Controlled page setup only: first navigation must reach genuinely blank
    # visible controls. The static chart is the existing intercepted reference.
    frame.locator("input").first.fill("")
    planet_html = snapshot(frame)
    write(tmp_path / "choice/confirmed.json", {"star": "JYREMIS"})
    stellar_html = stellar_sources(page, frame, tmp_path, selected_class="white_dwarf")
    source_dirs = {name + "_dir": tmp_path / name for name in ("numeric", "color", "class")}
    stellar = load_stellar_sources(_Evidence(tmp_path), **source_dirs)
    assert stellar["bundle"]["class"] == "white_dwarf"
    assert stellar["bundle"]["prefix"] is None
    assert set(stellar["bundle"]["readbacks"]) == {"distance", "luminosity", "temperature"}

    tabs = (
        '<nav style="display:flex;gap:10px">'
        + "".join(
            f'<span class="numbered" role="img" data-destination="{name}" style="display:inline-flex;width:24px;height:24px;align-items:center;justify-content:center;background:lightblue">{n}</span>'
            for n, name in enumerate(("stellar", "planet", "habitability"), 1)
        )
        + "</nav>"
    )
    screens = {
        name: html.replace('<img alt="1">', tabs)
        for name, html in (("stellar", stellar_html), ("planet", planet_html))
    }
    frame.evaluate(
        """data=>{
      window.fixtureScreens=data;window.fixtureSection='stellar';window.fixtureSaves=0;
      window.fixtureNavigate=destination=>{
        const clone=document.body.cloneNode(true),old=document.querySelectorAll('input,select');
        [...clone.querySelectorAll('input,select')].forEach((x,i)=>{
          if(x.tagName==='INPUT')x.setAttribute('value',old[i].value);
          else [...x.options].forEach(o=>o.toggleAttribute('selected',o.value===old[i].value));
        });
        window.fixtureScreens[window.fixtureSection]=clone.innerHTML;
        window.fixtureSection=destination;
        document.body.innerHTML=window.fixtureScreens[destination];window.fixtureInstall();
      };
      window.fixtureInstall=()=>{
        for(const tab of document.querySelectorAll('.numbered'))tab.onclick=()=>window.fixtureNavigate(tab.dataset.destination);
        if(window.fixtureSection==='planet'){
          const combo=document.querySelector('select');
          combo.onchange=()=>{window.fixtureSelections++;combo.nextElementSibling.hidden=combo.value!=='Yes'};
          document.querySelectorAll('.choice label').forEach(label=>label.onclick=()=>label.parentElement.classList.add('selected'));
          if(!document.querySelector('#save-notice')){
            const style=document.createElement('style');style.textContent='.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}';document.body.append(style);
            const button=[...document.querySelectorAll('button')].find(e=>e.textContent==='Save');
            const footer=document.createElement('div'),notice=document.createElement('div');notice.id='save-notice';button.before(footer);footer.append(notice,button);
          }
          [...document.querySelectorAll('button')].find(e=>e.textContent==='Save').onclick=()=>{
            window.fixtureSaves++;document.querySelector('#save-notice').textContent='Data saved';
          };
        }
      };
      document.body.innerHTML=data.stellar;window.fixtureInstall();
    }""",
        screens,
    )
    navigation = tmp_path / "to-planet"
    nav = navigate_project(page, config(), navigation, "planet", expected_star="JYREMIS")
    assert nav["navigation_clicks"] == 1 and nav["same_star_verified"]
    frame.locator("input").first.fill("5000")
    source = sources(raster_page, tmp_path)
    select(raster_page, tmp_path, source)
    copy(raster_page, tmp_path)
    current = tmp_path / "copy/native-copies/copy-03-after"
    supplied = build_supplied_planet_stellar_inputs(
        tmp_path,
        **source_dirs,
        navigation_dir=navigation,
        raw_dir=tmp_path / "copy",
        current_capture_dir=current,
        current_capture_sha256=sha(current / "observation.json"),
        expected_star="JYREMIS",
        selected_class="white_dwarf",
        expected_pack_hash=load_supplied_planet_pack().checksum,
        expected_adapter_sha256=adapter_manifest()["sha256"],
    )
    assert supplied["actual_class"] == "white_dwarf"
    assert supplied["inputs"]["stellar_mass"]["display_text"] == "1"
    assert supplied["inputs"]["stellar_radius"]["display_text"] == "0.01"
    supplied_path = tmp_path / "supplied/receipt.json"
    write(supplied_path, supplied)

    # Only this private-proof reader is injected. It pins its explicit fixture
    # bytes into the shared source book; public receipt reconstruction is real.
    gate_path = tmp_path / "fixture-private-gate.json"
    identity = {
        "parent_metadata_sha256": "d" * 64,
        "parent_content_hash": "e" * 64,
        "parent_pack_hash": BASE_PACK_HASH,
        "parent_scope": LEGACY_SCOPE,
    }
    write(gate_path, {"synthetic_fixture_only": True, "identity": identity})
    gate_link = {"path": gate_path.name, "sha256": sha(gate_path)}
    gate_calls = []

    def recorded_gate(book, link, *, task, checkpoint_sha256, graph_hash, pack_hash, adapter_sha256):
        assert link == gate_link and task == "planet"
        assert checkpoint_sha256 == "a" * 64 and graph_hash == "b" * 64
        assert pack_hash == load_supplied_planet_pack().checksum
        assert adapter_sha256 == adapter_manifest()["sha256"]
        gate_calls.append(task)
        return book.json(gate_path)

    monkeypatch.setattr(provenance, "load_archived_supplied_input_transfer_gate", recorded_gate)

    payloads, directory = [], tmp_path / "derived"
    native = PlanetNumericSession(
        page, config(), directory / "native-copies", lambda *args: payloads.append(args)
    )
    try:
        for name in sorted(DERIVED):
            native.copy(name, "1", ANSWER_UNITS[name], source="checkpoint")
        fields = deepcopy(native.verified)
    finally:
        native.close()
    derived_stream(
        directory,
        {
            "scope": DERIVED_SUPPLIED_MODE,
            "outcome": "planet_derived_transport_verified",
            "checkpoint_unchanged": True,
            "sources_unchanged": True,
            "planet_transport_verified": True,
            "optimizer_updates": 0,
            "task_completed": False,
            "browser_acceptance_passed": False,
            "saved": False,
            "assessment_performed": False,
            "submitted": False,
            "write_attempts": sorted(DERIVED),
            "verified_fields": fields,
            "provenance": {
                "checkpoint_sha256": "a" * 64,
                "checkpoint_metadata_sha256": "d" * 64,
                "graph_hash": "b" * 64,
                "knowledge_pack_hash": load_supplied_planet_pack().checksum,
                "optimizer_updates": 0,
                "supplied_star_class": "white_dwarf",
                "classification_source": "supplied_not_learned",
                "synthetic_fixture_only": True,
                "supplied_stellar_inputs": {"path": "supplied/receipt.json", "sha256": sha(supplied_path)},
                "supplied_input_transfer_gate": gate_link,
                "supplied_input_adapter_sha256": adapter_manifest()["sha256"],
                "original_knowledge_pack_hash": BASE_PACK_HASH,
                "original_scope": LEGACY_SCOPE,
                "original_checkpoint_content_hash": "e" * 64,
            },
        },
        payloads,
    )
    select_planet_class(page, config(), tmp_path / "planet-class", "gas_giant", source="reference_diagnostic")
    owner = PositiveFinalizationSteps(
        page,
        config(),
        tmp_path / "finalize",
        run_history=tmp_path,
        **source_dirs,
        raw_dir=tmp_path / "copy",
        derived_dir=directory,
        planet_class_dir=tmp_path / "planet-class",
        planet_class_sha256=sha(tmp_path / "planet-class/confirmed.json"),
        supplied_inputs=True,
    )
    assert owner.advance()["phase"] == "save"
    assert frame.evaluate("window.fixtureSaves") == 0
    assert owner.advance()["phase"] == "verify"
    assert frame.evaluate("window.fixtureSaves") == 1
    assert owner.advance()["task_completed"], owner.state()
    assert owner.workflow_receipt["mode"] == POSITIVE_SUPPLIED_MODE
    inv = inventory(tmp_path, ["JYREMIS"], "wd-native", classes={"JYREMIS": "white_dwarf"})
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="synthetic-wd-transport").create()
    result = import_verified_positive_planet(journal, tmp_path, inv, tmp_path / "finalize/workflow")
    assert result["progress"]["verified"] == 1 and result["appended_records"] == 7
    assert not result["progress"]["project_completed"]
    assert result["evidence_scopes"]["supplied_stellar_inputs"]["actual_class"] == "white_dwarf"
    assert not result["evidence_scopes"]["supplied_stellar_inputs"]["learned_stellar_mass_radius"]
    assert gate_calls and frame.evaluate("window.fixtureSaves") == 1
    assert import_verified_positive_planet(journal, tmp_path, inv, tmp_path / "finalize/workflow")[
        "idempotent"
    ]
