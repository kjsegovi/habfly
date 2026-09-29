"""Opt-in genuine adapter chain on disposable, entirely intercepted HTML.

No source validator, model loader, policy, native action, or navigation is
patched. Existing frozen checkpoints run on this public fixture, not sealed
final cases. The HTML's gas curves and chamber phase are test paint, not science.
This intentionally does NOT reuse the synthetic receipt builders in the older
workflow tests, nor invoke a preliminary Save and erase its artifacts.
"""
# ruff: noqa: F811

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from test_browser_class_setup_steps_intercepted import (
    create_steps,
    isolated_chromium,  # noqa: F401
    native_fresh_star,  # noqa: F401
)
from test_browser_gas_controls import gas_page
from test_browser_habitability_numeric import habitat_page
from test_browser_numeric import config
from test_browser_planet_window_choice import window_page
from test_browser_project_inventory import row
from test_browser_raster_planet_evidence import raster_page
from test_browser_water_chamber import helper_html, install_helper

from habfly.browser_observation_progress import capture_observation_progress
from habfly.browser_planet_classification import select_planet_class
from habfly.browser_planet_spectrum import capture_spectrum_excursion
from habfly.browser_planet_steps import PlanetDerivedSteps
from habfly.browser_project_inventory_steps import ProjectInventorySteps
from habfly.browser_project_navigation import LIST_LABELS, navigate_project
from habfly.browser_raster_planet_evidence import copy_raster_measured_inputs, select_raster_detected_planet
from habfly.browser_stellar_steps import FullStellarColorSteps, FullStellarNumericSteps
from habfly.browser_terrestrial_steps import TerrestrialSteps
from habfly.data import load_graph
from habfly.project_progress import ProjectJournal
from habfly.project_terrestrial_evidence import import_verified_terrestrial
from habfly.runtime import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Requires the sole live browser owner's explicit idle window; never run alongside preview work",
)

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "data/processed/graphs-v2/graph-2000"
STELLAR = ROOT / "experiments/lifetime-003"
COLOR = ROOT / "experiments/color-pilot-006"
PLANET = ROOT / "experiments/planet-pilot-001"
PLANET_FINAL = ROOT / "experiments/planet-final-001"
HABITAT = ROOT / "experiments/habitability-pilot-001"
HABITAT_FINAL = ROOT / "experiments/habitability-final-001"
RATIONALE = (
    "Explicit disposable-fixture visual reference decision for transport testing only. "
    "This is neither a scientific inference nor a learned gas, planet-class, or habitability label."
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def drive(component, success_key, *, max_advances=512):
    try:
        for _ in range(max_advances):
            if component.finished:
                break
            component.advance()
        assert component.finished, component.state()
        assert component.report[success_key] is True, component.report
        return component.report
    finally:
        component.close()


def snapshot(frame):
    """Serialize only test DOM presentation to retain native values across tabs."""
    return frame.locator("body").evaluate("""e=>{
      const clone=e.cloneNode(true),old=e.querySelectorAll('input,select');
      [...clone.querySelectorAll('input,select')].forEach((x,i)=>{
        if(x.tagName==='INPUT'){
          x.setAttribute('value',old[i].value);
          if(x.type==='checkbox')x.toggleAttribute('checked',old[i].checked);
        } else [...x.options].forEach(o=>o.toggleAttribute('selected',o.value===old[i].value));
      });return clone.innerHTML;
    }""")


def install_project_shell(page, frame, stellar):
    """Compose existing HTML fixtures; no evidence/artifact is manufactured.

    Adapters see only the resulting ordinary controls and public readouts.
    Fixture event handlers persist tab values and expose a one-row collection.
    """
    habitat_page.__wrapped__(page)
    gas_page.__wrapped__((page, frame))
    frame.locator("svg").last.evaluate("e=>document.querySelector('button').before(e)")
    frame.locator("div").first.evaluate("e=>e.textContent='Althinagon'")
    habitat = snapshot(frame)
    raster_page.__wrapped__(window_page.__wrapped__(page))
    frame.locator("div").first.evaluate("e=>e.textContent='Althinagon'")
    planet = snapshot(frame)
    install_helper(page, html=helper_html("gas"))
    tabs = (
        '<nav style="display:flex;gap:10px">'
        + "".join(
            f'<span class="numbered" role="img" data-destination="{name}" '
            'style="display:inline-flex;width:24px;height:24px;align-items:center;justify-content:center;background:lightblue">'
            f"{number}</span>"
            for number, name in enumerate(("stellar", "planet", "habitability"), 1)
        )
        + "</nav>"
    )
    details = {
        name: html.replace('<img alt="1">', "")
        .replace("<img alt='1'>", "")
        .replace("</div>", "</div>" + tabs, 1)
        for name, html in (("stellar", stellar), ("planet", planet), ("habitability", habitat))
    }
    list_body = (
        "<div>Funding $50000 Data Quality 0% Scavenger Hunt 0/8 Observations Analyzed Data Star "
        + " ".join(LIST_LABELS["stellar"])
        + "</div>"
        + row("Althinagon")
        + "<div>viewing 1-1 of 1 total collected 1</div><button>Save</button>"
    )
    frame.locator("body").evaluate(
        """(body,data)=>{
      body.innerHTML=`<style>body{margin:0}#content{padding-top:50px}.header{position:absolute;top:0;width:40px;height:40px}
      .choice label.selected{border-color:white}.choice label.selected::after{opacity:1;background:white}</style>
      <svg role='img' class='header' id='sky' style='left:0'><rect width='40' height='40' fill='gray'/></svg>
      <svg role='img' class='header' id='list' style='left:50px'><rect width='40' height='40' fill='gray'/></svg><div id='content'></div>`;
      window.fixtureDetails=data.details;window.fixtureSurface='detail';window.fixtureSection='stellar';
      window.fixtureNativeSaves=0;window.fixtureNativeNavigation=0;
      const content=document.querySelector('#content');
      const remember=()=>{
        if(window.fixtureSurface!=='detail')return;
        const clone=content.cloneNode(true), old=content.querySelectorAll('input,select');
        [...clone.querySelectorAll('input,select')].forEach((x,i)=>{
          if(x.tagName==='INPUT'){
            x.setAttribute('value',old[i].value);if(x.type==='checkbox')x.toggleAttribute('checked',old[i].checked);
          }else [...x.options].forEach(o=>o.toggleAttribute('selected',o.value===old[i].value));
        });window.fixtureDetails[window.fixtureSection]=clone.innerHTML;
      };
      const inputFrom=(section,id)=>{
        const e=document.createElement('div');e.innerHTML=window.fixtureDetails[section];
        return e.querySelector('#'+id).getAttribute('value');
      };
      window.fixtureNavigate=destination=>{
        remember();window.fixtureNativeNavigation++;
        if(destination==='list')window.fixtureSurface='list';
        else if(destination==='starfield')throw new Error('Unrequested fixture starfield navigation');
        else window.fixtureSection=destination;
        window.fixtureRender();
      };
      window.fixtureRender=()=>{
        content.innerHTML=window.fixtureSurface==='list'?data.list:window.fixtureDetails[window.fixtureSection];
        if(window.fixtureSurface==='list'){
          const v=id=>inputFrom('stellar',id);
          const detail=document.createElement('div');detail.innerHTML=window.fixtureDetails.stellar;
          const color=detail.querySelector('select').value;
          const prefix=detail.querySelector('#prefix').selectedOptions[0].textContent;
          content.querySelector('.star-row .name').parentElement.nextElementSibling.textContent=
            ` 0.045 370 5.68E-10 ${v('distance')} ${v('luminosity')} ${color} ${v('temperature')} Main Sequence ${v('mass')} ${v('radius')} ${v('lifetime')} ${prefix}`;
        }else{
          for(const tab of content.querySelectorAll('.numbered'))tab.onclick=()=>window.fixtureNavigate(tab.dataset.destination);
          for(const label of content.querySelectorAll('.choice label'))label.onclick=()=>{
            content.querySelectorAll('.choice label').forEach(e=>e.classList.remove('selected'));label.classList.add('selected');
          };
          if(window.fixtureSection==='planet'){
            content.querySelector('#derived').textContent=`STAR MASS (Ms) ${inputFrom('stellar','mass')} STAR RADIUS (Rs) ${inputFrom('stellar','radius')} ORBIT (years) 0.000`;
            const combo=content.querySelector('select'), panel=combo.nextElementSibling;
            combo.onchange=()=>{panel.hidden=combo.value!=='Yes'};
          }
          if(window.fixtureSection==='habitability'){
            const luminosity=inputFrom('stellar','luminosity'),radius=inputFrom('planet','orbital_radius');
            [...content.querySelectorAll('div')].find(e=>e.textContent.startsWith('STAR LUMINOSITY (Ls)')).textContent=
              `STAR LUMINOSITY (Ls) ${luminosity} ORBIT RADIUS (AU) ${radius}`;
            const gases=content.querySelector('#gases'), choices=gases.parentElement.nextElementSibling;
            gases.parentElement.lastElementChild.onclick=()=>{choices.hidden=false};
            for(const input of choices.querySelectorAll('input'))input.onchange=()=>{
              const checked=[...choices.querySelectorAll('input')].filter(e=>e.checked).map(e=>e.getAttribute('aria-label'));
              gases.replaceChildren(new Option(checked.join(', '),checked.join(', '),true,true));
              content.querySelector('#absorption').textContent=String(checked.length*10);
            };
            content.querySelector('#temperature').onblur=e=>{
              e.target.value=Number(e.target.value).toPrecision(4);
              content.querySelector('#surface').textContent=e.target.value;
            };
            content.querySelector('#greenhouse').onchange=e=>{
              const extra={Weak:10,Moderate:30,Strong:100}[e.target.value];
              content.querySelector('#surface').textContent=String(Number(content.querySelector('#temperature').value)+extra);
            };
          }
          const save=[...content.querySelectorAll('button')].find(e=>e.textContent==='Save');
          if(!save.parentElement.classList.contains('fixture-footer')){
            const footer=document.createElement('div'),notice=document.createElement('div');
            footer.className='fixture-footer';notice.className='fixture-notice';save.before(footer);footer.append(notice,save);
          }
          save.onclick=()=>{
            if(window.fixtureSection!=='habitability')throw new Error('Premature fixture Save');
            window.fixtureNativeSaves++;save.parentElement.querySelector('.fixture-notice').textContent='Data saved';
          };
        }
      };
      document.querySelector('#sky').onclick=()=>window.fixtureNavigate('starfield');
      document.querySelector('#list').onclick=()=>window.fixtureNavigate('list');
      window.fixtureRender();
    }""",
        {"details": details, "list": list_body},
    )


@pytest.mark.parametrize("pinned", [False, True])
def test_real_terrestrial_owner_chain_native_receipts_and_inventory_import(
    native_fresh_star, monkeypatch, pinned
):
    """One public fixture, real frozen inference, zero source-validator seams."""
    required = (
        STELLAR / "training/checkpoint.pt",
        COLOR / "training/checkpoint.pt",
        PLANET / "training/checkpoint.pt",
        PLANET_FINAL / "report.json",
        HABITAT / "training/checkpoint.pt",
        HABITAT_FINAL / "report.json",
        GRAPH / "graph_manifest.json",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Local frozen checkpoints/graph are not distributed; do not synthesize replacements")
    # Choose only a public capture setting; no reader, validator or model is
    # replaced. The legacy fixture/setup reader remains independently tested.
    original_config = config

    def selected_config():
        value = original_config()
        value.pinned_control_capture = pinned
        return value

    monkeypatch.setattr(sys.modules[__name__], "config", selected_config)
    # All URLs are fulfilled or aborted by native_fresh_star's context route.
    # install_helper adds a local fulfillment for the known chamber URL only.
    page, frame, history, fresh, _ = native_fresh_star(inherited_main=False)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    owner = None
    try:
        class_steps = create_steps(page, history, fresh, "main_sequence", "Ga", [])
        drive(class_steps, "setup_verified", max_advances=4)
        numeric = FullStellarNumericSteps(
            page,
            config(),
            history / "numeric",
            selected_class="main_sequence",
            lifetime_prefix="Ga",
            dataset=STELLAR,
            checkpoint=STELLAR / "training/checkpoint.pt",
            graph_path=GRAPH,
            seed=8500000,
        )
        drive(numeric, "full_stellar_numeric_transport_verified")
        color = FullStellarColorSteps(
            page,
            config(),
            history / "color",
            selected_class="main_sequence",
            lifetime_prefix="Ga",
            experiment=COLOR,
            graph_path=GRAPH,
        )
        drive(color, "color_transport_verified")
        stellar = snapshot(frame)
        install_project_shell(page, frame, stellar)
        settings = config()
        navigate_project(page, settings, history / "to-planet", "planet", expected_star="Althinagon")
        capture_observation_progress(page, settings, history / "window", requested_days=5000)
        capture_spectrum_excursion(page, settings, history / "spectrum", expected_star="Althinagon")
        select_raster_detected_planet(
            page,
            settings,
            history / "presence",
            run_history=history,
            window_report=history / "window/report.json",
            window_report_sha256=sha(history / "window/report.json"),
            spectrum_path=history / "spectrum/spectrum.json",
            spectrum_sha256=sha(history / "spectrum/spectrum.json"),
        )
        copy_raster_measured_inputs(
            page,
            settings,
            history / "raw",
            run_history=history,
            presence_path=history / "presence/confirmed.json",
            presence_sha256=sha(history / "presence/confirmed.json"),
        )
        graph = load_graph(GRAPH)
        derived = PlanetDerivedSteps(
            page,
            settings,
            history / "derived",
            pilot=PLANET,
            final_evaluation=PLANET_FINAL,
            graph=graph,
            supplied_star_class="main_sequence",
        )
        drive(derived, "planet_transport_verified")
        select_planet_class(
            page, settings, history / "planet-class", "terrestrial", source="reference_diagnostic"
        )
        assert frame.evaluate("window.fixtureNativeSaves") == 0
        assert not list(history.rglob("acknowledgement.json"))
        events = []
        owner = TerrestrialSteps(
            page,
            settings,
            history / "terrestrial",
            run_history=history,
            numeric_dir=history / "numeric",
            color_dir=history / "color",
            class_dir=history / "class-setup/class",
            raw_dir=history / "raw",
            derived_dir=history / "derived",
            planet_class_dir=history / "planet-class",
            planet_class_sha256=sha(history / "planet-class/confirmed.json"),
            candidates=["CO2"],
            pilot=HABITAT,
            final_evaluation=HABITAT_FINAL,
            graph=graph,
            emit=events.append,
        )
        for _ in range(160):
            if owner.finished:
                break
            if owner.phase == "awaiting_gases":
                owner.provide_gases(gases=["CO2"], rationale=RATIONALE, supplied_greenhouse_increment=10)
            elif owner.phase == "awaiting_habitability":
                owner.provide_habitability(choice="not_habitable", rationale=RATIONALE)
            else:
                owner.advance()
        assert owner.finished and owner.report["task_completed"], owner.report or owner.state()
        assert frame.evaluate("window.fixtureNativeSaves") == 1
        assert [e.model_dump(mode="json") for e in read_trace(owner.output / "events.jsonl")] == events
        proof = json.loads((owner.output / "workflow/confirmed.json").read_bytes())
        assert proof["habitability"]["outcome"] == "not_habitable"
        assert not proof["scientific_verified"] and not proof["training_label"]
        collection = ProjectInventorySteps(
            page,
            settings,
            history / "collection",
            run_history=history,
            workflow_dir=owner.output / "workflow",
            workflow_sha256=owner.report["workflow_sha256"],
            expected_star="Althinagon",
            expected_stars=["Althinagon"],
        )
        for _ in range(3):
            if collection.finished:
                break
            collection.advance()
        assert collection.finished and collection.state()["phase"] == "inventory_verified", collection.state()
        journal = ProjectJournal(
            history, project_id="intercepted-fixture", attempt_id="terrestrial-owner"
        ).create()
        imported = import_verified_terrestrial(
            journal, history, history / "collection/inventory", owner.output / "workflow"
        )
        assert imported["progress"]["verified"] == 1 and imported["appended_records"] == 7
        assert imported["progress"]["project_completed"] is False
        assert frame.evaluate("window.fixtureNativeSaves") == 1
        assert (
            page.get_by_role("checkbox", name="I am ready to submit project.", exact=True).is_checked()
            is False
        )
    finally:
        if owner is not None:
            owner.close()
        torch.set_num_threads(old_threads)
