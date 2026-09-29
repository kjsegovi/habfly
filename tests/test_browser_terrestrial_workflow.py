"""Intercepted workflow receipts, not learned scores or scientific labels.

Policy events and chamber icons are explicit synthetic fixture evidence. All
native source actions run against intercepted HTML, never a HabWorlds session.
"""
# ruff: noqa: F811

import hashlib
from types import SimpleNamespace

import pytest
from test_browser_gas_controls import gas_page
from test_browser_habitability_numeric import habitat_page
from test_browser_habitability_save import prepare_phase_choice_sources
from test_browser_no_planet_workflow import read, snapshot, write
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_positive_planet_workflow import kwargs as positive_kwargs
from test_browser_positive_planet_workflow import positive as make_positive
from test_browser_raster_planet_evidence import raster_page  # noqa: F401

import habfly.browser_terrestrial_workflow as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_gas_controls import compare_visible_gas_candidates, select_reference_gases
from habfly.browser_habitability_actions import HabitabilityMenuSession
from habfly.browser_habitability_numeric import HabitabilityNumericSession
from habfly.browser_habitability_save import save_habitability_work


@pytest.fixture
def terrestrial(raster_page, tmp_path, request):
    page, frame, screens = make_positive.__wrapped__(
        raster_page, tmp_path, SimpleNamespace(param="terrestrial")
    )
    habitat_page.__wrapped__(page)
    gas_page.__wrapped__((page, frame))
    frame.locator("svg").last.evaluate("e=>document.querySelector('button').before(e)")
    frame.get_by_text("STAR LUMINOSITY (Ls) 20.44 ORBIT RADIUS (AU) 0.5919", exact=True).evaluate(
        "e=>e.textContent='STAR LUMINOSITY (Ls) 1 ORBIT RADIUS (AU) 1'"
    )
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
    compare_visible_gas_candidates(page, config(), tmp_path / "gas-comparison", candidates=["CO2"])
    select_reference_gases(
        page,
        config(),
        tmp_path / "gas-selection",
        comparison=tmp_path / "gas-comparison",
        gases=["CO2"],
        rationale="Explicit synthetic fixture reference selection, not a learned or scientific label.",
    )
    temperature = tmp_path / "temperature"
    session = HabitabilityNumericSession(page, config(), temperature / "native-copy")
    try:
        receipt = session.copy("759.4", "K", source="checkpoint")
    finally:
        session.close()
    provenance = {
        "checkpoint_sha256": "a" * 64,
        "graph_hash": "b" * 64,
        "knowledge_pack_hash": "c" * 64,
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
        "scope": "one_equilibrium_copy_with_supplied_warming_local_proposal",
        "provenance": provenance,
        "outcome": "equilibrium_transport_verified",
        "steps": 1,
        "checkpoint_unchanged": True,
        "optimizer_updates": 0,
        "equilibrium_transport_verified": True,
        "local_proposals": {"equilibrium_temp": 759.4, "surface_temp": 769.4},
        "native_receipt": receipt,
        "task_completed": False,
        "course_acceptance_passed": False,
        "saved": False,
        "assessed": False,
        "submitted": False,
    }
    # Same event wire format, explicitly fake policy events; no training claims.
    from habfly.contracts import RuntimeEvent

    events = [
        ("hello", {"task": "habitability_calculations", "policy": "checkpoint", **provenance}),
        ("action_proposed", {"kind": "STOP", "action_source": "checkpoint"}),
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
        ("action_result", {"native_copy": receipt}),
        ("episode_summary", report),
    ]
    raw = "".join(
        RuntimeEvent(event=kind, sequence=i, run_id="synthetic-temperature", payload=data).model_dump_json()
        + "\n"
        for i, (kind, data) in enumerate(events)
    )
    (temperature / "events.jsonl").write_text(raw)
    write(temperature / "report.json", {**report, "events_sha256": hashlib.sha256(raw.encode()).hexdigest()})
    session = HabitabilityMenuSession(page, config(), tmp_path / "greenhouse")
    try:
        session.greenhouse_reference()
    finally:
        session.close()
    phase, decision = getattr(request, "param", ("gas", "not_habitable"))
    phase_dir, choice_dir = prepare_phase_choice_sources(
        page, frame, tmp_path / "habitability", phase=phase, choice=decision
    )
    save_habitability_work(
        page,
        config(),
        tmp_path / "final-save",
        run_history=tmp_path,
        phase_dir=phase_dir,
        choice_dir=choice_dir,
    )
    # The shared two-tab snapshot helper predates native gas checkboxes.
    for checkbox in frame.get_by_role("checkbox").all():
        checkbox.evaluate("e=>e.toggleAttribute('checked',e.checked)")
    screens["habitability"] = snapshot(frame)
    return page, frame, screens


def kwargs(history):
    return {
        **positive_kwargs(history),
        **{
            name + "_dir": history / value
            for name, value in (
                ("gas_comparison", "gas-comparison"),
                ("gas_selection", "gas-selection"),
                ("temperature", "temperature"),
                ("greenhouse", "greenhouse"),
                ("phase", "habitability/phase"),
                ("choice", "habitability/choice"),
                ("save", "final-save"),
            )
        },
    }


def load(history):
    return module._load_sources(module._Evidence(history), **kwargs(history))


def seam(monkeypatch, frame, screens, mutate=None):
    calls = []

    def navigate(page, config, output, destination, expected_star=None):
        calls.append(destination)
        frame.locator("body").evaluate("(e,h)=>e.innerHTML=h", screens[destination])
        if mutate:
            mutate(destination)
        return {"same_star_verified": True, "destination_verified": True, "suggested_required_text": []}

    monkeypatch.setattr(module, "navigate_project", navigate)
    return calls


def run(page, history, output="terrestrial-workflow"):
    return module.verify_terrestrial_workflow(
        page, config(), history / output, run_history=history, **kwargs(history)
    )


def test_native_three_tab_workflow_retains_reference_and_learned_scopes(terrestrial, tmp_path, monkeypatch):
    page, frame, screens = terrestrial
    calls = seam(monkeypatch, frame, screens)
    result = run(page, tmp_path)
    assert calls == ["stellar", "planet", "habitability"]
    assert result["task_completed"] and result["authority"] == "visible_workflow_readback"
    assert result["planet"]["value"] == "terrestrial" and result["planet"]["approximate"]
    assert result["planet"]["measurement_provenance"] == "approximate_reference_raster"
    assert (
        result["habitability"]["outcome"] == "not_habitable" and result["habitability"]["transport_verified"]
    )
    assert not result["habitability"]["learned_habitability_decision"]
    assert (
        result["gas_selection"]["gases"] == ["CO2"]
        and not result["gas_selection"]["learned_gas_identification"]
    )
    assert result["temperature_provenance"]["surface_proposal_is_not_browser_readback"] is True
    assert set(result["current_screen_sha256"]) == {"stellar", "planet", "habitability"}
    assert result["save_acknowledgement_verified"] and result["source_save_click_delivered"]
    assert all(
        result[k] is False
        for k in (
            "scientific_verified",
            "correctness_verified",
            "project_completed",
            "submitted",
            "course_completion_verified",
            "training_label",
            "cross_session_persistence_verified",
        )
    )
    assert result["save_clicks"] == result["answer_writes"] == result["assessment_clicks"] == 0
    assert frame.evaluate("window.finalSaveClicks") == 1
    assert not any(name.startswith("save/") for name in result["source_sha256"])


@pytest.mark.parametrize(
    "terrestrial",
    [("liquid", "habitable"), ("liquid", "not_habitable"), ("solid", "not_habitable")],
    indirect=True,
)
def test_explicit_decision_not_inferred_from_liquid_or_planet(terrestrial, tmp_path):
    source = load(tmp_path)
    assert source["choice"]["choice"] == read(tmp_path / "habitability/choice/confirmed.json")["choice"]
    assert source["planet_class"] == "terrestrial"


@pytest.mark.parametrize(
    "mutation",
    [
        "gas_crop",
        "gas_flags",
        "gas_write",
        "gas_uncertain",
        "temperature_events",
        "temperature_value",
        "temperature_updates",
        "temperature_uncertain",
        "greenhouse_increment",
        "phase",
        "choice",
        "save_ack",
        "save_dispatch",
        "save_claim",
        "save_uncertain",
        "foreign_source",
    ],
)
def test_missing_changed_or_uncertain_source_never_navigates(terrestrial, tmp_path, monkeypatch, mutation):
    page, frame, screens = terrestrial
    calls = seam(monkeypatch, frame, screens)
    if mutation == "gas_crop":
        (tmp_path / "gas-comparison/CO2.png").write_bytes(b"not a chart")
    elif mutation == "temperature_events":
        (tmp_path / "temperature/events.jsonl").write_text("{}\n")
    elif mutation in {"gas_uncertain", "temperature_uncertain", "save_uncertain"}:
        path = {
            "gas_uncertain": "gas-selection/write-01-stopped.json",
            "temperature_uncertain": "temperature/finalization_failed.json",
            "save_uncertain": "final-save/stopped.json",
        }[mutation]
        write(tmp_path / path, {"automatic_retry": False})
    else:
        name, key, value = {
            "gas_flags": ("gas-comparison/report.json", "learned_gas_identification", True),
            "gas_write": ("gas-selection/write-01-confirmed.json", "selected", ["NH3"]),
            "temperature_value": ("temperature/native-copy/confirmed.json", "value", "1"),
            "temperature_updates": ("temperature/report.json", "optimizer_updates", 1),
            "greenhouse_increment": ("greenhouse/confirmed.json", "label", "Strong (+100)"),
            "phase": ("habitability/phase/confirmed.json", "label", "Solid"),
            "choice": ("habitability/choice/confirmed.json", "choice", "habitable"),
            "save_ack": ("final-save/acknowledgement.json", "notice_was_already_present", True),
            "save_dispatch": ("final-save/dispatch.json", "max_clicks", 2),
            "save_claim": ("final-save/reserved.json", "output", "another-output"),
            "foreign_source": ("final-save/confirmed.json", "phase_dir", "../foreign"),
        }[mutation]
        data = read(tmp_path / name)
        data[key] = value
        write(tmp_path / name, data)
    with pytest.raises(BrowserSafetyStop):
        run(page, tmp_path)
    assert not calls and not read(tmp_path / "terrestrial-workflow/stopped.json")["task_completed"]


@pytest.mark.parametrize(
    "mutation",
    ["current_temperature", "current_gas", "current_choice", "current_star", "source_after_navigation"],
)
def test_changed_current_context_stops_without_repairs(terrestrial, tmp_path, monkeypatch, mutation):
    page, frame, screens = terrestrial

    def mutate(section):
        if section == "habitability":
            if mutation == "source_after_navigation":
                write(tmp_path / "gas-selection/selection-stopped.json", {"automatic_retry": False})
            else:
                frame.locator("body").evaluate(
                    "(e,code)=>new Function(code)()",
                    {
                        "current_temperature": "document.querySelector('#temperature').value='1'",
                        "current_gas": "document.querySelector('#gases').innerHTML='<option selected>NH3</option>'",
                        "current_choice": "document.querySelectorAll('.choice label').forEach(e=>e.classList.toggle('picked'))",
                        "current_star": "document.querySelector('div').textContent='OTHER STAR'",
                    }[mutation],
                )

    seam(monkeypatch, frame, screens, mutate)
    with pytest.raises(BrowserSafetyStop):
        run(page, tmp_path)
    assert not (tmp_path / "terrestrial-workflow/confirmed.json").exists()
    assert frame.evaluate("window.finalSaveClicks") == 1
