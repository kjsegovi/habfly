"""Injected owner scheduling tests, not native acceptance or scientific labels."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

import habfly.browser_terrestrial_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.contracts import RuntimeEvent

REAL_SOURCES = module._sources


def read(path):
    return json.loads(path.read_bytes())


def capture(report, directory):
    directory.mkdir(parents=True)
    persist_json(directory / "observation.json", report)
    digest = hashlib.sha256((directory / "observation.json").read_bytes()).hexdigest()
    persist_json(directory / "manifest.json", {"observation_sha256": digest})
    return {"observation_sha256": digest}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    state = SimpleNamespace(
        now=0,
        calls=[],
        events=[],
        hook=None,
        failure=None,
        owner=None,
        child_failure=False,
        verifier="normal",
        save_clicks=0,
    )
    graph = SimpleNamespace(checksum="graph")
    monkeypatch.setattr(module.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(module, "graph_fingerprint", lambda g: g.checksum)
    report = {
        "ignored_frame_urls": [],
        "frames": [],
        "mapping": {
            "star_name": "Terra",
            "observation": {
                "values": {
                    "measurements": {
                        "stellar_luminosity": {"display_text": "1"},
                        "orbital_radius": {"display_text": "1"},
                        "pressure": {"display_text": "0.8"},
                    },
                    "readouts": {"absorption": {"display_text": "50"}, "surface_temp": {"display_text": ""}},
                    "selected_gases": [],
                    "equilibrium_temp": {"value": ""},
                    "greenhouse": None,
                    "water_phase": None,
                    "choice": None,
                }
            },
        },
    }
    values = lambda: report["mapping"]["observation"]["values"]
    bundle = {
        "star": "Terra",
        "planet_class": "terrestrial",
        "class_capture": {"fixture": True},
        "save_capture": {"fixture": True},
        "readbacks": {"luminosity": {"display_value": "1"}},
        "planet_readbacks": {"orbital_radius": {"display": {"display_value": "1"}}},
    }
    directories = {}
    for name in ("numeric", "color", "class", "raw", "derived", "planet_class"):
        directory = tmp_path / "source" / name
        directory.mkdir(parents=True)
        persist_json(directory / "confirmed.json", {"fixture": name})
        directories[name + "_dir"] = directory
    pin = directories["planet_class_dir"] / "confirmed.json"

    def sources(book, dirs, class_sha):
        for directory in dirs.values():
            book.clean(directory)
            book.read(directory / "confirmed.json")
        assert hashlib.sha256(pin.read_bytes()).hexdigest() == class_sha
        return deepcopy(bundle)

    monkeypatch.setattr(module, "_sources", sources)
    monkeypatch.setattr(module, "_mapping", lambda r: deepcopy(r["mapping"]))
    monkeypatch.setattr(module, "_same_habitat", lambda a, b: a["mapping"] == b["mapping"])
    monkeypatch.setattr(module, "save_probe", capture)

    class Config:
        def __init__(self):
            self.frames = [SimpleNamespace(url=module.SIMULATION_URL, required_text=["Planet"])]

        def model_copy(self, deep=False):
            return deepcopy(self)

        def model_dump(self, **kwargs):
            return {"frames": [vars(f).copy() for f in self.frames]}

    def called(name):
        state.calls.append(name)
        if state.hook:
            state.hook(name)
        if state.failure == name:
            raise BrowserSafetyStop("fixture_uncertain_adapter")

    def inspect(*args):
        called("read")
        return deepcopy(report)

    monkeypatch.setattr(module, "inspect_page", inspect)
    monkeypatch.setattr(module, "_read_planet", lambda *args: called("planet_readback"))

    def navigate(page, config, output, destination, **kwargs):
        called("navigate")
        output.mkdir()
        result = {
            "same_star_verified": True,
            "destination_verified": True,
            "suggested_required_text": ["Habitability"],
        }
        persist_json(output / "confirmed.json", result)
        return result

    monkeypatch.setattr(module, "navigate_project", navigate)

    def compare(page, config, output, *, candidates):
        called("compare")
        capture(deepcopy(report), output / "initial")
        capture(deepcopy(report), output / "restored")
        result = {
            "star": "Terra",
            "baseline_restored": True,
            "task_completed": False,
            "candidates": candidates,
        }
        persist_json(output / "report.json", result)
        persist_json(output / "write-01-reserved.json", {"fixture_native_intent": True})
        return result

    monkeypatch.setattr(module, "compare_visible_gas_candidates", compare)
    monkeypatch.setattr(
        module,
        "_gas_writes",
        lambda book, directory, *args: (
            book.capture(directory / "initial"),
            book.capture(directory / "restored"),
        ),
    )

    def select(page, config, output, *, comparison, gases, rationale):
        called("select_gases")
        output.mkdir()
        values()["selected_gases"] = gases
        result = {"star": "Terra", "gases": gases, "rationale": rationale, "readback_verified": True}
        persist_json(output / "report.json", result)
        capture(deepcopy(report), output / "after")
        return result

    monkeypatch.setattr(module, "select_reference_gases", select)
    monkeypatch.setattr(
        module,
        "_gas_sources",
        lambda book, comparison, directory, star: (
            book.json(directory / "report.json"),
            book.capture(directory / "after"),
        ),
    )

    class Temperature:
        def __init__(self, page, config, output, **kwargs):
            called("temperature_init")
            self.finished = False
            self.output, self.callback, self.increment = (
                output,
                kwargs["emit"],
                kwargs["supplied_greenhouse_increment"],
            )
            self.index = 0
            self.session = SimpleNamespace(report=deepcopy(report))
            self.callback(
                RuntimeEvent(event="hello", sequence=0, run_id="temperature", payload={}).model_dump(
                    mode="json"
                )
            )
            self.index += 1

        def state(self):
            return {
                "finished": self.finished,
                "equilibrium_transport_verified": self.finished and not state.child_failure,
                "task_completed": False,
                "sources_unchanged": True,
                "checkpoint_unchanged": True,
                "optimizer_updates": 0,
            }

        def advance(self):
            called("temperature_step")
            self.callback(
                RuntimeEvent(
                    event="action_proposed",
                    sequence=self.index,
                    run_id="temperature",
                    payload={"native_write_boundary": True},
                ).model_dump(mode="json")
            )
            self.index += 1
            values()["equilibrium_temp"]["value"] = "250"
            self.output.mkdir()
            result = {"provenance": {"supplied_greenhouse_increment": self.increment}}
            persist_json(self.output / "report.json", result)
            capture(deepcopy(report), self.output / "after")
            self.finished = True

        def abort(self):
            called("temperature_abort")
            self.finished = True

    monkeypatch.setattr(module, "HabitabilityTemperatureSteps", Temperature)
    monkeypatch.setattr(
        module,
        "_temperature_source",
        lambda book, directory, before: (
            book.json(directory / "report.json"),
            book.capture(directory / "after"),
        ),
    )

    class Menu:
        def __init__(self, page, config, output):
            self.output = output
            self.report = deepcopy(report)

        def greenhouse_reference(self):
            called("greenhouse")
            values()["greenhouse"] = "Moderate (+30)"
            values()["readouts"]["surface_temp"]["display_text"] = "280"
            return self.finish("greenhouse")

        def water_phase_from_chamber(self, directory):
            called("phase")
            values()["water_phase"] = "Liquid"
            return self.finish("water_phase")

        def finish(self, field):
            self.output.mkdir()
            result = {"field": field, "readback_verified": True}
            persist_json(self.output / "confirmed.json", result)
            capture(deepcopy(report), self.output / "after")
            return result

        def close(self):
            called("menu_close")

    monkeypatch.setattr(module, "HabitabilityMenuSession", Menu)
    monkeypatch.setattr(
        module,
        "_greenhouse_source",
        lambda book, directory, *args: (
            book.json(directory / "confirmed.json"),
            book.capture(directory / "after"),
        ),
    )

    class Chamber:
        def __init__(self, page, config, output, emit):
            self.output, self.emit = output, emit

        def query(self, pressure, temperature, *, source):
            called("chamber")
            state.chamber_inputs = pressure, temperature, source
            self.emit("action_proposed", {"tool": "water_chamber"})
            self.output.mkdir()
            result = {
                "phase": "liquid",
                "pressure": pressure,
                "temperature": temperature,
                "conditions_verified": True,
                "task_completed": False,
            }
            persist_json(self.output / "confirmed.json", result)
            return result

    monkeypatch.setattr(module, "WaterChamberSession", Chamber)
    monkeypatch.setattr(
        module,
        "confirmed_phase_reference",
        lambda directory, mapping: {
            "phase": mapping["observation"]["values"]["water_phase"],
            "source": "fixture_reference",
        },
    )

    def choose(page, config, output, *, phase_record, name, rationale):
        called("choice")
        if name == "habitable" and values()["water_phase"] != "Liquid":
            raise BrowserSafetyStop("habitability_reference_requires_liquid_water")
        output.mkdir()
        values()["choice"] = name
        result = {"choice": name, "rationale": rationale, "readback_verified": True}
        persist_json(output / "confirmed.json", result)
        capture(deepcopy(report), output / "after")
        return result

    monkeypatch.setattr(module, "select_habitability_reference", choose)

    def save(page, config, output, **kwargs):
        called("save")
        kwargs["cancelled"]()
        output.mkdir()
        persist_json(output / "dispatch.json", {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1})
        state.save_clicks += 1
        result = {"save_click_delivered": True, "data_saved_notice_observed": True}
        persist_json(output / "confirmed.json", result)
        capture(deepcopy(report), output / "after")
        return result

    monkeypatch.setattr(module, "save_habitability_work", save)
    monkeypatch.setattr(
        module,
        "_final_save",
        lambda book, directory, *args: (
            {},
            book.json(directory / "confirmed.json"),
            book.capture(directory / "after"),
        ),
    )

    def verify(page, config, output, **kwargs):
        called("verify")
        state.verifier_sources = kwargs
        result = {
            "mode": "terrestrial_visible_workflow_readback",
            "task_completed": True,
            "authority": "visible_workflow_readback",
            "star": "TERRA",
        }
        if state.verifier == "false":
            result["task_completed"] = False
        elif state.verifier == "foreign":
            result["star"] = "Different"
        output.mkdir()
        if state.verifier != "missing":
            persist_json(output / "confirmed.json", result if state.verifier != "mismatch" else {})
        return result

    monkeypatch.setattr(module, "verify_terrestrial_workflow", verify)
    pilot, final = tmp_path / "pilot", tmp_path / "final"
    (pilot / "training").mkdir(parents=True)
    final.mkdir()
    for path in (
        pilot / "training/checkpoint.pt",
        pilot / "training/checkpoint.pt.json",
        pilot / "report.json",
        final / "report.json",
    ):
        persist_json(path, {"fixture_not_a_real_checkpoint": True})
    options = {
        "run_history": tmp_path,
        **directories,
        "planet_class_sha256": hashlib.sha256(pin.read_bytes()).hexdigest(),
        "candidates": ["CO2", "H2O"],
        "pilot": pilot,
        "final_evaluation": final,
        "graph": graph,
    }

    def make(**kwargs):
        output = kwargs.pop("output", tmp_path / "owner")
        callback = kwargs.pop("emit", state.events.append)
        state.owner = module.TerrestrialSteps(object(), Config(), output, emit=callback, **(options | kwargs))
        return state.owner

    return SimpleNamespace(
        state=state,
        make=make,
        options=options,
        report=report,
        values=values,
        pin=pin,
        root=tmp_path,
        bundle=bundle,
    )


def reach(owner, phase):
    for _ in range(15):
        if owner.phase == phase or owner.finished:
            return owner.state()
        owner.advance()
    raise AssertionError("Fixture stage limit")


def gases(owner, increment=30):
    reach(owner, "awaiting_gases")
    return owner.provide_gases(
        gases=["CO2"],
        rationale="Explicit fixture-only visual comparison rationale.",
        supplied_greenhouse_increment=increment,
    )


def habitat(owner, choice="not_habitable"):
    reach(owner, "awaiting_habitability")
    return owner.provide_habitability(
        choice=choice, rationale="Explicit fixture-only reference choice; not a scientific training label."
    )


def test_constructor_and_decisions_are_offline_and_handoffs_do_not_complete(rig):
    owner = rig.make()
    assert not rig.state.calls and not owner.claimed
    reach(owner, "awaiting_gases")
    calls, advances = list(rig.state.calls), owner.advances
    assert owner.state()["status"] == "paused"
    assert owner.advance()["task_completed"] is False
    assert owner.advances == advances and rig.state.calls == calls
    gases(owner)
    assert rig.state.calls == calls and owner.phase == "select_gases"
    reach(owner, "awaiting_habitability")
    calls = list(rig.state.calls)
    assert not owner.state()["task_completed"] and not owner.finished
    habitat(owner)
    assert rig.state.calls == calls and owner.phase == "select_habitability"
    owner.abort()


@pytest.mark.parametrize("choice", ["habitable", "not_habitable"])
def test_complete_chain_preserves_explicit_choice_exact_chamber_inputs_and_events(rig, choice):
    owner = rig.make()
    gases(owner)
    habitat(owner, choice)
    reach(owner, "finished")
    assert owner.report["task_completed"] is True
    assert rig.values()["choice"] == choice
    assert rig.state.chamber_inputs == ("0.8", "280", "reference_diagnostic")
    assert rig.state.save_clicks == 1
    assert owner.report["journal_writes"] == 0
    assert owner.report["scientific_verified"] is False
    assert rig.state.verifier_sources["save_dir"] == owner.output / "save"
    assert len(rig.state.verifier_sources) == 14  # thirteen dirs plus run_history
    events = [
        RuntimeEvent.model_validate_json(line)
        for line in (owner.output / "events.jsonl").read_text().splitlines()
    ]
    assert [e.sequence for e in events] == list(range(len(events)))
    assert [e.model_dump(mode="json") for e in events] == rig.state.events
    assert events[-1].event == "episode_summary" and events[-1].payload["task_completed"] is True
    assert (
        hashlib.sha256((owner.output / "workflow/confirmed.json").read_bytes()).hexdigest()
        == owner.report["workflow_sha256"]
    )
    before = list(rig.state.calls)
    owner.advance()
    owner.close()
    assert rig.state.calls == before


@pytest.mark.parametrize("field", ["source", "model", "graph", "config", "marker"])
def test_changed_pins_or_uncertainty_stop_before_next_stage(rig, field):
    owner = rig.make()
    reach(owner, "awaiting_gases")
    if field == "source":
        rig.pin.write_text("changed")
    elif field == "model":
        (rig.options["pilot"] / "training/checkpoint.pt").write_text("changed")
    elif field == "graph":
        rig.options["graph"].checksum = "different"
    elif field == "config":
        owner.config.frames[0].required_text = ["escape"]
    else:
        persist_json(owner.output / "gas-comparison/write-01-stopped.json", {"uncertain": True})
    calls = list(rig.state.calls)
    gases(owner)
    assert owner.finished and not owner.report["task_completed"]
    assert rig.state.calls == calls


def test_changed_current_values_stop_without_gas_comparison(rig):
    owner = rig.make()
    owner.advance()
    rig.values()["measurements"]["orbital_radius"]["display_text"] = "2"
    owner.advance()
    assert owner.finished and "compare" not in rig.state.calls


def test_cross_star_live_capture_never_starts_gas_work(rig):
    owner = rig.make()
    rig.report["mapping"]["star_name"] = "Wrong"
    owner.advance()
    assert owner.finished and "compare" not in rig.state.calls


def test_increment_disagreement_stops_before_greenhouse_selection(rig):
    owner = rig.make()
    gases(owner, 10)
    reach(owner, "greenhouse")
    owner.advance()
    assert owner.finished and "greenhouse" not in rig.state.calls and "chamber" not in rig.state.calls
    assert owner.failure == "terrestrial_steps_supplied_greenhouse_disagrees"


def test_no_automatic_habitability_choice_even_when_phase_is_liquid(rig):
    owner = rig.make()
    gases(owner)
    reach(owner, "awaiting_habitability")
    for _ in range(3):
        owner.advance()
    assert not owner.finished and "choice" not in rig.state.calls and "save" not in rig.state.calls
    owner.abort()


@pytest.mark.parametrize(
    "stage", ["compare", "select_gases", "greenhouse", "chamber", "phase", "choice", "save", "verify"]
)
def test_uncertain_stage_is_never_retried_or_bypassed_with_new_output(rig, stage):
    owner = rig.make()
    rig.state.failure = stage
    reach(owner, "awaiting_gases")
    if not owner.finished:
        gases(owner)
        reach(owner, "awaiting_habitability")
    if not owner.finished:
        habitat(owner)
        reach(owner, "finished")
    assert owner.finished and not owner.report["task_completed"]
    calls = list(rig.state.calls)
    owner.advance()
    assert calls == rig.state.calls and calls.count(stage) == 1
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        rig.make(output=rig.root / "different-output")


@pytest.mark.parametrize("mode", ["missing", "mismatch", "false", "foreign"])
def test_exact_persisted_workflow_receipt_required(rig, mode):
    owner = rig.make()
    gases(owner)
    habitat(owner)
    rig.state.verifier = mode
    reach(owner, "finished")
    assert owner.finished and not owner.report["task_completed"]
    assert owner.report["workflow_sha256"] is None and rig.state.save_clicks == 1


def test_pause_deadline_is_fixed_no_automatic_budget_increase(rig):
    owner = rig.make(max_seconds=10)
    reach(owner, "awaiting_gases")
    calls = list(rig.state.calls)
    rig.state.now = 11
    owner.advance()
    assert owner.finished and owner.failure == "terrestrial_steps_deadline_exhausted"
    assert rig.state.calls == calls


@pytest.mark.parametrize("event_kind", ["hello", "temperature", "final"])
def test_callback_failure_stops_without_false_success(rig, event_kind):
    def callback(event):
        rig.state.events.append(event)
        if (
            event_kind == "hello"
            and event["event"] == "hello"
            or event_kind == "temperature"
            and event["payload"].get("native_write_boundary")
            or event_kind == "final"
            and event["event"] == "episode_summary"
        ):
            raise RuntimeError("private callback diagnostic")

    owner = rig.make(emit=callback)
    if not owner.finished:
        gases(owner)
        reach(owner, "awaiting_habitability")
    if not owner.finished:
        habitat(owner)
        reach(owner, "finished")
    assert owner.finished and not owner.report["task_completed"]
    assert "private callback diagnostic" not in json.dumps(owner.report)
    if event_kind != "final":
        assert rig.state.save_clicks == 0


def test_abort_at_native_temperature_boundary_prevents_copy(rig):
    def callback(event):
        if event["payload"].get("native_write_boundary"):
            rig.state.owner.abort()

    owner = rig.make(emit=callback)
    gases(owner)
    reach(owner, "finished")
    assert owner.finished and rig.values()["equilibrium_temp"]["value"] == ""
    assert "greenhouse" not in rig.state.calls


def test_abort_before_work_touches_no_page(rig):
    owner = rig.make()
    owner.abort()
    assert owner.finished and not rig.state.calls and not owner.claim.exists()


def test_child_failure_does_not_start_greenhouse(rig):
    owner = rig.make()
    gases(owner)
    rig.state.child_failure = True
    reach(owner, "finished")
    assert owner.finished and "greenhouse" not in rig.state.calls


def test_decision_validation_has_no_browser_actions(rig):
    owner = rig.make()
    with pytest.raises(BrowserSafetyStop, match="handoff_not_ready"):
        owner.provide_habitability(choice="habitable", rationale="a" * 45)
    reach(owner, "awaiting_gases")
    calls = list(rig.state.calls)
    with pytest.raises(BrowserSafetyStop, match="invalid_supplied_gases"):
        owner.provide_gases(gases=["NH3"], rationale="a" * 30, supplied_greenhouse_increment=30)
    assert rig.state.calls == calls and owner.phase == "awaiting_gases"
    owner.abort()


@pytest.mark.parametrize("mutation", ["wrong_hash", "non_terrestrial", "cross_star", "mass", "radius"])
def test_offline_constructor_source_binder_reuses_existing_validators(rig, monkeypatch, mutation):
    dirs = {k: v for k, v in rig.options.items() if k.endswith("_dir")}
    stellar = {
        "star": "Terra",
        "readbacks": {"mass": {"display_value": "1"}, "radius": {"display_value": "2"}},
    }
    planet = {"star": "TERRA", "planet_class": "terrestrial", "class_capture": {}}
    inputs = {"stellar_mass": {"display_text": "1"}, "stellar_radius": {"display_text": "2"}}
    checksum = rig.options["planet_class_sha256"]
    if mutation == "wrong_hash":
        checksum = "0" * 64
    elif mutation == "non_terrestrial":
        planet["planet_class"] = "ice_giant"
    elif mutation == "cross_star":
        planet["star"] = "Wrong"
    else:
        inputs["stellar_" + mutation]["display_text"] = "99"
    monkeypatch.setattr(module, "_stellar_sources", lambda *args: stellar)
    monkeypatch.setattr(module, "_planet_sources", lambda *args: planet)
    monkeypatch.setattr(module, "_planet", lambda _: {"observation": {"values": {"stellar_inputs": inputs}}})
    with pytest.raises((BrowserSafetyStop, ValueError)):
        REAL_SOURCES(module._Evidence(rig.root), dirs, checksum)
    assert not rig.state.calls


def test_reentrant_advance_fails_closed_without_dispatch(rig):
    def callback(event):
        if event["payload"].get("scheduled_call") == "compare_gases":
            rig.state.owner.advance()

    owner = rig.make(emit=callback)
    reach(owner, "finished")
    assert owner.finished and "compare" not in rig.state.calls and not owner.report["task_completed"]


def test_final_callback_source_mutation_revokes_completion(rig):
    def callback(event):
        if event["event"] == "episode_summary" and event["payload"]["task_completed"]:
            rig.pin.write_text("changed at final notification")

    owner = rig.make(emit=callback)
    gases(owner)
    habitat(owner)
    reach(owner, "finished")
    assert owner.finished and not owner.report["task_completed"]
    assert rig.state.save_clicks == 1 and owner.report["workflow_sha256"] is None


def test_cancellation_in_save_guard_keeps_one_owner_claim_and_no_dispatch(rig):
    owner = rig.make()
    gases(owner)
    habitat(owner)
    reach(owner, "save")
    rig.state.hook = lambda name: owner.abort() if name == "save" else None
    owner.advance()
    assert owner.finished and not rig.state.save_clicks and owner.claim.exists()
    assert not (owner.output / "save/dispatch.json").exists()


def test_current_pressure_change_does_not_query_chamber(rig):
    owner = rig.make()
    gases(owner)
    reach(owner, "chamber")
    rig.values()["measurements"]["pressure"]["display_text"] = "9"
    owner.advance()
    assert owner.finished and "chamber" not in rig.state.calls


def test_new_chamber_uncertainty_blocks_phase_write(rig):
    owner = rig.make()
    gases(owner)
    reach(owner, "water_phase")
    persist_json(owner.output / "chamber/cleanup-stopped.json", {"uncertain": True})
    owner.advance()
    assert owner.finished and "phase" not in rig.state.calls
