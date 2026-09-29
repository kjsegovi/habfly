"""Offline new-mode boundaries; source/gate seams are explicit, not acceptance.

No browser/model runs occur. Generic Stellar receipts use their real validator.
Importer fixtures inject only the independently tested upstream source reader.
"""
# ruff: noqa: F811

import hashlib
import sys
from types import ModuleType

import pytest
from test_browser_star_preflight import read, save_capture, write
from test_browser_stellar_sources import class_evidence, sources  # noqa: F401
from test_browser_terrestrial_steps import gases, reach
from test_browser_terrestrial_steps import rig as terrestrial_rig  # noqa: F401
from test_planet_supplied_stellar_source import build, case  # noqa: F401
from test_project_evidence import current_stellar, inventory, sha
from test_project_positive_evidence import offline as positive_offline  # noqa: F401
from test_project_terrestrial_evidence import offline as terrestrial_offline  # noqa: F401

import habfly.browser_positive_finalize as finalizer
import habfly.browser_positive_planet_workflow as positive
import habfly.browser_terrestrial_steps as terrestrial_steps
import habfly.browser_terrestrial_workflow as terrestrial
import habfly.planet_supplied_stellar_source as supplied_source
import habfly.project_positive_evidence as positive_import
import habfly.project_terrestrial_evidence as terrestrial_import
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_numeric import screen_identity
from habfly.contracts import RuntimeEvent
from habfly.project_progress import ProgressError
from habfly.supplied_browser_modes import (
    DERIVED_SUPPLIED_MODE,
    NON_MAIN_CLASSES,
    POSITIVE_SUPPLIED_MODE,
    TEMPERATURE_SUPPLIED_MODE,
    TERRESTRIAL_SUPPLIED_MODE,
    positive_mode,
    source_options,
    terrestrial_mode,
)


@pytest.mark.parametrize("bad", [None, 0, 1, "true", {}, []])
def test_opt_in_is_strict_and_not_truthy(bad):
    with pytest.raises(BrowserSafetyStop, match="invalid_opt_in"):
        source_options(bad)


def test_modes_are_distinct_and_legacy_options_stay_empty():
    assert source_options(False) == {}
    assert source_options(True) == {"supplied_inputs": True}
    assert positive_mode(False) == positive_import.MODE
    assert terrestrial_mode(False) == terrestrial_import.MODE
    assert positive_mode(True) == POSITIVE_SUPPLIED_MODE
    assert terrestrial_mode(True) == TERRESTRIAL_SUPPLIED_MODE
    assert (
        len(
            {
                DERIVED_SUPPLIED_MODE,
                TEMPERATURE_SUPPLIED_MODE,
                POSITIVE_SUPPLIED_MODE,
                TERRESTRIAL_SUPPLIED_MODE,
            }
        )
        == 4
    )


@pytest.mark.parametrize("actual_class", NON_MAIN_CLASSES)
def test_new_stellar_reader_keeps_exact_three_and_old_path_rejects(sources, actual_class):
    item = sources(actual_class)
    args = (item.numeric_dir, item.color_dir, item.class_dir)
    bundle = positive._stellar_sources(_Evidence(item.history), *args, supplied_inputs=True)
    assert bundle["class"] == actual_class and bundle["prefix"] is None
    assert set(bundle["readbacks"]) == {"distance", "luminosity", "temperature"}
    with pytest.raises(BrowserSafetyStop, match="unsupported_stellar_numeric"):
        positive._stellar_sources(_Evidence(item.history), *args)


def test_main_stays_legacy_not_migrated(sources):
    item = sources()
    args = (item.numeric_dir, item.color_dir, item.class_dir)
    assert positive._stellar_sources(_Evidence(item.history), *args)["class"] == "main_sequence"
    with pytest.raises(BrowserSafetyStop, match="supplied_non_main"):
        positive._stellar_sources(_Evidence(item.history), *args, supplied_inputs=True)


@pytest.mark.parametrize("mutation", [None, "class", "star", "mass", "radius", "stellar_copy", "checkpoint"])
def test_independent_stellar_and_visible_supplied_inputs_must_match(case, mutation):
    item = case("white_dwarf")
    receipt = build(item)
    stellar = positive._stellar_sources(
        _Evidence(item.history), item.numeric_dir, item.color_dir, item.class_dir, supplied_inputs=True
    )
    planet = {"star": receipt["star"], "supplied_stellar_inputs": receipt}
    capture = read(item.current / "observation.json")
    if mutation == "class":
        stellar["class"] = "red_giant"
    elif mutation == "star":
        planet["star"] = "Other"
    elif mutation in {"mass", "radius"}:
        receipt["inputs"]["stellar_" + mutation]["display_text"] = "999"
    elif mutation == "stellar_copy":
        stellar["readbacks"]["luminosity"]["display_value"] = "999"
    elif mutation == "checkpoint":
        stellar["numeric_provenance"]["checkpoint_sha256"] = "f" * 64
    if mutation:
        with pytest.raises(BrowserSafetyStop):
            positive._match_stellar_planet_sources(stellar, planet, capture, supplied_inputs=True)
    else:
        positive._match_stellar_planet_sources(stellar, planet, capture, supplied_inputs=True)
        assert not {"mass", "radius", "lifetime"} & set(stellar["readbacks"])


def _derived_report(directory, actual_class, scope):
    report = {
        "scope": scope,
        "outcome": "planet_derived_transport_verified",
        "write_attempts": sorted(positive.DERIVED),
        "verified_fields": {name: {} for name in positive.DERIVED},
        "checkpoint_unchanged": True,
        "planet_transport_verified": True,
        "optimizer_updates": 0,
        **dict.fromkeys(
            ("task_completed", "browser_acceptance_passed", "saved", "assessment_performed", "submitted"),
            False,
        ),
        "provenance": {
            "checkpoint_sha256": "a" * 64,
            "graph_hash": "b" * 64,
            "knowledge_pack_hash": "c" * 64,
            "optimizer_updates": 0,
            "supplied_star_class": actual_class,
            "classification_source": "supplied_not_learned",
        },
    }
    raw = "".join(
        RuntimeEvent(
            event=kind, sequence=i, run_id="declared-synthetic-gate-boundary", payload=payload
        ).model_dump_json()
        + "\n"
        for i, (kind, payload) in enumerate((("hello", {}), ("episode_summary", report)))
    )
    directory.mkdir()
    (directory / "events.jsonl").write_text(raw)
    write(directory / "report.json", {**report, "events_sha256": hashlib.sha256(raw.encode()).hexdigest()})


@pytest.mark.parametrize(
    "opt_in,scope,actual_class,reaches_gate",
    [
        (True, DERIVED_SUPPLIED_MODE, "white_dwarf", True),
        (True, "four_derived_planet_browser_transport", "white_dwarf", False),
        (True, DERIVED_SUPPLIED_MODE, "main_sequence", False),
        (False, DERIVED_SUPPLIED_MODE, "white_dwarf", False),
    ],
)
def test_new_derived_requires_distinct_scope_and_mandatory_gate(
    case, monkeypatch, opt_in, scope, actual_class, reaches_gate
):
    item = case()
    for name in ("_presence", "_reload", "_fresh_evidence"):
        monkeypatch.setattr(positive, name, getattr(supplied_source, name))
    derived, classification = item.history / "derived", item.history / "planet-class"
    classification.mkdir()
    _derived_report(derived, actual_class, scope)

    # The independent gate is intentionally not forged by these upstream fixtures.
    class RequiredGate(Exception):
        pass

    calls = []
    helper = ModuleType("habfly.browser_supplied_provenance")

    def validate(*args):
        calls.append(args)
        raise RequiredGate

    helper.validate_supplied_derived_provenance = validate
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    with pytest.raises(RequiredGate if reaches_gate else BrowserSafetyStop):
        positive._planet_sources(
            _Evidence(item.history), item.raw, derived, classification, supplied_inputs=opt_in
        )
    assert bool(calls) is reaches_gate


def _convert(offline, *, terrestrial_branch, actual_class="white_dwarf"):
    journal, history, _, directory, registry, calls = offline
    key = str(history / "source-first/numeric")
    hashes, bundle = registry[key]
    capture = current_stellar(bundle["star"], actual_class)
    mapped = positive._stellar(capture)
    bundle["class"], bundle["prefix"] = actual_class, None
    bundle["readbacks"] = {
        k: v for k, v in bundle["readbacks"].items() if k in {"distance", "luminosity", "temperature"}
    }
    bundle["numeric_provenance"].update(selected_class=actual_class, lifetime_prefix=None)
    link = {"path": "supplied/confirmed.json", "sha256": "2" * 64}
    bundle["derived"]["provenance"].update(supplied_star_class=actual_class, supplied_stellar_inputs=link)
    base = history / "source-first"
    derived = read(base / "derived/report.json")
    derived.update(scope=DERIVED_SUPPLIED_MODE, provenance=bundle["derived"]["provenance"])
    write(base / "derived/report.json", derived)
    if terrestrial_branch:
        temperature = read(base / "temperature/report.json")
        temperature["scope"] = TEMPERATURE_SUPPLIED_MODE
        bundle["temperature"]["provenance"].update(
            supplied_star_class=actual_class, supplied_stellar_inputs=link
        )
        temperature["provenance"] = bundle["temperature"]["provenance"]
        write(base / "temperature/report.json", temperature)
    hashes = {name: sha(history / name) for name in hashes}
    registry[key] = (hashes, bundle)
    # Existing importer fixture deliberately supplies upstream validation, not
    # a fake gate. Actual current captures, fields/units, inventory and journal
    # still run through their production checks.
    save_capture(directory / "stellar-readback/verified", capture)
    receipt = read(directory / "confirmed.json")
    receipt.update(
        mode=TERRESTRIAL_SUPPLIED_MODE if terrestrial_branch else POSITIVE_SUPPLIED_MODE,
        classification=actual_class,
        stellar_fields=bundle["readbacks"],
        numeric_provenance=bundle["numeric_provenance"],
        derived_provenance=bundle["derived"]["provenance"],
        source_sha256=hashes,
    )
    if terrestrial_branch:
        receipt["temperature_provenance"] = bundle["temperature"]["provenance"]
    receipt["current_screen_sha256"]["stellar"] = screen_identity(capture)
    write(directory / "confirmed.json", receipt)
    inv = inventory(history, [bundle["star"], "Beta"], "supplied", classes={bundle["star"]: actual_class})
    assert set(mapped["observation"]["values"]["browser_field_map"]) == set(bundle["readbacks"])
    return journal, history, inv, directory, registry, calls


@pytest.mark.parametrize("actual_class", NON_MAIN_CLASSES)
def test_positive_import_explicit_new_mode_and_idempotency(positive_offline, actual_class):
    data = _convert(positive_offline, terrestrial_branch=False, actual_class=actual_class)
    first = positive_import.import_verified_positive_planet(*data[:4])
    second = positive_import.import_verified_positive_planet(*data[:4])
    assert first["progress"]["verified"] == 1 and second["idempotent"]
    assert data[-1][-1]["supplied_inputs"] is True
    scope = first["evidence_scopes"]
    assert scope["derived_calculations"]["scope"] == DERIVED_SUPPLIED_MODE
    assert scope["supplied_stellar_inputs"]["actual_class"] == actual_class
    assert scope["supplied_stellar_inputs"]["learned_stellar_mass_radius"] is False
    assert first["progress"]["project_completed"] is False


@pytest.mark.parametrize("actual_class", NON_MAIN_CLASSES)
def test_terrestrial_import_explicit_new_modes_and_idempotency(terrestrial_offline, actual_class):
    data = _convert(terrestrial_offline, terrestrial_branch=True, actual_class=actual_class)
    first = terrestrial_import.import_verified_terrestrial(*data[:4])
    second = terrestrial_import.import_verified_terrestrial(*data[:4])
    assert first["progress"]["verified"] == 1 and second["idempotent"]
    assert data[-1][-1]["supplied_inputs"] is True
    assert first["evidence_scopes"]["temperature_calculations"]["scope"] == TEMPERATURE_SUPPLIED_MODE
    assert not first["progress"]["project_completed"]


@pytest.mark.parametrize("component", ["derived", "temperature", "workflow"])
def test_mixed_terrestrial_sources_rejected_before_journal_write(terrestrial_offline, component):
    data = _convert(terrestrial_offline, terrestrial_branch=True)
    journal, history, _, directory, _registry, _ = data
    before = journal.path.read_bytes()
    receipt = read(directory / "confirmed.json")
    if component == "workflow":
        receipt["mode"] = terrestrial_import.MODE
    else:
        path = history / "source-first" / component / "report.json"
        value = read(path)
        value["scope"] = (
            "four_derived_planet_browser_transport"
            if component == "derived"
            else "one_equilibrium_copy_with_supplied_warming_local_proposal"
        )
        write(path, value)
        receipt["source_sha256"][str(path.relative_to(history))] = sha(path)
    write(directory / "confirmed.json", receipt)
    with pytest.raises((ProgressError, BrowserSafetyStop)):
        terrestrial_import.import_verified_terrestrial(*data[:4])
    assert journal.path.read_bytes() == before and not list(history.glob("project-evidence-*.json"))


@pytest.mark.parametrize("factory", [finalizer.PositiveFinalizationSteps, terrestrial_steps.TerrestrialSteps])
def test_constructor_rejects_non_boolean_opt_in_before_any_source_or_page(factory):
    import inspect

    options = {
        name: None
        for name, parameter in inspect.signature(factory).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    options["supplied_inputs"] = 1
    with pytest.raises(BrowserSafetyStop, match="invalid_opt_in"):
        factory(**options)


@pytest.mark.parametrize("opt_in,new_scope", [(True, True), (True, False), (False, True)])
def test_temperature_new_mode_cannot_skip_independent_gate(tmp_path, monkeypatch, opt_in, new_scope):
    provenance = {
        "checkpoint_sha256": "a" * 64,
        "graph_hash": "b" * 64,
        "knowledge_pack_hash": "c" * 64,
        "optimizer_updates": 0,
        "surface_proposal_is_not_browser_readback": True,
        "supplied_greenhouse_increment": 30,
        "calibration_scope": "browser_transfer_not_calibrated",
        **dict.fromkeys(
            (
                "greenhouse_selection_learned",
                "gas_identification_learned",
                "water_phase_learned",
                "habitability_decision_learned",
            ),
            False,
        ),
    }
    report = {
        "scope": TEMPERATURE_SUPPLIED_MODE
        if new_scope
        else "one_equilibrium_copy_with_supplied_warming_local_proposal",
        "outcome": "equilibrium_transport_verified",
        "steps": 1,
        "checkpoint_unchanged": True,
        "equilibrium_transport_verified": True,
        "optimizer_updates": 0,
        "provenance": provenance,
        **dict.fromkeys(
            ("task_completed", "course_acceptance_passed", "saved", "assessed", "submitted"), False
        ),
    }
    pairs = [
        ("hello", {"task": "habitability_calculations", "policy": "checkpoint", **provenance}),
        ("action_proposed", {"action_source": "checkpoint"}),
        ("neural_activity", {"activity_source": "checkpoint"}),
        ("state", {}),
        ("episode_summary", report),
    ]
    raw = "".join(
        RuntimeEvent(event=k, sequence=i, run_id="synthetic-temp-gate-boundary", payload=v).model_dump_json()
        + "\n"
        for i, (k, v) in enumerate(pairs)
    )
    directory = tmp_path / "temperature"
    directory.mkdir()
    (directory / "events.jsonl").write_text(raw)
    write(directory / "report.json", {**report, "events_sha256": hashlib.sha256(raw.encode()).hexdigest()})

    class RequiredGate(Exception):
        pass

    helper = ModuleType("habfly.browser_supplied_provenance")
    calls = []

    def validate(*args):
        calls.append(args)
        raise RequiredGate

    helper.validate_supplied_temperature_provenance = validate
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    with pytest.raises(RequiredGate if opt_in and new_scope else BrowserSafetyStop):
        terrestrial._temperature_source(_Evidence(tmp_path), directory, None, supplied_inputs=opt_in)
    assert bool(calls) is (opt_in and new_scope)


def test_terrestrial_scheduler_forwards_actual_class_and_owned_source_without_extra_writes(
    terrestrial_rig, monkeypatch
):
    rig = terrestrial_rig
    legacy_sources = terrestrial_steps._sources
    seen = {}

    def sources(book, dirs, class_sha, *, supplied_inputs=False):
        assert supplied_inputs is True
        value = legacy_sources(book, dirs, class_sha)
        value.update(
            {
                "class": "white_dwarf",
                "derived": {
                    "provenance": {
                        "supplied_stellar_inputs": {"path": "owned/confirmed.json", "sha256": "d" * 64}
                    }
                },
            }
        )
        return value

    monkeypatch.setattr(terrestrial_steps, "_sources", sources)
    parent = terrestrial_steps.HabitabilityTemperatureSteps

    class Supplied(parent):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    helper = ModuleType("habfly.browser_supplied_steps")
    helper.SuppliedHabitabilityTemperatureSteps = Supplied
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    owner = rig.make(supplied_inputs=True)
    assert owner.state()["supplied_inputs"] is True and not rig.state.calls
    gases(owner)
    reach(owner, "temperature_initializing")
    owner.advance()
    assert not owner.finished and owner.phase == "temperature"
    assert seen["supplied_star_class"] == "white_dwarf"
    assert seen["expected_star"] == "Terra"
    assert seen["supplied_stellar_inputs_path"] == rig.root / "owned/confirmed.json"
    assert seen["supplied_stellar_inputs_sha256"] == "d" * 64
    assert seen["final_evaluation"] == rig.options["final_evaluation"]
    assert rig.state.calls.count("temperature_init") == 1
    assert not {"temperature_step", "save"} & set(rig.state.calls)
    owner.abort()
    before = list(rig.state.calls)
    owner.advance()
    assert rig.state.calls == before and not owner.state()["task_completed"]
