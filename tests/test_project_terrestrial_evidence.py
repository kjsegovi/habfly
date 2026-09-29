"""Offline journal behavior and one intercepted native terrestrial chain.

The explicit upstream-validator seam is synthetic, never a science/training
oracle. The final integration removes that seam and uses only intercepted UI.
"""
# ruff: noqa: F811

import hashlib
import socket
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
import yaml
from test_browser_habitability import capture as habitat_capture
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_raster_planet_evidence import raster_page  # noqa: F401
from test_browser_terrestrial_workflow import run, seam, terrestrial  # noqa: F401
from test_project_evidence import inventory, read, sha, write
from test_project_positive_evidence import workflow as positive_workflow

import habfly.project_positive_evidence as positive_module
import habfly.project_terrestrial_evidence as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_habitability_save import MODE as SAVE_MODE
from habfly.browser_numeric import screen_identity
from habfly.browser_probe import save_probe
from habfly.project_progress import Collected, ProgressError, ProjectJournal, WriteReserved


def current_habitat(star, outcome):
    phase = "Liquid" if outcome == "habitable" else "Gas"
    report = habitat_capture("CO2", False, "759.4", "Weak (+10)", phase)
    frame = report["frames"][0]
    atoms = yaml.safe_load(frame["accessibility"])
    atoms[0]["text"] = star
    for atom in atoms:
        if isinstance(atom, dict) and "text" in atom:
            atom["text"] = atom["text"].replace("20.44", "1").replace("0.5919", "1")
    frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)
    frame["text"] = (
        star.upper() + "\nObservations Modeled Albedo 0.05 Modeled Pressure (atm) 9 "
        "Your Reconstruction Equilibrium Temp (K) Trace Gases Present CO2 Absorption % 6.882% "
        "Greenhouse Effect Surface Temp (K) 769.4 Water Phase Not Habitable Habitable "
        "STAR LUMINOSITY (Ls) 1 ORBIT RADIUS (AU) 1 Save"
    )
    return report


def workflow(history, star, suffix, registry, *, outcome="not_habitable"):
    directory = positive_workflow(history, star, suffix, registry, planet_class="terrestrial")
    base = history / ("source-" + suffix)
    _, bundle = registry[str(base / "numeric")]
    habitat = current_habitat(star, outcome)
    temperature = {
        "provenance": {
            "checkpoint_sha256": "d" * 64,
            "graph_hash": "b" * 64,
            "knowledge_pack_hash": "e" * 64,
            "optimizer_updates": 0,
            "surface_proposal_is_not_browser_readback": True,
        },
        "native_receipt": {
            "value": "759.4",
            "unit": "K",
            "action_source": "checkpoint",
            "readback_verified": True,
            "correctness_verified": False,
        },
    }
    greenhouse = {
        "evidence": {
            "name": "weak",
            "increment_kelvin": 10,
            "source": "course_greenhouse_reference",
            "gas_identification_verified": False,
        }
    }
    choice = {
        "choice": outcome,
        "evidence": {
            "phase": "liquid" if outcome == "habitable" else "gas",
            "source": "confirmed_visible_chamber_indicator",
        },
    }
    sources = {
        "gas-comparison/report.json": {"baseline_restored": True, "candidates": ["CO2"]},
        "gas-selection/report.json": {
            "action_source": "explicit_visual_reference_not_learned",
            "gases": ["CO2"],
        },
        "temperature/report.json": {
            "scope": "one_equilibrium_copy_with_supplied_warming_local_proposal",
            **temperature,
        },
        "greenhouse/confirmed.json": {"kind": "SELECT", "field": "greenhouse", **greenhouse},
        "phase/confirmed.json": {"kind": "SELECT", "field": "water_phase", "evidence": choice["evidence"]},
        "choice/confirmed.json": {"kind": "SELECT", **choice},
        "save/confirmed.json": {"kind": "CLICK", "visible_label": "Save", "mode": SAVE_MODE},
    }
    for name, value in sources.items():
        write(base / name, {**value, "fixture_star": star})
    hashes = {str(p.relative_to(history)): sha(p) for p in base.rglob("*.json")}
    bundle.update(
        class_capture=bundle["save_capture"],
        habitability_capture=habitat,
        gas={"gases": ["CO2"]},
        temperature=temperature,
        greenhouse=greenhouse,
        choice=choice,
    )
    registry[str(base / "numeric")] = (hashes, bundle)
    receipt = read(directory / "confirmed.json")
    receipt.update(
        {
            "mode": module.MODE,
            "navigation_clicks": 3,
            "source_sha256": hashes,
            "save_acknowledgement_source": "explicit_final_habitability_save_fresh_visible_footer",
            "gas_selection": {
                "gases": ["CO2"],
                "provenance": "reference_prediction",
                "transport_verified": True,
                "learned_gas_identification": False,
                "scientific_verified": False,
            },
            "temperature_provenance": temperature["provenance"],
            "equilibrium_transport": temperature["native_receipt"],
            "greenhouse_reference": greenhouse["evidence"],
            "phase_reference": choice["evidence"],
            "habitability": {
                "outcome": outcome,
                "provenance": "reference_prediction",
                "transport_verified": True,
                "scientific_verified": False,
                "learned_habitability_decision": False,
                "source": "explicit_native_choice_with_confirmed_chamber_phase",
            },
        }
    )
    save_probe(habitat, directory / "habitability-readback/verified")
    receipt["current_screen_sha256"]["habitability"] = screen_identity(habitat)
    write(directory / "confirmed.json", receipt)
    return directory


@pytest.fixture
def offline(tmp_path, monkeypatch, request):
    registry, calls = {}, []

    def validated(book, **directories):
        calls.append(directories)
        hashes, bundle = registry[str(directories["numeric_dir"])]
        for name in hashes:
            book.clean((book.history / name).parent)
            book.read(book.history / name)
        return deepcopy(bundle)

    monkeypatch.setattr(module, "_load_sources", validated)
    monkeypatch.setattr(positive_module, "_load_sources", validated)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network access forbidden"))
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="synthetic-terrestrial").create()
    inv = inventory(tmp_path, ["Dulat", "Beta"], "first")
    work = workflow(tmp_path, "Dulat", "first", registry, outcome=getattr(request, "param", "not_habitable"))
    return journal, tmp_path, inv, work, registry, calls


def ingest(offline):
    return module.import_verified_terrestrial(*offline[:4])


@pytest.mark.parametrize("offline", ["habitable", "not_habitable"], indirect=True)
def test_import_explicit_outcome_preserves_scopes_and_exact_idempotency(offline):
    journal, history, _, work, _, calls = offline
    result = ingest(offline)
    assert result["appended_records"] == 8 and result["progress"]["verified"] == 1
    assert result["star_id"] == "star:" + hashlib.sha256(b"dulat").hexdigest()
    state = journal.load().reduce()
    star = state.stars[result["star_id"]]
    assert star.planet.decision.value == "terrestrial" and star.task_completed
    assert star.habitability.outcome == read(work / "confirmed.json")["habitability"]["outcome"]
    assert star.habitability.transport_verified and star.habitability.applicability_reason is None
    assert not star.habitability.branch_applicability_verified
    assert star.habitability.decision.provenance == "reference_prediction"
    assert not state.reservations and not state.receipts and not result["progress"]["project_completed"]
    scopes = result["evidence_scopes"]
    assert scopes["raw_measurements"]["provenance"] == "approximate_reference_raster"
    assert (
        scopes["derived_calculations"]["provenance"]
        == scopes["temperature_calculations"]["provenance"]
        == "learned_prediction"
    )
    assert all(
        scopes[name]["provenance"] == "reference_prediction"
        for name in ("classification", "gas_selection", "greenhouse_selection", "water_phase", "habitability")
    )
    assert all(s["scientific_verified"] is False and s["training_label"] is False for s in scopes.values())
    assert (
        scopes["raw_measurements"]["uncertainty"]
        == read(work / "confirmed.json")["raw_measurement_evidence"]["uncertainty"]
    )
    before = journal.path.read_bytes()
    again = ingest(offline)
    assert again["idempotent"] and again["appended_records"] == 0 and journal.path.read_bytes() == before
    binding = next(history.glob("project-evidence-*.json"))
    assert binding.stat().st_mode & 0o777 == 0o600 and read(binding)["evidence_scopes"] == scopes
    assert len(calls) == 2


@pytest.mark.parametrize(
    "kind",
    [
        "source",
        "missing_source",
        "ambiguous_source",
        "capture",
        "inventory",
        "task_integer",
        "science",
        "exact_planet",
        "learned_gas",
        "learned_habitability",
        "not_applicable",
        "unresolved",
        "decision_change",
        "gas_change",
        "temperature_change",
        "phase_change",
        "save_integer",
        "two_tabs",
        "wrong_star",
        "upstream_stop",
        "invalidated",
        "uncertain_gas",
    ],
)
def test_invalid_or_revised_evidence_never_partially_appends(offline, monkeypatch, kind):
    journal, history, inv, work, registry, _ = offline
    before = journal.path.read_bytes()
    data = read(work / "confirmed.json")
    if kind == "source":
        write(history / "source-first/temperature/report.json", {})
    elif kind == "missing_source":
        data["source_sha256"].pop("source-first/choice/confirmed.json")
    elif kind == "ambiguous_source":
        path = history / "duplicate/confirmed.json"
        write(path, {"mode": SAVE_MODE, "kind": "CLICK"})
        data["source_sha256"][str(path.relative_to(history))] = sha(path)
    elif kind == "capture":
        write(work / "habitability-readback/verified/observation.json", {})
    elif kind == "inventory":
        inv = inventory(history, ["Other", "Beta"], "missing")
    elif kind == "wrong_star":
        registry[str(history / "source-first/numeric")][1]["star"] = "Other"
    elif kind == "upstream_stop":
        monkeypatch.setattr(
            module,
            "_load_sources",
            lambda *a, **k: (_ for _ in ()).throw(BrowserSafetyStop("source_not_verified")),
        )
    elif kind == "invalidated":
        write(work / "invalidated.json", {})
    elif kind == "uncertain_gas":
        write(history / "source-first/gas-selection/selection-stopped.json", {"automatic_retry": False})
    elif kind == "task_integer":
        data["task_completed"] = 1
    elif kind == "science":
        data["scientific_verified"] = True
    elif kind == "exact_planet":
        data["planet"]["approximate"] = False
    elif kind == "learned_gas":
        data["gas_selection"]["learned_gas_identification"] = True
    elif kind == "learned_habitability":
        data["habitability"]["provenance"] = "learned_prediction"
    elif kind in {"not_applicable", "unresolved", "decision_change"}:
        data["habitability"]["outcome"] = {"decision_change": "habitable"}.get(kind, kind)
    elif kind == "gas_change":
        data["gas_selection"]["gases"] = ["NH3"]
    elif kind == "temperature_change":
        data["temperature_provenance"]["optimizer_updates"] = 1
    elif kind == "phase_change":
        data["phase_reference"]["phase"] = "solid"
    elif kind == "save_integer":
        data["source_save_click_delivered"] = 1
    elif kind == "two_tabs":
        data["navigation_clicks"] = 2
    write(work / "confirmed.json", data)
    with pytest.raises((ProgressError, BrowserSafetyStop, ValueError)):
        module.import_verified_terrestrial(journal, history, inv, work)
    assert journal.path.read_bytes() == before and not list(history.glob("project-evidence-*.json"))


def test_concurrent_import_and_interrupted_prefix_do_not_duplicate(offline):
    journal, history, _, _, _, _ = offline
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ingest(offline), range(2)))
    assert sorted(r["appended_records"] for r in results) == [0, 8]
    complete = journal.path.read_bytes()
    for prefix in (0, 2, 4, 7):
        journal.path.write_bytes(b"\n".join(complete.splitlines()[: prefix + 1]) + b"\n")
        assert ingest(offline)["appended_records"] == 8 - prefix
        assert journal.path.read_bytes() == complete
    assert len(list(history.glob("project-evidence-*.json"))) == 1


def test_positive_and_terrestrial_share_ids_without_revising_prior_work(offline):
    journal, history, inv, _, registry, _ = offline
    giant = positive_workflow(history, "Beta", "giant", registry)
    first = positive_module.import_verified_positive_planet(journal, history, inv, giant)
    previous = journal.load().records
    result = ingest(offline)
    assert result["appended_records"] == 6 and result["progress"]["verified"] == 2
    assert journal.load().records[: len(previous)] == previous
    assert journal.load().reduce().stars[first["star_id"]].habitability.outcome == "not_applicable"
    assert positive_module.import_verified_positive_planet(journal, history, inv, giant)["idempotent"]
    with pytest.raises(ProgressError, match="unsupported_workflow"):
        module.import_verified_terrestrial(journal, history, inv, giant)


def test_uncertain_write_is_not_retroactively_completed(offline):
    journal, _, _, _, _, _ = offline
    journal.append(Collected(star_id="existing", name="Dulat", source_sha256="a" * 64))
    journal.append(
        WriteReserved(
            action_id="uncertain",
            write_kind="save_star",
            star_id="existing",
            revision=1,
            before_sha256="b" * 64,
        )
    )
    before = journal.path.read_bytes()
    with pytest.raises(ProgressError, match="uncertain"):
        ingest(offline)
    assert journal.path.read_bytes() == before and len(journal.load().reduce().pending) == 1


@pytest.mark.parametrize("kind", ["bytes", "uncertainty_marker", "nan"])
def test_last_moment_failure_is_atomic(offline, monkeypatch, kind):
    journal, history, _, _, _, _ = offline
    before = journal.path.read_bytes()
    original = module._stage_records
    if kind == "nan":
        monkeypatch.setattr(module, "_scopes", lambda *a: {"bad": float("nan")})
    else:

        def changed(*args):
            result = original(*args)
            name = (
                "source-first/temperature/report.json"
                if kind == "bytes"
                else "source-first/gas-selection/selection-stopped.json"
            )
            write(history / name, {})
            return result

        monkeypatch.setattr(module, "_stage_records", changed)
    with pytest.raises((BrowserSafetyStop, ValueError)):
        ingest(offline)
    assert journal.path.read_bytes() == before and not list(history.glob("project-evidence-*.json"))


@pytest.mark.parametrize("kind", ["receipt", "binding", "inventory", "rehash_capture", "symlink"])
def test_exact_reimport_cannot_revise_immutable_binding(offline, kind):
    journal, history, inv, work, _, _ = offline
    ingest(offline)
    before = journal.path.read_bytes()
    binding = next(history.glob("project-evidence-*.json"))
    if kind == "receipt":
        data = read(work / "confirmed.json")
        data["note"] = "Not an exact reimport"
        write(work / "confirmed.json", data)
    elif kind == "binding":
        data = read(binding)
        data["evidence_scopes"]["gas_selection"]["provenance"] = "learned_prediction"
        write(binding, data)
    elif kind == "inventory":
        inv = inventory(history, ["Dulat", "Beta"], "different")
    elif kind == "rehash_capture":
        capture_dir = work / "habitability-readback/verified"
        data = read(capture_dir / "observation.json")
        old = screen_identity(data)
        data["captured_at"] = "changed fixture metadata"
        assert screen_identity(data) == old
        write(capture_dir / "observation.json", data)
        manifest = read(capture_dir / "manifest.json")
        manifest["observation_sha256"] = sha(capture_dir / "observation.json")
        write(capture_dir / "manifest.json", manifest)
    else:
        target = history / "redirected.json"
        binding.rename(target)
        binding.symlink_to(target)
    with pytest.raises((ProgressError, BrowserSafetyStop)):
        module.import_verified_terrestrial(journal, history, inv, work)
    assert journal.path.read_bytes() == before


def test_30_collected_without_assessment_and_submission_is_not_project_completion(offline):
    journal, history, _, _, registry, _ = offline
    ingest(offline)
    names = ["Dulat", "Beta", *[f"Star{n}" for n in range(28)]]
    inv = inventory(history, names, "thirty")
    work = workflow(history, "Beta", "second", registry, outcome="habitable")
    result = module.import_verified_terrestrial(journal, history, inv, work)
    assert result["progress"]["collected"] == 30 and result["progress"]["verified"] == 2
    assert not result["progress"]["project_completed"] and not result["progress"]["submitted"]
    assert not result["progress"]["score_transfer_verified"] and not any(
        result["progress"]["assessment"].values()
    )
    before = journal.path.read_bytes()
    for bad in (
        inventory(history, names + ["Extra"], "too-many"),
        inventory(history, ["Dulat", "Beta"], "dropped"),
    ):
        with pytest.raises((ProgressError, BrowserSafetyStop)):
            module.import_verified_terrestrial(journal, history, bad, work)
    assert journal.path.read_bytes() == before


def test_foreign_or_ambiguous_attempt_cannot_receive_evidence(offline):
    journal, history, inv, work, _, _ = offline
    other = history / "other"
    other.mkdir()
    foreign = ProjectJournal(other, project_id="habworlds", attempt_id="other").create()
    with pytest.raises(ProgressError, match="journal_outside_attempt"):
        module.import_verified_terrestrial(foreign, history, inv, work)
    ProjectJournal(history, project_id="habworlds", attempt_id="duplicate").create()
    with pytest.raises(ProgressError, match="ambiguous_canonical_attempt"):
        ingest(offline)
    assert not journal.load().records


def test_intercepted_native_chain_imports_without_upstream_seam(terrestrial, tmp_path, monkeypatch):
    page, frame, screens = terrestrial
    seam(monkeypatch, frame, screens)
    receipt = run(page, tmp_path)
    inv = inventory(tmp_path, [receipt["star"]], "native")
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="intercepted-only").create()
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("Network forbidden during import")
    )
    result = module.import_verified_terrestrial(journal, tmp_path, inv, tmp_path / "terrestrial-workflow")
    assert result["appended_records"] == 7 and result["progress"]["verified"] == 1
    assert module.import_verified_terrestrial(journal, tmp_path, inv, tmp_path / "terrestrial-workflow")[
        "idempotent"
    ]
