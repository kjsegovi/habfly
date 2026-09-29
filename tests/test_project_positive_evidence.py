"""Offline journal import and one fully intercepted native-source integration.

Most cases use an explicit upstream-validator seam, as the No importer does;
this tests atomic journal behavior without generating browser actions or model
work. Synthetic reports here are not evidence of real task/science completion.
"""
# ruff: noqa: F811

import hashlib
import socket
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
import yaml
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet import capture as planet_capture
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_positive_planet_workflow import positive, run, seam  # noqa: F401
from test_browser_raster_planet_evidence import raster_page  # noqa: F401
from test_project_evidence import current_stellar, inventory, read, sha, write
from test_project_evidence import workflow as no_workflow

import habfly.project_evidence as no_module
import habfly.project_positive_evidence as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_workflow import _planet, _stellar
from habfly.browser_numeric import committed_display, screen_identity
from habfly.browser_probe import save_probe
from habfly.project_progress import Collected, ProgressError, ProjectJournal, WriteReserved


def current_planet(star):
    report = planet_capture("Yes")
    frame = report["frames"][0]
    frame["text"] = star.upper() + "\nOBSERVATIONS"
    atoms = yaml.safe_load(frame["accessibility"])
    atoms[0]["text"] = star
    ordinal = 0
    for index, atom in enumerate(atoms):
        if atom == 'textbox "0"' or isinstance(atom, dict) and 'textbox "0"' in atom:
            atoms[index] = {'textbox "0"': "5000" if ordinal == 0 else "1"}
            ordinal += 1
        elif isinstance(atom, dict) and "text" in atom:
            atom["text"] = atom["text"].replace("2.368", "1").replace("1.961", "1")
    frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)
    for index, control in enumerate(c for c in frame["controls"] if c["role"] == "textbox"):
        value = "5000" if index == 0 else "1"
        control.update(value=value, accessibility=yaml.safe_dump([{'textbox "0"': value}]))
    return report


def workflow(history, star, suffix, registry, *, planet_class="gas_giant"):
    base, directory = history / ("source-" + suffix), history / ("workflow-" + suffix)
    stellar, planet = current_stellar(star), current_planet(star)
    values = _stellar(stellar)["observation"]["values"]
    readbacks = {
        name: {"display_value": field["current_value"], "unit": field["unit"]}
        for name, field in values["browser_field_map"].items()
    }
    numeric = {
        "checkpoint_sha256": "a" * 64,
        "graph_hash": "b" * 64,
        "knowledge_pack_hash": "c" * 64,
        "optimizer_updates": 0,
        "selected_class": "main_sequence",
        "lifetime_prefix": "Ga",
    }
    color = {"color_checkpoint_sha256": "d" * 64, "color_reference_hash": "e" * 64, "optimizer_updates": 0}
    derived = {
        "checkpoint_sha256": "f" * 64,
        "graph_hash": "b" * 64,
        "knowledge_pack_hash": "1" * 64,
        "optimizer_updates": 0,
        "classification_source": "supplied_not_learned",
        "supplied_star_class": "main_sequence",
    }
    planet_readbacks = {
        name: {"display": committed_display("1", "1"), "unit": field["unit"]}
        for name, field in _planet(planet)["observation"]["values"]["browser_field_map"].items()
        if name != "observation_days"
    }
    raw_evidence = {
        "star": star.upper(),
        "measurements": {
            name: {"value": "1", "unit": planet_readbacks[name]["unit"]}
            for name in ("line_shift", "brightness_drop", "period_days")
        },
        "uncertainty": {
            "kind": "conditional_pixel_bounds_not_statistical_confidence",
            "period_days": {"value": 1.0, "lower": 0.9, "upper": 1.1},
            "limitations": ["physical period may be aliased"],
        },
    }
    payloads = {
        "numeric/manifest.json": {
            "outcome": "full_stellar_numeric_transport_verified",
            "provenance": numeric,
        },
        "color/manifest.json": {"outcome": "color_transport_verified", "provenance": color},
        "class/confirmed.json": {"action_source": "explicit_fresh_star_class_setup", "star": star},
        "raw/report.json": {"mode": module.MEASUREMENTS, "stage": "inputs", "evidence": raw_evidence},
        "derived/report.json": {"scope": "four_derived_planet_browser_transport", "provenance": derived},
        "planet-class/confirmed.json": {"kind": "SELECT", "value": planet_class, "star": star},
        "save/confirmed.json": {"kind": "CLICK", "visible_label": "Save", "star": star},
    }
    for name, payload in payloads.items():
        write(base / name, {**payload, "fixture_star": star})
    hashes = {str((base / name).relative_to(history)): sha(base / name) for name in payloads}
    bundle = {
        "star": star,
        "readbacks": readbacks,
        "color": "UV",
        "class": "main_sequence",
        "prefix": "Ga",
        "measurements": values["measurements"],
        "numeric_provenance": numeric,
        "color_provenance": color,
        "planet_class": planet_class,
        "raw": {"evidence": raw_evidence},
        "derived": {"provenance": derived},
        "planet_readbacks": planet_readbacks,
        "save_capture": planet,
    }
    registry[str(base / "numeric")] = (hashes, bundle)
    for name, report in (("stellar", stellar), ("planet", planet)):
        save_probe(report, directory / f"{name}-readback/verified")
    receipt = {
        "schema_version": 1,
        "mode": module.MODE,
        "authority": "visible_workflow_readback",
        "task_completed": True,
        "save_acknowledgement_verified": True,
        "star": star,
        "stellar_fields": readbacks,
        "color": "UV",
        "classification": "main_sequence",
        "classification_provenance": "reference_prediction",
        "numeric_provenance": numeric,
        "color_provenance": color,
        "navigation_clicks": 2,
        "source_save_click_delivered": True,
        "save_acknowledgement_source": "explicit_save_fresh_visible_footer",
        "planet_fields": planet_readbacks,
        "raw_measurement_evidence": raw_evidence,
        "derived_provenance": derived,
        "planet": {
            "outcome": "planet",
            "value": planet_class,
            "provenance": "reference_prediction",
            "measurement_provenance": module.MEASUREMENTS,
            "approximate": True,
            "transport_verified": True,
            "scientific_verified": False,
            "training_label": False,
        },
        "habitability": {
            "outcome": "not_applicable",
            "applicability_reason": "non_terrestrial_planet",
            "provenance": "reference_prediction",
            "branch_applicability_verified": True,
            "transport_verified": False,
            "scientific_verified": False,
            "source": "explicit_branch_applicability_not_field_write",
        },
        "source_sha256": hashes,
        "current_screen_sha256": {"stellar": screen_identity(stellar), "planet": screen_identity(planet)},
    }
    receipt.update(
        dict.fromkeys(
            (
                "learned_classification",
                "scientific_verified",
                "correctness_verified",
                "browser_acceptance_passed",
                "course_completion_verified",
                "project_completed",
                "submitted",
                "training_label",
                "cross_session_persistence_verified",
                "hidden_values_inspected",
                "hidden_values_blank_verified",
            ),
            False,
        )
    )
    receipt.update(
        dict.fromkeys(
            (
                "answer_writes",
                "habitability_writes",
                "na_writes",
                "save_clicks",
                "assessment_clicks",
                "score_transfer_clicks",
                "submission_clicks",
            ),
            0,
        )
    )
    write(directory / "confirmed.json", receipt)
    return directory


@pytest.fixture
def offline(tmp_path, monkeypatch):
    registry, calls = {}, []

    def validated(book, **directories):
        calls.append(directories)
        hashes, bundle = registry[str(directories["numeric_dir"])]
        for name in hashes:
            book.clean((book.history / name).parent)
            book.read(book.history / name)
        return deepcopy(bundle)

    monkeypatch.setattr(module, "_load_sources", validated)
    monkeypatch.setattr(no_module, "_load_sources", validated)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network access forbidden"))
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="synthetic-positive").create()
    inv = inventory(tmp_path, ["Dulat", "Beta"], "first")
    work = workflow(tmp_path, "Dulat", "first", registry)
    return journal, tmp_path, inv, work, registry, calls


def ingest(offline):
    return module.import_verified_positive_planet(*offline[:4])


def test_idempotent_import_preserves_scopes_without_fake_save_or_science(offline):
    journal, history, _, work, _, calls = offline
    result = ingest(offline)
    assert result["appended_records"] == 8 and result["progress"]["verified"] == 1
    assert result["star_id"] == "star:" + hashlib.sha256(b"dulat").hexdigest()
    assert result["progress"]["collected"] == 2 and not result["progress"]["project_completed"]
    assert not result["progress"]["submitted"] and not result["progress"]["score_transfer_verified"]
    assert not any(result["progress"]["assessment"].values())
    state = journal.load().reduce()
    assert not state.reservations and not state.receipts
    star = state.stars[result["star_id"]]
    assert star.name == "Dulat" and star.revision == 5 and star.task_completed
    assert star.stellar_numeric.decision.provenance == "learned_prediction"
    assert star.planet.decision.provenance == "reference_prediction"
    assert star.planet.decision.value == "gas_giant" and star.planet.policy_label == module.MEASUREMENTS
    assert star.habitability.applicability_reason == "non_terrestrial_planet"
    assert star.habitability.branch_applicability_verified and not star.habitability.transport_verified
    assert star.completion.source_sha256 == sha(work / "confirmed.json")
    binding_path = next(history.glob("project-evidence-*.json"))
    binding = read(binding_path)
    assert binding["evidence_scopes"] == result["evidence_scopes"]
    scopes = result["evidence_scopes"]
    assert scopes["raw_measurements"]["provenance"] == module.MEASUREMENTS
    assert (
        scopes["raw_measurements"]["uncertainty"]
        == read(work / "confirmed.json")["raw_measurement_evidence"]["uncertainty"]
    )
    assert scopes["derived_calculations"]["provenance"] == "learned_prediction"
    assert scopes["derived_calculations"]["input_provenance"] == module.MEASUREMENTS
    assert scopes["classification"]["provenance"] == "reference_prediction"
    assert all(s["scientific_verified"] is False and s["training_label"] is False for s in scopes.values())
    before = journal.path.read_bytes()
    again = ingest(offline)
    assert again["idempotent"] and again["appended_records"] == 0 and journal.path.read_bytes() == before
    assert len(calls) == 2 and binding_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "kind",
    [
        "source_hash",
        "inventory_hash",
        "capture",
        "task_integer",
        "science",
        "planet_exact",
        "planet_learned",
        "na_transport",
        "na_habitable",
        "uncertainty",
        "derived",
        "save_integer",
        "wrong_star",
        "source_chain",
        "invalidated",
        "inventory_missing",
        "incomplete_inventory",
        "ambiguous_source",
        "source_hash_omitted",
    ],
)
def test_invalid_sources_do_not_create_binding_or_append(offline, kind, monkeypatch):
    journal, history, inv, work, registry, _ = offline
    original = journal.path.read_bytes()
    data = read(work / "confirmed.json")
    if kind == "source_hash":
        (history / "source-first/raw/report.json").write_text("{}")
    elif kind == "inventory_hash":
        (inv / "after/observation.json").write_text("{}")
    elif kind == "capture":
        (work / "planet-readback/verified/observation.json").write_text("{}")
    elif kind == "source_chain":
        monkeypatch.setattr(
            module, "_load_sources", lambda *a, **k: (_ for _ in ()).throw(BrowserSafetyStop("bad_source"))
        )
    elif kind == "invalidated":
        write(work / "invalidated.json", {})
    elif kind == "wrong_star":
        registry[str(history / "source-first/numeric")][1]["star"] = "Other"
    elif kind == "inventory_missing":
        inv = inventory(history, ["Other", "Beta"], "missing")
    elif kind == "incomplete_inventory":
        receipt = read(inv / "confirmed.json")
        receipt["complete_visible_list_verified"] = False
        write(inv / "confirmed.json", receipt)
    elif kind == "ambiguous_source":
        path = history / "duplicate/confirmed.json"
        write(path, {"kind": "CLICK", "visible_label": "Save"})
        data["source_sha256"][str(path.relative_to(history))] = sha(path)
    elif kind == "source_hash_omitted":
        data["source_sha256"].pop("source-first/raw/report.json")
    elif kind == "task_integer":
        data["task_completed"] = 1
    elif kind == "science":
        data["scientific_verified"] = True
    elif kind == "planet_exact":
        data["planet"]["approximate"] = False
    elif kind == "planet_learned":
        data["planet"]["provenance"] = "learned_prediction"
    elif kind == "na_transport":
        data["habitability"]["transport_verified"] = True
    elif kind == "na_habitable":
        data["habitability"]["outcome"] = "not_habitable"
    elif kind == "uncertainty":
        data["raw_measurement_evidence"]["uncertainty"]["period_days"]["lower"] = 1
    elif kind == "derived":
        data["derived_provenance"]["optimizer_updates"] = 1
    elif kind == "save_integer":
        data["source_save_click_delivered"] = 1
    write(work / "confirmed.json", data)
    with pytest.raises((ProgressError, BrowserSafetyStop, ValueError)):
        module.import_verified_positive_planet(journal, history, inv, work)
    assert journal.path.read_bytes() == original and not list(history.glob("project-evidence-*.json"))


def test_concurrent_imports_and_interrupted_prefix_are_exact(offline):
    journal, history, _, _, _, _ = offline
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ingest(offline), range(2)))
    assert sorted(r["appended_records"] for r in results) == [0, 8]
    expected = journal.path.read_bytes()
    for prefix in (0, 2, 4, 7):
        journal.path.write_bytes(b"\n".join(expected.splitlines()[: prefix + 1]) + b"\n")
        assert ingest(offline)["appended_records"] == 8 - prefix
        assert journal.path.read_bytes() == expected
    assert len(list(history.glob("project-evidence-*.json"))) == 1


def test_revision_and_binding_tampering_never_update_existing_work(offline):
    journal, history, _, work, _, _ = offline
    ingest(offline)
    before = journal.path.read_bytes()
    path = work / "confirmed.json"
    original = path.read_bytes()
    data = read(path)
    data["extra_note"] = "A changed receipt is not exact reimport"
    write(path, data)
    with pytest.raises(ProgressError, match="revised_stage_evidence"):
        ingest(offline)
    path.write_bytes(original)
    binding_path = next(history.glob("project-evidence-*.json"))
    binding = read(binding_path)
    binding["evidence_scopes"]["derived_calculations"]["provenance"] = "scientific_observation"
    write(binding_path, binding)
    with pytest.raises(ProgressError, match="revised_import_binding"):
        ingest(offline)
    assert journal.path.read_bytes() == before


def test_no_and_positive_importers_share_ids_without_revising_prior_records(offline):
    journal, history, inv, work, registry, _ = offline
    old_work = no_workflow(history, "Beta", "no-planet", registry)
    first = no_module.import_verified_no_planet(journal, history, inv, old_work)
    previous = journal.load().records
    result = ingest(offline)
    assert result["appended_records"] == 6 and result["progress"]["verified"] == 2
    assert result["newly_collected"] == [] and journal.load().records[: len(previous)] == previous
    assert result["star_id"] == "star:" + hashlib.sha256(b"dulat").hexdigest()
    assert no_module.import_verified_no_planet(journal, history, inv, old_work)["idempotent"]
    assert journal.load().reduce().stars[first["star_id"]].planet.outcome == "no_planet"
    with pytest.raises(ProgressError, match="unsupported_workflow"):
        no_module.import_verified_no_planet(journal, history, inv, work)


def test_expansion_to_thirty_collected_is_not_project_completion(offline):
    journal, history, _, _, registry, _ = offline
    ingest(offline)
    names = ["Dulat", "Beta", *[f"Star{n}" for n in range(28)]]
    inv = inventory(history, names, "thirty")
    work = workflow(history, "Beta", "second", registry, planet_class="ice_giant")
    result = module.import_verified_positive_planet(journal, history, inv, work)
    assert result["progress"]["collected"] == 30 and result["progress"]["verified"] == 2
    assert not result["progress"]["project_completed"] and not result["progress"]["submitted"]
    before = journal.path.read_bytes()
    too_many = inventory(history, [*names, "Extra"], "too-many")
    with pytest.raises((ProgressError, BrowserSafetyStop)):
        module.import_verified_positive_planet(journal, history, too_many, work)
    dropped = inventory(history, ["Dulat", "Beta"], "dropped")
    with pytest.raises(ProgressError, match="inventory_dropped"):
        module.import_verified_positive_planet(journal, history, dropped, work)
    assert journal.path.read_bytes() == before


def test_uncertain_browser_write_is_not_retroactively_acknowledged(offline):
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


@pytest.mark.parametrize("kind", ["bytes", "stopped"])
def test_last_moment_source_change_is_atomic(offline, monkeypatch, kind):
    journal, history, _, _, _, _ = offline
    old = module._stage_records

    def changed(*args):
        result = old(*args)
        if kind == "bytes":
            (history / "source-first/raw/report.json").write_text("{}")
        else:
            write(history / "source-first/derived/stopped.json", {"write_may_have_occurred": True})
        return result

    monkeypatch.setattr(module, "_stage_records", changed)
    before = journal.path.read_bytes()
    with pytest.raises(BrowserSafetyStop, match="evidence_changed|failed_evidence"):
        ingest(offline)
    assert journal.path.read_bytes() == before and not list(history.glob("project-evidence-*.json"))


def test_wrong_or_ambiguous_attempt_cannot_receive_evidence(offline):
    journal, history, inv, work, _, _ = offline
    other = history / "other-attempt"
    other.mkdir()
    elsewhere = ProjectJournal(other, project_id="habworlds", attempt_id="other").create()
    with pytest.raises(ProgressError, match="journal_outside_attempt"):
        module.import_verified_positive_planet(elsewhere, history, inv, work)
    with pytest.raises(BrowserSafetyStop, match="outside_history"):
        module.import_verified_positive_planet(elsewhere, other, inv, work)
    ProjectJournal(history, project_id="habworlds", attempt_id="duplicate").create()
    with pytest.raises(ProgressError, match="ambiguous_canonical_attempt_journal"):
        ingest(offline)
    assert not journal.load().records and not list(history.glob("project-evidence-*.json"))


def test_revised_inventory_is_not_an_exact_reimport(offline):
    journal, history, _, work, _, _ = offline
    ingest(offline)
    before = journal.path.read_bytes()
    replacement = inventory(history, ["Dulat", "Beta"], "different-source-pair")
    with pytest.raises(ProgressError, match="revised_import_binding"):
        module.import_verified_positive_planet(journal, history, replacement, work)
    assert journal.path.read_bytes() == before


def test_invalid_serialization_creates_no_empty_binding(offline, monkeypatch):
    journal, history, _, _, _, _ = offline
    monkeypatch.setattr(module, "_scopes", lambda *args: {"invalid_fixture_value": float("nan")})
    before = journal.path.read_bytes()
    with pytest.raises(ValueError):
        ingest(offline)
    assert journal.path.read_bytes() == before and not list(history.glob("project-evidence-*.json"))


def test_binding_symlink_never_replaces_canonical_evidence(offline):
    journal, history, _, _, _, _ = offline
    ingest(offline)
    before = journal.path.read_bytes()
    binding = next(history.glob("project-evidence-*.json"))
    target = history / "redirected-binding.json"
    binding.rename(target)
    binding.symlink_to(target)
    with pytest.raises(BrowserSafetyStop, match="symlink_evidence"):
        ingest(offline)
    assert journal.path.read_bytes() == before and binding.is_symlink()


def test_rehashed_semantically_identical_capture_is_not_immutable_reimport(offline):
    journal, _, _, work, _, _ = offline
    ingest(offline)
    before = journal.path.read_bytes()
    directory = work / "planet-readback/verified"
    report = read(directory / "observation.json")
    old_identity = screen_identity(report)
    report["captured_at"] = "Changed synthetic metadata"
    assert screen_identity(report) == old_identity
    write(directory / "observation.json", report)
    manifest = read(directory / "manifest.json")
    manifest["observation_sha256"] = sha(directory / "observation.json")
    write(directory / "manifest.json", manifest)
    with pytest.raises(ProgressError, match="revised_import_binding"):
        ingest(offline)
    assert journal.path.read_bytes() == before


@pytest.mark.parametrize("positive", ["mixed_case"], indirect=True)
def test_real_intercepted_receipt_chain_imports_without_source_validator_seam(
    positive, tmp_path, monkeypatch
):
    page, frame, screens = positive
    seam(monkeypatch, frame, screens)
    receipt = run(page, tmp_path)
    assert receipt["star"] == "Dulat" and receipt["raw_measurement_evidence"]["star"] == "DULAT"
    inv = inventory(tmp_path, ["Dulat"], "actual-chain")
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="intercepted-only").create()
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("Network forbidden during import")
    )
    result = module.import_verified_positive_planet(journal, tmp_path, inv, tmp_path / "workflow")
    assert result["appended_records"] == 7 and result["progress"]["verified"] == 1
    assert module.import_verified_positive_planet(journal, tmp_path, inv, tmp_path / "workflow")["idempotent"]
