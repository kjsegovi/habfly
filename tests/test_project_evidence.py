"""Offline ingestion tests with public capture transcriptions and source seam.

The existing browser source validator has its own complete fixture suite. Here
its explicit seam returns prevalidated synthetic bundles, allowing journal and
artifact validation to run without a browser, network, checkpoints, or training.
These fixture receipts are not evidence of any real star's completion.
"""

import hashlib
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
import yaml
from test_browser_planet import capture as planet_capture
from test_browser_stellar import capture as stellar_capture

import habfly.project_evidence as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_save import CONDITIONAL_FIELDS, RAW_FIELDS
from habfly.browser_no_planet_workflow import _planet, _stellar
from habfly.browser_numeric import screen_identity
from habfly.browser_planet_window_choice import MODE as CHOICE_MODE
from habfly.browser_probe import save_probe
from habfly.browser_project_inventory import _capture_inventory
from habfly.browser_project_navigation import LIST_LABELS
from habfly.browser_stellar import SIMULATION_URL
from habfly.planet_window_policy import policy_manifest
from habfly.project_evidence import import_verified_no_planet
from habfly.project_progress import Collected, ProgressError, ProjectJournal, WriteReserved


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_stellar(star, selected_class="main_sequence"):
    report = stellar_capture()
    report["ignored_frame_urls"] = []
    frame = report["frames"][0]
    frame["text"] = star + "\nOBSERVATIONS"
    atoms = yaml.safe_load(frame["accessibility"])
    atoms[1]["text"] = star
    if selected_class == "main_sequence":
        atoms = [
            a for a in atoms if not isinstance(a, dict) or not str(a.get("text", "")).startswith("mass,")
        ]
    for atom in atoms:
        if isinstance(atom, dict) and "combobox" in atom:
            atom["combobox"][-1] += " [selected]"
            frame["controls"][3]["accessibility"] = yaml.safe_dump([atom], sort_keys=False)
            frame["controls"][3]["value"] = "UV"
    for control in frame["controls"][:3]:
        control["value"] = "1"
    if selected_class != "main_sequence":
        frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)
        return report
    for label in ("mass (Ms)", "radius (Rs)", "lifetime (years)"):
        atoms.extend([{"text": label}, 'textbox "0"'])
        frame["controls"].append(
            {
                "id": f"simulation-0:c{len(frame['controls']) + 10}",
                "role": "textbox",
                "enabled": True,
                "value": "1",
                "accessibility": '- textbox "0"',
            }
        )
    prefix = {"combobox": ["option", 'option "ka"', 'option "Ma"', 'option "Ga" [selected]', 'option "Ta"']}
    atoms.extend([prefix, {"text": "main sequence red giant supergiant white dwarf"}])
    frame["controls"].append(
        {
            "id": "simulation-0:c30",
            "role": "combobox",
            "enabled": True,
            "value": "Ga",
            "accessibility": yaml.safe_dump([prefix], sort_keys=False),
        }
    )
    frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)
    return report


def current_planet(star):
    report = planet_capture("No")
    frame = report["frames"][0]
    frame["text"] = star + "\nOBSERVATIONS"
    frame["accessibility"] = frame["accessibility"].replace("Jyremis", star)
    return report


def inventory(history, names, suffix, *, classes=None):
    directory = history / ("inventory-" + suffix)
    atoms = [{"text": "Funding Observations Analyzed Data Star " + " ".join(LIST_LABELS["stellar"])}]
    for name in names:
        selected = (classes or {}).get(name, "main_sequence")
        label = selected.replace("_", " ").title()
        conditional = "1 1 1 Ga" if selected == "main_sequence" else "- - -"
        atoms.extend(["img", {"text": f"{name} 0.045 370 5.68E-10 1 1 UV 1 {label} {conditional}"}])
    atoms.append({"text": f"viewing 1-{len(names)} of {len(names)} total collected {len(names)}"})
    report = {
        "schema_version": 1,
        "mode": "read_only_browser_preflight",
        "ignored_frame_urls": [],
        "actions_executed": 0,
        "allow_submission": False,
        "frames": [
            {
                "id": "simulation-0",
                "url": SIMULATION_URL,
                "text": "VISIBLE LIST",
                "accessibility": yaml.safe_dump(atoms, sort_keys=False),
                "controls": [],
            }
        ],
    }
    before, after = (save_probe(report, directory / name) for name in ("before", "after"))
    counts, rows = _capture_inventory(report)
    receipt = {
        "schema_version": 1,
        "mode": "read_only_collected_star_inventory",
        "authority": "visible_collected_list_readback",
        "section": "stellar",
        "collection_count_verified": True,
        "complete_visible_list_verified": True,
        "viewing": counts,
        "total_collected": len(names),
        "visible_row_count": len(names),
        "expected_stars_verified": False,
        "rows": [
            {
                "name": r["name"],
                "source_sha256": after["observation_sha256"],
                "row_accessibility_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
                "visible_name_box": {"x": 20, "y": 20 + i * 25, "width": 40, "height": 20},
            }
            for i, r in enumerate(rows)
        ],
        "source_sha256": {
            "before/observation.json": before["observation_sha256"],
            "after/observation.json": after["observation_sha256"],
        },
    }
    receipt.update(
        dict.fromkeys(
            (
                "hidden_catalog_read",
                "learned_perception",
                "scientific_verified",
                "task_completed",
                "project_completed",
                "cross_session_persistence_verified",
                "config_mutated",
                "automatic_retry",
            ),
            False,
        )
    )
    receipt.update(
        dict.fromkeys(
            (
                "browser_actions",
                "navigation_clicks",
                "scrolls",
                "answer_writes",
                "save_clicks",
                "deletion_clicks",
                "assessment_clicks",
                "submission_clicks",
            ),
            0,
        )
    )
    write(directory / "confirmed.json", receipt)
    return directory


def workflow(history, star, suffix, registry, *, selected_class="main_sequence"):
    base = history / ("source-" + suffix)
    directory = history / ("workflow-" + suffix)
    stellar, planet = current_stellar(star, selected_class), current_planet(star)
    values = _stellar(stellar)["observation"]["values"]
    readbacks = {
        name: {"display_value": field["current_value"], "unit": field["unit"]}
        for name, field in values["browser_field_map"].items()
    }
    numeric_provenance = {
        "checkpoint_sha256": "a" * 64,
        "graph_hash": "b" * 64,
        "knowledge_pack_hash": "c" * 64,
        "optimizer_updates": 0,
        "selected_class": selected_class,
        "lifetime_prefix": "Ga" if selected_class == "main_sequence" else None,
    }
    color_provenance = {
        "color_checkpoint_sha256": "d" * 64,
        "color_reference_hash": "e" * 64,
        "optimizer_updates": 0,
    }
    payloads = {
        "numeric/manifest.json": {
            "outcome": "full_stellar_numeric_transport_verified",
            "provenance": numeric_provenance,
        },
        "color/manifest.json": {"outcome": "color_transport_verified", "provenance": color_provenance},
        "class/confirmed.json": {"action_source": "explicit_fresh_star_class_setup", "star": star},
        "choice/confirmed.json": {"mode": CHOICE_MODE, "star": star, "policy": policy_manifest()},
        "save/confirmed.json": {"mode": "bounded_window_no_planet_save_acknowledgement", "star": star},
    }
    for name, payload in payloads.items():
        payload["fixture_star"] = star
        write(base / name, payload)
    hashes = {str((base / name).relative_to(history)): sha(base / name) for name in payloads}
    bundle = {
        "star": star,
        "readbacks": readbacks,
        "color": "UV",
        "class": selected_class,
        "prefix": "Ga" if selected_class == "main_sequence" else None,
        "measurements": values["measurements"],
        "numeric_provenance": numeric_provenance,
        "color_provenance": color_provenance,
        "choice": payloads["choice/confirmed.json"],
        "save_capture": planet,
        "save_mapping": _planet(planet),
        "save_click_delivered": True,
        "save_resumed_same_intent": False,
        "save_continuation_stopped_predispatch": False,
        "acknowledgement_source": "explicit_save_fresh_visible_footer",
    }
    registry[str(base / "numeric")] = (hashes, bundle)
    for name, report in (("stellar", stellar), ("planet", planet)):
        save_probe(report, directory / f"{name}-readback/verified")
    receipt = {
        "schema_version": 1,
        "mode": "no_planet_visible_workflow_readback",
        "authority": "visible_workflow_readback",
        "task_completed": True,
        "save_acknowledgement_verified": True,
        "star": star,
        "stellar_fields": readbacks,
        "color": "UV",
        "classification": selected_class,
        "classification_provenance": "reference_prediction",
        "numeric_provenance": numeric_provenance,
        "color_provenance": color_provenance,
        "navigation_clicks": 2,
        "source_save_click_delivered": True,
        "source_save_resumed_same_intent": False,
        "source_save_continuation_stopped_predispatch": False,
        "save_acknowledgement_source": bundle["acknowledgement_source"],
        "visible_blank_raw_fields": list(RAW_FIELDS),
        "conditionally_absent_derived_fields": list(CONDITIONAL_FIELDS),
        "planet": {
            "outcome": "no_planet",
            "provenance": "reference_prediction",
            "transport_verified": True,
            "scientific_verified": False,
            "policy_label": policy_manifest()["version"],
            "observation_limit_days": 5000,
        },
        "habitability": {
            "outcome": "not_applicable",
            "applicability_reason": "no_planet",
            "provenance": "reference_prediction",
            "transport_verified": False,
            "branch_applicability_verified": True,
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
def sources(tmp_path, monkeypatch):
    registry = {}
    calls = []

    def validated_sources(book, **directories):
        calls.append(directories)
        hashes, bundle = registry[str(directories["numeric_dir"])]
        for name in hashes:
            book.read(book.history / name)
        return deepcopy(bundle)

    monkeypatch.setattr(module, "_load_sources", validated_sources)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network access forbidden"))
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="synthetic-attempt").create()
    inv = inventory(tmp_path, ["Althinagon", "Beta"], "first")
    work = workflow(tmp_path, "Althinagon", "first", registry)
    return journal, tmp_path, inv, work, registry, calls


def ingest(sources):
    return import_verified_no_planet(*sources[:4])


def test_offline_import_is_exactly_idempotent_and_truthful(sources):
    journal, history, _, work, _, calls = sources
    result = ingest(sources)
    assert result["appended_records"] == 8 and result["newly_collected"] == ["Althinagon", "Beta"]
    assert result["progress"]["collected"] == 2 and result["progress"]["verified"] == 1
    assert not result["progress"]["project_completed"] and not result["progress"]["submitted"]
    assert (
        not any(result["progress"]["assessment"].values())
        and not result["progress"]["score_transfer_verified"]
    )
    star = journal.load().reduce().stars[result["star_id"]]
    assert star.revision == 5 and star.task_completed
    assert (
        star.stellar_numeric.decision.provenance
        == star.stellar_color.decision.provenance
        == "learned_prediction"
    )
    assert (
        star.stellar_classification.decision.provenance
        == star.planet.decision.provenance
        == "reference_prediction"
    )
    assert star.habitability.branch_applicability_verified and not star.habitability.transport_verified
    assert not star.planet.report()["scientific_verified"] and star.planet.observation_limit_days == 5000
    assert star.completion.source_sha256 == sha(work / "confirmed.json")
    assert not journal.load().reduce().reservations and not journal.load().reduce().receipts
    raw = journal.path.read_bytes()
    again = ingest(sources)
    assert again["idempotent"] and again["appended_records"] == 0 and journal.path.read_bytes() == raw
    assert len(calls) == 2 and len(list(history.glob("project-evidence-*.json"))) == 1
    assert (next(history.glob("project-evidence-*.json")).stat().st_mode & 0o777) == 0o600


@pytest.mark.parametrize("selected_class", ["red_giant", "supergiant", "white_dwarf"])
def test_non_main_no_branch_preserves_supplied_class_and_only_applicable_fields(sources, selected_class):
    journal, history, _, _, registry, _ = sources
    inv = inventory(history, ["Althinagon"], "non-main", classes={"Althinagon": selected_class})
    work = workflow(history, "Althinagon", "non-main", registry, selected_class=selected_class)
    result = import_verified_no_planet(journal, history, inv, work)
    star = journal.load().reduce().stars[result["star_id"]]
    assert star.task_completed and star.stellar_classification.decision.value == selected_class
    assert star.stellar_classification.decision.provenance == "reference_prediction"
    assert star.planet.outcome == "no_planet" and star.habitability.outcome == "not_applicable"
    assert star.habitability.branch_applicability_verified and not star.habitability.transport_verified
    assert not star.planet.report()["scientific_verified"]
    assert set(read(work / "confirmed.json")["stellar_fields"]) == {"distance", "luminosity", "temperature"}
    before = journal.path.read_bytes()
    assert import_verified_no_planet(journal, history, inv, work)["idempotent"]
    assert journal.path.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    ["extra_mass", "missing_temperature", "prefix", "missing_prefix", "class", "habitable", "hidden_blank"],
)
def test_non_main_applicability_mismatch_never_mutates_ledger(sources, change):
    journal, history, _, _, registry, _ = sources
    inv = inventory(history, ["Althinagon"], "non-main", classes={"Althinagon": "white_dwarf"})
    work = workflow(history, "Althinagon", "non-main", registry, selected_class="white_dwarf")
    path = work / "confirmed.json"
    receipt = read(path)
    if change == "extra_mass":
        receipt["stellar_fields"]["mass"] = {"display_value": "0", "unit": "M_sun"}
    elif change == "missing_temperature":
        receipt["stellar_fields"].pop("temperature")
    elif change == "prefix":
        receipt["numeric_provenance"]["lifetime_prefix"] = "Ga"
    elif change == "missing_prefix":
        receipt["numeric_provenance"].pop("lifetime_prefix")
    elif change == "class":
        receipt["numeric_provenance"]["selected_class"] = "red_giant"
    elif change == "habitable":
        receipt["habitability"]["outcome"] = "not_habitable"
    else:
        receipt["hidden_values_blank_verified"] = True
    write(path, receipt)
    before = journal.path.read_bytes()
    with pytest.raises(ProgressError):
        import_verified_no_planet(journal, history, inv, work)
    assert journal.path.read_bytes() == before and not list(history.glob("project-evidence-*.json"))


def test_expanded_inventory_adds_stars_without_revising_existing_records(sources):
    journal, history, _, first_work, registry, _ = sources
    first = ingest(sources)
    old_records = journal.load().records
    inv = inventory(history, ["Beta", "Althinagon", "Gamma"], "expanded")
    work = workflow(history, "Beta", "second", registry)
    result = import_verified_no_planet(journal, history, inv, work)
    assert result["newly_collected"] == ["Gamma"] and result["appended_records"] == 7
    assert result["progress"]["collected"] == 3 and result["progress"]["verified"] == 2
    assert journal.load().records[: len(old_records)] == old_records
    assert journal.load().reduce().stars[first["star_id"]].revision == 5
    before = journal.path.read_bytes()
    with pytest.raises(ProgressError, match="revised_import_binding"):
        import_verified_no_planet(journal, history, inv, first_work)
    assert journal.path.read_bytes() == before


@pytest.mark.parametrize(
    "kind",
    [
        "source_hash",
        "inventory_hash",
        "workflow_capture",
        "task_false",
        "task_integer",
        "science",
        "planet_science",
        "planet_outcome",
        "days",
        "na_transport",
        "na_applicability",
        "source_dispatch_truthy",
        "submitted",
        "write_count",
        "source_chain",
        "inventory_missing_name",
        "covered_flag",
        "invalidated",
        "wrong_star",
    ],
)
def test_invalid_evidence_never_mutates_journal_or_creates_binding(sources, kind, monkeypatch):
    journal, history, inv, work, registry, _ = sources
    before = journal.path.read_bytes()
    path = work / "confirmed.json"
    data = read(path)
    if kind == "source_hash":
        next((history / "source-first").rglob("manifest.json")).write_text("{}")
    elif kind == "inventory_hash":
        (inv / "after/observation.json").write_text("{}")
    elif kind == "workflow_capture":
        (work / "stellar-readback/verified/observation.json").write_text("{}")
    elif kind == "source_chain":
        monkeypatch.setattr(
            module,
            "_load_sources",
            lambda *a, **k: (_ for _ in ()).throw(BrowserSafetyStop("invalid_source")),
        )
    elif kind == "inventory_missing_name":
        data["star"] = "Missing"
    elif kind == "covered_flag":
        receipt = read(inv / "confirmed.json")
        receipt["complete_visible_list_verified"] = False
        write(inv / "confirmed.json", receipt)
    elif kind == "invalidated":
        write(work / "invalidated.json", {})
    elif kind == "wrong_star":
        registry[str(history / "source-first/numeric")][1]["star"] = "Other"
    else:
        keys = {
            "task_false": ("task_completed", False),
            "task_integer": ("task_completed", 1),
            "science": ("scientific_verified", True),
            "submitted": ("submitted", True),
            "write_count": ("save_clicks", 1),
            "source_dispatch_truthy": ("source_save_click_delivered", 1),
        }
        if kind in keys:
            key, value = keys[kind]
            data[key] = value
        elif kind == "planet_science":
            data["planet"]["scientific_verified"] = True
        elif kind == "planet_outcome":
            data["planet"]["outcome"] = "planet"
        elif kind == "days":
            data["planet"]["observation_limit_days"] = 10000
        elif kind == "na_transport":
            data["habitability"]["transport_verified"] = True
        elif kind == "na_applicability":
            data["habitability"]["branch_applicability_verified"] = False
    write(path, data)
    with pytest.raises((ProgressError, BrowserSafetyStop, ValueError)):
        ingest(sources)
    assert journal.path.read_bytes() == before and not list(history.glob("project-evidence-*.json"))


def test_same_star_revised_workflow_is_not_an_update(sources):
    journal, _, _, work, _, _ = sources
    ingest(sources)
    before = journal.path.read_bytes()
    receipt = read(work / "confirmed.json")
    receipt["extra_note"] = "A changed artifact is not exact re-ingestion"
    write(work / "confirmed.json", receipt)
    with pytest.raises(ProgressError, match="revised_stage_evidence"):
        ingest(sources)
    assert journal.path.read_bytes() == before


def test_wrong_attempt_journal_and_sources_rejected(sources, tmp_path):
    journal, history, inv, work, _, _ = sources
    other = history / "other-attempt"
    other.mkdir()
    elsewhere = ProjectJournal(other, project_id="habworlds", attempt_id="other").create()
    with pytest.raises(ProgressError, match="journal_outside_attempt"):
        import_verified_no_planet(elsewhere, history, inv, work)
    with pytest.raises(BrowserSafetyStop, match="outside_history"):
        import_verified_no_planet(elsewhere, other, inv, work)
    ProjectJournal(history, project_id="habworlds", attempt_id="ambiguous").create()
    with pytest.raises(ProgressError, match="ambiguous_canonical_attempt_journal"):
        ingest(sources)
    assert len(journal.load().records) == 0


def test_concurrent_identical_ingestion_has_only_one_batch(sources):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ingest(sources), range(2)))
    assert sorted(result["appended_records"] for result in results) == [0, 8]
    assert len(sources[0].load().records) == 8


@pytest.mark.parametrize("prefix_records", [0, 4])
def test_matching_prejournal_binding_can_finish_a_prevalidated_batch(sources, prefix_records):
    journal, history, _, _, _, _ = sources
    ingest(sources)
    # Controlled crash-recovery fixture: preserve original immutable binding,
    # restore the pre-append journal header, then re-ingest the exact sources.
    original = journal.path.read_bytes()
    journal.path.write_text("\n".join(journal.path.read_text().splitlines()[: prefix_records + 1]) + "\n")
    assert len(list(history.glob("project-evidence-*.json"))) == 1
    assert ingest(sources)["appended_records"] == 8 - prefix_records
    assert journal.path.read_bytes() == original


def test_no_click_acknowledgement_completes_workflow_without_inventing_dispatch(sources):
    journal, history, _, work, registry, _ = sources
    hashes, bundle = registry[str(history / "source-first/numeric")]
    bundle.update(
        save_click_delivered=False,
        save_continuation_stopped_predispatch=True,
        acknowledgement_source="visible_footer_no_explicit_save_dispatch",
    )
    path = history / "source-first/save/confirmed.json"
    source = read(path)
    source["mode"] = "no_planet_predispatch_read_only_reconciliation"
    write(path, source)
    hashes[str(path.relative_to(history))] = sha(path)
    receipt = read(work / "confirmed.json")
    receipt.update(
        source_sha256=hashes,
        source_save_click_delivered=False,
        source_save_continuation_stopped_predispatch=True,
        save_acknowledgement_source=bundle["acknowledgement_source"],
    )
    write(work / "confirmed.json", receipt)
    result = ingest(sources)
    assert result["progress"]["verified"] == 1
    assert not journal.load().reduce().reservations and not journal.load().reduce().receipts


def test_uncertain_write_does_not_become_a_retroactive_save_receipt(sources):
    journal, _, _, _, _, _ = sources
    journal.append(Collected(star_id="existing", name="Althinagon", source_sha256="a" * 64))
    journal.append(
        WriteReserved(
            action_id="pending",
            write_kind="save_star",
            star_id="existing",
            revision=1,
            before_sha256="b" * 64,
        )
    )
    before = journal.path.read_bytes()
    with pytest.raises(ProgressError, match="uncertain"):
        ingest(sources)
    assert journal.path.read_bytes() == before and len(journal.load().reduce().pending) == 1
