"""Collection counts never promote unverified work; no network/browser needed."""

import hashlib
import json
import socket
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_project_evidence import inventory, write

import habfly.project_inventory_evidence as module
from habfly.browser import BrowserSafetyStop
from habfly.project_progress import (
    Collected,
    Evidence,
    ProjectJournal,
    StageEvidence,
    StageRecorded,
    WriteReserved,
)


@pytest.fixture
def case(tmp_path, monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Inventory ingestion must work offline")

    monkeypatch.setattr(socket, "create_connection", deny)
    journal = ProjectJournal(tmp_path, project_id="fixture", attempt_id="attempt").create()
    source = inventory(tmp_path, ["Alpha", "Beta"], "two")
    return tmp_path, journal, source


def ingest(case):
    root, journal, source = case
    return module.import_verified_inventory(journal, root, source)


def test_collection_only_is_idempotent_without_completion_or_fake_writes(case):
    root, journal, _ = case
    result = ingest(case)
    assert result["appended_records"] == 2
    assert result["newly_collected"] == ["Alpha", "Beta"]
    assert result["progress"]["collected"] == result["progress"]["unresolved"] == 2
    assert result["progress"]["verified"] == 0
    assert not result["progress"]["project_completed"]
    assert (
        result["scientific_stages_added"]
        == result["task_receipts_added"]
        == result["write_receipts_added"]
        == 0
    )
    records = journal.load().records
    assert all(record.payload.kind == "collected" for record in records)
    saved = journal.path.read_bytes()
    assert ingest(case)["idempotent"]
    assert journal.path.read_bytes() == saved
    assert {star.id for star in journal.load().reduce().stars.values()} == {
        "star:" + hashlib.sha256(name.encode()).hexdigest() for name in ("alpha", "beta")
    }
    assert not list(root.glob("project-evidence-*.json"))


def test_expansion_preserves_all_previous_records(case):
    root, journal, _ = case
    ingest(case)
    saved = journal.path.read_bytes()
    source = inventory(root, ["Alpha", "Beta", "Gamma"], "three")
    result = ingest((root, journal, source))
    assert result["newly_collected"] == ["Gamma"]
    assert journal.path.read_bytes().startswith(saved)
    assert result["progress"]["verified"] == 0


def test_existing_scientific_stage_is_not_modified_or_promoted(case):
    root, journal, source = case
    ingest(case)
    star_id = "star:" + hashlib.sha256(b"alpha").hexdigest()
    journal.append(
        StageRecorded(
            star_id=star_id,
            stage="stellar_numeric",
            evidence=StageEvidence(
                decision=Evidence(source_sha256="a" * 64, provenance="learned_prediction")
            ),
        )
    )
    prior = journal.load().reduce().stars[star_id]
    saved = journal.path.read_bytes()
    result = ingest((root, journal, source))
    assert result["idempotent"]
    assert journal.path.read_bytes() == saved
    assert journal.load().reduce().stars[star_id] == prior
    assert result["progress"]["verified"] == 0


@pytest.mark.parametrize(
    "change", ["receipt", "capture", "dropped", "ambiguous", "uncertain", "partial_journal"]
)
def test_invalid_evidence_never_appends(case, change):
    root, journal, source = case
    if change == "receipt":
        receipt = json.loads((source / "confirmed.json").read_bytes())
        receipt["task_completed"] = True
        write(source / "confirmed.json", receipt)
    elif change == "capture":
        (source / "after/observation.json").write_text("{}")
    elif change == "dropped":
        journal.append(Collected(star_id="missing", name="Gamma", source_sha256="a" * 64))
    elif change == "ambiguous":
        ProjectJournal(root, project_id="fixture", attempt_id="another").create()
    elif change == "uncertain":
        journal.append(
            WriteReserved(
                action_id="pending",
                write_kind="assessment_data_quality",
                revision=0,
                before_sha256="a" * 64,
                project_rows_sha256="b" * 64,
            )
        )
    else:
        with journal.path.open("a") as stream:
            stream.write("{")
    saved = journal.path.read_bytes()
    with pytest.raises((ValueError, BrowserSafetyStop)):
        ingest(case)
    assert journal.path.read_bytes() == saved


def test_concurrent_import_appends_each_name_once(case):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ingest(case), range(2)))
    assert sorted(r["appended_records"] for r in results) == [0, 2]
    assert case[1].load().reduce().report()["collected"] == 2


def test_source_change_at_final_recheck_never_appends(case, monkeypatch):
    _, journal, source = case
    saved = journal.path.read_bytes()
    original = module._Evidence.unchanged

    def changed(book):
        (source / "confirmed.json").write_text("{}")
        original(book)

    monkeypatch.setattr(module._Evidence, "unchanged", changed)
    with pytest.raises(BrowserSafetyStop, match="evidence_changed"):
        ingest(case)
    assert journal.path.read_bytes() == saved
