"""Offline archive/integration seams, not execution of held transfer cases.

Synthetic gate files exercise ownership, identity and archival byte checks. The
expensive independent trajectory validator is explicitly injected; its own test
suite proves gate arithmetic. Real class/navigation/raw M/R validators remain.
"""

# ruff: noqa: F811

import hashlib
import socket
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_habitability import capture as habitat_capture
from test_browser_star_preflight import read, save_capture, write
from test_planet_supplied_stellar_source import build, case, class_evidence, sources, store  # noqa: F401

import habfly.browser_supplied_provenance as module
import habfly.training.supplied_input_transfer as transfer
from habfly.browser import BrowserSafetyStop


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network, evaluation, model, expert, or private-case generation")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for name in ("run", "_materialize", "_inverse_case", "_env", "load_checkpoint"):
        if hasattr(transfer, name):
            monkeypatch.setattr(transfer, name, forbidden)


@pytest.fixture
def gate(tmp_path, monkeypatch):
    originals, calls = {}, []

    def create(task="planet"):
        source, pack, adapter = module._task(task)
        directory = tmp_path / ("fixture-gate-" + task)
        directory.mkdir()
        original_source = tmp_path / ("fixture-source-" + task)
        original_source.mkdir()
        checkpoint, metadata = original_source / "checkpoint.pt", original_source / "checkpoint.pt.json"
        checkpoint.write_bytes(b"synthetic non-model bytes")
        write(metadata, {"fixture_only": True})
        identity = {
            "checkpoint_sha256": sha(checkpoint),
            "parent_metadata_sha256": sha(metadata),
            "parent_content_hash": "c" * 64,
            "parent_pack_hash": source.BASE_PACK_HASH,
            "parent_scope": source.LEGACY_SCOPE,
            "graph_hash": "d" * 64,
            "pack_hash": pack.checksum,
            "adapter_sha256": adapter["sha256"],
            "parent_checkpoint_path": str(checkpoint),
            "parent_metadata_path": str(metadata),
            "source_sha256": {str(p): sha(p) for p in (checkpoint, metadata)},
        }
        public = {"fixture_only": True, "identity": identity, "task": task, "aggregate_only": True}
        for name in module._artifact_names(task)[0]:
            write(directory / name, public if name == "report.json" else {"synthetic_proof_leaf": name})
        reservation = tmp_path / "fixture-registry" / module._artifact_names(task)[1]
        write(reservation, {"synthetic_reservation": True})
        artifacts = {str(p): sha(p) for p in (*directory.iterdir(), reservation)}
        validated = {
            "scope": "supplied_input_transfer_gate",
            "task": task,
            "report": public,
            "identity": identity,
            "scores": {"episodes": 100, "completed": 100},
            "report_sha256": sha(directory / "report.json"),
            "source_sha256": identity["source_sha256"],
            "artifact_sha256": artifacts,
            "artifact_origins": {p: p for p in artifacts},
            "source_origins": {p: p for p in identity["source_sha256"]},
            "transfer_gate_passed": True,
            "native_browser_enabled": False,
            "course_acceptance_passed": False,
            "scientific_verified": False,
            "current_sources_verified": True,
            "historical_provenance_verified": True,
        }
        originals[task] = deepcopy(validated)
        return validated

    def validate(
        directory,
        *,
        task,
        checkpoint_sha256,
        graph_hash,
        pack_hash,
        adapter_sha256,
        artifact_paths,
        source_paths,
    ):
        calls.append({"task": task, "artifacts": dict(artifact_paths), "sources": dict(source_paths)})
        expected = originals[task]
        assert str(directory) == str(
            Path(next(p for p in expected["artifact_sha256"] if Path(p).name == "report.json")).parent
        )
        for key, actual in {
            "checkpoint_sha256": checkpoint_sha256,
            "graph_hash": graph_hash,
            "pack_hash": pack_hash,
            "adapter_sha256": adapter_sha256,
        }.items():
            if expected["identity"][key] != actual:
                raise ValueError("fixture_gate_identity_changed")
        assert set(artifact_paths) == set(expected["artifact_sha256"])
        assert set(source_paths) == set(expected["source_sha256"])
        for pins, paths in (
            (expected["artifact_sha256"], artifact_paths),
            (expected["source_sha256"], source_paths),
        ):
            for origin, digest in pins.items():
                assert paths[origin] != origin  # No fallback to original private files.
                if sha(Path(paths[origin])) != digest:
                    raise ValueError("fixture_owned_bytes_changed")
        return {
            **deepcopy(expected),
            "artifact_sha256": {
                actual: expected["artifact_sha256"][origin] for origin, actual in artifact_paths.items()
            },
            "source_sha256": {
                actual: expected["source_sha256"][origin] for origin, actual in source_paths.items()
            },
            "artifact_origins": dict(artifact_paths),
            "source_origins": dict(source_paths),
            "current_sources_verified": False,
            "historical_provenance_verified": True,
        }

    monkeypatch.setattr(transfer, "require_supplied_input_transfer_gate", validate)
    return SimpleNamespace(create=create, calls=calls, originals=originals)


def archive(history, gate, task="planet"):
    history.mkdir(exist_ok=True)
    return module.archive_supplied_input_transfer_gate(history, gate, task=task)


def load(history, link, gate, task="planet"):
    return module.load_archived_supplied_input_transfer_gate(
        module._Evidence(history), link, task=task, **module._expected(gate["identity"])
    )


def test_archive_is_once_per_task_digest_and_has_no_original_path_reads_on_replay(tmp_path, gate):
    history = tmp_path / "history"
    validated = gate.create()
    link = archive(history, validated)
    assert set(link) == {"path", "sha256"}
    assert link["path"] == f"frozen-transfer-gates/planet/{validated['report_sha256']}/record.json"
    record = read(history / link["path"])
    assert record["policy_observation_authorized"] is False
    assert record["current_sources_verified"] is False
    assert record["historical_provenance_verified"] is True
    assert len(record["artifacts"]) == 204
    assert len(record["sources"]) == 2
    assert archive(history, validated) == link
    for path in (*validated["artifact_sha256"], *validated["source_sha256"]):
        Path(path).unlink()  # Replay cannot consult any original leaf.
    result = load(history, link, validated)
    assert result["identity"] == validated["identity"]
    assert not result["current_sources_verified"] and result["historical_provenance_verified"]
    assert all(Path(p).is_relative_to(history) for p in gate.calls[-1]["artifacts"].values())
    assert all(Path(p).is_relative_to(history) for p in gate.calls[-1]["sources"].values())


@pytest.mark.parametrize(
    "mutation",
    ["artifact", "source", "record", "new_file", "missing", "symlink", "traversal", "source_fallback"],
)
def test_archive_rejects_mutation_replacement_unlisted_and_fallback_paths(tmp_path, gate, mutation):
    history = tmp_path / "history"
    validated = gate.create()
    link = archive(history, validated)
    path = history / link["path"]
    record = read(path)
    leaf = path.parent / next(iter(record["artifacts"].values()))["path"]
    if mutation == "source":
        leaf = path.parent / next(iter(record["sources"].values()))["path"]
        leaf.write_bytes(b"changed")
    elif mutation == "artifact":
        leaf.write_bytes(b"changed")
    elif mutation == "record":
        record["task_completed"] = True
        write(path, record)
        link["sha256"] = sha(path)
    elif mutation == "new_file":
        write(path.parent / "unexpected.json", {})
    elif mutation == "missing":
        leaf.unlink()
    elif mutation == "symlink":
        leaf.unlink()
        leaf.symlink_to(next(iter(validated["artifact_sha256"])))
    elif mutation == "traversal":
        next(iter(record["artifacts"].values()))["path"] = "../outside.json"
        write(path, record)
        link["sha256"] = sha(path)
    else:
        del record["sources"][next(iter(record["sources"]))]
        write(path, record)
        link["sha256"] = sha(path)
    with pytest.raises((BrowserSafetyStop, ValueError)):
        load(history, link, validated)


def test_different_task_archive_does_not_invalidate_existing_gate_book(tmp_path, gate):
    history = tmp_path / "history"
    first = gate.create("planet")
    link = archive(history, first)
    book = module._Evidence(history)
    module.load_archived_supplied_input_transfer_gate(
        book, link, task="planet", **module._expected(first["identity"])
    )
    archive(history, gate.create("temperature"), "temperature")
    book.unchanged()


def test_partial_archive_cannot_be_repaired_or_retried(tmp_path, gate):
    history = tmp_path / "history"
    validated = gate.create()
    root = history / "frozen-transfer-gates/planet" / validated["report_sha256"]
    root.mkdir(parents=True)
    write(root / "partial.json", {})
    with pytest.raises(BrowserSafetyStop):
        archive(history, validated)
    assert not (root / "record.json").exists()


@pytest.fixture
def provenance(case, gate):
    def create(task="planet", actual_class="white_dwarf"):
        item = case(actual_class)
        receipt = build(item)
        path, digest = store(item, receipt)
        validated = gate.create(task)
        link = archive(item.history, validated, task)
        identity = validated["identity"]
        source, pack, adapter = module._task(task)
        value = {
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "checkpoint_metadata_sha256": identity["parent_metadata_sha256"],
            "graph_hash": identity["graph_hash"],
            "knowledge_pack_hash": pack.checksum,
            "supplied_input_adapter_sha256": adapter["sha256"],
            "original_checkpoint_content_hash": identity["parent_content_hash"],
            "original_knowledge_pack_hash": source.BASE_PACK_HASH,
            "original_scope": source.LEGACY_SCOPE,
            "supplied_stellar_inputs": {"path": str(path.relative_to(item.history)), "sha256": digest},
            "supplied_input_transfer_gate": link,
            "supplied_star_class": actual_class,
            "classification_source": "supplied_not_learned",
            "optimizer_updates": 0,
        }
        report = {
            "scope": module.DERIVED_SUPPLIED_MODE if task == "planet" else module.TEMPERATURE_SUPPLIED_MODE,
            "provenance": value,
            "optimizer_updates": 0,
            "checkpoint_unchanged": True,
            "sources_unchanged": True,
            "task_completed": False,
            "saved": False,
            "submitted": False,
        }
        if task == "planet":
            report.update(
                planet_transport_verified=True,
                outcome="planet_derived_transport_verified",
                browser_acceptance_passed=False,
                assessment_performed=False,
            )
            initial = read(item.current / "observation.json")
            name = "native-copies"
        else:
            report.update(
                equilibrium_transport_verified=True,
                outcome="equilibrium_transport_verified",
                course_acceptance_passed=False,
                assessed=False,
            )
            initial = habitat_capture()
            initial["frames"][0]["accessibility"] = initial["frames"][0]["accessibility"].replace(
                "Jyremis", "Althinagon"
            )
            initial["frames"][0]["text"] = "ALTHINAGON\nOBSERVATIONS"
            name = "native-copy"
        directory = item.history / ("derived" if task == "planet" else "temperature")
        save_capture(directory / name / "initial", initial)
        write(directory / "report.json", report)
        item.report, item.transport_dir, item.task, item.receipt = report, directory, task, receipt
        return item

    return create


def validate(item, book=None):
    function = (
        module.validate_supplied_derived_provenance
        if item.task == "planet"
        else module.validate_supplied_temperature_provenance
    )
    return function(book or module._Evidence(item.history), item.transport_dir, item.report)


@pytest.mark.parametrize("task", ["planet", "temperature"])
@pytest.mark.parametrize("actual_class", module.NON_MAIN_CLASSES)
def test_new_provenance_retains_actual_class_and_exact_owned_receipt(provenance, task, actual_class):
    item = provenance(task, actual_class)
    book = module._Evidence(item.history)
    result = validate(item, book)
    assert result == {
        "star": "Althinagon",
        "supplied_star_class": actual_class,
        "receipt": item.receipt,
        "provenance": item.report["provenance"],
    }
    assert item.receipt["source_sha256"].items() <= book.hashes.items()
    assert any("frozen-transfer-gates" in p and "private/source" in p for p in book.hashes)
    assert not result["receipt"]["scientific_verified"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("supplied_star_class", "main_sequence"),
        ("supplied_star_class", "red_giant"),
        ("knowledge_pack_hash", "a" * 64),
        ("supplied_input_adapter_sha256", "a" * 64),
        ("original_checkpoint_content_hash", "a" * 64),
        ("original_knowledge_pack_hash", "a" * 64),
        ("original_scope", "wrong"),
        ("checkpoint_metadata_sha256", "a" * 64),
        ("checkpoint_sha256", "a" * 64),
        ("graph_hash", "a" * 64),
        ("optimizer_updates", False),
    ],
)
def test_changed_or_legacy_identity_cannot_claim_new_transport(provenance, key, value):
    item = provenance()
    item.report["provenance"][key] = value
    write(item.transport_dir / "report.json", item.report)
    with pytest.raises(BrowserSafetyStop):
        validate(item)


@pytest.mark.parametrize("task", ["planet", "temperature"])
def test_native_initial_capture_must_match_the_source_star(provenance, task):
    item = provenance(task)
    directory = item.transport_dir / ("native-copies" if task == "planet" else "native-copy") / "initial"
    value = read(directory / "observation.json")
    value["frames"][0]["accessibility"] = value["frames"][0]["accessibility"].replace("Althinagon", "Other")
    save_capture(directory, value)
    with pytest.raises(BrowserSafetyStop):
        validate(item)


def test_adopted_source_changes_stop_later_parent_book_check(provenance):
    item = provenance()
    book = module._Evidence(item.history)
    validate(item, book)
    write(item.color_dir / "manifest.json", {"changed": True})
    with pytest.raises(BrowserSafetyStop):
        book.unchanged()


def test_strict_gate_failure_is_never_replaced_by_report_pass_flag(provenance, monkeypatch):
    item = provenance()

    def reject(*args, **kwargs):
        raise ValueError("recorded_trajectory_invalid")

    monkeypatch.setattr(transfer, "require_supplied_input_transfer_gate", reject)
    with pytest.raises(BrowserSafetyStop):
        validate(item)
