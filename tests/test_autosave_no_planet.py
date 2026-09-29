"""Offline autosave authority/No readback tests; source/native seams declared.

The synthetic source seam does not prove a real scientific choice. Current
capture parsing, exact receipt validation, mutation guards and journal importer
stay active; the separately gated native fixture checks genuine choice chains.
"""

# ruff: noqa: F811 - imported fixtures

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_star_session import advance_to, complete, create
from test_browser_star_session import rig as star_rig  # noqa: F401
from test_project_evidence import current_planet, ingest, read, sha, sources, write  # noqa: F401

import habfly.browser_no_planet_readback as module
import habfly.browser_star_session as star_module
from habfly.browser import BrowserSafetyStop
from habfly.browser_autosave import (
    ACKNOWLEDGEMENT_SOURCE,
    MODE,
    autosave_flags,
    autosave_manifest,
    autosave_workflow_flags,
    validate_autosave_receipt,
)
from habfly.browser_no_planet_workflow import _Evidence, _planet
from habfly.browser_numeric import screen_identity
from habfly.browser_probe import save_probe
from habfly.project_progress import ProgressError


def valid_receipt():
    return {
        "schema_version": 1,
        **autosave_flags(),
        "branch": "no_planet",
        "star": "Alpha",
        "output": "current/readback",
        "source_sha256": {"choice/confirmed.json": "a" * 64},
        "before_sha256": "b" * 64,
        "after_sha256": "c" * 64,
    }


def validate(receipt):
    return validate_autosave_receipt(receipt, branch="no_planet", star="ALPHA", output="current/readback")


def test_common_receipt_and_manifest_are_deterministic_without_persistence():
    assert validate(valid_receipt()) == valid_receipt()
    assert autosave_manifest() == autosave_manifest()
    assert autosave_manifest()["receipt_flags"] == autosave_flags()
    assert autosave_workflow_flags()["persistence_verified"] is False
    assert "task_completed" not in autosave_workflow_flags()


@pytest.mark.parametrize(
    "key,value",
    [
        ("save_click_delivered", True),
        ("save_acknowledgement_verified", True),
        ("persistence_verified", True),
        ("scientific_verified", True),
        ("task_completed", True),
        ("visible_readback_verified", 1),
        ("save_click_delivered", 0),
        ("browser_actions", False),
        ("schema_version", True),
        ("star", "Beta"),
        ("output", "other"),
        ("branch", "positive"),
        ("authority", "persistence"),
        ("source_sha256", {}),
        ("source_sha256", {"../escape": "a" * 64}),
        ("source_sha256", {"/absolute": "a" * 64}),
        ("source_sha256", {"x": "bad"}),
        ("after_sha256", "A" * 64),
    ],
)
def test_common_authority_rejects_aliases_and_changed_scope(key, value):
    with pytest.raises(BrowserSafetyStop):
        validate({**valid_receipt(), key: value})


def test_common_unknown_fields_do_not_smuggle_acknowledgement():
    with pytest.raises(BrowserSafetyStop):
        validate({**valid_receipt(), "acknowledgement": "Data saved"})


@pytest.fixture
def readback(tmp_path, monkeypatch):
    report = current_planet("Althinagon")
    mapping = _planet(report)
    calls, state = [], {"report": report, "cancelled": False}
    path = tmp_path / "choice/confirmed.json"
    write(path, {"synthetic_source_seam": True})

    def source(book, source_path, expected):
        raw = book.read(source_path)
        assert hashlib.sha256(raw).hexdigest() == expected
        return {"star": "Althinagon", "painted_class_after": None}, deepcopy(report), deepcopy(mapping)

    class Session:
        def __init__(self, page, config, output, **kwargs):
            assert kwargs["_allow_no_planet"] is True
            calls.append("open_readonly")
            save_probe(deepcopy(report), output / "initial")
            write(output / "scope.json", {"synthetic_native_session": True})

        def current(self):
            calls.append("current")
            captured = deepcopy(state["report"])
            return captured, _planet(captured), {"selected": None}, None

        def close(self):
            calls.append("close")

    monkeypatch.setattr(module, "_choice_sources", source)
    monkeypatch.setattr(module, "PlanetNumericSession", Session)
    monkeypatch.setattr(star_module, "save_no_planet_work", lambda *a, **k: pytest.fail("Save forbidden"))
    return SimpleNamespace(root=tmp_path, source=path, state=state, calls=calls)


def produce(rig, **kwargs):
    return module.readback_no_planet_work(
        object(),
        object(),
        rig.root / "readback",
        run_history=rig.root,
        choice_path=rig.source,
        choice_sha256=sha(rig.source),
        **kwargs,
    )


def load(rig):
    return module.load_no_planet_readback(
        _Evidence(rig.root), rig.root / "readback", choice_dir=rig.source.parent, expected_star="Althinagon"
    )


def test_readback_has_no_save_wait_or_fabricated_write_and_reloads(readback):
    receipt = produce(readback)
    assert readback.calls == ["open_readonly", "current", "current", "close"]
    assert all(
        receipt[key] is False
        for key in (
            "save_click_delivered",
            "save_acknowledgement_verified",
            "persistence_verified",
            "task_completed",
        )
    )
    bundle = load(readback)
    assert screen_identity(bundle["save_capture"]) == receipt["after_sha256"]
    assert not any(
        p.name in {"reserved.json", "dispatch.json", "acknowledgement.json"} for p in readback.root.rglob("*")
    )
    before = {str(p.relative_to(readback.root)): sha(p) for p in readback.root.rglob("*") if p.is_file()}
    assert load(readback) == bundle
    assert before == {
        str(p.relative_to(readback.root)): sha(p) for p in readback.root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "change", ["ack", "dispatch", "stopped", "source", "capture", "symlink", "authority", "source_hash"]
)
def test_readback_mutations_rejected(readback, change):
    produce(readback)
    directory = readback.root / "readback"
    if change in {"ack", "dispatch", "stopped"}:
        write(
            directory
            / {"ack": "acknowledgement.json", "dispatch": "dispatch.json", "stopped": "stopped.json"}[change],
            {},
        )
    elif change == "source":
        readback.source.write_bytes(b"{}")
    elif change == "capture":
        path = directory / "after/observation.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "symlink":
        path = directory / "after/observation.json"
        path.rename(directory / "actual.json")
        path.symlink_to(directory / "actual.json")
    else:
        path = directory / "confirmed.json"
        value = read(path)
        if change == "authority":
            value["persistence_verified"] = True
        else:
            value["source_sha256"] = {"choice/confirmed.json": "a" * 64}
        write(path, value)
    with pytest.raises((BrowserSafetyStop, AssertionError)):
        load(readback)


def test_prior_explicit_claim_cannot_be_reinterpreted(readback):
    key = hashlib.sha256(b"althinagon").hexdigest()
    write(readback.root / "no-planet-save-reservations" / f"{key}.json", {})
    with pytest.raises(BrowserSafetyStop, match="prior_explicit_save_attempt"):
        produce(readback)
    assert not readback.calls and not (readback.root / "readback").exists()


def test_cancelled_readback_never_constructs_browser_or_save(readback):
    with pytest.raises(BrowserSafetyStop, match="cancelled"):
        produce(readback, cancelled=lambda: True)
    assert not readback.calls
    assert not read(readback.root / "readback/stopped.json")["task_completed"]


@pytest.mark.parametrize("value", [None, True, 1, [], {}, "AUTO", ""])
def test_star_strategy_is_explicit_strict_before_output(star_rig, value):
    with pytest.raises(ValueError):
        create(star_rig, save_strategy=value)
    assert not star_rig.calls and not (star_rig.root / "run").exists()


def test_autosave_star_routes_readback_only_then_original_workflow(star_rig, monkeypatch):
    monkeypatch.setattr(
        module, "readback_no_planet_work", lambda *a, **k: star_rig.calls.append(("readback", "No"))
    )
    monkeypatch.setattr(star_module, "save_no_planet_work", lambda *a, **k: pytest.fail("Save forbidden"))
    session = create(star_rig, save_strategy="autosave")
    complete(session)
    assert session.phase == "verified_no_planet"
    assert ("readback", "No") in star_rig.calls and ("verify", "No") in star_rig.calls
    assert session.state()["persistence_verified"] is False


@pytest.mark.parametrize("change", ["attribute", "scope", "disk", "source"])
def test_strategy_mutation_before_readback_blocks_followup(star_rig, monkeypatch, change):
    session = create(star_rig, save_strategy="autosave")
    advance_to(session, "save_no_planet")
    calls = list(star_rig.calls)
    if change == "attribute":
        session.save_strategy = "explicit"
    elif change == "scope":
        session.scope["save_strategy"] = "explicit"
    elif change == "disk":
        path = session.output / "scope.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        original = Path.read_bytes
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda p: original(p) + b" " if p in session._autosave_sources else original(p),
        )
    session.advance()
    assert session.phase == "stopped" and star_rig.calls == calls


def autosave_import(sources):
    _, history, _, work, registry, _ = sources
    hashes, bundle = registry[str(history / "source-first/numeric")]
    bundle.update(
        save_strategy="autosave", save_click_delivered=False, acknowledgement_source=ACKNOWLEDGEMENT_SOURCE
    )
    path = history / "source-first/save/confirmed.json"
    value = read(path)
    value.update(mode=MODE, branch="no_planet")
    write(path, value)
    hashes[str(path.relative_to(history))] = sha(path)
    receipt = read(work / "confirmed.json")
    receipt.update(
        **autosave_workflow_flags(),
        source_sha256=hashes,
        source_save_click_delivered=False,
        save_acknowledgement_verified=False,
        save_acknowledgement_source=ACKNOWLEDGEMENT_SOURCE,
    )
    write(work / "confirmed.json", receipt)


def test_canonical_import_is_idempotent_visible_completion_without_save_receipt(sources):
    autosave_import(sources)
    journal = sources[0]
    result = ingest(sources)
    assert result["progress"]["verified"] == 1
    progress = journal.load().reduce()
    assert not progress.reservations and not progress.receipts and not progress.report()["submitted"]
    before = journal.path.read_bytes()
    ingest(sources)
    assert journal.path.read_bytes() == before


def test_canonical_import_preserves_readback_tree_closure_through_final_precommit(sources, monkeypatch):
    """Declared source seam closes a real fixture tree; importer must retain it."""
    import habfly.project_evidence as evidence

    autosave_import(sources)
    journal, history, _, _, _, _ = sources
    source_directory = history / "source-first/save"
    original_loader, original_plan = evidence._load_sources, evidence._plan

    def load_with_closed_tree(book, **directories):
        result = original_loader(book, **directories)
        book.closed_trees[source_directory] = book.tree(source_directory)
        return result

    def mutate_after_validated_source(*args, **kwargs):
        result = original_plan(*args, **kwargs)
        write(source_directory / "dispatch.json", {"synthetic_late_mutation": True})
        return result

    monkeypatch.setattr(evidence, "_load_sources", load_with_closed_tree)
    monkeypatch.setattr(evidence, "_plan", mutate_after_validated_source)
    before = journal.path.read_bytes()
    with pytest.raises(BrowserSafetyStop, match="evidence_tree_changed"):
        ingest(sources)
    assert journal.path.read_bytes() == before
    assert not list(history.glob("project-evidence-*.json"))


@pytest.mark.parametrize("change", ["ack", "persistence", "missing_flag", "legacy_source"])
def test_canonical_import_rejects_autosave_authority_upgrade_without_writes(sources, change):
    autosave_import(sources)
    journal, history, _, work, registry, _ = sources
    path = work / "confirmed.json"
    value = read(path)
    if change == "ack":
        value["save_acknowledgement_verified"] = True
    elif change == "persistence":
        value["persistence_verified"] = True
    elif change == "missing_flag":
        del value["autosave_readback_mode"]
    else:
        registry[str(history / "source-first/numeric")][1].pop("save_strategy")
    write(path, value)
    before = journal.path.read_bytes()
    with pytest.raises((BrowserSafetyStop, ProgressError)):
        ingest(sources)
    assert journal.path.read_bytes() == before
