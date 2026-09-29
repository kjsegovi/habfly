"""Offline source dispatch/import gates; synthetic receipts are not live proof.

The declared LIVE-contract seam isolates importer integration. Separate tests
use the real pager producer/loader with only native transport injected, without
launching Chromium, and never promote the offline-only consistency result.
"""
# ruff: noqa: F811

import socket
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_project_paginated_inventory_steps import complete, subject  # noqa: F401
from test_project_evidence import inventory, read, sha, write
from test_project_evidence import workflow as no_workflow
from test_project_paginated_inventory import capture
from test_project_positive_evidence import workflow as positive_workflow
from test_project_terrestrial_evidence import workflow as terrestrial_workflow

import habfly.project_evidence as no_module
import habfly.project_inventory_evidence as collection_module
import habfly.project_inventory_source as module
import habfly.project_positive_evidence as positive_module
import habfly.project_terrestrial_evidence as terrestrial_module
from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import rows_hash
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_probe import save_probe
from habfly.project_progress import Collected, ProgressError, ProjectJournal

WORKFLOWS = {
    "no": (no_module, no_workflow, no_module.import_verified_no_planet),
    "positive": (positive_module, positive_workflow, positive_module.import_verified_positive_planet),
    "terrestrial": (
        terrestrial_module,
        terrestrial_workflow,
        terrestrial_module.import_verified_terrestrial,
    ),
}


@pytest.fixture
def upstream(monkeypatch):
    """Declared workflow-source seam, not a claim about actual browser work."""
    registry = {}

    def validated(book, **directories):
        hashes, bundle = registry[str(directories["numeric_dir"])]
        for name in hashes:
            book.clean((book.history / name).parent)
            book.read(book.history / name)
        return deepcopy(bundle)

    for importer, _, _ in WORKFLOWS.values():
        monkeypatch.setattr(importer, "_load_sources", validated)
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("Network forbidden"))
    return registry


def declared_live(history, monkeypatch, *, names=None):
    """Construct only the explicitly injected loader contract, not native proof."""
    names = names or [f"Star{i:02d}" for i in range(11)]
    legacy = inventory(history, names, "row-material")
    _, parsed = no_module._inventory(_Evidence(history), legacy)
    directory = history / "live-inventory"
    anchor_dir = directory / "pages" / "anchor" / "after"
    anchor = capture(names[:10], 1, len(names))
    save_probe(anchor, anchor_dir)
    leaf = directory / "pages" / "second" / "page-source.json"
    write(leaf, {"declared_injected_contract": True, "rows": parsed[10:]})
    files = [anchor_dir / "observation.json", anchor_dir / "manifest.json", leaf]
    sources = {str(p.relative_to(history)): sha(p) for p in files}
    receipt = {
        "schema_version": 1,
        "mode": module.LIVE_MODE,
        "authority": "guarded_native_pager_visible_readback",
        "collection_count_verified": True,
        "live_pagination_verified": True,
        "complete_collection_traversal_verified": True,
        "complete_visible_list_verified": False,
        "total_collected": len(names),
        "rows": [{"name": n} for n in names],
        "whole_collection_sha256": module._whole(parsed),
        "collection_digest_domain": "recorded_stellar_collection_v1",
        "anchor_capture": str(anchor_dir.relative_to(history)),
        "anchor_capture_sha256": sha(anchor_dir / "observation.json"),
        "visible_assessment_anchor_sha256": rows_hash(anchor["frames"][0]["text"]),
        "source_sha256": sources,
        "validated_directories": sorted({str(p.parent.relative_to(history)) for p in files}),
    }
    write(directory / "confirmed.json", receipt)

    def injected(run_history, source, expected_sha256):
        assert run_history == history and source == directory
        assert expected_sha256 == sha(directory / "confirmed.json")
        return read(directory / "confirmed.json"), deepcopy(parsed), deepcopy(anchor)

    monkeypatch.setattr(module, "_load_live", injected)
    return SimpleNamespace(directory=directory, anchor=anchor_dir, leaf=leaf, rows=parsed, names=names)


def journal_at(history):
    return ProjectJournal(history, project_id="habworlds", attempt_id="synthetic-pager-import").create()


def import_case(kind, history, inv, registry):
    journal = journal_at(history)
    if kind == "collection":
        return journal, lambda: collection_module.import_verified_inventory(journal, history, inv)
    _, builder, importer = WORKFLOWS[kind]
    work = builder(history, "Star10", "current", registry)
    return journal, lambda: importer(journal, history, inv, work)


def test_legacy_accessor_keeps_exact_validator_contract_and_actual_after(tmp_path):
    directory = inventory(tmp_path, ["Dulat", "Beta"], "legacy")
    book = _Evidence(tmp_path)
    source = module.load_inventory_source(book, directory, expected_sha256=sha(directory / "confirmed.json"))
    assert (source.receipt, source.rows) == no_module._inventory(_Evidence(tmp_path), directory)
    assert source.anchor_dir == directory / "after" and source.anchor == read(
        directory / "after/observation.json"
    )
    assert source.kind == "legacy_single_page" and source.binding_fields() == {}
    assert not any("inventory_evidence" in name for name in book.hashes)
    with pytest.raises(ProgressError, match="receipt_hash_mismatch"):
        module.load_inventory_source(_Evidence(tmp_path), directory, expected_sha256="0" * 64)


@pytest.mark.parametrize("kind", list(WORKFLOWS))
def test_legacy_binding_bytes_match_original_validator_path(tmp_path, upstream, monkeypatch, kind):
    importer_module, builder, importer = WORKFLOWS[kind]
    bindings, journals = [], []
    for index in range(2):
        history = tmp_path / str(index)
        history.mkdir()
        journal = journal_at(history)
        inv = inventory(history, ["Star10"], "same")
        work = builder(history, "Star10", "same", upstream)
        if index:

            def original_path(book, directory):
                receipt, rows = no_module._inventory(book, directory)
                return SimpleNamespace(receipt=receipt, rows=rows, binding_fields=dict)

            monkeypatch.setattr(importer_module, "load_inventory_source", original_path)
        importer(journal, history, inv, work)
        bindings.append(next(history.glob("project-evidence-*.json")).read_bytes())
        journals.append(journal.path.read_bytes())
    assert bindings[0] == bindings[1] and journals[0] == journals[1]


def test_declared_live_contract_adopts_every_source_and_actual_anchor(tmp_path, monkeypatch):
    fixture = declared_live(tmp_path, monkeypatch)
    book = _Evidence(tmp_path)
    source = module.load_inventory_source(book, fixture.directory)
    receipt = read(fixture.directory / "confirmed.json")
    assert source.rows == fixture.rows and source.kind == "live_paginated"
    assert source.anchor_dir == fixture.anchor and not (fixture.directory / "after").exists()
    assert source.anchor == read(fixture.anchor / "observation.json")
    assert source.whole_collection_sha256 != sha(fixture.directory / "confirmed.json")
    assert set(receipt["source_sha256"]) <= book.hashes.keys()
    assert {tmp_path / p for p in receipt["validated_directories"]} <= book.clean_directories
    expected = {
        "inventory_evidence_version": 1,
        "inventory_kind": "live_paginated",
        "whole_collection_sha256": receipt["whole_collection_sha256"],
        "inventory_anchor": {
            "capture_dir": receipt["anchor_capture"],
            "observation_sha256": receipt["anchor_capture_sha256"],
            "manifest_sha256": sha(fixture.anchor / "manifest.json"),
            "visible_rows_sha256": receipt["visible_assessment_anchor_sha256"],
        },
    }
    assert source.binding_fields() == expected
    copy = source.binding_fields()
    copy["inventory_anchor"]["capture_dir"] = "changed"
    assert source.binding_fields() == expected


@pytest.mark.parametrize("kind", [*WORKFLOWS, "collection"])
def test_declared_live_all_importers_use_union_not_anchor_and_are_idempotent(
    tmp_path, monkeypatch, upstream, kind
):
    fixture = declared_live(tmp_path, monkeypatch)
    journal, ingest = import_case(kind, tmp_path, fixture.directory, upstream)
    result = ingest()
    assert result["progress"]["collected"] == 11
    assert result["progress"]["verified"] == (0 if kind == "collection" else 1)
    assert not result["progress"]["project_completed"] and not result["progress"]["submitted"]
    assert not result["progress"]["score_transfer_verified"] and not any(
        result["progress"]["assessment"].values()
    )
    progress = journal.load()
    collected = [r.payload for r in progress.records if isinstance(r.payload, Collected)]
    assert len(collected) == 11 and {r.source_sha256 for r in collected} == {
        sha(fixture.directory / "confirmed.json")
    }
    state = progress.reduce()
    assert not state.reservations and not state.receipts
    fields = module.load_inventory_source(_Evidence(tmp_path), fixture.directory).binding_fields()
    assert {k: result[k] for k in fields} == fields
    bindings = list(tmp_path.glob("project-evidence-*.json"))
    assert len(bindings) == (0 if kind == "collection" else 1)
    if bindings:
        binding = read(bindings[0])
        assert binding["schema_version"] == progress.header()["schema_version"]
        assert {k: binding[k] for k in fields} == fields
        star = state.stars[result["star_id"]]
        assert star.name == "Star10" and star.task_completed
        assert star.stellar_numeric.decision.provenance == "learned_prediction"
        assert star.stellar_classification.decision.provenance == "reference_prediction"
        assert not star.planet.report()["scientific_verified"]
        if kind != "no":
            assert (
                set(read(fixture.directory / "confirmed.json")["source_sha256"])
                <= binding["validated_artifact_sha256"].keys()
            )
    journal_bytes = journal.path.read_bytes()
    binding_bytes = [p.read_bytes() for p in bindings]
    assert ingest()["idempotent"] and journal.path.read_bytes() == journal_bytes
    assert [p.read_bytes() for p in bindings] == binding_bytes


@pytest.mark.parametrize("kind", [*WORKFLOWS, "collection"])
@pytest.mark.parametrize("mutation", ["leaf", "stopped"])
def test_late_paged_source_or_stop_is_rechecked_before_journal_mutation(
    tmp_path, monkeypatch, upstream, kind, mutation
):
    fixture = declared_live(tmp_path, monkeypatch)
    journal, ingest = import_case(kind, tmp_path, fixture.directory, upstream)
    before = journal.path.read_bytes()
    unchanged = _Evidence.unchanged

    def changing(book):
        book.test_rechecks = getattr(book, "test_rechecks", 0) + 1
        if book.test_rechecks == 2:
            if mutation == "leaf":
                fixture.leaf.write_text("changed late")
            else:
                write(fixture.leaf.parent / "stopped.json", {"reason": "late fixture stop"})
        return unchanged(book)

    monkeypatch.setattr(_Evidence, "unchanged", changing)
    with pytest.raises((ProgressError, BrowserSafetyStop)):
        ingest()
    assert journal.path.read_bytes() == before and not list(tmp_path.glob("project-evidence-*.json"))


@pytest.mark.parametrize(
    "mutation",
    [
        "whole",
        "row_name",
        "partial_flag",
        "source",
        "anchor_hash",
        "dirs_missing",
        "dirs_invalid",
        "dirs_incomplete",
    ],
)
def test_declared_contract_mismatch_fails_closed(tmp_path, monkeypatch, mutation):
    fixture = declared_live(tmp_path, monkeypatch)
    path = fixture.directory / "confirmed.json"
    value = read(path)
    if mutation == "whole":
        value["whole_collection_sha256"] = "f" * 64
    elif mutation == "row_name":
        value["rows"][-1]["name"] = "Other"
    elif mutation == "partial_flag":
        value["complete_visible_list_verified"] = True
    elif mutation == "source":
        value["source_sha256"][str(fixture.leaf.relative_to(tmp_path))] = "f" * 64
    elif mutation == "anchor_hash":
        value["anchor_capture_sha256"] = "f" * 64
    elif mutation == "dirs_missing":
        value.pop("validated_directories")
    elif mutation == "dirs_invalid":
        value["validated_directories"] = [{}]
    else:
        value["validated_directories"] = [str(fixture.anchor.relative_to(tmp_path))]
    write(path, value)
    with pytest.raises((ProgressError, BrowserSafetyStop)):
        module.load_inventory_source(_Evidence(tmp_path), fixture.directory)


@pytest.mark.parametrize("mode", ["recorded_paginated_collection_consistency", "offline", None])
def test_offline_receipts_never_reach_live_validator(tmp_path, monkeypatch, mode):
    directory = tmp_path / "offline"
    write(directory / "confirmed.json", {"mode": mode, "collection_count_verified": True})
    monkeypatch.setattr(module, "_load_live", lambda *_a: pytest.fail("Offline mode dispatched as live"))
    with pytest.raises(ProgressError, match="unsupported_inventory"):
        module.load_inventory_source(_Evidence(tmp_path), directory)


@pytest.mark.parametrize("total", [11, 21, 30])
def test_real_live_loader_and_accessor_share_actual_anchor_and_sources(subject, total):
    subject.state.names = [f"Star{i:02d}" for i in range(total)]
    owner = complete(subject)
    book = _Evidence(subject.root)
    source = module.load_inventory_source(
        book, owner.output, expected_sha256=sha(owner.output / "confirmed.json")
    )
    assert source.receipt == owner.report and len(source.rows) == total
    assert source.anchor_dir == owner.output / "anchor-01/after"
    assert source.anchor == subject.current() or source.anchor["frames"] == subject.current()["frames"]
    assert len(source.rows) > 10 and len(source.receipt["source_sha256"]) < 500
    assert set(source.receipt["source_sha256"]) <= book.hashes.keys()
    assert not (owner.output / "after").exists()


@pytest.mark.parametrize("kind", [*WORKFLOWS, "collection"])
def test_real_pager_loader_integrates_each_importer_without_inventory_validator_seam(subject, upstream, kind):
    subject.state.names = [f"Star{i:02d}" for i in range(11)]
    owner = complete(subject)
    journal, ingest = import_case(kind, subject.root, owner.output, upstream)
    clicks_before = list(subject.state.clicks)
    result = ingest()
    assert result["progress"]["collected"] == 11 and result["progress"]["verified"] == (kind != "collection")
    assert result["inventory_anchor"]["capture_dir"] == "inventory/anchor-01/after"
    assert result["inventory_sha256"] == sha(owner.output / "confirmed.json")
    assert not result["progress"]["project_completed"] and result["browser_actions"] == 0
    assert subject.state.clicks == clicks_before
    raw = journal.path.read_bytes()
    assert ingest()["idempotent"] and journal.path.read_bytes() == raw


def test_real_loader_rejects_spoofed_live_flags_on_offline_consistency(subject):
    owner = complete(subject)
    fake = subject.root / "spoofed"
    value = read(owner.output / "consistency.json")
    value.update(mode=module.LIVE_MODE, collection_count_verified=True, live_pagination_verified=True)
    write(fake / "confirmed.json", value)
    with pytest.raises((ProgressError, BrowserSafetyStop, ValueError)):
        module.load_inventory_source(_Evidence(subject.root), fake)


def test_real_source_tampering_and_new_stop_are_not_valid_live_inventory(subject):
    owner = complete(subject)
    source = module.load_inventory_source(_Evidence(subject.root), owner.output)
    target = subject.root / next(p for p in source.receipt["source_sha256"] if "native" in p)
    write(target.parent / "stopped.json", {"reason": "postcapture fixture failure"})
    with pytest.raises((ProgressError, BrowserSafetyStop, ValueError)):
        module.load_inventory_source(_Evidence(subject.root), owner.output)
