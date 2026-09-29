"""Actual paginated receipt/accessor/import composition with injected native I/O.

The old synthetic scientific-source seam is explicit; all new inventory page,
hash, anchor, journal and completed-owner validators execute unchanged offline.
No fixture is claimed as real HabWorlds completion.
"""
# ruff: noqa: F811

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_numeric import config as probe_config
from test_browser_project_paginated_inventory_steps import complete, subject  # noqa: F401
from test_browser_project_steps_inventory import inventory_rig, rig, to_inventory  # noqa: F401
from test_project_evidence import read, sha, workflow, write

import habfly.browser_project_campaign_steps as campaign
import habfly.browser_project_inventory_steps as inventory_steps
import habfly.browser_project_next_star_steps as transition
import habfly.project_evidence as evidence
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_probe import save_probe
from habfly.browser_project_navigation import project_view
from habfly.contracts import RuntimeEvent
from habfly.project_inventory_source import load_inventory_source
from habfly.project_progress import Collected, ProjectJournal


@pytest.fixture
def paged_sources(subject, monkeypatch):
    subject.state.names = [f"Star{i:02d}" for i in range(11)]
    pager = complete(subject)
    registry = {}

    def validated(book, **directories):
        hashes, bundle = registry[str(directories["numeric_dir"])]
        for name in hashes:
            book.clean((book.history / name).parent)
            book.read(book.history / name)
        return deepcopy(bundle)

    monkeypatch.setattr(evidence, "_load_sources", validated)
    journal = ProjectJournal(subject.root, project_id="habworlds", attempt_id="injected-pages").create()
    for index, name in enumerate(subject.state.names):
        source = workflow(subject.root, name, f"star-{index}", registry)
        imported = evidence.import_verified_no_planet(journal, subject.root, pager.output, source)
    owner = subject.root / "completed-owner"
    report = {
        "mode": "bounded_single_star_project_runtime",
        "status": "completed",
        "phase": "verified_no_planet",
        "star": subject.state.names[-1],
        "finished": True,
        "task_completed": True,
        "project_completed": False,
        "failure_reason": None,
        "event_forwarding_failed": False,
        "project_progress": imported["progress"],
    }
    write(owner / "report.json", report)
    write(owner / "workflow-import.json", imported)
    event = RuntimeEvent(
        event="episode_summary",
        run_id="injected-owned-pages",
        sequence=0,
        payload={**report, "completed": True},
    )
    (owner / "events.jsonl").write_text(event.model_dump_json() + "\n")
    binding = subject.root / (
        "project-evidence-" + evidence._sha((journal.path.name + imported["star_id"]).encode()) + ".json"
    )
    return SimpleNamespace(
        subject=subject,
        pager=pager,
        journal=journal,
        owner=owner,
        imported=imported,
        binding=binding,
        workflow=source,
    )


def test_actual_paged_anchor_not_synthetic_full_list_feeds_both_consumers(paged_sources):
    item = paged_sources
    history, names = item.subject.root, item.subject.state.names
    source = load_inventory_source(_Evidence(history), item.pager.output)
    assert source.kind == "live_paginated" and len(source.rows) == 11
    assert not (item.pager.output / "after").exists()
    assert "viewing 1-10 of 11" in source.anchor["frames"][0]["text"]
    assert item.imported["whole_collection_sha256"] == source.whole_collection_sha256
    assert (
        source.whole_collection_sha256 != source.binding_fields()["inventory_anchor"]["visible_rows_sha256"]
    )
    journal_before = item.journal.path.read_bytes()
    for consumer in ("transition", "campaign"):
        book = _Evidence(history)
        if consumer == "transition":
            state, previous, anchor = transition._completed_owner(book, item.journal, item.owner, names)
            assert previous.name == names[-1]
        else:
            state, anchor, sources = campaign._verified_owner(
                book, item.journal, item.owner, names, names[-1]
            )
            assert all(sources[key] == value for key, value in source.binding_fields().items())
        assert state.report()["verified"] == 11 and not state.report()["project_completed"]
        assert anchor == source.anchor
        assert str(source.anchor_dir.relative_to(history) / "observation.json") in book.hashes
        assert str(source.anchor_dir.relative_to(history) / "manifest.json") in book.hashes
        assert str(item.pager.output.relative_to(history) / "confirmed.json") in book.hashes
        book.unchanged()
    assert item.journal.path.read_bytes() == journal_before


@pytest.mark.parametrize(
    "target,key",
    [
        ("binding", "whole_collection_sha256"),
        ("binding", "inventory_anchor"),
        ("import", "whole_collection_sha256"),
        ("import", "inventory_anchor"),
    ],
)
def test_consumer_rejects_missing_or_rebound_live_metadata(paged_sources, target, key):
    item = paged_sources
    path = item.binding if target == "binding" else item.owner / "workflow-import.json"
    value = read(path)
    value.pop(key)
    write(path, value)
    for consumer in ("transition", "campaign"):
        with pytest.raises(Exception, match="inventory_binding_changed|import_mismatch"):
            if consumer == "transition":
                transition._completed_owner(
                    _Evidence(item.subject.root), item.journal, item.owner, item.subject.state.names
                )
            else:
                campaign._verified_owner(
                    _Evidence(item.subject.root),
                    item.journal,
                    item.owner,
                    item.subject.state.names,
                    item.subject.state.names[-1],
                )


@pytest.mark.parametrize("which", ["anchor", "forward-page", "transition"])
def test_actual_page_sources_remain_pinned_after_consumer_validation(paged_sources, which):
    item = paged_sources
    book = _Evidence(item.subject.root)
    transition._completed_owner(book, item.journal, item.owner, item.subject.state.names)
    path = (
        item.pager.output
        / {
            "anchor": "anchor-01/after/observation.json",
            "forward-page": "forward-02/after/observation.json",
            "transition": "transitions/01/confirmed.json",
        }[which]
    )
    value = read(path)
    value["injected_change"] = True
    write(path, value)
    with pytest.raises(BrowserSafetyStop):
        book.unchanged()


def test_single_owner_opts_into_live_only_after_ten_existing_stars(inventory_rig):
    fixture = inventory_rig.rig
    for index in range(10):
        fixture.journal.append(
            Collected(star_id=f"previous-{index}", name=f"Star{index:02d}", source_sha256="a" * 64)
        )
    owner = to_inventory(fixture)
    owner.step()
    child = inventory_rig.children[-1]
    assert child.options["inventory_mode"] == "live_paginated"
    assert child.options["max_seconds"] == 420
    assert len(child.options["expected_stars"]) == 11
    assert owner.phase == "inventory_active" and child.steps == 0
    assert not owner.state()["project_completed"]
    owner.abort()


def test_single_page_owner_keeps_original_constructor_contract(inventory_rig):
    owner = to_inventory(inventory_rig.rig)
    owner.step()
    child = inventory_rig.children[-1]
    assert "inventory_mode" not in child.options
    assert child.options["max_seconds"] == 180
    owner.abort()


@pytest.mark.parametrize("start_page", [0, 1])
def test_real_pager_and_loader_inside_cooperative_wrapper(subject, monkeypatch, start_page):
    """Real child/loader, injected page methods only; not a native browser test."""
    subject.state.names = [f"Star{i:02d}" for i in range(11)]
    subject.state.index = start_page
    source = workflow(subject.root, "Star10", "wrapper", {})
    subject.page.current = read(source / "planet-readback/verified/observation.json")
    listeners = {}

    def listen(kind, callback):
        listeners.setdefault(kind, []).append(callback)

    def remove(kind, callback):
        listeners[kind].remove(callback)

    # Real EventEmitter supports multiple parent/child listeners; fixture's
    # original one-callback map is intentionally upgraded for this composition.
    subject.page.on = listen
    subject.page.remove_listener = remove
    subject.page.context.on = lambda kind, callback: listen("context-" + kind, callback)
    subject.page.context.remove_listener = lambda kind, callback: remove("context-" + kind, callback)
    monkeypatch.setattr(inventory_steps, "inspect_page", lambda *_: deepcopy(subject.page.current))

    def navigate(_page, _config, output, destination, expected_star=None):
        assert destination == "list" and expected_star == "Star10"
        before = deepcopy(subject.page.current)
        save_probe(before, output / "before")
        save_probe(before, output / "pre-click")
        intent = {"mode": "bounded_project_navigation", "max_clicks": 1, "automatic_retry": False}
        write(output / "reserved.json", intent)
        subject.page.current = subject.current()
        save_probe(subject.page.current, output / "after")
        receipt = {
            **intent,
            "from": project_view(before),
            "to": project_view(subject.page.current),
            "destination": destination,
            "destination_verified": True,
            "navigation_clicks": 1,
            "task_completed": False,
            **dict.fromkeys(
                (
                    "answer_writes",
                    "collection_clicks",
                    "save_clicks",
                    "assessment_clicks",
                    "submission_clicks",
                ),
                0,
            ),
        }
        write(output / "confirmed.json", receipt)
        return receipt

    monkeypatch.setattr(inventory_steps, "navigate_project", navigate)
    native_factory = inventory_steps.PaginatedInventorySteps

    def factory(*args, **kwargs):
        child = native_factory(*args, **kwargs)
        subject.state.owner = child
        return child

    monkeypatch.setattr(inventory_steps, "PaginatedInventorySteps", factory)
    owner = inventory_steps.ProjectInventorySteps(
        subject.page,
        probe_config(),
        subject.root / "collection",
        run_history=subject.root,
        workflow_dir=source,
        workflow_sha256=sha(source / "confirmed.json"),
        expected_star="Star10",
        expected_stars=subject.state.names,
        inventory_mode="live_paginated",
        max_seconds=420,
    )
    assert subject.state.reads == 0 and subject.state.clicks == []
    for _ in range(11):
        if owner.finished:
            break
        clicks = len(subject.state.clicks)
        owner.advance()
        assert len(subject.state.clicks) - clicks <= 1
    if start_page:
        assert owner.status == "stopped"
        assert owner.failure == "paginated_inventory_live_unexpected_page_transition"
        assert subject.state.clicks == []
    else:
        assert owner.status == "completed", owner.state()
        assert owner.advances == 7
        assert subject.state.clicks == [1, 0] and subject.state.index == 0
        assert owner.state()["inventory_anchor"]["capture_dir"] == "collection/inventory/anchor-01/after"
        assert len(owner.report["source_sha256"]) > 25
    assert not any(listeners.values())
