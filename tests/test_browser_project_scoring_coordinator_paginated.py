"""Real cooperative children/receipts/ledgers with explicit injected page seams.

No native browser, network, training or real project completion. The disposable
thirty-task prerequisites are synthetic; every generated pagination receipt is
checked by the real live-mode validator and shared accessor.
"""
# ruff: noqa: F811

from copy import deepcopy

import pytest
import test_browser_project_paginated_inventory_steps as page_fixture
import yaml
from test_browser_numeric import config
from test_browser_project_paginated_inventory_steps import complete, subject  # noqa: F401
from test_browser_project_scoring_coordinator import rig  # noqa: F401
from test_browser_project_scoring_steps import case  # noqa: F401
from test_project_assessment import visible
from test_project_evidence import sha, write

import habfly.browser_project_paginated_inventory_steps as inventories
import habfly.browser_project_scoring_coordinator as module
from habfly.browser_project_navigation import LIST_LABELS
from habfly.runtime import read_trace


@pytest.fixture
def paged(rig, subject, monkeypatch):
    subject.state.names = [f"Star{i:02d}" for i in range(30)]
    original_capture = page_fixture.capture

    def capture(names, start, total):
        value = original_capture(names, start, total)
        frame = value["frames"][0]
        panel_text = visible(mode=rig.page.mode, funding=rig.page.funding, collected=30, ack=rig.page.ack)
        header = panel_text.split("OBSERVATIONS")[0]
        frame["text"] = header + "OBSERVATIONS" + frame["text"].split("Observations", 1)[1]
        atoms = yaml.safe_load(frame["accessibility"])
        atoms[0]["text"] = header + "Observations Analyzed Data Star " + " ".join(LIST_LABELS["stellar"])
        if rig.page.ack:
            acknowledgement = {
                "data_quality": "DATA QUALITY UPDATED OK",
                "scavenger_hunt": "SCAVENGER HUNT UPDATED OK",
            }[rig.page.mode]
            frame["text"] += "\n" + acknowledgement
            atoms.append({"text": acknowledgement})
        frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)
        frame["controls"] += [
            {
                "id": f"simulation-0:c{i + 2}",
                "role": "button",
                "accessibility": f'- button "{label}"',
                "enabled": True,
                "actions": [],
                "protected": False,
            }
            for i, label in enumerate(("ASSESSMENT", "DATA QUALITY", "SCAVENGER HUNT", "Assess", "Save"))
        ]
        return value

    monkeypatch.setattr(page_fixture, "capture", capture)
    initial = complete(subject, output="original-live-inventory")
    subject.state.clicks.clear()
    subject.state.events.clear()
    subject.state.owner = None
    rig.page.report = subject.current
    # These are the explicit native exposure seams. Public capture parsing,
    # receipt hashes, full forward/revisit proof and canonical charges stay real.
    rig.page.frames[0].locator = lambda _: type(
        "VisibleText",
        (),
        {
            "evaluate": lambda _self, _code: [
                {
                    "text": subject.current()["frames"][0]["text"],
                    "boxes": [{"x": 0, "y": 0, "width": 1, "height": 1}],
                }
            ],
        },
    )()
    monkeypatch.setattr(inventories, "visible_inventory_page", lambda *_: None)
    created = []
    constructor = inventories.PaginatedInventorySteps.__init__

    def construct(self, *args, **kwargs):
        constructor(self, *args, **kwargs)
        subject.state.owner = self
        created.append(self)

    monkeypatch.setattr(inventories.PaginatedInventorySteps, "__init__", construct)
    rig.values["inventory_dir"] = initial.output
    rig.initial_inventory, rig.subject, rig.inventory_children = initial, subject, created
    return rig


def make(paged, **options):
    return module.BrowserProjectScoringCoordinator(
        paged.page,
        config(),
        paged.root / options.pop("output", "paged-scoring"),
        **(paged.values | {"cancelled": lambda: paged.flags.cancelled} | options),
    )


def until(owner, predicate):
    for _ in range(46):
        if owner.finished or predicate():
            return
        owner.advance()
    pytest.fail("Fixed paginated coordinator work budget exhausted")


@pytest.mark.parametrize("transfer", [False, True])
def test_real_sweeps_precede_every_real_assessment_and_optional_transfer(paged, transfer):
    owner = make(paged, allow_score_transfer=transfer)
    assert owner.paginated and owner.scope["max_advances"] == 46 and owner.max_seconds == 600
    assert not paged.calls and not paged.page.clicks and not paged.inventory_children
    checks = []

    def before_charge(label):
        if label not in {"ASSESS", "Update Score"}:
            return
        kind = paged.page.mode if label == "ASSESS" else "score_transfer"
        assert kind in owner._completed_inventories
        assert kind in owner.scoring.state()["inventory_refreshes"]
        assert len(paged.journal.load().reduce().pending) == 1
        checks.append(kind)

    paged.page.preclick = before_charge
    for _ in range(46):
        if owner.finished:
            break
        before = (
            len(paged.page.clicks)
            + len(paged.subject.state.clicks)
            + sum(call[0] == "panel_click" for call in paged.calls)
        )
        owner.advance()
        after = (
            len(paged.page.clicks)
            + len(paged.subject.state.clicks)
            + sum(call[0] == "panel_click" for call in paged.calls)
        )
        assert after - before <= 1
    assert owner.finished and owner.failure is None, owner.report
    kinds = ["data_quality", "scavenger_hunt"] + (["score_transfer"] if transfer else [])
    assert checks == kinds
    assert list(owner.report["verified_inventory_refreshes"]) == kinds
    assert len(paged.inventory_children) == len(kinds)
    assert all(child.report["navigation_clicks"] == 4 for child in paged.inventory_children)
    assert all(
        child.report["whole_collection_sha256"] == owner.inventory_source.whole_collection_sha256
        for child in paged.inventory_children
    )
    assert owner.report["score_transfer_verified"] is transfer
    assert all(owner.report[k] is False for k in ("submitted", "task_completed", "project_completed"))
    assert owner.advances <= 46 and owner.scoring.advances == (6 if transfer else 5)
    assert paged.page.clicks == ["ASSESS", "OK", "ASSESS", "OK"] + (["Update Score"] if transfer else [])
    assert not paged.journal.load().reduce().pending
    outer = [e.model_dump(mode="json") for e in read_trace(owner.output / "events.jsonl")]
    for kind, child in zip(kinds, paged.inventory_children):
        original = [e.model_dump(mode="json") for e in read_trace(child.output / "events.jsonl")]
        assert [
            e["payload"]["component_event"]
            for e in outer
            if e["payload"].get("component") == "project.inventory." + kind
        ] == original
        assert owner.report["source_sha256"][
            str((child.output / "confirmed.json").relative_to(paged.root))
        ] == sha(child.output / "confirmed.json")
    assert owner.report["inventory_anchor"] == owner.inventory_source.binding_fields()["inventory_anchor"]


def test_refresh_proof_handoff_is_a_separate_offline_call(paged):
    owner = make(paged)
    until(owner, lambda: owner.phase == "inventory_handoff")
    assert not owner.finished, owner.report
    assert owner.scoring.phase == "awaiting_data_quality_panel"
    reads, clicks = paged.subject.state.reads, list(paged.subject.state.clicks)
    owner.advance()
    assert owner.phase == "panel_handoff" and "data_quality" in owner.scoring.state()["inventory_refreshes"]
    assert paged.subject.state.reads == reads and paged.subject.state.clicks == clicks
    owner.advance()
    assert owner.phase == "scoring_active" and not paged.page.clicks
    owner.abort()


def test_fresh_sweep_fixed_budget_is_capped_by_original_parent_deadline(paged):
    owner = make(paged)
    until(owner, lambda: owner.phase == "inventory_initializing")
    paged.clock[0] = 450
    owner.advance()
    assert owner.inventory.scope["max_seconds"] == 150
    assert owner.inventory.scope["max_clicks"] == 4 and owner.inventory.scope["max_advances"] == 8
    assert owner.scoring.max_seconds == owner.max_seconds == 600
    paged.clock[0] = 600
    owner.advance()
    assert owner.finished and owner.phase == "stopped" and not paged.page.clicks


@pytest.mark.parametrize("phase", ["inventory_initializing", "inventory_active", "inventory_handoff"])
def test_abort_refresh_boundary_is_sticky_and_cannot_charge(paged, phase):
    owner = make(paged)
    until(owner, lambda: owner.phase == phase)
    before = list(paged.page.clicks), list(paged.subject.state.clicks)
    owner.abort()
    owner.advance()
    assert owner.phase == "stopped" and (paged.page.clicks, paged.subject.state.clicks) == before
    assert not paged.journal.load().reduce().pending


@pytest.mark.parametrize("how", ["abort", "callback_error", "source_change", "cancel", "reentry"])
def test_inventory_event_callbacks_cannot_dispatch_after_parent_stop(paged, how):
    holder = {}

    def emit(event):
        if event["event"] != "action_proposed" or not event["payload"].get("component", "").startswith(
            "project.inventory."
        ):
            return
        if how == "abort":
            holder["owner"].abort()
        elif how == "callback_error":
            raise RuntimeError("PRIVATE SESSION DETAILS")
        elif how == "source_change":
            write(paged.initial_inventory.output / "stopped.json", {})
        elif how == "cancel":
            paged.flags.cancelled = True
        else:
            holder["owner"].advance()

    holder["owner"] = owner = make(paged, emit=emit)
    until(owner, lambda: False)
    assert owner.finished and owner.phase == "stopped"
    assert not paged.page.clicks and not paged.subject.state.clicks
    assert "PRIVATE" not in (owner.output / "events.jsonl").read_text()


@pytest.mark.parametrize("what", ["receipt", "source", "added_file", "late_stop"])
def test_pinned_refresh_source_changes_block_offline_handoff_and_charge(paged, what):
    owner = make(paged)
    until(owner, lambda: owner.phase == "inventory_handoff")
    directory = owner.inventory.output
    if what == "receipt":
        write(directory / "confirmed.json", {})
    elif what == "source":
        write(directory / "anchor-01/after/observation.json", {})
    elif what == "added_file":
        write(directory / "unrecorded.json", {})
    else:
        write(directory / "revisit-02/stopped.json", {})
    owner.advance()
    assert owner.finished and owner.phase == "stopped" and not paged.page.clicks


def test_changed_noncurrent_row_cannot_reach_assess_even_when_anchor_matches(paged):
    owner = make(paged)
    until(owner, lambda: owner.phase == "inventory_active")
    paged.subject.state.row_overrides["Star20 0.045"] = "Star20 0.046"
    until(owner, lambda: False)
    assert owner.finished and owner.phase == "stopped" and not paged.page.clicks


def test_scavenger_ack_requires_third_sweep_before_transfer(paged):
    owner = make(paged, allow_score_transfer=True)
    until(
        owner, lambda: owner.phase == "inventory_initializing" and owner._inventory_kind == "score_transfer"
    )
    assert not owner.finished, owner.report
    assert paged.page.clicks == ["ASSESS", "OK", "ASSESS", "OK"]
    assert "score_transfer" not in owner.scoring.state()["inventory_refreshes"]
    owner.abort()
    owner.advance()
    assert "Update Score" not in paged.page.clicks


def test_parent_work_cap_is_fixed_not_mutable_scope(paged):
    owner = make(paged)
    owner.max_advances = 47
    owner.advance()
    assert owner.finished and owner.phase == "stopped" and not paged.page.clicks


def test_panel_requires_original_live_binding_not_only_total30(paged, monkeypatch):
    owner = make(paged)
    owner.advance()
    original = owner.panel.advance

    def altered():
        result = original()
        owner.panel.report["whole_collection_sha256"] = "f" * 64
        write(owner.panel.output / "confirmed.json", owner.panel.report)
        return result

    monkeypatch.setattr(owner.panel, "advance", altered)
    owner.advance()
    assert owner.finished and not paged.inventory_children and not paged.page.clicks


def test_completed_coordinator_cannot_reschedule_refreshes(paged):
    owner = make(paged)
    until(owner, lambda: False)
    assert owner.failure is None, owner.report
    original = deepcopy(paged.page.clicks), deepcopy(paged.subject.state.clicks)
    owner.advance()
    owner.close()
    assert (paged.page.clicks, paged.subject.state.clicks) == original
