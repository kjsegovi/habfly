"""Cooperative composition tests: injected pager, no browser/network/model calls.

Producer and accessor validation have their own strict-source tests. These tests
exercise the scheduler boundary with conspicuously synthetic evidence, never a
source receipt intended for live reuse.
"""
# ruff: noqa: F811

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_project_inventory_steps import capture, events, rig, sha  # noqa: F401

import habfly.browser_project_inventory_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_probe import save_probe
from habfly.contracts import RuntimeEvent

NAMES = ["FIXTURE", *[f"Star{i}" for i in range(10)]]


@pytest.fixture
def paged(rig, monkeypatch):
    flags = SimpleNamespace(children=[], finish_hook=None, step_hook=None, fault=None, steps=5)

    class Pager:
        def __init__(self, page, config, output, **options):
            self.page, self.output, self.options = page, output, options
            self.finished, self.report = False, None
            self.status, self.phase, self.failure = "paused", "initial_capture", None
            self.advances = 0
            self.anchor = deepcopy(page.capture)
            output.mkdir()
            flags.children.append(self)
            rig.flags.calls.append("pager.construct")

        def state(self):
            return {
                "mode": "live_paginated_collected_stellar_inventory",
                "phase": self.phase,
                "status": self.status,
                "finished": self.finished,
                "advances": self.advances,
                "failure_reason": self.failure,
                "inventory_dir": str(self.output.relative_to(rig.root))
                if self.status == "completed"
                else None,
                "inventory_sha256": sha(self.output / "confirmed.json")
                if self.status == "completed"
                else None,
                "task_completed": False,
                "project_completed": False,
            }

        def advance(self):
            rig.flags.calls.append("pager.advance")
            self.advances += 1
            event = RuntimeEvent(
                event="state", sequence=self.advances, run_id="injected-pager", payload=self.state()
            )
            self.options["emit"](event.model_dump(mode="json"))
            if flags.step_hook:
                flags.step_hook(self)
            if self.options["cancelled"]():
                return self.abort()
            if self.advances < flags.steps:
                # Each native child owns current page guards. Parent must not
                # reject page two because it differs from its initial page one.
                self.page.capture["frames"][0]["text"] = "injected second page"
                self.phase = "forward_next"
                return self.state()
            self.page.capture = deepcopy(self.anchor)
            save_probe(self.anchor, self.output / "anchor-01/after")
            self.report = {
                "fixture_only": True,
                "never_live_evidence": True,
                "rows": [{"name": n, "text": n} for n in self.options["expected_stars"]],
            }
            module.persist_json(self.output / "confirmed.json", self.report)
            self.status, self.phase, self.finished = "completed", "inventory_verified", True
            if flags.finish_hook:
                flags.finish_hook(self)
            return self.state()

        def abort(self):
            self.finished = True
            self.status, self.phase = "aborted", "stopped"
            self.failure = "operator_aborted"
            rig.flags.calls.append("pager.abort")
            return self.state()

    def source(book, directory, *, expected_sha256):
        rig.flags.calls.append("source.load")
        assert expected_sha256 == sha(directory / "confirmed.json")
        child = flags.children[-1]
        receipt = book.json(directory / "confirmed.json")
        anchor_dir = directory / "anchor-01/after"
        anchor = book.capture(anchor_dir)
        binding = {
            "inventory_evidence_version": 1,
            "inventory_kind": "live_paginated",
            "whole_collection_sha256": "a" * 64,
            "inventory_anchor": {
                "capture_dir": str(anchor_dir.relative_to(rig.root)),
                "observation_sha256": sha(anchor_dir / "observation.json"),
                "manifest_sha256": sha(anchor_dir / "manifest.json"),
                "visible_rows_sha256": "b" * 64,
            },
        }
        rows = deepcopy(receipt["rows"])
        kind = "live_paginated"
        if flags.fault == "rows":
            rows[-1]["name"] = "Alien"
        elif flags.fault == "legacy":
            kind = "legacy_single_page"
        elif flags.fault == "receipt":
            child.report = {"different": True}
        elif flags.fault == "anchor":
            anchor["frames"][0]["text"] = "wrong current first page"
        return SimpleNamespace(
            receipt=receipt,
            rows=rows,
            anchor=anchor,
            anchor_dir=anchor_dir,
            kind=kind,
            binding_fields=lambda: deepcopy(binding),
        )

    monkeypatch.setattr(module, "PaginatedInventorySteps", Pager)
    monkeypatch.setattr(module, "load_inventory_source", source)

    def create(**options):
        return rig.create(
            inventory_mode="live_paginated", expected_stars=NAMES, **{"max_seconds": 420, **options}
        )

    return SimpleNamespace(rig=rig, flags=flags, create=create)


def construct_pager(paged, **options):
    owner = paged.create(**options)
    assert paged.rig.flags.calls == []
    for phase in ("to_stellar", "verify_inventory", "paginated_active"):
        owner.advance()
        assert owner.phase == phase
    assert len(paged.flags.children) == 1
    assert paged.flags.children[0].advances == 0
    return owner


def complete(owner):
    for _ in range(12):
        if owner.finished:
            return owner.state()
        owner.advance()
    pytest.fail("bounded component did not stop")


def test_explicit_mode_one_scheduled_child_call_and_actual_anchor(paged):
    owner = construct_pager(paged)
    child = paged.flags.children[0]
    assert child.options["max_seconds"] == 240
    assert child.options["max_advances"] == 8 and child.options["max_clicks"] == 4
    for index in range(5):
        old = len(paged.rig.flags.calls)
        owner.advance()
        calls = paged.rig.flags.calls[old:]
        assert calls.count("pager.advance") == 1
        assert "read" not in calls and "verify" not in calls
        assert child.advances == index + 1
    state = owner.state()
    assert state["status"] == "completed" and state["advances"] == 8
    assert state["max_advances"] == 11 and state["max_seconds"] == 420
    assert state["inventory_kind"] == "live_paginated"
    assert state["whole_collection_sha256"] != state["inventory_anchor"]["visible_rows_sha256"]
    assert state["inventory_anchor"]["capture_dir"].endswith("/anchor-01/after")
    assert not (owner.output / "inventory/after").exists()
    assert (
        state["collection_count_verified"] and not state["task_completed"] and not state["project_completed"]
    )
    assert any(e["payload"].get("component") == "project.inventory.pages" for e in events(owner.output))
    assert all(
        sha(paged.rig.root / name) == checksum for name, checksum in owner.report["source_sha256"].items()
    )


@pytest.mark.parametrize("count,budget", [(10, 420), (31, 420), (11, 421), (11, True), (11, float("nan"))])
def test_paginated_scope_is_not_a_silent_legacy_budget_increase(rig, count, budget):
    with pytest.raises(BrowserSafetyStop):
        rig.create(
            inventory_mode="live_paginated",
            expected_stars=["FIXTURE", *[f"S{i}" for i in range(count - 1)]],
            max_seconds=budget,
        )
    assert rig.flags.calls == []


def test_legacy_rejects_large_budget_and_never_falls_back(paged):
    with pytest.raises(BrowserSafetyStop, match="invalid_time_budget"):
        paged.rig.create(max_seconds=181)
    paged.rig.flags.inventory_failure = "project_inventory_incomplete_visible_list"
    owner = paged.rig.create()
    complete(owner)
    assert owner.failure == "project_inventory_steps_unsupported_pagination"
    assert not paged.flags.children


def test_remaining_budget_caps_child_without_new_deadline(paged):
    owner = paged.create()
    owner.advance()
    owner.advance()
    paged.rig.flags.now = 415
    owner.advance()
    child = paged.flags.children[0]
    assert child.options["max_seconds"] == 5
    paged.rig.flags.now = 420
    owner.advance()
    assert owner.failure == "project_inventory_steps_time_limit"
    assert child.advances == 0 and child.finished


@pytest.mark.parametrize("failure", ["rows", "legacy", "receipt", "anchor"])
def test_terminal_child_claims_require_matching_sources_and_actual_anchor(paged, failure):
    paged.flags.fault = failure
    owner = construct_pager(paged)
    complete(owner)
    assert owner.status == "stopped" and not owner.state()["collection_count_verified"]
    assert owner.state()["inventory_sha256"] is None


@pytest.mark.parametrize("where", ["before_child", "inside_child", "final_callback"])
def test_cancellation_preserves_bounded_dispatch_and_never_retries(paged, where):
    def emit(kind, payload):
        if where == "final_callback" and kind == "episode_summary":
            paged.rig.flags.cancel = True

    owner = construct_pager(paged, emit=emit)
    if where == "before_child":
        paged.rig.flags.cancel = True
    elif where == "inside_child":
        paged.flags.step_hook = lambda _: setattr(paged.rig.flags, "cancel", True)
    complete(owner)
    count = paged.flags.children[0].advances
    owner.advance()
    assert count == {"before_child": 0, "inside_child": 1, "final_callback": 5}[where]
    assert paged.flags.children[0].advances == count and owner.status != "completed"
    assert not owner.state()["collection_count_verified"]


@pytest.mark.parametrize("where", ["between", "child_callback", "terminal_callback"])
def test_source_mutation_invalidates_terminal_success(paged, where):
    path = paged.rig.directory / "upstream.json"

    def mutate():
        path.write_text('{"changed":true}')

    def emit(kind, payload):
        if where == "terminal_callback" and kind == "episode_summary":
            mutate()
        if where == "child_callback" and payload.get("component"):
            mutate()

    owner = construct_pager(paged, emit=emit)
    if where == "between":
        mutate()
    complete(owner)
    assert owner.status == "stopped" and not owner.state()["collection_count_verified"]
    if where == "terminal_callback":
        assert (owner.output / "invalidated.json").exists()


def test_parent_absolute_advance_limit_and_frozen_scope(paged):
    paged.flags.steps = 100
    owner = construct_pager(paged)
    complete(owner)
    assert owner.advances == 11 and paged.flags.children[0].advances == 8
    assert owner.failure == "project_inventory_steps_advance_limit"
    assert not owner.state()["collection_count_verified"]


@pytest.mark.parametrize(
    "attribute,value",
    [("inventory_mode", "legacy_single_page"), ("max_seconds", 1000), ("expected_stars", ("FIXTURE",))],
)
def test_mutating_selected_mode_or_budget_stops_before_next_child(paged, attribute, value):
    owner = construct_pager(paged)
    setattr(owner, attribute, value)
    owner.advance()
    assert owner.failure == "project_inventory_steps_scope_changed"
    assert paged.flags.children[0].advances == 0


def test_no_assessment_or_submission_action_is_added(paged):
    owner = construct_pager(paged)
    complete(owner)
    report = json.loads((owner.output / "report.json").read_text())
    assert all(report[key] == 0 for key in module.ZERO)
    assert report["journal_writes"] == 0
    assert not report["task_completed"] and not report["project_completed"]
