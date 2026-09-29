"""Pure fresh-sweep gates, using real pager proofs and injected visible clicks.

No browser, course, network, or learned policy runs here. Only the native pager
and assessment surfaces are injected; production loaders, scoring actuators,
canonical reservations and immutable source guards execute unchanged.
"""
# ruff: noqa: F811

import json
import re
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from test_browser_project_paginated_inventory_steps import complete, subject  # noqa: F401
from test_browser_project_scoring_steps import case, make  # noqa: F401
from test_project_evidence import read, sha, write

import habfly.browser_project_scoring_steps as module
from habfly.browser import BrowserSafetyStop


@pytest.fixture
def paged(case, subject):
    subject.state.names = [f"Star{i:02d}" for i in range(30)]
    initial = complete(subject, output="initial-live-inventory")
    anchor = read(case.root / initial.report["anchor_capture"] / "observation.json")
    case.page.rows = re.findall(
        r"OBSERVATIONS(.*?)TOTAL COLLECTED", anchor["frames"][0]["text"].upper(), re.DOTALL
    )[0].strip()
    case.values["inventory_dir"] = initial.output
    original = case.page.report
    settings = SimpleNamespace(timestamp="valid")

    def report():
        value = original()
        subject.state.reads += 1
        stamp = datetime(2026, 9, 26, tzinfo=UTC) + timedelta(seconds=subject.state.reads)
        if settings.timestamp == "valid":
            value["captured_at"] = stamp.isoformat()
        elif settings.timestamp != "missing":
            value["captured_at"] = (
                stamp.replace(tzinfo=None).isoformat() if settings.timestamp == "naive" else "not-a-date"
            )
        return value

    case.page.report = report
    return SimpleNamespace(case=case, subject=subject, initial=initial, settings=settings)


def sweep(paged, component, kind, *, handoff=True, output=None):
    owner = complete(paged.subject, output=output or f"fresh-{kind}")
    if handoff:
        component.provide_inventory_refresh(
            kind, inventory_dir=owner.output, inventory_sha256=sha(owner.output / "confirmed.json")
        )
    return owner


def reach(paged, component, kind):
    """Reach one distinct boundary without providing its new proof yet."""
    if kind == "data_quality":
        return
    sweep(paged, component, "data_quality")
    component.provide_panel("data_quality")
    component.advance()
    component.advance()
    component.advance()
    assert component.phase == "awaiting_scavenger_hunt_panel", component.report
    paged.case.page.mode = "scavenger_hunt"
    if kind == "scavenger_hunt":
        return
    sweep(paged, component, "scavenger_hunt")
    component.provide_panel("scavenger_hunt")
    component.advance()
    component.advance()
    assert component.phase == "score_transfer_ready", component.report


def schedule(component, kind):
    if kind != "score_transfer":
        component.provide_panel(kind)
        if kind == "data_quality":
            component.advance()


def test_three_fresh_sweeps_bind_separate_boundaries_without_false_completion(paged):
    component = make(paged.case, allow_score_transfer=True)
    reach(paged, component, "score_transfer")
    sweep(paged, component, "score_transfer")
    component.advance()
    assert component.status == "completed", component.report
    assert paged.case.page.clicks == ["ASSESS", "OK", "ASSESS", "OK", "Update Score"]
    records = component.report["inventory_refreshes"]
    assert set(records) == {*module.KINDS, "score_transfer"}
    expected = {
        "data_quality": "initial-live-inventory/anchor-01/after",
        "scavenger_hunt": str((component.stage_dir / "ack-000-after").relative_to(component.history)),
        "score_transfer": str((component.stage_dir / "ack-001-after").relative_to(component.history)),
    }
    for kind, record in records.items():
        assert record == read(component.output / f"inventory-refresh-{kind}.json")
        proof = record["freshness"]
        assert proof["boundary_capture"] == expected[kind]
        assert datetime.fromisoformat(proof["sweep_started_at"]) > datetime.fromisoformat(
            proof["boundary_captured_at"]
        )
        assert proof["started_after_boundary_verified"] and not proof["atomic_server_snapshot_verified"]
        assert record["whole_collection_sha256"] == component.inventory_source.whole_collection_sha256
    assert len({r["inventory_sha256"] for r in records.values()}) == 3
    state = paged.case.journal.load().reduce()
    assert len(state.receipts) == 3 and not state.pending
    assert component.report["score_transfer_verified"] and not component.report["submitted"]
    assert not component.report["task_completed"] and not component.report["project_completed"]


@pytest.mark.parametrize("kind", [*module.KINDS, "score_transfer"])
def test_each_boundary_without_its_own_proof_cannot_dispatch(paged, kind):
    component = make(paged.case, allow_score_transfer=True)
    reach(paged, component, kind)
    before = list(paged.case.page.clicks)
    if kind == "score_transfer":
        component.advance()
        assert component.failure == "project_scoring_fresh_collection_sweep_required"
    else:
        with pytest.raises(BrowserSafetyStop, match="fresh_collection_sweep_required"):
            component.provide_panel(kind)
        assert not (component.output / f"panel-{kind}.json").exists()
        component.abort()
    assert paged.case.page.clicks == before and not paged.case.journal.load().reduce().pending


@pytest.mark.parametrize("mismatch", ["rows", "whole", "anchor"])
def test_refresh_comparison_rejects_changed_cached_loader_result(paged, monkeypatch, mismatch):
    # Explicit loader-result corruption seam isolates child comparisons. The
    # actual underlying source first passes the production pager validator.
    component = make(paged.case)
    fresh = sweep(paged, component, "data_quality", handoff=False)
    loader = module.load_inventory_source

    def changed(*args, **kwargs):
        source = loader(*args, **kwargs)
        if mismatch == "rows":
            rows = deepcopy(source.rows)
            rows[-1]["text"] += " altered"
            return replace(source, rows=rows)
        if mismatch == "whole":
            return replace(source, whole_collection_sha256="f" * 64)
        anchor = deepcopy(source.anchor)
        anchor["frames"][0]["text"] = anchor["frames"][0]["text"].replace("Star00", "Other")
        return replace(source, anchor=anchor)

    monkeypatch.setattr(module, "load_inventory_source", changed)
    with pytest.raises(BrowserSafetyStop, match="refreshed_collection_changed"):
        component.provide_inventory_refresh(
            "data_quality", inventory_dir=fresh.output, inventory_sha256=sha(fresh.output / "confirmed.json")
        )
    assert not component._inventory_refreshes and not paged.case.page.clicks
    component.abort()


@pytest.mark.parametrize("kind", [*module.KINDS, "score_transfer"])
def test_sweep_started_before_required_boundary_is_rejected(paged, kind):
    component = make(paged.case, allow_score_transfer=True)
    reach(paged, component, kind)
    paged.subject.state.reads = 0  # Independent, internally ordered but old sweep.
    before = list(paged.case.page.clicks)
    with pytest.raises(BrowserSafetyStop, match="stale_collection_sweep"):
        sweep(paged, component, kind)
    assert kind not in component._inventory_refreshes and paged.case.page.clicks == before
    component.abort()


@pytest.mark.parametrize("timestamp", ["missing", "naive", "invalid"])
@pytest.mark.parametrize("kind", ["scavenger_hunt", "score_transfer"])
def test_boundary_requires_timezone_aware_timestamp(paged, timestamp, kind):
    component = make(paged.case, allow_score_transfer=True)
    paged.settings.timestamp = timestamp
    if kind == "score_transfer":
        # Keep the first acknowledgement valid; corrupt only the second one.
        paged.settings.timestamp = "valid"
        reach(paged, component, "scavenger_hunt")
        sweep(paged, component, "scavenger_hunt")
        component.provide_panel("scavenger_hunt")
        component.advance()
        paged.settings.timestamp = timestamp
        component.advance()
    else:
        reach(paged, component, kind)
    before = list(paged.case.page.clicks)
    with pytest.raises(BrowserSafetyStop, match="refresh_timestamp"):
        sweep(paged, component, kind)
    assert kind not in component._inventory_refreshes and paged.case.page.clicks == before
    component.abort()


def test_initial_and_previous_sweeps_cannot_be_reused_for_new_boundary(paged):
    component = make(paged.case, allow_score_transfer=True)
    with pytest.raises(BrowserSafetyStop, match="collection_sweep_reused"):
        component.provide_inventory_refresh(
            "data_quality",
            inventory_dir=paged.initial.output,
            inventory_sha256=sha(paged.initial.output / "confirmed.json"),
        )
    first = sweep(paged, component, "data_quality")
    with pytest.raises(BrowserSafetyStop, match="unexpected_inventory_refresh"):
        component.provide_inventory_refresh(
            "data_quality", inventory_dir=first.output, inventory_sha256=sha(first.output / "confirmed.json")
        )
    component.provide_panel("data_quality")
    component.advance()
    component.advance()
    component.advance()
    with pytest.raises(BrowserSafetyStop, match="collection_sweep_reused"):
        component.provide_inventory_refresh(
            "scavenger_hunt",
            inventory_dir=first.output,
            inventory_sha256=sha(first.output / "confirmed.json"),
        )
    assert paged.case.page.clicks == ["ASSESS", "OK"]
    component.abort()


@pytest.mark.parametrize("checksum", [None, "", 123, "f" * 63])
def test_refresh_requires_explicit_receipt_hash(paged, checksum):
    component = make(paged.case)
    with pytest.raises(BrowserSafetyStop, match="invalid_inventory_hash"):
        component.provide_inventory_refresh(
            "data_quality", inventory_dir=paged.initial.output, inventory_sha256=checksum
        )
    assert not component._inventory_refreshes and not paged.case.page.clicks
    component.abort()


@pytest.mark.parametrize(
    "change",
    ["cache", "deleted", "injected", "record", "record_and_cache", "source", "mode", "original_rows"],
)
def test_accepted_refresh_and_original_inventory_are_immutable_after_handoff(paged, change):
    component = make(paged.case)
    fresh = sweep(paged, component, "data_quality")
    record = component._inventory_refreshes["data_quality"]
    path = component.output / "inventory-refresh-data_quality.json"
    if change == "cache":
        record["inventory_sha256"] = "f" * 64
    elif change == "deleted":
        component._inventory_refreshes.clear()
    elif change == "injected":
        component._inventory_refreshes["score_transfer"] = deepcopy(record)
    elif change in {"record", "record_and_cache"}:
        altered = {**record, "revision": 999}
        write(path, altered)
        if change == "record_and_cache":
            component._inventory_refreshes["data_quality"] = altered
    elif change == "source":
        source = fresh.output / "plan.json"
        source.write_bytes(source.read_bytes() + b"\n")
    elif change == "mode":
        component.paginated = False
    else:
        component.inventory_source.rows[-1]["text"] += " changed"
    component.advance()
    assert component.status == "stopped" and not paged.case.page.clicks
    assert not paged.case.journal.load().reduce().reservations


@pytest.mark.parametrize("kind", [*module.KINDS, "score_transfer"])
@pytest.mark.parametrize("change", ["cache", "artifact"])
def test_callback_mutation_after_reservation_cannot_dispatch(paged, kind, change):
    holder = {}

    def callback(event):
        paged.case.events.append(event)
        target = "score_transfer" if kind == "score_transfer" else "assessment_" + kind
        if event["event"] != "action_proposed" or event["payload"]["target"] != target:
            return
        owner = holder["owner"]
        if change == "cache":
            owner._inventory_refreshes[kind]["freshness"]["started_after_boundary_verified"] = False
        else:
            path = owner.output / f"inventory-refresh-{kind}.json"
            path.write_bytes(path.read_bytes() + b"\n")

    component = make(paged.case, allow_score_transfer=True, emit=callback)
    holder["owner"] = component
    reach(paged, component, kind)
    sweep(paged, component, kind)
    schedule(component, kind)
    before = list(paged.case.page.clicks)
    component.advance()
    assert component.status == "stopped" and paged.case.page.clicks == before
    pending = paged.case.journal.load().reduce().pending
    assert len(pending) == 1 and pending[0].write_kind == (
        "score_transfer" if kind == "score_transfer" else "assessment_" + kind
    )
    component.advance()
    assert paged.case.page.clicks == before


def test_terminal_callback_cache_drift_cannot_report_clean_completion(paged):
    holder = {}

    def callback(event):
        if event["event"] == "episode_summary":
            holder["owner"]._inventory_refreshes["score_transfer"]["inventory_sha256"] = "f" * 64

    component = make(paged.case, allow_score_transfer=True, emit=callback)
    holder["owner"] = component
    reach(paged, component, "score_transfer")
    sweep(paged, component, "score_transfer")
    component.advance()
    assert component.status == "stopped" and component.failure == "project_scoring_terminal_validation_failed"
    assert len(paged.case.page.clicks) == 5 and len(paged.case.journal.load().reduce().receipts) == 3
    assert not component.report["project_completed"] and not component.report["submitted"]
    events = [json.loads(line) for line in (component.output / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "state" and events[-1]["payload"]["status"] == "stopped"
