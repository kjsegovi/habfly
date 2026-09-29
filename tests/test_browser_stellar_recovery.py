"""Read-only final-check reconciliation, with no repeated answer copies."""
# ruff: noqa: F811

import hashlib
import json
from copy import deepcopy

import pytest
from test_browser_full_stellar import full_page, select  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.browser_full_stellar import UNITS
from habfly.browser_probe import inspect_page
from habfly.browser_stellar import SIMULATION_URL
from habfly.browser_stellar_recovery import reconcile_stellar_footer


@pytest.fixture
def completed(full_page, tmp_path):
    select(full_page)
    frame = full_page.frame(url=SIMULATION_URL)
    for key in UNITS:
        frame.locator("#" + key).fill("1")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"fixture frozen checkpoint")
    source = tmp_path / "source"
    source.mkdir()
    after = inspect_page(full_page, config())
    before = deepcopy(after)
    sim = next(f for f in before["frames"] if f["url"] == SIMULATION_URL)
    sim["accessibility"] = sim["accessibility"].replace(
        '1 Rs\n- button "Save"', '1 Rs Data saved\n- button "Save"'
    )
    persist_json(
        source / "screen-change-001.json",
        {
            "before": before,
            "after": after,
            "change_count": 1,
            "truncated": False,
            "changes": [{"path": "/frames/0/accessibility"}],
        },
    )
    events = [
        {
            "event": "action_proposed",
            "payload": {
                "action_source": "frozen_lifetime_checkpoint",
                "action": {"kind": "CLICK", "target": "59:check"},
            },
        },
        {
            "event": "state",
            "payload": {
                "screen_change": {
                    "path": "screen-change-001.json",
                    "sha256": hashlib.sha256((source / "screen-change-001.json").read_bytes()).hexdigest(),
                }
            },
        },
    ]
    (source / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    persist_json(
        source / "manifest.json",
        {
            "events_sha256": hashlib.sha256((source / "events.jsonl").read_bytes()).hexdigest(),
            "outcome": "screen_changed_during_binding",
            "full_stellar_numeric_transport_verified": False,
            "steps": 59,
            "write_attempts": 6,
            "verified_fields": list(UNITS),
            "optimizer_updates": 0,
            "checkpoint_unchanged": True,
            "provenance": {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
            "numeric_readbacks": {k: {"display_value": "1", "exact_input_verified": True} for k in UNITS},
        },
    )
    return full_page, frame, source, checkpoint


def test_reconcile_is_read_only_and_preserves_failure(completed, tmp_path):
    page, frame, source, checkpoint = completed
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    receipt = reconcile_stellar_footer(page, config(), source, tmp_path / "reconciled", checkpoint=checkpoint)
    assert receipt["full_stellar_numeric_transport_reconciled"] and receipt["browser_actions"] == 0
    assert not receipt["task_completed"] and not receipt["saved"]
    assert {p.name: p.read_bytes() for p in source.iterdir()} == originals
    assert all(frame.locator("#" + k).input_value() == "1" for k in UNITS)
    assert not page.get_by_role("checkbox").is_checked()


@pytest.mark.parametrize("bad", ["value", "class", "other_feedback", "events", "diff", "checkpoint"])
def test_reconcile_rejects_any_changed_evidence(completed, tmp_path, bad):
    page, frame, source, checkpoint = completed
    if bad == "value":
        frame.locator("#mass").fill("2")
    elif bad == "class":
        frame.locator(".choice label").nth(3).click()
    elif bad == "other_feedback":
        frame.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<p>Data saved in another panel</p>')"
        )
    elif bad in {"events", "diff"}:
        p = source / ("events.jsonl" if bad == "events" else "screen-change-001.json")
        p.write_bytes(p.read_bytes() + b" ")
    else:
        checkpoint.write_bytes(b"changed")
    with pytest.raises(BrowserSafetyStop):
        reconcile_stellar_footer(page, config(), source, tmp_path / "reconciled", checkpoint=checkpoint)
    assert not (tmp_path / "reconciled/reconciled.json").exists()
