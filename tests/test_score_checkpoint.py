"""Offline reporting checks; no browser, score calculation or new assessment."""

import io
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_project_progress import complete_star, fresh, receipt_for, reserve, write

from habfly.browser_project_finalize_steps import BrowserProjectFinalizeSteps
from habfly.project_wire import compact_project_event
from habfly.runtime import Runtime


@pytest.fixture(scope="module")
def assessed():
    progress = fresh()
    for index in range(30):
        progress = complete_star(progress, f"Star{index:02d}")
    for kind in ("assessment_data_quality", "assessment_scavenger_hunt"):
        progress = write(progress, kind)
    progress = reserve(progress, "score_transfer")
    return progress.append(receipt_for(progress).model_copy(update={"score": 0.0})).reduce()


def finalizer(current, tmp_path):
    owner = object.__new__(BrowserProjectFinalizeSteps)
    owner._score_checkpoint_sha = "a" * 64
    owner._score_checkpoint_closed = True
    owner.status, owner.phase, owner.failure = "handoff", "score_transferred_not_submitted", None
    owner._abort_requested = owner._forward_failed = owner._cleanup_failed = False
    owner.allow_score_transfer, owner.allow_submission = True, False
    owner.revision = current.revision
    owner.book = SimpleNamespace(unchanged=lambda: None, hashes={})
    owner._completed_trees = {}
    owner._current = lambda: (b"", None, current)
    owner.scope = {"task_completed": False, "project_completed": False, "submitted": False}
    owner.submission = owner.scoring = owner._component = None
    owner.advances = 0
    owner.output = tmp_path / "finalization"
    owner.output.mkdir()
    owner.history = tmp_path
    return owner


def test_zero_score_is_current_receipt_not_truthiness_or_computation(assessed, tmp_path, monkeypatch):
    owner = finalizer(assessed, tmp_path)
    monkeypatch.setattr("habfly.browser_project_finalize_steps._sha", lambda _: "a" * 64)
    assert owner._score_checkpoint(assessed, "a" * 64)
    state = owner.state()
    assert state["score_checkpoint_completed"] is True and state["reported_score"] == 0.0
    assert all(state[k] is False for k in ("task_completed", "project_completed", "submitted"))


@pytest.mark.parametrize(
    "change",
    [
        "running",
        "stopped",
        "aborted",
        "unknown_pending",
        "callback",
        "cleanup",
        "not_closed",
        "source_changed",
        "pending",
        "stale_score",
        "missing_assessment",
        "missing_star",
        "unverified_star",
    ],
)
def test_checkpoint_requires_clean_closed_current_thirty(assessed, tmp_path, change):
    current = deepcopy(assessed)
    owner = finalizer(current, tmp_path)
    digest = "a" * 64
    if change in {"running", "stopped", "aborted"}:
        owner.status = change
    elif change == "unknown_pending":
        owner.phase = change
    elif change == "callback":
        owner._forward_failed = True
    elif change == "cleanup":
        owner._cleanup_failed = True
    elif change == "not_closed":
        owner._score_checkpoint_closed = False
    elif change == "source_changed":
        digest = "b" * 64
    elif change == "pending":
        current.receipts.pop()
    elif change == "stale_score":
        current.receipts[-1].revision -= 1
    elif change == "missing_assessment":
        current.receipts = [r for r in current.receipts if r.write_kind != "assessment_data_quality"]
    elif change == "missing_star":
        current.stars.pop("Star00")
    elif change == "unverified_star":
        current.stars["Star00"].completion = None
    assert owner._score_checkpoint(current, digest) is False


def test_runtime_failure_clears_even_finished_child_checkpoint():
    runtime = Runtime(output=io.StringIO())
    runtime.finalization = SimpleNamespace(
        state=lambda: {
            "mode": "bounded_post_campaign_finalization",
            "status": "handoff",
            "phase": "score_transferred_not_submitted",
            "task_completed": False,
            "project_completed": False,
            "score_checkpoint_completed": True,
            "reported_score": 0.0,
        }
    )
    runtime._finalization_terminal = {"status": "stopped", "failure_reason": "callback_failed"}
    result = runtime._finalization_state()
    assert result["score_checkpoint_completed"] is False and result["reported_score"] is None

    def unavailable():
        raise ValueError("fixture source unavailable")

    runtime.finalization.state = unavailable
    unavailable_state = runtime._finalization_state()
    assert unavailable_state["source_state_unavailable"] is True
    assert unavailable_state["score_checkpoint_completed"] is False
    assert unavailable_state["reported_score"] is None


@pytest.mark.parametrize("failure", ["close", "report"])
def test_closure_or_report_failure_never_latches_checkpoint(assessed, tmp_path, monkeypatch, failure):
    owner = finalizer(assessed, tmp_path)
    owner.status = "running"
    owner._score_checkpoint_closed = False
    owner._check = lambda: None
    owner._emit = lambda *_: None
    owner._relay = SimpleNamespace(retire=lambda: None)
    (owner.output / "events.jsonl").write_bytes(b"")

    def fail(*_):
        raise OSError("injected fixture closure failure")

    owner._stream = SimpleNamespace(close=fail if failure == "close" else lambda: None)
    if failure == "report":
        monkeypatch.setattr("habfly.browser_project_finalize_steps.persist_json", fail)
    with pytest.raises(OSError):
        owner._finish("score_transferred_not_submitted")
    assert owner._score_checkpoint_closed is False and not owner.finished
    assert owner._score_checkpoint(assessed, "a" * 64) is False
    assert not (owner.output / "report.json").exists()


def test_compact_parent_only_checkpoint_fields_and_legacy_omission():
    legacy = {"task": "browser_project", "task_completed": False}
    old = compact_project_event("state", legacy)
    assert {k: v for k, v in old.items() if k != "project_wire"} == legacy
    assert "score_checkpoint_completed" not in old and "reported_score" not in old
    state = {
        **legacy,
        "score_checkpoint_completed": True,
        "reported_score": 0.0,
        "project_owner": {"score_checkpoint_completed": True, "reported_score": 99},
    }
    result = compact_project_event("state", state)
    assert result["score_checkpoint_completed"] is True and result["reported_score"] == 0.0
    assert "score_checkpoint_completed" not in result["project_owner"]
    assert "reported_score" not in result["project_owner"]
