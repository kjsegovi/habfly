"""Playback context only: no model loads, browser, network, or live commands."""

import io
import json
from copy import deepcopy

import pytest

from habfly.runtime import Runtime, read_trace


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not initialize an environment or contact a service")

    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("habfly.browser.TorusBrowser.__init__", forbidden)
    monkeypatch.setattr("habfly.runtime.Runtime.start", forbidden)


def source(tmp_path, payloads, *, run_id="saved"):
    path = tmp_path / "recorded.jsonl"
    path.write_text(
        "".join(
            json.dumps({"version": 1, "event": kind, "sequence": i, "run_id": run_id, "payload": payload})
            + "\n"
            for i, (kind, payload) in enumerate(payloads)
        )
    )
    return path


def replay(path):
    runtime = Runtime(io.StringIO())
    runtime.command({"command": "replay", "payload": {"path": str(path)}})
    return runtime


def last(runtime):
    return json.loads(runtime.output.getvalue().splitlines()[-1])["payload"]


@pytest.mark.parametrize("recorded_status", ["running", "paused", "stopped", "handoff", "completed"])
def test_eof_retains_latest_star_progress_and_explicit_failure_without_success(tmp_path, recorded_status):
    progress = {"collected": 2, "verified": 1, "submitted": False}
    path = source(
        tmp_path,
        [
            ("hello", {"task": "browser_project", "policy": "checkpoint"}),
            ("state", {"status": "paused", "phase": "launch_pending", "current_star": None}),
            (
                "state",
                {
                    "status": recorded_status,
                    "phase": "active",
                    "current_star": {"star": "Second", "ordinal": 2},
                    "project_progress": progress,
                    "task_completed": False,
                    "project_completed": False,
                    "failure_reason": "saved_stop" if recorded_status == "stopped" else None,
                },
            ),
            ("action_result", {"terminated": True, "reward": 1}),
        ],
    )
    before = path.read_bytes()
    runtime = replay(path)
    while runtime.replay_events is not None:
        runtime.tick()
    final = last(runtime)
    assert final["replay_finished"] is True and runtime.status == "completed"
    assert final["recorded_status"] == recorded_status
    assert final["phase"] == "active" and final["current_star"]["star"] == "Second"
    assert final["project_progress"] == progress
    assert final["task_completed"] is False and final["project_completed"] is False
    assert final["failure_reason"] == ("saved_stop" if recorded_status == "stopped" else None)
    assert path.read_bytes() == before and runtime.env is None


def test_pause_resume_uses_latest_context_and_keeps_partial_state_fields(tmp_path):
    path = source(
        tmp_path,
        [
            ("state", {"status": "running", "current_star": {"star": "First"}}),
            ("state", {"current_star": {"star": "Second"}, "project_progress": {"verified": 1}}),
            ("state", {"policy_stage": "stellar_color"}),
        ],
    )
    runtime = replay(path)
    for _ in range(3):
        runtime.tick()
    for command in ["pause", "resume"]:
        runtime.command({"command": command})
        assert last(runtime)["current_star"]["star"] == "Second"
        assert last(runtime)["policy_stage"] == "stellar_color"
        assert last(runtime)["project_progress"] == {"verified": 1}
        assert last(runtime)["replay_finished"] is False
        assert last(runtime)["recorded_status"] == "running"
    runtime.tick()
    assert last(runtime)["replay_finished"] is True


def test_legacy_recorded_events_remain_unchanged_and_missing_success_stays_missing(tmp_path):
    path = source(
        tmp_path,
        [
            ("hello", {"activity_source": "untrained_observer"}),
            ("state", {"status": "running", "seed": 0}),
            ("state", {"status": "paused", "steps": 4}),
            ("episode_summary", {"completed": False, "failure_reason": "operator_aborted"}),
        ],
    )
    original = read_trace(path)
    runtime = replay(path)
    while runtime.replay_events is not None:
        runtime.tick()
    events = [json.loads(line) for line in runtime.output.getvalue().splitlines()]
    for emitted, recorded in zip(events[1:-1], original, strict=True):
        expected = {**recorded.payload, "replay": True}
        if recorded.event == "state":
            expected.update(status="running", recorded_status=recorded.payload.get("status"))
        assert emitted["payload"] == expected
    assert last(runtime)["steps"] == 4 and last(runtime)["recorded_status"] == "paused"
    assert "task_completed" not in last(runtime) and "project_completed" not in last(runtime)
    assert events[-2]["payload"]["completed"] is False


def test_recorded_explicit_success_is_preserved_not_created_by_eof(tmp_path):
    runtime = replay(source(tmp_path, [("state", {"status": "completed", "task_completed": True})]))
    runtime.tick()
    runtime.tick()
    assert last(runtime)["task_completed"] is True
    assert "project_completed" not in last(runtime)


def test_empty_trace_finishes_without_any_task_claim(tmp_path):
    runtime = replay(source(tmp_path, []))
    runtime.tick()
    assert last(runtime)["replay_finished"] is True
    assert last(runtime)["recorded_status"] is None
    assert "task_completed" not in last(runtime)


def raw_finalizer_scope(**changes):
    return {
        "schema_version": 1,
        "mode": "bounded_post_campaign_finalization",
        "project_id": "project",
        "attempt_id": "attempt",
        "data_revision": 90,
        "campaign_handoff_sha256": "d" * 64,
        "allow_score_transfer": True,
        "allow_submission": True,
        "scoring_max_seconds": 600,
        "submission_max_seconds": 180,
        "scoring_max_advances": 46,
        "submission_max_advances": 4,
        "task_completed": False,
        "project_completed": False,
        "submitted": False,
        "status": "paused",
        "phase": "submission_verifying",
        "finished": False,
        "project_progress": {"submitted": False, "project_completed": False},
        **changes,
    }


@pytest.mark.parametrize(
    "status,phase",
    [
        ("handoff", "unknown_pending"),
        ("handoff", "score_transferred_not_submitted"),
        ("handoff", "assessed_score_transfer_disabled"),
        ("stopped", "stopped"),
        ("aborted", "aborted"),
    ],
)
def test_raw_finalizer_terminal_summary_supplies_eof_context_without_success(tmp_path, status, phase):
    summary = raw_finalizer_scope(
        status=status, phase=phase, finished=True, failure_reason="recorded_boundary", project_progress=None
    )
    path = source(
        tmp_path,
        [("hello", raw_finalizer_scope()), ("episode_summary", summary)],
        run_id="project-finalize-" + "d" * 16,
    )
    before = path.read_bytes()
    runtime = replay(path)
    runtime.tick()
    runtime.tick()
    # The actual event remains unchanged; only playback context is repaired.
    assert last(runtime) == {**summary, "replay": True}
    for command in ("pause", "resume"):
        runtime.command({"command": command})
        assert last(runtime)["phase"] == phase
        assert last(runtime)["recorded_status"] == status
    runtime.tick()
    final = last(runtime)
    assert final["replay_finished"] is True
    assert final["phase"] == phase and final["recorded_status"] == status
    assert final["failure_reason"] == "recorded_boundary" and final["project_progress"] is None
    assert all(final[k] is False for k in ("submitted", "task_completed", "project_completed"))
    assert runtime.env is None and path.read_bytes() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"task_completed": True},
        {"project_completed": True},
        {"submitted": True},
        {"submitted": 0},
        {"mode": "child_finalizer"},
        {"finished": 1},
        {"status": "completed", "phase": "completed"},
        {"phase": "submission_active"},
        {"campaign_handoff_sha256": "e" * 64},
        {"attempt_id": "different"},
        {"data_revision": 91},
        {"allow_submission": False},
        {"allow_score_transfer": False},
        {"scoring_max_seconds": 601},
        {"submission_max_advances": 4.0},
        {"component": "project.finalization.submission"},
        {"component_event": {"event": "episode_summary", "payload": {"completed": True}}},
        {"component_summary": {"completed": True}},
        {"component_state": {}},
        {"component_hello": {}},
    ],
)
def test_raw_finalizer_replay_never_promotes_child_or_changed_terminal_summary(tmp_path, changes):
    summary = raw_finalizer_scope(
        **{"status": "handoff", "phase": "unknown_pending", "finished": True, **changes}
    )
    runtime = replay(
        source(
            tmp_path,
            [("state", raw_finalizer_scope()), ("episode_summary", summary)],
            run_id="project-finalize-" + "d" * 16,
        )
    )
    while runtime.replay_events is not None:
        runtime.tick()
    final = last(runtime)
    assert final["phase"] == "submission_verifying" and final["recorded_status"] == "paused"
    assert all(final[k] is False for k in ("submitted", "task_completed", "project_completed"))


@pytest.mark.parametrize("mutation", ["no_parent", "foreign_run", "intervening_scope", "bad_run_id"])
def test_raw_finalizer_summary_requires_matching_previously_seen_parent(tmp_path, mutation):
    summary = raw_finalizer_scope(status="handoff", phase="unknown_pending", finished=True)
    records = [("state", raw_finalizer_scope()), ("episode_summary", summary)]
    if mutation == "no_parent":
        records = records[1:]
    elif mutation == "intervening_scope":
        records.insert(1, ("state", raw_finalizer_scope(mode="another_owner")))
    path = source(tmp_path, records, run_id="project-finalize-" + "d" * 16)
    if mutation in {"foreign_run", "bad_run_id"}:
        items = [json.loads(line) for line in path.read_text().splitlines()]
        for item in items[-1:] if mutation == "foreign_run" else items:
            item["run_id"] = "another-owner"
        path.write_text("".join(json.dumps(item) + "\n" for item in items))
    runtime = replay(path)
    while runtime.replay_events is not None:
        runtime.tick()
    assert last(runtime).get("phase") != "unknown_pending"
    assert runtime.env is None


def test_raw_finalizer_corrective_stop_overrides_preliminary_terminal_summary(tmp_path):
    runtime = replay(
        source(
            tmp_path,
            [
                ("state", raw_finalizer_scope()),
                (
                    "episode_summary",
                    raw_finalizer_scope(status="handoff", phase="unknown_pending", finished=True),
                ),
                (
                    "episode_summary",
                    raw_finalizer_scope(
                        status="stopped",
                        phase="stopped",
                        finished=True,
                        failure_reason="project_finalize_source_changed",
                        project_progress=None,
                    ),
                ),
            ],
            run_id="project-finalize-" + "d" * 16,
        )
    )
    while runtime.replay_events is not None:
        runtime.tick()
    assert last(runtime)["phase"] == last(runtime)["recorded_status"] == "stopped"
    assert last(runtime)["failure_reason"] == "project_finalize_source_changed"
    assert last(runtime)["project_progress"] is None
    assert last(runtime)["project_completed"] is False


def campaign_state(*, compact=False, terminal=False):
    payload = {
        "mode": "fresh_browser_campaign_project_runtime",
        "task": "browser_project",
        "runtime_task": "browser_project",
        "environment": "browser",
        "project_campaign": True,
        "target_stars": 3,
        "stars": 3,
        "current_star": {"star": "Daphion", "ordinal": 3, "fresh_dir": "picker", "fresh_sha256": "a" * 64},
        "status": "handoff" if terminal else "running",
        "phase": "awaiting_assessment" if terminal else "verifying_star",
        "browser_phase": "awaiting_assessment" if terminal else "verifying_star",
        "finished": terminal,
        "task_completed": False,
        "project_completed": False,
        "target_workflows_verified": terminal,
        "failure_reason": None,
        "project_progress": {
            "project_id": "habworlds",
            "attempt_id": "browser-project-attempt",
            "target": 30,
            "revision": 18 if terminal else 12,
            "collected": 3 if terminal else 2,
            "verified": 3 if terminal else 2,
            "unresolved": 0,
            "uncertain_actions": [],
            "submitted": False,
            "project_completed": False,
        },
    }
    if terminal:
        payload["completed"] = False
    if compact:
        payload.update(
            project_compact_wire=True,
            project_wire={
                "version": 1,
                "mode": "compact_project_display",
                "ancestry": [],
                "leaf_header": {"event": "episode_summary" if terminal else "state"},
                "evidence_receipt": False,
            },
        )
    return payload


@pytest.mark.parametrize("compact", [False, True])
def test_outer_campaign_terminal_summary_updates_eof_after_child_state(tmp_path, compact):
    summary = campaign_state(compact=compact, terminal=True)
    child = campaign_state(compact=compact)
    child["component"] = "project.campaign"
    child["current_star"] = {"star": "not-parent", "ordinal": 1}
    if compact:
        child["project_wire"]["ancestry"] = [{"component": "project.campaign", "star": None}]
    path = source(
        tmp_path, [("state", campaign_state(compact=compact)), ("state", child), ("episode_summary", summary)]
    )
    before = path.read_bytes()
    runtime = replay(path)
    while runtime.replay_events is not None:
        runtime.tick()
    final = last(runtime)
    assert final["replay_finished"] and final["recorded_status"] == "handoff"
    assert final["phase"] == "awaiting_assessment" and final["target_workflows_verified"] is True
    assert "component" not in final
    assert (
        final["current_star"] == summary["current_star"]
        and final["project_progress"] == summary["project_progress"]
    )
    assert (
        final["task_completed"]
        is final["project_completed"]
        is final["project_progress"]["submitted"]
        is False
    )
    emitted = json.loads(runtime.output.getvalue().splitlines()[-2])["payload"]
    assert emitted == {**summary, "replay": True}
    assert runtime.env is None and path.read_bytes() == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("task", "mini_habworlds"),
        ("runtime_task", "child"),
        ("environment", "simulator"),
        ("mode", "fresh_cooperative_project_campaign"),
        ("project_campaign", 1),
        ("target_stars", 2),
        ("stars", 3.0),
        ("finished", 1),
        ("completed", True),
        ("task_completed", True),
        ("project_completed", True),
        ("submitted", True),
        ("submitted", 0),
        ("target_workflows_verified", 1),
        ("phase", "active"),
        ("browser_phase", "active"),
        ("failure_reason", "source_changed"),
        ("current_star.star", "Other"),
        ("current_star.fresh_sha256", "b" * 64),
        ("current_star.ordinal", 3.0),
        ("project_progress.attempt_id", "other"),
        ("project_progress.project_id", "other"),
        ("project_progress.revision", 11),
        ("project_progress.revision", True),
        ("project_progress.verified", 2),
        ("project_progress.collected", 2),
        ("project_progress.unresolved", 1),
        ("project_progress.uncertain_actions", ["pending"]),
        ("project_progress.submitted", True),
        ("project_progress.project_completed", True),
        ("component", "project.campaign"),
        ("component_event", {}),
        ("component_summary", {}),
        ("component_state", {}),
        ("component_hello", {}),
        ("ancestry", []),
        ("project_wire.ancestry", [{"component": "project.campaign"}]),
        ("project_wire.version", True),
        ("project_wire.leaf_header", {"event": "state"}),
        ("project_wire.evidence_receipt", True),
        ("project_wire", None),
    ],
)
def test_campaign_replay_rejects_foreign_child_or_unverified_summary(tmp_path, key, value):
    summary = campaign_state(compact=True, terminal=True)
    target = summary
    parts = key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    runtime = replay(
        source(tmp_path, [("state", campaign_state(compact=True)), ("episode_summary", summary)])
    )
    while runtime.replay_events is not None:
        runtime.tick()
    assert last(runtime)["recorded_status"] == "running" and last(runtime)["phase"] == "verifying_star"
    assert last(runtime)["target_workflows_verified"] is False
    assert last(runtime)["task_completed"] is last(runtime)["project_completed"] is False


@pytest.mark.parametrize(
    "mutation", ["no_parent", "different_run", "new_context", "missing_current", "intervening_scope"]
)
def test_campaign_summary_requires_previously_seen_matching_outer_context(tmp_path, mutation):
    records = [("state", campaign_state()), ("episode_summary", campaign_state(terminal=True))]
    if mutation == "no_parent":
        records = records[1:]
    elif mutation == "new_context":
        parent = deepcopy(records[0][1])
        parent["current_star"]["fresh_sha256"] = "c" * 64
        records.insert(1, ("state", parent))
    elif mutation == "missing_current":
        del records[1][1]["current_star"]
    elif mutation == "intervening_scope":
        records.insert(1, ("state", {"status": "paused", "mode": "different"}))
    path = source(tmp_path, records)
    if mutation == "different_run":
        items = [json.loads(line) for line in path.read_text().splitlines()]
        items[-1]["run_id"] = "different"
        path.write_text("".join(json.dumps(item) + "\n" for item in items))
    runtime = replay(path)
    while runtime.replay_events is not None:
        runtime.tick()
    assert last(runtime).get("phase") != "awaiting_assessment"


@pytest.mark.parametrize("status", ["stopped", "aborted"])
def test_outer_campaign_terminal_failure_preserved_without_success(tmp_path, status):
    summary = campaign_state(terminal=True)
    summary.update(
        status=status,
        phase=status,
        browser_phase=status,
        target_workflows_verified=False,
        failure_reason="recorded_failure",
    )
    runtime = replay(source(tmp_path, [("state", campaign_state()), ("episode_summary", summary)]))
    runtime.tick()
    runtime.tick()
    runtime.command({"command": "pause"})
    assert (
        last(runtime)["recorded_status"] == status and last(runtime)["failure_reason"] == "recorded_failure"
    )
    runtime.command({"command": "resume"})
    runtime.tick()
    assert last(runtime)["phase"] == status and last(runtime)["replay_finished"] is True
    assert last(runtime)["target_workflows_verified"] is False
