"""Real post-owner and source validators behind declared upstream/native seams.

No browser launch, trained policy execution, charge, or submission is performed.
The imported fixture supplies synthetic upstream workflow bundles; canonical
imports, actual paginated receipt loading, historical replay and owner classes
remain the production implementations.
"""
# ruff: noqa: F811

import io
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_numeric import config
from test_browser_project_finalize_steps import case, finished_campaign, paged, rig, subject  # noqa: F401

from habfly.browser_project_finalize_steps import BrowserProjectFinalizeSteps
from habfly.project_wire import reconstruct_native_event
from habfly.runtime import RunOptions, Runtime


@pytest.mark.parametrize("compact", (False, True))
def test_runtime_constructs_real_owner_and_coordinator_offline_then_aborts(finished_campaign, compact):
    rig = finished_campaign
    runtime = Runtime(io.StringIO())
    runtime.options = RunOptions(
        task="browser_project",
        stars=30,
        project_campaign=True,
        project_campaign_max_seconds=10800,
        project_allow_scoring=True,
        project_allow_submission=True,
        project_compact_wire=compact,
        artifact_dir=rig.root.parent,
    )
    runtime._finalization_authorization = runtime._authorization()
    runtime.run_id, runtime.trace_path = rig.root.name, rig.root.parent / "outer-runtime.jsonl"
    runtime.status = "paused"
    before = (rig.root / "report.json").read_bytes()
    report = json.loads(before)
    bridge = SimpleNamespace(
        page=rig.page,
        config=config(),
        output=rig.root,
        journal=rig.journal,
        closed=False,
        finished=True,
        status="handoff",
        phase="awaiting_assessment",
        report=report,
        state=lambda: deepcopy(report),
    )
    bridge.close = lambda: setattr(bridge, "closed", True)
    runtime.env = bridge
    try:
        runtime._queue_finalization(paused=True)
        assert runtime.finalization is None and not rig.calls and not rig.page.clicks
        runtime.command({"command": "step"})
        assert isinstance(runtime.finalization, BrowserProjectFinalizeSteps)
        assert runtime.finalization.phase == "scoring_initializing"
        assert runtime.finalization.scoring is None
        runtime.command({"command": "step"})
        assert runtime.finalization.phase == "scoring_active"
        assert runtime.finalization.scoring is not None
        assert runtime.finalization.scoring.advances == 0
        assert not rig.calls and not rig.page.clicks
        assert rig.journal.path.read_bytes() == rig.historical
        runtime.command({"command": "abort"})
        assert runtime.status == "aborted" and runtime.finalization.finished
        assert (rig.root / "report.json").read_bytes() == before
        assert rig.journal.path.read_bytes() == rig.historical
        events = [json.loads(line) for line in runtime.output.getvalue().splitlines()]
        lifecycle = [event for event in events if event["event"] in {"hello", "state", "episode_summary"}]
        assert lifecycle and all(event["payload"]["project_completed"] is False for event in lifecycle)
        assert all(event["payload"]["task_completed"] is False for event in lifecycle)
        if compact:
            assert any(event["payload"]["project_wire"]["ancestry"] for event in events)
            assert all("project_wire" in event["payload"] for event in events)
            raw = [
                json.loads(line) for line in (rig.root / "finalization/events.jsonl").read_text().splitlines()
            ]
            data = [event for event in raw if event["event"] not in {"hello", "state", "episode_summary"}]
            for native in data:
                leaf = native
                while "component_event" in leaf["payload"]:
                    leaf = leaf["payload"]["component_event"]
                assert any(
                    reconstruct_native_event(event["payload"]) == leaf
                    for event in events
                    if event["event"] == native["event"]
                )
    finally:
        runtime.close()
    assert bridge.closed
