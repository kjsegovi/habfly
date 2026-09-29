"""Explicit30 no-overall-timer contracts; injected clocks/pages, no native run."""
# ruff: noqa: F811, F401

import io
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_browser_project_campaign_runtime import make, ready
from test_browser_project_campaign_runtime import rig as bridge_rig
from test_browser_project_campaign_steps import rig as campaign_rig
from test_browser_project_reference_class import RATIONALE
from test_browser_project_runtime import rig as base_rig
from test_browser_project_tui_launcher import launcher
from test_runtime_browser_project import options

from habfly.browser import BrowserSafetyStop
from habfly.browser_project_campaign_steps import BrowserProjectCampaignSteps
from habfly.project_wire import compact_project_event
from habfly.runtime import RunOptions, Runtime, parse_run_options


def payload(tmp_path, **changes):
    return options(
        tmp_path, stars=30, project_campaign=True, project_campaign_max_seconds="uncapped", **changes
    )


def test_explicit_sentinel_roundtrips_without_changing_child_limits(tmp_path):
    value = parse_run_options(payload(tmp_path))
    assert value.project_campaign_max_seconds == "uncapped"
    assert RunOptions.model_validate_json(value.model_dump_json()) == value
    assert value.project_max_seconds == 1800 and value.project_max_advances == 512
    assert value.project_class_max_seconds == 180
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("target", [1, 2, 3, 4, 29, 31])
def test_uncapped_is_not_an_implicit_option_for_other_targets(tmp_path, target):
    with pytest.raises(ValueError):
        parse_run_options(payload(tmp_path) | {"stars": target})


@pytest.mark.parametrize("changes", [{"task": "stellar"}, {"project_campaign": False}])
def test_uncapped_requires_project_campaign(tmp_path, changes):
    with pytest.raises(ValueError):
        parse_run_options(payload(tmp_path) | changes)


@pytest.mark.parametrize(
    "budget",
    [
        None,
        True,
        False,
        0,
        -1,
        10801,
        float("inf"),
        float("nan"),
        "Uncapped",
        " uncapped",
        "uncapped ",
        b"uncapped",
        "10800",
        {},
        [],
    ],
)
def test_unknown_or_missing_budget_still_rejects(tmp_path, budget):
    with pytest.raises(ValueError):
        parse_run_options(payload(tmp_path) | {"project_campaign_max_seconds": budget})


@pytest.mark.parametrize("target", [2, 3, 30])
def test_finite_profile_contract_remains_numeric(tmp_path, target):
    value = parse_run_options(payload(tmp_path) | {"stars": target, "project_campaign_max_seconds": 10800})
    assert type(value.project_campaign_max_seconds) is float and value.project_campaign_max_seconds == 10800


def uncapped_owner(rig):
    return BrowserProjectCampaignSteps(
        rig.page,
        rig.config,
        rig.root / "uncapped",
        **(rig.kwargs | {"target_stars": 30, "max_seconds": "uncapped"}),
    )


@pytest.mark.parametrize("target", [2, 3])
def test_direct_owner_constructor_does_not_enable_uncapped_small_campaign(campaign_rig, target):
    rig = campaign_rig
    with pytest.raises(BrowserSafetyStop, match="invalid_fixed_campaign_deadline"):
        BrowserProjectCampaignSteps(
            rig.page,
            rig.config,
            rig.root / "invalid-uncapped",
            **(rig.kwargs | {"target_stars": target, "max_seconds": "uncapped"}),
        )
    assert not (rig.root / "invalid-uncapped").exists() and not rig.flags.calls


@pytest.mark.parametrize("budget", [3600, "uncapped"])
def test_terminal_guard_skips_only_explicit_overall_timer(campaign_rig, budget):
    rig = campaign_rig
    owner = BrowserProjectCampaignSteps(
        rig.page,
        rig.config,
        rig.root / "terminal-clock",
        **(rig.kwargs | {"target_stars": 30, "max_seconds": budget}),
    )
    try:
        owner.start(paused=True)
        # Exercise only the final guard: no stars/receipts are invented, so
        # target_workflows_verified must remain false even in this fake handoff.
        owner.status, owner.phase = "handoff", "awaiting_assessment"
        owner._callback = lambda *_: setattr(rig.flags, "now", 100000)
        owner._finish()
        assert owner.status == ("handoff" if budget == "uncapped" else "stopped")
        assert owner.report["target_workflows_verified"] is False
        assert owner.report["task_completed"] is owner.report["project_completed"] is False
    finally:
        owner.close()


def test_owner_no_overall_clock_but_original_child_limits_and_abort_remain(campaign_rig):
    rig = campaign_rig
    owner = uncapped_owner(rig)
    try:
        owner.start(paused=True)
        rig.flags.now = 100000
        owner.step()  # Only now construct the first child, with fixed1800/512.
        assert not owner.finished and rig.flags.calls == ["owner.init"]
        assert owner.scope["pause_counts_toward_deadline"] is False
        assert owner.scope["owner_limits"] == {"max_seconds": 1800, "max_advances": 512}
        assert owner.scope["transition_limits"] == {"max_seconds": 180, "max_advances": 128}
        assert owner.scope["automatic_deadline_increase"] is False
        owner.abort()
        before = list(rig.flags.calls)
        owner.step()
        assert owner.status == "aborted" and rig.flags.calls == before
        assert not owner.state()["target_workflows_verified"]
    finally:
        owner.close()


@pytest.mark.parametrize("budget", [3600, "uncapped"])
@pytest.mark.parametrize("mutation", ["budget", "scope", "source", "cancel"])
def test_owner_sources_and_budget_identity_remain_guarded(campaign_rig, budget, mutation):
    rig = campaign_rig
    owner = BrowserProjectCampaignSteps(
        rig.page,
        rig.config,
        rig.root / "guarded",
        **(rig.kwargs | {"target_stars": 30, "max_seconds": budget}),
    )
    try:
        owner.start(paused=True)
        if mutation == "budget":
            owner.max_seconds = 3600 if budget == "uncapped" else "uncapped"
        elif mutation == "scope":
            owner.scope["max_seconds"] = 3600 if budget == "uncapped" else "uncapped"
        elif mutation == "source":
            rig.source.write_bytes(b"changed fixture source")
        else:
            rig.flags.cancel = True
        owner.step()
        assert owner.status in {"stopped", "aborted"} and not rig.flags.calls
    finally:
        owner.close()


def test_uncapped_bridge_survives_old_overall_deadline_without_actions(bridge_rig):
    rig = bridge_rig
    rig.options = rig.options.model_copy(update={"stars": 30, "project_campaign_max_seconds": "uncapped"})
    bridge = ready(rig)
    try:
        rig.clock.now = 100000
        before = deepcopy(rig.calls)
        bridge._check_initial()
        bridge.step()  # An unresolved explicit class handoff is still guidance only.
        assert bridge.status == "paused" and rig.calls == before
        assert bridge.project.seconds == "uncapped"
        bridge.abort()
        assert bridge.status == "aborted"
    finally:
        bridge.close()


@pytest.mark.parametrize("target", [2, 3])
def test_direct_bridge_constructor_rejects_uncapped_small_campaign_before_page(bridge_rig, target):
    rig = bridge_rig
    rig.options = rig.options.model_copy(update={"stars": target, "project_campaign_max_seconds": "uncapped"})
    with pytest.raises(BrowserSafetyStop, match="explicit_campaign_budget_required"):
        make(rig)
    assert not rig.calls


@pytest.mark.parametrize("budget", [600, "uncapped"])
@pytest.mark.parametrize("location", ["option", "scope", "bridge", "all", "child"])
def test_bridge_budget_cannot_be_rewritten_midrun(bridge_rig, budget, location):
    rig = bridge_rig
    rig.options = rig.options.model_copy(update={"stars": 30, "project_campaign_max_seconds": budget})
    bridge = ready(rig)
    changed = 600 if budget == "uncapped" else "uncapped"
    try:
        if location != "child":
            bridge.provide_reference_class(
                selected_class="white_dwarf", reference_rationale=RATIONALE, lifetime_prefix=None
            )
        before = list(rig.calls)
        if location in {"option", "all"}:
            rig.options.project_campaign_max_seconds = changed
        if location in {"scope", "all"}:
            bridge.scope["campaign_max_seconds"] = changed
        if location in {"bridge", "all"}:
            bridge.campaign_max_seconds = changed
        if location == "child":
            bridge.project.seconds = changed
            with pytest.raises(BrowserSafetyStop, match="invalid_campaign_state"):
                bridge._sync_campaign()
        else:
            bridge.step()
            assert bridge.status == "stopped"
        assert [
            call for call in rig.calls[len(before) :] if call not in {("campaign_abort",), ("setup_close",)}
        ] == []
        assert bridge.class_setup is None or not bridge.class_setup.writes
    finally:
        bridge.close()


@pytest.mark.parametrize("original,changed", [(10800.0, "uncapped"), ("uncapped", 10800.0), (10800.0, 10800)])
def test_runtime_budget_authorization_is_sticky_and_keeps_flag_indices(tmp_path, original, changed):
    runtime = Runtime(output=io.StringIO())
    runtime.options = parse_run_options(
        payload(tmp_path)
        | {
            "project_campaign_max_seconds": original,
            "project_allow_scoring": True,
            "project_allow_submission": False,
        }
    )
    runtime._finalization_authorization = runtime._authorization()
    assert runtime._finalization_authorization[3:5] == (True, False)
    runtime.options.project_campaign_max_seconds = changed
    with pytest.raises(ValueError, match="authorization_changed"):
        runtime._check_project_authorization()
    runtime.options.project_campaign_max_seconds = original
    with pytest.raises(ValueError, match="authorization_changed"):
        runtime._check_project_authorization()


def test_wire_replay_and_check_preserve_explicit_uncapped_without_completion(tmp_path, launcher, capsys):
    state = {
        "task": "browser_project",
        "runtime_task": "browser_project",
        "project_campaign": True,
        "campaign_max_seconds": "uncapped",
        "status": "paused",
        "task_completed": False,
        "project_completed": False,
        "campaign": {"max_seconds": "uncapped", "target_stars": 30},
    }
    projected = compact_project_event("state", state)
    assert projected["campaign_max_seconds"] == projected["campaign"]["max_seconds"] == "uncapped"
    legacy = compact_project_event("state", {**state, "campaign_max_seconds": 3600})
    assert "campaign_max_seconds" not in legacy  # Existing finite compact shape unchanged.
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"version": 1, "event": "state", "sequence": 0, "payload": projected}) + "\n")
    replay = Runtime(output=io.StringIO())
    replay.command({"command": "replay", "payload": {"path": str(trace)}})
    while replay.replay_events is not None:
        replay.tick()
    eof = json.loads(replay.output.getvalue().splitlines()[-1])["payload"]
    assert eof["campaign_max_seconds"] == "uncapped" and eof["recorded_status"] == "paused"
    assert eof["task_completed"] is eof["project_completed"] is False
    monkeypatch_profile = {**launcher.test_payload, "stars": 30, "project_campaign_max_seconds": "uncapped"}
    launcher.test_profile.write_text(json.dumps(monkeypatch_profile))
    assert launcher.main(["--check"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["campaign_timer"] == "no_overall_timer" and result["launch_authorized"] is False
    with pytest.raises(SystemExit) as exc:
        launcher.main([])
    assert exc.value.code == 2  # No build/credentials/browser; explicit30 launch hold stays.


def test_only_held_profile_selects_uncapped():
    selected = []
    for path in Path("configs").glob("browser_project*.json"):
        value = json.loads(path.read_bytes())
        if value.get("project_campaign_max_seconds") == "uncapped":
            selected.append(path.name)
            assert value["stars"] == 30 and value["paused"] is True
            assert value["project_allow_scoring"] is True and value["project_allow_submission"] is False
    assert selected == ["browser_project_thirty_star.json"]
