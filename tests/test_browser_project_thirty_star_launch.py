"""Explicit30-target launch contracts, injected lifecycle only; no live browser.

Accepting configuration is not evidence of30-task completion. The partial-count
case deliberately lies in the injected child's completion counter and verifies
the real outer canonical/source gate refuses it.
"""
# ruff: noqa: F811

import json
from pathlib import Path

import pytest
from test_browser_project_campaign_runtime import classify, make, reach, ready, rig  # noqa: F401
from test_browser_project_reference_class import RATIONALE
from test_browser_project_runtime import rig as base_rig  # noqa: F401
from test_browser_project_tui_launcher import launcher  # noqa: F401
from test_runtime_browser_project import options

from habfly.browser import BrowserSafetyStop
from habfly.runtime import RunOptions, parse_run_options


@pytest.mark.parametrize("target", [2, 3, 30])
def test_only_explicit_supported_campaign_targets_parse_without_launch(tmp_path, target):
    parsed = parse_run_options(
        options(tmp_path, stars=target, project_campaign=True, project_campaign_max_seconds=10800)
    )
    assert parsed.stars == target and parsed.project_campaign
    assert parsed.project_max_advances == 512 and parsed.project_max_seconds == 1800
    assert not (tmp_path / "runs").exists()


def test_shallow_three_star_profile_is_separate_bounded_reference_acceptance():
    small = parse_run_options(json.loads(Path("configs/browser_project_two_star.json").read_text()))
    value = parse_run_options(json.loads(Path("configs/browser_project_shallow_three_star.json").read_text()))
    assert value.stars == 3 and value.project_campaign and value.paused
    assert value.project_reference_shallow_transits and value.project_reference_planet_continuation
    assert not small.project_reference_shallow_transits
    assert value.project_compact_wire and value.project_campaign_max_seconds == 5400
    assert value.project_max_seconds == 1800 and value.project_max_advances == 512
    assert not value.project_allow_scoring and not value.project_allow_submission
    assert value.artifact_dir == Path("experiments/browser-project-shallow-three-star")
    assert value.checkpoint == small.checkpoint and value.graph == small.graph


@pytest.mark.parametrize("target", [1, 4, 10, 11, 29, 31, 0])
def test_no_other_launch_target_is_silently_enabled(tmp_path, target):
    with pytest.raises(ValueError):
        parse_run_options(
            options(tmp_path, stars=target, project_campaign=True, project_campaign_max_seconds=10800)
        )
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"project_campaign": False},
        {"project_campaign_max_seconds": None},
        {"project_campaign_max_seconds": 10801},
        {"project_campaign_max_seconds": True},
        {"project_max_advances": 513},
        {"project_max_seconds": 1801},
        {"browser_setup": "manual"},
        {"browser_execution": "supervised"},
    ],
)
def test_thirty_launch_keeps_explicit_fixed_budgets_and_setup(tmp_path, change):
    value = options(tmp_path, stars=30, project_campaign=True, project_campaign_max_seconds=10800)
    value.update(change)
    with pytest.raises(ValueError):
        parse_run_options(value)
    assert not (tmp_path / "runs").exists()


def thirty(rig):
    rig.options = rig.options.model_copy(update={"stars": 30, "project_campaign_max_seconds": 10800})
    return rig


def test_thirty_outer_constructor_and_start_remain_page_free(rig):
    bridge = make(thirty(rig))
    assert rig.calls == [] and not bridge.output.exists()
    bridge.start()
    assert rig.calls == [] and not bridge.credentials_consumed
    assert bridge.state()["target_stars"] == 30
    assert bridge.scope["campaign_max_seconds"] == 10800
    assert bridge.scope["project_limits"] == {"max_seconds": 1800, "max_advances": 512}
    assert not bridge.scope["assessment_enabled"] and not bridge.scope["submission_enabled"]
    assert not bridge.state()["target_workflows_verified"]
    bridge.close()


def test_thirty_still_requires_explicit_current_star_class_before_work(rig):
    bridge = ready(thirty(rig))
    assert bridge.phase == "awaiting_class_source"
    assert bridge.project.target == 30 and bridge.project.seconds == 10800
    assert not rig.classes and not bridge.journal.load().records
    before = list(rig.calls)
    bridge.step()
    assert rig.calls == before
    assert bridge.phase == "awaiting_class_source"
    assert not bridge.state()["task_completed"] and not bridge.state()["project_completed"]
    bridge.close()


def test_thirty_budget_expiry_never_expands_or_dispatches_again(rig):
    bridge = ready(thirty(rig))
    before = list(rig.calls)
    rig.clock.now = 10801
    bridge.provide_reference_class(
        selected_class="white_dwarf", reference_rationale=RATIONALE, lifetime_prefix=None
    )
    assert bridge.finished and bridge.status == "stopped"
    assert "time_limit" in bridge.failure
    assert not any(call[0] in {"campaign_step", "class_advance"} for call in rig.calls[len(before) :])
    assert bridge.scope["campaign_max_seconds"] == 10800
    assert not bridge.report["target_workflows_verified"]
    bridge.close()


def test_two_tasks_cannot_satisfy_thirty_target_even_if_child_claims_it(rig):
    bridge = ready(thirty(rig))
    classify(bridge)
    reach(bridge, "adopting_fresh_star")
    bridge.step()
    reach(bridge, "awaiting_class_source")
    classify(bridge)
    reach(bridge, "verifying_star")
    bridge.project.verified = 29  # Explicitly forged test-only child counter.
    bridge.step()
    assert bridge.status == "stopped" and bridge.finished
    assert bridge.journal.load().reduce().report()["verified"] == 2
    assert not bridge.report["target_workflows_verified"]
    assert not bridge.report["task_completed"] and not bridge.report["project_completed"]
    assert not bridge.journal.load().reduce().pending
    bridge.close()


@pytest.mark.parametrize("target", [4, 29, 31])
def test_outer_constructor_independently_refuses_unapproved_targets(rig, target):
    rig.options = rig.options.model_copy(update={"stars": target, "project_campaign_max_seconds": 10800})
    with pytest.raises(BrowserSafetyStop):
        make(rig)
    assert rig.calls == []


def test_thirty_profile_authorizes_score_checkpoint_but_defers_submission():
    before = Path("configs/browser_project_two_star.json").read_bytes()
    small = json.loads(before)
    full = json.loads(Path("configs/browser_project_thirty_star.json").read_bytes())
    parsed = parse_run_options(full)
    assert parsed.stars == 30 and parsed.project_campaign and parsed.paused
    assert parsed.project_compact_wire is True
    assert parsed.project_reference_shallow_transits is True
    assert parsed.project_reference_planet_continuation is True
    assert parsed.project_autonomous_decisions is True
    assert parsed.project_supplied_stellar_inputs is True
    assert parsed.project_baseline_edge_reference is True
    assert parsed.project_baseline_band_reference is True
    assert parsed.project_single_event_reference is True
    assert parsed.project_save_strategy == "autosave"
    assert parsed.project_two_event_reference is True
    assert parsed.browser_planet_supplied_evaluation == Path("experiments/planet-supplied-input-transfer-001")
    assert parsed.browser_habitability_supplied_evaluation == Path(
        "experiments/habitability-supplied-input-transfer-001"
    )
    assert parsed.project_campaign_max_seconds == "uncapped"
    assert parsed.project_viewport.model_dump() == {"width": 1600, "height": 1100}
    assert parsed.project_max_seconds == 1800 and parsed.project_max_advances == 512
    assert parsed.artifact_dir == Path("experiments/browser-project-thirty-star")
    differences = {key for key in small.keys() | full.keys() if small.get(key) != full.get(key)}
    assert differences == {
        "stars",
        "project_campaign_max_seconds",
        "artifact_dir",
        "project_allow_scoring",
        "project_allow_submission",
        "project_reference_shallow_transits",
        "project_autonomous_decisions",
        "project_supplied_stellar_inputs",
        "project_baseline_edge_reference",
        "project_baseline_band_reference",
        "project_single_event_reference",
        "project_save_strategy",
        "project_two_event_reference",
        "browser_planet_supplied_evaluation",
        "browser_habitability_supplied_evaluation",
        "browser_config",
    } | ({"project_compact_wire"} if small.get("project_compact_wire") is not True else set())
    assert "allow_submission" not in full and "allow_score_transfer" not in full
    assert parsed.project_allow_scoring is True and parsed.project_allow_submission is False
    assert RunOptions.model_validate(small).project_allow_scoring is False
    assert RunOptions.model_validate(small).project_allow_submission is False
    assert Path("configs/browser_project_two_star.json").read_bytes() == before
    assert RunOptions.model_validate_json(parsed.model_dump_json()) == parsed


def test_prepared_thirty_profile_does_not_bypass_explicit_launch_hold(launcher, monkeypatch, capsys):
    # Reuse the launcher fixture's forbidden prompt/process/browser seams and
    # injected source-check result; this never reruns transfer validation.
    profile = Path("configs/browser_project_thirty_star.json")
    before = profile.read_bytes()
    output = Path(json.loads(before)["artifact_dir"])
    existed = output.exists()
    monkeypatch.setattr(launcher, "parse_run_options", parse_run_options)
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--profile", str(profile)])
    assert exc.value.code == 2
    assert "30-star launch requires --allow-thirty-star" in capsys.readouterr().err
    assert len(launcher.test_source_checks) == 1
    assert launcher.test_source_checks[0].stars == 30
    assert launcher.test_source_checks[0].project_supplied_stellar_inputs is True
    assert profile.read_bytes() == before and output.exists() is existed


def test_all_bounded_three_star_profiles_keep_assessment_and_submission_disabled():
    profiles = list(Path("configs").glob("browser_project*three_star.json"))
    assert profiles
    for path in profiles:
        options = json.loads(path.read_bytes())
        assert options["stars"] == 3
        assert options.get("project_allow_scoring", False) is False, path
        assert options.get("project_allow_submission", False) is False, path
