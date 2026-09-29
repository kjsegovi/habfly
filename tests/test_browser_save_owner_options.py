"""Offline fixed-budget/settlement authorization checks; no native browser."""

import pytest
import test_browser_positive_finalize as positive_fixtures
import test_browser_terrestrial_steps as terrestrial_fixtures

from habfly.browser import BrowserSafetyStop

injected, rig = positive_fixtures.injected, terrestrial_fixtures.rig


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout", 2),
        ("timeout", True),
        ("settle_timeout", 3),
        ("settle_timeout", 2.0),
        ("settle_reserved_notice", True),
        ("settle_reserved_notice", 0),
    ],
)
def test_positive_fixed_save_options_cannot_change_between_steps(injected, field, value):
    state, make, history, _ = injected
    owner = make()
    owner.advance()
    before = (state.reads, state.clicks, state.verifications)
    setattr(owner, field, value)
    owner.advance()
    assert owner.finished and owner.report["outcome"] == "positive_finalization_save_options_changed"
    assert (state.reads, state.clicks, state.verifications) == before
    assert not (history / "owner/save").exists()
    assert not owner.state()["task_completed"]


def test_positive_callback_budget_change_cannot_reach_save_constructor(injected):
    state, make, history, _ = injected

    def callback(event):
        state.events.append(event)
        if event["event"] == "action_proposed":
            state.owner.timeout = 30

    owner = make(emit=callback)
    owner.advance()
    owner.advance()
    assert owner.finished and owner.report["outcome"] == "positive_finalization_save_options_changed"
    assert not state.clicks and not (history / "owner/save").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_seconds", 1801),
        ("max_seconds", 1800.0),
        ("timeout", 30),
        ("timeout", 20.0),
        ("settle_timeout", 30),
        ("settle_timeout", 20.0),
        ("settle_reserved_notice", True),
        ("settle_reserved_notice", 0),
    ],
)
def test_terrestrial_fixed_save_and_owner_budget_cannot_change(rig, field, value):
    owner = rig.make()
    setattr(owner, field, value)
    owner.advance()
    assert owner.finished and owner.failure == "terrestrial_steps_fixed_options_changed"
    assert not rig.state.calls and not rig.state.save_clicks
    assert not (owner.output / "save").exists()
    assert not owner.state()["task_completed"]


def test_terrestrial_callback_cannot_disable_reserved_policy_before_save(rig):
    owners = []

    def callback(event):
        rig.state.events.append(event)
        if owners and event["event"] == "state" and event["payload"].get("phase") == "save":
            owners[0].settle_reserved_notice = False

    owner = rig.make(emit=callback, settle_reserved_notice=True)
    owners.append(owner)
    terrestrial_fixtures.gases(owner)
    terrestrial_fixtures.habitat(owner)
    terrestrial_fixtures.reach(owner, "finished")
    assert owner.finished and owner.failure == "terrestrial_steps_fixed_options_changed"
    assert rig.state.save_clicks == 0 and not (owner.output / "save").exists()


def test_terrestrial_save_cancellation_guard_detects_fixed_option_drift(rig):
    owner = rig.make()
    owner.timeout = 30
    with pytest.raises(BrowserSafetyStop, match="terrestrial_steps_fixed_options_changed"):
        owner._save_cancelled()
    owner.abort()
