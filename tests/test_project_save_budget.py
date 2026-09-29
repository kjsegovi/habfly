"""Explicit fixed Save settling budgets; no browser or network resources."""
# ruff: noqa: F811

import pytest
from test_browser_project_runtime import rig as bridge_rig  # noqa: F401
from test_browser_star_session import complete, create, rig  # noqa: F401
from test_runtime_browser_project import integrated, start  # noqa: F401

import habfly.browser_star_session as module
from habfly.runtime import RunOptions


@pytest.mark.parametrize("seconds", [0.1, 20, 30])
def test_explicit_budget_is_recorded_and_forwarded_once(rig, monkeypatch, seconds):
    session = create(rig, no_planet_save_settle_seconds=seconds)
    calls = []

    def save(*args, **kwargs):
        assert not kwargs["cancelled"]()
        assert kwargs["settle_reserved_notice"] is True
        calls.append(kwargs["settle_timeout_seconds"])
        return {"save_click_delivered": True}

    monkeypatch.setattr(module, "save_no_planet_work", save)
    complete(session)
    assert calls == [seconds]
    assert session.state()["no_planet_save_settle_seconds"] == seconds
    assert session.state()["no_planet_save_pre_dispatch_policy"] == "bounded_read_only_pre_dispatch_settle_v1"
    assert not session.state()["automatic_deadline_increase"]
    session.advance()
    assert calls == [seconds]


def test_legacy_default_stays_twenty_seconds(rig):
    assert create(rig).state()["no_planet_save_settle_seconds"] == 20
    assert RunOptions().browser_no_planet_save_settle_seconds == 20


@pytest.mark.parametrize("value", [True, False, None, "30", 0, 30.1, float("nan"), float("inf")])
def test_invalid_budget_rejected_before_actions(rig, value):
    with pytest.raises(ValueError):
        create(rig, no_planet_save_settle_seconds=value)
    with pytest.raises(ValueError):
        RunOptions(browser_no_planet_save_settle_seconds=value)
    assert not rig.calls and not (rig.root / "run").exists()


def test_abort_during_save_remains_terminal_and_never_schedules_verifier(rig, monkeypatch):
    session = create(rig)

    def save(*args, **kwargs):
        assert not kwargs["cancelled"]()
        session.abort()
        assert kwargs["cancelled"]()
        return {"save_click_delivered": False}

    monkeypatch.setattr(module, "save_no_planet_work", save)
    complete(session)
    assert session.phase == "aborted" and not session.state()["task_completed"]
    assert not any(call[0] == "verify" for call in rig.calls)
    session.advance()
    assert session.phase == "aborted"


def test_runtime_bridge_keeps_explicit_budget_without_launch(integrated):
    bridge = start(integrated, browser_no_planet_save_settle_seconds=30)
    assert bridge.model_options["no_planet_save_settle_seconds"] == 30
    assert bridge.state()["no_planet_save_settle_seconds"] == 30
    assert bridge.state()["observation_start_max_seconds"] == 60
    assert not integrated.calls and not bridge.credentials_consumed


def test_predispatch_window_stop_preserves_cause_without_inventing_star(rig, monkeypatch):
    class FailedWindow:
        def __init__(self, *args, **kwargs):
            self.finished = False

        def advance(self):
            self.finished = True

        def state(self):
            return {
                "star": None,
                "phase": "stopped" if self.finished else "ready_to_start",
                "failure_reason": "planet_copy_time_limit" if self.finished else None,
            }

        def abort(self):
            self.finished = True

    monkeypatch.setattr(module, "PlanetWindowSession", FailedWindow)
    session = create(rig)
    complete(session)
    assert session.failure == "planet_copy_time_limit"
    assert session.phase == "stopped" and not session.state()["task_completed"]
    assert not any(call[0] in {"save", "verify"} for call in rig.calls)
