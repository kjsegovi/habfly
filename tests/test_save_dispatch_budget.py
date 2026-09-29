"""No-Save admission: reject a predictably late write without extending time."""
# ruff: noqa: F401,F811

import json
from types import SimpleNamespace

import pytest
from test_browser_no_planet_save import no_save_page, save
from test_browser_numeric import chromium, config, page
from test_browser_planet_window_choice import window_page

import habfly.browser_no_planet_save as module
from habfly.browser import BrowserSafetyStop


@pytest.mark.parametrize("remaining", [0.1, 5.393, 9.5])
def test_known_slow_read_stops_before_dispatch_at_same_deadline(tmp_path, monkeypatch, remaining):
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    session = SimpleNamespace(longest_current_seconds=6.5)
    with pytest.raises(BrowserSafetyStop, match="insufficient_readback_budget"):
        module._check_dispatch_readback_budget(session, tmp_path, 100 + remaining)
    record = json.loads((tmp_path / "dispatch-budget-rejected.json").read_bytes())
    assert record["required_seconds"] == 9.5
    assert record["remaining_seconds"] == pytest.approx(remaining)
    assert record["deadline_monotonic_seconds"] == 100 + remaining
    assert record["estimate_not_guarantee"]
    for name in ("deadline_extended", "save_click_dispatched", "automatic_retry", "task_completed"):
        assert record[name] is False
    assert not (tmp_path / "dispatch.json").exists()


def test_sufficient_time_does_not_create_new_receipt_or_extend_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(module.time, "monotonic", lambda: 104.0)
    module._check_dispatch_readback_budget(SimpleNamespace(longest_current_seconds=2.5), tmp_path, 120.0)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("duration", [0, -1, True, None, "1", float("nan"), float("inf")])
def test_invalid_latency_estimates_never_authorize_save(tmp_path, monkeypatch, duration):
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    with pytest.raises(BrowserSafetyStop, match="invalid_readback_timing"):
        module._check_dispatch_readback_budget(
            SimpleNamespace(longest_current_seconds=duration), tmp_path, 120
        )
    assert not list(tmp_path.iterdir())


def test_intercepted_slow_read_budget_retains_claim_without_click_or_retry(
    no_save_page, tmp_path, monkeypatch
):
    browser_page, frame, options = no_save_page
    original = module.PlanetNumericSession.current

    def measured_slow_read(session):
        result = original(session)
        # Inject a measured duration, not a delay or an extra allowance.
        session.longest_current_seconds = 20.0
        return result

    monkeypatch.setattr(module.PlanetNumericSession, "current", measured_slow_read)
    with pytest.raises(BrowserSafetyStop, match="insufficient_readback_budget"):
        save(no_save_page, tmp_path, settle_reserved_notice=True)
    assert frame.evaluate("window.fixtureSaves") == 0
    stopped = json.loads((tmp_path / "save/stopped.json").read_bytes())
    assert stopped["reservation_created"] is True
    assert stopped["save_may_have_occurred"] is False
    assert not (tmp_path / "save/dispatch.json").exists()
    assert not (tmp_path / "save/confirmed.json").exists()
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        module.save_no_planet_work(
            browser_page, config(), tmp_path / "other", **options, settle_reserved_notice=True
        )
    assert frame.evaluate("window.fixtureSaves") == 0


def test_intercepted_final_callback_cannot_spend_previously_admitted_readback_budget(
    no_save_page, tmp_path, monkeypatch
):
    _, frame, _ = no_save_page
    original_clock = module.time.monotonic
    original_admit = module._check_dispatch_readback_budget
    offset = 0.0
    changed = False
    admitted_deadline = None

    def admit(session, directory, deadline):
        nonlocal admitted_deadline
        original_admit(session, directory, deadline)
        admitted_deadline = deadline

    def cancelled():
        nonlocal offset, changed
        if (tmp_path / "save/dispatch.json").is_file() and not changed:
            assert admitted_deadline is not None
            offset, changed = admitted_deadline - original_clock() - 2.0, True
        return False

    monkeypatch.setattr(module, "_check_dispatch_readback_budget", admit)
    monkeypatch.setattr(module.time, "monotonic", lambda: original_clock() + offset)
    try:
        with pytest.raises(BrowserSafetyStop, match="insufficient_readback_budget"):
            save(no_save_page, tmp_path, settle_reserved_notice=True, cancelled=cancelled)
    finally:
        offset = 0.0
    assert changed and frame.evaluate("window.fixtureSaves") == 0
    assert (tmp_path / "save/dispatch.json").is_file()
    assert (tmp_path / "save/dispatch-budget-rejected.json").is_file()
    stopped = json.loads((tmp_path / "save/stopped.json").read_bytes())
    # The durable pre-click dispatch marker remains conservative even though
    # this fixture independently observed zero native clicks. No retry allowed.
    assert stopped["reservation_created"] is True and stopped["save_may_have_occurred"] is True
    assert stopped["automatic_retry"] is False and not (tmp_path / "save/confirmed.json").exists()
