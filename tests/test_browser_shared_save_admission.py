"""Pure admission, timing and opt-in forwarding; no browser or scientific task."""

import json
import math
from types import SimpleNamespace

import pytest
import test_browser_positive_finalize as positive_fixtures
import test_browser_save_settlement as settlement_fixtures

import habfly.browser_habitability_actions as menus
import habfly.browser_positive_finalize as positive
import habfly.browser_save_settlement as module
from habfly.browser import BrowserSafetyStop

rig, injected = settlement_fixtures.rig, positive_fixtures.injected


def longer_phase(rig):
    """Declare a sufficient test budget up front; fake time stays monotonic."""
    rig.directory = rig.root / "admission"
    rig.directory.mkdir()
    rig.claim, rig.intent = rig.root / "admission-claim.json", module.intent_options(True, 5)
    claims = {rig.claim: rig.intent, rig.directory / "reserved.json": rig.intent}
    for path, value in claims.items():
        module.persist_json(path, value)

    def check():
        if rig.cancelled or rig.changed:
            raise BrowserSafetyStop("cancelled_or_changed")

    rig.phase = module.ReservedSavePhase(
        rig.directory, seconds=5, claims=claims, check=check, quick_check=check
    )


@pytest.mark.parametrize("remaining", [3.1, 3.09])
def test_insufficient_admission_retains_claim_without_dispatch_and_cannot_reenter(rig, remaining):
    longer_phase(rig)
    rig.samples[:] = [False]
    rig.phase.settle(**rig.callbacks)
    # Fixed fake deadline chosen before admission; no native or clock sleep.
    rig.now = rig.phase.deadline - remaining
    with pytest.raises(BrowserSafetyStop, match="insufficient_readback_budget"):
        rig.phase.admit_dispatch()
    rejected = json.loads((rig.directory / "dispatch-budget-rejected.json").read_bytes())
    assert rejected["longest_guarded_read_seconds"] == pytest.approx(0.1)
    assert rejected["required_seconds"] == pytest.approx(3.1)
    assert rejected["deadline_monotonic_seconds"] == 105
    assert rejected["estimate_not_guarantee"] is True
    assert rejected["deadline_extended"] is rejected["save_click_dispatched"] is False
    assert rig.claim.exists() and not (rig.directory / "dispatch.json").exists()
    with pytest.raises(BrowserSafetyStop, match="attempt_closed"):
        rig.phase.admit_dispatch()


def test_admission_estimate_does_not_extend_deadline_or_write_success_authority(rig):
    longer_phase(rig)
    rig.samples[:] = [False]
    rig.phase.settle(**rig.callbacks)
    deadline = rig.phase.deadline
    rig.now = deadline - 4
    rig.phase.admit_dispatch()
    assert rig.phase.deadline == deadline
    assert not (rig.directory / "dispatch-budget-rejected.json").exists()
    assert not (rig.directory / "dispatch.json").exists()
    # Passing admission is not permission to accept an unexpectedly late read.
    rig.now = deadline
    with pytest.raises(BrowserSafetyStop, match="phase_timeout"):
        rig.phase.click_timeout()


def test_longest_whole_current_callback_is_used_not_last_fast_read(rig):
    longer_phase(rig)
    rig.phase._current(lambda: setattr(rig, "now", rig.now + 0.7))
    rig.phase._current(lambda: setattr(rig, "now", rig.now + 0.1))
    assert rig.phase.longest_current_seconds == pytest.approx(0.7)
    rig.now = rig.phase.deadline - 3.5
    with pytest.raises(BrowserSafetyStop, match="insufficient_readback_budget"):
        rig.phase.admit_dispatch()


@pytest.mark.parametrize("stage", ["admission_quick", "timeout_full"])
def test_slow_final_callback_cannot_use_previous_headroom(rig, stage):
    longer_phase(rig)
    rig.phase._current(lambda: setattr(rig, "now", rig.now + 0.1))

    def slow_check():
        rig.now += 2

    if stage == "admission_quick":
        rig.phase.quick_check = slow_check
        operation = rig.phase.admit_dispatch
    else:
        rig.phase.admit_dispatch()
        # Mimic the actual boundary: a durable dispatch marker precedes the
        # final source check. Rejection must retain it, not authorize retry.
        module.persist_json(rig.directory / "dispatch.json", {"max_clicks": 1})
        rig.phase.full_check = slow_check
        operation = rig.phase.click_timeout
    with pytest.raises(BrowserSafetyStop, match="insufficient_readback_budget"):
        operation()
    assert rig.now < rig.phase.deadline  # headroom, not merely timeout, stopped it
    assert (rig.directory / "dispatch.json").exists() is (stage == "timeout_full")
    assert (rig.directory / "dispatch-budget-rejected.json").exists()
    assert rig.claim.exists()


@pytest.mark.parametrize("value", [0, -1, True, None, math.inf, math.nan])
def test_invalid_or_missing_read_timing_never_dispatches(rig, value):
    rig.phase.longest_current_seconds = value
    with pytest.raises(BrowserSafetyStop, match="invalid_readback_timing"):
        rig.phase.admit_dispatch()
    assert not (rig.directory / "dispatch.json").exists()


@pytest.mark.parametrize("changed", ["cancelled", "changed"])
def test_admission_keeps_source_and_cancellation_checks(rig, changed):
    setattr(rig, changed, True)
    with pytest.raises(BrowserSafetyStop, match="cancelled_or_changed"):
        rig.phase.admit_dispatch()
    assert not (rig.directory / "dispatch.json").exists()


def test_completed_receipt_cannot_adopt_a_dispatch_rejection(rig):
    before, after = settlement_fixtures.complete(rig)
    (rig.directory / "dispatch-budget-rejected.json").write_text("{}")
    with pytest.raises(BrowserSafetyStop, match="dispatch_budget_rejected"):
        settlement_fixtures.validate(rig, before, after)


@pytest.mark.parametrize("optin", [False, True])
def test_positive_pin_and_admission_only_on_explicit_settlement(injected, monkeypatch, optin):
    state, make, history, _ = injected
    original_owner, original_session = positive.PositiveFinalizationSteps, positive.PlanetNumericSession
    observed = []

    def owner(*args, **kwargs):
        return original_owner(*args, **{**kwargs, "settle_reserved_notice": optin, "timeout_seconds": 5})

    def session(*args, **kwargs):
        observed.append(dict(kwargs))
        return original_session(*args, **kwargs)

    monkeypatch.setattr(positive, "PositiveFinalizationSteps", owner)
    monkeypatch.setattr(positive, "PlanetNumericSession", session)
    subject = make()
    subject.advance()
    subject.advance()
    assert subject.state()["phase"] == "verify" and state.clicks == 1
    assert observed == [{"max_seconds": 120, **({"_pin_controls": True} if optin else {})}]
    assert (history / "owner/save/reserved-phase-confirmed.json").exists() is optin
    assert not (history / "owner/save/dispatch-budget-rejected.json").exists()
    subject.close()


def test_positive_admission_rejects_before_dispatch_with_original_reservation(injected, monkeypatch):
    state, make, history, _ = injected
    original_owner = positive.PositiveFinalizationSteps

    def owner(*args, **kwargs):
        return original_owner(*args, **{**kwargs, "settle_reserved_notice": True, "timeout_seconds": 2})

    monkeypatch.setattr(positive, "PositiveFinalizationSteps", owner)
    subject = make()
    subject.advance()
    subject.advance()
    assert subject.finished and state.clicks == 0
    assert subject.report["outcome"] == "reserved_save_insufficient_readback_budget"
    assert not subject.report["save_may_have_occurred"]
    assert (history / "owner/save/reserved.json").exists()
    assert (history / "owner/save/dispatch-budget-rejected.json").exists()
    assert not (history / "owner/save/dispatch.json").exists()
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        make("second-output")


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_habitability_pin_option_is_strict_before_page_access(value):
    with pytest.raises(ValueError, match="Pinned control"):
        menus.HabitabilityMenuSession(None, None, None, _pin_controls=value)


@pytest.mark.parametrize("optin", [False, True])
def test_habitability_menu_forwards_pin_only_on_true(monkeypatch, optin):
    calls = []
    page = SimpleNamespace(
        frames=[], context=SimpleNamespace(pages=[object()]), wait_for_timeout=lambda _: None
    )
    subject = object.__new__(menus.HabitabilityMenuSession)
    subject.stopped, subject.deadline, subject.unexpected_dialog = False, math.inf, False
    subject.page, subject.config, subject.frames, subject._pin_controls = page, object(), [], optin

    def inspect(*args, **kwargs):
        calls.append(kwargs)
        raise BrowserSafetyStop("fixture_inspection_boundary")

    monkeypatch.setattr(menus, "inspect_page", inspect)
    with pytest.raises(BrowserSafetyStop, match="fixture_inspection_boundary"):
        subject.read()
    assert calls == [{"pin_controls": True} if optin else {}]
