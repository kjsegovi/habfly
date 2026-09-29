"""Late public cookie dialog: cooperative time, no credentials in evidence."""

import json
from types import SimpleNamespace

import pytest
from test_browser_setup_diagnostics import EMAIL, PASSWORD, Page

import habfly.browser_setup as subject
from habfly.browser_setup import BrowserSetup, SetupStop


@pytest.fixture
def case(monkeypatch, tmp_path):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(subject, "time", SimpleNamespace(monotonic=lambda: clock.now))
    page = Page()
    config = SimpleNamespace(
        url="http://localhost/activity?preview_sequence_id=fixture-only",
        allows=lambda url: url == "http://localhost/activity?preview_sequence_id=fixture-only",
    )
    setup = BrowserSetup(page, config, (EMAIL, PASSWORD), emit=page.emit, output=tmp_path)
    page.setup = setup
    yield SimpleNamespace(page=page, setup=setup, clock=clock, output=tmp_path)
    setup.close()


def test_settling_has_no_sleep_or_credential_write_and_keeps_absolute_budget(case):
    p, setup, clock = case.page, case.setup, case.clock
    assert setup.advance() == "waiting" and setup.stage == "waiting_for_login_ui"
    for tick in (0.1, 0.3, 0.75, 1.49):
        clock.now = 100 + tick
        assert setup.advance() == "waiting"
        assert p.fills == p.clicks == []
    clock.now = 101.5
    setup.advance()
    assert p.fills == [("email", 3000), ("password", 3000)]
    assert p.clicks == [("submit", 3000)] and setup.submitted_login
    assert setup.started == 100 and setup.MAX_SECONDS == 90
    clock.now = 191
    with pytest.raises(SetupStop, match="^setup_time_limit$"):
        setup.advance()
    assert p.clicks == [("submit", 3000)]
    assert not list(case.output.iterdir())
    assert EMAIL not in json.dumps(p.events) and PASSWORD not in json.dumps(p.events)


def test_late_cookie_resets_settlement_before_any_credentials(case):
    p, setup, clock = case.page, case.setup, case.clock
    setup.advance()
    clock.now, p.cookie = 101.07, True
    setup.advance()
    assert setup.stage == "cookie_notice_closed" and setup.login_ui_since is None
    assert p.fills == [] and p.clicks == [("cookie_close", 3000)]
    p.cookie = False
    setup.advance()
    clock.now = 102.56
    setup.advance()
    assert not p.fills
    clock.now = 102.58
    setup.advance()
    assert p.clicks == [("cookie_close", 3000), ("submit", 3000)]


def test_cookie_arriving_during_password_fill_closes_before_first_submit(case):
    p, setup, clock = case.page, case.setup, case.clock
    original = p.trip

    def arrive_after_fill(call):
        original(call)
        if call == "fill" and setup._login_operation == "fill_password":
            p.cookie = True

    p.trip = arrive_after_fill
    setup.advance()
    clock.now = 101.5
    setup.advance()
    assert p.fills == [("email", 3000), ("password", 3000)]
    assert p.clicks == [("cookie_close", 3000)] and not setup.submitted_login
    assert setup.login_ui_since is None
    p.trip, p.cookie = original, False
    setup.advance()
    clock.now = 103.01
    setup.advance()
    assert p.clicks == [("cookie_close", 3000), ("submit", 3000)]
    setup.advance()
    assert p.clicks == [("cookie_close", 3000), ("submit", 3000)]


def test_cookie_arriving_during_email_fill_prevents_password_fill(case):
    p, setup, clock = case.page, case.setup, case.clock
    original = p.trip

    def arrive_after_email(call):
        original(call)
        if call == "fill" and setup._login_operation == "fill_email":
            p.cookie = True

    p.trip = arrive_after_email
    setup.advance()
    clock.now = 101.5
    setup.advance()
    assert p.fills == [("email", 3000)]
    assert p.clicks == [("cookie_close", 3000)] and not setup.submitted_login


@pytest.mark.parametrize("known", [False, True])
def test_signing_callback_overlay_prevents_first_fill(case, monkeypatch, known):
    p, setup, clock = case.page, case.setup, case.clock
    exposed = True

    def emit(kind, payload):
        nonlocal exposed
        p.events.append((kind, payload))
        if payload["setup_stage"] == "signing_in":
            p.cookie, exposed = known, False

    setup.emit = emit
    monkeypatch.setattr(subject, "rendered_control", lambda _: exposed)
    setup.advance()
    clock.now = 101.5
    if known:
        setup.advance()
        assert p.clicks == [("cookie_close", 3000)] and setup.login_ui_since is None
    else:
        setup.advance()  # A non-modal cover may settle, but no fill is allowed.
        clock.now += 3
        with pytest.raises(SetupStop, match="^setup_login_control_not_exposed$"):
            setup.advance()
        assert p.clicks == []
    assert p.fills == []


@pytest.mark.parametrize("phase", ["waiting_for_login_ui", "signing_in", "cookie_notice_closed"])
def test_callback_abort_prevents_subsequent_writes(case, phase):
    p, setup, clock = case.page, case.setup, case.clock

    def emit(kind, payload):
        p.events.append((kind, payload))
        if payload["setup_stage"] == phase:
            setup.close()

    setup.emit = emit
    if phase == "cookie_notice_closed":
        p.cookie = True
        setup.advance()  # Closing the known notice precedes this callback.
    elif phase == "signing_in":
        setup.advance()
        clock.now = 101.5
    with pytest.raises(SetupStop, match="^setup_closed$"):
        setup.advance()
    assert not p.fills
    assert p.clicks == ([("cookie_close", 3000)] if phase == "cookie_notice_closed" else [])


@pytest.mark.parametrize("late", [False, True])
def test_unknown_occluder_stops_without_submit(case, monkeypatch, late):
    p, setup, clock = case.page, case.setup, case.clock
    if late:
        setup.advance()
        clock.now = 101.5
    monkeypatch.setattr(subject, "rendered_control", lambda _: False)
    assert setup.advance() == "waiting"
    assert p.fills == p.clicks == []
    clock.now += 3
    with pytest.raises(SetupStop, match="^setup_login_control_not_exposed$"):
        setup.advance()
    assert p.fills == p.clicks == [] and setup.closed


def test_nonmodal_backdrop_then_known_heading_requires_fresh_clean_interval(case, monkeypatch):
    p, setup, clock = case.page, case.setup, case.clock
    exposed = True
    monkeypatch.setattr(subject, "rendered_control", lambda _: exposed)
    setup.advance()
    clock.now, exposed = 100.6, False
    assert setup.advance() == "waiting"
    assert setup.login_ui_since is None and not p.fills and not p.clicks
    clock.now, p.cookie = 100.77, True
    setup.advance()
    assert p.clicks == [("cookie_close", 3000)] and not p.fills
    exposed = True
    setup.advance()
    clock.now = 102.26
    setup.advance()
    assert not p.fills
    clock.now = 102.28
    setup.advance()
    assert p.clicks == [("cookie_close", 3000), ("submit", 3000)]


def test_unknown_cover_after_email_does_not_get_settling_allowance(case, monkeypatch):
    p, setup, clock = case.page, case.setup, case.clock
    original = p.trip
    exposed = True

    def fill_then_cover(call):
        nonlocal exposed
        original(call)
        if call == "fill" and setup._login_operation == "fill_email":
            exposed = False

    p.trip = fill_then_cover
    monkeypatch.setattr(subject, "rendered_control", lambda _: exposed)
    setup.advance()
    clock.now = 101.5
    with pytest.raises(SetupStop, match="^setup_login_control_not_exposed$"):
        setup.advance()
    assert p.fills == [("email", 3000)] and not p.clicks and setup.closed


def test_pending_cookie_close_is_one_shot_until_disappearance(case):
    p, setup, clock = case.page, case.setup, case.clock
    p.cookie = True
    setup.advance()
    p.cookie = True  # Model the same retained node during its dismissal animation.
    for now in (100.05, 100.1, 100.3, 100.5):
        clock.now = now
        assert setup.advance() == "waiting"
        assert p.clicks == [("cookie_close", 3000)] and not p.fills
    p.cookie = False
    setup.advance()
    clock.now = 102.01
    setup.advance()
    assert p.clicks == [("cookie_close", 3000), ("submit", 3000)]


def test_pending_cookie_never_disappears_stops_without_another_close(case):
    p, setup, clock = case.page, case.setup, case.clock
    p.cookie = True
    setup.advance()
    p.cookie = True
    clock.now += 3
    with pytest.raises(SetupStop, match="^setup_cookie_notice_close_unresolved$"):
        setup.advance()
    assert setup.closed and p.clicks == [("cookie_close", 3000)] and not p.fills


def test_cookie_replacement_cannot_substitute_for_pending_notice(case):
    from test_browser_setup_diagnostics import Control

    p, setup = case.page, case.setup
    p.cookie = True
    setup.advance()
    p.cookie = True
    setup.login_cookie_closing["notice"] = Control(p, "replaced")
    with pytest.raises(SetupStop, match="^setup_cookie_notice_changed_during_close$"):
        setup.advance()
    assert p.clicks == [("cookie_close", 3000)] and not p.fills


def test_close_handle_replacement_is_rejected_before_dispatch(case, monkeypatch):
    from test_browser_setup_diagnostics import Control

    p, setup = case.page, case.setup
    original = Control.element_handle

    def substituted(control, *, timeout):
        return (
            Control(p, "replaced_close")
            if control.name == "cookie_close"
            else original(control, timeout=timeout)
        )

    monkeypatch.setattr(Control, "element_handle", substituted)
    p.cookie = True
    with pytest.raises(SetupStop, match="^setup_cookie_notice_changed_before_close$"):
        setup.advance()
    assert p.clicks == p.fills == []


def test_unknown_modal_during_cookie_animation_stops_immediately(case):
    from test_browser_setup_diagnostics import Controls

    p, setup = case.page, case.setup
    p.cookie = True
    setup.advance()
    p.cookie = True
    original = p.get_by_role
    modal = SimpleNamespace(is_visible=lambda: True, evaluate=lambda *_: False)

    def dialogs(role, *, name=None, exact=False):
        return (
            Controls(p, [modal])
            if role == "dialog" and name is None
            else original(role, name=name, exact=exact)
        )

    p.get_by_role = dialogs
    with pytest.raises(SetupStop, match="^setup_unexpected_modal$"):
        setup.advance()
    assert p.clicks == [("cookie_close", 3000)] and not p.fills


def test_new_dialog_enclosing_original_notice_is_still_unknown(case):
    from test_browser_setup_diagnostics import Controls

    p, setup = case.page, case.setup
    p.cookie = True
    setup.advance()
    p.cookie = True
    original = p.get_by_role
    # It could contain the original notice, but was not one of its pinned dialogs.
    modal = SimpleNamespace(is_visible=lambda: True, evaluate=lambda source, _: "contains" in source)

    def dialogs(role, *, name=None, exact=False):
        if role == "dialog" and name is None:
            return Controls(p, [modal])
        return original(role, name=name, exact=exact)

    p.get_by_role = dialogs
    with pytest.raises(SetupStop, match="^setup_unexpected_modal$"):
        setup.advance()
    assert p.clicks == [("cookie_close", 3000)] and not p.fills


def test_known_cookie_animation_still_obeys_absolute_setup_deadline(case):
    p, setup, clock = case.page, case.setup, case.clock
    clock.now, p.cookie = 189, True
    setup.advance()
    p.cookie, clock.now = True, 190.01
    with pytest.raises(SetupStop, match="^setup_time_limit$"):
        setup.advance()
    assert p.clicks == [("cookie_close", 3000)] and not p.fills


def test_known_notice_arriving_during_exposure_is_closed_not_ignored(case, monkeypatch):
    p, setup = case.page, case.setup

    def covered(_):
        p.cookie = True
        return False

    monkeypatch.setattr(subject, "rendered_control", covered)
    assert setup.advance() == "waiting"
    assert p.fills == [] and p.clicks == [("cookie_close", 3000)]
    assert setup.login_ui_since is None and not setup.submitted_login


def test_form_changed_while_settling_stops_before_credentials(case):
    p, setup, clock = case.page, case.setup, case.clock
    setup.advance()
    p.form_action = "http://outside.invalid/not-allowed"
    clock.now = 101.5
    with pytest.raises(SetupStop, match="^setup_untrusted_login_form$"):
        setup.advance()
    assert p.fills == p.clicks == []


def test_submit_timeout_remains_terminal_no_recovery(case):
    from playwright.sync_api import TimeoutError

    p, setup, clock = case.page, case.setup, case.clock
    setup.advance()
    clock.now = 101.5
    p.fail_operation = "submit_click"
    p.failure = TimeoutError("- <omitted> intercepts pointer events")
    with pytest.raises(SetupStop, match="_at_login_pointer_interception$"):
        setup.advance()
    assert len(p.clicks) == 1 and setup.closed and not setup.submitted_login
    with pytest.raises(SetupStop, match="^setup_closed$"):
        setup.advance()
    assert len(p.clicks) == 1
