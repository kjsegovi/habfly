"""One-shot post-login refresh; every request is a disposable local fixture."""

# ruff: noqa: F811 - pytest fixtures are imported for discovery and used as parameters.

import json
import time

import pytest
from test_browser_setup import (  # noqa: F401 - shared fully intercepted native fixtures
    EMAIL,
    OUTER,
    PASSWORD,
    begin,
    chromium,
    finish,
    setup_page,
)

from habfly.browser_setup import SetupStop


def authenticated_preview(setup):
    for _ in range(10):
        if setup.submitted_login and setup.page.url == OUTER:
            assert not setup.visited and not setup.post_login_refresh_attempted
            return
        setup.advance()
        setup.page.wait_for_timeout(20)
    pytest.fail("Fixture did not reach the authenticated preview")


@pytest.mark.parametrize("landing", [False, True])
def test_login_refresh_removes_welcome_once_preserving_session_and_untouched_star(
    setup_page, tmp_path, landing
):
    page, state = setup_page
    state["landing_after_login"] = landing
    events = []
    setup = begin(page, tmp_path, events)
    authenticated_preview(setup)
    context = page.context
    assert page.get_by_role("alert").inner_text() == "Welcome back!"
    assert setup.advance() == "waiting"  # One refresh, no intro/star action.
    assert setup.post_login_refresh_attempted and not setup.post_login_refresh_verified
    assert setup.stage == "refreshing_authenticated_preview"
    assert not setup.visited and not setup.star_clicked
    assert page.get_by_role("alert").count() == 0
    assert setup.advance() == "waiting"  # A separate read verifies the result.
    assert setup.post_login_refresh_verified and not setup.visited
    finish(setup)
    assert state["posts"] == 1 and state["preview_loads"] == 2
    assert setup.page is page and page.context is context and context.pages == [page]
    stages = [payload["setup_stage"] for _, payload in events if "setup_stage" in payload]
    assert stages.count("refreshing_authenticated_preview") == 1
    assert stages.index("post_login_refresh_verified") < stages.index("intro_screen_1")
    assert setup._email == setup._password == ""
    assert EMAIL not in json.dumps(events) and PASSWORD not in json.dumps(events)
    assert OUTER not in json.dumps(events)


def test_already_authenticated_preview_is_not_refreshed(setup_page, tmp_path):
    page, state = setup_page
    state["authenticated"] = True
    setup = begin(page, tmp_path)
    finish(setup)
    assert state["posts"] == 0 and state["preview_loads"] == 1
    assert not setup.post_login_refresh_attempted and not setup.post_login_refresh_verified


@pytest.mark.parametrize("failure", ["welcome", "authentication"])
def test_refresh_failure_stops_before_intro_without_retry(setup_page, tmp_path, failure):
    page, state = setup_page
    state["persist_welcome"] = failure == "welcome"
    state["logout_on_refresh"] = failure == "authentication"
    events = []
    setup = begin(page, tmp_path, events)
    authenticated_preview(setup)
    reason = (
        "setup_welcome_back_persisted_after_refresh"
        if failure == "welcome"
        else "setup_authentication_lost_after_refresh"
    )
    with pytest.raises(SetupStop, match=f"^{reason}$"):
        setup.advance()
        setup.advance()
    assert setup.closed and not setup.visited and not setup.star_clicked
    assert not setup.post_login_refresh_verified and setup._email == setup._password == ""
    assert not list(tmp_path.iterdir())
    with pytest.raises(SetupStop, match="^setup_closed$"):
        setup.advance()
    assert state["posts"] == 1 and state["preview_loads"] == 2
    assert EMAIL not in json.dumps(events) and PASSWORD not in json.dumps(events)


@pytest.mark.parametrize(
    "markup,visible",
    [
        ('<span style="pointer-events:none">Welcome back!</span>', True),
        ("Welcome <span>back</span>!", True),
        ('<span style="opacity:0">Welcome back!</span>', False),
        ('<span aria-hidden="true">Welcome back!</span>', False),
        (
            (
                '<span style="display:inline-block;position:relative">Welcome back!'
                '<span style="position:absolute;inset:0;background:white"></span></span>'
            ),
            False,
        ),
    ],
)
def test_welcome_check_uses_painted_text_not_clickability(setup_page, tmp_path, markup, visible):
    page, state = setup_page
    state.update(welcome_back=markup, persist_welcome=True)
    setup = begin(page, tmp_path)
    authenticated_preview(setup)
    setup.advance()
    if visible:
        with pytest.raises(SetupStop, match="^setup_welcome_back_persisted_after_refresh$"):
            setup.advance()
        assert not setup.post_login_refresh_verified
    else:
        setup.advance()
        assert setup.post_login_refresh_verified
    assert not setup.visited and state["preview_loads"] == 2
    setup.close()


def test_late_authentication_loss_after_refresh_never_submits_again(setup_page, tmp_path):
    page, state = setup_page
    setup = begin(page, tmp_path)
    authenticated_preview(setup)
    setup.advance()
    page.goto(setup.login_url)
    with pytest.raises(SetupStop, match="^setup_authentication_lost_after_refresh$"):
        setup.advance()
    assert state["posts"] == 1 and setup.closed and not setup.post_login_refresh_verified


@pytest.mark.parametrize("effect", ["abort", "deadline", "popup", "modal", "navigation", "callback_error"])
def test_refresh_rechecks_after_callback_before_dispatch(setup_page, tmp_path, effect):
    page, state = setup_page
    setup = begin(page, tmp_path)
    authenticated_preview(setup)

    def changed(kind, payload):
        if payload.get("setup_stage") != "refreshing_authenticated_preview":
            return
        if effect == "abort":
            setup.close()
        elif effect == "deadline":
            setup.started = time.monotonic() - setup.MAX_SECONDS - 1
        elif effect == "popup":
            page.context.new_page()
        elif effect == "modal":
            page.evaluate("document.body.insertAdjacentHTML('beforeend', '<div role=dialog>Unknown</div>')")
        elif effect == "navigation":
            page.goto(setup.landing_url)
        else:
            raise RuntimeError(PASSWORD)

    setup.emit = changed
    with pytest.raises(SetupStop) as stopped:
        setup.advance()
    assert PASSWORD not in str(stopped.value)
    assert setup.closed and setup.post_login_refresh_attempted and not setup.post_login_refresh_verified
    assert state["preview_loads"] == 1 and state["posts"] == 1
    assert not setup.visited and not list(tmp_path.iterdir())


def test_refresh_uses_remaining_setup_budget_and_never_retries_timeout(setup_page, tmp_path, monkeypatch):
    page, state = setup_page
    setup = begin(page, tmp_path)
    authenticated_preview(setup)
    setup.started = time.monotonic() - setup.MAX_SECONDS + 2
    started = setup.started
    calls = []

    def timeout(**kwargs):
        calls.append(kwargs)
        raise TimeoutError(PASSWORD)

    monkeypatch.setattr(page, "reload", timeout)
    with pytest.raises(SetupStop, match="^setup_post_login_refresh_failed$"):
        setup.advance()
    assert len(calls) == 1 and calls[0]["wait_until"] == "domcontentloaded"
    assert 0 < calls[0]["timeout"] <= 2000 and setup.started == started
    with pytest.raises(SetupStop, match="^setup_closed$"):
        setup.advance()
    assert len(calls) == 1 and state["preview_loads"] == 1
    assert setup._email == setup._password == "" and not list(tmp_path.iterdir())


def test_refresh_never_resets_activity_started_by_this_setup(setup_page, tmp_path):
    page, state = setup_page
    setup = begin(page, tmp_path)
    authenticated_preview(setup)
    setup.visited.add(0)
    with pytest.raises(SetupStop, match="^setup_post_login_refresh_after_activity$"):
        setup.advance()
    assert state["preview_loads"] == 1 and not setup.post_login_refresh_attempted
