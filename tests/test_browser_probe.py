"""Read-only preflight exercised in Chromium with all network requests intercepted."""

import hashlib
import json
import time

import pytest
from playwright.sync_api import sync_playwright

from habfly.browser import BrowserSafetyStop
from habfly.browser_probe import (
    BrowserProbeConfig,
    ProbeFrame,
    boundary_diagnostic,
    capture_interactively,
    inspect_page,
    public_url,
    save_probe,
)

OUTER = "http://localhost/activity?preview_sequence_id=q%3A123%3A946"
SIM = "https://frames.fixture.invalid/simulation"
WIDGET = "https://frames.fixture.invalid/widget"


def config(**fields):
    return BrowserProbeConfig(
        url=OUTER,
        frames=[
            ProbeFrame(name="simulation", url=SIM, required_text=["OBSERVATIONS", "CRABILTIA"]),
            ProbeFrame(name="widgets", url=WIDGET, count=2),
        ],
        **fields,
    )


@pytest.fixture(scope="module")
def chromium():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(chromium):
    context = chromium.new_context()
    page = context.new_page()
    pages = {
        OUTER: f'''<h1>Project</h1>
            <label><input type=checkbox>I am ready to submit project.</label>
            <iframe src="{SIM}"></iframe>
            <iframe src="{WIDGET}"></iframe><iframe src="{WIDGET}"></iframe>
            <iframe src="https://unrelated.fixture.invalid/widget?secret=do-not-record"></iframe>
            <iframe hidden src="{SIM}"></iframe>''',
        SIM: """<h1>OBSERVATIONS: CRABILTIA</h1>
            <p>PARALLAX 0.032; Peak wavelength 212 nm; Flux 5.15E-13 W/m²</p>
            <label>Distance (ly)<input value="0"></label>
            <label>Color<select><option>IR</option><option>UV</option></select></label>
            <button>Save</button><button>ASSESS</button>
            <div hidden>Private answer 999<input type=password value=not-recorded></div>""",
        WIDGET: "<button>SUBMIT PROJECT</button>",
        "https://unrelated.fixture.invalid/widget?secret=do-not-record": "<p>Unrelated private text</p>",
    }
    context.route(
        "**/*",
        lambda route: (
            route.fulfill(status=200, content_type="text/html", body=pages[route.request.url])
            if route.request.url in pages
            else route.abort()
        ),
    )
    page.goto(OUTER, wait_until="load")
    try:
        yield page
    finally:
        context.close()


def test_preview_query_is_pinned_and_only_explicit_query_is_supported():
    settings = config()
    assert settings.allows(OUTER)
    assert settings.allows(OUTER.replace("%3A", ":"))
    for url in (
        OUTER.replace("123", "124"),
        OUTER + "&secret=x",
        OUTER + "&preview_sequence_id=other",
        OUTER + "#outside",
        OUTER.replace("localhost", "example.com"),
        OUTER.replace("/activity", "/activity/next"),
        OUTER.replace("/activity", "/activity/%2e%2e/escape"),
        OUTER.replace("localhost", "user:secret@localhost"),
    ):
        assert not settings.allows(url)
    with pytest.raises(ValueError):
        BrowserProbeConfig(url=OUTER, frames=settings.frames, mode="delivery")
    with pytest.raises(ValueError):
        config(allow_submission=True)
    with pytest.raises(ValueError):
        ProbeFrame(name="bad", url=SIM + "?answer=1")
    assert (
        public_url("https://user:secret@example.com/path?token=secret#fragment") == "https://example.com/path"
    )


def test_read_only_frame_inventory_excludes_hidden_and_unknown_content(page, tmp_path):
    before = page.frames[1].locator("input").first.input_value()
    report = inspect_page(page, config())
    assert report["actions_executed"] == 0
    assert not report["browser_acceptance_passed"]
    assert report["outer_url"] == "http://localhost/activity"
    assert [frame["id"] for frame in report["frames"]] == ["simulation-0", "widgets-0", "widgets-1"]
    encoded = json.dumps(report)
    for text in (
        "Private answer",
        "not-recorded",
        "Unrelated private text",
        "do-not-record",
        "preview_sequence_id",
    ):
        assert text not in encoded
    assert "0.032" in encoded and "5.15E-13" in encoded
    assert page.frames[1].locator("input").first.input_value() == before
    assert not page.get_by_role("checkbox").is_checked()
    textbox = next(c for c in report["frames"][0]["controls"] if c["role"] == "textbox")
    assert textbox["value"] == "0"
    assert report["outer_controls"][0]["protected"]
    assert all(c["actions"] == [] for f in report["frames"] for c in f["controls"])
    assert all(f["controls"][0]["protected"] for f in report["frames"][1:])
    manifest = save_probe(report, tmp_path / "capture")
    raw = (tmp_path / "capture" / "observation.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["observation_sha256"]
    with pytest.raises(FileExistsError):
        save_probe(report, tmp_path / "capture")


@pytest.mark.parametrize("in_frame", [False, True])
def test_login_and_modal_fail_before_capture(page, in_frame):
    frame = page.frames[1] if in_frame else page.main_frame
    frame.locator("body").evaluate("node => node.insertAdjacentHTML('beforeend', '<input type=password>')")
    with pytest.raises(BrowserSafetyStop, match="authentication_required"):
        inspect_page(page, config())
    frame.locator("input[type=password]").last.evaluate("node => node.remove()")
    frame.locator("body").evaluate(
        "node => node.insertAdjacentHTML('beforeend', '<div role=dialog>Unexpected</div>')"
    )
    with pytest.raises(BrowserSafetyStop, match="unexpected_modal"):
        inspect_page(page, config())


def test_missing_frame_and_hidden_frame_fail_closed(page):
    page.locator("iframe").nth(1).evaluate("node => node.remove()")
    with pytest.raises(BrowserSafetyStop, match="frame_count_mismatch:widgets"):
        inspect_page(page, config())


def test_frame_loading_and_wrong_screen_are_not_ready(page):
    page.frames[1].get_by_role("heading").evaluate("node => node.textContent='Loading'")
    with pytest.raises(BrowserSafetyStop, match="frame_not_ready:simulation"):
        inspect_page(page, config())


def test_navigation_escape_is_not_captured(page):
    page.evaluate("history.replaceState(null, '', '/outside')")
    with pytest.raises(BrowserSafetyStop, match="navigation_outside_activity"):
        inspect_page(page, config())


@pytest.mark.parametrize(
    "url,difference",
    [
        ("http://localhost/authors/log_in", "path"),
        (OUTER.replace("123", "124"), "preview_sequence_id"),
        (OUTER.replace("localhost", "127.0.0.1"), "host_or_port"),
        (OUTER + "#fragment", "unsupported_url_components"),
        ("http://user:secret@localhost/activity?token=secret", "unsupported_url_components"),
        ("http://[", "unsupported_url_components"),
    ],
)
def test_boundary_diagnostic_is_specific_but_redacts_secrets(url, difference):
    diagnostic = boundary_diagnostic(config(), url)
    assert difference in diagnostic["different_components"]
    encoded = json.dumps(diagnostic)
    for value in ("secret", "q%3A", "q:123", "q:124", "#fragment", "token="):
        assert value not in encoded
    assert not config().allows(url)


def test_manual_return_after_login_succeeds_without_repinning(page):
    page.evaluate("history.replaceState(null, '', '/authors/log_in')")
    messages, confirmations = [], []
    settings = config()

    def confirm(prompt):
        confirmations.append(prompt)
        if len(confirmations) == 2:
            # Simulate the HUMAN returning to the original URL. The probe never navigates.
            page.evaluate("url => history.replaceState(null, '', url)", OUTER)
        return True

    report = capture_interactively(page, settings, confirm=confirm, echo=messages.append)
    assert len(confirmations) == 2
    assert report["actions_executed"] == 0
    assert settings.url == OUTER
    assert any("/authors/log_in" in message for message in messages)
    assert not page.get_by_role("checkbox").is_checked()


def test_independent_setup_is_explicitly_recorded_without_probe_actions(page):
    report = capture_interactively(
        page, config(), confirm=lambda _: True, echo=lambda _: None, new_test_session=True
    )
    assert report["setup_mode"] == "independent_test_session"
    assert report["actions_executed"] == 0
    assert not page.get_by_role("checkbox").is_checked()


def test_manual_navigation_while_terminal_prompt_blocks_driver_is_refreshed(page):
    page.evaluate("history.replaceState(null, '', '/authors/log_in')")
    confirmations, messages = [], []

    def confirm(prompt):
        confirmations.append(prompt)
        if len(confirmations) == 1:
            # Browser changes while the synchronous terminal prompt is waiting.
            # Do not pump Playwright after scheduling this until confirm returns.
            page.evaluate("url => {setTimeout(() => history.replaceState(null, '', url), 50)}", OUTER)
            time.sleep(0.2)
            assert page.url.endswith("/authors/log_in")  # Cached until the next protocol round trip.
        return True

    report = capture_interactively(page, config(), confirm=confirm, echo=messages.append)
    assert len(confirmations) == 1
    assert report["actions_executed"] == 0
    assert page.url == OUTER
    assert not messages


def test_cached_allowed_url_cannot_hide_navigation_away_during_prompt(page):
    confirmations, messages = [], []

    def confirm(prompt):
        confirmations.append(prompt)
        if len(confirmations) > 1:
            return False
        page.evaluate("() => {setTimeout(() => history.replaceState(null, '', '/authors/log_in'), 50)}")
        time.sleep(0.2)
        assert page.url == OUTER  # Stale, apparently allowed URL must not authorize capture.
        return True

    assert capture_interactively(page, config(), confirm=confirm, echo=messages.append) is None
    assert page.url.endswith("/authors/log_in")
    assert "navigation_outside_activity" in " ".join(messages)


def test_changed_preview_never_silently_becomes_allowed(page):
    page.evaluate("url => history.replaceState(null, '', url)", OUTER.replace("123", "124"))
    messages, confirmations = [], []

    def confirm(prompt):
        confirmations.append(prompt)
        return True

    with pytest.raises(BrowserSafetyStop, match="capture_attempt_limit"):
        capture_interactively(page, config(), confirm=confirm, echo=messages.append)
    assert len(confirmations) == 3
    assert "preview_sequence_id" in " ".join(messages)
    assert "q%3A123" not in " ".join(messages)
    assert "q%3A124" not in " ".join(messages)
    assert not config().allows(page.url)


def test_cancel_does_not_read_or_navigate_page():
    class UnreadablePage:
        def __getattr__(self, name):
            raise AssertionError("Cancellation must not access the page")

    assert (
        capture_interactively(UnreadablePage(), config(), confirm=lambda _: False, echo=lambda _: None)
        is None
    )


def test_observation_budget_failure_is_not_retried(page):
    confirmations = []

    def confirm(prompt):
        confirmations.append(prompt)
        return True

    with pytest.raises(BrowserSafetyStop, match="observation_budget_exceeded"):
        capture_interactively(page, config(max_text_chars=100), confirm=confirm, echo=lambda _: None)
    assert len(confirmations) == 1


def test_snapshot_budget_does_not_silently_truncate(page):
    with pytest.raises(BrowserSafetyStop, match="observation_budget_exceeded"):
        inspect_page(page, config(max_text_chars=100))


def test_nested_unlisted_frame_is_not_read(page):
    page.frames[1].locator("body").evaluate(
        """node => node.insertAdjacentHTML('beforeend',
        '<iframe src="https://unrelated.fixture.invalid/widget?secret=do-not-record"></iframe>')"""
    )
    page.frames[1].locator("iframe").content_frame.get_by_text("Unrelated private text").wait_for()
    report = inspect_page(page, config())
    assert "Unrelated private text" not in json.dumps(report)


def test_real_stellar_checkpoint_still_cannot_execute_browser_actions():
    from habfly.runtime import Runtime

    runtime = Runtime()
    try:
        with pytest.raises(ValueError, match="cannot run against the HabWorlds browser"):
            runtime.start(
                {
                    "task": "lifetime",
                    "environment": "browser",
                    "policy": "checkpoint",
                    "checkpoint": "unused.pt",
                }
            )
    finally:
        runtime.close()
