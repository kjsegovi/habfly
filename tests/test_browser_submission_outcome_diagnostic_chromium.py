"""Gated, fully intercepted outcome diagnostics; never a real submission."""

import json
import os
import time

import pytest
from playwright.sync_api import sync_playwright

from habfly.browser_probe import BrowserProbeConfig
from habfly.browser_submission_outcome_diagnostic import (
    native_dialog_disposition,
    preserve_submission_outcome,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must explicitly release the real preview before native fixtures",
)

OUTER = "http://localhost/activity?preview_sequence_id=q%3A123%3A946"
KNOWN = "https://fixture.invalid/outcome"
UNKNOWN = "https://unknown.invalid/private"


@pytest.fixture(scope="module")
def chromium():
    if os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1":
        pytest.skip("Explicit native slot required")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(chromium):
    context = chromium.new_context(viewport={"width": 1200, "height": 900}, service_workers="block")
    pages = {
        OUTER: f'<h1>Public project fixture</h1><iframe src="{KNOWN}"></iframe><iframe src="{KNOWN}"></iframe>',
        KNOWN: "<p>Known visible response frame</p>",
        UNKNOWN: "<p>PRIVATE unknown response</p>",
    }
    context.route(
        "**/*",
        lambda route: (
            route.fulfill(status=200, content_type="text/html", body=pages[route.request.url])
            if route.request.url in pages
            else route.abort()
        ),
    )
    current = context.new_page()
    current.goto(OUTER, wait_until="load")
    yield current
    context.close()


def capture(page, tmp_path, **options):
    config = BrowserProbeConfig(url=OUTER, frames=[{"name": "known", "url": KNOWN, "count": 2}])
    return preserve_submission_outcome(
        page,
        config,
        tmp_path / "diagnostic",
        deadline=time.monotonic() + 10,
        original_failure="project_submission_context_changed",
        **options,
    )


def test_native_replaced_known_frames_and_html_outcome_dialog_are_preserved(page, tmp_path):
    # Fixture-only mutation models a public post-submit response. The diagnostic
    # itself never clicks, edits the DOM, navigates, or infers an acknowledgement.
    page.locator("body").evaluate(
        """(body,url)=>{
        for(const iframe of body.querySelectorAll('iframe'))iframe.remove();
        const frame=document.createElement('iframe');frame.src=url;body.append(frame);
        const dialog=document.createElement('section');dialog.setAttribute('role','dialog');
        dialog.setAttribute('aria-label','Public outcome');dialog.textContent='Visible fixture response';
        body.append(dialog);
        }""",
        KNOWN,
    )
    page.frame_locator("iframe").get_by_text("Known visible response frame").wait_for(state="visible")
    before = page.locator("body").inner_text()
    result = capture(page, tmp_path)
    assert result["disposition"] == "public_outcome_captured"
    assert result["consistency_unverified"] and not result["submission_verified"]
    assert not result["submitted"] and not result["project_completed"]
    assert page.locator("body").inner_text() == before
    saved = json.loads((tmp_path / "diagnostic/visible-outcome.json").read_bytes())
    assert len(saved["frames"]) == 2
    assert "Visible fixture response" in saved["frames"][0]["text"]
    assert 'dialog "Public outcome"' in saved["frames"][0]["accessibility"]
    assert (tmp_path / "diagnostic/viewport.png").is_file()


@pytest.mark.parametrize("scope", ["outer_auth", "frame_auth", "unknown_frame"])
def test_native_private_surface_is_never_saved(page, tmp_path, scope):
    if scope == "unknown_frame":
        page.locator("iframe").first.evaluate("(frame,url)=>frame.src=url", UNKNOWN)
        page.frame_locator("iframe").first.get_by_text("PRIVATE unknown response").wait_for(state="visible")
    else:
        surface = (
            page if scope == "outer_auth" else next(frame for frame in page.frames if frame.url == KNOWN)
        )
        surface.locator("body").evaluate(
            "body=>{const input=document.createElement('input');input.type='password';body.append(input)}"
        )
    result = capture(page, tmp_path)
    assert result["disposition"] == (
        "unknown_visible_frame" if scope == "unknown_frame" else "authentication_required"
    )
    assert result["content_saved"] is False
    assert {path.name for path in (tmp_path / "diagnostic").iterdir()} == {"disposition.json"}
    assert "PRIVATE" not in (tmp_path / "diagnostic/disposition.json").read_text()


def test_native_prompt_is_not_read_or_dismissed_by_diagnostic(page, tmp_path):
    dialogs = []
    page.on("dialog", lambda dialog: dialogs.append(native_dialog_disposition(dialog)))
    with page.expect_event("dialog"):
        page.evaluate("setTimeout(()=>prompt('PRIVATE prompt details','PRIVATE default'),0)")
    result = capture(page, tmp_path, native_dialogs=dialogs)
    assert result["disposition"] == "native_dialog_content_withheld"
    assert result["native_dialogs"] == [
        {"type": "prompt", "message_disposition": "withheld_unverified_native_dialog_context"}
    ]
    assert result["content_saved"] is False
    assert {path.name for path in (tmp_path / "diagnostic").iterdir()} == {"disposition.json"}
    assert "PRIVATE" not in (tmp_path / "diagnostic/disposition.json").read_text()
    # Context cleanup owns the still-open fixture prompt; production never
    # accepts or dismisses it and no success is inferred from its existence.
