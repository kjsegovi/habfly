"""Real Chromium checks against a synthetic, loopback-only student interface."""

import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from habfly.browser import BrowserConfig, BrowserSafetyStop, TorusBrowser
from habfly.contracts import Action, ActionKind


@pytest.fixture(scope="module")
def fixture_url():
    body = (Path(__file__).parent / "fixtures" / "browser.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/activity"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture(scope="module")
def chromium():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def adapter_factory(chromium, fixture_url, tmp_path):
    @contextmanager
    def factory(**settings):
        context = chromium.new_context()
        page = context.new_page()
        page.goto(fixture_url, wait_until="domcontentloaded")
        config = BrowserConfig(
            url=fixture_url,
            allowed_path="/activity",
            action_interval=0,
            artifact_dir=tmp_path,
            headless=True,
            **settings,
        )
        adapter = TorusBrowser(config, page=page)
        try:
            adapter.start()
            yield adapter
        finally:
            adapter.close()
            context.close()

    return factory


def target(adapter, label):
    return next(control for control in adapter.observation.controls if control.label == label)


def action(adapter, label, kind=ActionKind.CLICK, **fields):
    return Action(
        kind=kind,
        target=target(adapter, label).id,
        observation_revision=adapter.observation.revision,
        **fields,
    )


def test_config_restricts_origin_and_path_components():
    config = BrowserConfig(url="http://localhost:1234/activity", allowed_path="/activity")
    assert config.allows("http://localhost:1234/activity/next")
    assert not config.allows("http://localhost:1234/activity-escape")
    assert not config.allows("http://localhost:4321/activity")
    assert not config.allows("https://localhost:1234/activity")
    assert not config.allows("http://example.com/activity")
    assert not config.allows("http://user:password@localhost:1234/activity")
    for path in (
        "/activity/../outside",
        "/activity/%2e%2e/outside",
        "/activity/%252e%252e/outside",
        "/activity/%2f..%2foutside",
        "/activity/..%5coutside",
    ):
        assert not config.allows(f"http://localhost:1234{path}")
    for url in (
        "https://example.com/activity",
        "http://user:password@localhost/activity",
        "http://localhost/activity?secret=1",
    ):
        with pytest.raises(ValueError):
            BrowserConfig(url=url)


def test_visible_controls_values_options_and_artifacts(adapter_factory):
    with adapter_factory() as adapter:
        observation = adapter.observation
        labels = [control.label for control in observation.controls]
        assert "Collect star" in labels
        assert "Hidden answer key" not in labels
        assert "Hidden password" not in labels
        assert "Secret answer 999" not in observation.instruction
        assert target(adapter, "Unavailable star").enabled is False
        assert target(adapter, "Orbital period").value == "3"
        assert target(adapter, "Time units").options == ["days", "years"]
        assert observation.chart_crop is not None
        assert Path(observation.chart_crop).is_file()
        assert "chart_pixels" in observation.modalities
        result = adapter.step(action(adapter, "Collect star"))
        assert not result.truncated
        assert result.observation.feedback == "Star collected"
        run_dir = adapter.run_dir
    assert (run_dir / "actions.jsonl").is_file()
    assert (run_dir / "trace.zip").is_file()


def test_native_form_actions(adapter_factory):
    with adapter_factory() as adapter:
        result = adapter.step(action(adapter, "Orbital period", ActionKind.TYPE, value="12"))
        assert not result.truncated
        assert target(adapter, "Orbital period").value == "12"
        result = adapter.step(action(adapter, "Time units", ActionKind.SELECT, value="years"))
        assert not result.truncated
        assert target(adapter, "Time units").value == "y"
        result = adapter.step(action(adapter, "Potentially habitable"))
        assert not result.truncated
        assert target(adapter, "Potentially habitable").value == "true"


def test_chart_semantics_after_ordinary_hover(adapter_factory):
    with adapter_factory() as adapter:
        result = adapter.step(action(adapter, "lightcurve", ActionKind.HOVER, x=0.4, y=0.4))
        assert not result.truncated
        assert result.observation.chart["tooltip_text"] == "Time 3 days; flux 0.96"
        assert result.observation.chart_crop is None
        assert "chart_pixels" not in result.observation.modalities


@pytest.mark.parametrize(
    "kind,fields,feedback",
    [
        (ActionKind.SCROLL, {"dy": -0.5}, "Chart zoom changed"),
        (ActionKind.DRAG, {"dx": 0.3}, "Chart pan changed"),
    ],
)
def test_chart_pointer_actions(adapter_factory, kind, fields, feedback):
    with adapter_factory() as adapter:
        result = adapter.step(action(adapter, "lightcurve", kind, **fields))
        assert not result.truncated
        assert result.observation.feedback == feedback


def test_changed_order_stops_before_clicking_wrong_target(adapter_factory):
    with adapter_factory() as adapter:
        selected = action(adapter, "Collect star")
        # The synthetic page changes independently between observation and execution.
        adapter.page.locator("#collection").evaluate("node => node.prepend(node.querySelector('#delete'))")
        result = adapter.step(selected)
        assert result.truncated
        assert result.failure_reason == "stale_page_observation"
        assert adapter.page.get_by_role("status").inner_text() == "Ready"


def test_previous_observation_target_cannot_be_reused(adapter_factory):
    with adapter_factory() as adapter:
        selected = action(adapter, "Collect star")
        assert not adapter.step(selected).truncated
        result = adapter.step(selected)
        assert result.truncated
        assert result.failure_reason == "stale_observation"


@pytest.mark.parametrize(
    "label,reason",
    [
        ("Show modal", "unexpected_modal"),
        ("Show browser alert", "unexpected_browser_dialog"),
        ("Show login", "authentication_required"),
        ("Open popup", "unexpected_popup"),
        ("Open scripted popup", "unexpected_popup"),
    ],
)
def test_unexpected_ui_stops(adapter_factory, label, reason):
    with adapter_factory() as adapter:
        result = adapter.step(action(adapter, label))
        assert result.truncated
        assert result.failure_reason == reason
        if reason == "authentication_required":
            assert not (adapter.run_dir / "failure.png").exists()
        else:
            assert (adapter.run_dir / "failure.png").exists()
        run_dir = adapter.run_dir
    if reason == "authentication_required":
        assert not (run_dir / "trace.zip").exists()


@pytest.mark.parametrize("label", ["Leave activity", "Similar path prefix"])
def test_navigation_boundary_is_enforced_before_navigation(adapter_factory, label):
    with adapter_factory() as adapter:
        result = adapter.step(action(adapter, label))
        assert result.truncated
        assert result.failure_reason == "navigation_outside_activity"
        assert not adapter.page.url.endswith(("/outside", "/activity-escape"))


def test_in_scope_navigation_is_allowed(adapter_factory):
    with adapter_factory() as adapter:
        result = adapter.step(action(adapter, "Next activity page"))
        assert not result.truncated
        assert adapter.page.url.endswith("/activity/next")


@pytest.mark.parametrize(
    "label,kind,fields",
    [
        ("Submit Project", ActionKind.CLICK, {}),
        ("Student note", ActionKind.KEYPRESS, {"value": "Enter"}),
    ],
)
def test_submission_default_blocks_button_and_enter(adapter_factory, label, kind, fields):
    with adapter_factory() as adapter:
        result = adapter.step(action(adapter, label, kind, **fields))
        assert result.truncated
        assert result.failure_reason == "submission_disabled"
        assert adapter.page.get_by_role("status").inner_text() == "Ready"


def test_explicit_submission_permission_allows_synthetic_acceptance(adapter_factory):
    with adapter_factory(allow_submission=True) as adapter:
        result = adapter.step(action(adapter, "Submit Project"))
        assert result.terminated
        assert not result.truncated
        assert result.observation.progress["submitted"] is True


@pytest.mark.parametrize("label", ["SUBMIT PROJECT.", "I am ready to submit project."])
def test_submission_punctuation_does_not_bypass_guard(adapter_factory, label):
    with adapter_factory() as adapter:
        adapter.page.get_by_role("button", name="Submit Project", exact=True).evaluate(
            "(node, text) => node.textContent = text", label
        )
        adapter.observation = adapter.observe()
        result = adapter.step(action(adapter, label))
        assert result.failure_reason == "submission_disabled"
        assert adapter.page.get_by_role("status").inner_text() == "Ready"


def test_non_submission_keypress_remains_available(adapter_factory):
    with adapter_factory() as adapter:
        result = adapter.step(action(adapter, "Student note", ActionKind.KEYPRESS, value="Tab"))
        assert not result.truncated
        assert not result.terminated


def test_unsupported_custom_combobox_fails_explicitly(adapter_factory):
    with adapter_factory() as adapter:
        adapter.page.locator("body").evaluate(
            "node => node.insertAdjacentHTML('beforeend', '<div role=combobox aria-label=Custom>Visible option</div>')"
        )
        with pytest.raises(BrowserSafetyStop, match="unsupported_custom_combobox"):
            adapter.observe()


def test_contenteditable_textbox_uses_visible_content(adapter_factory):
    with adapter_factory() as adapter:
        adapter.page.locator("body").evaluate(
            "node => node.insertAdjacentHTML('beforeend', '<div role=textbox aria-label=Notes contenteditable=true>Existing note</div>')"
        )
        adapter.observation = adapter.observe()
        assert target(adapter, "Notes").value == "Existing note"
        result = adapter.step(action(adapter, "Notes", ActionKind.TYPE, value="Edited note"))
        assert not result.truncated
        assert target(adapter, "Notes").value == "Edited note"


def test_authentication_during_start_discards_trace_and_cleans_callbacks(chromium, fixture_url, tmp_path):
    context = chromium.new_context()
    try:
        page = context.new_page()
        page.goto(fixture_url)
        page.locator('[aria-label="Sign in password"]').evaluate(
            "node => {node.hidden=false; node.value='synthetic-secret';}"
        )
        adapter = TorusBrowser(BrowserConfig(url=fixture_url, artifact_dir=tmp_path), page=page)
        with pytest.raises(BrowserSafetyStop, match="authentication_required"):
            adapter.start()
        assert adapter._closed
        assert not adapter._tracing
        assert not (adapter.run_dir / "trace.zip").exists()
        assert not (adapter.run_dir / "failure.png").exists()
        # External callers retain their page, but the failed adapter no longer owns callbacks.
        assert not page.is_closed()
        adapter.close()
    finally:
        context.close()


def test_repeated_unchanged_actions_stop(adapter_factory):
    with adapter_factory(repeat_limit=2) as adapter:
        assert not adapter.step(Action(kind=ActionKind.WAIT)).truncated
        result = adapter.step(Action(kind=ActionKind.WAIT))
        assert result.truncated
        assert result.failure_reason == "repeated_unchanged_state"


def test_step_and_time_limits(adapter_factory):
    with adapter_factory(max_steps=1) as adapter:
        result = adapter.step(Action(kind=ActionKind.WAIT))
        assert result.truncated
        assert result.failure_reason == "step_limit"
    with adapter_factory(max_seconds=60) as adapter:
        adapter.started -= 61
        result = adapter.step(Action(kind=ActionKind.WAIT))
        assert result.truncated
        assert result.failure_reason == "time_limit"
