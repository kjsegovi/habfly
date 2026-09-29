"""Actual renderer on intercepted fixture pages only. No real activity or model."""

from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

import pytest
from PIL import Image
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser_mascot import ASSETS, HOST_ID, BrowserMascot, preview_html
from habfly.browser_probe import inspect_page
from habfly.contracts import RuntimeEvent
from habfly.presentation_capture import evidence_screenshot
from habfly.runtime import RunOptions, Runtime, parse_run_options


def event(kind="state", **payload):
    return {"version": 1, "event": kind, "payload": payload}


def test_generated_asset_has_real_transparency():
    image = Image.open(ASSETS / "sporky-fly-v1.png")
    assert image.mode == "RGBA"
    assert image.getchannel("A").getextrema() == (0, 255)
    assert image.getpixel((0, 0))[3] == 0


def test_preview_is_offline_explicitly_demonstrational():
    html = preview_html()
    assert "data:image/png;base64," in html and "Demo only" in html
    assert "No HabWorlds connection" in html and "not a running model" in html
    assert "https://" not in html and "http://" not in html


def test_option_off_by_default_and_strictly_project_scoped():
    assert RunOptions().project_mascot is False
    with pytest.raises(ValueError, match="requires_project_task"):
        parse_run_options({"project_mascot": True})
    with pytest.raises(ValueError):
        RunOptions(project_mascot=1)


def test_actual_canvas_is_capture_and_observation_neutral(page):  # noqa: F811
    baseline = inspect_page(page, config())
    clean = page.screenshot()
    presenter = BrowserMascot()
    presenter.consume(
        event(status="paused"), page=page, allowed_frame_urls=[rule.url for rule in config().frames]
    )
    page.wait_for_function("id=>Boolean(document.getElementById(id))", arg=HOST_ID)
    page.mouse.move(100, 140)
    page.wait_for_timeout(100)
    assert page.screenshot() != clean  # Presentation really is visible.
    assert evidence_screenshot(page) == clean
    after = inspect_page(page, config())
    baseline.pop("captured_at")
    after.pop("captured_at")
    assert after == baseline
    assert page.locator("body").inner_text().find("Paused") == -1
    assert "Paused" not in page.locator("body").aria_snapshot()
    # A closed shadow tree cannot be accidentally inventoried as another chart.
    assert page.locator("canvas").count() == 0
    checkbox = page.get_by_role("checkbox")
    checkbox.click()
    assert checkbox.is_checked()
    presenter.close()
    assert page.locator("#" + HOST_ID).count() == 0
    for frame in page.frames:
        assert not frame.evaluate("c=>Boolean(window[Symbol.for(c+'-pointer')])", presenter.channel)


def test_iframe_pointer_does_not_change_tooltip_or_chart_and_survives_reload(page):  # noqa: F811
    presenter = BrowserMascot()
    allowed = [rule.url for rule in config().frames]
    presenter.consume(event(status="running"), page=page, allowed_frame_urls=allowed)
    frame = page.frames[1]
    target = frame.get_by_role("textbox").first
    target.hover()
    page.wait_for_timeout(100)
    before = evidence_screenshot(frame.locator("body"))
    raw = page.screenshot()
    presenter.consume(event(status="paused"), page=page, allowed_frame_urls=allowed)
    assert before == evidence_screenshot(frame.locator("body"))
    assert raw != page.screenshot()
    # Same-URL iframe navigation retires the tracker; it must be reinstalled.
    frame.goto(frame.url)
    presenter.consume(event(status="paused"), page=page, allowed_frame_urls=allowed)
    assert frame.evaluate("c=>Boolean(window[Symbol.for(c+'-pointer')])", presenter.channel)
    page.reload(wait_until="load")
    presenter.consume(event(status="paused"), page=page, allowed_frame_urls=allowed)
    assert page.locator("#" + HOST_ID).count() == 1 and not presenter.disabled
    presenter.close()


def test_presentation_failure_is_sanitized_and_cannot_rewrite_event(capsys):
    presenter = BrowserMascot()

    class Broken:
        def evaluate(self, *_):
            raise RuntimeError("private password/session URL")

    source = event(status="running")
    before = deepcopy(source)
    presenter.consume(source, page=Broken())
    assert presenter.disabled and source == before
    assert "private" not in capsys.readouterr().err


def test_runtime_display_is_downstream_of_unchanged_wire_and_skipped_during_replay():
    output = StringIO()
    runtime = Runtime(output)
    calls = []
    runtime.mascot = SimpleNamespace(consume=lambda *a, **k: calls.append((a, k)), close=lambda: None)
    runtime.emit("state", {"status": "paused"})
    recorded = RuntimeEvent.model_validate_json(output.getvalue())
    assert recorded.payload == {"status": "paused"}
    assert calls[0][0][0] == recorded.model_dump(mode="json")
    assert calls[0][1]["page"] is None  # No browser at login/start, or for replay.
    runtime.replay_context = {"replay": True}
    runtime.emit("state", {"status": "paused"})
    assert len(calls) == 1
    runtime.close()


def test_runtime_drops_broken_display_without_stopping_or_leaking(capsys):
    runtime = Runtime(StringIO())

    def broken(*_args, **_kwargs):
        raise RuntimeError("private password")

    runtime.mascot = SimpleNamespace(consume=broken, close=broken)
    runtime.emit("state", {"status": "running"})
    assert runtime.mascot is None
    assert "private" not in capsys.readouterr().err


def test_runtime_project_display_uses_raw_context_once_with_compact_wire():
    output = StringIO()
    runtime = Runtime(output)
    runtime.run_id = "presentation-fixture"
    runtime.options = RunOptions(task="browser_project", project_compact_wire=True)
    runtime.trace_path = "presentation-only.jsonl"
    calls = []
    runtime.mascot = SimpleNamespace(consume=lambda *a, **k: calls.append((a, k)), close=lambda: None)
    raw = {
        "component": "project.owner",
        "component_star": "Example",
        "component_event": {"event": "action_proposed", "payload": {"kind": "WAIT"}},
        "kind": "WAIT",
    }
    source = deepcopy(raw)
    runtime.project_event("action_proposed", raw)
    assert len(calls) == 1 and calls[0][0][0]["payload"] == source
    assert raw == source
    recorded = RuntimeEvent.model_validate_json(output.getvalue())
    assert "project_wire" in recorded.payload
    assert "component_event" not in recorded.payload
    runtime.close()


def test_broken_display_cleanup_cannot_prevent_owned_browser_cleanup(capsys):
    runtime = Runtime(StringIO())
    closed = []

    def broken():
        raise RuntimeError("private driver text")

    runtime.mascot = SimpleNamespace(close=broken)
    runtime.env = SimpleNamespace(close=lambda: closed.append(True))
    runtime.close()
    assert closed == [True] and runtime.mascot is None and runtime.env is None
    assert "private" not in capsys.readouterr().err
