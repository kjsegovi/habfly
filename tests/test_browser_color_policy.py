"""Actual gated checkpoint on fully intercepted Chromium fixtures, never a live preview."""

import io
import json
import os
from pathlib import Path

import pytest
from test_browser_numeric import chromium, config, page, stellar_html  # noqa: F401
from test_browser_setup import EMAIL, OUTER, PASSWORD, setup_page  # noqa: F401

from habfly.browser_color_policy import BrowserColorPolicyBridge, load_browser_color_policy
from habfly.color_reference import ColorReference
from habfly.runtime import RunOptions, Runtime, read_trace


def settings(tmp_path, **updates):
    payload = json.loads(Path("configs/browser_color_tui.json").read_text())
    payload.update(artifact_dir=str(tmp_path / "runtime"), browser_setup="manual")
    payload.update(updates)
    return RunOptions.model_validate(payload)


@pytest.fixture
def trained_artifacts():
    for path in ("experiments/color-pilot-006/final/report.json", "data/processed/graphs-v2/graph-2000"):
        if not Path(path).exists():
            pytest.skip("Requires the gated color checkpoint and real 2000-node graph")


def start(page, tmp_path, monkeypatch, *, automatic=False, autonomous=False):  # noqa: F811
    from habfly import browser_color_policy

    monkeypatch.setattr(
        browser_color_policy,
        "BrowserColorPolicyBridge",
        lambda options, output, provenance: BrowserColorPolicyBridge(
            options, output, provenance, page=page, config=config()
        ),
    )
    options = settings(tmp_path)
    if automatic:
        options.browser_setup = "automatic"
        monkeypatch.setenv("HABFLY_LOGIN_EMAIL", EMAIL)
        monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", PASSWORD)
    if autonomous:
        options.browser_execution, options.paused, options.interval = "autonomous", False, 0
    current = Runtime(io.StringIO())
    try:
        current.command({"command": "start", "payload": options.model_dump(mode="json")})
    except BaseException:
        current.close()
        raise

    # Gate validation can inspect private test cases; live inference cannot.
    def forbidden(*args, **kwargs):
        raise AssertionError("Private label oracle called during browser inference")

    monkeypatch.setattr(ColorReference, "private_label", forbidden)
    return current


def step(current, **payload):
    current.command({"command": "step", "payload": payload})


def pending(current):
    step(current, browser_ready=True)
    step(current)  # learned source
    step(current)  # learned color proposal, NOT a native write
    assert current.env.phase == "awaiting_color"
    assert current.env.session.attempts == 0


@pytest.mark.parametrize(
    "wavelength,label",
    [
        (370, "UV"),
        (420, "Violet"),
        (460, "Blue"),
        (485, "Cyan"),
        (530, "Green"),
        (580, "Yellow"),
        (600, "Orange"),
        (680, "Red"),
        (1000, "IR"),
    ],
)
def test_real_policy_one_approved_selection_all_bands(
    page,  # noqa: F811
    tmp_path,
    monkeypatch,
    trained_artifacts,
    wavelength,
    label,
):
    page.frames[1].set_content(stellar_html().replace("370", str(wavelength)))
    for frame in page.frames:
        for button in frame.get_by_role("button").all():
            button.evaluate("b => b.onclick=()=>document.body.replaceChildren()")
    current = start(page, tmp_path, monkeypatch)
    try:
        pending(current)
        bridge = current.env
        assert bridge.pending_color()["selected_color"] == label
        assert bridge.pending_color()["measurement"]["value"] == wavelength
        assert bridge.session.frame.get_by_role("combobox").input_value() == ""
        step(current)  # Extra n is harmless guidance, not a failed action.
        assert bridge.phase == "awaiting_color" and bridge.session.attempts == 0
        for payload in ({"approve_copy": True}, {"browser_ready": True}):
            with pytest.raises(ValueError):
                step(current, **payload)
        with pytest.raises(ValueError, match="single-step"):
            current.command({"command": "resume"})
        with pytest.raises(ValueError, match="explicit human"):
            bridge.approve(automatic=True)
        assert bridge.session.attempts == 0
        step(current, approve_color=True)
        assert bridge.session.attempts == 1 and bridge.journal.receipt["selected_color"] == label
        assert bridge.state()["pending_browser_color"] is None
        with pytest.raises(ValueError, match="No pending color"):
            step(current, approve_color=True)
        step(current)  # local receipt check, never a site button
        assert current.status == "stopped" and bridge.phase == "finished"
        assert bridge.summary["color_transport_verified"]
        assert bridge.summary["write_attempts"] == 1
        assert not bridge.summary["task_completed"] and not bridge.summary["browser_acceptance_passed"]
        assert not bridge.summary["receipt"]["correctness_verified"]
        assert bridge.browser_held and not page.is_closed()
        assert all(
            bridge.session.frame.locator(f"#{field}").input_value() == ""
            for field in ("distance", "luminosity", "temperature")
        )
        assert not page.get_by_role("checkbox").is_checked()
        step(current)  # Finished runs cannot restart or write on extra n presses.
        assert bridge.session.attempts == 1
        events = read_trace(current.trace_path)
        proposals = [e.payload for e in events if e.event == "action_proposed"]
        assert len(proposals) == 3
        assert all(p["action_confidence"] is None and not p["calibrated"] for p in proposals)
        activity = [e.payload for e in events if e.event == "neural_activity"]
        assert any(a.get("activity_pathway") == "selected_measurement_color_graph" for a in activity)
        assert all(
            a.get("action_probability") is None and a.get("target_probability") is None for a in activity
        )
        raw = current.trace_path.read_text()
        assert "expected_color" not in raw and "preview_sequence_id" not in raw
        assert "Hidden answers" not in raw
        path = current.trace_path
    finally:
        current.close()
    # No credentials, policy load, or browser use on replay, even with browser options retained.
    current.command({"command": "replay", "payload": {"path": str(path)}})
    while current.status == "running":
        current.tick()
    assert current.env is None
    current.close()


@pytest.mark.parametrize("mutation", ["native", "local", "replacement"])
def test_approval_revalidates_pending_state(page, tmp_path, monkeypatch, trained_artifacts, mutation):  # noqa: F811
    current = start(page, tmp_path, monkeypatch)
    try:
        pending(current)
        if mutation == "native":
            current.env.session.frame.locator("#distance").fill("12")
        elif mutation == "replacement":
            current.env.session.frame.get_by_role("combobox").evaluate("e => e.outerHTML=e.outerHTML")
        else:
            current.env.tool.source = "browser_flux"
        step(current, approve_color=True)
        assert current.status == "stopped"
        assert current.env.summary["write_attempts"] == 0
        assert not current.env.summary["color_transport_verified"]
        assert current.env.state()["pending_browser_color"] is None
    finally:
        current.close()


def test_abort_pending_never_selects(page, tmp_path, monkeypatch, trained_artifacts):  # noqa: F811
    current = start(page, tmp_path, monkeypatch)
    pending(current)
    bridge = current.env
    current.command({"command": "abort"})
    assert bridge.summary["write_attempts"] == 0
    assert not bridge.summary["color_transport_verified"]
    assert page.frames[1].get_by_role("combobox").input_value() == ""


def test_automatic_setup_stops_before_learned_actions(setup_page, tmp_path, monkeypatch, trained_artifacts):  # noqa: F811
    page, state = setup_page  # noqa: F811
    current = start(page, tmp_path, monkeypatch, automatic=True)
    try:
        assert "HABFLY_LOGIN_PASSWORD" not in os.environ
        page.goto(OUTER)
        for _ in range(150):
            current.advance_if_due()
            if current.env.phase != "setting_up":
                break
            page.wait_for_timeout(20)
        assert current.env.phase == "ready", current.output.getvalue()
        assert current.status == "paused" and state["posts"] == 1
        for _ in range(5):
            current.advance_if_due()
        assert current.observation.revision == 0 and current.env.session.attempts == 0
        step(current)
        step(current)
        step(current, approve_color=True)
        step(current)
        assert current.env.summary["color_transport_verified"]
        for path in tmp_path.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                raw = path.read_text()
                assert EMAIL not in raw and PASSWORD not in raw and "preview_sequence_id" not in raw
    finally:
        current.close()


@pytest.mark.parametrize(
    "change",
    [
        {"browser_execution": "autonomous"},
        {"paused": False},
        {"stars": 2},
        {"task": "browser_numeric"},
        {"policy": "expert"},
        {"environment": "simulator"},
        {"checkpoint": "experiments/temperature-source-003/training/checkpoint.pt"},
    ],
)
def test_unsafe_profiles_rejected_without_browser(tmp_path, change):
    with pytest.raises(ValueError):
        load_browser_color_policy(settings(tmp_path, **change))


def test_missing_gate_before_browser_launch(tmp_path, monkeypatch):
    from habfly import browser_color_policy

    monkeypatch.setattr(
        browser_color_policy,
        "BrowserColorPolicyBridge",
        lambda *a, **k: pytest.fail("Browser touched before gate"),
    )
    current = Runtime(io.StringIO())
    options = settings(
        tmp_path, dataset=tmp_path / "missing", checkpoint=tmp_path / "missing/training/checkpoint.pt"
    )
    with pytest.raises(FileNotFoundError):
        current.command({"command": "start", "payload": options.model_dump(mode="json")})
    assert current.env is None and current.trace is None


def test_manual_setup_clears_unused_secrets_before_driver(tmp_path, monkeypatch):
    from habfly.browser_color import ColorJournal
    from habfly.color_reference import load_color_reference

    # Stop before driver launch, after the bridge has cleared inherited secrets.
    keys = ("HABFLY_LOGIN_EMAIL", "HABFLY_LOGIN_PASSWORD", "DEBUG", "PWDEBUG", "DEBUG_FILE")
    for key in keys:
        monkeypatch.setenv(key, "fixture-only-unused-secret")

    def journal(*args, **kwargs):
        assert all(key not in os.environ for key in keys)
        raise RuntimeError("stop_before_driver")

    monkeypatch.setattr(ColorJournal, "__init__", journal)
    with pytest.raises(RuntimeError, match="stop_before_driver"):
        BrowserColorPolicyBridge(
            settings(tmp_path),
            tmp_path / "unused",
            {"color_gate_passed": True, "color_reference_hash": load_color_reference().checksum},
            config=config(),
        )


def advance_until(current, predicate):
    for _ in range(150):
        if predicate():
            return
        current.advance_if_due()
        current.env.page.wait_for_timeout(20)
    pytest.fail(current.output.getvalue())


def test_autonomous_color_pause_cancel_and_one_write(setup_page, tmp_path, monkeypatch, trained_artifacts):  # noqa: F811
    fixture_page, _ = setup_page
    current = start(fixture_page, tmp_path, monkeypatch, automatic=True, autonomous=True)
    try:
        fixture_page.goto(OUTER)
        advance_until(current, lambda: current.env.phase == "awaiting_color")
        bridge = current.env
        assert bridge.session.attempts == 0  # proposal and native selection are separate ticks
        current.command({"command": "pause"})
        for _ in range(3):
            current.advance_if_due()
            step(current)  # guidance only, never authorizes a pending color
        assert bridge.session.attempts == 0 and bridge.pending is not None
        current.command({"command": "resume"})
        advance_until(current, lambda: current.status == "stopped")
        assert bridge.summary["color_transport_verified"]
        assert bridge.summary["write_attempts"] == 1
        assert bridge.summary["receipt"]["color_authorization"] == "autonomous_opt_in"
        assert all(
            bridge.session.frame.locator(f"#{field}").input_value() == ""
            for field in ("distance", "luminosity", "temperature")
        )
        assert len([e for e in read_trace(current.trace_path) if e.event == "action_proposed"]) == 3
        assert not any(e.event == "error" for e in read_trace(current.trace_path))
        path = current.trace_path
    finally:
        current.close()
    current.command({"command": "replay", "payload": {"path": str(path)}})
    while current.status == "running":
        current.advance_if_due()
    assert current.env is None
    current.close()


@pytest.mark.parametrize("stop", ["abort", "stale", "timeout"])
def test_autonomous_pending_stops_without_write(setup_page, tmp_path, monkeypatch, trained_artifacts, stop):  # noqa: F811
    fixture_page, _ = setup_page
    current = start(fixture_page, tmp_path, monkeypatch, automatic=True, autonomous=True)
    try:
        fixture_page.goto(OUTER)
        advance_until(current, lambda: current.env.phase == "awaiting_color")
        bridge = current.env
        if stop == "abort":
            current.command({"command": "abort"})
        else:
            if stop == "stale":
                bridge.session.frame.locator("#temperature").fill("5")
            else:
                bridge.session.started -= 121
            current.advance_if_due()
        assert bridge.summary["write_attempts"] == 0
        assert not bridge.summary["color_transport_verified"]
    finally:
        current.close()
