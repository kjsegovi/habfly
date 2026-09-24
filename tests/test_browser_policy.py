"""Frozen learned-policy transfer on intercepted disposable pages, not HabWorlds."""

import io
import json
from pathlib import Path

import pytest
from test_browser_numeric import (  # noqa: F401 - shared intercepted fixtures
    OUTER,
    chromium,
    config,
    page,
    stellar_autosave_html,
    stellar_html,
)
from test_browser_setup import (  # noqa: F401 - fully intercepted automatic-setup fixture
    EMAIL,
    PASSWORD,
    canvas_html,
    decorative_heading_html,
    setup_page,
)

from habfly.browser_policy import BrowserPolicyBridge, browser_config, load_browser_policy
from habfly.contracts import Action
from habfly.runtime import RunOptions, Runtime, read_trace


def settings(tmp_path):
    payload = json.loads(Path("configs/browser_numeric_tui.json").read_text())
    payload["artifact_dir"] = str(tmp_path / "runtime")
    return RunOptions.model_validate(payload)


@pytest.fixture
def trained_artifacts():
    # Learned-transfer checks are local integration tests, not a requirement to
    # commit private experiment data or replace it with an untrained policy.
    for name in (
        "experiments/temperature-source-003/training/checkpoint.pt",
        "experiments/temperature-source-003/final/report.json",
        "data/processed/graphs-v2/graph-2000",
    ):
        if not Path(name).exists():
            pytest.skip("Requires the promoted temperature checkpoint and real 2000-node graph")


@pytest.fixture
def runtime(page, tmp_path, monkeypatch, trained_artifacts):  # noqa: F811 - imported pytest fixture
    from habfly import browser_policy

    monkeypatch.setattr(
        browser_policy,
        "BrowserPolicyBridge",
        lambda options, output, provenance: BrowserPolicyBridge(
            options, output, provenance, page=page, config=config()
        ),
    )
    current = Runtime(io.StringIO())
    current.command({"command": "start", "payload": settings(tmp_path).model_dump(mode="json")})
    yield current
    current.close()


def command(runtime, **payload):
    runtime.command({"command": "step", "payload": payload})


@pytest.fixture
def autonomous_runtime(setup_page, tmp_path, monkeypatch, trained_artifacts):  # noqa: F811
    from habfly import browser_policy

    fixture_page, state = setup_page
    state["canvas"] = canvas_html(detail=stellar_autosave_html())
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", PASSWORD)
    monkeypatch.setattr(
        browser_policy,
        "BrowserPolicyBridge",
        lambda options, output, provenance: BrowserPolicyBridge(
            options, output, provenance, page=fixture_page, config=config()
        ),
    )
    current = Runtime(io.StringIO())
    options = settings(tmp_path).model_copy(
        update={
            "browser_execution": "autonomous",
            "browser_setup": "automatic",
            "paused": False,
        }
    )
    current.command({"command": "start", "payload": options.model_dump(mode="json")})
    fixture_page.goto(OUTER)
    yield current, state
    current.close()


def autonomous_until(current, predicate):
    for _ in range(150):
        if predicate():
            return
        current.last_tick = 0
        current.advance_if_due()
    pytest.fail(f"Autonomous fixture did not reach expected state: {current.status}")


def test_autonomous_frozen_policy_completes_without_human_step_or_copy_commands(
    autonomous_runtime, monkeypatch
):
    from habfly.environments import local_stellar

    def forbidden(*args, **kwargs):
        raise AssertionError("No expert, hidden grades or repair allowed")

    monkeypatch.setattr(local_stellar, "grade_fields", forbidden)
    monkeypatch.setattr(local_stellar.LocalStellarEnv, "expert_action", forbidden)
    current, _ = autonomous_runtime
    assert current.status == "running"
    autonomous_until(current, lambda: current.env.phase == "ready")
    bridge = current.env
    # All protected buttons destroy the fixture if touched.
    for fixture_frame in bridge.page.frames:
        for button in fixture_frame.get_by_role("button").all():
            button.evaluate("e => e.onclick=()=>document.body.replaceChildren()")
    bridge.session.frame.locator("#autosave").evaluate("e=>e.textContent='Data saved'")
    autonomous_until(current, lambda: current.status == "stopped")
    assert bridge.summary["numeric_transport_passed"]
    assert bridge.summary["write_attempts"] == 3 and bridge.tool.steps == 31
    assert not bridge.summary["task_completed"] and not bridge.summary["browser_acceptance_passed"]
    events = read_trace(current.trace_path)
    assert not any(e.event == "error" for e in events)
    assert sum(e.event == "action_proposed" for e in events) == 31
    inner = read_trace(bridge.journal.output / "events.jsonl")
    receipts = [e.payload for e in inner if e.event == "action_result"]
    assert len(receipts) == 3 and all(r["copy_authorization"] == "autonomous_opt_in" for r in receipts)
    assert bridge.summary["provenance"]["optimizer_updates"] == 0
    assert bridge.summary["provenance"]["evaluation_scope"] == "autonomous_browser_numeric_transfer"
    raw = current.trace_path.read_text() + (bridge.journal.output / "events.jsonl").read_text()
    assert PASSWORD not in raw and EMAIL not in raw
    path = current.trace_path
    monkeypatch.setattr("habfly.browser_policy.BrowserPolicyBridge", forbidden)
    current.command({"command": "replay", "payload": {"path": str(path)}})
    while current.replay_events is not None:
        current.last_tick = 0
        current.advance_if_due()
    assert current.status == "completed"  # Playback only; no browser reopened.


def test_autonomous_pause_stops_setup_and_pending_copy_then_resume_or_abort(autonomous_runtime):
    current, state = autonomous_runtime
    current.command({"command": "pause"})
    for _ in range(5):
        current.advance_if_due()
    assert state["posts"] == 0 and current.env.phase == "setting_up"
    current.command({"command": "resume"})
    autonomous_until(current, lambda: current.env.phase == "awaiting_copy")
    bridge = current.env
    current.command({"command": "pause"})
    for _ in range(5):
        current.last_tick = 0
        current.advance_if_due()
    assert bridge.session.attempts == 0 and bridge.pending is not None
    with pytest.raises(ValueError, match="Use b"):
        command(current)
    current.command({"command": "resume"})
    current.last_tick = 0
    current.advance_if_due()
    assert bridge.session.attempts == 1 and bridge.pending is None and not bridge.tool.copy_approved
    autonomous_until(current, lambda: bridge.phase == "awaiting_copy")
    current.command({"command": "abort"})
    for _ in range(5):
        current.advance_if_due()
    assert current.status == "aborted" and current.env is None
    assert bridge.summary["write_attempts"] == 1 and bridge.pending is None
    assert not bridge.summary["numeric_transport_passed"]


def test_autonomous_stale_page_stops_before_automatic_copy_without_retry(autonomous_runtime):
    current, _ = autonomous_runtime
    autonomous_until(current, lambda: current.env.phase == "awaiting_copy")
    bridge = current.env
    bridge.session.handles["distance"].fill("123")
    current.last_tick = 0
    current.advance_if_due()
    assert current.status == "stopped" and bridge.outcome == "stale_numeric_observation"
    for _ in range(5):
        current.advance_if_due()
    assert bridge.summary["write_attempts"] == 0
    assert bridge.pending is None


def test_supervised_copy_cannot_claim_autonomous_authorization(runtime):
    command(runtime, browser_ready=True)
    until_copy(runtime)
    with pytest.raises(ValueError, match="not enabled"):
        runtime.env.approve(automatic=True)
    assert runtime.env.session.attempts == 0 and runtime.env.pending is not None


def test_batch_identity_is_rechecked_before_browser_launch(tmp_path, monkeypatch, trained_artifacts):
    from habfly.browser import BrowserSafetyStop

    def forbidden(*args, **kwargs):
        pytest.fail("Changed checkpoint identity reached the browser")

    monkeypatch.setattr("habfly.browser_policy.BrowserPolicyBridge", forbidden)
    current = Runtime(io.StringIO())
    current.expected_browser_identity = {"checkpoint_sha256": "0" * 64}
    payload = settings(tmp_path).model_dump(mode="json")
    payload.update(browser_execution="autonomous", browser_setup="automatic", paused=False)
    with pytest.raises(BrowserSafetyStop, match="batch_provenance_changed"):
        current.command({"command": "start", "payload": payload})
    assert current.env is None and current.status == "stopped"


@pytest.mark.parametrize("change", [{"task": "mini_habworlds"}, {"browser_setup": "manual"}, {"stars": 3}])
def test_autonomous_options_reject_scope_expansion_before_launch(tmp_path, monkeypatch, change):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid autonomous profile reached the browser")

    monkeypatch.setattr("habfly.browser_policy.BrowserPolicyBridge", forbidden)
    payload = settings(tmp_path).model_dump(mode="json")
    payload.update(browser_execution="autonomous", browser_setup="automatic", paused=False)
    payload.update(change)
    with pytest.raises(ValueError, match="Autonomous execution|Automatic setup"):
        Runtime(io.StringIO()).command({"command": "start", "payload": payload})


def until_copy(runtime):
    for _ in range(64):
        if runtime.env.phase == "awaiting_copy" or runtime.status != "paused":
            break
        command(runtime)
    assert runtime.env.phase == "awaiting_copy", runtime.output.getvalue()


def test_automatic_setup_handoff_accepts_n_but_waits_for_copy_approval(
    setup_page,  # noqa: F811 - imported pytest fixture
    tmp_path,
    monkeypatch,
    trained_artifacts,
):
    from habfly import browser_policy

    fixture_page, state = setup_page
    state["canvas"] = canvas_html(detail=decorative_heading_html())
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", PASSWORD)
    monkeypatch.setattr(
        browser_policy,
        "BrowserPolicyBridge",
        lambda options, output, provenance: BrowserPolicyBridge(
            options, output, provenance, page=fixture_page, config=config()
        ),
    )
    options = settings(tmp_path).model_copy(update={"browser_setup": "automatic"})
    current = Runtime(io.StringIO())
    try:
        current.command({"command": "start", "payload": options.model_dump(mode="json")})
        fixture_page.goto(OUTER)
        for _ in range(30):
            current.setup_tick()
            if current.env.phase == "ready":
                break
        assert current.env.phase == "ready" and current.status == "paused"
        assert current.env.tool.steps == 0
        assert current.neural_state is None
        assert not any(e.event == "action_proposed" for e in read_trace(current.trace_path))
        until_copy(current)  # Same protocol commands as n, using the frozen real-graph model.
        assert current.env.tool.steps > 0
        assert current.env.session.attempts == 0
        assert current.env.pending_copy() is not None
        with pytest.raises(ValueError, match="Use b"):
            command(current)  # n still cannot approve a browser write.
        assert current.env.session.attempts == 0
    finally:
        current.close()


@pytest.mark.parametrize("variant", ["normal", "ilnidel_rounded_reordered", "autosave_notification"])
def test_frozen_model_completes_three_confirmed_copies_without_class_or_grading(
    runtime, monkeypatch, variant
):
    from habfly.environments import local_stellar

    def forbidden(*args, **kwargs):
        raise AssertionError("No expert, private grading or Sheets allowed")

    monkeypatch.setattr(local_stellar, "grade_fields", forbidden)
    monkeypatch.setattr(local_stellar.LocalStellarEnv, "expert_action", forbidden)
    if variant == "autosave_notification":
        from habfly.browser_stellar import SIMULATION_URL

        frame = next(f for f in runtime.env.page.frames if f.url == SIMULATION_URL)
        frame.set_content(stellar_autosave_html())
    if variant == "ilnidel_rounded_reordered":
        from habfly.browser_stellar import SIMULATION_URL

        frame = next(f for f in runtime.env.page.frames if f.url == SIMULATION_URL)
        fixture = stellar_html(("luminosity", "temperature", "distance"))
        fixture = fixture.replace("0.045", "0.036").replace("370", "1709").replace("5.68E-10", "6.00E-15")
        frame.set_content(fixture)
        frame.locator("input:not([hidden])").evaluate_all(
            "xs => xs.forEach(e => e.onblur = () => { if(e.value) e.value=Number(e.value).toPrecision(4); })"
        )
    assert runtime.status == "paused" and runtime.env.session is None
    with pytest.raises(ValueError, match="Use b"):
        command(runtime)
    with pytest.raises(ValueError, match="single-step"):
        runtime.command({"command": "resume"})
    command(runtime, browser_ready=True)
    assert runtime.observation.values["star_class"] is None
    assert "expected" not in runtime.env.tool.case
    for count in range(3):
        until_copy(runtime)
        assert runtime.env.session.attempts == count
        with pytest.raises(ValueError, match="Use b"):
            command(runtime)
        if variant == "autosave_notification":
            frame.locator("#autosave").evaluate("e=>e.textContent='Data saved'")
        command(runtime, approve_copy=True)
        assert runtime.env.session.attempts == count + 1
        with pytest.raises(ValueError, match="No pending"):
            command(runtime, approve_copy=True)
        if variant == "autosave_notification":
            frame.locator("#autosave").evaluate("e=>e.textContent=''")
    while runtime.status == "paused":
        command(runtime)
    assert runtime.env.summary["numeric_transport_passed"], runtime.output.getvalue()
    assert runtime.env.state()["browser_guidance"].startswith("PASS:")
    assert not runtime.env.summary["task_completed"]
    assert runtime.env.summary["learned_policy"]
    assert runtime.env.tool.steps == 31
    assert runtime.env.summary["provenance"]["optimizer_updates"] == 0
    events = read_trace(runtime.trace_path)
    assert sum(e.event == "neural_activity" for e in events) == 31
    assert all(e.payload["calibrated"] is False for e in events if e.event == "action_proposed")
    assert all(e.payload["reward"] == 0 for e in events if e.event == "action_result")
    assert all("q%3A123" not in e.model_dump_json() for e in events)
    assert runtime.env.tool.results["r2"]["bindings"]["distance"] == "r1"
    trace = runtime.trace_path
    runtime.command({"command": "replay", "payload": {"path": str(trace)}})
    while runtime.replay_events is not None:
        runtime.tick()
    assert runtime.status == "completed"  # Completed playback, not a completed star.


def test_changed_page_during_approval_stops_before_write(runtime):
    command(runtime, browser_ready=True)
    until_copy(runtime)
    runtime.env.session.handles["distance"].fill("123")
    command(runtime, approve_copy=True)
    assert runtime.status == "stopped"
    assert runtime.env.summary["outcome"] == "stale_numeric_observation"
    assert runtime.env.summary["write_attempts"] == 0
    assert not runtime.env.summary["numeric_transport_passed"]
    with pytest.raises(ValueError, match="STOP: stale_numeric_observation"):
        command(runtime)


def test_abort_pending_copy_never_writes(runtime):
    command(runtime, browser_ready=True)
    until_copy(runtime)
    bridge = runtime.env
    runtime.command({"command": "abort"})
    assert bridge.summary["write_attempts"] == 0
    assert bridge.summary["outcome"] == "operator_aborted"
    assert bridge.pending is None


def test_repeated_noops_are_bounded(runtime):
    command(runtime, browser_ready=True)
    for _ in range(4):
        result = runtime.env.propose(Action(kind="WAIT"))
    assert result.terminated
    assert result.failure_reason == "repeated_unchanged_tool_actions"
    assert runtime.env.session.attempts == 0


def test_abort_before_capture_records_no_writes(runtime):
    bridge = runtime.env
    runtime.command({"command": "abort"})
    assert bridge.summary["write_attempts"] == 0
    assert bridge.summary["outcome"] == "operator_aborted"
    assert bridge.session is None


def test_driver_error_is_redacted_and_stops(runtime, monkeypatch):
    command(runtime, browser_ready=True)

    def broken(*args):
        raise RuntimeError("https://secret.invalid/?token=private")

    monkeypatch.setattr(runtime.env.session, "_current", broken)
    command(runtime)
    assert runtime.status == "stopped"
    assert "token=private" not in runtime.output.getvalue()
    assert "secret.invalid" not in runtime.output.getvalue()
    assert runtime.env.summary["outcome"] == "browser_operation_failed"


def test_ready_failure_and_abort_before_ready_are_recorded(runtime):
    runtime.env.page.goto("about:blank")
    command(runtime, browser_ready=True)
    assert runtime.status == "stopped"
    assert runtime.env.summary["write_attempts"] == 0
    assert not runtime.env.summary["numeric_transport_passed"]


def test_wrong_bindings_are_not_repaired(runtime):
    command(runtime, browser_ready=True)
    env = runtime.env.tool

    def act(key, value=None):
        action = Action(
            kind="SELECT" if value is not None else "CLICK",
            target=f"{env.steps}:{key}",
            value=value,
            observation_revision=env.steps,
        )
        return env.step(action)

    act("operation", "distance")
    act("parameter", "parallax")
    act("source", "browser_flux")
    act("bind")
    assert env.bindings == {"parallax": "browser_flux"}
    result = act("execute")
    assert "incompatible_unit" in result.failure_reason
    assert result.terminated
    assert not env.results and not env.answers


def test_wrong_units_are_not_repaired(runtime):
    command(runtime, browser_ready=True)
    env = runtime.env.tool
    result = env.step(Action(kind="SELECT", target="0:unit_distance", value="K", observation_revision=0))
    assert result.failure_reason == "selected_unit_does_not_match_visible_field"
    assert env.units["distance"] == "K"
    assert not env.session.verified


def test_offline_preflight_and_replay_do_not_need_sheets_or_network(tmp_path, monkeypatch, trained_artifacts):
    import socket

    from habfly import spreadsheet

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline check attempted external access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(spreadsheet.SpreadsheetAdapter, "__init__", forbidden)
    _, provenance = load_browser_policy(settings(tmp_path))
    assert provenance["optimizer_updates"] == 0
    if not Path("experiments/browser-numeric-003/events.jsonl").is_file():
        pytest.skip("Legacy live numeric replay is retained locally, not committed")
    replay = Runtime(io.StringIO())
    replay.command({"command": "replay", "payload": {"path": "experiments/browser-numeric-003/events.jsonl"}})
    while replay.replay_events is not None:
        replay.tick()
    replay.close()


def test_config_origin_cannot_be_widened(tmp_path, monkeypatch):
    monkeypatch.setenv("HABFLY_PREVIEW_URL", "https://example.com/unrelated")
    with pytest.raises(ValueError):
        browser_config(settings(tmp_path))


@pytest.mark.parametrize(
    "change",
    [
        {"paused": False},
        {"policy": "expert"},
        {"environment": "simulator"},
        {"calculation_backend": "google_sheets"},
        {"graph": None},
    ],
)
def test_unsafe_profiles_rejected_before_browser(tmp_path, change):
    options = settings(tmp_path).model_copy(update=change)
    with pytest.raises(ValueError, match="paused local checkpoint"):
        load_browser_policy(options)
