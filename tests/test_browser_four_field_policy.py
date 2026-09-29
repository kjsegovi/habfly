"""Both frozen models on intercepted browser fixtures, never a live preview."""

import hashlib
import io
import json
from pathlib import Path

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_setup import EMAIL, OUTER, PASSWORD, setup_page  # noqa: F401

from habfly.browser_four_field_policy import BrowserFourFieldBridge, load_four_field_policy
from habfly.color_reference import ColorReference
from habfly.runtime import RunOptions, Runtime, read_trace


def settings(tmp_path, **updates):
    payload = json.loads(Path("configs/browser_four_field_tui.json").read_text())
    payload.update(artifact_dir=str(tmp_path / "runtime"), browser_setup="manual")
    payload.update(updates)
    return RunOptions.model_validate(payload)


@pytest.fixture
def trained_artifacts():
    for path in (
        "experiments/color-pilot-006/final/report.json",
        "experiments/temperature-source-003/training/checkpoint.pt",
        "data/processed/graphs-v2/graph-2000",
    ):
        if not Path(path).exists():
            pytest.skip("Requires both frozen models and real 2000-node graph")


def start(fixture_page, tmp_path, monkeypatch, *, automatic=False, autonomous=False):
    from habfly import browser_four_field_policy

    monkeypatch.setattr(
        browser_four_field_policy,
        "BrowserFourFieldBridge",
        lambda options, output, provenance: BrowserFourFieldBridge(
            options, output, provenance, page=fixture_page, config=config()
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
    # Gates can validate private cases; inference cannot access labels or grading.
    monkeypatch.setattr(
        ColorReference, "private_label", lambda *_: pytest.fail("Private label called during inference")
    )
    return current


def step(current, **payload):
    current.command({"command": "step", "payload": payload})


def to_handoff(current, *, capture=True):
    if capture:
        step(current, browser_ready=True)
    approvals = 0
    for _ in range(40):
        bridge = current.env
        if bridge.phase == "awaiting_handoff":
            break
        assert current.status == "paused", current.output.getvalue()
        assert bridge.active_stage == "numeric"
        if bridge.phase == "awaiting_copy":
            attempts = bridge.session.attempts
            step(current)  # Extra n cannot authorize a write.
            current.advance_if_due()
            assert bridge.session.attempts == attempts
            step(current, approve_copy=True)
            approvals += 1
        else:
            step(current)
    assert bridge.phase == "awaiting_handoff", current.output.getvalue()
    assert approvals == 3 and bridge.numeric_steps == 31
    assert bridge.session.attempts == 3 and not bridge.session.stopped
    assert bridge.state()["pending_browser_copy"] is None
    assert bridge.state()["pending_browser_color"] is None


def test_complete_four_fields_same_page_and_offline_replay(
    page,  # noqa: F811
    tmp_path,
    monkeypatch,
    trained_artifacts,
):
    for frame in page.frames:
        for button in frame.get_by_role("button").all():
            button.evaluate("b => b.onclick=()=>document.body.replaceChildren()")
    current = start(page, tmp_path, monkeypatch)
    try:
        to_handoff(current)
        bridge = current.env
        before = {
            name: bridge.session.frame.locator(f"#{name}").input_value()
            for name in ("distance", "luminosity", "temperature")
        }
        assert all(before.values())
        for _ in range(3):
            current.advance_if_due()
        assert bridge.phase == "awaiting_handoff" and set(bridge.children) == {"numeric"}
        for payload in ({"approve_copy": True}, {"approve_color": True}):
            with pytest.raises(ValueError):
                step(current, **payload)
        with pytest.raises(ValueError, match="single-step"):
            current.command({"command": "resume"})
        with pytest.raises(ValueError, match="human approval"):
            bridge.approve(automatic=True)
        step(current)  # Scripted read-only handoff, not a policy action.
        assert bridge.active_stage == "color" and bridge.phase == "ready"
        assert current.neural_state is None
        assert bridge.children["numeric"].session.stopped
        assert bridge.child.page is page and bridge.browser is None
        assert bridge.handoff_receipt["same_star_and_native_controls"]
        assert bridge.state()["checkpoint"] == str(current.options.color_checkpoint)
        assert bridge.state()["seed"] == 0
        assert bridge.children["numeric"].options.seed == 5500000
        step(current)
        assert current.policy.active_stage == "color"
        step(current)
        assert bridge.phase == "awaiting_color" and bridge.session.attempts == 0
        assert bridge.state()["pending_browser_color"]["selected_color"] == "UV"
        with pytest.raises(ValueError):
            step(current, approve_copy=True)
        step(current)
        assert bridge.session.attempts == 0
        step(current, approve_color=True)
        step(current)
        assert current.status == "stopped" and bridge.browser_held
        assert not page.is_closed() and not page.get_by_role("checkbox").is_checked()
        report = bridge.summary
        assert report["four_field_transport_verified"] and report["write_attempts"] == 4
        assert report["numeric_transport_passed"] and report["color_transport_verified"]
        assert not report["task_completed"] and not report["browser_acceptance_passed"]
        assert not report["color_receipt"]["correctness_verified"]
        assert report["color_receipt"]["color_authorization"] == "human_confirmation"
        for name, value in before.items():
            assert bridge.session.frame.locator(f"#{name}").input_value() == value
        assert bridge.session.frame.get_by_role("combobox").input_value() == "UV"
        for component in report["components"].values():
            path = bridge.journal.output / component["manifest"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == component["sha256"]
        step(current)  # Harmless after completion.
        events = read_trace(current.trace_path)
        proposals = [e.payload for e in events if e.event == "action_proposed"]
        assert len(proposals) == 34
        assert [p["policy_stage"] for p in proposals] == ["numeric"] * 31 + ["color"] * 3
        assert all(p["action_confidence"] is None and not p["calibrated"] for p in proposals)
        activity = [e.payload for e in events if e.event == "neural_activity"]
        assert len(activity) == 34
        assert any(
            a.get("activity_pathway") == "selected_measurement_color_graph"
            for a in activity
            if a["policy_stage"] == "color"
        )
        assert not any(e.event == "error" for e in events)
        results = [e.payload for e in events if e.event == "action_result"]
        assert [r["steps"] for r in results] == list(range(1, 35))
        path = current.trace_path
        for forbidden in ("expected_color", "preview_sequence_id", "Hidden answers", PASSWORD, EMAIL):
            assert forbidden not in path.read_text()
        journal_paths = [bridge.journal.output / "events.jsonl"] + [
            child.journal.output / "events.jsonl" for child in bridge.children.values()
        ]
    finally:
        current.close()
    from habfly import browser_four_field_policy

    monkeypatch.setattr(
        browser_four_field_policy, "load_four_field_policy", lambda *_: pytest.fail("Replay loaded a policy")
    )
    for trace in [path, *journal_paths]:
        replay = Runtime(io.StringIO())
        replay.command({"command": "replay", "payload": {"path": str(trace)}})
        while replay.status == "running":
            replay.tick()
        recorded = [json.loads(line) for line in replay.output.getvalue().splitlines()][1:-1]
        original = read_trace(trace)
        assert len(recorded) == len(original)
        for emitted, source in zip(recorded, original, strict=True):
            expected = {**source.payload, "replay": True}
            if source.event == "state":
                expected.update(recorded_status=expected.get("status"), status="running")
            assert emitted["event"] == source.event and emitted["payload"] == expected
        assert replay.env is None
        replay.close()


@pytest.mark.parametrize("mutation", ["star", "measurement", "value", "numeric_node", "color_node", "race"])
def test_handoff_fails_closed(
    page,  # noqa: F811
    tmp_path,
    monkeypatch,
    trained_artifacts,
    mutation,
):
    from habfly.browser_color_policy import BrowserColorPolicyBridge

    current = start(page, tmp_path, monkeypatch)
    try:
        to_handoff(current)
        bridge = current.env
        frame = bridge.session.frame
        if mutation == "star":
            frame.locator("div").first.evaluate("e => e.textContent='Different star'")
        elif mutation == "measurement":
            frame.locator("div").nth(1).evaluate(
                "e => e.firstChild.textContent=e.firstChild.textContent.replace('370','380')"
            )
        elif mutation == "value":
            frame.locator("#distance").fill("12")
        elif mutation in {"numeric_node", "color_node"}:
            selector = "#distance" if mutation == "numeric_node" else "select"
            frame.locator(selector).evaluate(
                "e => {const c=e.cloneNode(true);c.value=e.value;e.replaceWith(c)}"
            )
        else:
            original = BrowserColorPolicyBridge.ready

            def race(child):
                observation = original(child)
                frame.locator("#distance").fill("12")
                return observation

            monkeypatch.setattr(BrowserColorPolicyBridge, "ready", race)
        step(current)
        assert current.status == "stopped", current.output.getvalue()
        assert bridge.summary["write_attempts"] == 3
        assert not bridge.summary["four_field_transport_verified"]
        assert bridge.summary["handoff"] is None
        assert frame.get_by_role("combobox").input_value() == ""
    finally:
        current.close()


@pytest.mark.parametrize("stop", ["abort", "timeout", "stale", "side_effect"])
def test_pending_color_never_retries_or_claims_partial_success(
    page,  # noqa: F811
    tmp_path,
    monkeypatch,
    trained_artifacts,
    stop,
):
    current = start(page, tmp_path, monkeypatch)
    try:
        to_handoff(current)
        step(current)
        step(current)
        step(current)
        bridge = current.env
        assert bridge.phase == "awaiting_color"
        if stop == "abort":
            current.command({"command": "abort"})
        else:
            if stop == "timeout":
                bridge.started -= bridge.MAX_SECONDS + 1
            elif stop == "stale":
                bridge.session.frame.locator("#distance").fill("12")
            else:
                bridge.session.frame.get_by_role("combobox").evaluate(
                    "e => e.onchange=()=>document.querySelector('#distance').value='12'"
                )
            step(current, approve_color=True)
        assert bridge.summary["numeric_transport_passed"]  # Earlier receipts only.
        assert not bridge.summary["four_field_transport_verified"]
        assert bridge.summary["write_attempts"] == (4 if stop == "side_effect" else 3)
        assert bridge.state()["pending_browser_color"] is None
    finally:
        current.close()


@pytest.mark.parametrize("existing", ["number", "color"])
def test_nonblank_start_rejected_before_writes(
    page,  # noqa: F811
    tmp_path,
    monkeypatch,
    trained_artifacts,
    existing,
):
    frame = page.frames[1]
    if existing == "number":
        frame.locator("#distance").fill("12")
    else:
        frame.get_by_role("combobox").select_option(label="UV")
    current = start(page, tmp_path, monkeypatch)
    try:
        step(current, browser_ready=True)
        assert current.status == "stopped"
        assert current.env.summary["write_attempts"] == 0
    finally:
        current.close()


@pytest.mark.parametrize("phase", ["awaiting_copy", "awaiting_handoff"])
def test_abort_numeric_phase_cannot_start_color(
    page,  # noqa: F811
    tmp_path,
    monkeypatch,
    trained_artifacts,
    phase,
):
    current = start(page, tmp_path, monkeypatch)
    bridge = current.env
    try:
        if phase == "awaiting_handoff":
            to_handoff(current)
        else:
            step(current, browser_ready=True)
            for _ in range(15):
                if bridge.phase == "awaiting_copy":
                    break
                step(current)
            assert bridge.phase == "awaiting_copy"
        current.command({"command": "abort"})
        assert not bridge.summary["four_field_transport_verified"]
        assert bridge.summary["write_attempts"] == (3 if phase == "awaiting_handoff" else 0)
        assert set(bridge.children) == {"numeric"}
        assert bridge.state()["pending_browser_copy"] is None
        assert bridge.session.frame.get_by_role("combobox").input_value() == ""
    finally:
        current.close()


def test_automatic_setup_remains_paused_before_any_learned_action(
    setup_page,  # noqa: F811
    tmp_path,
    monkeypatch,
    trained_artifacts,
):
    fixture_page, state = setup_page
    current = start(fixture_page, tmp_path, monkeypatch, automatic=True)
    try:
        fixture_page.goto(OUTER)
        for _ in range(150):
            current.advance_if_due()
            if current.env.phase != "setting_up":
                break
            fixture_page.wait_for_timeout(20)
        assert current.env.phase == "ready", current.output.getvalue()
        assert current.status == "paused" and state["posts"] == 1
        for _ in range(5):
            current.advance_if_due()
        assert current.env.session.attempts == 0 and current.observation.revision == 0
        assert not any(e.event == "action_proposed" for e in read_trace(current.trace_path))
        current.command({"command": "abort"})
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
        {"policy": "expert"},
        {"environment": "simulator"},
        {"color_checkpoint": None},
        {"color_dataset": None},
        {"calculation_backend": "google_sheets"},
    ],
)
def test_unsafe_options_rejected_before_models(tmp_path, monkeypatch, change):
    from habfly import browser_four_field_policy

    monkeypatch.setattr(
        browser_four_field_policy,
        "load_browser_policy",
        lambda *_: pytest.fail("Invalid profile reached model loading"),
    )
    with pytest.raises(ValueError):
        load_four_field_policy(settings(tmp_path, **change))


def test_graph_mismatch_and_color_gate_fail_without_browser(tmp_path, monkeypatch):
    from habfly import browser_four_field_policy

    monkeypatch.setattr(
        browser_four_field_policy, "load_browser_policy", lambda *_: (None, {"graph_hash": "a"})
    )
    monkeypatch.setattr(
        browser_four_field_policy, "load_browser_color_policy", lambda *_: (None, {"graph_hash": "b"})
    )
    with pytest.raises(ValueError, match="same validated graph"):
        load_four_field_policy(settings(tmp_path))

    def failed_gate(*_):
        raise ValueError("Color learning gate failed")

    monkeypatch.setattr(browser_four_field_policy, "load_browser_color_policy", failed_gate)
    with pytest.raises(ValueError, match="gate"):
        load_four_field_policy(settings(tmp_path))


@pytest.mark.parametrize("phase", ["awaiting_copy", "awaiting_handoff", "awaiting_color"])
def test_autonomous_pause_and_abort_cancel_next_write_or_handoff(
    setup_page,  # noqa: F811
    tmp_path,
    monkeypatch,
    trained_artifacts,
    phase,
):
    fixture_page, _ = setup_page
    current = start(fixture_page, tmp_path, monkeypatch, automatic=True, autonomous=True)
    try:
        fixture_page.goto(OUTER)
        bridge = current.env
        for _ in range(250):
            if bridge.phase == phase:
                break
            current.advance_if_due()
            assert current.status == "running", current.output.getvalue()
            fixture_page.wait_for_timeout(5)
        assert bridge.phase == phase
        before = sum(c.session.attempts for c in bridge.children.values())
        stages = set(bridge.children)
        current.command({"command": "pause"})
        for _ in range(3):
            current.advance_if_due()
        assert sum(c.session.attempts for c in bridge.children.values()) == before
        assert set(bridge.children) == stages and bridge.phase == phase
        current.command({"command": "abort"})
        assert bridge.summary["write_attempts"] == before
        assert not bridge.summary["four_field_transport_verified"]
    finally:
        current.close()
