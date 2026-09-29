"""Bounded post-click paint reads, with no browser/network launch."""
# ruff: noqa: F401 -- imported pytest autouse fixture

import json
from types import SimpleNamespace

import pytest
from test_browser_class_rendering_diagnostic import offline_only, paint, real_session

import habfly.browser_classification as module
from habfly.browser import BrowserSafetyStop


def animated(monkeypatch, *, transitions=2, rendering=None, affected="white_dwarf", on_wait=None):
    session, state = real_session(monkeypatch)
    state.paint_reads, state.waits, state.now = 0, [], 0.0
    original = module.read_class_choices

    def read(frame):
        if state.clicks:
            state.paint_reads += 1
            if state.paint_reads <= transitions:
                raise module.ClassCircleRenderingStop(
                    affected,
                    rendering or {**paint(True), "dotOpacity": "0.431"},
                )
        return original(frame)

    def wait(milliseconds):
        state.waits.append(milliseconds)
        state.now += milliseconds / 1000
        if on_wait:
            on_wait(session, state)

    session.page.wait_for_timeout = wait
    monkeypatch.setattr(module, "read_class_choices", read)
    monkeypatch.setattr(module.time, "monotonic", lambda: state.now)
    return session, state


def select(session):
    return session.select_class("white_dwarf", source="reference_diagnostic")


def assert_stopped_single_click(session, state):
    assert session.stopped and state.clicks == 1
    assert [event[0] for event in state.events] == ["action_proposed"]
    with pytest.raises(BrowserSafetyStop):
        select(session)
    assert state.clicks == 1


def test_known_transition_reads_to_exact_final_paint_and_original_event_shape(monkeypatch):
    session, state = animated(monkeypatch)
    receipt = select(session)
    assert receipt["readback_verified"] is True
    assert not receipt["correctness_verified"] and not receipt["task_completed"]
    assert state.clicks == 1 and len(state.waits) == 2
    assert session.choices["rendering"]["white_dwarf"] == paint(True)
    assert "settling" not in json.dumps(state.events)
    assert set(receipt) == {
        "selected_class",
        "selection_source",
        "readback_verified",
        "correctness_verified",
        "rendering_sha256",
        "task_completed",
    }


@pytest.mark.parametrize("opacity", ["1e-07", "1e-12", "1e-38", "1.4013e-45"])
def test_native_tiny_numeric_formats_settle_but_are_never_final_paint(monkeypatch, opacity):
    rendering = {**paint(True), "dotOpacity": opacity}
    assert module._diagnostic_property("dotOpacity", opacity) == opacity
    with pytest.raises(BrowserSafetyStop, match="unsupported_class_circle_rendering"):
        module.painted_selection(rendering)
    session, state = animated(monkeypatch, rendering=rendering)
    receipt = select(session)
    assert receipt["readback_verified"] and state.clicks == 1
    assert len(state.waits) == 2
    assert session.choices["rendering"]["white_dwarf"]["dotOpacity"] == "1"


@pytest.mark.parametrize(
    "opacity",
    [
        "NaN",
        "nan",
        "Infinity",
        "inf",
        "-0.1",
        "-1e-07",
        "1.00001",
        "1.0000000000000001",
        "5e+1",
        "1e-46",
        "1e-999999999",
        "1e+999999999",
        "1e",
        "1e--7",
        "1;url(secret)",
        "0." + "1" * 200,
        "",
        None,
        True,
        0.5,
    ],
)
def test_invalid_or_unbounded_opacity_is_redacted_and_not_polled(monkeypatch, opacity):
    assert module._diagnostic_property("dotOpacity", opacity) is None
    session, state = animated(monkeypatch, rendering={**paint(True), "dotOpacity": opacity})
    with pytest.raises(module.ClassCircleRenderingStop) as caught:
        select(session)
    assert caught.value.diagnostic["visible_rendering"]["dotOpacity"] is None
    assert state.waits == []
    assert_stopped_single_click(session, state)


@pytest.mark.parametrize(
    "rendering,affected",
    [
        ({**paint(True), "dotOpacity": "0.5", "width": "27px"}, "white_dwarf"),
        ({**paint(True), "dotOpacity": "0.5", "dotColor": "rgb(0, 200, 220)"}, "white_dwarf"),
        ({**paint(True), "dotOpacity": "0.5", "border": "1px solid rgb(0, 200, 220)"}, "white_dwarf"),
        ({**paint(True), "dotOpacity": "0"}, "white_dwarf"),
        ({**paint(True), "dotOpacity": "0.5"}, "red_giant"),
    ],
)
def test_unknown_shape_or_unrelated_circle_never_polled(monkeypatch, rendering, affected):
    session, state = animated(monkeypatch, rendering=rendering, affected=affected)
    with pytest.raises(module.ClassCircleRenderingStop):
        select(session)
    assert state.waits == []
    assert_stopped_single_click(session, state)


def test_permanent_animation_obeys_fixed_time_and_probe_cap(monkeypatch):
    session, state = animated(monkeypatch, transitions=10000)
    with pytest.raises(module.ClassCircleRenderingStop) as caught:
        select(session)
    proof = caught.value.diagnostic["post_click_settling"]
    assert proof["max_seconds"] == 2 and proof["max_probes"] == 40
    assert proof["settled"] is False and proof["additional_clicks"] == 0
    assert state.now <= 2.000001 and len(state.waits) <= 40
    assert_stopped_single_click(session, state)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("abort", "stellar_selection_stopped"),
        ("popup", "unexpected_popup"),
        ("frame", "simulation_frame_changed"),
        ("detached", "class_control_replaced"),
    ],
)
def test_context_or_cancel_changes_stop_between_paint_reads(monkeypatch, mutation, reason):
    def change(session, state):
        if mutation == "abort":
            session.stopped = True
        elif mutation == "popup":
            session.page.context.pages.append(object())
        elif mutation == "frame":
            session.page.frames = []
        else:
            session.handles["white_dwarf"].evaluate = lambda *_: False

    session, state = animated(monkeypatch, on_wait=change)
    with pytest.raises(BrowserSafetyStop, match=reason):
        select(session)
    assert len(state.waits) == 1 and state.paint_reads == 1
    assert_stopped_single_click(session, state)


def test_replaced_connected_handle_rejected_at_final_paint(monkeypatch):
    session, state = animated(monkeypatch, transitions=1)
    old = session.handles["white_dwarf"]
    old.evaluate = lambda script, *args: not state.clicks or not args
    with pytest.raises(BrowserSafetyStop, match="class_control_replaced"):
        select(session)
    assert_stopped_single_click(session, state)


def test_late_exact_paint_does_not_escape_subdeadline(monkeypatch):
    def late(session, state):
        state.now += 3

    session, state = animated(monkeypatch, transitions=1, on_wait=late)
    with pytest.raises(module.ClassCircleRenderingStop):
        select(session)
    assert_stopped_single_click(session, state)


def test_driver_exception_is_not_treated_as_retryable_animation(monkeypatch):
    session, state = animated(monkeypatch)
    previous = module.read_class_choices

    def read(frame):
        if state.waits:
            raise RuntimeError("driver failure")
        return previous(frame)

    monkeypatch.setattr(module, "read_class_choices", read)
    with pytest.raises(RuntimeError, match="driver failure"):
        select(session)
    assert len(state.waits) == 1
    assert_stopped_single_click(session, state)


def test_post_animation_full_capture_still_rejects_changed_public_data(monkeypatch):
    session, state = animated(monkeypatch)
    original = module.inspect_page

    def inspect(*args):
        report = original(*args)
        if state.waits:
            report["frames"][0]["text"] += " unexpected changed information"
        return report

    monkeypatch.setattr(module, "inspect_page", inspect)
    with pytest.raises(BrowserSafetyStop, match="screen_changed_during_class_read"):
        select(session)
    assert_stopped_single_click(session, state)


def test_animation_of_previous_selected_circle_is_known_only_after_original_click(monkeypatch):
    session, state = animated(
        monkeypatch,
        rendering={**paint(), "dotOpacity": "0.00683295"},
        affected="main_sequence",
    )
    # Use a fresh inherited-paint proof, as in the production observed failure.
    session.choices["selected"] = "main_sequence"
    session.choices["rendering"]["main_sequence"] = paint(True)
    session.current = lambda: None
    session.fresh_star = None
    # Explicit blank revision authorizes the original click; it adds no retry.
    receipt = session.select_class(
        "white_dwarf",
        source="reference_diagnostic",
        expected_previous="main_sequence",
        revision_reason="Explicit fixture decision on blank fields.",
    )
    assert receipt["readback_verified"] and state.clicks == 1 and len(state.waits) == 2
