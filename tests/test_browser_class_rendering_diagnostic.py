"""Failure-only class paint evidence; all browser and network seams injected."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by argument name

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_class_setup_steps import create, offline_only, rig  # noqa: F401
from test_browser_star_preflight import stellar

import habfly.browser_class_setup_steps as steps_module
import habfly.browser_classification as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_numeric import digest
from habfly.browser_stellar import CLASSES, SIMULATION_URL


def paint(selected=False):
    return {
        "tag": "LABEL",
        "width": "26px",
        "height": "26px",
        "border": "1px solid rgb(255, 255, 255)" if selected else "1px solid rgb(0, 200, 220)",
        "display": "block",
        "opacity": "1",
        "content": '""',
        "dotWidth": "14px",
        "dotHeight": "14px",
        "dotDisplay": "block",
        "dotOpacity": "1" if selected else "0",
        "dotColor": "rgb(255, 255, 255)" if selected else "rgba(0, 0, 0, 0)",
    }


def unsupported():
    return {**paint(), "dotOpacity": "0.431", "dotColor": "rgba(255, 255, 255, 0.431)"}


@pytest.mark.parametrize("selected", [False, True])
def test_exact_accepted_paint_unchanged(selected):
    assert module.painted_selection(paint(selected)) is selected


@pytest.mark.parametrize(
    "key,value",
    [("dotOpacity", "0.431"), ("width", "26.1px"), ("border", "1px solid rgb(254, 255, 255)")],
)
def test_new_diagnostics_do_not_accept_intermediate_paint(key, value):
    with pytest.raises(BrowserSafetyStop, match="^unsupported_class_circle_rendering$"):
        module.painted_selection({**paint(), key: value})


@pytest.mark.parametrize("key", tuple(paint()))
def test_unknown_computed_strings_redacted(key):
    data = {
        **paint(),
        key: "secret http://localhost?session=private",
        "checked": True,
        "className": "private-css",
        "onclick": "private-handler",
        "radioValue": "private",
    }
    error = module.ClassCircleRenderingStop("white_dwarf", data).at_read("initial_read")
    assert error.diagnostic["visible_rendering"][key] is None
    assert set(error.diagnostic["visible_rendering"]) == set(paint())
    assert not any(word in json.dumps(error.diagnostic) for word in ("private", "secret", "http", "checked"))


def test_bounded_visible_fractional_paint_retained_without_acceptance():
    error = module.ClassCircleRenderingStop("white_dwarf", unsupported())
    assert error.diagnostic["visible_rendering"] == unsupported()
    assert error.diagnostic["validation_stage"] == "paint_state"
    assert error.diagnostic["additional_dom_queries"] == 0
    assert error.diagnostic["paint_acceptance_changed"] is False
    assert (
        module.ClassCircleRenderingStop("white_dwarf", {**paint(), "width": "27px"}).diagnostic[
            "validation_stage"
        ]
        == "common_geometry"
    )
    assert module.ClassCircleRenderingStop("secret", {}).diagnostic["affected_choice"] is None


@pytest.mark.parametrize(
    "args",
    [
        {"phase": "private-url"},
        {"phase": "initial_read", "click_invoked": 1},
        {"phase": "post_click_read", "click_returned": True},
    ],
)
def test_phase_and_dispatch_flags_are_fixed_typed_values(args):
    with pytest.raises(ValueError):
        module.ClassCircleRenderingStop("white_dwarf", unsupported()).at_read(**args)


def test_reader_failure_uses_existing_single_computed_read(monkeypatch):
    calls = []

    def evaluate(script):
        calls.append(script)
        return unsupported()

    circle = SimpleNamespace(
        is_visible=lambda: True,
        evaluate=evaluate,
        bounding_box=lambda: {"x": 0, "y": 0, "width": 26, "height": 26},
    )
    parent = SimpleNamespace(
        inner_text=lambda: "White Dwarf",
        locator=lambda selector: SimpleNamespace(all=lambda: [circle]),
    )
    label = SimpleNamespace(
        is_visible=lambda: True,
        locator=lambda selector: parent,
        bounding_box=lambda: {"x": 0, "y": 30, "width": 26, "height": 10},
    )
    frame = SimpleNamespace(get_by_text=lambda _: SimpleNamespace(all=lambda: [label]))
    monkeypatch.setattr(module, "rendered_control", lambda e: e is circle)
    with pytest.raises(module.ClassCircleRenderingStop) as caught:
        module._read_choice_circles(frame, ("white_dwarf",))
    assert calls == [module.RENDERING]
    assert caught.value.diagnostic["affected_choice"] == "white_dwarf"
    assert str(caught.value) == "unsupported_class_circle_rendering"
    assert all(hidden not in module.RENDERING for hidden in ("checked", "className", "onclick", ".value"))


def real_session(monkeypatch, failure=None):
    state = SimpleNamespace(selected=None, reads=0, clicks=0, events=[])
    report = stellar()
    frame = SimpleNamespace(url=SIMULATION_URL)
    page = SimpleNamespace(frames=[frame], main_frame=object())
    page.context = SimpleNamespace(pages=[page])

    def click(*, timeout):
        assert timeout == 3000
        state.clicks += 1
        state.selected = "white_dwarf"

    handle = SimpleNamespace(click=click, evaluate=lambda *args: True)
    handles = {name: handle for name in CLASSES}

    def read_choices(_):
        state.reads += 1
        if state.reads == failure:
            raise module.ClassCircleRenderingStop("main_sequence", unsupported())
        return {
            "selected": state.selected,
            "rendering": {name: paint(state.selected == name) for name in CLASSES},
        }, handles

    monkeypatch.setattr(module, "inspect_page", lambda *_: deepcopy(report))
    monkeypatch.setattr(module, "_visible_frame", lambda *_: True)
    monkeypatch.setattr(module, "read_class_choices", read_choices)
    session = module.StellarSelectionSession(
        page, None, lambda kind, payload: state.events.append((kind, payload))
    )
    return session, state


@pytest.mark.parametrize(
    "failure,phase,clicks",
    [(2, "pre_selection_read", 0), (3, "post_click_read", 1)],
)
def test_real_session_distinguishes_preclick_and_returned_click(monkeypatch, failure, phase, clicks):
    session, state = real_session(monkeypatch, failure)
    with pytest.raises(module.ClassCircleRenderingStop) as caught:
        session.select_class("white_dwarf", source="reference_diagnostic")
    diagnostic = caught.value.diagnostic
    assert diagnostic["failure_phase"] == phase
    assert diagnostic["native_class_click_invoked"] is bool(clicks)
    assert diagnostic["native_class_click_returned"] is bool(clicks)
    assert diagnostic["readback_verified"] is False
    assert state.clicks == clicks
    assert len(state.events) == clicks  # post-click only the proposal survives
    if clicks:
        assert session.stopped


def test_real_success_payload_bytes_remain_legacy(monkeypatch):
    session, state = real_session(monkeypatch)
    receipt = session.select_class("white_dwarf", source="reference_diagnostic")
    expected = {
        "selected_class": "white_dwarf",
        "selection_source": "reference_diagnostic",
        "readback_verified": True,
        "correctness_verified": False,
        "rendering_sha256": digest(
            {
                "selected": "white_dwarf",
                "rendering": {name: paint(name == "white_dwarf") for name in CLASSES},
            }
        ),
        "task_completed": False,
    }
    events = [
        (
            "action_proposed",
            {
                "kind": "SELECT",
                "target": "stellar_class",
                "value": "white_dwarf",
                "action_source": "reference_diagnostic",
                "correctness_verified": False,
            },
        ),
        ("action_result", expected),
    ]
    assert json.dumps(receipt).encode() == json.dumps(expected).encode()
    assert json.dumps(state.events).encode() == json.dumps(events).encode()
    assert state.clicks == 1


def test_setup_records_callback_failure_before_any_native_dispatch(rig, monkeypatch):
    rig.fresh_receipt("main_sequence")

    def fail(_):
        raise module.ClassCircleRenderingStop("main_sequence", unsupported())

    monkeypatch.setattr(steps_module, "read_class_choices", fail)
    owner = create(rig)
    result = owner.advance()
    assert result["failure_reason"] == "unsupported_class_circle_rendering"
    assert result["class_clicks_may_have_occurred"] == 0
    assert not rig.page.writes and not result["setup_verified"]
    source = result["class_rendering_failure"]
    path = rig.history / source["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    diagnostic = json.loads(path.read_bytes())
    assert diagnostic["requested_choice"] == "white_dwarf"
    assert diagnostic["affected_choice"] == "main_sequence"
    assert diagnostic["failure_phase"] == "pre_dispatch_native_binding"
    assert diagnostic["native_class_click_invoked"] is False
    assert diagnostic["native_class_click_returned"] is False
    assert not diagnostic["task_completed"] and not diagnostic["automatic_retry"]
    assert not (owner.output / "class/confirmed.json").exists()
    before = deepcopy(rig.page.writes)
    owner.advance()
    assert rig.page.writes == before


def test_setup_records_postclick_failure_without_claiming_verified_selection(rig, monkeypatch):
    def fail(self, name, *, source, **_):
        self.emit(
            "action_proposed",
            {
                "kind": "SELECT",
                "target": "stellar_class",
                "value": name,
                "action_source": source,
                "correctness_verified": False,
            },
        )
        rig.page.writes.append(("class", name))
        raise module.ClassCircleRenderingStop(name, unsupported()).at_read(
            "post_click_read",
            click_invoked=True,
            click_returned=True,
        )

    monkeypatch.setattr(steps_module.StellarSelectionSession, "select_class", fail)
    owner = create(rig, "white_dwarf")
    result = owner.advance()
    assert result["class_clicks_may_have_occurred"] == 1
    assert len(rig.page.writes) == 1 and not result["setup_verified"]
    diagnostic = json.loads((owner.output / "class-rendering-failure.json").read_bytes())
    assert diagnostic["native_class_click_invoked"] is True
    assert diagnostic["native_class_click_returned"] is True
    assert diagnostic["readback_verified"] is False
    assert not (owner.output / "class/confirmed.json").exists()


def test_setup_success_has_no_diagnostic_fields_or_files(rig):
    owner = create(rig, "white_dwarf")
    result = owner.advance()
    assert result["setup_verified"]
    assert "class_rendering_failure" not in result
    assert not (owner.output / "class-rendering-failure.json").exists()
    assert all("class_rendering_failure" not in event["payload"] for event in rig.events)


@pytest.mark.parametrize("failed_operation", ["persist", "pin"])
def test_diagnostic_io_failure_still_stops_original_cause_without_retry(rig, monkeypatch, failed_operation):
    rig.fresh_receipt("main_sequence")

    def fail(_):
        raise module.ClassCircleRenderingStop("main_sequence", unsupported())

    monkeypatch.setattr(steps_module, "read_class_choices", fail)
    owner = create(rig)
    previous_persist, previous_pin = steps_module.persist_json, owner._pin

    def failing_persist(path, payload):
        if path.name == "class-rendering-failure.json":
            raise OSError("secret diagnostic disk failure")
        return previous_persist(path, payload)

    def failing_pin(path):
        if path.name == "class-rendering-failure.json":
            raise OSError("secret diagnostic pin failure")
        return previous_pin(path)

    if failed_operation == "persist":
        monkeypatch.setattr(steps_module, "persist_json", failing_persist)
    else:
        monkeypatch.setattr(owner, "_pin", failing_pin)
    result = owner.advance()
    assert result["status"] == "stopped"
    assert result["failure_reason"] == "unsupported_class_circle_rendering"
    assert result["class_rendering_failure"]["artifact_recording_failed"] is True
    assert "path" not in result["class_rendering_failure"]
    assert "sha256" not in result["class_rendering_failure"]
    assert "secret" not in json.dumps(result)
    assert json.loads((owner.output / "stopped.json").read_bytes()) == result
    owner.advance()
    assert not rig.page.writes and result["class_clicks_may_have_occurred"] == 0
