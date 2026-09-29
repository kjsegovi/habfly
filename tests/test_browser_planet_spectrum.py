import json
from types import SimpleNamespace

import pytest
from test_browser_numeric import config
from test_browser_planet_chart import chart_page, chromium, page, session  # noqa: F401

import habfly.browser_planet_spectrum as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_spectrum import capture_spectrum_excursion, hover_spectrum_marker


@pytest.fixture
def spectrum_page(chart_page):  # noqa: F811
    frame = chart_page[1]
    frame.locator("body").evaluate("""e=>e.insertAdjacentHTML('beforeend',`
        <div style='position:relative;width:285px;height:100px'>
          <div style='position:relative;width:285px;height:40px;background:rgb(193,58,44)'>
            <div style='position:absolute;left:100px;width:1px;height:38px;background:black' onmousemove="this.parentElement.nextElementSibling.style.display='block';this.parentElement.nextElementSibling.innerText='656.299999371nm'"></div>
            <div style='position:absolute;left:160px;width:1px;height:38px;background:black' onmousemove="this.parentElement.nextElementSibling.style.display='block';this.parentElement.nextElementSibling.innerText='656.300000629nm'"></div>
          </div>
          <div style='background:rgb(250,250,220);display:none;position:absolute;top:45px;left:60px'></div>
          <div style='position:absolute;top:75px'>656.3nm</div>
        </div>`)
    """)
    return chart_page


@pytest.mark.parametrize("iframe_offset", [None, 200.0, 200.25, 200.5, 200.84375])
def test_only_selected_visible_marker_is_hovered(spectrum_page, iframe_offset):
    if iframe_offset is not None:
        spectrum_page[1].frame_element().evaluate(
            "(e,x)=>{e.style.position='absolute';e.style.left=x+'px';e.style.top='10px'}", iframe_offset
        )
    s, events = session(spectrum_page, max_actions=2)
    try:
        assert hover_spectrum_marker(s, "blue")["wavelength_text"] == "656.299999371nm"
        assert hover_spectrum_marker(s, "red")["wavelength_text"] == "656.300000629nm"
        assert s.actions == 2
        assert all(not p["task_completed"] for k, p in events if k == "action_result")
    finally:
        s.close()


def test_capture_is_compatible_with_positive_measurement_loader(spectrum_page, tmp_path):
    from habfly.browser_raster_planet_evidence import _spectrum

    observed = []
    s, _ = session(spectrum_page)
    star = s.star
    s.close()
    output = tmp_path / "spectrum"
    result = capture_spectrum_excursion(
        spectrum_page[0],
        config(),
        output,
        expected_star=star.title(),
        emit=lambda *args: observed.append(args),
    )
    assert _spectrum(result)["star"].casefold() == star.casefold()
    assert [event[0] for event in observed] == [
        "observation",
        "action_proposed",
        "action_result",
        "action_proposed",
        "action_result",
    ]
    assert json.loads((output / "spectrum.json").read_text()) == result
    assert len(list(output.glob("event-*.json"))) == 5
    assert not (output / "stopped.json").exists()


def test_capture_other_star_stops_before_hover(spectrum_page, tmp_path):
    output = tmp_path / "spectrum"
    with pytest.raises(BrowserSafetyStop, match="spectrum_capture_star_changed"):
        capture_spectrum_excursion(spectrum_page[0], config(), output, expected_star="Another Star")
    stopped = json.loads((output / "stopped.json").read_text())
    assert stopped["hover_actions_started"] == 0
    assert stopped["answer_writes"] == 0
    assert not (output / "spectrum.json").exists()


def test_capture_callback_failure_stops_before_pointer_action(spectrum_page, tmp_path):
    s, _ = session(spectrum_page)
    star = s.star
    s.close()

    def fail(kind, payload):
        if kind == "action_proposed":
            raise RuntimeError("secret-url")

    output = tmp_path / "spectrum"
    with pytest.raises(BrowserSafetyStop, match="^spectrum_event_forwarding_failed$"):
        capture_spectrum_excursion(spectrum_page[0], config(), output, expected_star=star, emit=fail)
    assert not (output / "spectrum.json").exists()
    assert len(list(output.glob("event-*.json"))) == 2
    assert "secret" not in (output / "stopped.json").read_text()


@pytest.mark.parametrize("seconds", [0, 121, True, "60", float("nan")])
def test_capture_budget_rejects_before_io(tmp_path, seconds):
    output = tmp_path / "spectrum"
    with pytest.raises(ValueError):
        capture_spectrum_excursion(None, None, output, expected_star="Example", max_seconds=seconds)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["offstrip", "ambiguous", "covered"])
def test_invalid_visible_marker_stops(spectrum_page, mutation):
    s, _ = session(spectrum_page)
    strip = spectrum_page[1].get_by_text("656.3nm", exact=True).locator("..").locator(":scope > div").first
    if mutation == "offstrip":
        strip.locator(":scope > div").first.evaluate("e=>e.style.left='-100px'")
    elif mutation == "ambiguous":
        strip.locator(":scope > div").first.evaluate("e=>e.after(e.cloneNode(true))")
    else:
        strip.evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<div style=\"position:absolute;inset:0;background:red;z-index:100\"></div>')"
        )
    try:
        with pytest.raises(BrowserSafetyStop):
            hover_spectrum_marker(s, "blue")
        assert s.stopped
    finally:
        s.close()


@pytest.fixture
def offline_settle(monkeypatch):
    """Only the read-only poll is injected; no browser or pointer action exists."""
    state = SimpleNamespace(
        clock=0.0,
        readings=["656.29999844nm"],
        reads=0,
        guards=0,
        waits=[],
        marker={"bound": True, "hovered": True, "hit": True},
        exposed=True,
        multiple=False,
    )
    monkeypatch.setattr(module.time, "monotonic", lambda: state.clock)

    class Tip:
        def is_visible(self):
            return True

        def inner_text(self):
            return state.current

        def element_handle(self, **_):
            return self

        def evaluate(self, script, arg=None):
            if script.startswith("(a,b)"):
                return arg is self
            return state.exposed if script == module.VISIBLE_TOOLTIP else True

    tip = Tip()

    def tips():
        state.current = state.readings[min(state.reads, len(state.readings) - 1)]
        state.reads += 1
        return [] if state.current is None else [tip] * (2 if state.multiple else 1)

    def guard():
        state.guards += 1
        if state.clock > s.started + s.max_seconds:
            raise BrowserSafetyStop("chart_time_limit")

    def wait(milliseconds):
        state.waits.append(milliseconds)
        state.clock += milliseconds / 1000

    parent = SimpleNamespace(get_by_text=lambda *_: SimpleNamespace(all=tips))
    line = SimpleNamespace(
        evaluate=lambda script, *_: state.marker if script == module._BOUND_MARKER else True
    )
    s = SimpleNamespace(
        started=0,
        max_seconds=120,
        _guard=guard,
        _spectrum_diagnostic={},
        page=SimpleNamespace(wait_for_timeout=wait),
    )
    return state, s, parent, line


@pytest.mark.parametrize("old", [None, "656.30000156nm", "656.3nm"])
def test_offline_missing_or_wrong_side_tip_must_settle_without_another_hover(offline_settle, old):
    state, s, parent, line = offline_settle
    state.readings = [old, old, "656.29999844nm", "656.29999844nm"]
    assert module._settled_reading(s, parent, line, {}, "blue") == "656.29999844nm"
    assert state.reads == 4 and 0 < state.clock < module._SETTLE_SECONDS
    assert s._spectrum_diagnostic["marker_readback"] == {"bound": True, "hovered": True, "hit": True}


@pytest.mark.parametrize("value", ["656.30000156nm", "656.3nm", "0nm"])
def test_offline_wrong_side_equal_or_zero_never_substitutes_an_endpoint(offline_settle, value):
    state, s, parent, line = offline_settle
    state.readings = [value]
    with pytest.raises(BrowserSafetyStop, match="^spectrum_tooltip_wrong_side$"):
        module._settled_reading(s, parent, line, {}, "blue")
    assert state.clock == module._SETTLE_SECONDS
    assert s._spectrum_diagnostic["last_visible_wavelength_text"] == value


@pytest.mark.parametrize("guard", ["bound", "hit"])
def test_offline_unbound_or_wrong_native_hover_stops_before_tooltip_read(offline_settle, guard):
    state, s, parent, line = offline_settle
    state.marker[guard] = False
    with pytest.raises(BrowserSafetyStop, match="^spectrum_selected_marker_not_hovered$"):
        module._settled_reading(s, parent, line, {}, "blue")
    assert state.reads == 0 and state.waits == []


def test_offline_css_hover_is_diagnostic_not_a_fabricated_native_delivery_claim(offline_settle):
    state, s, parent, line = offline_settle
    state.marker["hovered"] = False
    assert module._settled_reading(s, parent, line, {}, "blue") == "656.29999844nm"
    assert s._spectrum_diagnostic["marker_readback"]["hovered"] is False


@pytest.mark.parametrize("problem", ["multiple", "unexposed", "invalid"])
def test_offline_ambiguity_or_unexposed_tooltips_are_not_smoothed_over(offline_settle, problem):
    state, s, parent, line = offline_settle
    if problem == "multiple":
        state.multiple = True
    elif problem == "unexposed":
        state.exposed = False
    else:
        state.readings = ["private exception text"]
    with pytest.raises(BrowserSafetyStop):
        module._settled_reading(s, parent, line, {}, "blue")
    assert state.clock == 0 and state.waits == []


def test_offline_settle_cannot_extend_original_session_deadline(offline_settle):
    state, s, parent, line = offline_settle
    state.readings, s.max_seconds = [None], 0.075
    with pytest.raises(BrowserSafetyStop, match="missing_or_ambiguous_spectrum_tooltip"):
        module._settled_reading(s, parent, line, {}, "blue")
    assert state.clock == 0.075 and state.waits == [50, pytest.approx(25)]


@pytest.mark.parametrize(
    "blue,red,reason",
    [
        ("656.30000156nm", "656.30000156nm", "invalid_spectrum_excursion_order"),
        ("656.3nm", "656.3nm", "invalid_spectrum_excursion_order"),
        ("656.299999371nm", "656.30000063nm", "asymmetric_visible_spectrum_excursion"),
    ],
)
def test_offline_post_event_four_validation_has_precise_safe_cause(tmp_path, monkeypatch, blue, red, reason):
    from habfly import browser_planet_chart

    sessions = []

    class Session:
        def __init__(self, _page, _config, emit, **_):
            self.star, self.actions, self.emit, self.closed = "EXAMPLE", 0, emit, False
            sessions.append(self)
            emit("observation", {"chart": {"source": "visible_tooltips", "star": self.star}})

        def _guard(self):
            pass

        def close(self):
            self.closed = True

    def saved_reading(s, side):
        s.actions += 1
        s.emit(
            "action_proposed", {"kind": "HOVER", "surface": "spectrum", "sequence": s.actions, "marker": side}
        )
        reading = {
            "marker": side,
            "wavelength_text": blue if side == "blue" else red,
            "source": "visible_spectrum_tooltip",
        }
        s.emit("action_result", {"sequence": s.actions, "spectrum_sample": reading, "task_completed": False})
        return reading

    monkeypatch.setattr(browser_planet_chart, "FluxChartSession", Session)
    monkeypatch.setattr(module, "hover_spectrum_marker", saved_reading)
    output = tmp_path / "offline"
    with pytest.raises(BrowserSafetyStop, match="^spectrum_" + reason + "$"):
        capture_spectrum_excursion(None, None, output, expected_star="Example")
    stopped = json.loads((output / "stopped.json").read_text())
    assert stopped["reason"] == "spectrum_" + reason
    assert stopped["failure_stage"] == "excursion_validation"
    assert stopped["hover_actions_started"] == 2
    assert len(list(output.glob("event-*.json"))) == 5
    assert sessions[0].closed and not (output / "spectrum.json").exists()


@pytest.mark.parametrize(
    "error",
    [
        ValueError("private password"),
        RuntimeError("https://private/token"),
        BrowserSafetyStop("private_password"),
    ],
)
def test_offline_unexpected_exception_diagnostic_is_sanitized(tmp_path, monkeypatch, error):
    from habfly import browser_planet_chart

    def fail(*_, **__):
        raise error

    monkeypatch.setattr(browser_planet_chart, "FluxChartSession", fail)
    output = tmp_path / "offline"
    with pytest.raises(BrowserSafetyStop, match="^spectrum_capture_failed$"):
        capture_spectrum_excursion(None, None, output, expected_star="Example")
    raw = (output / "stopped.json").read_text()
    assert "private" not in raw and "password" not in raw and "token" not in raw
    stopped = json.loads(raw)
    assert stopped["failure_stage"] == "session_initialization"
    assert stopped["hover_actions_started"] == 0


@pytest.mark.parametrize("initial", ["old_red", "missing"])
def test_native_tooltip_settle_requires_new_side_after_single_move(spectrum_page, initial):
    _, frame = spectrum_page
    parent = frame.get_by_text("656.3nm", exact=True).locator("..")
    parent.evaluate(
        """(p,initial)=>{
      const strip=p.children[0],tip=p.children[1];
      if(initial==='old_red'){tip.style.display='block';tip.innerText='656.300000629nm';}
      strip.children[0].onmousemove=()=>setTimeout(()=>{tip.style.display='block';tip.innerText='656.299999371nm'},150);
    }""",
        initial,
    )
    s, events = session(spectrum_page, max_actions=2)
    try:
        assert hover_spectrum_marker(s, "blue")["wavelength_text"] == "656.299999371nm"
        assert s.actions == 1 and len([e for e in events if e[0] == "action_proposed"]) == 1
        assert hover_spectrum_marker(s, "red")["wavelength_text"] == "656.300000629nm"
        assert s.actions == 2
    finally:
        s.close()


def test_native_identical_wrong_side_tip_stops_and_preserves_rendered_geometry(spectrum_page, tmp_path):
    _, frame = spectrum_page
    parent = frame.get_by_text("656.3nm", exact=True).locator("..")
    parent.evaluate("""p=>{const tip=p.children[1];tip.style.display='block';tip.innerText='656.30000156nm';
      p.children[0].children[0].onmousemove=()=>{};}""")
    output = tmp_path / "wrong-side"
    with pytest.raises(BrowserSafetyStop, match="^spectrum_tooltip_wrong_side$"):
        capture_spectrum_excursion(spectrum_page[0], config(), output, expected_star="Jyremis")
    stopped = json.loads((output / "stopped.json").read_text())
    assert stopped["failure_stage"] == "blue_hover" and stopped["hover_actions_started"] == 1
    assert stopped["visible_diagnostic"]["last_visible_wavelength_text"] == "656.30000156nm"
    assert len(stopped["visible_diagnostic"]["marker_boxes"]) == 2
    assert stopped["visible_diagnostic"]["marker_readback"]["bound"] is True
    assert stopped["visible_diagnostic"]["marker_readback"]["hit"] is True
    assert type(stopped["visible_diagnostic"]["marker_readback"]["hovered"]) is bool
    assert not (output / "spectrum.json").exists()


@pytest.mark.parametrize("separation", [3, 1, 0])
def test_native_adjacent_or_overlapping_markers_never_accept_wrong_delivery(spectrum_page, separation):
    _, frame = spectrum_page
    strip = frame.get_by_text("656.3nm", exact=True).locator("..").locator(":scope > div").first
    strip.locator(":scope > div").nth(1).evaluate("(e,gap)=>e.style.left=(100+gap)+'px'", separation)
    s, _ = session(spectrum_page, max_actions=2)
    try:
        if separation == 3:
            assert hover_spectrum_marker(s, "blue")["wavelength_text"] == "656.299999371nm"
            assert hover_spectrum_marker(s, "red")["wavelength_text"] == "656.300000629nm"
        else:
            with pytest.raises(BrowserSafetyStop, match="occluded|selected_marker_not_hovered"):
                hover_spectrum_marker(s, "blue")
        assert s.actions <= 2
    finally:
        s.close()
