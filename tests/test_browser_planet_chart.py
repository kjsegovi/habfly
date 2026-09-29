"""All requests are fulfilled by the controlled local fixture."""

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_chart import FluxChartSession
from habfly.browser_stellar import SIMULATION_URL


@pytest.fixture
def chart_page(page):  # noqa: F811
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("iframe").first.evaluate("e=>{e.style.width='950px';e.style.height='600px'}")
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.locator("body").evaluate("""e=>e.innerHTML=`
        <div>JYREMIS</div><div>OBSERVATIONS SPECTRUM DOPPLER SHIFT BRIGHTNESS DROP</div>
        <label>observe for<input value='10000'></label><label>doppler shift<input value=''></label>
        <select><option selected></option><option>Yes</option><option>No</option></select>
        <div><svg role='img' width='280' height='195' style='background:black'>
          <title>Normalized Flux Days Observed</title>
          <rect width='280' height='195' fill='black'/>
          <text x='25' y='25' fill='white'>100</text>
          <text id='tip' x='40' y='160' font-size='10' fill='white' style='display:none'></text>
        </svg></div><button>Save</button><input hidden value='secret'>
    `""")
    frame.evaluate("""()=>document.querySelector('svg').onmousemove=e=>{
        const t=document.querySelector('#tip');t.style.display='block';
        t.textContent='Brightness: 99.992% , Day: 2902';
    }""")
    return page, frame


def session(chart_page, **kwargs):
    browser_page, _ = chart_page
    events = []
    s = FluxChartSession(browser_page, config(), lambda *e: events.append(e), **kwargs)
    return s, events


def test_visible_hover_and_bounded_navigation(chart_page):
    s, events = session(chart_page)
    try:
        sample = s.hover(0.5, 0.4)
        assert sample.day == 2902 and sample.brightness_percent == "99.992"
        s.zoom(0.5, 0.4, -500)
        s.pan((0.3, 0.4), (0.7, 0.4))
        assert s.actions == 3
        assert all(not payload["task_completed"] for kind, payload in events if kind == "action_result")
        assert chart_page[1].get_by_role("textbox").nth(1).input_value() == ""
    finally:
        s.close()


def test_explicit_probe_returns_no_sample_for_unexposed_text_without_weakening_strict_hover(chart_page):
    _, frame = chart_page
    frame.locator("#tip").evaluate("e=>e.setAttribute('y','300')")
    s, events = session(chart_page)
    try:
        assert s.probe_hover(0.2, 0.4) is None
        assert not s.stopped and events[-1][1]["chart_readout_unavailable"]
        assert "chart_sample" not in events[-1][1]
        with pytest.raises(BrowserSafetyStop, match="missing_or_ambiguous_visible_tooltip"):
            s.hover(0.2, 0.4)
        assert s.stopped
    finally:
        s.close()


def test_probe_cannot_downgrade_changed_context_to_missing_evidence(chart_page):
    _, frame = chart_page
    s, events = session(chart_page)
    frame.get_by_role("textbox").nth(1).fill("7")
    try:
        with pytest.raises(BrowserSafetyStop, match="context_or_answers_changed"):
            s.probe_hover(0.2, 0.4)
        assert s.stopped and len(events) == 1
    finally:
        s.close()


@pytest.mark.parametrize("stale", [False, True])
def test_strict_hover_checks_only_visible_time_ticks(chart_page, stale):
    browser_page, frame = chart_page
    frame.locator("svg").evaluate(
        """(e,stale)=>{
        document.querySelector('#tip').setAttribute('y','120');
        for(const [x,value] of [[40,1000],[140,2902],[240,4804]]){
            const t=document.createElementNS('http://www.w3.org/2000/svg','text');
            t.setAttribute('x',x);t.setAttribute('y',165);t.setAttribute('text-anchor','middle');
            t.setAttribute('font-size','10');t.setAttribute('fill','white');t.textContent=value;e.append(t);
        }
        if(stale)e.onmousemove=()=>{const t=document.querySelector('#tip');t.style.display='block';t.textContent='Brightness: 100% , Day: 5919'};
    }""",
        stale,
    )
    session = FluxChartSession(browser_page, config(), lambda *_: None, verify_day_axis=True)
    try:
        if stale:
            with pytest.raises(BrowserSafetyStop, match="disagrees_with_visible_axis"):
                session.hover(0.5, 0.4)
            assert session.stopped
        else:
            assert session.hover(0.5, 0.4).day == 2902
    finally:
        session.close()


def test_time_axis_validation_rejects_missing_nonlinear_and_disagreeing_ticks():
    from habfly.browser_planet_chart import verify_tooltip_day

    labels = [{"value": str(day), "center_x": x} for x, day in [(30, 0), (145, 5000), (260, 10000)]]
    verify_tooltip_day(labels, 31, 43)
    with pytest.raises(BrowserSafetyStop, match="disagrees_with_visible_axis"):
        verify_tooltip_day(labels, 31, 5919)
    with pytest.raises(BrowserSafetyStop, match="insufficient_visible_time_axis"):
        verify_tooltip_day(labels[:2], 31, 43)
    labels[1]["value"] = "5500"
    with pytest.raises(BrowserSafetyStop, match="nonlinear_visible_time_axis"):
        verify_tooltip_day(labels, 31, 43)
    zoomed = [{"value": str(v), "center_x": x} for x, v in [(43.5, 500), (66.5, 1000), (89.5, 1500)]]
    verify_tooltip_day(zoomed, 31, 239)
    with pytest.raises(BrowserSafetyStop, match="disagrees_with_visible_axis"):
        verify_tooltip_day(zoomed, 31, 5919)


def navigation_fixture(frame):
    frame.locator("svg").evaluate("""e=>{
      const p=e.parentElement;p.style='position:relative;width:285.5px;height:198.5px';
      const button=document.createElement('button');button.style='position:absolute;left:256.5px;top:-8px;width:27px;height:27px';
      button.innerHTML='<svg width="10" height="10"></svg>';p.append(button);
      e.onwheel=()=>{if(p.querySelectorAll('button').length===1){const b=button.cloneNode(true);b.style.left='223.5px';p.append(b)}};
    }""")


def test_first_zoom_can_reveal_only_known_plot_local_reset_icon(chart_page):
    _, frame = chart_page
    navigation_fixture(frame)
    s, _ = session(chart_page)
    try:
        s.zoom(0.4, 0.1, -500)
        assert frame.locator("button").count() == 3  # Two plot icons and Save.
        assert s.hover(0.5, 0.4).day == 2902
    finally:
        s.close()


@pytest.mark.parametrize("mutation", ["text", "outside", "large", "third", "input", "reparent"])
def test_chart_navigation_exception_does_not_hide_other_changes(chart_page, mutation):
    _, frame = chart_page
    navigation_fixture(frame)
    s, _ = session(chart_page)
    commands = {
        "text": 'const b=document.querySelector("svg").parentElement.querySelector("button");b.textContent="Submit"',
        "outside": 'document.body.append(document.createElement("button"))',
        "large": 'document.querySelector("svg").parentElement.querySelector("button").style.width="28px"',
        "third": 'const p=document.querySelector("svg").parentElement,b=p.querySelector("button");p.append(b.cloneNode(true),b.cloneNode(true))',
        "input": 'document.querySelectorAll("input")[1].value="7"',
        "reparent": 'const e=document.querySelector("svg"),p=document.createElement("div");e.parentElement.before(p);p.append(e)',
    }
    if mutation == "outside":
        commands[mutation] = (
            'const b=document.createElement("button");b.textContent="Unexpected";document.body.append(b)'
        )
    frame.evaluate("body=>{new Function(body)()}", commands[mutation])
    try:
        with pytest.raises(BrowserSafetyStop):
            s.hover(0.5, 0.4)
    finally:
        s.close()


@pytest.mark.parametrize("mutation", ["answer", "outer", "star", "frame", "modal", "password"])
def test_context_change_stops_before_hover(chart_page, mutation):
    browser_page, frame = chart_page
    s, events = session(chart_page)
    if mutation == "answer":
        frame.get_by_role("textbox").nth(1).fill("7")
    elif mutation == "outer":
        browser_page.get_by_role("checkbox").check()
    elif mutation == "star":
        frame.locator("div").first.evaluate("e=>e.innerText='OTHERSTAR'")
    elif mutation == "frame":
        frame.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<iframe src=\"about:blank\"></iframe>')"
        )
    elif mutation == "modal":
        frame.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<div role=dialog>Unexpected</div>')"
        )
    else:
        frame.locator("body").evaluate("e=>e.insertAdjacentHTML('beforeend','<input type=password>')")
    try:
        with pytest.raises(BrowserSafetyStop):
            s.hover(0.5, 0.4)
        assert s.stopped and s.actions == 0
        assert len(events) == 1
    finally:
        s.close()


def test_covered_chart_cannot_be_interacted_with(chart_page):
    s, _ = session(chart_page)
    chart_page[1].locator("body").evaluate(
        "e=>e.insertAdjacentHTML('beforeend','<div style=\"position:fixed;inset:0;background:red;z-index:1000\"></div>')"
    )
    try:
        with pytest.raises(BrowserSafetyStop, match="occluded"):
            s.hover(0.5, 0.4)
        assert s.stopped
    finally:
        s.close()


def test_replaced_chart_is_stale(chart_page):
    s, _ = session(chart_page)
    chart_page[1].locator("svg").evaluate("e=>e.replaceWith(e.cloneNode(true))")
    try:
        with pytest.raises(BrowserSafetyStop, match="replaced"):
            s.hover(0.5, 0.4)
    finally:
        s.close()


def test_replaced_outer_document_body_invalidates_cached_guard_handle(chart_page):
    s, events = session(chart_page)
    browser_page, _ = chart_page
    widget = next(f for f in browser_page.frames if f != browser_page.main_frame and f.url != SIMULATION_URL)
    widget.locator("body").evaluate("e=>e.replaceWith(e.cloneNode(true))")
    try:
        with pytest.raises(BrowserSafetyStop, match="body_replaced"):
            s.hover(0.5, 0.4)
        assert s.stopped and s.actions == 0
        assert len(events) == 1
    finally:
        s.close()


def test_hidden_values_are_not_read(chart_page):
    s, _ = session(chart_page)
    chart_page[1].locator("input[hidden]").evaluate("e=>e.value='changed private data'")
    try:
        assert s.hover(0.5, 0.4).day == 2902
    finally:
        s.close()


@pytest.mark.parametrize("mutation", ["hidden", "transparent", "collapsed", "hidden_shown", "after_move"])
def test_bundled_embedding_guards_run_before_and_after_every_hover(chart_page, mutation):
    browser_page, frame = chart_page
    widget = browser_page.locator("iframe").nth(1)
    options = config()
    if mutation == "hidden_shown":
        widget.evaluate("e=>e.style.display='none'")
        options.frames[1].count = 1
    events = []
    s = FluxChartSession(browser_page, options, lambda *e: events.append(e))
    assert s.bundle_embeddings
    try:
        if mutation == "after_move":
            browser_page.evaluate(
                "addEventListener('message',e=>{if(e.data==='hide-fixture-frame')document.querySelectorAll('iframe')[1].style.display='none'})"
            )
            frame.evaluate(
                "()=>{document.querySelector('svg').onmousemove=()=>parent.postMessage('hide-fixture-frame','*')}"
            )
        else:
            widget.evaluate(
                "(e,v)=>e.style.cssText=v",
                {
                    "hidden": "display:none",
                    "transparent": "opacity:0",
                    "collapsed": "visibility:collapse",
                    "hidden_shown": "display:block",
                }[mutation],
            )
        with pytest.raises(BrowserSafetyStop, match="frame_hidden|hidden_frame_became"):
            s.hover(0.5, 0.4)
        assert s.stopped and not any(kind == "action_result" for kind, _ in events)
        assert s.actions == (1 if mutation == "after_move" else 0)
    finally:
        s.close()


def test_nested_frame_guard_uses_unchanged_fallback(chart_page):
    browser_page, frame = chart_page
    # The inert nested frame changes no scientific data. Configured boundary
    # inventory is inspected normally, then hidden-frame exposure is guarded.
    frame.locator("body").evaluate(
        "e=>e.insertAdjacentHTML('beforeend','<iframe style=\"display:none\" src=\"about:blank\"></iframe>')"
    )
    browser_page.wait_for_timeout(50)
    s, _ = session(chart_page)
    try:
        assert not s.bundle_embeddings
        assert s.hover(0.5, 0.4).day == 2902
        frame.locator("iframe").evaluate("e=>e.style.display='block'")
        with pytest.raises(BrowserSafetyStop, match="hidden_frame_became"):
            s.hover(0.5, 0.4)
    finally:
        s.close()


def test_embedding_rejected_before_any_now_hidden_child_values_are_read(chart_page):
    browser_page, frame = chart_page
    s, _ = session(chart_page)
    try:
        frame.locator("input").first.evaluate(
            "e=>Object.defineProperty(e,'value',{get(){throw Error('hidden value read')}})"
        )
        browser_page.locator("iframe").first.evaluate("e=>e.style.display='none'")
        with pytest.raises(BrowserSafetyStop, match="frame_hidden"):
            s.hover(0.5, 0.4)
        assert s.actions == 0
    finally:
        s.close()


@pytest.mark.parametrize("hidden", [True, False])
def test_tooltip_visibility_and_ambiguity(chart_page, hidden):
    s, _ = session(chart_page)
    chart_page[1].locator("svg").evaluate(
        "(e,hidden)=>{const t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x','40');t.setAttribute('y','185');t.setAttribute('fill','white');t.style.display=hidden?'none':'block';t.textContent='Brightness: 50% , Day: 5';e.append(t)}",
        hidden,
    )
    try:
        if hidden:
            assert s.hover(0.5, 0.4).day == 2902
        else:
            with pytest.raises(BrowserSafetyStop, match="ambiguous"):
                s.hover(0.5, 0.4)
    finally:
        s.close()


def test_action_budget_no_retry(chart_page):
    s, _ = session(chart_page, max_actions=1)
    try:
        s.hover(0.5, 0.4)
        with pytest.raises(BrowserSafetyStop, match="action_limit"):
            s.hover(0.5, 0.4)
        with pytest.raises(BrowserSafetyStop, match="stopped"):
            s.hover(0.5, 0.4)
        assert s.actions == 1
    finally:
        s.close()


def test_unexpected_side_effect_during_hover_stops(chart_page):
    s, _ = session(chart_page)
    chart_page[1].evaluate(
        "()=>document.querySelector('svg').onmousemove=()=>document.querySelector('input').value='123'"
    )
    try:
        with pytest.raises(BrowserSafetyStop, match="answers_changed"):
            s.hover(0.5, 0.4)
        assert s.stopped
    finally:
        s.close()


@pytest.mark.parametrize("mutation", ["clipped", "partial", "transparent", "covered"])
def test_dom_tooltip_must_actually_be_painted(chart_page, mutation):
    s, _ = session(chart_page)
    frame = chart_page[1]
    if mutation == "clipped":
        frame.locator("#tip").evaluate("e=>e.setAttribute('y','450')")
    elif mutation == "partial":
        frame.locator("#tip").evaluate("e=>e.setAttribute('x','180')")
    elif mutation == "transparent":
        frame.locator("#tip").evaluate("e=>e.style.opacity='0'")
    else:
        frame.locator("svg").evaluate("""e=>{
            const r=document.createElementNS('http://www.w3.org/2000/svg','rect');
            r.setAttribute('x','30');r.setAttribute('y','140');
            r.setAttribute('width','245');r.setAttribute('height','30');r.setAttribute('fill','red');e.append(r);
        }""")
    try:
        with pytest.raises(BrowserSafetyStop, match="visible_tooltip"):
            s.hover(0.5, 0.4)
        assert s.stopped
    finally:
        s.close()
