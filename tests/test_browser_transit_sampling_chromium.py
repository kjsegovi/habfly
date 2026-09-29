"""Synthetic chart model is fixture-owned; production reads only visible tips."""
# ruff: noqa: F811

from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_chart import chart_page  # noqa: F401

from habfly.browser_planet_chart import FluxChartSession
from habfly.browser_transit_sampling import sample_transits


def test_fractional_native_pointer_coordinates_do_not_guarantee_daily_resolution(chart_page):
    page, frame = chart_page
    frame.evaluate("""()=>{
        const chart=document.querySelector('svg'), tip=document.querySelector('#tip');
        chart.onmousemove=e=>{
            const day=Math.round(1000+(e.clientX-chart.getBoundingClientRect().left)*4);
            tip.style.display='block';tip.textContent=`Brightness: 100% , Day: ${day}`;
        };
    }""")
    session = FluxChartSession(page, config(), lambda *_: None, max_actions=5, max_seconds=30)
    try:
        days = [session.hover(x / 280, 80 / 195).day for x in (40, 40.25, 40.5, 40.75, 41)]
        # Chromium mousemove coordinates are quantized in this native fixture.
        # Fractional Playwright inputs must not be treated as daily coverage.
        assert days == [1160, 1160, 1160, 1160, 1164]
    finally:
        session.close()


def test_bounded_sampler_uses_native_zoom_hover_and_pan_with_visible_labels(chart_page):
    page, frame = chart_page
    frame.evaluate("""()=>{
        const chart=document.querySelector('svg'), tip=document.querySelector('#tip');
        for(const text of chart.querySelectorAll('text')){if(text!==tip)text.remove()}
        let scale=2, offset=1000, down=null, baseline=21;
        const labels=[80,90,100].map(v=>{
            const e=document.createElementNS('http://www.w3.org/2000/svg','text');
            e.setAttribute('x','27');e.setAttribute('text-anchor','end');e.setAttribute('font-size','10');
            e.setAttribute('fill','white');e.textContent=String(v);chart.append(e);return {e,v};
        });
        const update=()=>labels.forEach(({e,v})=>e.setAttribute('y',String(baseline+100-v+4)));
        update();
        chart.onwheel=e=>{e.preventDefault();scale/=2};
        chart.onmousedown=e=>{down=[e.clientX,e.clientY]};
        chart.onmouseup=e=>{if(down!==null){offset+=(down[0]-e.clientX)*scale;baseline+=e.clientY-down[1];down=null;update()}};
        chart.onmousemove=e=>{
            const day=Math.round(offset+(e.clientX-chart.getBoundingClientRect().left)*scale);
            tip.style.display='block';
            tip.textContent=`Brightness: ${day%40===0?'99.99':'100'}% , Day: ${day}`;
        };
    }""")
    events = []
    session = FluxChartSession(
        page,
        config(),
        lambda event, payload: events.append((event, payload)),
        max_actions=200,
        max_seconds=60,
    )
    try:
        report = sample_transits(session, max_windows=3, max_zooms=2)
        assert report["status"] == "periodic_transits_observed" and report["period_days"] == "40"
        assert report["brightness_drop_percent"] == "0.01" and report["coverage_complete"]
        assert report["has_planet_answer"] is None and not report["task_completed"]
        assert {p["kind"] for e, p in events if e == "action_proposed"} == {"HOVER", "SCROLL", "DRAG"}
        assert frame.get_by_role("textbox").nth(1).input_value() == ""
    finally:
        session.close()
