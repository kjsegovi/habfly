"""Fixture-owned deep dip; production can read only exposed labels/tooltips."""
# ruff: noqa: F811

from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_chart import chart_page  # noqa: F401

from habfly.browser_flux_exposure import exposed_flux_sample
from habfly.browser_planet_chart import FluxChartSession


def test_native_vertical_search_reads_deep_dip_and_restores_axes(chart_page):
    page, frame = chart_page
    frame.evaluate("""()=>{
        const chart=document.querySelector('svg'),tip=document.querySelector('#tip');
        for(const e of chart.querySelectorAll('text'))if(e!==tip)e.remove();
        let offset=0,down=null;
        const labels=[98.4,98.8,99.2,99.6,100].map(v=>{
            const e=document.createElementNS('http://www.w3.org/2000/svg','text');
            e.setAttribute('x','27');e.setAttribute('text-anchor','end');e.setAttribute('font-size','10');
            e.setAttribute('fill','white');e.setAttribute('y',String(35+(100-v)*60));
            chart.append(e);return {e,v};
        });
        const update=()=>labels.forEach(({e,v})=>e.textContent=(v+offset).toFixed(1));
        update();tip.setAttribute('x','40');
        chart.onmousedown=e=>down=e.clientY;
        chart.onmouseup=e=>{if(down!==null){offset+=(e.clientY-down)/60;down=null;update()}};
        chart.onmousemove=e=>{
            tip.setAttribute('y',String(25+(100+offset-95)*60));
            tip.style.display='block';tip.textContent='Brightness: 95% , Day: 2625';
        };
    }""")
    events = []
    session = FluxChartSession(page, config(), lambda *e: events.append(e), max_actions=30, max_seconds=60)
    try:
        original = session.flux_axis_labels()
        sample = exposed_flux_sample(session, 0.2, max_pans=8)
        assert sample.day == 2625 and sample.brightness_percent == "95"
        assert session.flux_axis_labels() == original
        assert any(payload.get("chart_readout_unavailable") for _, payload in events)
        assert events[-1][1]["chart_visibility_recovery"]["view_restored"]
        assert frame.get_by_role("textbox").nth(1).input_value() == ""
    finally:
        session.close()
