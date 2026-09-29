"""Raster capture exposure uses the visible plot, never a covered background."""
# ruff: noqa: F811

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_chart import chart_page  # noqa: F401

from habfly.browser_observation_progress import PLOT_EXPOSED


def test_known_chart_is_exposed_even_with_an_underlying_screen(chart_page):
    _, frame = chart_page
    chart = frame.get_by_role("img")
    assert chart.evaluate(PLOT_EXPOSED)
    frame.locator("body").evaluate("""e=>{
        const background=document.createElement('div');
        background.style='position:fixed;inset:0;background:blue;z-index:-1';
        e.prepend(background);
    }""")
    assert chart.evaluate(PLOT_EXPOSED)


@pytest.mark.parametrize("change", ["overlay", "thin", "hidden", "opacity", "clipped", "resized"])
def test_unknown_or_unexposed_plot_is_rejected(chart_page, change):
    _, frame = chart_page
    chart = frame.get_by_role("img")
    handle = chart.element_handle()
    if change in {"overlay", "thin"}:
        chart.evaluate(
            """(e,thin)=>{
            const b=e.getBoundingClientRect(),cover=document.createElement('div');
            cover.style=`position:fixed;left:${b.x+80}px;top:${b.y+20}px;width:${thin?5:80}px;height:130px;background:black;z-index:999`;
            document.body.append(cover);
        }""",
            change == "thin",
        )
    elif change == "hidden":
        chart.evaluate("e=>e.style.visibility='hidden'")
    elif change == "opacity":
        chart.evaluate("e=>e.parentElement.style.opacity='0'")
    elif change == "clipped":
        chart.evaluate("e=>e.parentElement.style='width:100px;overflow:hidden'")
    else:
        chart.evaluate("e=>e.setAttribute('width','300')")
    assert not handle.evaluate(PLOT_EXPOSED)
