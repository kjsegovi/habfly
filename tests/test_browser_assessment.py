"""Real Chromium capture with every request fulfilled by the local fixture."""

from html import escape

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_project_assessment import visible

from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment import map_assessment_capture
from habfly.browser_probe import inspect_page
from habfly.browser_stellar import SIMULATION_URL


@pytest.mark.parametrize("mode", ["data_quality", "scavenger_hunt", "automation"])
def test_frame_scoped_read_only_assessment_mapping(page, mode):  # noqa: F811
    simulation = page.frame(url=SIMULATION_URL)
    simulation.locator("body").evaluate(
        "(body, text) => body.innerHTML = '<div>' + text + '</div>' + "
        "'<p hidden>FUNDING $999999 Private answer</p>' + "
        "'<button onclick=\"document.body.replaceChildren()\">ASSESS</button>'",
        escape(visible(mode=mode)),
    )
    report = inspect_page(page, config())
    mapped = map_assessment_capture(report, capture_sha256="1" * 64)
    assert mapped["assessment"]["funding"] == 50000
    assert mapped["assessment"]["mode"] == mode
    assert mapped["actions_executed"] == 0 and not mapped["task_completed"]
    assert simulation.get_by_role("button", name="ASSESS").is_visible()
    assert not page.get_by_role("checkbox").is_checked()


def test_assessment_probe_still_rejects_navigation_escape(page):  # noqa: F811
    page.evaluate("history.pushState({}, '', '/outside')")
    with pytest.raises(BrowserSafetyStop, match="navigation_outside_activity"):
        inspect_page(page, config())
