"""One visible hint is neither a repeat interval nor a justified No fallback.

Synthetic regression pixels reproduce only relevant Ianu plot colors, not the
course UI. Session/probe seams are injected; no native/browser/model execution.
"""

import io
from pathlib import Path

import pytest
import test_browser_shallow_transit_steps as fixtures
from PIL import Image
from test_shallow_two_hint_stop import axes

from habfly.browser import BrowserSafetyStop
from habfly.browser_shallow_transit_probe import EXACT_TWO_HINT_POLICY, candidate_columns
from habfly.planet_window_baseline_edge import analyze_planet_window

rig = fixtures.rig
SAVED = Path(__file__).resolve().parents[1] / (
    "experiments/browser-project-two-event-three-star/1c1e53a561e74d3c9fcc89e369e50987/"
    "campaign/stars/002/owner/star/window/progress-013"
)
PNG_SHA = "20c85ad0e800754bfa58ced35a79200b4b8385ae713d38bef5497a1c0c24f8e8"


def one_hint_png():
    picture = Image.new("RGB", (280, 196), "black")
    for x in range(30, 261):
        picture.putpixel((x, 21), (26, 26, 26))
    for x in range(30, 261, 23):
        for y in range(10, 152):
            picture.putpixel((x, y), (51, 51, 51))
    for x in range(30, 260):
        picture.putpixel((x, 20), (80, 131, 153))
    picture.putpixel((241, 21), (29, 36, 38))
    picture.putpixel((242, 21), (33, 46, 51))
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    return stream.getvalue()


def assert_insufficient(raw, times, flux):
    assert candidate_columns(raw, flux) == [[242]]
    with pytest.raises(BrowserSafetyStop, match="exact_two_overview_hints_required"):
        candidate_columns(raw, flux, hint_policy=EXACT_TWO_HINT_POLICY)
    negative = analyze_planet_window(raw, times, flux, requested_days=5000)
    assert negative["reason"] == "unknown_plot_palette"
    assert negative["endpoint_visible"] is True
    assert negative["planet_decision"] is None
    assert negative["absence_proven"] is negative["task_completed"] is False


def test_one_localized_hint_is_not_silently_repaired_or_called_no():
    assert_insufficient(one_hint_png(), *axes())


def test_new_optin_stops_before_sensor_reservation_or_native_actions(rig):
    data = one_hint_png()
    rig.state.image = data
    (rig.source / "chart.png").write_bytes(data)
    fixtures.save(rig.source / "report.json", rig.report(data))
    with pytest.raises(BrowserSafetyStop, match="exact_two_overview_hints_required"):
        rig.make(allow_two_events=True, source_report_sha256=fixtures.sha(rig.source / "report.json"))
    assert not rig.state.calls and not rig.state.native
    assert not (rig.root / "sensor").exists()
    assert not (rig.root / "shallow-transit-owner-claims").exists()


def test_optional_saved_ianu_stays_immutable_and_has_no_spacing():
    if not SAVED.is_dir():
        pytest.skip("Optional Ianu capture not distributed")
    raw, report = (SAVED / "chart.png").read_bytes(), fixtures.load(SAVED / "report.json")
    assert fixtures.sha(SAVED / "chart.png") == PNG_SHA
    assert report["star"] == "IANU" and report["chart_sha256"] == PNG_SHA
    assert_insufficient(raw, report["time_axis_labels"], report["flux_axis_labels"])
    assert (SAVED / "chart.png").read_bytes() == raw
