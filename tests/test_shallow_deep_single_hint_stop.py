"""One deeper overview feature cannot supply a repeat interval or justify No.

Synthetic pixels reproduce Jarlarth's decision-relevant plot geometry only,
not a physical transit or native tooltip. The optional saved-crop test is
hash-pinned and read-only. Session seams are injected; no browser/model runs.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import test_browser_shallow_transit_steps as fixtures
from PIL import Image
from test_shallow_single_hint_stop import one_hint_png
from test_shallow_two_hint_stop import POLICIES, axes

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_mask
from habfly.browser_shallow_transit_probe import EXACT_TWO_HINT_POLICY, candidate_columns
from habfly.planet_window_baseline_band import analyze_planet_window
from habfly.planet_window_dip import analyze_window_dip

rig = fixtures.rig
SAVED = Path(__file__).resolve().parents[1] / (
    "experiments/browser-pinned-capture-three-star/5e00bb5681c240c4b376000f041cb464/"
    "campaign/stars/001/owner/star/window/progress-013"
)
PNG_SHA = "6a9882170844288c3f546a96e2debb22a24b761f92dcb5811e7bed6e58bda749"


def deep_single_png():
    picture = Image.open(io.BytesIO(one_hint_png())).convert("RGB")
    # Replace the separate shallow fixture's tiny feature. One baseline-linked
    # two-column feature remains; there is no fabricated second event.
    for x in (241, 242):
        picture.putpixel((x, 21), (26, 26, 26))
    for x, upper, stem, bottom in (
        (243, (60, 92, 105), (50, 82, 96), (45, 74, 86)),
        (244, (46, 65, 74), (30, 49, 58), (25, 41, 48)),
    ):
        picture.putpixel((x, 21), upper)
        for y in range(22, 28):
            picture.putpixel((x, y), stem)
        picture.putpixel((x, 28), bottom)
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    return stream.getvalue()


def assert_deep_single_unresolved(raw, times, flux):
    pixels = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    mask = trace_mask(raw)
    baseline = next(row["center_y"] for row in flux if row["value"] == "100")
    below = mask[20:152, 30:261] & (np.arange(20, 152)[:, None] + 0.5 > baseline + 1.5)
    assert np.argwhere(below).tolist() == [[y - 20, x - 30] for y in range(22, 29) for x in (243, 244)]
    assert pixels[21, 244].tolist() == [46, 65, 74]
    assert candidate_columns(raw, flux) == [[243, 244]]
    for policy in POLICIES:
        with pytest.raises(BrowserSafetyStop, match="^shallow_probe_three_overview_hints_required$"):
            candidate_columns(raw, flux, hint_policy=policy)
    with pytest.raises(BrowserSafetyStop, match="^shallow_probe_exact_two_overview_hints_required$"):
        candidate_columns(raw, flux, hint_policy=EXACT_TWO_HINT_POLICY)

    negative = analyze_planet_window(raw, times, flux, requested_days=5000)
    assert negative["policy"]["version"] == "user_approved_5000_day_baseline_band_no_dip_v1"
    assert negative["status"] == "insufficient_visual_evidence"
    assert negative["reason"] == "unknown_plot_palette"
    assert negative["baseline_band_rejection"] == "unsupported_baseline_band"
    assert negative["endpoint_visible"] is True
    positive = analyze_window_dip(raw, times, flux)
    assert positive["status"] == "insufficient_visual_evidence"
    assert positive["reason"] == "insufficient_independent_dip_support"
    assert positive["antialias_candidates"] == positive["explained_antialias_pixels"] == 1
    for report in (negative, positive):
        assert report["planet_decision"] is None
        assert report["answer_writes"] == 0
        assert report["absence_proven"] is report["task_completed"] is False
        assert report["scientific_verified"] is report["learned_perception"] is False


def test_one_deeper_feature_stays_unresolved_under_explicit_baseline_band():
    assert_deep_single_unresolved(deep_single_png(), *axes())


def test_missing_second_stops_before_sensor_reservation_or_native_actions(rig):
    raw = deep_single_png()
    rig.state.image = raw
    (rig.source / "chart.png").write_bytes(raw)
    fixtures.save(rig.source / "report.json", rig.report(raw))
    before = {leaf: (rig.source / leaf).read_bytes() for leaf in ("chart.png", "report.json")}
    with pytest.raises(BrowserSafetyStop, match="^shallow_probe_exact_two_overview_hints_required$"):
        rig.make(allow_two_events=True, source_report_sha256=fixtures.sha(rig.source / "report.json"))
    assert not rig.state.calls and not rig.state.native
    assert not (rig.root / "sensor").exists()
    assert not (rig.root / "shallow-transit-owner-claims").exists()
    assert {leaf: (rig.source / leaf).read_bytes() for leaf in before} == before


def test_optional_saved_jarlarth_exact_hash_still_has_only_one_feature():
    if not SAVED.is_dir():
        pytest.skip("Optional immutable Jarlarth capture is not distributed")
    original = {path.name: path.read_bytes() for path in SAVED.iterdir() if path.is_file()}
    raw, report = original["chart.png"], json.loads(original["report.json"])
    assert hashlib.sha256(raw).hexdigest() == PNG_SHA
    assert report["star"] == "JARLARTH" and report["chart_sha256"] == PNG_SHA
    assert_deep_single_unresolved(raw, report["time_axis_labels"], report["flux_axis_labels"])
    assert {path.name: path.read_bytes() for path in SAVED.iterdir() if path.is_file()} == original
