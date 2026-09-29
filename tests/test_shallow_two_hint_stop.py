"""Offline Bisperon regression: two hints never become three or a No answer.

The required tests use deterministic synthetic pixels reproducing the saved
plot's two localized rows, not a physical transit simulation. Browser capture
and session seams are explicitly injected; no native browser or model runs.
The optional saved-source test only reads and hashes the immutable public crop.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import test_browser_shallow_transit_steps as steps_fixtures
from PIL import Image

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_mask
from habfly.browser_shallow_transit_probe import (
    FIRST_THREE_HINT_POLICY,
    FIRST_THREE_LOW_INTENSITY_HINT_POLICY,
    FIRST_THREE_QUANTIZED_HINT_POLICY,
    FIRST_THREE_TWO_ROW_HINT_POLICY,
    _low_intensity_hints,
    candidate_columns,
)
from habfly.planet_window_baseline_edge import analyze_planet_window as baseline_no
from habfly.planet_window_dip import analyze_window_dip
from habfly.planet_window_policy import analyze_planet_window as frozen_no

rig = steps_fixtures.rig
POLICIES = (
    FIRST_THREE_HINT_POLICY,
    FIRST_THREE_LOW_INTENSITY_HINT_POLICY,
    FIRST_THREE_QUANTIZED_HINT_POLICY,
    FIRST_THREE_TWO_ROW_HINT_POLICY,
)
FEATURES = {
    102: (57, 88, 102),
    103: (41, 57, 64),
    194: (50, 75, 86),
    195: (48, 70, 80),
}
SAVED = Path(__file__).resolve().parents[1] / (
    "experiments/browser-project-supplied-three-star/736e75273b6c424d970dc3c17a95dbc2/"
    "campaign/stars/001/owner/star/shallow/initial/progress"
)
SAVED_PNG_SHA = "2e9c16bf4488e3cb587ba33b2f3ab3d6349c061e960e3976f48f4c357ea74cc5"
SAVED_REPORT_SHA = "c20feb595199555c81484f499d4fbc9096ec963eb3693d57d3dd85bc394eaec8"


def representative_png():
    # This deliberately reproduces only decision-relevant plot geometry and
    # colors. It does not pretend to reproduce the course's axis glyphs/UI.
    picture = Image.new("RGB", (280, 196), "black")
    for x in range(30, 261):
        picture.putpixel((x, 21), (26, 26, 26))
    for x in range(30, 261, 23):
        for y in range(10, 152):
            picture.putpixel((x, y), (51, 51, 51))
    for x in range(30, 260):
        picture.putpixel((x, 20), (80, 131, 153))
    for x, rgb in FEATURES.items():
        picture.putpixel((x, 21), rgb)
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    return stream.getvalue()


def axes():
    return (
        [{"value": str(day), "center_x": 30.5 + day * 0.046} for day in range(0, 5001, 500)],
        [
            {"value": str(value), "center_y": 20.45000457763672 + (100 - value) * 1.3}
            for value in range(10, 101, 10)
        ],
    )


def assert_two_hints_without_answer(raw, times, flux):
    original = raw
    colors = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=float)
    mask = trace_mask(raw)
    assert {x: tuple(map(int, colors[21, x])) for x in FEATURES} == FEATURES
    assert not np.ptp(colors[22:152, 30:261], axis=2).any()
    assert mask[20, 30:260].all()
    unknown = (np.ptp(colors[20:152, 30:261], axis=2) > 8) & ~mask[20:152, 30:261]
    assert np.argwhere(unknown).tolist() == [[1, 73]]  # Original crop coordinate (103,21).
    baseline = next(row["center_y"] for row in flux if row["value"] == "100")
    low_candidates, _ = _low_intensity_hints(colors, mask, baseline)
    assert not low_candidates.any()  # No v2/v3/v4 context relaxation can create a third group.
    assert candidate_columns(raw, flux) == [[102, 103], [194, 195]]
    for policy in POLICIES:
        with pytest.raises(BrowserSafetyStop, match="^shallow_probe_three_overview_hints_required$"):
            candidate_columns(raw, flux, hint_policy=policy)
    for analyzer in (frozen_no, baseline_no):
        result = analyzer(raw, times, flux, requested_days=5000)
        assert result["status"] == "insufficient_visual_evidence"
        assert result["reason"] == "unknown_plot_palette"
        assert result["endpoint_visible"] is True
        assert result["planet_decision"] is None
        assert result["answer_writes"] == 0
        assert result["absence_proven"] is result["task_completed"] is False
    positive = analyze_window_dip(raw, times, flux)
    assert positive["reason"] == "insufficient_independent_dip_support"
    assert positive["supported_dip_components"] == 0
    assert positive["planet_decision"] is None
    assert raw == original


def test_deterministic_two_hint_pixels_never_invent_third_hint_or_no():
    assert_two_hints_without_answer(representative_png(), *axes())


def test_missing_three_gate_stops_owner_before_any_injected_native_action(rig):
    raw = representative_png()
    rig.state.image = raw
    (rig.source / "chart.png").write_bytes(raw)
    steps_fixtures.save(rig.source / "report.json", rig.report(raw))
    source_bytes = {name: (rig.source / name).read_bytes() for name in ("chart.png", "report.json")}
    probe_calls = []

    def forbidden_probe(*args, **kwargs):
        probe_calls.append((args, kwargs))
        raise AssertionError("A missing-third-hint stop must precede probe construction")

    owner = rig.make(
        source_report_sha256=steps_fixtures.sha(rig.source / "report.json"),
        _probe_factory=forbidden_probe,
    )
    try:
        assert not rig.state.calls and not rig.state.native
        owner.advance()
        assert owner.phase == "initial_capture" and not owner.finished
        owner.advance()
        assert owner.finished and owner.phase == "stopped"
        state = owner.state()
        assert state["failure_reason"] == "shallow_probe_three_overview_hints_required"
        assert state["advances"] == 2
        assert state["native_action_attempts"] == state["native_actions_confirmed"] == 0
        assert state["answer_writes"] == 0 and state["planet_decision"] is None
        assert state["answer_authorized"] is state["task_completed"] is state["project_completed"] is False
        assert state["completed_probes"] == [] and state["measurements"] is None
        assert not probe_calls and not rig.state.native
        assert [name for name, _ in rig.state.calls] == ["session", "capture"]
        assert all(kind not in {"action_proposed", "action_result"} for kind, _ in rig.state.events)
        assert not list(owner.output.glob("probe-*"))
        assert not (owner.output / "report.json").exists()
        stopped = (owner.output / "stopped.json").read_bytes()
        assert json.loads(stopped)["failure_reason"] == state["failure_reason"]
        for _ in range(3):
            owner.advance()
        assert (owner.output / "stopped.json").read_bytes() == stopped
        assert not probe_calls and not rig.state.native
        assert {name: (rig.source / name).read_bytes() for name in source_bytes} == source_bytes
    finally:
        owner.close()


def test_optional_saved_bisperon_exact_hash_and_same_two_hint_stop():
    if not SAVED.is_dir():
        pytest.skip("Optional immutable public Bisperon capture is not distributed")
    raw, report_raw = (SAVED / "chart.png").read_bytes(), (SAVED / "report.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SAVED_PNG_SHA
    assert hashlib.sha256(report_raw).hexdigest() == SAVED_REPORT_SHA
    report = json.loads(report_raw)
    assert report["star"] == "BISPERON" and report["endpoint_visible"] is True
    assert report["chart_sha256"] == SAVED_PNG_SHA
    assert_two_hints_without_answer(raw, report["time_axis_labels"], report["flux_axis_labels"])
    assert (SAVED / "chart.png").read_bytes() == raw
    assert (SAVED / "report.json").read_bytes() == report_raw
