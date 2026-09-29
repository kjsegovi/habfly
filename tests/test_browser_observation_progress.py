"""Synthetic rendered pixels test readiness, not scientific absence labels."""

import io

import pytest
from PIL import Image, ImageDraw

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_progress


def crop(end=130, *, color=(80, 132, 154), stray=False, size=(280, 195)):
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(20, 151, 13):
        draw.line((30, y, 260, y), fill=(60, 60, 60))
    draw.line((30, 50, end, 50), fill=color)
    # A cyan zoom button or numeral must not masquerade as plotted progress.
    draw.rectangle((230, 0, 260, 17), fill=(80, 132, 154))
    draw.line((230, 167, 260, 167), fill=(80, 132, 154))
    if stray:
        draw.point((260, 80), fill=color)
    result = io.BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


def labels():
    return [{"value": str(day), "center_x": 30.5 + day * 0.023} for day in range(0, 10001, 1000)]


def test_partial_trace_does_not_inherit_axis_completion():
    report = trace_progress(crop(), labels(), requested_days=10000)
    assert report["status"] == "trace_partial_or_unresolved"
    assert report["approximate_rendered_day"] == 4326.1
    assert not report["endpoint_visible"] and not report["observation_completed"]
    assert report["planet_presence"] is None


def test_endpoint_only_claims_visible_trace_not_science_or_completion():
    report = trace_progress(crop(260), labels(), requested_days=10000)
    assert report["endpoint_visible"] and report["status"] == "trace_reaches_requested_end"
    assert not report["observation_completed"] and not report["task_completed"]
    assert not report["learned_perception"] and report["planet_presence"] is None


def test_requested_half_of_legacy_window_and_fractional_crop_height():
    report = trace_progress(crop(180, size=(280, 196)), labels(), requested_days=5000)
    assert report["endpoint_visible"] and report["requested_days"] == 5000
    assert report["rightmost_trace_pixel"] <= 146


def test_neutral_grid_and_tiny_isolated_pixels_are_not_progress():
    report = trace_progress(crop(30, stray=True), labels(), requested_days=10000)
    assert not report["endpoint_visible"]
    stray = trace_progress(crop(130, stray=True), labels(), requested_days=10000)
    assert not stray["endpoint_visible"]
    neutral = trace_progress(crop(260, color=(150, 150, 150)), labels(), requested_days=10000)
    assert neutral["trace_columns"] == 0 and neutral["rightmost_trace_pixel"] is None


@pytest.mark.parametrize("mode", ["short", "nonlinear", "duplicate", "nan", "offscreen", "size"])
def test_unknown_views_fail_without_inference(mode):
    ticks = labels()
    if mode == "short":
        ticks = ticks[:2]
    elif mode == "nonlinear":
        ticks[4]["center_x"] += 10
    elif mode == "duplicate":
        ticks[4]["value"] = ticks[3]["value"]
    elif mode == "nan":
        ticks[4]["center_x"] = float("nan")
    elif mode == "offscreen":
        ticks = [{**row, "center_x": row["center_x"] + 50} for row in ticks]
    with pytest.raises(BrowserSafetyStop):
        trace_progress(crop(size=(300, 200) if mode == "size" else (280, 195)), ticks, requested_days=10000)


@pytest.mark.parametrize("budget", [True, False, 0, -1, 121, float("nan"), float("inf"), "30"])
def test_capture_budget_rejected_before_artifacts_or_browser(tmp_path, budget):
    from habfly.browser_observation_progress import capture_observation_progress

    output = tmp_path / "capture"
    with pytest.raises(ValueError, match="capture budget"):
        capture_observation_progress(None, None, output, requested_days=5000, max_seconds=budget)
    assert not output.exists()


@pytest.mark.parametrize("budget", [30, 120])
def test_explicit_capture_budget_is_fixed_recorded_and_has_no_browser_actions(tmp_path, monkeypatch, budget):
    import habfly.browser_observation_progress as module

    calls = []

    class Session:
        star = "FIXTURE"

        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            self.handle = self.chart = self

        def _guard(self):
            calls.append("guard")

        def time_axis_labels(self):
            return labels()

        def flux_axis_labels(self):
            return []

        def evaluate(self, code):
            return True

        def screenshot(self, **options):
            from habfly.presentation_capture import EVIDENCE_SCREENSHOT_STYLE

            assert options == {"style": EVIDENCE_SCREENSHOT_STYLE}
            return crop(260)

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(module, "FluxChartSession", Session)
    report = module.capture_observation_progress(
        None, None, tmp_path / "capture", requested_days=5000, max_seconds=budget
    )
    assert calls == [{"max_actions": 1, "max_seconds": budget}, "guard", "guard", "closed"]
    assert report["capture_max_seconds"] == budget
    assert report["requested_days"] == 5000
    assert report["browser_actions"] == report["answer_writes"] == 0
