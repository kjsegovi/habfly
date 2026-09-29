"""The bounded-window replay command has no browser or network dependency."""

import hashlib
import json
import socket

import pytest
from test_planet_window_policy import crop, labels
from typer.testing import CliRunner

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_progress
from habfly.browser_window_replay import load_planet_window_capture
from habfly.cli import app


def capture(tmp_path, *, end=260, dip=None):
    png = crop(end=end, dip=dip)
    time, flux = labels(5000)
    report = {
        **trace_progress(png, time, requested_days=5000),
        "star": "FIXTURE",
        "time_axis_labels": time,
        "flux_axis_labels": flux,
        "chart_sha256": hashlib.sha256(png).hexdigest(),
        "browser_actions": 0,
        "answer_writes": 0,
    }
    (tmp_path / "chart.png").write_bytes(png)
    (tmp_path / "report.json").write_text(json.dumps(report))


@pytest.mark.parametrize(
    "end,dip,status",
    [(260, None, "assume_no_planet"), (130, None, "still_collecting"), (260, (80, 35), "dip_observed")],
)
def test_offline_cli_replays_without_action_or_network(tmp_path, monkeypatch, end, dip, status):
    capture(tmp_path, end=end, dip=dip)

    def no_network(*args, **kwargs):
        raise AssertionError("Replay must not contact a network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    result = CliRunner().invoke(app, ["browser", "analyze-planet-window", str(tmp_path)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["status"] == status
    assert report["mode"] == "offline_window_replay"
    assert not report["browser_opened"] and not report["write_authorized"]
    assert not report["task_completed"] and not report["training_label"]
    if status != "dip_observed":
        assert report["approximate_measurements"] is None
    else:
        # One visible dip is not enough to infer a recurrence period.
        assert report["approximate_measurements"]["status"] == "measurement_error"
        assert report["approximate_measurements"]["period_days"] is None


def test_multiple_dip_replay_exposes_uncertainty_without_authorizing_writes(tmp_path, monkeypatch):
    from test_planet_window_measurements import axes, png

    image = png()
    times, flux = axes()
    report = {
        **trace_progress(image, times, requested_days=5000),
        "star": "FIXTURE",
        "time_axis_labels": times,
        "flux_axis_labels": flux,
        "chart_sha256": hashlib.sha256(image).hexdigest(),
        "browser_actions": 0,
        "answer_writes": 0,
    }
    (tmp_path / "chart.png").write_bytes(image)
    (tmp_path / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("Offline replay contacted network"))
    replay = load_planet_window_capture(tmp_path)
    estimate = replay["approximate_measurements"]
    assert estimate["status"] == "approximate_reference_measurements"
    assert estimate["period_days"]["value"] == pytest.approx(1000)
    assert estimate["period_days"]["lower"] < 1000 < estimate["period_days"]["upper"]
    assert estimate["brightness_drop_percent"]["value"] == pytest.approx(10)
    assert not replay["write_authorized"]
    assert not estimate["training_label"] and not estimate["scientific_verified"]


@pytest.mark.parametrize("change", ["crop", "days", "readiness", "stopped", "invalidated", "bool", "missing"])
def test_mutated_or_failed_capture_rejected(tmp_path, change):
    capture(tmp_path)
    path = tmp_path / "report.json"
    report = json.loads(path.read_text())
    if change == "crop":
        (tmp_path / "chart.png").write_bytes(crop(end=80))
    elif change == "days":
        report["requested_days"] = 10000
    elif change == "readiness":
        report["endpoint_visible"] = False
    elif change in {"stopped", "invalidated"}:
        (tmp_path / f"{change}.json").write_text("{}")
    elif change == "bool":
        report["answer_writes"] = False
    else:
        del report["time_axis_labels"]
    path.write_text(json.dumps(report))
    with pytest.raises(BrowserSafetyStop):
        load_planet_window_capture(tmp_path)
