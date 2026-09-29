"""Offline, integrity-checked replay of the bounded public chart capture."""

import hashlib
import json
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_observation_progress import trace_progress
from .browser_planet_window_choice import (
    analyze_recorded_planet_window,
    recorded_capture_policy,
    recorded_policy_manifest,
)
from .planet_window_dip import analyze_window_dip
from .planet_window_measurements import load_planet_window_measurements


def load_planet_window_capture(directory):
    """Replay all policy outcomes; never open a browser or authorize a write."""
    directory = Path(directory)
    try:
        if any((directory / name).exists() for name in ("stopped.json", "invalidated.json")):
            raise BrowserSafetyStop("planet_window_capture_not_valid")
        raw = (directory / "report.json").read_bytes()
        capture = json.loads(raw)
        png = (directory / "chart.png").read_bytes()
        if (
            not isinstance(capture, dict)
            or capture.get("chart_sha256") != hashlib.sha256(png).hexdigest()
            or type(capture.get("requested_days")) is not int
            or capture["requested_days"] != 5000
            or type(capture.get("browser_actions")) is not int
            or capture["browser_actions"] != 0
            or type(capture.get("answer_writes")) is not int
            or capture["answer_writes"] != 0
            or not isinstance(capture.get("star"), str)
            or not capture["star"].strip()
        ):
            raise BrowserSafetyStop("invalid_planet_window_capture")
        readiness = trace_progress(png, capture["time_axis_labels"], requested_days=5000)
        if any(capture.get(key) != value for key, value in readiness.items()):
            raise BrowserSafetyStop("planet_window_readiness_mismatch")
        policy = recorded_capture_policy(directory)
        saved_analysis = directory / "analysis.json"
        if saved_analysis.exists():
            saved = json.loads(saved_analysis.read_bytes())
            recorded = saved.get("policy", saved.get("negative_policy"))
            if recorded_policy_manifest(recorded) != policy:
                raise BrowserSafetyStop("planet_window_replay_policy_mismatch")
        analysis = analyze_recorded_planet_window(
            png, capture["time_axis_labels"], capture["flux_axis_labels"], policy=policy
        )
        if analysis["status"] == "insufficient_visual_evidence":
            positive = analyze_window_dip(png, capture["time_axis_labels"], capture["flux_axis_labels"])
            if positive["status"] == "dip_observed":
                analysis = {
                    **positive,
                    "negative_policy_status": analysis["status"],
                    "negative_policy_reason": analysis["reason"],
                    "negative_policy": analysis["policy"],
                }
        # The public CLI may show reference estimates, but neither positive
        # detection nor replay grants permission to select or enter answers.
        measurements = None
        if analysis["status"] == "dip_observed":
            measurements = load_planet_window_measurements(
                directory / "report.json", expected_report_sha256=hashlib.sha256(raw).hexdigest()
            )
        return {
            **analysis,
            "approximate_measurements": measurements,
            "star": capture["star"],
            "source_report_sha256": hashlib.sha256(raw).hexdigest(),
            "mode": "offline_window_replay",
            "browser_opened": False,
            "write_authorized": False,
        }
    except (OSError, ValueError, KeyError, TypeError):
        raise BrowserSafetyStop("unreadable_planet_window_capture") from None
