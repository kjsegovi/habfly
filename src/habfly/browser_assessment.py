"""Offline mapping for visible assessment captures; deliberately no UI writes."""

import hashlib
import json
from pathlib import Path

from .browser_stellar import SIMULATION_URL
from .project_assessment import AssessmentError, parse_assessment_text


def map_assessment_capture(report, *, capture_sha256):
    if (
        not isinstance(report, dict)
        or report.get("mode") != "read_only_browser_preflight"
        or type(report.get("actions_executed")) is not int
        or report["actions_executed"] != 0
    ):
        raise AssessmentError("unsupported_capture")
    if not isinstance(report.get("frames"), list) or any(not isinstance(f, dict) for f in report["frames"]):
        raise AssessmentError("invalid_capture_frames")
    frames = [frame for frame in report.get("frames", []) if frame.get("url") == SIMULATION_URL]
    if len(frames) != 1:
        raise AssessmentError("ambiguous_simulation_frame")
    snapshot = parse_assessment_text(frames[0].get("text"))
    return {
        "schema_version": 1,
        "mode": "offline_assessment_mapping",
        "capture_sha256": capture_sha256,
        "assessment": snapshot.model_dump(mode="json"),
        "actions_executed": 0,
        "assessment_executed_by_habfly": False,
        "save_verified": False,
        "score_transfer_verified": False,
        "submitted": False,
        "task_completed": False,
        "browser_acceptance_passed": False,
    }


def load_assessment_capture(directory: Path):
    raw = (directory / "observation.json").read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    manifest = json.loads((directory / "manifest.json").read_text())
    if checksum != manifest.get("observation_sha256"):
        raise AssessmentError("capture_hash_mismatch")
    return map_assessment_capture(json.loads(raw), capture_sha256=checksum)
