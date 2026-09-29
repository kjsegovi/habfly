"""Student-visible evidence only; no private course answers or network required."""

import hashlib
import json
from copy import deepcopy

import pytest
from typer.testing import CliRunner

from habfly.browser_assessment import load_assessment_capture, map_assessment_capture
from habfly.browser_stellar import SIMULATION_URL
from habfly.cli import app
from habfly.project_assessment import (
    AssessmentError,
    AssessmentLedger,
    AssessmentSnapshot,
    parse_assessment_text,
)


def visible(
    *, mode="data_quality", funding=50000, overall=0, stars=0, found=0, collected=2, ack=False, cost=100
):
    panel = {
        "data_quality": f"STARS {stars}% PLANETS 0.0% HABITABILITY 0.0% OVERALL {overall}%",
        "scavenger_hunt": "MAIN SEQUENCE WHITE DWARF RED GIANT SUPERGIANT GAS GIANT ICE GIANT TERRESTRIAL HABITABLE WORLD",
        "automation": "ASSESSMENT FIELD STARS SELECTED",
    }[mode]
    return (
        f"FUNDING ${funding}\nDATA QUALITY {overall}%\nSCAVENGER HUNT {found}/8\n"
        f"EDIT DATA ASSESSMENT {panel} ASSESS COST ${cost}\n"
        "AUTOMATION SCAVENGER HUNT DATA QUALITY OBSERVATIONS ANALYZED DATA STAR\n"
        f"VIEWING 1-{collected} OF {collected} TOTAL COLLECTED {collected}\nSave\n"
        + (
            {"data_quality": "DATA QUALITY UPDATED OK", "scavenger_hunt": "SCAVENGER HUNT UPDATED OK"}[mode]
            if ack
            else ""
        )
    )


def snapshot(**kwargs):
    return parse_assessment_text(visible(**kwargs))


def test_observed_quality_feedback_is_not_project_completion():
    ledger = AssessmentLedger(budget=200, max_attempts=2)
    before = snapshot()
    proposal = ledger.reserve("data_quality", before, data_revision=1)
    after = snapshot(funding=49900, stars=50, overall=16.7, ack=True)
    ledger.confirm(proposal.sequence, after, data_revision=1)
    report = ledger.report(data_revision=1)
    assert report["quality"] == {"stars": 50, "planets": 0, "habitability": 0, "overall": 16.7}
    assert report["data_quality_assessed"] and not report["scavenger_hunt_assessed"]
    for key in (
        "task_completed",
        "submitted",
        "save_verified",
        "score_transfer_verified",
        "browser_acceptance_passed",
    ):
        assert report[key] is False
    # Stale grading must not be reported for edited data.
    assert not ledger.report(data_revision=2)["data_quality_assessed"]


def test_scavenger_is_a_separate_receipt_and_no_category_is_invented():
    ledger = AssessmentLedger(budget=200, max_attempts=2)
    first = ledger.reserve("data_quality", snapshot(), data_revision=1)
    ledger.confirm(first.sequence, snapshot(funding=49900, stars=50, overall=16.7, ack=True), data_revision=1)
    before = snapshot(mode="scavenger_hunt", funding=49900, overall=16.7)
    second = ledger.reserve("scavenger_hunt", before, data_revision=1)
    ledger.confirm(
        second.sequence,
        snapshot(mode="scavenger_hunt", funding=49800, overall=16.7, found=1, ack=True),
        data_revision=1,
    )
    restored = AssessmentLedger.model_validate_json(ledger.model_dump_json())
    assert restored.report(data_revision=1)["scavenger_found"] == 1
    assert restored.reserved == 200
    assert "red_giant" not in restored.model_dump_json()
    assert not restored.report(data_revision=1)["task_completed"]


def test_uncertain_click_stays_reserved_across_serialization():
    ledger = AssessmentLedger(budget=300, max_attempts=3)
    ledger.reserve("data_quality", snapshot(), data_revision=1)
    restored = AssessmentLedger.model_validate_json(ledger.model_dump_json())
    with pytest.raises(AssessmentError, match="outcome_uncertain"):
        restored.reserve("scavenger_hunt", snapshot(mode="scavenger_hunt"), data_revision=2)
    assert restored.report(data_revision=1)["outcome_uncertain"]
    assert restored.reserved == 100


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"funding": 50000, "ack": True}, "funding_mismatch"),
        ({"funding": 49800, "ack": True}, "funding_mismatch"),
        ({"funding": 49900, "ack": False}, "acknowledgement_missing"),
        ({"funding": 49900, "ack": True, "collected": 3}, "project_changed"),
        ({"funding": 49900, "ack": True, "cost": 200}, "project_changed"),
        ({"funding": 49900, "ack": True, "mode": "scavenger_hunt"}, "acknowledgement_missing"),
    ],
)
def test_receipt_requires_matching_funding_mode_and_collection(change, reason):
    ledger = AssessmentLedger(budget=100, max_attempts=1)
    ledger.reserve("data_quality", snapshot(), data_revision=1)
    with pytest.raises(AssessmentError, match=reason):
        ledger.confirm(0, snapshot(**change), data_revision=1)
    assert ledger.pending is not None


def test_no_repeat_assessment_of_unchanged_data_or_mutable_alias():
    ledger = AssessmentLedger(budget=300, max_attempts=3)
    before = snapshot()
    attempt = ledger.reserve("data_quality", before, data_revision=1)
    before.funding = 1
    attempt.before.funding = 2
    assert ledger.pending.before.funding == 50000
    ledger.confirm(0, snapshot(funding=49900, ack=True), data_revision=1)
    with pytest.raises(AssessmentError, match="already_assessed_revision"):
        ledger.reserve("data_quality", snapshot(funding=49900), data_revision=1)
    with pytest.raises(AssessmentError, match="stale_project_revision"):
        ledger.reserve("scavenger_hunt", snapshot(mode="scavenger_hunt"), data_revision=0)
    ledger.reserve("data_quality", snapshot(funding=49900), data_revision=2)


@pytest.mark.parametrize(
    "ledger,before,kind,reason",
    [
        (AssessmentLedger(), snapshot(), "data_quality", "budget_exhausted"),
        (AssessmentLedger(budget=99, max_attempts=1), snapshot(), "data_quality", "budget_exhausted"),
        (AssessmentLedger(budget=100, max_attempts=0), snapshot(), "data_quality", "budget_exhausted"),
        (AssessmentLedger(budget=100, max_attempts=1), snapshot(funding=99), "data_quality", "insufficient"),
        (AssessmentLedger(budget=200, max_attempts=1), snapshot(cost=200), "data_quality", "cost_changed"),
        (
            AssessmentLedger(budget=100, max_attempts=1),
            snapshot(ack=True),
            "data_quality",
            "stale_assessment_modal",
        ),
        (
            AssessmentLedger(budget=100, max_attempts=1),
            snapshot(mode="automation"),
            "automation",
            "not_authorized",
        ),
    ],
)
def test_fail_closed_spending_defaults(ledger, before, kind, reason):
    with pytest.raises(AssessmentError, match=reason):
        ledger.reserve(kind, before, data_revision=1)
    assert not ledger.attempts


@pytest.mark.parametrize(
    "replacement",
    [
        "",
        "unknown",
        "NaN",
        "Infinity",
        "-1",
        "101",
    ],
)
def test_blank_invalid_scores_are_not_zero(replacement):
    with pytest.raises(ValueError):
        parse_assessment_text(visible().replace("DATA QUALITY 0%", f"DATA QUALITY {replacement}%"))


def test_parser_rejects_duplicate_and_inconsistent_evidence():
    for text in (
        visible() + " FUNDING $50000",
        visible() + " TOTAL COLLECTED 2",
        visible() + " DATA QUALITY 0%",
        visible() + " COST $100",
        visible() + " ASSESSMENT FIELD",
        visible().replace("OVERALL 0%", "OVERALL 1%"),
        visible() + " SCAVENGER HUNT UPDATED",
        visible(ack=True) + " SCAVENGER HUNT UPDATED",
    ):
        with pytest.raises(ValueError):
            parse_assessment_text(text)
    assert snapshot().quality.stars == 0


def test_invalid_history_and_revision_are_rejected():
    ledger = AssessmentLedger(budget=200, max_attempts=2)
    with pytest.raises(AssessmentError):
        ledger.reserve("data_quality", snapshot(), data_revision=True)
    ledger.reserve("data_quality", snapshot(), data_revision=2)
    with pytest.raises(AssessmentError, match="project_changed"):
        ledger.confirm(0, snapshot(funding=49900, ack=True), data_revision=3)
    with pytest.raises(AssessmentError, match="unknown_pending"):
        ledger.confirm(1, snapshot(funding=49900, ack=True), data_revision=2)
    dumped = ledger.model_dump()
    dumped["attempts"].append(deepcopy(dumped["attempts"][0]))
    with pytest.raises(ValueError):
        AssessmentLedger.model_validate(dumped)
    for mode, quality in (("data_quality", None), ("scavenger_hunt", snapshot().quality)):
        item = snapshot().model_dump()
        item.update(mode=mode, quality=quality)
        with pytest.raises(ValueError):
            AssessmentSnapshot.model_validate(item)


def test_offline_capture_cli_checks_hash_and_ignores_outer_scores(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("network access"))
    report = {
        "mode": "read_only_browser_preflight",
        "actions_executed": 0,
        "outer_text": "Score: 100.00",  # Not assessment evidence.
        "frames": [{"url": SIMULATION_URL, "text": visible(funding=49900, stars=50, overall=16.7, ack=True)}],
    }
    raw = json.dumps(report).encode()
    (tmp_path / "observation.json").write_bytes(raw)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"observation_sha256": hashlib.sha256(raw).hexdigest()})
    )
    mapped = load_assessment_capture(tmp_path)
    assert mapped["assessment"]["quality"]["stars"] == 50
    assert not mapped["assessment_executed_by_habfly"] and not mapped["task_completed"]
    result = CliRunner().invoke(app, ["browser", "map-assessment", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == mapped
    (tmp_path / "observation.json").write_bytes(raw + b" ")
    result = CliRunner().invoke(app, ["browser", "map-assessment", str(tmp_path)])
    assert result.exit_code == 1 and "capture_hash_mismatch" in result.output
    report["frames"] *= 2
    with pytest.raises(AssessmentError, match="ambiguous_simulation_frame"):
        map_assessment_capture(report, capture_sha256="0" * 64)


@pytest.mark.parametrize(
    "report",
    [
        None,
        [],
        {},
        {"mode": "read_only_browser_preflight", "actions_executed": False},
        {"mode": "read_only_browser_preflight", "actions_executed": 0, "frames": [None]},
    ],
)
def test_malformed_capture_is_a_structured_stop(report):
    with pytest.raises(AssessmentError):
        map_assessment_capture(report, capture_sha256="0" * 64)
