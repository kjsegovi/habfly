"""Bounded orchestration fixtures; browser actions belong to tested transport."""

import json

import pytest

from habfly.browser import BrowserSafetyStop
from habfly.browser_transit_campaign import record_transit_campaign


@pytest.mark.parametrize("complete_on", [2, None])
def test_campaign_only_continues_completed_batches_and_never_claims_task_success(
    tmp_path, monkeypatch, complete_on
):
    calls, progress = [], []

    def record(page, config, output, *, prior_report, notify):
        number = len(calls) + 1
        assert prior_report == (None if number == 1 else calls[-1] / "report.json")
        calls.append(output)
        output.mkdir()
        result = {
            "period_evidence_verified": number == complete_on,
            "has_planet_answer": None,
            "task_completed": False,
            "learned_chart_perception": False,
        }
        (output / "report.json").write_text(json.dumps(result))
        notify({"window": 1})
        return result

    monkeypatch.setattr("habfly.browser_transit_campaign.record_transit_sampling", record)
    result = record_transit_campaign(None, None, tmp_path / "campaign", max_batches=3, notify=progress.append)
    expected = complete_on or 3
    assert len(calls) == expected and len(result["campaign_batches"]) == expected
    assert [p["batch"] for p in progress] == list(range(1, expected + 1))
    assert result["campaign_status"] == (
        "period_evidence_verified" if complete_on else "batch_limit_inconclusive"
    )
    assert not result["task_completed"] and result["has_planet_answer"] is None
    assert result["answer_writes"] == result["optimizer_updates"] == 0
    with pytest.raises(FileExistsError):
        record_transit_campaign(None, None, tmp_path / "campaign", max_batches=3)
    assert len(calls) == expected


def test_campaign_failure_does_not_retry_or_start_later_batches(tmp_path, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise BrowserSafetyStop("chart_time_limit")

    monkeypatch.setattr("habfly.browser_transit_campaign.record_transit_sampling", fail)
    with pytest.raises(BrowserSafetyStop, match="chart_time_limit"):
        record_transit_campaign(None, None, tmp_path / "campaign", max_batches=4)
    assert calls == [1]
    failure = json.loads((tmp_path / "campaign/campaign-stopped.json").read_text())
    assert not failure["automatic_retry"] and not failure["task_completed"]
    assert not (tmp_path / "campaign/report.json").exists()


@pytest.mark.parametrize("budget", [0, 11, True, 2.5, None])
def test_invalid_campaign_budget_never_opens_a_browser_or_creates_artifacts(tmp_path, budget):
    with pytest.raises(ValueError, match="explicit"):
        record_transit_campaign(None, None, tmp_path / "none", max_batches=budget)
    assert not (tmp_path / "none").exists()
