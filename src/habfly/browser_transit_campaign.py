"""Explicitly bounded sequential sensor batches; no answers or training.

Only a completed, continuous, axis-checked batch can start the next batch.
Timeouts and other failures stop the campaign without recovery or retry. The
caller can inspect those journals separately. This is scripted UI sensing, not
learned chart perception and not a declaration that a star has no planet.
"""

import hashlib
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_transit_sampling import MAX_RECORDED_SAMPLES, record_transit_sampling


def record_transit_campaign(page, config, output, *, max_batches, prior_report=None, notify=lambda _: None):
    if type(max_batches) is not int or not 1 <= max_batches <= 10:
        raise ValueError("Choose an explicit 1 through 10 sensor batch limit")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    persist_json(
        directory / "campaign-scope.json",
        {
            "max_batches": max_batches,
            "max_actions_per_batch": 2048,
            "max_seconds_per_batch": 900,
            "max_cumulative_samples": MAX_RECORDED_SAMPLES,
            "policy": "scripted_reference_sensor",
            "answer_writes": 0,
            "optimizer_updates": 0,
            "retry_on_failure": False,
        },
    )
    receipts, current = [], prior_report
    try:
        for number in range(1, max_batches + 1):
            batch = directory / f"batch-{number:03d}"
            report = record_transit_sampling(
                page,
                config,
                batch,
                prior_report=current,
                notify=lambda progress, number=number: notify({"batch": number, **progress}),
            )
            current = batch / "report.json"
            receipts.append(
                {"batch": number, "report_sha256": hashlib.sha256(current.read_bytes()).hexdigest()}
            )
            persist_json(directory / f"completed-{number:03d}.json", receipts[-1])
            if report["period_evidence_verified"]:
                break
        result = {
            **report,
            "campaign_batches": receipts,
            "campaign_max_batches": max_batches,
            "campaign_status": "period_evidence_verified"
            if report["period_evidence_verified"]
            else "batch_limit_inconclusive",
            "answer_writes": 0,
            "optimizer_updates": 0,
            "automatic_retry": False,
        }
        persist_json(directory / "report.json", result)
        return result
    except BaseException as exc:
        persist_json(
            directory / "campaign-stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "transit_campaign_failed",
                "completed_batches": receipts,
                "automatic_retry": False,
                "answer_writes": 0,
                "task_completed": False,
            },
        )
        raise
