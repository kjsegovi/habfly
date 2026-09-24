"""Bounded, sequential three-field browser reliability runs, with no training.

Each run owns a fresh browser context, never a reset of an existing attempt.
Credentials live only in memory and are consumed before Playwright starts.
This is not fresh-account, scoring, persistence, or full-project acceptance.
"""

import hashlib
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .runtime import RunOptions, Runtime, read_trace

MAX_RUNS = 10
RUN_SECONDS = 1050  # Includes model loading, 90s setup and 900s diagnostic limits.
FIELDS = {"distance", "luminosity", "temperature"}
LOGIN_KEYS = ("HABFLY_LOGIN_EMAIL", "HABFLY_LOGIN_PASSWORD")


class ProgressOutput:
    """Protocol sink: never echo raw observations, credentials, or driver errors."""

    def __init__(self, notify):
        self.notify = notify
        self.stage = None

    def write(self, line):
        event = json.loads(line)
        if event["event"] == "state":
            payload = event["payload"]
            stage = (
                payload.get("setup_stage")
                if payload.get("browser_phase") == "setting_up"
                else payload.get("browser_phase")
            )
            if stage != self.stage and stage:
                self.stage = stage
                self.notify(f"Stage: {stage}")
        return len(line)

    def flush(self):
        pass


def _measurements(runtime):
    session = runtime.env.session if runtime.env else None
    if not session or not hasattr(session, "mapping"):
        return None, None
    values = session.mapping["observation"]["values"]
    case = {
        key: {name: value[name] for name in ("kind", "value", "unit")}
        for key, value in values["measurements"].items()
    }
    return values["star_name"], hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()


def _result(runtime, output, *, elapsed, failure, star, case_hash):
    trace = output / "runs" / f"{runtime.run_id}.jsonl"
    manifest_path = output / "runs" / str(runtime.run_id) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    events = read_trace(trace) if trace.exists() else []
    journal_path = manifest_path.parent / "events.jsonl"
    journal = read_trace(journal_path) if journal_path.exists() else []
    receipts = [
        e.payload for e in journal if e.event == "action_result" and e.payload.get("numeric_copy_verified")
    ]
    errors = sum(e.event == "error" for e in events)
    decisions = sum(e.event == "action_proposed" for e in events)
    summaries = [e.payload for e in events if e.event == "episode_summary"]
    integrity = bool(manifest) and hashlib.sha256(journal_path.read_bytes()).hexdigest() == manifest.get(
        "events_sha256"
    )
    passed = (
        failure is None
        and integrity
        and errors == 0
        and len(summaries) == 1
        and summaries[0].get("numeric_transport_passed") is True
        and summaries[0].get("outcome") == "numeric_transport_verified"
        and manifest.get("numeric_transport_passed") is True
        and manifest.get("outcome") == "numeric_transport_verified"
        and manifest.get("write_attempts") == 3
        and set(manifest.get("verified_fields", [])) == FIELDS
        and 0 < decisions <= 64
        and len(receipts) == 3
        and all(r.get("copy_authorization") == "autonomous_opt_in" for r in receipts)
        and manifest.get("provenance", {}).get("browser_execution") == "autonomous"
        and manifest.get("provenance", {}).get("optimizer_updates") == 0
        and all(
            re.fullmatch(r"[0-9a-f]{64}", manifest.get("provenance", {}).get(key, ""))
            for key in ("checkpoint_sha256", "graph_hash", "knowledge_pack_hash")
        )
        and manifest.get("task_completed") is False
        and manifest.get("browser_acceptance_passed") is False
        and manifest.get("allow_submission") is False
    )
    reason = failure or (None if passed else manifest.get("outcome", "missing_run_manifest"))
    if not passed and reason == "numeric_transport_verified":
        reason = "reliability_evidence_gate_failed"
    return {
        "run_id": runtime.run_id,
        "numeric_transport_passed": bool(passed),
        "failure_reason": reason,
        "star_name": star,
        "measurement_case_sha256": case_hash,
        "learned_decisions": decisions,
        "write_attempts": manifest.get("write_attempts", 0),
        "verified_fields": manifest.get("verified_fields", []),
        "runtime_errors": errors,
        "autonomous_copy_receipts": len(receipts),
        "journal_hash_verified": integrity,
        "elapsed_seconds": round(elapsed, 3),
        "trace": str(trace.relative_to(output)),
        "manifest": str(manifest_path.relative_to(output)),
        "provenance": manifest.get("provenance", {}),
    }


def run_reliability(options: RunOptions, output: Path, *, runs: int, credentials, notify=print):
    """No retries, no budget expansion; the first failed/aborted run ends the batch."""
    if not 1 <= runs <= MAX_RUNS:
        raise ValueError(f"Choose 1–{MAX_RUNS} reliability runs")
    if (
        options.task != "browser_numeric"
        or options.browser_execution != "autonomous"
        or options.browser_setup != "automatic"
        or options.paused
        or options.stars != 1
    ):
        raise ValueError("Reliability runs require the running one-star autonomous profile")
    if len(credentials) != 2 or not all(credentials):
        raise ValueError("Both local login credentials are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    options = options.model_copy(update={"artifact_dir": output / "runs"})
    report = {
        "schema_version": 1,
        "scope": "autonomous_three_field_browser_reliability",
        "requested_runs": runs,
        "attempted_runs": 0,
        "passed_runs": 0,
        "unique_stars": 0,
        "unique_measurement_cases": 0,
        "all_requested_runs_passed": False,
        "distinct_star_target_met": False,
        "task_completed": False,
        "browser_acceptance_passed": False,
        "fresh_accounts_provisioned": False,
        "allow_submission": False,
        "stop_on_first_failure": True,
        "run_time_limit_seconds": RUN_SECONDS,
        "calibration_scope": "browser_transfer_not_calibrated",
        "memory_scope": "python_runner_peak_rss_excludes_browser",
        "runs": [],
    }

    def save():
        report["attempted_runs"] = len(report["runs"])
        report["passed_runs"] = sum(r["numeric_transport_passed"] for r in report["runs"])
        report["unique_stars"] = len({r["star_name"] for r in report["runs"] if r["star_name"]})
        report["unique_measurement_cases"] = len(
            {r["measurement_case_sha256"] for r in report["runs"] if r["measurement_case_sha256"]}
        )
        report["all_requested_runs_passed"] = report["passed_runs"] == runs
        report["distinct_star_target_met"] = (
            report["all_requested_runs_passed"] and report["unique_stars"] == runs
        )
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_process_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
        # The output directory is exclusive to this batch; prior experiments are untouched.
        pending = output / "report.tmp"
        pending.write_text(json.dumps(report, indent=2) + "\n")
        pending.replace(output / "report.json")

    save()
    try:
        for index in range(runs):
            notify(f"Run {index + 1}/{runs}: fresh browser, at most three writes; Ctrl-C ends the batch.")
            runtime = Runtime(ProgressOutput(notify))
            if report["runs"]:
                runtime.expected_browser_identity = {
                    key: report["runs"][0]["provenance"][key]
                    for key in ("checkpoint_sha256", "graph_hash", "knowledge_pack_hash")
                }
            started, failure, star, case_hash = time.monotonic(), None, None, None
            try:
                for key, value in zip(LOGIN_KEYS, credentials):
                    os.environ[key] = value
                runtime.command({"command": "start", "payload": options.model_dump(mode="json")})
                while runtime.status == "running":
                    if time.monotonic() - started >= RUN_SECONDS:
                        failure = "reliability_time_limit"
                        runtime.env.outcome = failure
                        runtime.command({"command": "abort"})
                        break
                    runtime.advance_if_due()
                    if runtime.env and runtime.env.session:
                        star, case_hash = _measurements(runtime)
                    time.sleep(0.02)
            except KeyboardInterrupt:
                failure = "operator_aborted"
                runtime.command({"command": "abort"})
            except Exception as exc:  # noqa: BLE001 - never serialize driver/config/credential exception text
                failure = (
                    "batch_provenance_changed"
                    if isinstance(exc, BrowserSafetyStop) and str(exc) == "batch_provenance_changed"
                    else "runtime_exception"
                )
            finally:
                for key in LOGIN_KEYS:
                    os.environ.pop(key, None)
                try:
                    runtime.close()
                except Exception:  # noqa: BLE001 - do not start another browser after failed cleanup
                    failure = "browser_cleanup_failed"
                    if runtime.trace:
                        runtime.trace.close()
            try:
                result = _result(
                    runtime,
                    output,
                    elapsed=time.monotonic() - started,
                    failure=failure,
                    star=star,
                    case_hash=case_hash,
                )
            except Exception:  # noqa: BLE001 - retain the failed attempt without echoing artifact contents
                result = {
                    "run_id": runtime.run_id,
                    "numeric_transport_passed": False,
                    "failure_reason": failure or "artifact_verification_failed",
                    "star_name": star,
                    "measurement_case_sha256": case_hash,
                    "learned_decisions": 0,
                    "write_attempts": 0,
                    "provenance": {},
                    "metrics_available": False,
                }
            # All cases in the batch must use the same frozen identities.
            if report["runs"] and result["numeric_transport_passed"]:
                keys = ("checkpoint_sha256", "graph_hash", "knowledge_pack_hash")
                if any(result["provenance"].get(k) != report["runs"][0]["provenance"].get(k) for k in keys):
                    result.update(numeric_transport_passed=False, failure_reason="batch_provenance_changed")
            report["runs"].append(result)
            save()
            notify(
                f"Run {index + 1}: {'PASS' if result['numeric_transport_passed'] else 'STOP'}; "
                f"{result['learned_decisions']} decisions, {result['write_attempts']} writes."
            )
            if not result["numeric_transport_passed"]:
                break
    except KeyboardInterrupt:
        report["batch_stop_reason"] = "operator_aborted"
    finally:
        credentials = None
        for key in LOGIN_KEYS:
            os.environ.pop(key, None)
        save()
    return report
