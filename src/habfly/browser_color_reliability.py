"""Bounded color transport + post-run reference checks, never course grading.

The runner closes each browser before this module inspects its evidence. Private
reference labels are used only here, never to select, repair or retry an action.
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .browser_color_policy import load_browser_color_policy
from .browser_policy import browser_config
from .browser_reliability import LOGIN_KEYS, _run_reliability, validate_batch
from .color_reference import load_color_reference
from .runtime import read_trace

RUN_SECONDS = 240  # 90s scripted setup + 120s color session + model/cleanup overhead.
IDENTITY_KEYS = ("checkpoint_sha256", "graph_hash", "color_reference_hash", "final_report_sha256")


def _result(runtime, output, *, elapsed, failure, star, case_hash):
    trace = output / "runs" / f"{runtime.run_id}.jsonl"
    manifest_path = trace.with_suffix("") / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    journal_path = manifest_path.parent / "events.jsonl"
    events = read_trace(trace) if trace.exists() else []
    journal = read_trace(journal_path) if journal_path.exists() else []
    integrity = (
        bool(manifest)
        and journal_path.exists()
        and (hashlib.sha256(journal_path.read_bytes()).hexdigest() == manifest.get("events_sha256"))
    )
    proposals = [e.payload for e in events if e.event == "action_proposed"]
    summaries = [e.payload for e in events if e.event == "episode_summary"]
    receipts = [
        e.payload for e in journal if e.event == "action_result" and e.payload.get("color_transport_verified")
    ]
    receipt = manifest.get("receipt") or {}
    provenance = manifest.get("provenance", {})
    errors = sum(e.event == "error" for e in events)
    transport = bool(
        integrity
        and not errors
        and len(summaries) == 1
        and len(receipts) == 1
        and manifest.get("color_transport_verified") is True
        and manifest.get("outcome") == "color_transport_verified"
        and manifest.get("write_attempts") == 1
        and all(
            summaries[0].get(k) == manifest.get(k)
            for k in (
                "outcome",
                "color_transport_verified",
                "write_attempts",
                "receipt",
                "provenance",
                "events_sha256",
            )
        )
        and receipt.get("readback_verified") is True
        and receipt.get("correctness_verified") is False
        and receipt.get("color_authorization") == "autonomous_opt_in"
        and receipts[0].get("receipt") == receipt
        and receipts[0].get("action", {}).get("kind") == "SELECT"
        and receipts[0].get("action", {}).get("value") == receipt.get("selected_color")
        and provenance.get("browser_execution") == "autonomous"
        and provenance.get("color_gate_passed") is True
        and provenance.get("optimizer_updates") == 0
        and 0 < len(proposals) <= 8
        and all(
            manifest.get(k) is False and summaries[0].get(k) is False
            for k in ("task_completed", "browser_acceptance_passed", "allow_submission")
        )
    )
    # Snapshot before the learned color proposal; do not inspect hidden DOM or
    # generate a corrected action. The browser/runtime is already closed.
    snapshot, selected_snapshot, proposed_colors = None, None, []
    for event in events:
        if event.event == "observation":
            snapshot = event.payload
        if (
            event.event == "action_proposed"
            and event.payload.get("kind") == "SELECT"
            and str(event.payload.get("target", "")).endswith(":color")
        ):
            selected_snapshot = snapshot
            proposed_colors.append(event.payload.get("value"))
    selected_snapshot = selected_snapshot or {}
    source = selected_snapshot.get("calculation", {}).get("source")
    measurement = selected_snapshot.get("values", {}).get("measurements", {}).get(source, {})
    native_values = next((e.payload.get("values", {}) for e in journal if e.event == "observation"), {})
    native_measurement = native_values.get("measurements", {}).get("browser_wavelength", {})
    source_match = (
        measurement.get("kind") == "wavelength"
        and measurement.get("unit") == "nm"
        and measurement.get("source") == "current star"
        and measurement == native_measurement
    )
    reference = load_color_reference()
    expected = reference.private_label(measurement["value"], measurement["unit"]) if source_match else None
    identity_ok = (
        reference.checksum == provenance.get("color_reference_hash") == receipt.get("reference_hash")
    )
    proposal_match = proposed_colors == [receipt.get("selected_color")]
    reference_match = (
        receipt.get("selected_color") == expected
        if transport and identity_ok and proposal_match and expected is not None
        else None
    )
    passed = bool(
        failure is None and transport and identity_ok and proposal_match and source_match and reference_match
    )
    reason = failure
    if not passed and reason is None:
        if not transport or not identity_ok or not proposal_match:
            reason = manifest.get("outcome", "missing_run_manifest")
            if reason == "color_transport_verified":
                reason = "reliability_evidence_gate_failed"
        else:
            reason = "incorrect_measurement_source" if not source_match else "incorrect_color_selection"
    return {
        "run_id": runtime.run_id,
        "color_reliability_passed": passed,
        "color_transport_verified": transport,
        "failure_reason": reason,
        "failure_category": (
            None
            if passed
            else "policy"
            if reason
            in {
                "incorrect_measurement_source",
                "incorrect_color_selection",
                "incompatible_measurement_kind",
                "invalid_action",
                "policy_stopped",
                "color_step_limit",
                "repeated_unchanged_tool_actions",
            }
            else "operator"
            if reason == "operator_aborted"
            else "safety_or_infrastructure"
        ),
        "star_name": star,
        "measurement_case_sha256": case_hash,
        "learned_decisions": len(proposals),
        "write_attempts": manifest.get("write_attempts", 0),
        "runtime_errors": errors,
        "journal_hash_verified": integrity,
        "autonomous_color_receipts": len(receipts),
        "selected_source": source,
        "source_selection_correct": source_match,
        "selected_measurement": measurement,
        "selected_color": receipt.get("selected_color"),
        "post_run_reference_color": expected,
        "post_run_reference_match": reference_match,
        "elapsed_seconds": round(elapsed, 3),
        "trace": str(trace.relative_to(output)),
        "manifest": str(manifest_path.relative_to(output)),
        "provenance": provenance,
    }


def run_color_reliability(options, output, *, runs, credentials, notify=print):
    try:
        return _run_color_reliability(options, output, runs=runs, credentials=credentials, notify=notify)
    finally:
        credentials = None
        for key in LOGIN_KEYS:
            os.environ.pop(key, None)


def _run_color_reliability(options, output, *, runs, credentials, notify):
    validate_batch(options, runs, credentials, task="browser_color")
    # Before any login/browser, pin the gated model, graph, reference and final
    # report. Runtime revalidates those exact identities before every new session.
    _, provenance = load_browser_color_policy(options)
    browser_config(options)
    identity = {key: provenance[key] for key in IDENTITY_KEYS}
    report = _run_reliability(
        options,
        output,
        runs=runs,
        credentials=credentials,
        notify=notify,
        task="browser_color",
        scope="autonomous_color_browser_reliability",
        pass_key="color_reliability_passed",
        identity_keys=IDENTITY_KEYS,
        result_reader=_result,
        run_seconds=RUN_SECONDS,
        write_limit=1,
        initial_identity=identity,
    )
    rows = report["runs"]
    report.update(
        color_transport_verified_runs=sum(r.get("color_transport_verified", False) for r in rows),
        reference_matching_runs=sum(r.get("post_run_reference_match") is True for r in rows),
        reference_checked_runs=sum(r.get("post_run_reference_match") is not None for r in rows),
        source_correct_runs=sum(r.get("source_selection_correct", False) for r in rows),
        reference_band_counts=dict(
            Counter(r["post_run_reference_color"] for r in rows if r.get("post_run_reference_color"))
        ),
        policy_failure_runs=sum(r.get("failure_category") == "policy" for r in rows),
        safety_or_infrastructure_failure_runs=sum(
            not r["color_reliability_passed"] and r.get("failure_category") not in {"policy", "operator"}
            for r in rows
        ),
        scoring_scope="post_run_local_reference_not_habworlds_grading",
        expected_identity=identity,
    )
    pending = Path(output) / "report.tmp"
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(Path(output) / "report.json")
    return report
