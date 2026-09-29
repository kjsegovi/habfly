"""Post-run four-field accuracy audit and capped sequential browser batch.

Reference answers are computed only after browser teardown. They never select,
repair, retry or grade a live policy action. No HabWorlds grading is performed.
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .browser_four_field_policy import load_four_field_policy
from .browser_policy import FIELDS, UNITS, browser_config
from .browser_reliability import LOGIN_KEYS, _run_reliability, validate_batch
from .color_reference import load_color_reference
from .contracts import Action, Observation, validate_action
from .knowledge import LocalCalculator, load_knowledge_pack
from .runtime import read_trace

RUN_SECONDS = 1260  # Parent 1200s guard plus model load/teardown; no budget expansion.
IDENTITY_KEYS = ("graph_hash", "numeric_policy", "color_policy", "numeric_seed", "color_seed", "coordinator")


def _json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _events(path):
    return read_trace(path) if path.exists() else []


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def audit_saved_run(root, trace, *, authorization="autonomous_opt_in"):
    """Read-only saved evidence; callers must close the browser before invoking."""
    root, trace = Path(root), Path(trace)
    manifest = _json(root / "manifest.json")
    provenance = manifest.get("provenance", {})
    events = _events(trace)
    children, journals = {}, {}
    integrity = bool(manifest) and _sha(root / "events.jsonl") == manifest.get("events_sha256")
    _events(root / "events.jsonl")  # Protocol parse as well as byte integrity.
    for stage in ("numeric", "color"):
        path = root / stage / "manifest.json"
        children[stage] = _json(path)
        journals[stage] = _events(path.parent / "events.jsonl")
        linked = manifest.get("components", {}).get(stage, {})
        summaries = [e.payload for e in journals[stage] if e.event == "episode_summary"]
        integrity = bool(
            integrity
            and children[stage]
            and linked.get("manifest") == f"{stage}/manifest.json"
            and linked.get("sha256") == _sha(path)
            and children[stage].get("events_sha256") == _sha(path.parent / "events.jsonl")
            and len(summaries) == 1
            and all(summaries[0].get(k) == v for k, v in children[stage].items() if k != "events_sha256")
        )

    proposals, results, snapshot = [], [], None
    invalid = 0
    selected_color_snapshot = None
    for event in events:
        if event.event == "observation":
            snapshot = Observation.model_validate(event.payload)
        elif event.event == "action_proposed":
            proposals.append(event.payload)
            try:
                action = Action.model_validate(
                    {k: v for k, v in event.payload.items() if k in Action.model_fields}
                )
                validate_action(snapshot, action)
            except (ValueError, AttributeError):
                invalid += 1
            if event.payload.get("policy_stage") == "color" and str(event.payload.get("target")).endswith(
                ":color"
            ):
                selected_color_snapshot = snapshot
        elif event.event == "action_result":
            results.append(event.payload)
    errors = sum(e.event == "error" for stream in (events, *journals.values()) for e in stream)
    counts = Counter(p.get("policy_stage") for p in proposals)
    summaries = [e.payload for e in events if e.event == "episode_summary"]
    numeric = children["numeric"]
    color = children["color"]
    numeric_copies = [
        e.payload
        for e in journals["numeric"]
        if e.event == "action_result" and e.payload.get("numeric_copy_verified")
    ]
    color_copies = [
        e.payload
        for e in journals["color"]
        if e.event == "action_result" and e.payload.get("color_transport_verified")
    ]
    native_proposals = [e.payload for e in journals["numeric"] if e.event == "action_proposed"]
    receipt = manifest.get("color_receipt") or {}
    readbacks = manifest.get("numeric_readbacks", {})
    handoff = manifest.get("handoff") or {}
    mode = "autonomous" if authorization == "autonomous_opt_in" else "supervised"
    transport = bool(
        integrity
        and not errors
        and not invalid
        and len(summaries) == 1
        and all(summaries[0].get(k) == v for k, v in manifest.items())
        and manifest.get("four_field_transport_verified") is True
        and manifest.get("outcome") == "four_field_transport_verified"
        and manifest.get("write_attempts") == 4
        and numeric.get("numeric_transport_passed") is True
        and numeric.get("write_attempts") == 3
        and color.get("color_transport_verified") is True
        and color.get("write_attempts") == 1
        and numeric.get("numeric_readbacks") == readbacks
        and color.get("receipt") == receipt
        and provenance.get("browser_execution") == mode
        and provenance.get("optimizer_updates") == 0
        and all(children[s].get("provenance") == provenance.get(f"{s}_policy") for s in children)
        and set(readbacks) == set(FIELDS)
        and len(numeric_copies) == len(native_proposals) == 3
        and len(color_copies) == 1
        and all(r.get("copy_authorization") == authorization for r in numeric_copies)
        and receipt.get("color_authorization") == authorization
        and receipt.get("readback_verified") is True
        and color_copies[0].get("receipt") == receipt
        and color_copies[0].get("action", {}).get("kind") == "SELECT"
        and color_copies[0].get("action", {}).get("value") == receipt.get("selected_color")
        and handoff.get("same_star_and_native_controls") is True
        and handoff.get("numeric_readbacks_preserved") is True
        and handoff.get("recurrent_state_reset") is True
        and handoff.get("numeric_screen_sha256") == handoff.get("color_screen_sha256")
        and 0 < counts["numeric"] <= 64
        and 0 < counts["color"] <= 8
        and [p.get("policy_stage") for p in proposals]
        == ["numeric"] * counts["numeric"] + ["color"] * counts["color"]
        and all(
            p.get("action_source") == "checkpoint"
            and p.get("calibrated") is False
            and p.get("action_confidence") is None
            and p.get("target_confidence") is None
            for p in proposals
        )
        and len(results) == len(proposals)
        and [r.get("steps") for r in results] == list(range(1, len(results) + 1))
        and results[-1].get("terminated")
        and all(r.get("failure_reason") is None for r in results)
        and all(
            manifest.get(k) is False
            for k in ("task_completed", "browser_acceptance_passed", "allow_submission")
        )
    )
    numeric_matches, unit_matches, expected, source_match = {}, {}, {}, None
    expected_color = color_match = preserved = None
    if transport:
        initial = next(e.payload["values"] for e in journals["numeric"] if e.event == "observation")
        color_start = next(e.payload["values"] for e in journals["color"] if e.event == "observation")
        final = color_copies[0]["observation"]["values"]
        measurements = initial["measurements"]
        preserved = (
            initial["star_name"] == color_start["star_name"] == final["star_name"] == handoff["star_name"]
            and measurements == color_start["measurements"] == final["measurements"]
            and all(not f["current_value"] for f in initial["browser_field_map"].values())
            and color_start["color"]["selected"] is None
            and final["color"]["selected"] == receipt["selected_color"]
            and all(
                color_start["browser_field_map"][f]["current_value"]
                == final["browser_field_map"][f]["current_value"]
                == readbacks[f]["display_value"]
                for f in FIELDS
            )
        )
        calculator = LocalCalculator(load_knowledge_pack())
        reference = load_color_reference()
        if (
            calculator.pack.checksum != provenance["numeric_policy"]["knowledge_pack_hash"]
            or reference.checksum != provenance["color_policy"]["color_reference_hash"]
            or reference.checksum != receipt["reference_hash"]
        ):
            raise ValueError("Reference identity changed after run")
        inputs = {
            "distance": {"parallax": measurements["browser_parallax"]},
            "temperature": {"wavelength": measurements["browser_wavelength"]},
        }
        for field in FIELDS:
            if field == "luminosity":
                inputs[field] = {
                    "flux": measurements["browser_flux"],
                    "distance": {"value": expected["distance"], "unit": "ly"},
                }
            result = calculator.execute_unclassified_common(field, inputs[field])
            if not result.ok:
                raise ValueError("Post-run reference calculation failed")
            expected[field] = result.value
            numeric_matches[field] = float(readbacks[field]["exact_copied"]) == result.value
            unit_matches[field] = readbacks[field]["unit"] == UNITS[field]
        for proposed, copied, field in zip(native_proposals, numeric_copies, FIELDS, strict=True):
            if (
                proposed.get("copy_authorization") != authorization
                or proposed["action"]["kind"] != "TYPE"
                or not proposed["action"]["target"].endswith(f":{field}")
                or proposed["action"]["value"] != readbacks[field]["exact_copied"]
                or copied["numeric_readback"] != readbacks[field]
                or copied["action"] != proposed["action"]
                or copied["commit_action"] != proposed["commit_action"]
                or proposed["commit_action"]["kind"] != "KEYPRESS"
                or proposed["commit_action"]["value"] != "Tab"
                or not readbacks[field]["exact_input_verified"]
            ):
                transport = False
        source = selected_color_snapshot.calculation.get("source") if selected_color_snapshot else None
        source_match = bool(
            selected_color_snapshot
            and selected_color_snapshot.values["measurements"].get(source)
            == measurements["browser_wavelength"]
        )
        expected_color = reference.private_label(measurements["browser_wavelength"]["value"], "nm")
        color_match = receipt["selected_color"] == expected_color
    passed = bool(
        transport
        and preserved
        and source_match
        and color_match
        and all(numeric_matches.values())
        and all(unit_matches.values())
    )
    reason = None if passed else (manifest.get("outcome") or "missing_run_manifest")
    if reason == "four_field_transport_verified":
        reason = (
            "reliability_evidence_gate_failed"
            if not transport or not preserved
            else "incorrect_numeric_answer"
            if not all(numeric_matches.values()) or not all(unit_matches.values())
            else "incorrect_measurement_source"
            if not source_match
            else "incorrect_color_selection"
        )
    return {
        "four_field_reliability_passed": passed,
        "four_field_transport_verified": transport,
        "failure_reason": reason,
        "learned_decisions": len(proposals),
        "policy_stage_decisions": dict(counts),
        "write_attempts": manifest.get("write_attempts", 0),
        "runtime_errors": errors,
        "invalid_actions": invalid,
        "journal_hash_verified": integrity,
        "numeric_values_preserved_after_color": preserved,
        "numeric_reference_matches": numeric_matches,
        "unit_matches": unit_matches,
        "post_run_reference_values": expected,
        "post_run_reference_color": expected_color,
        "selected_color": receipt.get("selected_color"),
        "color_source_correct": source_match,
        "color_reference_match": color_match,
        "provenance": provenance,
    }


def _result(runtime, output, *, elapsed, failure, star, case_hash):
    if runtime.env is not None:
        raise ValueError("Post-run accuracy audit requires closed runtime")
    trace = output / "runs" / f"{runtime.run_id}.jsonl"
    root = trace.with_suffix("")
    result = audit_saved_run(root, trace)
    if failure:
        result.update(four_field_reliability_passed=False, failure_reason=failure)
    reason = result["failure_reason"]
    result.update(
        run_id=runtime.run_id,
        star_name=star,
        measurement_case_sha256=case_hash,
        elapsed_seconds=round(elapsed, 3),
        trace=str(trace.relative_to(output)),
        manifest=str((root / "manifest.json").relative_to(output)),
        failure_category=None
        if result["four_field_reliability_passed"]
        else "operator"
        if reason == "operator_aborted"
        else "policy"
        if reason
        in {
            "incorrect_numeric_answer",
            "incorrect_measurement_source",
            "incorrect_color_selection",
            "invalid_action",
            "policy_stopped",
            "incompatible_measurement_kind",
            "color_step_limit",
            "repeated_unchanged_tool_actions",
        }
        else "safety_or_infrastructure",
    )
    return result


def run_four_field_reliability(options, output, *, runs, credentials, notify=print):
    try:
        validate_batch(options, runs, credentials, task="browser_four_field")
        _, provenance = load_four_field_policy(options)
        browser_config(options)
        identity = {key: provenance[key] for key in IDENTITY_KEYS}
        report = _run_reliability(
            options,
            output,
            runs=runs,
            credentials=credentials,
            notify=notify,
            task="browser_four_field",
            scope="autonomous_four_field_browser_reliability",
            pass_key="four_field_reliability_passed",
            identity_keys=IDENTITY_KEYS,
            result_reader=_result,
            run_seconds=RUN_SECONDS,
            write_limit=4,
            initial_identity=identity,
        )
        rows = report["runs"]
        report.update(
            four_field_transport_verified_runs=sum(
                r.get("four_field_transport_verified", False) for r in rows
            ),
            field_reference_matching_runs={
                f: sum(r.get("numeric_reference_matches", {}).get(f) is True for r in rows) for f in FIELDS
            },
            color_reference_matching_runs=sum(r.get("color_reference_match") is True for r in rows),
            color_source_correct_runs=sum(r.get("color_source_correct") is True for r in rows),
            reference_band_counts=dict(
                Counter(r["post_run_reference_color"] for r in rows if r.get("post_run_reference_color"))
            ),
            policy_failure_runs=sum(r.get("failure_category") == "policy" for r in rows),
            safety_or_infrastructure_failure_runs=sum(
                not r["four_field_reliability_passed"]
                and r.get("failure_category") not in {"policy", "operator"}
                for r in rows
            ),
            scoring_scope="post_run_local_reference_not_habworlds_grading",
            expected_identity=identity,
        )
        pending = Path(output) / "report.tmp"
        pending.write_text(json.dumps(report, indent=2) + "\n")
        pending.replace(Path(output) / "report.json")
        return report
    finally:
        credentials = None
        for key in LOGIN_KEYS:
            os.environ.pop(key, None)
