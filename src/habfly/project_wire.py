"""Optional compact display projection; never an evidence receipt or a journal.

The raw bridge and component JSONL journals remain authoritative and unchanged.
Data events carry exactly one native payload at the existing top-level keys.
Their original envelope can be reconstructed from the small ``project_wire``
header/key list. Lifecycle snapshots are deliberately summaries, not lossless
copies; only the incoming parent may supply top-level completion/status fields.
The source digest binds an incoming event/payload pair, not a mutable trace file.
"""

import hashlib
import json
import re
from copy import deepcopy

LIFECYCLE = frozenset({"hello", "state", "episode_summary"})
EVENTS = LIFECYCLE | {"observation", "action_proposed", "action_result", "neural_activity", "error"}
MAX_ANCESTRY = 8
_ROOTS = {
    "browser.setup",
    "browser.reference_class",
    "project.owner",
    "project.campaign",
    "project.finalization.scoring",
    "project.finalization.submission",
}
_CHILDREN = {
    "project.campaign": {"campaign.owner", "campaign.next_star"},
    "project.owner": {"star.workflow", "project.inventory", "planet.finalization", "planet.terrestrial"},
    "campaign.owner": {"star.workflow", "project.inventory", "planet.finalization", "planet.terrestrial"},
    "campaign.next_star": {"next_star.picker"},
    "star.workflow": {
        "stellar.numeric",
        "stellar.color",
        "planet.window",
        "planet.spectrum",
        "planet.positive",
        "planet.shallow",
    },
    "planet.shallow": {"shallow.probe"},
    "planet.positive": {"raw", "derived"},
    "project.inventory": {"project.inventory.pages"},
    "planet.terrestrial": {"terrestrial.temperature", "terrestrial.chamber"},
    "project.finalization.scoring": {
        "project.assessment",
        "project.panel.data_quality",
        "project.panel.scavenger_hunt",
        "project.inventory.data_quality",
        "project.inventory.scavenger_hunt",
        "project.inventory.score_transfer",
    },
}
_DETAILS = frozenset(
    [
        "current_star",
        "initial_star",
        "project_progress",
        "reference_measurements",
        "artifact_paths",
        "class_handoff",
        "inventory_handoff",
        "planet_class_handoff",
        "planet_class_decision",
        "gas_handoff",
        "habitability_handoff",
        "autonomous_decision",
        "submission_feedback",
        "owner_limits",
        "project_limits",
        "transition_limits",
        "class_setup_limits",
        "project_viewport",
        "pending_canonical_action",
        "pending_acknowledgement",
        "reservation_intent",
        "save_outcome",
    ]
)
_SUMMARY_SCALARS = frozenset(
    [
        "mode",
        "status",
        "phase",
        "stage",
        "browser_phase",
        "browser_status",
        "policy_stage",
        "finished",
        "failure_reason",
        "task_completed",
        "project_completed",
        "target_workflows_verified",
        "completed",
        "component_only",
        "event_forwarding_failed",
        "star",
        "advances",
        "max_advances",
        "max_seconds",
        "target_stars",
        "verified_stars",
        "selected_class",
        "lifetime_prefix",
        "setup_verified",
        "observation_limit_days",
        "polls",
        "max_polls",
        "window_status",
        "waiting_for_clean_partial_chart",
        "scheduling_rule",
        "prior_dip_observed",
        "waiting_for_positive_endpoint",
        "decision_readback_verified",
        "collection_count_verified",
        "navigation_clicks",
        "scoring_max_seconds",
        "submission_max_seconds",
        "max_scoring_advances",
        "max_submission_advances",
        "scoring_max_advances",
        "submission_max_advances",
        "allow_score_transfer",
        "allow_submission",
        "score_transfer_verified",
        "submitted",
        "submission_verified",
        "submission_outcome",
        "canonical_reservation_attempted",
        "canonical_reservation_verified",
        "assessment_outcome_uncertain",
        "readiness_readback_verified",
        "readiness_write_may_have_occurred",
        "readiness_click_returned",
        "submit_click_returned",
        "submit_write_may_have_occurred",
        "campaign_budget_extended",
        "source_state_unavailable",
        "acknowledgement_recognizer_implemented",
        "canonical_success_receipt_implemented",
        "requested_panel",
        "requested_inventory_refresh",
        "scoring_dir",
        "shallow_reference_enabled",
        "measurement_mode",
        "confirmed_feature_count",
        "observed_interval_count",
        "consistency_redundancy",
        "consecutive_events_assumed",
        "recurrence_confirmed",
        "observed_recurrence_compatible",
        "single_spacing_compatible",
        "native_action_attempts",
        "native_actions_confirmed",
        "max_native_actions",
        "probe_index",
        "native_action_outcome_uncertain",
        "save_outcome_uncertain",
        "learned_perception",
        "scientific_verified",
        "training_label",
    ]
)
_ROOT_SCALARS = _SUMMARY_SCALARS | frozenset(
    [
        "protocol_version",
        "project_campaign",
        "setup_stage",
        "setup_advances",
        "assessment_enabled",
        "submission_enabled",
        "automatic_retry",
        "automatic_deadline_increase",
        "scientific_verified",
        "automatic_classification",
        "automatic_planet_classification",
        "automatic_gas_identification",
        "automatic_habitability_decision",
        "autonomous_decisions_enabled",
        "supplied_stellar_inputs_enabled",
        "baseline_edge_reference_enabled",
        "save_strategy",
        "persistence_verified",
        "score_checkpoint_completed",
        "reported_score",
        "decision_source",
        "experimental_hr_matcher_enabled",
        "reference_decisions",
        "classification_source",
        "learned_policy",
        "seed",
        "stars",
        "task",
        "runtime_task",
        "environment",
        "policy",
        "graph",
        "checkpoint",
        "activity_source",
        "trace_path",
        "project_trace_path",
        "artifact_root",
    ]
)
_HANDOFFS = frozenset(
    {
        "class_handoff",
        "inventory_handoff",
        "planet_class_handoff",
        "planet_class_decision",
        "gas_handoff",
        "habitability_handoff",
    }
)
_CHILD_SUMMARIES = frozenset(
    {
        "project_owner",
        "star_component",
        "inventory_component",
        "positive_component",
        "terrestrial_component",
        "next_star",
        "class_setup",
        "component",
        "finalization_component",
        "scoring_component",
        "panel_component",
    }
)
_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "email",
        "credentials",
        "credential",
        "authorization",
        "cookie",
        "cookies",
        "storage_state",
        "access_token",
        "refresh_token",
        "api_key",
        "secret",
        "session_url",
        "preview_url",
        "login_url",
    }
)


def _require(condition):
    if not condition:
        raise ValueError("project_compact_wire_invalid_event")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _envelope(value):
    _require(isinstance(value, dict) and {"event", "payload"} <= value.keys())
    _require(value.keys() <= {"version", "event", "payload", "sequence", "run_id"})
    _require(value["event"] in EVENTS and isinstance(value["payload"], dict))
    if len(value) != 2:
        _require(type(value.get("sequence")) is int and value["sequence"] >= 0)
        _require("version" not in value or type(value["version"]) is int and value["version"] == 1)
        run_id = value.get("run_id")
        _require(run_id is None or isinstance(run_id, str) and 0 < len(run_id) <= 128)
    return value


def _native(kind, payload):
    """Follow only recognized relay edges with their actual envelope contract."""
    current = _envelope({"event": kind, "payload": payload})
    ancestry = []
    previous = None
    while "component_event" in current["payload"]:
        data = current["payload"]
        _require("project_wire" not in data)
        special = previous == "planet.positive"
        identity = data.get("component_identity") if special else data.get("component")
        _require(isinstance(identity, str))
        _require(identity in (_ROOTS if previous is None else _CHILDREN.get(previous, set())))
        _require(len(ancestry) < MAX_ANCESTRY)
        child = _envelope(data["component_event"])
        child_kind = child["event"]
        if child_kind in LIFECYCLE:
            _require(current["event"] == "state")
            duplicate = (
                "component_summary"
                if special
                else {
                    "hello": "component_hello",
                    "state": "component_state",
                    "episode_summary": "component_summary",
                }[child_kind]
            )
            _require(data.get(duplicate) == child["payload"])
        else:
            _require(current["event"] == child_kind)
        star = data.get("component_star") if not special else data.get("star")
        _require(
            star is None
            or isinstance(star, str)
            and 0 < len(star) <= 100
            and star == star.strip()
            and all(ord(c) >= 32 for c in star)
        )
        ancestry.append({"component": identity, "star": star})
        previous, current = identity, child
    _require("project_wire" not in current["payload"])
    return current, ancestry


def _public(value, *, depth=0):
    """Reject secret material rather than silently changing exact native data."""
    _require(depth <= 64)
    if isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str) and key.casefold() not in _SENSITIVE_KEYS)
            _public(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _public(item, depth=depth + 1)
    elif isinstance(value, str):
        # Student prose and harmless public URLs remain exact; session query
        # URLs/userinfo must not enter stdout even if upstream misroutes them.
        _require(not re.search(r"https?://[^\s]*[?@#]", value, re.IGNORECASE))
        _require(not re.search(r"(?:preview_sequence_id|access_token|password)=", value, re.IGNORECASE))
    else:
        _require(value is None or type(value) in {bool, int, float})


def _detail(value, *, depth=0):
    """Bound summary structure; leaf observations themselves are not truncated."""
    _require(depth <= 8)
    if isinstance(value, dict):
        _require(len(value) <= 256)
        return {key: _detail(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        _require(len(value) <= 256)
        return [_detail(item, depth=depth + 1) for item in value]
    _require(not isinstance(value, str) or len(value) <= 8192)
    return value


def _summary(payload, *, depth=0, campaign=False, children=True):
    _require(isinstance(payload, dict))
    output = {}
    if depth == 0 and payload.get("campaign_max_seconds") == "uncapped":
        output["campaign_max_seconds"] = "uncapped"
    for key in (_ROOT_SCALARS if depth == 0 else _SUMMARY_SCALARS) & payload.keys():
        value = payload[key]
        if key == "policy" and isinstance(value, dict):
            # A window policy manifest is raw evidence, not the runtime's
            # short policy label. Its exact source remains in the journal.
            continue
        _require(not isinstance(value, (dict, list)))
        output[key] = _detail(value)
    for key in (
        _DETAILS
        if depth == 0
        else _HANDOFFS | {"pending_canonical_action", "pending_acknowledgement", "reservation_intent"}
    ) & payload.keys():
        output[key] = _detail(payload[key])
    if payload.get("mode") == "cooperative_shallow_transit_measurements":
        # A maximum of three completed probe receipts and the current overview
        # pins are useful progress, not cached observations or whole source trees.
        for key in {"completed_probes", "window_evidence"} & payload.keys():
            output[key] = _detail(payload[key])
    if (
        payload.get("mode") == "bounded_planet_window"
        and isinstance(payload.get("policy"), dict)
        and payload["policy"].get("version") == "user_approved_5000_day_single_event_no_planet_v1"
    ):
        # Display-only provenance for this explicitly recorded policy. Legacy
        # window bytes stay unchanged; analysis never becomes a parent result.
        output["policy"] = {}
        for key in {"version", "sha256"} & payload["policy"].keys():
            value = payload["policy"][key]
            _require(isinstance(value, str))
            output["policy"][key] = _detail(value)
        analysis = payload.get("analysis")
        if isinstance(analysis, dict):
            selected = {}
            for key in {
                "status",
                "reason",
                "approximation",
                "visible_candidate_events",
                "possible_planet_ignored",
            } & analysis.keys():
                value = analysis[key]
                _require(not isinstance(value, (dict, list)))
                selected[key] = _detail(value)
            if selected:
                output["analysis"] = selected
    # Keep the existing UI paths. A campaign's owner/progress are already in
    # the parent and must not be copied again inside the campaign summary.
    for key in _CHILD_SUMMARIES & payload.keys():
        if not children:
            continue
        if campaign and key != "next_star":
            continue
        value = payload[key]
        if value is None:
            output[key] = None
        elif isinstance(value, dict) and depth < 3:
            output[key] = _summary(value, depth=depth + 1)
        elif isinstance(value, str) and key == "component":
            output[key] = value
    if depth == 0 and "campaign" in payload:
        value = payload["campaign"]
        output["campaign"] = _summary(value, depth=1, campaign=True) if isinstance(value, dict) else None
    if campaign:
        for key in _DETAILS:
            output.pop(key, None)
    return output


def compact_project_event(kind, payload, *, source_trace="events.jsonl", source_header=None):
    """Return a new v1 payload; do not mutate source objects or perform any IO.

    ``source_event_sha256`` is canonical JSON of ``{event, payload}`` as received
    from the bridge, before runtime metadata. It is a lookup binding, NOT a
    receipt, trust promotion, whole-file checksum or authentication guarantee.
    ``source_trace=None`` is reserved for runtime-generated lifecycle snapshots:
    no raw component record/file is claimed for those display-only messages.
    """
    try:
        _require(isinstance(payload, dict) and "project_wire" not in payload)
        if source_trace is None:
            _require(kind in LIFECYCLE)
        else:
            _require(isinstance(source_trace, str) and len(source_trace) <= 256)
            _require(re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.jsonl", source_trace) is not None)
            _require(not {".", ".."} & set(source_trace.split("/")))
        native, ancestry = _native(kind, payload)
        if source_header is not None:
            _require(isinstance(source_header, dict) and "payload" not in source_header)
            supplied = _envelope({**source_header, "payload": payload})
            _require(supplied["event"] == kind and source_trace is not None)
            if not ancestry:
                native = supplied
        header = {key: value for key, value in native.items() if key != "payload"}
        wire = {
            "version": 1,
            "mode": "compact_project_display",
            "source_trace": source_trace,
            **(
                {"source_origin": "runtime_generated_no_raw_component_record"} if source_trace is None else {}
            ),
            "source_event_sha256": _sha({"event": kind, "payload": payload}),
            "source_hash_algorithm": "sha256_canonical_json",
            "ancestry": ancestry,
            "leaf_header": header,
            "leaf_event_sha256": _sha(native),
            "leaf_payload_sha256": _sha(native["payload"]),
            "leaf_payload_inline": kind not in LIFECYCLE,
            "evidence_receipt": False,
        }
        if kind in LIFECYCLE:
            result = _summary(payload)
            # Component claims stay scoped, never merged with parent status.
            wire["leaf_summary"] = _summary(native["payload"], depth=1, children=False)
        else:
            _require(native["event"] == kind)
            wire["leaf_payload_keys"] = sorted(native["payload"])
            wire["parent"] = {
                key: _detail(payload[key])
                for key in {
                    "status",
                    "phase",
                    "browser_phase",
                    "current_star",
                    "project_progress",
                    "reference_measurements",
                    "task_completed",
                    "project_completed",
                    "target_workflows_verified",
                }
                & payload.keys()
            }
            # Native status/phase/etc belongs to the leaf, so do not repeat it
            # as parent state unless a genuine owner lifecycle is present.
            if "browser_phase" not in payload or "project_owner" not in payload:
                wire["parent"] = {}
            result = deepcopy(native["payload"])
        result["project_wire"] = wire
        _public(result)
        _json(result)  # reject nonfinite/unsupported data without echoing it
        return result
    except Exception:  # noqa: BLE001 - public boundary must never echo rejected source material
        raise ValueError("project_compact_wire_invalid_event") from None


def reconstruct_native_event(payload):
    """Validate/reconstruct a compact DATA envelope without trusting its claims.

    Lifecycle summaries are intentionally non-reconstructible; consult their
    source digest in the untouched raw bridge journal instead.
    """
    try:
        wire = payload["project_wire"]
        _require(wire["version"] == 1 and wire["mode"] == "compact_project_display")
        _require(wire["leaf_payload_inline"] is True and wire["evidence_receipt"] is False)
        keys = wire["leaf_payload_keys"]
        _require(isinstance(keys, list) and all(isinstance(key, str) for key in keys))
        _require(keys == sorted(set(keys)) and "project_wire" not in keys)
        _require(set(payload) == set(keys) | {"project_wire"})
        native = {**deepcopy(wire["leaf_header"]), "payload": {key: deepcopy(payload[key]) for key in keys}}
        _envelope(native)
        _require(native["event"] not in LIFECYCLE)
        _require(_sha(native["payload"]) == wire["leaf_payload_sha256"])
        _require(_sha(native) == wire["leaf_event_sha256"])
        _public(native)
        return native
    except Exception:  # noqa: BLE001 - corrupt metadata/native values remain private
        raise ValueError("project_compact_wire_invalid_native_envelope") from None
