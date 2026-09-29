"""Read-only verification of one saved stellar / reference-No / explicit-N/A branch.

Only two numbered-tab navigation clicks are permitted. Existing scientific
choices are never repaired. A receipt proves current visible workflow readback,
not correct astronomy, perfect score, hidden values or project submission.
"""

import hashlib
import json
import math
import re
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import read_class_choices
from .browser_full_stellar import UNITS, FullStellarSession
from .browser_no_planet_save import (
    CONDITIONAL_FIELDS,
    RAW_FIELDS,
    _blank_no,
    _load_choice,
    _predispatch_source,
    _reconciliation_source,
    no_planet_visit_equivalence,
)
from .browser_no_planet_save import MODE as SAVE_MODE
from .browser_numeric import NumericJournal, committed_display, digest, screen_identity
from .browser_planet import map_planet_capture
from .browser_planet_numeric import PlanetNumericSession, planet_projection
from .browser_probe import save_probe
from .browser_project_navigation import navigate_project
from .browser_setup import rendered_control
from .browser_star_preflight import validate_star_class_source
from .browser_stellar import CLASSES, SIMULATION_URL, map_stellar_capture
from .contracts import RuntimeEvent


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("no_planet_workflow_" + reason)


class _Evidence:
    def __init__(self, history):
        self.history = Path(history).resolve(strict=True)
        self.hashes = {}
        self.clean_directories = set()
        self.closed_trees = {}

    def path(self, path):
        path = Path(path).absolute()
        _require(path.resolve().is_relative_to(self.history), "evidence_outside_history")
        _require(
            not any(p.is_symlink() for p in (path, *path.parents) if p != self.history), "symlink_evidence"
        )
        return path.resolve()

    def read(self, path):
        path = self.path(path)
        _require(path.is_file() and path.stat().st_size <= 32_000_000, "missing_or_oversized_evidence")
        raw = path.read_bytes()
        key, sha = str(path.relative_to(self.history)), hashlib.sha256(raw).hexdigest()
        _require(key not in self.hashes or self.hashes[key] == sha, "evidence_changed")
        self.hashes[key] = sha
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def capture(self, directory):
        manifest = self.json(directory / "manifest.json")
        raw = self.read(directory / "observation.json")
        _require(
            manifest.get("observation_sha256") == hashlib.sha256(raw).hexdigest(), "capture_hash_mismatch"
        )
        report = json.loads(raw)
        _require(report.get("ignored_frame_urls") == [], "unknown_capture_frame")
        return report

    def clean(self, directory):
        directory = self.path(directory)
        _require(directory.is_dir(), "missing_evidence_directory")
        _require(
            not any((directory / name).exists() for name in ("stopped.json", "invalidated.json")),
            "failed_evidence",
        )
        self.clean_directories.add(directory)
        return directory

    def unchanged(self):
        for directory in list(self.clean_directories):
            self.clean(directory)
        for directory, tree in self.closed_trees.items():
            _require(self.tree(directory) == tree, "evidence_tree_changed")
        for name in list(self.hashes):
            self.read(self.history / name)

    def tree(self, directory):
        """Bounded inventory for the explicitly opted-in Save evidence only."""
        directory = self.path(directory)
        pending, entries = [directory], []
        while pending:
            parent = pending.pop()
            for child in parent.iterdir():
                _require(len(entries) < 4096, "oversized_evidence_tree")
                path = self.path(child)
                name = str(path.relative_to(directory))
                _require(len(Path(name).parts) <= 4, "unexpected_evidence_depth")
                kind = "directory" if path.is_dir() else "file" if path.is_file() else None
                _require(kind is not None, "unexpected_evidence_type")
                entries.append((name, kind))
                if kind == "directory":
                    pending.append(path)
        return tuple(sorted(entries))


_SETTLE_POLICY = "bounded_read_only_pre_dispatch_settle_v1"
_SETTLE_KEYS = {"settle_reserved_notice", "reserved_notice_policy", "reserved_phase_timeout_seconds"}


def _reserved_notice_options_match(receipt, intent, *, reconciled=False, resumed=False):
    _require(
        _SETTLE_KEYS.intersection(receipt) == _SETTLE_KEYS.intersection(intent)
        and all(
            type(receipt[key]) is type(intent[key]) and receipt[key] == intent[key]
            for key in _SETTLE_KEYS.intersection(intent)
        )
        and (not (reconciled or resumed) or not _SETTLE_KEYS.intersection(intent)),
        "reserved_notice_intent_mismatch",
    )


def _typed_fields(value, expected, reason):
    _require(
        isinstance(value, dict)
        and set(value) == set(expected)
        and all(type(value[key]) is type(item) and value[key] == item for key, item in expected.items()),
        reason,
    )


def _reserved_notice_evidence(evidence, directory, intent, before, before_map):
    """Validate diagnostics as source evidence, never as Save acknowledgement.

    Existing dispatch, fresh acknowledgement and before/after guards remain
    required by _load_sources. Recorded times establish only consistency of
    these diagnostics, not an independently measured duration or UI success.
    """
    selected = _SETTLE_KEYS.intersection(intent)
    if not selected:
        _require(
            not (directory / "reserved-notice-settled.json").exists()
            and not (directory / "reserved-observations").exists(),
            "unselected_reserved_notice_evidence",
        )
        return
    timeout = intent.get("reserved_phase_timeout_seconds")
    _require(
        selected == _SETTLE_KEYS
        and intent["settle_reserved_notice"] is True
        and intent["reserved_notice_policy"] == _SETTLE_POLICY
        and type(timeout) in {int, float}
        and math.isfinite(timeout)
        and 0.1 <= timeout <= 30
        and not intent.get("resumed_same_intent")
        and intent.get("mode") == SAVE_MODE,
        "invalid_reserved_notice_policy",
    )
    _reserved_notice_options_match(evidence.json(directory / "confirmed.json"), intent)
    settled = evidence.json(directory / "reserved-notice-settled.json")
    _require(isinstance(settled, dict), "invalid_reserved_notice_settled")
    count, waits, deadline = (
        settled.get("full_revalidations"),
        settled.get("waiting_footer_probes"),
        settled.get("deadline_monotonic_seconds"),
    )
    _require(
        type(count) is int
        and 1 <= count <= 301
        and type(waits) is int
        and 0 <= waits <= 300
        and type(deadline) in {int, float}
        and math.isfinite(deadline)
        and deadline >= timeout,
        "invalid_reserved_notice_limits",
    )
    tree = evidence.tree(directory)
    files = {name for name, kind in tree if kind == "file"}
    static = {
        "reserved.json",
        "confirmed.json",
        "dispatch.json",
        "acknowledgement.json",
        "reserved-notice-settled.json",
        "pre-reservation-settled.json",
        "read-guard/scope.json",
    }
    base_captures = {"before", "after", "pre-dispatch-observation", "read-guard/initial"}
    capture_names = set(base_captures)
    for name in files:
        if re.fullmatch(r"pre-reservation-observations/[0-9]{3}/(?:observation|manifest)\.json", name):
            _require(int(name.split("/")[1]) <= 300, "invalid_reserved_notice_ordinal")
            capture_names.add(str(Path(name).parent))
    capture_names |= {f"reserved-observations/{index:03d}" for index in range(count)}
    reserved_probes = {f"footer-probes/reserved-final-{index:03d}.json" for index in range(count)}
    reserved_probes |= {f"footer-probes/reserved-wait-{index:03d}.json" for index in range(waits)}
    pre_probes = {
        name
        for name in files
        if re.fullmatch(r"footer-probes/(?:before-reservation|settle-(?:wait|final)-[0-9]{3})\.json", name)
    }
    _require("footer-probes/before-reservation.json" in pre_probes, "missing_preflight_footer_evidence")
    expected = (
        static
        | reserved_probes
        | pre_probes
        | {f"{name}/{leaf}.json" for name in capture_names for leaf in ("observation", "manifest")}
    )
    directories = {str(parent) for name in expected for parent in Path(name).parents if parent != Path(".")}
    _require(
        files == expected and {name for name, kind in tree if kind == "directory"} == directories,
        "unexpected_reserved_notice_files",
    )
    projection = planet_projection(before, before_map)
    captures = {}
    for name in sorted(capture_names):
        report = evidence.capture(directory / name)
        mapping = _planet(report)
        _blank_no(mapping)
        _require(planet_projection(report, mapping) == projection, "reserved_notice_visible_state_changed")
        captures[name] = report
    final_capture = f"reserved-observations/{count - 1:03d}"
    _require(
        captures[final_capture] == captures["pre-dispatch-observation"],
        "reserved_notice_final_capture_changed",
    )
    _typed_fields(
        settled,
        {
            "schema_version": 1,
            "policy": _SETTLE_POLICY,
            "full_revalidations": count,
            "waiting_footer_probes": waits,
            "last_capture": final_capture,
            "last_footer_probe": f"footer-probes/reserved-final-{count - 1:03d}.json",
            "compared_capture_sha256": screen_identity(captures[final_capture]),
            "deadline_monotonic_seconds": deadline,
            "notice_present": False,
            "reservation_retained": True,
            "save_click_dispatched": False,
            "fresh_save_acknowledgement_verified": False,
            "later_notice_excluded": False,
            "automatic_retry": False,
            "task_completed": False,
        },
        "invalid_reserved_notice_settled",
    )
    previous_end = deadline - timeout

    def probe(stage, capture):
        nonlocal previous_end
        record = evidence.json(directory / "footer-probes" / (stage + ".json"))
        _require(isinstance(record, dict), "invalid_reserved_footer_probe")
        start, finish, present = (
            record.get("probe_started_monotonic_seconds"),
            record.get("probe_finished_monotonic_seconds"),
            record.get("notice_present"),
        )
        _require(
            type(start) in {int, float}
            and type(finish) in {int, float}
            and math.isfinite(start)
            and math.isfinite(finish)
            and previous_end <= start <= finish < deadline
            and type(present) is bool,
            "invalid_reserved_footer_timing_or_verdict",
        )
        _typed_fields(
            record,
            {
                "schema_version": 1,
                "mode": "no_planet_preclick_footer_diagnostic",
                "stage": stage,
                "source": "ordinary_visible_footer_probe",
                "compared_capture_sha256": screen_identity(capture),
                "compared_capture_is_simultaneous": False,
                "probe_started_monotonic_seconds": start,
                "probe_finished_monotonic_seconds": finish,
                "diagnostic_only": True,
                "fresh_save_acknowledgement_verified": False,
                "save_click_dispatched": False,
                "retry_authorized": False,
                "task_completed": False,
                "verdict": "present" if present else "absent",
                "notice_present": present,
                "visible_text": "Data saved" if present else None,
                "same_footer_and_exposure_verified": present,
            },
            "invalid_reserved_footer_probe",
        )
        previous_end = finish
        return present

    consumed_waits = 0
    for index in range(count):
        report = captures[f"reserved-observations/{index:03d}"]
        present = probe(f"reserved-final-{index:03d}", report)
        _require(present is (index < count - 1), "invalid_reserved_footer_sequence")
        while present:
            _require(consumed_waits < waits, "missing_reserved_wait_probe")
            present = probe(f"reserved-wait-{consumed_waits:03d}", report)
            consumed_waits += 1
    _require(consumed_waits == waits, "unexpected_reserved_wait_probe")
    pre = evidence.json(directory / "pre-reservation-settled.json")
    _require(isinstance(pre, dict), "invalid_preflight_settled")
    cycles = pre.get("read_only_cycles")
    _require(type(cycles) is int and 1 <= cycles <= 301, "invalid_preflight_cycles")
    _require(
        pre.get("last_footer_probe") == f"footer-probes/settle-final-{cycles - 1:03d}.json"
        and pre.get("compared_capture_sha256") == screen_identity(before)
        and pre.get("choice_sha256") == intent.get("choice_sha256")
        and captures.get(f"pre-reservation-observations/{cycles - 1:03d}") == before,
        "invalid_preflight_capture_binding",
    )
    _typed_fields(
        pre,
        {
            "schema_version": 1,
            "scope": "bounded_read_only_fresh_no_save_precondition",
            "read_only_cycles": cycles,
            "compared_capture_sha256": screen_identity(before),
            "choice_sha256": intent.get("choice_sha256"),
            "last_footer_probe": f"footer-probes/settle-final-{cycles - 1:03d}.json",
            "notice_present": False,
            "save_enabled": True,
            "reservation_created": False,
            "save_click_dispatched": False,
            "fresh_save_acknowledgement_verified": False,
            "later_notice_excluded": False,
            "task_completed": False,
        },
        "invalid_preflight_settled",
    )
    known_pre_probes = {"footer-probes/before-reservation.json"} | {
        f"footer-probes/settle-wait-{index:03d}.json" for index in range(1, cycles)
    }
    pre_captures = {name for name in capture_names if name.startswith("pre-reservation-observations/")}
    _require(all(int(name.split("/")[1]) < cycles for name in pre_captures), "invalid_preflight_ordinal")
    known_pre_probes |= {f"footer-probes/settle-final-{Path(name).name}.json" for name in pre_captures}
    _require(pre_probes == known_pre_probes, "unexpected_preflight_footer_files")
    _typed_fields(
        evidence.json(directory / "dispatch.json"),
        {
            "kind": "CLICK",
            "visible_label": "Save",
            "max_clicks": 1,
        },
        "invalid_reserved_notice_dispatch",
    )
    for name in sorted(files):
        evidence.read(directory / name)
    evidence.closed_trees[directory] = tree
    evidence.unchanged()


def _stellar(report):
    return map_stellar_capture(
        report,
        capture_sha256=screen_identity(report),
        allow_color_selection=True,
        allow_main_sequence_fields=True,
    )


def _planet(report):
    return map_planet_capture(report, capture_sha256=screen_identity(report))


def _same_star(mapping, star):
    _require(mapping["star_name"].casefold() == star.casefold(), "star_mismatch")


def _hash_provenance(provenance, names):
    _require(
        all(
            isinstance(provenance.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", provenance[key])
            for key in names
        ),
        "missing_model_provenance",
    )
    _require(
        type(provenance.get("optimizer_updates")) is int and provenance["optimizer_updates"] == 0,
        "unexpected_browser_optimizer_updates",
    )


def _events(evidence, directory):
    manifest = evidence.json(directory / "manifest.json")
    raw = evidence.read(directory / "events.jsonl")
    _require(hashlib.sha256(raw).hexdigest() == manifest.get("events_sha256"), "event_hash_mismatch")
    events = [RuntimeEvent.model_validate_json(line) for line in raw.splitlines()]
    _require(2 <= len(events) <= 2000 and events[0].run_id is not None, "invalid_event_stream")
    _require(
        all(
            e.sequence == i and e.run_id == events[0].run_id and e.event != "error"
            for i, e in enumerate(events)
        ),
        "invalid_event_sequence_or_failure",
    )
    _require(
        events[-1].event == "episode_summary"
        and all(
            events[-1].payload.get(key) == value for key, value in manifest.items() if key != "events_sha256"
        ),
        "manifest_summary_mismatch",
    )
    return manifest, events


def _values_match(mapping, bundle, *, color=True):
    _same_star(mapping, bundle["star"])
    values = mapping["observation"]["values"]
    fields = values["browser_field_map"]
    _require(values["measurements"] == bundle["measurements"], "stellar_measurements_changed")
    _require(set(fields) == set(bundle["readbacks"]), "applicable_stellar_fields_changed")
    for name, receipt in bundle["readbacks"].items():
        _require(
            fields[name]["current_value"] == receipt["display_value"]
            and fields[name]["unit"] == receipt["unit"]
            and fields[name]["value_known"],
            "stellar_readback_changed",
        )
    _require(
        values["conditional_fields_visible"] == (bundle["class"] == "main_sequence"),
        "stellar_applicability_changed",
    )
    _require(
        (values.get("lifetime_prefix") or {}).get("selected") == bundle["prefix"], "lifetime_unit_changed"
    )
    if color:
        _require(values["color"]["selected"] == bundle["color"], "stellar_color_changed")


def _load_sources(evidence, *, numeric_dir, color_dir, class_dir, choice_dir, save_dir):
    numeric_dir, color_dir, class_dir, choice_dir, save_dir = [
        evidence.clean(Path(p)) for p in (numeric_dir, color_dir, class_dir, choice_dir, save_dir)
    ]
    numeric, events = _events(evidence, numeric_dir)
    _hash_provenance(numeric["provenance"], ("checkpoint_sha256", "graph_hash", "knowledge_pack_hash"))
    initial = _stellar(evidence.capture(numeric_dir / "capture"))
    star = initial["star_name"]
    selected = numeric["provenance"]["selected_class"]
    prefix = numeric["provenance"]["lifetime_prefix"]
    required = set(UNITS) if selected == "main_sequence" else {"distance", "luminosity", "temperature"}
    _require(
        selected in CLASSES
        and numeric.get("full_stellar_numeric_transport_verified") is True
        and numeric.get("outcome") == "full_stellar_numeric_transport_verified"
        and numeric.get("learned_policy") is True
        and numeric.get("checkpoint_unchanged") is True
        and type(numeric.get("optimizer_updates")) is int
        and numeric["optimizer_updates"] == 0
        and numeric.get("write_attempts") == len(required)
        and len(numeric.get("verified_fields", [])) == len(required)
        and set(numeric.get("verified_fields", [])) == required
        and set(numeric.get("numeric_readbacks", {})) == required,
        "incomplete_learned_numeric_receipt",
    )
    _require(
        (selected == "main_sequence" and prefix in {"ka", "Ma", "Ga", "Ta"})
        or (selected != "main_sequence" and prefix is None),
        "invalid_lifetime_unit",
    )
    _require(
        any(
            e.event == "action_proposed" and e.payload.get("action_source") == "frozen_lifetime_checkpoint"
            for e in events
        ),
        "missing_learned_numeric_proposals",
    )
    copies = [
        e.payload
        for e in events
        if e.event == "action_result" and e.payload.get("numeric_copy_verified") is True
    ]
    _require(len(copies) == len(required), "incomplete_numeric_copy_events")
    for name, receipt in numeric["numeric_readbacks"].items():
        unit = prefix if name == "lifetime" else UNITS[name]
        formatting = committed_display(receipt["exact_copied"], receipt["display_value"])
        _require(
            receipt["unit"] == unit
            and receipt.get("exact_input_verified") is True
            and receipt.get("commit_key") == "Tab"
            and all(receipt.get(k) == v for k, v in formatting.items()),
            "invalid_numeric_copy_receipt",
        )
        matching = [p for p in copies if p.get("numeric_readback") == receipt]
        _require(len(matching) == 1, "numeric_copy_event_mismatch")
        copy = matching[0]
        field = copy["observation"]["values"]["browser_field_map"][name]
        _require(
            copy["observation"]["values"]["star_name"].casefold() == star.casefold()
            and copy["action"]["kind"] == "TYPE"
            and copy["action"]["value"] == receipt["exact_copied"]
            and field["current_value"] == receipt["display_value"]
            and field["unit"] == unit,
            "numeric_copy_readback_mismatch",
        )
    color, color_events = _events(evidence, color_dir)
    _hash_provenance(color["provenance"], ("color_checkpoint_sha256", "color_reference_hash"))
    _require(
        any(
            e.event == "action_proposed" and e.payload.get("action_source") == "checkpoint"
            for e in color_events
        ),
        "missing_learned_color_proposal",
    )
    _require(
        color.get("color_transport_verified") is True
        and color.get("outcome") == "color_transport_verified"
        and color.get("write_attempts") == 1
        and color["receipt"].get("readback_verified") is True
        and color["provenance"].get("color_gate_passed") is True
        and type(color["provenance"].get("optimizer_updates")) is int
        and color["provenance"]["optimizer_updates"] == 0,
        "incomplete_learned_color_receipt",
    )
    confirmed_colors = [
        e.payload
        for e in color_events
        if e.event == "action_result" and e.payload.get("color_transport_verified") is True
    ]
    _require(
        len(confirmed_colors) == 1 and confirmed_colors[0]["receipt"] == color["receipt"],
        "color_event_mismatch",
    )
    bundle = {
        "star": star,
        "class": selected,
        "prefix": prefix,
        "readbacks": numeric["numeric_readbacks"],
        "color": color["receipt"]["selected_color"],
        "measurements": initial["observation"]["values"]["measurements"],
    }
    _values_match(_stellar(evidence.capture(color_dir / "capture")), bundle, color=False)
    _values_match(
        {
            "star_name": confirmed_colors[0]["observation"]["values"]["star_name"],
            "observation": confirmed_colors[0]["observation"],
        },
        bundle,
    )
    # One shared source contract covers legacy Main receipts and all explicitly
    # supplied native classes. It validates only visible applicable fields; it
    # never claims hidden non-main conditional answers are blank.
    class_source = validate_star_class_source(evidence.history, class_dir, star, selected)
    _require(class_source["measurements"] == bundle["measurements"], "class_measurements_changed")
    for name, expected_hash in class_source["source_hashes"].items():
        path = evidence.history / name
        evidence.clean(path.parent)
        _require(hashlib.sha256(evidence.read(path)).hexdigest() == expected_hash, "class_source_changed")
    bundle["class_rendering_sha256"] = class_source["class_rendering_sha256"]
    save = evidence.json(save_dir / "confirmed.json")
    from .browser_autosave import ACKNOWLEDGEMENT_SOURCE
    from .browser_autosave import MODE as AUTOSAVE_MODE

    if save.get("mode") == AUTOSAVE_MODE:
        from .browser_no_planet_readback import load_no_planet_readback

        bundle.update(load_no_planet_readback(evidence, save_dir, choice_dir=choice_dir, expected_star=star))
        bundle.update(
            save_strategy="autosave",
            save_click_delivered=False,
            save_resumed_same_intent=False,
            save_continuation_stopped_predispatch=False,
            acknowledgement_source=ACKNOWLEDGEMENT_SOURCE,
            numeric_provenance=numeric["provenance"],
            color_provenance=color["provenance"],
        )
        evidence.unchanged()
        return bundle
    reconciled = save.get("mode") == "no_planet_predispatch_read_only_reconciliation"
    resumed = save.get("resumed_same_intent") is True
    continuation = None
    if reconciled or resumed:
        failed_dir = evidence.path(evidence.history / save["source_output"])
        if reconciled:
            original_intent, _, _, _, hashes, continuation = _reconciliation_source(
                failed_dir, evidence.history
            )
            _require(
                save.get("continuation_disposition") == continuation,
                "reconciliation_continuation_disposition_mismatch",
            )
            if continuation is not None:
                recorded = save["continuation_disposition"]
                _require(
                    recorded.get("save_click_dispatched") is False
                    and recorded.get("reservation_retained") is True
                    and recorded.get("further_continuation_allowed") is False,
                    "reconciliation_continuation_disposition_mismatch",
                )
                for name, checksum in continuation["source_sha256"].items():
                    _require(
                        hashlib.sha256(evidence.read(evidence.history / name)).hexdigest() == checksum,
                        "reconciliation_continuation_hash_mismatch",
                    )
        else:
            original_intent, _, _, _, hashes = _predispatch_source(failed_dir, evidence.history)
        _require(save.get("source_sha256") == hashes, "reconciliation_source_hash_mismatch")
        for name, checksum in hashes.items():
            _require(
                hashlib.sha256(evidence.read(evidence.history / name)).hexdigest() == checksum,
                "reconciliation_source_hash_mismatch",
            )
    else:
        failed_dir = save_dir
        original_intent = evidence.json(save_dir / "reserved.json")
    if reconciled:
        intent = original_intent
        disposition = evidence.json(save_dir / "disposition.json")
        _require(
            all(save.get(k) == v for k, v in disposition.items())
            and save.get("outcome") == "current_save_acknowledgement_without_explicit_click"
            and save.get("save_acknowledgement_verified") is True
            and save.get("save_click_delivered") is False
            and save.get("source_save_click_dispatched") is False
            and save.get("site_autosave_trigger_verified") is False
            and save.get("reservation_retained") is True
            and save.get("blocked_before_explicit_save") is True
            and type(save.get("browser_actions")) is int
            and save["browser_actions"] == 0,
            "unconfirmed_autosave_receipt",
        )
        expected_ack = {
            "visible_text": "Data saved",
            "source": "current_fully_exposed_footer_no_explicit_save_dispatch",
            "trigger_verified": False,
            "browser_actions": 0,
        }
    else:
        intent = evidence.json(save_dir / "reserved.json")
        if resumed:
            _require(
                save_dir == failed_dir / "resume"
                and evidence.json(failed_dir / "resume-reserved.json") == intent
                and all(intent.get(k) == v for k, v in original_intent.items() if k != "output")
                and intent.get("source_save_click_dispatched") is False
                and intent.get("reservation_retained") is True
                and intent.get("further_continuation_allowed") is False
                and type(intent.get("max_total_save_clicks")) is int
                and intent["max_total_save_clicks"] == 1
                and evidence.json(save_dir / "stale-notice-cleared.json")
                == {"data_saved_notice_present": False}
                and evidence.json(save_dir / "dispatch.json")
                == {
                    "kind": "CLICK",
                    "visible_label": "Save",
                    "max_clicks": 1,
                    "resumed_same_intent": True,
                    "source_output": intent["source_output"],
                    "source_sha256": hashes,
                },
                "unconfirmed_same_intent_continuation",
            )
        _require(
            save.get("mode") == SAVE_MODE
            and all(save.get(k) == v for k, v in intent.items())
            and save.get("save_click_delivered") is True
            and save.get("notice_was_already_present") is False,
            "unconfirmed_save_receipt",
        )
        expected_ack = {
            "visible_text": "Data saved",
            "source": "fully_exposed_footer_text",
            "notice_was_already_present": False,
        }
    _require(
        save.get("star", "").casefold() == star.casefold()
        and save.get("choice_path")
        == str((choice_dir / "confirmed.json").resolve().relative_to(evidence.history))
        and intent.get("output")
        == str((failed_dir if reconciled else save_dir).relative_to(evidence.history))
        and all(
            save.get(k) is True
            for k in ("data_saved_notice_observed", "answers_unchanged", "duration_and_class_paint_unchanged")
        )
        and intent.get("has_planet") == "No"
        and intent.get("kind") == "CLICK"
        and intent.get("visible_label") == "Save"
        and type(intent.get("max_save_clicks")) is int
        and intent["max_save_clicks"] == 1
        and all(
            save.get(k) is False
            for k in (
                "task_completed",
                "automatic_retry",
                "correctness_verified",
                "cross_session_persistence_verified",
                "course_completion_verified",
                "assessed",
                "score_updated",
                "submitted",
                "hidden_values_inspected",
                "hidden_values_blank_verified",
            )
        ),
        "unconfirmed_save_receipt",
    )
    key = hashlib.sha256(star.casefold().encode()).hexdigest()
    _require(
        evidence.json(evidence.history / "no-planet-save-reservations" / f"{key}.json") == original_intent,
        "save_reservation_mismatch",
    )
    _require(evidence.json(save_dir / "acknowledgement.json") == expected_ack, "save_acknowledgement_missing")
    for name in (
        "numeric_writes",
        "class_writes",
        "habitability_writes",
        "na_writes",
        "assessment_clicks",
        "score_transfer_clicks",
        "submission_clicks",
    ):
        _require(type(intent.get(name)) is int and intent[name] == 0, "unexpected_save_side_effect")
    choice, choice_capture, choice_mapping = _load_choice(
        choice_dir / "confirmed.json", save["choice_sha256"], evidence.history
    )
    _require(
        choice["star"].casefold() == star.casefold() and save.get("policy") == choice["policy"],
        "choice_star_or_policy_mismatch",
    )
    for path in (
        choice_dir / "confirmed.json",
        choice_dir / "reserved.json",
        choice_dir / "preselect.json",
        evidence.history / "planet-window-choice-reservations" / f"{key}.json",
    ):
        evidence.read(path)
    evidence.capture(choice_dir / "after")
    for label, source in (
        ("saved_evidence", evidence.history / choice["source"]),
        ("fresh_evidence", choice_dir / "fresh-progress/report.json"),
        ("preselect_evidence", choice_dir / "preselect-progress/report.json"),
    ):
        evidence.read(source)
        evidence.read(source.parent / "chart.png")
        if "policy_sha256" in choice["saved_evidence"]:
            _require(
                hashlib.sha256(evidence.read(source.parent / "policy.json")).hexdigest()
                == choice[label].get("policy_sha256"),
                "choice_policy_source_changed",
            )
    before, after = evidence.capture(save_dir / "before"), evidence.capture(save_dir / "after")
    _reserved_notice_options_match(save, intent, reconciled=reconciled, resumed=resumed)
    if resumed:
        _require(save.get("resume_before_sha256") == screen_identity(before), "resumed_before_hash_mismatch")
    before_map, after_map = _planet(before), _planet(after)
    if not reconciled and not resumed:
        _reserved_notice_evidence(evidence, save_dir, intent, before, before_map)
    _same_star(after_map, star)
    _blank_no(before_map)
    _blank_no(after_map)
    original = evidence.capture(failed_dir / "before") if reconciled or resumed else before
    if reconciled:
        equivalence = no_planet_visit_equivalence(original, _planet(original), before, before_map)
        _require(
            json.dumps(save.get("historical_view_equivalence"), sort_keys=True)
            == json.dumps(equivalence, sort_keys=True),
            "reconciliation_view_equivalence_mismatch",
        )
    _require(
        screen_identity(original) == intent["before_sha256"]
        and planet_projection(before, before_map) == planet_projection(after, after_map)
        and planet_projection(
            original if reconciled else before, _planet(original) if reconciled else before_map
        )
        == planet_projection(choice_capture, choice_mapping),
        "saved_visible_state_changed",
    )
    bundle.update(
        choice=choice,
        save_capture=after,
        save_mapping=after_map,
        save_click_delivered=not reconciled,
        save_resumed_same_intent=resumed,
        save_continuation_stopped_predispatch=continuation is not None,
        acknowledgement_source="visible_footer_no_explicit_save_dispatch"
        if reconciled
        else "explicit_save_fresh_visible_footer",
        numeric_provenance=numeric["provenance"],
        color_provenance=color["provenance"],
    )
    evidence.unchanged()
    return bundle


def _read_stellar(page, config, directory, bundle):
    journal = NumericJournal(directory, provenance={"scope": "read_only_no_planet_workflow"})
    session = None
    try:
        session = FullStellarSession(
            page, config, journal, selected_class=bundle["class"], lifetime_prefix=bundle["prefix"]
        )
        session.start()
        report, mapping, handles, frame = session._current()
        _require(all(rendered_control(handle) for handle in handles.values()), "stellar_fields_occluded")
        menus = [m for m in frame.get_by_role("combobox").all() if m.is_visible()]
        expected = [bundle["color"]] + ([bundle["prefix"]] if bundle["prefix"] else [])
        _require(
            len(menus) == len(expected)
            and all(rendered_control(m) and m.input_value() == v for m, v in zip(menus, expected)),
            "stellar_menus_changed",
        )
        choices, _ = read_class_choices(frame)
        _require(digest(choices) == bundle["class_rendering_sha256"], "painted_stellar_class_changed")
        _values_match(mapping, bundle)
        session._current()
        save_probe(report, directory / "verified")
        return report
    finally:
        if session is not None:
            session.stopped = True
            page.remove_listener("dialog", session._dialog)
        journal.stream.close()


def _read_planet(page, config, directory, bundle):
    session = PlanetNumericSession(page, config, directory, max_seconds=60, _allow_no_planet=True)
    try:
        report, mapping, choices, _ = session.current()
        _same_star(mapping, bundle["star"])
        _blank_no(mapping)
        _require(
            choices["selected"] == bundle["choice"]["painted_class_after"]
            and planet_projection(report, mapping)
            == planet_projection(bundle["save_capture"], bundle["save_mapping"]),
            "current_saved_planet_state_changed",
        )
        session.current()
        save_probe(report, directory / "verified")
        return report
    finally:
        session.close()


def verify_no_planet_workflow(
    page, config, output, *, run_history, numeric_dir, color_dir, class_dir, choice_dir, save_dir
):
    """Verify owned receipts then current same-star Stellar and Planet readbacks."""
    evidence = _Evidence(run_history)
    directory = evidence.path(output)
    directory.mkdir(parents=True, exist_ok=False)
    navigation = []
    try:
        bundle = _load_sources(
            evidence,
            numeric_dir=numeric_dir,
            color_dir=color_dir,
            class_dir=class_dir,
            choice_dir=choice_dir,
            save_dir=save_dir,
        )
        boundary = config.model_copy(deep=True)
        captures = {}
        for section, reader in (("stellar", _read_stellar), ("planet", _read_planet)):
            evidence.unchanged()
            result = navigate_project(
                page, boundary, directory / ("to-" + section), section, expected_star=bundle["star"]
            )
            _require(
                result.get("same_star_verified") is True and result.get("destination_verified") is True,
                "navigation_not_verified",
            )
            navigation.append(result)
            for rule in boundary.frames:
                if rule.url == SIMULATION_URL:
                    rule.required_text = result["suggested_required_text"]
            captures[section] = reader(page, boundary, directory / (section + "-readback"), bundle)
        evidence.unchanged()
        receipt = {
            "schema_version": 1,
            "mode": "no_planet_visible_workflow_readback",
            "star": bundle["star"],
            "authority": "visible_workflow_readback",
            "task_completed": True,
            "numeric_provenance": bundle["numeric_provenance"],
            "color_provenance": bundle["color_provenance"],
            "stellar_fields": bundle["readbacks"],
            "color": bundle["color"],
            "classification": bundle["class"],
            "classification_provenance": "reference_prediction",
            "learned_classification": False,
            "planet": {
                "outcome": "no_planet",
                "provenance": "reference_prediction",
                "transport_verified": True,
                "scientific_verified": False,
                "policy_label": bundle["choice"]["policy"]["version"],
                "observation_limit_days": 5000,
            },
            "habitability": {
                "outcome": "not_applicable",
                "applicability_reason": "no_planet",
                "provenance": "reference_prediction",
                "transport_verified": False,
                "branch_applicability_verified": True,
                "scientific_verified": False,
                "source": "explicit_branch_applicability_not_field_write",
            },
            "visible_blank_raw_fields": list(RAW_FIELDS),
            "conditionally_absent_derived_fields": list(CONDITIONAL_FIELDS),
            "save_acknowledgement_verified": bundle.get("save_strategy") != "autosave",
            "navigation_clicks": len(navigation),
            "source_save_click_delivered": bundle["save_click_delivered"],
            "source_save_resumed_same_intent": bundle["save_resumed_same_intent"],
            "source_save_continuation_stopped_predispatch": bundle["save_continuation_stopped_predispatch"],
            "save_acknowledgement_source": bundle["acknowledgement_source"],
            "answer_writes": 0,
            "habitability_writes": 0,
            "na_writes": 0,
            "save_clicks": 0,
            "assessment_clicks": 0,
            "score_transfer_clicks": 0,
            "submission_clicks": 0,
            "hidden_values_inspected": False,
            "hidden_values_blank_verified": False,
            "scientific_verified": False,
            "correctness_verified": False,
            "browser_acceptance_passed": False,
            "course_completion_verified": False,
            "project_completed": False,
            "submitted": False,
            "training_label": False,
            "cross_session_persistence_verified": False,
            "source_sha256": dict(evidence.hashes),
            "current_screen_sha256": {key: screen_identity(value) for key, value in captures.items()},
        }
        if bundle.get("save_strategy") == "autosave":
            from .browser_autosave import autosave_workflow_flags

            receipt.update(autosave_workflow_flags())
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "no_planet_workflow_evidence_or_read_failed",
                "navigation_completed": len(navigation),
                "answer_writes": 0,
                "task_completed": False,
                "automatic_retry": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("no_planet_workflow_evidence_or_read_failed") from None
