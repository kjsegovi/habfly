"""Read-only positive-planet workflow verification, not scientific acceptance.

Version one supports a supplied main-sequence star and a visibly selected gas
or ice giant. Terrestrial completion is deliberately unsupported until the gas,
habitability decision and final Save evidence have a complete native contract.
N/A is an explicit applicability rule, never a hidden/blank field inference.
Only ordinary Stellar/Planet tab navigation is performed; no answers or Save
are dispatched. Explicit approximate raster or sampled-tooltip references retain
their distinct limitations; neither supplies scientific period/depth evidence.
"""

import hashlib
import re

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_autosave import (
    ACKNOWLEDGEMENT_SOURCE as AUTOSAVE_ACKNOWLEDGEMENT_SOURCE,
)
from .browser_autosave import (
    MODE as AUTOSAVE_MODE,
)
from .browser_autosave import (
    autosave_workflow_flags,
    validate_autosave_receipt,
)
from .browser_full_stellar import UNITS
from .browser_no_planet_workflow import (
    _events,
    _Evidence,
    _hash_provenance,
    _planet,
    _read_stellar,
    _same_star,
    _stellar,
    _values_match,
)
from .browser_numeric import committed_display, screen_identity
from .browser_planet_numeric import (
    ANSWER_UNITS,
    PlanetNumericSession,
    planet_projection,
    validate_orbit_readout_change,
)
from .browser_probe import save_probe
from .browser_project_navigation import navigate_project
from .browser_raster_planet_evidence import (
    MODE,  # noqa: F401 - legacy module constant
    RAW,
    _action_source,
    _flags_for,
    _fresh_evidence,
    _Owned,
    _presence,
    _reload,
    _reservation,
)
from .browser_save_settlement import OPTION_KEYS, typed_equal, validate_reserved_phase
from .browser_stellar import SIMULATION_URL
from .contracts import RuntimeEvent
from .supplied_browser_modes import (
    DERIVED_SUPPLIED_MODE,
    NON_MAIN_CLASSES,
    positive_mode,
    source_options,
)

DERIVED = {"orbital_radius", "planet_mass", "planet_radius", "planet_density"}


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("positive_planet_workflow_" + reason)


def _identity_matches(left, right):
    """AX/native names can be Title Case while rendered chart labels are uppercase.

    Normalize identity comparisons only, never stored text, receipts or hashes.
    Whitespace differences and missing/non-string names are not equivalent.
    """
    return (
        isinstance(left, str)
        and isinstance(right, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", left) is not None
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", right) is not None
        and left.casefold() == right.casefold()
    )


def _flags(value, *, true=(), false=(), zero=()):
    _require(
        all(value.get(k) is True for k in true)
        and all(value.get(k) is False for k in false)
        and all(type(value.get(k)) is int and value[k] == 0 for k in zero),
        "invalid_evidence_flags",
    )


class _RasterEvidence(_Owned):
    """Use the shared reference validator while retaining every consumed hash."""

    def __init__(self, book):
        super().__init__(book.history)
        self.book = book

    def read(self, path, expected=None, limit=2_000_000):
        raw = super().read(path, expected, limit)
        _require(raw == self.book.read(path), "evidence_changed")
        return raw

    def clean(self, directory):
        return self.book.clean(super().clean(directory))


def _measurement_mode(evidence):
    """Dispatch only a shared transport mode, after upstream source validation."""
    mode = _flags_for(evidence)["provenance"]
    if mode == "approximate_reference_two_visible_tooltips_v1":
        from .browser_raster_planet_evidence import _two_tooltip_metadata

        _two_tooltip_metadata(evidence.get("tooltip_measurements", {}))
    return mode


def _stellar_sources(book, numeric_dir, color_dir, class_dir, *, supplied_inputs=False):
    """Validate existing frozen Stellar streams without requiring a No receipt."""
    source_options(supplied_inputs)
    if supplied_inputs:
        from .browser_stellar_sources import load_stellar_sources

        source = load_stellar_sources(book, numeric_dir=numeric_dir, color_dir=color_dir, class_dir=class_dir)
        _require(source["bundle"]["class"] in NON_MAIN_CLASSES, "supplied_non_main_class_required")
        return {
            **source["bundle"],
            "numeric_provenance": source["numeric_provenance"],
            "color_provenance": source["color_provenance"],
        }
    numeric_dir, color_dir, class_dir = [book.clean(p) for p in (numeric_dir, color_dir, class_dir)]
    numeric, events = _events(book, numeric_dir)
    provenance = numeric["provenance"]
    _hash_provenance(provenance, ("checkpoint_sha256", "graph_hash", "knowledge_pack_hash"))
    initial = _stellar(book.capture(numeric_dir / "capture"))
    star = initial["star_name"]
    _flags(
        numeric,
        true=("full_stellar_numeric_transport_verified", "learned_policy", "checkpoint_unchanged"),
        zero=("optimizer_updates",),
    )
    readbacks = numeric["numeric_readbacks"]
    _require(
        provenance.get("selected_class") == "main_sequence"
        and provenance.get("lifetime_prefix") in {"ka", "Ma", "Ga", "Ta"}
        and numeric.get("outcome") == "full_stellar_numeric_transport_verified"
        and type(numeric.get("write_attempts")) is int
        and numeric["write_attempts"] == 6
        and set(numeric.get("verified_fields", [])) == set(UNITS)
        and len(numeric["verified_fields"]) == 6
        and set(readbacks) == set(UNITS)
        and any(
            e.event == "action_proposed" and e.payload.get("action_source") == "frozen_lifetime_checkpoint"
            for e in events
        ),
        "unsupported_stellar_numeric_receipt",
    )
    copies = [
        e.payload
        for e in events
        if e.event == "action_result" and e.payload.get("numeric_copy_verified") is True
    ]
    _require(len(copies) == 6, "stellar_copy_count")
    for name, item in readbacks.items():
        unit = provenance["lifetime_prefix"] if name == "lifetime" else UNITS[name]
        _require(
            item.get("exact_input_verified") is True
            and item.get("commit_key") == "Tab"
            and item.get("unit") == unit,
            "stellar_copy_contract",
        )
        _require(
            all(
                item.get(k) == v
                for k, v in committed_display(item["exact_copied"], item["display_value"]).items()
            ),
            "stellar_copy_display",
        )
        matching = [
            p
            for p in copies
            if p.get("numeric_readback") == item
            and p.get("action", {}).get("kind") == "TYPE"
            and p["action"].get("value") == item["exact_copied"]
            and p["observation"]["values"]["star_name"].casefold() == star.casefold()
            and p["observation"]["values"]["browser_field_map"][name]["current_value"]
            == item["display_value"]
            and p["observation"]["values"]["browser_field_map"][name]["unit"] == unit
        ]
        _require(bool(matching), "missing_stellar_copy_event")
    color, color_events = _events(book, color_dir)
    cp = color["provenance"]
    _hash_provenance(cp, ("color_checkpoint_sha256", "color_reference_hash"))
    _flags(color, true=("color_transport_verified",))
    _flags(cp, true=("color_gate_passed",), zero=("optimizer_updates",))
    confirmed = [
        e.payload
        for e in color_events
        if e.event == "action_result" and e.payload.get("color_transport_verified") is True
    ]
    _require(
        color.get("outcome") == "color_transport_verified"
        and type(color.get("write_attempts")) is int
        and color["write_attempts"] == 1
        and color["receipt"].get("readback_verified") is True
        and len(confirmed) == 1
        and confirmed[0].get("receipt") == color["receipt"]
        and any(
            e.event == "action_proposed" and e.payload.get("action_source") == "checkpoint"
            for e in color_events
        ),
        "unsupported_stellar_color_receipt",
    )
    bundle = {
        "star": star,
        "class": "main_sequence",
        "prefix": provenance["lifetime_prefix"],
        "readbacks": readbacks,
        "color": color["receipt"]["selected_color"],
        "measurements": initial["observation"]["values"]["measurements"],
        "numeric_provenance": provenance,
        "color_provenance": cp,
    }
    _values_match(_stellar(book.capture(color_dir / "capture")), bundle, color=False)
    color_observation = confirmed[0]["observation"]
    _values_match(
        {"star_name": color_observation["values"]["star_name"], "observation": color_observation}, bundle
    )
    choice, scope = book.json(class_dir / "confirmed.json"), book.json(class_dir / "scope.json")
    _flags(
        choice,
        true=("readback_verified",),
        false=(
            "learned_classification",
            "correctness_verified",
            "task_completed",
            "intermediate_is_training_label",
        ),
        zero=("numeric_writes",),
    )
    _flags(
        scope,
        false=("learned_classification", "automatic_retry", "intermediate_is_training_label"),
        zero=("numeric_writes",),
    )
    _require(
        choice.get("star", "").casefold() == star.casefold()
        and scope.get("star", "").casefold() == star.casefold()
        and choice.get("selected_class") == scope.get("intended_class") == "main_sequence"
        and choice.get("action_source") == "explicit_fresh_star_class_setup"
        and type(choice.get("class_clicks")) is int
        and choice["class_clicks"] in {1, 2}
        and scope.get("max_class_clicks") == choice["class_clicks"],
        "unsupported_stellar_class_receipt",
    )
    fresh = book.clean(scope["fresh_star"])
    receipt = book.json(fresh / "confirmed.json")
    _flags(
        receipt,
        true=("fresh_blank_numeric_answers_verified",),
        false=("class_selection_verified",),
        zero=("answer_writes",),
    )
    _require(
        receipt.get("star", "").casefold() == star.casefold()
        and receipt.get("action_source") == "deterministic_navigation"
        and hashlib.sha256(book.read(fresh / "stellar/observation.json")).hexdigest()
        == scope["source_capture_sha256"],
        "stellar_class_source_mismatch",
    )
    for directory in (fresh / "stellar", class_dir / "before"):
        mapping = _stellar(book.capture(directory))
        _same_star(mapping, star)
        values = mapping["observation"]["values"]
        _require(
            values["measurements"] == bundle["measurements"]
            and values["color"]["selected"] is None
            and all(
                f["value_known"] and f["current_value"] == "" for f in values["browser_field_map"].values()
            ),
            "stellar_class_source_not_blank",
        )
    for name in ["intermediate", "after"] if choice["class_clicks"] == 2 else ["after"]:
        mapping = _stellar(book.capture(class_dir / name))
        _same_star(mapping, star)
        _require(
            mapping["observation"]["values"]["measurements"] == bundle["measurements"],
            "stellar_measurements_changed",
        )
    class_events = [book.json(p) for p in sorted(class_dir.glob("event-*.json"))]
    expected = ["white_dwarf", "main_sequence"] if choice["class_clicks"] == 2 else ["main_sequence"]
    _require(
        [e.get("event") for e in class_events] == ["action_proposed", "action_result"] * len(expected),
        "missing_stellar_class_events",
    )
    reservation = book.json(fresh / "class-selection-reserved.json")
    _require(
        reservation.get("star", "").casefold() == star.casefold()
        and reservation.get("source_capture_sha256") == scope["source_capture_sha256"]
        and reservation.get("selected_class") == expected[0]
        and reservation.get("action_source") == "reference_diagnostic"
        and reservation.get("max_clicks") == 1
        and reservation.get("automatic_retry") is False,
        "stellar_class_reservation",
    )
    for index, value in enumerate(expected):
        proposed, result = [class_events[2 * index + i]["payload"] for i in (0, 1)]
        _require(
            proposed.get("kind") == "SELECT"
            and proposed.get("target") == "stellar_class"
            and proposed.get("value") == value
            and proposed.get("action_source") == "reference_diagnostic"
            and result.get("selected_class") == value
            and result.get("selection_source") == "reference_diagnostic",
            "stellar_class_event_mismatch",
        )
        _flags(result, true=("readback_verified",), false=("correctness_verified", "task_completed"))
    bundle["class_rendering_sha256"] = class_events[-1]["payload"]["rendering_sha256"]
    return bundle


def _copy_chain(book, directory, fields, *, source, initial, expected=None):
    directory = book.clean(directory)
    _require(not list(directory.glob("*-stopped.json")), "uncertain_native_copy")
    confirmed = sorted(directory.glob("copy-*-confirmed.json"))
    _require(len(confirmed) == len(fields), "native_copy_count")
    previous = initial
    receipts = {}
    for index, path in enumerate(confirmed, 1):
        prefix = f"copy-{index:02d}"
        _require(path.name == prefix + "-confirmed.json", "native_copy_sequence")
        intent, receipt = book.json(directory / (prefix + "-reserved.json")), book.json(path)
        _flags(receipt, true=("readback_verified",), false=("correctness_verified", "task_completed"))
        name = intent.get("destination")
        _require(
            name in fields
            and name not in receipts
            and intent.get("kind") == "TYPE"
            and intent.get("action_source") == source
            and intent.get("unit") == ANSWER_UNITS[name]
            and intent.get("task_completed") is False,
            "native_copy_intent",
        )
        before, after = (
            book.capture(directory / (prefix + "-before")),
            book.capture(directory / (prefix + "-after")),
        )
        old, new = _planet(before), _planet(after)
        _same_star(old, _planet(previous)["star_name"])
        _same_star(new, old["star_name"])
        before_fields, after_fields = [m["observation"]["values"]["browser_field_map"] for m in (old, new)]
        _require(
            before_fields[name]["current_value"] == ""
            and before_fields[name]["capture_target_id"] == intent.get("target")
            and planet_projection(previous, _planet(previous)) == planet_projection(before, old)
            and planet_projection(before, old, name) == planet_projection(after, new, name),
            "native_copy_capture_mismatch",
        )
        display = committed_display(intent["value"], after_fields[name]["current_value"])
        validate_orbit_readout_change(before, after, name)
        _require(
            receipt
            == {**intent, "display": display, "readback_verified": True, "correctness_verified": False},
            "native_copy_receipt_mismatch",
        )
        if expected:
            _require(
                intent["value"] == expected[name]["value"] and intent["unit"] == expected[name]["unit"],
                "raw_copy_not_exact_reference",
            )
        receipts[name], previous = receipt, after
    return receipts, previous


def _planet_sources(book, raw_dir, derived_dir, planet_class_dir, *, supplied_inputs=False):
    """Validated raw/derived/class evidence, without asserting any Save."""
    source_options(supplied_inputs)
    owner = _RasterEvidence(book)
    raw_dir, derived_dir, planet_class_dir = [book.clean(p) for p in (raw_dir, derived_dir, planet_class_dir)]
    raw = book.json(raw_dir / "report.json")
    _flags(raw, true=("raw_measurement_transport_verified",), false=("saved", "assessed", "submitted"))
    _require(type(raw.get("numeric_writes")) is int and raw["numeric_writes"] == 3, "raw_write_count")
    intent = book.json(raw_dir / "reserved.json")
    presence, initial, _ = _presence(owner, owner.history / raw["presence_path"], raw["presence_sha256"])
    evidence, star = presence["evidence"], presence["star"]
    flags, mode = _flags_for(evidence), _measurement_mode(evidence)
    _require(
        all(type(raw.get(k)) is type(v) and raw[k] == v for k, v in flags.items())
        and type(raw.get("schema_version")) is int
        and raw["schema_version"] == 1
        and raw.get("mode") == mode
        and raw.get("stage") == "inputs"
        and _identity_matches(raw.get("star"), star)
        and raw.get("evidence") == evidence
        and raw.get("output") == owner.relative(raw_dir)
        and type(raw.get("maximum_numeric_writes")) is int
        and raw.get("maximum_numeric_writes") == 3
        and raw.get("destinations") == list(RAW)
        and raw.get("action_source") == _action_source(evidence),
        "unsupported_raw_provenance",
    )
    _require(
        book.json(_reservation(owner, star, "inputs")) == intent
        and raw
        == {
            **intent,
            "raw_measurement_transport_verified": True,
            "numeric_writes": 3,
            "verified_fields": raw["verified_fields"],
            "precopy": raw["precopy"],
            "native_events": raw["native_events"],
            "saved": False,
            "assessed": False,
            "submitted": False,
        },
        "raw_reservation_mismatch",
    )
    _reload(owner, evidence)
    _require(len(raw["precopy"]) == 3, "raw_precopy_count")
    for index, name in enumerate(("fresh", "precopy-1", "precopy-2", "precopy-3")):
        recorded = raw["fresh"] if index == 0 else raw["precopy"][index - 1]
        if index:
            _require(book.json(raw_dir / name / "evidence.json") == recorded, "raw_precopy_record_mismatch")
        _fresh_evidence(owner, raw_dir / name, recorded, evidence)
    raw_receipts, last = _copy_chain(
        book,
        raw_dir / "native-copies",
        set(RAW),
        source="reference_diagnostic",
        initial=initial,
        expected=evidence["measurements"],
    )
    _require(raw_receipts == raw["verified_fields"], "raw_readbacks_changed")
    expected_native_events = []
    for name in RAW:
        receipt = raw_receipts[name]
        expected_native_events.extend(
            {"kind": kind, "payload": payload, "provenance": mode}
            for kind, payload in (
                (
                    "action_proposed",
                    {
                        k: v
                        for k, v in receipt.items()
                        if k not in {"display", "readback_verified", "correctness_verified"}
                    },
                ),
                ("action_result", receipt),
            )
        )
    _require(
        raw["native_events"] == expected_native_events
        and [book.json(p) for p in sorted(raw_dir.glob("native-event-*.json"))] == expected_native_events,
        "raw_native_events_mismatch",
    )
    derived = book.json(derived_dir / "report.json")
    stream = book.read(derived_dir / "events.jsonl")
    _require(hashlib.sha256(stream).hexdigest() == derived.get("events_sha256"), "derived_events_hash")
    events = [RuntimeEvent.model_validate_json(line) for line in stream.splitlines()]
    _require(
        2 <= len(events) <= 2000
        and events[0].event == "hello"
        and events[0].run_id is not None
        and all(
            e.sequence == i and e.run_id == events[0].run_id and e.event != "error"
            for i, e in enumerate(events)
        )
        and events[-1].event == "episode_summary"
        and events[-1].payload == {k: v for k, v in derived.items() if k != "events_sha256"},
        "derived_event_sequence",
    )
    _flags(
        derived,
        true=("checkpoint_unchanged", "planet_transport_verified"),
        false=("task_completed", "browser_acceptance_passed", "saved", "assessment_performed", "submitted"),
        zero=("optimizer_updates",),
    )
    provenance = derived["provenance"]
    _hash_provenance(provenance, ("checkpoint_sha256", "graph_hash", "knowledge_pack_hash"))
    _require(
        derived.get("scope")
        == (DERIVED_SUPPLIED_MODE if supplied_inputs else "four_derived_planet_browser_transport")
        and derived.get("outcome") == "planet_derived_transport_verified"
        and derived.get("write_attempts") == sorted(DERIVED)
        and set(derived["verified_fields"]) == DERIVED
        and (
            provenance.get("supplied_star_class") in NON_MAIN_CLASSES
            if supplied_inputs
            else provenance.get("supplied_star_class") == "main_sequence"
        )
        and provenance.get("classification_source") == "supplied_not_learned"
        and "recovery" not in provenance,
        "unsupported_derived_receipt",
    )
    supplied = None
    if supplied_inputs:
        from .browser_supplied_provenance import validate_supplied_derived_provenance
        from .planet_supplied_stellar_source import matches_mapping

        supplied = validate_supplied_derived_provenance(book, derived_dir, derived)
        _require(
            _identity_matches(supplied["star"], star)
            and supplied["supplied_star_class"] == provenance["supplied_star_class"]
            and book.path(book.history / supplied["receipt"]["source_arguments"]["raw_dir"]) == raw_dir,
            "supplied_planet_source_mismatch",
        )
        _require(matches_mapping(supplied["receipt"], _planet(last)), "supplied_planet_readouts_changed")
    derived_receipts, last = _copy_chain(
        book, derived_dir / "native-copies", DERIVED, source="checkpoint", initial=last
    )
    _require(derived_receipts == derived["verified_fields"], "derived_readbacks_changed")
    for receipt in derived_receipts.values():
        _require(
            sum(e.event == "action_result" and e.payload == receipt for e in events) == 1
            and sum(
                e.event == "action_proposed"
                and e.payload
                == {
                    k: v
                    for k, v in receipt.items()
                    if k not in {"display", "readback_verified", "correctness_verified"}
                }
                for e in events
            )
            == 1,
            "missing_derived_copy_event",
        )
    choice, choice_intent = (
        book.json(planet_class_dir / "confirmed.json"),
        book.json(planet_class_dir / "reserved.json"),
    )
    name = choice.get("value")
    _require(
        name in {"gas_giant", "ice_giant", "terrestrial"}
        and _identity_matches(choice.get("star"), star)
        and choice.get("kind") == "SELECT"
        and choice.get("action_source") == "reference_diagnostic"
        and type(choice.get("max_class_writes")) is int
        and choice.get("max_class_writes") == 1
        and "previous" not in choice
        and choice == {**choice_intent, "readback_verified": True},
        "unsupported_planet_class_receipt",
    )
    _flags(
        choice,
        true=("readback_verified",),
        false=("correctness_verified", "task_completed"),
        zero=("numeric_writes",),
    )
    class_before, class_after = [book.capture(planet_class_dir / p) for p in ("read-guard/initial", "after")]
    for report in (class_before, class_after):
        _require(
            planet_projection(report, _planet(report)) == planet_projection(last, _planet(last)),
            "planet_class_changed_answers",
        )
    return {
        "star": star,
        "planet_class": name,
        "raw": raw,
        "derived": derived,
        "planet_readbacks": {**raw_receipts, **derived_receipts},
        "class_capture": class_after,
        **({"supplied_stellar_inputs": supplied["receipt"]} if supplied_inputs else {}),
    }


def _autosave_sources(book, directory, source_book, source_capture, *, branch, star, same):
    """Rebuild only current visible readback; no persistence or Save authority."""
    directory = book.clean(directory)
    receipt = book.json(directory / "confirmed.json")
    validate_autosave_receipt(
        receipt, branch=branch, star=star, output=str(directory.relative_to(book.history))
    )
    _require(receipt["source_sha256"] == source_book.hashes, "autosave_sources_mismatch")
    for path in source_book.clean_directories:
        book.clean(path)
    for name, sha in source_book.hashes.items():
        _require(hashlib.sha256(book.read(book.history / name)).hexdigest() == sha, "autosave_source_changed")
    book.closed_trees.update(source_book.closed_trees)
    before, after = [book.capture(directory / p) for p in ("read-guard/initial", "after")]
    _require(
        screen_identity(before) == receipt["before_sha256"]
        and screen_identity(after) == receipt["after_sha256"]
        and same(source_capture, before)
        and same(before, after),
        "autosave_capture_mismatch",
    )
    files = {
        "confirmed.json",
        "read-guard/initial/manifest.json",
        "read-guard/initial/observation.json",
        "after/manifest.json",
        "after/observation.json",
    }
    if branch == "positive":
        scope = book.json(directory / "read-guard/scope.json")
        _require(
            typed_equal(
                scope,
                {
                    "scope": "planet_exact_copy_transport",
                    "max_writes": 7,
                    "max_seconds": 60,
                    "star": _planet(before)["star_name"],
                    "task_completed": False,
                    "scientific_choices": "caller_supplied",
                    "retries": False,
                },
            ),
            "autosave_guard_scope_mismatch",
        )
        files.add("read-guard/scope.json")
    tree = book.tree(directory)
    _require(
        {name for name, kind in tree if kind == "file"} == files
        and {name for name, kind in tree if kind == "directory"}
        == {"read-guard", "read-guard/initial", "after"},
        "autosave_unexpected_artifacts",
    )
    book.closed_trees[directory] = tree
    source_book.unchanged()
    book.unchanged()
    return receipt, after


def _positive_sources(book, raw_dir, derived_dir, planet_class_dir, save_dir, *, supplied_inputs=False):
    planet = _planet_sources(book, raw_dir, derived_dir, planet_class_dir, **source_options(supplied_inputs))
    star, class_after = planet["star"], planet["class_capture"]
    save_dir = book.clean(save_dir)
    save = book.json(save_dir / "confirmed.json")
    if save.get("mode") == AUTOSAVE_MODE:
        source_book = _Evidence(book.history)
        source = _planet_sources(
            source_book, raw_dir, derived_dir, planet_class_dir, **source_options(supplied_inputs)
        )
        _, after = _autosave_sources(
            book,
            save_dir,
            source_book,
            source["class_capture"],
            branch="positive",
            star=star,
            same=lambda a, b: planet_projection(a, _planet(a)) == planet_projection(b, _planet(b)),
        )
        return {
            **{k: v for k, v in planet.items() if k != "class_capture"},
            "save_capture": after,
            **autosave_workflow_flags(),
        }
    save_intent = book.json(save_dir / "reserved.json")
    _require(
        save
        == {
            **save_intent,
            "save_click_delivered": True,
            "data_saved_notice_observed": True,
            "notice_was_already_present": False,
            "answers_unchanged": True,
            "cross_session_persistence_verified": False,
            "course_completion_verified": False,
            "assessed": False,
            "score_updated": False,
            "submitted": False,
        },
        "unsupported_save_receipt",
    )
    _require(
        save.get("kind") == "CLICK"
        and save.get("visible_label") == "Save"
        and _identity_matches(save.get("star"), star)
        and type(save.get("max_save_clicks")) is int
        and save.get("max_save_clicks") == 1
        and save.get("selection_source") == "scripted_setup_not_learned",
        "save_intent_mismatch",
    )
    _flags(
        save,
        true=("save_click_delivered", "data_saved_notice_observed", "answers_unchanged"),
        false=(
            "task_completed",
            "automatic_retry",
            "notice_was_already_present",
            "cross_session_persistence_verified",
            "course_completion_verified",
            "assessed",
            "score_updated",
            "submitted",
        ),
        zero=("numeric_writes",),
    )
    _require(
        book.json(save_dir / "acknowledgement.json")
        == {
            "visible_text": "Data saved",
            "source": "fully_exposed_footer_text",
            "notice_was_already_present": False,
        },
        "save_acknowledgement_mismatch",
    )
    before, after = [book.capture(save_dir / p) for p in ("read-guard/initial", "after")]
    _require(
        screen_identity(before) == save["before_sha256"]
        and planet_projection(before, _planet(before)) == planet_projection(class_after, _planet(class_after))
        and planet_projection(after, _planet(after)) == planet_projection(before, _planet(before)),
        "save_capture_mismatch",
    )
    if OPTION_KEYS & set(save_intent):
        _require(
            all(typed_equal(save.get(key), save_intent.get(key)) for key in OPTION_KEYS),
            "save_policy_receipt_mismatch",
        )
        claim = book.json(
            book.history
            / "positive-finalization-reservations"
            / (hashlib.sha256(star.casefold().encode()).hexdigest() + ".json")
        )
        hashes = claim.get("source_sha256")
        _require(isinstance(hashes, dict) and hashes, "save_claim_sources_missing")
        for name, digest in hashes.items():
            _require(
                isinstance(name, str)
                and isinstance(digest, str)
                and re.fullmatch(r"[a-f0-9]{64}", digest)
                and hashlib.sha256(book.read(book.history / name)).hexdigest() == digest,
                "save_claim_sources_changed",
            )
        _require(
            _identity_matches(claim.get("star"), star)
            and typed_equal(
                claim,
                {
                    "schema_version": 1,
                    "mode": "cooperative_positive_planet_finalization",
                    "star": claim["star"],
                    "planet_class": planet["planet_class"],
                    "output": str(save_dir.parent.relative_to(book.history)),
                    "save_output": str(save_dir.relative_to(book.history)),
                    "planet_class_sha256": hashlib.sha256(
                        book.read(book.path(planet_class_dir) / "confirmed.json")
                    ).hexdigest(),
                    "source_sha256": hashes,
                    "native_intent": save_intent,
                    "maximum_save_dispatches": 1,
                    "automatic_retry": False,
                    "task_completed": False,
                },
            ),
            "save_claim_mismatch",
        )
        predispatched = book.capture(save_dir / "pre-dispatch-observation")
        _require(
            planet_projection(predispatched, _planet(predispatched))
            == planet_projection(before, _planet(before))
            and typed_equal(
                book.json(save_dir / "pre-dispatch-footer.json"),
                {
                    "notice_present": False,
                    "save_click_dispatched": False,
                    "compared_capture_sha256": screen_identity(predispatched),
                    "compared_capture_is_simultaneous": False,
                },
            )
            and typed_equal(
                book.json(save_dir / "dispatch.json"),
                {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1},
            ),
            "save_dispatch_evidence_mismatch",
        )
    validate_reserved_phase(
        book,
        save_dir,
        save_intent,
        after=after,
        predispatched=predispatched if OPTION_KEYS & set(save_intent) else None,
        same_view=lambda report: (
            planet_projection(report, _planet(report)) == planet_projection(before, _planet(before))
        ),
    )
    return {
        **{k: v for k, v in planet.items() if k != "class_capture"},
        "save_capture": after,
    }


def _match_stellar_planet_sources(stellar, planet, capture, *, supplied_inputs=False):
    """Bind independently validated sources without inventing non-main answers."""
    source_options(supplied_inputs)
    _require(_identity_matches(stellar["star"], planet["star"]), "cross_star_receipts")
    if supplied_inputs:
        from .planet_supplied_stellar_source import matches_mapping

        receipt = planet["supplied_stellar_inputs"]
        core = {key: stellar[key] for key in receipt["stellar_source"]["bundle"]}
        _require(
            stellar["class"] in NON_MAIN_CLASSES
            and stellar["class"] == receipt["actual_class"]
            and typed_equal(core, receipt["stellar_source"]["bundle"])
            and typed_equal(stellar["numeric_provenance"], receipt["stellar_source"]["numeric_provenance"])
            and typed_equal(stellar["color_provenance"], receipt["stellar_source"]["color_provenance"]),
            "supplied_stellar_chain_mismatch",
        )
        _require(matches_mapping(receipt, _planet(capture)), "supplied_planet_readouts_changed")
        return
    values = _planet(capture)["observation"]["values"]
    for planet_key, stellar_key in (("stellar_mass", "mass"), ("stellar_radius", "radius")):
        committed_display(
            stellar["readbacks"][stellar_key]["display_value"],
            values["stellar_inputs"][planet_key]["display_text"],
        )


def _load_sources(
    book,
    *,
    numeric_dir,
    color_dir,
    class_dir,
    raw_dir,
    derived_dir,
    planet_class_dir,
    save_dir,
    supplied_inputs=False,
):
    options = source_options(supplied_inputs)
    stellar = _stellar_sources(book, numeric_dir, color_dir, class_dir, **options)
    planet = _positive_sources(book, raw_dir, derived_dir, planet_class_dir, save_dir, **options)
    _require(stellar["star"].casefold() == planet["star"].casefold(), "cross_star_receipts")
    _require(planet["planet_class"] != "terrestrial", "terrestrial_completion_evidence_not_supported")
    # Visible stellar reconstruction inputs must belong to this verified star.
    _match_stellar_planet_sources(stellar, planet, planet["save_capture"], **options)
    book.unchanged()
    # The public workflow keeps the native stellar display spelling. Embedded
    # chart/reference evidence retains its original spelling and pinned hashes.
    return {**stellar, **planet, "star": stellar["star"]}


def _read_planet(page, config, directory, bundle):
    session = PlanetNumericSession(page, config, directory, max_seconds=60)
    try:
        report, mapping, choices, handles = session.current()
        _same_star(mapping, bundle["star"])
        _require(
            choices["selected"] == bundle["planet_class"]
            and planet_projection(report, mapping)
            == planet_projection(bundle["save_capture"], _planet(bundle["save_capture"])),
            "current_planet_or_class_changed",
        )
        _require(set(handles) == set(ANSWER_UNITS), "current_answer_controls_missing")
        session.current()
        save_probe(report, directory / "verified")
        return report
    finally:
        session.close()


def verify_positive_planet_workflow(
    page,
    config,
    output,
    *,
    run_history,
    numeric_dir,
    color_dir,
    class_dir,
    raw_dir,
    derived_dir,
    planet_class_dir,
    save_dir,
    supplied_inputs=False,
):
    """Verify one saved non-terrestrial positive branch; no repair or Save retry."""
    book = _Evidence(run_history)
    directory = book.path(output)
    directory.mkdir(parents=True, exist_ok=False)
    navigation = 0
    try:
        bundle = _load_sources(
            book,
            numeric_dir=numeric_dir,
            color_dir=color_dir,
            class_dir=class_dir,
            raw_dir=raw_dir,
            derived_dir=derived_dir,
            planet_class_dir=planet_class_dir,
            save_dir=save_dir,
            **source_options(supplied_inputs),
        )
        boundary, captures = config.model_copy(deep=True), {}
        for section, reader in (("stellar", _read_stellar), ("planet", _read_planet)):
            book.unchanged()
            result = navigate_project(
                page, boundary, directory / ("to-" + section), section, expected_star=bundle["star"]
            )
            _require(
                result.get("same_star_verified") is True and result.get("destination_verified") is True,
                "navigation_not_verified",
            )
            navigation += 1
            for rule in boundary.frames:
                if rule.url == SIMULATION_URL:
                    rule.required_text = result["suggested_required_text"]
            captures[section] = reader(page, boundary, directory / (section + "-readback"), bundle)
        book.unchanged()
        receipt = {
            "schema_version": 1,
            "mode": positive_mode(supplied_inputs),
            "star": bundle["star"],
            "authority": "visible_workflow_readback",
            "task_completed": True,
            "stellar_fields": bundle["readbacks"],
            "numeric_provenance": bundle["numeric_provenance"],
            "color_provenance": bundle["color_provenance"],
            "color": bundle["color"],
            "classification": bundle["class"],
            "classification_provenance": "reference_prediction",
            "learned_classification": False,
            "planet": {
                "outcome": "planet",
                "value": bundle["planet_class"],
                "provenance": "reference_prediction",
                "measurement_provenance": _measurement_mode(bundle["raw"]["evidence"]),
                "approximate": True,
                "transport_verified": True,
                "scientific_verified": False,
                "training_label": False,
            },
            "planet_fields": bundle["planet_readbacks"],
            "raw_measurement_evidence": bundle["raw"]["evidence"],
            "derived_provenance": bundle["derived"]["provenance"],
            "habitability": {
                "outcome": "not_applicable",
                "applicability_reason": "non_terrestrial_planet",
                "provenance": "reference_prediction",
                "branch_applicability_verified": True,
                "transport_verified": False,
                "scientific_verified": False,
                "source": "explicit_branch_applicability_not_field_write",
            },
            **(autosave_workflow_flags() if bundle.get("save_strategy") == "autosave" else {}),
            "save_acknowledgement_verified": bundle.get("save_strategy") != "autosave",
            "source_save_click_delivered": bundle.get("save_strategy") != "autosave",
            "save_acknowledgement_source": AUTOSAVE_ACKNOWLEDGEMENT_SOURCE
            if bundle.get("save_strategy") == "autosave"
            else "explicit_save_fresh_visible_footer",
            "navigation_clicks": navigation,
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
            "source_sha256": dict(book.hashes),
            "current_screen_sha256": {name: screen_identity(report) for name, report in captures.items()},
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "positive_planet_workflow_evidence_or_read_failed",
                "navigation_completed": navigation,
                "answer_writes": 0,
                "task_completed": False,
                "automatic_retry": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("positive_planet_workflow_evidence_or_read_failed") from None
