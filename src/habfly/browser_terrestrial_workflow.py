"""Read-only terrestrial workflow proof, never science or project acceptance.

The supplied main-sequence/terrestrial branch keeps explicit approximate references,
frozen learned calculations, and explicit reference judgments separate. Only
three ordinary tab navigations occur. No answer, helper, Save, assessment, or
submission action is dispatched, and missing evidence is never repaired.
"""

import hashlib
import math
from copy import deepcopy
from decimal import Decimal

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
)
from .browser_gas_controls import gas_projection
from .browser_habitability import GASES
from .browser_habitability_actions import HabitabilityMenuSession, menu_projection
from .browser_habitability_numeric import equilibrium_projection
from .browser_habitability_save import MODE as SAVE_MODE
from .browser_habitability_save import _complete, _mapping
from .browser_habitability_save import _load_sources as _phase_choice_sources
from .browser_no_planet_workflow import _Evidence, _hash_provenance, _planet, _read_stellar
from .browser_numeric import committed_display, screen_identity
from .browser_positive_planet_workflow import (
    _autosave_sources,
    _flags,
    _identity_matches,
    _match_stellar_planet_sources,
    _measurement_mode,
    _planet_sources,
    _read_planet,
    _stellar_sources,
)
from .browser_probe import save_probe
from .browser_project_navigation import navigate_project
from .browser_save_settlement import OPTION_KEYS, typed_equal, validate_reserved_phase
from .browser_stellar import SIMULATION_URL
from .contracts import RuntimeEvent
from .habitability_knowledge import HabitabilityCalculator
from .supplied_browser_modes import TEMPERATURE_SUPPLIED_MODE, source_options, terrestrial_mode


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("terrestrial_workflow_" + reason)


def _values(report):
    return _mapping(report)["observation"]["values"]


def _same_habitat(left, right):
    """Allow only gas-menu expansion/observation-local IDs and Save busy paint.

    Unlike gas_projection alone, this comparison retains the selected gases,
    absorption, exact temperatures, units, and every other visible value.
    """
    values = [deepcopy(_values(r)) for r in (left, right)]
    for value in values:
        value["equilibrium_temp"].pop("capture_target_id")
    projections = [gas_projection(r, _mapping(r)) for r in (left, right)]
    for report, (projection, _) in zip((left, right), projections, strict=True):
        frame = next(f for f in projection["frames"] if f["url"] == SIMULATION_URL)
        original = next(f for f in report["frames"] if f["url"] == SIMULATION_URL)
        # AX may expose the native footer notice as a separate text atom. Only
        # that exact last-row notice is normalized; arbitrary feedback remains.
        if (
            " ".join(original["text"].split()).endswith(" Data saved Save")
            and original["text"].count("Data saved") == 1
            and frame["accessibility"][-2:] == [("text", "Data saved"), ('button "Save"', None)]
        ):
            del frame["accessibility"][-2]
    return values[0] == values[1] and projections[0] == projections[1]


def _clean(book, directory):
    directory = book.clean(directory)
    _require(
        not list(directory.glob("*-stopped.json"))
        and not any(
            (directory / p).exists() for p in ("finalization_failed.json", "event_forward_failed.json")
        ),
        "uncertain_source",
    )
    return directory


def _unchanged(book):
    # Additional stage-specific uncertainty markers can appear without changing
    # an already-read file; checking hashes alone would miss these failures.
    for directory in list(book.clean_directories):
        _clean(book, directory)
    book.unchanged()


def _gas_writes(book, directory, writes, star):
    directory = _clean(book, directory)
    scope = book.json(directory / "scope.json")
    _require(
        _identity_matches(scope.get("star"), star)
        and type(scope.get("max_checkbox_writes")) is int
        and scope["max_checkbox_writes"] == len(writes)
        and scope.get("selection_source") == "explicit_reference_comparison_not_learned",
        "gas_scope_mismatch",
    )
    _flags(
        scope,
        false=("hidden_spectrum_data_read", "task_completed"),
        zero=("numeric_writes", "assessment_clicks"),
    )
    before, last = (book.capture(directory / p) for p in ("initial", "menu-opened"))
    _require(
        _same_habitat(before, last)
        and not _values(before)["selected_gases"]
        and _identity_matches(_mapping(before)["star_name"], star),
        "gas_initial_state_mismatch",
    )
    opening = directory / "menu-open-reserved.json"
    if opening.exists():
        _require(book.json(opening) == {"clicks": 1, "checkbox_writes": 0}, "gas_open_mismatch")
    expected_names = {
        f"write-{i:02d}-{suffix}.json"
        for i in range(1, len(writes) + 1)
        for suffix in ("reserved", "confirmed")
    }
    _require({p.name for p in directory.glob("write-*.json")} == expected_names, "gas_write_count")
    for index, (gas, checked) in enumerate(writes, 1):
        intent = book.json(directory / f"write-{index:02d}-reserved.json")
        receipt = book.json(directory / f"write-{index:02d}-confirmed.json")
        prior = _values(last)["selected_gases"]
        expected = set(prior) | {gas} if checked else set(prior) - {gas}
        _require(
            type(intent.get("checked")) is bool
            and intent
            == {
                "gas": gas,
                "checked": checked,
                "prior": prior,
                "star": star,
                "action_source": "reference_diagnostic",
                "automatic_retry": False,
            }
            and (gas in prior) != checked,
            "gas_write_intent_mismatch",
        )
        after = book.capture(directory / f"write-{index:02d}-after")
        _require(
            gas_projection(last, _mapping(last)) == gas_projection(after, _mapping(after))
            and set(_values(after)["selected_gases"]) == expected
            and receipt
            == {
                **intent,
                "selected": sorted(expected),
                "absorption": _values(after)["readouts"]["absorption"],
                "readback_verified": True,
                "correctness_verified": False,
                "task_completed": False,
            },
            "gas_write_readback_mismatch",
        )
        _flags(
            receipt,
            true=("readback_verified",),
            false=("correctness_verified", "task_completed", "automatic_retry"),
        )
        last = after
    return before, last


def _gas_sources(book, comparison_dir, selection_dir, star):
    comparison_dir, selection_dir = [_clean(book, p) for p in (comparison_dir, selection_dir)]
    comparison = book.json(comparison_dir / "report.json")
    candidates, hashes = comparison.get("candidates"), comparison.get("chart_sha256")
    _require(
        isinstance(candidates, list)
        and 1 <= len(candidates) <= len(GASES)
        and len(set(candidates)) == len(candidates)
        and set(candidates) <= set(GASES)
        and isinstance(hashes, dict)
        and set(hashes) == {"baseline", *candidates}
        and _identity_matches(comparison.get("star"), star)
        and type(comparison.get("checkbox_writes")) is int
        and comparison["checkbox_writes"] == 2 * len(candidates),
        "invalid_gas_comparison",
    )
    _flags(
        comparison,
        true=("baseline_restored",),
        false=("gas_selection_inferred", "learned_gas_identification", "task_completed"),
    )
    for name, digest in hashes.items():
        raw = book.read(comparison_dir / (name + ".png"))
        _require(
            raw.startswith(b"\x89PNG\r\n\x1a\n") and hashlib.sha256(raw).hexdigest() == digest,
            "gas_crop_hash_mismatch",
        )
    first, restored = _gas_writes(
        book,
        comparison_dir,
        [(g, selected) for g in candidates for selected in (True, False)],
        comparison["star"],
    )
    _require(_same_habitat(first, restored), "gas_baseline_not_restored")
    intent, selected = book.json(selection_dir / "selection.json"), book.json(selection_dir / "report.json")
    gases = intent.get("gases")
    _require(
        isinstance(gases, list)
        and gases
        and len(set(gases)) == len(gases)
        and set(gases) <= set(candidates)
        and _identity_matches(intent.get("star"), star)
        and intent.get("action_source") == "explicit_visual_reference_not_learned"
        and isinstance(intent.get("rationale"), str)
        and 20 <= len(intent["rationale"].strip()) <= 2000
        and intent.get("comparison_sha256")
        == hashlib.sha256(book.read(comparison_dir / "report.json")).hexdigest()
        and intent.get("chart_sha256") == hashes,
        "unsupported_gas_selection",
    )
    _flags(intent, false=("correctness_verified", "automatic_retry", "task_completed"))
    initial, last = _gas_writes(book, selection_dir, [(g, True) for g in gases], intent["star"])
    _require(
        _same_habitat(restored, initial)
        and selected
        == {
            **intent,
            "readback_verified": True,
            "checkbox_writes": len(gases),
            "absorption": _values(last)["readouts"]["absorption"],
        }
        and type(selected.get("checkbox_writes")) is int
        and selected.get("readback_verified") is True
        and book.read(selection_dir / "combined.png").startswith(b"\x89PNG\r\n\x1a\n"),
        "gas_selection_readback_mismatch",
    )
    return selected, last


def _temperature_source(book, directory, gas_after, *, supplied_inputs=False, stellar=None, planet=None):
    source_options(supplied_inputs)
    directory = _clean(book, directory)
    report = book.json(directory / "report.json")
    raw = book.read(directory / "events.jsonl")
    events = [RuntimeEvent.model_validate_json(line) for line in raw.splitlines()]
    _require(
        hashlib.sha256(raw).hexdigest() == report.get("events_sha256")
        and 5 <= len(events) <= 1000
        and events[0].event == "hello"
        and events[0].run_id is not None
        and all(
            e.sequence == i and e.run_id == events[0].run_id and e.event != "error"
            for i, e in enumerate(events)
        )
        and events[-1].event == "episode_summary"
        and events[-1].payload == {k: v for k, v in report.items() if k != "events_sha256"},
        "temperature_event_chain",
    )
    _flags(
        report,
        true=("checkpoint_unchanged", "equilibrium_transport_verified"),
        false=("task_completed", "course_acceptance_passed", "saved", "assessed", "submitted"),
        zero=("optimizer_updates",),
    )
    if "sources_unchanged" in report:
        _require(report["sources_unchanged"] is True, "temperature_sources_changed")
    provenance = report["provenance"]
    _hash_provenance(provenance, ("checkpoint_sha256", "graph_hash", "knowledge_pack_hash"))
    _flags(
        provenance,
        true=("surface_proposal_is_not_browser_readback",),
        false=(
            "greenhouse_selection_learned",
            "gas_identification_learned",
            "water_phase_learned",
            "habitability_decision_learned",
        ),
        zero=("optimizer_updates",),
    )
    _require(
        report.get("scope")
        == (
            TEMPERATURE_SUPPLIED_MODE
            if supplied_inputs
            else "one_equilibrium_copy_with_supplied_warming_local_proposal"
        )
        and report.get("outcome") == "equilibrium_transport_verified"
        and type(report.get("steps")) is int
        and 1 <= report["steps"] <= 128
        and type(provenance.get("supplied_greenhouse_increment")) in {int, float}
        and provenance["supplied_greenhouse_increment"] in {10, 30, 100}
        and provenance.get("calibration_scope") == "browser_transfer_not_calibrated"
        and "recovery" not in provenance
        and events[0].payload.get("task") == "habitability_calculations"
        and events[0].payload.get("policy") == "checkpoint"
        and all(events[0].payload.get(k) == v for k, v in provenance.items())
        and sum(
            e.event == "action_proposed" and e.payload.get("action_source") == "checkpoint" for e in events
        )
        == report["steps"]
        and sum(
            e.event == "neural_activity" and e.payload.get("activity_source") == "checkpoint" for e in events
        )
        == report["steps"],
        "unsupported_temperature_provenance",
    )
    if supplied_inputs:
        from .browser_supplied_provenance import validate_supplied_temperature_provenance

        supplied = validate_supplied_temperature_provenance(book, directory, report)
        _require(
            stellar is not None
            and planet is not None
            and _identity_matches(supplied["star"], stellar["star"])
            and supplied["supplied_star_class"] == stellar["class"]
            and typed_equal(supplied["receipt"], planet["supplied_stellar_inputs"]),
            "supplied_temperature_source_mismatch",
        )
    proposals = report["local_proposals"]
    _require(
        set(proposals) == {"equilibrium_temp", "surface_temp"}
        and all(type(v) in {int, float} and math.isfinite(v) and v > 0 for v in proposals.values()),
        "invalid_temperature_proposal",
    )
    native = _clean(book, directory / "native-copy")
    before, after = (book.capture(native / p) for p in ("initial", "after"))
    scope, intent, receipt = (
        book.json(native / p) for p in ("scope.json", "reserved.json", "confirmed.json")
    )
    _require(
        scope.get("scope") == "one_equilibrium_copy_before_greenhouse_selection"
        and type(scope.get("max_writes")) is int
        and scope["max_writes"] == 1
        and _identity_matches(scope.get("star"), _mapping(before)["star_name"])
        and scope.get("selection_source") == "caller_supplied"
        and _same_habitat(gas_after, before)
        and _values(before)["equilibrium_temp"]["value"] in {"", "0"}
        and _values(before)["greenhouse"] is None
        and equilibrium_projection(before, _mapping(before))
        == equilibrium_projection(after, _mapping(after)),
        "temperature_native_context_mismatch",
    )
    _flags(scope, false=("task_completed", "automatic_retry"))
    _require(
        intent
        == {
            "kind": "TYPE",
            "destination": "equilibrium_temp",
            "value": str(proposals["equilibrium_temp"]),
            "unit": "K",
            "action_source": "checkpoint",
            "target": _values(before)["equilibrium_temp"]["capture_target_id"],
            "task_completed": False,
        },
        "temperature_copy_not_checkpoint_result",
    )
    eq, surface = (
        _values(after)["equilibrium_temp"]["value"],
        _values(after)["readouts"]["surface_temp"]["display_text"],
    )
    _require(
        receipt
        == {
            **intent,
            "display": committed_display(intent["value"], eq),
            "surface_display": committed_display(eq, surface),
            "readback_verified": True,
            "correctness_verified": False,
        }
        and report["native_receipt"] == receipt,
        "temperature_native_readback_mismatch",
    )
    _flags(receipt, true=("readback_verified",), false=("correctness_verified", "task_completed"))
    proposals_events = [
        e.payload
        for e in events
        if e.event == "action_proposed"
        and e.payload.get("action_source") == "deterministic_exact_transport_of_checkpoint_result"
    ]
    _require(
        len(proposals_events) == 1
        and (
            "native_write_boundary" not in proposals_events[0]
            or proposals_events[0]["native_write_boundary"] is True
        )
        and {k: v for k, v in proposals_events[0].items() if k != "native_write_boundary"}
        == {
            "kind": "TYPE",
            "destination": "equilibrium_temp",
            "value": intent["value"],
            "action_source": "deterministic_exact_transport_of_checkpoint_result",
        }
        and sum(e.event == "action_result" and e.payload == {"native_copy": receipt} for e in events) == 1,
        "missing_temperature_copy_event",
    )
    return report, after


def _greenhouse_source(book, directory, temperature, before):
    directory = _clean(book, directory)
    initial, after = (book.capture(directory / p) for p in ("initial", "after"))
    intent, receipt = (book.json(directory / p) for p in ("reserved.json", "confirmed.json"))
    calculator = HabitabilityCalculator()
    reference = calculator.greenhouse_reference(_values(initial)["readouts"]["absorption"]["display_text"])
    label = {"weak": "Weak (+10)", "moderate": "Moderate (+30)", "strong": "Strong (+100)"}.get(
        reference.get("name")
    )
    _require(reference.get("ok") is True and label is not None, "unsupported_greenhouse_reference")
    _require(
        _same_habitat(before, initial)
        and _values(initial)["greenhouse"] is None
        and menu_projection(initial, _mapping(initial), "greenhouse")
        == menu_projection(after, _mapping(after), "greenhouse")
        and _values(after)["greenhouse"] == label
        and intent
        == {
            "star": _mapping(initial)["star_name"],
            "kind": "SELECT",
            "field": "greenhouse",
            "label": label,
            "action_source": "reference_diagnostic",
            "evidence": {
                **reference,
                "knowledge_pack_sha256": calculator.pack.checksum,
                "gas_identification_verified": False,
            },
            "max_menu_writes": 1,
            "numeric_writes": 0,
            "task_completed": False,
            "correctness_verified": False,
            "automatic_retry": False,
        }
        and type(intent.get("max_menu_writes")) is int
        and type(intent.get("numeric_writes")) is int
        and receipt
        == {**intent, "readback_verified": True, "surface_temp": _values(after)["readouts"]["surface_temp"]}
        and reference["increment_kelvin"] == temperature["provenance"]["supplied_greenhouse_increment"],
        "greenhouse_chain_mismatch",
    )
    _flags(
        receipt,
        true=("readback_verified",),
        false=("task_completed", "correctness_verified", "automatic_retry"),
        zero=("numeric_writes",),
    )
    committed_display(
        str(Decimal(_values(after)["equilibrium_temp"]["value"]) + reference["increment_kelvin"]),
        _values(after)["readouts"]["surface_temp"]["display_text"],
    )
    return receipt, after


def _final_save(book, directory, phase_dir, choice_dir, greenhouse_after):
    source_book, source = _phase_choice_sources(book.history, phase_dir, choice_dir)
    for path in source_book.clean_directories:
        _clean(book, path)
    for name, digest in source_book.hashes.items():
        _require(
            hashlib.sha256(book.read(book.history / name)).hexdigest() == digest, "phase_sources_changed"
        )
    phase_before = book.capture(book.path(phase_dir) / "initial")
    _require(_same_habitat(greenhouse_after, phase_before), "greenhouse_phase_transition_mismatch")
    directory = _clean(book, directory)
    receipt = book.json(directory / "confirmed.json")
    star = source["mapping"]["star_name"]
    if receipt.get("mode") == AUTOSAVE_MODE:
        receipt, after = _autosave_sources(
            book,
            directory,
            source_book,
            source["capture"],
            branch="terrestrial",
            star=star,
            same=_same_habitat,
        )
        _complete(_mapping(after))
        return source, receipt, after
    intent = book.json(directory / "reserved.json")
    claim = (
        book.history
        / "habitability-save-reservations"
        / (hashlib.sha256(star.casefold().encode()).hexdigest() + ".json")
    )
    before, predispatched, after = [
        book.capture(directory / p) for p in ("before", "pre-dispatch-observation", "after")
    ]
    guard = book.capture(directory / "read-guard/initial")
    _require(
        book.json(claim) == intent
        and intent.get("schema_version") == 1
        and type(intent.get("schema_version")) is int
        and intent.get("mode") == SAVE_MODE
        and _identity_matches(intent.get("star"), star)
        and intent.get("kind") == "CLICK"
        and intent.get("visible_label") == "Save"
        and intent.get("output") == str(directory.relative_to(book.history))
        and intent.get("phase_dir") == str(book.path(phase_dir).relative_to(book.history))
        and intent.get("choice_dir") == str(book.path(choice_dir).relative_to(book.history))
        and intent.get("source_sha256") == source_book.hashes
        and intent.get("choice") == source["choice"]["choice"]
        and intent.get("choice_provenance") == "reference_prediction"
        and type(intent.get("max_save_clicks")) is int
        and intent["max_save_clicks"] == 1
        and intent.get("before_sha256") == screen_identity(before)
        and receipt
        == {
            **intent,
            "after_sha256": screen_identity(after),
            "save_click_delivered": True,
            "data_saved_notice_observed": True,
            "notice_was_already_present": False,
            "answers_unchanged": True,
            "choice_paint_unchanged": True,
            "cross_session_persistence_verified": False,
            "scientific_verified": False,
            "correctness_verified": False,
            "course_completion_verified": False,
            "assessed": False,
            "score_updated": False,
            "submitted": False,
        },
        "final_save_chain_mismatch",
    )
    _flags(
        receipt,
        true=(
            "save_click_delivered",
            "data_saved_notice_observed",
            "answers_unchanged",
            "choice_paint_unchanged",
        ),
        false=(
            "hidden_values_inspected",
            "learned_habitability_decision",
            "task_completed",
            "automatic_retry",
            "notice_was_already_present",
            "cross_session_persistence_verified",
            "scientific_verified",
            "correctness_verified",
            "course_completion_verified",
            "assessed",
            "score_updated",
            "submitted",
        ),
        zero=(
            "numeric_writes",
            "selection_writes",
            "assessment_clicks",
            "score_transfer_clicks",
            "submission_clicks",
        ),
    )
    settled = book.json(directory / "pre-reservation-settled.json")
    _flags(settled, false=("notice_present", "later_notice_excluded", "task_completed"))
    _require(
        type(settled.get("read_only_cycles")) is int
        and 1 <= settled["read_only_cycles"] <= 301
        and settled
        == {
            "read_only_cycles": settled["read_only_cycles"],
            "notice_present": False,
            "later_notice_excluded": False,
            "compared_capture_sha256": screen_identity(before),
            "task_completed": False,
        },
        "save_settlement_mismatch",
    )
    footer = book.json(directory / "pre-dispatch-footer.json")
    dispatch = book.json(directory / "dispatch.json")
    acknowledgement = book.json(directory / "acknowledgement.json")
    _flags(footer, false=("notice_present", "compared_capture_is_simultaneous", "save_click_dispatched"))
    _flags(acknowledgement, false=("notice_was_already_present",))
    _require(
        footer
        == {
            "notice_present": False,
            "compared_capture_sha256": screen_identity(predispatched),
            "compared_capture_is_simultaneous": False,
            "save_click_dispatched": False,
        }
        and type(dispatch.get("max_clicks")) is int
        and dispatch == {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1}
        and acknowledgement
        == {
            "visible_text": "Data saved",
            "source": "fully_exposed_footer_text",
            "notice_was_already_present": False,
        }
        and all(_same_habitat(source["capture"], r) for r in (guard, before, predispatched, after)),
        "final_save_acknowledgement_mismatch",
    )
    if OPTION_KEYS & set(intent):
        _require(
            typed_equal(book.json(claim), intent)
            and all(typed_equal(receipt.get(key), intent.get(key)) for key in OPTION_KEYS),
            "save_policy_receipt_mismatch",
        )
    validate_reserved_phase(
        book,
        directory,
        intent,
        after=after,
        predispatched=predispatched,
        same_view=lambda report: _same_habitat(source["capture"], report),
    )
    return source, receipt, after


def _load_sources(
    book,
    *,
    numeric_dir,
    color_dir,
    class_dir,
    raw_dir,
    derived_dir,
    planet_class_dir,
    gas_comparison_dir,
    gas_selection_dir,
    temperature_dir,
    greenhouse_dir,
    phase_dir,
    choice_dir,
    save_dir,
    supplied_inputs=False,
):
    options = source_options(supplied_inputs)
    stellar = _stellar_sources(book, numeric_dir, color_dir, class_dir, **options)
    planet = _planet_sources(book, raw_dir, derived_dir, planet_class_dir, **options)
    star = stellar["star"]
    _require(
        _identity_matches(planet["star"], star) and planet["planet_class"] == "terrestrial",
        "terrestrial_branch_required",
    )
    if supplied_inputs:
        _match_stellar_planet_sources(stellar, planet, planet["class_capture"], **options)
    else:
        planetary = _planet(planet["class_capture"])["observation"]["values"]
        for p, s in (("stellar_mass", "mass"), ("stellar_radius", "radius")):
            committed_display(
                stellar["readbacks"][s]["display_value"], planetary["stellar_inputs"][p]["display_text"]
            )
    gas, gas_after = _gas_sources(book, gas_comparison_dir, gas_selection_dir, star)
    temperature, temperature_after = _temperature_source(
        book,
        temperature_dir,
        gas_after,
        **({**options, "stellar": stellar, "planet": planet} if supplied_inputs else {}),
    )
    greenhouse, greenhouse_after = _greenhouse_source(book, greenhouse_dir, temperature, temperature_after)
    choice, save, saved_capture = _final_save(book, save_dir, phase_dir, choice_dir, greenhouse_after)
    _require(_identity_matches(choice["mapping"]["star_name"], star), "cross_star_habitability")
    measurements = _values(saved_capture)["measurements"]
    committed_display(
        stellar["readbacks"]["luminosity"]["display_value"],
        measurements["stellar_luminosity"]["display_text"],
    )
    committed_display(
        planet["planet_readbacks"]["orbital_radius"]["display"]["display_value"],
        measurements["orbital_radius"]["display_text"],
    )
    _unchanged(book)
    return {
        **stellar,
        **planet,
        "star": star,
        "gas": gas,
        "temperature": temperature,
        "greenhouse": greenhouse,
        "choice": choice["choice"],
        "final_save": save,
        **(autosave_workflow_flags() if save.get("mode") == AUTOSAVE_MODE else {}),
        "habitability_capture": saved_capture,
        # The planet reader compares a capture, not a Planet-tab Save claim.
        "save_capture": planet["class_capture"],
    }


def _read_habitability(page, config, directory, bundle):
    session = HabitabilityMenuSession(page, config, directory, max_seconds=60)
    try:
        report, mapping, choices, _ = session.current()
        _complete(mapping)
        _require(
            _identity_matches(mapping["star_name"], bundle["star"])
            and choices["selected"] == bundle["choice"]["choice"]
            and _same_habitat(report, bundle["habitability_capture"]),
            "current_habitability_changed",
        )
        session.current()
        save_probe(report, directory / "verified")
        return report
    finally:
        session.close()


def verify_terrestrial_workflow(page, config, output, *, run_history, supplied_inputs=False, **sources):
    """Verify visible saved work, not a course grade, science, or persistence."""
    book = _Evidence(run_history)
    directory = book.path(output)
    directory.mkdir(parents=True, exist_ok=False)
    navigation = 0
    try:
        bundle = _load_sources(book, **sources, **source_options(supplied_inputs))
        boundary, captures = config.model_copy(deep=True), {}
        for section, reader in (
            ("stellar", _read_stellar),
            ("planet", _read_planet),
            ("habitability", _read_habitability),
        ):
            _unchanged(book)
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
        _unchanged(book)
        receipt = {
            "schema_version": 1,
            "mode": terrestrial_mode(supplied_inputs),
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
                "value": "terrestrial",
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
            "gas_selection": {
                "gases": bundle["gas"]["gases"],
                "provenance": "reference_prediction",
                "transport_verified": True,
                "learned_gas_identification": False,
                "scientific_verified": False,
            },
            "temperature_provenance": bundle["temperature"]["provenance"],
            "equilibrium_transport": bundle["temperature"]["native_receipt"],
            "greenhouse_reference": bundle["greenhouse"]["evidence"],
            "phase_reference": bundle["choice"]["evidence"],
            "habitability": {
                "outcome": bundle["choice"]["choice"],
                "provenance": "reference_prediction",
                "transport_verified": True,
                "scientific_verified": False,
                "learned_habitability_decision": False,
                "source": "explicit_native_choice_with_confirmed_chamber_phase",
            },
            **(autosave_workflow_flags() if bundle.get("save_strategy") == "autosave" else {}),
            "save_acknowledgement_verified": bundle.get("save_strategy") != "autosave",
            "source_save_click_delivered": bundle.get("save_strategy") != "autosave",
            "save_acknowledgement_source": AUTOSAVE_ACKNOWLEDGEMENT_SOURCE
            if bundle.get("save_strategy") == "autosave"
            else "explicit_final_habitability_save_fresh_visible_footer",
            "navigation_clicks": navigation,
            "answer_writes": 0,
            "habitability_writes": 0,
            "na_writes": 0,
            "save_clicks": 0,
            "assessment_clicks": 0,
            "score_transfer_clicks": 0,
            "submission_clicks": 0,
            "hidden_values_inspected": False,
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
                else "terrestrial_workflow_evidence_or_read_failed",
                "navigation_completed": navigation,
                "answer_writes": 0,
                "task_completed": False,
                "automatic_retry": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("terrestrial_workflow_evidence_or_read_failed") from None
