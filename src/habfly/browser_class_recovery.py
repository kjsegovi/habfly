"""Explicit two-click recovery of verified blank data with stale radio paint.

Not a classifier: the caller supplies the desired class. The intermediate choice
is a UI synchronization step, never a training label or a graded hypothesis.
"""

import hashlib
import json
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import StellarSelectionSession, _stable_mapping
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_stellar import CLASSES, map_stellar_capture


def synchronize_fresh_main_sequence(page, config, output, *, fresh_star):
    """Compatibility entrypoint for explicitly supplied Main Sequence setup."""
    return synchronize_fresh_stellar_class(
        page, config, output, fresh_star=fresh_star, selected_class="main_sequence"
    )


def synchronize_fresh_stellar_class(page, config, output, *, fresh_star, selected_class):
    """Synchronize an explicitly supplied class on a verified blank new star.

    The page can retain the previous star's selected radio paint without its
    conditional fields. Only in that exact proven state, clear the stale option
    with an intermediate choice before selecting the caller's intended class.
    No failed same-option click is required. Neither choice is an inference or
    training label. A different non-main intermediate never exposes conditional
    numeric answers; the class selection is never inferred from measurements.
    """
    if not isinstance(selected_class, str) or selected_class not in CLASSES:
        raise BrowserSafetyStop("invalid_explicit_fresh_stellar_class")
    directory = Path(output)
    session = StellarSelectionSession(page, config, lambda *_: None, fresh_star=fresh_star)
    values = session.mapping["observation"]["values"]
    fields = values["browser_field_map"]
    if (
        values["conditional_fields_visible"]
        or set(fields) != {"distance", "luminosity", "temperature"}
        or values["color"]["selected"] is not None
        or any(f["value_known"] is not True or f["current_value"] != "" for f in fields.values())
    ):
        raise BrowserSafetyStop("fresh_stellar_class_requires_blank_data")
    stale = session.choices["selected"] == selected_class
    intermediate = "red_giant" if selected_class == "white_dwarf" else "white_dwarf"
    directory.mkdir(parents=True, exist_ok=False)
    save_probe(session.report, directory / "before")
    persist_json(
        directory / "scope.json",
        {
            "star": session.mapping["star_name"],
            "intended_class": selected_class,
            "fresh_star": str(fresh_star),
            "source_capture_sha256": session.fresh_star["hash"],
            "max_class_clicks": 2 if stale else 1,
            "numeric_writes": 0,
            "clears_inherited_paint": stale,
            "intermediate_is_training_label": False,
            "learned_classification": False,
            "scientific_verified": False,
            "training_label": False,
            "automatic_retry": False,
        },
    )
    events = []

    def emit(event, payload):
        item = {"event": event, "payload": payload}
        persist_json(directory / f"event-{len(events)}.json", item)
        events.append(item)

    session.emit = emit
    try:
        if stale:
            session.select_class(intermediate, source="reference_diagnostic")
            save_probe(session.report, directory / "intermediate")
            session = StellarSelectionSession(page, config, emit)
            session.select_class(
                selected_class,
                source="reference_diagnostic",
                expected_previous=intermediate,
                revision_reason="Set the explicitly supplied intended class after clearing inherited radio paint on verified fresh blank data. The intermediate choice is not a scientific inference or training label.",
            )
        else:
            session.select_class(selected_class, source="reference_diagnostic")
        after = inspect_page(page, config)
        save_probe(after, directory / "after")
        if screen_identity(after) != screen_identity(session.report):
            raise BrowserSafetyStop("fresh_stellar_class_changed_after_readback")
        values = session.mapping["observation"]["values"]
        fields = values["browser_field_map"]
        main = selected_class == "main_sequence"
        expected_fields = {"distance", "luminosity", "temperature"}
        if main:
            expected_fields |= {"mass", "radius", "lifetime"}
        if (
            values["conditional_fields_visible"] is not main
            or set(fields) != expected_fields
            or values["color"]["selected"] is not None
            or any(f["value_known"] is not True or f["current_value"] != "" for f in fields.values())
            or (main and values["lifetime_prefix"]["selected"] is not None)
        ):
            raise BrowserSafetyStop("fresh_stellar_class_requires_blank_data")
        receipt = {
            "star": session.mapping["star_name"],
            "selected_class": selected_class,
            "class_clicks": sum(e["event"] == "action_proposed" for e in events),
            "numeric_writes": 0,
            "readback_verified": True,
            "correctness_verified": False,
            "learned_classification": False,
            "scientific_verified": False,
            "training_label": False,
            "intermediate_is_training_label": False,
            "action_source": "explicit_fresh_star_class_setup",
            "task_completed": False,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "fresh_class_setup_failed",
                "class_clicks_may_have_occurred": sum(e["event"] == "action_proposed" for e in events),
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        raise


def recover_blank_main_sequence(page, config, output, *, fresh_star, failed_selection):
    failed, directory = Path(failed_selection), Path(output)
    proposed = json.loads((failed / "action_proposed.json").read_text())
    raw = (failed / "after/observation.json").read_bytes()
    manifest = json.loads((failed / "after/manifest.json").read_text())
    if (
        proposed.get("target") != "stellar_class"
        or proposed.get("value") != "main_sequence"
        or proposed.get("action_source") != "reference_diagnostic"
        or (failed / "action_result.json").exists()
        or (failed / "blank-recovery-reserved.json").exists()
        or hashlib.sha256(raw).hexdigest() != manifest["observation_sha256"]
    ):
        raise BrowserSafetyStop("unsupported_blank_class_recovery")
    earlier = map_stellar_capture(
        json.loads(raw),
        capture_sha256=manifest["observation_sha256"],
        allow_color_selection=True,
        allow_main_sequence_fields=True,
    )
    events = []

    def emit(event, payload):
        item = {"event": event, "payload": payload, "recovery_step": len(events) // 2 + 1}
        persist_json(directory / f"event-{len(events)}.json", item)
        events.append(item)

    session = StellarSelectionSession(page, config, emit, fresh_star=fresh_star)
    if (
        session.choices["selected"] != "main_sequence"
        or session.mapping["observation"]["values"]["conditional_fields_visible"]
        or _stable_mapping(session.mapping) != _stable_mapping(earlier)
    ):
        raise BrowserSafetyStop("blank_class_recovery_evidence_changed")
    # One durable claim in the failed attempt prevents retrying via a new folder.
    persist_json(
        failed / "blank-recovery-reserved.json",
        {
            "output": str(directory),
            "fresh_star": str(fresh_star),
            "max_class_clicks": 2,
            "intermediate": "white_dwarf",
            "intended": "main_sequence",
            "automatic_retry": False,
        },
    )
    directory.mkdir(parents=True, exist_ok=False)
    save_probe(session.report, directory / "before")
    try:
        session.select_class("white_dwarf", source="reference_diagnostic")
        save_probe(session.report, directory / "intermediate")
        next_session = StellarSelectionSession(page, config, emit)
        next_session.select_class(
            "main_sequence",
            source="reference_diagnostic",
            expected_previous="white_dwarf",
            revision_reason="Restore the explicitly intended class after clearing stale same-option paint on verified blank data; not a scientific reclassification.",
        )
        after = inspect_page(page, config)
        save_probe(after, directory / "after")
        if screen_identity(after) != screen_identity(next_session.report):
            raise BrowserSafetyStop("blank_class_recovery_changed_after_readback")
        receipt = {
            "star": next_session.mapping["star_name"],
            "selected_class": "main_sequence",
            "class_clicks": 2,
            "numeric_writes": 0,
            "readback_verified": True,
            "correctness_verified": False,
            "learned_classification": False,
            "action_source": "explicit_blank_radio_recovery",
            "original_failure_preserved": True,
            "intermediate_is_training_label": False,
            "automatic_retry": False,
            "task_completed": False,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "blank_class_recovery_failed",
                "class_clicks_may_have_occurred": sum(e["event"] == "action_proposed" for e in events),
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        raise
