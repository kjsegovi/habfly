"""One caller-selected painted planet class, never a scientific classifier."""

from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_numeric import committed_display
from .browser_planet_numeric import ANSWER_UNITS, PlanetNumericSession, planet_projection
from .browser_probe import save_probe

CLASSES = ("gas_giant", "ice_giant", "terrestrial")


def select_planet_class(
    page, config, output, name, *, source, expected_previous=None, revision_reason=None, before_dispatch=None
):
    """Select one explicit class; an optional caller guard may veto dispatch.

    The guard receives the current visible capture, mapping and painted choices.
    A fresh guarded read follows it, so callback side effects cannot silently
    change the authorized screen or replace native controls. Legacy calls keep
    their original receipt shape and read count.
    """
    if name not in CLASSES or source not in {"reference_diagnostic", "checkpoint"}:
        raise BrowserSafetyStop("invalid_planet_class_request")
    if before_dispatch is not None and not callable(before_dispatch):
        raise BrowserSafetyStop("invalid_planet_class_dispatch_guard")
    revising = expected_previous is not None
    if (
        revising
        and (
            expected_previous not in CLASSES
            or expected_previous == name
            or source != "reference_diagnostic"
            or not isinstance(revision_reason, str)
            or not revision_reason.strip()
            or len(revision_reason) > 1000
        )
    ) or (not revising and revision_reason is not None):
        raise BrowserSafetyStop("invalid_planet_class_revision")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    session = None
    attempted = False
    try:
        # Reuse the known same-star, field, frame, modal, native-identity and
        # deadline guards. No numeric copy method is called in this function.
        session = PlanetNumericSession(page, config, directory / "read-guard", max_seconds=60)
        before, mapping, choices, _ = session.current()
        fields = mapping["observation"]["values"]["browser_field_map"]
        for field in ANSWER_UNITS:
            committed_display(fields[field]["current_value"], fields[field]["current_value"])
        if choices["selected"] != expected_previous:
            raise BrowserSafetyStop(
                "planet_class_revision_precondition" if revising else "planet_class_already_selected"
            )
        from .browser_classification import read_planet_class_choices

        visible, handles = read_planet_class_choices(session.frame)
        if visible != choices:
            raise BrowserSafetyStop("planet_class_changed_during_binding")
        intent = {
            "kind": "SELECT",
            "value": name,
            "star": mapping["star_name"],
            "action_source": source,
            "max_class_writes": 1,
            "numeric_writes": 0,
            "correctness_verified": False,
            "task_completed": False,
        }
        if revising:
            intent.update(previous=expected_previous, revision_reason=revision_reason)
        persist_json(directory / "reserved.json", intent)
        current, current_mapping, current_choices, _ = session.current()
        if before_dispatch is not None:
            before_dispatch(deepcopy(current), deepcopy(current_mapping), deepcopy(current_choices))
            guarded, guarded_mapping, guarded_choices, _ = session.current()
            if (
                planet_projection(current, current_mapping) != planet_projection(guarded, guarded_mapping)
                or current_choices != guarded_choices
            ):
                raise BrowserSafetyStop("planet_class_dispatch_source_changed")
        latest, rebound = read_planet_class_choices(session.frame)
        if latest != visible or any(
            not handles[k].evaluate("(a,b)=>a.isConnected&&a===b", rebound[k]) for k in CLASSES
        ):
            raise BrowserSafetyStop("planet_class_control_replaced")
        attempted = True
        handles[name].click(timeout=3000)
        after, newer, selected, _ = session.read()
        save_probe(after, directory / "after")
        expected_rendering = dict(choices["rendering"])
        if revising:
            # The existing selected and unselected painted circles swap. No
            # hidden radio values or application state establish this transition.
            expected_rendering[name] = choices["rendering"][expected_previous]
            expected_rendering[expected_previous] = choices["rendering"][name]
        if (
            selected["selected"] != name
            or any(
                selected["rendering"][k] != expected_rendering[k] for k in CLASSES if revising or k != name
            )
            or planet_projection(before, mapping) != planet_projection(after, newer)
        ):
            raise BrowserSafetyStop("unexpected_planet_class_side_effect")
        receipt = {**intent, "readback_verified": True}
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "planet_class_operation_failed",
                "write_may_have_occurred": attempted,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        raise
    finally:
        if session is not None:
            session.close()
