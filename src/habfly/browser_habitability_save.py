"""One final terrestrial Save, not scientific acceptance or durable persistence.

Only current public fields and an immutable explicit phase/choice chain authorize
the click. A prior footer notice may settle before reservation. An explicit
new-owner opt-in also permits pre-dispatch settling within one shared deadline;
all failures consume the attempt. No continuation or retry API is provided.
"""

import hashlib
import time
from decimal import Decimal
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_habitability import map_habitability_capture
from .browser_habitability_actions import HabitabilityMenuSession, menu_projection
from .browser_habitability_choice import confirmed_phase_reference
from .browser_habitability_numeric import equilibrium_projection
from .browser_no_planet_save import _sync_directory
from .browser_no_planet_workflow import _Evidence
from .browser_numeric import committed_display, comparable_screen, screen_identity
from .browser_planet_chart import VISIBLE_TOOLTIP
from .browser_planet_save import save_control_exposed
from .browser_probe import save_probe
from .browser_save_settlement import ReservedSavePhase, intent_options
from .browser_stellar import SIMULATION_URL

MODE = "terrestrial_habitability_save"


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("habitability_save_" + reason)


def _mapping(report):
    return map_habitability_capture(report, capture_sha256=screen_identity(report))


def _same_view(left, right):
    """Permit only the existing known Save footer transition between stages.

    Restoring all mapped values makes the numeric projection's answer/readout
    normalization unavailable here. Scientific fields must remain identical.
    """
    a, b = _mapping(left), _mapping(right)
    return a["observation"]["values"] == b["observation"]["values"] and equilibrium_projection(
        left, a
    ) == equilibrium_projection(right, b)


def _complete(mapping):
    values = mapping["observation"]["values"]
    _require(bool(values["selected_gases"]), "explicit_nonempty_gas_selection_required")
    _require(values["greenhouse"] in {"Weak (+10)", "Moderate (+30)", "Strong (+100)"}, "greenhouse_required")
    _require(values["water_phase"] in {"Solid", "Liquid", "Gas"}, "phase_required")
    _require(values["equilibrium_temp"]["unit"] == "K", "temperature_unit_mismatch")
    committed_display(values["equilibrium_temp"]["value"], values["equilibrium_temp"]["value"])
    surface = values["readouts"]["surface_temp"]
    _require(surface["unit"] == "K" and surface["display_text"] is not None, "surface_required")
    committed_display(surface["display_text"], surface["display_text"])
    increment = {"Weak (+10)": 10, "Moderate (+30)": 30, "Strong (+100)": 100}[values["greenhouse"]]
    committed_display(str(Decimal(values["equilibrium_temp"]["value"]) + increment), surface["display_text"])
    _require(values["measurements"]["pressure"]["value"] > 0, "positive_pressure_required")
    return values


def _load_sources(history, phase_dir, choice_dir):
    """Offline source validator also usable by the eventual workflow verifier.

    This proves the explicit phase/choice chain only. Gas identification and
    learned-temperature provenance belong to the separate workflow validator.
    """
    book = _Evidence(history)
    phase_dir, choice_dir = book.clean(phase_dir), book.clean(choice_dir)
    intent, choice = book.json(choice_dir / "reserved.json"), book.json(choice_dir / "confirmed.json")
    before, after = book.capture(choice_dir / "initial"), book.capture(choice_dir / "after")
    mapping = _mapping(after)
    values = _complete(mapping)
    _require(
        choice == {**intent, "readback_verified": True}
        and choice.get("readback_verified") is True
        and choice.get("kind") == "SELECT"
        and choice.get("star") == mapping["star_name"]
        and choice.get("choice") in {"habitable", "not_habitable"}
        and choice.get("previous_paint") in {None, "habitable", "not_habitable"}
        and choice.get("action_source") == "reference_diagnostic"
        and isinstance(choice.get("rationale"), str)
        and 40 <= len(choice["rationale"].strip()) <= 2000
        and choice.get("learned_habitability_decision") is False
        and choice.get("correctness_verified") is False
        and choice.get("task_completed") is False
        and choice.get("automatic_retry") is False
        and type(choice.get("numeric_writes")) is int
        and choice["numeric_writes"] == 0
        and type(choice.get("max_choice_clicks")) is int
        and choice["max_choice_clicks"] == 1
        and _mapping(before)["observation"]["values"] == values
        and equilibrium_projection(before, _mapping(before)) == equilibrium_projection(after, mapping)
        and (choice["choice"] != "habitable" or values["water_phase"] == "Liquid"),
        "invalid_explicit_choice",
    )
    phase = book.json(phase_dir / "confirmed.json")
    book.json(phase_dir / "reserved.json")
    _require(
        phase.get("readback_verified") is True
        and phase.get("automatic_retry") is False
        and type(phase.get("max_menu_writes")) is int
        and phase["max_menu_writes"] == 1
        and type(phase.get("numeric_writes")) is int
        and phase["numeric_writes"] == 0,
        "invalid_phase_transport",
    )
    phase_before, phase_after = book.capture(phase_dir / "initial"), book.capture(phase_dir / "after")
    _require(
        menu_projection(phase_before, _mapping(phase_before), "water_phase")
        == menu_projection(phase_after, _mapping(phase_after), "water_phase")
        and _mapping(phase_before)["observation"]["values"]["water_phase"] is None
        and _same_view(phase_after, before),
        "phase_choice_transition_mismatch",
    )
    hashes = phase.get("evidence", {}).get("source_sha256")
    _require(isinstance(hashes, dict) and len(hashes) == 4, "invalid_chamber_sources")
    for name, expected in hashes.items():
        _require(hashlib.sha256(book.read(name)).hexdigest() == expected, "chamber_source_hash_mismatch")
    candidates = [book.path(name).parent for name in hashes if Path(name).name == "confirmed.json"]
    _require(len(candidates) == 1, "ambiguous_chamber_source")
    chamber = book.clean(candidates[0])
    _require(
        not any(
            (chamber / name).exists()
            for name in ("cleanup-stopped.json", "unconfirmed-controls.json", "unconfirmed-helper.png")
        ),
        "uncertain_chamber_source",
    )
    _require(
        {book.path(name) for name in hashes}
        == {
            chamber / name
            for name in (
                "reserved.json",
                "confirmed.json",
                "task-before/observation.json",
                "task-before/manifest.json",
            )
        },
        "unowned_chamber_chain",
    )
    query, result = book.json(chamber / "reserved.json"), book.json(chamber / "confirmed.json")
    chamber_before, chamber_after = (
        book.capture(chamber / "task-before"),
        book.capture(chamber / "task-after"),
    )
    icons = result.get("icons", [])
    _require(
        set(query)
        == {
            "pressure",
            "pressure_unit",
            "temperature",
            "temperature_unit",
            "action_source",
            "task_answer_write",
        }
        and query.get("task_answer_write") is False
        and query.get("action_source") in {"reference_diagnostic", "checkpoint"}
        and (query.get("pressure_unit"), query.get("temperature_unit")) == ("atm", "K")
        and all(result.get(k) == v for k, v in query.items())
        and result.get("conditions_verified") is True
        and result.get("task_completed") is False
        and result.get("source") == "visible_chamber_indicator"
        and result.get("phase") == values["water_phase"].lower()
        and len(icons) == 3
        and {r.get("phase") for r in icons} == {"solid", "liquid", "gas"}
        and all(
            type(r.get("paint", {}).get("opacity")) in {int, float}
            and r["paint"]["opacity"] == int(r["phase"] == result["phase"])
            for r in icons
        )
        and next(r for r in icons if r["phase"] == result["phase"]).get("fully_exposed") is True
        and book.json(chamber / "observed.json") == result
        and book.json(chamber / "close-reserved.json") == {"retry_allowed": False}
        and book.read(chamber / "chamber.png").startswith(b"\x89PNG\r\n\x1a\n")
        and comparable_screen(chamber_before) == comparable_screen(chamber_after)
        and _same_view(chamber_after, phase_before),
        "unconfirmed_chamber_query",
    )
    for unit, field, expected in (
        ("atm", "pressure", values["measurements"]["pressure"]["display_text"]),
        ("K", "temperature", values["readouts"]["surface_temp"]["display_text"]),
    ):
        number = Decimal(query[field])
        _require(
            number.is_finite()
            and number > 0
            and number == Decimal(expected)
            and number == Decimal(result["visible_readback"][unit]),
            "chamber_conditions_mismatch",
        )
    reference = confirmed_phase_reference(phase_dir, mapping)
    _require(choice.get("evidence") == reference, "choice_phase_evidence_mismatch")
    for name, expected in reference["source_sha256"].items():
        _require(hashlib.sha256(book.read(name)).hexdigest() == expected, "phase_source_hash_mismatch")
    book.unchanged()
    return book, {"choice": choice, "capture": after, "mapping": mapping}


def _cancel(cancelled):
    try:
        value = cancelled()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001 - callback diagnostics may be private
        raise BrowserSafetyStop("habitability_save_cancellation_failed") from None
    _require(type(value) is bool, "cancellation_failed")
    _require(not value, "cancelled")


def _button(session, handle=None):
    session.page.wait_for_timeout(0)
    _require(
        not session.unexpected_dialog
        and session.page.frames == session.frames
        and len(session.page.context.pages) == 1
        and session.frame.url == SIMULATION_URL
        and session.config.allows(session.page.url),
        "context_changed",
    )
    button = session.frame.get_by_role("button", name="Save", exact=True)
    _require(
        button.count() == 1 and button.is_visible() and save_control_exposed(button), "control_unavailable"
    )
    if handle is not None:
        _require(
            handle.evaluate("(a,b)=>a.isConnected&&a===b", button.element_handle(timeout=2000)),
            "control_replaced",
        )
    return button


def _notice(session, handle):
    notices = [n for n in session.frame.get_by_text("Data saved", exact=True).all() if n.is_visible()]
    if not notices:
        return False
    _require(
        len(notices) == 1
        and notices[0].evaluate(
            "(e,b)=>b.parentElement.contains(e)&&b.parentElement.getBoundingClientRect().height<=64", handle
        )
        and notices[0].evaluate(VISIBLE_TOOLTIP),
        "unverified_acknowledgement",
    )
    return True


def save_habitability_work(
    page,
    config,
    output,
    *,
    run_history,
    phase_dir,
    choice_dir,
    timeout_seconds=20,
    settle_timeout_seconds=20,
    settle_reserved_notice=False,
    cancelled=lambda: False,
):
    """A fresh final Save; neither a source decision nor a task receipt.

    Deadlines are fixed and include read-only work. A canonical star claim is
    retained on every post-reservation stop, even cancellation before dispatch.
    """
    history, directory = Path(run_history).resolve(strict=True), Path(output).absolute()
    if (
        not history.is_dir()
        or not directory.resolve().is_relative_to(history)
        or not callable(cancelled)
        or type(settle_reserved_notice) is not bool
        or any(
            type(v) not in {int, float} or not 0.1 <= v <= 30
            for v in (timeout_seconds, settle_timeout_seconds)
        )
    ):
        raise ValueError("Owned history, bounded deadlines and cancellation callback required")
    _Evidence(history).path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    session, last_report, reserved, attempted = None, None, False, False
    popup = []

    def on_popup(_):
        popup.append(True)

    page.on("popup", on_popup)
    try:
        deadline = time.monotonic() + settle_timeout_seconds
        _cancel(cancelled)
        book, source = _load_sources(history, phase_dir, choice_dir)
        star = source["mapping"]["star_name"]
        claim = (
            history
            / "habitability-save-reservations"
            / (hashlib.sha256(star.casefold().encode()).hexdigest() + ".json")
        )
        book.path(claim)
        _require(not claim.exists(), "already_reserved")
        session = HabitabilityMenuSession(
            page,
            config,
            directory / "read-guard",
            max_seconds=120,
            **({"_pin_controls": True} if settle_reserved_notice else {}),
        )

        def current():
            nonlocal last_report
            _cancel(cancelled)
            _require(not popup, "unexpected_popup")
            book.unchanged()
            report, mapping, choices, handles = session.current()
            last_report = report
            _require(not popup, "unexpected_popup")
            _require(
                _complete(mapping) == source["mapping"]["observation"]["values"]
                and choices["selected"] == source["choice"]["choice"]
                and equilibrium_projection(report, mapping)
                == equilibrium_projection(source["capture"], source["mapping"]),
                "source_current_state_mismatch",
            )
            return report, mapping, choices, handles

        before, _, _, _ = current()
        handle = _button(session).element_handle(timeout=2000)
        for index in range(301):
            _cancel(cancelled)
            _require(not popup, "unexpected_popup")
            _require(time.monotonic() < deadline, "preflight_timeout")
            button = _button(session, handle)
            if not _notice(session, handle) and button.is_enabled():
                before, _, _, _ = current()
                book.unchanged()
                button = _button(session, handle)
                if not _notice(session, handle) and button.is_enabled():
                    _require(time.monotonic() < deadline, "preflight_timeout")
                    break
            page.wait_for_timeout(100)
        else:
            raise BrowserSafetyStop("habitability_save_preflight_timeout")
        save_probe(before, directory / "before")
        intent = {
            "schema_version": 1,
            "mode": MODE,
            "star": star,
            "kind": "CLICK",
            "visible_label": "Save",
            "output": str(directory.resolve().relative_to(history)),
            "phase_dir": str(book.path(phase_dir).relative_to(history)),
            "choice_dir": str(book.path(choice_dir).relative_to(history)),
            "source_sha256": dict(book.hashes),
            "before_sha256": screen_identity(before),
            "choice": source["choice"]["choice"],
            "choice_provenance": "reference_prediction",
            "max_save_clicks": 1,
            "numeric_writes": 0,
            "selection_writes": 0,
            "assessment_clicks": 0,
            "score_transfer_clicks": 0,
            "submission_clicks": 0,
            "hidden_values_inspected": False,
            "learned_habitability_decision": False,
            "task_completed": False,
            "automatic_retry": False,
            **intent_options(settle_reserved_notice, timeout_seconds),
        }
        persist_json(
            directory / "pre-reservation-settled.json",
            {
                "read_only_cycles": index + 1,
                "notice_present": False,
                "later_notice_excluded": False,
                "compared_capture_sha256": screen_identity(before),
                "task_completed": False,
            },
        )
        claim.parent.mkdir(exist_ok=True)
        _cancel(cancelled)
        _require(time.monotonic() < deadline, "preflight_timeout")
        _require(not popup and _button(session, handle).is_enabled(), "control_unavailable")
        _require(not _notice(session, handle), "stale_acknowledgement")
        _cancel(cancelled)
        _require(not popup, "unexpected_popup")
        _require(time.monotonic() < deadline, "preflight_timeout")
        book.unchanged()
        try:
            persist_json(claim, intent)
        except FileExistsError:
            raise BrowserSafetyStop("habitability_save_already_reserved") from None
        reserved = True
        _sync_directory(claim.parent)
        _sync_directory(history)
        persist_json(directory / "reserved.json", intent)
        phase = None
        if settle_reserved_notice:

            def quick_check():
                _cancel(cancelled)
                _require(not popup, "unexpected_popup")

            def check():
                quick_check()
                book.unchanged()

            phase = ReservedSavePhase(
                directory,
                seconds=timeout_seconds,
                claims={claim: intent, directory / "reserved.json": intent},
                check=check,
                quick_check=quick_check,
            )
            before = phase.settle(
                current=lambda: current()[0],
                button=lambda: _button(session, handle),
                notice=lambda: _notice(session, handle),
                wait=page.wait_for_timeout,
            )
        else:
            before, _, _, _ = current()
        save_probe(before, directory / "pre-dispatch-observation")
        book.unchanged()
        button = _button(session, handle)
        _require(button.is_enabled(), "control_unavailable")
        present = _notice(session, handle)
        persist_json(
            directory / "pre-dispatch-footer.json",
            {
                "notice_present": present,
                "compared_capture_sha256": screen_identity(before),
                "compared_capture_is_simultaneous": False,
                "save_click_dispatched": False,
            },
        )
        _require(not present, "stale_acknowledgement")
        _cancel(cancelled)
        _require(not popup, "unexpected_popup")
        book.unchanged()
        if phase:
            phase.admit_dispatch()
        attempted = True
        persist_json(directory / "dispatch.json", {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1})
        _sync_directory(directory)
        handle.click(timeout=phase.click_timeout() if phase else 3000)
        if phase:
            after = phase.acknowledge(
                current=lambda: current()[0],
                button=lambda: _button(session, handle),
                notice=lambda: _notice(session, handle),
                wait=page.wait_for_timeout,
            )
            save_probe(after, directory / "after")
            phase.finish(after)
            receipt = {
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
            }
            persist_json(directory / "confirmed.json", receipt)
            return receipt
        deadline = time.monotonic() + timeout_seconds
        while True:
            _cancel(cancelled)
            _require(not popup, "unexpected_popup")
            button = _button(session, handle)
            if _notice(session, handle) and button.is_enabled():
                persist_json(
                    directory / "acknowledgement.json",
                    {
                        "visible_text": "Data saved",
                        "source": "fully_exposed_footer_text",
                        "notice_was_already_present": False,
                    },
                )
                after, _, _, _ = current()
                save_probe(after, directory / "after")
                book.unchanged()
                _cancel(cancelled)
                receipt = {
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
                }
                persist_json(directory / "confirmed.json", receipt)
                return receipt
            _require(time.monotonic() < deadline, "acknowledgement_timeout")
            page.wait_for_timeout(100)
    except BaseException as exc:
        if last_report is not None:
            save_probe(last_report, directory / "last-verified-observation")
        persist_json(
            directory / "stopped.json",
            {
                "mode": MODE,
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "habitability_save_failed",
                "reservation_created": reserved,
                "save_may_have_occurred": attempted,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("habitability_save_failed") from None
    finally:
        page.remove_listener("popup", on_popup)
        if session is not None:
            session.close()
