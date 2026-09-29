"""Cooperative gas/ice finalization from immutable supplied source receipts.

Construction is offline. Separate advances read the current Planet tab, perform
one guarded Save, then run the existing two-tab read-only verifier. There is no
classification, recovery, journal import, scoring, or submission capability.
"""

import hashlib
import json
import re
import time

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_autosave import autosave_flags, autosave_workflow_flags, validate_autosave_workflow_flags
from .browser_no_planet_save import _sync_directory
from .browser_no_planet_workflow import _Evidence, _planet
from .browser_numeric import committed_display, screen_identity
from .browser_planet_chart import VISIBLE_TOOLTIP
from .browser_planet_numeric import ANSWER_UNITS, PlanetNumericSession, planet_projection
from .browser_planet_save import save_control_exposed
from .browser_positive_planet_workflow import (
    _identity_matches,
    _match_stellar_planet_sources,
    _planet_sources,
    _read_planet,
    _stellar_sources,
    verify_positive_planet_workflow,
)
from .browser_probe import save_probe
from .browser_save_settlement import ReservedSavePhase, intent_options
from .browser_stellar import SIMULATION_URL
from .contracts import RuntimeEvent
from .supplied_browser_modes import positive_mode, source_options

MODE = "cooperative_positive_planet_finalization"


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("positive_finalization_" + reason)


def _sources(book, directories, class_sha, *, supplied_inputs=False):
    options = source_options(supplied_inputs)
    _require(isinstance(class_sha, str) and re.fullmatch(r"[a-f0-9]{64}", class_sha), "class_hash_required")
    class_path = book.path(directories["planet_class_dir"]) / "confirmed.json"
    _require(hashlib.sha256(book.read(class_path)).hexdigest() == class_sha, "supplied_class_changed")
    stellar = _stellar_sources(
        book, *(directories[k] for k in ("numeric_dir", "color_dir", "class_dir")), **options
    )
    planet = _planet_sources(
        book, *(directories[k] for k in ("raw_dir", "derived_dir", "planet_class_dir")), **options
    )
    _require(_identity_matches(stellar["star"], planet["star"]), "cross_star_sources")
    _require(planet["planet_class"] in {"gas_giant", "ice_giant"}, "explicit_non_terrestrial_class_required")
    if supplied_inputs:
        _match_stellar_planet_sources(stellar, planet, planet["class_capture"], **options)
    else:
        values = _planet(planet["class_capture"])["observation"]["values"]
        for p, s in (("stellar_mass", "mass"), ("stellar_radius", "radius")):
            committed_display(
                stellar["readbacks"][s]["display_value"], values["stellar_inputs"][p]["display_text"]
            )
    book.unchanged()
    return {**stellar, **planet, "star": stellar["star"], "save_capture": planet["class_capture"]}


def _claim_path(history, star):
    return (
        history
        / "positive-finalization-reservations"
        / (hashlib.sha256(star.casefold().encode()).hexdigest() + ".json")
    )


def _known_save_attempts(book, star):
    """Refuse legacy output-local Save intents too; never retrofit receipts.

    The bounded artifact inventory is read-only and contains no application or
    storage state. Malformed/oversized reservation artifacts fail closed.
    """
    paths = sorted(book.history.rglob("reserved.json"))
    _require(len(paths) <= 10000, "reservation_inventory_limit")
    for path in paths:
        value = book.json(path)
        if (
            value.get("kind") == "CLICK"
            and value.get("visible_label") == "Save"
            and _identity_matches(value.get("star"), star)
        ):
            raise BrowserSafetyStop("positive_finalization_prior_save_intent")


def _capture_autosave_readback(
    page, config, output, *, source_book, star, branch, session_factory, validate, check, timeout_seconds
):
    """Two guarded visible snapshots, deliberately without any Save control read."""
    directory = source_book.path(output)
    directory.mkdir(parents=True, exist_ok=False)
    deadline, session = time.monotonic() + timeout_seconds, None

    def guard():
        check()
        source_book.unchanged()
        _require(time.monotonic() < deadline, "autosave_readback_timeout")

    try:
        guard()
        session = session_factory(page, config, directory / "read-guard", max_seconds=60, _pin_controls=True)
        guard()
        validate(session.report, session.mapping, session.choices, session.handles)
        before = session.report
        report, mapping, choices, handles = session.current()
        guard()
        validate(report, mapping, choices, handles)
        save_probe(report, directory / "after")
        guard()
        receipt = {
            "schema_version": 1,
            **autosave_flags(),
            "branch": branch,
            "star": star,
            "output": str(directory.relative_to(source_book.history)),
            "source_sha256": dict(source_book.hashes),
            "before_sha256": screen_identity(before),
            "after_sha256": screen_identity(report),
        }
        persist_json(directory / "confirmed.json", receipt)
        guard()
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "autosave_readback_failed",
                "save_click_delivered": False,
                "persistence_verified": False,
                "task_completed": False,
                "automatic_retry": False,
            },
        )
        raise
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - driver cleanup text may be private
                if not (directory / "stopped.json").exists():
                    persist_json(
                        directory / "stopped.json",
                        {
                            "reason": "autosave_readback_cleanup_failed",
                            "save_click_delivered": False,
                            "persistence_verified": False,
                            "task_completed": False,
                            "automatic_retry": False,
                        },
                    )
                raise BrowserSafetyStop("autosave_readback_cleanup_failed") from None


def _positive_autosave(
    page, config, output, *, history, directories, supplied_inputs, check, timeout_seconds
):
    source_book = _Evidence(history)
    source = _planet_sources(
        source_book,
        *(directories[k] for k in ("raw_dir", "derived_dir", "planet_class_dir")),
        **source_options(supplied_inputs),
    )

    def validate(report, mapping, choices, handles):
        _require(
            _identity_matches(mapping["star_name"], source["star"])
            and choices["selected"] == source["planet_class"]
            and set(handles) == set(ANSWER_UNITS)
            and planet_projection(report, mapping)
            == planet_projection(source["class_capture"], _planet(source["class_capture"])),
            "autosave_current_planet_changed",
        )

    return _capture_autosave_readback(
        page,
        config,
        output,
        source_book=source_book,
        star=source["star"],
        branch="positive",
        session_factory=PlanetNumericSession,
        validate=validate,
        check=check,
        timeout_seconds=timeout_seconds,
    )


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
        button.count() == 1 and button.is_visible() and save_control_exposed(button),
        "save_control_unavailable",
    )
    if handle is not None:
        _require(
            handle.evaluate("(a,b)=>a.isConnected&&a===b", button.element_handle(timeout=2000)),
            "save_control_replaced",
        )
    return button


def _notice(session, handle):
    notices = [e for e in session.frame.get_by_text("Data saved", exact=True).all() if e.is_visible()]
    if not notices:
        return False
    _require(
        len(notices) == 1
        and notices[0].evaluate(
            "(e,b)=>b.parentElement.contains(e)&&b.parentElement.getBoundingClientRect().height<=64", handle
        )
        and notices[0].evaluate(VISIBLE_TOOLTIP),
        "unverified_save_notice",
    )
    return True


def _strict_save(
    page,
    config,
    directory,
    *,
    book,
    bundle,
    owner_output,
    class_sha,
    check,
    quick_check,
    timeout_seconds,
    settle_timeout_seconds,
    settle_reserved_notice=False,
):
    """One native Save with the unchanged legacy positive receipt format.

    Additional canonical and dispatch evidence are real new actions only. A
    late stale banner consumes the reservation and stops before dispatch by
    default. Explicit new-owner settling retains the same claim and shares one
    unchanged final deadline through readback; failed attempts never resume.
    """
    directory = book.path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    session, last, attempted, reserved = None, None, False, False
    popups = []

    def popup(_):
        popups.append(True)

    page.on("popup", popup)
    try:
        deadline = time.monotonic() + settle_timeout_seconds
        check()
        claim = _claim_path(book.history, bundle["star"])
        book.path(claim)
        _require(not claim.exists(), "already_reserved")
        _known_save_attempts(book, bundle["star"])
        session = PlanetNumericSession(
            page,
            config,
            directory / "read-guard",
            max_seconds=120,
            **({"_pin_controls": True} if settle_reserved_notice else {}),
        )

        def current():
            nonlocal last
            check()
            _require(not popups, "unexpected_popup")
            report, mapping, choices, _ = session.current()
            last = report
            _require(
                _identity_matches(mapping["star_name"], bundle["star"])
                and choices["selected"] == bundle["planet_class"]
                and planet_projection(report, mapping)
                == planet_projection(bundle["class_capture"], _planet(bundle["class_capture"])),
                "current_planet_changed",
            )
            fields = mapping["observation"]["values"]["browser_field_map"]
            _require(set(fields) >= set(ANSWER_UNITS), "missing_planet_fields")
            for name in ANSWER_UNITS:
                _require(fields[name]["unit"] == ANSWER_UNITS[name], "planet_unit_changed")
                committed_display(fields[name]["current_value"], fields[name]["current_value"])
            return report

        before = current()
        handle = _button(session).element_handle(timeout=2000)
        for cycle in range(301):
            quick_check()
            _require(not popups and time.monotonic() < deadline, "save_settlement_timeout")
            button = _button(session, handle)
            if button.is_enabled() and not _notice(session, handle):
                before = current()
                if _button(session, handle).is_enabled() and not _notice(session, handle):
                    _require(time.monotonic() < deadline, "save_settlement_timeout")
                    break
            page.wait_for_timeout(100)
        else:
            raise BrowserSafetyStop("positive_finalization_save_settlement_timeout")
        # Retain the real initial capture; never rewrite it after settlement.
        # The existing comparison permits only known footer/busy/tooltip paint.
        initial = book.capture(directory / "read-guard/initial")
        _require(
            planet_projection(initial, _planet(initial)) == planet_projection(before, _planet(before)),
            "settled_initial_identity_changed",
        )
        intent = {
            "kind": "CLICK",
            "visible_label": "Save",
            "star": bundle["star"],
            "before_sha256": screen_identity(initial),
            "max_save_clicks": 1,
            "numeric_writes": 0,
            "selection_source": "scripted_setup_not_learned",
            "task_completed": False,
            "automatic_retry": False,
            **intent_options(settle_reserved_notice, timeout_seconds),
        }
        canonical = {
            "schema_version": 1,
            "mode": MODE,
            "star": bundle["star"],
            "planet_class": bundle["planet_class"],
            "output": str(owner_output.relative_to(book.history)),
            "save_output": str(directory.relative_to(book.history)),
            "planet_class_sha256": class_sha,
            "source_sha256": dict(book.hashes),
            "native_intent": intent,
            "maximum_save_dispatches": 1,
            "automatic_retry": False,
            "task_completed": False,
        }
        persist_json(
            directory / "pre-reservation-settled.json",
            {
                "notice_present": False,
                "read_only_cycles": cycle + 1,
                "compared_capture_sha256": screen_identity(before),
                "task_completed": False,
            },
        )
        check()
        _require(time.monotonic() < deadline and not popups, "save_settlement_timeout")
        _require(
            _button(session, handle).is_enabled() and not _notice(session, handle),
            "stale_save_acknowledgement",
        )
        claim.parent.mkdir(exist_ok=True)
        try:
            persist_json(claim, canonical)
        except FileExistsError:
            raise BrowserSafetyStop("positive_finalization_already_reserved") from None
        reserved = True
        _sync_directory(claim.parent)
        _sync_directory(book.history)
        persist_json(directory / "reserved.json", intent)
        phase = None
        if settle_reserved_notice:
            phase = ReservedSavePhase(
                directory,
                seconds=timeout_seconds,
                claims={claim: canonical, directory / "reserved.json": intent},
                check=check,
                quick_check=quick_check,
            )
            fresh = phase.settle(
                current=current,
                button=lambda: _button(session, handle),
                notice=lambda: _notice(session, handle),
                wait=page.wait_for_timeout,
            )
        else:
            fresh = current()
        save_probe(fresh, directory / "pre-dispatch-observation")
        check()
        button = _button(session, handle)
        present = _notice(session, handle)
        persist_json(
            directory / "pre-dispatch-footer.json",
            {
                "notice_present": present,
                "save_click_dispatched": False,
                "compared_capture_sha256": screen_identity(fresh),
                "compared_capture_is_simultaneous": False,
            },
        )
        _require(not popups and button.is_enabled() and not present, "stale_save_acknowledgement")
        check()
        _require(
            _button(session, handle).is_enabled() and not _notice(session, handle),
            "stale_save_acknowledgement",
        )
        if phase:
            phase.admit_dispatch()
        persist_json(directory / "dispatch.json", {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1})
        _sync_directory(directory)
        attempted = True
        handle.click(timeout=phase.click_timeout() if phase else 3000)
        if phase:
            after = phase.acknowledge(
                current=current,
                button=lambda: _button(session, handle),
                notice=lambda: _notice(session, handle),
                wait=page.wait_for_timeout,
            )
            save_probe(after, directory / "after")
            phase.finish(after)
            receipt = {
                **intent,
                "save_click_delivered": True,
                "data_saved_notice_observed": True,
                "notice_was_already_present": False,
                "answers_unchanged": True,
                "cross_session_persistence_verified": False,
                "course_completion_verified": False,
                "assessed": False,
                "score_updated": False,
                "submitted": False,
            }
            persist_json(directory / "confirmed.json", receipt)
            return receipt
        deadline = time.monotonic() + timeout_seconds
        while True:
            quick_check()
            _require(not popups, "unexpected_popup")
            button = _button(session, handle)
            # Lightweight polling precedes expensive source/full-view checks.
            if _notice(session, handle) and button.is_enabled():
                persist_json(
                    directory / "acknowledgement.json",
                    {
                        "visible_text": "Data saved",
                        "source": "fully_exposed_footer_text",
                        "notice_was_already_present": False,
                    },
                )
                after = current()
                save_probe(after, directory / "after")
                check()
                receipt = {
                    **intent,
                    "save_click_delivered": True,
                    "data_saved_notice_observed": True,
                    "notice_was_already_present": False,
                    "answers_unchanged": True,
                    "cross_session_persistence_verified": False,
                    "course_completion_verified": False,
                    "assessed": False,
                    "score_updated": False,
                    "submitted": False,
                }
                persist_json(directory / "confirmed.json", receipt)
                return receipt
            if time.monotonic() >= deadline:
                check()
                raise BrowserSafetyStop("positive_finalization_save_acknowledgement_timeout")
            page.wait_for_timeout(100)
    except BaseException as exc:
        if last is not None:
            save_probe(last, directory / "last-verified-observation")
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "positive_finalization_save_failed",
                "reservation_created": reserved,
                "save_may_have_occurred": attempted,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("positive_finalization_save_failed") from None
    finally:
        page.remove_listener("popup", popup)
        if session is not None:
            session.close()


class PositiveFinalizationSteps:
    """Caller owns scheduling; each phase requires a separate advance()."""

    def __init__(
        self,
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
        planet_class_sha256,
        emit=lambda _: None,
        timeout_seconds=20,
        settle_timeout_seconds=20,
        settle_reserved_notice=False,
        supplied_inputs=False,
        save_strategy="explicit",
    ):
        options = source_options(supplied_inputs)
        if (
            type(settle_reserved_notice) is not bool
            or type(save_strategy) is not str
            or save_strategy not in {"explicit", "autosave"}
            or not callable(emit)
            or any(
                type(v) not in {int, float} or not 0.1 <= v <= 30
                for v in (timeout_seconds, settle_timeout_seconds)
            )
        ):
            raise ValueError("Callable event sink and fixed bounded Save deadlines required")
        self.page, self.config, self.book = page, config.model_copy(deep=True), _Evidence(run_history)
        self.output = self.book.path(output)
        self.directories = {
            name + "_dir": self.book.path(value)
            for name, value in (
                ("numeric", numeric_dir),
                ("color", color_dir),
                ("class", class_dir),
                ("raw", raw_dir),
                ("derived", derived_dir),
                ("planet_class", planet_class_dir),
            )
        }
        self.bundle = _sources(self.book, self.directories, planet_class_sha256, **options)
        self.supplied_inputs = supplied_inputs
        _require(not _claim_path(self.book.history, self.bundle["star"]).exists(), "already_reserved")
        _known_save_attempts(self.book, self.bundle["star"])
        self.class_sha = planet_class_sha256
        self.source_hashes = dict(self.book.hashes)
        self.timeout, self.settle_timeout = timeout_seconds, settle_timeout_seconds
        self.settle_reserved_notice = settle_reserved_notice
        self.save_strategy = save_strategy
        self._fixed_save_options = tuple(
            (type(value), value)
            for value in (self.timeout, self.settle_timeout, self.settle_reserved_notice, self.save_strategy)
        )
        self._callback, self._sequence, self._phase = emit, 0, "readback"
        self._advancing = self._cancelled = self._forward_failed = self._verified = self._finalizing = False
        self._dispatch_recorded = False
        self.finished, self.report, self.save_receipt, self.workflow_receipt = False, None, None, None
        self._workflow_sha = None
        self.output.mkdir(parents=True, exist_ok=False)
        self._stream = (self.output / "events.jsonl").open("x")
        try:
            self._emit(
                "hello",
                {
                    "protocol_version": 1,
                    "component": MODE,
                    "star": self.bundle["star"],
                    "planet_class": self.bundle["planet_class"],
                    "classification_source": "immutable_supplied_reference_receipt",
                    "planet_class_sha256": self.class_sha,
                    "source_sha256": self.source_hashes,
                    "constructor_browser_actions": 0,
                    "optimizer_updates": 0,
                    **(autosave_workflow_flags() if self.save_strategy == "autosave" else {}),
                    **intent_options(settle_reserved_notice, timeout_seconds),
                },
            )
            self._check()
            self._emit("state", self.state())
            self._check()
        except BaseException as exc:
            self._stop(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise

    def _emit(self, event, payload):
        item = RuntimeEvent(event=event, sequence=self._sequence, run_id=self.output.name, payload=payload)
        line = item.model_dump_json()
        self._stream.write(line + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            try:
                self._callback(json.loads(line))
            except Exception:  # noqa: BLE001 - callback diagnostics may contain private data
                self._forward_failed = True

    def _quick_check(self):
        _require(not self._cancelled, "operator_aborted")
        _require(not self._forward_failed, "event_forwarding_failed")
        _require(
            tuple(
                (type(value), value)
                for value in (
                    self.timeout,
                    self.settle_timeout,
                    self.settle_reserved_notice,
                    self.save_strategy,
                )
            )
            == self._fixed_save_options,
            "save_options_changed",
        )

    def _check(self):
        self._quick_check()
        self.book.unchanged()

    def state(self):
        return {
            **source_options(self.supplied_inputs),
            **(autosave_workflow_flags() if self.save_strategy == "autosave" else {}),
            **intent_options(self.settle_reserved_notice, self.timeout),
            "component": MODE,
            "star": self.bundle["star"],
            "planet_class": self.bundle["planet_class"],
            "phase": self._phase,
            "finished": self.finished,
            "status": "finished" if self.finished else "ready",
            "save_acknowledgement_verified": self.save_strategy == "explicit"
            and self.save_receipt is not None,
            **(
                {"visible_readback_verified": self.save_receipt is not None}
                if self.save_strategy == "autosave"
                else {}
            ),
            "save_dispatch_attempts_recorded": int(self._dispatch_recorded),
            "save_may_have_occurred": self._dispatch_recorded,
            "task_completed": self._verified,
            "automatic_retry": False,
            "optimizer_updates": 0,
        }

    def advance(self):
        if self.finished:
            return self.state()
        _require(not self._advancing, "reentrant_advance")
        self._advancing = True
        try:
            self._check()
            if self._phase == "readback":
                _read_planet(self.page, self.config, self.output / "initial-readback", self.bundle)
                self._check()
                self._phase = "save"
            elif self._phase == "save" and self.save_strategy == "autosave":
                self._emit("state", {**self.state(), "scheduled_call": "autosave_visible_readback"})
                self._check()
                self.save_receipt = _positive_autosave(
                    self.page,
                    self.config,
                    self.output / "save",
                    history=self.book.history,
                    directories=self.directories,
                    supplied_inputs=self.supplied_inputs,
                    check=self._check,
                    timeout_seconds=self.timeout,
                )
                self._emit("observation", self.save_receipt)
                self._check()
                self._phase = "verify"
            elif self._phase == "save":
                self._emit(
                    "action_proposed",
                    {
                        "kind": "CLICK",
                        "visible_label": "Save",
                        "action_source": "guarded_setup_not_learned",
                        "maximum_native_dispatches": 1,
                    },
                )
                self._check()
                self.save_receipt = _strict_save(
                    self.page,
                    self.config,
                    self.output / "save",
                    book=self.book,
                    bundle=self.bundle,
                    owner_output=self.output,
                    class_sha=self.class_sha,
                    check=self._check,
                    quick_check=self._quick_check,
                    timeout_seconds=self.timeout,
                    settle_timeout_seconds=self.settle_timeout,
                    settle_reserved_notice=self.settle_reserved_notice,
                )
                self._dispatch_recorded = True
                self._emit("action_result", self.save_receipt)
                self._check()
                self._phase = "verify"
            elif self._phase == "verify":
                self.workflow_receipt = verify_positive_planet_workflow(
                    self.page,
                    self.config,
                    self.output / "workflow",
                    run_history=self.book.history,
                    save_dir=self.output / "save",
                    **self.directories,
                    **source_options(self.supplied_inputs),
                )
                self._check()
                validate_autosave_workflow_flags(
                    self.workflow_receipt, enabled=self.save_strategy == "autosave"
                )
                _require(
                    self.workflow_receipt.get("task_completed") is True
                    and self.workflow_receipt.get("mode") == positive_mode(self.supplied_inputs)
                    and self.workflow_receipt.get("planet", {}).get("value") == self.bundle["planet_class"],
                    "workflow_not_verified",
                )
                raw = self.book.read(self.output / "workflow/confirmed.json")
                _require(json.loads(raw) == self.workflow_receipt, "workflow_receipt_mismatch")
                self._workflow_sha = hashlib.sha256(raw).hexdigest()
                self._finish("positive_workflow_verified", True)
            if not self.finished:
                self._emit("state", self.state())
                self._check()
        except BaseException as exc:
            self._stop(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            self._advancing = False
        return self.state()

    def _stop(self, exc):
        if not self.finished:
            reason = str(exc) if isinstance(exc, BrowserSafetyStop) else "positive_finalization_failed"
            try:
                self._emit("error", {"reason": reason, "exception_type": type(exc).__name__})
            finally:
                self._finish(reason, False)

    def _finish(self, outcome, verified):
        if self.finished:
            return
        self.finished, self._phase, self._finalizing = True, "finished", True
        self._verified = verified
        self._dispatch_recorded = (self.output / "save/dispatch.json").exists()
        self.report = {
            **source_options(self.supplied_inputs),
            **(autosave_workflow_flags() if self.save_strategy == "autosave" else {}),
            "schema_version": 1,
            "mode": MODE,
            "star": self.bundle["star"],
            "planet_class": self.bundle["planet_class"],
            "outcome": outcome,
            "task_completed": verified,
            "save_acknowledgement_verified": self.save_strategy == "explicit"
            and self.save_receipt is not None,
            "save_dispatch_attempts_recorded": int(self._dispatch_recorded),
            "save_may_have_occurred": self._dispatch_recorded,
            "save_click_delivery_confirmed": self.save_strategy == "explicit"
            and self.save_receipt is not None,
            "maximum_save_dispatches": 1,
            "workflow_dir": str((self.output / "workflow").relative_to(self.book.history))
            if verified
            else None,
            "workflow_sha256": self._workflow_sha if verified else None,
            "source_sha256": self.source_hashes,
            "planet_class_sha256": self.class_sha,
            "automatic_retry": False,
            "optimizer_updates": 0,
            "scientific_verified": False,
            "project_completed": False,
            "assessed": False,
            "score_updated": False,
            "submitted": False,
            "journal_writes": 0,
        }
        try:
            self._emit("episode_summary", self.report)
            if verified:
                try:
                    self._check()
                except BrowserSafetyStop as exc:
                    outcome, self._verified = str(exc), False
                    self.report.update(
                        outcome=outcome, task_completed=False, workflow_dir=None, workflow_sha256=None
                    )
                    self._emit("error", {"reason": outcome, "exception_type": "BrowserSafetyStop"})
                    self._emit("episode_summary", self.report)
            self._stream.close()
            self.report["events_sha256"] = hashlib.sha256(
                (self.output / "events.jsonl").read_bytes()
            ).hexdigest()
            persist_json(self.output / "report.json", self.report)
            if not self._verified:
                persist_json(
                    self.output / "stopped.json",
                    {"reason": outcome, "automatic_retry": False, "task_completed": False},
                )
        except BaseException:
            self._verified = False
            self.report.update(
                outcome="positive_finalization_artifact_failed",
                task_completed=False,
                workflow_dir=None,
                workflow_sha256=None,
                artifact_finalization_failed=True,
            )
            self._stream.close()
            try:
                persist_json(self.output / "finalization_failed.json", self.report)
            except OSError:
                pass
            raise
        finally:
            self._finalizing = False

    def abort(self):
        if not self.finished or self._finalizing:
            self._cancelled = True
            if not self._advancing and not self._finalizing:
                self._stop(BrowserSafetyStop("positive_finalization_operator_aborted"))
        return self.state()

    def close(self):
        return self.abort()
