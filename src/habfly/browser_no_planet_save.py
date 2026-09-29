"""One Save acknowledgement for a confirmed bounded-window No hypothesis.

Only three raw answer fields are visible when No is selected; those must remain
blank and the four conditional derived fields must remain absent from the visible
contract. Hidden values are never read and are NOT certified blank. The choice
and Save are reference/transport evidence, not scientific or course completion,
cross-session persistence, assessment, score transfer, or submission.
"""

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_numeric import screen_identity
from .browser_planet import BASE_FIELDS, load_planet_capture
from .browser_planet_chart import VISIBLE_TOOLTIP
from .browser_planet_numeric import PlanetNumericSession, planet_projection
from .browser_planet_save import save_control_exposed
from .browser_planet_window_choice import MODE as CHOICE_MODE
from .browser_planet_window_choice import _evidence, _same_context, recorded_policy_manifest
from .browser_probe import save_probe
from .browser_stellar import SIMULATION_URL

MODE = "bounded_window_no_planet_save_acknowledgement"
RAW_FIELDS = ("line_shift", "brightness_drop", "period_days")
CONDITIONAL_FIELDS = ("orbital_radius", "planet_mass", "planet_radius", "planet_density")
ZERO_FIELDS = (
    "numeric_writes",
    "class_writes",
    "na_writes",
    "assessment_clicks",
    "save_clicks",
    "submission_clicks",
)
FALSE_FIELDS = (
    "class_selection_verified",
    "absence_proven",
    "scientific_verified",
    "correctness_verified",
    "training_label",
    "learned_perception",
    "task_completed",
    "automatic_retry",
)


def no_planet_visit_equivalence(original, original_mapping, current, current_mapping):
    """Compare historical No answers after a revisit, not within a live guard.

    Only the observed full-to-5000-day view reset, enabling the second header
    navigation button, and the AX-absent collected-star tooltip may differ.
    This does not certify current chart measurements, science or persistence.
    Fresh session.current() comparisons deliberately remain fully strict.
    """
    reason = "no_planet_autosave_current_state_mismatch"
    for mapping in (original_mapping, current_mapping):
        _blank_no(mapping)
    if {k: v for k, v in original_mapping.items() if k != "capture_sha256"} != {
        k: v for k, v in current_mapping.items() if k != "capture_sha256"
    }:
        raise BrowserSafetyStop(reason)
    old, new = planet_projection(original, original_mapping), planet_projection(current, current_mapping)
    if old == new:
        return None  # Legacy, same-view receipts require no exception descriptor.
    frames = [[f for f in report["frames"] if f["url"] == SIMULATION_URL] for report in (old, new)]
    if any(len(items) != 1 for items in frames):
        raise BrowserSafetyStop(reason)
    left, right = (items[0] for items in frames)
    changes = []
    star = original_mapping["star_name"]
    header = [("img", None), ("img", None), ("button", None)]
    tail = [("text", star), ("img", "1"), ("img", "2"), ("img", "3"), ("img", None)]
    known_headers = all(
        f["accessibility"][:3] == header
        and f["accessibility"][4:9] == tail
        and f["accessibility"][3] in {("button", None), ("button [disabled]", None)}
        for f in (left, right)
    )
    if known_headers and left["accessibility"][3] != right["accessibility"][3]:
        a, b = left["controls"][1], right["controls"][1]
        if (
            left["accessibility"][3] != ("button [disabled]", None)
            or right["accessibility"][3] != ("button", None)
            or a.get("role") != "button"
            or a.get("enabled") is not False
            or b.get("enabled") is not True
            or a.get("accessibility") != "- button [disabled]"
            or b.get("accessibility") != "- button"
            or {**a, "enabled": True, "accessibility": "- button"} != b
        ):
            raise BrowserSafetyStop(reason)
        changes.append({"view": "second_header_navigation_enabled", "before": False, "after": True})
        left["accessibility"][3] = right["accessibility"][3]
        left["controls"][1] = b.copy()
    # Exact observed axis glyphs, not arbitrary ranges, labels, pans or flux zooms.
    flux = [str(v) for v in range(0, 101, 10)]
    labels = [[str(v) for v in range(0, maximum + 1, maximum // 10)] + flux for maximum in (10000, 5000)]
    names = ["Normalized Flux Days Observed " + " ".join(values) for values in labels]
    charts = [
        [i for i, atom in enumerate(f["accessibility"]) if atom == ("img", name)]
        for f, name in zip((left, right), names)
    ]
    if all(len(indices) == 1 for indices in charts) and charts[0] == charts[1]:
        raw = ["NORMALIZED FLUX\nDAYS OBSERVED\n" + "\n".join(values) for values in labels]
        if any(f["text"].count(text) != 1 for f, text in zip((left, right), raw)):
            raise BrowserSafetyStop(reason)
        changes.append({"view": "known_time_axis_reset", "before": [0, 10000], "after": [0, 5000]})
        left["accessibility"][charts[0][0]] = right["accessibility"][charts[1][0]]
        left["text"] = left["text"].replace(raw[0], raw[1], 1)
    if known_headers:
        pattern = rf"^{re.escape(star.upper())}\n1\n2\n3\n([A-Z][A-Z -]{{0,79}})\nSTAR COLLECTED\nVIEW STAR DATA\n(?=OBSERVATIONS\n)"
        tips = [re.match(pattern, f["text"]) for f in (left, right)]
        if all(tips) and tips[0][1] != tips[1][1]:
            # Never discard an exposed tooltip or another AX current-star label.
            for f in (left, right):
                if any(
                    word in json.dumps(f["accessibility"]).upper()
                    for word in ("STAR COLLECTED", "VIEW STAR DATA")
                ):
                    raise BrowserSafetyStop(reason)
            changes.append({"view": "ax_absent_collected_tooltip", "before": tips[0][1], "after": tips[1][1]})
            left["text"] = tips[1][0] + left["text"][tips[0].end() :]
    if old != new or not changes:
        raise BrowserSafetyStop(reason)
    return {
        "schema_version": 1,
        "mode": "same_no_planet_answers_known_revisit_view_v1",
        "original_screen_sha256": screen_identity(original),
        "current_screen_sha256": screen_identity(current),
        "differences": changes,
        "current_chart_measurements_verified": False,
        "scientific_verified": False,
    }


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_choice(path, checksum, history):
    """Validate the immutable choice chain, not merely its success flag."""
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise BrowserSafetyStop("invalid_no_planet_choice_hash")
    try:
        raw = path.read_bytes()
        if _sha(raw) != checksum:
            raise BrowserSafetyStop("no_planet_choice_hash_mismatch")
        receipt = json.loads(raw)
        intent = json.loads((path.parent / "reserved.json").read_bytes())
        if not isinstance(receipt, dict) or not isinstance(intent, dict):
            raise BrowserSafetyStop("invalid_no_planet_choice_receipt")
        expected_keys = {
            "schema_version",
            "mode",
            "star",
            "kind",
            "value",
            "provenance",
            "action_source",
            "policy",
            "examined_day_interval",
            "observation_limit_days",
            "saved_evidence",
            "fresh_evidence",
            "output",
            "source",
            "preserved_painted_class",
            "max_presence_writes",
            *ZERO_FIELDS,
            *FALSE_FIELDS,
        }
        if (
            set(intent) != expected_keys
            or intent["schema_version"] != 1
            or intent["mode"] != CHOICE_MODE
            or intent["kind"] != "SELECT"
            or intent["value"] != "No"
            or intent["provenance"] != "reference_prediction"
            or intent["action_source"] != "user_approved_window_reference_not_learned"
            or intent["policy"] != recorded_policy_manifest(intent["policy"])
            or intent["examined_day_interval"] != [0, 5000]
            or type(intent["observation_limit_days"]) is not int
            or intent["observation_limit_days"] != 5000
            or type(intent["max_presence_writes"]) is not int
            or intent["max_presence_writes"] != 1
            or any(type(intent[key]) is not int or intent[key] != 0 for key in ZERO_FIELDS)
            or any(intent[key] is not False for key in FALSE_FIELDS)
            or receipt.get("readback_verified") is not True
            or type(receipt.get("inherited_paint_cleared")) is not bool
            or receipt.get("painted_class_after") not in {None, "gas_giant", "ice_giant", "terrestrial"}
            or intent["preserved_painted_class"] not in {None, "gas_giant", "ice_giant", "terrestrial"}
            or receipt
            != {
                **intent,
                **{
                    key: receipt[key]
                    for key in (
                        "preselect_evidence",
                        "readback_verified",
                        "painted_class_after",
                        "inherited_paint_cleared",
                    )
                },
            }
            or not isinstance(intent["star"], str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", intent["star"])
            or (path.parent / "stopped.json").exists()
            or (path.parent / "invalidated.json").exists()
        ):
            raise BrowserSafetyStop("unsupported_no_planet_choice_receipt")
        expected_cleared = (
            intent["preserved_painted_class"] is not None and receipt["painted_class_after"] is None
        )
        if receipt["inherited_paint_cleared"] != expected_cleared or receipt["painted_class_after"] not in {
            None,
            intent["preserved_painted_class"],
        }:
            raise BrowserSafetyStop("inconsistent_no_planet_choice_paint")
        if intent["output"] != str(path.parent.resolve().relative_to(history.resolve())):
            raise BrowserSafetyStop("no_planet_choice_output_mismatch")
        source = (history / intent["source"]).resolve()
        if not source.is_relative_to(history.resolve()) or source.name != "report.json":
            raise BrowserSafetyStop("no_planet_choice_source_outside_history")
        key = _sha(intent["star"].casefold().encode())
        claim = history / "planet-window-choice-reservations" / f"{key}.json"
        if json.loads(claim.read_bytes()) != intent:
            raise BrowserSafetyStop("no_planet_choice_reservation_mismatch")
        records = (
            ("saved_evidence", source),
            ("fresh_evidence", path.parent / "fresh-progress/report.json"),
            ("preselect_evidence", path.parent / "preselect-progress/report.json"),
        )
        for label, report_path in records:
            policy_options = (
                {"policy": intent["policy"]} if intent["policy"] != recorded_policy_manifest() else {}
            )
            evidence = _evidence(report_path, receipt[label]["report_sha256"], **policy_options)
            if evidence != receipt[label] or evidence["star"].casefold() != intent["star"].casefold():
                raise BrowserSafetyStop("no_planet_choice_evidence_mismatch")
            _same_context(receipt["saved_evidence"], evidence)
        if json.loads((path.parent / "preselect.json").read_bytes()) != receipt["preselect_evidence"]:
            raise BrowserSafetyStop("no_planet_choice_preselect_mismatch")
        mapping = load_planet_capture(path.parent / "after")
        capture = json.loads((path.parent / "after/observation.json").read_bytes())
        if mapping["star_name"].casefold() != intent["star"].casefold():
            raise BrowserSafetyStop("no_planet_choice_capture_star_mismatch")
        _blank_no(mapping)
        return receipt, capture, mapping
    except (OSError, ValueError, KeyError, TypeError):
        raise BrowserSafetyStop("unreadable_no_planet_choice_evidence") from None


def _blank_no(mapping):
    values = mapping["observation"]["values"]
    fields = values["browser_field_map"]
    if values["has_planet"] != "No" or set(fields) != BASE_FIELDS:
        raise BrowserSafetyStop("no_planet_visible_branch_required")
    if any(fields[name]["current_value"] != "" for name in RAW_FIELDS):
        raise BrowserSafetyStop("no_planet_raw_answers_must_remain_blank")


def _button(session, *, allow_disabled=False):
    button = session.frame.get_by_role("button", name="Save", exact=True)
    if (
        button.count() != 1
        or not button.is_visible()
        or not save_control_exposed(button)
        or not allow_disabled
        and not button.is_enabled()
    ):
        raise BrowserSafetyStop("no_planet_save_control_unavailable")
    return button


def _notice(session, handle):
    notices = [n for n in session.frame.get_by_text("Data saved", exact=True).all() if n.is_visible()]
    if not notices:
        return False
    if (
        len(notices) != 1
        or not notices[0].evaluate(
            "(e,b)=>b.parentElement.contains(e)&&b.parentElement.getBoundingClientRect().height<=64", handle
        )
        or not notices[0].evaluate(VISIBLE_TOOLTIP)
    ):
        raise BrowserSafetyStop("unverified_no_planet_save_acknowledgement")
    return True


def _record_notice_probe(session, handle, directory, stage, compared_report):
    """Retain the actual lightweight pre-click decision, not a later snapshot.

    Full capture can outlast this footer. These records explain a guard stop;
    they never authorize a Save, acknowledge a dispatch or permit a retry.
    The linked full snapshot is explicitly older than this separate probe.
    """
    probes = directory / "footer-probes"
    probes.mkdir(exist_ok=True)
    started = time.monotonic()
    record = {
        "schema_version": 1,
        "mode": "no_planet_preclick_footer_diagnostic",
        "stage": stage,
        "source": "ordinary_visible_footer_probe",
        "compared_capture_sha256": screen_identity(compared_report),
        "compared_capture_is_simultaneous": False,
        "probe_started_monotonic_seconds": started,
        "diagnostic_only": True,
        "fresh_save_acknowledgement_verified": False,
        "save_click_dispatched": False,
        "retry_authorized": False,
        "task_completed": False,
    }
    try:
        present = _notice(session, handle)
    except BaseException as exc:
        persist_json(
            probes / f"{stage}.json",
            {
                **record,
                "probe_finished_monotonic_seconds": time.monotonic(),
                "verdict": "rejected",
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "footer_probe_failed",
                "notice_present": None,
            },
        )
        raise
    persist_json(
        probes / f"{stage}.json",
        {
            **record,
            "probe_finished_monotonic_seconds": time.monotonic(),
            "verdict": "present" if present else "absent",
            "notice_present": present,
            "visible_text": "Data saved" if present else None,
            "same_footer_and_exposure_verified": present,
        },
    )
    return present


def _poll_button(session, handle):
    """Lightweight, read-only identity checks for transient footer polling."""
    page = session.page
    page.wait_for_timeout(0)
    if session.unexpected_dialog:
        raise BrowserSafetyStop("unexpected_browser_dialog")
    if page.frames != session.frames or len(page.context.pages) != 1:
        raise BrowserSafetyStop("no_planet_save_context_changed")
    button = _button(session, allow_disabled=True)
    if not handle.evaluate("(a,b)=>a.isConnected&&a===b", button.element_handle(timeout=2000)):
        raise BrowserSafetyStop("no_planet_save_control_replaced")
    return button


def _check_save_cancelled(cancelled):
    try:
        value = cancelled()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001 - caller errors may contain private data
        raise BrowserSafetyStop("no_planet_save_cancellation_check_failed") from None
    if type(value) is not bool:
        raise BrowserSafetyStop("no_planet_save_cancellation_check_failed")
    if value:
        raise BrowserSafetyStop("no_planet_save_cancelled")


def _settle_before_reservation(
    session, handle, directory, initial_report, *, deadline, source, choice_sha256, history, choice, cancelled
):
    """Repeat only read-only preflight, never a reserved or dispatched Save.

    A notice can reappear while expensive guards run. Recheck after those
    guards, without weakening their comparisons. This improves the fresh
    precondition but cannot guarantee that no later notice will appear.
    """
    latest = initial_report
    for index in range(301):
        _check_save_cancelled(cancelled)
        if time.monotonic() >= deadline:
            break
        button = _poll_button(session, handle)
        present = _record_notice_probe(
            session,
            handle,
            directory,
            "before-reservation" if index == 0 else f"settle-wait-{index:03d}",
            latest,
        )
        if not present and button.is_enabled():
            if _load_choice(source, choice_sha256, history)[0] != choice:
                raise BrowserSafetyStop("no_planet_choice_changed_before_save")
            _check_save_cancelled(cancelled)
            candidate = session.current()
            latest, mapping, _, _ = candidate
            _blank_no(mapping)
            save_probe(latest, directory / "pre-reservation-observations" / f"{index:03d}")
            _check_save_cancelled(cancelled)
            button = _poll_button(session, handle)
            present = _record_notice_probe(session, handle, directory, f"settle-final-{index:03d}", latest)
            _check_save_cancelled(cancelled)
            if time.monotonic() >= deadline:
                break
            if not present and button.is_enabled():
                persist_json(
                    directory / "pre-reservation-settled.json",
                    {
                        "schema_version": 1,
                        "scope": "bounded_read_only_fresh_no_save_precondition",
                        "read_only_cycles": index + 1,
                        "compared_capture_sha256": screen_identity(latest),
                        "choice_sha256": choice_sha256,
                        "last_footer_probe": f"footer-probes/settle-final-{index:03d}.json",
                        "notice_present": False,
                        "save_enabled": True,
                        "reservation_created": False,
                        "save_click_dispatched": False,
                        "fresh_save_acknowledgement_verified": False,
                        "later_notice_excluded": False,
                        "task_completed": False,
                    },
                )
                return candidate
        _check_save_cancelled(cancelled)
        session.page.wait_for_timeout(100)
    raise BrowserSafetyStop("no_planet_save_preflight_timeout")


def _reserved_phase_check(deadline, cancelled):
    _check_save_cancelled(cancelled)
    if time.monotonic() >= deadline:
        raise BrowserSafetyStop("no_planet_save_reserved_phase_timeout")


def _check_dispatch_readback_budget(session, directory, deadline):
    """Do not dispatch when recent full reads already predict a late finish.

    This admission estimate reserves the existing maximum three-second click
    plus the slowest full guarded read observed in this session. This is a
    minimum admission estimate, not a bound on acknowledgement/persistence or
    future latency. It cannot guarantee a successful finish;
    the original absolute deadline still governs every post-click check.
    """
    longest = session.longest_current_seconds
    remaining = deadline - time.monotonic()
    if type(longest) not in {int, float} or not math.isfinite(longest) or longest <= 0:
        raise BrowserSafetyStop("no_planet_save_invalid_readback_timing")
    required = 3.0 + longest
    if not remaining > required:
        persist_json(
            directory / "dispatch-budget-rejected.json",
            {
                "schema_version": 1,
                "policy": "observed_readback_dispatch_admission_v1",
                "longest_guarded_read_seconds": longest,
                "remaining_seconds": remaining,
                "required_seconds": required,
                "click_allowance_seconds": 3.0,
                "readback_multiplier": 1,
                "deadline_monotonic_seconds": deadline,
                "estimate_not_guarantee": True,
                "deadline_extended": False,
                "save_click_dispatched": False,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        raise BrowserSafetyStop("no_planet_save_insufficient_readback_budget")


def _settle_reserved_notice(session, directory, handle, *, deadline, cancelled, check_sources, remember):
    """One fresh intent, before its first dispatch; only footer waiting repeats.

    Each observed absence is followed by the original full native guard and a
    new immediate footer probe. Failed outputs never enter this helper again.
    Neither a preexisting notice nor its disappearance acknowledges our Save.
    """
    checks, probes = 0, 0
    while True:
        _reserved_phase_check(deadline, cancelled)
        check_sources()
        latest, mapping, _, _ = session.current()
        remember(latest)
        _blank_no(mapping)
        capture_name = f"reserved-observations/{checks:03d}"
        save_probe(latest, directory / capture_name)
        checks += 1
        _reserved_phase_check(deadline, cancelled)
        check_sources()
        button = _poll_button(session, handle)
        stage = f"reserved-final-{checks - 1:03d}"
        present = _record_notice_probe(session, handle, directory, stage, latest)
        _reserved_phase_check(deadline, cancelled)
        if not present:
            if not button.is_enabled():
                raise BrowserSafetyStop("no_planet_save_control_unavailable")
            save_probe(latest, directory / "pre-dispatch-observation")
            persist_json(
                directory / "reserved-notice-settled.json",
                {
                    "schema_version": 1,
                    "policy": "bounded_read_only_pre_dispatch_settle_v1",
                    "full_revalidations": checks,
                    "waiting_footer_probes": probes,
                    "last_capture": capture_name,
                    "last_footer_probe": f"footer-probes/{stage}.json",
                    "compared_capture_sha256": screen_identity(latest),
                    "deadline_monotonic_seconds": deadline,
                    "notice_present": False,
                    "reservation_retained": True,
                    "save_click_dispatched": False,
                    "fresh_save_acknowledgement_verified": False,
                    "later_notice_excluded": False,
                    "automatic_retry": False,
                    "task_completed": False,
                },
            )
            _reserved_phase_check(deadline, cancelled)
            return latest
        while present:
            _reserved_phase_check(deadline, cancelled)
            check_sources()
            session.page.wait_for_timeout(100)
            _reserved_phase_check(deadline, cancelled)
            _poll_button(session, handle)
            stage = f"reserved-wait-{probes:03d}"
            present = _record_notice_probe(session, handle, directory, stage, latest)
            probes += 1
            _reserved_phase_check(deadline, cancelled)


def _wait_for_save_acknowledgement(
    session, directory, handle, deadline, *, strict_deadline=False, cancelled=lambda: False
):
    # A full snapshot can outlast the entire painted acknowledgement. Persist
    # a fresh observed notice first, then validate all guarded visible state.
    while True:
        if strict_deadline:
            _reserved_phase_check(deadline, cancelled)
        button = _poll_button(session, handle)
        if _notice(session, handle) and button.is_enabled():
            if strict_deadline:
                _reserved_phase_check(deadline, cancelled)
            persist_json(
                directory / "acknowledgement.json",
                {
                    "visible_text": "Data saved",
                    "source": "fully_exposed_footer_text",
                    "notice_was_already_present": False,
                },
            )
            if strict_deadline:
                _reserved_phase_check(deadline, cancelled)
            after, mapping, _, _ = session.current()
            _blank_no(mapping)
            if strict_deadline:
                _reserved_phase_check(deadline, cancelled)
            return after
        if strict_deadline:
            _reserved_phase_check(deadline, cancelled)
        if time.monotonic() >= deadline:
            session.current()
            raise BrowserSafetyStop("no_planet_save_acknowledgement_timeout")
        session.page.wait_for_timeout(100)


def save_no_planet_work(
    page,
    config,
    output,
    *,
    run_history,
    choice_path,
    choice_sha256,
    timeout_seconds=20,
    settle_timeout_seconds=20,
    settle_reserved_notice=False,
    cancelled=lambda: False,
):
    """Save once, requiring a newly visible acknowledgement and unchanged work.

    The canonical per-star reservation stays even on timeout, an uncertain
    click, or a pre-click failure after reservation. New output directories
    cannot authorize another Save. Before reservation only, a separate bounded
    read-only preflight may wait for an existing footer notice to clear and
    revalidate the entire source/page. It is not a guarantee against a later
    notice: the default still stops on any unexpected post-reservation notice.
    With the explicit fresh-owner opt-in, only that known footer may settle
    before the first click. One timeout_seconds deadline covers the final guard,
    waiting, click, acknowledgement and final readback; it is never restarted.
    No hidden derived values are inspected. Neither deadline is extended.
    """
    history, directory, source = Path(run_history), Path(output), Path(choice_path)
    if (
        not history.is_dir()
        or not directory.resolve().is_relative_to(history.resolve())
        or not source.resolve().is_relative_to(history.resolve())
        or source.name != "confirmed.json"
        or type(timeout_seconds) not in {int, float}
        or not 0.1 <= timeout_seconds <= 30
        or type(settle_timeout_seconds) not in {int, float}
        or not 0.1 <= settle_timeout_seconds <= 30
        or type(settle_reserved_notice) is not bool
        or not callable(cancelled)
    ):
        raise ValueError(
            "Owned history, confirmed choice, bounded deadlines and a cancellation callback are required"
        )
    directory.mkdir(parents=True, exist_ok=False)
    session, last_report, reserved, attempted = None, None, False, False
    try:
        settle_deadline = time.monotonic() + settle_timeout_seconds
        _check_save_cancelled(cancelled)
        choice, choice_capture, choice_mapping = _load_choice(source, choice_sha256, history)
        key = _sha(choice["star"].casefold().encode())
        reservation = history / "no-planet-save-reservations" / f"{key}.json"
        if reservation.exists():
            raise BrowserSafetyStop("no_planet_save_already_reserved")
        session = PlanetNumericSession(
            page, config, directory / "read-guard", max_seconds=120, _allow_no_planet=True, _pin_controls=True
        )
        before, mapping, choices, _ = session.current()
        last_report = before
        _blank_no(mapping)
        if (
            mapping["star_name"].casefold() != choice["star"].casefold()
            or choices["selected"] != choice["painted_class_after"]
            or planet_projection(before, mapping) != planet_projection(choice_capture, choice_mapping)
        ):
            raise BrowserSafetyStop("no_planet_choice_current_state_mismatch")
        button = _button(session, allow_disabled=True)
        handle = button.element_handle(timeout=2000)
        before, mapping, choices, _ = _settle_before_reservation(
            session,
            handle,
            directory,
            before,
            deadline=settle_deadline,
            source=source,
            choice_sha256=choice_sha256,
            history=history,
            choice=choice,
            cancelled=cancelled,
        )
        last_report = before
        _check_save_cancelled(cancelled)
        save_probe(before, directory / "before")
        intent = {
            "schema_version": 1,
            "mode": MODE,
            "star": mapping["star_name"],
            "kind": "CLICK",
            "visible_label": "Save",
            "choice_sha256": choice_sha256,
            "choice_path": str(source.resolve().relative_to(history.resolve())),
            "before_sha256": screen_identity(before),
            "output": str(directory.resolve().relative_to(history.resolve())),
            "policy": choice["policy"],
            "has_planet": "No",
            "max_save_clicks": 1,
            "visible_blank_raw_fields": list(RAW_FIELDS),
            "conditionally_absent_derived_fields": list(CONDITIONAL_FIELDS),
            "hidden_values_inspected": False,
            "hidden_values_blank_verified": False,
            "numeric_writes": 0,
            "class_writes": 0,
            "habitability_writes": 0,
            "na_writes": 0,
            "assessment_clicks": 0,
            "score_transfer_clicks": 0,
            "submission_clicks": 0,
            "task_completed": False,
            "automatic_retry": False,
        }
        if settle_reserved_notice:
            intent.update(
                settle_reserved_notice=True,
                reserved_notice_policy="bounded_read_only_pre_dispatch_settle_v1",
                reserved_phase_timeout_seconds=timeout_seconds,
            )
        reservation.parent.mkdir(exist_ok=True)
        _check_save_cancelled(cancelled)
        if time.monotonic() >= settle_deadline:
            raise BrowserSafetyStop("no_planet_save_preflight_timeout")
        try:
            persist_json(reservation, intent)
        except FileExistsError:
            raise BrowserSafetyStop("no_planet_save_already_reserved") from None
        reserved = True
        for folder in (reservation.parent, history):
            _sync_directory(folder)
        persist_json(directory / "reserved.json", intent)
        _check_save_cancelled(cancelled)
        reserved_deadline = time.monotonic() + timeout_seconds if settle_reserved_notice else None
        if settle_reserved_notice:
            reservation_bytes = {
                path: path.read_bytes() for path in (reservation, directory / "reserved.json")
            }
            if any(
                path.is_symlink()
                or json.dumps(json.loads(raw), sort_keys=True, allow_nan=False)
                != json.dumps(intent, sort_keys=True, allow_nan=False)
                for path, raw in reservation_bytes.items()
            ):
                raise BrowserSafetyStop("no_planet_save_reserved_intent_changed")
            pinned_reservations = {path: _sha(raw) for path, raw in reservation_bytes.items()}

            def check_sources():
                if any(
                    path.is_symlink() or _sha(path.read_bytes()) != checksum
                    for path, checksum in pinned_reservations.items()
                ) or any(
                    (directory / name).exists()
                    for name in ("dispatch.json", "stopped.json", "confirmed.json", "acknowledgement.json")
                ):
                    raise BrowserSafetyStop("no_planet_save_reserved_intent_changed")
                if _load_choice(source, choice_sha256, history)[0] != choice:
                    raise BrowserSafetyStop("no_planet_choice_changed_before_save")

            def remember(report):
                nonlocal last_report
                last_report = report

            last_report = _settle_reserved_notice(
                session,
                directory,
                handle,
                deadline=reserved_deadline,
                cancelled=cancelled,
                check_sources=check_sources,
                remember=remember,
            )
            check_sources()
            _reserved_phase_check(reserved_deadline, cancelled)
        else:
            last_report, _, _, _ = session.current()
            save_probe(last_report, directory / "pre-dispatch-observation")
            if _load_choice(source, choice_sha256, history)[0] != choice:
                raise BrowserSafetyStop("no_planet_choice_changed_before_save")
            if not handle.evaluate(
                "(a,b)=>a.isConnected&&a===b", _button(session).element_handle(timeout=2000)
            ):
                raise BrowserSafetyStop("no_planet_save_control_replaced")
            if _record_notice_probe(session, handle, directory, "after-reservation", last_report):
                raise BrowserSafetyStop("no_planet_save_stale_acknowledgement")
        _check_save_cancelled(cancelled)
        if settle_reserved_notice:
            _reserved_phase_check(reserved_deadline, cancelled)
            _check_dispatch_readback_budget(session, directory, reserved_deadline)
        attempted = True
        persist_json(directory / "dispatch.json", {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1})
        _sync_directory(directory)
        if settle_reserved_notice:
            _reserved_phase_check(reserved_deadline, cancelled)
            # The final cancellation/source callback and durable marker writes
            # may consume the previously admitted headroom. Recheck after them,
            # before the native click, without another callback in between.
            # A retained dispatch marker still means a conservative uncertain
            # attempt; it is never erased or used to authorize a retry.
            _check_dispatch_readback_budget(session, directory, reserved_deadline)
        click_timeout = 3000
        if settle_reserved_notice:
            remaining_ms = int((reserved_deadline - time.monotonic()) * 1000)
            if remaining_ms < 1:
                raise BrowserSafetyStop("no_planet_save_reserved_phase_timeout")
            click_timeout = min(click_timeout, remaining_ms)
        handle.click(timeout=click_timeout)
        after = _wait_for_save_acknowledgement(
            session,
            directory,
            handle,
            reserved_deadline if settle_reserved_notice else time.monotonic() + timeout_seconds,
            **({"strict_deadline": True, "cancelled": cancelled} if settle_reserved_notice else {}),
        )
        last_report = after
        save_probe(after, directory / "after")
        _load_choice(source, choice_sha256, history)
        if settle_reserved_notice:
            if any(_sha(path.read_bytes()) != checksum for path, checksum in pinned_reservations.items()):
                raise BrowserSafetyStop("no_planet_save_reserved_intent_changed")
            _reserved_phase_check(reserved_deadline, cancelled)
        receipt = {
            **intent,
            "save_click_delivered": True,
            "data_saved_notice_observed": True,
            "notice_was_already_present": False,
            "answers_unchanged": True,
            "duration_and_class_paint_unchanged": True,
            "cross_session_persistence_verified": False,
            "correctness_verified": False,
            "course_completion_verified": False,
            "assessed": False,
            "score_updated": False,
            "submitted": False,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        if last_report is not None:
            save_probe(last_report, directory / "last-verified-observation")
        persist_json(
            directory / "stopped.json",
            {
                "mode": MODE,
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "no_planet_save_failed",
                "reservation_created": reserved,
                "save_may_have_occurred": attempted,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("no_planet_save_failed") from None
    finally:
        if session is not None:
            session.close()


def _predispatch_source(failed, history):
    """Only the known stale-banner guard proves this reservation did not click."""
    try:
        stopped_path, reserved_path = failed / "stopped.json", failed / "reserved.json"
        stopped, intent = json.loads(stopped_path.read_bytes()), json.loads(reserved_path.read_bytes())
        if (
            stopped
            != {
                "mode": MODE,
                "reason": "no_planet_save_stale_acknowledgement",
                "reservation_created": True,
                "save_may_have_occurred": False,
                "automatic_retry": False,
                "task_completed": False,
            }
            or (failed / "confirmed.json").exists()
            or (failed / "acknowledgement.json").exists()
            or (failed / "dispatch.json").exists()
        ):
            raise BrowserSafetyStop("unsupported_no_planet_predispatch_disposition")
        if (
            stopped.get("reservation_created") is not True
            or stopped.get("save_may_have_occurred") is not False
            or stopped.get("automatic_retry") is not False
            or stopped.get("task_completed") is not False
        ):
            raise BrowserSafetyStop("unsupported_no_planet_predispatch_disposition")
        choice_path = (history / intent["choice_path"]).resolve()
        if not choice_path.is_relative_to(history.resolve()) or choice_path.name != "confirmed.json":
            raise BrowserSafetyStop("no_planet_choice_source_outside_history")
        choice, choice_capture, choice_mapping = _load_choice(choice_path, intent["choice_sha256"], history)
        claim = history / "no-planet-save-reservations" / f"{_sha(choice['star'].casefold().encode())}.json"
        mapping = load_planet_capture(failed / "before")
        capture = json.loads((failed / "before/observation.json").read_bytes())
        if (
            json.loads(claim.read_bytes()) != intent
            or type(intent.get("schema_version")) is not int
            or intent["schema_version"] != 1
            or intent["mode"] != MODE
            or intent["kind"] != "CLICK"
            or intent["visible_label"] != "Save"
            or intent["has_planet"] != "No"
            or type(intent["max_save_clicks"]) is not int
            or intent["max_save_clicks"] != 1
            or intent["automatic_retry"] is not False
            or intent["task_completed"] is not False
            or intent.get("hidden_values_inspected") is not False
            or intent.get("hidden_values_blank_verified") is not False
            or intent.get("visible_blank_raw_fields") != list(RAW_FIELDS)
            or intent.get("conditionally_absent_derived_fields") != list(CONDITIONAL_FIELDS)
            or intent["policy"] != choice["policy"]
            or intent["output"] != str(failed.resolve().relative_to(history.resolve()))
            or intent["star"].casefold() != choice["star"].casefold()
            or mapping["star_name"].casefold() != choice["star"].casefold()
            or intent["before_sha256"] != screen_identity(capture)
            or planet_projection(capture, mapping) != planet_projection(choice_capture, choice_mapping)
            or any(
                type(intent.get(key)) is not int or intent.get(key) != 0
                for key in (
                    "numeric_writes",
                    "class_writes",
                    "habitability_writes",
                    "na_writes",
                    "assessment_clicks",
                    "score_transfer_clicks",
                    "submission_clicks",
                )
            )
        ):
            raise BrowserSafetyStop("inconsistent_no_planet_predispatch_source")
        _blank_no(mapping)
        paths = (
            stopped_path,
            reserved_path,
            claim,
            choice_path,
            failed / "before/observation.json",
            failed / "before/manifest.json",
        )
        hashes = {
            str(path.resolve().relative_to(history.resolve())): _sha(path.read_bytes()) for path in paths
        }
        return intent, choice, capture, mapping, hashes
    except (OSError, ValueError, KeyError, TypeError):
        raise BrowserSafetyStop("unreadable_no_planet_predispatch_source") from None


def resume_no_planet_save(page, config, failed_output, *, run_history, timeout_seconds=20):
    """Continue the one original Save only after its exact proven no-click stop.

    This is explicitly invoked, never an automatic retry. The stale notice must
    clear, all original evidence and visible work must remain unchanged, and a
    new acknowledgement must follow the sole native Save dispatch. The timeout
    is shared by the stale-notice wait, final guards, and acknowledgement wait.
    A durable exclusive continuation claim is consumed even if this attempt
    stops before clicking. Neither original nor continuation may be retried.
    """
    history, failed = Path(run_history), Path(failed_output)
    if (
        not history.is_dir()
        or not failed.is_dir()
        or failed.resolve() == history.resolve()
        or not failed.resolve().is_relative_to(history.resolve())
        or type(timeout_seconds) not in {int, float}
        or not 0.1 <= timeout_seconds <= 30
    ):
        raise ValueError("The original Save must belong to owned history with a 0.1..30 second deadline")
    directory, claim_path = failed / "resume", failed / "resume-reserved.json"
    if claim_path.exists() or directory.exists():
        raise BrowserSafetyStop("no_planet_save_continuation_already_reserved")
    intent, choice, original, original_mapping, hashes = _predispatch_source(failed, history)
    continued = {
        **intent,
        "output": str(directory.resolve().relative_to(history.resolve())),
        "resumed_same_intent": True,
        "source_output": str(failed.resolve().relative_to(history.resolve())),
        "source_sha256": hashes,
        "source_save_click_dispatched": False,
        "reservation_retained": True,
        "max_total_save_clicks": 1,
        "further_continuation_allowed": False,
        "timeout_seconds": timeout_seconds,
    }
    try:
        persist_json(claim_path, continued)
    except FileExistsError:
        raise BrowserSafetyStop("no_planet_save_continuation_already_reserved") from None
    _sync_directory(failed)
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError:
        raise BrowserSafetyStop("no_planet_save_continuation_already_reserved") from None
    session, last_report, attempted = None, None, False
    claim_hash = _sha(claim_path.read_bytes())

    def unchanged_sources():
        if (
            _predispatch_source(failed, history)[-1] != hashes
            or _sha(claim_path.read_bytes()) != claim_hash
            or json.loads(claim_path.read_bytes()) != continued
            or json.loads((directory / "reserved.json").read_bytes()) != continued
        ):
            raise BrowserSafetyStop("no_planet_predispatch_sources_changed")

    try:
        persist_json(directory / "reserved.json", continued)
        session = PlanetNumericSession(
            page, config, directory / "read-guard", max_seconds=120, _allow_no_planet=True
        )
        before, mapping, choices, _ = session.current()
        last_report = before
        _blank_no(mapping)
        if (
            mapping["star_name"].casefold() != choice["star"].casefold()
            or choices["selected"] != choice["painted_class_after"]
            or planet_projection(before, mapping) != planet_projection(original, original_mapping)
        ):
            raise BrowserSafetyStop("no_planet_resume_current_state_mismatch")
        save_probe(before, directory / "before")
        handle = _button(session, allow_disabled=True).element_handle(timeout=2000)
        deadline = time.monotonic() + timeout_seconds
        probe_index = 0
        while True:
            _poll_button(session, handle)
            present = _record_notice_probe(
                session, handle, directory, f"clear-wait-{probe_index:03d}", before
            )
            probe_index += 1
            if not present:
                break
            if time.monotonic() >= deadline:
                raise BrowserSafetyStop("no_planet_save_stale_acknowledgement_timeout")
            page.wait_for_timeout(100)
        persist_json(directory / "stale-notice-cleared.json", {"data_saved_notice_present": False})
        unchanged_sources()
        last_report, mapping, _, _ = session.current()
        save_probe(last_report, directory / "pre-dispatch-observation")
        _blank_no(mapping)
        if not handle.evaluate("(a,b)=>a.isConnected&&a===b", _button(session).element_handle(timeout=2000)):
            raise BrowserSafetyStop("no_planet_save_control_replaced")
        if _record_notice_probe(session, handle, directory, "after-clear-and-full-guard", last_report):
            raise BrowserSafetyStop("no_planet_save_stale_acknowledgement")
        if time.monotonic() >= deadline:
            raise BrowserSafetyStop("no_planet_save_continuation_time_limit")
        attempted = True
        persist_json(
            directory / "dispatch.json",
            {
                "kind": "CLICK",
                "visible_label": "Save",
                "max_clicks": 1,
                "resumed_same_intent": True,
                "source_output": continued["source_output"],
                "source_sha256": hashes,
            },
        )
        _sync_directory(directory)
        handle.click(timeout=3000)
        after = _wait_for_save_acknowledgement(session, directory, handle, deadline)
        last_report = after
        save_probe(after, directory / "after")
        unchanged_sources()
        receipt = {
            **continued,
            "resume_before_sha256": screen_identity(before),
            "save_click_delivered": True,
            "data_saved_notice_observed": True,
            "notice_was_already_present": False,
            "answers_unchanged": True,
            "duration_and_class_paint_unchanged": True,
            "cross_session_persistence_verified": False,
            "correctness_verified": False,
            "course_completion_verified": False,
            "assessed": False,
            "score_updated": False,
            "submitted": False,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        if last_report is not None:
            save_probe(last_report, directory / "last-verified-observation")
        persist_json(
            directory / "stopped.json",
            {
                "mode": MODE,
                "resumed_same_intent": True,
                "source_output": continued["source_output"],
                "source_sha256": hashes,
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "no_planet_save_resume_failed",
                "reservation_retained": True,
                "resume_claim_created": True,
                "save_may_have_occurred": attempted,
                "automatic_retry": False,
                "further_continuation_allowed": False,
                "task_completed": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("no_planet_save_resume_failed") from None
    finally:
        if session is not None:
            session.close()


def _reconciliation_source(failed, history):
    """Prove that neither the original intent nor any continuation dispatched.

    A continuation directory or claim is never ignored, including a partial or
    uncertain continuation. Only its exact final pre-click stale-notice stop is
    supported. These records acknowledge no historical banner themselves.
    """
    source = _predispatch_source(failed, history)
    intent, _, original, original_mapping, original_hashes = source
    resumed, claim = failed / "resume", failed / "resume-reserved.json"
    if not any(path.exists() or path.is_symlink() for path in (resumed, claim)):
        return (*source, None)
    try:
        reserved = json.loads((resumed / "reserved.json").read_bytes())
        timeout = reserved.get("timeout_seconds")
        expected_intent = {
            **intent,
            "output": str(resumed.resolve().relative_to(history.resolve())),
            "resumed_same_intent": True,
            "source_output": str(failed.resolve().relative_to(history.resolve())),
            "source_sha256": original_hashes,
            "source_save_click_dispatched": False,
            "reservation_retained": True,
            "max_total_save_clicks": 1,
            "further_continuation_allowed": False,
            "timeout_seconds": timeout,
        }
        stopped = json.loads((resumed / "stopped.json").read_bytes())
        expected_stopped = {
            "mode": MODE,
            "resumed_same_intent": True,
            "source_output": expected_intent["source_output"],
            "source_sha256": original_hashes,
            "reason": "no_planet_save_stale_acknowledgement",
            "reservation_retained": True,
            "resume_claim_created": True,
            "save_may_have_occurred": False,
            "automatic_retry": False,
            "further_continuation_allowed": False,
            "task_completed": False,
        }
        cleared = json.loads((resumed / "stale-notice-cleared.json").read_bytes())
        if (
            not resumed.resolve().is_relative_to(failed.resolve())
            or reserved != expected_intent
            or json.dumps(json.loads(claim.read_bytes()), sort_keys=True)
            != json.dumps(expected_intent, sort_keys=True)
            or type(timeout) not in {int, float}
            or not 0.1 <= timeout <= 30
            or type(reserved.get("max_total_save_clicks")) is not int
            or any(
                type(reserved[key]) is not bool
                for key in (
                    "resumed_same_intent",
                    "source_save_click_dispatched",
                    "reservation_retained",
                    "further_continuation_allowed",
                )
            )
            or stopped != expected_stopped
            or any(
                type(stopped[key]) is not bool
                for key, value in expected_stopped.items()
                if type(value) is bool
            )
            or cleared != {"data_saved_notice_present": False}
            or cleared.get("data_saved_notice_present") is not False
            or any(
                (resumed / name).exists()
                for name in ("dispatch.json", "acknowledgement.json", "confirmed.json")
            )
        ):
            raise BrowserSafetyStop("unsupported_no_planet_continuation_disposition")
        paths = [
            claim,
            resumed / "reserved.json",
            resumed / "stopped.json",
            resumed / "stale-notice-cleared.json",
        ]
        for name in ("before", "last-verified-observation"):
            mapping = load_planet_capture(resumed / name)
            capture_path = resumed / name / "observation.json"
            capture = json.loads(capture_path.read_bytes())
            _blank_no(mapping)
            if planet_projection(capture, mapping) != planet_projection(original, original_mapping):
                raise BrowserSafetyStop("no_planet_continuation_state_mismatch")
            paths.extend((capture_path, resumed / name / "manifest.json"))
        disposition = {
            "output": expected_intent["output"],
            "disposition": "stopped_before_explicit_save_after_stale_acknowledgement",
            "source_sha256": {
                str(path.resolve().relative_to(history.resolve())): _sha(path.read_bytes()) for path in paths
            },
            "save_click_dispatched": False,
            "reservation_retained": True,
            "further_continuation_allowed": False,
        }
        return (*source, disposition)
    except (OSError, ValueError, KeyError, TypeError):
        raise BrowserSafetyStop("unreadable_no_planet_continuation_disposition") from None


def reconcile_no_planet_autosave(page, config, failed_output, output, *, run_history, timeout_seconds=20):
    """Read-only review of the exact no-click stale-acknowledgement stop.

    A currently visible Data saved banner may acknowledge unchanged work, but
    its triggering site operation remains unknown. No Save is dispatched, the
    old reservation is never released, and an absent banner yields only a
    blocked disposition, not a successful acknowledgement or permission to retry.
    Both consumed claims are checked if a pre-click continuation exists. The
    read-only notice wait is bounded below 30 seconds, including time spent on
    its initial full guard. A mandatory full readback follows an observed notice
    or deadline; its duration is additional, never another polling window.
    """
    history, failed, directory = Path(run_history), Path(failed_output), Path(output)
    if (
        not history.is_dir()
        or not all(path.resolve().is_relative_to(history.resolve()) for path in (failed, directory))
        or type(timeout_seconds) not in {int, float}
        or not 0 <= timeout_seconds < 30
    ):
        raise ValueError("Reconciliation needs owned history and a read-only wait below 30 seconds")
    directory.mkdir(parents=True, exist_ok=False)
    session = None
    deadline = time.monotonic() + timeout_seconds
    try:
        intent, choice, original, original_mapping, hashes, continuation = _reconciliation_source(
            failed, history
        )

        def unchanged_sources():
            current = _reconciliation_source(failed, history)
            if current[-2:] != (hashes, continuation):
                raise BrowserSafetyStop("no_planet_predispatch_sources_changed")

        session = PlanetNumericSession(
            page, config, directory / "read-guard", max_seconds=120, _allow_no_planet=True
        )
        # The constructor has already captured the entire guarded screen. Use
        # that read rather than missing a brief notice in a redundant snapshot.
        before, mapping, choices = session.report, session.mapping, session.choices
        _blank_no(mapping)
        if (
            mapping["star_name"].casefold() != choice["star"].casefold()
            or choices["selected"] != choice["painted_class_after"]
        ):
            raise BrowserSafetyStop("no_planet_autosave_current_state_mismatch")
        revisit = no_planet_visit_equivalence(original, original_mapping, before, mapping)
        save_probe(before, directory / "before")
        disposition = {
            "schema_version": 1,
            "mode": "no_planet_predispatch_read_only_reconciliation",
            "star": mapping["star_name"],
            "source_output": str(failed.resolve().relative_to(history.resolve())),
            "source_sha256": hashes,
            "continuation_disposition": continuation,
            "notice_wait_seconds": timeout_seconds,
            "choice_path": intent["choice_path"],
            "choice_sha256": intent["choice_sha256"],
            "policy": choice["policy"],
            "source_save_click_dispatched": False,
            "reservation_retained": True,
            "blocked_before_explicit_save": True,
            "browser_actions": 0,
            "save_click_delivered": False,
            "site_autosave_trigger_verified": False,
            "numeric_writes": 0,
            "class_writes": 0,
            "habitability_writes": 0,
            "task_completed": False,
            "automatic_retry": False,
            "cross_session_persistence_verified": False,
            "correctness_verified": False,
            "course_completion_verified": False,
            "assessed": False,
            "score_updated": False,
            "submitted": False,
            "hidden_values_inspected": False,
            "hidden_values_blank_verified": False,
            "visible_blank_raw_fields": list(RAW_FIELDS),
            "conditionally_absent_derived_fields": list(CONDITIONAL_FIELDS),
        }
        if revisit is not None:
            disposition["historical_view_equivalence"] = revisit
        persist_json(directory / "disposition.json", disposition)
        handle = _button(session, allow_disabled=True).element_handle(timeout=2000)
        observed = False
        while True:
            button = _poll_button(session, handle)
            if _notice(session, handle) and button.is_enabled():
                observed = True
                break
            if time.monotonic() >= deadline:
                break
            page.wait_for_timeout(min(100, max(0, (deadline - time.monotonic()) * 1000)))
        if not observed:
            session.current()
            unchanged_sources()
            blocked = {
                **disposition,
                "outcome": "no_current_save_acknowledgement",
                "data_saved_notice_observed": False,
                "save_acknowledgement_verified": False,
            }
            persist_json(directory / "blocked.json", blocked)
            return blocked
        persist_json(
            directory / "acknowledgement.json",
            {
                "visible_text": "Data saved",
                "source": "current_fully_exposed_footer_no_explicit_save_dispatch",
                "trigger_verified": False,
                "browser_actions": 0,
            },
        )
        after, newer, _, _ = session.current()
        _blank_no(newer)
        if not handle.evaluate("(a,b)=>a.isConnected&&a===b", _button(session).element_handle(timeout=2000)):
            raise BrowserSafetyStop("no_planet_save_control_replaced")
        save_probe(after, directory / "after")
        unchanged_sources()
        receipt = {
            **disposition,
            "outcome": "current_save_acknowledgement_without_explicit_click",
            "data_saved_notice_observed": True,
            "save_acknowledgement_verified": True,
            "acknowledgement_source": "visible_footer_no_explicit_save_dispatch",
            "answers_unchanged": True,
            "duration_and_class_paint_unchanged": True,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "no_planet_autosave_review_failed",
                "browser_actions": 0,
                "save_click_delivered": False,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("no_planet_autosave_review_failed") from None
    finally:
        if session is not None:
            session.close()
