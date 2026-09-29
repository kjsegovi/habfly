"""One native No under the user's explicitly pragmatic 5,000-day policy.

Saved hashes and fresh, guarded chart crops must independently pass the raster
policy. This is a reference hypothesis, not learned perception, proven absence,
grading truth, task completion, or permission to write zero/N/A answers.
"""

import hashlib
import json
import re
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_observation_progress import capture_observation_progress, trace_progress
from .browser_planet_numeric import PlanetNumericSession
from .browser_planet_presence import presence_projection, preserved_paint_transition
from .browser_probe import save_probe
from .browser_setup import rendered_control
from .planet_window_policy import _digest
from .planet_window_policy import analyze_planet_window as frozen_analyze
from .planet_window_policy import policy_manifest as frozen_manifest

MODE = "user_approved_5000_day_planet_window_choice"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def recorded_policy_manifest(policy=None):
    """Keep the legacy decision path independent of the optional engine source."""
    frozen = frozen_manifest()
    if policy is None or policy == frozen and _digest(policy) == _digest(frozen):
        return frozen
    from .planet_window_baseline_edge import policy_manifest as edge_manifest

    edge = edge_manifest()
    if policy == edge and _digest(policy) == _digest(edge):
        return edge
    from .planet_window_baseline_band import policy_manifest as band_manifest

    band = band_manifest()
    if policy == band and _digest(policy) == _digest(band):
        return band
    from .planet_window_single_event import recorded_policy_manifest as optional_manifest

    return optional_manifest(policy)


def analyze_recorded_planet_window(png, times, fluxes, *, requested_days=5000, policy=None):
    selected = recorded_policy_manifest(policy)
    if selected == frozen_manifest():
        return frozen_analyze(png, times, fluxes, requested_days=requested_days)
    from .planet_window_baseline_edge import policy_manifest as edge_manifest

    if selected == edge_manifest():
        from .planet_window_baseline_edge import analyze_recorded_planet_window as optional_analyze
    else:
        from .planet_window_baseline_band import policy_manifest as band_manifest

        if selected == band_manifest():
            from .planet_window_baseline_band import analyze_recorded_planet_window as optional_analyze
        else:
            from .planet_window_single_event import analyze_recorded_planet_window as optional_analyze

    return optional_analyze(png, times, fluxes, requested_days=requested_days, policy=selected)


def _recorded_capture_policy(directory):
    """A sidecar is explicit new-policy evidence; its absence remains frozen v2."""
    sidecar = Path(directory) / "policy.json"
    if not sidecar.exists() and not sidecar.is_symlink():
        return recorded_policy_manifest(), None
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.stat().st_size > 32768:
        raise BrowserSafetyStop("invalid_planet_window_policy_sidecar")
    try:
        raw = sidecar.read_bytes()
        policy = recorded_policy_manifest(json.loads(raw))
    except (OSError, ValueError, TypeError):
        raise BrowserSafetyStop("invalid_planet_window_policy_sidecar") from None
    if policy == recorded_policy_manifest():
        raise BrowserSafetyStop("unexpected_legacy_planet_window_policy_sidecar")
    return policy, _sha(raw)


def recorded_capture_policy(directory):
    return _recorded_capture_policy(directory)[0]


def _evidence(path, expected_sha256, *, policy=None):
    """Recompute from saved bytes, not a caller-authored decision field."""
    if not isinstance(expected_sha256, str) or not re.fullmatch("[a-f0-9]{64}", expected_sha256):
        raise BrowserSafetyStop("invalid_planet_window_evidence_hash")
    try:
        selected_policy = recorded_policy_manifest(policy)
        captured_policy, policy_sha256 = _recorded_capture_policy(path.parent)
        if captured_policy != selected_policy:
            raise BrowserSafetyStop("planet_window_recorded_policy_mismatch")
        raw = path.read_bytes()
        if _sha(raw) != expected_sha256:
            raise BrowserSafetyStop("planet_window_report_hash_mismatch")
        report = json.loads(raw)
        png = (path.parent / "chart.png").read_bytes()
        if not isinstance(report, dict) or report.get("chart_sha256") != _sha(png):
            raise BrowserSafetyStop("planet_window_chart_hash_mismatch")
        if (
            report.get("requested_days") != 5000
            or report.get("browser_actions") != 0
            or report.get("answer_writes") != 0
            or not isinstance(report.get("star"), str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", report["star"])
            or (path.parent / "stopped.json").exists()
            or (path.parent / "invalidated.json").exists()
        ):
            raise BrowserSafetyStop("unsupported_planet_window_evidence")
        recomputed = trace_progress(png, report["time_axis_labels"], requested_days=5000)
        if any(report.get(key) != value for key, value in recomputed.items()):
            raise BrowserSafetyStop("planet_window_progress_report_disagrees")
        policy = analyze_recorded_planet_window(
            png,
            report["time_axis_labels"],
            report["flux_axis_labels"],
            requested_days=5000,
            policy=selected_policy,
        )
        if policy["status"] != "assume_no_planet" or policy["planet_decision"] != "No":
            raise BrowserSafetyStop("planet_window_does_not_support_reference_no")
        return {
            "star": report["star"],
            "report_sha256": expected_sha256,
            "chart_sha256": report["chart_sha256"],
            "time_axis_labels": report["time_axis_labels"],
            "flux_axis_labels": report["flux_axis_labels"],
            "analysis": policy,
            **({"policy_sha256": policy_sha256} if policy_sha256 is not None else {}),
        }
    except (OSError, ValueError, KeyError, TypeError):
        raise BrowserSafetyStop("unreadable_planet_window_evidence") from None


def _fresh(page, config, directory, *, policy=None):
    report = capture_observation_progress(page, config, directory, requested_days=5000)
    selected = recorded_policy_manifest(policy)
    if selected != recorded_policy_manifest():
        persist_json(directory / "policy.json", selected)
    path = directory / "report.json"
    evidence = _evidence(
        path, _sha(path.read_bytes()), **({"policy": selected} if policy is not None else {})
    )
    if json.loads(path.read_text()) != report:
        raise BrowserSafetyStop("planet_window_capture_readback_mismatch")
    return evidence


def _same_context(source, fresh):
    if (
        source["star"].casefold() != fresh["star"].casefold()
        or source["time_axis_labels"] != fresh["time_axis_labels"]
        or source["flux_axis_labels"] != fresh["flux_axis_labels"]
        or source["analysis"]["policy"] != fresh["analysis"]["policy"]
    ):
        raise BrowserSafetyStop("planet_window_evidence_context_changed")


def _control(session):
    control = session.frame.get_by_role("combobox")
    if (
        control.count() != 1
        or not control.is_enabled()
        or not rendered_control(control)
        or control.input_value() != ""
        or control.locator("option").all_text_contents() != ["", "Yes", "No"]
    ):
        raise BrowserSafetyStop("unverified_planet_presence_control")
    return control.element_handle(timeout=2000)


def select_no_planet_from_window(
    page,
    config,
    output,
    *,
    run_history,
    evidence_path,
    evidence_sha256,
    preserve_painted_class=None,
    policy=None,
):
    """Select No once; capture/readback failure never permits another attempt.

    The source is a hashed report.json from capture_observation_progress with
    requested_days=5000. Fresh evidence is captured twice: before reservation,
    then immediately before selection. A legacy 10,000-day axis/duration may
    provide its first 5,000 days; it is never restarted or rewritten here.
    """
    directory, history, source = Path(output), Path(run_history), Path(evidence_path)
    selected = recorded_policy_manifest(policy)
    policy_options = {"policy": selected} if selected != recorded_policy_manifest() else {}
    if (
        not history.is_dir()
        or not directory.resolve().is_relative_to(history.resolve())
        or not source.resolve().is_relative_to(history.resolve())
        or source.name != "report.json"
    ):
        raise ValueError("Window evidence and output must belong to the explicit run history")
    if preserve_painted_class not in {None, "gas_giant", "ice_giant", "terrestrial"}:
        raise BrowserSafetyStop("invalid_preserved_planet_paint")
    directory.mkdir(parents=True, exist_ok=False)
    session, attempted, reserved = None, False, False
    try:
        saved = _evidence(source, evidence_sha256, **policy_options)
        key = _sha(saved["star"].casefold().encode())
        reservation = history / "planet-window-choice-reservations" / f"{key}.json"
        if reservation.exists():
            raise BrowserSafetyStop("planet_window_choice_already_reserved")
        session = PlanetNumericSession(
            page, config, directory / "read-guard", max_seconds=120, _allow_unset_planet=True
        )
        before, mapping, choices, _ = session.current()
        values = mapping["observation"]["values"]
        if (
            mapping["star_name"].casefold() != saved["star"].casefold()
            or values["has_planet"] is not None
            or choices["selected"] != preserve_painted_class
        ):
            raise BrowserSafetyStop("planet_window_star_or_selection_changed")
        fields = values["browser_field_map"]
        if fields["observation_days"]["current_value"] not in {"5000", "10000"} or any(
            field["current_value"] for name, field in fields.items() if name != "observation_days"
        ):
            raise BrowserSafetyStop("planet_window_would_replace_work")
        handle = _control(session)
        save_probe(before, directory / "before")
        fresh = _fresh(page, config, directory / "fresh-progress", **policy_options)
        _same_context(saved, fresh)
        session.current()
        if not handle.evaluate("(a,b)=>a.isConnected&&a===b", _control(session)):
            raise BrowserSafetyStop("planet_presence_control_replaced")
        intent = {
            "schema_version": 1,
            "mode": MODE,
            "star": mapping["star_name"],
            "kind": "SELECT",
            "value": "No",
            "provenance": "reference_prediction",
            "action_source": "user_approved_window_reference_not_learned",
            "policy": fresh["analysis"]["policy"],
            "examined_day_interval": [0, 5000],
            "observation_limit_days": 5000,
            "saved_evidence": saved,
            "fresh_evidence": fresh,
            "output": str(directory.resolve().relative_to(history.resolve())),
            "source": str(source.resolve().relative_to(history.resolve())),
            "preserved_painted_class": preserve_painted_class,
            "class_selection_verified": False,
            "numeric_writes": 0,
            "class_writes": 0,
            "na_writes": 0,
            "assessment_clicks": 0,
            "save_clicks": 0,
            "submission_clicks": 0,
            "absence_proven": False,
            "scientific_verified": False,
            "correctness_verified": False,
            "training_label": False,
            "learned_perception": False,
            "task_completed": False,
            "automatic_retry": False,
            "max_presence_writes": 1,
        }
        reservation.parent.mkdir(exist_ok=True)
        try:
            persist_json(reservation, intent)
        except FileExistsError:
            raise BrowserSafetyStop("planet_window_choice_already_reserved") from None
        reserved = True
        persist_json(directory / "reserved.json", intent)
        # This crop is deliberately after the durable reservation. A stale
        # report or late dip cannot become a native selection through reuse.
        preselect = _fresh(page, config, directory / "preselect-progress", **policy_options)
        _same_context(saved, preselect)
        persist_json(directory / "preselect.json", preselect)
        session.current()
        if (
            _evidence(source, evidence_sha256, **policy_options) != saved
            or _evidence(directory / "fresh-progress/report.json", fresh["report_sha256"], **policy_options)
            != fresh
            or _evidence(
                directory / "preselect-progress/report.json", preselect["report_sha256"], **policy_options
            )
            != preselect
            or not handle.evaluate("(a,b)=>a.isConnected&&a===b", _control(session))
        ):
            raise BrowserSafetyStop("planet_window_evidence_or_control_changed")
        attempted = True
        handle.select_option(label="No", timeout=3000)
        after, newer, actual, _ = session.read()
        save_probe(after, directory / "after")
        control = session.frame.get_by_role("combobox")
        if (
            newer["observation"]["values"]["has_planet"] != "No"
            or presence_projection(before, mapping) != presence_projection(after, newer)
            or not preserved_paint_transition(choices, actual)
            or control.count() != 1
            or not rendered_control(control)
            or not handle.evaluate("(a,b)=>a.isConnected&&a===b", control.element_handle(timeout=2000))
        ):
            raise BrowserSafetyStop("unexpected_planet_window_choice_side_effect")
        receipt = {
            **intent,
            "preselect_evidence": preselect,
            "readback_verified": True,
            "painted_class_after": actual["selected"],
            "inherited_paint_cleared": choices["selected"] is not None and actual["selected"] is None,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "mode": MODE,
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "planet_window_choice_failed",
                "reservation_created": reserved,
                "write_may_have_occurred": attempted,
                "automatic_retry": False,
                "absence_proven": False,
                "training_label": False,
                "task_completed": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("planet_window_choice_failed") from None
    finally:
        if session is not None:
            session.close()
