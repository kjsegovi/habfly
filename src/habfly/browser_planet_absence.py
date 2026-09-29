"""Explicit no-planet reference hypothesis, never a measured absence theorem.

A finite flat interval cannot rule out a planet. This one-shot transport lets
an explicit caller test that hypothesis against the course, keeping the limited
evidence and unverified status. It provides no learner/expert training label,
zero-amplitude answer, assessment, automatic correction, or retry.
"""

import hashlib
import json
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_planet_chart import VISIBLE_TOOLTIP
from .browser_planet_numeric import PlanetNumericSession
from .browser_planet_presence import presence_projection, preserved_paint_transition
from .browser_probe import save_probe
from .browser_setup import rendered_control
from .browser_transit_sampling import validated_scan_samples
from .presentation_capture import evidence_screenshot


def visible_spectrum_inventory(frame):
    label = frame.get_by_text("656.3nm", exact=True)
    if label.count() != 1 or not label.is_visible():
        raise BrowserSafetyStop("ambiguous_rest_wavelength")
    strips = [
        e
        for e in label.locator("..").locator(":scope > div").all()
        if e.is_visible() and e.evaluate("e=>getComputedStyle(e).backgroundColor==='rgb(193, 58, 44)'")
    ]
    if len(strips) != 1:
        raise BrowserSafetyStop("unsupported_spectral_strip")
    strip = strips[0]
    lines = [
        e
        for e in strip.locator(":scope > div").all()
        if e.is_visible() and e.evaluate("e=>getComputedStyle(e).backgroundColor==='rgb(0, 0, 0)'")
    ]
    if len(lines) != 1 or not lines[0].evaluate(VISIBLE_TOOLTIP):
        raise BrowserSafetyStop("absence_hypothesis_requires_single_exposed_line")
    width = lines[0].bounding_box()["width"]
    if width != 2:
        raise BrowserSafetyStop("absence_hypothesis_excursion_marker_present")
    return strip, {
        "visible_black_lines": 1,
        "line_width_css_pixels": 2,
        "excursion_markers_detected": False,
        "zero_amplitude_measured": False,
    }


def select_no_planet_hypothesis(page, config, output, *, transit_path, rationale):
    if not isinstance(rationale, str) or not 40 <= len(rationale.strip()) <= 2000:
        raise ValueError("Explicit limited-evidence reference rationale required")
    source = Path(transit_path)
    if (source.parent / "invalidated.json").exists():
        raise BrowserSafetyStop("invalidated_transit_scan_evidence")
    raw = source.read_bytes()
    evidence = json.loads(raw)
    validated_scan_samples(evidence, star=evidence.get("star"))
    if (
        evidence.get("time_axis_validation") != "each_hover_against_visible_tick_glyphs"
        or evidence.get("status") != "no_transit_in_observed_interval"
        or evidence.get("unique_days", 0) < 1000
        or evidence.get("complete_transits")
        or evidence.get("period_evidence_verified") is not False
    ):
        raise BrowserSafetyStop("unsupported_absence_hypothesis_evidence")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    session, attempted = None, False
    try:
        session = PlanetNumericSession(
            page, config, directory / "read-guard", max_seconds=90, _allow_unset_planet=True
        )
        before, mapping, choices, _ = session.current()
        values = mapping["observation"]["values"]
        if mapping["star_name"].upper() != evidence["star"] or values["has_planet"] is not None:
            raise BrowserSafetyStop("absence_hypothesis_star_or_selection_changed")
        if any(v["current_value"] for k, v in values["browser_field_map"].items() if k != "observation_days"):
            raise BrowserSafetyStop("absence_hypothesis_would_replace_work")
        strip, spectrum = visible_spectrum_inventory(session.frame)
        evidence_screenshot(strip, path=str(directory / "spectrum.png"))
        control = session.frame.get_by_role("combobox")
        if (
            control.count() != 1
            or not control.is_enabled()
            or not rendered_control(control)
            or control.input_value() != ""
            or control.locator("option").all_text_contents() != ["", "Yes", "No"]
        ):
            raise BrowserSafetyStop("unverified_planet_presence_control")
        handle = control.element_handle(timeout=1000)
        intent = {
            "star": mapping["star_name"],
            "value": "No",
            "kind": "SELECT",
            "action_source": "explicit_reference_hypothesis_not_learned",
            "rationale": rationale,
            "observed_day_min": min(s["day"] for s in evidence["samples"]),
            "observed_day_max": max(s["day"] for s in evidence["samples"]),
            "observed_unique_days": evidence["unique_days"],
            "spectrum": spectrum,
            "transit_sha256": hashlib.sha256(raw).hexdigest(),
            "spectrum_crop_sha256": hashlib.sha256((directory / "spectrum.png").read_bytes()).hexdigest(),
            "absence_proven": False,
            "correctness_verified": False,
            "training_label": False,
            "numeric_writes": 0,
            "class_writes": 0,
            "assessment_clicks": 0,
            "automatic_retry": False,
            "task_completed": False,
        }
        persist_json(directory / "reserved.json", intent)
        session.current()
        if (
            hashlib.sha256(source.read_bytes()).hexdigest() != intent["transit_sha256"]
            or visible_spectrum_inventory(session.frame)[1] != spectrum
            or not handle.evaluate("(a,b)=>a.isConnected&&a===b", control.element_handle())
        ):
            raise BrowserSafetyStop("absence_hypothesis_evidence_changed")
        attempted = True
        handle.select_option(label="No", timeout=3000)
        after, newer, actual, _ = session.read()
        save_probe(after, directory / "after")
        if (
            newer["observation"]["values"]["has_planet"] != "No"
            or presence_projection(before, mapping) != presence_projection(after, newer)
            or not preserved_paint_transition(choices, actual)
            or not handle.evaluate("(a,b)=>a.isConnected&&a===b", control.element_handle())
        ):
            raise BrowserSafetyStop("unexpected_absence_hypothesis_side_effect")
        receipt = {**intent, "readback_verified": True, "painted_class_after": actual["selected"]}
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "absence_hypothesis_uncertain",
                "write_may_have_occurred": attempted,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        if isinstance(exc, Exception) and not isinstance(exc, BrowserSafetyStop):
            raise BrowserSafetyStop("absence_hypothesis_uncertain") from None
        raise
    finally:
        if session is not None:
            session.close()
