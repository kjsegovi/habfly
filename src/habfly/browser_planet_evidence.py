"""Explicit scripted transfer of same-star, student-visible measurement evidence.

This is not a learned selection stage. Three raw inputs are transferred exactly;
the promoted biological policy still chooses its four derived calculations.
"""

import hashlib
import json
import re
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_planet_numeric import PlanetNumericSession
from .browser_transit_sampling import validated_scan_samples
from .planet_charts import analyze_flux_samples, spectrum_excursion


def measurement_evidence(spectrum, transit):
    fail = BrowserSafetyStop
    if not isinstance(spectrum, dict) or not isinstance(transit, dict):
        raise fail("invalid_planet_measurement_evidence")
    star = transit.get("star")
    if not isinstance(star, str) or not re.fullmatch(r"[A-Z][A-Z -]{1,39}", star):
        raise fail("missing_planet_evidence_star")
    if (
        spectrum.get("visibility") != "full_glyph_and_occlusion_checked"
        or transit.get("sampling_method") != "scripted_bounded_visible_tooltips"
        or transit.get("period_evidence_verified") is not True
        or transit.get("has_planet_answer") is not None
    ):
        raise fail("unverified_planet_measurement_evidence")
    events = spectrum.get("events", [])
    stars = [
        r.get("payload", {}).get("chart", {}).get("star") for r in events if r.get("kind") == "observation"
    ]
    readings = [
        r["payload"]["spectrum_sample"]
        for r in events
        if r.get("kind") == "action_result" and "spectrum_sample" in r.get("payload", {})
    ]
    if stars != [star] or len(readings) != 2 or {r.get("marker") for r in readings} != {"blue", "red"}:
        raise fail("conflicting_planet_measurement_identity")
    if any(r.get("source") != "visible_spectrum_tooltip" for r in readings):
        raise fail("unverified_spectrum_source")
    selected = {r["marker"]: r["wavelength_text"] for r in readings}
    result = spectrum_excursion("656.3nm", selected["blue"], selected["red"])
    if result != spectrum.get("result"):
        raise fail("spectrum_evidence_recalculation_mismatch")
    samples = validated_scan_samples(transit, star=star)
    measured = analyze_flux_samples(samples)
    if measured["status"] != "periodic_transits_observed" or any(
        transit.get(k) != v for k, v in measured.items()
    ):
        raise fail("transit_evidence_recalculation_mismatch")
    return {
        "star": star,
        "measurements": {
            "line_shift": {"value": result["line_shift_nm"], "unit": "nm"},
            "brightness_drop": {"value": measured["brightness_drop_percent"], "unit": "%"},
            "period_days": {"value": measured["period_days"], "unit": "day"},
        },
        "selection_source": "scripted_visible_evidence_transfer_not_learned",
        "raw_observation_interval_complete": False,
        "period_evidence_verified": True,
        "scientific_assumptions_verified": False,
        "task_completed": False,
    }


def copy_measured_inputs(page, config, output, *, spectrum_path, transit_path):
    raw = {
        name: Path(path).read_bytes()
        for name, path in (("spectrum", spectrum_path), ("transit", transit_path))
    }
    evidence = measurement_evidence(json.loads(raw["spectrum"]), json.loads(raw["transit"]))
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    persist_json(
        directory / "evidence.json",
        {**evidence, "source_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}},
    )
    session = None
    try:
        session = PlanetNumericSession(page, config, directory / "native-copies")
        if session.mapping["star_name"].upper() != evidence["star"]:
            raise BrowserSafetyStop("planet_evidence_star_changed")
        fields = session.mapping["observation"]["values"]["browser_field_map"]
        if any(fields[name]["current_value"] != "" for name in evidence["measurements"]):
            raise BrowserSafetyStop("planet_raw_measurements_already_populated")
        for name, spec in evidence["measurements"].items():
            session.copy(name, spec["value"], spec["unit"], source="reference_diagnostic")
        report = {
            **evidence,
            "raw_measurement_transport_verified": True,
            "verified_fields": session.verified,
            "saved": False,
            "assessed": False,
            "submitted": False,
        }
        persist_json(directory / "report.json", report)
        return report
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "raw_measurement_transfer_failed",
                "attempted": sorted(session.attempted) if session else [],
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        raise
    finally:
        if session is not None:
            session.close()
