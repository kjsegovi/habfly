"""Opt-in approximate raster-reference transfer, separate from daily semantics.

One Yes selection and a separate at-most-once three-input batch are supported.
Every stage replays owned hashed evidence and rechecks the current visible
chart/spectrum. Neither stage classifies, calculates derived answers, saves,
assesses, trains, or claims scientific correctness or complete daily coverage.
"""

import hashlib
import json
import math
import re
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_numeric import screen_identity
from .browser_observation_progress import capture_observation_progress, trace_progress
from .browser_planet import map_planet_capture
from .browser_planet_chart import FluxChartSession
from .browser_planet_numeric import PlanetNumericSession, planet_projection
from .browser_planet_presence import DERIVED, presence_projection, preserved_paint_transition
from .browser_planet_spectrum import hover_spectrum_marker
from .browser_probe import save_probe
from .browser_setup import rendered_control
from .planet_charts import spectrum_excursion
from .planet_overview_anchor import overview_anchor
from .planet_window_measurements import load_planet_window_measurements

MODE = "approximate_reference_raster"
TOOLTIP_MODE = "approximate_reference_visible_tooltips_v1"
TWO_TOOLTIP_MODE = "approximate_reference_two_visible_tooltips_v1"
TOOLTIP_MODES = (TOOLTIP_MODE, TWO_TOOLTIP_MODE)
TWO_TOOLTIP_FLAGS = {
    "confirmed_feature_count": 2,
    "observed_interval_count": 1,
    "consistency_redundancy": 0,
    "consecutive_events_assumed": True,
    "recurrence_confirmed": False,
    "observed_recurrence_compatible": False,
    "single_spacing_compatible": True,
}
RAW = ("line_shift", "brightness_drop", "period_days")
FLAGS = {
    "provenance": MODE,
    "approximate": True,
    "scientific_verified": False,
    "correctness_verified": False,
    "learned_perception": False,
    "training_label": False,
    "task_completed": False,
    "daily_coverage_verified": False,
    "missing_events_ruled_out": False,
    "aliasing_ruled_out": False,
    "period_evidence_verified": False,
    "observation_completed": False,
    "class_selection_verified": False,
    "class_writes": 0,
    "derived_answer_writes": 0,
    "save_clicks": 0,
    "assessment_clicks": 0,
    "submission_clicks": 0,
    "automatic_retry": False,
}


def _flags_for(evidence=None):
    """Legacy evidence has no mode key; explicit unknown modes never fall back."""
    mode = evidence.get("mode", MODE) if isinstance(evidence, dict) else MODE
    _require(type(mode) is str and mode in (MODE, *TOOLTIP_MODES), "unsupported_measurement_mode")
    if mode == MODE:
        return dict(FLAGS)
    return {
        **FLAGS,
        "provenance": mode,
        "physical_period_verified": False,
        "minimum_depth_verified": False,
        **(TWO_TOOLTIP_FLAGS if mode == TWO_TOOLTIP_MODE else {}),
    }


def _two_tooltip_metadata(reference):
    """Keep one-spacing evidence distinct from repeated-spacing confirmation."""
    _require(
        all(type(reference.get(k)) is type(v) and reference[k] == v for k, v in TWO_TOOLTIP_FLAGS.items())
        and reference.get("mode") == TWO_TOOLTIP_MODE
        and reference.get("period_days", {}).get("estimate_kind") == "single_spacing",
        "invalid_two_tooltip_metadata",
    )
    return {key: reference[key] for key in TWO_TOOLTIP_FLAGS}


def _action_source(evidence):
    return "explicit_" + _flags_for(evidence)["provenance"] + "_not_learned"


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("raster_planet_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _star(value):
    _require(
        isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9 '-]{1,79}", value), "invalid_visible_star"
    )
    return value


def _same_star(left, right):
    # The chart guard uppercases the visible heading; the numeric mapper
    # preserves its original paint spelling. Compare identity without changing
    # either saved representation or relaxing any other spelling difference.
    return (
        isinstance(left, str)
        and isinstance(right, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", left) is not None
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", right) is not None
        and left.casefold() == right.casefold()
    )


class _Owned:
    def __init__(self, history):
        self.history = Path(history).resolve(strict=True)
        _require(self.history.is_dir(), "missing_history")

    def path(self, path):
        path = Path(path).absolute()
        _require(path.resolve().is_relative_to(self.history), "evidence_outside_history")
        _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_evidence")
        return path.resolve()

    def relative(self, path):
        return str(self.path(path).relative_to(self.history))

    def clean(self, directory):
        directory = self.path(directory)
        _require(directory.is_dir(), "missing_evidence_directory")
        _require(
            not any((directory / name).exists() for name in ("stopped.json", "invalidated.json")),
            "failed_evidence",
        )
        return directory

    def read(self, path, expected=None, limit=2_000_000):
        path = self.path(path)
        self.clean(path.parent)
        _require(path.is_file() and path.stat().st_size <= limit, "missing_or_oversized_evidence")
        raw = path.read_bytes()
        if expected is not None:
            _require(
                isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected), "invalid_source_hash"
            )
            _require(_sha(raw) == expected, "source_hash_mismatch")
        return raw

    def json(self, path, expected=None):
        value = json.loads(self.read(path, expected))
        _require(isinstance(value, dict), "invalid_evidence_object")
        return value

    def capture(self, directory):
        directory = self.clean(directory)
        manifest = self.json(directory / "manifest.json")
        raw = self.read(directory / "observation.json", manifest.get("observation_sha256"))
        report = json.loads(raw)
        _require(report.get("ignored_frame_urls") == [], "unknown_capture_frame")
        mapping = map_planet_capture(report, capture_sha256=screen_identity(report))
        return report, mapping, _sha(raw)

    def output(self, output):
        directory = self.path(output)
        _require(directory != self.history, "output_is_history")
        directory.mkdir(parents=True, exist_ok=False)
        return directory


def _window(owner, path, checksum):
    path = owner.path(path)
    _require(path.name == "report.json", "unsupported_window_path")
    report = owner.json(path, checksum)
    owner.read(path.parent / "chart.png", report.get("chart_sha256"), limit=1024 * 1024)
    star = _star(report.get("star"))
    estimate = load_planet_window_measurements(path, expected_report_sha256=checksum)
    _require(estimate["status"] == "approximate_reference_measurements", "window_measurements_unresolved")
    progress = trace_progress(
        owner.read(path.parent / "chart.png", report["chart_sha256"]),
        report["time_axis_labels"],
        requested_days=5000,
    )
    _require(all(report.get(key) == value for key, value in progress.items()), "window_progress_disagrees")
    return {
        "star": star,
        "path": owner.relative(path),
        "report_sha256": checksum,
        "chart_sha256": report["chart_sha256"],
        "time_axis_labels": report["time_axis_labels"],
        "flux_axis_labels": report["flux_axis_labels"],
        "estimate": estimate,
    }


def _spectrum(value):
    if "marker_mode" in value:
        return _geometric_spectrum(value)
    _require(value.get("visibility") == "full_glyph_and_occlusion_checked", "unverified_spectrum_visibility")
    events = value.get("events")
    _require(
        isinstance(events, list)
        and [e.get("kind") for e in events]
        == ["observation", "action_proposed", "action_result", "action_proposed", "action_result"],
        "unsupported_spectrum_events",
    )
    initial = events[0]["payload"]["chart"]
    star = _star(initial.get("star"))
    _require(initial.get("source") == "visible_tooltips", "unsupported_spectrum_source")
    readings = {}
    for index, marker in enumerate(("blue", "red"), 1):
        proposed, result = events[index * 2 - 1]["payload"], events[index * 2]["payload"]
        sample = result["spectrum_sample"]
        _require(
            proposed.get("kind") == "HOVER"
            and proposed.get("surface") == "spectrum"
            and proposed.get("marker") == marker
            and type(proposed.get("sequence")) is int
            and proposed["sequence"] == index
            and result.get("sequence") == index
            and result.get("task_completed") is False
            and sample.get("marker") == marker
            and sample.get("source") == "visible_spectrum_tooltip",
            "spectrum_event_mismatch",
        )
        readings[marker] = sample["wavelength_text"]
    result = spectrum_excursion("656.3nm", readings["blue"], readings["red"])
    _require(result == value.get("result"), "spectrum_recalculation_mismatch")
    return {"star": star, "readings": readings, "result": result}


def _geometric_spectrum(value):
    """Explicit left/right captures retain their raw identity before normalization."""
    from .browser_planet_spectrum import GEOMETRIC_PAIR_MODE, normalize_geometric_spectrum_pair

    _require(value.get("marker_mode") == GEOMETRIC_PAIR_MODE, "unsupported_spectrum_mode")
    _require(value.get("visibility") == "full_glyph_and_occlusion_checked", "unverified_spectrum_visibility")
    events = value.get("events")
    _require(
        type(events) is list
        and len(events) == 5
        and all(type(event) is dict and type(event.get("payload")) is dict for event in events)
        and [event.get("kind") for event in events]
        == ["observation", "action_proposed", "action_result", "action_proposed", "action_result"],
        "unsupported_spectrum_events",
    )
    initial = events[0]["payload"].get("chart")
    _require(type(initial) is dict, "unsupported_spectrum_source")
    star = _star(initial.get("star"))
    _require(initial.get("source") == "visible_tooltips", "unsupported_spectrum_source")
    readings = {}
    for index, marker in enumerate(("left", "right"), 1):
        proposed, result = events[index * 2 - 1]["payload"], events[index * 2]["payload"]
        sample = result.get("spectrum_sample")
        _require(
            proposed.get("kind") == "HOVER"
            and proposed.get("surface") == "spectrum"
            and proposed.get("marker") == marker
            and type(proposed.get("sequence")) is int
            and proposed["sequence"] == index
            and type(result.get("sequence")) is int
            and result["sequence"] == index
            and result.get("task_completed") is False
            and type(sample) is dict
            and sample.get("marker") == marker
            and sample.get("source") == "visible_spectrum_tooltip",
            "spectrum_event_mismatch",
        )
        readings[marker] = sample.get("wavelength_text")
    try:
        pair = normalize_geometric_spectrum_pair(readings["left"], readings["right"])
    except ValueError:
        raise BrowserSafetyStop("raster_planet_invalid_geometric_spectrum_pair") from None
    result = pair["result"]
    recorded = value.get("result")
    _require(
        type(recorded) is dict
        and recorded.keys() == result.keys()
        and all(type(recorded[key]) is type(item) and recorded[key] == item for key, item in result.items()),
        "spectrum_recalculation_mismatch",
    )
    return {"star": star, "readings": pair["readings"], "result": result, "marker_mode": GEOMETRIC_PAIR_MODE}


def _overview_window(owner, path, checksum):
    """An exact settled overview anchor, not a raster measurement estimate."""
    path = owner.path(path)
    _require(path.name == "report.json", "unsupported_window_path")
    report = owner.json(path, checksum)
    png = owner.read(path.parent / "chart.png", report.get("chart_sha256"), limit=1024 * 1024)
    star = _star(report.get("star"))
    progress = trace_progress(png, report["time_axis_labels"], requested_days=5000)
    _require(
        all(type(report.get(k)) is type(v) and report[k] == v for k, v in progress.items())
        and progress["endpoint_visible"] is True
        and type(report.get("browser_actions")) is int
        and report["browser_actions"] == 0
        and type(report.get("answer_writes")) is int
        and report["answer_writes"] == 0,
        "unverified_settled_overview",
    )
    fingerprint = overview_anchor(png, report["time_axis_labels"], report["flux_axis_labels"])
    flux = report["flux_axis_labels"]
    _require(isinstance(flux, list) and len(flux) >= 3, "invalid_overview_flux_axis")
    _require(
        all(
            isinstance(row, dict)
            and set(row) == {"value", "center_y"}
            and type(row["center_y"]) in {int, float}
            and math.isfinite(row["center_y"])
            and Decimal(str(row["value"])).is_finite()
            for row in flux
        ),
        "invalid_overview_flux_axis",
    )
    return {
        "star": star,
        "path": owner.relative(path),
        "report_sha256": checksum,
        "chart_sha256": report["chart_sha256"],
        "time_axis_labels": report["time_axis_labels"],
        "flux_axis_labels": flux,
        "source": "settled_full_5000_overview_anchor",
        "fingerprint": fingerprint,
    }


def _same_overview(source, current):
    _require(
        _same_star(source["star"], current["star"])
        and source["source"] == current["source"]
        and source["fingerprint"] == current["fingerprint"],
        "current_settled_overview_changed",
    )


def _tooltip_source(owner, descriptor, window):
    # Lazy import: the independent offline validator reuses _Owned, not this
    # adapter's native actions. Every source it consumes is adopted into this
    # owner's read/clean hooks so downstream _Evidence retains the whole chain.
    from .planet_tooltip_reference import load_tooltip_reference

    _require(isinstance(descriptor, dict), "invalid_tooltip_descriptor")
    two = set(descriptor) == {"diagnostics", "expected_star", "mode"}
    _require(
        (two and descriptor["mode"] == TWO_TOOLTIP_MODE)
        or (not two and set(descriptor) == {"diagnostics", "expected_star"}),
        "invalid_tooltip_descriptor",
    )
    mode = TWO_TOOLTIP_MODE if two else TOOLTIP_MODE
    _require(_same_star(descriptor["expected_star"], window["star"]), "tooltip_star_changed")
    result = load_tooltip_reference(
        owner.history,
        descriptor["diagnostics"],
        expected_star=descriptor["expected_star"],
        **({"mode": mode} if two else {}),
    )
    _require(result.get("mode") == mode, "unsupported_tooltip_result")
    if two:
        _two_tooltip_metadata(result)
    for directory in result["validated_directories"]:
        owner.clean(owner.history / directory)
    for name, checksum in result["source_sha256"].items():
        owner.read(owner.history / name, checksum)
    for feature in result["features"]:
        path = owner.history / feature["source_dir"] / "report.json"
        source = _overview_window(owner, path, result["source_sha256"][owner.relative(path)])
        _same_overview(window, source)
    normalized = {
        "expected_star": descriptor["expected_star"],
        "diagnostics": [
            {
                "directory": owner.relative(
                    Path(spec["directory"])
                    if Path(spec["directory"]).is_absolute()
                    else owner.history / spec["directory"]
                ),
                "files": deepcopy(spec["files"]),
            }
            for spec in descriptor["diagnostics"]
        ],
    }
    if two:
        normalized["mode"] = mode
    return normalized, result


def _evidence(
    owner, window_report, window_report_sha256, spectrum_path, spectrum_sha256, *, tooltip_reference=None
):
    if tooltip_reference is not None:
        window = _overview_window(owner, window_report, window_report_sha256)
        descriptor, reference = _tooltip_source(owner, tooltip_reference, window)
        mode = reference["mode"]
        spectrum = _spectrum(owner.json(spectrum_path, spectrum_sha256))
        _require(_same_star(window["star"], spectrum["star"]), "source_star_disagrees")
        period, drop = reference["period_days"], reference["brightness_drop_percent"]
        _require(
            all(
                isinstance(item["value"], str)
                and Decimal(item["value"]).is_finite()
                and Decimal(item["value"]) > 0
                for item in (period, drop)
            )
            and Decimal(period["value"]) <= 5000
            and Decimal(drop["value"]) <= 100
            and drop["physical_bounds"] is None,
            "invalid_tooltip_measurements",
        )
        return {
            "mode": mode,
            "star": window["star"],
            "window": window,
            "spectrum": {**spectrum, "path": owner.relative(spectrum_path), "sha256": spectrum_sha256},
            "tooltip_reference": descriptor,
            "tooltip_measurements": reference,
            "measurements": {
                "line_shift": {
                    "value": spectrum["result"]["line_shift_nm"],
                    "unit": "nm",
                    "source": "visible_spectrum_markers",
                },
                "brightness_drop": {
                    "value": drop["value"],
                    "unit": "%",
                    "physical_bounds": None,
                    "source": mode,
                },
                "period_days": {
                    "value": period["value"],
                    "unit": "day",
                    "compatibility_interval": deepcopy(period["compatibility_interval"]),
                    "exact_representative": deepcopy(period["exact_representative"]),
                    "source": mode,
                    **({"estimate_kind": "single_spacing"} if mode == TWO_TOOLTIP_MODE else {}),
                },
            },
            "uncertainty": {
                "kind": "sampled_tooltip_bracket_compatibility_not_physical_bounds",
                "period_days": deepcopy(period),
                "brightness_drop_percent": deepcopy(drop),
                "limitations": deepcopy(reference["method"]["limitations"]),
                "spectrum": "Exact visible tooltip spelling; physical and systematic uncertainty unverified",
                **(_two_tooltip_metadata(reference) if mode == TWO_TOOLTIP_MODE else {}),
            },
        }
    window = _window(owner, window_report, window_report_sha256)
    spectrum = _spectrum(owner.json(spectrum_path, spectrum_sha256))
    _require(_same_star(window["star"], spectrum["star"]), "source_star_disagrees")
    estimate = window["estimate"]
    measurements = {
        "line_shift": {
            "value": spectrum["result"]["line_shift_nm"],
            "unit": "nm",
            "source": "visible_spectrum_markers",
        }
    }
    for field, key, unit in (
        ("brightness_drop", "brightness_drop_percent", "%"),
        ("period_days", "period_days", "day"),
    ):
        spec = estimate[key]
        _require(
            all(math.isfinite(spec[k]) and spec[k] > 0 for k in ("value", "lower", "upper")),
            "invalid_estimate",
        )
        measurements[field] = {
            "value": repr(spec["value"]),
            "unit": unit,
            "lower": repr(spec["lower"]),
            "upper": repr(spec["upper"]),
            "source": "approximate_reference_raster",
        }
    return {
        "star": window["star"],
        "window": window,
        "spectrum": {**spectrum, "path": owner.relative(spectrum_path), "sha256": spectrum_sha256},
        "measurements": measurements,
        "uncertainty": {
            "kind": "conditional_pixel_bounds_not_statistical_confidence",
            "period_days": estimate["period_days"],
            "brightness_drop_percent": estimate["brightness_drop_percent"],
            "limitations": estimate["method"]["limitations"],
            "spectrum": "Exact visible tooltip spelling; physical and systematic uncertainty unverified",
        },
    }


def _reload(owner, evidence):
    w, s = evidence["window"], evidence["spectrum"]
    mode = _flags_for(evidence)["provenance"]
    tooltip = mode in TOOLTIP_MODES
    current = _evidence(
        owner,
        owner.history / w["path"],
        w["report_sha256"],
        owner.history / s["path"],
        s["sha256"],
        **({"tooltip_reference": evidence["tooltip_reference"]} if tooltip else {}),
    )
    equal = current == evidence
    if mode == TWO_TOOLTIP_MODE:
        # New mode's count/assumption metadata must not accept bool/int aliases.
        equal = json.dumps(current, sort_keys=True, allow_nan=False) == json.dumps(
            evidence, sort_keys=True, allow_nan=False
        )
    _require(equal, "source_evidence_changed")


def _same_window(source, current):
    _require(
        _same_star(source["star"], current["star"])
        and source["time_axis_labels"] == current["time_axis_labels"]
        and source["flux_axis_labels"] == current["flux_axis_labels"]
        and source["estimate"]["method"] == current["estimate"]["method"],
        "current_window_context_changed",
    )
    old, new = source["estimate"], current["estimate"]
    bounds, newer = old["support_pixel_bounds"], new["support_pixel_bounds"]
    _require(
        len(bounds) == len(newer)
        and all(
            abs(a - b) <= 1
            for first, second in zip(bounds, newer, strict=True)
            for a, b in zip(first, second, strict=True)
        ),
        "current_positive_features_changed",
    )
    for key in ("period_days", "brightness_drop_percent"):
        _require(
            old[key]["lower"] <= new[key]["value"] <= old[key]["upper"]
            and new[key]["lower"] <= old[key]["value"] <= new[key]["upper"],
            "current_measurements_outside_uncertainty",
        )


def _fresh(page, config, owner, directory, evidence):
    directory.mkdir(exist_ok=False)
    events, chart = [], None
    try:
        chart = FluxChartSession(
            page,
            config,
            lambda kind, payload: events.append({"kind": kind, "payload": payload}),
            max_actions=2,
            max_seconds=120,
        )
        _require(_same_star(chart.star, evidence["star"]), "current_spectrum_star_changed")
        if "marker_mode" in evidence["spectrum"]:
            from .browser_planet_spectrum import GEOMETRIC_PAIR_MODE, read_geometric_spectrum_pair

            _require(evidence["spectrum"]["marker_mode"] == GEOMETRIC_PAIR_MODE, "unsupported_spectrum_mode")
            pair = read_geometric_spectrum_pair(chart)
            result = pair["result"]
        else:
            readings = {
                side: hover_spectrum_marker(chart, side)["wavelength_text"] for side in ("blue", "red")
            }
            result = spectrum_excursion("656.3nm", readings["blue"], readings["red"])
        artifact = {"events": events, "result": result, "visibility": "full_glyph_and_occlusion_checked"}
        if "marker_mode" in evidence["spectrum"]:
            artifact["marker_mode"] = evidence["spectrum"]["marker_mode"]
        persist_json(directory / "spectrum.json", artifact)
        measured = _spectrum(artifact)
        _require(
            measured["readings"] == evidence["spectrum"]["readings"]
            and measured["result"] == evidence["spectrum"]["result"],
            "current_spectrum_changed",
        )
    finally:
        if chart is not None:
            chart.close()
        persist_json(
            directory / "spectrum-events.json",
            {"events": events, "maximum_hover_actions": 2, "max_seconds": 120},
        )
    # The ordinary marker HOVER has moved the pointer off the flux chart. No
    # DOM is hidden, curve path inspected, chart panned, or observation restarted.
    capture_observation_progress(page, config, directory / "progress", requested_days=5000, max_seconds=120)
    path = directory / "progress/report.json"
    mode = _flags_for(evidence)["provenance"]
    tooltip = mode in TOOLTIP_MODES
    current = (_overview_window if tooltip else _window)(owner, path, _sha(owner.read(path)))
    (_same_overview if tooltip else _same_window)(evidence["window"], current)
    return {
        "window": current,
        "spectrum_sha256": _sha(owner.read(directory / "spectrum.json")),
        **({"measurement_mode": mode} if mode == TWO_TOOLTIP_MODE else {}),
    }


def _fresh_evidence(owner, directory, recorded, evidence):
    """Replay one fresh/preselect/precopy result with the exact selected method."""
    mode = _flags_for(evidence)["provenance"]
    tooltip = mode in TOOLTIP_MODES
    _require(
        recorded.get("measurement_mode") == TWO_TOOLTIP_MODE
        if mode == TWO_TOOLTIP_MODE
        else "measurement_mode" not in recorded,
        "fresh_measurement_mode_changed",
    )
    path = directory / "progress/report.json"
    current = (_overview_window if tooltip else _window)(owner, path, _sha(owner.read(path)))
    _require(recorded["window"] == current, "fresh_window_evidence_changed")
    (_same_overview if tooltip else _same_window)(evidence["window"], current)
    measured = _spectrum(owner.json(directory / "spectrum.json", recorded["spectrum_sha256"]))
    keys = ("star", "readings", "result") + (
        ("marker_mode",) if "marker_mode" in evidence["spectrum"] else ()
    )
    _require(
        measured == {k: evidence["spectrum"][k] for k in keys},
        "fresh_spectrum_evidence_changed",
    )


def _blank_current(mapping, choices, evidence, *, presence, painted):
    values = mapping["observation"]["values"]
    _require(
        _same_star(mapping["star_name"], evidence["star"]) and values["has_planet"] == presence,
        "current_star_or_presence_changed",
    )
    _require(choices["selected"] == painted, "current_class_paint_changed")
    fields = values["browser_field_map"]
    _require(fields["observation_days"]["current_value"] == "5000", "exact_5000_duration_required")
    expected = {"observation_days", *RAW} | (DERIVED if presence == "Yes" else set())
    _require(
        set(fields) == expected
        and all(spec["current_value"] == "" for name, spec in fields.items() if name != "observation_days"),
        "answers_already_populated",
    )


def _control(session):
    control = session.frame.get_by_role("combobox")
    _require(
        control.count() == 1
        and rendered_control(control)
        and control.is_enabled()
        and control.input_value() == ""
        and control.locator("option").all_text_contents() == ["", "Yes", "No"],
        "unverified_presence_control",
    )
    return control.element_handle(timeout=2000)


def _reservation(owner, star, stage):
    return owner.path(
        owner.history / f"planet-raster-{stage}-reservations" / f"{_sha(star.casefold().encode())}.json"
    )


def _reserve(owner, directory, star, stage, intent):
    path = _reservation(owner, star, stage)
    path.parent.mkdir(exist_ok=True)
    try:
        persist_json(path, intent)
    except FileExistsError:
        raise BrowserSafetyStop("raster_planet_" + stage + "_already_reserved") from None
    persist_json(directory / "reserved.json", intent)


def _failure(directory, exc, *, attempted, reserved, evidence=None):
    reason = str(exc) if isinstance(exc, BrowserSafetyStop) else "raster_planet_operation_failed"
    flags = _flags_for(evidence)
    if flags["provenance"] == TWO_TOOLTIP_MODE:
        validated = isinstance(evidence.get("tooltip_measurements"), dict)
        if not validated:
            # Merely requesting the new mode is not proof of two confirmed
            # features. Early descriptor/source failures have no count claim.
            flags = {key: value for key, value in flags.items() if key not in TWO_TOOLTIP_FLAGS}
        flags["measurement_evidence_validated"] = validated
    persist_json(
        directory / "stopped.json",
        {
            **flags,
            "reason": reason,
            "write_may_have_occurred": attempted,
            "reservation_created": reserved,
        },
    )


def select_raster_detected_planet(
    page,
    config,
    output,
    *,
    run_history,
    window_report,
    window_report_sha256,
    spectrum_path,
    spectrum_sha256,
    preserve_painted_class=None,
    max_seconds=600,
    tooltip_reference=None,
    before_dispatch=None,
):
    """One explicitly reference Yes; uncertain dispatch permanently retains claim.

    An optional source/cancellation guard receives the current public mapping
    after reservation, before the native selection. It cannot replace answers
    or extend the session deadline; native state and target identity are checked
    again after it returns. Legacy calls retain their original read sequence.
    """
    _require(before_dispatch is None or callable(before_dispatch), "invalid_before_dispatch_guard")
    _require(
        preserve_painted_class in {None, "gas_giant", "ice_giant", "terrestrial"}, "invalid_preserved_paint"
    )
    _require(type(max_seconds) in {int, float} and 30 <= max_seconds <= 900, "invalid_time_budget")
    owner = _Owned(run_history)
    directory = owner.output(output)
    session, attempted, reserved = None, False, False
    evidence = (
        {
            "mode": TWO_TOOLTIP_MODE
            if isinstance(tooltip_reference, dict) and tooltip_reference.get("mode") == TWO_TOOLTIP_MODE
            else TOOLTIP_MODE
        }
        if tooltip_reference is not None
        else None
    )
    try:
        evidence = _evidence(
            owner,
            window_report,
            window_report_sha256,
            spectrum_path,
            spectrum_sha256,
            **({"tooltip_reference": tooltip_reference} if tooltip_reference is not None else {}),
        )
        flags = _flags_for(evidence)
        _require(not _reservation(owner, evidence["star"], "presence").exists(), "presence_already_reserved")
        _require(
            not (
                owner.history
                / "planet-window-choice-reservations"
                / f"{_sha(evidence['star'].casefold().encode())}.json"
            ).exists(),
            "conflicting_no_reservation",
        )
        session = PlanetNumericSession(
            page, config, directory / "read-guard", max_seconds=max_seconds, _allow_unset_planet=True
        )
        before, mapping, choices, _ = session.current()
        _blank_current(mapping, choices, evidence, presence=None, painted=preserve_painted_class)
        handle = _control(session)
        save_probe(before, directory / "before")
        fresh = _fresh(page, config, owner, directory / "fresh", evidence)
        session.current()
        intent = {
            **flags,
            "schema_version": 1,
            "mode": flags["provenance"],
            "stage": "presence",
            "kind": "SELECT",
            "value": "Yes",
            "star": evidence["star"],
            "evidence": evidence,
            "fresh": fresh,
            "output": owner.relative(directory),
            "preserved_painted_class": preserve_painted_class,
            "painted_choices_before": choices,
            "numeric_writes": 0,
            "maximum_presence_writes": 1,
            "max_seconds": max_seconds,
            "action_source": _action_source(evidence),
        }
        _reserve(owner, directory, evidence["star"], "presence", intent)
        reserved = True
        preselect = _fresh(page, config, owner, directory / "preselect", evidence)
        persist_json(directory / "preselect.json", preselect)
        _reload(owner, evidence)
        _, current_mapping, _, _ = session.current()
        if before_dispatch is not None:
            before_dispatch(deepcopy(current_mapping))
            session.current()
        _require(
            handle.evaluate("(a,b)=>a.isConnected&&a===b", _control(session)), "presence_control_replaced"
        )
        attempted = True
        handle.select_option(label="Yes", timeout=3000)
        after, newer, actual, _ = session.read()
        save_probe(after, directory / "after")
        _blank_current(newer, actual, evidence, presence="Yes", painted=actual["selected"])
        control = session.frame.get_by_role("combobox")
        _require(
            presence_projection(before, mapping) == presence_projection(after, newer)
            and preserved_paint_transition(choices, actual)
            and control.count() == 1
            and control.is_enabled()
            and rendered_control(control)
            and control.input_value() == "Yes"
            and handle.evaluate(
                "(a,b)=>a.isConnected&&a===b",
                control.element_handle(timeout=2000),
            ),
            "unexpected_presence_side_effect",
        )
        _reload(owner, evidence)
        receipt = {
            **intent,
            "preselect": preselect,
            "readback_verified": True,
            "painted_choices_after": actual,
            "painted_class_after": actual["selected"],
            "inherited_paint_cleared": choices["selected"] is not None and actual["selected"] is None,
            "before_capture_sha256": _sha(owner.read(directory / "before/observation.json")),
            "after_capture_sha256": _sha(owner.read(directory / "after/observation.json")),
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        _failure(directory, exc, attempted=attempted, reserved=reserved, evidence=evidence)
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("raster_planet_operation_failed") from None
    finally:
        if session is not None:
            session.close()


def _presence(owner, path, checksum):
    path = owner.path(path)
    _require(path.name == "confirmed.json", "invalid_presence_path")
    receipt = owner.json(path, checksum)
    intent = owner.json(path.parent / "reserved.json")
    evidence = receipt["evidence"]
    flags = _flags_for(evidence)
    _require(
        receipt.get("mode") == flags["provenance"]
        and receipt.get("stage") == "presence"
        and receipt.get("value") == "Yes"
        and receipt.get("kind") == "SELECT"
        and receipt.get("readback_verified") is True
        and receipt.get("numeric_writes") == 0
        and all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in flags.items())
        and receipt.get("action_source") == _action_source(evidence)
        and all(receipt.get(k) == v for k, v in intent.items())
        and receipt.get("output") == owner.relative(path.parent),
        "unverified_presence_receipt",
    )
    _require(
        owner.json(_reservation(owner, _star(receipt["star"]), "presence")) == intent,
        "presence_reservation_mismatch",
    )
    _reload(owner, evidence)
    _require(_same_star(receipt["star"], evidence["star"]), "presence_evidence_star_mismatch")
    _require(owner.json(path.parent / "preselect.json") == receipt["preselect"], "preselect_receipt_mismatch")
    for name in ("fresh", "preselect"):
        _fresh_evidence(owner, path.parent / name, receipt[name], evidence)
    before, old, before_hash = owner.capture(path.parent / "before")
    after, newer, after_hash = owner.capture(path.parent / "after")
    choices, actual = receipt["painted_choices_before"], receipt["painted_choices_after"]
    _blank_current(old, choices, evidence, presence=None, painted=receipt["preserved_painted_class"])
    _blank_current(newer, actual, evidence, presence="Yes", painted=receipt["painted_class_after"])
    _require(
        before_hash == receipt["before_capture_sha256"]
        and after_hash == receipt["after_capture_sha256"]
        and preserved_paint_transition(choices, actual)
        and presence_projection(before, old) == presence_projection(after, newer),
        "presence_capture_mismatch",
    )
    return receipt, after, newer


def copy_raster_measured_inputs(
    page, config, output, *, run_history, presence_path, presence_sha256, max_seconds=900
):
    """Copy exactly three selected raw spellings once; no derived calculations."""
    _require(type(max_seconds) in {int, float} and 30 <= max_seconds <= 900, "invalid_time_budget")
    owner = _Owned(run_history)
    directory = owner.output(output)
    session, reserved, evidence = None, False, None
    try:
        presence, previous, previous_map = _presence(owner, presence_path, presence_sha256)
        evidence, star = presence["evidence"], presence["star"]
        flags = _flags_for(evidence)
        _require(not _reservation(owner, star, "inputs").exists(), "inputs_already_reserved")
        events = []

        def native_event(kind, payload):
            # Existing copy() emits after its durable field reservation and
            # immediately before its final current-state check/native fill.
            # Recheck source identity at that boundary, not only per batch.
            if kind == "action_proposed":
                _presence(owner, presence_path, presence_sha256)
            event = {"kind": kind, "payload": payload, "provenance": flags["provenance"]}
            persist_json(directory / f"native-event-{len(events):02d}.json", event)
            events.append(event)

        session = PlanetNumericSession(
            page, config, directory / "native-copies", native_event, max_seconds=max_seconds
        )
        before, mapping, choices, _ = session.current()
        _blank_current(mapping, choices, evidence, presence="Yes", painted=presence["painted_class_after"])
        _require(
            planet_projection(before, mapping) == planet_projection(previous, previous_map)
            and choices == presence["painted_choices_after"],
            "presence_current_state_changed",
        )
        fresh = _fresh(page, config, owner, directory / "fresh", evidence)
        session.current()
        intent = {
            **flags,
            "schema_version": 1,
            "mode": flags["provenance"],
            "stage": "inputs",
            "star": star,
            "evidence": evidence,
            "fresh": fresh,
            "output": owner.relative(directory),
            "presence_path": owner.relative(presence_path),
            "presence_sha256": presence_sha256,
            "maximum_numeric_writes": 3,
            "max_seconds": max_seconds,
            "destinations": list(RAW),
            "action_source": _action_source(evidence),
        }
        _reserve(owner, directory, star, "inputs", intent)
        reserved = True
        precopy = []
        for index, name in enumerate(RAW):
            _presence(owner, presence_path, presence_sha256)
            current = _fresh(page, config, owner, directory / f"precopy-{index + 1}", evidence)
            precopy.append(current)
            persist_json(directory / f"precopy-{index + 1}/evidence.json", current)
            session.current()
            spec = evidence["measurements"][name]
            session.copy(name, spec["value"], spec["unit"], source="reference_diagnostic")
        session.current()
        _reload(owner, evidence)
        report = {
            **intent,
            "raw_measurement_transport_verified": True,
            "numeric_writes": 3,
            "verified_fields": session.verified,
            "precopy": precopy,
            "native_events": events,
            "saved": False,
            "assessed": False,
            "submitted": False,
        }
        persist_json(directory / "report.json", report)
        return report
    except BaseException as exc:
        _failure(
            directory,
            exc,
            attempted=bool(session and session.attempted),
            reserved=reserved,
            evidence=evidence,
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("raster_planet_operation_failed") from None
    finally:
        if session is not None:
            session.close()
