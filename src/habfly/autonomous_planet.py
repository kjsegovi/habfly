"""Offline, source-bound reference decisions; never learned or course truth.

The Solar-System nearest-prototype metric and pixel-template matcher are
explicit engineering heuristics. The habitability rule is the public course
rule, not a general biosignature claim. No browser, network, grading data or
selected-control-as-answer inference is available in this module.
"""

import hashlib
import io
import itertools
import json
import re
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .browser import BrowserSafetyStop
from .browser_habitability import GASES
from .browser_habitability_actions import menu_projection
from .browser_habitability_choice import confirmed_phase_reference
from .browser_habitability_save import _mapping, _same_view
from .browser_no_planet_workflow import _Evidence
from .browser_numeric import comparable_screen, screen_identity
from .browser_planet import map_planet_capture
from .browser_terrestrial_workflow import _clean, _gas_sources, _gas_writes, _same_habitat, _unchanged
from .habitability_knowledge import HabitabilityCalculator

POLICY = "autonomous_planet_reference_v1"
REFERENCE_PACK = {
    "schema_version": 1,
    "id": POLICY,
    "retrieved": "2026-09-26",
    "planet_policy": "nearest_solar_prototype_squared_log_mass_radius_density_v1",
    "planet_policy_scope": "approximate_analogy_not_course_thresholds",
    "metric_source": "explicit engineering policy; NASA supplies data and labels, not this metric",
    "planet_sources": [
        "https://nssdc.gsfc.nasa.gov/planetary/factsheet/",
        "https://science.nasa.gov/solar-system/planets/",
    ],
    "prototype_units": ["1e24_kg", "diameter_km", "kg/m3"],
    "earth_reference": ["5.97", "12756", "5514"],
    "prototypes": [
        ["Mercury", "terrestrial", "0.330", "4879", "5429"],
        ["Venus", "terrestrial", "4.87", "12104", "5243"],
        ["Earth", "terrestrial", "5.97", "12756", "5514"],
        ["Mars", "terrestrial", "0.642", "6792", "3934"],
        ["Jupiter", "gas_giant", "1898", "142984", "1326"],
        ["Saturn", "gas_giant", "568", "120536", "687"],
        ["Uranus", "ice_giant", "86.8", "51118", "1270"],
        ["Neptune", "ice_giant", "102", "49528", "1638"],
    ],
    "habitability_source": "https://kb.inspark.education/habworlds-project",
    "habitability_rule": "terrestrial AND confirmed liquid surface-water conditions",
    "lifetime_required": False,
    "water_vapor_detection_required": False,
    "greenhouse_source": "student-visible Greenhouse Strength help, preserved habitability knowledge pack",
    "gas_policy": "blue_target_grey_template_pixel_depth_union_v1",
    "gas_policy_scope": "approximate visible-profile matching, not chemistry inference or calibrated confidence",
}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def reference_pack():
    return json.loads(_canonical(REFERENCE_PACK))


def _require(value, reason):
    if not value:
        raise BrowserSafetyStop("autonomous_planet_" + reason)


def _identity(value, expected):
    _require(
        isinstance(expected, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", expected)
        and isinstance(value, str)
        and value.casefold() == expected.casefold(),
        "star_mismatch",
    )


def _path(book, value):
    path = Path(value)
    return book.path(path if path.is_absolute() else book.history / path)


def _pinned(book, path, expected):
    _require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected), "source_hash_required")
    raw = book.read(_path(book, path))
    _require(_sha(raw) == expected, "source_hash_mismatch")
    return json.loads(raw)


def _number(value):
    _require(isinstance(value, str) and 0 < len(value) <= 64, "numeric_text_required")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise BrowserSafetyStop("autonomous_planet_invalid_number") from None
    _require(number.is_finite() and number > 0 and abs(number.adjusted()) <= 100, "invalid_number")
    return number


def _result(book, kind, star, *, payload=None, reason=None, evidence=None):
    _unchanged(book)
    value = {
        "schema_version": 1,
        "mode": POLICY,
        "star": star,
        "decision_kind": kind,
        "status": "decided" if payload is not None else "abstained",
        "payload": payload,
        "reason": reason,
        "evidence": evidence or {},
        "reference_pack": reference_pack(),
        "reference_pack_sha256": _sha(_canonical(REFERENCE_PACK)),
        "implementation_sha256": _sha(Path(__file__).read_bytes()),
        "source_sha256": dict(sorted(book.hashes.items())),
        "provenance": "reference_prediction",
        "learned": False,
        "scientific_verified": False,
        "correctness_verified": False,
        "course_thresholds_verified": False,
        "training_label": False,
        "confidence_calibrated": False,
        "browser_actions": 0,
        "task_completed": False,
        "project_completed": False,
    }
    value["decision_sha256"] = _sha(_canonical(value))
    return value


def decide_planet_class(run_history, capture_dir, capture_sha256, *, expected_star):
    """Choose the nearest labelled Solar-System analogue, not a course cutoff."""
    book = _Evidence(run_history)
    directory = _clean(book, _path(book, capture_dir))
    report = _pinned(book, directory / "observation.json", capture_sha256)
    _require(book.capture(directory) == report, "capture_changed")
    mapping = map_planet_capture(report, capture_sha256=capture_sha256)
    _identity(mapping["star_name"], expected_star)
    values = mapping["observation"]["values"]
    _require(values["has_planet"] == "Yes", "positive_planet_required")
    fields = values["browser_field_map"]
    names = ("planet_mass", "planet_radius", "planet_density")
    units = ("MEarth", "REarth", "g/cm3")
    _require(all(fields[n]["unit"] == u for n, u in zip(names, units, strict=True)), "units_changed")
    numbers = [_number(fields[n]["current_value"]) for n in names]
    distances = []
    with localcontext() as context:
        context.prec = 50
        for name, kind, mass, diameter, density in REFERENCE_PACK["prototypes"]:
            prototype = (Decimal(mass) / Decimal("5.97"), Decimal(diameter) / 12756, Decimal(density) / 1000)
            distance = sum((actual / ref).ln() ** 2 for actual, ref in zip(numbers, prototype, strict=True))
            distances.append({"name": name, "class": kind, "squared_log_distance": str(distance)})
    distances.sort(key=lambda entry: (Decimal(entry["squared_log_distance"]), entry["name"]))
    best = distances[0]
    runner = next(entry for entry in distances if entry["class"] != best["class"])
    evidence = {
        "measurements": {
            n: {"value": fields[n]["current_value"], "unit": u} for n, u in zip(names, units, strict=True)
        },
        "ranked_prototypes": distances,
        "metric_is_not_probability": True,
        "measurement_uncertainty_propagated": False,
    }
    if abs(Decimal(best["squared_log_distance"]) - Decimal(runner["squared_log_distance"])) <= Decimal(
        "1e-20"
    ):
        return _result(
            book, "planet_class", expected_star, reason="ambiguous_prototype_distance", evidence=evidence
        )
    rationale = (
        f"Approximate NASA Solar-System analogue: {best['name']} is nearest in equal-weight log mass, "
        "radius and density. This is a versioned reference heuristic, not a HabWorlds class boundary, "
        "learned classification, or scientific confirmation."
    )
    return _result(
        book,
        "planet_class",
        expected_star,
        payload={"name": best["class"], "rationale": rationale},
        evidence=evidence,
    )


def _profile(png):
    _require(isinstance(png, bytes) and len(png) <= 4_000_000, "unsupported_spectrum_image")
    with Image.open(io.BytesIO(png)) as image:
        _require(image.format == "PNG" and image.mode in {"RGB", "RGBA"}, "unsupported_spectrum_image")
        width, height = image.size
        _require(
            200 <= width <= 1200 and 120 <= height <= 800 and width * height <= 600_000,
            "unsupported_spectrum_geometry",
        )
        if image.mode == "RGBA":
            _require(np.all(np.asarray(image.getchannel("A")) == 255), "transparent_spectrum_pixels")
        pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    r, g, b = pixels.transpose(2, 0, 1)
    grid = (r == g) & (g == b) & (r >= 8) & (r <= 60)
    lines = np.flatnonzero(grid.sum(axis=1) >= width * 0.85)
    groups = np.split(lines, np.flatnonzero(np.diff(lines) > 1) + 1)
    _require(len(groups) >= 3 and all(len(group) for group in groups), "spectrum_grid_unresolved")
    y0, y1 = int(groups[0][0]), int(groups[-1][-1])
    columns = np.flatnonzero(grid[lines].sum(axis=0) >= len(lines) * 0.8)
    _require(len(columns) >= width * 0.85, "spectrum_grid_unresolved")
    x0, x1 = int(columns[0]), int(columns[-1])
    _require(y1 - y0 >= 50 and x1 - x0 >= 180, "spectrum_grid_unresolved")
    region = pixels[y0 : y1 + 1, x0 : x1 + 1]
    rr, gg, bb = region.transpose(2, 0, 1)
    grey = (rr == gg) & (gg == bb) & (rr >= 65)
    blue = (gg - rr >= 20) & (bb - gg >= 8) & (bb - rr >= 30)
    _require(not np.any((region.max(axis=2) >= 65) & ~grey & ~blue), "unknown_spectrum_palette")
    present = grey.any(axis=0)
    indices = np.flatnonzero(present)
    _require(
        len(indices) >= len(present) * 0.7 and indices[0] <= 2 and indices[-1] >= len(present) - 3,
        "incomplete_model_trace",
    )
    _require(np.diff(indices).max() <= 4, "incomplete_model_trace")
    model = np.max(np.where(grey, np.arange(len(grey))[:, None], -1), axis=0)
    model = np.interp(np.arange(len(model)), indices, model[indices])
    actual = np.max(np.where(blue, np.arange(len(blue))[:, None], -1), axis=0).astype(float)
    _require(
        np.all((model > 0) & (model < len(grey) - 1)) and np.all(actual < len(blue) - 1),
        "clipped_spectrum_trace",
    )
    outside = pixels.copy()
    outside[y0 : y1 + 1, x0 : x1 + 1] = 0
    return (x0, y0, x1, y1), model, actual, _sha(outside.tobytes())


def match_visible_gas_crops(crops):
    """Conservative raster-only union fit. Scores are pixel errors, not confidence.

    All seven candidate crops are required. Grid, palette and complete model
    traces must agree. A gas subset is accepted only with a unique <=1.5 pixel
    RMS fit and >=90% observed dip-column coverage; every selected gas must
    contribute at least three uniquely covered columns. Thresholds describe the
    image heuristic, not chemistry or grading boundaries.
    """
    _require(
        isinstance(crops, dict) and set(crops) == {"baseline", *GASES}, "complete_gas_comparison_required"
    )
    geometry, baseline, actual, outside = _profile(crops["baseline"])
    observed = np.maximum(actual - baseline, 0)
    observed[observed < 2] = 0
    _require(np.count_nonzero(observed) >= 3, "target_dips_unresolved")
    templates = []
    for gas in GASES:
        bounds, model, candidate_actual, other = _profile(crops[gas])
        _require(
            bounds == geometry and other == outside and len(model) == len(baseline),
            "spectrum_alignment_changed",
        )
        exposed = candidate_actual >= 0
        _require(
            np.all(np.abs(candidate_actual[exposed] - np.maximum(actual, baseline)[exposed]) <= 1),
            "target_spectrum_changed",
        )
        depth = np.maximum(model - baseline, 0)
        depth[depth < 2] = 0
        _require(np.count_nonzero(depth) >= 3, "candidate_dips_unresolved")
        templates.append(depth)
    fits = []
    for count in range(1, len(GASES) + 1):
        for subset in itertools.combinations(range(len(GASES)), count):
            prediction = np.maximum.reduce([templates[i] for i in subset])
            coverage = float(np.count_nonzero((prediction > 0) & (observed > 0)) / np.count_nonzero(observed))
            active = (prediction > 0) | (observed > 0)
            rms = float(np.sqrt(np.mean((prediction[active] - observed[active]) ** 2)))
            unique = all(
                np.count_nonzero(
                    (templates[i] > 0)
                    & (
                        (
                            np.maximum.reduce([templates[j] for j in subset if j != i])
                            if len(subset) > 1
                            else np.zeros_like(observed)
                        )
                        == 0
                    )
                )
                >= 3
                for i in subset
            )
            fits.append((rms, tuple(GASES[i] for i in subset), coverage, unique))
    fits.sort()
    best = fits[0]
    _require(
        best[0] <= 1.5 and best[2] >= 0.9 and best[3] and fits[1][0] - best[0] > 0.25,
        "ambiguous_gas_pixel_match",
    )
    return {
        "gases": list(best[1]),
        "rms_pixels": best[0],
        "runner_up_rms_pixels": fits[1][0],
        "observed_dip_coverage": best[2],
        "plot_pixel_bounds": list(geometry),
        "physical_gas_identity_verified": False,
    }


def _comparison(book, directory, checksum, star):
    directory = _clean(book, _path(book, directory))
    report = _pinned(book, directory / "report.json", checksum)
    _identity(report.get("star"), star)
    candidates = report.get("candidates")
    _require(
        isinstance(candidates, list) and len(candidates) == len(GASES) and set(candidates) == set(GASES),
        "complete_gas_comparison_required",
    )
    _require(
        report.get("baseline_restored") is True
        and report.get("gas_selection_inferred") is False
        and report.get("learned_gas_identification") is False
        and report.get("task_completed") is False
        and type(report.get("checkbox_writes")) is int
        and report["checkbox_writes"] == 14,
        "invalid_gas_comparison",
    )
    before, after = _gas_writes(
        book, directory, [(g, v) for g in candidates for v in (True, False)], report["star"]
    )
    _require(_same_habitat(before, after), "gas_baseline_changed")
    hashes = report.get("chart_sha256")
    _require(isinstance(hashes, dict) and set(hashes) == {"baseline", *GASES}, "gas_crop_inventory")
    crops = {}
    for name, expected in hashes.items():
        crops[name] = book.read(directory / (name + ".png"))
        _require(_sha(crops[name]) == expected, "gas_crop_changed")
    return directory, crops


def decide_gases(run_history, comparison_dir, comparison_sha256, *, expected_star):
    book = _Evidence(run_history)
    _, crops = _comparison(book, comparison_dir, comparison_sha256, expected_star)
    try:
        match = match_visible_gas_crops(crops)
    except BrowserSafetyStop as exc:
        return _result(book, "gases", expected_star, reason=str(exc))
    rationale = "Approximate pixel-template comparison of the visible blue target and grey candidate spectra; selected combination has a unique supported profile fit. Not learned gas identity, a grading answer, or physical confirmation."
    return _result(
        book,
        "gases",
        expected_star,
        payload={"gases": match["gases"], "rationale": rationale},
        evidence={**match, "greenhouse_increment_requires_selected_combination_readback": True},
    )


def decide_greenhouse(
    run_history, comparison_dir, comparison_sha256, selection_dir, selection_sha256, *, expected_star
):
    book = _Evidence(run_history)
    comparison, _ = _comparison(book, comparison_dir, comparison_sha256, expected_star)
    selection = _clean(book, _path(book, selection_dir))
    report = _pinned(book, selection / "report.json", selection_sha256)
    _identity(report.get("star"), expected_star)
    selected, after = _gas_sources(book, comparison, selection, report["star"])
    _require(report == selected, "selection_changed")
    text = _mapping(after)["observation"]["values"]["readouts"]["absorption"]["display_text"]
    calculator = HabitabilityCalculator()
    result = calculator.greenhouse_reference(text)
    evidence = {
        "selected_gases": selected["gases"],
        "absorption_percent": text,
        "knowledge_pack_sha256": calculator.pack.checksum,
        "lookup": result,
    }
    if not result["ok"] or result["increment_kelvin"] == 0:
        return _result(
            book,
            "greenhouse",
            expected_star,
            reason=result.get("error", "no_explicit_native_none_option"),
            evidence=evidence,
        )
    return _result(
        book,
        "greenhouse",
        expected_star,
        payload={"supplied_greenhouse_increment": result["increment_kelvin"]},
        evidence=evidence,
    )


def _phase(book, directory, checksum, star):
    directory = _clean(book, _path(book, directory))
    phase = _pinned(book, directory / "confirmed.json", checksum)
    before, after = book.capture(directory / "initial"), book.capture(directory / "after")
    mapping = _mapping(after)
    _identity(mapping["star_name"], star)
    hashes = phase.get("evidence", {}).get("source_sha256")
    _require(isinstance(hashes, dict) and len(hashes) == 4, "phase_sources_required")
    for path, expected in hashes.items():
        _pinned(book, path, expected)
    source = [book.path(path).parent for path in hashes if Path(path).name == "confirmed.json"]
    _require(len(source) == 1, "chamber_identity_required")
    chamber = _clean(book, source[0])
    _require(
        not any(
            (chamber / name).exists()
            for name in ("cleanup-stopped.json", "unconfirmed-controls.json", "unconfirmed-helper.png")
        ),
        "uncertain_chamber",
    )
    query, result = book.json(chamber / "reserved.json"), book.json(chamber / "confirmed.json")
    _require(
        set(map(book.path, hashes))
        == {
            chamber / p
            for p in (
                "reserved.json",
                "confirmed.json",
                "task-before/observation.json",
                "task-before/manifest.json",
            )
        },
        "chamber_source_inventory",
    )
    reference = confirmed_phase_reference(directory, mapping)
    for path, expected in reference["source_sha256"].items():
        _require(_sha(book.read(path)) == expected, "phase_source_changed")
    cb, ca = book.capture(chamber / "task-before"), book.capture(chamber / "task-after")
    values = mapping["observation"]["values"]
    icons = result.get("icons", [])
    _require(
        result.get("conditions_verified") is True
        and result.get("task_completed") is False
        and result.get("source") == "visible_chamber_indicator"
        and result.get("phase") == values["water_phase"].lower()
        and len(icons) == 3
        and {v.get("phase") for v in icons} == {"solid", "liquid", "gas"},
        "chamber_indicator_unverified",
    )
    _require(
        all(
            type(v.get("paint", {}).get("opacity")) in {float, int}
            and v["paint"]["opacity"] == int(v["phase"] == result["phase"])
            for v in icons
        )
        and next(v for v in icons if v["phase"] == result["phase"]).get("fully_exposed") is True,
        "ambiguous_chamber_paint",
    )
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
        and query["task_answer_write"] is False
        and query["action_source"] in {"reference_diagnostic", "checkpoint"}
        and (query["pressure_unit"], query["temperature_unit"]) == ("atm", "K")
        and all(result.get(k) == v for k, v in query.items()),
        "chamber_query_changed",
    )
    for unit, field, expected in (
        ("atm", "pressure", values["measurements"]["pressure"]["display_text"]),
        ("K", "temperature", values["readouts"]["surface_temp"]["display_text"]),
    ):
        _require(
            _number(query[field]) == _number(expected) == _number(result["visible_readback"][unit]),
            "chamber_conditions_changed",
        )
    _require(
        book.json(chamber / "observed.json") == result
        and book.json(chamber / "close-reserved.json") == {"retry_allowed": False}
        and book.read(chamber / "chamber.png").startswith(b"\x89PNG\r\n\x1a\n")
        and comparable_screen(cb) == comparable_screen(ca)
        and _same_view(ca, before),
        "chamber_cleanup_unverified",
    )
    _require(
        menu_projection(before, _mapping(before), "water_phase")
        == menu_projection(after, mapping, "water_phase")
        and _mapping(before)["observation"]["values"]["water_phase"] is None
        and phase.get("automatic_retry") is False
        and type(phase.get("max_menu_writes")) is int
        and phase["max_menu_writes"] == 1
        and type(phase.get("numeric_writes")) is int
        and phase["numeric_writes"] == 0,
        "phase_transition_changed",
    )
    return reference


def decide_habitability(
    run_history, phase_dir, phase_sha256, planet_class_dir, planet_class_sha256, *, expected_star
):
    book = _Evidence(run_history)
    directory = _clean(book, _path(book, planet_class_dir))
    receipt = _pinned(book, directory / "confirmed.json", planet_class_sha256)
    intent = book.json(directory / "reserved.json")
    _require(
        receipt == {**intent, "readback_verified": True}
        and receipt.get("readback_verified") is True
        and receipt.get("value") == "terrestrial"
        and receipt.get("kind") == "SELECT"
        and receipt.get("action_source") == "reference_diagnostic"
        and receipt.get("correctness_verified") is False
        and receipt.get("task_completed") is False
        and type(receipt.get("numeric_writes")) is int
        and receipt["numeric_writes"] == 0
        and type(receipt.get("max_class_writes")) is int
        and receipt["max_class_writes"] == 1,
        "terrestrial_receipt_required",
    )
    _identity(receipt.get("star"), expected_star)
    capture = book.capture(directory / "after")
    _identity(
        map_planet_capture(capture, capture_sha256=screen_identity(capture))["star_name"], expected_star
    )
    reference = _phase(book, phase_dir, phase_sha256, expected_star)
    choice = "habitable" if reference["phase"] == "Liquid" else "not_habitable"
    rationale = f"Course-reference rule: this explicitly reconstructed terrestrial planet has confirmed {reference['phase'].lower()} water-phase conditions at the displayed surface pressure and temperature. Only terrestrial composition and liquid-water-compatible conditions are required; lifetime and water-vapor detection are not added criteria. Scientific correctness remains unverified."
    return _result(
        book,
        "habitability",
        expected_star,
        payload={"choice": choice, "rationale": rationale},
        evidence={
            "planet_class": "terrestrial",
            "water_phase": reference["phase"],
            "pressure_atm": reference["pressure"],
            "temperature_kelvin": reference["temperature"],
            "lifetime_used": False,
            "water_vapor_used": False,
        },
    )


def decide_planet_handoff(*, phase, run_history, owner_state, expected_star):
    """Use explicit owner descriptors only; no directory discovery or UI calls."""
    _require(
        isinstance(owner_state, dict)
        and owner_state.get("phase") == phase
        and owner_state.get("finished") is False
        and owner_state.get("failure_reason") is None,
        "inactive_handoff",
    )
    _identity(owner_state.get("star"), expected_star)
    paths = owner_state.get("artifact_paths", {})
    book = _Evidence(run_history)
    if phase == "awaiting_planet_class":
        positive, derived = (_path(book, paths[k]) for k in ("positive_dir", "derived_dir"))
        descriptor = book.json(positive / "derived-completed.json")
        files = descriptor.get("files")
        _require(isinstance(files, dict) and 1 <= len(files) <= 1024, "derived_descriptor_required")
        for path, digest in files.items():
            _require(_path(book, path).is_relative_to(derived), "foreign_derived_source")
            _require(_sha(book.read(_path(book, path))) == digest, "derived_source_changed")
        report = descriptor.get("report", {})
        _require(
            _path(book, report["path"]) == derived / "report.json" and report.get("files") == len(files),
            "derived_report_descriptor",
        )
        _pinned(book, derived / "report.json", report.get("sha256"))
        capture = derived / "native-copies/copy-04-after"
        key = str((capture / "observation.json").relative_to(book.history))
        result = decide_planet_class(book.history, capture, files.get(key), expected_star=expected_star)
    elif phase in {"awaiting_gases", "awaiting_habitability"}:
        component = owner_state.get("terrestrial_component", {})
        _require(
            component.get("phase") == phase
            and component.get("finished") is False
            and component.get("failure_reason") is None,
            "inactive_terrestrial_handoff",
        )
        _identity(component.get("star"), expected_star)
        sources = component.get("decision_sources", {})
        directory = _path(book, paths["terrestrial_dir"])
        if phase == "awaiting_gases":
            source = sources["gas_comparison"]
            _require(
                _path(book, source["directory"]) == directory / "gas-comparison", "gas_source_path_changed"
            )
            result = decide_gases(
                book.history,
                directory / "gas-comparison",
                source["report_sha256"],
                expected_star=expected_star,
            )
        else:
            source, classification = sources["phase"], sources["planet_class"]
            _require(
                _path(book, source["directory"]) == directory / "phase"
                and _path(book, classification["directory"]) == _path(book, paths["planet_class_dir"]),
                "habitability_source_path_changed",
            )
            result = decide_habitability(
                book.history,
                directory / "phase",
                source["confirmed_sha256"],
                _path(book, classification["directory"]),
                classification["confirmed_sha256"],
                expected_star=expected_star,
            )
    else:
        raise BrowserSafetyStop("autonomous_planet_unsupported_handoff")
    _unchanged(book)
    result["source_sha256"].update(book.hashes)
    result.pop("decision_sha256")
    result["decision_sha256"] = _sha(_canonical(result))
    return result
