"""Offline, hash-pinned sampled-tooltip reference measurements.

No browser, answer authority, physical period/minimum claim, or legacy raster
receipt is produced here. Compatibility intervals describe recorded brackets,
not scientific uncertainty. Overview hint positions never enter the fit.
"""

import hashlib
import io
import json
import math
import re
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from PIL import Image

from .browser import BrowserSafetyStop
from .browser_observation_progress import trace_progress
from .browser_planet_chart import verify_tooltip_day
from .browser_raster_planet_evidence import _Owned
from .browser_transit_sampling import baseline_position
from .contracts import RuntimeEvent

MODE = "approximate_reference_visible_tooltips_v1"
TWO_MODE = "approximate_reference_two_visible_tooltips_v1"
SENSOR = "bounded_shallow_feature_visible_tooltip_diagnostic"
_LEGACY_CAPTURES = ("before", "zoom-1", "zoom-2", "zoom-3", "zoom-4", "readable")
CAPTURES = ("before", "zoom-1", "zoom-2", "zoom-3", "zoom-4", "zoom-5", "zoom-6", "readable")
DIAGNOSTIC_FILES = frozenset(
    ("scope.json", "report.json", "events.jsonl")
    + tuple(name + suffix for name in CAPTURES for suffix in (".json", ".png"))
)
_LEGACY_FILES = frozenset(
    ("scope.json", "report.json", "events.jsonl")
    + tuple(name + suffix for name in _LEGACY_CAPTURES for suffix in (".json", ".png"))
)
SENSOR_FLAGS = {
    "answer_writes": 0,
    "planet_decision": None,
    "period_evidence_verified": False,
    "minimum_depth_verified": False,
    "scientific_verified": False,
    "learned_perception": False,
    "training_label": False,
    "task_completed": False,
    "answer_authorized": False,
    "observation_restart": False,
    "automatic_retry": False,
}
METHOD = {
    "version": MODE,
    "diagnostics": [3, 8],
    "samples_per_diagnostic": [3, 8],
    "observed_day_interval": [0, 5000],
    "bracket": "nearest sampled 100% before and after one contiguous sampled decline run",
    "recurrence": "one arithmetic progression point strictly inside each ordered bracket; consecutive indices only",
    "representative": "midpoint of feasible recurrence interval and then midpoint of feasible phase interval",
    "compatibility_interval": "open deterministic bracket compatibility; NOT physical uncertainty or confidence",
    "brightness": "exact Decimal(100) minus minimum confirmed sampled brightness; NOT resolved transit minimum",
    "decimal_precision": 50,
    "overview_hint_positions_used": False,
    "sensor_recipes": {
        "guarded_pointer_clear_v1": "four zooms, optional vertical pan, pointer clear, adjacent pixel hovers",
        "daily_focus_v2": "six zooms, optional vertical pan, pointer clear, consecutive exact requested-day hovers",
    },
    "mixed_sensor_recipes_allowed": False,
    "limitations": [
        "Unobserved days and missed or aliased events remain unresolved.",
        "A bracket contains a sampled decline, not a proven unique physical transit.",
        "The maximum sampled decline can underestimate the physical minimum's drop.",
        "Tooltip display precision does not supply physical error bounds.",
        "Hash and event validation is not independent proof of browser execution or science.",
    ],
}
TWO_METHOD = {
    **json.loads(json.dumps(METHOD)),
    "version": TWO_MODE,
    "diagnostics": [2, 2],
    "recurrence": "No recurrence confirmation: one spacing between two ordered bracketed sampled declines, assumed consecutive",
    "representative": "Midpoint of the open interval of compatible single spacings; midpoint of the compatible first bracket position",
    "sensor_recipes": {"daily_focus_v2": METHOD["sensor_recipes"]["daily_focus_v2"]},
    "overview_hint_policy": "exact_two_overview_two_row_blue_hints_v1",
    "confirmed_feature_count": 2,
    "observed_interval_count": 1,
    "consistency_redundancy": 0,
    "consecutive_events_assumed": True,
    "recurrence_confirmed": False,
    "limitations": [
        *METHOD["limitations"],
        "Two sampled declines provide one spacing and zero repeated-spacing consistency checks.",
        "Treating the two declines as consecutive events is an explicit assumption, not an observed fact.",
    ],
}


def _require(value, reason):
    if not value:
        raise BrowserSafetyStop("tooltip_reference_" + reason)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def measurement_manifest(mode=MODE):
    _require(type(mode) is str and mode in {MODE, TWO_MODE}, "unsupported_measurement_mode")
    value = json.loads(json.dumps(METHOD if mode == MODE else TWO_METHOD))
    return {**value, "sha256": _digest(value)}


def _exact(actual, expected):
    return type(actual) is type(expected) and actual == expected


def _flags(value):
    _require(all(_exact(value.get(k), v) for k, v in SENSOR_FLAGS.items()), "unsupported_flags")


def _number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _star(value):
    _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", value), "invalid_star")
    return value.casefold()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda _: _require(False, "nonfinite_json")
    )


class _Sources:
    def __init__(self, history):
        self.owner, self.hashes, self.directories = _Owned(history), {}, set()

    def directory(self, value):
        _require(isinstance(value, (str, Path)), "invalid_directory")
        path = Path(value)
        path = self.owner.history / path if not path.is_absolute() else path
        path = self.owner.clean(path)
        self.directories.add(self.owner.relative(path))
        return path

    def read(self, path, checksum):
        _require(
            isinstance(checksum, str) and re.fullmatch(r"[a-f0-9]{64}", checksum), "explicit_hash_required"
        )
        raw = self.owner.read(path, checksum, limit=2_000_000)
        name = self.owner.relative(path)
        _require(name not in self.hashes or self.hashes[name] == checksum, "conflicting_source_pin")
        self.hashes[name] = checksum
        self.directories.add(self.owner.relative(path.parent))
        return raw

    def unchanged(self):
        for directory in self.directories:
            self.owner.clean(self.owner.history / directory)
        for name, checksum in self.hashes.items():
            self.owner.read(self.owner.history / name, checksum)


def _capture(raw, name):
    value = _json(raw[name + ".json"])
    _require(isinstance(value, dict), "invalid_capture")
    _flags(value)
    png = raw[name + ".png"]
    _require(value.get("chart_sha256") == hashlib.sha256(png).hexdigest(), "capture_png_disagrees")
    with Image.open(io.BytesIO(png)) as image:
        _require(image.size in {(280, 195), (280, 196)}, "unsupported_crop")
        _require(image.convert("RGBA").getextrema()[3] == (255, 255), "transparent_crop")
    for key, coordinate in (("time_axis_labels", "center_x"), ("flux_axis_labels", "center_y")):
        labels = value.get(key)
        _require(isinstance(labels, list) and 1 <= len(labels) <= 64, "invalid_capture_axes")
        for row in labels:
            _require(isinstance(row, dict) and set(row) == {"value", coordinate}, "invalid_axis_label")
            _require(
                type(row["value"]) in {str, int, float} and _number(row[coordinate]), "invalid_axis_label"
            )
            _require(math.isfinite(float(row["value"])), "invalid_axis_label")
    return value


def _sample(value):
    _require(
        isinstance(value, dict) and set(value) == {"day", "brightness_percent", "source"}, "invalid_sample"
    )
    _require(type(value["day"]) is int and 0 <= value["day"] <= 5000, "sample_outside_window")
    _require(value["source"] == "visible_hover_tooltip", "unsupported_sample_source")
    text = value["brightness_percent"]
    _require(
        isinstance(text, str) and re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{1,60})?", text), "invalid_brightness"
    )
    number = Decimal(text)
    _require(0 <= number <= 100, "invalid_brightness")
    return number


def _events(raw, report, scope, current, run_id, recipe):
    lines = raw.splitlines()
    _require(18 <= len(lines) <= 34 and all(lines), "invalid_event_count")
    events = [_json(line) for line in lines]
    for index, event in enumerate(events):
        RuntimeEvent.model_validate(event)
        _require(
            _exact(event.get("version"), 1)
            and _exact(event.get("sequence"), index)
            and event.get("run_id") == run_id,
            "event_identity_or_order",
        )
    _require(
        events[0]["event"] == "observation"
        and events[0]["payload"] == {"chart": {"source": "visible_tooltips", "star": report["star"]}},
        "initial_observation_mismatch",
    )
    _require(events[-1]["event"] == "episode_summary" and events[-1]["payload"] == report, "summary_mismatch")
    middle = events[1:-1]
    _require(len(middle) % 2 == 0, "missing_action_confirmation")
    actions, samples, coordinates = [], [], []
    for i in range(0, len(middle), 2):
        proposed, confirmed = middle[i : i + 2]
        p, c = proposed["payload"], confirmed["payload"]
        _require(
            proposed["event"] == "action_proposed" and confirmed["event"] == "action_result",
            "missing_action_confirmation",
        )
        _require(
            _exact(p.get("sequence"), i // 2 + 1)
            and _exact(c.get("sequence"), i // 2 + 1)
            and p.get("surface") == "chart"
            and c.get("task_completed") is False,
            "action_sequence_or_scope",
        )
        kind = p.get("kind")
        _require(kind in {"SCROLL", "DRAG", "HOVER"}, "unsupported_action")
        if kind == "HOVER" and "chart_sample" in c:
            keys = {"kind", "surface", "sequence", "x_fraction", "y_fraction"}
            if recipe == "daily_focus_v2":
                keys.add("requested_day")
            _require(
                set(p) == keys and set(c) == {"sequence", "chart_sample", "task_completed"},
                "unsupported_sample_event",
            )
            _sample(c["chart_sample"])
            if recipe == "daily_focus_v2":
                _require(_exact(p.get("requested_day"), c["chart_sample"]["day"]), "requested_day_disagrees")
            x, y = p["x_fraction"], p["y_fraction"]
            _require(
                _number(x) and _number(y) and 30 <= x * 280 <= 260 and y == 80 / 195,
                "invalid_sample_coordinate",
            )
            verify_tooltip_day(current["time_axis_labels"], x * 280, c["chart_sample"]["day"])
            samples.append(c["chart_sample"])
            coordinates.append(x * 280)
            actions.append("sample")
        elif kind == "HOVER":
            _require(
                set(p) == {"kind", "surface", "sequence", "purpose", "outside_chart"}
                and p.get("purpose") == "remove_pointer_overlay"
                and p.get("outside_chart") is True
                and set(c) == {"sequence", "pointer_outside_chart", "task_completed"}
                and c.get("pointer_outside_chart") is True,
                "unsupported_hover",
            )
            actions.append("clear")
        elif kind == "SCROLL":
            _require(
                set(p) == {"kind", "surface", "sequence", "x_fraction", "y_fraction", "delta_y"}
                and set(c) == {"sequence", "chart_accessibility", "task_completed"}
                and isinstance(c["chart_accessibility"], str),
                "unsupported_zoom_event",
            )
            _require(_number(p.get("delta_y")), "invalid_zoom")
            _require(
                all(_number(p.get(k)) and 0 < p[k] < 1 for k in ("x_fraction", "y_fraction")), "invalid_zoom"
            )
            actions.append(p["delta_y"])
        else:
            _require(
                set(p) == {"kind", "surface", "sequence", "start_fraction", "end_fraction"}
                and set(c) == {"sequence", "chart_accessibility", "task_completed"}
                and isinstance(c["chart_accessibility"], str),
                "unsupported_pan_event",
            )
            a, b = p.get("start_fraction"), p.get("end_fraction")
            _require(
                isinstance(a, list)
                and isinstance(b, list)
                and len(a) == len(b) == 2
                and all(_number(v) and 0 < v < 1 for v in a + b)
                and a[0] == b[0],
                "invalid_pan",
            )
            actions.append("pan")
    zooms = [-500] * 5 + [-250] if recipe == "daily_focus_v2" else [-500, -500, -500, 250]
    _require(actions[: len(zooms)] == zooms, "unsupported_sensor_recipe")
    tail = actions[len(zooms) :]
    if tail and tail[0] == "pan":
        tail = tail[1:]
    _require(tail and tail[0] == "clear", "pointer_clear_required")
    tail = tail[1:]
    _require(3 <= len(samples) <= 8 and tail == ["sample"] * len(samples), "unsupported_sensor_recipe")
    if recipe == "daily_focus_v2":
        _require(all(b["day"] - a["day"] == 1 for a, b in pairwise(samples)), "nonconsecutive_requested_days")
    else:
        _require(
            all(abs(b - a - 1) <= 1e-7 for a, b in pairwise(coordinates)), "nonadjacent_hover_coordinates"
        )
    _require(
        _exact(report.get("browser_actions"), len(actions)) and len(actions) <= scope["max_actions"],
        "action_limit",
    )
    _require(samples == report.get("samples"), "sample_journal_mismatch")
    return samples, recipe


def _diagnostic(book, spec, expected_star):
    _require(isinstance(spec, dict) and set(spec) == {"directory", "files"}, "invalid_diagnostic_spec")
    directory = book.directory(spec["directory"])
    pins = spec["files"]
    _require(isinstance(pins, dict) and "scope.json" in pins, "incomplete_file_pins")
    scope = _json(book.read(directory / "scope.json", pins["scope.json"]))
    _require(isinstance(scope, dict), "invalid_sensor_scope")
    declared = scope.get("sensor_recipe")
    _require(declared in {None, "daily_focus_v2"}, "unsupported_sensor_recipe")
    recipe = declared or "guarded_pointer_clear_v1"
    captures = CAPTURES if declared else _LEGACY_CAPTURES
    _require(set(pins) == (DIAGNOSTIC_FILES if declared else _LEGACY_FILES), "incomplete_file_pins")
    raw = {name: book.read(directory / name, pins[name]) for name in sorted(pins)}
    scope, report = (_json(raw[name]) for name in ("scope.json", "report.json"))
    _require(isinstance(scope, dict) and isinstance(report, dict), "invalid_sensor_report")
    for item in (scope, report):
        _flags(item)
        _require(
            item.get("mode") == SENSOR and _star(item.get("star")) == expected_star, "mixed_method_or_star"
        )
        _require(item.get("sensor_recipe") == declared, "mixed_sensor_recipes")
    _require(
        _exact(scope.get("max_actions"), 16)
        and _exact(scope.get("max_seconds"), 180)
        and _exact(scope.get("requested_days"), 5000)
        and scope.get("source_owner_recovery") is False,
        "unsupported_sensor_scope",
    )
    source = book.directory(scope.get("source_dir"))
    _require(
        not directory.is_relative_to(source) and not source.is_relative_to(directory),
        "overlapping_diagnostic",
    )
    hashes = scope.get("source_hashes")
    _require(
        isinstance(hashes, dict)
        and set(hashes) == {"report.json", "chart.png"}
        and report.get("source_hashes") == hashes,
        "source_hash_map_mismatch",
    )
    original = _json(book.read(source / "report.json", hashes["report.json"]))
    png = book.read(source / "chart.png", hashes["chart.png"])
    _require(
        isinstance(original, dict)
        and original.get("method") == "rendered_blue_trace_frontier_v1"
        and original.get("source") == "chart_crop_pixels_and_visible_axis_labels"
        and _star(original.get("star")) == expected_star
        and _exact(original.get("requested_days"), 5000)
        and _exact(original.get("browser_actions"), 0)
        and _exact(original.get("answer_writes"), 0)
        and original.get("chart_sha256") == hashes["chart.png"],
        "unsupported_overview_source",
    )
    progress = trace_progress(png, original["time_axis_labels"], requested_days=5000)
    _require(
        progress["endpoint_visible"] and all(_exact(original.get(k), v) for k, v in progress.items()),
        "overview_progress_disagrees",
    )
    hint_metadata = _selected_overview_hint(scope, report, original, png)
    captures = {name: _capture(raw, name) for name in captures}
    before, current = captures["before"], captures["readable"]
    _require(
        before["chart_sha256"] == hashes["chart.png"]
        and all(before[k] == original[k] for k in ("time_axis_labels", "flux_axis_labels")),
        "before_source_mismatch",
    )
    _require(30 <= baseline_position(current["flux_axis_labels"]) <= 120, "unreadable_current_baseline")
    samples, recipe = _events(raw["events.jsonl"], report, scope, current, directory.name, recipe)
    days, values = [s["day"] for s in samples], [_sample(s) for s in samples]
    _require(all(a < b for a, b in pairwise(days)), "nonmonotonic_samples")
    _require(values[0] == values[-1] == 100 and min(values) < 100, "unbracketed_decline")
    below = [i for i, value in enumerate(values) if value < 100]
    _require(below == list(range(below[0], below[-1] + 1)), "multiple_declines_in_bracket")
    linked = report.get("source_hint_linked")
    hint = report.get("source_hint_day_interval")
    _require(
        isinstance(hint, list)
        and len(hint) == 2
        and all(_number(value) for value in hint)
        and 0 <= hint[0] < hint[1] <= 5000
        and hint == scope.get("source_hint_day_interval"),
        "hint_metadata_disagrees",
    )
    _require(
        type(linked) is bool
        and report.get("status")
        == ("visible_bracketed_dip" if linked else "visible_decline_not_source_linked"),
        "invalid_linkage_status",
    )
    _require(
        report.get("sampled_decline_verified") is True
        and report.get("sampled_day_interval") == [days[0], days[-1]]
        and report.get("minimum_sampled_brightness_percent") == str(min(values))
        and _exact(report.get("daily_coverage_verified"), all(b - a == 1 for a, b in pairwise(days))),
        "sample_summary_disagrees",
    )
    if hint_metadata:
        _require(
            linked is True and all(hint[0] <= days[i] <= hint[1] for i in below),
            "selected_hint_not_linked",
        )
    return {
        "star": report["star"],
        "directory": book.owner.relative(directory),
        "source_dir": book.owner.relative(source),
        "report_sha256": pins["report.json"],
        "sensor_recipe": recipe,
        "current_capture": book.owner.relative(directory / "readable.json"),
        "current_png_sha256": captures["readable"]["chart_sha256"],
        "samples": samples,
        "bracket_days": [days[below[0] - 1], days[below[-1] + 1]],
        "minimum_sampled_brightness_percent": str(min(values)),
        "source_hint_linked": linked,
        "source_hint_day_interval": report.get("source_hint_day_interval"),
        "overview_hint_positions_used": False,
        **hint_metadata,
    }


def _selected_overview_hint(scope, report, original, png):
    """Check new bounded search provenance; hint positions never enter the fit."""
    key = "overview_hint_policy"
    if key not in scope and key not in report:
        _require(
            "overview_hint_recipe" not in scope and "overview_hint_recipe" not in report,
            "unsupported_overview_hint_policy",
        )
        return {}  # Exact legacy sensor evidence remains readable.
    from .browser_shallow_transit_probe import (
        EXACT_TWO_HINT_POLICY,
        OVERVIEW_HINT_POLICIES,
        candidate_columns,
        matches_overview_hint_metadata,
        overview_hint_metadata,
    )

    _require(
        type(scope.get(key)) is str
        and scope.get(key) in OVERVIEW_HINT_POLICIES
        and scope.get(key) == report.get(key)
        and scope.get("sensor_recipe") == "daily_focus_v2",
        "unsupported_overview_hint_policy",
    )
    metadata = overview_hint_metadata(scope[key])
    _require(
        matches_overview_hint_metadata(scope, scope[key])
        and matches_overview_hint_metadata(report, scope[key]),
        "unsupported_overview_hint_recipe",
    )
    index = scope.get("candidate_index")
    count = 2 if scope[key] == EXACT_TWO_HINT_POLICY else 3
    _require(type(index) is int and 0 <= index < count, "invalid_overview_hint_index")
    groups = candidate_columns(png, original["flux_axis_labels"], hint_policy=scope[key])
    group = groups[index]
    columns = scope.get("candidate_columns")
    _require(
        isinstance(columns, list) and all(type(column) is int for column in columns) and columns == group,
        "selected_overview_hint_changed",
    )
    ticks = sorted((float(row["center_x"]), float(row["value"])) for row in original["time_axis_labels"])
    day_per_pixel = (ticks[-1][1] - ticks[0][1]) / (ticks[-1][0] - ticks[0][0])
    expected = [
        max(0.0, ticks[0][1] + (group[0] - 0.5 - ticks[0][0]) * day_per_pixel),
        min(5000.0, ticks[0][1] + (group[-1] + 1.5 - ticks[0][0]) * day_per_pixel),
    ]
    _require(
        scope.get("source_hint_day_interval") == report.get("source_hint_day_interval") == expected,
        "selected_overview_hint_interval_changed",
    )
    return {**metadata, "overview_candidate_index": index}


def _fraction(value):
    with localcontext() as context:
        context.prec = METHOD["decimal_precision"]
        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def _recurrence(brackets):
    lower, upper = Fraction(0), Fraction(5000)
    for i, (left, right) in enumerate(brackets):
        for j in range(i + 1, len(brackets)):
            lower = max(lower, Fraction(brackets[j][0] - right, j - i))
            upper = min(upper, Fraction(brackets[j][1] - left, j - i))
    _require(0 < lower < upper <= 5000, "inconsistent_observed_recurrence")
    period = (lower + upper) / 2
    first = max(Fraction(left) - i * period for i, (left, _) in enumerate(brackets))
    last = min(Fraction(right) - i * period for i, (_, right) in enumerate(brackets))
    _require(first < last, "inconsistent_observed_recurrence")
    phase = (first + last) / 2
    return {
        "value": _fraction(period),
        "unit": "days",
        "exact_representative": {"numerator": period.numerator, "denominator": period.denominator},
        "compatibility_interval": {"lower": _fraction(lower), "upper": _fraction(upper), "endpoints": "open"},
        "fitted_bracket_points": [_fraction(phase + i * period) for i in range(len(brackets))],
        "interpretation": "Observed bracket recurrence only; compatibility is NOT physical uncertainty or confidence.",
    }


def load_tooltip_reference(run_history, diagnostics, *, expected_star, mode=MODE):
    """Read explicitly pinned diagnostic bundles; never write any artifact.

    The unchanged default requires 3–8 diagnostics. TWO_MODE requires exactly
    two daily-focus diagnostics with explicit exact-two search provenance; it
    estimates one assumed-consecutive spacing, never confirmed recurrence.
    Each spec is ``{directory, files: {name: sha256}}``. Daily v2 recipes pin
    exactly DIAGNOSTIC_FILES; read-only v1 compatibility uses its four-zoom
    file set. Recipes cannot mix. Paths belong to run_history. Reports contain
    measurements only, not permission to select Yes, fill answers, or revive a
    stopped owner.
    """
    try:
        _require(type(mode) is str and mode in {MODE, TWO_MODE}, "unsupported_measurement_mode")
        two = mode == TWO_MODE
        star = _star(expected_star)
        _require(
            isinstance(diagnostics, list) and (len(diagnostics) == 2 if two else 3 <= len(diagnostics) <= 8),
            "diagnostic_count",
        )
        book = _Sources(run_history)
        features = [_diagnostic(book, spec, star) for spec in diagnostics]
        _require(len({f["directory"] for f in features}) == len(features), "duplicate_diagnostic")
        _require(len({f["sensor_recipe"] for f in features}) == 1, "mixed_sensor_recipes")
        features.sort(key=lambda feature: feature["bracket_days"])
        if two:
            _require(
                all(
                    feature["sensor_recipe"] == "daily_focus_v2"
                    and feature.get("overview_hint_policy") == TWO_METHOD["overview_hint_policy"]
                    for feature in features
                )
                and [feature.get("overview_candidate_index") for feature in features] == [0, 1],
                "two_mode_requires_exact_two_daily_hints",
            )
        elif any("overview_hint_policy" in feature for feature in features):
            _require(
                len({feature.get("overview_hint_policy") for feature in features}) == 1
                and all(
                    feature.get("overview_hint_policy") != TWO_METHOD["overview_hint_policy"]
                    for feature in features
                )
                and [feature.get("overview_candidate_index") for feature in features] == [0, 1, 2],
                "mixed_or_nonconsecutive_overview_hints",
            )
        brackets = [feature["bracket_days"] for feature in features]
        _require(all(a[1] < b[0] for a, b in pairwise(brackets)), "overlapping_features")
        with localcontext() as context:
            context.prec = 128
            decline = Decimal(100) - min(Decimal(f["minimum_sampled_brightness_percent"]) for f in features)
        period = _recurrence(brackets)
        if two:
            period = {
                **period,
                "estimate_kind": "single_spacing",
                "interpretation": "One spacing between bracketed sampled declines, ASSUMED consecutive; NOT confirmed recurrence, physical uncertainty or confidence.",
            }
        receipt = {
            "schema_version": 1,
            "mode": mode,
            "method": measurement_manifest(mode),
            "status": "approximate_reference_measurements",
            "star": features[0]["star"],
            "observation_limit_days": 5000,
            "features": features,
            "observed_recurrence_compatible": not two,
            "period_days": period,
            "brightness_drop_percent": {
                "value": format(decline, "f"),
                "unit": "percent",
                "physical_bounds": None,
                "interpretation": "Maximum confirmed sampled decline, NOT a resolved physical transit minimum.",
            },
            "approximate": True,
            "source_sha256": dict(sorted(book.hashes.items())),
            "validated_directories": sorted(book.directories),
            **SENSOR_FLAGS,
            "browser_actions": 0,
            "source_browser_actions_recorded": True,
            "daily_coverage_verified": False,
            "missing_events_ruled_out": False,
            "aliasing_ruled_out": False,
            "absence_proven": False,
            "overview_hint_positions_used": False,
        }
        if two:
            receipt.update(
                confirmed_feature_count=2,
                observed_interval_count=1,
                consistency_redundancy=0,
                consecutive_events_assumed=True,
                recurrence_confirmed=False,
                single_spacing_compatible=True,
            )
        book.unchanged()
        return {**receipt, "receipt_sha256": _digest(receipt)}
    except BrowserSafetyStop:
        raise
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError):
        raise BrowserSafetyStop("tooltip_reference_invalid_evidence") from None
