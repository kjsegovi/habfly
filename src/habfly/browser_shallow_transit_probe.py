"""Bounded chart-only follow-up of a caller-selected shallow raster feature.

This is a diagnostic, not a replacement for either frozen window policy. It
never supplies a planet answer, orbit, minimum depth, training label or task
receipt. Every number comes from an exposed, axis-checked native tooltip.
"""

import hashlib
import io
import json
import math
import re
from decimal import Decimal
from fractions import Fraction
from itertools import pairwise

import numpy as np
from PIL import Image

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_observation_progress import PLOT_EXPOSED, trace_mask, trace_progress
from .browser_planet_chart import FluxChartSession, verify_tooltip_day
from .browser_raster_planet_evidence import _Owned
from .browser_transit_sampling import baseline_position, center_baseline
from .contracts import RuntimeEvent
from .presentation_capture import evidence_screenshot

MODE = "bounded_shallow_feature_visible_tooltip_diagnostic"
RECIPE = "daily_focus_v2"
FIRST_THREE_HINT_POLICY = "first_three_overview_groups_v1"
FIRST_THREE_LOW_INTENSITY_HINT_POLICY = "first_three_overview_low_intensity_blue_hints_v2"
FIRST_THREE_QUANTIZED_HINT_POLICY = "first_three_overview_quantized_blue_hints_v3"
FIRST_THREE_TWO_ROW_HINT_POLICY = "first_three_overview_two_row_blue_hints_v4"
EXACT_TWO_HINT_POLICY = "exact_two_overview_two_row_blue_hints_v1"
OVERVIEW_HINT_POLICIES = frozenset(
    {
        FIRST_THREE_HINT_POLICY,
        FIRST_THREE_LOW_INTENSITY_HINT_POLICY,
        FIRST_THREE_QUANTIZED_HINT_POLICY,
        FIRST_THREE_TWO_ROW_HINT_POLICY,
        EXACT_TWO_HINT_POLICY,
    }
)
LOW_INTENSITY_HINT_RECIPE = {
    "version": FIRST_THREE_LOW_INTENSITY_HINT_POLICY,
    "purpose": "Search hints only; ordinary native bracketed tooltip confirmation remains mandatory",
    "blue_direction_rgb": [80, 132, 154],
    "maximum_channel": 64,
    "alpha_range": [0.06, 0.25],
    "maximum_channel_residual": 1,
    "neutral_background": "Within min/max exact-neutral 0..64 pixels in a 3x3 neighborhood",
    "baseline_support": "Original frozen trace-mask pixel immediately above in the same column",
    "baseline_band_pixels": [0.75, 2.5],
    "ordering": "Group raw blue-direction candidates before context validation; union with existing hints; first three without skipping",
    "source": "Student-visible Kanzain crop 1cf0a39fad897ba8c54697f4be74024aa782925c3f2e6eda8f0299b8aff61591",
    "learned_perception": False,
    "scientific_verified": False,
    "answer_authorized": False,
}
QUANTIZED_HINT_RECIPE = {
    **LOW_INTENSITY_HINT_RECIPE,
    "version": FIRST_THREE_QUANTIZED_HINT_POLICY,
    "neutral_background": "One common neutral value within min/max exact-neutral 0..64 pixels in a 3x3 neighborhood",
    "quantization": "Closed 8-bit rounding bins: observed channel minus/plus 0.5",
    "context_solver": "Exact rational feasibility of one shared alpha and one shared neutral for all three RGB channels",
    "raw_candidates": "Unchanged v2 chromatic eligibility, union/grouping/order; context never drops or skips a selected candidate",
    "source": "Student-visible Kanivermyr crop 24292999f3348821a9c03a7d76e0f150fb00db1fbf7bd2390d60ed53b0c07206",
}
TWO_ROW_HINT_RECIPE = {
    **QUANTIZED_HINT_RECIPE,
    "version": FIRST_THREE_TWO_ROW_HINT_POLICY,
    "baseline_support": "Unchanged v3 direct support, or one raw candidate immediately above with unchanged-v3-valid context and original frozen support exactly two rows above in the same column",
    "maximum_support_edges": 2,
    "context_extension": "Nonrecursive: use only the immutable v3 context map; current pixel independently requires the same exact 3x3 observed-neutral common-alpha RGB feasibility",
    "source": "Student-visible Belebron crop e36d8c12feb0b706a3f5417dc769c266c3307ec7a308f9227c367111ed69eae2",
}
EXACT_TWO_HINT_RECIPE = {
    **TWO_ROW_HINT_RECIPE,
    "version": EXACT_TWO_HINT_POLICY,
    "ordering": "Exactly two total current groups, ordered indices 0 and 1; no truncation, skipping or failed-third fallback",
    "purpose": "Search hints only; both require independent native daily baseline-bracketed decline confirmation",
    "measurement_scope": "Single assumed-consecutive event spacing; recurrence is not confirmed",
}
MAX_ACTIONS, MAX_SECONDS = 16, 180
FLAGS = {
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


def _require(value, reason):
    if not value:
        raise BrowserSafetyStop("shallow_probe_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def overview_hint_metadata(policy):
    """Historical recipes stay unchanged; each new hint policy pins its recipe."""
    if policy is None:
        return {}
    _require(type(policy) is str and policy in OVERVIEW_HINT_POLICIES, "invalid_overview_hint_policy")
    value = {"overview_hint_policy": policy}
    recipes = {
        FIRST_THREE_LOW_INTENSITY_HINT_POLICY: LOW_INTENSITY_HINT_RECIPE,
        FIRST_THREE_QUANTIZED_HINT_POLICY: QUANTIZED_HINT_RECIPE,
        FIRST_THREE_TWO_ROW_HINT_POLICY: TWO_ROW_HINT_RECIPE,
        EXACT_TWO_HINT_POLICY: EXACT_TWO_HINT_RECIPE,
    }
    if policy in recipes:
        recipe = json.loads(json.dumps(recipes[policy]))
        value["overview_hint_recipe"] = {
            **recipe,
            "sha256": _sha(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()),
        }
    return value


def matches_overview_hint_metadata(value, policy):
    """Compare canonical JSON, including numeric/boolean types and recipe hash."""
    expected = overview_hint_metadata(policy)
    actual = {key: value[key] for key in expected if key in value}
    if "overview_hint_recipe" not in expected and "overview_hint_recipe" in value:
        return False
    try:
        return json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        return False


def _low_intensity_hints(colors, original_mask, baseline):
    """Chromatic search direction, NOT trace support or a physical blend claim."""
    red, green, blue = (colors[:, :, i] for i in range(3))
    alpha = ((green - red) * 52 + (blue - red) * 74) / (52**2 + 74**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        neutral = (red - 80 * alpha) / (1 - alpha)
        predicted = (
            alpha[:, :, None] * np.array([80, 132, 154]) + (1 - alpha[:, :, None]) * neutral[:, :, None]
        )
    gray = (red == green) & (green == blue) & (red <= 64)
    low = np.full(red.shape, np.inf)
    high = np.full(red.shape, -np.inf)
    lows = np.pad(np.where(gray, red, np.inf), 1, constant_values=np.inf)
    highs = np.pad(np.where(gray, red, -np.inf), 1, constant_values=-np.inf)
    height, width = red.shape
    for dy in range(3):
        for dx in range(3):
            low = np.minimum(low, lows[dy : dy + height, dx : dx + width])
            high = np.maximum(high, highs[dy : dy + height, dx : dx + width])
    supported = np.zeros(original_mask.shape, dtype=bool)
    supported[1:] = original_mask[:-1]
    rows = np.arange(height)[:, None] + 0.5
    candidates = (
        (colors.max(axis=2) <= 64)
        & (alpha >= 0.06)
        & (alpha <= 0.25)
        & (np.abs(predicted - colors).max(axis=2) <= 1)
        & (rows > baseline + 0.75)
        & (rows <= baseline + 2.5)
    )
    return candidates, (neutral >= low) & (neutral <= high) & supported


def _quantized_gray_feasible(rgb, low, high):
    """Search-context compatibility only, not proof of a physical pixel blend.

    Set b=2*(1-alpha)*neutral. Every RGB rounding bin and the observed gray
    range supplies a lower/upper affine bound on b. A common b exists exactly
    when every lower bound is <= every upper bound for the SAME alpha. Pairwise
    inequalities reduce to one exact rational alpha interval; independent
    per-channel fits or floating-point tolerances cannot authorize a hint.
    """
    lower = [(-2 * low, 2 * low)]
    upper = [(-2 * high, 2 * high)]
    for value, direction in zip(rgb, (80, 132, 154), strict=True):
        lower.append((-2 * direction, 2 * value - 1))
        upper.append((-2 * direction, 2 * value + 1))
    start, end = Fraction(3, 50), Fraction(1, 4)
    for lower_slope, lower_intercept in lower:
        for upper_slope, upper_intercept in upper:
            slope = lower_slope - upper_slope
            bound = upper_intercept - lower_intercept
            if slope > 0:
                end = min(end, Fraction(bound, slope))
            elif slope < 0:
                start = max(start, Fraction(bound, slope))
            elif bound < 0:
                return False
            if start > end:
                return False
    return True


def _quantized_hint_context(colors, original_mask, candidates):
    context = np.zeros(candidates.shape, dtype=bool)
    for y, x in zip(*np.nonzero(candidates), strict=True):
        if y == 0 or not original_mask[y - 1, x]:
            continue
        neighborhood = colors[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
        red, green, blue = (neighborhood[:, :, i] for i in range(3))
        gray = red[(red == green) & (green == blue) & (red <= 64)]
        if gray.size:
            context[y, x] = _quantized_gray_feasible(
                tuple(map(int, colors[y, x])), int(gray.min()), int(gray.max())
            )
    return context


def _two_row_hint_context(colors, original_mask, candidates):
    """Extend search context once through an unchanged-v3-valid raw parent.

    Newly accepted pixels cannot support another pixel. Neither this context
    map nor its original/parent support changes the frozen trace mask or the
    raw hint candidates; native bracketed tooltip confirmation is still needed.
    """
    direct = _quantized_hint_context(colors, original_mask, candidates)
    context = direct.copy()
    for y, x in zip(*np.nonzero(candidates & ~direct), strict=True):
        if y < 2 or not candidates[y - 1, x] or not direct[y - 1, x] or not original_mask[y - 2, x]:
            continue
        neighborhood = colors[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
        red, green, blue = (neighborhood[:, :, i] for i in range(3))
        gray = red[(red == green) & (green == blue) & (red <= 64)]
        if gray.size:
            context[y, x] = _quantized_gray_feasible(
                tuple(map(int, colors[y, x])), int(gray.min()), int(gray.max())
            )
    return context


def candidate_columns(png, flux_labels, *, overview=True, hint_policy=None):
    """Return image-column hints only. Even an empty result never implies No.

    Blue-tinted overview pixels, including ambiguous grid blends, can suggest
    where to look. They are not reclassified as trace support. A hint merely
    selects an ordinary zoom; a later native tooltip must prove a decline.
    The explicit first-three policy truncates only after grouping, never after
    filtering malformed early hints. Unexamined groups have no inferred outcome.
    """
    _require(
        hint_policy is None
        or (type(hint_policy) is str and hint_policy in OVERVIEW_HINT_POLICIES and overview is True),
        "invalid_overview_hint_policy",
    )
    baseline = baseline_position(flux_labels)
    _require(19 <= baseline <= 22 if overview else 30 <= baseline <= 120, "unsupported_baseline")
    mask = trace_mask(png).copy()
    unsupported = np.zeros(mask.shape, dtype=bool)
    if overview:
        # Deliberately permissive search hints, not the frozen detector mask.
        # A grid blend can hide a subpixel feature from the strict palette.
        with Image.open(io.BytesIO(png)) as image:
            colors = np.asarray(image.convert("RGB"), dtype=float)
        red, green, blue = (colors[:, :, i] for i in range(3))
        if hint_policy in {
            FIRST_THREE_LOW_INTENSITY_HINT_POLICY,
            FIRST_THREE_QUANTIZED_HINT_POLICY,
            FIRST_THREE_TWO_ROW_HINT_POLICY,
            EXACT_TWO_HINT_POLICY,
        }:
            candidates, context = _low_intensity_hints(colors, mask.copy(), baseline)
            if hint_policy == FIRST_THREE_QUANTIZED_HINT_POLICY:
                context = _quantized_hint_context(colors, mask, candidates)
            elif hint_policy in {FIRST_THREE_TWO_ROW_HINT_POLICY, EXACT_TWO_HINT_POLICY}:
                context = _two_row_hint_context(colors, mask, candidates)
            mask |= candidates
            unsupported = candidates & ~context
        mask |= (red <= 110) & (green >= red + 5) & (blue >= green + 4) & (blue >= red + 8)
    mask[:, :31] = False
    mask[:, 260:] = False
    centers = np.arange(mask.shape[0]) + 0.5
    # Hints only: a one-row feature can survive a 0..100 overview but be
    # shallower than 1.5px. It may also be raster noise. Only later native
    # tooltip readings can establish a decline; this never changes either
    # frozen positive/negative detector or authorizes an answer.
    mask[centers <= baseline + (0.75 if overview else 1.5)] = False
    mask[centers > baseline + (8 if overview else 40)] = False
    columns = np.flatnonzero(mask.any(axis=0))
    groups = []
    for column in map(int, columns):
        if groups and column == groups[-1][-1] + 1:
            groups[-1].append(column)
        else:
            groups.append([column])
    if hint_policy == EXACT_TWO_HINT_POLICY:
        _require(len(groups) == 2, "exact_two_overview_hints_required")
    elif hint_policy is not None:
        _require(len(groups) >= 3, "three_overview_hints_required")
        groups = groups[:3]
    else:
        _require(len(groups) <= 8, "too_many_candidate_groups")
    _require(
        all(len(group) <= 6 and 32 <= group[0] <= group[-1] <= 258 for group in groups),
        "broad_or_edge_feature",
    )
    _require(
        not any(unsupported[:, group].any() for group in groups),
        "unsupported_low_intensity_hint_context",
    )
    return groups


def summarize_samples(samples):
    """Only a bracketed, visibly sampled decline; never a period or exact depth."""
    _require(3 <= len(samples) <= 8, "invalid_sample_count")
    values = []
    days = []
    for sample in samples:
        _require(sample.source == "visible_hover_tooltip", "unsupported_sample_source")
        value = Decimal(sample.brightness_percent)
        _require(value.is_finite() and 0 <= value <= 100, "invalid_visible_brightness")
        _require(type(sample.day) is int and 0 <= sample.day <= 5000, "sample_outside_observed_window")
        values.append(value)
        days.append(sample.day)
    _require(all(a < b for a, b in pairwise(days)), "nonincreasing_sample_days")
    observed = values[0] == values[-1] == 100 and min(values[1:-1]) < 100
    return {
        "status": "visible_bracketed_dip" if observed else "unresolved",
        "sampled_day_interval": [days[0], days[-1]],
        "minimum_sampled_brightness_percent": str(min(values)),
        "sampled_decline_verified": observed,
        "daily_coverage_verified": all(b - a == 1 for a, b in pairwise(days)),
        **FLAGS,
    }


def probe_shallow_feature(
    page,
    config,
    output,
    *,
    run_history,
    source_dir,
    source_report_sha256,
    candidate_index,
    overview_hint_policy=None,
):
    """Synchronous diagnostic wrapper over the cooperative bounded recipe."""
    steps = iterate_shallow_feature(
        page,
        config,
        output,
        run_history=run_history,
        source_dir=source_dir,
        source_report_sha256=source_report_sha256,
        candidate_index=candidate_index,
        overview_hint_policy=overview_hint_policy,
    )
    try:
        while True:
            next(steps)
    except StopIteration as finished:
        return finished.value
    finally:
        steps.close()


def iterate_shallow_feature(
    page,
    config,
    output,
    *,
    run_history,
    source_dir,
    source_report_sha256,
    candidate_index,
    overview_hint_policy=None,
):
    """Explicit diagnostic using one current, completed 5,000-day overview.

    Fixed recipe: five -500 wheel steps and one -250 step, at most one vertical
    pan and pointer exit, then at most eight exact consecutive-day hovers. Each
    step keeps the original 180-second/16-action deadline. Failed owners cannot
    be resumed by this function, and this report is not a workflow handoff.
    """
    _require(
        overview_hint_policy is None
        or (type(overview_hint_policy) is str and overview_hint_policy in OVERVIEW_HINT_POLICIES),
        "invalid_overview_hint_policy",
    )
    _require(
        type(candidate_index) is int
        and 0
        <= candidate_index
        < (2 if overview_hint_policy == EXACT_TWO_HINT_POLICY else 3 if overview_hint_policy else 8),
        "invalid_candidate_index",
    )
    policy_metadata = overview_hint_metadata(overview_hint_policy)
    _require(
        isinstance(source_report_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", source_report_sha256),
        "explicit_report_hash_required",
    )
    book = _Owned(run_history)
    source = book.clean(source_dir)
    raw = book.read(source / "report.json", source_report_sha256)
    report = json.loads(raw)
    _require(
        isinstance(report.get("chart_sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", report["chart_sha256"]),
        "explicit_chart_hash_required",
    )
    png = book.read(source / "chart.png", report.get("chart_sha256"), limit=1024 * 1024)
    _require(
        report.get("requested_days") == 5000
        and report.get("browser_actions") == 0
        and report.get("answer_writes") == 0,
        "unsupported_source",
    )
    readiness = trace_progress(png, report["time_axis_labels"], requested_days=5000)
    _require(readiness["endpoint_visible"] and report.get("endpoint_visible") is True, "incomplete_overview")
    groups = candidate_columns(png, report["flux_axis_labels"], hint_policy=overview_hint_policy)
    _require(candidate_index < len(groups), "missing_candidate")
    group = groups[candidate_index]
    ticks = sorted((float(row["center_x"]), float(row["value"])) for row in report["time_axis_labels"])
    day_per_pixel = (ticks[-1][1] - ticks[0][1]) / (ticks[-1][0] - ticks[0][0])
    hint_interval = [
        max(0.0, ticks[0][1] + (group[0] - 0.5 - ticks[0][0]) * day_per_pixel),
        min(5000.0, ticks[0][1] + (group[-1] + 1.5 - ticks[0][0]) * day_per_pixel),
    ]
    x = (group[0] + group[-1] + 1) / 2 / 280
    y = baseline_position(report["flux_axis_labels"]) / 195
    directory = book.path(output)
    _require(
        not directory.is_relative_to(source) and not source.is_relative_to(directory), "overlapping_output"
    )
    directory = book.output(directory)
    source_hashes = {"report.json": _sha(raw), "chart.png": _sha(png)}
    sequence, session = 0, None

    def unchanged():
        for name, checksum in source_hashes.items():
            book.read(source / name, checksum)

    with (directory / "events.jsonl").open("x", encoding="utf-8") as stream:

        def emit(kind, payload):
            nonlocal sequence
            event = RuntimeEvent(event=kind, sequence=sequence, run_id=directory.name, payload=payload)
            stream.write(event.model_dump_json() + "\n")
            stream.flush()
            sequence += 1

        def capture(name):
            unchanged()
            session._guard()
            times, fluxes = session.time_axis_labels(), session.flux_axis_labels()
            _require(session.handle.evaluate(PLOT_EXPOSED), "plot_occluded")
            image = evidence_screenshot(session.chart)
            session._guard()
            _require(session.handle.evaluate(PLOT_EXPOSED), "plot_occluded")
            _require(
                times == session.time_axis_labels() and fluxes == session.flux_axis_labels(),
                "axes_changed_during_capture",
            )
            (directory / (name + ".png")).write_bytes(image)
            persist_json(
                directory / (name + ".json"),
                {
                    "time_axis_labels": times,
                    "flux_axis_labels": fluxes,
                    "chart_sha256": _sha(image),
                    **FLAGS,
                },
            )
            return image, times, fluxes

        try:
            persist_json(
                directory / "scope.json",
                {
                    "mode": MODE,
                    "sensor_recipe": RECIPE,
                    **policy_metadata,
                    "star": report.get("star"),
                    "max_actions": MAX_ACTIONS,
                    "max_seconds": MAX_SECONDS,
                    "requested_days": 5000,
                    "source_dir": book.relative(source),
                    "source_hashes": source_hashes,
                    "candidate_index": candidate_index,
                    "candidate_columns": group,
                    "source_hint_day_interval": hint_interval,
                    "source_owner_recovery": False,
                    **FLAGS,
                },
            )
            session = FluxChartSession(
                page, config, emit, max_actions=MAX_ACTIONS, max_seconds=MAX_SECONDS, verify_day_axis=True
            )
            _require(session.star == report.get("star"), "current_star_changed")
            image, times, fluxes = capture("before")
            _require(
                times == report["time_axis_labels"] and fluxes == report["flux_axis_labels"],
                "current_axes_changed",
            )
            _require(_sha(image) == source_hashes["chart.png"], "current_overview_changed")
            yield {"stage": "source_verified", "browser_actions": session.actions}
            for index, delta in enumerate((-500, -500, -500, -500, -500, -250), start=1):
                unchanged()
                session.zoom(x, y, delta)
                capture(f"zoom-{index}")
                yield {"stage": f"zoom_{index}", "browser_actions": session.actions}
            unchanged()
            center_baseline(session)
            unchanged()
            session.clear_pointer()
            image, times, fluxes = capture("readable")
            yield {"stage": "readable_capture", "browser_actions": session.actions}
            groups = candidate_columns(image, fluxes, overview=False)
            _require(len(groups) == 1, "focused_feature_not_unique")
            group = groups[0]
            # Prove readable affine time labels before dispatching a hover.
            points = sorted((float(row["center_x"]), float(row["value"])) for row in times)
            _require(len(points) >= 3, "insufficient_visible_time_axis")
            slope = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
            _require(math.isfinite(slope) and slope > 0, "invalid_time_axis")
            day_at = lambda pixel: points[0][1] + (pixel - points[0][0]) * slope
            first_day = math.floor(day_at(group[0] - 1))
            last_day = math.ceil(day_at(group[-1] + 2))
            _require(0 <= first_day < last_day <= 5000, "probe_outside_observed_window")
            _require(3 <= last_day - first_day + 1 <= 8, "daily_feature_exceeds_sample_budget")
            samples = []
            for day in range(first_day, last_day + 1):
                unchanged()
                pixel = points[0][0] + (day - points[0][1]) / slope
                verify_tooltip_day(times, pixel, day)
                sample = session.hover_day(pixel / 280, 80 / 195, day)
                verify_tooltip_day(times, pixel, sample.day)
                _require(sample.day == day, "requested_day_changed")
                _require(
                    session.time_axis_labels() == times and session.flux_axis_labels() == fluxes,
                    "axes_changed_during_probe",
                )
                samples.append(sample)
                yield {"stage": "sample", "sample": sample.model_dump(), "browser_actions": session.actions}
            sampled = summarize_samples(samples)
            linked = sampled["sampled_decline_verified"] and all(
                hint_interval[0] <= row.day <= hint_interval[1]
                for row in samples
                if Decimal(row.brightness_percent) < 100
            )
            # A zoom can expose a different feature, including when a course
            # chart carries stale transforms across stars. Preserve that new
            # reading, but never fabricate correspondence to the original hint.
            result = {
                "mode": MODE,
                "sensor_recipe": RECIPE,
                **policy_metadata,
                "star": session.star,
                "source_hashes": source_hashes,
                "samples": [sample.model_dump() for sample in samples],
                "browser_actions": session.actions,
                **sampled,
                "source_hint_day_interval": hint_interval,
                "source_hint_linked": linked,
            }
            if sampled["sampled_decline_verified"] and not linked:
                result["status"] = "visible_decline_not_source_linked"
            unchanged()
            session._guard()
            persist_json(directory / "report.json", result)
            emit("episode_summary", result)
            return result
        except BaseException as exc:
            reason = (
                "operator_aborted"
                if isinstance(exc, (GeneratorExit, KeyboardInterrupt, SystemExit))
                else str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "shallow_probe_operation_failed"
            )
            persist_json(directory / "stopped.json", {"reason": reason, **FLAGS})
            emit("error", {"type": "ShallowProbeStop", "message": reason, **FLAGS})
            raise
        finally:
            if session is not None:
                session.close()
