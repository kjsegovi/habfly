"""Bounded scripted visible-tooltip sampling, not learned chart perception.

The fixed 280x195 chart interaction region was verified separately. Calibration
uses visible day tooltips, never axes alone or SVG path/data properties. This
tool reports evidence; it does not choose Has Planet or write an answer.
"""

import math
import re
from itertools import pairwise

from .browser import BrowserSafetyStop
from .browser_flux_exposure import exposed_flux_sample
from .planet_charts import FluxSample, analyze_flux_samples
from .presentation_capture import evidence_screenshot

MAX_RECORDED_SAMPLES = 17600  # Ten explicit 32-window batches, not an unbounded retry.


def validated_scan_samples(report, *, star):
    """A complete recorded batch may be continued, never cross-star or repaired."""
    if (
        not isinstance(report, dict)
        or report.get("star") != star
        or report.get("sampling_method") != "scripted_bounded_visible_tooltips"
        or report.get("has_planet_answer") is not None
        or report.get("learned_chart_perception") is not False
        or report.get("task_completed") is not False
    ):
        raise BrowserSafetyStop("incompatible_transit_scan_evidence")
    rows = report.get("samples")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_RECORDED_SAMPLES:
        raise BrowserSafetyStop("unbounded_or_missing_transit_samples")
    samples = [FluxSample.model_validate(row) for row in rows]
    if any(s.source != "visible_hover_tooltip" for s in samples):
        raise BrowserSafetyStop("unverified_transit_source")
    try:
        measured = analyze_flux_samples(samples)
    except ValueError as exc:
        raise BrowserSafetyStop("conflicting_transit_scan_evidence") from exc
    if not measured["coverage_complete"] or any(report.get(k) != v for k, v in measured.items()):
        raise BrowserSafetyStop("transit_evidence_recalculation_mismatch")
    if report.get("period_evidence_verified") is not (measured["status"] == "periodic_transits_observed"):
        raise BrowserSafetyStop("conflicting_transit_scan_status")
    return samples


def recover_timed_out_scan(directory, *, prior_report=None):
    """Read-only evidence recovery for a pre-checkpoint budget stop.

    A deadline is not an uncertain answer write: the sensor never writes answers.
    Only samples matching already-confirmed hover results are retained. This
    function performs no browser action and never resumes a browser by itself.
    """
    import hashlib
    import json
    from pathlib import Path

    from .browser_assessment_actions import persist_json

    directory = Path(directory)
    if (directory / "invalidated.json").exists():
        raise BrowserSafetyStop("invalidated_transit_scan_evidence")
    names = ("scope.json", "stopped.json", "samples.jsonl", "events.jsonl")
    raw = {name: (directory / name).read_bytes() for name in names}
    scope, stopped = (json.loads(raw[name]) for name in names[:2])
    if (
        scope.get("sampling_method") != "scripted_bounded_visible_tooltips"
        or scope.get("answer_writes") is not False
        or scope.get("learned_chart_perception") is not False
        or not re.fullmatch(r"[A-Z][A-Z -]{1,39}", str(scope.get("star")))
        or stopped.get("reason") not in {"chart_time_limit", "chart_action_limit"}
        or stopped.get("exception_type") != "BrowserSafetyStop"
    ):
        raise BrowserSafetyStop("unsupported_transit_journal_recovery")
    rows = [json.loads(line) for line in raw["samples.jsonl"].splitlines()]
    if not 1 <= len(rows) <= 1760:
        raise BrowserSafetyStop("unbounded_or_missing_transit_samples")
    events = [json.loads(line) for line in raw["events.jsonl"].splitlines()]
    observations = [e["payload"] for e in events if e.get("event") == "observation"]
    if observations != [{"chart": {"source": "visible_tooltips", "star": scope["star"]}}]:
        raise BrowserSafetyStop("conflicting_transit_scan_identity")
    proposed, confirmed = {}, []
    for event in events:
        payload = event.get("payload", {})
        sequence = payload.get("sequence")
        if event.get("event") == "action_proposed":
            if sequence in proposed or payload.get("kind") not in {"HOVER", "DRAG", "SCROLL"}:
                raise BrowserSafetyStop("conflicting_transit_action_journal")
            proposed[sequence] = payload["kind"]
        if event.get("event") == "action_result" and "chart_sample" in payload:
            if proposed.get(sequence) != "HOVER" or payload.get("task_completed") is not False:
                raise BrowserSafetyStop("unconfirmed_transit_sample")
            confirmed.append(payload["chart_sample"])
            proposed[sequence] = "confirmed_hover"
    if len(confirmed) < len(rows) or confirmed[-len(rows) :] != rows:
        raise BrowserSafetyStop("transit_samples_do_not_match_confirmed_actions")
    inherited = []
    source_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    if scope.get("prior_samples", 0):
        # New batches carry an exact immutable snapshot. Older in-flight
        # journals may supply the original explicitly; its recorded hash and
        # sample count must agree before any evidence is joined.
        snapshot = directory / "prior-report.json"
        source = snapshot if snapshot.exists() else Path(prior_report) if prior_report else None
        if source is None:
            raise BrowserSafetyStop("missing_prior_transit_evidence")
        if (source.parent / "invalidated.json").exists():
            raise BrowserSafetyStop("invalidated_transit_scan_evidence")
        earlier_raw = source.read_bytes()
        prior_hash = hashlib.sha256(earlier_raw).hexdigest()
        if prior_hash != scope.get("prior_report_sha256"):
            raise BrowserSafetyStop("prior_transit_evidence_hash_mismatch")
        earlier = json.loads(earlier_raw)
        inherited = validated_scan_samples(earlier, star=scope["star"])
        if len(inherited) != scope["prior_samples"] or (
            scope.get("time_axis_validation") == "each_hover_against_visible_tick_glyphs"
            and earlier.get("time_axis_validation") != scope["time_axis_validation"]
        ):
            raise BrowserSafetyStop("incompatible_prior_transit_evidence")
        source_hashes["prior-report.json"] = prior_hash
    elif prior_report is not None or scope.get("prior_report_sha256"):
        raise BrowserSafetyStop("unexpected_prior_transit_evidence")
    samples = [*inherited, *(FluxSample.model_validate(row) for row in rows)]
    summary = analyze_flux_samples(samples)
    report = {
        **summary,
        "star": scope["star"],
        "sampling_method": "scripted_bounded_visible_tooltips",
        "learned_chart_perception": False,
        "task_completed": False,
        "period_evidence_verified": summary["status"] == "periodic_transits_observed",
        "samples": [sample.model_dump() for sample in samples],
        "recovery": "confirmed_read_only_samples_after_budget_stop",
        "source_sha256": source_hashes,
        "prior_report_sha256": scope.get("prior_report_sha256"),
        "inherited_samples": len(inherited),
        "new_samples": len(rows),
        "automatic_browser_actions": False,
        "time_axis_validation": scope.get("time_axis_validation", "legacy_not_recorded"),
    }
    validated_scan_samples(report, star=scope["star"])
    persist_json(directory / "recovered-report.json", report)
    return report


def baseline_position(labels):
    """Interpolate 100% using visible y-axis labels; bounded one-tick extrapolation."""
    points = sorted((float(r["value"]), float(r["center_y"])) for r in labels)
    if len(points) < 3 or not all(math.isfinite(v) and math.isfinite(y) for v, y in points):
        raise BrowserSafetyStop("insufficient_visible_flux_axis_labels")
    values = [v for v, _ in points]
    if len(set(values)) != len(values):
        raise BrowserSafetyStop("ambiguous_visible_flux_axis_labels")
    low, high = points[0], points[-1]
    slope = (high[1] - low[1]) / (high[0] - low[0])
    if slope >= 0 or any(abs(low[1] + (v - low[0]) * slope - y) > 1 for v, y in points):
        raise BrowserSafetyStop("nonlinear_visible_flux_axis")
    tick = max(b - a for a, b in pairwise(values))
    if not low[0] - tick <= 100 <= high[0] + tick:
        raise BrowserSafetyStop("flux_baseline_outside_supported_axis")
    return low[1] + (100 - low[0]) * slope


def center_baseline(session):
    """Keep the line in a readable band, not an unnecessarily exact pixel.

    The chart can limit a requested pan. Every eventual tooltip still requires
    full glyph exposure, regardless of the inferred axis position.
    """
    position = baseline_position(session.flux_axis_labels())
    if 30 <= position <= 120:
        return position
    shift = 80 - position
    if abs(shift) > 3:
        if not 5 <= 70 + shift <= 150:
            raise BrowserSafetyStop("flux_baseline_pan_out_of_bounds")
        session.pan((140 / 280, 70 / 195), (140 / 280, (70 + shift) / 195))
        position = baseline_position(session.flux_axis_labels())
        if not 30 <= position <= 120:
            raise BrowserSafetyStop("flux_baseline_readability_unverified")
    return position


def record_transit_sampling(page, config, output, *, prior_report=None, notify=lambda _: None):
    """One explicit bounded live diagnostic; retain partial evidence on failure."""
    import hashlib
    import json
    from pathlib import Path

    from .browser_assessment_actions import persist_json
    from .browser_planet_chart import FluxChartSession
    from .contracts import RuntimeEvent

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    session = None
    with (directory / "events.jsonl").open("x") as events, (directory / "samples.jsonl").open("x") as samples:
        current_samples = []
        sequence = 0

        def emit(event, payload):
            nonlocal sequence
            events.write(
                RuntimeEvent(
                    event=event, sequence=sequence, run_id=directory.name, payload=payload
                ).model_dump_json()
                + "\n"
            )
            events.flush()
            sequence += 1

        emit(
            "hello",
            {
                "protocol_version": 1,
                "task": "browser_transit_sampling",
                "policy": "scripted_reference_sensor",
                "learned_chart_perception": False,
                "answer_writes": 0,
            },
        )

        def sample(value):
            current_samples.append(value)
            samples.write(value.model_dump_json() + "\n")
            samples.flush()

        def completed_window(progress):
            collected = [*prior, *current_samples]
            summary = analyze_flux_samples(collected)
            persist_json(
                directory / f"window-{progress['window']:02d}.json",
                {
                    **summary,
                    "star": session.star,
                    "sampling_method": "scripted_bounded_visible_tooltips",
                    "time_axis_validation": "each_hover_against_visible_tick_glyphs",
                    "learned_chart_perception": False,
                    "task_completed": False,
                    "period_evidence_verified": summary["status"] == "periodic_transits_observed",
                    "samples": [s.model_dump() for s in collected],
                    "prior_report_sha256": source_hash,
                    "new_samples": len(current_samples),
                    "inherited_samples": len(prior),
                    "completed_window": progress["window"],
                },
            )
            notify(progress)
            emit("state", {"progress": progress, "task_completed": False})

        try:
            session = FluxChartSession(
                page, config, emit, max_actions=2048, max_seconds=900, verify_day_axis=True
            )
            prior, source_hash = [], None
            if prior_report is not None:
                if (Path(prior_report).parent / "invalidated.json").exists():
                    raise BrowserSafetyStop("invalidated_transit_scan_evidence")
                raw = Path(prior_report).read_bytes()
                earlier = json.loads(raw)
                if earlier.get("time_axis_validation") != "each_hover_against_visible_tick_glyphs":
                    raise BrowserSafetyStop("unverified_prior_time_axis")
                prior = validated_scan_samples(earlier, star=session.star)
                if len(prior) + 1760 > MAX_RECORDED_SAMPLES:
                    raise BrowserSafetyStop("cumulative_transit_sample_limit")
                source_hash = hashlib.sha256(raw).hexdigest()
                with (directory / "prior-report.json").open("xb") as snapshot:
                    snapshot.write(raw)
            persist_json(
                directory / "scope.json",
                {
                    "star": session.star,
                    "sampling_method": "scripted_bounded_visible_tooltips",
                    "max_actions": 2048,
                    "max_seconds": 900,
                    "max_windows": 32,
                    "max_zooms": 12,
                    "max_vertical_search_pans_per_read": 64,
                    "visibility_recovery": "ordinary_vertical_pan_and_verified_restore",
                    "answer_writes": False,
                    "learned_chart_perception": False,
                    "time_axis_validation": "each_hover_against_visible_tick_glyphs",
                    "prior_report_sha256": source_hash,
                    "prior_samples": len(prior),
                    "max_cumulative_samples": MAX_RECORDED_SAMPLES,
                },
            )
            report = sample_transits(session, prior_samples=prior, on_sample=sample, notify=completed_window)
            report["star"] = session.star
            report["time_axis_validation"] = "each_hover_against_visible_tick_glyphs"
            report["period_evidence_verified"] = report["status"] == "periodic_transits_observed"
            report["prior_report_sha256"] = source_hash
            evidence_screenshot(session.chart, path=str(directory / "final-chart.png"))
            persist_json(directory / "report.json", report)
            emit(
                "episode_summary",
                {k: v for k, v in report.items() if k not in {"samples", "calibration_samples"}},
            )
            return report
        except Exception as exc:
            emit(
                "error",
                {
                    "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "transit_sampling_failed",
                    "automatic_retry": False,
                    "task_completed": False,
                },
            )
            persist_json(
                directory / "stopped.json",
                {
                    "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "transit_sampling_failed",
                    "exception_type": type(exc).__name__,
                    "automatic_retry": False,
                    "partial_samples_are_not_period_evidence": True,
                    "task_completed": False,
                },
            )
            raise
        finally:
            if session is not None:
                session.close()


def sample_transits(
    session,
    *,
    max_windows=32,
    max_zooms=12,
    prior_samples=(),
    on_sample=lambda _: None,
    notify=lambda _: None,
):
    if type(max_windows) is not int or not 1 <= max_windows <= 32:
        raise ValueError("Transit window budget must be 1 through 32")
    if type(max_zooms) is not int or not 0 <= max_zooms <= 12:
        raise ValueError("Transit zoom budget must be 0 through 12")
    box = session.chart.bounding_box()
    if not box or abs(box["width"] - 280) > 1 or abs(box["height"] - 195) > 1:
        raise BrowserSafetyStop("unsupported_transit_sampling_geometry")
    # Keep the complete tooltip in the left-side region. Each operation still
    # goes through the session's exposure/context/action/deadline guards.
    if len(prior_samples) + max_windows * 55 > MAX_RECORDED_SAMPLES:
        raise BrowserSafetyStop("cumulative_transit_sample_limit")
    calibration, samples, zooms = [], list(prior_samples), 0
    inherited = len(samples)
    # The unzoomed chart clamps vertical panning. If the exposed baseline
    # cannot be moved into the readable band, spend an existing zoom-budget
    # unit before trying to position it again; never invent a tooltip reading.
    while True:
        try:
            baseline = center_baseline(session)
            break
        except BrowserSafetyStop as exc:
            if str(exc) != "flux_baseline_readability_unverified" or zooms >= max_zooms:
                raise
            baseline = baseline_position(session.flux_axis_labels())
            session.zoom(40 / 280, baseline / 195, -500)
            zooms += 1
    remaining_zooms = max_zooms - zooms
    for attempt in range(remaining_zooms + 1):
        left, right = exposed_flux_sample(session, 31 / 280), exposed_flux_sample(session, 85 / 280)
        calibration.extend((left.model_dump(), right.model_dump()))
        day_span = right.day - left.day
        if 42 <= day_span <= 54:
            break
        if day_span <= 0:
            raise BrowserSafetyStop("unresolved_visible_transit_time_window")
        if attempt == remaining_zooms:
            raise BrowserSafetyStop("daily_transit_resolution_not_reached")
        # Keep the visible baseline central and anchor near the left edge;
        # centering the 10,000-day axis can select not-yet-observed data.
        # Keep approximately one observed day per native pixel. Overshooting
        # to a much narrower scale creates duplicate-day work; smaller bounded
        # wheel steps approach the range without increasing the zoom budget.
        if day_span > 54:
            delta = -125 if day_span <= 64 else -250 if day_span < 108 else -500
        else:
            delta = 500 if day_span < 21 else 250 if day_span < 30 else 125
        session.zoom(40 / 280, baseline / 195, delta)
        baseline = center_baseline(session)
        zooms += 1
    for window in range(max_windows):
        for x in range(31, 86):
            sample = exposed_flux_sample(session, x / 280)
            samples.append(sample)
            on_sample(sample)
        report = analyze_flux_samples(samples)
        notify(
            {
                "window": window + 1,
                "unique_days": report["unique_days"],
                "complete_transits": len(report["complete_transits"]),
                "status": report["status"],
            }
        )
        if not report["coverage_complete"]:
            raise BrowserSafetyStop("daily_transit_coverage_gap")
        if report["status"] == "periodic_transits_observed":
            break
        if window + 1 < max_windows:
            session.pan((150 / 280, 80 / 195), (100 / 280, 80 / 195))
    return {
        **report,
        "sampling_method": "scripted_bounded_visible_tooltips",
        "learned_chart_perception": False,
        "task_completed": False,
        "zooms": zooms,
        "windows": window + 1,
        "inherited_samples": inherited,
        "new_samples": len(samples) - inherited,
        "calibration_samples": calibration,
        "samples": [s.model_dump() for s in samples],
        "has_planet_answer": None,
    }
