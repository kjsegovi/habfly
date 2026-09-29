"""Interpret recorded, student-visible chart tooltips, never simulation arrays.

Sparse hover samples can establish a transit candidate but cannot establish its
period or prove absence. Period estimation requires continuous daily coverage
between at least three complete transits, avoiding a silently aliased period.
"""

import re
from decimal import Decimal, InvalidOperation
from itertools import pairwise

from pydantic import Field, field_validator

from .contracts import Contract

TOOLTIP = re.compile(r"Brightness:\s*(\d+(?:\.\d+)?)%\s*,\s*Day:\s*(\d+)(?![\d.])")


def spectrum_excursion(rest, blue, red):
    """Explicitly selected visible endpoints -> semi-amplitude, not peak-to-peak."""
    values = []
    for text in (rest, blue, red):
        if not isinstance(text, str) or not re.fullmatch(r"\d+(?:\.\d+)?nm", text):
            raise ValueError("invalid_visible_wavelength")
        values.append(Decimal(text.removesuffix("nm")))
    center, low, high = values
    if not 0 < low < center < high:
        raise ValueError("invalid_spectrum_excursion_order")
    a, b = center - low, high - center
    # The marker tooltips share displayed precision. Do not silently average
    # asymmetric endpoints or substitute the known central wavelength.
    if a != b:
        raise ValueError("asymmetric_visible_spectrum_excursion")
    return {
        "rest_nm": str(center),
        "blue_nm": str(low),
        "red_nm": str(high),
        "line_shift_nm": format(a, "f"),
        "peak_to_peak_nm": format(high - low, "f"),
        "source": "explicit_visible_excursion_markers",
        "assumptions_verified": False,
    }


class FluxSample(Contract):
    day: int = Field(ge=0, le=100000)
    brightness_percent: str
    source: str = "visible_hover_tooltip"

    @field_validator("brightness_percent")
    @classmethod
    def finite_percent(cls, value):
        try:
            number = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("Invalid brightness") from exc
        if not number.is_finite() or not 0 <= number <= 100:
            raise ValueError("Brightness must be a finite percentage")
        return str(number)


def parse_flux_tooltip(text):
    matches = TOOLTIP.findall(text)
    if len(matches) != 1:
        raise ValueError("missing_or_ambiguous_flux_tooltip")
    brightness, day = matches[0]
    return FluxSample(day=int(day), brightness_percent=brightness)


def analyze_flux_samples(samples):
    """Return evidence, not an automatically selected browser answer."""
    by_day = {}
    for sample in samples:
        sample = sample if isinstance(sample, FluxSample) else FluxSample.model_validate(sample)
        value = Decimal(sample.brightness_percent)
        if sample.day in by_day and by_day[sample.day] != value:
            raise ValueError("conflicting_visible_flux_readings")
        by_day[sample.day] = value
    days = sorted(by_day)
    dips = [day for day in days if by_day[day] < 100]
    report = {
        "source": "visible_hover_tooltips",
        "unique_days": len(days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "coverage_complete": bool(days) and len(days) == days[-1] - days[0] + 1,
        "transit_candidate": bool(dips),
        "brightness_drop_percent": str(100 - min(by_day.values())) if dips else None,
        "period_days": None,
        "complete_transits": [],
        "status": "insufficient_coverage" if days else "no_samples",
        "has_planet_answer": None,
    }
    groups = []
    for day in dips:
        if groups and day == groups[-1][-1] + 1:
            groups[-1].append(day)
        else:
            groups.append([day])
    complete = [g for g in groups if by_day.get(g[0] - 1) == 100 and by_day.get(g[-1] + 1) == 100]
    report["complete_transits"] = [
        {"start": g[0], "end": g[-1], "center": (g[0] + g[-1]) / 2} for g in complete
    ]
    if not dips and report["coverage_complete"]:
        report["status"] = "no_transit_in_observed_interval"
    elif dips and report["coverage_complete"]:
        report["status"] = "insufficient_complete_transits"
        if len(complete) >= 3:
            centers = [Decimal(g[0] + g[-1]) / 2 for g in complete]
            intervals = [b - a for a, b in pairwise(centers)]
            if max(intervals) - min(intervals) <= 1:
                period = sum(intervals) / len(intervals)
                report["period_days"] = str(period)
                report["status"] = "periodic_transits_observed"
            else:
                report["status"] = "inconsistent_transit_intervals"
    # The agent still chooses whether and where to use this evidence. Neither
    # missing samples nor a clean finite interval is converted to "No planet".
    return report
