"""Incomplete, inconsistent and cross-star readings cannot feed native answers."""

from copy import deepcopy

import pytest

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_evidence import measurement_evidence
from habfly.planet_charts import FluxSample, analyze_flux_samples, spectrum_excursion


def evidence():
    readings = [
        {"marker": side, "wavelength_text": value, "source": "visible_spectrum_tooltip"}
        for side, value in (("blue", "656.299nm"), ("red", "656.301nm"))
    ]
    spectrum = {
        "visibility": "full_glyph_and_occlusion_checked",
        "events": [
            {"kind": "observation", "payload": {"chart": {"star": "FIXTURE"}}},
            *[{"kind": "action_result", "payload": {"spectrum_sample": row}} for row in readings],
        ],
        "result": spectrum_excursion("656.3nm", "656.299nm", "656.301nm"),
    }
    samples = [
        FluxSample(day=d, brightness_percent="99.99" if d in (10, 30, 50) else "100") for d in range(60)
    ]
    transit = {
        **analyze_flux_samples(samples),
        "star": "FIXTURE",
        "samples": [s.model_dump() for s in samples],
        "sampling_method": "scripted_bounded_visible_tooltips",
        "period_evidence_verified": True,
        "learned_chart_perception": False,
        "task_completed": False,
    }
    return spectrum, transit


def test_measurement_evidence_recomputes_from_visible_same_star_samples():
    spectrum, transit = evidence()
    before = deepcopy((spectrum, transit))
    result = measurement_evidence(spectrum, transit)
    assert result["measurements"] == {
        "line_shift": {"value": "0.001", "unit": "nm"},
        "brightness_drop": {"value": "0.01", "unit": "%"},
        "period_days": {"value": "20", "unit": "day"},
    }
    assert not result["task_completed"] and result["selection_source"].endswith("not_learned")
    assert before == (spectrum, transit)


@pytest.mark.parametrize("change", ["star", "sparse", "depth", "period", "amplitude", "visibility", "source"])
def test_invalid_or_tampered_evidence_is_not_repaired(change):
    spectrum, transit = evidence()
    if change == "star":
        transit["star"] = "OTHER"
    elif change == "sparse":
        transit["samples"] = transit["samples"][::2]
    elif change == "depth":
        transit["brightness_drop_percent"] = "99"
    elif change == "period":
        transit["period_days"] = "999"
    elif change == "amplitude":
        spectrum["result"]["line_shift_nm"] = "1"
    elif change == "visibility":
        spectrum["visibility"] = "partial"
    elif change == "source":
        transit["samples"][0]["source"] = "private_array"
    with pytest.raises(BrowserSafetyStop):
        measurement_evidence(spectrum, transit)
