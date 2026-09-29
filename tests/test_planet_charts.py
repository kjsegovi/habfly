import pytest
from pydantic import ValidationError

from habfly.planet_charts import FluxSample, analyze_flux_samples, parse_flux_tooltip, spectrum_excursion


def samples(days, dips=()):
    return [FluxSample(day=day, brightness_percent="99.992" if day in dips else "100") for day in days]


def test_visible_tooltip_decimal_copy():
    sample = parse_flux_tooltip("before Brightness: 99.992% , Day: 2902 after")
    assert sample.day == 2902 and sample.brightness_percent == "99.992"
    assert analyze_flux_samples([sample])["brightness_drop_percent"] == "0.008"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Brightness: nan% , Day: 1",
        "Brightness: 100% , Day: 1.5",
        "Brightness: 100% , Day: 1 Brightness: 99% , Day: 2",
    ],
)
def test_invalid_tooltips(text):
    with pytest.raises(ValueError):
        parse_flux_tooltip(text)


@pytest.mark.parametrize("value", ["nan", "Infinity", "-1", "101"])
def test_percent_domain(value):
    with pytest.raises(ValidationError):
        FluxSample(day=0, brightness_percent=value)


def test_three_regular_complete_transits():
    report = analyze_flux_samples(samples(range(330), (10, 118, 226)))
    assert report["status"] == "periodic_transits_observed"
    assert report["period_days"] == "108"
    assert len(report["complete_transits"]) == 3
    assert report["has_planet_answer"] is None


def test_sparse_aliasing_is_not_a_period():
    report = analyze_flux_samples(samples([9, 10, 11, 225, 226, 227, 441, 442, 443], (10, 226, 442)))
    assert report["transit_candidate"] and len(report["complete_transits"]) == 3
    assert not report["coverage_complete"] and report["period_days"] is None


def test_missing_baseline_does_not_complete_transit():
    report = analyze_flux_samples(samples(range(11), (0, 5, 10)))
    assert len(report["complete_transits"]) == 1 and report["period_days"] is None


def test_nonperiodic_transits_fail_period_gate():
    report = analyze_flux_samples(samples(range(400), (20, 128, 350)))
    assert report["status"] == "inconsistent_transit_intervals"


def test_empty_flat_incomplete_flat_and_complete_flat_are_not_no_planet():
    for seq in ([], samples([1, 10000]), samples(range(500))):
        report = analyze_flux_samples(seq)
        assert report["has_planet_answer"] is None and report["period_days"] is None


def test_conflicting_tooltip_values_rejected_but_duplicates_ok():
    seq = samples([1, 1, 2])
    assert analyze_flux_samples(seq)["unique_days"] == 2
    seq.append(FluxSample(day=1, brightness_percent="99"))
    with pytest.raises(ValueError, match="conflicting"):
        analyze_flux_samples(seq)


def test_multi_day_transit_centers():
    dips = [10, 11, 12, 30, 31, 32, 50, 51, 52]
    report = analyze_flux_samples(samples(range(60), dips))
    assert report["period_days"] == "20" and len(report["complete_transits"]) == 3


def test_spectrum_decimal_amplitude_not_full_excursion():
    result = spectrum_excursion("656.3nm", "656.299999371nm", "656.300000629nm")
    assert result["line_shift_nm"] == "0.000000629"
    assert result["peak_to_peak_nm"] == "0.000001258"
    assert not result["assumptions_verified"]


@pytest.mark.parametrize(
    "blue,red",
    [
        ("656.300000629nm", "656.299999371nm"),
        ("656.299999371nm", "656.30000063nm"),
        ("656.3nm", "656.3nm"),
        ("nan", "656.4nm"),
        ("650km", "660nm"),
    ],
)
def test_invalid_excursion_is_not_repaired(blue, red):
    with pytest.raises(ValueError):
        spectrum_excursion("656.3nm", blue, red)
