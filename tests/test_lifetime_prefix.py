from decimal import Decimal

import pytest

from habfly.lifetime_prefix import lifetime_to_prefix, lifetime_to_years


@pytest.mark.parametrize(
    "years,prefix,quantity",
    [
        (1000, "ka", "1"),
        (1000000, "Ma", "1"),
        (10000000000, "Ga", "10"),
        (2006995712068.1611, "Ta", "2.0069957120681611"),
        (312500000, "Ma", "312.5"),
        (320000000000, "Ga", "320"),
        (1, "Ta", "0.000000000001"),
    ],
)
def test_independent_prefix_cases_and_exact_roundtrip(years, prefix, quantity):
    assert lifetime_to_prefix(years, prefix) == quantity
    assert Decimal(lifetime_to_years(quantity, prefix)) == Decimal(str(years))


@pytest.mark.parametrize(
    "value", [None, True, False, "", "NaN", "Infinity", "-Infinity", 0, -1, [], "1e10000", "0.1" * 100]
)
def test_missing_zero_and_nonfinite_inputs_rejected(value):
    with pytest.raises(ValueError):
        lifetime_to_prefix(value, "Ga")
    with pytest.raises(ValueError):
        lifetime_to_years(value, "Ga")


@pytest.mark.parametrize("prefix", ["", "yr", "Gyr", "ga", "billion", None])
def test_prefix_is_explicit_never_guessed(prefix):
    with pytest.raises(ValueError, match="unsupported_lifetime_prefix"):
        lifetime_to_prefix(10000000000, prefix)
