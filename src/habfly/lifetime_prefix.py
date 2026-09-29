"""Deterministic unit transport for the lesson's visible lifetime dropdown.

The caller/policy selects the prefix; this module does not choose or round it.
The learned local tool still returns years. This separate transport conversion
leaves all existing knowledge-pack and checkpoint hashes unchanged.
"""

from decimal import Decimal, InvalidOperation, localcontext

PREFIX_YEARS = {"ka": 1000, "Ma": 1000000, "Ga": 1000000000, "Ta": 1000000000000}


def _number(value):
    if type(value) not in {int, float, str, Decimal} or len(str(value)) > 128:
        raise ValueError("invalid_lifetime_number")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("invalid_lifetime_number") from None
    if not number.is_finite() or number <= 0 or abs(number.adjusted()) > 100:
        raise ValueError("invalid_lifetime_domain")
    return number


def _convert(value, prefix, *, to_years):
    if not isinstance(prefix, str) or prefix not in PREFIX_YEARS:
        raise ValueError("unsupported_lifetime_prefix")
    number = _number(value)
    with localcontext() as context:
        context.prec = 256
        factor = Decimal(PREFIX_YEARS[prefix])
        result = number * factor if to_years else number / factor
    return format(result, "f")


def lifetime_to_prefix(years, prefix):
    return _convert(years, prefix, to_years=False)


def lifetime_to_years(quantity, prefix):
    return _convert(quantity, prefix, to_years=True)
