"""Independent physical candidate tests; no course answers or network access."""

import copy

import pytest
from pydantic import ValidationError

from habfly.planet_knowledge import PlanetCalculator, PlanetKnowledgePack, load_planet_pack


@pytest.fixture
def calculator():
    return PlanetCalculator()


def execute(calculator, operation, bindings):
    return calculator.execute(operation, bindings, assumptions=calculator.pack.assumptions)


def test_independent_goldens_are_not_course_promotion(calculator):
    report = calculator.verify()
    assert report["valid"] and report["golden_cases"] == 6
    assert not report["course_validation_passed"]
    assert not report["training_oracle_enabled"]
    with pytest.raises(ValueError, match="no training oracle"):
        calculator.reference_answers({})


@pytest.mark.parametrize("assumptions", [None, [], "circular_orbit", [{}], [False]])
def test_assumptions_must_be_explicit(calculator, assumptions):
    result = calculator.execute(
        "period_years", {"period_days": {"value": 365, "unit": "day"}}, assumptions=assumptions
    )
    assert result.error == "planet_assumptions_not_explicit"


def test_duplicate_or_extra_assumptions_rejected(calculator):
    for extra in [calculator.pack.assumptions[0], "invented"]:
        result = calculator.execute("period_years", {}, assumptions=calculator.pack.assumptions + [extra])
        assert result.error == "planet_assumptions_not_explicit"


@pytest.mark.parametrize("value", [None, "", "0", True, float("nan"), float("inf"), 10**400])
def test_numeric_only(calculator, value):
    result = execute(calculator, "period_years", {"period_days": {"value": value, "unit": "day"}})
    assert result.error == "input_not_finite_number"


@pytest.mark.parametrize(
    "bindings,error",
    [
        ({}, "missing_or_extra_inputs"),
        ({"period_days": None}, "invalid_binding"),
        ({"period_days": {"value": 0, "unit": "day"}}, "input_must_be_positive"),
        ({"period_days": {"value": -1, "unit": "day"}}, "input_must_be_positive"),
        ({"period_days": {"value": 365, "unit": "yr"}}, "incompatible_unit"),
    ],
)
def test_missing_zero_units_are_distinct(calculator, bindings, error):
    assert execute(calculator, "period_years", bindings).error == error


@pytest.mark.parametrize(
    "binding,error",
    [
        (None, "invalid_binding"),
        (5, "invalid_binding"),
        ({"value": 101, "unit": "%"}, "brightness_drop_exceeds_100_percent"),
        ({"value": 0, "unit": "%"}, "input_must_be_positive"),
        ({"value": 0.1, "unit": "fraction"}, "incompatible_unit"),
    ],
)
def test_depth_domains(calculator, binding, error):
    result = execute(
        calculator,
        "planet_radius",
        {"brightness_drop": binding, "stellar_radius": {"value": 1, "unit": "Rsun"}},
    )
    assert result.error == error


def test_wrong_same_unit_bindings_are_not_corrected(calculator):
    first = execute(
        calculator,
        "orbital_radius",
        {"period_years": {"value": 1, "unit": "yr"}, "stellar_mass": {"value": 1, "unit": "Msun"}},
    )
    second = execute(
        calculator,
        "orbital_radius",
        {"period_years": {"value": 8, "unit": "yr"}, "stellar_mass": {"value": 1, "unit": "Msun"}},
    )
    assert first.value == pytest.approx(1) and second.value == pytest.approx(4)
    assert execute(calculator, "unknown", {}).error == "unknown_operation"


def test_mass_requires_mass_not_spreadsheet_radius(calculator):
    result = execute(
        calculator,
        "planet_mass",
        {
            "radial_velocity": {"value": 1, "unit": "m/s"},
            "orbital_radius": {"value": 1, "unit": "au"},
            "stellar_mass": {"value": 1, "unit": "Rsun"},
        },
    )
    assert result.error == "incompatible_unit"
    assert "H2" in calculator.pack.source_conflicts[0]["original_formula"]


@pytest.mark.parametrize("field", ["output_unit", "quantity", "applicability"])
def test_candidate_contract_is_not_silently_reinterpreted(field):
    raw = load_planet_pack().model_dump()
    operation = raw["operations"][0]
    if field == "output_unit":
        operation[field] = "day"
    elif field == "quantity":
        operation["inputs"]["period_days"]["quantity"] = "stellar_mass"
    else:
        operation["applicable_classes"] = ["giant"]
    with pytest.raises(ValidationError):
        PlanetKnowledgePack.model_validate(raw)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "sqrt(4)",
        "1 if True else 2",
        "[x for x in range(3)]",
        "unknown + 1",
    ],
)
def test_no_executable_expressions(expression):
    raw = load_planet_pack().model_dump()
    raw["operations"][0]["expression"] = expression
    with pytest.raises(ValidationError):
        PlanetKnowledgePack.model_validate(raw)


@pytest.mark.parametrize(
    "field", ["provenance", "source_conflicts", "pending", "assumptions", "constant", "expression"]
)
def test_full_provenance_and_formula_hash(field):
    pack = load_planet_pack()
    raw = copy.deepcopy(pack.model_dump())
    if field == "provenance":
        raw[field]["date"] = "different"
    elif field == "source_conflicts":
        raw[field][0]["status"] += " annotation"
    elif field in {"pending", "assumptions"}:
        raw[field].append("additional item")
    elif field == "constant":
        raw["operations"][0]["constants"]["days_per_year"] = 365.25
    else:
        raw["operations"][0]["expression"] += " + 1"
    assert PlanetKnowledgePack.model_validate(raw).checksum != pack.checksum


def test_reset_and_offline_calculation(calculator, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    generation = calculator.generation
    calculator.reset()
    assert calculator.generation == generation + 1
    assert calculator.verify()["valid"]


def test_cli_planet_is_explicitly_candidate_and_stellar_default_unchanged():
    import json

    from typer.testing import CliRunner

    from habfly.cli import app

    runner = CliRunner()
    planet = runner.invoke(app, ["knowledge", "validate", "--task", "planet"])
    assert planet.exit_code == 0, planet.output
    assert json.loads(planet.output)["course_validation_passed"] is False
    stellar = runner.invoke(app, ["knowledge", "validate"])
    assert stellar.exit_code == 0, stellar.output
    assert json.loads(stellar.output)["calculation_mode"] == "local_tool_assisted"
    invalid = runner.invoke(app, ["knowledge", "inspect", "--task", "invented"])
    assert invalid.exit_code != 0
