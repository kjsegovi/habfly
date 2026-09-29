import copy
import json
import math

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from habfly.cli import app
from habfly.habitability_knowledge import (
    HabitabilityCalculator,
    HabitabilityKnowledgePack,
    load_habitability_pack,
)


def inputs():
    return {
        "stellar_luminosity": {"value": 1, "unit": "Lsun"},
        "orbital_radius": {"value": 1, "unit": "au"},
        "albedo": {"value": 0, "unit": "fraction"},
    }


def execute(bindings, op="equilibrium_temp"):
    calc = HabitabilityCalculator()
    return calc.execute(op, bindings, assumptions=calc.pack.assumptions)


def test_independent_goldens_and_cli_remain_course_pending():
    report = HabitabilityCalculator().verify()
    assert report["golden_cases"] == 6 and report["valid"]
    assert not report["course_validation_passed"] and not report["training_oracle_enabled"]
    for command in ("inspect", "validate"):
        result = CliRunner().invoke(app, ["knowledge", command, "--task", "habitability"])
        assert result.exit_code == 0
        assert isinstance(json.loads(result.output), dict)
    with pytest.raises(ValueError, match="pending"):
        HabitabilityCalculator().reference_answers()


def test_explicit_inputs_scaling_and_wrong_bindings_are_not_repaired():
    base = execute(inputs()).value
    values = inputs()
    values["stellar_luminosity"]["value"] = 16
    assert math.isclose(execute(values).value, base * 2)
    values = inputs()
    values["orbital_radius"]["value"] = 4
    assert math.isclose(execute(values).value, base / 2)
    values = inputs()
    values["albedo"]["value"] = 0.9375
    assert math.isclose(execute(values).value, base / 2)


@pytest.mark.parametrize("value", [-1, 1, 2, True, None, "0.3", float("inf"), float("nan"), 10**400])
def test_invalid_albedo(value):
    values = inputs()
    values["albedo"]["value"] = value
    assert not execute(values).ok


@pytest.mark.parametrize("change", ["missing", "zero_radius", "wrong_unit", "extra", "negative_luminosity"])
def test_invalid_bindings(change):
    values = inputs()
    if change == "missing":
        del values["albedo"]
    elif change == "zero_radius":
        values["orbital_radius"]["value"] = 0
    elif change == "wrong_unit":
        values["albedo"]["unit"] = "%"
    elif change == "extra":
        values["unrequested"] = {"value": 1, "unit": "K"}
    else:
        values["stellar_luminosity"]["value"] = -1
    assert not execute(values).ok


@pytest.mark.parametrize("assumptions", [None, [], [None], ["terrestrial_planet"]])
def test_assumptions_not_silently_inferred(assumptions):
    assert not HabitabilityCalculator().execute("equilibrium_temp", inputs(), assumptions=assumptions).ok


@pytest.mark.parametrize(
    "value,name",
    [
        ("0", "none"),
        ("0.49", "none"),
        ("0.5", "weak"),
        ("39.99", "weak"),
        ("40", "moderate"),
        ("59.99", "moderate"),
        ("60", "strong"),
        ("100", "strong"),
    ],
)
def test_published_greenhouse_bands(value, name):
    result = HabitabilityCalculator().greenhouse_reference(value)
    assert result["ok"] and result["name"] == name


@pytest.mark.parametrize("value", ["0.495", "39.995", "59.995"])
def test_published_gaps_do_not_get_rounded(value):
    assert HabitabilityCalculator().greenhouse_reference(value) == {
        "ok": False,
        "error": "unresolved_published_band_gap",
    }


@pytest.mark.parametrize("value", [None, True, 50, "nan", "inf", "101", "-1", "50%"])
def test_invalid_absorption(value):
    assert not HabitabilityCalculator().greenhouse_reference(value)["ok"]


@pytest.mark.parametrize("value", [-10, 9, 50, 101, True])
def test_increment_must_be_a_published_course_value(value):
    assert not execute(
        {
            "equilibrium_temp": {"value": 250, "unit": "K"},
            "greenhouse_increment": {"value": value, "unit": "K"},
        },
        "surface_temp",
    ).ok


def test_pack_hash_covers_provenance_and_contract_forbids_band_repairs():
    pack = load_habitability_pack()
    data = pack.model_dump(mode="json")
    other = copy.deepcopy(data)
    other["provenance"]["test"] = "changed"
    assert HabitabilityKnowledgePack.model_validate(other).checksum != pack.checksum
    other = copy.deepcopy(data)
    other["greenhouse_bands"][0]["maximum"] = "0.5"
    with pytest.raises(ValidationError):
        HabitabilityKnowledgePack.model_validate(other)


def test_arbitrary_expressions_are_rejected():
    data = load_habitability_pack().model_dump(mode="json")
    data["operations"][0]["expression"] = "__import__('os').system('echo unsafe')"
    with pytest.raises(ValidationError):
        HabitabilityKnowledgePack.model_validate(data)
