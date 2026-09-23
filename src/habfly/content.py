"""Declarative lesson packs. The bundled pack is explicitly synthetic."""

import ast
import hashlib
import math
import operator
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .contracts import Contract


class CalculationField(Contract):
    name: str
    label: str
    formula: str
    tolerance: float = Field(default=0.01, gt=0)
    units: list[str] = Field(default_factory=list)
    expected_unit: str = ""


class ContentPack(Contract):
    version: Literal[1] = 1
    id: str
    title: str
    synthetic: bool = True
    description: str
    fields: list[CalculationField]
    terrestrial_required: bool = True
    liquid_temperature: tuple[float, float] = (273, 373)
    minimum_pressure: float = 0.01

    @model_validator(mode="after")
    def check_fields(self):
        names = [f.name for f in self.fields]
        if not names or len(names) != len(set(names)):
            raise ValueError("Content fields must have unique names")
        for field in self.fields:
            if field.units and field.expected_unit not in field.units:
                raise ValueError("expected_unit must be a declared unit")
        return self

    @property
    def checksum(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def load_content_pack(path: Path | str | None = None) -> ContentPack:
    if path is None:
        path = Path(__file__).parent / "packs" / "synthetic.json"
    return ContentPack.model_validate_json(Path(path).read_text())


def calculate(formula: str, values: dict[str, float]) -> float:
    """A small arithmetic grammar, without Python eval or arbitrary calls."""
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    try:
        tree = ast.parse(formula, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid formula syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Formula is too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = float(node.value)
            if not math.isfinite(value):
                raise ValueError("Non-finite formula constant")
            return value
        if isinstance(node, ast.Name) and node.id in values:
            value = values[node.id]
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError("Formula inputs must be finite numbers, not booleans or strings")
            return float(value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent exceeds bound")
            result = operations[type(node.op)](left, right)
            if not isinstance(result, (float, int)) or not math.isfinite(result):
                raise ValueError("Non-finite formula result")
            return result
        raise ValueError("Formula contains an unsupported operation or missing visible value")

    try:
        result = float(visit(tree.body))
    except (ArithmeticError, TypeError, RecursionError) as exc:
        raise ValueError("Formula domain or arithmetic error") from exc
    if not math.isfinite(result):
        raise ValueError("Non-finite formula result")
    return result
