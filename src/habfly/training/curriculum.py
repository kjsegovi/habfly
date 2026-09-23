"""Reproducible recognition, composition, and arithmetic curricula."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

from habfly.contracts import Action, ActionKind, Control, Observation


@dataclass
class Example:
    observation: Observation
    action: Action
    answer: str = ""
    group: str = ""
    value_target: float | None = None


def _answer_example(instruction: str, answer: str, group: str) -> Example:
    control = Control(id="answer", label="Answer", role="textbox", actions=[ActionKind.TYPE])
    return Example(Observation(instruction=instruction, controls=[control]),
                   Action(kind=ActionKind.TYPE, target="answer", value=answer), answer, group)


def curriculum_examples(stage: str, *, seed: int = 0, count: int = 96) -> tuple[list[Example], list[Example]]:
    """Split by templates, operand ranges, or composed phrases before sampling."""
    if count < 4:
        raise ValueError("A curriculum requires at least four examples")
    rng = random.Random(seed)
    train, test = [], []
    if stage == "recognize":
        symbols = list(string.ascii_letters + string.digits + "+-*/=.?!") + [
            "click", "type", "select", "star", "planet", "transit", "period", "radius", "yes", "no", "days", "units",
            "CLICK", "Type", "Select", "Star", "Planet", "WAIT", "stop", "hover", "drag", "scroll",
        ]
        for index in range(count):
            symbol = symbols[index % len(symbols)]
            train.append(_answer_example(f"Copy exactly [{symbol}].", symbol, f"copy:{symbol}"))
            test.append(_answer_example(f"  Repeat exactly: {symbol}  !", symbol, f"noise:{symbol}"))
    elif stage == "arithmetic":
        # Split unordered operand pairs first, so a reversed pair cannot leak.
        pairs = [(a, b) for a in range(10) for b in range(a, 10)]
        rng.shuffle(pairs)
        for split, selected_pairs, template in ((train, pairs[:44], "Calculate {a} {op} {b}."),
                                                (test, pairs[44:], "What is ({a}) {op} ({b})?")):
            candidates = [(a, b, op) for lo, hi in selected_pairs for a, b in {(lo, hi), (hi, lo)}
                          for op in ("+", "-", "*")]
            rng.shuffle(candidates)
            for a, b, op in candidates[:count]:
                result = {"+": a + b, "-": a - b, "*": a * b}[op]
                split.append(_answer_example(template.format(a=a, b=b, op=op), str(result), f"pair:{min(a,b)}:{max(a,b)}"))
    elif stage == "ground":
        colors = ["red", "blue", "green", "yellow", "white", "black"]
        objects = ["star", "planet", "moon", "comet", "button", "chart"]
        phrases = [(f"{color} {noun}", ci, ni) for ci, color in enumerate(colors) for ni, noun in enumerate(objects)]
        for split, parity, prefix in ((train, 0, "Click the"), (test, 1, "Please select the")):
            goals = [phrase for phrase, ci, ni in phrases if (ci + ni) % 2 == parity]
            for index in range(count):
                goal = goals[index % len(goals)]
                alternatives = [p[0] for p in phrases if p[0] != goal]
                options = rng.sample(alternatives, 3) + [goal]
                rng.shuffle(options)
                kind = (ActionKind.CLICK, ActionKind.TYPE, ActionKind.SELECT)[index % 3]
                role = {ActionKind.CLICK: "button", ActionKind.TYPE: "textbox", ActionKind.SELECT: "combobox"}[kind]
                choices = ["yes", "no", "unknown"] if kind == ActionKind.SELECT else []
                controls = [Control(id=f"option-{j}", label=label, role=role, actions=[kind], options=choices) for j, label in enumerate(options)]
                target = controls[options.index(goal)].id
                value = None
                instruction = f"{prefix} {goal}."
                if kind == ActionKind.TYPE:
                    value = ("star", "planet", "42", "yes")[index % 4]
                    instruction = f"Type '{value}' into the {goal}." if parity == 0 else f"In {goal}, enter '{value}'."
                elif kind == ActionKind.SELECT:
                    value = choices[(index // 3) % len(choices)]
                    instruction = f"Set the {goal} to '{value}'." if parity == 0 else f"Choose '{value}' for {goal}."
                split.append(Example(Observation(instruction=instruction, controls=controls),
                                     Action(kind=kind, target=target, value=value), value or "", goal))
    else:
        raise ValueError(f"Unknown curriculum: {stage}")
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def arithmetic_ood_examples(seed=0, count=32) -> list[Example]:
    """Two-digit evaluation only. Never included in the first training curriculum."""
    rng = random.Random(seed)
    candidates = [(a, b, op) for a in range(10, 30) for b in range(10, 30) for op in ("+", "-", "*")]
    rng.shuffle(candidates)
    result = []
    for a, b, op in candidates[:count]:
        answer = {"+": a + b, "-": a - b, "*": a * b}[op]
        result.append(_answer_example(f"Calculate {a} {op} {b}.", str(answer), f"ood:{a}:{op}:{b}"))
    return result
