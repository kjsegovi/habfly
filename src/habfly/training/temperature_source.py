"""Balanced source-language repair; this generator is never used at inference.

The first temperature test's wording is explicitly consumed rehearsal. New
development, test and manual templates, candidate payloads and numeric cases are
disjoint. This is known-vocabulary composition, not arbitrary-language coverage.
"""

import itertools
import random
from copy import deepcopy

from habfly.environments.temperature import temperature_cases

from .source_request import validate_pairs

VERSION = "temperature-source-v2"
TEMPLATES = {
    "train": (
        "Find the {source}'s distance, luminosity and temperature. Use its parallax, flux and peak wavelength. Copy each result and select its unit.",
        "Use the {source}'s measurements. Calculate distance, luminosity using that distance, and temperature using peak wavelength. Fill all three answers and units.",
        # Consumed failed-test wording, now explicitly training/rehearsal.
        "Calculate the {source}'s distance, luminosity and temperature, not those of the {other}. Enter the tool results and select their units.",
        "Analyze the {source}: determine distance, luminosity and temperature, then copy each result with its unit. Ignore the {other}'s measurements.",
        "For the {source}, obtain distance, luminosity and temperature with the local tool. Transfer the three values and their units.",
        "Calculate distance, luminosity and temperature for the {source}, not the {other}. Copy the values and units.",
        "Ignore the {other}. Use measurements of the {source} to calculate distance, luminosity and temperature.",
        "Use the {source}'s readings, not those of the {other}. Calculate and copy distance, luminosity and temperature, with units.",
        "Determine the {source}'s distance, luminosity and temperature instead of the {other}'s values. Copy each result and select its unit.",
        "Instead of using the {other}'s data, use the {source}'s parallax, flux and wavelength. Fill distance, luminosity and temperature with units.",
        "The {source} is requested, not the {other}. Calculate distance, luminosity and temperature using the requested star's data.",
        "Use the {source}'s parallax, flux and peak wavelength. The {other}'s measurements are distractors. Fill the three calculated answers and units.",
        "Not the {other}: analyze the {source}. Calculate distance, luminosity and temperature, then copy each value and unit.",
        "Disregard the {other}'s readings. For the {source}, calculate distance, luminosity and temperature using only its measurements.",
    ),
    "development": (
        "Use readings from the {source}, not those belonging to the {other}, to calculate distance, luminosity and temperature. Copy the results and select their units.",
        "Ignore measurements for the {other}; determine all three answers for the {source}: distance, luminosity and temperature, with units.",
        "The {source} is the requested star. Calculate its distance, luminosity and temperature, not the values for the {other}.",
        "Determine the {source}'s distance, luminosity and temperature. Use that star's parallax, flux and wavelength; the {other} is a distractor.",
    ),
    "calibration": (
        "Complete distance, luminosity and temperature for the {source}. Use its measurements; ignore data from the {other}. Select each answer's unit.",
        "For the {source}, calculate and copy the three answers: distance, luminosity and temperature. Do not use the {other}'s readings.",
    ),
    "test": (
        "For the {source}, calculate distance, luminosity and temperature using only its readings. Ignore the {other}; copy each result and select each unit.",
        "Use the {source}'s parallax, flux and peak wavelength to fill distance, luminosity and temperature. Disregard readings from the {other}.",
        "Analyze the {source}, not the {other}: determine distance, luminosity and temperature. Copy each calculated value and its unit.",
        "Find distance and luminosity, then temperature, for the {source}. Use its measurements instead of the {other}'s data.",
    ),
    "manual": (
        "For the {source}, calculate distance, luminosity and temperature. Use its measurements, not the {other}'s; copy each value and unit.",
        "The task concerns the {source}. Ignore the {other}, calculate distance, luminosity and temperature, and fill the three answers and units.",
    ),
}


def workflow_cases(split, count, calculator):
    rows = temperature_cases(split, count, calculator, offset=1000000)
    for i, case in enumerate(rows):
        case["instruction"] = TEMPLATES[split][i % len(TEMPLATES[split])].format(
            source="current star", other="reference star"
        )
    return rows


def source_records(split):
    if split not in {"train", "development"}:
        raise ValueError("Only source training and development are exposed")
    rng = random.Random(5600000 if split == "train" else 5700000)
    quantities = ("parallax", "flux", "wavelength", "distance", "luminosity")
    units = ("arcsec", "W/m2", "nm", "ly", "Lsun")
    records = []
    for template, quantity, unit in itertools.product(TEMPLATES[split], quantities, units):
        candidates = [
            {
                "id": f"{split}-{rng.getrandbits(96):x}",
                "kind": k,
                "unit": u,
                "source": s,
                "value": rng.uniform(0.001, 1e6),
            }
            for k, u, s in itertools.product(quantities, units, ("current star", "reference star"))
        ]
        rng.shuffle(candidates)
        for source, other in (("current star", "reference star"), ("reference star", "current star")):
            records.append(
                {
                    "instruction": template.format(source=source, other=other),
                    "requirement": {"quantity": quantity, "unit": unit},
                    "candidates": deepcopy(candidates),
                    "expected_id": next(
                        c["id"]
                        for c in candidates
                        if (c["kind"], c["unit"], c["source"]) == (quantity, unit, source)
                    ),
                }
            )
    validate_pairs(records)
    return records


def validate_separation(cases, recognition):
    for left, right in itertools.combinations(cases.values(), 2):
        for key in ("seed", "case_id", "instruction"):
            if {c[key] for c in left} & {c[key] for c in right}:
                raise ValueError("Workflow split overlap")
        for key in ("parallax", "flux", "wavelength"):
            if {c["inputs"][key] for c in left} & {c["inputs"][key] for c in right}:
                raise ValueError("Numeric workflow split overlap")
    training_instructions = {r["instruction"] for r in recognition["train"]}
    for split in ("calibration", "development", "test", "manual"):
        if training_instructions & {c["instruction"] for c in cases[split]}:
            raise ValueError("Training wording overlaps held-out workflow")
    for left, right in itertools.combinations(recognition.values(), 2):
        if {r["instruction"] for r in left} & {r["instruction"] for r in right}:
            raise ValueError("Source instruction overlap")
        for key in ("id", "value"):
            if {c[key] for r in left for c in r["candidates"]} & {
                c[key] for r in right for c in r["candidates"]
            }:
                raise ValueError("Source candidate overlap")


def composition_records(*, punctuation=False):
    """Balanced clause composition across six orders; never called at inference."""
    positives = (
        "Use the {source}'s readings.",
        "Select measurements for the {source}.",
        "Use readings belonging to the {source}.",
        "The {source} is the requested star.",
        "The task concerns the {source}.",
        "Find values for the {source}.",
        "Analyze the {source}.",
        "Determine the {source}'s values.",
        "Obtain all three answers for the {source}.",
        "Use only the measurements from the {source}.",
        "Choose the {source}'s parallax, flux and peak wavelength.",
        "Use measurements of the {source}.",
    )
    negatives = (
        "Ignore the {other}.",
        "Disregard readings from the {other}.",
        "Not those of the {other}.",
        "Not the values for the {other}.",
        "Do not use the {other}'s readings.",
        "Ignore measurements for the {other}.",
        "The {other} is a distractor.",
        "Not those belonging to the {other}.",
        "The {other}'s measurements are distractors.",
        "Not the {other}.",
        "Avoid using data from the {other}.",
        "Disregard the {other}'s data.",
    )
    if punctuation:
        positives += ("Use readings from the {source}.",)
    tasks = (
        "Calculate distance, luminosity and temperature, then copy each result and unit.",
        "Fill the distance, luminosity and temperature fields using the local tool.",
        "Determine distance and luminosity, then temperature. Select all three units.",
        "Read the input requirements, calculate the three answers, and copy each value and its unit.",
    )
    rng, records = random.Random(5800000), []
    fields = (
        ("parallax", "arcsec"),
        ("flux", "W/m2"),
        ("wavelength", "nm"),
        ("distance", "ly"),
        ("luminosity", "Lsun"),
    )
    for positive, negative, task, order in itertools.product(
        positives, negatives, tasks, itertools.permutations(range(3))
    ):
        quantity, unit = rng.choice(fields)
        candidates = [
            {
                "id": f"composition-{rng.getrandbits(96):x}",
                "kind": quantity,
                "unit": unit,
                "source": s,
                "value": rng.uniform(0.001, 1e6),
            }
            for s in ("current star", "reference star")
        ]
        rng.shuffle(candidates)
        separator = rng.choice((". ", "; ", ", ")) if punctuation else ". "
        purpose_clause = punctuation and rng.choice((False, True))
        for source, other in (("current star", "reference star"), ("reference star", "current star")):
            clauses = (positive.format(source=source), negative.format(other=other), task)
            ordered = [clauses[i].rstrip(".") for i in order]
            if separator != ". ":
                ordered = [text if i == 0 else text[0].lower() + text[1:] for i, text in enumerate(ordered)]
                if purpose_clause and order[-1] == 2:
                    ordered[-1] = "to " + ordered[-1]
            records.append(
                {
                    "instruction": separator.join(ordered) + ".",
                    "requirement": {"quantity": quantity, "unit": unit},
                    "candidates": deepcopy(candidates),
                    "expected_id": next(c["id"] for c in candidates if c["source"] == source),
                }
            )
    validate_pairs(records)
    return records
