"""Source-reference curriculum v2; label-generation rules are never used at inference.

Known aliases appear in training. Development/test hold out entire instruction
templates and numeric cases, not untaught vocabulary or general English ability.
"""

import itertools
import random
from copy import deepcopy

from habfly.environments.distance_diagnostic import distance_case

from .measurement_identity import QUANTITIES, SOURCES, UNITS
from .source_curriculum import curriculum_records
from .source_request import paired_records, validate_pairs

VERSION = "source-aliases-v3"
ALIASES = (
    ("current star", "reference star"),
    ("supplied star", "comparison star"),
    ("target star", "distractor star"),
    ("star being analyzed", "example star"),
    ("current star", "comparison star"),
    ("supplied star", "reference star"),
)
TEMPLATES = {
    "train": (
        "Find the {source}'s distance from its {quantity} using the local tool. Copy the distance and select its unit.",
        "Use the {source}'s {quantity}.",
        "Select {quantity} from the {source}, not the {other}.",
        "Ignore the {other}. Obtain the {source}'s {quantity}.",
        "The {source} is being analyzed. Bind its {quantity} and copy the result.",
        "Find distance for the {source}; use its {quantity} and select its unit.",
        "Determine distance from the {source}'s {quantity}. Copy the result and select its unit.",
        "Use data belonging to the {source}. The {other}'s readings are distractors.",
        "Disregard the {other}'s readings. Use {quantity} for the {source}.",
        "Analyze the {source}: calculate distance using its {quantity}, then fill the distance and unit.",
        "The {other} is not the requested source. Bind {quantity} for the {source}.",
        "For the {source}, read its {quantity}. Calculate and transfer the result.",
        "Read the input requirements and choose a measurement for the {source}, not the {other}.",
    ),
    "development": (
        "For the {source}, obtain {quantity}; ignore the {other}.",
        "Analyze the {source}. Use its {quantity} and select the distance unit.",
        "Use {quantity} belonging to the {source}; readings of the {other} are distractors.",
    ),
    "test": (
        "Obtain distance for the {source} using its {quantity}. Copy the result and its unit.",
        "Ignore the {other}'s data and determine distance from the {source}'s {quantity}.",
        "For the {source}, calculate distance; bind its {quantity}, then select the distance unit.",
    ),
    "manual": (
        "Calculate the distance of the {source} from its {quantity}. Copy the result and select its unit.",
        "Determine distance for the {source}. Use its {quantity}, not readings from the {other}.",
    ),
}
SEEDS = {"train": 2100000, "development": 2200000, "test": 2300000, "manual": 2400000}


def alias_records(split):
    rng, records = random.Random(SEEDS[split]), []
    for template, aliases, quantity, unit in itertools.product(TEMPLATES[split], ALIASES, QUANTITIES, UNITS):
        candidates = [
            {
                "kind": k,
                "unit": u,
                "source": s,
                "value": rng.uniform(0.01, 1e6),
                "id": f"alias-{split}-{rng.getrandbits(96):024x}",
            }
            for k, u, s in itertools.product(QUANTITIES, UNITS, SOURCES)
        ]
        rng.shuffle(candidates)
        for i, source in enumerate(SOURCES):
            records.append(
                {
                    "instruction": template.format(
                        source=aliases[i], other=aliases[1 - i], quantity=quantity
                    ),
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


def rehearsal_records():
    """Previously consumed sets become explicit rehearsal, never fresh validation."""
    return (
        curriculum_records("vocabulary")
        + curriculum_records("train")
        + curriculum_records("development")
        + paired_records("train")
        + paired_records("development")
    )


def workflow_cases(split, count, calculator):
    cases = []
    for index in range(count):
        case = distance_case(SEEDS[split] + index, "test" if split == "manual" else split, calculator)
        source, other = ALIASES[index % len(ALIASES)]
        template = TEMPLATES[split][(index // len(ALIASES)) % len(TEMPLATES[split])]
        case["instruction"] = template.format(source=source, other=other, quantity="parallax")
        cases.append(case)
    return cases


def validate_separation(recognition, workflows):
    for left, right in itertools.combinations(recognition, 2):
        if {r["instruction"] for r in recognition[left]} & {r["instruction"] for r in recognition[right]}:
            raise ValueError("Source instruction split overlap")
        for key in ("id", "value"):
            if {c[key] for r in recognition[left] for c in r["candidates"]} & {
                c[key] for r in recognition[right] for c in r["candidates"]
            }:
                raise ValueError("Recognition payload split overlap")
    for left, right in itertools.combinations(workflows, 2):
        for key in ("case_id", "seed", "instruction"):
            if {c[key] for c in workflows[left]} & {c[key] for c in workflows[right]}:
                raise ValueError("Workflow split overlap")
        if {c["inputs"]["parallax"] for c in workflows[left]} & {
            c["inputs"]["parallax"] for c in workflows[right]
        }:
            raise ValueError("Workflow numeric split overlap")
