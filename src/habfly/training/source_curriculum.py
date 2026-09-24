"""Versioned lexical grounding, known-word composition, and unseen-synonym diagnostics.

This grammar generates supervised practice only; inference never uses its rules.
Single negative directives are exercises with exactly two possible source labels.
"""

from __future__ import annotations

import itertools
import random
import re
from copy import deepcopy

from .measurement_identity import QUANTITIES, SOURCES, UNITS
from .source_request import validate_pairs

VERSION = "source-vocabulary-v1"
POSITIVE = ("use", "select", "obtain", "find")
NEGATIVE = ("ignore", "disregard", "avoid", "not")
# Exclude combinations used in the source parent's training or subsequent probes.
PRIOR_COMBINATIONS = frozenset(
    {
        ("use", "ignore"),
        ("find", "ignore"),
        ("select", "not"),
        ("obtain", "disregard"),
        ("obtain", "ignore"),
        ("use", "disregard"),
        ("select", "ignore"),
    }
)
HELD_OUT = frozenset({("use", "avoid"), ("select", "disregard"), ("obtain", "not"), ("find", "avoid")})
UNKNOWN_POSITIVE = ("retrieve", "gather")
UNKNOWN_NEGATIVE = ("exclude", "omit")
SEEDS = {"vocabulary": 1500000, "train": 1600000, "development": 1700000, "synonym_diagnostic": 1800000}


def curriculum_records(split):
    if split not in SEEDS:
        raise ValueError("Unsupported source curriculum split")
    rng = random.Random(SEEDS[split])
    plans = []
    if split == "vocabulary":
        plans = [(word, None, "single_positive") for word in POSITIVE]
        plans += [(None, word, "single_negative") for word in NEGATIVE]
    else:
        positive, negative = (
            (UNKNOWN_POSITIVE, UNKNOWN_NEGATIVE) if split == "synonym_diagnostic" else (POSITIVE, NEGATIVE)
        )
        for pos, neg in itertools.product(positive, negative):
            if split == "train" and (pos, neg) in HELD_OUT:
                continue
            if split == "development" and (pos, neg) not in HELD_OUT:
                continue
            plans.extend((pos, neg, order) for order in ("positive_first", "negative_first"))
    records = []
    for positive, negative, order in plans:
        for quantity, unit in itertools.product(QUANTITIES, UNITS):
            candidates = [
                {
                    "kind": k,
                    "unit": u,
                    "source": s,
                    "value": rng.uniform(0.01, 1e6),
                    "id": f"{split}-{rng.getrandbits(96):024x}",
                }
                for k, u, s in itertools.product(QUANTITIES, UNITS, SOURCES)
            ]
            rng.shuffle(candidates)
            for source in SOURCES:
                other = next(s for s in SOURCES if s != source)
                yes = f"{positive.capitalize()} the {source}'s {quantity}." if positive else ""
                no = f"{negative.capitalize()} the {other}'s {quantity}." if negative else ""
                clauses = (no, yes) if order == "negative_first" else (yes, no)
                records.append(
                    {
                        "instruction": " ".join(c for c in clauses if c),
                        "requirement": {"quantity": quantity, "unit": unit},
                        "candidates": deepcopy(candidates),
                        "expected_id": next(
                            c["id"]
                            for c in candidates
                            if (c["kind"], c["unit"], c["source"]) == (quantity, unit, source)
                        ),
                        # Analysis metadata is explicitly NOT consumed by the policy.
                        "curriculum": {
                            "positive": positive,
                            "negative": negative,
                            "order": order,
                            "split": split,
                        },
                    }
                )
    validate_pairs(records)
    return records


def words(records):
    return {word for r in records for word in re.findall(r"[a-z]+", r["instruction"].lower())}


def validate_curriculum(datasets):
    if set(datasets) != set(SEEDS):
        raise ValueError("Incomplete curriculum splits")
    for records in datasets.values():
        validate_pairs(records)
    train_words = words(datasets["vocabulary"] + datasets["train"])
    unknown = words(datasets["development"]) - train_words
    if unknown:
        raise ValueError(f"Composition development contains untaught words: {sorted(unknown)}")
    pairs = {
        s: {(r["curriculum"]["positive"], r["curriculum"]["negative"]) for r in datasets[s]}
        for s in ("train", "development")
    }
    if (
        pairs["train"] & pairs["development"]
        or pairs["development"] != HELD_OUT
        or HELD_OUT & PRIOR_COMBINATIONS
    ):
        raise ValueError("Composition holdout is not separate from training and previous probes")
    for left, right in itertools.combinations(datasets, 2):
        for key in ("id", "value"):
            if {c[key] for r in datasets[left] for c in r["candidates"]} & {
                c[key] for r in datasets[right] for c in r["candidates"]
            }:
                raise ValueError("Numeric payloads or IDs overlap across curriculum splits")
        if {r["instruction"] for r in datasets[left]} & {r["instruction"] for r in datasets[right]}:
            raise ValueError("Instruction overlap across curriculum splits")
    diagnostic_words = words(datasets["synonym_diagnostic"]) - train_words
    if diagnostic_words != set(UNKNOWN_POSITIVE + UNKNOWN_NEGATIVE):
        raise ValueError("Unseen-synonym diagnostic lost its separate lexical boundary")
    return {
        "version": VERSION,
        "composition_unknown_words": sorted(unknown),
        "held_out_combinations": sorted([list(pair) for pair in HELD_OUT]),
        "unseen_diagnostic_words": sorted(diagnostic_words),
        "split_examples": {s: len(r) for s, r in datasets.items()},
        "single_negative_scope": "exactly two visible source labels; choose the nonexcluded source",
        "synonym_diagnostic_is_gate": False,
    }


def resume_source_pointer(policy):
    if policy.selection_mode != "measurement_source_v2" or not hasattr(
        policy.measurement_identity, "source_request"
    ):
        raise ValueError("Vocabulary curriculum requires an existing measurement_source_v2 checkpoint")
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith("measurement_identity.source_request."))
    policy.action_temperature = policy.target_temperature = 1.0
    policy.calibration = {"status": "uncalibrated", "reason": "source_vocabulary_curriculum"}
