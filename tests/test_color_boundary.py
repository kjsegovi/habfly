"""Independent boundary coverage and separation; no policy final evaluation."""

from collections import Counter
from copy import deepcopy

import pytest

from habfly.color_reference import COLOR_LABELS, load_color_reference
from habfly.environments.color import color_cases
from habfly.environments.color_boundary import BOUNDARY_SEED, boundary_curriculum, new_boundary_cases
from habfly.training.color import development_gate, record, validate_splits


def test_boundary_curriculum_balances_32_original_and_32_new_without_mutating_parent():
    ref = load_color_reference()
    original = color_cases("train", 64, ref)
    before = deepcopy(original)
    mixed = boundary_curriculum(ref, original)
    assert mixed == boundary_curriculum(ref, original) and original == before
    assert len(mixed) == len({c["case_id"] for c in mixed}) == 64
    retained = [c for c in mixed if c["seed"] < BOUNDARY_SEED]
    fresh = [c for c in mixed if c["seed"] >= BOUNDARY_SEED]
    assert len(retained) == len(fresh) == 32
    assert all(c in original for c in retained)
    counts = Counter(c["expected_color"] for c in mixed)
    assert set(counts) == set(COLOR_LABELS) and sorted(counts.values()) == [7] * 8 + [8]
    for case in mixed:
        examples, summary = record(ref, case)
        assert summary["completed"] and summary["steps"] == 3
        for example in examples:
            text = example.observation.model_dump_json()
            assert all(
                key not in text
                for key in ("curriculum_sampling", "expected_color", "expected_source", "case_id")
            )


def test_all_eight_transitions_have_two_unique_samples_on_each_side_of_printed_edges():
    cases = new_boundary_cases(load_color_reference())
    edges = [(380, 380), (450, 450), (475, 475), (494, 495), (570, 570), (590, 590), (620, 620), (744, 744)]
    counts = Counter()
    values = []
    for case in cases:
        meta = case["curriculum_sampling"]
        pair, side = meta["transition"], meta["side"]
        counts[pair, side] += 1
        value = case["measurements"][case["expected_source"]]["value"]
        values.append(value)
        assert case["expected_color"] == COLOR_LABELS[pair + side]
        assert 0 < (edges[pair][0] - value if side == 0 else value - edges[pair][1]) < 5
        assert value not in (450, 475, 570, 590, 620) and not 494 < value < 495
    assert counts == Counter({(pair, side): 2 for pair in range(8) for side in (0, 1)})
    assert len(set(values)) == 32


def test_new_and_inherited_training_are_disjoint_from_all_reserved_splits():
    ref = load_color_reference()
    original = color_cases("train", 64, ref)
    mixed = boundary_curriculum(ref, original)
    union = list({c["case_id"]: c for c in original + mixed}.values())
    assert len(union) == 96
    # Labels are checked for separation only; no model sees or evaluates final cases.
    validate_splits(
        {"train": union}
        | {s: color_cases(s, 100, ref) for s in ("calibration", "development", "test", "manual", "expert")}
    )


def test_boundary_budgets_and_regression_gate_fail_closed():
    ref = load_color_reference()
    with pytest.raises(ValueError, match="budget"):
        new_boundary_cases(ref, 64)
    with pytest.raises(ValueError, match="unique"):
        boundary_curriculum(ref, color_cases("train", 3, ref))
    smoke = color_cases("train", 4, ref)
    assert len(boundary_curriculum(ref, smoke)) == 4
    with pytest.raises(ValueError, match="unique"):
        boundary_curriculum(ref, smoke[:1] * 4)
    report = {
        "content": {"profile": "pilot", "boundary_curriculum": {"version": 1}},
        "development": {"episodes": 16, "completed": 16, "invalid_actions": 0, "reference_errors": 0},
    }
    assert not development_gate(report)
    report["regression_gate_passed"] = False
    assert not development_gate(report)
    report["regression_gate_passed"] = True
    assert development_gate(report)
