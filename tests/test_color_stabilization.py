"""Checkpoint selection uses development metrics only, with deterministic ties."""

from copy import deepcopy

import pytest

from habfly.training.color_stabilization import selection_key


def candidate(epoch, completed, nll, errors=0):
    return {
        "epoch": epoch,
        "development": {
            "completed": completed,
            "invalid_actions": errors,
            "reference_errors": 0,
            "infrastructure_failures": 0,
        },
        "fixed_weight_losses": {"development": {"color_nll": nll}},
    }


def test_selection_prefers_safe_completion_then_nll_and_keeps_earliest_tie():
    baseline = candidate(0, 11, 0.8)
    winner = candidate(1, 15, 0.4)
    later_worse = candidate(2, 14, 0.1)
    unsafe = candidate(3, 16, 0.05, errors=1)
    tied = candidate(4, 15, 0.4)
    assert max([baseline, winner, later_worse, unsafe, tied], key=selection_key) is winner
    assert max([baseline, candidate(1, 10, 0.1)], key=selection_key) is baseline
    assert selection_key(candidate(2, 11, 0.3)) > selection_key(baseline)


def test_training_and_calibration_scores_cannot_select_checkpoint():
    epoch = candidate(1, 12, 0.4)
    changed = deepcopy(epoch)
    changed.update(train={"completed": 64}, calibration={"completed": 16}, final={"completed": 100})
    assert selection_key(changed) == selection_key(epoch)
    with pytest.raises(ValueError, match="Nonfinite"):
        selection_key(candidate(2, 16, float("nan")))
