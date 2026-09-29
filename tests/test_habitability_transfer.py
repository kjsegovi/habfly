"""Only actual shared meanings transfer; new temperature quantities start neutral."""

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.training.habitability_transfer import transfer_planet_policy
from habfly.training.planet_calculations import new_policy


def test_transfer_preserves_core_and_parent_and_does_not_alias_unrelated_physics():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        parent = new_policy(make_demo_graph(16))
        before = {k: v.detach().clone() for k, v in parent.state_dict().items()}
        child, metadata = transfer_planet_policy(parent)
        assert child.graph_hash == parent.graph_hash
        assert child.calibration["status"] == "uncalibrated"
        assert metadata["shared_quantities"] == ["orbital_radius"]
        assert not metadata["formula_or_correct_action_rules_added"]
        for name, value in before.items():
            assert torch.equal(value, parent.state_dict()[name])
            if name not in metadata["projection_columns"]:
                assert torch.equal(value, child.state_dict()[name])
            else:
                copied = set()
                for pair in metadata["projection_columns"][name]:
                    a, b = pair["old_column"], pair["new_column"]
                    assert torch.equal(value[:, a], child.state_dict()[name][:, b])
                    copied.add(b)
                for column in set(range(child.state_dict()[name].shape[1])) - copied:
                    assert child.state_dict()[name][:, column].count_nonzero() == 0
        with pytest.raises(ValueError, match="explicit planet v1"):
            transfer_planet_policy(child)
    finally:
        torch.set_num_threads(previous)
