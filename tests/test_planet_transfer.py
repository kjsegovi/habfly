"""Only shared semantics transfer; new identities start neutral and learn."""

import pytest
import torch

from habfly.data import make_demo_graph
from habfly.model import ConnectomePolicy
from habfly.training.planet_transfer import transfer_stellar_policy


def test_explicit_transfer_preserves_parent_core_and_neutral_new_columns():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        graph = make_demo_graph(16)
        parent = ConnectomePolicy(
            graph,
            hidden_size=16,
            observation_encoding="structured_tool_v6",
            control_encoding="semantic_tool_v1",
            selection_mode="measurement_result_v3",
        )
        before = {k: v.detach().clone() for k, v in parent.state_dict().items()}
        child, metadata = transfer_stellar_policy(parent)
        assert child.graph_hash == parent.graph_hash
        assert child.calibration["status"] == "uncalibrated"
        assert metadata["quantity_aliases"] == {"mass": "stellar_mass", "radius": "stellar_radius"}
        for name, value in before.items():
            assert torch.equal(value, parent.state_dict()[name])
            if name not in metadata["projection_columns"]:
                assert torch.equal(value, child.state_dict()[name])
            else:
                transferred = set()
                for pair in metadata["projection_columns"][name]:
                    a, b = pair["old_column"], pair["new_column"]
                    assert torch.equal(value[:, a], child.state_dict()[name][:, b])
                    transferred.add(b)
                for index in set(range(child.state_dict()[name].shape[1])) - transferred:
                    assert child.state_dict()[name][:, index].count_nonzero() == 0
        assert not metadata["formula_or_correct_action_rules_added"]
        with pytest.raises(ValueError, match="explicit stellar v6"):
            transfer_stellar_policy(child)
    finally:
        torch.set_num_threads(old_threads)
