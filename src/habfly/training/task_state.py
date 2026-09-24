"""Explicit task-state/unit migrations; preserve old encodings and neutral new inputs."""

import torch

from habfly.model.tool_state import OPTION_WIDTH, STATE_WIDTH, TASK_STATE_WIDTH


def migrate_task_state(parent, policy):
    state = parent.state_dict()
    if (
        parent.observation_encoding == "structured_tool_v4"
        and policy.observation_encoding == "structured_tool_v5"
    ):
        weight = state["tool_state_projection.weight"]
        if weight.shape != (policy.hidden_size, STATE_WIDTH):
            raise ValueError("Unexpected v4 projection dimensions")
        expanded = weight.new_zeros((policy.hidden_size, TASK_STATE_WIDTH))
        expanded[:, :STATE_WIDTH] = weight
        state["tool_state_projection.weight"] = expanded
    elif (
        parent.observation_encoding == "structured_tool_v5"
        and policy.observation_encoding == "structured_tool_v6"
    ):
        # The last unit slot changes from Gyr to yr. Do not transfer learned Gyr
        # meaning or modify the parent tensor; the newly represented yr starts neutral.
        weight = state["option_projection.weight"].clone()
        if weight.shape != (policy.hidden_size, OPTION_WIDTH):
            raise ValueError("Unexpected option projection dimensions")
        weight[:, -1] = 0
        state["option_projection.weight"] = weight
    elif parent.observation_encoding != policy.observation_encoding:
        raise ValueError("Unsupported task-state migration")
    policy.load_state_dict(state, strict=True)
    # New assignment inputs start neutral; training must learn their effect.
    if (
        parent.observation_encoding == "structured_tool_v4"
        and policy.observation_encoding == "structured_tool_v5"
    ):
        assert torch.count_nonzero(policy.tool_state_projection.weight[:, STATE_WIDTH:]) == 0
    if (
        parent.observation_encoding == "structured_tool_v5"
        and policy.observation_encoding == "structured_tool_v6"
    ):
        assert torch.count_nonzero(policy.option_projection.weight[:, -1]) == 0
