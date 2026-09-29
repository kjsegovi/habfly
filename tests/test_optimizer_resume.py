"""Same-task continuation preserves moments and never mutates the supplied state."""

from copy import deepcopy

import pytest
import torch

from habfly.contracts import Action, Control, Observation
from habfly.data import make_demo_graph
from habfly.model import ConnectomePolicy
from habfly.training.checkpoints import load_checkpoint
from habfly.training.train import train_behavioral_cloning


def test_bc_resumes_step_counters_without_mutating_parent_optimizer(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        graph = make_demo_graph(8)
        policy = ConnectomePolicy(graph, hidden_size=8, max_answer_length=2)

        def episode(instruction):
            return [
                {
                    "observation": Observation(
                        instruction=instruction, controls=[Control(id="a", label="A")]
                    ).model_dump(mode="json"),
                    "action": Action(kind="CLICK", target="a").model_dump(mode="json"),
                }
            ]

        train, validation = [episode("train")], [episode("validation")]
        train_behavioral_cloning(
            train, graph, tmp_path / "first", policy=policy, validation_episodes=validation, epochs=1
        )
        parent = torch.load(tmp_path / "first/checkpoint.pt", weights_only=True)
        original = deepcopy(parent["optimizer"])
        restored, _ = load_checkpoint(tmp_path / "first/checkpoint.pt", graph)
        report = train_behavioral_cloning(
            train,
            graph,
            tmp_path / "resumed",
            policy=restored,
            validation_episodes=validation,
            epochs=1,
            optimizer_state=parent["optimizer"],
        )
        assert report["optimizer_steps"] == 1 and report["optimizer_initial_step"] == 1
        assert report["optimizer_resumed"]
        after = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)["optimizer"]
        for key, state in original["state"].items():
            assert int(after["state"][key]["step"]) == 2
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, parent["optimizer"]["state"][key][name])
        with pytest.raises(ValueError, match="matching policy"):
            train_behavioral_cloning(
                train, graph, tmp_path / "bad", validation_episodes=validation, optimizer_state=original
            )
        with pytest.raises(ValueError, match="learning rate"):
            train_behavioral_cloning(
                train,
                graph,
                tmp_path / "bad_rate",
                policy=restored,
                validation_episodes=validation,
                optimizer_state=original,
                learning_rate=0.02,
            )
        broken = deepcopy(original)
        next(iter(broken["state"].values()))["exp_avg"].fill_(float("nan"))
        with pytest.raises(ValueError, match="nonfinite"):
            train_behavioral_cloning(
                train,
                graph,
                tmp_path / "bad_state",
                policy=restored,
                validation_episodes=validation,
                optimizer_state=broken,
            )
    finally:
        torch.set_num_threads(previous)
