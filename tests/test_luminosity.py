import copy
import io
import itertools
import socket

import pytest
import torch

from habfly.contracts import Action
from habfly.data import make_demo_graph
from habfly.environments.luminosity import MAX_STEPS, LuminosityEnv, luminosity_cases
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy
from habfly.model.measurement_identity import measurement_request
from habfly.model.tool_state import option_features, state_features, visible_sources
from habfly.runtime import Runtime
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.frozen_text import frozen_text_cache
from habfly.training.luminosity import audit_chain
from habfly.training.stellar import record_episode
from habfly.training.train import episodes_to_examples, supervised_loss


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("No network or spreadsheet access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    torch.set_num_threads(1)


@pytest.fixture
def chain():
    calculator = LocalCalculator(load_knowledge_pack())
    case = luminosity_cases("train", 1, calculator)[0]
    env = LuminosityEnv(calculator, [case], max_steps=MAX_STEPS)
    trajectory, summary = record_episode(env, case["seed"])
    return calculator, case, env, trajectory, summary


def test_expert_chain_reuses_visible_result_and_exact_copy(chain):
    calculator, case, _, trajectory, summary = chain
    assert calculator.verify()["cases"] == 9
    assert summary["completed"] and summary["steps"] == 22
    audit = audit_chain(case, trajectory)
    assert audit["distance_result_reuses"] == audit["current_star_flux_uses"] == 1
    assert audit["exact_copies"] == 2
    assert audit["selection_correct"] == audit["selection_attempts"]
    for step in trajectory:
        assert "expected" not in step["observation"]["values"]
        assert "stage" not in step["observation"]["calculation"]
    # All six operation choices stay visible; no forced next-action surface.
    assert (
        len(
            next(
                c["options"] for c in trajectory[0]["observation"]["controls"] if c["label"] == "Calculation"
            )
        )
        == 6
    )


def test_mixed_pointer_and_truthful_lineage(chain):
    _, _, _, trajectory, _ = chain
    obs = copy.deepcopy(trajectory[14]["observation"])
    control = next(c for c in obs["controls"] if c["label"] == "Measurement or result")
    assert measurement_request(obs, control) is None  # Old checkpoint behavior unchanged.
    request = measurement_request(obs, control, include_results=True)
    assert len(request["candidates"]) == 7
    distance = next(c for c in request["candidates"] if c["kind"] == "distance")
    assert distance["unit"] == "ly" and distance["source"] == "current star"
    reference = next(
        k
        for k, c in obs["values"]["measurements"].items()
        if c["kind"] == "parallax" and c["source"] == "reference star"
    )
    obs["calculation"]["results"]["r1"]["bindings"]["parallax"] = reference
    assert visible_sources(obs)["r1"]["source"] == "reference star"
    obs["calculation"]["results"]["r1"]["bindings"]["parallax"] = "r1"
    assert visible_sources(obs)["r1"]["source"] == "unknown source"


def test_wrong_flux_is_executed_not_repaired(chain):
    calculator, case, _, trajectory, _ = chain
    env = LuminosityEnv(calculator, [case], max_steps=MAX_STEPS)
    env.reset(seed=case["seed"])
    wrong = next(
        k for k, c in case["measurements"].items() if c["kind"] == "flux" and c["source"] == "reference star"
    )
    for index, step in enumerate(trajectory[:17]):
        action = Action.model_validate(step["action"])
        if index == 11:
            action.value = wrong
        env.step(action)
    assert env.results["r2"]["value"] == pytest.approx(case["expected"]["luminosity"] * 1.7)
    assert visible_sources(env.observe().model_dump())["r2"]["source"] == "mixed or unknown sources"
    env.reset(seed=case["seed"])
    assert not env.results and not env.answers and not env.bindings


def test_features_ignore_payloads_and_represent_second_operation(chain):
    _, _, _, trajectory, _ = chain
    before = copy.deepcopy(trajectory[14]["observation"])
    changed = copy.deepcopy(before)
    for row in changed["values"]["measurements"].values():
        row["value"] = -9000
    for row in changed["calculation"]["results"].values():
        row["value"] = 9000
    assert state_features(before) == state_features(changed)
    assert option_features(before, "r1") == option_features(changed, "r1")
    assert state_features(trajectory[6]["observation"]) != state_features(trajectory[18]["observation"])


def test_split_numeric_and_template_separation():
    calculator = LocalCalculator(load_knowledge_pack())
    splits = {
        s: luminosity_cases(s, 16, calculator)
        for s in ("train", "calibration", "development", "test", "manual")
    }
    for left, right in itertools.combinations(splits.values(), 2):
        for key in ("case_id", "seed", "instruction"):
            assert not {c[key] for c in left} & {c[key] for c in right}
        for quantity in ("parallax", "flux"):
            assert not {c["inputs"][quantity] for c in left} & {c["inputs"][quantity] for c in right}


def test_v4_finite_gradients_cache_equivalence_and_reload(chain, tmp_path):
    _, _, _, trajectory, _ = chain
    graph = make_demo_graph(16)
    model = ConnectomePolicy(
        graph,
        hidden_size=16,
        observation_encoding="structured_tool_v4",
        selection_mode="measurement_result_v3",
        control_encoding="semantic_tool_v1",
    )
    model.embedding.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    examples = episodes_to_examples([trajectory])
    ordinary, output = supervised_loss(model, [examples[14]])
    with frozen_text_cache(model):
        cached, cached_output = supervised_loss(model, [examples[14]])
        assert torch.allclose(ordinary, cached, atol=1e-5)
        assert torch.allclose(output.pooled, cached_output.pooled, atol=1e-5)
        cached.backward()
    for name in ("tool_state_projection", "control_projection", "cell", "measurement_identity"):
        gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, stage="synthetic_test_only", seed=0, content_pack={"scope": "chain"})
    restored, _ = load_checkpoint(path, graph, content_pack={"scope": "chain"})
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError):
        load_checkpoint(path, graph, content_pack={"scope": "distance"})
    model.text_encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"), frozen_text_cache(model):
        pass


@pytest.mark.parametrize(
    "settings",
    [
        {"environment": "browser"},
        {"calculation_backend": "google_sheets"},
        {"policy": "expert"},
        {"graph": None},
        {"spreadsheet_config": "ignored"},
    ],
)
def test_luminosity_rejects_other_backends_before_access(tmp_path, settings):
    runtime = Runtime(io.StringIO())
    with pytest.raises(ValueError):
        runtime.command(
            {
                "command": "start",
                "payload": {
                    "task": "luminosity",
                    "policy": "checkpoint",
                    "dataset": "missing",
                    "graph": "missing",
                    "checkpoint": "missing",
                    "artifact_dir": str(tmp_path / "never"),
                    **settings,
                },
            }
        )
    assert not (tmp_path / "never").exists()
