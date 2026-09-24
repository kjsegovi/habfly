import copy
import io
import itertools
import socket

import pytest
import torch

from habfly.contracts import Action
from habfly.data import make_demo_graph
from habfly.environments.temperature import TemperatureEnv, temperature_cases
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy
from habfly.model.measurement_identity import measurement_request
from habfly.model.tool_state import state_features, visible_sources
from habfly.runtime import Runtime, read_trace
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.frozen_text import frozen_text_cache
from habfly.training.luminosity import audit_chain, clean, rollout
from habfly.training.stellar import record_episode
from habfly.training.train import episodes_to_examples, supervised_loss


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Temperature task attempted network or Sheets access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    torch.set_num_threads(1)


@pytest.fixture
def chain():
    calculator = LocalCalculator(load_knowledge_pack())
    case = temperature_cases("train", 1, calculator, offset=50000)[0]
    env = TemperatureEnv(calculator, [case], max_steps=64)
    trajectory, summary = record_episode(env, case["seed"])
    return calculator, case, trajectory, summary


def act(env, key, value=None):
    obs = env.observe()
    target = next(c.id for c in obs.controls if c.id.partition(":")[2] == key)
    return env.step(
        Action(
            kind="SELECT" if value is not None else "CLICK",
            target=target,
            value=value,
            observation_revision=obs.revision,
        )
    )


def test_golden_temperature_expert_and_replay(chain, tmp_path):
    calculator, case, trajectory, summary = chain
    assert calculator.verify()["cases"] == 9
    result = calculator.execute("temperature", {"wavelength": {"value": 212, "unit": "nm"}}, "giant")
    assert result.ok and result.unit == "K"
    assert result.value == pytest.approx(13668.719339622641, rel=1e-12)
    assert summary["completed"] and summary["steps"] == 31
    audit = audit_chain(case, trajectory)
    assert (
        audit["current_star_wavelength_uses"]
        == audit["distance_result_reuses"]
        == audit["current_star_flux_uses"]
        == 1
    )
    assert audit["exact_copies"] == 3 and audit["selection_correct"] == audit["selection_attempts"]
    report = rollout(None, calculator, [case], tmp_path / "expert", environment=TemperatureEnv)
    assert clean(report, 1, steps=31) and not clean(report, 1)  # No two-field gate confusion.
    events = read_trace(tmp_path / "expert" / f"{case['seed']}.events.jsonl")
    assert next(e.payload for e in events if e.event == "hello")["policy"] == "scripted_expert"
    assert next(e.payload for e in events if e.event == "episode_summary")["completed"]
    for step in trajectory:
        assert "expected" not in step["observation"]["values"]
        assert "stage" not in step["observation"]["calculation"]
        calculation = next(c for c in step["observation"]["controls"] if c["label"] == "Calculation")
        assert len(calculation["options"]) == 6


def test_temperature_wrong_source_staleness_and_reset_isolation(chain):
    calculator, case, trajectory, _ = chain
    env = TemperatureEnv(calculator, [case], max_steps=64)
    env.reset(seed=case["seed"])
    wrong = next(
        k
        for k, v in case["measurements"].items()
        if v["kind"] == "wavelength" and v["source"] == "reference star"
    )
    right = next(
        k
        for k, v in case["measurements"].items()
        if v["kind"] == "wavelength" and v["source"] == "current star"
    )
    for step in trajectory[:26]:
        action = Action.model_validate(step["action"])
        if (
            action.target.partition(":")[2] == "source"
            and step["observation"]["calculation"]["operation"] == "temperature"
        ):
            action.value = wrong
        env.step(action)
    assert env.results["r3"]["value"] == pytest.approx(case["expected"]["temperature"] / 1.7)
    assert visible_sources(env.observe().model_dump())["r3"]["source"] == "reference star"
    old_results = copy.deepcopy({k: env.results[k] for k in ("r1", "r2")})
    act(env, "result", "r3")
    act(env, "destination", "temperature")
    act(env, "copy")
    assert env.answers["temperature"] == env.results["r3"]["value"]
    act(env, "source", right)
    act(env, "bind")
    assert not env.results["r3"]["valid"]
    assert {k: env.results[k] for k in ("r1", "r2")} == old_results
    env.reset(seed=case["seed"])
    assert not env.results and not env.answers and not env.units and not env.bindings


def test_two_answers_do_not_complete_three_field_task(chain):
    calculator, case, trajectory, _ = chain
    env = TemperatureEnv(calculator, [case], max_steps=64)
    env.reset(seed=case["seed"])
    for step in trajectory[:21]:
        env.step(Action.model_validate(step["action"]))
    obs, _, terminated, truncated, _ = act(env, "check")
    assert not terminated and not truncated and not obs["progress"]["task_completed"]
    assert set(obs["values"]["answers"]) == {"distance", "luminosity"}


def test_temperature_mixed_sources_and_payload_independence(chain):
    _, _, trajectory, _ = chain
    obs = copy.deepcopy(trajectory[23]["observation"])
    control = next(c for c in obs["controls"] if c["label"] == "Measurement or result")
    request = measurement_request(obs, control, include_results=True)
    assert request["requirement"] == {"quantity": "wavelength", "unit": "nm"}
    assert len(request["candidates"]) == 8
    assert {c["kind"] for c in request["candidates"]} == {
        "parallax",
        "flux",
        "wavelength",
        "distance",
        "luminosity",
    }
    before = state_features(obs)
    for row in obs["values"]["measurements"].values():
        row["value"] = 9999
    for row in obs["calculation"]["results"].values():
        row["value"] = 0
    assert state_features(obs) == before


def test_split_numbers_and_instructions_are_separate():
    calculator = LocalCalculator(load_knowledge_pack())
    splits = {
        s: temperature_cases(s, 16, calculator, offset=50000)
        for s in ("train", "calibration", "development", "test", "manual")
    }
    for left, right in itertools.combinations(splits.values(), 2):
        for key in ("seed", "case_id", "instruction"):
            assert not {c[key] for c in left} & {c[key] for c in right}
        for key in ("parallax", "flux", "wavelength"):
            assert not {c["inputs"][key] for c in left} & {c["inputs"][key] for c in right}


def test_three_result_option_gradients_and_reload(chain, tmp_path):
    _, _, trajectory, _ = chain
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
    with frozen_text_cache(model):
        state, losses = None, []
        for example in examples:
            loss, output = supervised_loss(model, [example], state)
            state = output.state
            losses.append(loss)
        combined = torch.stack(losses).mean()
        assert torch.isfinite(combined)
        combined.backward()
    for module in (
        "cell",
        "tool_state_projection",
        "option_projection",
        "control_projection",
        "measurement_identity",
    ):
        grads = [p.grad for p in getattr(model, module).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        stage="synthetic_test_only",
        seed=0,
        content_pack={"required_fields": ["distance", "luminosity", "temperature"]},
    )
    restored, _ = load_checkpoint(
        path, graph, content_pack={"required_fields": ["distance", "luminosity", "temperature"]}
    )
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="content-pack mismatch"):
        load_checkpoint(path, graph, content_pack={"required_fields": ["distance", "luminosity"]})


@pytest.mark.parametrize(
    "settings",
    [
        {"environment": "browser"},
        {"calculation_backend": "google_sheets"},
        {"spreadsheet_config": "ignored"},
        {"policy": "expert"},
        {"graph": None},
    ],
)
def test_temperature_runtime_rejects_out_of_scope_before_access(tmp_path, settings):
    runtime = Runtime(io.StringIO())
    with pytest.raises(ValueError):
        runtime.command(
            {
                "command": "start",
                "payload": {
                    "task": "temperature",
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
