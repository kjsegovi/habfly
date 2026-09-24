import copy
import io
import itertools
import socket

import pytest
import torch

from habfly.contracts import Action
from habfly.data import make_demo_graph
from habfly.environments.radius import RadiusEnv, radius_cases
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy
from habfly.model.measurement_identity import measurement_request
from habfly.model.tool_state import task_state_features, visible_sources
from habfly.runtime import Runtime, read_trace
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.chained_workflow import workflow_spec
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.frozen_text import frozen_text_cache
from habfly.training.luminosity import audit_chain, clean, rollout
from habfly.training.stellar import record_episode
from habfly.training.train import episodes_to_examples, supervised_loss


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Radius task attempted network or Sheets access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    torch.set_num_threads(1)


@pytest.fixture
def calculator():
    return LocalCalculator(load_knowledge_pack())


def act(env, key, value=None):
    obs = env.observe()
    return env.step(
        Action(
            kind="SELECT" if value is not None else "CLICK",
            target=next(c.id for c in obs.controls if c.id.partition(":")[2] == key),
            value=value,
            observation_revision=obs.revision,
        )
    )


def bindings(luminosity=1, temperature=5800):
    return {
        "luminosity": {"value": luminosity, "unit": "Lsun"},
        "temperature": {"value": temperature, "unit": "K"},
    }


def test_independent_radius_golden_and_errors(calculator):
    assert calculator.verify()["cases"] == 9
    # Spreadsheet H2: sqrt(L)/(T/5800)^2, independently simplified integer cases.
    for luminosity, temperature, expected in ((1, 5800, 1), (16, 11600, 1), (4, 2900, 8)):
        result = calculator.execute("radius", bindings(luminosity, temperature), "main_sequence")
        assert result.ok and result.unit == "Rsun" and result.value == pytest.approx(expected)
    for cls in ("white_dwarf", "giant"):
        assert calculator.execute("radius", bindings(), cls).error == "operation_not_applicable"
    for inputs, error in (
        ({"luminosity": {"value": 1, "unit": "Lsun"}}, "missing_or_extra_inputs"),
        (bindings(1, 0), "input_must_be_positive"),
        (bindings(0, 5800), "input_must_be_positive"),
        (bindings(-1, 5800), "input_must_be_positive"),
        (bindings(1, None), "input_not_finite_number"),
        (bindings(1, float("inf")), "input_not_finite_number"),
        (
            {"luminosity": {"value": 1, "unit": "Lsun"}, "temperature": {"value": 5800, "unit": "nm"}},
            "incompatible_unit",
        ),
    ):
        assert calculator.execute("radius", inputs, "main_sequence").error == error


def test_expert_both_branches_pair_reuse_and_replay(calculator, tmp_path):
    cases = radius_cases("train", 4, calculator, offset=50000)
    report = rollout(None, calculator, cases, tmp_path / "expert", environment=RadiusEnv)
    assert clean(report, 4, steps=workflow_spec("radius").expected_steps)
    assert [e["steps"] for e in report["episodes"]] == [51, 31, 51, 31]
    assert all(
        e["mass_applicability_correct"] and e["radius_applicability_correct"] for e in report["episodes"]
    )
    for case in cases:
        trajectory, _ = record_episode(RadiusEnv(calculator, [case], max_steps=64), case["seed"])
        assert audit_chain(case, trajectory)["radius_result_pair_reuses"] == int(
            case["star_class"] == "main_sequence"
        )
        events = read_trace(tmp_path / "expert" / f"{case['seed']}.events.jsonl")
        assert next(e.payload for e in events if e.event == "episode_summary")["completed"]
        for row in trajectory:
            obs = row["observation"]
            assert "expected" not in obs["values"] and "stage" not in obs["calculation"]
            assert len(next(c for c in obs["controls"] if c["label"] == "Calculation")["options"]) == 6


def test_wrong_binding_is_calculated_exactly_and_invalidated(calculator):
    case = radius_cases("train", 1, calculator, offset=50000)[0]
    env = RadiusEnv(calculator, [case], max_steps=64)
    trajectory, _ = record_episode(env, case["seed"])
    case["measurements"]["wrong-L"] = {
        "kind": "luminosity",
        "value": 16,
        "unit": "Lsun",
        "source": "reference star",
    }
    env.reset(seed=case["seed"])
    for row in trajectory[:39]:
        env.step(Action.model_validate(row["action"]))
    previous = copy.deepcopy(env.results)
    act(env, "operation", "radius")
    act(env, "parameter", "luminosity")
    act(env, "source", "wrong-L")
    act(env, "bind")
    act(env, "parameter", "temperature")
    act(env, "source", "r3")
    act(env, "bind")
    act(env, "execute")
    wrong = 4 / (env.results["r3"]["value"] / 5800) ** 2
    assert env.results["r5"]["value"] == pytest.approx(wrong)
    assert visible_sources(env.observe().model_dump())["r5"]["source"] == "mixed or unknown sources"
    act(env, "result", "r5")
    act(env, "destination", "radius")
    act(env, "copy")
    assert env.answers["radius"] == env.results["r5"]["value"]
    act(env, "parameter", "luminosity")
    act(env, "source", "r2")
    act(env, "bind")
    assert not env.results["r5"]["valid"]
    assert {k: env.results[k] for k in previous} == previous
    env.reset(seed=case["seed"])
    assert not env.results and not env.answers and not env.bindings


def test_mass_completion_not_radius_completion_and_class_guards(calculator):
    for case in radius_cases("train", 4, calculator, offset=50000):
        env = RadiusEnv(calculator, [case], max_steps=64)
        trajectory, _ = record_episode(env, case["seed"])
        env.reset(seed=case["seed"])
        for row in trajectory[: 39 if case["star_class"] == "main_sequence" else 30]:
            env.step(Action.model_validate(row["action"]))
        if case["star_class"] == "main_sequence":
            obs, _, terminated, _, _ = act(env, "check")
            assert not terminated and not obs["progress"]["task_completed"]
            assert "radius" not in obs["values"]["answers"]
        else:
            act(env, "operation", "radius")
            for parameter, result in (("luminosity", "r2"), ("temperature", "r3")):
                act(env, "parameter", parameter)
                act(env, "source", result)
                act(env, "bind")
            act(env, "execute")
            assert env.tool_error == "operation_not_applicable" and len(env.results) == 3


def test_radius_sources_features_gradients_and_reload(calculator, tmp_path):
    case = radius_cases("train", 1, calculator, offset=50000)[0]
    trajectory, _ = record_episode(RadiusEnv(calculator, [case], max_steps=64), case["seed"])
    obs = copy.deepcopy(trajectory[43]["observation"])
    control = next(c for c in obs["controls"] if c["label"] == "Measurement or result")
    request = measurement_request(obs, control, include_results=True)
    assert request["requirement"] == {"quantity": "temperature", "unit": "K"}
    assert len(request["candidates"]) == 10 and any(c["kind"] == "mass" for c in request["candidates"])
    features = task_state_features(obs)
    for row in obs["calculation"]["results"].values():
        row["value"] = 0
    assert task_state_features(obs) == features
    graph = make_demo_graph(16)
    policy = ConnectomePolicy(
        graph,
        hidden_size=16,
        observation_encoding="structured_tool_v5",
        selection_mode="measurement_result_v3",
        control_encoding="semantic_tool_v1",
    )
    policy.embedding.requires_grad_(False)
    policy.text_encoder.requires_grad_(False)
    with frozen_text_cache(policy):
        state, losses = None, []
        for example in episodes_to_examples([trajectory]):
            loss, output = supervised_loss(policy, [example], state)
            losses.append(loss)
            state = output.state
        loss = torch.stack(losses).mean()
        assert torch.isfinite(loss)
        loss.backward()
    for module in (
        "cell",
        "tool_state_projection",
        "option_projection",
        "control_projection",
        "measurement_identity",
    ):
        gradients = [p.grad for p in getattr(policy, module).parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
    content = {"scope": "test_only_radius", "required_fields": case["required"]}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, policy, stage="synthetic_test_only", seed=0, content_pack=content)
    restored, _ = load_checkpoint(path, graph, content_pack=content)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in policy.state_dict().items())
    with pytest.raises(ValueError, match="content-pack mismatch"):
        load_checkpoint(path, graph, content_pack={"scope": "mass"})


def test_split_separation_and_all_class_templates(calculator):
    splits = {
        s: radius_cases(s, 16, calculator, offset=50000)
        for s in ("train", "calibration", "development", "test", "manual")
    }
    for a, b in itertools.combinations(splits.values(), 2):
        for key in ("seed", "case_id", "instruction"):
            assert not {c[key] for c in a} & {c[key] for c in b}
        for key in ("parallax", "flux", "wavelength"):
            assert not {c["inputs"][key] for c in a} & {c["inputs"][key] for c in b}
    for rows in splits.values():
        for template in {c["instruction"] for c in rows}:
            assert {c["star_class"] for c in rows if c["instruction"] == template} == {
                "main_sequence",
                "white_dwarf",
                "giant",
            }


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
def test_radius_runtime_scope(tmp_path, settings):
    runtime = Runtime(io.StringIO())
    with pytest.raises(ValueError):
        runtime.command(
            {
                "command": "start",
                "payload": {
                    "task": "radius",
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
