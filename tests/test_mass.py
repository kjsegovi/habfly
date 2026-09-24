import copy
import io
import itertools
import runpy
import socket
from types import SimpleNamespace

import pytest
import torch

from habfly.contracts import Action
from habfly.data import make_demo_graph
from habfly.environments.mass import MassEnv, mass_cases
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy
from habfly.model.measurement_identity import measurement_request
from habfly.model.tool_state import STATE_WIDTH, state_features, task_state_features, visible_sources
from habfly.runtime import Runtime, read_trace
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.chained_workflow import workflow_spec
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.frozen_text import frozen_text_cache
from habfly.training.luminosity import audit_chain, clean, rollout
from habfly.training.stellar import record_episode
from habfly.training.task_state import migrate_task_state
from habfly.training.train import episodes_to_examples, supervised_loss


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Mass task attempted network or Sheets access")

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


def test_independent_mass_golden_and_errors(calculator):
    assert calculator.verify()["cases"] == 9
    # Spreadsheet G2 = F2^(1/3.5): 128 = 4^3.5, so 128 Lsun gives 4 Msun.
    assert calculator.execute(
        "mass", {"luminosity": {"value": 128, "unit": "Lsun"}}, "main_sequence"
    ).value == pytest.approx(4)
    for cls in ("white_dwarf", "giant"):
        assert (
            calculator.execute("mass", {"luminosity": {"value": 1, "unit": "Lsun"}}, cls).error
            == "operation_not_applicable"
        )
    for binding, expected in (
        ({}, "missing_or_extra_inputs"),
        ({"luminosity": {"value": 0, "unit": "Lsun"}}, "input_must_be_positive"),
        ({"luminosity": {"value": -1, "unit": "Lsun"}}, "input_must_be_positive"),
        ({"luminosity": {"value": None, "unit": "Lsun"}}, "input_not_finite_number"),
        ({"luminosity": {"value": float("inf"), "unit": "Lsun"}}, "input_not_finite_number"),
        ({"luminosity": {"value": 1, "unit": "K"}}, "incompatible_unit"),
    ):
        assert calculator.execute("mass", binding, "main_sequence").error == expected


def test_expert_both_branches_replay_and_public_contract(calculator, tmp_path):
    cases = mass_cases("train", 4, calculator, offset=50000)
    report = rollout(None, calculator, cases, tmp_path / "expert", environment=MassEnv)
    assert clean(report, 4, steps=workflow_spec("mass").expected_steps)
    assert [e["steps"] for e in report["episodes"]] == [40, 31, 40, 31]
    assert all(e["mass_applicability_correct"] for e in report["episodes"])
    for case in cases:
        env = MassEnv(calculator, [case], max_steps=64)
        trajectory, _ = record_episode(env, case["seed"])
        events = read_trace(tmp_path / "expert" / f"{case['seed']}.events.jsonl")
        assert next(e.payload for e in events if e.event == "episode_summary")["completed"]
        assert audit_chain(case, trajectory)["mass_luminosity_reuses"] == int(
            case["star_class"] == "main_sequence"
        )
        for row in trajectory:
            obs = row["observation"]
            assert "expected" not in obs["values"] and "stage" not in obs["calculation"]
            assert len(next(c for c in obs["controls"] if c["label"] == "Calculation")["options"]) == 6


def test_wrong_luminosity_is_not_repaired_stale_and_reset(calculator):
    case = mass_cases("train", 1, calculator, offset=50000)[0]
    # Same-unit reference distractor must be calculated as selected, not repaired.
    case["measurements"]["wrong-luminosity"] = {
        "kind": "luminosity",
        "unit": "Lsun",
        "value": 128,
        "source": "reference star",
    }
    env = MassEnv(calculator, [case], max_steps=64)
    env.reset(seed=case["seed"])
    act(env, "operation", "mass")
    act(env, "parameter", "luminosity")
    act(env, "source", "wrong-luminosity")
    act(env, "bind")
    act(env, "execute")
    assert env.results["r1"]["value"] == pytest.approx(4)
    assert visible_sources(env.observe().model_dump())["r1"]["source"] == "reference star"
    act(env, "result", "r1")
    act(env, "destination", "mass")
    act(env, "copy")
    assert env.answers["mass"] == env.results["r1"]["value"]
    other = next(k for k, v in case["measurements"].items() if v["kind"] == "wavelength")
    act(env, "source", other)
    act(env, "bind")
    assert not env.results["r1"]["valid"]
    act(env, "execute")
    assert env.tool_error == "incompatible_unit"
    env.reset(seed=case["seed"])
    assert not env.results and not env.answers and not env.bindings


def test_no_premature_completion_or_inapplicable_mass(calculator):
    cases = mass_cases("train", 4, calculator, offset=50000)
    for case in cases:
        env = MassEnv(calculator, [case], max_steps=64)
        trajectory, _ = record_episode(env, case["seed"])
        env.reset(seed=case["seed"])
        for row in trajectory[:30]:
            env.step(Action.model_validate(row["action"]))
        if case["star_class"] == "main_sequence":
            obs, _, terminated, _, _ = act(env, "check")
            assert not terminated and not obs["progress"]["task_completed"]
        else:
            act(env, "operation", "mass")
            act(env, "parameter", "luminosity")
            act(env, "source", "r2")
            act(env, "bind")
            act(env, "execute")
            assert env.tool_error == "operation_not_applicable" and len(env.results) == 3


def test_mass_gradients_mixed_results_reload(calculator, tmp_path):
    case = mass_cases("train", 1, calculator, offset=50000)[0]
    trajectory, _ = record_episode(MassEnv(calculator, [case], max_steps=64), case["seed"])
    obs = copy.deepcopy(trajectory[32]["observation"])
    control = next(c for c in obs["controls"] if c["label"] == "Measurement or result")
    request = measurement_request(obs, control, include_results=True)
    assert request["requirement"] == {"quantity": "luminosity", "unit": "Lsun"}
    assert len(request["candidates"]) == 9
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
    content = {"scope": "test_only_mass", "required_fields": case["required"]}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, policy, stage="synthetic_test_only", seed=0, content_pack=content)
    restored, _ = load_checkpoint(path, graph, content_pack=content)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in policy.state_dict().items())
    with pytest.raises(ValueError, match="content-pack mismatch"):
        load_checkpoint(path, graph, content_pack={"scope": "temperature"})


def test_splits_separate_and_templates_cover_both_branches(calculator):
    splits = {
        s: mass_cases(s, 16, calculator, offset=50000)
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


def test_v5_assignment_metadata_and_neutral_parent_migration(calculator):
    case = mass_cases("train", 1, calculator, offset=50000)[0]
    trajectory, _ = record_episode(MassEnv(calculator, [case], max_steps=64), case["seed"])
    main = copy.deepcopy(trajectory[30]["observation"])
    giant = copy.deepcopy(main)
    giant["values"]["star_class"] = "giant"
    giant["values"]["required_fields"].remove("mass")
    assert state_features(main) == state_features(giant)
    assert task_state_features(main) != task_state_features(giant)
    numeric = copy.deepcopy(main)
    numeric["values"]["expected"] = {"mass": 999}
    for row in numeric["values"]["measurements"].values():
        row["value"] = 0
    assert task_state_features(numeric) == task_state_features(main)
    assert task_state_features({})[-3:] == [False, False, False]
    graph = make_demo_graph(16)
    settings = {
        "hidden_size": 16,
        "selection_mode": "measurement_result_v3",
        "control_encoding": "semantic_tool_v1",
    }
    parent = ConnectomePolicy(graph, observation_encoding="structured_tool_v4", **settings).eval()
    policy = ConnectomePolicy(graph, observation_encoding="structured_tool_v5", **settings).eval()
    migrate_task_state(parent, policy)
    assert not policy.tool_state_projection.weight[:, STATE_WIDTH:].any()
    with torch.inference_mode():
        for observation in (main, giant):
            a, b = parent([observation]), policy([observation])
            assert torch.allclose(a.state, b.state, atol=1e-6)
            assert torch.allclose(a.action_logits, b.action_logits, atol=1e-6)
            assert torch.allclose(a.target_logits, b.target_logits, atol=1e-6)


@pytest.mark.parametrize("updates, recognition", [(0, 200), (801, 200), (1, -1), (1, 201)])
def test_training_caps_reject_before_creating_artifacts(tmp_path, updates, recognition):
    train = runpy.run_path("scripts/train_luminosity.py")["train"]
    args = SimpleNamespace(
        task="mass", output=tmp_path / "never", updates=updates, recognition_updates=recognition
    )
    with pytest.raises(ValueError, match="capped"):
        train(args, None, None)
    assert not args.output.exists()


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
def test_mass_runtime_scope(tmp_path, settings):
    runtime = Runtime(io.StringIO())
    with pytest.raises(ValueError):
        runtime.command(
            {
                "command": "start",
                "payload": {
                    "task": "mass",
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
