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
from habfly.environments.lifetime import LifetimeEnv, lifetime_cases
from habfly.environments.radius import RadiusEnv, radius_cases
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy
from habfly.model.measurement_identity import measurement_request
from habfly.model.tool_state import UNITS, YEAR_UNITS, option_features, task_state_features, visible_sources
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
        raise AssertionError("Lifetime task attempted network or Sheets access")

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


def bindings(mass=1, unit="Msun"):
    return {"mass": {"value": mass, "unit": unit}}


def test_independent_lifetime_golden_and_errors(calculator):
    assert calculator.verify()["cases"] == 9
    # Spreadsheet I2: 10^10 * M^(-2.5). Integer powers, independently simplified.
    for mass, expected in ((1, 10000000000), (4, 312500000), (0.25, 320000000000)):
        result = calculator.execute("lifetime", bindings(mass), "main_sequence")
        assert result.ok and result.unit == "yr" and result.value == pytest.approx(expected)
    for cls in ("white_dwarf", "giant"):
        assert calculator.execute("lifetime", bindings(), cls).error == "operation_not_applicable"
    for inputs, error in (
        ({}, "missing_or_extra_inputs"),
        (bindings(0), "input_must_be_positive"),
        (bindings(-1), "input_must_be_positive"),
        (bindings(None), "input_not_finite_number"),
        (bindings(float("inf")), "input_not_finite_number"),
        (bindings(1, "Lsun"), "incompatible_unit"),
    ):
        assert calculator.execute("lifetime", inputs, "main_sequence").error == error


def test_expert_branches_mass_reuse_and_replay(calculator, tmp_path):
    cases = lifetime_cases("train", 4, calculator, offset=50000)
    report = rollout(None, calculator, cases, tmp_path / "expert", environment=LifetimeEnv)
    assert clean(report, 4, steps=workflow_spec("lifetime").expected_steps)
    assert [e["steps"] for e in report["episodes"]] == [60, 31, 60, 31]
    for case, summary in zip(cases, report["episodes"]):
        assert all(summary[k] for k in workflow_spec("lifetime").applicability_metrics)
        trajectory, _ = record_episode(LifetimeEnv(calculator, [case], max_steps=64), case["seed"])
        assert audit_chain(case, trajectory)["lifetime_mass_reuses"] == int(
            case["star_class"] == "main_sequence"
        )
        events = read_trace(tmp_path / "expert" / f"{case['seed']}.events.jsonl")
        assert next(e.payload for e in events if e.event == "episode_summary")["completed"]
        for row in trajectory:
            obs = row["observation"]
            assert "expected" not in obs["values"] and "stage" not in obs["calculation"]
            assert len(next(c for c in obs["controls"] if c["label"] == "Calculation")["options"]) == 6


def test_wrong_mass_binding_is_not_repaired_and_reset_isolated(calculator):
    case = lifetime_cases("train", 1, calculator, offset=50000)[0]
    env = LifetimeEnv(calculator, [case], max_steps=64)
    trajectory, _ = record_episode(env, case["seed"])
    case["measurements"]["wrong-M"] = {"kind": "mass", "value": 4, "unit": "Msun", "source": "reference star"}
    env.reset(seed=case["seed"])
    for row in trajectory[:50]:
        env.step(Action.model_validate(row["action"]))
    previous = copy.deepcopy(env.results)
    act(env, "operation", "lifetime")
    act(env, "parameter", "mass")
    act(env, "source", "wrong-M")
    act(env, "bind")
    act(env, "execute")
    assert env.results["r6"]["value"] == 312500000
    assert visible_sources(env.observe().model_dump())["r6"]["source"] == "reference star"
    act(env, "result", "r6")
    act(env, "destination", "lifetime")
    act(env, "copy")
    assert env.answers["lifetime"] == env.results["r6"]["value"]
    act(env, "source", "r4")
    act(env, "bind")
    assert not env.results["r6"]["valid"]
    assert {k: env.results[k] for k in previous} == previous
    env.reset(seed=case["seed"])
    assert not env.results and not env.answers and not env.bindings


def test_five_answers_not_completed_and_class_guards(calculator):
    for case in lifetime_cases("train", 4, calculator, offset=50000):
        env = LifetimeEnv(calculator, [case], max_steps=64)
        trajectory, _ = record_episode(env, case["seed"])
        env.reset(seed=case["seed"])
        for row in trajectory[: 50 if case["star_class"] == "main_sequence" else 30]:
            env.step(Action.model_validate(row["action"]))
        if case["star_class"] == "main_sequence":
            obs, _, terminated, _, _ = act(env, "check")
            assert not terminated and not obs["progress"]["task_completed"]
            assert "lifetime" not in obs["values"]["answers"]
        else:
            case["measurements"]["mass-reading"] = {
                "kind": "mass",
                "value": 1,
                "unit": "Msun",
                "source": "current star",
            }
            act(env, "operation", "lifetime")
            act(env, "parameter", "mass")
            act(env, "source", "mass-reading")
            act(env, "bind")
            act(env, "execute")
            assert env.tool_error == "operation_not_applicable" and len(env.results) == 3


def policy(graph, version="structured_tool_v6"):
    return ConnectomePolicy(
        graph,
        hidden_size=16,
        observation_encoding=version,
        selection_mode="measurement_result_v3",
        control_encoding="semantic_tool_v1",
    )


def test_year_encoding_neutral_migration_and_old_compatibility(calculator):
    assert UNITS[-1] == "Gyr" and YEAR_UNITS[-1] == "yr"
    assert not any(option_features({}, "yr"))
    assert option_features({}, "Gyr")[-1]
    assert option_features({}, "yr", years=True)[-1]
    assert not any(option_features({}, "Gyr", years=True))
    graph = make_demo_graph(16)
    parent, child = policy(graph, "structured_tool_v5").eval(), policy(graph).eval()
    original = {k: v.clone() for k, v in parent.state_dict().items()}
    migrate_task_state(parent, child)
    assert all(torch.equal(v, parent.state_dict()[k]) for k, v in original.items())
    assert not child.option_projection.weight[:, -1].any()
    case = radius_cases("train", 1, calculator, offset=50000)[0]
    trajectory, _ = record_episode(RadiusEnv(calculator, [case], max_steps=64), case["seed"])
    with torch.inference_mode():
        for step in trajectory[::10]:
            obs = step["observation"]
            a, b = parent([obs]), child([obs])
            for name in ("state", "action_logits", "target_logits"):
                assert torch.equal(getattr(a, name), getattr(b, name))
            for control in obs["controls"]:
                if control["options"]:
                    assert torch.equal(
                        parent.option_scores(obs, control, a.pooled),
                        child.option_scores(obs, control, b.pooled),
                    )
    with pytest.raises(ValueError, match="Unsupported"):
        migrate_task_state(child, parent)


def test_sources_features_gradients_reload_and_year_learning(calculator, tmp_path):
    case = lifetime_cases("train", 1, calculator, offset=50000)[0]
    trajectory, _ = record_episode(LifetimeEnv(calculator, [case], max_steps=64), case["seed"])
    obs = copy.deepcopy(trajectory[52]["observation"])
    control = next(c for c in obs["controls"] if c["label"] == "Measurement or result")
    request = measurement_request(obs, control, include_results=True)
    assert request["requirement"] == {"quantity": "mass", "unit": "Msun"}
    assert len(request["candidates"]) == 11 and any(c["kind"] == "radius" for c in request["candidates"])
    features = task_state_features(obs)
    for row in obs["calculation"]["results"].values():
        row["value"] = 0
    assert task_state_features(obs) == features
    graph = make_demo_graph(16)
    model = policy(graph)
    model.embedding.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    with frozen_text_cache(model):
        state, losses = None, []
        for example in episodes_to_examples([trajectory]):
            loss, output = supervised_loss(model, [example], state)
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
        gradients = [p.grad for p in getattr(model, module).parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert model.option_projection.weight.grad[:, -1].abs().sum() > 0
    content = {"scope": "test_only_lifetime", "required_fields": case["required"]}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, stage="synthetic_test_only", seed=0, content_pack=content)
    restored, _ = load_checkpoint(path, graph, content_pack=content)
    assert restored.observation_encoding == "structured_tool_v6"
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="content-pack mismatch"):
        load_checkpoint(path, graph, content_pack={"scope": "radius"})


def test_option_only_refinement_freezes_core_and_other_heads(calculator):
    runner = runpy.run_path("scripts/train_luminosity.py")
    model = policy(make_demo_graph(16)).eval()
    prefixes = runner["set_workflow_trainable"](model, "options")
    assert prefixes == ("option_projection.", "option_query.")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    case = lifetime_cases("train", 1, calculator, offset=50000)[0]
    trajectory, _ = record_episode(LifetimeEnv(calculator, [case], max_steps=64), case["seed"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.003)
    with frozen_text_cache(model):
        state, losses = None, []
        for example in episodes_to_examples([trajectory]):
            loss, output = supervised_loss(model, [example], state)
            losses.append(loss)
            state = output.state
        torch.stack(losses).mean().backward()
        assert not state.requires_grad
        optimizer.step()
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed == {"option_projection.weight", "option_query.weight"}


def test_option_only_gate_and_branch_loss_reporting(tmp_path):
    runner = runpy.run_path("scripts/train_luminosity.py")
    args = SimpleNamespace(
        task="lifetime",
        output=tmp_path / "never",
        updates=200,
        recognition_updates=1,
        train_components="options",
    )
    with pytest.raises(ValueError, match="keep recognition frozen"):
        runner["train"](args, None, None)
    assert not args.output.exists()
    cases = [{"star_class": c} for c in ("main_sequence", "white_dwarf", "main_sequence", "giant")]
    summary = runner["cycle_loss_summary"]([99, 99, 10, 1, 20, 3], cases)
    assert summary["mean_loss_last_cycle"] == 8.5
    assert summary["mean_loss_by_class_last_cycle"] == {"main_sequence": 15, "giant": 1, "white_dwarf": 3}


def test_splits_and_templates_separate(calculator):
    splits = {
        s: lifetime_cases(s, 16, calculator, offset=50000)
        for s in ("train", "calibration", "development", "test", "manual")
    }
    for a, b in itertools.combinations(splits.values(), 2):
        for key in ("seed", "case_id", "instruction"):
            assert not {c[key] for c in a} & {c[key] for c in b}
        for key in ("parallax", "flux", "wavelength"):
            assert not {c["inputs"][key] for c in a} & {c["inputs"][key] for c in b}
    for split, rows in splits.items():
        previous = radius_cases(split, 16, calculator, offset=50000)
        for key in ("seed", "case_id", "instruction"):
            assert not {c[key] for c in rows} & {c[key] for c in previous}
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
def test_runtime_scope(tmp_path, settings):
    runtime = Runtime(io.StringIO())
    with pytest.raises(ValueError):
        runtime.command(
            {
                "command": "start",
                "payload": {
                    "task": "lifetime",
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
