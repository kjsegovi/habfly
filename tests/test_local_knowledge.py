"""Offline stellar tooling, policy-visible decisions, provenance and replay."""

import io
import json
import math
import socket
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from habfly.cli import app
from habfly.config import Profile, calculation_config, load_profile
from habfly.content import calculate
from habfly.contracts import Action, Observation
from habfly.data import make_demo_graph
from habfly.environments.local_stellar import LocalStellarEnv
from habfly.environments.stellar_common import LOCAL_TEMPLATES, make_case
from habfly.knowledge import KnowledgePack, LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy, graph_fingerprint
from habfly.model.policy import calculation_sections
from habfly.runtime import Runtime, read_trace
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint
from habfly.training.stellar import (
    collect_demonstrations,
    evaluate_stellar,
    load_dataset,
    record_episode,
    train_stellar,
    training_content,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Local workflow attempted network or spreadsheet access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", forbidden)
    torch.set_num_threads(1)


def environment(seed=1, split="gate"):
    calculator = LocalCalculator(load_knowledge_pack())
    case = make_case(seed, split)
    case["expected"] = calculator.reference_answers(case["inputs"], case["star_class"])
    env = LocalStellarEnv(calculator, [case])
    env.reset(seed=seed)
    return env


def act(env, key, value=None):
    obs = env.observe()
    target = next(c.id for c in obs.controls if c.id.split(":", 1)[1] == key)
    return env.step(
        Action(
            kind="SELECT" if value is not None else "CLICK",
            target=target,
            value=value,
            observation_revision=obs.revision,
        )
    )


def bind_distance(env, source="current star"):
    act(env, "operation", "distance")
    act(env, "parameter", "parallax")
    ident = next(
        k for k, v in env.case["measurements"].items() if v["kind"] == "parallax" and v["source"] == source
    )
    act(env, "source", ident)
    act(env, "bind")
    act(env, "execute")
    return ident


def test_pack_golden_hash_provenance_and_separate_content():
    pack = load_knowledge_pack()
    assert LocalCalculator(pack).verify()["cases"] == 9
    assert len(pack.operations) == 6 and pack.pending
    assert pack.operation("luminosity").inputs["distance"].quantity == "distance"
    assert pack.operation("lifetime").inputs["mass"].quantity == "mass"
    for section in ("provenance", "references"):
        changed = pack.model_copy(deep=True)
        getattr(changed, section)["note"] = "changed"
        assert changed.checksum != pack.checksum
    changed = pack.model_copy(deep=True)
    changed.operation("distance").constants["parsec_to_ly"] = 3.2
    with pytest.raises(ValueError, match="golden"):
        LocalCalculator(changed).verify()
    assert "terrestrial_required" not in pack.model_dump()


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "x.__class__", "[x][0]", "1e999", "True", "(lambda: 1)()", "undeclared + 1"],
)
def test_restricted_pack_expressions(expression):
    data = load_knowledge_pack().model_dump()
    data["operations"][0]["expression"] = expression
    with pytest.raises(ValueError):
        KnowledgePack.model_validate(data)


@pytest.mark.parametrize(
    "expression,values",
    [
        ("1/0", {}),
        ("(-1)**.5", {}),
        ("2**9", {}),
        ("x", {"x": True}),
        ("x", {"x": float("inf")}),
        ("x", {"x": "1"}),
    ],
)
def test_arithmetic_errors_are_bounded(expression, values):
    with pytest.raises(ValueError):
        calculate(expression, values)


def test_arithmetic_ignores_unreferenced_observation_metadata():
    assert calculate("x * 2", {"x": 3, "label": "visible text", "checked": True}) == 6


@pytest.mark.parametrize(
    "value,error",
    [
        (None, "input_not_finite_number"),
        ("", "input_not_finite_number"),
        (0, "input_must_be_positive"),
        (-1, "input_must_be_positive"),
        (True, "input_not_finite_number"),
        (float("nan"), "input_not_finite_number"),
    ],
)
def test_missing_zero_and_domain(value, error):
    calc = LocalCalculator(load_knowledge_pack())
    result = calc.execute("distance", {"parallax": {"value": value, "unit": "arcsec"}}, "main_sequence")
    assert not result.ok and result.error == error and result.value is None


def test_applicability_units_overflow_and_dependencies():
    calc = LocalCalculator(load_knowledge_pack())
    for op in ("mass", "radius", "lifetime"):
        assert calc.execute(op, {}, "giant").error == "operation_not_applicable"
    assert calc.execute("distance", {}, "unknown").error == "operation_not_applicable"
    assert calc.execute("distance", {}, "main_sequence").error == "missing_or_extra_inputs"
    assert (
        calc.execute("distance", {"parallax": {"value": 1, "unit": "nm"}}, "main_sequence").error
        == "incompatible_unit"
    )
    assert calc.execute("distance", {"parallax": None}, "main_sequence").error == "invalid_binding"
    assert not calc.execute("distance", {"parallax": {"value": 1e-320, "unit": "arcsec"}}, "main_sequence").ok
    data = calc.pack.model_dump()
    data["operations"][0]["inputs"]["parallax"]["quantity"] = "distance"
    with pytest.raises(ValueError, match="cycle"):
        KnowledgePack.model_validate(data)


def test_wrong_bindings_not_repaired_exact_copy_and_staleness():
    env = environment()
    wrong = bind_distance(env, "reference star")
    value = env.results["r1"]["value"]
    assert value == 3.26 / env.case["measurements"][wrong]["value"]
    assert value != env.case["expected"]["distance"]
    assert not env.answers and not env.destination and not env.result
    act(env, "result", "r1")
    act(env, "destination", "distance")
    act(env, "copy")
    assert env.answers["distance"] == value
    # A different operation preserves this completed result.
    act(env, "operation", "temperature")
    assert env.results["r1"]["valid"]
    act(env, "execute")
    assert env.tool_error == "missing_or_extra_inputs" and env.metrics["tool_errors"] == 1
    assert env.metrics["invalid_actions"] == env.metrics["infrastructure_failures"] == 0
    bind_distance(env, "current star")
    assert env.results["r2"]["valid"]
    act(env, "source", wrong)
    act(env, "bind")
    assert not env.results["r2"]["valid"] and env.results["r1"]["valid"]
    _, _, _, _, info = env.step(Action(kind="SELECT", target="0:result", value="r2"))
    assert info["metrics"]["invalid_actions"] == 1
    first, _ = env.reset(seed=1)
    second, _ = env.reset(seed=1)
    assert first == second and not env.results and not env.answers and not env.bindings


def test_same_operation_self_reference_cannot_crash():
    env = environment()
    bind_distance(env)
    act(env, "source", "r1")
    act(env, "bind")
    assert env.tool_error == "stale_source"
    assert not env.results["r1"]["valid"]


def test_infrastructure_failure_is_separate_from_tool_error(tmp_path, monkeypatch):
    env = environment()

    def fail(*args, **kwargs):
        raise OSError("Local fixture failure")

    monkeypatch.setattr(env.adapter, "execute", fail)
    trajectory, report = record_episode(env, 1, event_path=tmp_path / "failure.jsonl")
    assert trajectory and not report["completed"]
    assert report["infrastructure_failures"] == 1 and report["tool_errors"] == 0
    assert report["api_failures"] == 0 and report["invalid_actions"] == 0
    assert read_trace(tmp_path / "failure.jsonl")[-1].payload == report


def test_expert_100_deterministic_cases_and_visible_budget(tmp_path):
    for seed in range(100):
        env = environment(seed)
        obs = env.observe()
        assert "expected" not in obs.model_dump_json()
        assert obs.calculation["reference_card"] is None and not env.bindings
        episode, summary = record_episode(env, seed, event_path=tmp_path / f"{seed}.jsonl")
        assert summary["completed"] and summary["steps"] < 128
        assert summary["invalid_actions"] == summary["tool_errors"] == 0
        assert summary["input_correct"] == summary["input_attempts"]
        assert summary["calculation_correct"] == summary["calculation_attempts"]
        for step in episode:
            sections = calculation_sections(step["observation"]["calculation"])
            assert all(len(s) < 900 for s in sections)
            assert not step["observation"]["spreadsheet"]
        assert read_trace(tmp_path / f"{seed}.jsonl")[-1].payload["completed"]


def test_model_observes_calculation_state_and_finite_gradient():
    env = environment()
    policy = ConnectomePolicy(make_demo_graph(), hidden_size=4)
    first = policy([env.observe()])
    bind_distance(env)
    second = policy([env.observe()])
    assert not torch.equal(first.action_logits, second.action_logits)
    second.action_logits.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    old = Observation.model_validate({"instruction": "old replay", "spreadsheet": {"results": {}}})
    assert not old.calculation


@pytest.mark.skipif(
    not Path("data/processed/graphs-v2/graph-2000").exists(), reason="Requires real local graph"
)
def test_real_2000_graph_with_calculation_observations():
    from habfly.data import load_graph

    env = environment()
    graph = load_graph("data/processed/graphs-v2/graph-2000")
    assert len(graph.body_ids) == 2000
    policy = ConnectomePolicy(graph, hidden_size=16)
    before = policy([env.observe()])
    bind_distance(env)
    after = policy([env.observe()], before.state.detach())
    assert torch.isfinite(after.action_logits).all()
    assert not torch.equal(before.action_logits, after.action_logits)
    after.action_logits.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_replay_fidelity_and_episode_determinism(tmp_path):
    env = environment()
    first, summary = record_episode(env, 1, event_path=tmp_path / "events.jsonl")
    second, again = record_episode(env, 1)
    assert first == second and summary == again
    recorded = read_trace(tmp_path / "events.jsonl")
    assert [e.payload for e in recorded if e.event == "action_result"] == [s["result"] for s in first]
    assert [e.payload for e in recorded if e.event == "observation"][:-1] == [s["observation"] for s in first]


def test_offline_collection_training_evaluation_runtime_replay_and_compatibility(tmp_path, monkeypatch):
    pack = load_knowledge_pack()
    graph = make_demo_graph()
    dataset = tmp_path / "data"
    collect_demonstrations(
        pack,
        dataset,
        {"train": 2, "calibration": 2, "development": 2, "test": 2},
        graph_hash=graph_fingerprint(graph),
    )
    manifest, data = load_dataset(dataset)
    assert len(set(LOCAL_TEMPLATES.values())) == 5
    assert all(c["split"] == split for split, records in data.items() for c in records["cases"])
    with pytest.raises(ValueError, match="backend"):
        train_stellar(
            dataset, graph, tmp_path / "wrong", expected_content={"calculation_mode": "google_sheets"}
        )
    # BC must replay records, never execute formulas.
    with monkeypatch.context() as patch:
        patch.setattr(LocalCalculator, "execute", lambda *a, **kw: pytest.fail("BC recalculated answers"))
        report = train_stellar(
            dataset,
            graph,
            tmp_path / "train",
            epochs=1,
            hidden_size=4,
            expected_content=pack.content_identity(),
        )
    assert report["calculation_mode"] == "local_tool_assisted"
    assert all(math.isfinite(loss) for loss in report["losses"])
    policy, _ = load_checkpoint(
        tmp_path / "train/checkpoint.pt", graph, content_pack=training_content(manifest)
    )
    with pytest.raises(ValueError):
        load_checkpoint(
            tmp_path / "train/checkpoint.pt",
            graph,
            content_pack={"task": "stellar", "calculation_mode": "google_sheets"},
        )
    report = evaluate_stellar(policy, pack, dataset, tmp_path / "evaluation", runs=2)
    assert report["api_failures"] == report["infrastructure_failures"] == 0
    assert report["calculation_mode"] == "local_tool_assisted" and not report["gate_passed"]
    runtime = Runtime(io.StringIO())
    runtime.start(
        {
            "task": "stellar",
            "calculation_backend": "local",
            "dataset": str(dataset),
            "seed": 300000,
            "paused": True,
            "artifact_dir": str(tmp_path / "runtime"),
        }
    )
    runtime.command({"command": "step"})
    assert runtime.env.steps == 1 and runtime.status == "paused"
    runtime.command({"command": "abort"})
    trace = runtime.trace_path
    assert any(e.payload.get("calculation") for e in read_trace(trace) if e.event == "observation")
    monkeypatch.setattr(LocalCalculator, "execute", lambda *a, **kw: pytest.fail("Replay used calculator"))
    runtime.command({"command": "replay", "payload": {"path": str(trace)}})
    while runtime.status == "running":
        runtime.tick()
    assert runtime.status == "completed"
    runtime.close()


def test_profiles_and_readonly_commands_offline():
    local = load_profile("configs/stellar_local_smoke.yaml")
    assert local.backend == "local" and local.epochs == 2 and local.training_episodes == 4
    assert local.hidden_size == 16 and local.seed == 0
    assert isinstance(calculation_config(local), KnowledgePack)
    assert Profile(spreadsheet_config="old.json").backend == "google_sheets"
    assert load_profile("configs/stellar_smoke.yaml").backend == "google_sheets"
    for command in ("inspect", "validate"):
        result = CliRunner().invoke(app, ["knowledge", command])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)
