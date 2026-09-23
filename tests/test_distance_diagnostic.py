"""The distance diagnostic must distinguish memorization, rollout and generalization."""

import io
import json
import math
import socket

import pytest
from typer.testing import CliRunner

from habfly.cli import app
from habfly.contracts import Action
from habfly.data import make_demo_graph, save_graph
from habfly.environments.distance_diagnostic import TEMPLATES, DistanceDiagnosticEnv, distance_case
from habfly.environments.local_stellar import local_stellar_expert
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.runtime import Runtime, read_trace
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint
from habfly.training.distance_diagnostic import (
    STAGES,
    DistanceDiagnosticConfig,
    closed_loop,
    fit_gate,
    load_distance_config,
    rollout_stages,
    run_distance_diagnostic,
    teacher_forced_stages,
)
from habfly.training.stellar import load_dataset, record_episode


@pytest.fixture(autouse=True)
def no_external_access(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Distance diagnostic attempted network or Sheets access")

    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", fail)


class ExpertPolicy:
    def __init__(self):
        self.calibration = {"status": "uncalibrated", "reason": "scripted_test_double"}

    def act(self, observation, state):
        return local_stellar_expert(observation, load_knowledge_pack()), None, {}


def case_and_env(seed=600000):
    calculator = LocalCalculator(load_knowledge_pack())
    case = distance_case(seed, "train", calculator)
    return case, DistanceDiagnosticEnv(calculator, [case], max_steps=32)


def test_default_budget_is_explicit_and_capped():
    settings = load_distance_config("configs/distance_diagnostic.yaml")
    assert settings == DistanceDiagnosticConfig()
    assert settings.training_cases * settings.epochs * 10 == 800
    for changes in (
        {"epochs": 21},
        {"training_cases": 5},
        {"unseen_cases": 9},
        {"threads": 2},
        {"hidden_size": 32},
        {"seed": 1},
        {"max_steps": 33},
        {"calibration_cases": 1},
        {"spreadsheet_config": "unused"},
        {"checkpoint": "old.pt"},
    ):
        with pytest.raises(ValueError):
            DistanceDiagnosticConfig(**changes)


def test_expert_completes_100_distance_cases_without_hints_or_leakage():
    assert len(set(TEMPLATES.values())) == 4
    orders, source_ids, parallaxes = set(), set(), set()
    for seed in range(600000, 600100):
        case, env = case_and_env(seed)
        first, _ = env.reset(seed=seed)
        assert first == env.reset(seed=seed)[0]
        assert first["values"]["required_fields"] == ["distance"]
        assert len(first["values"]["measurements"]) == 6
        assert len(next(c for c in first["controls"] if c["id"].endswith(":operation"))["options"]) == 6
        assert "expected" not in json.dumps(first) and "stages_reached" not in json.dumps(first)
        assert first["calculation"]["operation"] == "" and not first["calculation"]["bindings"]
        orders.add(tuple(c["label"] for c in first["controls"]))
        source_ids.add(
            next(
                k
                for k, v in case["measurements"].items()
                if v["kind"] == "parallax" and v["source"] == "current star"
            )
        )
        parallaxes.add(case["inputs"]["parallax"])
        trajectory, summary = record_episode(env, seed)
        assert summary["completed"] and summary["steps"] == 10
        assert not summary["invalid_actions"] and not summary["tool_errors"]
        metrics = rollout_stages(case, trajectory)
        assert metrics["stages_reached"] == list(STAGES.values())
        assert metrics["first_unreached_stage"] is None
        assert set(metrics["correct_attempts"].values()) == {1}
    assert len(orders) > 1 and len(source_ids) > 1 and len(parallaxes) == 100


def test_wrong_parallax_is_not_repaired_and_does_not_count_as_correct_execution():
    case, env = case_and_env()
    observation, _ = env.reset(seed=case["seed"])
    wrong = next(
        k
        for k, v in case["measurements"].items()
        if v["kind"] == "parallax" and v["source"] == "reference star"
    )
    actions = [
        ("operation", "distance"),
        ("parameter", "parallax"),
        ("source", wrong),
        ("bind", None),
        ("execute", None),
        ("result", "r1"),
        ("destination", "distance"),
        ("copy", None),
        ("unit_distance", "ly"),
        ("check", None),
    ]
    trajectory = []
    for key, value in actions:
        target = next(c["id"] for c in observation["controls"] if c["id"].endswith(":" + key))
        action = Action(kind="SELECT" if value is not None else "CLICK", target=target, value=value)
        following, _, _, _, info = env.step(action)
        trajectory.append(
            {"observation": observation, "action": action.model_dump(mode="json"), "result": info["result"]}
        )
        observation = following
    result = observation["calculation"]["results"]["r1"]["value"]
    assert result == 3.26 / case["measurements"][wrong]["value"]
    assert observation["values"]["answers"]["distance"] == result != case["expected"]["distance"]
    metrics = rollout_stages(case, trajectory)
    assert metrics["first_unreached_stage"] == "select_source"
    assert (
        "execute_calculation" not in metrics["stages_reached"]
        and "copy_result" not in metrics["stages_reached"]
    )
    assert not env.completed


def test_teacher_forced_metrics_and_value_error_attribution():
    case, env = case_and_env()
    episode, _ = record_episode(env, case["seed"])
    report = teacher_forced_stages(ExpertPolicy(), [episode])
    assert report["examples"] == 10 and report["exact_action_accuracy"] == 1
    assert all(row["exact_accuracy"] == 1 for row in report["stages"].values())

    class WrongSource(ExpertPolicy):
        def act(self, observation, state):
            action, state, diagnostics = super().act(observation, state)
            if action.target.endswith(":source"):
                action.value = next(
                    k
                    for k, v in observation.values["measurements"].items()
                    if v["source"] == "reference star"
                )
            return action, state, diagnostics

    wrong = teacher_forced_stages(WrongSource(), [episode])
    assert wrong["exact_action_accuracy"] == 0.9
    assert wrong["stages"]["select_source"]["target_correct"] == 1
    assert wrong["stages"]["select_source"]["value_correct"] == 0


def test_closed_loop_stage_counts_replay_and_gate_rejects_stop(tmp_path):
    case, env = case_and_env()
    report = closed_loop(ExpertPolicy(), env.adapter, [case], tmp_path / "expert", 32)
    assert report["completed"] == 1 and set(report["stage_reach"].values()) == {1}
    assert fit_gate({"exact_action_accuracy": 1}, report)
    assert not fit_gate({"exact_action_accuracy": 0.9}, report)
    path = tmp_path / "expert/600000.events.jsonl"
    events = read_trace(path)
    trajectory = json.loads((tmp_path / "expert/600000.trajectory.json").read_text())
    assert [e.payload for e in events if e.event == "action_result"] == [s["result"] for s in trajectory]
    runtime = Runtime(io.StringIO())
    runtime.command({"command": "replay", "payload": {"path": str(path)}})
    while runtime.status == "running":
        runtime.tick()
    assert runtime.status == "completed"
    runtime.close()

    class StopPolicy:
        def act(self, observation, state):
            return Action(kind="STOP"), None, {}

    stopped = closed_loop(StopPolicy(), env.adapter, [case], tmp_path / "stop", 32)
    assert stopped["completed"] == 0 and not fit_gate({"exact_action_accuracy": 1}, stopped)
    assert stopped["episodes"][0]["first_unreached_stage"] == "select_calculation"


def small_settings(tmp_path):
    graph = make_demo_graph(16)
    path = save_graph(graph, tmp_path / "graph")
    return graph, DistanceDiagnosticConfig(
        graph=path,
        epochs=1,
        training_cases=1,
        calibration_cases=1,
        development_cases=1,
        unseen_cases=2,
        max_steps=12,
    )


def test_actual_small_graph_training_reload_and_fresh_artifacts(tmp_path):
    graph, settings = small_settings(tmp_path)
    output = tmp_path / "diagnostic"
    report = run_distance_diagnostic(output, settings)
    assert report["training"]["epochs"] == 1 and report["max_optimizer_steps"] == 10
    assert all(math.isfinite(loss) for loss in report["training"]["losses"])
    assert not report["stellar_acceptance_gate_passed"]
    if not report["training_fit_gate_passed"]:
        assert report["unseen"]["status"] == "not_run"
        assert not (output / "unseen").exists()
    else:
        assert report["unseen"]["status"] == "evaluated"
    policy, _ = load_checkpoint(output / "training/checkpoint.pt", graph, content_pack=report["content"])
    assert policy.hidden_size == 16
    with pytest.raises(ValueError, match="content-pack"):
        load_checkpoint(
            output / "training/checkpoint.pt", graph, content_pack=load_knowledge_pack().content_identity()
        )
    before = (output / "report.json").read_bytes()
    with pytest.raises(ValueError, match="Diagnostic datasets"):
        load_dataset(output)
    with pytest.raises(FileExistsError):
        run_distance_diagnostic(output, settings)
    assert (output / "report.json").read_bytes() == before
    assert json.loads((output / "status.json").read_text())["stage"] == "complete"
    cases = [
        c
        for split in ("train", "calibration", "development", "test")
        for c in json.loads((output / f"{split}.json").read_text())["cases"]
    ]
    assert len({c["inputs"]["parallax"] for c in cases}) == len(cases)
    assert all(c["seed"] >= 600000 for c in cases)


def test_success_branch_evaluates_unseen_only_after_seen_passes(tmp_path, monkeypatch):
    import habfly.training.distance_diagnostic as module

    _, settings = small_settings(tmp_path)
    calls = []

    def fake_train(episodes, graph, output, **kwargs):
        calls.append("train")
        assert len(episodes) == 1 and kwargs["epochs"] == 1
        assert len(kwargs["validation_episodes"]) == 2
        assert all(
            e[0]["observation"]["instruction"] != TEMPLATES["test"] for e in kwargs["validation_episodes"]
        )
        return {"epochs": 1, "losses": [0.0], "elapsed_seconds": 0.0, "process_peak_rss_bytes": 0}

    real_closed_loop = module.closed_loop

    def recording_closed_loop(policy, calculator, cases, directory, max_steps):
        calls.append(directory.name)
        return real_closed_loop(policy, calculator, cases, directory, max_steps)

    monkeypatch.setattr(module, "train_behavioral_cloning", fake_train)
    monkeypatch.setattr(module, "load_checkpoint", lambda *args, **kwargs: (ExpertPolicy(), {}))
    monkeypatch.setattr(module, "closed_loop", recording_closed_loop)
    report = run_distance_diagnostic(tmp_path / "success-fixture", settings)
    assert calls == ["train", "seen", "unseen"]
    assert report["distance_gate_passed"] and report["unseen"]["completed"] == 2
    assert not report["stellar_acceptance_gate_passed"]


def test_failure_status_and_no_automatic_retry(tmp_path, monkeypatch):
    import habfly.training.distance_diagnostic as module

    _, settings = small_settings(tmp_path)
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("test failure")

    monkeypatch.setattr(module, "train_behavioral_cloning", fail)
    with pytest.raises(RuntimeError, match="test failure"):
        run_distance_diagnostic(tmp_path / "failed", settings)
    assert calls == [1]
    assert json.loads((tmp_path / "failed/status.json").read_text()) == {
        "stage": "failed",
        "error_type": "RuntimeError",
    }
    assert (tmp_path / "failed/manifest.json").exists()
    assert not (tmp_path / "failed/report.json").exists()


@pytest.mark.parametrize(
    "accuracy,broken,outcome",
    [
        (0.5, False, "training_imitation_failed"),
        (1, False, "training_rollout_failed"),
        (1, True, "infrastructure_failure"),
    ],
)
def test_failed_fit_holds_back_unseen_and_classifies_failure(
    tmp_path, monkeypatch, accuracy, broken, outcome
):
    import habfly.training.distance_diagnostic as module

    _, settings = small_settings(tmp_path)

    class FailingPolicy:
        def act(self, observation, state):
            if broken:
                raise OSError("fixture infrastructure failure")
            return Action(kind="STOP"), None, {}

    monkeypatch.setattr(
        module,
        "train_behavioral_cloning",
        lambda *args, **kwargs: {
            "epochs": 1,
            "losses": [1.0],
            "elapsed_seconds": 0.0,
            "process_peak_rss_bytes": 0,
        },
    )
    monkeypatch.setattr(module, "load_checkpoint", lambda *args, **kwargs: (FailingPolicy(), {}))
    monkeypatch.setattr(
        module, "teacher_forced_stages", lambda *args, **kwargs: {"exact_action_accuracy": accuracy}
    )
    output = tmp_path / "fit-failure"
    report = run_distance_diagnostic(output, settings)
    assert report["outcome"] == outcome and not report["training_fit_gate_passed"]
    assert report["unseen"]["status"] == "not_run" and not (output / "unseen").exists()
    assert "unseen" not in report["teacher_forced"]


def test_cli_dispatch_and_help(tmp_path, monkeypatch):
    import habfly.training.distance_diagnostic as module

    seen = []

    def fake_run(output, settings):
        seen.append((output, settings))
        return {
            "outcome": "training_imitation_failed",
            "training_fit_gate_passed": False,
            "seen": {"completed": 0, "requested_episodes": 4},
            "unseen": {"status": "not_run"},
            "distance_gate_passed": False,
        }

    monkeypatch.setattr(module, "run_distance_diagnostic", fake_run)
    result = CliRunner().invoke(app, ["diagnose", "distance", str(tmp_path / "output")])
    assert result.exit_code == 0, result.output
    assert seen[0][1].epochs == 20 and seen[0][1].training_cases == 4
    assert json.loads(result.stdout)["unseen_status"] == "not_run"
    assert CliRunner().invoke(app, ["diagnose", "distance", "--help"]).exit_code == 0
