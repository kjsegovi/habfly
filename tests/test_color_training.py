"""Real-graph bounded color training and compatibility. No browser/network access."""

import io
import json
import socket

import pytest
import torch

from habfly.runtime import Runtime
from habfly.training.color import (
    GRAPH,
    PARENT,
    file_hash,
    frozen_workflow_hash,
    load_color_experiment,
    read,
    record,
    refine_color,
    refinement_features,
    train_color,
)

pytestmark = pytest.mark.skipif(
    not (GRAPH.exists() and PARENT.exists()), reason="Requires local real graph and frozen parent"
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp("color-training") / "smoke"
    from habfly.spreadsheet import SpreadsheetAdapter

    def deny(*args, **kwargs):
        pytest.fail("Color training attempted network or Sheets access")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", deny)
        patch.setattr(socket.socket, "connect_ex", deny)
        patch.setattr(socket, "create_connection", deny)
        patch.setattr(SpreadsheetAdapter, "__init__", deny)
        report = train_color(output, notify=lambda _: None)
    return output, report


def test_real_graph_smoke_budget_losses_reload_and_sealed_final(trained, monkeypatch):
    from habfly.training.color import require_browser_color_gate, test_color

    output, report = trained
    assert report["optimizer_updates"] == 8 and len(report["losses"]) == 8
    assert report["checkpoint_reload_verified"] and report["expert_cases_passed"] == 100
    assert not report["browser_eligible"] and not report["ready_for_final_test"]
    policy, _reference, _ = load_color_experiment(output)
    assert len(policy.graph.body_ids) == 2000 and policy.hidden_size == 16
    assert policy.graph_hash == report["content"]["graph_hash"]
    with pytest.raises(ValueError, match="development gate"):
        test_color(output)
    assert not (output / "final").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        require_browser_color_gate(output)
    with pytest.raises(FileExistsError):
        train_color(output)


def test_runtime_color_is_local_experimental_and_bounded(trained, monkeypatch, tmp_path):
    output, _ = trained

    def deny(*a, **k):
        pytest.fail("Color runtime attempted network")

    monkeypatch.setattr(socket.socket, "connect", deny)
    from habfly.spreadsheet import SpreadsheetAdapter

    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    runtime = Runtime(io.StringIO())
    try:
        runtime.command(
            {
                "command": "start",
                "payload": {
                    "task": "color",
                    "policy": "checkpoint",
                    "graph": str(GRAPH),
                    "dataset": str(output),
                    "checkpoint": str(output / "training/checkpoint.pt"),
                    "seed": 10000000,
                    "paused": True,
                    "artifact_dir": str(tmp_path / "runtime"),
                },
            }
        )
        assert runtime.status == "paused" and runtime.observation.progress["task"] == "color"
        for _ in range(8):
            if runtime.status in {"completed", "stopped"}:
                break
            runtime.tick()
        assert runtime.status in {"completed", "stopped"}
        lines = [json.loads(line) for line in runtime.output.getvalue().splitlines()]
        assert lines[0]["payload"]["experimental_local_checkpoint"]
        assert lines[-1]["payload"]["stage"] == "color"
        assert any(e["event"] == "neural_activity" for e in lines)
    finally:
        runtime.close()


def test_dataset_corruption_rejected_before_loading_model(trained, tmp_path):
    import shutil

    output, _ = trained
    copied = tmp_path / "copied"
    shutil.copytree(output, copied)
    with (copied / "development.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        load_color_experiment(copied)


def test_offline_refinement_reuses_splits_freezes_workflow_and_counts_added_updates(
    trained, tmp_path, monkeypatch
):
    source, source_report = trained
    from habfly.spreadsheet import SpreadsheetAdapter

    def deny(*a, **k):
        pytest.fail("Refinement attempted network or Sheets access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    numeric_hash = file_hash(PARENT)
    output = tmp_path / "refined"
    report = refine_color(source, output, notify=lambda _: None)
    assert report["optimizer_updates"] == 8
    assert report["cumulative_color_optimizer_updates"] == 16
    assert torch.isfinite(torch.tensor(report["losses"])).all()
    assert report["frozen_workflow_verified"] and report["checkpoint_reload_verified"]
    assert not report["ready_for_final_test"] and not report["browser_eligible"]
    assert not (source / "final").exists() and not (output / "final").exists()
    for name in ("train.json", "calibration.json", "development.json", "demonstrations.json"):
        assert (source / name).read_bytes() == (output / name).read_bytes()
    assert file_hash(source / "training/checkpoint.pt") == source_report["checkpoint_sha256"]
    assert file_hash(PARENT) == numeric_hash
    original, reference, _ = load_color_experiment(source)
    original.requires_grad_(False)
    refined, _, _ = load_color_experiment(output)
    assert original.color_readout == "linear_v1" and refined.color_readout == "ordinal_v2"
    assert frozen_workflow_hash(original) == frozen_workflow_hash(refined)
    assert all(n.startswith("color_head.") for n in report["content"]["refinement"]["trainable_parameters"])
    episodes = [record(reference, c)[0] for c in read(source / "train.json")]
    features, _ = refinement_features(original, episodes)
    assert torch.equal(refined.color_head.feature_mean, features.mean(0))
    assert torch.equal(refined.color_head.feature_scale, features.std(0, unbiased=False).clamp_min(1e-4))
    with torch.no_grad():
        state_original = state_refined = None
        for example in episodes[0]:
            a = original([example.observation], state_original)
            b = refined([example.observation], state_refined)
            state_original, state_refined = a.state, b.state
            for name in ("pooled", "state", "action_logits", "target_logits"):
                assert torch.equal(getattr(a, name), getattr(b, name))
    with pytest.raises(ValueError, match="single bounded linear_v1"):
        refine_color(output, tmp_path / "recursive")
    with pytest.raises(FileExistsError):
        refine_color(source, output)
    # A valid old experiment stays loadable, but opened final tests cannot be reused for tuning.
    import shutil

    opened = tmp_path / "opened-final"
    shutil.copytree(source, opened)
    (opened / "final").mkdir()
    with pytest.raises(ValueError, match="final test was already opened"):
        refine_color(opened, tmp_path / "after-final")
