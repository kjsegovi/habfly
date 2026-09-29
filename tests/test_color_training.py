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


def test_stabilization_is_offline_bounded_and_restores_selected_head_and_optimizer(
    trained, tmp_path, monkeypatch
):
    from habfly.spreadsheet import SpreadsheetAdapter
    from habfly.training import color_stabilization as stabilization
    from habfly.training.checkpoints import load_checkpoint

    source, _ = trained
    ordinal = tmp_path / "ordinal"
    refine_color(source, ordinal, notify=lambda _: None)
    parent, _, parent_report = load_color_experiment(ordinal)
    numeric_hash = file_hash(PARENT)

    def deny(*a, **k):
        pytest.fail("Stabilization attempted network or spreadsheet access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    original_cases = stabilization.color_cases

    def only_expert(split, *args):
        assert split == "expert", "Stabilization must not generate new numeric cases"
        return original_cases(split, *args)

    monkeypatch.setattr(stabilization, "color_cases", only_expert)
    output = tmp_path / "stable"
    report = stabilization.stabilize_color(ordinal, output, notify=lambda _: None)
    policy, _, loaded = load_color_experiment(output)
    assert report == loaded
    assert report["optimizer_updates"] == len(report["losses"]) == 8
    assert report["cumulative_color_optimizer_updates"] == 24
    assert torch.isfinite(torch.tensor(report["losses"])).all()
    assert not report["ready_for_final_test"] and not report["browser_eligible"]
    assert len(report["epoch_history"]) == 3
    chosen = max(report["epoch_history"], key=stabilization.selection_key)
    assert report["selected_epoch"] == chosen["epoch"]
    assert report["selected_checkpoint_optimizer_updates"] == chosen["epoch"] * 4
    assert report["discarded_optimizer_updates"] == 8 - chosen["optimizer_updates"]
    assert (
        report["selected_checkpoint_cumulative_color_optimizer_updates"] == 16 + chosen["optimizer_updates"]
    )
    assert frozen_workflow_hash(parent) == frozen_workflow_hash(policy)
    assert all(
        torch.equal(a, dict(policy.color_head.named_buffers())[n])
        for n, a in parent.color_head.named_buffers()
    )
    assert file_hash(ordinal / "training/checkpoint.pt") == parent_report["checkpoint_sha256"]
    assert file_hash(PARENT) == numeric_hash
    for name in ("train.json", "development.json", "calibration.json", "demonstrations.json"):
        assert (output / name).read_bytes() == (ordinal / name).read_bytes()
    for entry in report["epoch_history"]:
        epoch = entry["epoch"]
        path = output / "epochs" / f"epoch-{epoch:03d}" / "training/checkpoint.pt"
        archived, manifest = load_checkpoint(path, policy.graph, content_pack=report["content"])
        assert manifest["evaluation_scores"]["epoch"] == epoch
        assert frozen_workflow_hash(archived) == frozen_workflow_hash(parent)
        assert entry["head_sha256"] == stabilization.head_hash(archived)
        for split in ("train", "development"):
            assert (path.parent.parent / f"{split}-rollouts/report.json").exists()
    saved = torch.load(output / "training/checkpoint.pt", weights_only=True)
    selected = torch.load(
        output / "epochs" / f"epoch-{chosen['epoch']:03d}" / "training/checkpoint.pt", weights_only=True
    )
    assert all(torch.equal(saved["model"][k], v) for k, v in selected["model"].items())
    for state in saved["optimizer"]["state"].values():
        assert float(state["step"]) == 8 + chosen["optimizer_updates"]
    assert all(group["lr"] == 0.003 for group in saved["optimizer"]["param_groups"])
    for key, state in saved["optimizer"]["state"].items():
        assert all(torch.equal(v, selected["optimizer"]["state"][key][n]) for n, v in state.items())
    assert not (output / "final").exists() and not (ordinal / "final").exists()
    with pytest.raises(ValueError, match="one ordinal refinement"):
        stabilization.stabilize_color(output, tmp_path / "recursive")
    with pytest.raises(ValueError, match="one ordinal refinement"):
        stabilization.stabilize_color(source, tmp_path / "linear")
    with pytest.raises(FileExistsError):
        stabilization.stabilize_color(ordinal, output)
    (ordinal / "final").mkdir()
    with pytest.raises(ValueError, match="final test was already opened"):
        stabilization.stabilize_color(ordinal, tmp_path / "after-final")
    # Tampering with selection accounting or an archived candidate fails closed.
    report["selected_epoch"] = 99
    (output / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="selected checkpoint mismatch"):
        load_color_experiment(output)
    report["selected_epoch"] = chosen["epoch"]
    (output / "report.json").write_text(json.dumps(report))
    with (output / "epochs/epoch-000/training/checkpoint.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="epoch checkpoint mismatch"):
        load_color_experiment(output)


def test_boundary_training_preserves_original_and_heldout_data_and_optimizer_lineage(
    trained, tmp_path, monkeypatch
):
    from habfly.spreadsheet import SpreadsheetAdapter
    from habfly.training import color_stabilization
    from habfly.training.color_boundary import all_training_cases, train_color_boundaries

    def deny(*a, **k):
        pytest.fail("Boundary training attempted network or spreadsheet access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    source, _ = trained
    ordinal, stable, output = (tmp_path / name for name in ("ordinal", "stable", "boundary"))
    refine_color(source, ordinal, notify=lambda _: None)
    color_stabilization.stabilize_color(ordinal, stable, notify=lambda _: None)
    parent, _, parent_report = load_color_experiment(stable)
    numeric_hash = file_hash(PARENT)
    original_cases = color_stabilization.color_cases

    def only_expert(split, *args):
        assert split == "expert", "Boundary training must not create new held-out or final cases"
        return original_cases(split, *args)

    monkeypatch.setattr(color_stabilization, "color_cases", only_expert)
    report = train_color_boundaries(stable, output, notify=lambda _: None)
    policy, _, _ = load_color_experiment(output)
    assert len(read(output / "train.json")) == len(read(output / "regression.json")) == 4
    assert len(all_training_cases(output)) == 6
    for split in ("calibration", "development"):
        assert (output / f"{split}.json").read_bytes() == (stable / f"{split}.json").read_bytes()
    assert (output / "regression.json").read_bytes() == (stable / "train.json").read_bytes()
    assert read(output / "train.json") != read(stable / "train.json")
    assert report["content"]["demonstrations_sha256"] != parent_report["content"]["demonstrations_sha256"]
    assert report["content"]["boundary_curriculum"]["new_numeric_cases"] == 2
    assert report["content"]["boundary_curriculum"]["boundary_expert_cases_passed"] == 4
    assert report["optimizer_updates"] == len(report["losses"]) == 8
    assert report["cumulative_color_optimizer_updates"] == 32
    selected_updates = report["selected_checkpoint_optimizer_updates"]
    inherited = parent_report["selected_checkpoint_cumulative_color_optimizer_updates"]
    assert report["selected_checkpoint_cumulative_color_optimizer_updates"] == inherited + selected_updates
    assert torch.isfinite(torch.tensor(report["losses"])).all()
    assert frozen_workflow_hash(policy) == frozen_workflow_hash(parent)
    assert all(
        torch.equal(a, dict(policy.color_head.named_buffers())[n])
        for n, a in parent.color_head.named_buffers()
    )
    saved = torch.load(output / "training/checkpoint.pt", weights_only=True)
    parent_saved = torch.load(stable / "training/checkpoint.pt", weights_only=True)
    for key, state in saved["optimizer"]["state"].items():
        assert (
            float(state["step"]) == float(parent_saved["optimizer"]["state"][key]["step"]) + selected_updates
        )
    assert all(group["lr"] == 0.003 for group in saved["optimizer"]["param_groups"])
    assert report["epoch_history"][0]["regression"]["completed"] == parent_report["train"]["completed"]
    for entry in report["epoch_history"]:
        assert entry["regression"]["episodes"] == 4
    assert report["regression"]["episodes"] == 4
    assert not report["ready_for_final_test"] and not report["browser_eligible"]
    assert not (output / "final").exists() and not (stable / "final").exists()
    assert file_hash(PARENT) == numeric_hash
    assert file_hash(stable / "training/checkpoint.pt") == parent_report["checkpoint_sha256"]
    with pytest.raises(ValueError, match="one stabilized color parent"):
        train_color_boundaries(output, tmp_path / "recursive")
    with pytest.raises(ValueError, match="one stabilized color parent"):
        train_color_boundaries(ordinal, tmp_path / "before-stabilization")
    with pytest.raises(FileExistsError):
        train_color_boundaries(stable, output)
    (stable / "final").mkdir()
    with pytest.raises(ValueError, match="final test was already opened"):
        train_color_boundaries(stable, tmp_path / "after-final")
    (output / "regression.json").write_text("[]")
    with pytest.raises(ValueError, match="original/held-out dataset mismatch"):
        load_color_experiment(output)


def test_selected_input_real_graph_offline_budget_reload_and_provenance(trained, tmp_path, monkeypatch):
    from habfly.runtime import read_trace
    from habfly.spreadsheet import SpreadsheetAdapter
    from habfly.training.color_boundary import train_color_boundaries
    from habfly.training.color_isolation import isolate_color
    from habfly.training.color_stabilization import stabilize_color

    def deny(*a, **k):
        pytest.fail("Selected-input experiment attempted network or Sheets access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    source, _ = trained
    ordinal, stable, boundary, output = (tmp_path / s for s in ("ordinal", "stable", "boundary", "isolated"))
    refine_color(source, ordinal, notify=lambda _: None)
    stabilize_color(ordinal, stable, notify=lambda _: None)
    train_color_boundaries(stable, boundary, notify=lambda _: None)
    parent, _, before = load_color_experiment(boundary)
    numeric = file_hash(PARENT)
    report = isolate_color(boundary, output, notify=lambda _: None)
    policy, reference, reloaded = load_color_experiment(output)
    assert report == reloaded
    policy.requires_grad_(False)
    assert policy.color_input == "selected_graph_v1"
    assert report["optimizer_updates"] == len(report["losses"]) == 8
    assert report["cumulative_color_optimizer_updates"] == 40
    assert torch.isfinite(torch.tensor(report["losses"])).all()
    assert report["invariance_after"]["prediction_changes"] == 0
    assert report["invariance_after"]["max_logit_delta"] == 0
    assert frozen_workflow_hash(policy) == frozen_workflow_hash(parent)
    for name in (
        "train.json",
        "regression.json",
        "calibration.json",
        "development.json",
        "demonstrations.json",
    ):
        assert (output / name).read_bytes() == (boundary / name).read_bytes()
    training_features, _ = refinement_features(
        policy, [record(reference, c)[0] for c in read(output / "train.json")]
    )
    assert torch.equal(training_features.mean(0), policy.color_head.feature_mean)
    assert not torch.equal(parent.color_head.feature_mean, policy.color_head.feature_mean)
    assert report["regression_gate_passed"] == (
        report["regression"]["completed"] >= before["regression"]["completed"]
    )
    assert file_hash(PARENT) == numeric
    assert file_hash(boundary / "training/checkpoint.pt") == before["checkpoint_sha256"]
    assert not (output / "final").exists() and not (boundary / "final").exists()
    for trace in output.rglob("*.jsonl"):
        assert list(read_trace(trace))  # Offline v1 replay accepts both activity paths.
    with pytest.raises(ValueError, match="one legacy boundary parent"):
        isolate_color(output, tmp_path / "recursive")
    with pytest.raises(FileExistsError):
        isolate_color(boundary, output)
    report["content"]["selected_input"]["new_numeric_cases"] = 1
    # Validator independently rejects forged provenance, not just the manifest hash.
    from habfly.training.color_isolation import validate_isolation

    with pytest.raises(ValueError, match="selected-input provenance"):
        validate_isolation(output, policy, reference, report)


def test_ordered_readout_real_graph_bounded_offline_fit_and_frozen_projection(trained, tmp_path, monkeypatch):
    from habfly.spreadsheet import SpreadsheetAdapter
    from habfly.training.color_boundary import train_color_boundaries
    from habfly.training.color_isolation import isolate_color
    from habfly.training.color_ordering import order_color, projection_hash, validate_ordering
    from habfly.training.color_stabilization import stabilize_color

    def deny(*a, **k):
        pytest.fail("Ordering attempted network or Sheets access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(SpreadsheetAdapter, "__init__", deny)
    source, _ = trained
    ordinal, stable, boundary, isolated, output = (
        tmp_path / n for n in ("ordinal", "stable", "boundary", "isolated", "ordered")
    )
    refine_color(source, ordinal, notify=lambda _: None)
    stabilize_color(ordinal, stable, notify=lambda _: None)
    train_color_boundaries(stable, boundary, notify=lambda _: None)
    isolate_color(boundary, isolated, notify=lambda _: None)
    parent, _, before = load_color_experiment(isolated)
    numeric_hash = file_hash(PARENT)
    report = order_color(isolated, output, notify=lambda _: None)
    policy, reference, loaded = load_color_experiment(output)
    assert loaded == report
    assert report["optimizer_updates"] == len(report["losses"]) == 8
    assert report["cumulative_color_optimizer_updates"] == 48
    assert report["closed_form_fits"] == 1 and report["closed_form_fit_retained"]
    assert report["ordering_after"]["score_decreases"] == 0
    assert report["invariance_after"]["max_logit_delta"] == 0
    assert torch.isfinite(torch.tensor(report["losses"])).all()
    config = report["content"]["readout_ordering"]
    assert config["new_numeric_cases"] == 0
    assert config["frozen_projection_sha256"] == projection_hash(policy.color_head)
    assert (
        config["regression_minimum_completed"]
        >= before["content"]["selected_input"]["regression_minimum_completed"]
    )
    assert frozen_workflow_hash(policy) == frozen_workflow_hash(parent)
    for name, tensor in parent.color_head.named_buffers():
        assert torch.equal(tensor, dict(policy.color_head.named_buffers())[name])
    for name in (
        "train.json",
        "regression.json",
        "development.json",
        "calibration.json",
        "demonstrations.json",
    ):
        assert (output / name).read_bytes() == (isolated / name).read_bytes()
    baseline = torch.load(output / "epochs/epoch-000/training/checkpoint.pt", weights_only=True)
    for checkpoint in (output / "epochs").glob("*/training/checkpoint.pt"):
        payload = torch.load(checkpoint, weights_only=True)
        for key in ("color_head.score.weight", "color_head.score.bias"):
            assert torch.equal(payload["model"][key], baseline["model"][key])
        assert sum(len(g["params"]) for g in payload["optimizer"]["param_groups"]) == 3
    assert not (output / "final").exists() and not (isolated / "final").exists()
    assert file_hash(PARENT) == numeric_hash
    assert file_hash(isolated / "training/checkpoint.pt") == before["checkpoint_sha256"]
    with pytest.raises(ValueError, match="one isolated color parent"):
        order_color(output, tmp_path / "recursive")
    with pytest.raises(FileExistsError):
        order_color(isolated, output)
    (isolated / "final").mkdir()
    with pytest.raises(ValueError, match="final test was already opened"):
        order_color(isolated, tmp_path / "after-final")
    report["content"]["readout_ordering"]["closed_form_fits"] = 2
    with pytest.raises(ValueError, match="readout-ordering provenance"):
        validate_ordering(output, policy, reference, report)
