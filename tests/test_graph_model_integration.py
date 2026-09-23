"""Graph/model contract checks; run directly for opt-in real-data resource smoke.

Default pytest runs use only tiny synthetic graphs. The direct script runs
100 deterministic 2,000-node forwards, one 30,000-node inference, and a tiny
gradient check, producing a machine-readable verification report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from habfly.contracts import Control, Observation
from habfly.data import load_graph, make_demo_graph, save_graph, validate_graph
from habfly.model import ConnectomePolicy
from habfly.training import load_checkpoint, save_checkpoint


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _observation(instruction="Click the star."):
    return Observation(
        instruction=instruction,
        controls=[Control(id="star", label="Star"), Control(id="planet", label="Planet")],
    )


def _finite_output(output):
    return all(
        torch.isfinite(getattr(output, name)).all().item()
        for name in (
            "action_logits",
            "target_logits",
            "typed_value_logits",
            "answer_logits",
            "value",
            "pointer",
            "state",
            "pooled",
        )
    )


def test_instruction_effect_requires_active_graph_path():
    torch.manual_seed(102)
    graph = make_demo_graph(16)
    normal = ConnectomePolicy(graph, hidden_size=8, max_answer_length=4).eval()
    disconnected_graph = deepcopy(graph)
    disconnected_graph.edge_weight[:] = 0
    disconnected = ConnectomePolicy(disconnected_graph, hidden_size=8, max_answer_length=4).eval()
    disconnected.load_state_dict(normal.state_dict())
    with torch.no_grad():
        a = normal([_observation("Click the star.")])
        b = normal([_observation("Type the planet's orbital period.")])
        isolated_a = disconnected([_observation("Click the star.")])
        isolated_b = disconnected([_observation("Type the planet's orbital period.")])
    assert not torch.equal(a.action_logits, b.action_logits)
    assert torch.equal(isolated_a.action_logits, isolated_b.action_logits)
    assert _finite_output(a) and _finite_output(isolated_a)


def test_signed_edges_change_readout_with_identical_learned_parameters():
    torch.manual_seed(103)
    graph = make_demo_graph(16)
    normal = ConnectomePolicy(graph, hidden_size=8, max_answer_length=4).eval()
    changed_graph = deepcopy(graph)
    changed_graph.edge_weight *= -1
    changed = ConnectomePolicy(changed_graph, hidden_size=8, max_answer_length=4).eval()
    changed.load_state_dict(normal.state_dict())
    with torch.no_grad():
        expected = normal([_observation()])
        actual = changed([_observation()])
    assert not torch.allclose(expected.action_logits, actual.action_logits)
    assert not normal.adjacency.requires_grad
    assert "adjacency" not in normal.state_dict()


def test_disk_graph_checkpoint_transfer_and_gradients(tmp_path):
    torch.manual_seed(104)
    original = load_graph(save_graph(make_demo_graph(16), tmp_path / "graph16"))
    larger = load_graph(save_graph(make_demo_graph(32), tmp_path / "graph32"))
    assert np.array_equal(original.body_ids, larger.body_ids[:16])
    policy = ConnectomePolicy(original, hidden_size=8, max_answer_length=4)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, policy, stage="integration_test", seed=104)
    transferred, _ = load_checkpoint(checkpoint, larger, allow_graph_transfer=True)
    assert transferred.calibration["status"] == "uncalibrated"
    assert sum(p.numel() for p in policy.parameters()) == sum(p.numel() for p in transferred.parameters())
    output = transferred([_observation()])
    assert output.state.shape == (1, 32, 8) and _finite_output(output)
    output.action_logits.square().sum().backward()
    for parameter in (
        transferred.embedding.weight,
        transferred.sensory_projection.weight,
        transferred.cell.weight_hh,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def real_graph_smoke(graphs_root: Path, report_path: Path) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(105)
    started = time.perf_counter()
    observations = [_observation()]
    records = []
    for size, repeats in ((2000, 100), (30000, 1)):
        graph = load_graph(graphs_root / f"graph-{size}")
        validation = validate_graph(graph)
        model_started = time.perf_counter()
        model = ConnectomePolicy(graph, hidden_size=32).eval()
        constructed = time.perf_counter()
        with torch.no_grad():
            expected = None
            for _ in range(repeats):
                output = model(observations)
                if not _finite_output(output):
                    raise AssertionError(f"Nonfinite {size}-node model output")
                if expected is None:
                    expected = output.action_logits.clone()
                elif not torch.equal(expected, output.action_logits):
                    raise AssertionError("Repeated forward pass was not deterministic")
            changed = model([_observation("Type the planet's orbital period.")])
            responsive = not torch.equal(expected, changed.action_logits)
            if not responsive:
                raise AssertionError("Real graph model was not instruction-responsive")
        records.append(
            {
                "graph_nodes": size,
                "graph_edges": graph.num_edges,
                "graph_hash": graph.manifest["graph_hash"],
                "validation": validation,
                "hidden_size": 32,
                "propagation_steps": model.propagation_steps,
                "deterministic_repeats": repeats,
                "finite": True,
                "instruction_responsive": responsive,
                "construction_seconds": constructed - model_started,
                "forward_seconds_including_changed_instruction": time.perf_counter() - constructed,
                "parameter_count": sum(p.numel() for p in model.parameters()),
            }
        )
        del model, output, changed, graph
    small = ConnectomePolicy(make_demo_graph(32), hidden_size=8)
    small(observations).action_logits.square().sum().backward()
    gradients = {
        name: bool(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
        )
        for name, parameter in small.named_parameters()
        if name in {"embedding.weight", "sensory_projection.weight", "cell.weight_hh"}
    }
    if not all(gradients.values()):
        raise AssertionError("Small-graph gradients failed")
    repository = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1,
        "purpose": "resource and numerical smoke; not trained task competence",
        "device": "cpu",
        "torch_threads": 1,
        "torch_version": torch.__version__,
        "seed": 105,
        "graphs": records,
        "small_graph_gradients": gradients,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verification_source_sha256": {
            str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (repository / "src/habfly/data", repository / "src/habfly/model")
            for path in sorted(folder.glob("*.py"))
        },
    }
    historical_hashes = {
        2000: "b3cb690fce4cb37d0cfa11d6097fae165b6c25e62bbe461ed794b870eaad37e1",
        30000: "e4f59934d3cf691bd3cd3a322112e0df59e8a7febd8424154006eee309d23726",
    }
    if all(record["graph_hash"] == historical_hashes[record["graph_nodes"]] for record in records):
        report["graphs_v2_build_provenance_supplement"] = {
            "note": "Captured exact builder hash after graph build, before adding provenance fields; artifact manifests preserved.",
            "git_dirty": True,
            "source_code_sha256": {
                "src/habfly/data/graphs.py": "5bb1636fe6df32fe818849e48edbec7a6d6c357fbd5acae83c632d85f2e823b6",
                "src/habfly/data/__init__.py": "813ee05d98e01a3f5cec67bac4f56bc823d3af864e1bf862e2934fcac7da20cb",
            },
            "uv_lock_sha256": "6122d785d18d4ffc83703e2a06efc8f35dcb48d4e3553fa02c21dffe27d0057b",
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-graphs", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(real_graph_smoke(arguments.real_graphs, arguments.report), indent=2))
