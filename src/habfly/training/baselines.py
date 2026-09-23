"""Topology controls with unchanged learned parameter counts."""

import json
from copy import deepcopy

import numpy as np
import torch


def matched_random_graph(graph, seed: int = 0):
    """Destination permutation within signs preserves in/out degree by sign.

    Parallel edges are retained (as in a weighted multigraph); edge weights stay
    attached to their original source. Renormalize signed raw counts at each
    destination to preserve the biological model's incoming-mass convention.
    """
    result = deepcopy(graph)
    result.edge_dst = np.asarray(graph.edge_dst).copy()
    rng = np.random.default_rng(seed)
    signs = np.sign(graph.edge_weight)
    for sign in np.unique(signs):
        group = np.flatnonzero(signs == sign)
        if len(group) < 2:
            continue
        result.edge_dst[group] = rng.permutation(result.edge_dst[group])
    signed = np.asarray(graph.edge_synapse_count, dtype=np.float64) * signs
    totals = np.bincount(result.edge_dst, weights=np.abs(signed), minlength=len(graph.body_ids))
    result.edge_weight = np.divide(signed, totals[result.edge_dst], out=np.zeros_like(signed), where=totals[result.edge_dst] > 0).astype(np.float32)
    result.manifest = {**graph.manifest, "baseline": "degree_sign_matched_randomization", "randomization_seed": seed}
    return result


def run_baselines(graph, output_dir, *, stage="ground", seeds=(0, 1, 2), epochs=1, count=32, hidden_size=32):
    from habfly.model import ConnectomePolicy, TopologyFreePolicy

    from .train import prepare_output_directory, run_curriculum

    directory = prepare_output_directory(output_dir)
    results = []
    for seed in seeds:
        for name, selected, cls in (
            ("biological", graph, ConnectomePolicy),
            ("degree_sign_randomized", matched_random_graph(graph, seed), ConnectomePolicy),
            ("topology_free", graph, TopologyFreePolicy),
        ):
            torch.manual_seed(seed)
            policy = cls(selected, hidden_size=hidden_size)
            report = run_curriculum(stage, directory / f"{name}-{seed}", selected, seed=seed,
                                    epochs=epochs, count=count, policy=policy)
            results.append({"baseline": name, "seed": seed, "report": report})
    report = {"stage": stage, "results": results,
              "interpretation": "Compare measured scores; no biological-topology benefit is assumed.",
              "parameters_matched": len({r["report"]["parameter_count"] for r in results}) <= 1}
    (directory / "baselines.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
