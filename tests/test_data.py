import json
from copy import deepcopy

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import feather

from habfly.data import (
    build_graphs,
    inspect_sources,
    load_graph,
    make_demo_graph,
    save_graph,
    validate_graph,
)
from habfly.data.graphs import _rank_nodes


def sources(tmp_path):
    canonical = tmp_path / "canonical.feather"
    edges = tmp_path / "edges.feather"
    n = 8
    feather.write_feather(
        pa.table(
            {
                "body_id": np.arange(10, 10 + n, dtype=np.int64),
                "node_index": np.arange(n, dtype=np.int32),
                "type": ["fixture"] * n,
                "class": ["test"] * n,
                "polarity": [1, -1, 1, 1, 1, 1, 1, 0],
                "is_sensory": [True] + [False] * (n - 1),
                "is_visual_projection": [False] * n,
                "is_intrinsic": [False] + [True] * (n - 2) + [False],
                "is_descending": [False, True] + [False] * (n - 2),
                "is_efferent": [False] * (n - 1) + [True],
            }
        ),
        canonical,
    )
    # Direct path to output 11, then chain through all other nodes. Backlinks
    # preserve output paths at each requested prefix. Unknown 17 is terminal.
    feather.write_feather(
        pa.table(
            {
                "body_pre": [
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    8,
                    99,
                    10,
                ],
                "body_post": [
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    11,
                    11,
                    11,
                    11,
                    11,
                    10,
                    10,
                    10,
                    99,
                ],
                "weight": [3, 2, 1, 4, 1, 2, 3, 1, 1, 1, 1, 1, 4, 1, 1, 1],
            }
        ),
        edges,
        chunksize=3,
    )
    return canonical, edges


def test_stream_build_nested_deterministic_and_round_trip(tmp_path):
    canonical, edges = sources(tmp_path)
    report = inspect_sources(canonical, edges)
    assert report["canonical"]["rows"] == 8
    assert report["edges"]["rows"] == 16
    first = build_graphs(canonical, edges, tmp_path / "a", sizes=(2, 4, 8))
    second = build_graphs(canonical, edges, tmp_path / "b", sizes=(2, 4, 8))
    graphs = [load_graph(path) for path in first]
    for path_a, path_b, graph in zip(first, second, graphs, strict=True):
        assert (path_a / "graph_manifest.json").read_bytes() == (path_b / "graph_manifest.json").read_bytes()
        assert validate_graph(graph)["valid"]
        assert graph.manifest["eligible_edges"] == 13
        assert validate_graph(graph)["nodes_reaching_outputs"] == graph.num_nodes
    assert np.array_equal(graphs[0].body_ids, graphs[2].body_ids[:2])
    assert np.array_equal(graphs[1].body_ids, graphs[2].body_ids[:4])
    assert graphs[2].edge_weight[-1] == 0
    assert np.any(graphs[2].edge_weight < 0)
    assert validate_graph(graphs[2])["unknown_polarity_nodes"] == 1
    with pytest.raises(FileExistsError):
        build_graphs(canonical, edges, tmp_path / "a", sizes=(2,))


def test_demo_graph_and_manifest_integrity(tmp_path):
    graph = make_demo_graph()
    assert graph.node_features.shape == (32, 5)
    path = save_graph(graph, tmp_path / "graph")
    assert validate_graph(path)["sensory_reachable_nodes"] == 32
    with pytest.raises(FileExistsError):
        save_graph(graph, path)
    data = np.load(path / "edge_weight.npy")
    data[0] *= -1
    np.save(path / "edge_weight.npy", data)
    with pytest.raises(ValueError, match="checksum"):
        load_graph(path)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda g: g.edge_dst.__setitem__(0, 1000), "indices"),
        (lambda g: g.edge_weight.__setitem__(0, float("nan")), "finite"),
        (lambda g: g.edge_weight.__setitem__(0, -g.edge_weight[0]), "signs"),
        (lambda g: g.sensory_mask.__setitem__(0, False), "masks"),
        (lambda g: g.edge_synapse_count.__setitem__(0, 0), "positive"),
    ],
)
def test_invalid_graphs_rejected(mutation, error):
    graph = deepcopy(make_demo_graph(8))
    mutation(graph)
    with pytest.raises(ValueError, match=error):
        validate_graph(graph)


def test_disconnected_active_path_rejected(tmp_path):
    canonical, edges = sources(tmp_path)
    table = feather.read_table(canonical)
    table = table.set_column(table.column_names.index("polarity"), "polarity", pa.array([0] * 8))
    feather.write_feather(table, canonical)
    with pytest.raises(ValueError, match="No nonzero directed"):
        build_graphs(canonical, edges, tmp_path / "bad", sizes=(2,))


def test_unsorted_canonical_rejected(tmp_path):
    canonical, edges = sources(tmp_path)
    table = feather.read_table(canonical)
    table = table.set_column(
        table.column_names.index("body_id"),
        "body_id",
        pa.array([11, 10, 12, 13, 14, 15, 16, 17]),
    )
    feather.write_feather(table, canonical)
    with pytest.raises(ValueError, match="sorted"):
        build_graphs(canonical, edges, tmp_path / "bad", sizes=(2,))


def test_manifest_path_escape_rejected(tmp_path):
    path = save_graph(make_demo_graph(), tmp_path / "graph")
    manifest = json.loads((path / "graph_manifest.json").read_text())
    manifest["arrays"]["body_ids"]["file"] = "../../canonical.npy"
    (path / "graph_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filename"):
        load_graph(path)


def test_searchsorted_does_not_admit_missing_in_range_ids(tmp_path):
    canonical, edges = sources(tmp_path)
    table = feather.read_table(canonical).slice(0, 2)
    table = table.set_column(table.column_names.index("body_id"), "body_id", pa.array([10, 20]))
    feather.write_feather(table, canonical)
    feather.write_feather(
        pa.table(
            {
                "body_pre": [10, 15, 10, 99],
                "body_post": [20, 20, 15, 20],
                "weight": [1, 1, 1, 1],
            }
        ),
        edges,
        chunksize=2,
    )
    path = build_graphs(canonical, edges, tmp_path / "sparse-ids", sizes=(2,))[0]
    graph = load_graph(path)
    assert graph.num_edges == 1
    assert graph.manifest["eligible_edges"] == 1
    assert graph.body_ids.tolist() == [10, 20]


def test_balanced_ranking_retains_connectors_and_excludes_dead_ends():
    features = np.zeros((61, 5), dtype=np.float32)
    features[:12, 0] = 1
    features[12:54, 2] = 1
    features[54:60, 4] = 1
    features[60, 2] = 1  # Reachable from input, but cannot reach any output.
    pairs = [(sensor, intrinsic) for sensor in range(12) for intrinsic in range(12, 54)]
    pairs += [(intrinsic, output) for intrinsic in range(12, 54) for output in range(54, 60)]
    pairs.append((0, 60))
    src, dst = np.asarray(pairs, dtype=np.int32).T
    ranking = _rank_nodes(src, dst, features, np.ones(61, dtype=np.int8), (20, 60), (0.2, 0.7, 0.1))
    assert 60 not in ranking
    assert features[ranking[:20]].sum(0).tolist() == [4, 0, 14, 0, 2]
    assert features[ranking].sum(0).tolist() == [12, 0, 42, 0, 6]
    assert np.array_equal(
        ranking, _rank_nodes(src, dst, features, np.ones(61, dtype=np.int8), (20, 60), (0.2, 0.7, 0.1))
    )


def test_validation_rejects_a_sensory_reachable_dead_end():
    graph = deepcopy(make_demo_graph(4))
    # Keep input->output through shortcut 1->3, but strand intrinsic neuron 2.
    keep = ~((graph.edge_src == 2) & (graph.edge_dst == 3))
    graph.edge_src = graph.edge_src[keep]
    graph.edge_dst = graph.edge_dst[keep]
    graph.edge_synapse_count = graph.edge_synapse_count[keep]
    from habfly.data.graphs import _normalize

    graph.edge_weight = _normalize(
        graph.edge_src,
        graph.edge_dst,
        graph.edge_synapse_count,
        np.array([row["polarity"] for row in graph.metadata]),
    )
    with pytest.raises(ValueError, match="Every selected neuron must have"):
        validate_graph(graph)


@pytest.mark.parametrize(
    "field,value", [("provenance", "not-an-object"), ("seed", "123"), ("unknown_field", True)]
)
def test_save_rejects_malformed_typed_manifest_before_writing(tmp_path, field, value):
    graph = make_demo_graph(8)
    graph.manifest[field] = value
    with pytest.raises(ValueError):
        save_graph(graph, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.update(num_nodes="32"),
        lambda manifest: manifest.update(num_edges=-1),
        lambda manifest: manifest["arrays"]["body_ids"].update(shape="32"),
        lambda manifest: manifest["arrays"].pop("body_ids"),
    ],
)
def test_load_rejects_malformed_typed_manifest(tmp_path, mutation):
    path = save_graph(make_demo_graph(), tmp_path / "graph")
    manifest_path = path / "graph_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_graph(path)
