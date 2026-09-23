"""Fixed topology graph data; no dense adjacency or whole-edge DataFrame.

The biological polarity mapping is a provisional modeling assumption, not a
claim that transmitter sign is universal across postsynaptic receptor types.
Unknown polarity is retained as zero-weight connectivity, and never supplies a
path in the active-connectivity reachability checks.
"""

from __future__ import annotations

import hashlib
import heapq
import importlib.metadata
import json
import subprocess
import tempfile
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from pyarrow import feather, ipc

from habfly.contracts import GraphManifest

SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "is_sensory",
    "is_visual_projection",
    "is_intrinsic",
    "is_descending",
    "is_efferent",
)
ARRAY_NAMES = (
    "body_ids",
    "edge_src",
    "edge_dst",
    "edge_weight",
    "edge_synapse_count",
    "node_features",
    "sensory_mask",
    "intrinsic_mask",
    "efferent_mask",
)
CHUNK_SIZE = 1_000_000


@dataclass
class GraphData:
    body_ids: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_weight: np.ndarray
    edge_synapse_count: np.ndarray
    node_features: np.ndarray
    sensory_mask: np.ndarray
    intrinsic_mask: np.ndarray
    efferent_mask: np.ndarray
    metadata: list[dict[str, Any]]
    manifest: dict[str, Any]

    @property
    def num_nodes(self) -> int:
        return len(self.body_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edge_src)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    view = memoryview(contiguous).cast("B")
    for start in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[start : start + 8 * 1024 * 1024])
    return digest.hexdigest()


def _graph_hash(graph: GraphData) -> str:
    identities = {
        name: {
            "hash": _array_hash(getattr(graph, name)),
            "dtype": str(getattr(graph, name).dtype),
            "shape": list(getattr(graph, name).shape),
        }
        for name in ARRAY_NAMES
    }
    identities["metadata"] = hashlib.sha256(_json_bytes(graph.metadata)).hexdigest()
    return hashlib.sha256(_json_bytes(identities)).hexdigest()


def _provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=repository,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=repository,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    lock = repository / "uv.lock"
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_code_sha256": {
            str(path.relative_to(repository)): _file_hash(path)
            for path in (Path(__file__).resolve(), Path(__file__).with_name("__init__.py").resolve())
        },
        "uv_lock_sha256": _file_hash(lock) if lock.exists() else None,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pyarrow")},
        "implementation": "habfly-sparse-graph-v2",
    }


def _source_info(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        info = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "rows": sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches)),
            "record_batches": reader.num_record_batches,
            "columns": {field.name: str(field.type) for field in reader.schema},
        }
    if hash_file:
        info["sha256"] = _file_hash(path)
    return info


def inspect_sources(canonical_path: str | Path, edges_path: str | Path) -> dict[str, Any]:
    """Inspect Arrow schemas/counts without materializing the edge table."""
    return {
        "canonical": _source_info(Path(canonical_path)),
        "edges": _source_info(Path(edges_path)),
    }


def _canonical(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    table = feather.read_table(path)
    required = {"body_id", "node_index", "polarity", *FEATURE_NAMES}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"Canonical table is missing columns: {sorted(missing)}")
    body_ids = table["body_id"].combine_chunks().to_numpy(zero_copy_only=False)
    if body_ids.dtype.kind not in "iu" or len(body_ids) == 0:
        raise ValueError("Canonical body IDs must be nonempty integers")
    if np.any(body_ids[1:] <= body_ids[:-1]):
        raise ValueError("Canonical body IDs must be unique and sorted ascending")
    indices = table["node_index"].combine_chunks().to_numpy(zero_copy_only=False)
    if not np.array_equal(indices, np.arange(len(body_ids))):
        raise ValueError("Canonical node indices must be contiguous and agree with body-ID ordering")
    polarity = table["polarity"].combine_chunks().to_numpy(zero_copy_only=False)
    if not np.all(np.isin(polarity, [-1, 0, 1])):
        raise ValueError("Polarity must be -1, 0, or +1")
    features = np.column_stack(
        [table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in FEATURE_NAMES]
    ).astype(np.float32)
    if not np.all(np.isin(features, [0, 1])):
        raise ValueError("Biological role flags must be non-null booleans")
    metadata = []
    selected = [
        name
        for name in (
            "body_id",
            "type",
            "class",
            "superclass",
            "polarity",
            *FEATURE_NAMES,
        )
        if name in table.column_names
    ]
    for row in table.select(selected).to_pylist():
        row.setdefault("type", "unknown")
        row.setdefault("class", "unknown")
        row["polarity_known"] = row["polarity"] != 0
        metadata.append(row)
    return body_ids.astype(np.int64, copy=False), features, metadata


def _mapped_batches(path: Path, body_ids: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Map endpoints with BOTH search bounds and equality checks per record batch."""
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        required = ("body_pre", "body_post", "weight")
        if any(name not in reader.schema.names for name in required):
            raise ValueError("Edge table requires body_pre, body_post, and weight")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            columns = [batch.column(batch.schema.get_field_index(name)) for name in required]
            if any(column.null_count for column in columns):
                raise ValueError(f"Edge batch {batch_index} contains null values")
            pre, post, counts = [column.to_numpy(zero_copy_only=False) for column in columns]
            if any(array.dtype.kind not in "iu" for array in (pre, post, counts)):
                raise ValueError("Edge IDs and raw synapse counts must be integers")
            if np.any(counts <= 0):
                raise ValueError("Raw synapse counts must be positive")
            src = np.searchsorted(body_ids, pre)
            dst = np.searchsorted(body_ids, post)
            valid = (src < len(body_ids)) & (dst < len(body_ids))
            candidates = np.flatnonzero(valid)
            valid[candidates] &= (body_ids[src[candidates]] == pre[candidates]) & (
                body_ids[dst[candidates]] == post[candidates]
            )
            yield (
                src[valid].astype(np.int32),
                dst[valid].astype(np.int32),
                counts[valid].astype(np.int64),
            )


def _eligible_edges(
    path: Path, body_ids: np.ndarray, work: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count first, then allocate exact disk-backed arrays and fill in source order."""
    count = sum(len(src) for src, _, _ in _mapped_batches(path, body_ids))
    arrays = [
        np.lib.format.open_memmap(work / f"{name}.npy", mode="w+", dtype=dtype, shape=(count,))
        for name, dtype in (("src", np.int32), ("dst", np.int32), ("count", np.int64))
    ]
    offset = 0
    for src, dst, counts in _mapped_batches(path, body_ids):
        end = offset + len(src)
        if end > count:
            raise ValueError("Edge source changed between count and fill passes")
        for target, values in zip(arrays, (src, dst, counts), strict=True):
            target[offset:end] = values
        offset = end
    if offset != count:
        raise ValueError("Edge source changed between count and fill passes")
    for array in arrays:
        array.flush()
    return tuple(arrays)


def _adjacency(src: np.ndarray, dst: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(src, kind="stable")
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=offsets[1:])
    return offsets, np.asarray(dst[order], dtype=np.int32)


def _bfs(offsets: np.ndarray, targets: np.ndarray, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(offsets) - 1
    distance = np.full(n, -1, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    queue = deque(int(seed) for seed in seeds)
    distance[seeds] = 0
    while queue:
        node = queue.popleft()
        neighbors = targets[offsets[node] : offsets[node + 1]]
        unseen = np.unique(neighbors[distance[neighbors] < 0])
        distance[unseen] = distance[node] + 1
        parent[unseen] = node
        queue.extend(int(neighbor) for neighbor in unseen)
    return distance, parent


def _rank_nodes(
    src: np.ndarray,
    dst: np.ndarray,
    features: np.ndarray,
    polarity: np.ndarray,
    sizes: tuple[int, ...],
    role_proportions: tuple[float, float, float],
) -> np.ndarray:
    """Grow complete sensor-output paths with deterministic role balancing.

    Every accepted candidate already has a sensory ancestor in the selected
    graph (or is a sensory root). Add its shortest outgoing path atomically,
    so each requested prefix puts every node on an active input-output path.
    Candidates that do not fit a prefix boundary are deferred to the next size.
    Role proportions are soft targets; necessary connectors take precedence.
    """
    n = len(features)
    active = polarity[src] != 0
    active_src, active_dst = np.asarray(src[active]), np.asarray(dst[active])
    forward = _adjacency(active_src, active_dst, n)
    reverse = _adjacency(active_dst, active_src, n)
    del active_src, active_dst, active
    sensory = np.flatnonzero((features[:, 0] + features[:, 1]) > 0)
    outputs = np.flatnonzero((features[:, 3] + features[:, 4]) > 0)
    if not len(sensory) or not len(outputs):
        raise ValueError("Graph requires biological sensory/visual and descending/efferent populations")
    distance, parent = _bfs(*forward, sensory)
    back_distance, next_to_output = _bfs(*reverse, outputs)
    reachable_outputs = outputs[distance[outputs] > 0]
    if not len(reachable_outputs):
        raise ValueError("No nonzero directed sensory-to-output path exists")
    target = min(reachable_outputs, key=lambda node: (distance[node], int(node)))
    path = [int(target)]
    while parent[path[-1]] >= 0:
        path.append(int(parent[path[-1]]))
    ranking = list(reversed(path))
    if len(ranking) > sizes[0]:
        raise ValueError(f"Smallest graph must fit its {len(ranking)}-node sensor-output seed path")
    selected = np.zeros(n, dtype=bool)
    selected[ranking] = True
    queued = selected.copy()
    roles = np.ones(n, dtype=np.int8)
    roles[sensory] = 0
    roles[outputs] = 2
    role_counts = np.bincount(roles[ranking], minlength=3)
    frontier: list[list[tuple[int, int]]] = [[], [], []]

    def enqueue(node: int) -> None:
        if queued[node] or back_distance[node] < 0:
            return
        queued[node] = True
        distance_to_output = int(back_distance[node])
        heapq.heappush(
            frontier[int(roles[node])],
            (distance_to_output, int(node)),
        )

    def expand(node: int) -> None:
        for neighbor in forward[1][forward[0][node] : forward[0][node + 1]]:
            enqueue(int(neighbor))

    for node in ranking:
        expand(node)
    # Additional sensory roots represent alternative input channels.
    for node in sensory:
        enqueue(int(node))
    for boundary in sizes:
        deferred: list[tuple[int, tuple[int, int]]] = []
        last_retry_length = -1
        while len(ranking) < boundary:
            available = [role for role in range(3) if frontier[role]]
            if not available and deferred and last_retry_length != len(ranking):
                # Newly selected tails can shorten a previously deferred path.
                last_retry_length = len(ranking)
                for role, item in deferred:
                    heapq.heappush(frontier[role], item)
                deferred.clear()
                available = [role for role in range(3) if frontier[role]]
            if not available:
                raise ValueError(
                    f"Cannot form an exact {boundary}-node path-preserving graph; "
                    f"only {len(ranking)} nodes fit its complete-path boundary"
                )
            role = min(
                available,
                key=lambda item: (role_counts[item] / role_proportions[item], item),
            )
            item = heapq.heappop(frontier[role])
            node = item[1]
            if selected[node]:
                continue
            path = []
            cursor = node
            while not selected[cursor]:
                path.append(cursor)
                successor = int(next_to_output[cursor])
                if successor < 0:  # Reached an output root.
                    break
                cursor = successor
            if len(path) > boundary - len(ranking):
                deferred.append((role, item))
                continue
            for node in path:
                selected[node] = True
                ranking.append(node)
                role_counts[roles[node]] += 1
            for node in path:
                expand(node)
        for role, item in deferred:
            heapq.heappush(frontier[role], item)
    return np.asarray(ranking, dtype=np.int32)


def _normalize(src: np.ndarray, dst: np.ndarray, counts: np.ndarray, polarity: np.ndarray) -> np.ndarray:
    signed = counts.astype(np.float64) * polarity[src]
    totals = np.bincount(dst, weights=np.abs(signed), minlength=len(polarity))
    return np.divide(signed, totals[dst], out=np.zeros_like(signed), where=totals[dst] > 0).astype(np.float32)


def _selected_graph(
    ranking: np.ndarray,
    body_ids: np.ndarray,
    features: np.ndarray,
    metadata: list[dict[str, Any]],
    src: np.ndarray,
    dst: np.ndarray,
    counts: np.ndarray,
    manifest: dict[str, Any],
) -> GraphData:
    mapping = np.full(len(body_ids), -1, dtype=np.int32)
    mapping[ranking] = np.arange(len(ranking), dtype=np.int32)
    total = 0
    for start in range(0, len(src), CHUNK_SIZE):
        end = start + CHUNK_SIZE
        total += np.count_nonzero((mapping[src[start:end]] >= 0) & (mapping[dst[start:end]] >= 0))
    selected_src = np.empty(total, dtype=np.int32)
    selected_dst = np.empty(total, dtype=np.int32)
    selected_counts = np.empty(total, dtype=np.int64)
    offset = 0
    for start in range(0, len(src), CHUNK_SIZE):
        end = start + CHUNK_SIZE
        mapped_src, mapped_dst = mapping[src[start:end]], mapping[dst[start:end]]
        valid = (mapped_src >= 0) & (mapped_dst >= 0)
        stop = offset + np.count_nonzero(valid)
        selected_src[offset:stop] = mapped_src[valid]
        selected_dst[offset:stop] = mapped_dst[valid]
        selected_counts[offset:stop] = counts[start:end][valid]
        offset = stop
    selected_features = features[ranking]
    selected_metadata = [metadata[int(index)] for index in ranking]
    polarity = np.array([row["polarity"] for row in selected_metadata], dtype=np.int8)
    graph = GraphData(
        body_ids=body_ids[ranking],
        edge_src=selected_src,
        edge_dst=selected_dst,
        edge_weight=_normalize(selected_src, selected_dst, selected_counts, polarity),
        edge_synapse_count=selected_counts,
        node_features=selected_features,
        sensory_mask=np.any(selected_features[:, :2] > 0, axis=1),
        intrinsic_mask=selected_features[:, 2] > 0,
        efferent_mask=np.any(selected_features[:, 3:] > 0, axis=1),
        metadata=selected_metadata,
        manifest=manifest,
    )
    return graph


def build_graphs(
    canonical_path: str | Path,
    edges_path: str | Path,
    output_root: str | Path,
    sizes: tuple[int, ...] = (2000, 5000, 10000, 30000),
    seed: int = 0,
    role_proportions: tuple[float, float, float] = (0.2, 0.7, 0.1),
) -> list[Path]:
    """Create immutable nested graph artifacts; never overwrite an output root.

    Seed is recorded for experiment provenance. Selection itself has no random
    tie breaks: canonical body ID wins equal priority. Prefixes are nested
    within a requested size set; changing boundaries may change connector
    choices. Role proportions target sensory, intrinsic/other, and output
    populations respectively; preserving complete paths takes precedence.
    """
    sizes = tuple(sorted(set(sizes)))
    if not sizes or any(not isinstance(size, int) or isinstance(size, bool) or size < 2 for size in sizes):
        raise ValueError("Graph sizes must be integers of at least two")
    if (
        len(role_proportions) != 3
        or not all(np.isfinite(value) and value > 0 for value in role_proportions)
        or not np.isclose(sum(role_proportions), 1.0)
    ):
        raise ValueError("Role proportions must be three positive finite fractions summing to one")
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite graph output: {root}")
    canonical_path, edges_path = Path(canonical_path), Path(edges_path)
    sources = inspect_sources(canonical_path, edges_path)
    fingerprints = [(path.stat().st_size, path.stat().st_mtime_ns) for path in (canonical_path, edges_path)]
    body_ids, features, metadata = _canonical(canonical_path)
    if sizes[-1] > len(body_ids):
        raise ValueError(f"Requested {sizes[-1]} nodes from only {len(body_ids)} canonical neurons")
    root.mkdir(parents=True)
    paths = []
    with tempfile.TemporaryDirectory(prefix="eligible-", dir=root) as temporary:
        src, dst, counts = _eligible_edges(edges_path, body_ids, Path(temporary))
        polarity = np.array([row["polarity"] for row in metadata], dtype=np.int8)
        ranking = _rank_nodes(src, dst, features, polarity, sizes, role_proportions)
        # Smallest requested artifact must include the complete seed path.
        for size in sizes:
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "source_kind": "biological",
                "sources": sources,
                "seed": seed,
                "provenance": _provenance(),
                "eligible_edges": len(src),
                "selection": {
                    "algorithm": "role-balanced-complete-path-frontier-v2",
                    "ordering": "path seed, role deficit, shortest output connector, canonical index tie-break",
                    "role_targets": dict(
                        zip(
                            (
                                "sensory_visual",
                                "intrinsic_other",
                                "descending_efferent",
                            ),
                            role_proportions,
                            strict=True,
                        )
                    ),
                    "path_policy": "every node sensory-reachable and output-reaching in every requested prefix; connectors override role targets",
                    "nested_sizes": list(sizes),
                },
                "node_feature_names": list(FEATURE_NAMES),
                "polarity_policy": "canonical provisional sign; unknown=0; no active path through zero-sign edges",
                "normalization": "incoming absolute signed synapse mass sums to one where nonzero",
            }
            graph = _selected_graph(ranking[:size], body_ids, features, metadata, src, dst, counts, manifest)
            paths.append(save_graph(graph, root / f"graph-{size}"))
    after = [(path.stat().st_size, path.stat().st_mtime_ns) for path in (canonical_path, edges_path)]
    if after != fingerprints:
        raise ValueError("Source files changed during graph construction; discard this artifact set")
    return paths


def validate_graph(graph_or_path: GraphData | str | Path) -> dict[str, Any]:
    """Raise on malformed arrays, polarity, normalization, or active reachability."""
    graph = (
        load_graph(graph_or_path, validate=False)
        if not isinstance(graph_or_path, GraphData)
        else graph_or_path
    )
    n, e = graph.num_nodes, graph.num_edges
    if n < 2 or len(np.unique(graph.body_ids)) != n or graph.body_ids.dtype != np.int64:
        raise ValueError("Body IDs must be unique int64 values for at least two nodes")
    if graph.node_features.shape != (n, 5) or graph.node_features.dtype != np.float32:
        raise ValueError("Node features must be float32 with shape [nodes, 5]")
    if not np.all(np.isin(graph.node_features, [0, 1])):
        raise ValueError("Role features must be binary")
    for name in ("sensory_mask", "intrinsic_mask", "efferent_mask"):
        array = getattr(graph, name)
        if array.shape != (n,) or array.dtype != bool:
            raise ValueError(f"{name} must be a boolean vector matching node count")
    expected_masks = (
        np.any(graph.node_features[:, :2] > 0, axis=1),
        graph.node_features[:, 2] > 0,
        np.any(graph.node_features[:, 3:] > 0, axis=1),
    )
    if any(
        not np.array_equal(getattr(graph, name), expected)
        for name, expected in zip(
            ("sensory_mask", "intrinsic_mask", "efferent_mask"),
            expected_masks,
            strict=True,
        )
    ):
        raise ValueError("Biological masks must agree with role features")
    for name in ("edge_src", "edge_dst"):
        array = getattr(graph, name)
        if array.shape != (e,) or array.dtype != np.int32 or np.any(array < 0) or np.any(array >= n):
            raise ValueError(f"{name} must contain valid contiguous int32 node indices")
    if (
        graph.edge_synapse_count.shape != (e,)
        or graph.edge_synapse_count.dtype != np.int64
        or np.any(graph.edge_synapse_count <= 0)
    ):
        raise ValueError("Raw synapse counts must be positive int64 values matching edge count")
    if (
        graph.edge_weight.shape != (e,)
        or graph.edge_weight.dtype != np.float32
        or not np.all(np.isfinite(graph.edge_weight))
    ):
        raise ValueError("Edge weights must be finite float32 values matching edge count")
    if len(graph.metadata) != n:
        raise ValueError("Metadata must match node count")
    polarity = np.array([row.get("polarity", 0) for row in graph.metadata])
    if not np.all(np.isin(polarity, [-1, 0, 1])):
        raise ValueError("Metadata polarity must be -1, 0, or +1")
    if not np.array_equal([row["body_id"] for row in graph.metadata], graph.body_ids):
        raise ValueError("Metadata body IDs must match node ordering")
    if not np.array_equal(np.sign(graph.edge_weight), polarity[graph.edge_src]):
        raise ValueError("Edge signs must follow presynaptic polarity, including unknown=zero")
    expected = _normalize(graph.edge_src, graph.edge_dst, graph.edge_synapse_count, polarity)
    if not np.allclose(graph.edge_weight, expected, rtol=1e-6, atol=1e-7):
        raise ValueError("Weights must use incoming absolute synapse normalization")
    active = graph.edge_weight != 0
    active_src, active_dst = graph.edge_src[active], graph.edge_dst[active]
    reached, _ = _bfs(*_adjacency(active_src, active_dst, n), np.flatnonzero(graph.sensory_mask))
    to_output, _ = _bfs(*_adjacency(active_dst, active_src, n), np.flatnonzero(graph.efferent_mask))
    output_count = int(np.count_nonzero(graph.efferent_mask & (reached > 0)))
    if not output_count:
        raise ValueError("No nonzero directed sensory-to-output path exists in selected graph")
    if np.any(reached < 0):
        raise ValueError("Every selected neuron must be reachable from a selected sensory root")
    if np.any(to_output < 0):
        raise ValueError("Every selected neuron must have an active directed path to a selected output")
    digest = _graph_hash(graph)
    if graph.manifest.get("graph_hash") not in (None, digest):
        raise ValueError("Graph content hash does not match manifest")
    return {
        "valid": True,
        "num_nodes": n,
        "num_edges": e,
        "active_edges": int(np.count_nonzero(active)),
        "sensory_reachable_nodes": int(np.count_nonzero(reached >= 0)),
        "nodes_reaching_outputs": int(np.count_nonzero(to_output >= 0)),
        "reachable_output_nodes": output_count,
        "unknown_polarity_nodes": int(np.count_nonzero(polarity == 0)),
        "graph_hash": digest,
    }


def save_graph(graph: GraphData, path: str | Path) -> Path:
    validation = validate_graph(graph)
    manifest = {
        **graph.manifest,
        "schema_version": SCHEMA_VERSION,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "graph_hash": validation["graph_hash"],
        "arrays": {},
        "metadata_sha256": "pending",
        "validation": validation,
    }
    # Reject malformed provenance/schema before creating any output files.
    GraphManifest.model_validate(manifest, strict=True)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for name in ARRAY_NAMES:
        array = getattr(graph, name)
        filename = path / f"{name}.npy"
        np.save(filename, array, allow_pickle=False)
        arrays[name] = {
            "file": filename.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _file_hash(filename),
        }
    metadata_path = path / "metadata.json"
    metadata_path.write_bytes(_json_bytes(graph.metadata) + b"\n")
    manifest.update(arrays=arrays, metadata_sha256=_file_hash(metadata_path))
    graph.manifest = GraphManifest.model_validate(manifest, strict=True).model_dump(
        mode="json", exclude_unset=True
    )
    (path / "graph_manifest.json").write_bytes(_json_bytes(graph.manifest) + b"\n")
    return path


def load_graph(path: str | Path, *, validate: bool = True) -> GraphData:
    path = Path(path)
    manifest = json.loads((path / "graph_manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported graph schema version")
    typed_manifest = GraphManifest.model_validate(manifest, strict=True)
    if set(typed_manifest.arrays) != set(ARRAY_NAMES):
        raise ValueError("Graph manifest must describe exactly the required graph arrays")
    arrays = {}
    for name in ARRAY_NAMES:
        description = manifest["arrays"][name]
        if description["file"] != f"{name}.npy":
            raise ValueError("Graph manifest contains an invalid array filename")
        filename = path / description["file"]
        if _file_hash(filename) != description["sha256"]:
            raise ValueError(f"Array checksum mismatch: {name}")
        arrays[name] = np.load(filename, mmap_mode="r", allow_pickle=False)
        if (
            list(arrays[name].shape) != description["shape"]
            or str(arrays[name].dtype) != description["dtype"]
        ):
            raise ValueError(f"Array schema mismatch: {name}")
    if _file_hash(path / "metadata.json") != manifest["metadata_sha256"]:
        raise ValueError("Metadata checksum mismatch")
    graph = GraphData(
        **arrays,
        metadata=json.loads((path / "metadata.json").read_text()),
        manifest=manifest,
    )
    if graph.num_nodes != manifest["num_nodes"] or graph.num_edges != manifest["num_edges"]:
        raise ValueError("Graph dimensions do not match manifest")
    if validate:
        validate_graph(graph)
    return graph


def make_demo_graph(n: int = 32, seed: int = 0) -> GraphData:
    """Small synthetic recurrent graph, explicitly labeled non-biological."""
    if n < 3:
        raise ValueError("Demo graph needs at least three nodes")
    rng = np.random.default_rng(seed)
    features = np.zeros((n, 5), dtype=np.float32)
    features[0, 0] = 1
    features[0, 1] = 1
    features[1:-1, 2] = 1
    features[-1, 3:] = 1
    polarity = np.where(np.arange(n) % 4 == 2, -1, 1).astype(np.int8)
    metadata = [
        dict(
            body_id=int(1000 + index),
            type="synthetic",
            **{"class": "demo"},
            polarity=int(polarity[index]),
            polarity_known=True,
            **{name: bool(features[index, i]) for i, name in enumerate(FEATURE_NAMES)},
        )
        for index in range(n)
    ]
    # A chain guarantees all nodes are reachable; a two-edge shortcut also
    # exposes sensory input to the readout during short smoke-test rollouts.
    src = np.concatenate([np.arange(n - 1), np.arange(n)]).astype(np.int32)
    dst = np.concatenate([np.arange(1, n), np.arange(n)]).astype(np.int32)
    if n > 3:
        src = np.append(src, np.int32(1))
        dst = np.append(dst, np.int32(n - 1))
    counts = rng.integers(1, 8, size=len(src), dtype=np.int64)
    graph = _selected_graph(
        np.arange(n, dtype=np.int32),
        np.arange(1000, 1000 + n, dtype=np.int64),
        features,
        metadata,
        src,
        dst,
        counts,
        {
            "schema_version": SCHEMA_VERSION,
            "source_kind": "synthetic-demo",
            "seed": seed,
            "node_feature_names": list(FEATURE_NAMES),
            "polarity_policy": "synthetic +1/-1 signs",
            "provenance": _provenance(),
        },
    )
    graph.manifest["graph_hash"] = _graph_hash(graph)
    graph.manifest["num_nodes"] = n
    graph.manifest["num_edges"] = len(src)
    graph.manifest["validation"] = validate_graph(graph)
    return graph
