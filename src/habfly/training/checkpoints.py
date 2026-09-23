"""Portable learned parameters with explicit graph/tokenizer/content provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from habfly.contracts import CheckpointManifest
from habfly.model import CharacterTokenizer, ConnectomePolicy, TopologyFreePolicy, graph_fingerprint


def source_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _source_provenance() -> tuple[str | None, dict]:
    package = Path(__file__).resolve().parents[1]
    repository = package.parents[1]
    commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=normal"],
                            capture_output=True, text=True, check=False)
    lockfile = repository / "uv.lock"
    versions = {}
    for name in ("torch", "numpy", "pyarrow", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    provenance = {
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "code_hashes": {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(package.rglob("*.py"))},
        "dependency_lock_hash": hashlib.sha256(lockfile.read_bytes()).hexdigest() if lockfile.is_file() else None,
        "dependency_versions": versions, "python_version": platform.python_version(),
    }
    return commit.stdout.strip() if commit.returncode == 0 else None, provenance


def save_checkpoint(path: str | Path, policy: ConnectomePolicy, *, stage: str, seed: int,
                    optimizer=None, content_pack: dict | None = None, evaluation: dict | None = None) -> dict:
    path = Path(path)
    if path.exists() or path.with_suffix(path.suffix + ".json").exists():
        raise FileExistsError(f"Checkpoint already exists: {path}; choose a new experiment directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    commit, provenance = _source_provenance()
    provenance["graph_source_kind"] = policy.graph.manifest.get("source_kind", "unspecified")
    provenance["graph_manifest_hash"] = source_hash(policy.graph.manifest)
    if content_pack and content_pack.get("task") == "stellar":
        provenance["stellar"] = content_pack
    manifest = CheckpointManifest.model_validate({
        "schema_version": 1, "model": policy.configuration(), "tokenizer": policy.tokenizer.config(),
        "tokenizer_hash": policy.tokenizer.fingerprint, "graph_hash": policy.graph_hash,
        "graph_size": len(policy.graph.body_ids), "content_pack_hash": source_hash(content_pack) if content_pack else None,
        "training_stage": stage, "seed": seed, "git_commit": commit,
        "evaluation_scores": evaluation or {}, "optimizer_state": optimizer is not None,
        "calibration": policy.calibration, "action_temperature": policy.action_temperature,
        "target_temperature": policy.target_temperature,
        "vision_trained": policy.vision_trained,
        "provenance": provenance,
    }).model_dump(mode="json")
    payload = {"manifest": manifest, "model": policy.state_dict(),
               "optimizer": optimizer.state_dict() if optimizer is not None else None}
    torch.save(payload, path)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_checkpoint(path: str | Path, graph: Any, *, allow_graph_transfer: bool = False,
                    content_pack: dict | None = None, device: str = "cpu") -> tuple[ConnectomePolicy, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    manifest = CheckpointManifest.model_validate(payload["manifest"]).model_dump(mode="json")
    same_graph = graph_fingerprint(graph) == manifest["graph_hash"]
    if not same_graph and not allow_graph_transfer:
        raise ValueError("Checkpoint graph mismatch; graph transfer must be explicitly enabled")
    if content_pack is not None and source_hash(content_pack) != manifest["content_pack_hash"]:
        raise ValueError("Checkpoint content-pack mismatch")
    tokenizer = CharacterTokenizer(**manifest["tokenizer"])
    if tokenizer.fingerprint != manifest["tokenizer_hash"]:
        raise ValueError("Checkpoint tokenizer hash mismatch")
    config = dict(manifest["model"])
    architecture = config.pop("architecture")
    classes = {"ConnectomePolicy": ConnectomePolicy, "TopologyFreePolicy": TopologyFreePolicy}
    if architecture not in classes:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    policy = classes[architecture](graph, tokenizer=tokenizer, **config).to(device)
    policy.load_state_dict(payload["model"], strict=True)
    policy.vision_trained = manifest.get("vision_trained", False)
    saved_code = manifest.get("provenance", {}).get("code_hashes", {})
    package = Path(__file__).resolve().parents[1]
    same_model_code = all(saved_code.get(f"src/habfly/model/{path.name}") == hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (package / "model").glob("*.py"))
    if same_graph and same_model_code:
        policy.action_temperature = float(manifest["action_temperature"])
        policy.target_temperature = float(manifest["target_temperature"])
        policy.calibration = manifest["calibration"]
    else:
        # Confidence calibration is distribution/graph-specific and must be refitted.
        policy.calibration = {"status": "uncalibrated", "reason": "graph_transfer" if not same_graph else "model_source_changed_or_unverified"}
    policy.eval()
    return policy, manifest
