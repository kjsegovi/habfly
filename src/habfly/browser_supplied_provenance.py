"""Owned, historical supplied-input gate and transport provenance.

Private gate artifacts belong under one run/task/digest archive, never a model
observation, event payload or per-star workflow tree. Replay validates recorded
evidence only; it does not execute a policy, expert or new evaluation.
"""

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path

from . import habitability_supplied_inputs as temperature
from . import planet_supplied_inputs as planet
from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_habitability import map_habitability_capture
from .browser_no_planet_workflow import _Evidence, _planet
from .browser_numeric import screen_identity
from .planet_supplied_stellar_source import load_supplied_planet_stellar_inputs, matches_mapping
from .supplied_browser_modes import DERIVED_SUPPLIED_MODE, NON_MAIN_CLASSES, TEMPERATURE_SUPPLIED_MODE

ARCHIVE_MODE = "owned_supplied_input_transfer_evidence_v1"
_PUBLIC_GATE_KEYS = {
    "scope",
    "task",
    "report",
    "identity",
    "scores",
    "report_sha256",
    "source_sha256",
    "transfer_gate_passed",
    "native_browser_enabled",
    "course_acceptance_passed",
    "scientific_verified",
}


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("supplied_provenance_" + reason)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _equal(left, right):
    return _json(left) == _json(right)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    _require(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value), "invalid_hash")
    return value


def _link(value):
    _require(type(value) is dict and set(value) == {"path", "sha256"}, "invalid_source_link")
    name = value["path"]
    _require(
        type(name) is str
        and name
        and not Path(name).is_absolute()
        and all(part not in {".", ".."} for part in name.split("/"))
        and str(Path(name)) == name,
        "invalid_owned_source_path",
    )
    _hash(value["sha256"])
    return value


def _task(task):
    _require(task in {"planet", "temperature"}, "unsupported_task")
    module = planet if task == "planet" else temperature
    pack = module.load_supplied_planet_pack() if task == "planet" else module.load_supplied_temperature_pack()
    return module, pack, module.adapter_manifest()


def _artifact_names(task):
    return {"report.json", "manifest.json", "private-transfer-cases.json"} | {
        f"episode-{i:03d}{suffix}" for i in range(100) for suffix in (".jsonl", "-summary.json")
    }, f"supplied-input-transfer-{task}-v1.reserved.json"


def _gate_shape(gate, task):
    _task(task)
    _require(type(gate) is dict and _PUBLIC_GATE_KEYS <= gate.keys(), "malformed_gate")
    _require(
        gate["scope"] == "supplied_input_transfer_gate"
        and gate["task"] == task
        and gate["transfer_gate_passed"] is True
        and gate.get("historical_provenance_verified") is True
        and type(gate.get("current_sources_verified")) is bool
        and all(
            gate[k] is False
            for k in ("native_browser_enabled", "course_acceptance_passed", "scientific_verified")
        ),
        "unsupported_gate_claims",
    )
    _hash(gate["report_sha256"])
    result = {key: deepcopy(gate[key]) for key in sorted(_PUBLIC_GATE_KEYS)}
    if "source_origins" in gate:
        _require(
            set(gate["source_origins"].values()) == set(gate["source_sha256"]),
            "source_origin_mismatch",
        )
        result["source_sha256"] = {
            original: gate["source_sha256"][actual]
            for original, actual in sorted(gate["source_origins"].items())
        }
    result.update(current_sources_verified=False, historical_provenance_verified=True)
    return result


def _origins(gate, task):
    pins = gate["artifact_sha256"]
    origins = gate.get("artifact_origins", {path: path for path in pins})
    _require(
        type(pins) is dict and type(origins) is dict and len(pins) == len(origins) == 204, "artifact_count"
    )
    _require(set(origins.values()) == set(pins), "artifact_origin_mismatch")
    report_paths = [Path(p) for p in origins if Path(p).name == "report.json"]
    _require(len(report_paths) == 1, "ambiguous_gate_report")
    original_dir = report_paths[0].parent
    expected, reservation = _artifact_names(task)
    _require(
        all(type(p) is str and Path(p).is_absolute() and str(Path(p)) == p for p in origins)
        and {Path(p).name for p in origins if Path(p).parent == original_dir} == expected
        and len([p for p in origins if Path(p).parent != original_dir]) == 1
        and next(Path(p).name for p in origins if Path(p).parent != original_dir) == reservation,
        "unsupported_artifact_paths",
    )
    return original_dir, {
        p: {"actual": origins[p], "sha256": _hash(pins[origins[p]])} for p in sorted(origins)
    }


def archive_supplied_input_transfer_gate(run_history, validated_gate, *, task):
    """Copy an already validated gate once; return only its owned path/hash.

    The caller must obtain validated_gate from the strict transfer validator.
    Byte hashes are rechecked during copying, and the finished private archive
    is independently validated before this link is returned. Partial archives
    are terminal; no missing files are repaired on a later call.
    """
    try:
        book = _Evidence(run_history)
        gate = _gate_shape(validated_gate, task)
        original_dir, artifacts = _origins(validated_gate, task)
        source_origins = validated_gate.get(
            "source_origins", {path: path for path in validated_gate["source_sha256"]}
        )
        _require(
            type(source_origins) is dict
            and 1 <= len(source_origins) <= 128
            and set(source_origins) == set(gate["source_sha256"]),
            "invalid_source_inventory",
        )
        directory = book.path(book.history / "frozen-transfer-gates" / task / gate["report_sha256"])
        record_path = directory / "record.json"
        if directory.exists():
            raw = book.read(record_path)
            link = {"path": str(record_path.relative_to(book.history)), "sha256": _sha(raw)}
            restored = load_archived_supplied_input_transfer_gate(
                book, link, task=task, **_expected(gate["identity"])
            )
            _require(_equal(restored, gate), "existing_archive_changed")
            return link
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "private").mkdir()
        copied, total = {}, 0
        for i, (original, item) in enumerate(artifacts.items()):
            path = Path(item["actual"])
            _require(
                path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)),
                "symlink_gate_artifact",
            )
            _require(path.is_file() and path.stat().st_size <= 32_000_000, "missing_or_large_gate_artifact")
            raw = path.read_bytes()
            total += len(raw)
            _require(total <= 512_000_000 and _sha(raw) == item["sha256"], "gate_artifact_changed")
            relative = f"private/artifact-{i:03d}{path.suffix}"
            with (directory / relative).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            copied[original] = {"path": relative, "sha256": item["sha256"]}
        copied_sources = {}
        for i, (original, actual) in enumerate(sorted(source_origins.items())):
            path = Path(actual)
            _require(
                type(original) is str
                and Path(original).is_absolute()
                and path.is_absolute()
                and not any(p.is_symlink() for p in (path, *path.parents))
                and path.is_file()
                and path.stat().st_size <= 32_000_000,
                "invalid_source_file",
            )
            raw = path.read_bytes()
            total += len(raw)
            _require(total <= 512_000_000 and _sha(raw) == gate["source_sha256"][original], "source_changed")
            relative = f"private/source-{i:03d}.bin"
            with (directory / relative).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            copied_sources[original] = {"path": relative, "sha256": gate["source_sha256"][original]}
        record = {
            "schema_version": 1,
            "mode": ARCHIVE_MODE,
            "task": task,
            "original_directory": str(original_dir),
            "gate": gate,
            "artifacts": copied,
            "sources": copied_sources,
            "current_sources_verified": False,
            "historical_provenance_verified": True,
            "private_evaluation_evidence": True,
            "policy_observation_authorized": False,
            "browser_actions": 0,
            "optimizer_updates": 0,
            "task_completed": False,
        }
        persist_json(record_path, record)
        link = {"path": str(record_path.relative_to(book.history)), "sha256": _sha(record_path.read_bytes())}
        restored = load_archived_supplied_input_transfer_gate(
            book, link, task=task, **_expected(gate["identity"])
        )
        _require(_equal(restored, gate), "archive_revalidation_changed")
        book.unchanged()
        return link
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError):
        raise BrowserSafetyStop("supplied_provenance_archive_failed") from None


def _expected(identity):
    return {key: identity[key] for key in ("checkpoint_sha256", "graph_hash", "pack_hash", "adapter_sha256")}


def load_archived_supplied_input_transfer_gate(
    book, link, *, task, checkpoint_sha256, graph_hash, pack_hash, adapter_sha256
):
    """Validate one owned archive, adopting every consumed leaf and tree."""
    # Lazy import cannot cause evaluation: only the recorded-evidence reader is
    # called. No _materialize/run/model/expert API belongs to this module.
    from .training.supplied_input_transfer import require_supplied_input_transfer_gate

    _link(link)
    path = book.path(book.history / link["path"])
    directory = book.clean(path.parent)
    raw = book.read(path)
    _require(_sha(raw) == link["sha256"], "archive_record_changed")
    record = json.loads(raw)
    gate = _gate_shape(record["gate"], task)
    _require(
        _equal(
            record,
            {
                "schema_version": 1,
                "mode": ARCHIVE_MODE,
                "task": task,
                "original_directory": record["original_directory"],
                "gate": gate,
                "artifacts": record["artifacts"],
                "sources": record["sources"],
                "current_sources_verified": False,
                "historical_provenance_verified": True,
                "private_evaluation_evidence": True,
                "policy_observation_authorized": False,
                "browser_actions": 0,
                "optimizer_updates": 0,
                "task_completed": False,
            },
        )
        and directory == book.history / "frozen-transfer-gates" / task / gate["report_sha256"]
        and path.name == "record.json",
        "invalid_archive_record",
    )
    entries = record["artifacts"]
    _require(type(entries) is dict and len(entries) == 204, "artifact_count")
    paths, pins, total = {}, {}, 0
    expected_tree = {("record.json", "file"), ("private", "directory")}
    for i, (original, item) in enumerate(sorted(entries.items())):
        _require(type(item) is dict and set(item) == {"path", "sha256"}, "invalid_archive_artifact")
        suffix = Path(original).suffix
        relative = f"private/artifact-{i:03d}{suffix}"
        _require(item["path"] == relative and suffix in {".json", ".jsonl"}, "invalid_archive_artifact_path")
        artifact = book.path(directory / relative)
        content = book.read(artifact)
        total += len(content)
        _require(total <= 512_000_000 and _sha(content) == _hash(item["sha256"]), "archive_artifact_changed")
        paths[original], pins[str(artifact)] = str(artifact), item["sha256"]
        expected_tree.add((relative, "file"))
    original_dir, _ = _origins({"artifact_origins": paths, "artifact_sha256": pins}, task)
    _require(str(original_dir) == record["original_directory"], "archive_original_directory_changed")
    entries = record["sources"]
    _require(
        type(entries) is dict and 1 <= len(entries) <= 128 and set(entries) == set(gate["source_sha256"]),
        "invalid_source_inventory",
    )
    source_paths, source_pins = {}, {}
    for i, (original, item) in enumerate(sorted(entries.items())):
        relative = f"private/source-{i:03d}.bin"
        _require(
            type(original) is str
            and Path(original).is_absolute()
            and type(item) is dict
            and set(item) == {"path", "sha256"}
            and item["path"] == relative
            and item["sha256"] == gate["source_sha256"][original],
            "invalid_source_archive_path",
        )
        source = book.path(directory / relative)
        content = book.read(source)
        total += len(content)
        _require(total <= 512_000_000 and _sha(content) == _hash(item["sha256"]), "archived_source_changed")
        source_paths[original], source_pins[str(source)] = str(source), item["sha256"]
        expected_tree.add((relative, "file"))
    tree = book.tree(directory)
    _require(set(tree) == expected_tree, "archive_tree_changed")
    book.closed_trees[directory] = tree
    validated = require_supplied_input_transfer_gate(
        original_dir,
        task=task,
        checkpoint_sha256=_hash(checkpoint_sha256),
        graph_hash=_hash(graph_hash),
        pack_hash=_hash(pack_hash),
        adapter_sha256=_hash(adapter_sha256),
        artifact_paths=paths,
        source_paths=source_paths,
    )
    _require(_equal(_gate_shape(validated, task), gate), "archive_gate_changed")
    _require(_equal(validated["artifact_sha256"], pins), "archive_consumed_files_changed")
    _require(
        _equal(validated["source_sha256"], source_pins)
        and validated["current_sources_verified"] is False
        and validated["historical_provenance_verified"] is True,
        "archive_source_scope_changed",
    )
    book.unchanged()
    return gate


def _adopt_receipt(book, link, actual_class):
    _link(link)
    path = book.path(book.history / link["path"])
    raw = book.read(path)
    _require(_sha(raw) == link["sha256"], "supplied_receipt_changed")
    saved = json.loads(raw)
    receipt = load_supplied_planet_stellar_inputs(
        book.history,
        path,
        expected_sha256=link["sha256"],
        expected_star=saved["star"],
        selected_class=actual_class,
        expected_pack_hash=planet.load_supplied_planet_pack().checksum,
        expected_adapter_sha256=planet.adapter_manifest()["sha256"],
    )
    for name, expected in receipt["source_sha256"].items():
        _require(_sha(book.read(book.history / name)) == expected, "supplied_source_changed")
    for name in receipt["validated_directories"]:
        book.clean(book.history / name)
    for name, value in receipt["directory_trees"].items():
        directory = book.path(book.history / name)
        tree = tuple(tuple(item) for item in value)
        _require(book.tree(directory) == tree, "supplied_source_tree_changed")
        book.closed_trees[directory] = tree
    return receipt


def _validate(book, directory, report, task):
    module, pack, adapter = _task(task)
    mode = DERIVED_SUPPLIED_MODE if task == "planet" else TEMPERATURE_SUPPLIED_MODE
    directory = book.clean(directory)
    _require(_equal(book.json(directory / "report.json"), report), "report_changed")
    provenance = report["provenance"]
    actual_class = provenance["supplied_star_class"]
    _require(
        report.get("scope") == mode
        and actual_class in NON_MAIN_CLASSES
        and provenance.get("classification_source") == "supplied_not_learned"
        and type(provenance.get("optimizer_updates")) is int
        and provenance["optimizer_updates"] == 0
        and type(report.get("optimizer_updates")) is int
        and report["optimizer_updates"] == 0
        and report.get("checkpoint_unchanged") is True
        and report.get("sources_unchanged") is True
        and all(report.get(key) is False for key in ("task_completed", "saved", "submitted")),
        "unsupported_transport_provenance",
    )
    _require(
        (
            task == "planet"
            and report.get("planet_transport_verified") is True
            and report.get("outcome") == "planet_derived_transport_verified"
            and report.get("browser_acceptance_passed") is False
            and report.get("assessment_performed") is False
        )
        or (
            task == "temperature"
            and report.get("equilibrium_transport_verified") is True
            and report.get("outcome") == "equilibrium_transport_verified"
            and report.get("course_acceptance_passed") is False
            and report.get("assessed") is False
        ),
        "incomplete_transport",
    )
    _require(
        provenance["knowledge_pack_hash"] == pack.checksum
        and provenance["supplied_input_adapter_sha256"] == adapter["sha256"]
        and provenance["original_knowledge_pack_hash"] == module.BASE_PACK_HASH
        and provenance["original_scope"] == module.LEGACY_SCOPE
        and pack.checksum != module.BASE_PACK_HASH,
        "pack_or_adapter_changed",
    )
    receipt = _adopt_receipt(book, provenance["supplied_stellar_inputs"], actual_class)
    gate = load_archived_supplied_input_transfer_gate(
        book,
        provenance["supplied_input_transfer_gate"],
        task=task,
        checkpoint_sha256=provenance["checkpoint_sha256"],
        graph_hash=provenance["graph_hash"],
        pack_hash=pack.checksum,
        adapter_sha256=adapter["sha256"],
    )
    identity = gate["identity"]
    for gate_key, provenance_key in (
        ("parent_metadata_sha256", "checkpoint_metadata_sha256"),
        ("parent_content_hash", "original_checkpoint_content_hash"),
        ("parent_pack_hash", "original_knowledge_pack_hash"),
        ("parent_scope", "original_scope"),
    ):
        _require(identity[gate_key] == provenance[provenance_key], "original_checkpoint_identity_changed")
    capture = book.capture(directory / ("native-copies" if task == "planet" else "native-copy") / "initial")
    if task == "planet":
        _require(matches_mapping(receipt, _planet(capture)), "current_supplied_readout_changed")
    else:
        mapped = map_habitability_capture(capture, capture_sha256=screen_identity(capture))
        _require(mapped["star_name"].casefold() == receipt["star"].casefold(), "temperature_star_changed")
    book.unchanged()
    return {
        "star": receipt["star"],
        "supplied_star_class": actual_class,
        "receipt": receipt,
        "provenance": deepcopy(provenance),
    }


def validate_supplied_derived_provenance(book, derived_dir, report):
    """Validate new-scope identities; the caller still validates events/copies."""
    try:
        return _validate(book, derived_dir, report, "planet")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError):
        raise BrowserSafetyStop("supplied_provenance_invalid_derived_sources") from None


def validate_supplied_temperature_provenance(book, temperature_dir, report):
    """Same exact M/R/class receipt, plus independently scoped temperature gate."""
    try:
        return _validate(book, temperature_dir, report, "temperature")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError):
        raise BrowserSafetyStop("supplied_provenance_invalid_temperature_sources") from None
