"""Read-only launch-source checks, not model loading or browser readiness.

Checkpoint bytes are hashed but never deserialized. Metadata/gate-report
consistency cannot establish that those bytes contain the stated architecture;
the existing frozen component loaders must still perform that check at use.
Legacy checks open no cases or trajectories. Explicit supplied-input checks
also validate existing recorded transfer proof; they never generate cases,
run policies, access credentials, launch browsers or write output files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import autonomous_planet
from .color_reference import COLOR_LABELS, load_color_reference, validate_color_reference
from .contracts import CheckpointManifest, GraphManifest
from .data.graphs import ARRAY_NAMES, load_graph
from .habitability_knowledge import DEFAULT_HABITABILITY_PACK, HabitabilityCalculator
from .knowledge import DEFAULT_PACK, LocalCalculator, load_knowledge_pack
from .model import CharacterTokenizer, graph_fingerprint
from .planet_knowledge import DEFAULT_PLANET_PACK, PlanetCalculator
from .stellar_reference import load_hr_reference, validate_hr_reference_source
from .training.chained_workflow import workflow_spec
from .training.checkpoints import source_hash
from .training.habitability_evaluation import require_frozen_final_gate as require_habitability_gate
from .training.planet_evaluation import require_frozen_final_gate as require_planet_gate

_ERRORS = ("invalid_actions", "tool_errors", "api_failures", "infrastructure_failures")
_PACKAGE = Path(__file__).parent


class AutonomousValidationError(ValueError):
    """A fixed-code local-source failure, never raw configuration or file data."""


def _require(value, reason):
    if not value:
        raise AutonomousValidationError("autonomous_validation_" + reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate_source_key")
            result[key] = value
        return result

    def reject(_):
        raise AutonomousValidationError("autonomous_validation_nonfinite_json")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    _require(isinstance(value, dict), "source_object_required")
    return value


class _Sources:
    def __init__(self):
        self.rows = {}

    def file(self, role, source, *, limit=64_000_000, read=False):
        path = Path(source).absolute()
        _require(
            path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "missing_or_symlinked_source",
        )
        _require(0 < path.stat().st_size <= limit, "invalid_source_size")
        digest, count, chunks = hashlib.sha256(), 0, []
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                count += len(chunk)
                _require(count <= limit, "invalid_source_size")
                digest.update(chunk)
                if read:
                    chunks.append(chunk)
        row = {"path": str(path), "sha256": digest.hexdigest(), "bytes": count}
        _require(role not in self.rows or self.rows[role] == row, "source_changed")
        self.rows[role] = row
        return b"".join(chunks) if read else row["sha256"]

    def json(self, role, source):
        return _json(self.file(role, source, limit=16_000_000, read=True))

    def unchanged(self):
        for role, row in list(self.rows.items()):
            self.file(role, row["path"])


def _zero(value):
    return type(value) is int and value == 0


def _graph(book, path):
    manifest = book.json("graph.manifest", path / "graph_manifest.json")
    GraphManifest.model_validate(manifest, strict=True)
    _require(
        manifest["num_nodes"] == 2000
        and manifest["source_kind"] == "biological"
        and set(manifest["arrays"]) == set(ARRAY_NAMES),
        "graph_scope_mismatch",
    )
    for name in ARRAY_NAMES:
        record = manifest["arrays"][name]
        _require(record["file"] == f"{name}.npy", "graph_array_filename")
        _require(book.file("graph." + name, path / record["file"]) == record["sha256"], "graph_array_hash")
    _require(
        book.file("graph.metadata", path / "metadata.json") == manifest["metadata_sha256"],
        "graph_metadata_hash",
    )
    graph = load_graph(path)
    return {
        "nodes": graph.num_nodes,
        "edges": graph.num_edges,
        "source_kind": graph.manifest["source_kind"],
        "artifact_graph_hash": graph.manifest["graph_hash"],
        "model_graph_fingerprint": graph_fingerprint(graph),
        "arrays_and_invariants_verified": True,
    }


def _checkpoint(book, role, path, graph, encoding, control_encoding, content):
    checksum = book.file(role + ".checkpoint", path)
    metadata = book.json(role + ".checkpoint_manifest", path.with_suffix(path.suffix + ".json"))
    CheckpointManifest.model_validate(metadata, strict=True)
    model = metadata["model"]
    _require(
        type(metadata["graph_size"]) is int
        and metadata["graph_size"] == 2000
        and metadata["graph_hash"] == graph["model_graph_fingerprint"]
        and model.get("architecture") == "ConnectomePolicy"
        and type(model.get("hidden_size")) is int
        and model["hidden_size"] == 16
        and model.get("observation_encoding") == encoding
        and model.get("control_encoding") == control_encoding
        and model.get("selection_mode") == "measurement_result_v3"
        and metadata["content_pack_hash"] == source_hash(content),
        "checkpoint_manifest_compatibility",
    )
    tokenizer = CharacterTokenizer(**metadata["tokenizer"])
    _require(tokenizer.fingerprint == metadata["tokenizer_hash"], "tokenizer_hash")
    return metadata, {
        "checkpoint_sha256": checksum,
        "checkpoint_manifest_sha256": book.rows[role + ".checkpoint_manifest"]["sha256"],
        "graph_fingerprint": metadata["graph_hash"],
        "content_pack_hash": metadata["content_pack_hash"],
        "encoding": encoding,
        "hidden_size": 16,
        "manifest_compatibility_verified": True,
        "weight_payload_deserialized": False,
        "weight_manifest_agreement_verified": False,
        "complete_promotion_chain_verified": False,
    }


def _stellar(book, options, graph, knowledge_hash):
    workflow = workflow_spec("lifetime")
    source = Path(options.dataset)
    manifest = book.json("stellar.dataset_manifest", source / "manifest.json")
    report = book.json("stellar.report", source / "report.json")
    final = book.json("stellar.final_report", source / "final/report.json")
    content = manifest["content"]
    metadata, summary = _checkpoint(
        book, "stellar", Path(options.checkpoint), graph, "structured_tool_v6", "semantic_tool_v1", content
    )
    scores = final["closed_loop"]
    _require(
        content == report["content"] == metadata["provenance"]["stellar"]
        and content.get("task") == "stellar"
        and content.get("calculation_backend") == "local"
        and content.get("knowledge_pack_hash") == knowledge_hash
        and content.get("scope") == workflow.scope
        and content.get("required_fields") == list(workflow.required)
        and content.get("max_steps") == workflow.max_steps
        and report.get("development_gate_passed") is True
        and final.get("chain_gate_passed") is True
        and final.get("model_state_unchanged") is True
        and final.get("scope") == content["scope"]
        and _zero(final.get("optimizer_updates"))
        and final.get("checkpoint_sha256") == report.get("checkpoint_sha256") == summary["checkpoint_sha256"]
        and type(scores.get("requested")) is int
        and scores["requested"] == 100
        and isinstance(scores.get("episodes"), list)
        and len(scores["episodes"]) == 100
        and type(scores.get("completed")) is int
        and 90 <= scores["completed"] <= 100
        and type(scores.get("chained")) is int
        and 90 <= scores["chained"] <= 100
        and all(_zero(scores.get(key)) for key in _ERRORS),
        "stellar_recorded_gate_mismatch",
    )
    return {**summary, "recorded_gate_report_verified": True}


def _color(book, options, graph, reference):
    source = Path(options.color_experiment)
    content = book.json("color.dataset_manifest", source / "dataset-manifest.json")
    report = book.json("color.report", source / "report.json")
    final = book.json("color.final_report", source / "final/report.json")
    metadata, summary = _checkpoint(
        book,
        "color",
        source / "training/checkpoint.pt",
        graph,
        "structured_color_v1",
        "semantic_color_v1",
        content,
    )
    bands, dev = final.get("per_color", {}), report.get("development", {})
    _require(
        content == report.get("content")
        and content.get("task") == "color"
        and content.get("scope") == "learned_peak_wavelength_color_v1"
        and content.get("profile") == "pilot"
        and content.get("reference_hash") == reference.checksum
        and content.get("graph_hash") == graph["model_graph_fingerprint"]
        and metadata["model"].get("color_readout", "linear_v1") == content.get("color_readout", "linear_v1")
        and metadata["model"].get("color_input", "workflow_v1") == content.get("color_input", "workflow_v1")
        and report.get("ready_for_final_test") is True
        and dev.get("episodes") == 16
        and type(dev.get("completed")) is int
        and 15 <= dev["completed"] <= 16
        and all(_zero(dev.get(key)) for key in ("invalid_actions", "reference_errors"))
        and (not content.get("boundary_curriculum") or report.get("regression_gate_passed") is True)
        and final.get("browser_eligible") is True
        and final.get("per_band_gate_passed") is True
        and final.get("checkpoint_sha256") == report.get("checkpoint_sha256") == summary["checkpoint_sha256"]
        and final.get("reference_hash") == reference.checksum
        and final.get("scope") == content["scope"]
        and type(final.get("episodes")) is int
        and final["episodes"] == 100
        and type(final.get("completed")) is int
        and 90 <= final["completed"] <= 100
        and all(_zero(final.get(key)) for key in ("invalid_actions", "reference_errors"))
        and set(bands) == set(COLOR_LABELS)
        and all(
            type(v.get("episodes")) is int
            and v["episodes"] > 0
            and type(v.get("completed")) is int
            and 0.8 <= v["completed"] / v["episodes"] <= 1
            for v in bands.values()
        ),
        "color_recorded_gate_mismatch",
    )
    return {**summary, "recorded_gate_report_verified": True}


def _derived(book, role, pilot, final_dir, graph, pack_hash, encoding, control, scope, gate):
    pilot = Path(pilot)
    metadata = book.json(role + ".checkpoint_manifest", pilot / "training/checkpoint.pt.json")
    content = metadata["provenance"][role + "_calculations"]
    _, summary = _checkpoint(book, role, pilot / "training/checkpoint.pt", graph, encoding, control, content)
    report = book.json(role + ".report", pilot / "report.json")
    final = book.json(role + ".final_report", Path(final_dir) / "report.json")
    _require(
        report.get("content") == content
        and content.get("scope") == scope
        and content.get("task") == role + "_calculations"
        and content.get("calculation_backend") == "local"
        and content.get("knowledge_pack_hash") == pack_hash
        and content.get("graph_hash") == graph["model_graph_fingerprint"]
        and content.get("observation_encoding") == encoding
        and content.get("control_encoding") == control,
        "derived_content_mismatch",
    )
    gate(final, summary["checkpoint_sha256"])
    # Older gate helpers allow bool/int equality for counts. This source checker
    # does not promote those aliases into a statement of recorded provenance.
    scores = final.get("closed_loop", {}).get("test", {})
    _require(
        final.get("checkpoint_unchanged") is True
        and final.get("learning_gate_passed") is True
        and _zero(final.get("optimizer_updates"))
        and type(final.get("final_test_episodes")) is int
        and type(scores.get("episodes")) is int
        and type(scores.get("completed")) is int
        and scores["completed"] <= scores["episodes"]
        and all(_zero(scores.get(k)) for k in _ERRORS),
        "derived_recorded_gate_mismatch",
    )
    return {**summary, "recorded_gate_report_verified": True}


def _supplied_transfer_sources(task, pilot, directory, graph_hash):
    """Read the new recorded gate, returning only public summary and exact pins.

    This is not the original final gate and is never an evaluation invocation.
    The strict reader reconstructs recorded results, without model loading.
    """
    from . import habitability_supplied_inputs, planet_supplied_inputs
    from .training.supplied_input_transfer import require_supplied_input_transfer_gate

    _require(task in {"planet", "temperature"}, "supplied_task")
    module = planet_supplied_inputs if task == "planet" else habitability_supplied_inputs
    pack = module.load_supplied_planet_pack() if task == "planet" else module.load_supplied_temperature_pack()
    checkpoint = Path(pilot) / "training/checkpoint.pt"
    metadata = checkpoint.with_suffix(".pt.json")
    book = _Sources()
    digest = book.file("checkpoint", checkpoint)
    parent = book.json("metadata", metadata)
    content = parent["provenance"]["planet_calculations" if task == "planet" else "habitability_calculations"]
    adapter = module.adapter_manifest()
    gate = require_supplied_input_transfer_gate(
        directory,
        task=task,
        checkpoint_sha256=digest,
        graph_hash=graph_hash,
        pack_hash=pack.checksum,
        adapter_sha256=adapter["sha256"],
    )
    identity = gate["identity"]
    _require(
        gate.get("current_sources_verified") is True
        and gate.get("transfer_gate_passed") is True
        and identity["parent_metadata_sha256"] == book.rows["metadata"]["sha256"]
        and identity["parent_content_hash"] == parent["content_pack_hash"] == source_hash(content)
        and identity["parent_scope"] == content.get("scope") == module.LEGACY_SCOPE
        and identity["parent_pack_hash"] == content.get("knowledge_pack_hash") == module.BASE_PACK_HASH,
        "supplied_original_identity_changed",
    )
    pins = {}
    for source in (gate["source_sha256"], gate["artifact_sha256"]):
        for path, expected in source.items():
            _require(path not in pins or pins[path] == expected, "supplied_source_changed")
            pins[path] = expected
            _require(book.file("pin:" + path, path) == expected, "supplied_source_changed")
    book.unchanged()
    _require(not (Path(directory) / "stopped.json").exists(), "supplied_gate_stopped")
    return {
        "task": task,
        "scope": module.SCOPE,
        "pack_sha256": pack.checksum,
        "adapter_sha256": adapter["sha256"],
        "checkpoint_sha256": digest,
        "report_sha256": gate["report_sha256"],
        "recorded_transfer_gate_verified": True,
        "scores": gate["scores"],
        "calibration_verified": False,
        "model_loaded": False,
        "evaluation_executed": False,
        "browser_readiness_verified": False,
        "scientific_verified": False,
        "task_completed": False,
    }, pins


def validate_autonomous_decision_sources(options):
    """Check a dict/RunOptions without launching or authorizing any run.

    Read-only checks may inspect a 30-star configuration, but never lift the
    separate execution hold. Existing runtime loaders remain mandatory at use.
    """
    try:
        from .runtime import RunOptions, parse_run_options

        payload = options.model_dump(mode="python") if isinstance(options, RunOptions) else options
        _require(isinstance(payload, dict), "invalid_options")
        _require(payload.get("project_autonomous_decisions") is True, "explicit_autonomous_flag_required")
        options = parse_run_options(payload)
        _require(options.task == "browser_project", "project_options_required")
        book = _Sources()
        book.json("browser.config", options.browser_config)
        graph = _graph(book, Path(options.graph))
        hr_pack, hr_sha = load_hr_reference()
        hr = validate_hr_reference_source(Path.cwd())
        _require(hr["pack_sha256"] == hr_sha, "hr_source_changed")
        _require(
            book.file("reference.hr_png", Path.cwd() / hr_pack["source"]["capture"])
            == hr_pack["source"]["sha256"],
            "hr_image_hash",
        )
        for role, path in (
            ("stellar", DEFAULT_PACK),
            ("planet", DEFAULT_PLANET_PACK),
            ("habitability", DEFAULT_HABITABILITY_PACK),
            ("color", _PACKAGE / "packs/stellar_color.json"),
        ):
            book.json("reference." + role + "_pack_file", path)
        knowledge = load_knowledge_pack()
        planet, habitability = PlanetCalculator(), HabitabilityCalculator()
        color = load_color_reference()
        references = {
            "hr_pack_sha256": hr_sha,
            "hr_png_sha256": hr_pack["source"]["sha256"],
            "stellar_knowledge_sha256": knowledge.checksum,
            "stellar_golden": LocalCalculator(knowledge).verify(),
            "planet_knowledge_sha256": planet.pack.checksum,
            "planet_golden": planet.verify(),
            "habitability_knowledge_sha256": habitability.pack.checksum,
            "habitability_golden": habitability.verify(),
            "color_reference_sha256": color.checksum,
            "color_golden": validate_color_reference(color),
        }
        autonomous_pack = autonomous_planet.reference_pack()
        _require(
            autonomous_pack.get("id") == autonomous_planet.POLICY
            and type(autonomous_pack.get("schema_version")) is int
            and autonomous_pack["schema_version"] == 1,
            "autonomous_planet_pack",
        )
        references["autonomous_planet_reference_pack_sha256"] = hashlib.sha256(
            _canonical(autonomous_pack)
        ).hexdigest()
        references["autonomous_planet_implementation_sha256"] = book.file(
            "reference.autonomous_planet_implementation", autonomous_planet.__file__
        )
        if getattr(options, "project_baseline_edge_reference", False):
            from . import planet_window_baseline_edge

            references["baseline_edge_policy"] = planet_window_baseline_edge.policy_manifest()
            references["baseline_edge_implementation_sha256"] = book.file(
                "reference.baseline_edge_implementation", planet_window_baseline_edge.__file__
            )
        if getattr(options, "project_baseline_band_reference", False):
            from . import planet_window_baseline_band

            references["baseline_band_policy"] = planet_window_baseline_band.policy_manifest()
            references["baseline_band_implementation_sha256"] = book.file(
                "reference.baseline_band_implementation", planet_window_baseline_band.__file__
            )
        if getattr(options, "project_single_event_reference", False):
            from . import planet_window_single_event

            references["single_event_policy"] = planet_window_single_event.policy_manifest()
            references["single_event_implementation_sha256"] = book.file(
                "reference.single_event_implementation", planet_window_single_event.__file__
            )
        if getattr(options, "project_save_strategy", "explicit") == "autosave":
            from . import browser_autosave

            references["autosave_manifest"] = browser_autosave.autosave_manifest()
            references["autosave_implementation_sha256"] = book.file(
                "reference.autosave_implementation", browser_autosave.__file__
            )
        if getattr(options, "project_two_event_reference", False):
            from . import (
                browser_shallow_transit_probe,
                browser_shallow_transit_steps,
                planet_tooltip_reference,
            )

            references["two_event_measurement_method"] = planet_tooltip_reference.measurement_manifest(
                planet_tooltip_reference.TWO_MODE
            )
            references["two_event_overview_hint"] = browser_shallow_transit_probe.overview_hint_metadata(
                browser_shallow_transit_probe.EXACT_TWO_HINT_POLICY
            )
            for name, module in (
                ("reference", planet_tooltip_reference),
                ("probe", browser_shallow_transit_probe),
                ("scheduler", browser_shallow_transit_steps),
            ):
                references["two_event_" + name + "_sha256"] = book.file(
                    "reference.two_event_" + name, module.__file__
                )
        checkpoints = {
            "stellar": _stellar(book, options, graph, knowledge.checksum),
            "color": _color(book, options, graph, color),
            "planet": _derived(
                book,
                "planet",
                options.browser_planet_pilot,
                options.browser_planet_final_evaluation,
                graph,
                planet.pack.checksum,
                "structured_planet_tool_v1",
                "semantic_planet_tool_v1",
                "independent_physics_planet_derived_four_field_v1",
                require_planet_gate,
            ),
            "habitability": _derived(
                book,
                "habitability",
                options.browser_habitability_pilot,
                options.browser_habitability_final_evaluation,
                graph,
                habitability.pack.checksum,
                "structured_habitability_tool_v1",
                "semantic_habitability_tool_v1",
                "independent_physics_supplied_habitability_temperatures_v1",
                require_habitability_gate,
            ),
        }
        supplied = {}
        if options.project_supplied_stellar_inputs:
            for task, pilot, directory in (
                ("planet", options.browser_planet_pilot, options.browser_planet_supplied_evaluation),
                (
                    "temperature",
                    options.browser_habitability_pilot,
                    options.browser_habitability_supplied_evaluation,
                ),
            ):
                summary, pins = _supplied_transfer_sources(
                    task, pilot, directory, graph["model_graph_fingerprint"]
                )
                supplied[task] = summary
                for index, (path, digest) in enumerate(sorted(pins.items())):
                    _require(
                        book.file(f"supplied.{task}.{index:03d}", path) == digest, "supplied_source_changed"
                    )
        book.unchanged()
        _require(
            load_hr_reference()[1] == hr_sha and validate_hr_reference_source(Path.cwd()) == hr,
            "hr_source_changed",
        )
        _require(autonomous_planet.reference_pack() == autonomous_pack, "autonomous_pack_changed")
        return {
            "schema_version": 1,
            "mode": "offline_autonomous_source_validation_v1",
            "sources_verified": True,
            "graph": graph,
            "references": references,
            "checkpoints": checkpoints,
            "source_files": book.rows,
            "configured_stars": options.stars,
            "project_autonomous_decisions": True,
            **(
                {
                    "project_supplied_stellar_inputs": True,
                    "supplied_transfer_gates": supplied,
                    "recorded_transfer_cases_opened": True,
                }
                if options.project_supplied_stellar_inputs
                else {}
            ),
            "budgets": {
                "project_max_advances": options.project_max_advances,
                "project_max_seconds": options.project_max_seconds,
                "campaign_max_seconds": options.project_campaign_max_seconds,
                **(
                    {"campaign_timer": "no_overall_timer"}
                    if options.project_campaign_max_seconds == "uncapped"
                    else {}
                ),
                "habitability_max_seconds": options.browser_habitability_max_seconds,
            },
            "model_loaded": False,
            "inference_executed": False,
            "training_executed": False,
            "dataset_cases_opened": options.project_supplied_stellar_inputs,
            "final_evaluation_rerun": False,
            "learned_readiness_verified": False,
            "complete_promotion_chains_verified": False,
            "browser_readiness_verified": False,
            "scientific_verified": False,
            "training_label": False,
            "task_completed": False,
            "project_completed": False,
            "browser_actions": 0,
            "launch_authorized": False,
            "thirty_star_launch_authorized": False,
            "limitations": [
                "Checkpoint sidecars and recorded report hashes are checked; weight payloads are not deserialized.",
                "Existing runtime loaders must verify full model, promotion and current browser contracts at use.",
                "Reference heuristics are not learned classification, exact course boundaries or scientific verification.",
                "Read-only source validation does not lift the thirty-star execution hold.",
            ],
        }
    except AutonomousValidationError:
        raise
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, OverflowError):
        raise AutonomousValidationError("autonomous_validation_source_or_options_invalid") from None
