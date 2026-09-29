"""Explicitly synthetic metadata fixtures; no weights, cases or browser runs."""

import hashlib
import json
import socket
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from test_browser_numeric import config as browser_config
from test_stellar_reference import source_fixture  # noqa: F401 - pytest fixture

import habfly.autonomous_validation as module
from habfly.autonomous_validation import AutonomousValidationError, validate_autonomous_decision_sources
from habfly.color_reference import COLOR_LABELS, load_color_reference
from habfly.data.graphs import make_demo_graph, save_graph
from habfly.habitability_knowledge import HabitabilityCalculator
from habfly.knowledge import load_knowledge_pack
from habfly.model import CharacterTokenizer, ConnectomePolicy, TopologyFreePolicy, graph_fingerprint
from habfly.planet_knowledge import PlanetCalculator
from habfly.runtime import parse_run_options
from habfly.training.chained_workflow import workflow_spec
from habfly.training.checkpoints import source_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def no_network_or_models(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Source validation cannot load a model, launch a browser or use the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(ConnectomePolicy, "__init__", forbidden)
    monkeypatch.setattr(TopologyFreePolicy, "__init__", forbidden)


@pytest.fixture
def options(tmp_path, source_fixture, monkeypatch):  # noqa: F811 - imported pytest fixture
    """Fake frozen files deliberately cannot deserialize; this checker says so.

    Synthetic graph carries biological source metadata solely to exercise this
    validator's declared metadata contract, never as a biological data claim.
    """
    monkeypatch.chdir(tmp_path)
    graph = make_demo_graph(2000)
    graph.manifest["source_kind"] = "biological"
    graph_dir = save_graph(graph, tmp_path / "graph")
    graph_hash = graph_fingerprint(graph)
    tokenizer = CharacterTokenizer()

    def checkpoint(role, content, encoding, control):
        directory = tmp_path / role
        path = directory / "training/checkpoint.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"SYNTHETIC SOURCE-CHECK FIXTURE, NOT A MODEL")
        manifest = {
            "schema_version": 1,
            "model": {
                "architecture": "ConnectomePolicy",
                "hidden_size": 16,
                "observation_encoding": encoding,
                "control_encoding": control,
                "selection_mode": "measurement_result_v3",
            },
            "tokenizer": tokenizer.config(),
            "tokenizer_hash": tokenizer.fingerprint,
            "graph_hash": graph_hash,
            "graph_size": 2000,
            "content_pack_hash": source_hash(content),
            "training_stage": "synthetic_metadata_only",
            "seed": 0,
            "provenance": {}
            if role == "color"
            else {role if role == "stellar" else role + "_calculations": content},
        }
        write(path.with_suffix(".pt.json"), manifest)
        return directory, path, sha(path)

    workflow = workflow_spec("lifetime")
    content = {
        "task": "stellar",
        "calculation_backend": "local",
        "knowledge_pack_hash": load_knowledge_pack().checksum,
        "scope": workflow.scope,
        "required_fields": list(workflow.required),
        "max_steps": workflow.max_steps,
    }
    directory, stellar_checkpoint, digest = checkpoint(
        "stellar", content, "structured_tool_v6", "semantic_tool_v1"
    )
    write(directory / "manifest.json", {"content": content})
    write(
        directory / "report.json",
        {"content": content, "checkpoint_sha256": digest, "development_gate_passed": True},
    )
    write(
        directory / "final/report.json",
        {
            "scope": workflow.scope,
            "checkpoint_sha256": digest,
            "chain_gate_passed": True,
            "model_state_unchanged": True,
            "optimizer_updates": 0,
            "closed_loop": {
                "requested": 100,
                "episodes": [{}] * 100,
                "completed": 100,
                "chained": 100,
                **dict.fromkeys(module._ERRORS, 0),
            },
        },
    )

    color_hash = load_color_reference().checksum
    content = {
        "task": "color",
        "scope": "learned_peak_wavelength_color_v1",
        "profile": "pilot",
        "reference_hash": color_hash,
        "graph_hash": graph_hash,
    }
    directory, _, digest = checkpoint("color", content, "structured_color_v1", "semantic_color_v1")
    write(directory / "dataset-manifest.json", content)
    write(
        directory / "report.json",
        {
            "content": content,
            "checkpoint_sha256": digest,
            "ready_for_final_test": True,
            "development": {"episodes": 16, "completed": 16, "invalid_actions": 0, "reference_errors": 0},
        },
    )
    write(
        directory / "final/report.json",
        {
            "scope": content["scope"],
            "checkpoint_sha256": digest,
            "reference_hash": color_hash,
            "browser_eligible": True,
            "per_band_gate_passed": True,
            "episodes": 100,
            "completed": 100,
            "invalid_actions": 0,
            "reference_errors": 0,
            "per_color": {
                label: {"episodes": 12 if i == 0 else 11, "completed": 12 if i == 0 else 11}
                for i, label in enumerate(COLOR_LABELS)
            },
        },
    )

    for role, calculator, scope in (
        ("planet", PlanetCalculator(), "independent_physics_planet_derived_four_field_v1"),
        (
            "habitability",
            HabitabilityCalculator(),
            "independent_physics_supplied_habitability_temperatures_v1",
        ),
    ):
        content = {
            "task": role + "_calculations",
            "scope": scope,
            "calculation_backend": "local",
            "knowledge_pack_hash": calculator.pack.checksum,
            "graph_hash": graph_hash,
            "observation_encoding": f"structured_{role}_tool_v1",
            "control_encoding": f"semantic_{role}_tool_v1",
        }
        directory, _, digest = checkpoint(
            role, content, content["observation_encoding"], content["control_encoding"]
        )
        write(directory / "report.json", {"content": content})
        write(
            tmp_path / (role + "-final") / "report.json",
            {
                "scope": scope,
                "checkpoint_sha256": digest,
                "checkpoint_unchanged": True,
                "optimizer_updates": 0,
                "final_test_episodes": 100,
                "learning_gate_passed": True,
                "closed_loop": {
                    "test": {"episodes": 100, "completed": 100, **dict.fromkeys(module._ERRORS, 0)}
                },
            },
        )

    browser = tmp_path / "browser.json"
    write(browser, browser_config().model_dump(mode="json"))
    return {
        "task": "browser_project",
        "environment": "browser",
        "policy": "checkpoint",
        "browser_setup": "automatic",
        "browser_execution": "autonomous",
        "calculation_backend": "local",
        "graph": str(graph_dir),
        "checkpoint": str(stellar_checkpoint),
        "dataset": str(tmp_path / "stellar"),
        "color_experiment": str(tmp_path / "color"),
        "browser_config": str(browser),
        "browser_planet_pilot": str(tmp_path / "planet"),
        "browser_planet_final_evaluation": str(tmp_path / "planet-final"),
        "browser_habitability_pilot": str(tmp_path / "habitability"),
        "browser_habitability_final_evaluation": str(tmp_path / "habitability-final"),
        "browser_habitability_gas_candidates": ["CH4", "CO2", "H2O", "H2S", "N2O", "NH3", "O3"],
        "project_reference_planet_continuation": True,
        "project_autonomous_decisions": True,
        "stars": 3,
        "project_campaign": True,
        "project_campaign_max_seconds": 5400,
        "artifact_dir": str(tmp_path / "must-not-be-created"),
    }


def test_sources_only_no_model_or_launch_authority(options):
    before = {p: sha(p) for p in Path.cwd().rglob("*") if p.is_file()}
    report = validate_autonomous_decision_sources(options)
    assert report["sources_verified"] is True and report["graph"]["nodes"] == 2000
    assert set(report["checkpoints"]) == {"stellar", "color", "planet", "habitability"}
    assert len(report["source_files"]) == 36
    for key in (
        "model_loaded",
        "inference_executed",
        "training_executed",
        "dataset_cases_opened",
        "final_evaluation_rerun",
        "learned_readiness_verified",
        "complete_promotion_chains_verified",
        "browser_readiness_verified",
        "scientific_verified",
        "training_label",
        "task_completed",
        "project_completed",
        "launch_authorized",
        "thirty_star_launch_authorized",
    ):
        assert report[key] is False
    for row in report["checkpoints"].values():
        assert row["manifest_compatibility_verified"] is True
        assert row["recorded_gate_report_verified"] is True
        assert row["weight_payload_deserialized"] is row["weight_manifest_agreement_verified"] is False
    assert report["browser_actions"] == 0
    assert before == {p: sha(p) for p in Path.cwd().rglob("*") if p.is_file()}
    assert not Path(options["artifact_dir"]).exists()
    assert validate_autonomous_decision_sources(parse_run_options(options)) == report
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    for row in report["source_files"].values():
        assert sha(Path(row["path"])) == row["sha256"]
    assert report["graph"]["artifact_graph_hash"] != report["graph"]["model_graph_fingerprint"]


def test_readonly_thirty_check_does_not_lift_hold(options):
    options.update(
        stars=30,
        project_campaign_max_seconds=10800,
        project_allow_scoring=True,
        project_allow_submission=True,
    )
    report = validate_autonomous_decision_sources(options)
    assert report["configured_stars"] == 30
    assert report["launch_authorized"] is report["thirty_star_launch_authorized"] is False
    assert not Path(options["artifact_dir"]).exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"project_autonomous_decisions": False},
        {"project_autonomous_decisions": 1},
        {"project_autonomous_decisions": "true"},
        {"task": "stellar"},
        {"environment": "simulator"},
        {"project_reference_planet_continuation": False},
        {"browser_habitability_pilot": None},
        {"browser_habitability_gas_candidates": []},
        {"project_max_advances": 513},
        {"project_max_seconds": 1801},
        {"project_campaign_max_seconds": 10801},
    ],
)
def test_invalid_optins_and_budgets_reject(options, changes):
    with pytest.raises(AutonomousValidationError):
        validate_autonomous_decision_sources(options | changes)
    assert not Path(options["artifact_dir"]).exists()


@pytest.mark.parametrize("role", ["stellar", "color", "planet", "habitability"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("graph_hash", "a" * 64),
        ("graph_size", 2001),
        ("content_pack_hash", "a" * 64),
        ("tokenizer_hash", "a" * 64),
        ("hidden_size", 8),
        ("architecture", "TopologyFreePolicy"),
        ("observation_encoding", "wrong_encoding"),
        ("control_encoding", "wrong_control"),
    ],
)
def test_all_checkpoint_manifest_contracts(options, role, field, value):
    path = Path(options["dataset"]).parent / role / "training/checkpoint.pt.json"
    data = json.loads(path.read_bytes())
    target = (
        data["model"]
        if field in {"hidden_size", "architecture", "observation_encoding", "control_encoding"}
        else data
    )
    target[field] = value
    write(path, data)
    with pytest.raises(AutonomousValidationError):
        validate_autonomous_decision_sources(options)


@pytest.mark.parametrize(
    "relative",
    [
        "stellar/training/checkpoint.pt",
        "color/training/checkpoint.pt",
        "planet/training/checkpoint.pt",
        "habitability/training/checkpoint.pt",
        "graph/body_ids.npy",
        "graph/metadata.json",
    ],
)
def test_changed_checkpoint_or_graph_bytes_reject(options, relative):
    path = Path.cwd() / relative
    path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(AutonomousValidationError):
        validate_autonomous_decision_sources(options)


@pytest.mark.parametrize(
    "relative,field,value",
    [
        ("stellar/final/report.json", "optimizer_updates", False),
        ("stellar/final/report.json", "chain_gate_passed", 1),
        ("color/final/report.json", "browser_eligible", False),
        ("color/final/report.json", "reference_hash", "a" * 64),
        ("planet-final/report.json", "optimizer_updates", False),
        ("planet-final/report.json", "learning_gate_passed", 1),
        ("habitability-final/report.json", "checkpoint_unchanged", False),
        ("habitability-final/report.json", "final_test_episodes", 99),
    ],
)
def test_recorded_final_report_mismatch_without_opening_test_cases(options, relative, field, value):
    path = Path.cwd() / relative
    data = json.loads(path.read_bytes())
    data[field] = value
    write(path, data)
    with pytest.raises(AutonomousValidationError):
        validate_autonomous_decision_sources(options)


def test_cases_are_not_required_or_opened(options, monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name in {"manual.json", "test.json", "train.json"} or "private-" in path.name:
            pytest.fail("Source check must not reopen cases")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert validate_autonomous_decision_sources(options)["dataset_cases_opened"] is False


@pytest.mark.parametrize(
    "relative", ["stellar/training/checkpoint.pt", "planet-final/report.json", "graph/body_ids.npy"]
)
def test_source_symlink_refused(options, relative):
    path = Path.cwd() / relative
    target = path.with_suffix(".source")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(AutonomousValidationError):
        validate_autonomous_decision_sources(options)


def test_source_changed_during_validation_detected(options, monkeypatch):
    original = module._derived
    count = 0

    def changed(*args, **kwargs):
        nonlocal count
        value = original(*args, **kwargs)
        count += 1
        if count == 2:
            path = Path(options["checkpoint"])
            path.write_bytes(path.read_bytes() + b"changed during check")
        return value

    monkeypatch.setattr(module, "_derived", changed)
    with pytest.raises(AutonomousValidationError, match="source_changed"):
        validate_autonomous_decision_sources(options)


def test_changed_autonomous_reference_detected(options, monkeypatch):
    original = module.autonomous_planet.reference_pack
    count = 0

    def changed():
        nonlocal count
        count += 1
        pack = original()
        if count > 1:
            pack["planet_policy"] += "changed"
        return pack

    monkeypatch.setattr(module.autonomous_planet, "reference_pack", changed)
    with pytest.raises(AutonomousValidationError, match="autonomous_pack_changed"):
        validate_autonomous_decision_sources(options)


def test_hr_source_missing_or_changed(options):
    pack, _ = module.load_hr_reference()
    path = Path.cwd() / pack["source"]["capture"]
    path.write_bytes(b"unverified image")
    with pytest.raises(AutonomousValidationError):
        validate_autonomous_decision_sources(options)


def test_config_and_errors_never_echo_credentials(options):
    bad = deepcopy(options)
    bad["unsupported_private_value"] = "private secret or session URL"
    with pytest.raises(AutonomousValidationError) as caught:
        validate_autonomous_decision_sources(bad)
    assert str(caught.value) == "autonomous_validation_source_or_options_invalid"


@pytest.mark.parametrize("suffix", [',"optimizer_updates":0}', ',"unexpected":NaN}'])
def test_noncanonical_report_objects_reject(options, suffix):
    path = Path.cwd() / "planet-final/report.json"
    path.write_text(path.read_text()[:-1] + suffix)
    with pytest.raises(AutonomousValidationError):
        validate_autonomous_decision_sources(options)


def test_autonomous_reference_version_requires_integer(options, monkeypatch):
    pack = module.autonomous_planet.reference_pack()
    pack["schema_version"] = True
    monkeypatch.setattr(module.autonomous_planet, "reference_pack", lambda: deepcopy(pack))
    with pytest.raises(AutonomousValidationError, match="autonomous_planet_pack"):
        validate_autonomous_decision_sources(options)


def test_actual_profile_readonly_when_local_artifacts_are_available():
    root = Path(__file__).resolve().parents[1]
    profile = root / "configs/browser_project_autonomous_three_star.json"
    if not profile.is_file() or not (root / "experiments/lifetime-003/training/checkpoint.pt").is_file():
        pytest.skip("Local frozen experiment artifacts are not shipped in source-only checkouts")
    report = validate_autonomous_decision_sources(json.loads(profile.read_bytes()))
    assert report["sources_verified"] is True
    assert (
        report["graph"]["model_graph_fingerprint"]
        == "580eaa081ccf57e466343352a9a61b16a4ec8f025055c0797afe0cca7a228bfd"
    )
    assert report["model_loaded"] is report["thirty_star_launch_authorized"] is False
