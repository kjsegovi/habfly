"""Offline supplied-scope plumbing; injected gate records are not promotion."""
# ruff: noqa: F811

import io
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_autonomous_validation import no_network_or_models, options, sha, source_fixture, write  # noqa: F401
from test_browser_project_campaign_steps import model_files  # noqa: F401
from test_browser_project_runtime import rig as bridge_rig  # noqa: F401

import habfly.autonomous_validation as validation
import habfly.browser_project_campaign_steps as campaign
import habfly.browser_project_runtime as bridge_module
from habfly.browser import BrowserSafetyStop
from habfly.runtime import RunOptions, Runtime, parse_run_options


def supplied_options(options):
    result = deepcopy(options)
    for role in ("planet", "habitability"):
        directory = Path(options["artifact_dir"]).parent / (role + "-supplied-gate")
        directory.mkdir(exist_ok=True)
        result[f"browser_{role}_supplied_evaluation"] = str(directory)
    result["project_supplied_stellar_inputs"] = True
    return result


@pytest.mark.parametrize("bad", [1, 0, None, "true", {}, []])
def test_opt_in_strict_bool(options, bad):
    payload = supplied_options(options)
    payload["project_supplied_stellar_inputs"] = bad
    with pytest.raises(ValueError, match="invalid_start_options"):
        parse_run_options(payload)


@pytest.mark.parametrize(
    "change",
    [
        "disabled",
        "manual",
        "no_planet",
        "no_temperature",
        "no_planet_gate",
        "no_temperature_gate",
        "absent_directory",
        "other_task",
    ],
)
def test_opt_in_requires_complete_explicit_owned_configuration(options, change):
    payload = supplied_options(options)
    key = {
        "disabled": "project_supplied_stellar_inputs",
        "manual": "project_autonomous_decisions",
        "no_planet": "browser_planet_pilot",
        "no_temperature": "browser_habitability_pilot",
        "no_planet_gate": "browser_planet_supplied_evaluation",
        "no_temperature_gate": "browser_habitability_supplied_evaluation",
    }.get(change)
    if key:
        payload[key] = False if change in {"disabled", "manual"} else None
    elif change == "absent_directory":
        payload["browser_planet_supplied_evaluation"] += "/missing"
    else:
        payload.update(
            task="stellar",
            project_autonomous_decisions=False,
            project_campaign=False,
            project_campaign_max_seconds=None,
        )
    with pytest.raises(ValueError):
        parse_run_options(payload)
    assert not Path(payload["artifact_dir"]).exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_only_forwards_gate_options_when_explicitly_enabled(options, monkeypatch, enabled):
    settings = supplied_options(options) if enabled else options
    seen = {}

    class Bridge:
        status = "paused"

        def __init__(self, options, output, **kwargs):
            seen.update(kwargs)

        def start(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(bridge_module, "BrowserProjectRuntime", Bridge)
    runtime = Runtime(io.StringIO())
    runtime.start(settings)
    models, terrestrial = seen["model_options"], seen["terrestrial_options"]
    assert ("planet_supplied_evaluation" in models) is enabled
    assert ("supplied_evaluation" in terrestrial) is enabled
    if enabled:
        assert models["planet_supplied_evaluation"] == Path(settings["browser_planet_supplied_evaluation"])
        assert terrestrial["supplied_evaluation"] == Path(
            settings["browser_habitability_supplied_evaluation"]
        )
    assert not runtime.env.__dict__.get("browser")
    runtime.close()


@pytest.mark.parametrize("mutation", [None, "scope", "flag", "planet", "temperature"])
def test_outer_scope_pins_explicit_option_without_native_calls(bridge_rig, mutation):
    rig = bridge_rig
    for role in ("planet", "habitability"):
        directory = rig.root / (role + "-supplied")
        directory.mkdir()
        setattr(rig.options, f"browser_{role}_supplied_evaluation", directory)
    rig.options.project_supplied_stellar_inputs = True
    rig.options.project_autonomous_decisions = True
    rig.options.task = "browser_project"
    models = {key: Path(key) for key in ("dataset", "checkpoint", "color_experiment", "graph_path")}
    models.update(
        planet_pilot=Path("pilot"),
        planet_final_evaluation=Path("legacy-final"),
        planet_supplied_evaluation=rig.options.browser_planet_supplied_evaluation,
    )
    terrestrial = {
        "supplied_evaluation": rig.options.browser_habitability_supplied_evaluation,
        "pilot": Path("pilot"),
        "final_evaluation": Path("legacy-final"),
        "graph": object(),
        "candidates": ["CO2"],
    }
    bridge = bridge_module.BrowserProjectRuntime(
        rig.options,
        rig.root / "run",
        model_options=models,
        reference_planet_continuation=True,
        terrestrial_options=terrestrial,
        config=rig.config,
        emit=lambda *e: rig.events.append(e),
        _driver_factory=rig.driver,
    )
    assert bridge.scope["supplied_stellar_inputs_enabled"] is True and not rig.calls
    if mutation == "scope":
        bridge.scope["supplied_stellar_inputs_enabled"] = 1
    elif mutation == "flag":
        bridge.supplied_stellar_inputs = False
    elif mutation == "planet":
        bridge.model_options["planet_supplied_evaluation"] = Path("other")
    elif mutation == "temperature":
        bridge.terrestrial_options["supplied_evaluation"] = Path("other")
    if mutation:
        with pytest.raises(BrowserSafetyStop, match="supplied_input_settings_changed"):
            bridge._check_viewport()
    else:
        bridge._check_viewport()
    assert not rig.calls
    bridge.close()


@pytest.mark.parametrize("change", [None, "mutated_pin", "gate_rejected"])
def test_source_check_adds_exact_gate_proof_without_model_or_browser(options, monkeypatch, change):
    payload = supplied_options(options)
    calls = []

    def gate(task, pilot, directory, graph_hash):
        calls.append(task)
        if change == "gate_rejected":
            raise ValueError("private case details must not escape")
        path = Path(directory) / "recorded-proof.json"
        write(path, {"declared_fixture": task})
        digest = sha(path)
        if change == "mutated_pin":
            write(path, {"changed": True})
        return {"recorded_transfer_gate_verified": True, "task": task}, {str(path): digest}

    monkeypatch.setattr(validation, "_supplied_transfer_sources", gate)
    if change:
        with pytest.raises(validation.AutonomousValidationError) as exc:
            validation.validate_autonomous_decision_sources(payload)
        assert "private case" not in str(exc.value)
    else:
        report = validation.validate_autonomous_decision_sources(payload)
        assert calls == ["planet", "temperature"]
        assert report["recorded_transfer_cases_opened"] is report["dataset_cases_opened"] is True
        assert set(report["supplied_transfer_gates"]) == {"planet", "temperature"}
        assert not any(
            report[k]
            for k in (
                "model_loaded",
                "inference_executed",
                "training_executed",
                "launch_authorized",
                "task_completed",
            )
        )
        assert len([k for k in report["source_files"] if k.startswith("supplied.")]) == 2
    assert not Path(payload["artifact_dir"]).exists()


def test_campaign_pins_gate_allowlist_not_unrelated_files(model_files, monkeypatch, tmp_path):
    models, terrestrial, _ = model_files
    models["planet_supplied_evaluation"] = tmp_path / "planet-new"
    terrestrial["supplied_evaluation"] = tmp_path / "temperature-new"
    proof = tmp_path / "owned-proof.json"
    write(proof, {"strict_reader_fixture": True})
    calls = []

    def gate(task, pilot, directory, graph_hash):
        calls.append(task)
        return {}, {str(proof): sha(proof)}

    monkeypatch.setattr(validation, "_supplied_transfer_sources", gate)
    monkeypatch.setattr("habfly.data.graphs.load_graph", lambda *_: SimpleNamespace())
    monkeypatch.setattr(campaign, "graph_fingerprint", lambda *_: "graph-fixture")
    result = campaign._model_dependencies(models, terrestrial)
    assert calls == ["planet", "temperature"] and result[str(proof)] == sha(proof)
    unrelated = tmp_path / "unrelated-trajectory.jsonl"
    write(unrelated, {"not_consumed": True})
    assert campaign._model_dependencies(models, terrestrial) == result
    write(proof, {"changed": True})
    assert campaign._model_dependencies(models, terrestrial) != result
    del terrestrial["supplied_evaluation"]
    with pytest.raises(BrowserSafetyStop, match="supplied_gate_pair_required"):
        campaign._model_dependencies(models, terrestrial)


def test_supplied_config_drift_revokes_runtime_authorization(options):
    runtime = Runtime(io.StringIO())
    runtime.options = RunOptions.model_validate(supplied_options(options))
    runtime._finalization_authorization = runtime._authorization()
    runtime.options.browser_planet_supplied_evaluation = Path("different")
    with pytest.raises(ValueError):
        runtime._check_project_authorization()


@pytest.mark.parametrize("task", ["planet", "temperature"])
@pytest.mark.parametrize(
    "change", [None, "metadata_identity", "original_scope", "content_hash", "changed_leaf", "stopped"]
)
def test_gate_adoption_rechecks_exact_original_metadata_and_owned_pins(options, monkeypatch, task, change):
    import json

    import habfly.training.supplied_input_transfer as transfer

    role = "planet" if task == "planet" else "habitability"
    pilot = Path(options[f"browser_{role}_pilot"])
    checkpoint = pilot / "training/checkpoint.pt"
    metadata = checkpoint.with_suffix(".pt.json")
    parent = json.loads(metadata.read_bytes())
    content = parent["provenance"][role + "_calculations"]
    directory = pilot / "new-transfer-proof"
    directory.mkdir()
    proof = directory / "report.json"
    write(proof, {"declared_strict_reader_fixture": True})
    gate = {
        "current_sources_verified": True,
        "transfer_gate_passed": True,
        "identity": {
            "parent_metadata_sha256": sha(metadata),
            "parent_content_hash": parent["content_pack_hash"],
            "parent_scope": content["scope"],
            "parent_pack_hash": content["knowledge_pack_hash"],
        },
        "source_sha256": {str(metadata): sha(metadata), str(checkpoint): sha(checkpoint)},
        "artifact_sha256": {str(proof): sha(proof)},
        "report_sha256": sha(proof),
        "scores": {"episodes": 100},
    }
    if change in {"metadata_identity", "original_scope", "content_hash"}:
        key = {
            "metadata_identity": "parent_metadata_sha256",
            "original_scope": "parent_scope",
            "content_hash": "parent_content_hash",
        }[change]
        gate["identity"][key] = "wrong"
    elif change == "changed_leaf":
        write(proof, {"modified": True})
    elif change == "stopped":
        write(directory / "stopped.json", {"failure": True})
    calls = []

    def reader(*args, **kwargs):
        calls.append((args, kwargs))
        return gate

    monkeypatch.setattr(transfer, "require_supplied_input_transfer_gate", reader)
    if change:
        with pytest.raises(validation.AutonomousValidationError):
            validation._supplied_transfer_sources(task, pilot, directory, parent["graph_hash"])
    else:
        summary, pins = validation._supplied_transfer_sources(task, pilot, directory, parent["graph_hash"])
        assert pins == gate["source_sha256"] | gate["artifact_sha256"]
        assert summary["recorded_transfer_gate_verified"] and not summary["model_loaded"]
        assert calls[0][1]["checkpoint_sha256"] == sha(checkpoint)
        assert calls[0][1]["task"] == task


def test_campaign_new_stop_marker_revokes_supplied_gate(tmp_path):
    directory = tmp_path / "gate"
    directory.mkdir()
    models = {"planet_supplied_evaluation": directory}
    campaign._supplied_gate_status(models, None)
    write(directory / "stopped.json", {"failed": True})
    with pytest.raises(BrowserSafetyStop, match="supplied_gate_stopped"):
        campaign._supplied_gate_status(models, None)
