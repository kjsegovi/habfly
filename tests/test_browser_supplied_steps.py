"""Controller boundary fixtures, not new learned or live-browser acceptance."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_habitability_policy import mapping
from test_browser_planet_policy import FakeSession

import habfly.browser_supplied_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.training.checkpoints import source_hash


@pytest.fixture(params=["planet", "temperature"])
def loader(request, tmp_path, monkeypatch):
    task = request.param
    parent = module._module(task)
    pilot = tmp_path / "pilot"
    (pilot / "training").mkdir(parents=True)
    checkpoint = pilot / "training/checkpoint.pt"
    checkpoint.write_bytes(b"fixture, not trained weights")
    content = {"scope": parent.LEGACY_SCOPE, "knowledge_pack_hash": parent.BASE_PACK_HASH}
    metadata = checkpoint.with_suffix(".pt.json")
    metadata.write_text(
        json.dumps(
            {
                "provenance": {
                    "planet_calculations" if task == "planet" else "habitability_calculations": content
                }
            }
        )
    )
    identity = {
        "parent_metadata_sha256": module.file_hash(metadata),
        "parent_content_hash": source_hash(content),
        "parent_pack_hash": parent.BASE_PACK_HASH,
        "parent_scope": parent.LEGACY_SCOPE,
        "graph_hash": "fixture-graph",
    }
    gate = {"identity": identity, "current_sources_verified": True, "transfer_gate_passed": True}
    manifest = {"graph_hash": "fixture-graph", "content_pack_hash": source_hash(content)}
    graph = SimpleNamespace(body_ids=list(range(2000)))
    policy = SimpleNamespace(
        hidden_size=16,
        selection_mode="measurement_result_v3",
        observation_encoding=(
            "structured_planet_tool_v1" if task == "planet" else "structured_habitability_tool_v1"
        ),
    )
    calls = []
    policy.requires_grad_ = lambda value: calls.append(("grad", value))
    policy.eval = lambda: calls.append(("eval",))

    def validate(directory, **kwargs):
        calls.append(("gate", directory, kwargs))
        return deepcopy(gate)

    def load(path, graph, *, content_pack):
        assert content_pack == content  # New formulas must not masquerade as original training content.
        calls.append(("load", path))
        return policy, manifest

    monkeypatch.setattr(module, "require_supplied_input_transfer_gate", validate)
    monkeypatch.setattr(module, "load_checkpoint", load)
    monkeypatch.setattr(module, "graph_fingerprint", lambda _: "fixture-graph")
    return SimpleNamespace(
        task=task,
        pilot=pilot,
        final=tmp_path / "new-transfer",
        graph=graph,
        gate=gate,
        policy=policy,
        calls=calls,
        manifest=manifest,
        metadata=metadata,
    )


def test_new_gate_is_required_before_original_weight_load(loader):
    rig = loader
    policy, _, gate = module.load_supplied_frozen_policy(rig.task, rig.pilot, rig.final, rig.graph)
    assert rig.calls[0][0:2] == ("gate", rig.final) and rig.calls[1][0] == "load"
    assert rig.calls[0][2]["pack_hash"] == module._pack(rig.task).checksum
    assert ("grad", False) in rig.calls and ("eval",) in rig.calls
    assert policy.calibration["status"] == "uncalibrated"
    assert policy.action_temperature == policy.target_temperature == 1
    assert gate["identity"]["parent_pack_hash"] != module._pack(rig.task).checksum


@pytest.mark.parametrize("mutation", ["historical", "failed", "metadata", "content", "scope"])
def test_mismatched_gate_cannot_load_weights(loader, mutation):
    rig = loader
    if mutation == "historical":
        rig.gate["current_sources_verified"] = False
    elif mutation == "failed":
        rig.gate["transfer_gate_passed"] = False
    else:
        key = {
            "metadata": "parent_metadata_sha256",
            "content": "parent_content_hash",
            "scope": "parent_scope",
        }[mutation]
        rig.gate["identity"][key] = "wrong"
    with pytest.raises(BrowserSafetyStop, match="identity_changed"):
        module.load_supplied_frozen_policy(rig.task, rig.pilot, rig.final, rig.graph)
    assert not any(call[0] == "load" for call in rig.calls)


@pytest.mark.parametrize(
    "attribute,value", [("hidden_size", 32), ("selection_mode", "other"), ("observation_encoding", "other")]
)
def test_model_contract_mismatch_not_promoted(loader, attribute, value):
    setattr(loader.policy, attribute, value)
    with pytest.raises(BrowserSafetyStop, match="model_contract"):
        module.load_supplied_frozen_policy(loader.task, loader.pilot, loader.final, loader.graph)


@pytest.mark.parametrize("actual_class", [None, True, "", "main_sequence", "giant"])
def test_supplied_controller_requires_explicit_non_main_class(tmp_path, actual_class):
    owner = module.SuppliedPlanetDerivedSteps.__new__(module.SuppliedPlanetDerivedSteps)
    with pytest.raises(BrowserSafetyStop, match="explicit_non_main"):
        owner._configure_supplied(
            run_history=tmp_path,
            expected_star="Example",
            supplied_star_class=actual_class,
            supplied_stellar_inputs_path="receipt.json",
            supplied_stellar_inputs_sha256="a" * 64,
        )


@pytest.mark.parametrize("actual_class", module.NON_MAIN_CLASSES)
@pytest.mark.parametrize("task", ["planet", "temperature"])
def test_controller_environment_and_model_view_do_not_relabel_class(task, actual_class, monkeypatch):
    cls = (
        module.SuppliedPlanetDerivedSteps if task == "planet" else module.SuppliedHabitabilityTemperatureSteps
    )
    owner = cls.__new__(cls)
    owner._actual_class, owner._expected_star = actual_class, "Example"
    owner._pack_hash = module._pack(task).checksum
    owner._adapter_hash = module._module(task).adapter_manifest()["sha256"]
    owner._supplied_receipt = {"fixture": True}
    owner.session = FakeSession() if task == "planet" else SimpleNamespace(mapping=mapping())
    owner.session.mapping["star_name"] = "Example"
    monkeypatch.setattr(module, "matches_mapping", lambda *args: True)
    env = owner._environment(
        seed=123,
        **(
            {"supplied_star_class": actual_class}
            if task == "planet"
            else {"supplied_greenhouse_increment": 30}
        ),
    )
    obs = env.observe()
    before = obs.model_dump(mode="json")
    view = owner._inference_observation(obs)
    assert "star_class" not in view.values
    assert obs.model_dump(mode="json") == before and obs.values["star_class"] == actual_class
    assert "expected" not in env.case


def test_changed_visible_mass_or_star_cannot_start_environment(monkeypatch):
    planet = module.SuppliedPlanetDerivedSteps.__new__(module.SuppliedPlanetDerivedSteps)
    planet._actual_class, planet._supplied_receipt, planet.session = "white_dwarf", {}, FakeSession()
    monkeypatch.setattr(module, "matches_mapping", lambda *args: False)
    with pytest.raises(BrowserSafetyStop, match="current_stellar_inputs_changed"):
        planet._environment(supplied_star_class="white_dwarf", seed=0)
    temperature = module.SuppliedHabitabilityTemperatureSteps.__new__(
        module.SuppliedHabitabilityTemperatureSteps
    )
    temperature._actual_class, temperature._expected_star = "white_dwarf", "Example"
    temperature.session = SimpleNamespace(mapping={"star_name": "Different"})
    with pytest.raises(BrowserSafetyStop, match="current_temperature_star_changed"):
        temperature._environment(supplied_greenhouse_increment=0, seed=0)


def test_event_provenance_contains_identities_only_not_private_gate_or_receipt():
    owner = module.SuppliedPlanetDerivedSteps.__new__(module.SuppliedPlanetDerivedSteps)
    owner._actual_class, owner._adapter_hash = "white_dwarf", "a" * 64
    owner._supplied_link = {"path": "receipt.json", "sha256": "b" * 64}
    owner._supplied_gate_link = {"path": "gate/record.json", "sha256": "c" * 64}
    owner._supplied_identity = {
        "parent_content_hash": "d" * 64,
        "parent_pack_hash": "e" * 64,
        "parent_scope": "original",
        "private_case": "must not leak",
    }
    provenance = owner._additional_provenance()
    assert "must not leak" not in json.dumps(provenance)
    assert provenance["supplied_stellar_inputs"] == owner._supplied_link
    assert provenance["original_scope"] == "original"
