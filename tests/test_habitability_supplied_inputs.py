"""Actual-class temperature arithmetic and inference contracts, never native proof."""

import copy
import hashlib
import json

import pytest

from habfly.contracts import Action
from habfly.environments.habitability_calculations import HabitabilityCalculationEnv, habitability_cases
from habfly.habitability_knowledge import (
    DEFAULT_HABITABILITY_PACK,
    HabitabilityCalculator,
    load_habitability_pack,
)
from habfly.habitability_supplied_inputs import (
    BASE_FILE_SHA256,
    BASE_PACK_HASH,
    CLASSES,
    SCOPE,
    SuppliedTemperatureCalculator,
    SuppliedTemperatureEnv,
    SuppliedTemperaturePack,
    adapter_manifest,
    inference_view,
    load_supplied_temperature_pack,
    supplied_case,
)
from habfly.model.habitability_tool_state import habitability_state_features


@pytest.fixture
def calculator():
    return SuppliedTemperatureCalculator()


def case(star_class="white_dwarf"):
    return supplied_case(habitability_cases("development", 1)[0], star_class)


def environment(calculator, star_class="white_dwarf"):
    item = case(star_class)
    env = SuppliedTemperatureEnv(calculator, [item])
    env.reset(seed=item["seed"])
    return env


def inputs():
    return {
        "stellar_luminosity": {"value": 1, "unit": "Lsun"},
        "orbital_radius": {"value": 1, "unit": "au"},
        "albedo": {"value": 0, "unit": "fraction"},
    }


def execute(calculator, bindings, operation="equilibrium_temp", star_class="white_dwarf"):
    return calculator.execute(
        operation, bindings, star_class=star_class, assumptions=calculator.pack.assumptions
    )


def action(env, key, value=None):
    obs = env.observe()
    target = next(control.id for control in obs.controls if control.id.split(":", 1)[1] == key)
    return env.step(
        Action(
            kind="SELECT" if value is not None else "CLICK",
            target=target,
            value=value,
            observation_revision=obs.revision,
        )
    )


def test_original_equations_units_constants_bands_and_pending_preserved():
    old, new = load_habitability_pack(), load_supplied_temperature_pack()
    assert hashlib.sha256(DEFAULT_HABITABILITY_PACK.read_bytes()).hexdigest() == BASE_FILE_SHA256
    assert old.checksum == BASE_PACK_HASH != new.checksum
    assert len(new.operations) == 2
    for original, supplied in zip(old.operations, new.operations, strict=True):
        expected = original.model_dump(mode="json")
        expected["applicable_classes"] = list(CLASSES)
        assert supplied.model_dump(mode="json") == expected
    for key in ("assumptions", "greenhouse_bands", "pending", "provenance"):
        assert getattr(new, key) == getattr(old, key)
    assert SuppliedTemperaturePack.model_validate_json(new.model_dump_json()).checksum == new.checksum


@pytest.mark.parametrize(
    "mutation",
    [
        "expression",
        "constant",
        "unit",
        "source",
        "class",
        "assumption",
        "pending",
        "provenance",
        "band",
        "band_type",
        "input_bool",
        "schema_bool",
        "declaration",
    ],
)
def test_exact_source_extension_rejects_mutation_before_coercion(mutation):
    raw = load_supplied_temperature_pack().model_dump(mode="json")
    op = raw["operations"][0]
    if mutation == "expression":
        op["expression"] += "+1"
    elif mutation == "constant":
        op["constants"]["solar_watts"] *= 2
    elif mutation == "unit":
        op["inputs"]["albedo"]["unit"] = "%"
    elif mutation == "source":
        op["sources"] = [{"reference": "invented"}]
    elif mutation == "class":
        op["applicable_classes"] = ["main_sequence"]
    elif mutation == "assumption":
        raw["assumptions"] = []
    elif mutation == "pending":
        raw["pending"] = []
    elif mutation == "provenance":
        raw["provenance"]["course_validation"] = "passed"
    elif mutation == "band":
        raw["greenhouse_bands"][0]["maximum"] = "0.5"
    elif mutation == "band_type":
        raw["greenhouse_bands"][0]["increment_kelvin"] = False
    elif mutation == "input_bool":
        op["inputs"]["albedo"]["positive"] = 0
    elif mutation == "schema_bool":
        raw["schema_version"] = True
    else:
        raw["declaration"]["base_pack_hash"] = "0" * 64
    with pytest.raises(ValueError, match="math_or_provenance_changed"):
        SuppliedTemperaturePack.model_validate(raw)


@pytest.mark.parametrize("path_name", ["DEFAULT_HABITABILITY_PACK", "PACK_PATH"])
def test_file_tamper_refused(monkeypatch, tmp_path, path_name):
    import habfly.habitability_supplied_inputs as module

    changed = tmp_path / "source.json"
    changed.write_text("{}")
    monkeypatch.setattr(module, path_name, changed)
    with pytest.raises(ValueError, match="changed"):
        load_supplied_temperature_pack()


def test_goldens_offline_and_not_acceptance(calculator, monkeypatch):
    import socket

    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: pytest.fail("network used"))
    report = calculator.verify()
    assert report["valid"] is True and report["operation_golden_cases"] == 24
    for key in (
        "scientific_verified",
        "course_acceptance_passed",
        "native_browser_enabled",
        "training_oracle_enabled",
    ):
        assert report[key] is False
    with pytest.raises(ValueError, match="no_training_oracle"):
        calculator.reference_answers({})


@pytest.mark.parametrize("star_class", CLASSES)
def test_actual_class_reaches_arithmetic_unchanged(calculator, monkeypatch, star_class):
    seen, original = [], calculator._execute

    def traced(*args, **kwargs):
        seen.append(kwargs["star_class"])
        return original(*args, **kwargs)

    monkeypatch.setattr(calculator, "_execute", traced)
    result = execute(calculator, inputs(), star_class=star_class)
    assert result.ok and result.unit == "K" and seen == [star_class]


@pytest.mark.parametrize("assumptions", [None, [], [True], [float("nan")], {}, "terrestrial_planet"])
def test_explicit_assumptions_required(calculator, assumptions):
    result = calculator.execute(
        "equilibrium_temp", inputs(), star_class="white_dwarf", assumptions=assumptions
    )
    assert result.error == "temperature_assumptions_not_explicit"


def test_assumptions_and_class_not_defaulted(calculator):
    for assumptions in (
        tuple(calculator.pack.assumptions),
        calculator.pack.assumptions[::-1],
        calculator.pack.assumptions + calculator.pack.assumptions[:1],
    ):
        result = calculator.execute(
            "equilibrium_temp", inputs(), star_class="red_giant", assumptions=assumptions
        )
        assert result.error == "temperature_assumptions_not_explicit"
    for actual in (None, True, "giant", "unknown"):
        assert execute(calculator, inputs(), star_class=actual).error == "unsupported_supplied_class"
    with pytest.raises(TypeError):
        calculator.execute("equilibrium_temp", inputs(), assumptions=calculator.pack.assumptions)


@pytest.mark.parametrize("value", [None, "0.3", True, float("nan"), float("inf"), 10**400, -1, 1])
def test_invalid_albedo_rejected_without_correction(calculator, value):
    bindings = inputs()
    bindings["albedo"]["value"] = value
    assert not execute(calculator, bindings).ok


def test_missing_zero_units_and_valid_wrong_inputs(calculator):
    baseline = execute(calculator, inputs()).value
    wrong = inputs()
    wrong["stellar_luminosity"]["value"] = 16
    assert execute(calculator, wrong).value == pytest.approx(2 * baseline)
    missing = inputs()
    del missing["albedo"]
    assert execute(calculator, missing).error == "missing_or_extra_inputs"
    wrong = inputs()
    wrong["orbital_radius"]["value"] = 0
    assert execute(calculator, wrong).error == "input_must_be_positive"
    wrong = inputs()
    wrong["albedo"]["unit"] = "%"
    assert execute(calculator, wrong).error == "incompatible_unit"
    for increment in (1, -1, 20, 300):
        assert (
            execute(
                calculator,
                {
                    "equilibrium_temp": {"value": 250, "unit": "K"},
                    "greenhouse_increment": {"value": increment, "unit": "K"},
                },
                "surface_temp",
            ).error
            == "invalid_greenhouse_increment_domain"
        )
    for forbidden in ("mass", "radius", "lifetime", "gas", "habitability"):
        assert execute(calculator, {}, forbidden).error == "unknown_operation"


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "0.49",
        "0.5",
        "39.99",
        "40",
        "59.99",
        "60",
        "100",
        "0.495",
        "39.995",
        "59.995",
        None,
        True,
        "nan",
        "101",
    ],
)
def test_exact_greenhouse_lookup_and_gaps_retained(calculator, value):
    assert calculator.greenhouse_reference(value) == HabitabilityCalculator().greenhouse_reference(value)


def test_in_memory_pack_mutation_stops_both_tool_and_lookup(calculator):
    calculator.pack.greenhouse_bands[0].maximum = "0.5"
    assert execute(calculator, inputs()).error == "supplied_temperature_pack_changed"
    assert calculator.greenhouse_reference("0.495")["error"] == "supplied_temperature_pack_changed"


@pytest.mark.parametrize("kind", ["stellar_luminosity", "orbital_radius", "albedo", "greenhouse_increment"])
def test_every_current_input_is_explicit_not_automatically_inferred(calculator, kind):
    item = case()
    key = next(
        k
        for k, row in item["measurements"].items()
        if row["kind"] == kind and row["source"] == "current star"
    )
    del item["measurements"][key]
    with pytest.raises(ValueError, match="explicit_input_required"):
        SuppliedTemperatureEnv(calculator, [item])


@pytest.mark.parametrize("mutation", ["class", "scope", "planet", "unit", "duplicate", "source", "domain"])
def test_environment_compatibility_and_measurement_guards(calculator, mutation):
    item = case()
    key = next(
        k
        for k, row in item["measurements"].items()
        if row["kind"] == "albedo" and row["source"] == "current star"
    )
    if mutation == "class":
        item["star_class"] = "giant"
    elif mutation == "scope":
        item["scope"] = "independent_physics_supplied_habitability_temperatures_v1"
    elif mutation == "planet":
        item["planet_class"] = "ice_giant"
    elif mutation == "unit":
        item["measurements"][key]["unit"] = "%"
    elif mutation == "duplicate":
        item["measurements"]["extra"] = dict(item["measurements"][key])
    elif mutation == "source":
        item["measurements"][key]["source"] = "reference star"
    else:
        item["measurements"][key]["value"] = 1
    with pytest.raises(ValueError):
        SuppliedTemperatureEnv(calculator, [item])


def test_legacy_and_new_scope_do_not_cross(calculator):
    legacy = habitability_cases("development", 1)[0]
    with pytest.raises(ValueError):
        SuppliedTemperatureEnv(calculator, [legacy])
    with pytest.raises(ValueError):
        HabitabilityCalculationEnv(HabitabilityCalculator(), [case()])
    with pytest.raises(ValueError):
        supplied_case(habitability_cases("train", 1)[0], "white_dwarf")
    with pytest.raises(ValueError):
        SuppliedTemperatureEnv(calculator, [case()], max_steps=129)
    with pytest.raises(ValueError):
        SuppliedTemperatureEnv(calculator, [case(), case()])


@pytest.mark.parametrize("actual", CLASSES)
def test_expert_controls_actual_class_reference_cards_reset_and_no_grade_leak(calculator, actual):
    env = environment(calculator, actual)
    first = copy.deepcopy(env.observe().model_dump(mode="json"))
    assert first["instruction"] and "current star" in first["instruction"]
    assert '"expected"' not in json.dumps(first)
    for _ in range(128):
        obs = env.observe()
        assert obs.values["star_class"] == actual and obs.progress["scope"] == SCOPE
        if obs.calculation["reference_card"]:
            assert obs.calculation["reference_card"]["applicable_classes"] == list(CLASSES)
        _, _, done, truncated, _ = env.step(env.expert_action(obs))
        if done or truncated:
            break
    assert env.completed and env.steps == 28
    assert env.metrics["invalid_actions"] == env.metrics["tool_errors"] == 0
    assert env.observe().progress["scientific_verified"] is False
    env.reset(seed=env.case["seed"])
    assert env.observe().model_dump(mode="json") == first


def test_no_automatic_operation_binding_copy_or_wrong_input_repair(calculator):
    env = environment(calculator)
    assert not env.operation and not env.bindings and not env.results
    action(env, "operation", "equilibrium_temp")
    action(env, "execute")
    assert env.tool_error == "missing_or_extra_inputs" and not env.results
    for kind in ("stellar_luminosity", "orbital_radius", "albedo"):
        source_kind = "reference star" if kind == "stellar_luminosity" else "current star"
        source = next(
            k
            for k, row in env.case["measurements"].items()
            if row["kind"] == kind and row["source"] == source_kind
        )
        action(env, "parameter", kind)
        action(env, "source", source)
        action(env, "bind")
    assert not env.results and not env.answers
    action(env, "execute")
    assert env.results["r1"]["value"] == pytest.approx(env.case["expected"]["equilibrium_temp"] * 1.1**0.25)
    assert env.metrics["input_correct"] == 2 and not env.answers


def test_inference_only_omits_class_and_preserves_authoritative_payload(calculator):
    views = []
    for actual in CLASSES:
        env = environment(calculator, actual)
        action(env, "operation", "equilibrium_temp")
        observation = env.observe()
        before = copy.deepcopy(observation.model_dump(mode="json"))
        view = inference_view(
            observation,
            expected_adapter_sha256=adapter_manifest()["sha256"],
            expected_pack_hash=calculator.pack.checksum,
        ).model_dump(mode="json")
        expected = copy.deepcopy(before)
        del expected["values"]["star_class"]
        assert view == expected and observation.model_dump(mode="json") == before
        assert observation.values["star_class"] == actual
        assert habitability_state_features(view)[-3:] == [False, False, False]
        views.append(view)
    assert all(view == views[0] for view in views)
    assert adapter_manifest()["class_relabelled_as_main_sequence"] is False
    assert adapter_manifest()["calibration_verified"] is False


@pytest.mark.parametrize("mutation", ["adapter", "expected_pack", "pack", "scope", "class", "planet"])
def test_new_adapter_and_pack_identity_required(calculator, mutation):
    observation = environment(calculator).observe()
    adapter_hash, pack_hash = adapter_manifest()["sha256"], calculator.pack.checksum
    if mutation == "adapter":
        adapter_hash = "0" * 64
    elif mutation == "expected_pack":
        pack_hash = BASE_PACK_HASH
    elif mutation == "pack":
        observation.calculation["pack_hash"] = BASE_PACK_HASH
    elif mutation == "scope":
        observation.progress["scope"] = "independent_physics_supplied_habitability_temperatures_v1"
    elif mutation == "class":
        observation.values["star_class"] = "unknown"
    else:
        observation.values["planet_class"] = "gas_giant"
    with pytest.raises(ValueError, match="mismatch"):
        inference_view(observation, expected_adapter_sha256=adapter_hash, expected_pack_hash=pack_hash)


def test_new_adapter_is_not_the_v1_final_gate():
    from habfly.training.habitability_evaluation import require_frozen_final_gate

    with pytest.raises(ValueError):
        require_frozen_final_gate(
            {"scope": SCOPE, "adapter": adapter_manifest(), "local_completed": 16}, "0" * 64
        )
