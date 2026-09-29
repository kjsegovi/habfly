"""Additive arithmetic/adapter contract, not native or scientific acceptance."""

import copy
import hashlib
import json

import pytest

from habfly.contracts import Action
from habfly.environments.planet_calculations import PlanetCalculationEnv, planet_cases
from habfly.model.planet_tool_state import planet_state_features
from habfly.planet_knowledge import DEFAULT_PLANET_PACK, PlanetCalculator, load_planet_pack
from habfly.planet_supplied_inputs import (
    BASE_FILE_SHA256,
    BASE_PACK_HASH,
    CLASSES,
    SCOPE,
    SuppliedPlanetCalculator,
    SuppliedPlanetEnv,
    SuppliedPlanetPack,
    adapter_manifest,
    inference_view,
    load_supplied_planet_pack,
    supplied_case,
)


@pytest.fixture
def calculator():
    return SuppliedPlanetCalculator()


def case(star_class="white_dwarf"):
    return supplied_case(planet_cases("development", 1)[0], star_class)


def environment(calculator, star_class="white_dwarf"):
    item = case(star_class)
    env = SuppliedPlanetEnv(calculator, [item])
    env.reset(seed=item["seed"])
    return env


def execute(calculator, operation, bindings, star_class="white_dwarf"):
    return calculator.execute(
        operation, bindings, star_class=star_class, assumptions=calculator.pack.assumptions
    )


def action(env, key, value=None):
    obs = env.observe()
    target = next(c.id for c in obs.controls if c.id.split(":", 1)[1] == key)
    return env.step(
        Action(
            kind="SELECT" if value is not None else "CLICK",
            target=target,
            value=value,
            observation_revision=obs.revision,
        )
    )


def test_six_operations_sources_and_original_hashes_exactly_preserved():
    original, pack = load_planet_pack(), load_supplied_planet_pack()
    assert hashlib.sha256(DEFAULT_PLANET_PACK.read_bytes()).hexdigest() == BASE_FILE_SHA256
    assert original.checksum == BASE_PACK_HASH != pack.checksum
    assert len(pack.operations) == 6
    for previous, current in zip(original.operations, pack.operations, strict=True):
        expected = previous.model_dump(mode="json")
        expected["applicable_classes"] = list(CLASSES)
        assert current.model_dump(mode="json") == expected
    for field in ("assumptions", "source_conflicts", "provenance", "pending"):
        assert getattr(pack, field) == getattr(original, field)
    assert "H2" in pack.source_conflicts[0]["original_formula"]
    assert pack.declaration["input_authority"] == "explicit_supplied_current_star_measurements_only"
    assert "native_browser_enablement" in pack.declaration["excluded"]
    assert SuppliedPlanetPack.model_validate_json(pack.model_dump_json()).checksum == pack.checksum


@pytest.mark.parametrize(
    "field",
    [
        "expression",
        "constant",
        "unit",
        "source",
        "input",
        "class",
        "assumptions",
        "source_conflicts",
        "pending",
        "provenance",
        "base_hash",
        "declaration",
        "schema_bool",
        "positive_number",
    ],
)
def test_pack_mutation_or_type_coercion_cannot_retain_source_identity(field):
    raw = load_supplied_planet_pack().model_dump(mode="json")
    op = raw["operations"][0]
    if field == "expression":
        op["expression"] += " + 1"
    elif field == "constant":
        op["constants"]["days_per_year"] = 365.25
    elif field == "unit":
        op["output_unit"] = "day"
    elif field == "source":
        op["sources"][0]["claim"] = "invented"
    elif field == "input":
        op["inputs"]["period_days"]["quantity"] = "stellar_radius"
    elif field == "class":
        op["applicable_classes"] = ["main_sequence"]
    elif field in {"assumptions", "pending"}:
        raw[field].append("invented")
    elif field == "source_conflicts":
        raw[field] = []
    elif field == "provenance":
        raw[field]["date"] = "invented"
    elif field == "base_hash":
        raw["declaration"]["base_pack_hash"] = "0" * 64
    elif field == "declaration":
        raw["declaration"]["input_authority"] = "guessed"
    elif field == "schema_bool":
        raw["schema_version"] = True
    else:
        op["inputs"]["period_days"]["positive"] = 1
    with pytest.raises(ValueError, match="math_or_provenance_changed"):
        SuppliedPlanetPack.model_validate(raw)


def test_original_and_extension_file_changes_rejected(monkeypatch, tmp_path):
    import habfly.planet_supplied_inputs as module

    changed = tmp_path / "changed.json"
    changed.write_text("{}")
    with monkeypatch.context() as context:
        context.setattr(module, "DEFAULT_PLANET_PACK", changed)
        with pytest.raises(ValueError, match="base_source_changed"):
            load_supplied_planet_pack()
    with monkeypatch.context() as context:
        context.setattr(module, "PACK_PATH", changed)
        with pytest.raises(ValueError, match="declaration_changed"):
            load_supplied_planet_pack()


def test_independent_goldens_never_promote_course_or_native_scope(calculator, monkeypatch):
    import socket

    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: pytest.fail("network used"))
    report = calculator.verify()
    assert report["valid"] and report["operation_golden_cases"] == 24
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
def test_actual_class_reaches_tool_unchanged(calculator, monkeypatch, star_class):
    seen = []
    original = calculator._execute

    def traced(*args, **kwargs):
        seen.append(kwargs["star_class"])
        return original(*args, **kwargs)

    monkeypatch.setattr(calculator, "_execute", traced)
    result = execute(calculator, "period_years", {"period_days": {"value": 365, "unit": "day"}}, star_class)
    assert result.ok and result.value == 1 and seen == [star_class]


@pytest.mark.parametrize("assumptions", [None, [], [True], [float("nan")], "circular_orbit", {}])
def test_assumptions_required_and_not_guessed(calculator, assumptions):
    result = calculator.execute("period_years", {}, star_class="white_dwarf", assumptions=assumptions)
    assert result.error == "planet_assumptions_not_explicit"


def test_assumptions_exact_list_and_unsupported_class(calculator):
    for assumptions in (
        tuple(calculator.pack.assumptions),
        calculator.pack.assumptions[::-1],
        calculator.pack.assumptions + calculator.pack.assumptions[:1],
    ):
        result = calculator.execute("period_years", {}, star_class="red_giant", assumptions=assumptions)
        assert result.error == "planet_assumptions_not_explicit"
    for label in (None, True, "giant", "unknown"):
        assert execute(calculator, "period_years", {}, label).error == "unsupported_supplied_class"


@pytest.mark.parametrize(
    "value,error",
    [
        (None, "input_not_finite_number"),
        (True, "input_not_finite_number"),
        ("365", "input_not_finite_number"),
        (float("inf"), "input_not_finite_number"),
        (float("nan"), "input_not_finite_number"),
        (10**400, "input_not_finite_number"),
        (0, "input_must_be_positive"),
        (-1, "input_must_be_positive"),
    ],
)
def test_numeric_domains_not_repaired(calculator, value, error):
    assert (
        execute(calculator, "period_years", {"period_days": {"value": value, "unit": "day"}}).error == error
    )


def test_units_missing_extra_depth_and_no_stellar_relations(calculator):
    assert execute(calculator, "period_years", {}).error == "missing_or_extra_inputs"
    assert (
        execute(calculator, "period_years", {"period_days": {"value": 365, "unit": "yr"}}).error
        == "incompatible_unit"
    )
    assert (
        execute(calculator, "period_years", {"period_days": {"value": 365, "unit": "day"}, "extra": {}}).error
        == "missing_or_extra_inputs"
    )
    assert (
        execute(
            calculator,
            "planet_radius",
            {"brightness_drop": {"value": 101, "unit": "%"}, "stellar_radius": {"value": 1, "unit": "Rsun"}},
        ).error
        == "brightness_drop_exceeds_100_percent"
    )
    for forbidden in ("mass", "radius", "lifetime"):
        assert execute(calculator, forbidden, {}).error == "unknown_operation"
    calculator.pack.operations[0].constants["days_per_year"] = 365.25
    assert execute(calculator, "period_years", {}).error == "supplied_planet_pack_changed"


@pytest.mark.parametrize("mutation", ["missing", "zero", "unit", "source", "duplicate"])
def test_both_current_stellar_inputs_required_not_inferred(calculator, mutation):
    item = case()
    key = next(
        k
        for k, row in item["measurements"].items()
        if row["kind"] == "stellar_radius" and row["source"] == "current star"
    )
    if mutation == "missing":
        del item["measurements"][key]
    elif mutation == "zero":
        item["measurements"][key]["value"] = 0
    elif mutation == "unit":
        item["measurements"][key]["unit"] = "REarth"
    elif mutation == "source":
        item["measurements"][key]["source"] = "reference star"
    else:
        item["measurements"]["extra"] = dict(item["measurements"][key])
    with pytest.raises(ValueError, match="explicit_stellar_input_required"):
        SuppliedPlanetEnv(calculator, [item])


def test_old_and_new_scope_cases_are_not_silently_interchangeable(calculator):
    legacy = planet_cases("development", 1)[0]
    with pytest.raises(ValueError, match="case_scope"):
        SuppliedPlanetEnv(calculator, [legacy])
    with pytest.raises(ValueError, match="Incompatible"):
        PlanetCalculationEnv(PlanetCalculator(), [case()])
    with pytest.raises(ValueError, match="development_fixture"):
        supplied_case(planet_cases("train", 1)[0], "white_dwarf")
    with pytest.raises(ValueError, match="development_fixture"):
        supplied_case(legacy, "giant")
    with pytest.raises(ValueError, match="environment_contract"):
        SuppliedPlanetEnv(calculator, [case()], max_steps=129)
    with pytest.raises(ValueError, match="duplicate"):
        SuppliedPlanetEnv(calculator, [case(), case()])


@pytest.mark.parametrize("star_class", CLASSES)
def test_expert_uses_visible_controls_and_actual_class_with_no_grade_leak(calculator, star_class):
    env = environment(calculator, star_class)
    initial = copy.deepcopy(env.observe().model_dump(mode="json"))
    assert initial["instruction"] and "supplied observations" in initial["instruction"]
    assert initial["values"]["star_class"] == star_class
    assert '"expected"' not in json.dumps(initial)
    assert '"expected_provenance"' not in json.dumps(initial)
    for _ in range(128):
        obs = env.observe()
        assert obs.values["star_class"] == star_class
        assert obs.progress["scientific_verified"] is False
        if obs.calculation["reference_card"]:
            assert obs.calculation["reference_card"]["applicable_classes"] == list(CLASSES)
        _, _, done, truncated, _ = env.step(env.expert_action(obs))
        if done or truncated:
            break
    assert env.completed and env.steps == 62
    assert env.metrics["invalid_actions"] == env.metrics["tool_errors"] == 0
    assert env.observe().progress["course_acceptance_passed"] is False
    env.reset(seed=env.case["seed"])
    assert env.observe().model_dump(mode="json") == initial


def test_missing_binding_and_distractor_choices_not_corrected(calculator):
    env = environment(calculator, "supergiant")
    action(env, "operation", "period_years")
    assert not env.results and not env.answers
    action(env, "execute")
    assert env.tool_error == "missing_or_extra_inputs" and not env.results
    wrong = next(
        k
        for k, row in env.case["measurements"].items()
        if row["kind"] == "period_days" and row["source"] == "reference star"
    )
    action(env, "parameter", "period_days")
    action(env, "source", wrong)
    action(env, "bind")
    assert not env.results
    action(env, "execute")
    assert env.results["r1"]["value"] == pytest.approx(env.case["measurements"][wrong]["value"] / 365)
    assert env.metrics["input_correct"] == 0 and not env.answers


def test_inference_removes_only_class_and_preserves_all_authoritative_content(calculator):
    views = []
    for star_class in CLASSES:
        env = environment(calculator, star_class)
        action(env, "operation", "orbital_radius")
        observation = env.observe()
        before = copy.deepcopy(observation.model_dump(mode="json"))
        view = inference_view(
            observation,
            expected_adapter_sha256=adapter_manifest()["sha256"],
            expected_pack_hash=calculator.pack.checksum,
        )
        after = view.model_dump(mode="json")
        expected = copy.deepcopy(before)
        del expected["values"]["star_class"]
        assert after == expected
        assert observation.model_dump(mode="json") == before
        assert observation.values["star_class"] == star_class
        assert planet_state_features(after)[-3:] == [False, False, False]
        views.append(after)
    assert all(view == views[0] for view in views)
    assert adapter_manifest()["calibration_verified"] is False
    assert adapter_manifest()["class_relabelled_as_main_sequence"] is False


@pytest.mark.parametrize("mutation", ["adapter", "expected_pack", "pack", "scope", "class"])
def test_inference_requires_explicit_new_provenance(calculator, mutation):
    obs = environment(calculator).observe()
    adapter_hash, pack_hash = adapter_manifest()["sha256"], calculator.pack.checksum
    if mutation == "adapter":
        adapter_hash = "0" * 64
    elif mutation == "expected_pack":
        pack_hash = BASE_PACK_HASH
    elif mutation == "pack":
        obs.calculation["pack_hash"] = BASE_PACK_HASH
    elif mutation == "scope":
        obs.progress["scope"] = "independent_physics_planet_derived_four_field_v1"
    else:
        obs.values["star_class"] = "unknown"
    with pytest.raises(ValueError, match="mismatch"):
        inference_view(obs, expected_adapter_sha256=adapter_hash, expected_pack_hash=pack_hash)


def test_adapter_is_not_a_v1_final_acceptance_report():
    from habfly.training.planet_evaluation import require_frozen_final_gate

    with pytest.raises(ValueError):
        require_frozen_final_gate(
            {"scope": SCOPE, "adapter": adapter_manifest(), "local_completed": 16}, "0" * 64
        )
