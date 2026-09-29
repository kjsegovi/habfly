"""Preparation gates only: synthetic TEST-band traces, no frozen100-policy run.

No test generates the planned30M/31M evaluation values. Full evidence-reader
tests substitute a90M seed schedule and use a reference expert, never weights.
"""

import copy
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from habfly.contracts import Action
from habfly.training import supplied_input_transfer as transfer

ROOT = Path(__file__).resolve().parents[1]
TEST_SEED = 90_000_000


@pytest.mark.parametrize(
    "raw", ['{"scope":1,"scope":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":{"y":1,"y":1}}']
)
def test_strict_json_rejects_ambiguous_and_nonfinite_evidence(raw):
    with pytest.raises(ValueError):
        transfer._json(raw)


def test_schedule_contains_only_metadata_and_disjoint_bands():
    for task in ("planet", "temperature"):
        rows = transfer.schedule(task)
        assert len(rows) == len({row["seed"] for row in rows}) == 100
        assert Counter(row["star_class"] for row in rows) == dict.fromkeys(transfer.CLASSES, 25)
        assert all(set(row) == {"seed", "star_class"} for row in rows)
        _, legacy, _, _ = transfer._config(task)
        assert transfer.TEMPLATES[task] not in legacy.TEMPLATES.values()
        assert not {row["seed"] for row in rows}.intersection(
            row["seed"] for row in transfer.schedule("temperature" if task == "planet" else "planet")
        )


@pytest.mark.parametrize("task", ["planet", "temperature"])
def test_parent_metadata_seed_disjointness_without_final_cases(task):
    _, _, _, key = transfer._config(task)
    pilot = "planet" if task == "planet" else "habitability"
    content = json.loads((ROOT / f"experiments/{pilot}-pilot-001/training/checkpoint.pt.json").read_bytes())[
        "provenance"
    ][key]
    transfer._disjoint(task, content)
    content["splits"]["development"]["seeds"].append(transfer.SEED_START[task])
    with pytest.raises(ValueError, match="parent_seed_overlap"):
        transfer._disjoint(task, content)


@pytest.mark.parametrize("task", ["planet", "temperature"])
def test_reservation_precedes_materialization_and_survives_changed_output(tmp_path, monkeypatch, task):
    calls = []
    monkeypatch.setattr(transfer, "_inverse_case", lambda *args: calls.append(args))
    path, reservation = transfer._reserve(
        tmp_path / "registry", task, tmp_path / "first", {"fake_test_identity": 1}
    )
    assert calls == []
    transfer._materialize(task, reservation, path)
    assert len(calls) == 100
    with pytest.raises(FileExistsError):
        transfer._reserve(tmp_path / "registry", task, tmp_path / "second", {"fake_test_identity": 2})
    reservation["automatic_retry"] = 0
    with pytest.raises(ValueError, match="reservation_changed"):
        transfer._materialize(task, reservation, path)


@pytest.mark.parametrize("actual_class", transfer.CLASSES)
def test_declared_stress_strata_keep_valid_depth_without_clipping(actual_class):
    for index in (0, 4, 5, 24):
        case = transfer._inverse_case("planet", TEST_SEED + index * 4, actual_class)
        transfer._recorded_case("planet", case)
        inputs = {
            row["kind"]: row["value"]
            for row in case["measurements"].values()
            if row["source"] == "current star"
        }
        assert 0 < inputs["brightness_drop"] <= 100
        if actual_class != "main_sequence" and index < 5:
            assert [inputs["stellar_mass"], inputs["stellar_radius"]] == transfer.PLANET_STRATA[actual_class][
                "anchor"
            ]


def trace(task, seed=TEST_SEED, actual_class="white_dwarf", stop=False):
    module, _, calculator_type, _ = transfer._config(task)
    case = transfer._inverse_case(task, seed, actual_class)
    calculator = calculator_type()
    env = transfer._env(task, calculator, case)
    env.reset(seed=seed)
    adapter = module.adapter_manifest()
    events = []
    for step in range(128):
        observation = env.observe()
        before = observation.model_dump(mode="json")
        view = module.inference_view(
            observation,
            expected_adapter_sha256=adapter["sha256"],
            expected_pack_hash=calculator.pack.checksum,
        )
        reference = env.expert_action(observation)
        action = Action(kind="STOP") if stop else reference
        _, _, done, truncated, info = env.step(action)
        events.append(
            {
                "step": step,
                "observation": before,
                "inference_view_sha256": transfer._digest(view.model_dump(mode="json")),
                "action": action.model_dump(mode="json"),
                "reference_action": reference.model_dump(mode="json"),
                "matches_expert": transfer._identity(action) == transfer._identity(reference),
                "result": info["result"],
                "failure_reason": info["result"]["failure_reason"],
                "tool_error": env.tool_error,
            }
        )
        if done or truncated:
            break
    row = transfer._summary(case, env, len(events), sum(event["matches_expert"] for event in events))
    env.close()
    return case, events, row, {"pack_hash": calculator.pack.checksum, "adapter_sha256": adapter["sha256"]}


@pytest.fixture(scope="module", params=["planet", "temperature"])
def recorded(request):
    return (request.param, *trace(request.param))


def test_replay_uses_saved_evidence_never_regenerates_or_executes(recorded, monkeypatch):
    task, case, events, row, identity = recorded

    def forbidden(*args, **kwargs):
        raise AssertionError("reader must not execute policy, expert, environment or case generator")

    for key in ("_env", "_inverse_case", "_materialize", "load_checkpoint"):
        monkeypatch.setattr(transfer, key, forbidden)
    assert transfer._recorded_episode(task, case, events, **identity) == row
    assert events[0]["observation"]["values"]["star_class"] == "white_dwarf"
    assert events[0]["observation"]["values"]["answers"] == {}
    assert all("expected" not in event["observation"]["values"] for event in events)


@pytest.mark.parametrize(
    "mutation",
    [
        "class",
        "measurement",
        "revision",
        "inference",
        "confidence",
        "reference",
        "completion",
        "units",
        "answer",
        "early_stop",
        "bool_step",
        "truncated",
        "tool_error",
    ],
)
def test_trace_corruption_rejected(recorded, mutation):
    task, case, original, _, identity = recorded
    events = copy.deepcopy(original)
    if mutation == "class":
        events[0]["observation"]["values"]["star_class"] = "main_sequence"
    elif mutation == "measurement":
        events[0]["observation"]["values"]["measurements"]["m0"]["value"] *= 2
    elif mutation == "revision":
        events[1]["observation"]["revision"] += 1
    elif mutation == "inference":
        events[0]["inference_view_sha256"] = "0" * 64
    elif mutation == "confidence":
        events[0]["action"]["calibrated"] = True
    elif mutation == "reference":
        events[0]["reference_action"]["value"] = "not_a_calculation"
    elif mutation == "completion":
        events[0]["result"]["observation"]["progress"]["task_completed"] = True
    elif mutation == "units":
        events[-1]["result"]["observation"]["values"]["units"] = {}
    elif mutation == "answer":
        events[-1]["result"]["observation"]["values"]["answers"] = {}
    elif mutation == "early_stop":
        events[0]["result"]["terminated"] = True
    elif mutation == "bool_step":
        events[0]["step"] = False
    elif mutation == "truncated":
        events[-1]["result"]["truncated"] = True
    elif mutation == "tool_error":
        events[0]["tool_error"] = "changed"
    with pytest.raises((ValueError, AssertionError)):
        transfer._recorded_episode(task, case, events, **identity)


@pytest.mark.parametrize("task", ["planet", "temperature"])
def test_termination_is_not_completion(task):
    case, events, row, identity = trace(task, stop=True)
    assert transfer._recorded_episode(task, case, events, **identity) == row
    assert row["completed"] is False and row["steps"] == 1 and row["exact_actions"] == 0


@pytest.mark.parametrize(
    "completed,invalid,tool,infrastructure,passed",
    [
        (90, 0, 0, 0, True),
        (89, 0, 0, 0, False),
        (100, 1, 0, 0, False),
        (100, 0, 1, 0, False),
        (100, 0, 0, 1, False),
    ],
)
def test_fixed_gate_threshold(completed, invalid, tool, infrastructure, passed):
    assert (
        transfer._passed(
            {
                "episodes": 100,
                "completed": completed,
                "invalid_actions": invalid,
                "tool_errors": tool,
                "infrastructure_failures": infrastructure,
            }
        )
        is passed
    )


@pytest.mark.parametrize("task", ["planet", "temperature"])
def test_sources_pin_original_parent_and_new_scopes_separately(task):
    pilot = "planet" if task == "planet" else "habitability"
    checkpoint = ROOT / f"experiments/{pilot}-pilot-001/training/checkpoint.pt"
    paths = transfer._source_paths(task, checkpoint, checkpoint.with_suffix(".pt.json"))
    assert str(checkpoint.resolve()) in paths
    assert str(Path(transfer.__file__).resolve()) in paths
    assert any("/model/" in path for path in paths)
    assert any("supplied_inputs_v1.json" in path for path in paths)
    assert any("physical_candidate.json" in path for path in paths)


@pytest.mark.parametrize(
    "task,same,changed",
    [("planet", False, {"policy.py", "habitability_tool_state.py"}), ("temperature", True, set())],
)
def test_explicit_execution_code_is_not_false_training_code_equality(task, same, changed):
    pilot = "planet" if task == "planet" else "habitability"
    checkpoint = ROOT / f"experiments/{pilot}-pilot-001/training/checkpoint.pt"
    parent = json.loads(checkpoint.with_suffix(".pt.json").read_bytes())
    module, _, _, _ = transfer._config(task)
    sources = transfer._sources(task, checkpoint, checkpoint.with_suffix(".pt.json"))
    result = transfer._parent_model(parent, module.adapter_manifest(), sources)
    assert result["same_as_training"] is same
    assert {Path(name).name for name in result["delta"]} == changed
    assert result["execution_map_sha256"] == transfer.EXECUTION_MODEL_MAP_SHA256
    assert result["historical_code_equivalence_claimed"] is False
    assert result["calibration_transferred"] is False
    sources[str(ROOT / "src/habfly/model/policy.py")] = "0" * 64
    with pytest.raises(ValueError, match="execution_model_sources_changed"):
        transfer._parent_model(parent, module.adapter_manifest(), sources)


def test_complete_reader_and_owned_relocation(tmp_path, monkeypatch):
    """Real source/trajectory reader, TEST-only schedule and reference actions.

    Fake checkpoint BYTES are intentionally never interpreted as a model; real
    recorded parent metadata is used solely to check its content identity.
    """
    task = "temperature"
    schedule = [{"seed": TEST_SEED + i, "star_class": transfer.CLASSES[i % 4]} for i in range(100)]
    monkeypatch.setattr(transfer, "schedule", lambda which: copy.deepcopy(schedule))
    module, _, calculator_type, _ = transfer._config(task)
    checkpoint = tmp_path / "test-only-parent.pt"
    checkpoint.write_bytes(b"TEST ONLY, NOT MODEL WEIGHTS")
    metadata = checkpoint.with_suffix(".pt.json")
    metadata.write_bytes(
        (ROOT / "experiments/habitability-pilot-001/training/checkpoint.pt.json").read_bytes()
    )
    parent = json.loads(metadata.read_bytes())
    adapter = module.adapter_manifest()
    identity = {
        "parent_checkpoint_path": str(checkpoint),
        "parent_metadata_path": str(metadata),
        "checkpoint_sha256": transfer._sha(checkpoint),
        "parent_metadata_sha256": transfer._sha(metadata),
        "parent_content_hash": parent["content_pack_hash"],
        "parent_pack_hash": module.BASE_PACK_HASH,
        "parent_scope": module.LEGACY_SCOPE,
        "graph_hash": parent["graph_hash"],
        "pack_hash": calculator_type().pack.checksum,
        "adapter_sha256": adapter["sha256"],
        "source_sha256": transfer._sources(task, checkpoint, metadata),
    }
    identity["model_source_provenance"] = transfer._parent_model(parent, adapter, identity["source_sha256"])
    directory = tmp_path / "synthetic-gate"
    directory.mkdir()
    reservation, _ = transfer._reserve(tmp_path / "registry", task, directory, identity)
    cases, rows = [], []
    for index, planned in enumerate(schedule):
        case, events, row, _ = trace(task, planned["seed"], planned["star_class"])
        path = directory / f"episode-{index:03d}.jsonl"
        path.write_text("".join(json.dumps(event, allow_nan=False) + "\n" for event in events))
        row.update(trace=path.name, trace_sha256=transfer._sha(path))
        transfer._write(directory / f"episode-{index:03d}-summary.json", row)
        cases.append(case)
        rows.append(row)
    transfer._write(directory / "private-transfer-cases.json", cases)
    manifest = {
        "scope": transfer.SCOPE,
        "version": 1,
        "task": task,
        "budget": transfer.BUDGET,
        "identity": identity,
        "adapter": adapter,
        "input_scope": module.SCOPE,
        "reservation": str(reservation),
        "reservation_sha256": transfer._sha(reservation),
        "cases_sha256": transfer._sha(directory / "private-transfer-cases.json"),
        "instruction": transfer.TEMPLATES[task],
        "instruction_sha256": transfer._digest(transfer.TEMPLATES[task]),
        "numeric_strata": None,
        "capability_rationale": {},
        "automatic_retry": False,
        "verification": {
            "valid": True,
            "operation_golden_cases": 24,
            "pack_hash": identity["pack_hash"],
            "scope": module.SCOPE,
            "course_acceptance_passed": False,
            "scientific_verified": False,
            "native_browser_enabled": False,
            "training_oracle_enabled": False,
        },
    }
    transfer._write(directory / "manifest.json", manifest)
    report = {
        "scope": transfer.SCOPE,
        "version": 1,
        "task": task,
        "input_scope": module.SCOPE,
        "budget": transfer.BUDGET,
        "identity": identity,
        "manifest_sha256": transfer._sha(directory / "manifest.json"),
        "rows": rows,
        "scores": transfer._scores(rows),
        "per_class": {
            name: transfer._scores([row for row in rows if row["actual_star_class"] == name])
            for name in transfer.CLASSES
        },
        "transfer_gate_passed": True,
        "weights_unchanged": True,
        "sources_unchanged": True,
        "calibration": {
            "status": "uncalibrated",
            "scope": transfer.SCOPE,
            "reason": "new_supplied_input_adapter",
        },
        "elapsed_seconds": 1,
        "optimizer_updates": 0,
        **dict.fromkeys(transfer.FALSE_CLAIMS, False),
    }
    transfer._write(directory / "report.json", report)

    def forbidden(*args, **kwargs):
        raise AssertionError("Consumer must not execute generator/environment/model")

    for name in ("_materialize", "_inverse_case", "_env", "load_checkpoint"):
        monkeypatch.setattr(transfer, name, forbidden)
    args = {key: identity[key] for key in ("checkpoint_sha256", "graph_hash", "pack_hash", "adapter_sha256")}
    validated = transfer.require_supplied_input_transfer_gate(directory, task=task, **args)
    assert validated["report"] == report and validated["scores"]["completed"] == 100
    assert validated["current_sources_verified"] is True
    assert len(validated["artifact_sha256"]) == 204
    artifact_paths, source_paths = {}, {}
    archive = tmp_path / "archive"
    archive.mkdir()
    for kind, pins, mapping in (
        ("artifact", validated["artifact_sha256"], artifact_paths),
        ("source", validated["source_sha256"], source_paths),
    ):
        for index, path in enumerate(pins):
            destination = archive / f"{kind}-{index}"
            shutil.copyfile(path, destination)
            mapping[path] = str(destination)
    # Originals are unavailable; neither fallback artifacts nor checkpoint paths
    # may be opened by historical verification.
    checkpoint.unlink()
    metadata.unlink()
    shutil.move(str(directory), str(tmp_path / "unavailable-original"))
    historical = transfer.require_supplied_input_transfer_gate(
        directory, task=task, artifact_paths=artifact_paths, source_paths=source_paths, **args
    )
    assert historical["report"] == report
    assert historical["current_sources_verified"] is False
    assert historical["historical_provenance_verified"] is True
    assert all(Path(path).parent == archive for path in historical["artifact_sha256"])
    missing = dict(artifact_paths)
    del missing[str(directory / "private-transfer-cases.json")]
    with pytest.raises(ValueError, match="relocation_map_missing"):
        transfer.require_supplied_input_transfer_gate(
            directory, task=task, artifact_paths=missing, source_paths=source_paths, **args
        )
    Path(source_paths[str(metadata)]).write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        transfer.require_supplied_input_transfer_gate(
            directory, task=task, artifact_paths=artifact_paths, source_paths=source_paths, **args
        )
