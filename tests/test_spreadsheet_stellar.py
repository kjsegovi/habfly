"""Arithmetic transport is a TEST DOUBLE only; production always calls Sheets."""

import copy
import json
import math

import pytest
import torch
from pydantic import ValidationError
from typer.testing import CliRunner

from habfly.cli import app
from habfly.contracts import Action, Observation, task_completed
from habfly.data import make_demo_graph
from habfly.environments.stellar import StellarAnalysisEnv, make_case, stellar_expert
from habfly.model import ConnectomePolicy
from habfly.runtime import read_trace
from habfly.spreadsheet import (
    FORMULA_CELLS,
    INPUTS,
    OUTPUTS,
    SOURCE_ID,
    GoogleSheetsTransport,
    SpreadsheetAdapter,
    SpreadsheetConfig,
    SpreadsheetError,
    digest,
)
from habfly.training.stellar import (
    collect_demonstrations,
    load_dataset,
    record_episode,
    train_stellar,
    training_content,
)

FORMULAS = dict(
    zip(
        FORMULA_CELLS,
        [
            "=(3.26)/B2",
            "=C2*9460500000000000",
            "=A2*4*pi()*(D2^2)",
            "=E2/3.827E+26",
            "=F2^(1/3.5)",
            "=(F2^(1/2))/((M2/5800)^2)",
            "=10^10*G2^(-2.5)",
            "=2897768.5/L2",
        ],
    )
)


class FakeTransport:
    """Independent Sheets fixture. Never shipped as a runtime fallback."""

    def __init__(self):
        self.cells = {
            f"{chr(65 + c)}{r}": {"formula": None, "value": None, "entered": None, "error": False}
            for c in range(13)
            for r in (1, 2)
        }
        for key, cell in {**INPUTS, **OUTPUTS}.items():
            self.cells[cell[0] + "1"]["value"] = key
        for cell, formula in FORMULAS.items():
            self.cells[cell]["formula"] = formula
        self.writes, self.clears, self.snapshots = [], [], 0
        self.fail = False
        self.stale_inputs = False
        self.bad_output = False

    def snapshot(self):
        self.snapshots += 1
        if self.fail:
            raise SpreadsheetError("spreadsheet_access_denied")
        cells = copy.deepcopy(self.cells)
        if self.stale_inputs:
            cells["A2"]["entered"] = -1
        if self.bad_output:
            cells["C2"]["error"] = True
        return cells

    def write(self, values):
        assert set(values) <= set(INPUTS.values())
        self.writes.append(dict(values))
        for c, v in values.items():
            self.cells[c].update(entered=v, value=v)
        vals = [self.cells[c]["entered"] for c in INPUTS.values()]
        if all(v is not None and v > 0 for v in vals):
            f, p, w = vals
            distance = 3.26 / p
            luminosity = f * 4 * math.pi * (distance * 9460500000000000) ** 2 / 3.827e26
            temp = 2897768.5 / w
            mass = luminosity ** (1 / 3.5)
            computed = {
                "distance": distance,
                "luminosity": luminosity,
                "temperature": temp,
                "mass": mass,
                "radius": luminosity**0.5 / (temp / 5800) ** 2,
                "lifetime": 1e10 * mass**-2.5,
            }
            for key, val in computed.items():
                self.cells[OUTPUTS[key]].update(value=val, error=False)

    def clear(self, cells):
        assert set(cells) <= set(INPUTS.values())
        self.clears.append(list(cells))
        for c in cells:
            self.cells[c].update(value=None, entered=None)

    def close(self):
        pass


@pytest.fixture
def sheet(tmp_path):
    config = SpreadsheetConfig(
        spreadsheet_id="test_copy_" + tmp_path.name.replace("-", "_") + "x" * 20,
        credentials_path=tmp_path / "unused.json",
    )
    fake = FakeTransport()
    adapter = SpreadsheetAdapter(config, fake)
    inspection = adapter.inspect()
    config.expected_formula_fingerprint = inspection["formula_fingerprint"]
    config.expected_headers = inspection["headers"]
    yield adapter, fake
    adapter.close()


def reference_case(adapter, seed=42, split="train"):
    case = make_case(seed, split)
    case["expected"] = adapter.calculate(case["inputs"])
    return case


def test_config_boundaries_and_redaction(tmp_path):
    for kwargs in ({"spreadsheet_id": SOURCE_ID}, {"inputs": {"flux": "F2"}}, {"outputs": {"mass": "A1"}}):
        with pytest.raises(ValidationError):
            SpreadsheetConfig(
                credentials_path=tmp_path / "secret.json", **({"spreadsheet_id": "a" * 30} | kwargs)
            )
    config = SpreadsheetConfig(spreadsheet_id="a" * 30, credentials_path=tmp_path / "secret.json")
    assert "secret" not in config.model_dump_json()
    assert "secret" not in repr(config)


def test_inspect_read_only_and_pin_before_write(sheet):
    adapter, fake = sheet
    assert adapter.inspect()["formula_fingerprint"] == digest(FORMULAS)
    assert not fake.writes and not fake.clears
    adapter.config.expected_formula_fingerprint = None
    with pytest.raises(SpreadsheetError, match="pin_inspected"):
        adapter.reset()
    assert not fake.writes and not fake.clears


def test_verify_restores_numeric_zero_and_blank_inputs(sheet):
    adapter, fake = sheet
    fake.write({"A2": 0, "B2": 0.05})
    before = fake.snapshot()
    assert adapter.verify()["verified"]
    assert fake.cells["A2"]["entered"] == 0
    assert fake.cells["B2"]["entered"] == 0.05
    assert fake.cells["L2"]["entered"] is None
    for cell in FORMULA_CELLS:
        assert before[cell]["formula"] == fake.cells[cell]["formula"]
    assert not adapter.results


def test_actual_transport_rejects_non_numeric_and_formula_cell_before_network():
    transport = object.__new__(GoogleSheetsTransport)
    for vals in ({"F2": 10}, {"A2": "=1+2"}, {"A2": float("nan")}, {"A2": True}, {}):
        with pytest.raises(SpreadsheetError):
            transport.write(vals)
    with pytest.raises(SpreadsheetError):
        transport.clear(["M2"])


@pytest.mark.parametrize("bad", [0, -1, float("inf"), "1", True, None])
def test_invalid_input_not_silently_repaired(sheet, bad):
    adapter, fake = sheet
    with pytest.raises(SpreadsheetError):
        adapter.calculate({"flux": bad, "parallax": 0.03, "wavelength": 500})
    assert not fake.writes


def test_change_auth_readback_output_failures(sheet):
    adapter, fake = sheet
    inputs = {"flux": 5.15e-13, "parallax": 0.032, "wavelength": 212}
    adapter.calculate(inputs)
    fake.cells["G2"]["formula"] = "=1"
    with pytest.raises(SpreadsheetError, match="formulas_changed"):
        adapter.calculate(inputs)
    assert not adapter.results
    fake.cells["G2"]["formula"] = FORMULAS["G2"]
    fake.stale_inputs = True
    with pytest.raises(SpreadsheetError, match="readback"):
        adapter.calculate(inputs)
    fake.stale_inputs = False
    fake.bad_output = True
    with pytest.raises(SpreadsheetError, match="outputs_unavailable"):
        adapter.calculate(inputs)
    fake.bad_output = False
    fake.fail = True
    with pytest.raises(SpreadsheetError, match="access_denied"):
        adapter.calculate(inputs)


def test_single_host_worker_lock(sheet):
    adapter, fake = sheet
    adapter.acquire()
    other = SpreadsheetAdapter(adapter.config, fake)
    try:
        with pytest.raises(SpreadsheetError, match="already_running"):
            other.acquire()
    finally:
        other.close()


def test_expert_100_deterministic_fixture_cases_without_grading_leak(sheet, tmp_path):
    adapter, _ = sheet
    for seed in range(100):
        case = reference_case(adapter, seed)
        env = StellarAnalysisEnv(adapter, [case])
        first, _ = env.reset(seed=seed)
        second, _ = env.reset(seed=seed)
        # Generation increases but all task state and controls are deterministic.
        first["spreadsheet"]["generation"] = second["spreadsheet"]["generation"]
        assert first == second
        assert "expected" not in json.dumps(first)
        assert not first["spreadsheet"]["results"]
        episode, summary = record_episode(env, seed, event_path=tmp_path / f"{seed}.jsonl")
        assert summary["completed"] and summary["invalid_actions"] == 0
        assert summary["steps"] <= 35
        assert "submitted" not in episode[-1]["result"]["observation"]["progress"]
        assert read_trace(tmp_path / f"{seed}.jsonl")[-1].event == "episode_summary"


def test_stop_invalid_action_and_api_failure_not_success(sheet):
    adapter, fake = sheet
    case = reference_case(adapter)
    env = StellarAnalysisEnv(adapter, [case])
    env.reset(seed=42)
    _, _, done, _, info = env.step(Action(kind="STOP"))
    assert done and not task_completed(Observation.model_validate(info["result"]["observation"]))
    env.reset(seed=42)
    _, _, _, _, info = env.step(Action(kind="CLICK", target="0:fake"))
    assert info["metrics"]["invalid_actions"] == 1
    while len(env.bindings) < 3:
        env.step(stellar_expert(env.observe()))
    fake.fail = True
    _, _, done, truncated, info = env.step(stellar_expert(env.observe()))
    assert truncated and not done and info["metrics"]["api_failures"] == 1


def test_wrong_measurement_is_sent_to_sheet_not_corrected(sheet):
    adapter, fake = sheet
    case = reference_case(adapter)
    env = StellarAnalysisEnv(adapter, [case])
    env.reset(seed=42)
    for kind, cell in INPUTS.items():
        env.bindings[cell] = next(
            k
            for k, m in case["measurements"].items()
            if m["kind"] == kind and m["source"] == "reference star"
        )
    env.step(Action(kind="CLICK", target="0:commit"))
    assert fake.writes[-1]["A2"] == case["inputs"]["flux"] * 1.7
    assert env.outputs != case["expected"]


def test_dataset_integrity_offline_training_and_reload(sheet, tmp_path, monkeypatch):
    from habfly.training.checkpoints import load_checkpoint

    adapter, _ = sheet
    dataset = tmp_path / "dataset"
    collect_demonstrations(
        adapter.config, dataset, {"train": 2, "calibration": 2, "development": 2, "test": 2}, adapter=adapter
    )
    manifest, data = load_dataset(dataset)
    assert manifest["splits"]["gate"]["count"] == 100
    assert set(data["train"]["cases"][0]) >= {"expected", "case_id"}
    # If offline training touches the adapter, fail immediately.
    monkeypatch.setattr(
        SpreadsheetAdapter, "__init__", lambda *a, **kw: pytest.fail("Network attempted during BC")
    )
    torch.set_num_threads(1)
    graph = make_demo_graph()
    report = train_stellar(dataset, graph, tmp_path / "training", epochs=1, hidden_size=4)
    assert all(math.isfinite(v) for v in report["losses"])
    policy, saved = load_checkpoint(
        tmp_path / "training/checkpoint.pt", graph, content_pack=training_content(manifest)
    )
    assert saved["provenance"]["stellar"]["calculation_mode"] == "google_sheets"
    assert policy.hidden_size == 4
    payload = json.loads((dataset / "train.json").read_text())
    payload["cases"][0]["inputs"]["flux"] = 1
    (dataset / "train.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(dataset)


def test_spreadsheet_observation_changes_model_and_has_finite_gradient(sheet):
    adapter, _ = sheet
    case = reference_case(adapter)
    env = StellarAnalysisEnv(adapter, [case])
    env.reset(seed=case["seed"])
    first = env.observe()
    policy = ConnectomePolicy(make_demo_graph(), hidden_size=4)
    a = policy([first])
    second = first.model_copy(deep=True)
    second.spreadsheet["results"] = {"distance": 101.875}
    b = policy([second])
    assert not torch.equal(a.action_logits, b.action_logits)
    b.action_logits.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_new_cli_help():
    for command in (["spreadsheet", "inspect"], ["spreadsheet", "verify"], ["data", "demonstrations"]):
        result = CliRunner().invoke(app, [*command, "--help"])
        assert result.exit_code == 0, result.output


def test_google_transport_raw_request_and_bounded_redacted_retries(monkeypatch, tmp_path):
    import habfly.spreadsheet as module

    class Response:
        def __init__(self, status):
            self.status_code = status

        def json(self):
            return {"ok": True}

    class Session:
        def __init__(self, statuses):
            self.statuses, self.calls = iter(statuses), []

        def request(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            status = next(self.statuses)
            if status == "exception":
                raise OSError("PRIVATE_TOKEN_MUST_NOT_ESCAPE")
            return Response(status)

    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    transport = object.__new__(GoogleSheetsTransport)
    transport.config = SpreadsheetConfig(
        spreadsheet_id="a" * 30, credentials_path=tmp_path / "key", retries=2
    )
    transport.base, transport.last_request = "https://sheets.googleapis.com/test", 0
    transport.session = Session([429, 503, 200])
    transport.write({"A2": 1.2e-12})
    assert len(transport.session.calls) == 3
    body = transport.session.calls[-1][1]["json"]
    assert body == {"valueInputOption": "RAW", "data": [{"range": "'Sheet1'!A2", "values": [[1.2e-12]]}]}
    transport.session = Session(["exception"] * 3)
    with pytest.raises(SpreadsheetError, match="retries_exhausted") as error:
        transport.request("GET")
    assert "PRIVATE_TOKEN" not in str(error.value)
    transport.session = Session([403])
    with pytest.raises(SpreadsheetError, match="access_denied"):
        transport.request("GET")
    assert len(transport.session.calls) == 1


def test_google_grid_preserves_blank_zero_and_boolean(monkeypatch, tmp_path):
    transport = object.__new__(GoogleSheetsTransport)
    transport.config = SpreadsheetConfig(spreadsheet_id="b" * 30, credentials_path=tmp_path / "key")
    data = {
        "spreadsheetId": "b" * 30,
        "sheets": [
            {
                "properties": {"title": "Sheet1"},
                "data": [
                    {
                        "rowData": [
                            {"values": []},
                            {
                                "values": [
                                    {
                                        "userEnteredValue": {"numberValue": 0},
                                        "effectiveValue": {"numberValue": 0},
                                    },
                                    {
                                        "userEnteredValue": {"boolValue": True},
                                        "effectiveValue": {"boolValue": True},
                                    },
                                ]
                            },
                        ]
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(transport, "request", lambda *args, **kwargs: data)
    cells = transport.snapshot()
    assert cells["A2"]["entered"] == 0
    assert cells["B2"]["entered"] is True
    assert cells["L2"]["entered"] is None


def test_runtime_stellar_step_pause_abort_and_offline_replay(sheet, tmp_path, monkeypatch):
    import io

    import habfly.spreadsheet as spreadsheet_module
    import habfly.training.stellar as training_module
    from habfly.runtime import Runtime

    adapter, _ = sheet
    case = reference_case(adapter, 300000, "test")
    manifest = {"content": adapter.config.content_identity(), "splits": {}}
    monkeypatch.setattr(training_module, "load_dataset", lambda _: (manifest, {"test": {"cases": [case]}}))
    monkeypatch.setattr(spreadsheet_module, "load_spreadsheet_config", lambda _: adapter.config)
    monkeypatch.setattr(spreadsheet_module, "SpreadsheetAdapter", lambda _: adapter)
    runtime = Runtime(io.StringIO())
    runtime.start(
        {
            "task": "stellar",
            "seed": 300000,
            "policy": "expert",
            "paused": True,
            "dataset": "unused",
            "spreadsheet_config": "unused",
            "artifact_dir": str(tmp_path),
        }
    )
    assert runtime.status == "paused"
    runtime.command({"command": "step"})
    assert runtime.env.steps == 1 and runtime.status == "paused"
    runtime.command({"command": "abort"})
    trace = runtime.trace_path
    assert any(e.event == "observation" and e.payload.get("spreadsheet") for e in read_trace(trace))
    monkeypatch.setattr(
        spreadsheet_module, "SpreadsheetAdapter", lambda _: pytest.fail("Replay contacted Sheets")
    )
    runtime.command({"command": "replay", "payload": {"path": str(trace)}})
    while runtime.status == "running":
        runtime.tick()
    assert runtime.status == "completed"
    runtime.close()
    with pytest.raises(ValueError, match="cannot run"):
        runtime.start(
            {"task": "stellar", "environment": "browser", "policy": "checkpoint", "checkpoint": "unused"}
        )


def test_live_evaluation_counts_policy_and_transport_failures_separately(sheet, tmp_path, monkeypatch):
    import habfly.training.stellar as training_module

    adapter, fake = sheet
    cases = [reference_case(adapter, 300000 + i, "test") for i in range(2)]
    manifest = {"content": adapter.config.content_identity(), "splits": {}}
    monkeypatch.setattr(training_module, "load_dataset", lambda _: (manifest, {"test": {"cases": cases}}))

    class StopPolicy:
        def __init__(self):
            self.calibration = {"status": "uncalibrated"}

        def act(self, observation, state):
            return Action(kind="STOP"), None, {}

    report = training_module.evaluate_stellar(
        StopPolicy(), adapter.config, "unused", tmp_path / "policy-failed", adapter=adapter
    )
    assert report["completion_rate"] == 0 and report["api_failures"] == 0 and not report["gate_passed"]
    assert all(e["failure_reason"] == "policy_stopped" for e in report["episodes"])
    fake.fail = True
    report = training_module.evaluate_stellar(
        StopPolicy(), adapter.config, "unused", tmp_path / "api-failed", adapter=adapter
    )
    assert report["api_failures"] == 1 and report["completion_rate"] == 0 and not report["gate_passed"]


def test_reset_preserves_unrelated_cells_and_stale_targets_do_not_mutate(sheet):
    adapter, fake = sheet
    case = reference_case(adapter)
    fake.cells["J2"].update(entered=777, value=777)
    env = StellarAnalysisEnv(adapter, [case])
    env.reset(seed=42)
    assert fake.cells["J2"]["value"] == 777
    action = stellar_expert(env.observe())
    env.step(Action(kind="WAIT"))
    _, _, _, _, info = env.step(action)
    assert info["result"]["failure_reason"] == "stale_observation"
    assert not env.bindings and not env.outputs


def test_documented_collection_command_accepts_profile_option(sheet, tmp_path, monkeypatch):
    import habfly.spreadsheet as spreadsheet_module
    import habfly.training.stellar as training_module

    adapter, _ = sheet
    profile = tmp_path / "profile.yaml"
    profile.write_text("task: stellar\nspreadsheet_config: unused.json\ntraining_episodes: 4\n")
    monkeypatch.setattr(spreadsheet_module, "load_spreadsheet_config", lambda _: adapter.config)
    monkeypatch.setattr(training_module, "collect_demonstrations", lambda *a, **k: {"configured": True})
    result = CliRunner().invoke(
        app, ["data", "demonstrations", str(tmp_path / "out"), "--task", "stellar", "--profile", str(profile)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["configured"]
