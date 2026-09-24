"""Batch orchestration tests without a browser, network, or model updates."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from habfly import browser_reliability as batch
from habfly.browser_numeric import NumericJournal
from habfly.browser_setup import consume_credentials
from habfly.contracts import RuntimeEvent
from habfly.runtime import RunOptions

EMAIL, PASSWORD = "batch@example.invalid", "batch-test-secret"


def options():
    payload = json.loads(Path("configs/browser_numeric_tui.json").read_text())
    payload.update(browser_execution="autonomous", browser_setup="automatic", paused=False)
    return RunOptions.model_validate(payload)


@pytest.fixture
def fake_runs(monkeypatch):
    instances, scenarios = [], []

    class FakeRuntime:
        def __init__(self, output):
            self.output, self.run_id, self.env = output, None, None
            self.status, self.closed, self.trace = "idle", False, None
            self.scenario = scenarios[len(instances)]
            instances.append(self)
            self.count = 0

        def emit(self, event, payload):
            line = (
                RuntimeEvent(
                    event=event, payload=payload, run_id=self.run_id, sequence=self.count
                ).model_dump_json()
                + "\n"
            )
            self.count += 1
            self.output.write(line)
            self.trace.write(line)
            self.trace.flush()

        def command(self, command):
            if command["command"] == "abort":
                self.status = "aborted"
                self.close()
                return
            assert consume_credentials() == (EMAIL, PASSWORD)
            payload = command["payload"]
            assert payload["browser_execution"] == "autonomous" and payload["stars"] == 1
            self.run_id = f"{len(instances):032x}"
            root = Path(payload["artifact_dir"])
            root.mkdir(parents=True, exist_ok=True)
            self.trace = (root / f"{self.run_id}.jsonl").open("x")
            self.journal = NumericJournal(
                root / self.run_id,
                learned_policy=True,
                provenance={
                    "browser_execution": "autonomous",
                    "optimizer_updates": 0,
                    "checkpoint_sha256": self.scenario.get("checkpoint", "a" * 64),
                    "graph_hash": "b" * 64,
                    "knowledge_pack_hash": "c" * 64,
                },
            )
            self.env = SimpleNamespace(
                outcome="operator_aborted",
                summary=None,
                session=SimpleNamespace(
                    mapping={
                        "observation": {
                            "values": {
                                "star_name": self.scenario.get("star", "Fixture Star"),
                                "measurements": {"p": {"kind": "parallax", "value": 0.05, "unit": "arcsec"}},
                            }
                        },
                    }
                ),
            )
            self.status = "running"

        def advance_if_due(self):
            kind = self.scenario.get("kind", "pass")
            if kind == "exception":
                raise RuntimeError(f"driver contained {EMAIL} {PASSWORD}")
            if kind == "interrupt":
                raise KeyboardInterrupt
            for _ in range(31):
                self.emit("action_proposed", {"action_source": "checkpoint"})
            if kind in {"pass", "corrupt", "missing_summary"}:
                for _ in range(3):
                    self.journal.emit(
                        "action_result",
                        {
                            "numeric_copy_verified": True,
                            "copy_authorization": "autonomous_opt_in",
                        },
                    )
                self.env.outcome = "numeric_transport_verified"
            else:
                self.env.outcome = "stale_numeric_observation"
                self.emit("error", {"message": self.env.outcome})
            self.status = "stopped"
            self.finish()
            if kind == "corrupt":
                self.trace.write("invalid-json\n")

        def finish(self):
            if self.env and self.env.summary is None:
                success = self.env.outcome == "numeric_transport_verified"
                self.env.summary = self.journal.finish(
                    outcome=self.env.outcome,
                    attempts=3 if success else 0,
                    verified=list(batch.FIELDS) if success else [],
                    pack_hash="c" * 64,
                )
                if self.scenario.get("kind") != "missing_summary":
                    self.emit("episode_summary", self.env.summary)

        def close(self):
            self.finish()
            if self.trace:
                self.trace.close()
            self.closed = True

    monkeypatch.setattr(batch, "Runtime", FakeRuntime)
    return scenarios, instances


def run(tmp_path, count=3):
    return batch.run_reliability(
        options(), tmp_path / "batch", runs=count, credentials=(EMAIL, PASSWORD), notify=lambda _: None
    )


def test_batch_runs_exact_budget_reports_duplicates_and_keeps_secrets_out(fake_runs, tmp_path):
    scenarios, instances = fake_runs
    scenarios.extend([{"star": "Same"}, {"star": "Same"}, {"star": "Other"}])
    result = run(tmp_path)
    assert result["all_requested_runs_passed"] and result["passed_runs"] == 3
    assert result["unique_stars"] == 2 and result["unique_measurement_cases"] == 1
    assert not result["distinct_star_target_met"] and not result["browser_acceptance_passed"]
    assert len(instances) == 3 and all(r.closed for r in instances)
    assert all(r["autonomous_copy_receipts"] == 3 for r in result["runs"])
    assert not any(os.environ.get(key) for key in batch.LOGIN_KEYS)
    for path in (tmp_path / "batch").rglob("*"):
        if path.is_file():
            assert PASSWORD not in path.read_text() and EMAIL not in path.read_text()
    assert json.loads((tmp_path / "batch/report.json").read_text()) == result
    with pytest.raises(FileExistsError):
        run(tmp_path)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("safety", "stale_numeric_observation"),
        ("exception", "runtime_exception"),
        ("interrupt", "operator_aborted"),
        ("corrupt", "artifact_verification_failed"),
        ("missing_summary", "reliability_evidence_gate_failed"),
    ],
)
def test_batch_stops_at_first_failure_and_preserves_prior_pass(fake_runs, tmp_path, kind, reason):
    scenarios, instances = fake_runs
    scenarios.extend([{}, {"kind": kind}, {}])
    result = run(tmp_path)
    assert len(instances) == 2 and all(r.closed for r in instances)
    assert result["attempted_runs"] == 2 and result["passed_runs"] == 1
    assert result["runs"][-1]["failure_reason"] == reason
    assert not result["all_requested_runs_passed"]
    assert not any(os.environ.get(key) for key in batch.LOGIN_KEYS)


def test_batch_deadline_aborts_without_retry(fake_runs, tmp_path, monkeypatch):
    scenarios, instances = fake_runs
    scenarios.extend([{}, {}, {}])
    monkeypatch.setattr(batch, "RUN_SECONDS", 0)
    result = run(tmp_path)
    assert len(instances) == 1 and instances[0].closed
    assert result["runs"][0]["failure_reason"] == "reliability_time_limit"
    assert result["passed_runs"] == 0


def test_batch_rejects_changed_checkpoint_identity(fake_runs, tmp_path):
    scenarios, instances = fake_runs
    scenarios.extend([{}, {"checkpoint": "d" * 64}, {}])
    result = run(tmp_path)
    assert len(instances) == 2 and result["passed_runs"] == 1
    assert result["runs"][-1]["failure_reason"] == "batch_provenance_changed"


@pytest.mark.parametrize("count", [0, 11])
def test_batch_rejects_out_of_budget_without_creating_output(fake_runs, tmp_path, count):
    with pytest.raises(ValueError, match="reliability runs"):
        run(tmp_path, count)
    assert not (tmp_path / "batch").exists()
    assert not fake_runs[1]


def test_batch_rejects_supervised_mode_before_any_session(tmp_path):
    with pytest.raises(ValueError, match="autonomous profile"):
        batch.run_reliability(
            options().model_copy(update={"browser_execution": "supervised"}),
            tmp_path / "no",
            runs=3,
            credentials=(EMAIL, PASSWORD),
        )
    assert not (tmp_path / "no").exists()
