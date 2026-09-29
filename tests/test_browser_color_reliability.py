"""Color batch budgets/evidence isolation without live browser or training."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_color_policy import trained_artifacts  # noqa: F401
from test_browser_numeric import OUTER, WIDGET, chromium, config, stellar_html  # noqa: F401
from test_browser_setup import canvas_html

from habfly import browser_color_reliability as color_batch
from habfly import browser_reliability as batch
from habfly.browser import BrowserSafetyStop
from habfly.browser_color import ColorJournal
from habfly.browser_color_policy import BrowserColorPolicyBridge
from habfly.browser_setup import consume_credentials
from habfly.browser_stellar import SIMULATION_URL
from habfly.color_reference import ColorReference, load_color_reference
from habfly.contracts import RuntimeEvent
from habfly.runtime import RunOptions

EMAIL, PASSWORD = "color-batch@example.invalid", "fixture-only-secret"


def options():
    payload = json.loads(Path("configs/browser_color_tui.json").read_text())
    payload.update(browser_execution="autonomous", paused=False)
    return RunOptions.model_validate(payload)


def identity():
    return {
        "checkpoint_sha256": "a" * 64,
        "graph_hash": "b" * 64,
        "color_reference_hash": load_color_reference().checksum,
        "final_report_sha256": "c" * 64,
    }


@pytest.fixture
def fake_runs(monkeypatch):
    scenarios, instances = [], []

    class FakeRuntime:
        def __init__(self, output):
            self.output, self.env, self.trace, self.run_id = output, None, None, None
            self.status, self.closed, self.sequence = "idle", False, 0
            self.scenario = scenarios[len(instances)]
            instances.append(self)

        def emit(self, event, payload):
            line = (
                RuntimeEvent(
                    event=event, payload=payload, run_id=self.run_id, sequence=self.sequence
                ).model_dump_json()
                + "\n"
            )
            self.sequence += 1
            self.output.write(line)
            self.trace.write(line)
            self.trace.flush()

        def command(self, command):
            if command["command"] == "abort":
                self.status = "aborted"
                self.close()
                return
            assert all(i.closed for i in instances[:-1])  # never simultaneous sessions
            assert consume_credentials() == (EMAIL, PASSWORD)
            provenance = {
                **identity(),
                "browser_execution": "autonomous",
                "color_gate_passed": True,
                "optimizer_updates": 0,
            }
            if self.scenario.get("changed_identity"):
                provenance["checkpoint_sha256"] = "d" * 64
            if any(provenance[k] != v for k, v in self.expected_browser_identity.items()):
                raise BrowserSafetyStop("batch_provenance_changed")
            self.run_id = f"{len(instances):032x}"
            root = Path(command["payload"]["artifact_dir"])
            root.mkdir(parents=True, exist_ok=True)
            self.trace = (root / f"{self.run_id}.jsonl").open("x")
            self.journal = ColorJournal(root / self.run_id, provenance)
            self.measurements = {
                "browser_wavelength": {
                    "kind": "wavelength",
                    "value": 379,
                    "unit": "nm",
                    "source": "current star",
                }
            }
            self.env = SimpleNamespace(
                outcome="operator_aborted",
                summary=None,
                session=SimpleNamespace(
                    mapping={
                        "observation": {
                            "values": {
                                "star_name": self.scenario.get("star", "Fixture Star"),
                                "measurements": self.measurements,
                            }
                        }
                    }
                ),
            )
            self.status = "running"

        def advance_if_due(self):
            kind = self.scenario.get("kind", "pass")
            if kind == "exception":
                raise RuntimeError(f"private driver text {EMAIL} {PASSWORD}")
            if kind == "interrupt":
                raise KeyboardInterrupt
            label = "Red" if kind == "wrong_color" else "UV"
            self.journal.emit("observation", {"values": {"measurements": self.measurements}})
            self.emit(
                "action_proposed", {"kind": "SELECT", "target": "0:source", "value": "browser_wavelength"}
            )
            self.emit(
                "observation",
                {
                    "values": {"measurements": self.measurements},
                    "calculation": {"source": "browser_wavelength"},
                },
            )
            self.emit("action_proposed", {"kind": "SELECT", "target": "1:color", "value": label})
            self.emit("action_proposed", {"kind": "CLICK", "target": "2:check"})
            self.journal.receipt = {
                "selected_color": label,
                "readback_verified": True,
                "correctness_verified": False,
                "color_authorization": "autonomous_opt_in",
                "reference_hash": load_color_reference().checksum,
            }
            self.journal.emit(
                "action_result",
                {
                    "color_transport_verified": True,
                    "receipt": self.journal.receipt,
                    "action": {"kind": "SELECT", "value": label},
                },
            )
            self.env.outcome = "color_transport_verified"
            self.status = "stopped"
            self.finish()
            if kind == "corrupt_journal":
                with (self.journal.output / "events.jsonl").open("a") as stream:
                    stream.write("\n")

        def finish(self):
            if self.env and self.env.summary is None:
                count = (
                    2
                    if self.scenario.get("kind") == "too_many_writes"
                    else int(self.journal.receipt is not None)
                )
                self.env.summary = self.journal.finish(self.env.outcome, count)
                if self.scenario.get("kind") != "missing_summary":
                    self.emit("episode_summary", self.env.summary)

        def close(self):
            self.finish()
            if self.trace:
                self.trace.close()
            self.closed = True

    monkeypatch.setattr(batch, "Runtime", FakeRuntime)
    monkeypatch.setattr(color_batch, "load_browser_color_policy", lambda _: (None, identity()))
    real_result = color_batch._result

    def closed_result(runtime, *args, **kwargs):
        assert runtime.closed  # the private post-run reference cannot guide policy
        return real_result(runtime, *args, **kwargs)

    monkeypatch.setattr(color_batch, "_result", closed_result)
    return scenarios, instances


def run(tmp_path, count=10):
    return color_batch.run_color_reliability(
        options(), tmp_path / "batch", runs=count, credentials=(EMAIL, PASSWORD), notify=lambda _: None
    )


def test_exact_ten_runs_counts_duplicates_and_never_claims_course_success(fake_runs, tmp_path):
    scenarios, instances = fake_runs
    scenarios.extend([{}] * 10)
    report = run(tmp_path)
    assert report["all_requested_runs_passed"] and report["passed_runs"] == 10
    assert report["color_transport_verified_runs"] == report["reference_matching_runs"] == 10
    assert report["reference_band_counts"] == {"UV": 10}
    assert report["unique_stars"] == report["unique_measurement_cases"] == 1
    assert not report["distinct_star_target_met"] and not report["fresh_accounts_provisioned"]
    assert (
        not report["task_completed"]
        and not report["browser_acceptance_passed"]
        and not report["allow_submission"]
    )
    assert report["max_native_writes_per_run"] == 1 and report["run_time_limit_seconds"] == 240
    assert len(instances) == 10 and all(i.closed for i in instances)
    assert not any(os.environ.get(key) for key in batch.LOGIN_KEYS)
    for file in (tmp_path / "batch").rglob("*"):
        if file.is_file():
            assert EMAIL not in file.read_text() and PASSWORD not in file.read_text()
    assert json.loads((tmp_path / "batch/report.json").read_text()) == report
    with pytest.raises(FileExistsError):
        run(tmp_path)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("wrong_color", "incorrect_color_selection"),
        ("exception", "runtime_exception"),
        ("interrupt", "operator_aborted"),
        ("corrupt_journal", "artifact_verification_failed"),
        ("missing_summary", "reliability_evidence_gate_failed"),
        ("too_many_writes", "reliability_evidence_gate_failed"),
    ],
)
def test_stop_first_failure_without_repair_or_retry(fake_runs, tmp_path, kind, reason):
    scenarios, instances = fake_runs
    scenarios.extend([{}, {"kind": kind}, {}])
    report = run(tmp_path, 3)
    assert len(instances) == report["attempted_runs"] == 2
    assert report["passed_runs"] == 1 and all(i.closed for i in instances)
    assert report["runs"][-1]["failure_reason"] == reason
    if kind == "wrong_color":
        assert report["runs"][-1]["color_transport_verified"]
        assert report["runs"][-1]["selected_color"] == "Red"
        assert report["runs"][-1]["post_run_reference_match"] is False
        assert report["policy_failure_runs"] == 1
        assert report["safety_or_infrastructure_failure_runs"] == 0


def test_changed_identity_stops_before_second_browser(fake_runs, tmp_path):
    scenarios, instances = fake_runs
    scenarios.extend([{}, {"changed_identity": True}, {}])
    report = run(tmp_path, 3)
    assert len(instances) == 2 and instances[-1].env is None
    assert report["runs"][-1]["failure_reason"] == "batch_provenance_changed"


def test_time_budget_aborts_without_retry(fake_runs, tmp_path, monkeypatch):
    scenarios, instances = fake_runs
    scenarios.extend([{}] * 10)
    monkeypatch.setattr(color_batch, "RUN_SECONDS", 0)
    report = run(tmp_path)
    assert len(instances) == 1 and instances[0].closed
    assert report["runs"][0]["failure_reason"] == "reliability_time_limit"


@pytest.mark.parametrize("count", [0, 11, True, 1.5])
def test_out_of_budget_creates_no_output(fake_runs, tmp_path, count):
    with pytest.raises(ValueError, match="reliability runs"):
        run(tmp_path, count)
    assert not fake_runs[1] and not (tmp_path / "batch").exists()


def test_supervision_not_silently_upgraded(fake_runs, tmp_path):
    with pytest.raises(ValueError, match="autonomous profile"):
        color_batch.run_color_reliability(
            options().model_copy(update={"browser_execution": "supervised"}),
            tmp_path / "no",
            runs=10,
            credentials=(EMAIL, PASSWORD),
        )
    assert not fake_runs[1] and not (tmp_path / "no").exists()


def test_failed_preflight_clears_credentials_before_any_browser(fake_runs, tmp_path, monkeypatch):
    def fail(_):
        raise ValueError("Color browser gate not passed")

    monkeypatch.setattr(color_batch, "load_browser_color_policy", fail)
    for key in batch.LOGIN_KEYS:
        monkeypatch.setenv(key, "fixture-unused-secret")
    with pytest.raises(ValueError, match="gate not passed"):
        run(tmp_path)
    assert not fake_runs[1] and not (tmp_path / "batch").exists()
    assert not any(os.environ.get(key) for key in batch.LOGIN_KEYS)


def test_real_gated_policy_batch_uses_fresh_intercepted_contexts(
    chromium,  # noqa: F811
    tmp_path,
    monkeypatch,
    trained_artifacts,  # noqa: F811
):
    from habfly import browser_color_policy
    from habfly.browser_setup import BrowserSetup

    pages, contexts = [], []
    monkeypatch.setattr(BrowserSetup, "STARFIELD_POLL_SECONDS", 0)
    private_label = ColorReference.private_label

    def offline_label(reference, *args, **kwargs):
        assert all(p.is_closed() for p in pages), "Private reference accessed while browser was active"
        return private_label(reference, *args, **kwargs)

    monkeypatch.setattr(ColorReference, "private_label", offline_label)

    def bridge_factory(settings, output, provenance):
        assert all(p.is_closed() for p in pages)
        context = chromium.new_context()
        index = len(contexts)
        detail = (
            stellar_html()
            .replace("370", str([379, 680][index]))
            .replace("Althinagon", ["Althinagon", "Gannost"][index])
        )
        documents = {
            OUTER: f'<iframe style="width:800px;height:500px" src="{SIMULATION_URL}"></iframe><iframe src="{WIDGET}"></iframe><iframe src="{WIDGET}"></iframe>',
            SIMULATION_URL: canvas_html(detail=detail),
            WIDGET: "<button onclick='document.body.replaceChildren()'>Update Score</button><button>Submit Project</button>",
        }
        context.route(
            "**/*",
            lambda route: (
                route.fulfill(content_type="text/html", body=documents[route.request.url])
                if route.request.url in documents
                else route.abort()
            ),
        )
        fixture_page = context.new_page()
        fixture_page.goto(OUTER)
        bridge = BrowserColorPolicyBridge(settings, output, provenance, page=fixture_page, config=config())
        bridge.context = context  # give this run ownership so close really closes its page
        pages.append(fixture_page)
        contexts.append(context)
        return bridge

    monkeypatch.setattr(browser_color_policy, "BrowserColorPolicyBridge", bridge_factory)
    try:
        report = color_batch.run_color_reliability(
            options().model_copy(update={"interval": 0}),
            tmp_path / "real-batch",
            runs=2,
            credentials=(EMAIL, PASSWORD),
            notify=lambda _: None,
        )
        assert report["all_requested_runs_passed"], report
        assert len(contexts) == 2 and contexts[0] is not contexts[1]
        assert all(p.is_closed() for p in pages)
        assert report["unique_stars"] == report["unique_measurement_cases"] == 2
        assert report["reference_band_counts"] == {"UV": 1, "Red": 1}
        assert all(row["learned_decisions"] == 3 and row["write_attempts"] == 1 for row in report["runs"])
    finally:
        for context in contexts:
            context.close()
