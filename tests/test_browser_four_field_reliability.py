"""Capped batch and post-run grading tests; all browser traffic is intercepted."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_four_field_policy import settings, trained_artifacts  # noqa: F401
from test_browser_numeric import OUTER, WIDGET, chromium, config, stellar_html  # noqa: F401
from test_browser_setup import canvas_html

from habfly import browser_four_field_reliability as four
from habfly import browser_reliability as batch
from habfly.browser_four_field_policy import BrowserFourFieldBridge
from habfly.browser_stellar import SIMULATION_URL
from habfly.color_reference import ColorReference
from habfly.knowledge import LocalCalculator

EMAIL, PASSWORD = "fixture@example.invalid", "fixture-batch-secret"


def options(tmp_path):
    return settings(
        tmp_path, browser_setup="automatic", browser_execution="autonomous", paused=False, interval=0
    )


def test_two_real_models_complete_two_fresh_contexts(
    chromium,  # noqa: F811
    tmp_path,
    monkeypatch,
    trained_artifacts,  # noqa: F811
):
    from habfly import browser_four_field_policy
    from habfly.browser_setup import BrowserSetup

    pages, contexts = [], []
    monkeypatch.setattr(BrowserSetup, "STARFIELD_POLL_SECONDS", 0)
    original_label = ColorReference.private_label
    original_result = four._result

    def private(reference, *args, **kwargs):
        assert all(p.is_closed() for p in pages), "Oracle used while browser active"
        return original_label(reference, *args, **kwargs)

    def result(*args, **kwargs):
        assert all(p.is_closed() for p in pages), "Post-run accuracy used while browser active"
        return original_result(*args, **kwargs)

    monkeypatch.setattr(ColorReference, "private_label", private)
    monkeypatch.setattr(four, "_result", result)

    def bridge_factory(settings, output, provenance):
        assert all(p.is_closed() for p in pages)
        context = chromium.new_context()
        index = len(contexts)
        detail = (
            stellar_html()
            .replace("370", str([459, 680][index]))
            .replace("Althinagon", ["First", "Second"][index])
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
        page = context.new_page()
        page.goto(OUTER)
        bridge = BrowserFourFieldBridge(settings, output, provenance, page=page, config=config())
        bridge.context = context
        pages.append(page)
        contexts.append(context)
        return bridge

    monkeypatch.setattr(browser_four_field_policy, "BrowserFourFieldBridge", bridge_factory)
    try:
        output = tmp_path / "real-batch"
        report = four.run_four_field_reliability(
            options(tmp_path), output, runs=2, credentials=(EMAIL, PASSWORD), notify=lambda _: None
        )
        assert report["all_requested_runs_passed"], report
        assert len(contexts) == 2 and contexts[0] is not contexts[1] and all(p.is_closed() for p in pages)
        assert report["unique_stars"] == report["unique_measurement_cases"] == 2
        assert report["reference_band_counts"] == {"Blue": 1, "Red": 1}
        assert report["field_reference_matching_runs"] == {"distance": 2, "luminosity": 2, "temperature": 2}
        assert report["color_reference_matching_runs"] == 2
        assert all(
            r["learned_decisions"] == 34 and r["write_attempts"] == 4 and r["invalid_actions"] == 0
            for r in report["runs"]
        )
        assert not report["task_completed"] and not report["browser_acceptance_passed"]
        assert not report["fresh_accounts_provisioned"] and report["max_native_writes_per_run"] == 4
        assert json.loads((output / "report.json").read_text()) == report
        for path in output.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                raw = path.read_text()
                assert EMAIL not in raw and PASSWORD not in raw and "preview_sequence_id" not in raw

        root = output / report["runs"][0]["manifest"]
        trace = output / report["runs"][0]["trace"]
        # Reference mismatches fail, but never alter the completed run or claim
        # transport failed. These are scoring-only tests after browser teardown.
        with monkeypatch.context() as scoring:
            scoring.setattr(ColorReference, "private_label", lambda *_: "IR")
            failed = four.audit_saved_run(root.parent, trace)
            assert failed["four_field_transport_verified"] and not failed["four_field_reliability_passed"]
            assert failed["failure_reason"] == "incorrect_color_selection"
        original_calc = LocalCalculator.execute_unclassified_common

        def wrong_reference(calculator, operation, bindings):
            result = original_calc(calculator, operation, bindings)
            if operation == "temperature":
                result.value += 1
            return result

        with monkeypatch.context() as scoring:
            scoring.setattr(LocalCalculator, "execute_unclassified_common", wrong_reference)
            failed = four.audit_saved_run(root.parent, trace)
            assert failed["failure_reason"] == "incorrect_numeric_answer"
            assert not failed["numeric_reference_matches"]["temperature"]
        journal = root.parent / "numeric/events.jsonl"
        journal.write_text(journal.read_text().replace('"sequence":0', '"sequence": 0', 1))
        failed = four.audit_saved_run(root.parent, trace)
        assert not failed["four_field_reliability_passed"] and not failed["journal_hash_verified"]
        with journal.open("a") as stream:
            stream.write("\n")
        with pytest.raises(ValueError):
            four.audit_saved_run(root.parent, trace)
    finally:
        for context in contexts:
            context.close()


@pytest.fixture
def fake_batch(monkeypatch):
    scenarios, instances = [], []
    identity = {key: "a" * 64 for key in four.IDENTITY_KEYS}

    class FakeRuntime:
        def __init__(self, output):
            self.env = None
            self.trace = None
            self.run_id = None
            self.closed = False
            self.scenario = scenarios[len(instances)]
            instances.append(self)

        def command(self, command):
            if command["command"] == "abort":
                self.status = "aborted"
                return
            from habfly.browser import BrowserSafetyStop
            from habfly.browser_setup import consume_credentials

            assert consume_credentials() == (EMAIL, PASSWORD)
            assert self.expected_browser_identity == identity
            self.run_id = str(len(instances))
            if self.scenario == "identity":
                raise BrowserSafetyStop("batch_provenance_changed")
            self.env = SimpleNamespace(session=None, outcome=None)
            self.status = "running"

        def advance_if_due(self):
            if self.scenario == "interrupt":
                raise KeyboardInterrupt
            if self.scenario == "exception":
                raise RuntimeError(PASSWORD)
            self.status = "stopped"

        def close(self):
            self.closed = True
            self.env = None

    def result(runtime, _output, *, failure, **_kwargs):
        assert runtime.closed and runtime.env is None
        if runtime.scenario == "corrupt":
            raise ValueError("Malformed evidence")
        passed = not failure and runtime.scenario == "pass"
        return {
            "run_id": runtime.run_id,
            "four_field_reliability_passed": passed,
            "failure_reason": failure or (None if passed else "incorrect_color_selection"),
            "star_name": "Repeated",
            "measurement_case_sha256": "case",
            "provenance": identity,
            "learned_decisions": 34,
            "write_attempts": 4,
        }

    monkeypatch.setattr(batch, "Runtime", FakeRuntime)
    monkeypatch.setattr(four, "load_four_field_policy", lambda *_: (None, identity))
    monkeypatch.setattr(four, "browser_config", lambda *_: None)
    monkeypatch.setattr(four, "_result", result)
    return scenarios, instances


def run_fake(tmp_path, count=10):
    return four.run_four_field_reliability(
        options(tmp_path),
        tmp_path / "batch",
        runs=count,
        credentials=(EMAIL, PASSWORD),
        notify=lambda _: None,
    )


def test_exact_ten_run_cap_and_duplicate_reporting(fake_batch, tmp_path):
    scenarios, instances = fake_batch
    scenarios.extend(["pass"] * 10)
    report = run_fake(tmp_path)
    assert len(instances) == report["passed_runs"] == 10 and all(i.closed for i in instances)
    assert report["unique_stars"] == report["unique_measurement_cases"] == 1
    assert not report["distinct_star_target_met"] and not report["fresh_accounts_provisioned"]
    assert report["all_requested_runs_passed"] and report["stop_on_first_failure"]
    assert report["max_native_writes_per_run"] == 4 and report["run_time_limit_seconds"] == 1260
    with pytest.raises(FileExistsError):
        run_fake(tmp_path)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("wrong", "incorrect_color_selection"),
        ("identity", "batch_provenance_changed"),
        ("exception", "runtime_exception"),
        ("interrupt", "operator_aborted"),
        ("corrupt", "artifact_verification_failed"),
    ],
)
def test_stops_at_first_failure_without_an_extra_attempt(fake_batch, tmp_path, kind, reason):
    scenarios, instances = fake_batch
    scenarios.extend(["pass", kind, "pass"])
    report = run_fake(tmp_path, 3)
    assert len(instances) == report["attempted_runs"] == 2 and report["passed_runs"] == 1
    assert report["runs"][-1]["failure_reason"] == reason
    assert all(i.closed for i in instances)
    assert PASSWORD not in (tmp_path / "batch/report.json").read_text()


@pytest.mark.parametrize("count", [0, 11, True, 1.5])
def test_rejects_invalid_budget_before_browser(fake_batch, tmp_path, count):
    with pytest.raises(ValueError, match="reliability runs"):
        run_fake(tmp_path, count)
    assert not fake_batch[1] and not (tmp_path / "batch").exists()


def test_deadline_and_closed_runtime_requirement(fake_batch, tmp_path, monkeypatch):
    fake_batch[0].extend(["pass"] * 10)
    monkeypatch.setattr(four, "RUN_SECONDS", 0)
    report = run_fake(tmp_path)
    assert report["attempted_runs"] == 1 and report["runs"][0]["failure_reason"] == "reliability_time_limit"


def test_audit_matches_previous_supervised_capture_without_modifying_it():
    root = Path("experiments/browser-four-field/dceb4c0a70aa450baa32bf9be5fa00b8")
    if not root.exists():
        pytest.skip("Optional prior supervised live capture")
    path = root / "manifest.json"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = four.audit_saved_run(root, root.with_suffix(".jsonl"), authorization="human_confirmation")
    assert audit["four_field_reliability_passed"] and audit["write_attempts"] == 4
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not four.audit_saved_run(root, root.with_suffix(".jsonl"))["four_field_reliability_passed"]
