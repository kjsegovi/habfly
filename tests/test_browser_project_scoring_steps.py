"""Offline injected visible transport; no browser, network, training, or grades.

Real assessment/score adapters, spending ledger, capture validators and canonical
journal run against deterministic injected page operations. Synthetic receipts
are test data only and do not establish real project completion.
"""

import json
import socket
from copy import deepcopy
from functools import lru_cache
from types import SimpleNamespace

import pytest
from test_browser_numeric import OUTER, WIDGET, config
from test_project_assessment import visible
from test_project_evidence import inventory, sha, write
from test_project_progress import complete_star, fresh

import habfly.browser_assessment_actions as actions
import habfly.browser_project_assessment as stages
import habfly.browser_project_scoring_steps as module
import habfly.browser_score_transfer as scores
from habfly.browser import BrowserSafetyStop
from habfly.browser_stellar import SIMULATION_URL
from habfly.project_progress import Active, ProjectJournal, _json
from habfly.runtime import read_trace


@lru_cache
def thirty_tasks():
    result = fresh()
    for i in range(30):
        result = complete_star(result, f"Star{i:02d}")
    return result


class Collection:
    def __init__(self, items):
        self.items = items

    def all(self):
        return self.items

    def count(self):
        return len(self.items)


class Control:
    def __init__(self, page, label):
        self.page, self.label = page, label

    def element_handle(self, **_):
        return self

    def evaluate(self, script, other=None):
        return other is None or other is self

    def is_visible(self):
        return True

    def is_enabled(self):
        return True

    def count(self):
        return 1

    def is_checked(self):
        return False

    def click(self, **_):
        if self.page.preclick:
            self.page.preclick(self.label)
        self.page.clicks.append(self.label)
        if self.page.dispatch_error:
            raise RuntimeError("secret://fixture-session credential-do-not-log")
        if self.label == "ASSESS":
            self.page.funding -= 100
            self.page.ack = True
        elif self.label == "OK":
            self.page.ack = False
        else:
            self.page.score, self.page.notice = "56.59", True


class Frame:
    def __init__(self, page, url):
        self.page, self.url = page, url

    def get_by_role(self, role, name):
        labels = ["Update Score"] if self.url == WIDGET else ["ASSESS", "OK"]
        return Collection([self.page.controls[label] for label in labels if name.fullmatch(label)])


class Page:
    def __init__(self, rows):
        self.rows, self.mode, self.funding = rows, "data_quality", 50000
        self.ack = self.notice = self.dispatch_error = False
        self.score = "0.00"
        self.clicks, self.preclick = [], None
        self.controls = {
            name: Control(self, name) for name in ("ASSESS", "OK", "Update Score", "checkbox", "notice")
        }
        self.frames = [Frame(self, SIMULATION_URL), Frame(self, WIDGET)]
        self.context, self.url = SimpleNamespace(pages=[self]), OUTER

    def report(self):
        text = visible(mode=self.mode, funding=self.funding, collected=30, ack=self.ack)
        text = text.replace("ANALYZED DATA STAR\nVIEWING 1-30 OF 30", self.rows)
        return {
            "schema_version": 1,
            "mode": "read_only_browser_preflight",
            "ignored_frame_urls": [],
            "actions_executed": 0,
            "allow_submission": False,
            "outer_controls": [],
            "frames": [
                {
                    "id": "simulation-0",
                    "url": SIMULATION_URL,
                    "text": text,
                    "accessibility": "- text: fixture",
                    "controls": [],
                }
            ],
        }

    def on(self, *args):
        pass

    def remove_listener(self, *args):
        pass

    def wait_for_timeout(self, _):
        pass

    def get_by_role(self, *_args, **_kwargs):
        return self.controls["checkbox"]

    def get_by_text(self, *_args, **_kwargs):
        return Collection([self.controls["notice"]] if self.notice else [])

    def locator(self, _):
        return SimpleNamespace(inner_text=lambda **_: "Score: " + self.score)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("Network forbidden"))
    progress = thirty_tasks()
    journal = ProjectJournal(
        tmp_path, project_id=progress.project_id, attempt_id=progress.attempt_id
    ).create()
    journal.path.write_text(
        _json(progress.header())
        + "\n"
        + "".join(_json(r.model_dump(mode="json")) + "\n" for r in progress.records)
    )
    names = [s.name for s in progress.reduce().stars.values()]
    inv = inventory(tmp_path, names, "thirty")
    row_text = "ANALYZED DATA STAR " + " ".join(names) + " VIEWING 1-30 OF 30"
    receipt = json.loads((inv / "confirmed.json").read_bytes())
    for section in ("before", "after"):
        path = inv / section / "observation.json"
        report = json.loads(path.read_bytes())
        report["frames"][0]["text"] = "OBSERVATIONS " + row_text + " TOTAL COLLECTED 30"
        write(path, report)
        manifest_path = inv / section / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["observation_sha256"] = sha(path)
        write(manifest_path, manifest)
        receipt["source_sha256"][section + "/observation.json"] = sha(path)
    for row in receipt["rows"]:
        row["source_sha256"] = receipt["source_sha256"]["after/observation.json"]
    write(inv / "confirmed.json", receipt)
    history_root = tmp_path / "assessments"
    history_root.mkdir()
    page = Page(row_text)
    for target in (actions, stages, scores):
        monkeypatch.setattr(target, "inspect_page", lambda p, _c: deepcopy(p.report()))
    monkeypatch.setattr(actions.AssessmentActuator, "button", lambda self, label: self.page.controls[label])
    events, clock = [], [0]
    values = {
        "journal": journal,
        "run_history": tmp_path,
        "inventory_dir": inv,
        "assessment_history_root": history_root,
        "emit": events.append,
        "_clock": lambda: clock[0],
    }
    return SimpleNamespace(
        root=tmp_path, journal=journal, inv=inv, page=page, values=values, events=events, clock=clock
    )


def make(case, **kwargs):
    return module.BrowserProjectAssessmentSteps(
        case.page, config(), case.root / "scoring", **(case.values | kwargs)
    )


def quality(component):
    component.provide_panel("data_quality")
    component.advance()  # read-only stage initialization
    component.advance()  # one charge


def both(case, component):
    quality(component)
    component.advance()  # acknowledgement only
    case.page.mode = "scavenger_hunt"  # explicit visible panel handoff, not an adapter click
    component.provide_panel("scavenger_hunt")
    component.advance()
    component.advance()


def test_constructor_offline_and_each_advance_has_one_distinct_write(case, monkeypatch):
    for target in (actions, stages, scores):
        monkeypatch.setattr(target, "inspect_page", lambda *_: pytest.fail("constructor touched page"))
    component = make(case)
    assert component.phase == "awaiting_data_quality_panel" and not case.page.clicks
    assert component.advance()["advances"] == 0
    assert not case.journal.load().reduce().reservations
    component.abort()
    component.advance()
    assert not case.page.clicks


def test_both_charges_and_transfer_reserved_before_actual_dispatch(case):
    component = make(case, allow_score_transfer=True)
    observed = []

    def prior(label):
        state = case.journal.load().reduce()
        if label in {"ASSESS", "Update Score"}:
            assert len(state.pending) == 1
            assert state.pending[0].revision == component.revision
            assert state.pending[0].project_rows_sha256 == component.rows_sha
            observed.append(state.pending[0].write_kind)
        else:
            assert not state.pending

    case.page.preclick = prior
    quality(component)
    assert case.page.clicks == ["ASSESS"]
    assert component.phase == "acknowledging_data_quality"
    component.pause()
    assert case.page.clicks == ["ASSESS"]
    component.advance()
    assert case.page.clicks == ["ASSESS", "OK"]
    assert component.phase == "awaiting_scavenger_hunt_panel"
    component.resume()
    component.advance()
    assert len(case.page.clicks) == 2
    case.page.mode = "scavenger_hunt"
    component.provide_panel("scavenger_hunt")
    component.advance()
    assert case.page.clicks == ["ASSESS", "OK", "ASSESS"]
    component.advance()
    assert component.phase == "score_transfer_ready"
    component.advance()
    assert component.status == "completed", component.report
    assert case.page.clicks == ["ASSESS", "OK", "ASSESS", "OK", "Update Score"]
    assert observed == ["assessment_data_quality", "assessment_scavenger_hunt", "score_transfer"]
    state = case.journal.load().reduce()
    assert not state.pending and len(state.receipts) == 3
    assert state.receipt("score_transfer").score == 56.59
    ledger = state.receipt("assessment_scavenger_hunt").assessment
    assert ledger.budget == ledger.reserved == 200 and ledger.max_attempts == 2
    assert len(ledger.attempts) == 2
    assert not component.report["project_completed"] and not state.report()["project_completed"]
    assert not component.report["task_completed"] and not state.report()["submitted"]
    assert component.report["score_transfer_verified"]
    saved = [e.model_dump(mode="json") for e in read_trace(component.output / "events.jsonl")]
    assert saved == case.events and all(e["version"] == 1 for e in saved)
    component.advance()
    assert len(case.page.clicks) == 5
    with pytest.raises(BrowserSafetyStop, match="revision_already_reserved"):
        module.BrowserProjectAssessmentSteps(case.page, config(), case.root / "alternate", **case.values)


def test_transfer_is_disabled_by_default_and_scoring_never_submits(case):
    component = make(case)
    both(case, component)
    assert component.finished and component.phase == "assessed_score_transfer_disabled"
    assert not component.report["score_transfer_verified"] and not component.report["project_completed"]
    assert case.page.clicks == ["ASSESS", "OK", "ASSESS", "OK"]
    assert len(case.journal.load().reduce().receipts) == 2


@pytest.mark.parametrize("boundary", ["assessment", "acknowledgement", "transfer"])
@pytest.mark.parametrize("failure", ["abort", "callback", "reentrant", "source", "journal"])
def test_predispatch_boundary_cannot_dispatch_after_abort_or_changed_source(case, boundary, failure):
    holder = {}

    def callback(event):
        case.events.append(event)
        if event["event"] != "action_proposed":
            return
        target = event["payload"]["target"]
        selected = {
            "assessment": "assessment_data_quality",
            "acknowledgement": "acknowledgement_data_quality",
            "transfer": "score_transfer",
        }[boundary]
        if target != selected:
            return
        if failure == "abort":
            holder["component"].abort()
        elif failure == "callback":
            raise RuntimeError("session-secret password must stay redacted")
        elif failure == "reentrant":
            holder["component"].advance()
        elif failure == "source":
            path = case.inv / "confirmed.json"
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            case.journal.append(Active(star_id=None, stage="assessment"))

    component = make(case, allow_score_transfer=True, emit=callback)
    holder["component"] = component
    if boundary == "transfer":
        both(case, component)
        component.advance()
        expected = 4
    else:
        quality(component)
        if boundary == "acknowledgement":
            component.advance()
        expected = int(boundary == "acknowledgement")
    assert component.finished and len(case.page.clicks) == expected, component.report
    state = case.journal.load().reduce()
    assert len(state.pending) == (0 if boundary == "acknowledgement" else 1)
    assert len(state.receipts) == (2 if boundary == "transfer" else expected)
    assert not component.report["project_completed"]
    assert "session-secret" not in (component.output / "report.json").read_text()
    component.advance()
    assert len(case.page.clicks) == expected


@pytest.mark.parametrize("boundary", ["assessment", "acknowledgement", "transfer"])
def test_dispatched_unknown_action_retains_pending_evidence_and_blocks_new_output(case, boundary):
    component = make(case, allow_score_transfer=True)
    if boundary == "transfer":
        both(case, component)
    elif boundary == "acknowledgement":
        quality(component)
    else:
        component.provide_panel("data_quality")
        component.advance()
    case.page.dispatch_error = True
    component.advance()
    assert component.finished and component.status == "stopped"
    state = case.journal.load().reduce()
    assert len(state.pending) == (0 if boundary == "acknowledgement" else 1)
    assert "credential-do-not-log" not in (component.output / "report.json").read_text()
    with pytest.raises(BrowserSafetyStop):
        module.BrowserProjectAssessmentSteps(case.page, config(), case.root / "retry", **case.values)


@pytest.mark.parametrize("fault", ["wrong_panel", "rows", "count"])
def test_wrong_visible_panel_or_collection_never_spends(case, fault):
    component = make(case)
    if fault == "wrong_panel":
        case.page.mode = "scavenger_hunt"
    elif fault == "rows":
        case.page.rows += " EXTRA STAR"
    else:
        original = case.page.report

        def report():
            result = original()
            result["frames"][0]["text"] = result["frames"][0]["text"].replace(
                "TOTAL COLLECTED 30", "TOTAL COLLECTED 29"
            )
            return result

        case.page.report = report
    component.provide_panel("data_quality")
    component.advance()
    component.advance()
    assert component.finished and not case.page.clicks
    assert not case.journal.load().reduce().reservations


def test_fixed_deadline_counts_pause_time_and_no_automatic_extension(case):
    component = make(case, max_seconds=5)
    component.provide_panel("data_quality")
    component.advance()
    case.clock[0] = 5
    component.advance()
    assert component.finished and not case.page.clicks
    assert not case.journal.load().reduce().reservations


def test_bounded_advance_limit_and_explicit_panel_handoff(case):
    component = make(case, max_advances=1)
    with pytest.raises(BrowserSafetyStop, match="unexpected_panel_handoff"):
        component.provide_panel("scavenger_hunt")
    component.provide_panel("data_quality")
    component.advance()
    component.advance()
    assert component.finished and not case.page.clicks


@pytest.mark.parametrize(
    "options",
    [
        {"allow_score_transfer": 1},
        {"max_seconds": float("nan")},
        {"max_seconds": 3601},
        {"max_advances": True},
        {"max_advances": 7},
        {"assessment_timeout_seconds": 31},
        {"score_timeout_seconds": 0.09},
    ],
)
def test_invalid_limits_rejected_offline_before_output(case, options):
    with pytest.raises(BrowserSafetyStop):
        make(case, **options)
    assert not (case.root / "scoring").exists() and not case.page.clicks


@pytest.mark.parametrize(
    "fault", ["less_than_thirty", "unverified", "inventory_other_names", "capture", "ambiguous"]
)
def test_invalid_canonical_or_inventory_source_rejected_offline(case, fault):
    if fault in {"less_than_thirty", "unverified"}:
        progress = thirty_tasks()
        records = progress.records[:-7] if fault == "less_than_thirty" else progress.records[:-1]
        case.journal.path.write_text(
            _json(progress.header())
            + "\n"
            + "".join(_json(r.model_dump(mode="json")) + "\n" for r in records)
        )
    elif fault == "capture":
        path = case.inv / "after/observation.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "ambiguous":
        ProjectJournal(case.root, project_id="other", attempt_id="other").create()
    else:
        case.values["inventory_dir"] = inventory(case.root, ["Other Star"], "wrong")
    with pytest.raises((ValueError, BrowserSafetyStop)):
        make(case)
    assert not (case.root / "scoring").exists() and not case.page.clicks


def test_owner_claim_blocks_alternate_output_even_before_dispatch(case):
    component = make(case)
    component.abort()
    with pytest.raises(BrowserSafetyStop, match="owner_already_reserved"):
        module.BrowserProjectAssessmentSteps(case.page, config(), case.root / "retry", **case.values)
    assert not case.page.clicks and not case.journal.load().reduce().reservations


@pytest.mark.parametrize("fault", ["confirmation", "capture", "ledger", "missing_dispatch"])
def test_malformed_assessment_evidence_never_gets_canonical_receipt(case, monkeypatch, fault):
    original = actions.AssessmentActuator.assess

    def malformed(actuator, kind, **kwargs):
        if fault == "missing_dispatch":
            return {"data_quality_assessed": True}
        result = original(actuator, kind, **kwargs)
        path = actuator.output / "attempt-000-confirmed.json"
        if fault == "capture":
            path = actuator.output / "attempt-000-after/observation.json"
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            data = json.loads(path.read_bytes())
            if fault == "confirmation":
                data["receipt"]["data_revision"] += 1
            else:
                data["ledger"]["max_attempts"] = 3
            write(path, data)
        return result

    monkeypatch.setattr(actions.AssessmentActuator, "assess", malformed)
    component = make(case)
    quality(component)
    assert component.status == "stopped"
    state = case.journal.load().reduce()
    assert not state.receipts
    assert len(state.pending) == (fault != "missing_dispatch")
    assert not component.report["assessment_verified"]["data_quality"]


@pytest.mark.parametrize("fault", ["confirmation", "capture", "acknowledgement", "return", "source"])
def test_malformed_transfer_evidence_preserves_real_pending_transfer(case, monkeypatch, fault):
    component = make(case, allow_score_transfer=True)
    both(case, component)
    original = module.transfer_assessed_score

    def malformed(*args, **kwargs):
        result = original(*args, **kwargs)
        directory = args[2]
        if fault == "return":
            return {**result, "submitted": True}
        if fault == "source":
            path = kwargs["assessment_receipt"]
            path.write_bytes(path.read_bytes() + b"\n")
        elif fault == "capture":
            path = directory / "after/observation.json"
            path.write_bytes(path.read_bytes() + b"\n")
        elif fault == "acknowledgement":
            write(directory / "acknowledgement.json", {"visible_text": "Generic success"})
        else:
            write(directory / "confirmed.json", {**result, "source_sha256": "a" * 64})
        return result

    monkeypatch.setattr(module, "transfer_assessed_score", malformed)
    component.advance()
    assert component.status == "stopped" and len(case.page.clicks) == 5
    state = case.journal.load().reduce()
    assert len(state.receipts) == 2 and len(state.pending) == 1
    assert state.pending[0].write_kind == "score_transfer"
    assert not component.report["score_transfer_verified"] and not component.report["project_completed"]


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("allow_score_transfer", True),
        ("max_seconds", 601),
        ("assessment_timeout_seconds", 30),
        ("score_timeout_seconds", 30),
        ("max_advances", 5),
    ],
)
def test_declared_options_cannot_change_after_construction(case, attribute, value):
    component = make(case)
    component.provide_panel("data_quality")
    setattr(component, attribute, value)
    component.advance()
    assert component.finished and not case.page.clicks
    assert not case.journal.load().reduce().reservations


def test_actuator_deadline_cannot_increase_after_stage_initialization(case):
    component = make(case)
    component.provide_panel("data_quality")
    component.advance()
    component.actuator.timeout_seconds = 30
    component.advance()
    assert component.finished and not case.page.clicks


def test_cancellation_after_native_guard_before_click_keeps_reservation(case, monkeypatch):
    flag = [False]
    original = Control.evaluate

    def changed(control, script, other=None):
        if other is not None and control.label == "ASSESS":
            flag[0] = True
        return original(control, script, other)

    monkeypatch.setattr(Control, "evaluate", changed)
    component = make(case, cancelled=lambda: flag[0])
    quality(component)
    assert component.status == "aborted" and not case.page.clicks
    assert len(case.journal.load().reduce().pending) == 1


def test_explicit_acknowledgement_uncertainty_is_not_hidden(case):
    component = make(case)
    quality(component)
    case.page.dispatch_error = True
    component.advance()
    assert component.report["pending_acknowledgement"] == "data_quality"
    assert component.report["pending_canonical_action"] is None
    assert not case.journal.load().reduce().pending
    assert (component.stage_dir / "ack-000-reserved.json").exists()
    assert not (component.stage_dir / "ack-000-confirmed.json").exists()


def test_terminal_callback_failure_is_redacted_and_persists_corrected_event(case):
    def callback(event):
        if event["event"] == "episode_summary":
            raise RuntimeError("terminal-private-credential")
        case.events.append(event)

    component = make(case, emit=callback)
    both(case, component)
    assert component.status == "stopped"
    assert component.report["event_forwarding_failed"]
    assert "terminal-private-credential" not in (component.output / "report.json").read_text()
    events = read_trace(component.output / "events.jsonl")
    assert events[-1].event == "state" and events[-1].payload["status"] == "stopped"
    assert component.report["events_sha256"] == sha(component.output / "events.jsonl")
    assert not events[-1].payload["project_completed"]


def test_legacy_adapters_keep_original_signature_defaults_and_validate_hooks_before_page():
    actuator = object.__new__(actions.AssessmentActuator)
    with pytest.raises(TypeError, match="callable"):
        actuator.assess("data_quality", data_revision=1, before_dispatch=True)
    with pytest.raises(TypeError, match="callable"):
        actuator.dismiss_receipt(check_cancelled="bad")
    with pytest.raises(TypeError, match="callable"):
        scores.transfer_assessed_score(None, None, None, assessment_receipt=None, before_dispatch=False)
