"""Real child owners and canonical ledgers, injected public page only.

No native browser/network/model training. Synthetic inventory/task receipts are
test inputs; passing this gate is not thirty-star real-project acceptance.
"""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml
from test_browser_numeric import config
from test_browser_project_inventory_steps import Events
from test_browser_project_scoring_steps import case  # noqa: F401 - shared offline fixture
from test_project_evidence import inventory, sha, write
from test_project_progress import complete_star, fresh

import habfly.browser_assessment_actions as actions
import habfly.browser_assessment_panel_steps as panels
import habfly.browser_project_assessment as stages
import habfly.browser_project_scoring_coordinator as module
import habfly.browser_score_transfer as scores
from habfly.browser import BrowserSafetyStop
from habfly.project_progress import Active, ProgressError, _json
from habfly.runtime import read_trace


@pytest.fixture
def rig(case, monkeypatch):  # noqa: F811
    """Add native-shaped navigation controls to the existing injected scorer."""
    report_path = case.inv / "after/observation.json"
    original = json.loads(report_path.read_bytes())
    receipt = json.loads((case.inv / "confirmed.json").read_bytes())
    for section in ("before", "after"):
        path = case.inv / section / "observation.json"
        report = json.loads(path.read_bytes())
        report.update(outer_url=config().url, outer_controls=[])
        write(path, report)
        manifest = json.loads((case.inv / section / "manifest.json").read_bytes())
        manifest["observation_sha256"] = sha(path)
        write(case.inv / section / "manifest.json", manifest)
        receipt["source_sha256"][section + "/observation.json"] = sha(path)
    for row in receipt["rows"]:
        row["source_sha256"] = receipt["source_sha256"]["after/observation.json"]
    write(case.inv / "confirmed.json", receipt)
    original_atoms = yaml.safe_load(original["frames"][0]["accessibility"])
    original_report = case.page.report
    calls = []
    flags = SimpleNamespace(reads=0, inspect_hook=None, click_hook=None, cancelled=False, native_reads=0)

    def current():
        report = original_report()
        report.update(outer_url=config().url)
        frame = report["frames"][0]
        header = frame["text"].split("OBSERVATIONS")[0]
        atoms = deepcopy(original_atoms)
        atoms[0]["text"] = header + atoms[0]["text"].removeprefix("Funding ")
        frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)
        labels = ["ASSESSMENT", "DATA QUALITY", "SCAVENGER HUNT", "Assess", "Save"]
        frame["controls"] = [
            {
                "id": f"simulation-0:c{i}",
                "role": "button",
                "accessibility": f'- button "{label}"',
                "enabled": True,
                "actions": [],
            }
            for i, label in enumerate(labels)
        ]
        return report

    case.page.report = current
    page_events, context_events = Events(), Events()
    case.page.on, case.page.remove_listener = page_events.on, page_events.remove_listener
    case.page.context.on, case.page.context.remove_listener = (
        context_events.on,
        context_events.remove_listener,
    )
    case.page.main_frame = object()
    frame = case.page.frames[0]
    frame.parent_frame = case.page.main_frame
    original_roles = frame.get_by_role

    class Button:
        def __init__(self, label):
            self.label = label

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def element_handle(self, **_):
            return self

        def evaluate(self, code, other=None):
            if "isConnected" in code:
                return other is self
            if "tagName" in code:
                return True
            return {"x": 10, "y": 10, "width": 40, "height": 20}

        def click(self, **_):
            assert self.label in panels.ALLOW
            calls.append(("panel_click", self.label))
            if flags.click_hook:
                flags.click_hook()
            case.page.mode = (
                "data_quality"
                if self.label == "ASSESSMENT"
                else next(key for key, value in panels.LABELS.items() if value == self.label)
            )

    buttons = {label: Button(label) for label in panels.ALLOW}

    def roles(role, name, exact=False):
        if exact:
            assert role == "button" and name in buttons
            return SimpleNamespace(all=lambda: [buttons[name]])
        return original_roles(role, name)

    frame.get_by_role = roles

    def native(_):
        flags.native_reads += 1
        return [
            {"text": current()["frames"][0]["text"], "boxes": [{"x": 0, "y": 0, "width": 1, "height": 1}]}
        ]

    frame.locator = lambda _: SimpleNamespace(evaluate=native)

    def inspect(page, _config):
        flags.reads += 1
        if flags.inspect_hook:
            flags.inspect_hook()
        return deepcopy(page.report())

    for target in (actions, stages, scores, panels):
        monkeypatch.setattr(target, "inspect_page", inspect)
    monkeypatch.setattr(panels, "_visible_rows", lambda *_: None)
    monkeypatch.setattr(panels, "_outer_exposed", lambda *_: True)
    monkeypatch.setattr(panels, "row_icon_exposed", lambda *_: True)
    monkeypatch.setattr(panels.time, "monotonic", lambda: case.clock[0])
    for cls, name in (
        (module.AssessmentPanelSteps, "panel"),
        (module.BrowserProjectAssessmentSteps, "scoring"),
    ):
        advance = cls.advance

        def counted(self, _advance=advance, _name=name):
            calls.append((_name, self.phase))
            return _advance(self)

        monkeypatch.setattr(cls, "advance", counted)
    return SimpleNamespace(**vars(case), flags=flags, calls=calls, listeners=(page_events, context_events))


def make(rig, **options):
    return module.BrowserProjectScoringCoordinator(
        rig.page,
        config(),
        rig.root / options.pop("output", "coordinator"),
        **(rig.values | {"cancelled": lambda: rig.flags.cancelled} | options),
    )


def advance_until(owner, predicate):
    for _ in range(16):
        if owner.finished or predicate():
            return
        owner.advance()
    pytest.fail("fixed scheduling budget exceeded")


def run(owner):
    advance_until(owner, lambda: False)
    assert owner.finished


@pytest.mark.parametrize("transfer", [False, True])
def test_genuine_child_owners_offline_constructor_one_child_per_call(rig, transfer):
    owner = make(rig, allow_score_transfer=transfer)
    assert not rig.flags.reads and not rig.calls and not rig.page.clicks
    assert not rig.journal.load().reduce().reservations
    assert owner.scoring.output == owner.output / "scoring"
    before_reservations = []

    def before_native(label):
        state = rig.journal.load().reduce()
        if label in {"ASSESS", "Update Score"}:
            assert len(state.pending) == 1
            assert state.pending[0].revision == owner.scoring.revision
            assert state.pending[0].project_rows_sha256 == owner.scoring.rows_sha
            before_reservations.append(state.pending[0].write_kind)
        else:
            assert not state.pending

    rig.page.preclick = before_native
    for _ in range(16):
        if owner.finished:
            break
        previous = len([c for c in rig.calls if c[0] != "panel_click"])
        native = len(rig.page.clicks) + len([c for c in rig.calls if c[0] == "panel_click"])
        owner.advance()
        assert len([c for c in rig.calls if c[0] != "panel_click"]) - previous <= 1
        assert len(rig.page.clicks) + len([c for c in rig.calls if c[0] == "panel_click"]) - native <= 1
    assert owner.finished and owner.failure is None, owner.report
    expected = "score_transferred_not_submitted" if transfer else "assessed_score_transfer_disabled"
    assert owner.phase == expected
    assert rig.page.clicks == ["ASSESS", "OK", "ASSESS", "OK"] + (["Update Score"] if transfer else [])
    assert [c for c in rig.calls if c[0] == "panel_click"] == [("panel_click", "SCAVENGER HUNT")]
    assert before_reservations == ["assessment_data_quality", "assessment_scavenger_hunt"] + (
        ["score_transfer"] if transfer else []
    )
    assert owner.scoring.advances == (6 if transfer else 5)
    assert owner.report["verified_panels"] == ["data_quality", "scavenger_hunt"]
    assert owner.report["score_transfer_verified"] is transfer
    assert all(owner.report[k] is False for k in ("task_completed", "project_completed", "submitted"))
    assert not rig.journal.load().reduce().pending
    assert not rig.journal.load().reduce().report()["project_completed"]
    assert all(not any(events.listeners.values()) for events in rig.listeners)
    events = [e.model_dump(mode="json") for e in read_trace(owner.output / "events.jsonl")]
    assert events == rig.events
    original = [e.model_dump(mode="json") for e in read_trace(owner.scoring.output / "events.jsonl")]
    assert [
        e["payload"]["component_event"]
        for e in events
        if e["payload"].get("component") == "project.assessment"
    ] == original
    assert owner.report["scoring_dir"] == "coordinator/scoring"
    assert (
        sha(owner.scoring.output / "report.json")
        == owner.report["source_sha256"]["coordinator/scoring/report.json"]
    )
    assert all(e["payload"].get("project_completed") is not True for e in events)
    unchanged = deepcopy(rig.calls), list(rig.page.clicks)
    owner.advance()
    owner.close()
    assert (rig.calls, rig.page.clicks) == unchanged


def test_panel_handoff_is_a_separate_offline_scheduling_call(rig):
    owner = make(rig)
    owner.advance()  # panel construction only
    assert owner.phase == "panel_active" and not rig.flags.reads
    owner.advance()  # already-selected panel verification
    assert owner.phase == "panel_handoff" and owner.scoring.advances == 0
    reads, calls = rig.flags.reads, list(rig.calls)
    owner.advance()
    assert owner.phase == "scoring_active" and owner.scoring.advances == 0
    assert rig.flags.reads == reads and rig.calls == calls
    owner.abort()
    assert not rig.page.clicks


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_seconds", 601),
        ("max_seconds", True),
        ("panel_max_seconds", 61),
        ("allow_score_transfer", 1),
        ("assessment_timeout_seconds", 31),
        ("score_timeout_seconds", float("nan")),
    ],
)
def test_fixed_budgets_reject_before_any_claim(rig, field, value):
    with pytest.raises(BrowserSafetyStop):
        make(rig, **{field: value})
    assert not (rig.root / "project-scoring-claims").exists() and not rig.flags.reads


@pytest.mark.parametrize("gate", ["29_tasks", "unverified", "partial_list", "different_names"])
def test_original_strict_task_and_inventory_gate_is_not_relaxed(rig, gate):
    if gate in {"29_tasks", "unverified"}:
        progress = fresh()
        for i in range(29):
            progress = complete_star(progress, f"Star{i:02d}")
        if gate == "unverified":
            from habfly.project_progress import Collected

            progress = progress.append(Collected(star_id="Star29", name="Star29", source_sha256="a" * 64))
        rig.journal.path.write_text(
            _json(progress.header())
            + "\n"
            + "".join(_json(r.model_dump(mode="json")) + "\n" for r in progress.records)
        )
    elif gate == "partial_list":
        receipt = json.loads((rig.inv / "confirmed.json").read_bytes())
        receipt["complete_visible_list_verified"] = False
        write(rig.inv / "confirmed.json", receipt)
    else:
        rig.values["inventory_dir"] = inventory(rig.root, [f"Other{i:02d}" for i in range(30)], "other")
    with pytest.raises((BrowserSafetyStop, ProgressError)):
        make(rig)
    assert not rig.flags.reads and not rig.page.clicks


@pytest.mark.parametrize("when", ["before_panel", "inside_panel", "before_scoring", "inside_scoring"])
def test_overall_deadline_includes_panel_and_pause_and_intracall_work(rig, when):
    owner = make(rig, max_seconds=60)
    if when in {"before_scoring", "inside_scoring"}:
        advance_until(owner, lambda: owner.phase == "scoring_active")
        owner.advance()  # read-only scoring initialization
    elif when == "inside_panel":
        owner.advance()
    if when.startswith("inside"):
        rig.flags.inspect_hook = lambda: rig.clock.__setitem__(0, 60)
    else:
        rig.clock[0] = 60
    owner.advance()
    assert owner.finished and owner.phase == "stopped"
    assert not rig.page.clicks and not [c for c in rig.calls if c[0] == "panel_click"]
    assert not owner.report["project_completed"]


@pytest.mark.parametrize(
    "change",
    ["source", "extra_failure", "panel_receipt", "child_budget", "parent_budget", "boundary", "journal"],
)
def test_completed_sources_and_fixed_scope_checked_before_next_child(rig, change):
    owner = make(rig)
    advance_until(owner, lambda: owner.phase == "panel_handoff")
    if change == "source":
        (rig.inv / "after/observation.json").write_text("{}")
    elif change == "extra_failure":
        write(owner.panel.output / "later-stopped.json", {"failure": True})
    elif change == "panel_receipt":
        (owner.panel.output / "confirmed.json").write_text("{}")
    elif change == "child_budget":
        owner.scoring.max_advances = 7
    elif change == "parent_budget":
        owner.max_seconds += 1
    elif change == "boundary":
        owner.config.url += "/different"
    else:
        rig.journal.append(Active(star_id="Star00", stage="stellar_numeric"))
    before = list(rig.calls)
    owner.advance()
    assert owner.phase == "stopped" and rig.calls == before and not rig.page.clicks
    assert (owner.output / "stopped.json").is_file()


@pytest.mark.parametrize(
    "event_target",
    [
        "assessment_data_quality",
        "acknowledgement_data_quality",
        "assessment_scavenger_hunt",
        "acknowledgement_scavenger_hunt",
        "score_transfer",
    ],
)
@pytest.mark.parametrize("how", ["abort", "cancel", "callback_error", "reentrant", "source_change"])
def test_pre_dispatch_callback_interrupt_never_advances_or_retries(rig, event_target, how):
    holder = []

    def emit(event):
        rig.events.append(event)
        if event["event"] != "action_proposed" or event["payload"].get("target") != event_target:
            return
        owner = holder[0]
        if how == "abort":
            owner.abort()
        elif how == "cancel":
            rig.flags.cancelled = True
        elif how == "callback_error":
            raise RuntimeError("secret://credential-do-not-log")
        elif how == "reentrant":
            owner.advance()
        else:
            write(owner.output / "panel-data_quality/later-stopped.json", {"failure": True})

    owner = make(rig, allow_score_transfer=True, emit=emit)
    holder.append(owner)
    run(owner)
    allowed_before = {
        "assessment_data_quality": [],
        "acknowledgement_data_quality": ["ASSESS"],
        "assessment_scavenger_hunt": ["ASSESS", "OK"],
        "acknowledgement_scavenger_hunt": ["ASSESS", "OK", "ASSESS"],
        "score_transfer": ["ASSESS", "OK", "ASSESS", "OK"],
    }
    assert owner.phase == "stopped" and rig.page.clicks == allowed_before[event_target]
    assert not owner.report["project_completed"]
    assert "credential-do-not-log" not in (owner.output / "events.jsonl").read_text()
    assert (owner.output / "stopped.json").is_file()
    previous = list(rig.page.clicks)
    owner.advance()
    assert rig.page.clicks == previous
    with pytest.raises(BrowserSafetyStop):
        make(rig, output="changed-output", allow_score_transfer=True)


def test_uncertain_native_charge_keeps_canonical_reservation(rig):
    owner = make(rig)
    rig.page.dispatch_error = True
    run(owner)
    assert rig.page.clicks == ["ASSESS"] and owner.phase == "stopped"
    assert len(rig.journal.load().reduce().pending) == 1
    assert not owner.report["score_transfer_verified"]
    with pytest.raises(BrowserSafetyStop):
        make(rig, output="new-output")


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("kind", "scavenger_hunt"),
        ("panel_verified", False),
        ("inventory_sha256", "b" * 64),
        ("inventory_path", "elsewhere/confirmed.json"),
        ("project_rows_sha256", "b" * 64),
        ("task_completed", True),
        ("assessment_clicks", False),
        ("navigation_dispatch_attempts", False),
        ("max_advances", 4),
        ("current_screen_sha256", "b" * 64),
    ],
)
def test_completed_panel_receipt_is_independently_validated(rig, monkeypatch, key, value):
    owner = make(rig)
    owner.advance()
    original = owner.panel.advance

    def altered():
        result = original()
        owner.panel.report[key] = value
        write(owner.panel.output / "confirmed.json", owner.panel.report)
        return result

    monkeypatch.setattr(owner.panel, "advance", altered)
    owner.advance()
    assert owner.finished and owner.phase == "stopped" and not rig.page.clicks
    assert not (owner.scoring.output / "panel-data_quality.json").exists()


@pytest.mark.parametrize("when", ["capture", "receipt_file", "events"])
def test_completed_panel_source_hashes_checked_before_handoff(rig, monkeypatch, when):
    owner = make(rig)
    owner.advance()
    original = owner.panel.advance

    def altered():
        result = original()
        if when == "capture":
            write(owner.panel.output / "verified/observation.json", {})
        elif when == "receipt_file":
            write(owner.panel.output / "confirmed.json", {})
        else:
            (owner.panel.output / "events.jsonl").write_text("")
        return result

    monkeypatch.setattr(owner.panel, "advance", altered)
    owner.advance()
    assert owner.phase == "stopped" and not rig.page.clicks


def test_parent_source_guard_applies_during_native_child_capture(rig):
    owner = make(rig)
    advance_until(owner, lambda: owner.phase == "scoring_active")
    owner.advance()
    rig.flags.inspect_hook = lambda: write(owner.panel.output / "stopped.json", {})
    owner.advance()
    assert owner.phase == "stopped" and not rig.page.clicks


def test_cleanup_error_cannot_erase_stop_or_allow_another_action(rig, monkeypatch):
    owner = make(rig)
    owner.advance()

    def failed_abort():
        raise RuntimeError("private-cleanup-driver-error")

    original_abort = owner.panel.abort
    monkeypatch.setattr(owner.panel, "abort", failed_abort)
    owner.abort()
    assert owner.report["cleanup_failed"] and (owner.output / "stopped.json").exists()
    owner.advance()
    assert not rig.page.clicks
    # Test teardown encounters the intentionally retired parent callback.
    with pytest.raises(BrowserSafetyStop, match="event_forwarding_failed"):
        original_abort()
    assert owner.panel._stream.closed


@pytest.mark.parametrize("how", ["cancel", "abort", "callback_error", "source_change"])
def test_panel_action_callback_cannot_dispatch_after_stop(rig, how):
    holder = []

    def emit(event):
        if event["event"] != "action_proposed" or event["payload"].get("visible_label") != "SCAVENGER HUNT":
            return
        if how == "cancel":
            rig.flags.cancelled = True
        elif how == "abort":
            holder[0].abort()
        elif how == "callback_error":
            raise RuntimeError("private-driver-error")
        else:
            write(rig.inv / "stopped.json", {})

    owner = make(rig, emit=emit)
    holder.append(owner)
    run(owner)
    assert owner.phase == "stopped" and rig.page.clicks == ["ASSESS", "OK"]
    assert not [c for c in rig.calls if c[0] == "panel_click"]


def test_missing_journal_preserves_stopped_report_without_claiming_current_hash(rig):
    owner = make(rig)
    rig.journal.path.rename(rig.journal.path.with_suffix(".unavailable"))
    owner.advance()
    assert owner.phase == "stopped" and owner.report["journal_sha256"] is None
    assert (owner.output / "stopped.json").exists() and not rig.page.clicks


def test_terminal_report_write_failure_is_not_success(rig, monkeypatch):
    original = module.persist_json

    def fail_final(path, value):
        if path.name == "report.json" and path.parent.name == "coordinator":
            raise OSError("fixture-no-space")
        return original(path, value)

    monkeypatch.setattr(module, "persist_json", fail_final)
    owner = make(rig)
    run(owner)
    assert owner.phase == "stopped" and owner.finished
    assert (owner.output / "stopped.json").exists()
    assert not (owner.output / "report.json").exists()


@pytest.mark.parametrize("change", ["report", "events"])
def test_scoring_terminal_artifacts_are_verified_before_parent_completion(rig, monkeypatch, change):
    owner = make(rig, allow_score_transfer=True)
    advance_until(
        owner, lambda: owner.phase == "scoring_active" and owner.scoring.phase == "score_transfer_ready"
    )
    original = owner.scoring.advance

    def altered():
        result = original()
        if change == "report":
            write(owner.scoring.output / "report.json", {})
        else:
            (owner.scoring.output / "events.jsonl").write_text("")
        return result

    monkeypatch.setattr(owner.scoring, "advance", altered)
    owner.advance()
    assert owner.phase == "stopped" and not owner.report["score_transfer_verified"]
    assert rig.journal.load().reduce().receipt("score_transfer") is not None
    assert not (owner.output / "report.json").exists()


@pytest.mark.parametrize("how", ["cancel", "source_change", "callback_error"])
def test_terminal_callback_cannot_promote_failed_coordinator(rig, how):
    def emit(event):
        rig.events.append(event)
        if event["event"] != "episode_summary" or event["payload"].get("mode") != module.MODE:
            return
        if how == "cancel":
            rig.flags.cancelled = True
        elif how == "source_change":
            write(rig.root / "coordinator/panel-data_quality/stopped.json", {})
        else:
            raise RuntimeError("redact-this")

    owner = make(rig, allow_score_transfer=True, emit=emit)
    run(owner)
    assert owner.phase == "stopped" and not owner.report["score_transfer_verified"]
    assert not (owner.output / "report.json").exists()
    assert (owner.output / "stopped.json").exists()
    assert rig.journal.load().reduce().receipt("score_transfer") is not None
    assert not rig.journal.load().reduce().report()["project_completed"]
