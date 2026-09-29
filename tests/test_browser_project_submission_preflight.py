"""Pure injected submission preflight: no launches, UI actions or journal writes."""
# ruff: noqa: F811

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_numeric import WIDGET, config
from test_browser_project_scoring_steps import both, case, make  # noqa: F401
from test_project_evidence import write

import habfly.browser_project_submission_preflight as module
from habfly.browser import BrowserSafetyStop
from habfly.project_progress import Active, ProjectJournal, WriteReserved, _json


@pytest.fixture
def subject(case, monkeypatch):
    scoring = make(case, allow_score_transfer=True)
    both(case, scoring)
    scoring.advance()
    assert scoring.report["status"] == "completed"
    page = case.page
    current_report = page.report
    state = SimpleNamespace(
        reads=0,
        clock=0,
        hook=None,
        popup_callbacks={},
        dialog_callbacks={},
        outer_exposed=True,
        cancelled=False,
        score="56.59",
    )

    class Control:
        def __init__(self, label, role):
            self.label, self.role = label, role
            self.visible = self.enabled = self.exposed = True
            self.checked, self.number = False, 1
            self.snapshot_override = None

        def count(self):
            return self.number

        def is_visible(self):
            return self.visible

        def is_enabled(self):
            return self.enabled

        def is_checked(self):
            return self.checked

        def aria_snapshot(self):
            if self.snapshot_override is not None:
                return self.snapshot_override
            return (
                f'- {self.role} "{self.label}"'
                + (" [checked]" if self.checked else "")
                + (" [disabled]" if not self.enabled else "")
            )

        def element_handle(self, **_):
            return self

        def evaluate(self, script, value=None):
            if script.startswith("(a,b)"):
                return self is value
            return {"x": 10, "y": 10, "width": 20, "height": 20} if self.exposed else None

        def click(self, **_):
            pytest.fail("Submission preflight must not click")

        def check(self, **_):
            pytest.fail("Submission preflight must not check readiness")

    ready, submit = Control(module.READY, "checkbox"), Control(module.SUBMIT, "button")
    state.ready, state.submit = ready, submit
    page.main_frame = object()
    simulation, widget = page.frames
    simulation.parent_frame = widget.parent_frame = page.main_frame
    update_widget = SimpleNamespace(
        url=WIDGET,
        parent_frame=page.main_frame,
        get_by_role=lambda *_a, **_k: SimpleNamespace(all=list),
    )
    page.frames.insert(1, update_widget)
    widget.get_by_role = lambda *_a, **_k: SimpleNamespace(all=lambda: [state.submit] * state.submit.number)
    page.context.on = lambda kind, callback: state.popup_callbacks.update({kind: callback})
    page.context.remove_listener = lambda kind, callback: state.popup_callbacks.pop(kind)
    page.on = lambda kind, callback: state.dialog_callbacks.update({kind: callback})
    page.remove_listener = lambda kind, callback: state.dialog_callbacks.pop(kind)
    page.get_by_role = lambda *_a, **_k: state.ready
    page.locator = lambda _selector: SimpleNamespace(inner_text=lambda **_: "Score: " + state.score)

    def report():
        state.reads += 1
        if state.hook:
            state.hook(state.reads)
        value = deepcopy(current_report())
        value["outer_controls"] = [
            {
                "id": "outer:c0",
                "role": "checkbox",
                "accessibility": state.ready.aria_snapshot(),
                "enabled": state.ready.enabled,
                "actions": [],
                "protected": True,
            }
        ]
        value["frames"].append(
            {
                "id": "widgets-1",
                "url": WIDGET,
                "text": module.SUBMIT,
                "accessibility": state.submit.aria_snapshot(),
                "controls": [
                    {
                        "id": "widgets-1:c0",
                        "role": "button",
                        "accessibility": state.submit.aria_snapshot(),
                        "enabled": state.submit.enabled,
                        "actions": [],
                        "protected": True,
                    }
                ],
            }
        )
        return value

    monkeypatch.setattr(module, "inspect_page", lambda _p, _c: report())
    monkeypatch.setattr(module, "_outer_exposed", lambda _f, _b: state.outer_exposed)
    return SimpleNamespace(case=case, page=page, state=state, scoring=scoring)


def run(subject, **options):
    item = subject.case
    return module.preflight_project_submission(
        subject.page,
        config(),
        item.root / "submission-preflight",
        run_history=item.root,
        journal=item.journal,
        scoring_dir=item.root / "scoring",
        cancelled=lambda: subject.state.cancelled,
        _clock=lambda: subject.state.clock,
        **options,
    )


def test_eligible_read_only_result_is_never_submission_or_write_authority(subject):
    before = subject.case.journal.path.read_bytes()
    actions = list(subject.page.clicks)
    result = run(subject)
    assert result["eligible_for_grounded_dispatch"] and result["eligible_for_readiness_selection"]
    assert result["readiness_selection_required"] and result["fresh_dispatch_validation_required"]
    assert (
        result["controls"]["readiness"]["checked"] is False
        and result["controls"]["submit"]["enabled"] is True
    )
    assert result["verified_tasks"] == 30 and result["visible_score"] == "56.59"
    assert all(
        result[key] is False
        for key in (
            "submit_click_authorized",
            "dispatch_implemented",
            "submitted",
            "task_completed",
            "project_completed",
            "scientific_verified",
            "acknowledgement_observed",
        )
    )
    assert all(
        result[key] == 0
        for key in ("browser_actions", "checkbox_writes", "submission_clicks", "journal_writes")
    )
    assert subject.case.journal.path.read_bytes() == before and subject.page.clicks == actions
    assert (
        subject.state.reads == 2 and not subject.state.dialog_callbacks and not subject.state.popup_callbacks
    )
    assert json.loads((subject.case.root / "submission-preflight/confirmed.json").read_bytes()) == result


def test_disabled_submit_is_only_readiness_eligibility_not_inferred_enablement(subject):
    subject.state.submit.enabled = False
    result = run(subject)
    assert result["eligible_for_readiness_selection"] and not result["eligible_for_grounded_dispatch"]
    assert not result["submission_button_enablement_inferred"]
    assert result["controls"]["submit"]["enabled"] is False


@pytest.mark.parametrize(
    "control,attribute,value",
    [
        ("ready", "number", 0),
        ("ready", "number", 2),
        ("ready", "visible", False),
        ("ready", "enabled", False),
        ("ready", "checked", True),
        ("ready", "exposed", False),
        ("submit", "number", 0),
        ("submit", "number", 2),
        ("submit", "visible", False),
        ("submit", "exposed", False),
        ("submit", "enabled", 1),
        ("submit", "snapshot_override", '- button "Other"'),
    ],
)
def test_unknown_missing_hidden_covered_or_changed_controls_fail_closed(subject, control, attribute, value):
    setattr(getattr(subject.state, control), attribute, value)
    before = subject.case.journal.path.read_bytes()
    with pytest.raises(BrowserSafetyStop):
        run(subject)
    assert subject.case.journal.path.read_bytes() == before and subject.state.reads <= 2
    assert not (subject.case.root / "submission-preflight/confirmed.json").exists()
    stopped = json.loads((subject.case.root / "submission-preflight/stopped.json").read_bytes())
    assert not stopped["eligible_for_grounded_dispatch"] and stopped["browser_actions"] == 0


def test_outer_iframe_overlay_cannot_certify_button(subject):
    subject.state.outer_exposed = False
    with pytest.raises(BrowserSafetyStop, match="unexposed"):
        run(subject)


def test_same_url_widget_reordering_is_not_observation_local_target_identity(subject):
    subject.page.frames[1:] = reversed(subject.page.frames[1:])
    with pytest.raises(BrowserSafetyStop, match="submit_capture_mismatch"):
        run(subject)


def test_nested_submit_widget_is_not_the_declared_outer_widget(subject):
    subject.page.frames[-1].parent_frame = subject.page.frames[0]
    with pytest.raises(BrowserSafetyStop, match="submit_control_unavailable"):
        run(subject)


@pytest.mark.parametrize(
    "mutation",
    [
        "control",
        "handle",
        "score",
        "journal",
        "source",
        "rows",
        "url",
        "frames",
        "popup",
        "dialog",
        "cancel",
        "deadline",
    ],
)
def test_second_read_changes_do_not_create_eligibility(subject, mutation):
    def change(index):
        if index != 2:
            return
        if mutation == "control":
            subject.state.submit.enabled = False
        elif mutation == "handle":
            subject.state.submit = deepcopy(subject.state.submit)
        elif mutation == "score":
            subject.state.score = "99"
        elif mutation == "journal":
            subject.case.journal.append(Active(stage="submission"))
        elif mutation == "source":
            write(subject.case.root / "scoring/scope.json", {"changed": True})
        elif mutation == "rows":
            subject.page.rows += " changed"
        elif mutation == "url":
            subject.page.url = "http://localhost/other"
        elif mutation == "frames":
            subject.page.frames.append(object())
        elif mutation == "popup":
            subject.state.popup_callbacks["page"](object())
        elif mutation == "dialog":
            subject.state.dialog_callbacks["dialog"](
                SimpleNamespace(dismiss=lambda: pytest.fail("Must not dismiss dialog"))
            )
        elif mutation == "cancel":
            subject.state.cancelled = True
        else:
            subject.state.clock = 30

    subject.state.hook = change
    with pytest.raises(BrowserSafetyStop):
        run(subject)
    assert not (subject.case.root / "submission-preflight/confirmed.json").exists()
    assert not subject.state.dialog_callbacks and not subject.state.popup_callbacks


@pytest.mark.parametrize(
    "kind",
    ["missing_assessment", "missing_score", "pending", "twenty_nine", "incomplete_task", "ambiguous_attempt"],
)
def test_canonical_prerequisites_fail_before_any_page_read(subject, kind):
    journal = subject.case.journal
    progress = journal.load()
    if kind == "ambiguous_attempt":
        ProjectJournal(subject.case.root, project_id="other", attempt_id="other").create()
    else:
        if kind == "missing_assessment":
            progress.records = progress.records[:-6]
        elif kind == "missing_score":
            progress.records = progress.records[:-2]
        elif kind == "pending":
            progress = progress.append(
                WriteReserved(
                    action_id="pending-save",
                    write_kind="save_star",
                    star_id=next(iter(progress.reduce().stars)),
                    revision=progress.reduce().revision,
                    before_sha256="a" * 64,
                )
            )
        elif kind == "twenty_nine":
            progress.records = progress.records[:-13]
        else:
            progress.records = progress.records[:-7]
        journal.path.write_text(
            _json(progress.header())
            + "\n"
            + "".join(_json(r.model_dump(mode="json")) + "\n" for r in progress.records)
        )
    with pytest.raises(BrowserSafetyStop):
        run(subject)
    assert subject.state.reads == 0 and not (subject.case.root / "submission-preflight").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "scoring/scope.json",
        "scoring/events.jsonl",
        "scoring/score-transfer/confirmed.json",
        "scoring/score-transfer/acknowledgement.json",
        "scoring/score-transfer/after/observation.json",
    ],
)
def test_frozen_scoring_sources_cannot_change_before_preflight(subject, relative):
    write(subject.case.root / relative, {"changed": True})
    with pytest.raises(BrowserSafetyStop):
        run(subject)
    assert subject.state.reads == 0


def test_cancelled_start_never_reads_or_mutates_page(subject):
    subject.state.cancelled = True
    with pytest.raises(BrowserSafetyStop, match="cancelled"):
        run(subject)
    assert subject.state.reads == 0


def test_duplicate_output_never_overwrites_prior_eligibility(subject):
    run(subject)
    path = subject.case.root / "submission-preflight/confirmed.json"
    before = path.read_bytes()
    with pytest.raises(BrowserSafetyStop, match="output_already_exists"):
        run(subject)
    assert path.read_bytes() == before and subject.state.reads == 2
