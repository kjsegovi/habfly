"""Read-only eligibility for a future, separately guarded submission stage.

No checkbox change, button click, acknowledgement, navigation or journal append
is performed. Even an eligible report is not a durable submission authorization:
the later readiness/dispatch stages must revalidate their own current sources.
"""

import hashlib
import json
import math
import re
import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import assessment_config, persist_json, rows_hash
from .browser_no_planet_workflow import _Evidence
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_score_transfer import load_assessed_revision, read_outer_score, score_projection
from .browser_setup import RENDER_VISIBILITY_JS
from .browser_stellar import SIMULATION_URL
from .browser_submission_controls import SUBMIT_EXPOSED, bind_readiness, outer_submit_box
from .browser_submission_controls import submit_outer_exposed as _outer_exposed
from .contracts import RuntimeEvent
from .project_assessment import AssessmentLedger, parse_assessment_text
from .project_progress import ProjectJournal

MODE = "read_only_project_submission_preflight"
READY = "I am ready to submit project."
SUBMIT = "Submit Project"
SCORING_MODE = "canonical_project_assessment_and_score_transfer"
KINDS = ("assessment_data_quality", "assessment_scavenger_hunt", "score_transfer")
_EXPOSED = (
    """(e,kind)=>{"""
    + RENDER_VISIBILITY_JS
    + """
  const r=e.getBoundingClientRect();
  if(!((kind==='checkbox'&&e.tagName==='INPUT'&&e.type==='checkbox')||
       (kind==='button'&&e.tagName==='BUTTON')) || !styled(e) ||
       r.width<=0||r.height<=0||r.left<0||r.top<0||r.right>innerWidth||r.bottom>innerHeight)return null;
  for(let p=e.parentElement;p;p=p.parentElement){const s=getComputedStyle(p),b=p.getBoundingClientRect();
    if((['hidden','clip','scroll','auto'].includes(s.overflowX)&&(r.left<b.left||r.right>b.right))||
       (['hidden','clip','scroll','auto'].includes(s.overflowY)&&(r.top<b.top||r.bottom>b.bottom)))return null;
  }
  for(const x of [r.left+.5,r.x+r.width/2,r.right-.5])
    for(const y of [r.top+.5,r.y+r.height/2,r.bottom-.5]){
      const h=document.elementFromPoint(x,y);if(!(h===e||e.contains(h)))return null;
    }
  return {x:r.x,y:r.y,width:r.width,height:r.height};
}"""
)


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("project_submission_preflight_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _simulation(report):
    frames = [f for f in report["frames"] if f["url"] == SIMULATION_URL]
    _require(report.get("ignored_frame_urls") == [] and len(frames) == 1, "invalid_capture")
    return frames[0]["text"]


def _sources(book, journal, scoring_dir):
    """Canonical eligibility plus the completed scoring owner's pinned chain."""
    _require(isinstance(journal, ProjectJournal), "canonical_journal_required")
    journal_path = book.path(journal.path)
    _require(journal_path.parent == book.history, "journal_outside_attempt")
    _require(
        {book.path(p) for p in book.history.glob("project-progress-*.jsonl")} == {journal_path},
        "ambiguous_canonical_journal",
    )
    progress = journal.load()
    state = progress.reduce()
    raw_journal = book.read(journal_path)
    _require(journal.load() == progress, "canonical_journal_changed")
    _require(
        len(state.stars) == 30 and all(star.task_completed for star in state.stars.values()),
        "thirty_exact_verified_tasks_required",
    )
    _require(
        not state.pending and not any(r.write_kind == "submission" for r in state.reservations),
        "pending_or_submitted_write",
    )
    canonical = [state.receipt(kind) for kind in KINDS]
    _require(all(canonical), "both_current_assessments_and_score_required")
    rows = canonical[0].project_rows_sha256
    _require(
        rows is not None
        and all(r.revision == state.revision and r.project_rows_sha256 == rows for r in canonical),
        "current_revision_or_rows_mismatch",
    )
    directory = book.clean(scoring_dir)
    report = book.json(directory / "report.json")
    scope = book.json(directory / "scope.json")
    _require(
        report.get("mode") == scope.get("mode") == SCORING_MODE
        and report.get("status") == "completed"
        and report.get("phase") == "score_transferred_not_submitted"
        and report.get("finished") is True
        and report.get("score_transfer_verified") is True
        and report.get("assessment_verified") == {"data_quality": True, "scavenger_hunt": True}
        and all(value is True for value in report["assessment_verified"].values())
        and type(report.get("acknowledgements_verified")) is int
        and report["acknowledgements_verified"] == 2
        and report.get("pending_canonical_action") is None
        and report.get("pending_acknowledgement") is None
        and report.get("failure_reason") is None
        and all(
            report.get(key) is False
            for key in (
                "event_forwarding_failed",
                "assessment_outcome_uncertain",
                "submitted",
                "task_completed",
                "project_completed",
                "scientific_verified",
                "submission_enabled",
                "automatic_retry",
            )
        )
        and report.get("allow_score_transfer") is True
        and type(report.get("data_revision")) is int
        and report["data_revision"] == state.revision
        and report.get("project_rows_sha256") == rows
        and report.get("journal_sha256") == _sha(raw_journal)
        and all(report.get(key) == value for key, value in scope.items()),
        "unsupported_scoring_completion",
    )
    declared = report.get("source_sha256")
    _require(isinstance(declared, dict) and 1 <= len(declared) <= 3000, "missing_scoring_sources")
    for name, digest in declared.items():
        _require(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and isinstance(digest, str)
            and re.fullmatch(r"[a-f0-9]{64}", digest),
            "invalid_source_hash",
        )
        _require(_sha(book.read(book.history / name)) == digest, "scoring_source_changed")
    stream = book.read(directory / "events.jsonl")
    _require(_sha(stream) == report.get("events_sha256"), "scoring_events_changed")
    events = [RuntimeEvent.model_validate_json(line) for line in stream.splitlines()]
    _require(
        1 <= len(events) <= 2000
        and events[0].run_id is not None
        and all(
            event.sequence == i and event.run_id == events[0].run_id and event.event != "error"
            for i, event in enumerate(events)
        )
        and events[-1].event == "episode_summary"
        and events[-1].payload == {key: value for key, value in report.items() if key != "events_sha256"},
        "scoring_event_completion_mismatch",
    )
    paths = report.get("artifact_paths", {})
    _require(set(paths) == {"assessments", "score_transfer"}, "unsupported_scoring_artifacts")
    assessments, score = [book.clean(book.history / paths[key]) for key in ("assessments", "score_transfer")]
    _require(score == directory / "score-transfer", "score_directory_changed")
    receipts = []
    for index, item in enumerate(canonical[:2]):
        path = assessments / f"attempt-{index:03d}-confirmed.json"
        raw = book.read(path)
        payload = json.loads(raw)
        ledger = AssessmentLedger.model_validate(payload["ledger"])
        _require(
            _sha(raw) == item.source_sha256
            and ledger == item.assessment
            and payload["receipt"] == ledger.attempts[-1].model_dump(mode="json")
            and payload["project_rows_sha256"] == rows,
            "assessment_canonical_source_mismatch",
        )
        for section in ("before", "after"):
            capture = book.capture(assessments / f"attempt-{index:03d}-{section}")
            _require(rows_hash(_simulation(capture)) == rows, "assessment_rows_changed")
        receipts.append(payload)
    latest = assessments / "attempt-001-confirmed.json"
    assessed, expected_rows, provenance = load_assessed_revision(latest)
    _require(
        expected_rows == rows and provenance["data_revision"] == state.revision and assessed.collected == 30,
        "assessed_revision_changed",
    )
    payload = book.json(score / "confirmed.json")
    _require(
        _sha(book.read(score / "confirmed.json")) == canonical[2].source_sha256
        and all(payload.get(k) == v for k, v in provenance.items())
        and payload.get("score_transfer_verified") is True
        and all(
            payload.get(k) is False
            for k in (
                "score_computed_locally",
                "submitted",
                "task_completed",
                "browser_acceptance_passed",
                "automatic_retry",
                "save_verified",
            )
        ),
        "score_canonical_source_mismatch",
    )
    _require(
        book.json(score / "acknowledgement.json") == {"visible_text": "Score updated."}
        and book.json(score / "score-readback.json")
        == {key: payload[key] for key in ("score_before", "score_after")},
        "score_acknowledgement_missing",
    )
    try:
        outer_score = float(payload["score_after"])
    except (ValueError, TypeError):
        raise BrowserSafetyStop("project_submission_preflight_invalid_score") from None
    _require(math.isfinite(outer_score) and outer_score == canonical[2].score, "canonical_score_changed")
    before, after = [book.capture(score / name) for name in ("before", "after")]
    _require(
        score_projection(before) == score_projection(after, feedback=True)
        and all(rows_hash(_simulation(capture)) == rows for capture in (before, after)),
        "score_capture_changed",
    )
    current_assessment = parse_assessment_text(_simulation(after))
    _require(
        current_assessment.acknowledgement is None and current_assessment.collected == 30,
        "unsettled_assessment_source",
    )
    required = {
        str(path.relative_to(book.history))
        for path in (directory / "scope.json", latest, score / "confirmed.json")
    }
    _require(required <= declared.keys(), "scoring_sources_not_pinned")
    book.unchanged()
    return state, rows, outer_score, current_assessment


def _native_controls(page, boundary, report):
    checkbox = page.get_by_role("checkbox", name=READY, exact=True)
    _require(
        checkbox.count() == 1 and checkbox.is_visible() and checkbox.is_enabled(),
        "readiness_control_unavailable",
    )
    _require(checkbox.is_checked() is False, "readiness_must_be_unchecked")
    readiness = bind_readiness(
        page, checkbox, checked=False, exposed=_EXPOSED, prefix="project_submission_preflight_"
    )
    checkbox_box = readiness.box
    allowed = {rule.url for rule in boundary.frames if rule.url != SIMULATION_URL}
    buttons = [
        (frame, button)
        for frame in page.frames
        if frame.url in allowed
        for button in frame.get_by_role("button", name=SUBMIT, exact=True).all()
    ]
    _require(len(buttons) == 1, "submit_control_ambiguous_or_missing")
    frame, button = buttons[0]
    _require(frame.parent_frame == page.main_frame and button.is_visible(), "submit_control_unavailable")
    button_box = button.evaluate(SUBMIT_EXPOSED)
    _require(
        isinstance(button_box, dict) and _outer_exposed(frame, [outer_submit_box(button_box)]),
        "submit_control_unexposed",
    )
    enabled = button.is_enabled()
    _require(type(enabled) is bool, "unknown_submit_enabled_state")
    expected_checkbox = '- checkbox "' + READY + '"'
    expected_button = '- button "' + SUBMIT + '"' + ("" if enabled else " [disabled]")
    outer = [
        control
        for control in report["outer_controls"]
        if control["role"] == "checkbox" and READY in control["accessibility"]
    ]
    captured = [
        (f, c)
        for f in report["frames"]
        for c in f["controls"]
        if c["role"] == "button" and SUBMIT in c["accessibility"]
    ]
    _require(
        not any(c["role"] == "button" and SUBMIT in c["accessibility"] for c in report["outer_controls"])
        and not any(
            c["role"] == "checkbox" and READY in c["accessibility"]
            for f in report["frames"]
            for c in f["controls"]
        ),
        "submission_control_wrong_surface",
    )
    _require(
        len(outer) == len(captured) == 1
        and outer[0]["accessibility"] == checkbox.aria_snapshot() == expected_checkbox
        and outer[0].get("enabled") is True
        and outer[0].get("protected") is True,
        "readiness_capture_mismatch",
    )
    captured_frame, control = captured[0]
    rule = next(rule for rule in boundary.frames if rule.url == frame.url)
    siblings = [candidate for candidate in page.frames if candidate.url == frame.url]
    _require(
        len(siblings) == rule.count
        and captured_frame["id"] == f"{rule.name}-{siblings.index(frame)}"
        and captured_frame["url"] == frame.url
        and control["accessibility"] == button.aria_snapshot() == expected_button
        and control.get("enabled") is enabled
        and control.get("protected") is True,
        "submit_capture_mismatch",
    )
    return {
        "readiness": {
            "target_id": outer[0]["id"],
            "label": READY,
            "checked": False,
            "enabled": True,
            "visible_box": checkbox_box,
            **(
                {"click_surface": readiness.mode, "checked_state_source": "native_checkbox"}
                if readiness.mode == "associated_painted_label"
                else {}
            ),
        },
        "submit": {
            "target_id": control["id"],
            "label": SUBMIT,
            "frame_id": captured_frame["id"],
            "enabled": enabled,
            "visible_box": button_box,
        },
    }, (readiness, button.element_handle(timeout=2000))


def preflight_project_submission(
    page,
    config,
    output,
    *,
    run_history,
    journal,
    scoring_dir,
    timeout_seconds=30,
    cancelled=lambda: False,
    _clock=time.monotonic,
):
    """Read prerequisites and controls twice; never reserve or execute a write."""
    _require(
        type(timeout_seconds) in {int, float}
        and math.isfinite(timeout_seconds)
        and 0.1 <= timeout_seconds <= 60
        and callable(cancelled)
        and callable(_clock),
        "invalid_runtime_options",
    )
    deadline = _clock() + timeout_seconds
    book = _Evidence(run_history)
    try:
        state, rows, expected_score, assessed = _sources(book, journal, scoring_dir)
    except BrowserSafetyStop:
        raise
    except Exception:  # noqa: BLE001 - malformed artifact errors must not leak private paths
        raise BrowserSafetyStop("project_submission_preflight_invalid_source_chain") from None
    directory = book.path(output)
    _require(directory != book.history and not directory.exists(), "output_already_exists")
    boundary = assessment_config(config)
    rules = [r for r in boundary.frames if r.url == SIMULATION_URL]
    _require(len(rules) == 1 and rules[0].count == 1, "unsupported_simulation_boundary")
    directory.mkdir(parents=True, exist_ok=False)
    frames, dialogs, popups = page.frames.copy(), [], []

    # Observe unexpected modal events without dismissing/accepting anything.
    def dialog_handler(_):
        dialogs.append(True)

    def popup_handler(_):
        popups.append(True)

    page.on("dialog", dialog_handler)
    page.context.on("page", popup_handler)
    try:

        def check():
            _require(not cancelled(), "cancelled")
            _require(_clock() < deadline, "time_limit")
            page.wait_for_timeout(0)
            _require(
                not dialogs
                and not popups
                and page.frames == frames
                and len(page.context.pages) == 1
                and boundary.allows(page.url),
                "context_changed",
            )
            _require(
                {book.path(p) for p in book.history.glob("project-progress-*.jsonl")}
                == {book.path(journal.path)},
                "ambiguous_canonical_journal",
            )
            book.unchanged()

        def read():
            check()
            report = inspect_page(page, boundary)
            text = _simulation(report)
            visible = parse_assessment_text(text)
            _require(visible == assessed and rows_hash(text) == rows, "current_assessment_or_rows_changed")
            score = read_outer_score(page)
            _require(float(score) == expected_score, "current_score_changed")
            controls, handles = _native_controls(page, boundary, report)
            check()
            return report, controls, handles, score

        before, controls, handles, score = read()
        first = save_probe(before, directory / "before")
        after, newer, newer_handles, score_after = read()
        _require(
            screen_identity(before) == screen_identity(after)
            and controls == newer
            and score == score_after
            and all(
                old.evaluate("(a,b)=>a.isConnected&&a===b", new) for old, new in zip(handles, newer_handles)
            ),
            "changed_during_readback",
        )
        last = save_probe(after, directory / "after")
        check()
        result = {
            "schema_version": 1,
            "mode": MODE,
            "authority": "read_only_prerequisites_and_visible_controls",
            "project_id": state.project_id,
            "attempt_id": state.attempt_id,
            "data_revision": state.revision,
            "journal_sha256": book.hashes[str(book.path(journal.path).relative_to(book.history))],
            "project_rows_sha256": rows,
            "verified_tasks": 30,
            "stars": [
                {
                    "id": star.id,
                    "name": star.name,
                    "revision": star.revision,
                    "task_receipt_sha256": star.completion.source_sha256,
                }
                for star in state.stars.values()
            ],
            "assessment_revision_verified": True,
            "score_transfer_revision_verified": True,
            "visible_score": score_after,
            "controls": controls,
            "eligible_for_readiness_selection": True,
            "readiness_selection_required": True,
            "eligible_for_grounded_dispatch": controls["submit"]["enabled"],
            "eligibility_scope": "future_guarded_readiness_and_submission_sequence_not_current_click",
            "dispatch_blocker": None
            if controls["submit"]["enabled"]
            else "submit_disabled_readiness_effect_unverified",
            "submission_button_enablement_inferred": False,
            "fresh_dispatch_validation_required": True,
            "submit_click_authorized": False,
            "dispatch_implemented": False,
            "source_sha256": dict(book.hashes),
            "capture_sha256": {"before": first["observation_sha256"], "after": last["observation_sha256"]},
            "browser_actions": 0,
            "checkbox_writes": 0,
            "submission_clicks": 0,
            "journal_writes": 0,
            "acknowledgement_observed": False,
            "submitted": False,
            "task_completed": False,
            "project_completed": False,
            "scientific_verified": False,
            "automatic_retry": False,
        }
        persist_json(directory / "confirmed.json", result)
        return result
    except BaseException as exc:
        reason = (
            str(exc) if isinstance(exc, BrowserSafetyStop) else "project_submission_preflight_read_failed"
        )
        persist_json(
            directory / "stopped.json",
            {
                "reason": reason,
                "eligible_for_grounded_dispatch": False,
                "browser_actions": 0,
                "journal_writes": 0,
                "submitted": False,
                "project_completed": False,
                "automatic_retry": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop(reason) from None
    finally:
        page.remove_listener("dialog", dialog_handler)
        page.context.remove_listener("page", popup_handler)
