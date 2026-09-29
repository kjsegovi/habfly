"""Injected local submission transport; no real browser or course receipt."""
# ruff: noqa: F811

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_numeric import config
from test_browser_project_scoring_steps import case  # noqa: F401
from test_browser_project_submission_preflight import subject  # noqa: F401

import habfly.browser_project_submission_preflight as preflight
import habfly.browser_project_submission_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.project_progress import Active, WriteReserved


@pytest.fixture
def setup(subject, monkeypatch):
    state, page = subject.state, subject.page
    state.native = []
    state.images = []
    state.dispatch_error = state.screenshot_error = state.readiness_noop = False
    state.image_hook = state.dispatch_hook = None
    state.after_text = "Awaiting a real course response; fixture only"
    state.image_bytes = b"\x89PNG\r\n\x1a\nfixture-not-real-course-image"
    state.body = SimpleNamespace(
        inner_text=lambda **_: state.after_text,
        aria_snapshot=lambda **_: "- text: " + state.after_text,
    )
    state.observer = None

    def click(label):
        pending = subject.case.journal.load().reduce().pending
        assert len(pending) == 1 and pending[0].write_kind == "submission"
        state.native.append(label)
        if state.dispatch_hook:
            state.dispatch_hook(label)
        if state.dispatch_error:
            raise RuntimeError("PRIVATE DRIVER SESSION URL")
        if label == "readiness" and not state.readiness_noop:
            state.ready.checked = True

    state.ready.click = lambda **_: click("readiness")
    state.submit.click = lambda **_: click("submit")

    def screenshot(**options):
        assert options["full_page"] is False and 0 < options["timeout"] <= 5000
        state.images.append(options)
        if state.image_hook:
            state.image_hook()
        if state.screenshot_error:
            raise RuntimeError("PRIVATE SCREENSHOT FAILURE")
        return state.image_bytes

    def add(kind, callback, *, store):
        store.setdefault(kind, []).append(callback)

    def remove(kind, callback, *, store):
        store[kind].remove(callback)

    page.on = lambda kind, callback: add(kind, callback, store=state.dialog_callbacks)
    page.remove_listener = lambda kind, callback: remove(kind, callback, store=state.dialog_callbacks)
    page.context.on = lambda kind, callback: add(kind, callback, store=state.popup_callbacks)
    page.context.remove_listener = lambda kind, callback: remove(kind, callback, store=state.popup_callbacks)
    page.locator = lambda selector: state.body
    page.screenshot = screenshot
    monkeypatch.setattr(module, "inspect_page", preflight.inspect_page)
    monkeypatch.setattr(module, "read_outer_score", lambda _: state.score)
    monkeypatch.setattr(preflight, "read_outer_score", lambda _: state.score)
    monkeypatch.setattr(module, "_outer_exposed", lambda *_: state.outer_exposed)
    # This suite injects non-browser frame objects; the separate feedback-frame
    # tests exercise the real embedding-chain visibility helper.
    monkeypatch.setattr(module, "_visible_frame", lambda *_: True)
    return subject


def create(setup, **options):
    return module.BrowserProjectSubmissionSteps(
        setup.page,
        config(),
        setup.case.root / options.pop("output", "submission"),
        run_history=setup.case.root,
        journal=setup.case.journal,
        scoring_dir=setup.case.root / "scoring",
        allow_submission=options.pop("allow_submission", True),
        cancelled=lambda: setup.state.cancelled,
        _clock=lambda: setup.state.clock,
        **options,
    )


def to_stage(setup, phase, **options):
    owner = create(setup, **options)
    for _ in range(4):
        if owner.phase == phase:
            return owner
        owner.advance()
        assert not owner.finished, owner.state()
    raise AssertionError("Requested fixture stage not reached")


def test_offline_constructor_and_explicit_opt_in(setup):
    before = setup.case.journal.path.read_bytes()
    for flag in (False, 1, "true", None):
        with pytest.raises(BrowserSafetyStop, match="explicit_opt_in"):
            create(setup, allow_submission=flag)
    assert setup.state.reads == 0 and not (setup.case.root / "submission").exists()
    owner = create(setup)
    assert setup.state.reads == 0 and not setup.state.native and not setup.state.images
    assert owner.phase == "preflight_pending" and owner.status == "paused"
    assert setup.case.journal.path.read_bytes() == before
    owner.close()


def test_exact_two_dispatches_end_unknown_with_preserved_evidence_not_receipt(setup, monkeypatch):
    monkeypatch.setattr(
        module, "preserve_submission_outcome", lambda *_a, **_k: pytest.fail("No fallback on normal capture")
    )
    events = []
    original = setup.case.journal.load()
    owner = create(setup, emit=events.append)
    assert owner.advance()["phase"] == "selecting_readiness", owner.state()
    assert setup.state.native == []
    reserved = setup.case.journal.load()
    assert reserved.records[:-1] == original.records
    assert reserved.records[-1].payload == owner._proposal
    assert len(reserved.reduce().pending) == 1
    assert owner.advance()["phase"] == "submitting", owner.state()
    assert setup.state.native == ["readiness"] and setup.state.ready.checked
    assert owner.advance()["phase"] == "capturing_outcome", owner.state()
    assert setup.state.native == ["readiness", "submit"]
    assert owner.advance()["phase"] == "unknown_pending", owner.state()
    assert owner.status == "stopped" and owner.failure == "project_submission_acknowledgement_not_grounded"
    assert owner.report["submit_click_returned"] and owner.report["submission_outcome"] == "unknown"
    assert not owner.report["submitted"] and not owner.report["project_completed"]
    assert setup.case.journal.load() == reserved
    assert setup.case.journal.load().reduce().receipt("submission") is None
    assert (owner.output / "before/viewport.png").read_bytes() == setup.state.image_bytes
    assert (owner.output / "post-submit/viewport.png").read_bytes() == setup.state.image_bytes
    assert not (owner.output / "confirmed.json").exists()
    assert not (owner.output / "post-submit-failure").exists()
    assert "failure_outcome_diagnostic" not in owner.report
    saved_events = [json.loads(line) for line in (owner.output / "events.jsonl").read_text().splitlines()]
    assert saved_events == events
    assert [e["sequence"] for e in events] == list(range(len(events)))
    assert events[-1]["event"] == "episode_summary"
    assert events[-1]["payload"] == {k: v for k, v in owner.report.items() if k != "events_sha256"}
    assert (
        owner.report["events_sha256"]
        == hashlib.sha256((owner.output / "events.jsonl").read_bytes()).hexdigest()
    )
    for _ in range(3):
        owner.advance()
        owner.tick()
        owner.resume()
        owner.close()
    assert setup.state.native == ["readiness", "submit"]
    with pytest.raises(BrowserSafetyStop, match="pending_or_submitted"):
        create(setup, output="another-output")
    with pytest.raises(ValueError, match="write_outcome_uncertain_no_retry"):
        setup.case.journal.append(
            WriteReserved(
                action_id="retry",
                write_kind="submission",
                revision=owner.revision,
                project_rows_sha256=owner.rows_sha,
                before_sha256="1" * 64,
            )
        )


@pytest.mark.parametrize(
    "boundary",
    [
        "hello",
        "before_reservation",
        "after_reservation",
        "readiness",
        "submit",
        "after_readiness",
        "after_submit",
    ],
)
@pytest.mark.parametrize("effect", ["abort", "fail", "reenter"])
def test_callbacks_cannot_dispatch_after_cancel_or_failure(setup, boundary, effect):
    holder = {}

    def callback(event):
        hit = (
            (boundary == "hello" and event["event"] == "hello")
            or (
                boundary == "before_reservation"
                and event["payload"].get("boundary") == "before_canonical_reservation"
            )
            or (
                boundary == "after_reservation"
                and event["payload"].get("boundary") == "canonical_reservation_verified"
            )
            or (
                boundary in {"readiness", "submit"}
                and event["event"] == "action_proposed"
                and event["payload"].get("target") == boundary
            )
            or (
                boundary in {"after_readiness", "after_submit"}
                and event["event"] == "action_result"
                and event["payload"].get("target") == boundary.removeprefix("after_")
            )
        )
        if not hit:
            return
        if effect == "fail":
            raise RuntimeError("PRIVATE SESSION")
        getattr(holder["owner"], "abort" if effect == "abort" else "advance")()

    holder["owner"] = owner = create(setup, emit=callback)
    for _ in range(4):
        owner.advance()
    assert owner.finished
    expected = (
        ["readiness", "submit"]
        if boundary == "after_submit"
        else ["readiness"]
        if boundary in {"submit", "after_readiness"}
        else []
    )
    assert setup.state.native == expected
    pending = setup.case.journal.load().reduce().pending
    assert bool(pending) == (boundary not in {"hello", "before_reservation"})
    assert (owner.phase == "unknown_pending") == bool(pending)
    assert "PRIVATE" not in (owner.output / "events.jsonl").read_text()
    assert "PRIVATE" not in (owner.output / "report.json").read_text()


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("preflight_pending", []),
        ("selecting_readiness", []),
        ("submitting", ["readiness"]),
        ("capturing_outcome", ["readiness", "submit"]),
    ],
)
def test_abort_every_stage_is_sticky_and_does_not_take_new_screenshots(setup, phase, expected):
    owner = to_stage(setup, phase)
    captures = len(setup.state.images)
    owner.abort()
    for _ in range(3):
        owner.advance()
        owner.tick()
        owner.resume()
        owner.close()
    assert owner.status == "aborted" and setup.state.native == expected
    assert len(setup.state.images) == captures


@pytest.mark.parametrize("phase", ["selecting_readiness", "submitting"])
@pytest.mark.parametrize(
    "mutation", ["source", "journal_append", "journal_rewrite", "reservation", "config", "limit"]
)
def test_changed_sources_or_exact_journal_prefix_block_each_write(setup, phase, mutation):
    owner = to_stage(setup, phase)
    if mutation == "source":
        (setup.case.root / "scoring/scope.json").write_text("{}")
    elif mutation == "journal_append":
        setup.case.journal.append(Active(stage="submission"))
    elif mutation == "journal_rewrite":
        path = setup.case.journal.path
        path.write_bytes(path.read_bytes().replace(b'"sequence":0', b'"sequence": 0', 1))
    elif mutation == "reservation":
        owner.claim.write_text("{}")
    elif mutation == "config":
        owner.config.max_controls += 1
    else:
        owner.max_seconds += 1
    before = list(setup.state.native)
    owner.advance()
    assert owner.finished and owner.phase == "unknown_pending"
    assert setup.state.native == before and not owner.report["submitted"]


@pytest.mark.parametrize("phase", ["selecting_readiness", "submitting"])
@pytest.mark.parametrize(
    "mutation",
    ["checkbox_overlay", "button_overlay", "frame_overlay", "replaced", "rows", "score", "checked"],
)
def test_native_changes_at_action_callback_block_the_dispatch(setup, phase, mutation):
    owner = to_stage(setup, phase)
    before = list(setup.state.native)

    def callback(event):
        if event["event"] != "action_proposed":
            return
        if mutation == "checkbox_overlay":
            setup.state.ready.exposed = False
        elif mutation == "button_overlay":
            setup.state.submit.exposed = False
        elif mutation == "frame_overlay":
            setup.state.outer_exposed = False
        elif mutation == "replaced":
            setup.state.submit = deepcopy(setup.state.submit)
        elif mutation == "rows":
            setup.page.rows += " changed"
        elif mutation == "score":
            setup.state.score = "57.01"
        else:
            setup.state.ready.checked = not setup.state.ready.checked

    owner._callback = callback
    owner.advance()
    assert owner.finished and owner.phase == "unknown_pending"
    assert setup.state.native == before


@pytest.mark.parametrize("phase", ["selecting_readiness", "submitting"])
def test_click_exception_is_uncertain_not_retried(setup, phase):
    owner = to_stage(setup, phase)
    setup.state.dispatch_error = True
    owner.advance()
    assert owner.phase == "unknown_pending" and owner.finished
    assert owner.readiness_may_have_occurred
    assert owner.submit_may_have_occurred is (phase == "submitting")
    calls = list(setup.state.native)
    owner.advance()
    assert setup.state.native == calls and "PRIVATE" not in json.dumps(owner.report)


def test_readiness_that_does_not_change_or_changes_other_values_cannot_submit(setup):
    owner = to_stage(setup, "selecting_readiness")
    setup.state.readiness_noop = True
    owner.advance()
    assert owner.phase == "unknown_pending" and not owner.readiness_verified
    assert setup.state.native == ["readiness"]


@pytest.mark.parametrize("when", ["before", "post_submit"])
@pytest.mark.parametrize(
    "failure", ["screenshot", "invalid_png", "changed_during_image", "auth", "dialog", "navigation"]
)
def test_diagnostic_failures_never_bypass_guards_or_produce_success(setup, monkeypatch, when, failure):
    owner = create(setup) if when == "before" else to_stage(setup, "capturing_outcome")
    if failure == "screenshot":
        setup.state.screenshot_error = True
    elif failure == "invalid_png":
        setup.state.image_bytes = b"not a PNG"
    elif failure == "changed_during_image":
        setup.state.image_hook = lambda: setattr(setup.state.submit, "enabled", False)
    elif failure == "auth":

        def authentication(*_):
            raise BrowserSafetyStop("authentication_required")

        monkeypatch.setattr(module, "inspect_page", authentication)
    elif failure == "dialog":
        owner._watch()
        owner._dialog_handler(None)
    else:
        setup.page.url = "http://localhost/not-authorized"
    before = len(setup.state.images)
    owner.advance()
    assert owner.finished and not owner.report["submitted"]
    path = owner.output / ("before" if when == "before" else "post-submit") / "viewport.png"
    assert not path.exists()
    if failure in {"auth", "dialog", "navigation"}:
        assert len(setup.state.images) == before
    assert setup.state.native == ([] if when == "before" else ["readiness", "submit"])
    assert "PRIVATE" not in json.dumps(owner.report)


def test_even_fixture_project_submitted_text_is_not_a_real_course_receipt(setup):
    owner = to_stage(setup, "capturing_outcome")
    setup.state.after_text = "Project submitted"
    owner.advance()
    assert owner.phase == "unknown_pending" and not owner.report["submitted"]
    raw = json.loads((owner.output / "post-submit/visible-outer.json").read_bytes())
    assert raw["text"] == "Project submitted"
    assert raw["authority"] == "diagnostic_only_not_submission_acknowledgement"


def test_failed_post_submit_capture_preserves_one_safe_disposition_and_original_pending(setup, monkeypatch):
    owner = to_stage(setup, "capturing_outcome")
    journal = setup.case.journal.path.read_bytes()
    calls, original = [], module.preserve_submission_outcome

    def diagnostic(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "preserve_submission_outcome", diagnostic)
    setup.page.url = "https://private.invalid/PRIVATE"
    owner.advance()
    assert owner.failure == "project_submission_context_changed" and owner.phase == "unknown_pending"
    assert len(calls) == 1 and calls[0]["deadline"] == owner._started + 180
    result = owner.report["failure_outcome_diagnostic"]
    assert result["disposition"] == "navigation_outside_activity" and not result["content_saved"]
    path = owner.output / "post-submit-failure/disposition.json"
    assert (
        owner.report["source_sha256"][str(path.relative_to(owner.history))]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert setup.case.journal.path.read_bytes() == journal
    assert setup.case.journal.load().reduce().receipt("submission") is None
    assert "PRIVATE" not in json.dumps(owner.report)
    for _ in range(3):
        owner.advance()
        owner.tick()
        owner.resume()
        owner.close()
    assert len(calls) == 1 and setup.state.native == ["readiness", "submit"]


def test_fallback_failure_cannot_overwrite_original_cause_or_reservation(setup, monkeypatch):
    owner = to_stage(setup, "submitting")
    journal = setup.case.journal.path.read_bytes()
    calls = []

    def failed(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError("PRIVATE DIAGNOSTIC FAILURE")

    monkeypatch.setattr(module, "preserve_submission_outcome", failed)
    setup.state.dispatch_error = True
    owner.advance()
    assert owner.failure == "project_submission_operation_failed"
    assert owner.report["failure_outcome_diagnostic"]["disposition"] == "diagnostic_preservation_failed"
    assert owner.submit_may_have_occurred and not owner.submit_click_returned
    assert setup.case.journal.path.read_bytes() == journal
    owner.close()
    assert calls == [True] and setup.state.native == ["readiness", "submit"]
    assert "PRIVATE" not in json.dumps(owner.report)


def test_pre_submit_stop_never_enters_fallback(setup, monkeypatch):
    monkeypatch.setattr(
        module, "preserve_submission_outcome", lambda *_a, **_k: pytest.fail("No possible Submit yet")
    )
    owner = to_stage(setup, "submitting")
    owner.abort()
    assert not owner.submit_may_have_occurred
    assert "failure_outcome_diagnostic" not in owner.report
    assert not (owner.output / "post-submit-failure").exists()


def test_fallback_uses_original_deadline_even_if_live_limit_is_mutated(setup, monkeypatch):
    owner = to_stage(setup, "capturing_outcome")
    deadlines, original = [], module.preserve_submission_outcome

    def diagnostic(*args, **kwargs):
        deadlines.append(kwargs["deadline"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "preserve_submission_outcome", diagnostic)
    owner.max_seconds = 600
    setup.state.clock = 180
    owner.advance()
    assert deadlines == [180]
    assert owner.failure == "project_submission_limits_changed"
    assert owner.report["failure_outcome_diagnostic"]["disposition"] == "deadline_exhausted"
    assert setup.state.native == ["readiness", "submit"]


def test_native_dialog_type_survives_failure_without_reading_private_prompt(setup):
    owner = to_stage(setup, "capturing_outcome")

    class Dialog:
        type = "prompt"

        @property
        def message(self):
            pytest.fail("Never read a prompt message")

        @property
        def default_value(self):
            pytest.fail("Never read a prompt default")

        def accept(self):
            pytest.fail("Never accept a dialog")

        dismiss = accept

    owner._dialog_handler(Dialog())
    images = len(setup.state.images)
    owner.advance()
    result = owner.report["failure_outcome_diagnostic"]
    assert result["disposition"] == "native_dialog_content_withheld"
    assert result["native_dialogs"] == [
        {"type": "prompt", "message_disposition": "withheld_unverified_native_dialog_context"}
    ]
    assert owner.failure == "project_submission_context_changed"
    assert len(setup.state.images) == images
    assert setup.state.native == ["readiness", "submit"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_diagnostic_interrupt_still_persists_first_cause_and_closes_stream(setup, monkeypatch, interrupt):
    owner = to_stage(setup, "submitting")
    journal = setup.case.journal.path.read_bytes()
    calls = []

    def interrupted(*_args, **_kwargs):
        calls.append(True)
        raise interrupt()

    monkeypatch.setattr(module, "preserve_submission_outcome", interrupted)
    setup.state.dispatch_error = True
    with pytest.raises(interrupt):
        owner.advance()
    assert owner.failure == "project_submission_operation_failed" and owner.phase == "unknown_pending"
    assert owner.finished and owner.report["failure_reason"] == owner.failure
    assert owner.report["failure_outcome_diagnostic"]["disposition"] == "diagnostic_interrupted"
    assert json.loads((owner.output / "report.json").read_bytes()) == owner.report
    assert owner._stream.closed and not owner._finalizing and not owner._busy
    assert setup.case.journal.path.read_bytes() == journal
    for _ in range(2):
        owner.advance()
        owner.tick()
        owner.resume()
        owner.close()
    assert calls == [True] and setup.state.native == ["readiness", "submit"]


@pytest.mark.parametrize("budget", ["time", "advances", "pause_time"])
def test_fixed_budgets_never_extend_or_retry(setup, budget):
    owner = create(setup, max_seconds=30, max_advances=1 if budget == "advances" else 4)
    owner.advance()
    if budget != "advances":
        setup.state.clock = 30
    if budget == "pause_time":
        owner.pause()
        owner.resume()
    owner.advance()
    assert owner.phase == "unknown_pending" and owner.finished and not setup.state.native
    assert owner.max_seconds == 30


def test_pause_tick_and_single_step_do_not_dispatch_early(setup):
    owner = create(setup)
    owner.tick()
    assert setup.state.reads == 0
    owner.advance()
    assert owner.phase == "selecting_readiness" and owner.status == "paused"
    owner.tick()
    assert setup.state.native == []
    owner.resume()
    owner.tick()
    owner.pause()
    owner.tick()
    assert setup.state.native == ["readiness"]
    owner.advance()
    assert setup.state.native == ["readiness", "submit"]
    owner.close()


def test_concurrent_journal_append_between_check_and_reserve_is_rejected(setup, monkeypatch):
    owner = create(setup)
    original = module.fcntl.flock
    changed = []

    def flock(fd, operation):
        if operation == module.fcntl.LOCK_EX and not changed:
            changed.append(True)
            setup.case.journal.append(Active(stage="submission"))
        return original(fd, operation)

    monkeypatch.setattr(module.fcntl, "flock", flock)
    owner.advance()
    assert owner.finished and owner.failure == "project_submission_canonical_journal_changed"
    assert not setup.state.native and not owner.state()["pending_canonical_action"]
    assert owner.state()["reservation_intent"] is not None
    assert not setup.case.journal.load().reduce().pending
    # The immutable local claim still prevents alternate-output replay.
    assert owner.claim.exists()


@pytest.mark.parametrize("action", ["readiness", "submit"])
def test_mutation_after_dispatch_marker_is_still_guarded(setup, monkeypatch, action):
    owner = to_stage(setup, "selecting_readiness" if action == "readiness" else "submitting")
    original = module.persist_json

    def write(path, value):
        original(path, value)
        if path.name == action + "-dispatch-reserved.json":
            setup.state.submit.exposed = False

    monkeypatch.setattr(module, "persist_json", write)
    before = list(setup.state.native)
    owner.advance()
    assert owner.phase == "unknown_pending" and setup.state.native == before
    assert (owner.output / (action + "-dispatch-reserved.json")).exists()


def test_terminal_callback_failure_is_recorded_without_clearing_pending(setup):
    owner = to_stage(setup, "capturing_outcome")

    def fail(event):
        if event["event"] == "episode_summary":
            raise RuntimeError("PRIVATE callback")

    owner._callback = fail
    owner.advance()
    assert owner.report["event_forwarding_failed"] and owner.phase == "unknown_pending"
    events = [json.loads(line) for line in (owner.output / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "state" and events[-1]["payload"]["event_forwarding_failed"]
    assert len(setup.case.journal.load().reduce().pending) == 1
    assert "PRIVATE" not in json.dumps(owner.report)


def test_failed_report_write_is_terminal_and_does_not_retry(setup, monkeypatch):
    owner = to_stage(setup, "capturing_outcome")
    original = module.persist_json

    def failed(path, value):
        if path == owner.output / "report.json":
            raise OSError("PRIVATE disk failure")
        original(path, value)

    monkeypatch.setattr(module, "persist_json", failed)
    owner.advance()
    assert owner.finished and owner.artifact_write_failed
    assert owner.phase == "unknown_pending" and len(setup.case.journal.load().reduce().pending) == 1
    before = list(setup.state.native)
    owner.advance()
    assert setup.state.native == before


@pytest.mark.parametrize(
    "options",
    [
        {"max_seconds": True},
        {"max_seconds": float("nan")},
        {"max_seconds": 601},
        {"max_advances": True},
        {"max_advances": 5},
        {"max_advances": 0},
        {"emit": None},
    ],
)
def test_invalid_options_stop_before_output_or_page_access(setup, options):
    with pytest.raises(BrowserSafetyStop):
        create(setup, **options)
    assert setup.state.reads == 0 and not (setup.case.root / "submission").exists()


def test_stale_scoring_input_rejects_offline_before_artifacts(setup):
    (setup.case.root / "scoring/scope.json").write_text("{}")
    with pytest.raises(BrowserSafetyStop):
        create(setup)
    assert setup.state.reads == 0 and not (setup.case.root / "submission").exists()


def test_diagnostic_event_urls_are_never_derived_from_visible_text(setup):
    owner = to_stage(setup, "capturing_outcome")
    setup.state.after_text = "Rendered activity link: " + setup.page.url
    owner.advance()
    report = json.loads((owner.output / "post-submit/visible-outer.json").read_bytes())
    assert report["text"] == "Rendered activity link: " + module.public_url(setup.page.url)
    assert "Rendered activity link" not in (owner.output / "events.jsonl").read_text()
