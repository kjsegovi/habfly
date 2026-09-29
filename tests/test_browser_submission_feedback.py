"""Exact negative feedback tests; fixtures cannot establish a positive receipt."""
# ruff: noqa: F811

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_project_scoring_steps import case  # noqa: F401
from test_browser_project_submission_preflight import subject  # noqa: F401
from test_browser_project_submission_steps import create, setup, to_stage  # noqa: F401

import habfly.browser_submission_feedback as module
from habfly.browser import BrowserSafetyStop


def snapshots():
    before = {
        "text": "Project\n" + module.READY,
        "accessibility": '- main:\n  - group "Select all that apply":\n    - text: Select all that apply\n'
        f'    - checkbox "{module.READY}" [checked]\n    - paragraph: {module.READY}',
    }
    after = {
        "text": before["text"] + "\nOK\n\n" + module.REFUSAL,
        "accessibility": before["accessibility"] + "\n" + "\n".join(module._BLOCK),
    }
    return before, after


def exposure():
    box = {"x": 10, "y": 10, "width": 40, "height": 20}
    return {
        "paragraph": dict(box),
        "buttons": [{"accessibility": ax, "box": dict(box)} for ax in module._BUTTON_AX],
        "same_outer_snapshot_verified": True,
    }


def classify(before=None, after=None, **kwargs):
    old, new = snapshots()
    return module.classify_submission_feedback(
        before if before is not None else old,
        after if after is not None else new,
        **({"submit_dispatched": True, "exposure": exposure()} | kwargs),
    )


def test_exact_refusal_is_diagnostic_only_never_success_or_retry():
    before, after = snapshots()
    result = classify(before, after)
    assert result["status"] == "course_refusal"
    assert result["reason"] == "at_least_thirty_analyzed_submitted_stars_required"
    assert result["refusal_visible"] is True and result["visible_text"] == module.REFUSAL
    assert all(
        result[k] is False
        for k in (
            "submitted",
            "submission_verified",
            "task_completed",
            "project_completed",
            "canonical_receipt",
            "automatic_retry",
            "positive_acknowledgement_recognized",
        )
    )
    assert (
        result["before_outer_sha256"]
        == hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    )
    assert (
        result["after_outer_sha256"] == hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "naked_string",
        "instructions_quote",
        "preexisting",
        "no_ok",
        "no_close",
        "extra_close",
        "wrong_order",
        "unchecked",
        "disabled_ok",
        "nested_feedback",
        "duplicate_refusal",
        "wrong_case",
        "positive",
        "unknown_dialog",
        "unverified_snapshot",
        "missing_main",
        "conflicting_outcome",
    ],
)
def test_old_text_naked_substrings_and_unsupported_structures_never_recognized(mutation):
    before, after = snapshots()
    if mutation == "naked_string":
        after["accessibility"] = "- text: " + module.REFUSAL
    elif mutation == "instructions_quote":
        after["text"] = "Instructions say: " + module.REFUSAL
    elif mutation == "preexisting":
        before = deepcopy(after)
    elif mutation == "no_ok":
        after["accessibility"] = after["accessibility"].replace(module._BLOCK[0], "")
    elif mutation == "no_close":
        after["accessibility"] = after["accessibility"].replace(module._BLOCK[3], "")
    elif mutation == "extra_close":
        after["accessibility"] += '\n  - button "Close feedback"'
    elif mutation == "wrong_order":
        after["accessibility"] = before["accessibility"] + "\n" + "\n".join(reversed(module._BLOCK))
    elif mutation == "unchecked":
        after["accessibility"] = after["accessibility"].replace(" [checked]", "")
    elif mutation == "disabled_ok":
        after["accessibility"] = after["accessibility"].replace(
            module._BLOCK[0], module._BLOCK[0] + " [disabled]"
        )
    elif mutation == "nested_feedback":
        after["accessibility"] = after["accessibility"].replace(
            "\n".join(module._BLOCK), "\n".join("  " + line for line in module._BLOCK)
        )
    elif mutation == "duplicate_refusal":
        after["text"] += "\n" + module.REFUSAL
    elif mutation == "wrong_case":
        after = {k: v.replace(module.REFUSAL, module.REFUSAL.upper()) for k, v in after.items()}
    elif mutation == "positive":
        after = {k: v.replace(module.REFUSAL, "Project successfully submitted.") for k, v in after.items()}
    elif mutation == "unknown_dialog":
        after["accessibility"] = after["accessibility"].replace("- main:", '- dialog "Outcome":')
    elif mutation == "unverified_snapshot":
        after["consistency_unverified"] = True
    elif mutation == "missing_main":
        before["accessibility"] = before["accessibility"].replace("- main:\n", "")
    else:
        after["accessibility"] += "\n  - paragraph: Project successfully submitted."
    assert classify(before, after)["status"] == "unrecognized"


@pytest.mark.parametrize("dispatch", [False, None, 1, "true"])
def test_only_actual_boolean_post_dispatch_boundary(dispatch):
    assert classify(submit_dispatched=dispatch)["status"] == "unrecognized"


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "unchecked_proof",
        "missing_button",
        "wrong_button",
        "occluded",
        "negative_box",
        "nonfinite_box",
        "bool_box",
    ],
)
def test_exposure_and_same_snapshot_are_required(change):
    proof = exposure()
    if change == "none":
        proof = None
    elif change == "unchecked_proof":
        proof["same_outer_snapshot_verified"] = 1
    elif change == "missing_button":
        proof["buttons"].pop()
    elif change == "wrong_button":
        proof["buttons"][0]["accessibility"] = '- button "Submit Project"'
    elif change == "occluded":
        proof["paragraph"] = None
    elif change == "negative_box":
        proof["buttons"][0]["box"]["x"] = -1
    elif change == "nonfinite_box":
        proof["paragraph"]["width"] = float("nan")
    else:
        proof["paragraph"]["width"] = True
    assert classify(exposure=proof)["status"] == "unrecognized"


def test_actual_disposable_capture_grounding_does_not_become_canonical_proof():
    root = Path(__file__).resolve().parents[1] / "experiments/browser-submission-shape-006"
    if not root.is_dir():
        pytest.skip("Disposable public diagnostic is local, not a shipped course fixture")
    captures = [json.loads((root / f"capture-{i:02d}/visible-outcome.json").read_bytes()) for i in (1, 2)]
    before, after = [{k: capture["frames"][0][k] for k in ("text", "accessibility")} for capture in captures]
    assert all(capture["consistency_unverified"] is True for capture in captures)
    assert module.submission_refusal_candidate(before, after) is True
    # Public capture grounds the structure, but it supplies no guarded exposure
    # proof or canonical 30-star submission acknowledgement.
    assert classify(before, after, exposure=None)["status"] == "unrecognized"


@pytest.fixture
def saved_production_refusal():
    """Optional immutable public evidence, not a dependency of shipped fixtures."""
    root = Path(__file__).resolve().parents[1] / "experiments/browser-submission-production-004"
    pins = {
        "ready/visible-outer.json": "015643f0f665592e991393106de35a26ad8445566e2e1e405d8329d658325b8b",
        "after/visible-outer.json": "dd906576e5659c544b60dd1b42e227f7281f9a02dd64fc702612a903f5ff908a",
        "submit-returned.json": "1fe0ef179d15519f2380811f990a739f5aaa32b698d2ce0451526f8b98007ae2",
        "submission-feedback.json": "e5161b352370e0c177d1f0bb9feb607f49b665bcbba36c3130c34525707b1d00",
        "report.json": "7c911506ebf53b824b6ee168c01f7e1cbffa2a7448e6bbb8f3afa3f7000d9486",
    }
    if not all((root / name).is_file() for name in pins):
        pytest.skip("Optional local public refusal capture is unavailable")
    records = {}
    for name, digest in pins.items():
        raw = (root / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, name
        records[name] = json.loads(raw)
    yield records
    assert all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in pins.items()
    )


def test_saved_guarded_production_refusal_rebuilds_exact_diagnostic_only(saved_production_refusal):
    records = saved_production_refusal
    feedback = records["submission-feedback.json"]
    result = module.classify_submission_feedback(
        records["ready/visible-outer.json"],
        records["after/visible-outer.json"],
        submit_dispatched=records["submit-returned.json"]["click_returned"],
        exposure=feedback["exposure"],
    )
    assert result == feedback
    assert result["status"] == "course_refusal"
    report = records["report.json"]
    assert report["canonical_analyzed_count_verified"] is False
    assert report["thirty_star_run"] is False
    assert report["submission_feedback"] == result
    # A genuine negative UI result does not supply the separate prerequisites
    # or authoritatively successful response required by the canonical owner.
    assert all(
        result[key] is False
        for key in (
            "submission_verified",
            "submitted",
            "task_completed",
            "project_completed",
            "canonical_receipt",
            "automatic_retry",
            "positive_acknowledgement_recognized",
        )
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "no_exposure",
        "unverified_capture",
        "preexisting",
        "no_dispatch",
        "positive_wording",
        "score_acknowledgement",
    ],
)
def test_saved_refusal_cannot_be_promoted_or_reused_as_success(saved_production_refusal, mutation):
    records = saved_production_refusal
    before, after = [deepcopy(records[f"{name}/visible-outer.json"]) for name in ("ready", "after")]
    proof = deepcopy(records["submission-feedback.json"]["exposure"])
    dispatched = True
    if mutation == "no_exposure":
        proof = None
    elif mutation == "unverified_capture":
        after["consistency_unverified"] = True
    elif mutation == "preexisting":
        before = deepcopy(after)
    elif mutation == "no_dispatch":
        dispatched = False
    else:
        text = "Project successfully submitted." if mutation == "positive_wording" else "Score updated."
        after = {key: value.replace(module.REFUSAL, text) for key, value in after.items()}
    result = module.classify_submission_feedback(before, after, submit_dispatched=dispatched, exposure=proof)
    assert result["status"] == "unrecognized"
    assert all(
        result[key] is False
        for key in (
            "submission_verified",
            "submitted",
            "task_completed",
            "project_completed",
            "canonical_receipt",
            "automatic_retry",
            "positive_acknowledgement_recognized",
        )
    )


@pytest.fixture
def native_surface():
    before, after = snapshots()
    state = SimpleNamespace(guards=0, replaced=False, changed=False, blocked=False, cancel=False)

    class Control:
        def __init__(self, ax):
            self.ax, self.enabled, self.covered = ax, True, False

        def count(self):
            return 1

        def filter(self, *, has_text):
            assert has_text.fullmatch(module.REFUSAL)
            return self

        def is_enabled(self, **_):
            return self.enabled

        def aria_snapshot(self, **_):
            return self.ax

        def element_handle(self, **_):
            return self

        def evaluate(self, script, argument=None):
            if script.startswith("(a,b)"):
                return argument is self and not state.replaced
            if self.covered:
                return None
            box = {"x": 1, "y": 1, "width": 40, "height": 10}
            if self.ax.startswith("- button"):
                from habfly.browser_submission_controls import SUBMIT_EXPOSED

                assert script == SUBMIT_EXPOSED
                box["hit_inset"] = 2
            return box

    paragraph = Control("- paragraph: " + module.REFUSAL)
    buttons = [Control(ax) for ax in module._BUTTON_AX]
    close = SimpleNamespace(count=lambda: 2, all=lambda: buttons[1:])
    page = SimpleNamespace(
        get_by_role=lambda role, name=None, exact=None: (
            paragraph if role == "paragraph" else buttons[0] if name == "OK" else close
        ),
        locator=lambda selector: SimpleNamespace(
            inner_text=lambda **_: after["text"] + ("changed" if state.changed else ""),
            aria_snapshot=lambda **_: after["accessibility"],
        ),
    )

    def guard():
        state.guards += 1
        if state.cancel:
            raise BrowserSafetyStop("operator_aborted")

    return SimpleNamespace(
        before=before, after=after, state=state, page=page, paragraph=paragraph, buttons=buttons, guard=guard
    )


@pytest.mark.parametrize(
    "change", [None, "covered_text", "covered_button", "replaced", "changed", "disabled"]
)
def test_readonly_native_binding_exposure_and_stability(native_surface, change):
    item = native_surface
    if change == "covered_text":
        item.paragraph.covered = True
    elif change == "covered_button":
        item.buttons[1].covered = True
    elif change == "disabled":
        item.buttons[0].enabled = False
    elif change:
        setattr(item.state, change, True)
    proof = module.read_submission_feedback_exposure(
        item.page, item.after, guard=item.guard, timeout=lambda: 1000
    )
    result = classify(item.before, item.after, exposure=proof)
    assert (result["status"] == "course_refusal") is (change is None)
    assert item.state.guards >= 1


def test_exposure_cancel_propagates_without_actions(native_surface):
    native_surface.state.cancel = True
    with pytest.raises(BrowserSafetyStop, match="operator_aborted"):
        module.read_submission_feedback_exposure(
            native_surface.page, native_surface.after, guard=native_surface.guard, timeout=lambda: 1000
        )


def test_late_overlay_after_snapshot_read_rejects(native_surface):
    item = native_surface
    body = item.page.locator("body")
    old = body.inner_text

    def late(**kwargs):
        item.buttons[0].covered = True
        return old(**kwargs)

    body.inner_text = late
    item.page.locator = lambda _: body
    proof = module.read_submission_feedback_exposure(
        item.page, item.after, guard=item.guard, timeout=lambda: 1000
    )
    assert proof is None


def test_injected_fourth_stage_recognizes_refusal_but_keeps_exact_pending_journal(setup, monkeypatch):
    before, after = snapshots()
    setup.state.body.inner_text = lambda **_: before["text"]
    setup.state.body.aria_snapshot = lambda **_: before["accessibility"]
    owner = to_stage(setup, "capturing_outcome")
    setup.state.body.inner_text = lambda **_: after["text"]
    setup.state.body.aria_snapshot = lambda **_: after["accessibility"]
    # Declared native-reader seam only: pure recognition and real durable owner
    # reservation/pins/report paths run unchanged. No native browser is claimed.
    monkeypatch.setattr(module, "read_submission_feedback_exposure", lambda *_a, **_k: exposure())
    journal = setup.case.journal.path.read_bytes()
    owner.advance()
    assert owner.advances == 4 and owner.phase == "unknown_pending" and owner.finished
    assert owner.report["submission_outcome"] == "unknown"
    assert owner.report["submission_feedback"]["status"] == "course_refusal"
    assert setup.state.native == ["readiness", "submit"]
    assert setup.case.journal.path.read_bytes() == journal
    assert setup.case.journal.load().reduce().receipt("submission") is None
    assert all(
        owner.report[k] is False
        for k in ("submitted", "submission_verified", "task_completed", "project_completed")
    )
    record = json.loads((owner.output / "submission-feedback.json").read_bytes())
    assert record == owner.report["submission_feedback"]
    for name, digest in record["source_sha256"].items():
        assert hashlib.sha256((owner.history / name).read_bytes()).hexdigest() == digest
    owner.advance()
    assert setup.state.native == ["readiness", "submit"]


def test_async_refusal_wait_is_bounded_inside_same_fourth_advance(setup, monkeypatch):
    before, after = snapshots()
    setup.state.body.inner_text = lambda **_: before["text"]
    setup.state.body.aria_snapshot = lambda **_: before["accessibility"]
    owner = to_stage(setup, "capturing_outcome")
    wait = setup.page.wait_for_timeout

    def advance_time(milliseconds):
        wait(milliseconds)
        setup.state.clock += milliseconds / 1000

    setup.page.wait_for_timeout = advance_time
    setup.state.body.inner_text = lambda **_: (after if setup.state.clock >= 0.5 else before)["text"]
    setup.state.body.aria_snapshot = lambda **_: (after if setup.state.clock >= 0.5 else before)[
        "accessibility"
    ]
    monkeypatch.setattr(module, "read_submission_feedback_exposure", lambda *_a, **_k: exposure())
    owner.advance()
    assert owner.advances == 4 and owner.report["submission_feedback"]["read_polls"] == 3
    assert owner.report["submission_feedback"]["status"] == "course_refusal"
    assert setup.state.clock == 0.5 and setup.state.native == ["readiness", "submit"]


def test_poll_deadline_and_cancellation_never_extend_or_retry(setup):
    owner = to_stage(setup, "capturing_outcome")
    setup.state.clock = 179.9
    wait = setup.page.wait_for_timeout

    def advance_time(milliseconds):
        wait(milliseconds)
        setup.state.clock += milliseconds / 1000

    setup.page.wait_for_timeout = advance_time
    owner.advance()
    assert owner.failure == "project_submission_time_limit" and owner.phase == "unknown_pending"
    assert setup.state.clock == 180
    assert owner.advances == 4 and setup.state.native == ["readiness", "submit"]
    assert not (owner.output / "submission-feedback.json").exists()
