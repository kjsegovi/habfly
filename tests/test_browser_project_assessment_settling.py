"""Offline raw-capture settling with injected visible transport, never Chromium.

The shared notice detector, spending ledger, at-most-once adapters and saved
capture hashes are real; page operations and clock are explicit fixture seams.
"""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_numeric import config
from test_browser_project_scoring_steps import Control, Page
from test_project_assessment import visible

import habfly.browser_assessment_actions as actions
import habfly.browser_project_assessment as stages
import habfly.browser_project_paginated_inventory_steps as pages
import habfly.browser_score_transfer as scores
from habfly.browser import BrowserSafetyStop
from habfly.project_assessment import ACKNOWLEDGEMENTS, AssessmentError


@pytest.fixture
def rig(tmp_path, monkeypatch):
    page = Page("ANALYZED DATA STAR")
    state = SimpleNamespace(
        now=100.0, notice_until=0.0, variant="paired", raw=[], waits=[], on_wait=None, unsafe=None
    )

    def report():
        value = Page.report(page)
        frame = value["frames"][0]
        prefix = visible(mode=page.mode, funding=page.funding, collected=30).split("OBSERVATIONS")[0]
        if page.ack:
            prefix += ACKNOWLEDGEMENTS[page.mode] + " OK\n"
        active = state.now < state.notice_until
        notice = "Data saved\n" if active else ""
        frame["text"] = (
            prefix
            + "OBSERVATIONS\nANALYZED DATA STAR\n"
            + notice
            + "VIEWING\n1-10 OF 30\nTOTAL COLLECTED\n30\nSave"
        )
        frame["accessibility"] = (
            "- text: fixture\n"
            + ("- text: Data saved\n" if active else "")
            + '- button [disabled]\n- button\n- text: viewing 1-10 of 30 total collected 30\n- button "Save"'
        )
        if active and state.variant == "unpaired":
            frame["accessibility"] = frame["accessibility"].replace("Data saved", "Different notice")
        if active and state.variant == "wrong_position":
            frame["text"] = frame["text"].replace("Data saved\n", "") + "\nData saved"
        return value

    def inspect(p, _c):
        if state.unsafe:
            raise BrowserSafetyStop(state.unsafe)
        value = p.report()
        state.raw.append(deepcopy(value))
        return value

    def wait(ms):
        state.waits.append(ms)
        state.now += ms / 1000
        if ms and state.on_wait:
            state.on_wait()

    page.report, page.wait_for_timeout = report, wait
    monkeypatch.setattr(actions.time, "monotonic", lambda: state.now)
    for module in (actions, stages, scores):
        monkeypatch.setattr(module, "inspect_page", inspect)
    monkeypatch.setattr(actions.AssessmentActuator, "button", lambda s, label: s.page.controls[label])

    def start(**kwargs):
        return stages.begin_assessment_stage(
            page,
            config(),
            tmp_path / "stage",
            history_root=tmp_path,
            data_revision=1,
            collected=30,
            reason="Explicit offline raw-list fixture",
            **kwargs,
        )

    return SimpleNamespace(page=page, state=state, root=tmp_path, start=start)


def raw(folder):
    return json.loads((folder / "observation.json").read_text())


def fade(rig, duration=0.2):
    rig.state.notice_until = rig.state.now + duration


def both(rig):
    actuator = rig.start(settle_inventory=True, deadline=rig.state.now + 20)
    for kind in ("data_quality", "scavenger_hunt"):
        rig.page.mode = kind
        actuator.assess(kind, data_revision=1)
        actuator.dismiss_receipt()
    return actuator


def transfer(rig, **kwargs):
    return scores.transfer_assessed_score(
        rig.page,
        config(),
        rig.root / "transfer",
        assessment_receipt=rig.root / "stage/attempt-001-confirmed.json",
        allow_score_transfer=True,
        **kwargs,
    )


def test_stage_fade_reads_raw_pair_and_never_normalizes_or_dispatches(rig):
    fade(rig)
    original = deepcopy(rig.page.report())
    assert pages.paired_footer_autosave(original)
    checks = []
    actuator = rig.start(
        settle_inventory=True,
        deadline=101,
        timeout_seconds=0.8,
        check_cancelled=lambda: checks.append(rig.state.now),
    )
    record = json.loads((rig.root / "stage/stage.json").read_text())
    final = rig.page.report()
    assert rig.state.raw[0] == original and "Data saved" in rig.state.raw[0]["frames"][0]["text"]
    assert record["new_project_rows_sha256"] == actions.rows_hash(final["frames"][0]["text"])
    assert record["new_project_rows_sha256"] != actions.rows_hash(original["frames"][0]["text"])
    assert actuator.settle_inventory and actuator.read_deadline == 101
    assert len(checks) >= 4 and not rig.page.clicks and max(rig.state.waits) <= 100


def test_charges_and_acknowledgements_settle_raw_captures_with_one_dispatch_each(rig):
    actuator = rig.start(settle_inventory=True, deadline=120)
    for kind in ("data_quality", "scavenger_hunt"):
        rig.page.mode = kind
        fade(rig)
        rig.page.preclick = lambda _label: fade(rig)
        callbacks = []
        actuator.assess(
            kind,
            data_revision=1,
            before_dispatch=lambda callbacks=callbacks: (callbacks.append("reserved"), fade(rig)),
            check_cancelled=lambda callbacks=callbacks: callbacks.append("checked"),
        )
        actuator.dismiss_receipt(before_dispatch=lambda: fade(rig))
        assert "reserved" in callbacks and callbacks.count("checked") > 4
    assert rig.page.clicks == ["ASSESS", "OK", "ASSESS", "OK"]
    for index in range(2):
        for name in (f"attempt-{index:03d}-before", f"attempt-{index:03d}-after", f"ack-{index:03d}-after"):
            saved = raw(rig.root / "stage" / name)
            assert saved in rig.state.raw
            assert "Data saved" not in saved["frames"][0]["text"]
    assert actuator.ledger.reserved == 200 and not actuator.ledger.pending


def test_transfer_settles_before_and_after_one_click_and_keeps_exact_raw_evidence(rig):
    both(rig)
    fade(rig)
    rig.page.preclick = lambda _label: fade(rig)
    checks = []
    receipt = transfer(
        rig,
        settle_inventory=True,
        deadline=rig.state.now + 3,
        before_dispatch=lambda: fade(rig),
        check_cancelled=lambda: checks.append(rig.state.now),
    )
    assert receipt["score_transfer_verified"] and not receipt["submitted"]
    assert rig.page.clicks.count("Update Score") == 1 and len(checks) > 4
    for name in ("before", "after"):
        captured = raw(rig.root / "transfer" / name)
        assert captured in rig.state.raw and "Data saved" not in captured["frames"][0]["text"]
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        transfer(rig, settle_inventory=True)


@pytest.mark.parametrize(
    "stage",
    [
        "setup",
        "charge_before",
        "charge_after",
        "ack_before",
        "ack_after",
        "transfer_before",
        "transfer_after",
    ],
)
def test_persistent_notice_uses_absolute_budget_and_never_retries(rig, stage):
    actuator = None
    if stage.startswith("transfer"):
        both(rig)
    elif stage != "setup":
        actuator = rig.start(settle_inventory=True, deadline=120)
        if stage.startswith("ack"):
            actuator.assess("data_quality", data_revision=1)
        actuator.timeout_seconds = 0.25
    baseline = len(rig.page.clicks)
    start = rig.state.now

    def forever(*_):
        rig.state.notice_until = float("inf")

    if stage.endswith("after"):
        rig.page.preclick = forever
    else:
        forever()
    with pytest.raises((AssessmentError, BrowserSafetyStop), match="deadline|did_not_settle"):
        if stage == "setup":
            rig.start(settle_inventory=True, deadline=start + 0.25)
        elif stage.startswith("charge"):
            actuator.assess("data_quality", data_revision=1)
        elif stage.startswith("ack"):
            actuator.dismiss_receipt()
        else:
            transfer(rig, settle_inventory=True, deadline=start + 0.25)
    assert rig.state.now - start <= 0.251
    assert len(rig.page.clicks) - baseline == int(stage.endswith("after"))
    if stage == "charge_after":
        assert actuator.ledger.pending
        with pytest.raises(AssessmentError, match="stopped"):
            actuator.assess("data_quality", data_revision=1)
    if stage == "transfer_after":
        assert json.loads((rig.root / "transfer/stopped.json").read_text())["write_may_have_occurred"]
        assert not (rig.root / "transfer/confirmed.json").exists()
        with pytest.raises(BrowserSafetyStop, match="already_reserved"):
            transfer(rig, settle_inventory=True)


def test_three_second_subbudget_is_not_extended_when_operation_budget_is_longer(rig):
    rig.state.notice_until = float("inf")
    with pytest.raises(BrowserSafetyStop, match="autosave_notice_did_not_settle"):
        rig.start(settle_inventory=True, deadline=130, timeout_seconds=10)
    assert rig.state.now == pytest.approx(103, abs=0.001)
    assert not (rig.root / "stage").exists() and not rig.page.clicks


@pytest.mark.parametrize("operation", ["charge", "ack", "transfer"])
def test_cancellation_during_reserved_wait_keeps_claim_and_prevents_dispatch(rig, operation):
    if operation == "transfer":
        both(rig)
    else:
        actuator = rig.start(settle_inventory=True)
        if operation == "ack":
            actuator.assess("data_quality", data_revision=1)
    baseline = list(rig.page.clicks)
    cancelled = [False]
    rig.state.on_wait = lambda: cancelled.__setitem__(0, True)

    def check():
        if cancelled[0]:
            raise BrowserSafetyStop("fixture_source_or_cancellation_changed")

    with pytest.raises(BrowserSafetyStop, match="fixture_source_or_cancellation_changed"):
        if operation == "charge":
            actuator.assess(
                "data_quality", data_revision=1, before_dispatch=lambda: fade(rig), check_cancelled=check
            )
        elif operation == "ack":
            actuator.dismiss_receipt(before_dispatch=lambda: fade(rig), check_cancelled=check)
        else:
            transfer(rig, settle_inventory=True, before_dispatch=lambda: fade(rig), check_cancelled=check)
    assert rig.page.clicks == baseline
    prefix = {"charge": "stage/attempt-000", "ack": "stage/ack-000", "transfer": "transfer/"}[operation]
    separator = "" if operation == "transfer" else "-"
    assert (rig.root / (prefix + separator + "reserved.json")).is_file()
    stopped = json.loads((rig.root / (prefix + separator + "stopped.json")).read_text())
    assert not stopped.get("click_dispatch_started", stopped.get("write_may_have_occurred"))
    assert not (rig.root / (prefix + separator + "confirmed.json")).exists()


@pytest.mark.parametrize("operation", ["charge", "ack", "transfer"])
def test_replaced_control_during_settling_is_not_rebound_for_dispatch(rig, monkeypatch, operation):
    if operation == "transfer":
        both(rig)
    else:
        actuator = rig.start(settle_inventory=True)
        if operation == "ack":
            actuator.assess("data_quality", data_revision=1)
            # A locator resolves its current native handle on each binding.
            monkeypatch.setattr(
                actions.AssessmentActuator,
                "button",
                lambda s, label: SimpleNamespace(element_handle=lambda **_: s.page.controls[label]),
            )
    baseline = list(rig.page.clicks)
    label = {"charge": "ASSESS", "ack": "OK", "transfer": "Update Score"}[operation]
    rig.state.on_wait = lambda: rig.page.controls.__setitem__(label, Control(rig.page, label))
    with pytest.raises(
        (AssessmentError, BrowserSafetyStop),
        match="changed_after_reservation|changed_before_dismiss_dispatch",
    ):
        if operation == "charge":
            actuator.assess("data_quality", data_revision=1, before_dispatch=lambda: fade(rig))
        elif operation == "ack":
            actuator.dismiss_receipt(before_dispatch=lambda: fade(rig))
        else:
            transfer(rig, settle_inventory=True, before_dispatch=lambda: fade(rig))
    assert rig.page.clicks == baseline


def test_repeated_reads_share_predispatch_deadline_not_fresh_timeout(rig):
    actuator = rig.start(settle_inventory=True)
    actuator.timeout_seconds = 0.3
    fade(rig)
    with pytest.raises(AssessmentError, match="deadline"):
        actuator.assess("data_quality", data_revision=1, before_dispatch=lambda: fade(rig))
    assert rig.state.now == pytest.approx(100.3, abs=0.001)
    assert actuator.ledger.pending and not rig.page.clicks


def test_postdispatch_capture_uses_remaining_receipt_deadline(rig):
    both(rig)
    original = rig.page.get_by_text
    polls = [0]

    def delayed(*a, **k):
        # Spend most of the receipt timeout on the real acknowledgement wait.
        if rig.page.notice:
            polls[0] += 1
            if polls[0] < 5:
                return SimpleNamespace(all=list)
        return original(*a, **k)

    rig.page.get_by_text = delayed
    rig.page.preclick = lambda _: fade(rig, 0.35)
    start = rig.state.now
    with pytest.raises(AssessmentError, match="deadline"):
        transfer(rig, settle_inventory=True, timeout_seconds=0.3)
    assert rig.state.now - start <= 0.301
    assert rig.page.clicks.count("Update Score") == 1
    assert not (rig.root / "transfer/confirmed.json").exists()


def test_changed_rows_after_notice_fade_still_leave_charge_unconfirmed(rig):
    actuator = rig.start(settle_inventory=True)
    original = rig.page.report

    def changed_report():
        value = original()
        if rig.page.clicks:
            value["frames"][0]["text"] = value["frames"][0]["text"].replace(
                "ANALYZED DATA STAR", "ANALYZED DATA STAR CHANGED"
            )
        return value

    rig.page.report = changed_report
    rig.page.preclick = lambda _: fade(rig)
    with pytest.raises(AssessmentError, match="project_rows_changed"):
        actuator.assess("data_quality", data_revision=1)
    assert rig.page.clicks == ["ASSESS"] and actuator.ledger.pending
    saved = raw(rig.root / "stage/attempt-000-changed-rows")
    assert "CHANGED" in saved["frames"][0]["text"] and "Data saved" not in saved["frames"][0]["text"]
    assert saved in rig.state.raw and not (rig.root / "stage/attempt-000-confirmed.json").exists()


@pytest.mark.parametrize("fault", ["cancel", "popup", "frame", "navigation", "source", "modal", "auth"])
def test_each_wait_rechecks_cancellation_sources_and_context_before_any_write(rig, monkeypatch, fault):
    fade(rig)
    bad = [False]

    def check():
        if bad[0] and fault in {"cancel", "source"}:
            raise BrowserSafetyStop("fixture_" + fault)

    def changed():
        bad[0] = True
        if fault == "popup":
            rig.page.context.pages.append(object())
        elif fault == "frame":
            rig.page.frames = rig.page.frames[:1]
        elif fault == "navigation":
            rig.page.url = "http://unapproved.invalid/"
        elif fault in {"modal", "auth"}:
            rig.state.unsafe = "unexpected_modal" if fault == "modal" else "authentication_required"

    rig.state.on_wait = changed
    with pytest.raises(BrowserSafetyStop):
        rig.start(settle_inventory=True, deadline=120, check_cancelled=check)
    assert not rig.page.clicks and not (rig.root / "stage").exists()


@pytest.mark.parametrize("variant", ["unpaired", "wrong_position"])
def test_other_notices_are_not_waited_for_or_stripped(rig, variant):
    fade(rig)
    rig.state.variant = variant
    actuator = rig.start(settle_inventory=True)
    report, _, hashed = actuator.read()
    assert "Data saved" in report["frames"][0]["text"] and not rig.state.waits
    assert hashed == actions.rows_hash(report["frames"][0]["text"])


@pytest.mark.parametrize("operation", ["setup", "assessment", "transfer"])
def test_legacy_default_never_calls_settling_helper(rig, monkeypatch, operation):
    if operation == "transfer":
        both(rig)
    monkeypatch.setattr(pages, "read_settled_inventory", lambda *_a, **_k: pytest.fail("legacy settling"))
    if operation == "setup":
        fade(rig)
        actuator = rig.start()
        assert not actuator.settle_inventory and not rig.state.waits
    elif operation == "assessment":
        actuator = rig.start()
        actuator.assess("data_quality", data_revision=1)
        actuator.dismiss_receipt()
    else:
        assert transfer(rig)["score_transfer_verified"]


@pytest.mark.parametrize(
    "options",
    [
        {"settle_inventory": 1},
        {"settle_inventory": "yes"},
        {"deadline": 120},
        {"settle_inventory": True, "deadline": float("nan")},
        {"settle_inventory": True, "deadline": True},
    ],
)
def test_invalid_optin_rejected_before_page_or_artifacts(tmp_path, options):
    with pytest.raises(ValueError):
        actions.AssessmentActuator(None, None, tmp_path / "actuator", **options)
    with pytest.raises(ValueError):
        stages.begin_assessment_stage(
            None,
            None,
            tmp_path / "stage",
            history_root=tmp_path,
            data_revision=1,
            collected=30,
            reason="fixture",
            **options,
        )
    with pytest.raises(ValueError):
        scores.transfer_assessed_score(
            None, None, tmp_path / "score", assessment_receipt=None, allow_score_transfer=True, **options
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("options", [{"timeout_seconds": 5}, {"check_cancelled": lambda: None}])
def test_stage_never_silently_ignores_new_options_without_optin(tmp_path, options):
    with pytest.raises(ValueError, match="require explicit inventory settling"):
        stages.begin_assessment_stage(
            None,
            None,
            tmp_path / "stage",
            history_root=tmp_path,
            data_revision=1,
            collected=30,
            reason="fixture",
            **options,
        )
    assert not list(tmp_path.iterdir())
