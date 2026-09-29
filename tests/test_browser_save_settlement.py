"""Pure callbacks and real immutable diagnostic validation; no browser launch."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import habfly.browser_save_settlement as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_probe import save_probe


@pytest.fixture
def rig(tmp_path, monkeypatch):
    state = SimpleNamespace(now=100.0, reads=0, samples=[True, False, False], changed=False, cancelled=False)
    directory = tmp_path / "save"
    directory.mkdir()
    claim, intent = tmp_path / "claim.json", module.intent_options(True, 2)
    for path in (claim, directory / "reserved.json"):
        persist_json(path, intent)
    monkeypatch.setattr(module.time, "monotonic", lambda: state.now)

    def check():
        if state.cancelled or state.changed:
            raise BrowserSafetyStop("cancelled_or_changed")

    state.report = {
        "schema_version": 1,
        "mode": "read_only_browser_preflight",
        "frames": [],
        "ignored_frame_urls": [],
        "value": 42,
    }

    def current():
        state.now += 0.1
        state.reads += 1
        return dict(state.report)

    def notice():
        return state.samples.pop(0) if len(state.samples) > 1 else state.samples[0]

    state.phase = module.ReservedSavePhase(
        directory,
        seconds=2,
        claims={claim: intent, directory / "reserved.json": intent},
        check=check,
        quick_check=check,
    )
    state.callbacks = {
        "current": current,
        "button": lambda: SimpleNamespace(is_enabled=lambda: True),
        "notice": notice,
        "wait": lambda ms: setattr(state, "now", state.now + ms / 1000),
    }
    state.root, state.directory, state.claim, state.intent = tmp_path, directory, claim, intent
    return state


def complete(rig):
    before = rig.phase.settle(**rig.callbacks)
    save_probe(before, rig.directory / "pre-dispatch-observation")
    persist_json(rig.directory / "dispatch.json", {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1})
    rig.samples[:] = [True]
    after = rig.phase.acknowledge(**rig.callbacks)
    save_probe(after, rig.directory / "after")
    rig.phase.finish(after)
    return before, after


def validate(rig, before, after):
    book = _Evidence(rig.root)
    module.validate_reserved_phase(
        book,
        rig.directory,
        rig.intent,
        predispatched=before,
        after=after,
        same_view=lambda r: r["value"] == 42,
    )
    return book


def test_complete_sources_and_tree_are_pinned(rig):
    before, after = complete(rig)
    book = validate(rig, before, after)
    assert rig.reads == 3
    assert "save/reserved-settlement/capture-001/observation.json" in book.hashes
    assert "save/reserved-phase-confirmed.json" in book.hashes
    (rig.directory / "reserved-settlement/added.json").write_text("{}")
    with pytest.raises(BrowserSafetyStop, match="tree_changed"):
        book.unchanged()


def test_reappearance_requires_another_guard(rig):
    rig.samples[:] = [True, False, True, False, False]
    before, after = complete(rig)
    validate(rig, before, after)
    assert rig.reads == 4


@pytest.mark.parametrize("failure", ["cancel", "source", "reservation", "deadline"])
def test_wait_never_dispatches_on_changes(rig, failure):
    original = rig.callbacks["wait"]

    def wait(ms):
        original(ms)
        if failure == "cancel":
            rig.cancelled = True
        if failure == "source":
            rig.changed = True
        if failure == "reservation":
            rig.claim.write_text("{}")
        if failure == "deadline":
            rig.now = 102

    rig.callbacks["wait"] = wait
    with pytest.raises(BrowserSafetyStop):
        rig.phase.settle(**rig.callbacks)
    assert not (rig.directory / "reserved-notice-settled.json").exists()
    assert not (rig.directory / "dispatch.json").exists()


@pytest.mark.parametrize("where", ["full_guard", "notice", "acknowledgement", "final_readback"])
def test_shared_deadline_does_not_accept_late_result(rig, where):
    if where in {"full_guard", "notice"}:
        key = "current" if where == "full_guard" else "notice"
        original = rig.callbacks[key]

        def late():
            result = original()
            rig.now = 102
            return result

        rig.callbacks[key] = late
        with pytest.raises(BrowserSafetyStop, match="phase_timeout"):
            rig.phase.settle(**rig.callbacks)
    else:
        rig.phase.settle(**rig.callbacks)
        rig.samples[:] = [True]
        key = "notice" if where == "acknowledgement" else "current"
        original = rig.callbacks[key]

        def late():
            result = original()
            rig.now = 102
            return result

        rig.callbacks[key] = late
        with pytest.raises(BrowserSafetyStop, match="phase_timeout"):
            rig.phase.acknowledge(**rig.callbacks)
    assert not (rig.directory / "reserved-phase-confirmed.json").exists()


@pytest.mark.parametrize(
    "mutation", ["flag", "complete_flag", "budget", "probe", "capture", "extra", "predispatched"]
)
def test_validator_rejects_changed_or_misrepresented_diagnostics(rig, mutation):
    before, after = complete(rig)
    if mutation in {"flag", "budget", "complete_flag"}:
        path = rig.directory / (
            "reserved-phase-confirmed.json" if mutation == "complete_flag" else "reserved-notice-settled.json"
        )
        value = json.loads(path.read_text())
        if mutation == "flag":
            value["notice_present"] = 0
        if mutation == "complete_flag":
            value["shared_deadline_verified"] = 1
        if mutation == "budget":
            value["deadline_monotonic_seconds"] = 104
        path.write_text(json.dumps(value))
    elif mutation == "probe":
        (rig.directory / "reserved-settlement/probe-000.json").write_text("{}")
    elif mutation == "capture":
        (rig.directory / "reserved-settlement/capture-000/observation.json").write_text("{}")
    elif mutation == "extra":
        (rig.directory / "reserved-settlement/unlisted.json").write_text("{}")
    else:
        before = {**before, "extra_non_scientific_paint": True}
    with pytest.raises(BrowserSafetyStop):
        validate(rig, before, after)


def test_legacy_optout_does_not_adopt_new_artifacts(rig):
    book = _Evidence(rig.root)
    module.validate_reserved_phase(book, rig.root, {}, after=None, same_view=lambda _: False)
    assert not book.hashes
    complete(rig)
    with pytest.raises(BrowserSafetyStop, match="undeclared_policy"):
        module.validate_reserved_phase(book, rig.directory, {}, after=None, same_view=lambda _: False)


@pytest.mark.parametrize("failure", ["between_reads", "bool_alias"])
def test_claim_validation_and_pin_use_same_typed_bytes(tmp_path, monkeypatch, failure):
    claim = tmp_path / "claim.json"
    intent = {"settle_reserved_notice": True}
    persist_json(claim, intent if failure == "between_reads" else {"settle_reserved_notice": 1})
    original = Path.read_bytes
    changed = False

    def read(path):
        nonlocal changed
        result = original(path)
        if path == claim and not changed and failure == "between_reads":
            changed = True
            claim.write_text("{}")
        return result

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(BrowserSafetyStop, match="reservation_changed"):
        phase = module.ReservedSavePhase(
            tmp_path, seconds=2, claims={claim: intent}, check=lambda: None, quick_check=lambda: None
        )
        phase.check()
