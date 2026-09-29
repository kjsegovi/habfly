"""Bounded new-intent footer settling; injected clocks plus intercepted pages."""
# ruff: noqa: F811

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_no_planet_save import no_save_page, save  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401

import habfly.browser_no_planet_save as module
from habfly.browser import BrowserSafetyStop


@pytest.fixture
def clocked(tmp_path, monkeypatch):
    state = SimpleNamespace(
        clock=100.0, present=[True, False, False], reads=0, probes=[], remembered=[], hook=None
    )
    button = SimpleNamespace(is_enabled=lambda: True)
    mapping = {
        "observation": {
            "values": {
                "has_planet": "No",
                "browser_field_map": {name: {"current_value": ""} for name in module.BASE_FIELDS},
            }
        }
    }

    def current():
        state.reads += 1
        state.clock += 2
        return {"read": state.reads}, mapping, {}, {}

    def capture(report, path):
        path.mkdir(parents=True)
        (path / "observation.json").write_text(json.dumps(report))

    def probe(session, handle, directory, stage, report):
        present = state.present.pop(0) if len(state.present) > 1 else state.present[0]
        state.probes.append((stage, present))
        if state.hook:
            state.hook(stage)
        return present

    monkeypatch.setattr(module.time, "monotonic", lambda: state.clock)
    monkeypatch.setattr(module, "save_probe", capture)
    monkeypatch.setattr(module, "screen_identity", lambda report: str(report["read"]))
    monkeypatch.setattr(module, "_record_notice_probe", probe)
    monkeypatch.setattr(module, "_poll_button", lambda *_: button)
    state.session = SimpleNamespace(
        current=current,
        page=SimpleNamespace(
            wait_for_timeout=lambda milliseconds: setattr(state, "clock", state.clock + milliseconds / 1000)
        ),
    )
    state.output = tmp_path
    state.button = button
    state.run = lambda **options: module._settle_reserved_notice(
        state.session,
        tmp_path,
        object(),
        deadline=110.0,
        cancelled=options.get("cancelled", lambda: False),
        check_sources=options.get("check_sources", lambda: None),
        remember=state.remembered.append,
    )
    return state


def test_injected_notice_clear_requires_new_full_guard_and_final_absence(clocked):
    result = clocked.run()
    assert result == {"read": 2} and clocked.reads == 2
    assert clocked.probes == [
        ("reserved-final-000", True),
        ("reserved-wait-000", False),
        ("reserved-final-001", False),
    ]
    settled = json.loads((clocked.output / "reserved-notice-settled.json").read_text())
    assert settled["full_revalidations"] == 2 and settled["waiting_footer_probes"] == 1
    assert settled["deadline_monotonic_seconds"] == 110
    assert settled["reservation_retained"] and not settled["save_click_dispatched"]
    assert not settled["fresh_save_acknowledgement_verified"] and not settled["task_completed"]


def test_injected_notice_reappears_after_expensive_guard_and_waits_again(clocked):
    clocked.present[:] = [True, False, True, False, False]
    assert clocked.run() == {"read": 3}
    assert clocked.reads == 3
    assert (clocked.output / "reserved-observations/002/observation.json").exists()


def test_injected_notice_never_clears_uses_one_original_deadline(clocked):
    clocked.present[:] = [True]
    with pytest.raises(BrowserSafetyStop, match="reserved_phase_timeout"):
        clocked.run()
    assert clocked.reads == 1 and clocked.clock >= 110
    assert not (clocked.output / "reserved-notice-settled.json").exists()


@pytest.mark.parametrize("where", ["full_guard", "final_probe", "waiting_probe"])
def test_injected_deadline_expiration_never_accepts_late_absence(clocked, where):
    if where == "full_guard":
        clocked.clock = 109
        clocked.present[:] = [False]
    else:
        clocked.hook = lambda stage: (
            setattr(clocked, "clock", 110)
            if stage.startswith("reserved-final" if where == "final_probe" else "reserved-wait")
            else None
        )
    with pytest.raises(BrowserSafetyStop, match="reserved_phase_timeout"):
        clocked.run()
    assert not (clocked.output / "reserved-notice-settled.json").exists()


@pytest.mark.parametrize("failure", ["source", "cancel", "context"])
def test_injected_wait_rechecks_sources_cancel_and_control_context(clocked, monkeypatch, failure):
    changed = False

    def hook(stage):
        nonlocal changed
        if stage == "reserved-wait-000":
            changed = True

    def sources():
        if changed and failure == "source":
            raise BrowserSafetyStop("source_changed")

    def button(*_):
        if changed and failure == "context":
            raise BrowserSafetyStop("context_changed")
        return clocked.button

    clocked.hook = hook
    monkeypatch.setattr(module, "_poll_button", button)
    with pytest.raises(BrowserSafetyStop):
        clocked.run(check_sources=sources, cancelled=lambda: changed and failure == "cancel")
    assert not (clocked.output / "reserved-notice-settled.json").exists()


@pytest.mark.parametrize("expire", ["before_notice", "during_notice", "after_readback"])
def test_acknowledgement_cannot_escape_shared_deadline(clocked, monkeypatch, expire):
    clocked.clock = 109.0 if expire == "after_readback" else 110 if expire == "before_notice" else 100

    def notice(*_):
        if expire == "during_notice":
            clocked.clock = 110
        return True

    monkeypatch.setattr(module, "_notice", notice)
    with pytest.raises(BrowserSafetyStop, match="reserved_phase_timeout"):
        module._wait_for_save_acknowledgement(
            clocked.session,
            clocked.output,
            object(),
            110,
            strict_deadline=True,
        )
    assert (clocked.output / "acknowledgement.json").exists() is (expire == "after_readback")


@pytest.mark.parametrize("reappear", [False, True])
def test_intercepted_reserved_notice_settles_with_single_click(no_save_page, tmp_path, monkeypatch, reappear):
    _, frame, _ = no_save_page
    original = module._record_notice_probe

    def probe(session, handle, directory, stage, report):
        if stage == "reserved-final-000" or reappear and stage == "reserved-final-001":
            frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
        elif stage.startswith("reserved-wait-"):
            frame.locator("#save-notice").evaluate("e=>e.textContent=''")
        return original(session, handle, directory, stage, report)

    monkeypatch.setattr(module, "_record_notice_probe", probe)
    receipt = save(no_save_page, tmp_path, settle_reserved_notice=True)
    assert frame.evaluate("window.fixtureSaves") == 1
    assert receipt["settle_reserved_notice"] is True
    assert receipt["reserved_phase_timeout_seconds"] == 20
    assert receipt["notice_was_already_present"] is False
    assert receipt["save_click_delivered"] and not receipt["task_completed"]
    settled = json.loads((tmp_path / "save/reserved-notice-settled.json").read_text())
    assert settled["full_revalidations"] == (3 if reappear else 2)
    assert len(list((tmp_path / "no-planet-save-reservations").glob("*.json"))) == 1


@pytest.mark.parametrize("mutation", ["choice", "reservation", "answer", "button", "cancel"])
def test_intercepted_reserved_wait_mutation_stops_without_click_or_reclaim(
    no_save_page, tmp_path, monkeypatch, mutation
):
    page, frame, options = no_save_page
    original = module._record_notice_probe
    cancelled = False

    def probe(session, handle, directory, stage, report):
        nonlocal cancelled
        if stage == "reserved-final-000":
            frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
        if stage == "reserved-wait-000":
            frame.locator("#save-notice").evaluate("e=>e.textContent=''")
            if mutation == "choice":
                options["choice_path"].write_text("{}")
            elif mutation == "reservation":
                next((tmp_path / "no-planet-save-reservations").glob("*.json")).write_text("{}")
            elif mutation == "answer":
                frame.locator("#period_days").fill("5")
            elif mutation == "button":
                frame.get_by_role("button", name="Save").evaluate("e=>e.replaceWith(e.cloneNode(true))")
            else:
                cancelled = True
        return original(session, handle, directory, stage, report)

    monkeypatch.setattr(module, "_record_notice_probe", probe)
    with pytest.raises(BrowserSafetyStop):
        save(no_save_page, tmp_path, settle_reserved_notice=True, cancelled=lambda: cancelled)
    assert frame.evaluate("window.fixtureSaves") == 0
    stop = json.loads((tmp_path / "save/stopped.json").read_text())
    assert stop["reservation_created"] and not stop["save_may_have_occurred"]
    assert not (tmp_path / "save/dispatch.json").exists()
    with pytest.raises(BrowserSafetyStop):
        module.save_no_planet_work(page, config(), tmp_path / "other", **options, settle_reserved_notice=True)
    assert frame.evaluate("window.fixtureSaves") == 0


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_invalid_optin_rejected_before_output(no_save_page, tmp_path, value):
    with pytest.raises(ValueError):
        save(no_save_page, tmp_path, settle_reserved_notice=value)
    assert not (tmp_path / "save").exists()


@pytest.mark.parametrize("mutation", ["between_reads", "boolean_alias"])
def test_claim_validation_and_pin_use_identical_typed_bytes(no_save_page, tmp_path, monkeypatch, mutation):
    _, frame, _ = no_save_page
    original = Path.read_bytes
    target = tmp_path / "save/reserved.json"
    changed = False

    def read(path):
        nonlocal changed
        raw = original(path)
        if path == target and not changed:
            changed = True
            payload = json.loads(raw)
            payload["max_save_clicks"] = True
            replacement = json.dumps(payload).encode()
            path.write_bytes(replacement)
            return replacement if mutation == "boolean_alias" else raw
        return raw

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(BrowserSafetyStop, match="reserved_intent_changed"):
        save(no_save_page, tmp_path, settle_reserved_notice=True)
    assert changed
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not (tmp_path / "save/dispatch.json").exists()
    assert not (tmp_path / "save/confirmed.json").exists()


@pytest.mark.parametrize("failure", ["cancel_after_ack", "late_notice", "late_readback"])
def test_intercepted_postclick_failure_remains_uncertain_and_never_repeats(
    no_save_page, tmp_path, monkeypatch, failure
):
    page, frame, options = no_save_page
    cancelled, offset = False, 0
    original_time = module.time.monotonic
    original_notice, original_current, original_persist = (
        module._notice,
        module.PlanetNumericSession.current,
        module.persist_json,
    )

    def notice(session, handle):
        nonlocal offset
        result = original_notice(session, handle)
        if failure == "late_notice" and result and (tmp_path / "save/dispatch.json").exists():
            offset = 31
        return result

    def current(session):
        nonlocal offset
        result = original_current(session)
        if failure == "late_readback" and (tmp_path / "save/acknowledgement.json").exists():
            offset = 31
        return result

    def persist(path, payload):
        nonlocal cancelled
        result = original_persist(path, payload)
        if failure == "cancel_after_ack" and path == tmp_path / "save/acknowledgement.json":
            cancelled = True
        return result

    monkeypatch.setattr(module.time, "monotonic", lambda: original_time() + offset)
    monkeypatch.setattr(module, "_notice", notice)
    monkeypatch.setattr(module.PlanetNumericSession, "current", current)
    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop, match="cancelled|reserved_phase_timeout"):
        save(no_save_page, tmp_path, settle_reserved_notice=True, cancelled=lambda: cancelled)
    assert frame.evaluate("window.fixtureSaves") == 1
    stopped = json.loads((tmp_path / "save/stopped.json").read_text())
    assert stopped["reservation_created"] and stopped["save_may_have_occurred"]
    assert not (tmp_path / "save/confirmed.json").exists()
    with pytest.raises(BrowserSafetyStop):
        module.save_no_planet_work(page, config(), tmp_path / "other", **options, settle_reserved_notice=True)
    assert frame.evaluate("window.fixtureSaves") == 1


def test_intercepted_expired_final_guard_retains_claim_without_dispatch(no_save_page, tmp_path, monkeypatch):
    _, frame, _ = no_save_page
    original_time, original_probe = module.time.monotonic, module._record_notice_probe
    offset = 0

    def probe(session, handle, directory, stage, report):
        nonlocal offset
        result = original_probe(session, handle, directory, stage, report)
        if stage == "reserved-final-000":
            offset = 31
        return result

    monkeypatch.setattr(module.time, "monotonic", lambda: original_time() + offset)
    monkeypatch.setattr(module, "_record_notice_probe", probe)
    with pytest.raises(BrowserSafetyStop, match="reserved_phase_timeout"):
        save(no_save_page, tmp_path, settle_reserved_notice=True)
    assert frame.evaluate("window.fixtureSaves") == 0
    stopped = json.loads((tmp_path / "save/stopped.json").read_text())
    assert stopped["reservation_created"] and not stopped["save_may_have_occurred"]
    assert not (tmp_path / "save/dispatch.json").exists()
