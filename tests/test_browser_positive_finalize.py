"""Offline injected transport gates; native Chromium integration is separate."""
# ruff: noqa: F811

import hashlib
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_no_planet_workflow import snapshot, stellar_sources, write
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_positive_planet_workflow import derived_stream
from test_browser_positive_planet_workflow import kwargs as source_kwargs
from test_browser_positive_planet_workflow import seam as positive_seam
from test_browser_raster_planet_evidence import copy, raster_page, select, sources  # noqa: F401
from test_project_evidence import inventory

import habfly.browser_positive_finalize as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.browser_planet_classification import select_planet_class
from habfly.browser_planet_numeric import ANSWER_UNITS
from habfly.browser_planet_numeric import PlanetNumericSession as NativeSession
from habfly.browser_positive_planet_workflow import DERIVED
from habfly.browser_probe import save_probe
from habfly.contracts import RuntimeEvent
from habfly.project_positive_evidence import import_verified_positive_planet
from habfly.project_progress import ProjectJournal


def read(path):
    return json.loads(path.read_bytes())


@pytest.fixture
def injected(tmp_path, monkeypatch):
    state = SimpleNamespace(
        now=0.0,
        reads=0,
        readbacks=0,
        clicks=0,
        verifications=0,
        clicked_at=None,
        mode="normal",
        callbacks={},
        events=[],
        owner=None,
    )
    source = tmp_path / "source"
    source.mkdir()
    pin = source / "confirmed.json"
    persist_json(pin, {"fixture": "immutable supplied gas giant"})
    report = {
        "schema_version": 1,
        "mode": "read_only_browser_preflight",
        "frames": [],
        "ignored_frame_urls": [],
        "value": 42,
    }
    mapping = {
        "star_name": "Dulat",
        "observation": {
            "values": {
                "browser_field_map": {
                    name: {"current_value": "1", "unit": unit} for name, unit in ANSWER_UNITS.items()
                }
            }
        },
    }
    bundle = {"star": "Dulat", "planet_class": "gas_giant", "class_capture": report, "save_capture": report}

    def sources(book, directories, class_sha):
        book.clean(source)
        assert hashlib.sha256(book.read(pin)).hexdigest() == class_sha
        return deepcopy(bundle)

    monkeypatch.setattr(module, "_sources", sources)
    monkeypatch.setattr(module, "_planet", lambda _: deepcopy(mapping))
    monkeypatch.setattr(module, "planet_projection", lambda r, m: r["value"])
    monkeypatch.setattr(module.time, "monotonic", lambda: state.now)

    class Page:
        def on(self, kind, callback):
            state.callbacks[kind] = callback

        def remove_listener(self, kind, callback):
            state.callbacks.pop(kind, None)

        def wait_for_timeout(self, delay):
            state.now += delay / 1000

    page = Page()

    class Handle:
        def element_handle(self, **kwargs):
            return self

        def is_enabled(self):
            return True

        def click(self, **kwargs):
            state.clicks += 1
            state.clicked_at = state.now
            if state.mode == "click_failure":
                raise RuntimeError("private browser diagnostic")
            if state.mode == "popup":
                state.callbacks["popup"](None)
            if state.mode == "source_after_click":
                pin.write_text("changed")
            if state.mode == "abort_after_click":
                state.owner.abort()

    handle = Handle()

    class Session:
        def __init__(self, page, config, output, **kwargs):
            self.page, self.report = page, deepcopy(report)
            save_probe(self.report, output / "initial")

        def current(self):
            state.reads += 1
            state.now += 0.35
            if state.mode == "changed_answer":
                self.report["value"] = 99
            if state.mode == "source_during_settlement" and state.reads == 2:
                pin.write_text("changed")
            return self.report, deepcopy(mapping), {"selected": bundle["planet_class"]}, {}

        def close(self):
            pass

    monkeypatch.setattr(module, "PlanetNumericSession", Session)

    def button(session, target=None):
        if state.mode == "replaced" and module._claim_path(tmp_path, "Dulat").exists():
            raise BrowserSafetyStop("positive_finalization_save_control_replaced")
        return handle

    monkeypatch.setattr(module, "_button", button)

    def notice(session, target):
        if state.clicked_at is None:
            if state.mode == "stale_forever":
                return True
            if state.mode == "stale_then_clear":
                return state.now < 0.8
            if state.mode == "late_stale":
                return module._claim_path(tmp_path, "Dulat").exists()
            return False
        if state.mode == "no_ack":
            return False
        return 0.08 <= state.now - state.clicked_at <= 0.28

    monkeypatch.setattr(module, "_notice", notice)

    def readback(*args):
        state.readbacks += 1

    monkeypatch.setattr(module, "_read_planet", readback)

    def verify(page, config, output, **kwargs):
        state.verifications += 1
        if state.mode == "verify_failure":
            raise BrowserSafetyStop("current_star_changed")
        result = {
            "mode": "positive_planet_visible_workflow_readback",
            "task_completed": True,
            "planet": {"value": bundle["planet_class"]},
        }
        output.mkdir()
        if state.mode != "verify_missing":
            persist_json(
                output / "confirmed.json",
                {**result, "changed": True} if state.mode == "verify_mismatch" else result,
            )
        return result

    monkeypatch.setattr(module, "verify_positive_planet_workflow", verify)
    config = SimpleNamespace(model_copy=lambda **kwargs: None)

    def make(name="owner", emit=None):
        owner = module.PositiveFinalizationSteps(
            page,
            config,
            tmp_path / name,
            run_history=tmp_path,
            **dict.fromkeys(
                ("numeric_dir", "color_dir", "class_dir", "raw_dir", "derived_dir", "planet_class_dir"),
                source,
            ),
            planet_class_sha256=hashlib.sha256(pin.read_bytes()).hexdigest(),
            emit=emit or state.events.append,
            timeout_seconds=1,
            settle_timeout_seconds=2,
        )
        state.owner = owner
        return owner

    return state, make, tmp_path, pin


def test_constructor_no_actions_and_three_separate_advances(injected):
    state, make, history, _ = injected
    owner = make()
    assert (state.readbacks, state.reads, state.clicks, state.verifications) == (0, 0, 0, 0)
    assert owner.state()["phase"] == "readback" and not owner.state()["task_completed"]
    assert owner.advance()["phase"] == "save" and state.clicks == 0
    assert owner.advance()["phase"] == "verify" and state.clicks == 1 and state.verifications == 0
    assert not owner.state()["task_completed"]
    assert owner.advance()["task_completed"] and owner.finished and state.verifications == 1
    before = (state.reads, state.clicks, state.verifications)
    owner.advance()
    owner.close()
    assert (state.reads, state.clicks, state.verifications) == before
    native = read(history / "owner/save/confirmed.json")
    assert (
        native["save_click_delivered"]
        and not native["notice_was_already_present"]
        and not native["task_completed"]
    )
    assert read(history / "owner/save/dispatch.json") == {
        "kind": "CLICK",
        "visible_label": "Save",
        "max_clicks": 1,
    }
    assert owner.report["journal_writes"] == owner.report["optimizer_updates"] == 0
    assert not owner.report["scientific_verified"] and not owner.report["project_completed"]
    rows = [
        RuntimeEvent.model_validate_json(line)
        for line in (history / "owner/events.jsonl").read_text().splitlines()
    ]
    assert [r.sequence for r in rows] == list(range(len(rows)))
    assert rows[-1].event == "episode_summary" and rows[-1].payload["task_completed"]
    assert [r.model_dump(mode="json") for r in rows] == state.events
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        make("another-output")


@pytest.mark.parametrize("mode", ["normal", "stale_then_clear"])
def test_delayed_brief_ack_precedes_slow_full_guard(injected, mode):
    state, make, history, _ = injected
    state.mode = mode
    owner = make()
    owner.advance()
    owner.advance()
    assert state.clicks == 1 and owner.state()["phase"] == "verify"
    assert state.now - state.clicked_at > 0.35  # Full capture completed after the short notice vanished.
    assert read(history / "owner/save/acknowledgement.json")["visible_text"] == "Data saved"
    assert read(history / "owner/save/pre-reservation-settled.json")["notice_present"] is False
    owner.close()


@pytest.mark.parametrize(
    "mode,attempted,reserved",
    [
        ("stale_forever", False, False),
        ("changed_answer", False, False),
        ("source_during_settlement", False, False),
        ("late_stale", False, True),
        ("replaced", False, True),
        ("click_failure", True, True),
        ("no_ack", True, True),
        ("popup", True, True),
        ("source_after_click", True, True),
        ("abort_after_click", True, True),
    ],
)
def test_uncertainty_stops_without_retry_or_verifier(injected, mode, attempted, reserved):
    state, make, history, _ = injected
    state.mode = mode
    owner = make()
    owner.advance()
    owner.advance()
    assert owner.finished and not owner.state()["task_completed"] and not owner.report["task_completed"]
    assert state.clicks == int(attempted) and not state.verifications
    assert owner.report["save_dispatch_attempts_recorded"] == int(attempted)
    assert owner.report["save_may_have_occurred"] is attempted
    assert not owner.report["save_click_delivery_confirmed"]
    stopped = read(history / "owner/save/stopped.json")
    assert stopped["save_may_have_occurred"] is attempted and stopped["reservation_created"] is reserved
    assert module._claim_path(history, "Dulat").exists() is reserved
    owner.advance()
    assert state.clicks == int(attempted)
    if reserved and mode != "source_after_click":
        with pytest.raises(BrowserSafetyStop, match="already_reserved"):
            make("changed-output")


def test_known_legacy_save_intent_is_not_a_new_output_bypass(injected):
    state, make, history, _ = injected
    (history / "legacy").mkdir()
    persist_json(
        history / "legacy/reserved.json", {"kind": "CLICK", "visible_label": "Save", "star": "DULAT"}
    )
    with pytest.raises(BrowserSafetyStop, match="prior_save_intent"):
        make()
    assert not state.clicks and not (history / "owner").exists()


def test_source_changed_between_advances_and_abort_before_save(injected):
    state, make, history, pin = injected
    owner = make()
    owner.advance()
    pin.write_text("changed")
    owner.advance()
    assert owner.finished and not state.clicks and not module._claim_path(history, "Dulat").exists()


def test_abort_is_terminal_without_browser_or_save(injected):
    state, make, _, _ = injected
    owner = make()
    owner.abort()
    owner.advance()
    assert owner.finished and not owner.report["task_completed"]
    assert state.readbacks == state.reads == state.clicks == state.verifications == 0


@pytest.mark.parametrize("event", ["hello", "action_proposed", "episode_summary"])
def test_callback_failure_cannot_report_completion(injected, event):
    state, make, _, _ = injected

    def callback(item):
        if item["event"] == event:
            raise RuntimeError("private callback detail")

    owner = make(emit=callback)
    for _ in range(3):
        owner.advance()
    assert owner.finished and not owner.report["task_completed"] and not owner.state()["task_completed"]
    assert "private" not in json.dumps(owner.report)
    assert state.clicks == (1 if event == "episode_summary" else 0)


@pytest.mark.parametrize("mode", ["verify_failure", "verify_missing", "verify_mismatch"])
def test_verification_failure_keeps_save_but_never_completes(injected, mode):
    state, make, history, _ = injected
    state.mode = mode
    owner = make()
    owner.advance()
    owner.advance()
    owner.advance()
    assert state.clicks == 1 and owner.finished and not owner.state()["task_completed"]
    assert owner.report["save_acknowledgement_verified"] and not owner.report["task_completed"]
    assert (history / "owner/save/confirmed.json").exists()
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        make("changed-output")


def test_last_event_source_change_cannot_be_success(injected):
    _, make, _, pin = injected

    def callback(item):
        if item["event"] == "episode_summary" and item["payload"]["task_completed"]:
            pin.write_text("changed")

    owner = make(emit=callback)
    owner.advance()
    owner.advance()
    owner.advance()
    assert owner.finished and not owner.report["task_completed"] and not owner.state()["task_completed"]


@pytest.mark.parametrize("class_name", ["terrestrial", "main_sequence", None])
def test_source_validator_never_supplies_classification(tmp_path, monkeypatch, class_name):
    directory = tmp_path / "source"
    directory.mkdir()
    persist_json(directory / "confirmed.json", {"class": class_name})
    monkeypatch.setattr(module, "_stellar_sources", lambda *a: {"star": "Dulat"})
    monkeypatch.setattr(module, "_planet_sources", lambda *a: {"star": "DULAT", "planet_class": class_name})
    directories = dict.fromkeys(
        ("numeric_dir", "color_dir", "class_dir", "raw_dir", "derived_dir", "planet_class_dir"), directory
    )
    with pytest.raises(BrowserSafetyStop, match="explicit_non_terrestrial"):
        module._sources(
            module._Evidence(tmp_path),
            directories,
            hashlib.sha256((directory / "confirmed.json").read_bytes()).hexdigest(),
        )


@pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Native fixture gate requires sole live owner's explicit idle confirmation",
)
@pytest.mark.parametrize("planet_class", ["gas_giant", "ice_giant"])
def test_native_finalization_without_prior_save_preserves_import_contract(
    raster_page, tmp_path, monkeypatch, planet_class
):
    """No prior Save is erased; this fixture never invokes the legacy saver."""
    page, frame = raster_page
    frame.locator("#derived").evaluate(
        "e=>e.textContent='STAR MASS (Ms) 1 STAR RADIUS (Rs) 1 ORBIT (years) 0.000'"
    )
    frame.evaluate("""()=>{
      const style=document.createElement('style');
      style.textContent='.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}';document.body.append(style);
      document.querySelectorAll('.choice label').forEach(label=>label.onclick=()=>label.parentElement.classList.add('selected'));
      const button=[...document.querySelectorAll('button')].find(e=>e.textContent==='Save');
      const footer=document.createElement('div'),notice=document.createElement('div');
      notice.id='save-notice';button.before(footer);footer.append(notice,button);
    }""")
    source = sources(raster_page, tmp_path)
    select(raster_page, tmp_path, source)
    copy(raster_page, tmp_path)
    payloads = []
    directory = tmp_path / "derived"
    session = NativeSession(page, config(), directory / "native-copies", lambda *args: payloads.append(args))
    try:
        for name in sorted(DERIVED):
            session.copy(name, "1", ANSWER_UNITS[name], source="checkpoint")
        verified = deepcopy(session.verified)
    finally:
        session.close()
    derived_stream(
        directory,
        {
            "scope": "four_derived_planet_browser_transport",
            "outcome": "planet_derived_transport_verified",
            "checkpoint_unchanged": True,
            "planet_transport_verified": True,
            "optimizer_updates": 0,
            "task_completed": False,
            "browser_acceptance_passed": False,
            "saved": False,
            "assessment_performed": False,
            "submitted": False,
            "write_attempts": sorted(DERIVED),
            "verified_fields": verified,
            "provenance": {
                "checkpoint_sha256": "a" * 64,
                "graph_hash": "b" * 64,
                "knowledge_pack_hash": "c" * 64,
                "optimizer_updates": 0,
                "supplied_star_class": "main_sequence",
                "classification_source": "supplied_not_learned",
                "synthetic_fixture_only": True,
            },
        },
        payloads,
    )
    select_planet_class(
        page, config(), tmp_path / "planet-class", planet_class, source="reference_diagnostic"
    )
    planet = snapshot(frame)
    write(tmp_path / "choice/confirmed.json", {"star": "JYREMIS"})
    stellar = stellar_sources(page, frame, tmp_path)
    frame.locator("body").evaluate("(e,h)=>e.innerHTML=h", planet)
    # Fixture event handlers are not serialized by innerHTML. This only installs
    # the test Save acknowledgement; no persisted attempt is deleted or reset.
    frame.get_by_role("button", name="Save", exact=True).evaluate("""e=>e.onclick=()=>{
      window.finalizationSaves=(window.finalizationSaves||0)+1;
      document.querySelector('#save-notice').textContent='Data saved';
    }""")
    directories = {k: v for k, v in source_kwargs(tmp_path).items() if k != "save_dir"}
    owner = module.PositiveFinalizationSteps(
        page,
        config(),
        tmp_path / "finalize",
        run_history=tmp_path,
        **directories,
        planet_class_sha256=hashlib.sha256(
            (tmp_path / "planet-class/confirmed.json").read_bytes()
        ).hexdigest(),
    )
    assert frame.evaluate("window.finalizationSaves||0") == 0
    assert owner.advance()["phase"] == "save"
    assert frame.evaluate("window.finalizationSaves||0") == 0
    assert owner.advance()["phase"] == "verify"
    assert frame.evaluate("window.finalizationSaves") == 1
    calls = positive_seam(monkeypatch, frame, {"stellar": stellar, "planet": snapshot(frame)})
    assert owner.advance()["task_completed"] and calls == ["stellar", "planet"]
    inv = inventory(tmp_path, ["JYREMIS"], "native-finalized")
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="intercepted-finalization").create()
    result = import_verified_positive_planet(journal, tmp_path, inv, tmp_path / "finalize/workflow")
    assert result["progress"]["verified"] == 1 and result["appended_records"] == 7
    assert result["evidence_scopes"]["habitability"]["outcome"] == "not_applicable"
    assert not result["evidence_scopes"]["habitability"]["transport_verified"]
    assert frame.evaluate("window.finalizationSaves") == 1
