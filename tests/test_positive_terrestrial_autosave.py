"""Offline autosave authority tests; injected sessions are not native acceptance.

The capture/schema/projection and immutable receipt readers are real. Upstream
learned-source and owner schedulers use the explicitly declared existing seams.
"""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import test_browser_positive_finalize as positive_fixture
import test_browser_terrestrial_steps as terrestrial_fixture
import test_project_positive_evidence as positive_import_fixture
import test_project_terrestrial_evidence as terrestrial_import_fixture

import habfly.browser_positive_finalize as positive
import habfly.browser_positive_planet_workflow as workflow
import habfly.browser_terrestrial_workflow as terrestrial_workflow
import habfly.project_positive_evidence as positive_import
import habfly.project_terrestrial_evidence as terrestrial_import
from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.browser_autosave import (
    ACKNOWLEDGEMENT_SOURCE,
    MODE,
    autosave_flags,
    autosave_workflow_flags,
)
from habfly.browser_no_planet_workflow import _Evidence, _planet
from habfly.browser_numeric import screen_identity
from habfly.browser_planet_numeric import ANSWER_UNITS, planet_projection
from habfly.browser_probe import save_probe
from habfly.project_progress import ProgressError

injected = positive_fixture.injected
rig = terrestrial_fixture.rig
positive_offline = positive_import_fixture.offline
terrestrial_offline = terrestrial_import_fixture.offline


def read(path):
    return json.loads(path.read_bytes())


def replace_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture(params=["positive", "terrestrial"])
def capture_rig(tmp_path, monkeypatch, request):
    branch = request.param
    report = (
        positive_import_fixture.current_planet("Dulat")
        if branch == "positive"
        else terrestrial_import_fixture.current_habitat("Dulat", "not_habitable")
    )
    source_dir = tmp_path / "source"
    save_probe(report, source_dir)
    source_book = _Evidence(tmp_path)
    source_book.clean(source_dir)
    source_book.capture(source_dir)
    state = SimpleNamespace(now=0.0, checks=0, reads=0, closed=0, hook=None)
    monkeypatch.setattr(positive.time, "monotonic", lambda: state.now)

    def same(a, b):
        if branch == "positive":
            return planet_projection(a, _planet(a)) == planet_projection(b, _planet(b))
        return terrestrial_workflow._same_habitat(a, b)

    class Session:
        def __init__(self, page, config, output, **kwargs):
            assert kwargs == {"max_seconds": 60, "_pin_controls": True}
            self.report = deepcopy(report)
            self.mapping = _planet(report) if branch == "positive" else terrestrial_workflow._mapping(report)
            self.choices = {"selected": "gas_giant" if branch == "positive" else "not_habitable"}
            self.handles = dict.fromkeys(ANSWER_UNITS)
            save_probe(self.report, output / "initial")
            if branch == "positive":
                persist_json(
                    output / "scope.json",
                    {
                        "scope": "planet_exact_copy_transport",
                        "max_writes": 7,
                        "max_seconds": 60,
                        "star": "Dulat",
                        "task_completed": False,
                        "scientific_choices": "caller_supplied",
                        "retries": False,
                    },
                )

        def current(self):
            state.reads += 1
            if state.hook:
                state.hook(self)
            return self.report, self.mapping, self.choices, self.handles

        def close(self):
            state.closed += 1

    def check():
        state.checks += 1

    def produce(checker=check):
        return positive._capture_autosave_readback(
            None,
            None,
            tmp_path / "save",
            source_book=source_book,
            star="Dulat",
            branch=branch,
            session_factory=Session,
            validate=lambda r, *args: workflow._require(same(r, report), "fixture_changed"),
            check=checker,
            timeout_seconds=20,
        )

    def load(book=None):
        book = book or _Evidence(tmp_path)
        result = workflow._autosave_sources(
            book, tmp_path / "save", source_book, report, branch=branch, star="Dulat", same=same
        )
        return book, result

    return SimpleNamespace(
        root=tmp_path,
        source=source_book,
        report=report,
        state=state,
        branch=branch,
        produce=produce,
        load=load,
        Session=Session,
    )


def test_readback_receipt_has_no_save_or_persistence_authority(capture_rig):
    r = capture_rig
    result = r.produce()
    book, (loaded, after) = r.load()
    assert result == loaded and result["mode"] == MODE
    assert all(
        result[key] is False
        for key in (
            "save_click_delivered",
            "save_acknowledgement_verified",
            "persistence_verified",
            "cross_session_persistence_verified",
            "scientific_verified",
            "task_completed",
        )
    )
    assert result["browser_actions"] == 0 and result["visible_readback_verified"] is True
    assert (r.state.reads, r.state.closed) == (1, 1)
    assert screen_identity(after) == result["after_sha256"]
    assert not any(
        (r.root / "save" / name).exists()
        for name in ("reserved.json", "dispatch.json", "acknowledgement.json", "footer-probes")
    )
    assert r.root / "save" in book.closed_trees


@pytest.mark.parametrize(
    "kind",
    [
        "extra",
        "reservation",
        "failed",
        "missing",
        "symlink",
        "source",
        "after",
        "alias",
        "cross_branch",
        "hash",
    ],
)
def test_readback_reader_rejects_changed_or_fabricated_sources(capture_rig, kind):
    r = capture_rig
    receipt = r.produce()
    directory = r.root / "save"
    if kind in {"extra", "reservation", "failed"}:
        persist_json(
            directory
            / {"extra": "extra.json", "reservation": "dispatch.json", "failed": "stopped.json"}[kind],
            {},
        )
    elif kind == "missing":
        (directory / "after/manifest.json").unlink()
    elif kind == "symlink":
        (directory / "alias").symlink_to(r.root / "source", target_is_directory=True)
    elif kind == "source":
        (r.root / "source/observation.json").write_text("{}")
    elif kind == "after":
        after = read(directory / "after/observation.json")
        after["frames"][0]["text"] += " unexpected feedback"
        replace_json(directory / "after/observation.json", after)
        manifest = read(directory / "after/manifest.json")
        manifest["observation_sha256"] = hashlib.sha256(
            (directory / "after/observation.json").read_bytes()
        ).hexdigest()
        replace_json(directory / "after/manifest.json", manifest)
        receipt["after_sha256"] = screen_identity(after)
        replace_json(directory / "confirmed.json", receipt)
    else:
        receipt[
            {"alias": "save_click_delivered", "cross_branch": "branch", "hash": "before_sha256"}[kind]
        ] = {"alias": 0, "cross_branch": "no_planet", "hash": "f" * 64}[kind]
        replace_json(directory / "confirmed.json", receipt)
    with pytest.raises(BrowserSafetyStop):
        r.load()


def test_closed_tree_detects_later_added_file(capture_rig):
    r = capture_rig
    r.produce()
    book, _ = r.load()
    persist_json(r.root / "save/late.json", {})
    with pytest.raises(BrowserSafetyStop, match="tree_changed"):
        book.unchanged()


@pytest.mark.parametrize("kind", ["cancel", "source", "deadline", "answer"])
def test_readback_cancellation_and_changes_never_confirm(capture_rig, kind):
    r = capture_rig

    def hook(session):
        if kind == "source":
            (r.root / "source/observation.json").write_text("{}")
        elif kind == "deadline":
            r.state.now = 20
        elif kind == "answer":
            session.report["frames"][0]["text"] += " unexpected feedback"

    def check():
        if kind == "cancel" and r.state.reads:
            raise BrowserSafetyStop("operator_aborted")

    r.state.hook = hook
    with pytest.raises(BrowserSafetyStop):
        r.produce(check)
    assert not (r.root / "save/confirmed.json").exists()
    assert read(r.root / "save/stopped.json")["save_click_delivered"] is False
    assert r.state.closed == 1


def positive_owner(injected, monkeypatch, *, strategy="autosave", emit=None):
    original = positive.PositiveFinalizationSteps
    monkeypatch.setattr(
        positive, "PositiveFinalizationSteps", lambda *a, **kw: original(*a, **kw, save_strategy=strategy)
    )
    return injected[1](emit=emit)


def test_positive_owner_autosave_is_separate_readback_not_click(injected, monkeypatch):
    state, _, history, _ = injected
    owner = positive_owner(injected, monkeypatch)
    calls = []

    def readback(*args, **kwargs):
        kwargs["check"]()
        calls.append("visible")
        return autosave_flags()

    def verified(page, config, output, **kwargs):
        result = {
            "mode": "positive_planet_visible_workflow_readback",
            "task_completed": True,
            "planet": {"value": "gas_giant"},
            **autosave_workflow_flags(),
        }
        output.mkdir()
        persist_json(output / "confirmed.json", result)
        return result

    monkeypatch.setattr(positive, "_positive_autosave", readback)
    monkeypatch.setattr(positive, "verify_positive_planet_workflow", verified)
    owner.advance()
    owner.advance()
    owner.advance()
    assert owner.report["task_completed"] is True and calls == ["visible"]
    assert state.clicks == 0 and not owner.report["save_click_delivery_confirmed"]
    assert not owner.report["save_acknowledgement_verified"] and not owner.report["persistence_verified"]
    assert not any(e["event"] == "action_proposed" for e in state.events)
    assert not (history / "owner/save/dispatch.json").exists()


@pytest.mark.parametrize("value", [True, None, "assumed", 1])
def test_strict_strategy_is_rejected_before_owner_actions(injected, rig, monkeypatch, value):
    with pytest.raises((ValueError, BrowserSafetyStop)):
        positive_owner(injected, monkeypatch, strategy=value)
    with pytest.raises(BrowserSafetyStop):
        rig.make(save_strategy=value)
    assert not injected[0].clicks and not rig.state.calls


@pytest.mark.parametrize("callback", [False, True])
def test_positive_strategy_revocation_prevents_readback(injected, monkeypatch, callback):
    state, _, history, _ = injected

    def sink(event):
        if event["payload"].get("scheduled_call") == "autosave_visible_readback":
            state.owner.save_strategy = "explicit"

    owner = positive_owner(injected, monkeypatch, emit=sink if callback else None)
    owner.advance()
    if not callback:
        owner.save_strategy = "explicit"
    owner.advance()
    assert owner.finished and "save_options_changed" in owner.report["outcome"]
    assert not state.clicks and not (history / "owner/save").exists()


def test_terrestrial_autosave_schedule_never_invokes_save_adapter(rig, monkeypatch):
    owner = rig.make(save_strategy="autosave")
    terrestrial_fixture.gases(owner)
    terrestrial_fixture.habitat(owner)
    terrestrial_fixture.reach(owner, "save")

    def readback():
        directory = owner.output / "save"
        directory.mkdir()
        receipt = autosave_flags()
        persist_json(directory / "confirmed.json", receipt)
        terrestrial_fixture.capture(deepcopy(rig.report), directory / "after")
        return receipt

    monkeypatch.setattr(owner, "_autosave_readback", readback)
    owner.advance()
    assert owner.phase == "verify" and rig.state.save_clicks == 0
    assert owner.state()["visible_readback_verified"] and not owner.state()["save_acknowledgement_verified"]
    assert read(owner.output / "save-call-reserved.json")["native_write_dispatched"] is False
    owner.abort()


def test_terrestrial_strategy_callback_revocation_stops_before_save(rig):
    owner = rig.make(save_strategy="autosave")
    terrestrial_fixture.gases(owner)
    terrestrial_fixture.habitat(owner)
    terrestrial_fixture.reach(owner, "save")

    def callback(event):
        if event["payload"].get("scheduled_call") == "save":
            owner.save_strategy = "explicit"

    owner._callback = callback
    owner.advance()
    assert owner.finished and owner.failure == "terrestrial_steps_fixed_options_changed"
    assert not rig.state.save_clicks and not (owner.output / "save").exists()


def make_import_autosave(offline, branch):
    _, history, _, directory, registry, _ = offline
    receipt = read(directory / "confirmed.json")
    hashes, bundle = next(iter(registry.values()))
    save_name = next(name for name in hashes if name.endswith("save/confirmed.json"))
    replace_json(history / save_name, {"mode": MODE, "branch": branch})
    hashes[save_name] = hashlib.sha256((history / save_name).read_bytes()).hexdigest()
    bundle.update(autosave_workflow_flags())
    receipt.update(
        autosave_workflow_flags(),
        source_sha256=hashes,
        save_acknowledgement_verified=False,
        source_save_click_delivered=False,
        save_acknowledgement_source=ACKNOWLEDGEMENT_SOURCE,
    )
    replace_json(directory / "confirmed.json", receipt)
    return directory, receipt, bundle


@pytest.mark.parametrize("branch", ["positive", "terrestrial"])
def test_import_retains_visible_only_save_scope(request, branch):
    offline = request.getfixturevalue(branch + "_offline")
    module = positive_import if branch == "positive" else terrestrial_import
    directory, _, _ = make_import_autosave(offline, branch)
    # Declared upstream-source seam is inherited; source bytes remain hashed and
    # current readbacks/inventory/journal validation use the real importer.
    result = (
        module.import_verified_positive_planet(*offline[:4])
        if branch == "positive"
        else module.import_verified_terrestrial(*offline[:4])
    )
    assert result["progress"]["verified"] == 1
    assert result["evidence_scopes"]["save_readback"]["persistence_verified"] is False
    assert read(directory / "confirmed.json")["save_acknowledgement_verified"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("persistence_verified", True),
        ("persistence_verified", 0),
        ("source_save_click_delivered", True),
        ("save_acknowledgement_verified", True),
        ("autosave_readback_mode", "unknown"),
    ],
)
def test_import_cannot_upgrade_assumption_to_ack(positive_offline, field, value):
    directory, receipt, _ = make_import_autosave(positive_offline, "positive")
    receipt[field] = value
    replace_json(directory / "confirmed.json", receipt)
    with pytest.raises((BrowserSafetyStop, ProgressError)):
        positive_import_fixture.ingest(positive_offline)


def test_import_cannot_relabel_explicit_save_sources(positive_offline):
    _, history, _, directory, _, _ = positive_offline
    receipt = read(directory / "confirmed.json")
    receipt.update(
        autosave_workflow_flags(),
        save_acknowledgement_verified=False,
        source_save_click_delivered=False,
        save_acknowledgement_source=ACKNOWLEDGEMENT_SOURCE,
    )
    replace_json(directory / "confirmed.json", receipt)
    with pytest.raises(ProgressError, match="save_strategy_mismatch"):
        positive_import_fixture.ingest(positive_offline)
    assert not (history / "positive-evidence").exists()
