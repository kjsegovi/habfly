"""Offline joined finalization, not course or astronomical acceptance.

Thirty upstream workflow bundles are explicit synthetic fixtures at the same
source seam used by campaign/import tests. The canonical imports, historical
owner validators, live-mode pager receipts, cooperative scoring components and
submission preflight/actuator execute unchanged. Only native page operations
and the upstream already-tested numerical/source bundle are injected.
"""
# ruff: noqa: F811

import io
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_numeric import WIDGET, config
from test_browser_project_paginated_inventory_steps import subject  # noqa: F401
from test_browser_project_scoring_coordinator import rig  # noqa: F401
from test_browser_project_scoring_coordinator_paginated import paged  # noqa: F401
from test_browser_project_scoring_steps import case  # noqa: F401
from test_project_evidence import inventory, read, sha, workflow, write

import habfly.browser_project_finalize_steps as module
import habfly.browser_project_submission_preflight as preflight
import habfly.browser_project_submission_steps as submission
import habfly.project_evidence as evidence
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_project_campaign_steps import _verified_owner
from habfly.contracts import RuntimeEvent
from habfly.project_progress import Active, _json
from habfly.runtime import Runtime, read_trace

_TEMPLATE = None
_SCORING_TEMPLATE = None


@pytest.fixture
def finished_campaign(paged, monkeypatch):
    global _TEMPLATE
    registry = {}

    def validated(book, **directories):
        hashes, bundle = registry[str(directories["numeric_dir"])]
        for name in hashes:
            book.clean((book.history / name).parent)
            book.read(book.history / name)
        return deepcopy(bundle)

    monkeypatch.setattr(evidence, "_load_sources", validated)
    root, journal = paged.root, paged.journal
    if _TEMPLATE is not None:
        files, source_registry = _TEMPLATE
        for name, raw in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        registry.update({str(root / name): deepcopy(value) for name, value in source_registry.items()})
        paged.historical = journal.path.read_bytes()
        paged.before_campaign = {
            str(path.relative_to(root)): sha(path)
            for path in (root / "campaign").rglob("*")
            if path.is_file()
        }
        paged.post_events = []
        return paged
    # Replace only disposable synthetic setup, before any owner is constructed.
    journal.path.write_text(_json(journal.identity.header()) + "\n")
    book = _Evidence(root)
    scope = {
        "mode": module.CAMPAIGN_MODE,
        "target_stars": 30,
        "owner_limits": {"max_seconds": 1800, "max_advances": 512},
        "automatic_retry": False,
    }
    write(root / "campaign/scope.json", scope)
    book.read(root / "campaign/scope.json")
    names = []
    for index in range(30):
        ordinal, star = index + 1, f"Star{index:02d}"
        names.append(star)
        inv = paged.initial_inventory.output if ordinal == 30 else inventory(root, names, f"prefix-{ordinal}")
        work = workflow(root, star, f"star-{ordinal}", registry)
        imported = evidence.import_verified_no_planet(journal, root, inv, work)
        owner_dir = root / f"campaign/stars/{ordinal:03d}/owner"
        owner = {
            "mode": "bounded_single_star_project_runtime",
            "status": "completed",
            "phase": "verified_no_planet",
            "finished": True,
            "star": star,
            "task_completed": True,
            "project_completed": False,
            "failure_reason": None,
            "event_forwarding_failed": False,
            "project_progress": imported["progress"],
        }
        write(owner_dir / "report.json", owner)
        write(owner_dir / "workflow-import.json", imported)
        event = RuntimeEvent(
            event="episode_summary",
            run_id=f"owner-{ordinal}",
            sequence=0,
            payload={**owner, "completed": True},
        )
        (owner_dir / "events.jsonl").write_text(event.model_dump_json() + "\n")
        context = {
            "ordinal": ordinal,
            "star": star,
            "fresh_dir": f"fresh-{ordinal}",
            "fresh_sha256": "f" * 64,
            "class_output": f"source-star-{ordinal}/class",
            "owner_output": str(owner_dir.relative_to(root)),
        }
        write(root / f"campaign/star-{ordinal:03d}.json", context)
        book.read(root / f"campaign/star-{ordinal:03d}.json")
        _, _, sources = _verified_owner(book, journal, owner_dir, names, star)
        verified = {**context, **sources, "journal_sha256": sha(journal.path)}
        write(root / f"campaign/verified-{ordinal:03d}.json", verified)
        book.read(root / f"campaign/verified-{ordinal:03d}.json")
    campaign = {
        **scope,
        "status": "handoff",
        "phase": "awaiting_assessment",
        "finished": True,
        "target_workflows_verified": True,
        "task_completed": False,
        "project_completed": False,
        "failure_reason": None,
        "event_forwarding_failed": False,
        "project_progress": journal.load().reduce().report(),
        "verified_stars": 30,
        "journal_sha256": sha(journal.path),
        "source_sha256": dict(book.hashes),
        "current_star": context,
    }
    write(root / "campaign/report.json", campaign)
    write(
        root / "report.json",
        {
            **campaign,
            "mode": "fresh_browser_campaign_project_runtime",
            "project_campaign": True,
            "browser_status": "visible",
            "campaign": campaign,
        },
    )
    paged.historical = journal.path.read_bytes()
    paged.before_campaign = {
        str(path.relative_to(root)): sha(path) for path in (root / "campaign").rglob("*") if path.is_file()
    }
    paged.post_events = []
    _TEMPLATE = (
        {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()},
        {str(Path(name).relative_to(root)): deepcopy(value) for name, value in registry.items()},
    )
    return paged


def make(finished_campaign, **options):
    rig = finished_campaign
    return module.BrowserProjectFinalizeSteps(
        rig.page,
        config(),
        rig.root / options.pop("output", "finalization"),
        run_history=rig.root,
        journal=rig.journal,
        campaign_runtime_dir=rig.root,
        emit=options.pop("emit", rig.post_events.append),
        cancelled=lambda: rig.flags.cancelled,
        _clock=lambda: rig.clock[0],
        **options,
    )


def drive(owner, *, until=None):
    if owner.status == "idle":
        owner.start(paused=True)
    for _ in range(module.MAX_ADVANCES):
        if owner.finished or owner.phase == until:
            return
        owner.step()
    pytest.fail("fixed parent scheduling budget exhausted")


def completed_scoring_factory(rig):
    """Replay a genuinely generated disposable scoring bundle for final-guard tests.

    The joined tests above still run every real child. These tests target only
    the final callback boundary; replaying those same pinned synthetic files
    avoids charging/navigating the injected page again. No production validator
    is patched. Running a guard test alone falls back to the real coordinator.
    """
    if _SCORING_TEMPLATE is None:
        return None
    files, raw_journal, report = _SCORING_TEMPLATE

    class Completed:
        def __init__(self, _page, _config, output, *, journal, **_):
            self.output, self.journal = output, journal
            self.finished, self.report, self.failure, self.advances = False, None, None, 0
            self.current_sha = sha(journal.path)
            self.scoring = SimpleNamespace(
                output=output / "scoring", state=lambda: {"journal_sha256": self.current_sha}
            )

        def state(self):
            return (
                deepcopy(report)
                if self.finished
                else {
                    "max_seconds": 600,
                    "max_advances": 46,
                    "allow_score_transfer": False,
                    "phase": "injected_persisted_source",
                    "finished": False,
                }
            )

        def advance(self):
            self.advances += 1
            for name, raw in files.items():
                path = rig.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            self.journal.path.write_bytes(raw_journal)
            self.current_sha = sha(self.journal.path)
            self.finished, self.report = True, deepcopy(report)

        def abort(self):
            self.finished = True

    return Completed


def attach_submission_surface(rig, monkeypatch):
    page = rig.page
    state = SimpleNamespace(native=[], ready_checked=False, before_native=None, screenshot_count=0)
    original = page.report

    class Control:
        def __init__(self, role, label):
            self.role, self.label = role, label

        def count(self):
            return 1

        def is_visible(self):
            return True

        def is_enabled(self):
            return True

        def is_checked(self):
            return state.ready_checked if self.role == "checkbox" else False

        def aria_snapshot(self):
            return f'- {self.role} "{self.label}"' + (" [checked]" if self.is_checked() else "")

        def element_handle(self, **_):
            return self

        def evaluate(self, script, value=None):
            if script.startswith("(a,b)"):
                return self is value
            return {"x": 10, "y": 10, "width": 20, "height": 20}

        def click(self, **_):
            assert len(rig.journal.load().reduce().pending) == 1
            if state.before_native:
                state.before_native(self.label)
            state.native.append(self.label)
            if self.role == "checkbox":
                state.ready_checked = True

    ready, submit = Control("checkbox", preflight.READY), Control("button", preflight.SUBMIT)
    page.main_frame = object()
    simulation, widget = page.frames
    simulation.parent_frame = widget.parent_frame = page.main_frame
    update_widget = SimpleNamespace(
        url=WIDGET, parent_frame=page.main_frame, get_by_role=lambda *_a, **_k: SimpleNamespace(all=list)
    )
    page.frames.insert(1, update_widget)
    # This joined fixture already injects native page operations. Provide the
    # visible iframe-element contract so the real embedding-chain predicate can
    # pin the same three visible frames as its injected observation reports.
    # Hidden/new/replaced frames are tested separately; no production fallback.
    for frame in page.frames:
        frame.frame_element = lambda: SimpleNamespace(is_visible=lambda: True)
    widget.get_by_role = lambda *_a, **_k: SimpleNamespace(all=lambda: [submit])
    page.get_by_role = lambda *_a, **_k: ready

    def report():
        value = deepcopy(original())
        value["outer_controls"] = [
            {
                "id": "outer:c0",
                "role": "checkbox",
                "enabled": True,
                "accessibility": ready.aria_snapshot(),
                "actions": [],
                "protected": True,
            }
        ]
        value["frames"].append(
            {
                "id": "widgets-1",
                "url": WIDGET,
                "text": preflight.SUBMIT,
                "accessibility": submit.aria_snapshot(),
                "controls": [
                    {
                        "id": "widgets-1:c0",
                        "role": "button",
                        "enabled": True,
                        "accessibility": submit.aria_snapshot(),
                        "actions": [],
                        "protected": True,
                    }
                ],
            }
        )
        return value

    page.screenshot = lambda **_: b"\x89PNG\r\n\x1a\nsynthetic-not-course-evidence"
    page.locator = lambda _: SimpleNamespace(
        inner_text=lambda **_: "Fixture response, not a submission receipt",
        aria_snapshot=lambda **_: "- text: Fixture response",
    )
    for target in (preflight, submission):
        monkeypatch.setattr(target, "inspect_page", lambda *_: report())
        monkeypatch.setattr(target, "read_outer_score", lambda _: page.score)
        monkeypatch.setattr(target, "_outer_exposed", lambda *_: True)
    return state


@pytest.mark.parametrize("transfer", [False, True])
def test_joined_handoff_paginated_scoring_preserves_campaign(finished_campaign, transfer):
    global _SCORING_TEMPLATE
    rig = finished_campaign
    owner = make(rig, allow_score_transfer=transfer)
    assert owner.status == "idle" and not rig.calls and not rig.page.clicks
    assert owner.scoring is None and rig.journal.path.read_bytes() == rig.historical
    assert read(owner.output / "campaign-handoff.json")["historical_journal_sha256"] == sha(
        owner.output / "campaign-journal.jsonl"
    )
    owner.start(paused=True)
    for _ in range(module.MAX_ADVANCES):
        if owner.finished:
            break
        before = len(rig.calls) + len(rig.page.clicks) + len(rig.subject.state.clicks)
        child_advances = owner.scoring.advances if owner.scoring else 0
        owner.step()
        if owner.scoring:
            assert owner.scoring.advances - child_advances <= 1
        assert len(rig.calls) + len(rig.page.clicks) + len(rig.subject.state.clicks) >= before
    assert owner.status == "handoff", owner.report
    assert owner.phase == (
        "score_transferred_not_submitted" if transfer else "assessed_score_transfer_disabled"
    )
    assert rig.page.clicks == ["ASSESS", "OK", "ASSESS", "OK"] + (["Update Score"] if transfer else [])
    assert len(rig.inventory_children) == (3 if transfer else 2)
    assert rig.journal.path.read_bytes().startswith(rig.historical)
    assert sha(rig.journal.path) != owner.scope["campaign_journal_sha256"]
    assert all(sha(rig.root / path) == value for path, value in rig.before_campaign.items())
    assert not any(owner.report[k] for k in ("task_completed", "project_completed", "submitted"))
    assert owner.report["scoring_dir"] == "finalization/scoring/scoring"
    assert owner.report["score_checkpoint_completed"] is transfer
    assert owner.report["reported_score"] == (56.59 if transfer else None)
    assert owner.state()["score_checkpoint_completed"] is transfer
    assert owner._stream.closed
    # _finish's raw summary precedes closure. The subsequent actual Runtime
    # publication must carry the closed-owner result into v1 recording and EOF.
    runtime = Runtime(output=io.StringIO())
    runtime.finalization = owner
    runtime.env = SimpleNamespace(closed=True)
    runtime.run_id = "checkpoint-recording"
    runtime.options.project_compact_wire = True
    runtime.options.stars = 30
    runtime.options.project_allow_scoring = transfer
    runtime.options.project_allow_submission = False
    runtime._finalization_authorization = runtime._authorization()
    runtime._publish_finalization()
    emitted = json.loads(runtime.output.getvalue().splitlines()[-1])["payload"]
    assert emitted["score_checkpoint_completed"] is transfer
    assert emitted["reported_score"] == (56.59 if transfer else None)
    trace = rig.root / "checkpoint-replay.jsonl"
    trace.write_text(runtime.output.getvalue())
    playback = Runtime(output=io.StringIO())
    playback.command({"command": "replay", "payload": {"path": str(trace)}})
    while playback.replay_events is not None:
        playback.tick()
    eof = json.loads(playback.output.getvalue().splitlines()[-1])["payload"]
    assert eof["score_checkpoint_completed"] is transfer
    assert eof["reported_score"] == (56.59 if transfer else None)
    assert eof["recorded_status"] == "handoff"
    assert not any(eof[k] for k in ("task_completed", "project_completed", "submitted"))
    if not transfer:
        roots = [
            owner.scoring.output,
            owner.output / "assessment-history",
            rig.root / "project-scoring-claims",
        ]
        _SCORING_TEMPLATE = (
            {
                str(path.relative_to(rig.root)): path.read_bytes()
                for directory in roots
                for path in directory.rglob("*")
                if path.is_file()
            },
            rig.journal.path.read_bytes(),
            deepcopy(owner.scoring.report),
        )
    assert [
        event.model_dump(mode="json") for event in read_trace(owner.output / "events.jsonl")
    ] == rig.post_events
    before = rig.page.clicks[:]
    owner.close()
    owner.step()
    assert rig.page.clicks == before


def test_joined_paged_scoring_to_real_preflight_and_unknown_submission(finished_campaign, monkeypatch):
    rig = finished_campaign
    owner = make(rig, allow_score_transfer=True, allow_submission=True)
    drive(owner, until="submission_initializing")
    assert not owner.finished, owner.report
    state = attach_submission_surface(rig, monkeypatch)
    before = rig.journal.path.read_bytes()
    result = preflight.preflight_project_submission(
        rig.page,
        config(),
        rig.root / "read-only-preflight",
        run_history=rig.root,
        journal=rig.journal,
        scoring_dir=rig.root / owner.state()["scoring_dir"],
        _clock=lambda: rig.clock[0],
    )
    assert result["verified_tasks"] == 30 and result["eligible_for_grounded_dispatch"]
    assert not state.native and rig.journal.path.read_bytes() == before
    drive(owner)
    assert owner.phase == "unknown_pending" and owner.status == "handoff", owner.report
    assert state.native == [preflight.READY, preflight.SUBMIT]
    assert owner.failure == "project_submission_acknowledgement_not_grounded"
    assert owner.report["submission_outcome"] == "unknown"
    current = rig.journal.load().reduce()
    assert len(current.pending) == 1 and current.pending[0].write_kind == "submission"
    assert current.receipt("submission") is None
    assert all(sha(rig.root / path) == value for path, value in rig.before_campaign.items())
    assert not any(owner.report[k] for k in ("task_completed", "project_completed", "submitted"))


@pytest.mark.parametrize("bad", ["outer", "nested", "count", "record", "source", "cross_star"])
def test_handoff_flags_cannot_replace_owned_current_proof(finished_campaign, bad):
    rig = finished_campaign
    path = rig.root / ("report.json" if bad in {"outer", "nested", "count"} else "campaign/verified-030.json")
    value = read(path)
    if bad == "outer":
        value["phase"] = "stopped"
    if bad == "nested":
        value["campaign"]["failure_reason"] = "different"
    if bad == "count":
        value["target_stars"] = 3
    if bad == "record":
        value["ordinal"] = 29
    if bad == "source":
        value["inventory_sha256"] = "f" * 64
    if bad == "cross_star":
        value["star"] = "Other"
    write(path, value)
    with pytest.raises(BrowserSafetyStop):
        make(rig)
    assert not (rig.root / "finalization").exists() and not rig.page.clicks


def test_exclusive_continuation_claim_survives_abort_and_changed_output(finished_campaign):
    rig = finished_campaign
    owner = make(rig)
    owner.abort()
    with pytest.raises(BrowserSafetyStop, match="continuation_already_claimed"):
        make(rig, output="different-output")
    assert owner.claim.is_file() and not rig.page.clicks


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "journal",
        "extra_completed_file",
        "extra_completed_directory",
        "callback_abort",
        "callback_error",
    ],
)
def test_terminal_callback_cannot_certify_mutated_or_aborted_sources(finished_campaign, mutation):
    rig = finished_campaign
    holder = {}

    def emit(event):
        rig.post_events.append(event)
        if event["event"] != "episode_summary":
            return
        owner = holder["owner"]
        if mutation == "source":
            write(rig.root / "campaign/verified-030.json", {"changed": True})
        elif mutation == "journal":
            rig.journal.append(Active(stage="changed"))
        elif mutation == "extra_completed_file":
            write(owner.scoring.output / "extra.json", {})
        elif mutation == "extra_completed_directory":
            (owner.scoring.output / "unclaimed-attempt").mkdir()
        elif mutation == "callback_abort":
            owner.abort()
        elif mutation == "callback_error":
            raise RuntimeError("secret://private")

    owner = make(rig, emit=emit, _scoring_factory=completed_scoring_factory(rig))
    holder["owner"] = owner
    drive(owner)
    assert owner.status in {"stopped", "aborted"}, owner.report
    assert owner.state()["score_checkpoint_completed"] is False
    assert owner.state()["reported_score"] is None
    assert not (owner.output / "report.json").exists()
    assert "private" not in (owner.output / "stopped.json").read_text()
    assert owner.claim.is_file()


def test_pause_does_not_refresh_budget_and_running_start_advances(finished_campaign):
    rig = finished_campaign
    owner = make(rig)
    owner.start(paused=False)
    owner.tick()
    assert owner.phase == "scoring_active" and owner.scoring.advances == 0
    owner.pause()
    owner.tick()
    assert owner.scoring.advances == 0
    rig.clock[0] = 600
    owner.step()
    assert owner.finished and "time_limit" in owner.failure and not rig.page.clicks


@pytest.mark.parametrize("mutation", ["deadline", "malformed_journal", "symlink_journal"])
def test_current_source_and_budget_hardening_never_dispatches(finished_campaign, mutation):
    rig = finished_campaign
    owner = make(rig)
    owner.start(paused=True)
    if mutation == "deadline":
        owner._stage_seconds = 6000
    elif mutation == "malformed_journal":
        rig.journal.path.write_text("malformed canonical input")
    else:
        original = rig.journal.path
        saved = rig.root / "original-canonical.jsonl"
        original.rename(saved)
        original.symlink_to(saved)
    owner.step()
    assert owner.status == "stopped" and owner.scoring is None
    assert not rig.page.clicks and owner.claim.is_file()
    if mutation != "deadline":
        assert owner.state()["project_progress"] is None
        assert not owner.state()["canonical_progress_available"]


def test_callback_abort_before_charged_action_preserves_reservation(finished_campaign):
    rig = finished_campaign
    holder = {}

    def emit(event):
        rig.post_events.append(event)
        if (
            event["event"] == "action_proposed"
            and event["payload"].get("target") == "assessment_data_quality"
        ):
            holder["owner"].abort()

    owner = make(rig, emit=emit)
    holder["owner"] = owner
    drive(owner)
    assert owner.status == "aborted", owner.report
    assert not rig.page.clicks
    assert len(rig.journal.load().reduce().pending) == 1
    assert not owner.state()["project_completed"]


@pytest.mark.parametrize("change", ["bytes", "replacement", "parent", "failed"])
def test_verified_metadata_cache_cannot_hide_changed_sources(tmp_path, change):
    source = tmp_path / "source"
    source.mkdir()
    file = source / "receipt.json"
    file.write_text("original")
    book = module._PinnedEvidence(tmp_path)
    book.clean(source)
    book.read(file)
    book.unchanged()
    if change == "bytes":
        file.write_text("different")
    elif change == "replacement":
        new = tmp_path / "new"
        new.write_text("changed")
        new.replace(file)
    elif change == "parent":
        source.rename(tmp_path / "old")
        source.symlink_to(tmp_path / "old", target_is_directory=True)
    else:
        (source / "stopped.json").write_text("{}")
    with pytest.raises(BrowserSafetyStop):
        book.unchanged()


def test_completed_tree_rejects_separate_callback_failure_disposition(tmp_path):
    owner = object.__new__(module.BrowserProjectFinalizeSteps)
    owner.history, owner.book, owner._completed_trees = tmp_path, module._PinnedEvidence(tmp_path), {}
    directory = tmp_path / "completed"
    directory.mkdir()
    write(directory / "event-forwarding-failed.json", {"automatic_retry": False})
    with pytest.raises(BrowserSafetyStop, match="failed_completed_tree"):
        owner._pin_tree(directory)


@pytest.mark.parametrize(
    "kwargs", [{"allow_score_transfer": 1}, {"allow_submission": "yes"}, {"allow_submission": True}]
)
def test_flags_are_strict_and_submission_requires_transfer(finished_campaign, kwargs):
    with pytest.raises(BrowserSafetyStop):
        make(finished_campaign, **kwargs)
    assert not (finished_campaign.root / "project-finalization-claims").exists()
