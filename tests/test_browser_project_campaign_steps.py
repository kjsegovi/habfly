"""Offline composition with real journal/import validation and injected UI work.

The upstream source seam supplies synthetic, prevalidated browser bundles. No
browser, inference, credentials, network, grading oracle, or training is used.
"""

import socket
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_project_next_star_steps import fresh_capture, fresh_receipt
from test_project_evidence import inventory, read, sha, write
from test_project_evidence import workflow as no_workflow
from test_project_positive_evidence import workflow as positive_workflow
from test_project_terrestrial_evidence import workflow as terrestrial_workflow

import habfly.browser_project_campaign_steps as module
import habfly.project_evidence as no_evidence
import habfly.project_positive_evidence as positive_evidence
import habfly.project_terrestrial_evidence as terrestrial_evidence
from habfly.browser import BrowserSafetyStop
from habfly.browser_numeric import screen_identity
from habfly.browser_probe import BrowserProbeConfig, ProbeFrame
from habfly.browser_project_next_star_steps import MODE as TRANSITION_MODE
from habfly.browser_stellar import SIMULATION_URL
from habfly.contracts import RuntimeEvent
from habfly.project_progress import Collected, ProjectJournal


def replace_probe(capture, directory):
    """Amend only synthetic fixture text and its explicit hash, never live artifacts."""
    write(directory / "observation.json", capture)
    manifest = read(directory / "manifest.json")
    manifest["observation_sha256"] = sha(directory / "observation.json")
    write(directory / "manifest.json", manifest)
    return manifest


def fresh(directory, star, *, initial=False, names=(), excluded=()):
    entry, receipt = fresh_receipt(directory, star, initial=initial, names=names, excluded=excluded)
    capture = fresh_capture(star)
    capture["frames"][0]["text"] = f"{star}\nObservations Your Reconstruction parallax"
    replace_probe(capture, directory / "stellar")
    if initial:
        replace_probe(capture, directory / "before")
        scope = read(directory / "scope.json")
        scope.update(
            selection_schema_version=1,
            starfield_image="setup-starfield.png",
            starfield_anchor=[0.4, 0.55],
            excluded_points=[],
            ready_screen_sha256=screen_identity(capture),
        )
        write(directory / "scope.json", scope)
    else:
        before = {
            "schema_version": 1,
            "ignored_frame_urls": [],
            "frames": [
                {
                    "url": SIMULATION_URL,
                    "text": "Funding Data Quality Scavenger Hunt",
                    "accessibility": "- text: Funding Data Quality Scavenger Hunt",
                    "controls": [],
                }
            ],
        }
        replace_probe(before, directory / "before")
    return entry, receipt


def full_inventory(history, names, suffix):
    directory = inventory(history, names, suffix)
    receipt = read(directory / "confirmed.json")
    for part in ("before", "after"):
        capture = read(directory / part / "observation.json")
        capture["frames"][0]["text"] = "Funding Total Collected Analyzed Data"
        saved = replace_probe(capture, directory / part)
        checksum = saved["observation_sha256"]
        receipt["source_sha256"][part + "/observation.json"] = checksum
        if part == "after":
            for row in receipt["rows"]:
                row["source_sha256"] = checksum
    write(directory / "confirmed.json", receipt)
    return directory


@pytest.fixture
def rig(tmp_path, monkeypatch):
    registry = {}
    flags = SimpleNamespace(
        calls=[],
        now=0,
        cancel=False,
        branches=["no", "positive", "terrestrial"],
        owner_hook=None,
        transition_hook=None,
        fail=None,
        owners=[],
        transitions=[],
    )

    def validated(book, **directories):
        hashes, bundle = registry[str(directories["numeric_dir"])]
        for name in hashes:
            book.clean((book.history / name).parent)
            book.read(book.history / name)
        return deepcopy(bundle)

    for helper in (no_evidence, positive_evidence, terrestrial_evidence):
        monkeypatch.setattr(helper, "_load_sources", validated)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network forbidden"))
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="fresh-synthetic").create()
    first, _ = fresh(tmp_path / "initial", "Althinagon", initial=True)
    source = tmp_path / "frozen-checkpoint"
    source.write_bytes(b"injected model identity, never loaded")
    monkeypatch.setattr(module, "_model_dependencies", lambda *a: {str(source): sha(source)})
    config = BrowserProbeConfig(
        url="http://localhost/owned-preview",
        frames=[ProbeFrame(name="simulation", url=SIMULATION_URL, required_text=["old context"])],
    )
    page = SimpleNamespace(closed=False)
    models = {key: tmp_path / key for key in module._REQUIRED_MODELS}

    def source_class(history, directory, star, selected):
        context = campaign.current_star_context()
        return {
            "fresh_star": context["fresh_dir"],
            "source_capture_sha256": sha(history / context["fresh_dir"] / "stellar/observation.json"),
        }

    monkeypatch.setattr(module, "validate_star_class_source", source_class)

    class Owner:
        def __init__(self, page, config, output, **kwargs):
            assert kwargs["max_seconds"] == 1800 and kwargs["max_advances"] == 512
            assert kwargs["automatic_inventory"] is True
            assert config.frames[0].required_text == ["Observations", "Reconstruction", "parallax"]
            flags.calls.append("owner.init")
            self.index, self.output, self.star = len(flags.owners), output, kwargs["star"]
            self.branch = flags.branches[self.index]
            self.emit, self.events, self.report = kwargs["emit"], [], None
            self.phase, self.status, self.finished = "not_started", "idle", False
            self.failure = None
            output.mkdir(parents=True)
            flags.owners.append(self)

        def state(self):
            return (
                deepcopy(self.report)
                if self.report
                else {
                    "mode": "bounded_single_star_project_runtime",
                    "star": self.star,
                    "phase": self.phase,
                    "status": self.status,
                    "finished": self.finished,
                    "task_completed": False,
                    "project_completed": False,
                    "failure_reason": self.failure,
                    "event_forwarding_failed": False,
                }
            )

        def event(self, kind, payload):
            event = RuntimeEvent(
                event=kind, sequence=len(self.events), run_id=f"owner-{self.index}", payload=payload
            )
            self.events.append(event)
            (self.output / "events.jsonl").write_text(
                "\n".join(e.model_dump_json() for e in self.events) + "\n"
            )
            self.emit(kind, payload)

        def start(self, *, paused):
            assert paused is True
            self.status, self.phase = "paused", "awaiting_class_source"
            self.event("state", self.state())

        def provide_class(self, **kwargs):
            flags.calls.append("owner.class")
            self.phase = "ready"
            self.event("state", self.state())

        def provide_planet_class(self, **kwargs):
            flags.calls.append("owner.planet_class")
            self.phase = "awaiting_gases" if self.branch == "terrestrial" else "inventory_import"
            self.event("state", self.state())

        def provide_gases(self, **kwargs):
            flags.calls.append("owner.gases")
            self.phase = "awaiting_habitability"
            self.event("state", self.state())

        def provide_habitability(self, **kwargs):
            flags.calls.append("owner.habitability")
            self.phase = "inventory_import"
            self.event("state", self.state())

        def step(self):
            flags.calls.append("owner.step")
            if flags.owner_hook:
                flags.owner_hook(self)
            if self.finished:
                return
            if flags.fail:
                self.status, self.phase, self.failure, self.finished = "stopped", "stopped", flags.fail, True
            elif self.phase == "ready":
                self.phase = "inventory_import" if self.branch == "no" else "awaiting_planet_class"
            elif self.phase == "inventory_import":
                names = [s.name for s in journal.load().reduce().stars.values()] + [self.star]
                inv = full_inventory(tmp_path, names, str(self.index))
                creator, importer, phase = {
                    "no": (no_workflow, no_evidence.import_verified_no_planet, "verified_no_planet"),
                    "positive": (
                        positive_workflow,
                        positive_evidence.import_verified_positive_planet,
                        "verified_positive_planet",
                    ),
                    "terrestrial": (
                        terrestrial_workflow,
                        terrestrial_evidence.import_verified_terrestrial,
                        "verified_terrestrial",
                    ),
                }[self.branch]
                work = creator(tmp_path, self.star, str(self.index), registry)
                result = importer(journal, tmp_path, inv, work)
                write(self.output / "workflow-import.json", result)
                self.status, self.phase, self.finished = "completed", phase, True
                self.report = {**self.state(), "task_completed": True, "project_progress": result["progress"]}
                write(self.output / "report.json", self.report)
                self.event("episode_summary", {**self.report, "completed": True})
                return
            self.event("state", self.state())

        tick = step

        def resume(self):
            self.status = "running"

        def pause(self):
            self.status = "paused"

        def abort(self):
            flags.calls.append("owner.abort")
            self.status, self.phase, self.finished = "aborted", "aborted", True
            self.event("state", self.state())

    class Transition:
        def __init__(self, page, config, output, **kwargs):
            flags.calls.append("transition.init")
            assert kwargs["max_seconds"] == 180 and kwargs["max_advances"] == 128
            assert config.frames[0].required_text == ["Funding", "Total Collected", "Analyzed Data"]
            self.output, self.options, self.finished, self.report = output, kwargs, False, None
            output.mkdir(parents=True)
            self.value = {"phase": "to_starfield", "status": "ready", "task_completed": False}
            flags.transitions.append(self)

        def state(self):
            return deepcopy(self.value)

        def advance(self):
            flags.calls.append("transition.advance")
            if flags.transition_hook:
                flags.transition_hook(self)
            if self.finished:
                return
            names = self.options["expected_stars"]
            points = [read(Path(e["path"]))["selected_point"] for e in self.options["visited_receipts"]]
            entry, receipt = fresh(
                self.output / "picker", ["Beta", "Gamma"][len(names) - 1], names=names, excluded=points
            )
            self.value = {
                "mode": TRANSITION_MODE,
                "status": "completed",
                "phase": "fresh_star_handoff",
                "previous_star": names[-1],
                "expected_stars": names,
                "project_progress": journal.load().reduce().report(),
                "max_seconds": 180,
                "max_advances": 128,
                "failure_reason": None,
                "event_forwarding_failed": False,
                "fresh_star_dir": str((self.output / "picker").relative_to(tmp_path)),
                "fresh_star_sha256": entry["sha256"],
                "star": receipt["star"],
                "source_sha256": {journal.path.name: sha(journal.path)},
                "task_completed": False,
                "project_completed": False,
                "class_selection_verified": False,
                "collection_count_verified": False,
            }
            self.finished, self.report = True, deepcopy(self.value)
            write(self.output / "report.json", self.report)
            write(self.output / "confirmed.json", self.report)
            self.options["emit"]("episode_summary", self.report)

        def abort(self):
            flags.calls.append("transition.abort")
            self.finished, self.value = True, {"status": "aborted", "phase": "aborted"}

    kwargs = {
        "run_history": tmp_path,
        "journal": journal,
        "initial_star_dir": tmp_path / "initial",
        "initial_star_sha256": first["sha256"],
        "target_stars": 3,
        "max_seconds": 3600,
        "model_options": models,
        "reference_planet_continuation": True,
        "_owner_factory": Owner,
        "_transition_factory": Transition,
        "_clock": lambda: flags.now,
        "cancelled": lambda: flags.cancel,
    }
    campaign = module.BrowserProjectCampaignSteps(page, config, tmp_path / "campaign", **kwargs)
    yield SimpleNamespace(
        c=campaign,
        flags=flags,
        root=tmp_path,
        journal=journal,
        page=page,
        config=config,
        models=models,
        kwargs=kwargs,
        source=source,
    )
    campaign.close()


def drive(rig, *, until=None):
    c = rig.c
    if c.status == "idle":
        c.start(paused=True)
    for _ in range(70):
        if c.finished or c.phase == until:
            return c.state()
        phase = c.phase
        if phase == "awaiting_class_source":
            c.provide_class(class_dir="explicit-class", selected_class="main_sequence", lifetime_prefix="Ga")
        elif phase == "awaiting_planet_class":
            c.provide_planet_class(
                name="terrestrial" if c.owner.branch == "terrestrial" else "gas_giant",
                rationale="supplied reference",
            )
        elif phase == "awaiting_gases":
            c.provide_gases(
                gases=["CO2"], rationale="supplied spectral comparison", supplied_greenhouse_increment=10
            )
        elif phase == "awaiting_habitability":
            c.provide_habitability(choice="not_habitable", rationale="supplied chamber readback")
        else:
            before = len(rig.flags.calls)
            c.step()
            assert len(rig.flags.calls) - before <= 1
    pytest.fail("unbounded scheduler")


def test_all_three_branches_strict_import_and_assessment_handoff(rig):
    result = drive(rig)
    assert result["status"] == "handoff", result
    assert result["phase"] == "awaiting_assessment"
    assert result["verified_stars"] == 3 and result["target_workflows_verified"] is True
    assert result["task_completed"] is result["project_completed"] is False
    state = rig.journal.load().reduce()
    assert all(s.task_completed for s in state.stars.values())
    assert not state.reservations and not state.receipts and not state.pending
    assert [r["star"] for r in rig.c._completed] == ["Althinagon", "Beta", "Gamma"]
    assert rig.config.frames[0].required_text == ["old context"]
    assert rig.c.config.frames[0].required_text == ["Funding", "Total Collected", "Analyzed Data"]
    assert not rig.page.closed
    assert rig.journal.path.name not in rig.c.book.hashes
    events = [
        RuntimeEvent.model_validate_json(line)
        for line in (rig.c.output / "events.jsonl").read_text().splitlines()
    ]
    assert [e.sequence for e in events] == list(range(len(events)))
    assert all(not e.payload.get("task_completed") for e in events)
    nested = [e.payload for e in events if "component_summary" in e.payload]
    assert any(e["component_summary"].get("completed") is True for e in nested)
    assert read(rig.c.output / "report.json") == result


def test_constructor_is_offline_and_context_detached(rig):
    assert not rig.flags.calls
    assert not rig.journal.load().records
    context = rig.c.current_star_context()
    context["star"] = "Other"
    assert rig.c.current_star_context()["star"] == "Althinagon"
    assert rig.c.state()["thirty_star_live_launch_ready"] is False


@pytest.mark.parametrize("target", [True, 1, 31, 2.0, "3"])
def test_rejects_invalid_multistar_targets(rig, target):
    with pytest.raises(BrowserSafetyStop, match="explicit_multistar_target"):
        module.BrowserProjectCampaignSteps(
            rig.page, rig.config, rig.root / "bad", **(rig.kwargs | {"target_stars": target})
        )


@pytest.mark.parametrize("budget", [True, 0, -1, 10801, float("nan"), float("inf")])
def test_rejects_invalid_fixed_deadline(rig, budget):
    with pytest.raises(BrowserSafetyStop, match="deadline"):
        module.BrowserProjectCampaignSteps(
            rig.page, rig.config, rig.root / "bad", **(rig.kwargs | {"max_seconds": budget})
        )


def test_nonempty_attempt_cannot_be_retrofitted(rig):
    rig.journal.append(Collected(star_id="old", name="Old", source_sha256="a" * 64))
    with pytest.raises(BrowserSafetyStop, match="fresh_empty_attempt"):
        module.BrowserProjectCampaignSteps(rig.page, rig.config, rig.root / "bad", **rig.kwargs)


def test_explicit_handoffs_cannot_be_skipped(rig):
    c = rig.c
    c.start(paused=True)
    c.step()
    before = list(rig.flags.calls)
    assert c.step()["phase"] == "awaiting_class_source"
    with pytest.raises(BrowserSafetyStop, match="explicit_handoff"):
        c.resume()
    assert rig.flags.calls == before


@pytest.mark.parametrize("mutation", ["checkpoint", "scope", "config", "journal", "current", "settings"])
def test_frozen_input_mutation_stops_before_next_child(rig, mutation):
    rig.c.start(paused=True)
    if mutation == "checkpoint":
        rig.source.write_bytes(b"changed")
    elif mutation == "scope":
        write(rig.c.output / "scope.json", {})
    elif mutation == "config":
        rig.c.config.frames[0].url = "http://localhost/elsewhere"
    elif mutation == "journal":
        rig.journal.append(Collected(star_id="extra", name="Extra", source_sha256="a" * 64))
    elif mutation == "current":
        rig.c._current["star"] = "Other"
    else:
        rig.c.max_seconds += 1
    state = rig.c.step()
    assert state["status"] == "stopped"
    assert not rig.flags.calls


@pytest.mark.parametrize(
    "phase",
    [
        "initializing_star",
        "awaiting_class_source",
        "inventory_import",
        "next_star_active",
        "adopting_fresh_star",
    ],
)
def test_abort_sticky_no_hidden_work(rig, phase):
    drive(rig, until=phase)
    result = rig.c.abort()
    assert result["status"] == "aborted" and result["failure_reason"] == "operator_aborted"
    before = list(rig.flags.calls)
    for fn in (rig.c.step, rig.c.tick, rig.c.advance_if_due, rig.c.close):
        assert fn()["status"] == "aborted"
    assert before == rig.flags.calls and not rig.page.closed


def test_fixed_deadline_includes_paused_handoff(rig):
    drive(rig, until="awaiting_class_source")
    rig.flags.now = 3600
    rig.c.provide_class(class_dir="explicit", selected_class="main_sequence", lifetime_prefix="Ga")
    assert rig.c.status == "stopped" and "time_limit" in rig.c.failure
    assert "owner.class" not in rig.flags.calls


def test_partial_inventory_explicitly_blocks_campaign(rig):
    rig.flags.fail = "project_inventory_unsupported_pagination"
    result = drive(rig)
    assert result["phase"] == "pagination_required" and result["status"] == "handoff"
    assert not result["target_workflows_verified"] and not rig.flags.transitions


def test_final_callback_mutation_cannot_claim_target(rig):
    def emit(kind, payload):
        if kind == "episode_summary" and payload["phase"] == "awaiting_assessment":
            rig.source.write_bytes(b"changed at final callback")

    rig.c._callback = emit
    result = drive(rig)
    assert result["status"] == "stopped" and not result["target_workflows_verified"]


def test_callback_abort_stops_before_later_native_stage(rig):
    def emit(kind, payload):
        if payload.get("component") == "campaign.owner":
            rig.c.abort()

    rig.c._callback = emit
    rig.c.start(paused=True)
    rig.c.step()
    assert rig.c.status == "aborted"
    assert rig.flags.calls == ["owner.init", "owner.abort"]


def test_child_error_does_not_become_parent_success(rig):
    drive(rig, until="ready")
    rig.flags.owner_hook = lambda child: child.event("error", {"message": "injected error"})
    result = rig.c.step()
    assert result["status"] == "stopped" and not result["target_workflows_verified"]


@pytest.mark.parametrize("later", ["terminal_events", "second_error", "action_proposed"])
def test_child_error_retains_first_cause_without_fabricated_callback_failure(rig, later):
    drive(rig, until="ready")
    cause = "no_planet_save_stale_acknowledgement"
    journal_before = rig.journal.path.read_bytes()

    def stop(child):
        child.event("error", {"type": "BrowserStarStop", "message": cause})
        if later == "action_proposed":
            child.event("action_proposed", {"kind": "CLICK", "target": "forbidden-after-stop"})
            pytest.fail("No native dispatch may follow a child error")
        if later == "second_error":
            child.event("error", {"message": "project_steps_component_unverified"})
        child.status, child.phase, child.finished = "stopped", "stopped", True
        child.event("state", child.state())
        child.event("episode_summary", child.state())

    rig.flags.owner_hook = stop
    result = rig.c.step()
    assert result["failure_reason"] == cause and not result["event_forwarding_failed"]
    assert result["status"] == "stopped" and not result["target_workflows_verified"]
    before = list(rig.flags.calls)
    rig.c.step()
    rig.c.tick()
    assert rig.flags.calls == before and rig.journal.path.read_bytes() == journal_before
    assert not rig.flags.transitions


def test_actual_callback_exception_on_child_error_remains_forwarding_failure(rig):
    drive(rig, until="ready")

    def fail(kind, payload):
        if kind == "error":
            raise RuntimeError("private callback details")

    rig.c._callback = fail
    rig.flags.owner_hook = lambda child: child.event(
        "error", {"message": "no_planet_save_stale_acknowledgement"}
    )
    result = rig.c.step()
    assert result["event_forwarding_failed"]
    assert result["failure_reason"] == "project_campaign_event_forwarding_failed"
    assert "private callback details" not in (rig.c.output / "events.jsonl").read_text()


def test_child_error_reason_is_sanitized_before_becoming_terminal_reason(rig):
    drive(rig, until="ready")
    rig.flags.owner_hook = lambda child: child.event("error", {"message": "bad input with spaces"})
    result = rig.c.step()
    assert result["failure_reason"] == "project_campaign_operation_failed"
    assert not result["event_forwarding_failed"]


def test_changed_completed_source_prevents_next_star(rig):
    drive(rig, until="next_star_initializing")
    write(rig.root / "workflow-0/confirmed.json", {})
    result = rig.c.step()
    assert result["status"] == "stopped" and not rig.flags.transitions


def test_forged_transition_scope_not_adopted(rig):
    drive(rig, until="adopting_fresh_star")
    child = rig.c.transition
    child.value["expected_stars"] = ["Foreign"]
    child.report = deepcopy(child.value)
    write(child.output / "report.json", child.value)
    write(child.output / "confirmed.json", child.value)
    result = rig.c.step()
    assert result["status"] == "stopped" and result["star"] == "Althinagon"


def test_current_capture_class_binding_is_required(rig, monkeypatch):
    drive(rig, until="awaiting_class_source")
    monkeypatch.setattr(
        module,
        "validate_star_class_source",
        lambda *a: {
            "fresh_star": "another-star",
            "source_capture_sha256": "a" * 64,
        },
    )
    result = rig.c.provide_class(class_dir="foreign", selected_class="main_sequence", lifetime_prefix="Ga")
    assert result["status"] == "stopped" and "owner.class" not in rig.flags.calls


def test_running_mode_still_pauses_for_each_reference(rig):
    c = rig.c
    c.start(paused=False)
    c.tick()
    assert c.status == "paused" and c.phase == "awaiting_class_source"
    c.provide_class(class_dir="explicit", selected_class="main_sequence", lifetime_prefix="Ga")
    c.resume()
    c.tick()
    assert c.phase == "inventory_import"
    c.pause()
    before = list(rig.flags.calls)
    c.tick()
    assert before == rig.flags.calls
    c.step()
    c.step()
    c.resume()
    for _ in range(6):
        c.tick()
        if c.phase == "awaiting_class_source":
            break
    assert c.status == "paused" and c.current_star_context()["ordinal"] == 2


def test_callback_failures_and_unknown_events_fail_closed(rig):
    drive(rig, until="ready")
    rig.flags.owner_hook = lambda child: child.event("unknown_event", {})
    result = rig.c.step()
    assert result["status"] == "stopped" and not result["target_workflows_verified"]


def test_final_callback_abort_remains_abort(rig):
    rig.c._callback = lambda kind, payload: rig.c.abort() if kind == "episode_summary" else None
    result = drive(rig)
    assert result["status"] == "aborted" and not result["target_workflows_verified"]


def test_explicit_cancel_callback_prevents_next_child(rig):
    rig.c.start(paused=True)
    rig.flags.cancel = True
    result = rig.c.step()
    assert result["status"] == "aborted" and not rig.flags.calls


def test_target_thirty_does_not_claim_remaining_live_acceptance(rig):
    c = module.BrowserProjectCampaignSteps(
        rig.page, rig.config, rig.root / "thirty", **(rig.kwargs | {"target_stars": 30, "max_seconds": 10800})
    )
    try:
        assert c.state()["target_stars"] == 30
        assert (
            c.state()["pending_thirty_star_gate"] == "live_terrestrial_workflow_and_compact_event_acceptance"
        )
        assert c.state()["pagination_supported"]
        assert not c.state()["thirty_star_live_launch_ready"]
        assert not c.state()["target_workflows_verified"]
    finally:
        c.close()


def test_legacy_initial_receipt_cannot_seed_fresh_campaign(rig):
    scope = read(rig.root / "initial/scope.json")
    for key in ("selection_schema_version", "starfield_image", "starfield_anchor", "excluded_points"):
        del scope[key]
    write(rig.root / "initial/scope.json", scope)
    with pytest.raises(BrowserSafetyStop, match="initial_starfield_source_required"):
        module.BrowserProjectCampaignSteps(rig.page, rig.config, rig.root / "legacy", **rig.kwargs)


def test_ambiguous_canonical_attempt_rejected(rig):
    ProjectJournal(rig.root, project_id="habworlds", attempt_id="other").create()
    with pytest.raises(BrowserSafetyStop, match="ambiguous_attempt"):
        module.BrowserProjectCampaignSteps(rig.page, rig.config, rig.root / "other", **rig.kwargs)


def test_prior_complete_task_cannot_be_revised_during_import(rig):
    drive(rig, until="next_star_initializing")
    # Even append-only collection outside the sole current owner import stops.
    rig.journal.append(Collected(star_id="foreign", name="Foreign", source_sha256="f" * 64))
    result = rig.c.step()
    assert result["status"] == "stopped" and not rig.flags.transitions


def test_owner_success_flag_without_persisted_chain_is_rejected(rig):
    drive(rig, until="ready")
    child = rig.c.owner
    child.finished = True
    child.report = {**child.state(), "status": "completed", "task_completed": True}
    result = rig.c.step()
    if result["phase"] == "verifying_star":
        result = rig.c.step()
    assert result["status"] == "stopped" and not rig.flags.transitions


@pytest.fixture
def model_files(tmp_path):
    options = {key: tmp_path / key for key in module._REQUIRED_MODELS}
    options.update(planet_pilot=tmp_path / "planet", planet_final_evaluation=tmp_path / "planet-final")
    terrestrial = {"pilot": tmp_path / "temperature", "final_evaluation": tmp_path / "temperature-final"}
    files = {
        options["checkpoint"],
        options["checkpoint"].with_suffix(".json"),
        *(
            options["dataset"] / name
            for name in ("manifest.json", "report.json", "final/report.json", "manual.json")
        ),
        *(
            options["color_experiment"] / name
            for name in (
                "report.json",
                "dataset-manifest.json",
                "training/checkpoint.pt",
                "training/checkpoint.pt.json",
                "final/report.json",
                "train.json",
                "calibration.json",
                "development.json",
                "demonstrations.json",
                "regression.json",
                "epochs/epoch-000/training/checkpoint.pt",
                "epochs/epoch-001/training/checkpoint.pt",
            )
        ),
        options["graph_path"] / "graph_manifest.json",
        options["graph_path"] / "metadata.json",
        *(options["graph_path"] / (name + ".npy") for name in module.ARRAY_NAMES),
        *(
            options["planet_pilot"] / name
            for name in ("training/checkpoint.pt", "training/checkpoint.pt.json")
        ),
        options["planet_final_evaluation"] / "report.json",
        *(
            terrestrial["pilot"] / name
            for name in ("report.json", "training/checkpoint.pt", "training/checkpoint.pt.json")
        ),
        terrestrial["final_evaluation"] / "report.json",
        *(terrestrial["pilot"] / f"private-{split}-cases.json" for split in module.PILOT_COUNTS),
        *(
            terrestrial["pilot"] / f"expert-{split}/episodes.json"
            for split in module.PILOT_COUNTS
            if split != "gate"
        ),
        *(terrestrial["pilot"] / f"learned-{split}/summaries.json" for split in ("train", "development")),
    }
    for path in files:
        write(path, {"synthetic": True})
    write(
        options["color_experiment"] / "report.json",
        {
            "content": {"readout_ordering": {"version": 1}},
            "epoch_history": [{"epoch": 0}, {"epoch": 1}],
        },
    )
    write(
        options["graph_path"] / "graph_manifest.json",
        {
            "num_nodes": 2000,
            "arrays": {name: {"file": name + ".npy"} for name in module.ARRAY_NAMES},
        },
    )
    return options, terrestrial, files


def test_exact_loader_dependency_allowlist_ignores_unrelated_trajectories(model_files):
    options, terrestrial, files = model_files
    unused = options["color_experiment"] / "epochs/epoch-000/development/trajectories.jsonl"
    unused.parent.mkdir(parents=True)
    unused.write_text("not an inference dependency")
    forbidden = terrestrial["final_evaluation"] / "private-test-cases.json"
    forbidden.write_text("sealed and not read")
    actual = module._model_dependencies(options, terrestrial)
    sources = {
        module.DEFAULT_PACK,
        module.DEFAULT_PLANET_PACK,
        module.DEFAULT_HABITABILITY_PACK,
        Path(module.__file__).parent / "packs/stellar_color.json",
        *(Path(module.__file__).parent / "model").glob("*.py"),
    }
    assert set(actual) == {str(path.absolute()) for path in files | sources}
    unused.write_text("unrelated change")
    assert module._model_dependencies(options, terrestrial) == actual
    options["checkpoint"].write_bytes(b"changed inference input")
    assert module._model_dependencies(options, terrestrial) != actual


@pytest.mark.parametrize("kind", ["array_escape", "missing_array", "nonlinear_history", "symlink"])
def test_unknown_or_unowned_model_dependencies_rejected(model_files, kind):
    options, terrestrial, _ = model_files
    if kind == "symlink":
        path = options["checkpoint"]
        other = path.with_name("other-checkpoint")
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    elif kind == "nonlinear_history":
        path = options["color_experiment"] / "report.json"
        report = read(path)
        report["epoch_history"] = [{"epoch": 10}]
        write(path, report)
    else:
        path = options["graph_path"] / "graph_manifest.json"
        manifest = read(path)
        name = next(iter(manifest["arrays"]))
        if kind == "array_escape":
            manifest["arrays"][name]["file"] = "../other.npy"
        else:
            del manifest["arrays"][name]
        write(path, manifest)
    with pytest.raises(BrowserSafetyStop):
        module._model_dependencies(options, terrestrial)
