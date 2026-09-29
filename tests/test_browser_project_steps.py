"""Pure injected components: no browser, network, checkpoints or training."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_star_session import rig as star_rig  # noqa: F401
from test_project_progress import complete_star

import habfly.browser_project_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_probe import BrowserProbeConfig
from habfly.contracts import RuntimeEvent
from habfly.project_progress import Collected, ProjectJournal, WriteReserved
from habfly.runtime import read_trace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def rig(tmp_path, monkeypatch):
    journal = ProjectJournal(tmp_path, project_id="fixture", attempt_id="preview").create()
    calls, events, children = [], [], []
    clock = SimpleNamespace(now=0.0)
    outcome = {"phase": "planet_classification_required", "task_completed": False, "steps": 2}
    source = {"star": "ALPHA", "selected_class": "main_sequence", "source_hashes": {"class": "0" * 64}}
    write(tmp_path / "class/confirmed.json", {"star": "ALPHA"})

    def preflight(history, path, star, selected):
        calls.append(("preflight", selected))
        return deepcopy(source)

    monkeypatch.setattr(module, "validate_star_class_source", preflight)

    class NoBrowser:
        def __getattr__(self, name):
            raise AssertionError("Owner must not call browser APIs: " + name)

    class Component:
        def __init__(self, page, config, output, *, emit, **options):
            self.output, self.emit, self.options = output, emit, options
            self.finished = self.aborted = False
            self.steps, self.sequence = 0, 0
            self.phase = "stellar_numeric"
            self.output.mkdir()
            children.append(self)
            calls.append(("construct", options["star"]))
            self.send("hello", {"protocol_version": 1})

        def send(self, kind, payload):
            item = RuntimeEvent(
                event=kind, sequence=self.sequence, run_id="child", payload=payload
            ).model_dump(mode="json")
            self.sequence += 1
            self.emit(item)
            return item

        def state(self):
            return {
                "star": "ALPHA",
                "phase": self.phase,
                "task_completed": self.finished and outcome["task_completed"] and not self.aborted,
                "artifact_paths": {"workflow_dir": str((self.output / "workflow").relative_to(tmp_path))},
                "reference_measurements": {"fixture": "reference-only"},
            }

        def advance(self):
            calls.append(("advance", self.steps))
            self.send("action_proposed", {"fixture": self.steps})
            # The real BrowserStarSession and its child adapters honor this
            # boundary after the callback, before any later native write.
            if self.aborted:
                return
            calls.append(("dispatch", self.steps))
            self.send("action_result", {"fixture": self.steps})
            self.steps += 1
            if self.steps == outcome["steps"]:
                self.finished, self.phase = True, outcome["phase"]
                if self.phase == "verified_no_planet":
                    write(
                        self.output / "workflow/confirmed.json",
                        {
                            "schema_version": 1,
                            "mode": "no_planet_visible_workflow_readback",
                            "authority": "visible_workflow_readback",
                            "star": "ALPHA",
                            "task_completed": True,
                        },
                    )
                self.send("episode_summary", {"completed": outcome["task_completed"]})

        def abort(self):
            calls.append(("child_abort", None))
            self.aborted = self.finished = True

    return SimpleNamespace(
        root=tmp_path,
        journal=journal,
        calls=calls,
        events=events,
        children=children,
        clock=clock,
        outcome=outcome,
        source=source,
        factory=Component,
        page=NoBrowser(),
        config=BrowserProbeConfig(
            url="http://localhost/activity",
            frames=[
                {
                    "name": "simulation",
                    "url": "https://fixture.invalid/simulation",
                }
            ],
        ),
    )


def create(rig, **kwargs):
    return module.BrowserProjectSteps(
        rig.page,
        rig.config,
        rig.root / kwargs.pop("output", "runtime"),
        run_history=rig.root,
        journal=rig.journal,
        star="ALPHA",
        model_options={key: key for key in ("dataset", "checkpoint", "color_experiment", "graph_path")},
        component_factory=kwargs.pop("component_factory", rig.factory),
        emit=kwargs.pop("emit", lambda *event: rig.events.append(event)),
        _clock=lambda: rig.clock.now,
        **kwargs,
    )


def ready(rig, **kwargs):
    owner = create(rig, **kwargs)
    owner.start()
    owner.provide_class(class_dir=rig.root / "class", selected_class="main_sequence", lifetime_prefix="Ga")
    return owner


def finish(owner):
    for _ in range(8):
        owner.step()
        if owner.finished or owner.phase == "awaiting_inventory":
            return
    raise AssertionError("Fixture did not finish")


def test_constructor_and_class_handoff_never_construct_or_touch_browser(rig):
    owner = create(rig)
    assert owner.status == "idle" and not rig.calls and not rig.events
    before = rig.journal.path.read_bytes()
    owner.start(paused=False)
    assert owner.phase == "awaiting_class_source" and owner.status == "paused"
    with pytest.raises(BrowserSafetyStop, match="handoff"):
        owner.resume()
    with pytest.raises(BrowserSafetyStop, match="handoff"):
        owner.step()
    owner.provide_class(class_dir=rig.root / "class", selected_class="main_sequence", lifetime_prefix="Ga")
    assert owner.phase == "ready" and not rig.children and owner.advances == 0
    assert rig.journal.path.read_bytes() == before
    assert json.loads((owner.output / "class-handoff.json").read_text())["browser_writes"] == 0
    owner.close()


def test_single_step_constructs_then_exactly_one_decision_and_pauses(rig):
    owner = ready(rig)
    owner.step()
    assert len(rig.children) == 1 and not any(c[0] == "advance" for c in rig.calls)
    owner.step()
    assert rig.children[0].steps == 1 and owner.status == "paused"
    owner.advance_if_due()
    owner.tick()
    assert rig.children[0].steps == 1
    owner.close()


def test_interval_pause_resume_no_catch_up_burst(rig):
    owner = ready(rig, interval=2)
    owner.resume()
    owner.advance_if_due()
    owner.advance_if_due()
    assert rig.children[0].steps == 0
    rig.clock.now = 1
    owner.advance_if_due()
    owner.pause()
    rig.clock.now = 50
    owner.advance_if_due()
    assert rig.children[0].steps == 0
    owner.resume()
    owner.advance_if_due()
    assert rig.children[0].steps == 1
    owner.advance_if_due()
    assert rig.children[0].steps == 1
    owner.close()


@pytest.mark.parametrize("phase", sorted(module._HANDOFFS))
def test_positive_and_unsupported_handoffs_never_mean_completion(rig, phase):
    rig.outcome["phase"] = phase
    owner = ready(rig)
    finish(owner)
    assert owner.status == "handoff" and owner.phase == phase
    assert not owner.report["task_completed"] and not owner.report["project_completed"]
    assert owner.report["artifact_paths"] and owner.report["star_component"]["phase"] == phase
    old = list(rig.calls)
    owner.step()
    owner.tick()
    owner.advance_if_due()
    owner.abort()
    owner.close()
    assert rig.calls == old
    with pytest.raises(BrowserSafetyStop):
        owner.start()
    assert rig.journal.load().reduce().report()["verified"] == 0


def test_exact_v1_envelopes_and_child_summaries_replay_without_outer_completion(rig):
    owner = ready(rig)
    finish(owner)
    events = read_trace(owner.output / "events.jsonl")
    assert [e.sequence for e in events] == list(range(len(events)))
    assert len({e.run_id for e in events}) == 1
    assert [(e.event, e.payload) for e in events] == rig.events
    nested = [e.payload["component_event"] for e in events if "component_event" in e.payload]
    assert [RuntimeEvent.model_validate(e).sequence for e in nested] == list(range(6))
    assert nested[-1]["event"] == "episode_summary"
    assert sum(e.event == "episode_summary" for e in events) == 1
    assert events[-1].payload["completed"] is False


@pytest.mark.parametrize("after", [0, 1, 2])
def test_abort_is_sticky_at_every_boundary_without_browser_close(rig, after):
    owner = ready(rig)
    for _ in range(after):
        owner.step()
    owner.abort()
    before = list(rig.calls)
    owner.step()
    owner.tick()
    owner.abort()
    owner.close()
    assert rig.calls == before and owner.status == "aborted"
    assert owner.report["task_completed"] is False
    assert sum(e == "episode_summary" for e, _ in rig.events) == 1


@pytest.mark.parametrize("effect", ["abort", "failure", "reentry", "pause"])
def test_callback_boundary_cannot_dispatch_after_abort_failure_or_reentry(rig, effect):
    owner = None

    def callback(kind, payload):
        rig.events.append((kind, payload))
        if kind == "action_proposed":
            if effect == "abort":
                owner.abort()
            elif effect == "failure":
                raise RuntimeError("password=private http://secret")
            elif effect == "reentry":
                owner.step()
            else:
                owner.pause()

    owner = ready(rig, emit=callback)
    owner.step()
    owner.resume()
    owner.tick()
    assert sum(c[0] == "dispatch" for c in rig.calls) == (1 if effect == "pause" else 0)
    assert owner.status == ("paused" if effect == "pause" else "aborted" if effect == "abort" else "stopped")
    owner.close()
    assert "private" not in (owner.output / "events.jsonl").read_text()
    assert "private" not in (owner.output / "report.json").read_text()


def test_callback_abort_during_constructor_closes_returned_child(rig):
    owner = None

    def callback(kind, payload):
        if "component_hello" in payload:
            owner.abort()

    owner = ready(rig, emit=callback)
    owner.step()
    assert owner.status == "aborted"
    # The fixture's constructor raises when the relay observes cancellation;
    # it has not dispatched any browser action.
    assert not any(c[0] in {"advance", "dispatch"} for c in rig.calls)


@pytest.mark.parametrize("kind", ["duplicate_sequence", "nonfinite", "wrong_kind", "stale_callback"])
def test_malformed_or_stale_child_events_never_dispatch(rig, kind, monkeypatch):
    owner = ready(rig)
    owner.step()
    child = rig.children[0]
    if kind == "stale_callback":
        owner.abort()
        with pytest.raises(ValueError):
            child.send("observation", {"visible": True})
        assert not any(c[0] == "dispatch" for c in rig.calls)
        return

    def malformed():
        item = {"event": "observation", "sequence": 1, "run_id": "child", "payload": {}}
        if kind == "duplicate_sequence":
            item["sequence"] = 0
        elif kind == "nonfinite":
            item["payload"]["value"] = float("nan")
        else:
            item["event"] = "submit"
        child.emit(item)
        rig.calls.append(("dispatch", 0))

    monkeypatch.setattr(child, "advance", malformed)
    owner.step()
    assert owner.status == "stopped" and not any(c[0] == "dispatch" for c in rig.calls)


@pytest.mark.parametrize("change", ["class", "journal", "claim"])
def test_changed_pinned_sources_stop_before_next_decision(rig, change):
    owner = ready(rig)
    owner.step()
    if change == "class":
        rig.source["source_hashes"]["class"] = "1" * 64
    elif change == "journal":
        rig.journal.append(Collected(star_id="other", name="OTHER", source_sha256="1" * 64))
    else:
        owner._claim_path.write_text("{}")
    owner.step()
    assert owner.status == "stopped" and not any(c[0] == "dispatch" for c in rig.calls)


@pytest.mark.parametrize("change", ["class", "journal"])
def test_callback_source_mutation_stops_before_native_dispatch(rig, change):
    def callback(kind, payload):
        if kind == "action_proposed":
            if change == "class":
                rig.source["source_hashes"]["class"] = "2" * 64
            else:
                rig.journal.append(Collected(star_id="other", name="OTHER", source_sha256="1" * 64))

    owner = ready(rig, emit=callback)
    owner.step()
    owner.step()
    assert owner.status == "stopped" and not any(c[0] == "dispatch" for c in rig.calls)


@pytest.mark.parametrize("limit", ["count", "time", "pause_time"])
def test_explicit_budgets_stop_without_automatic_increase(rig, limit):
    owner = ready(rig, max_advances=1 if limit == "count" else 100, max_seconds=5)
    owner.step()
    if limit != "count":
        rig.clock.now = 5
    if limit == "pause_time":
        owner.pause()
        rig.clock.now = 500
    owner.step()
    assert owner.status == "stopped"
    assert not any(c[0] == "dispatch" for c in rig.calls)
    assert owner.failure.endswith("advance_limit" if limit == "count" else "time_limit")


def test_per_star_canonical_reservation_blocks_new_output_retry(rig):
    first = ready(rig)
    first.step()
    first.abort()
    second = ready(rig, output="retry")
    second.step()
    assert second.failure == "project_steps_star_already_reserved" and len(rig.children) == 1


@pytest.mark.parametrize("kind", ["unknown", "false_no", "wrong_star", "exception"])
def test_unverified_children_and_driver_errors_fail_closed(rig, kind, monkeypatch):
    owner = ready(rig)
    owner.step()
    child = rig.children[0]
    if kind == "unknown":
        rig.outcome["phase"] = "finished"
    elif kind == "false_no":
        rig.outcome["phase"] = "verified_no_planet"
    elif kind == "wrong_star":
        monkeypatch.setattr(child, "state", lambda: {"star": "OTHER"})
    else:
        monkeypatch.setattr(child, "advance", lambda: (_ for _ in ()).throw(RuntimeError("secret=private")))
    finish(owner)
    assert owner.status == "stopped" and not owner.report["task_completed"]
    assert "private" not in (owner.output / "events.jsonl").read_text()


@pytest.mark.parametrize("later", ["terminal_events", "second_error", "action_proposed"])
def test_ordinary_child_stop_preserves_cause_and_forbids_later_dispatch(rig, monkeypatch, later):
    owner = ready(rig)
    owner.step()
    child = rig.children[0]
    cause = "no_planet_save_stale_acknowledgement"
    journal_before = rig.journal.path.read_bytes()

    def stop():
        child.send("error", {"type": "BrowserStarStop", "message": cause})
        if later == "action_proposed":
            child.send("action_proposed", {"kind": "CLICK", "target": "forbidden-after-stop"})
            pytest.fail("No native dispatch may follow a child error")
        if later == "second_error":
            child.send("error", {"message": "project_steps_component_unverified"})
        child.finished, child.phase = True, "stopped"
        child.send("state", child.state())
        child.send("episode_summary", child.state())

    monkeypatch.setattr(child, "advance", stop)
    owner.step()
    assert owner.failure == cause and not owner.report["event_forwarding_failed"]
    assert owner.status == "stopped" and not owner.report["task_completed"]
    assert not (owner.output / "event-forwarding-failed.json").exists()
    before = list(rig.calls)
    owner.step()
    owner.tick()
    assert rig.calls == before and rig.journal.path.read_bytes() == journal_before
    assert not any(call[0] == "dispatch" for call in rig.calls)


def test_actual_receiver_exception_during_error_is_still_callback_failure(rig, monkeypatch):
    def fail(kind, payload):
        if kind == "error":
            raise RuntimeError("private callback details")

    owner = ready(rig, emit=fail)
    owner.step()
    child = rig.children[0]
    monkeypatch.setattr(
        child,
        "advance",
        lambda: child.send("error", {"message": "no_planet_save_stale_acknowledgement"}),
    )
    owner.step()
    assert owner.report["event_forwarding_failed"]
    assert owner.failure == "project_steps_event_forwarding_failed"
    assert "private callback details" not in (owner.output / "events.jsonl").read_text()


def prepare_no(rig):
    rig.outcome.update(phase="verified_no_planet", task_completed=True)
    owner = ready(rig)
    finish(owner)
    assert owner.phase == "awaiting_inventory" and not owner.finished and not owner.state()["task_completed"]
    inventory = rig.root / "inventory"
    write(inventory / "confirmed.json", {"fixture": "not real inventory evidence"})
    return owner, inventory


def test_no_workflow_waits_for_explicit_valid_inventory_not_child_success(rig):
    owner, inventory = prepare_no(rig)
    assert not any(e == "episode_summary" for e, _ in rig.events)
    with pytest.raises(BrowserSafetyStop, match="handoff"):
        owner.resume()
    with pytest.raises(BrowserSafetyStop, match="handoff"):
        owner.step()
    owner.provide_inventory(inventory_dir=inventory, inventory_sha256=sha(inventory / "confirmed.json"))
    assert owner.status == "stopped" and rig.journal.load().reduce().report()["verified"] == 0
    assert owner.report["task_completed"] is False


def test_only_importer_plus_canonical_completed_journal_can_complete(rig, monkeypatch):
    owner, inventory = prepare_no(rig)

    def importer(journal, history, inventory_dir, workflow_dir):
        assert inventory_dir == inventory and workflow_dir == owner.output / "star/workflow"
        # Inject only the importer seam, not the canonical reducer/completion rules.
        planned = complete_star(journal.load(), "ALPHA")
        for item in planned.records:
            journal.append(item.payload)
        return {"fixture": "strict importer seam", "task_completed": True}

    monkeypatch.setattr(module, "import_verified_no_planet", importer)
    owner.provide_inventory(inventory_dir=inventory, inventory_sha256=sha(inventory / "confirmed.json"))
    assert owner.status == "completed" and owner.report["task_completed"]
    assert owner.report["project_progress"]["verified"] == 1 and not owner.report["project_completed"]
    assert sum(e == "episode_summary" for e, _ in rig.events) == 1


@pytest.mark.parametrize("change", ["inventory_hash", "workflow", "lying_importer"])
def test_import_handoff_hashes_and_reducer_are_authoritative(rig, monkeypatch, change):
    owner, inventory = prepare_no(rig)
    digest = sha(inventory / "confirmed.json")
    if change == "inventory_hash":
        digest = "0" * 64
    elif change == "workflow":
        write(owner.output / "star/workflow/confirmed.json", {"star": "OTHER"})
    else:
        monkeypatch.setattr(module, "import_verified_no_planet", lambda *_: {"task_completed": True})
    owner.provide_inventory(inventory_dir=inventory, inventory_sha256=digest)
    assert owner.status == "stopped" and not owner.report["task_completed"]


def test_final_callback_failure_cannot_leave_a_successful_report(rig):
    def callback(kind, payload):
        if kind == "episode_summary":
            raise RuntimeError("private")

    owner = ready(rig, emit=callback)
    finish(owner)
    assert owner.report["event_forwarding_failed"] and owner.status == "stopped"
    assert not owner.report["task_completed"]
    assert read_trace(owner.output / "events.jsonl")[-1].payload["status"] == "stopped"


def test_state_callback_pause_does_not_recursively_emit(rig):
    owner = None

    def callback(kind, payload):
        rig.events.append((kind, payload))
        if kind == "state":
            owner.pause()

    owner = create(rig, emit=callback)
    owner.start()
    assert owner.status == "paused" and len(rig.events) == 2
    owner.abort()


def test_keyboard_interrupt_aborts_component_and_preserves_failure(rig, monkeypatch):
    owner = ready(rig)
    owner.step()
    monkeypatch.setattr(rig.children[0], "advance", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        owner.step()
    assert owner.report["status"] == "aborted" and rig.children[0].aborted
    old = list(rig.calls)
    owner.step()
    assert rig.calls == old


def test_actual_star_scheduler_outer_style_events_and_explicit_dip_handoff(star_rig, monkeypatch):  # noqa: F811
    rig = star_rig
    rig.outcome["window"] = "dip_observed"
    journal = ProjectJournal(rig.root, project_id="fixture", attempt_id="actual-star-scheduler").create()
    monkeypatch.setattr(module, "validate_star_class_source", lambda *_: {"star": "Fixture"})
    owner = module.BrowserProjectSteps(
        object(),
        rig.config,
        rig.root / "runtime",
        run_history=rig.root,
        journal=journal,
        star="Fixture",
        model_options={key: key for key in ("dataset", "checkpoint", "color_experiment", "graph_path")},
        emit=lambda *e: rig.events.append(e),
    )
    owner.start()
    owner.provide_class(class_dir=rig.root / "class", selected_class="main_sequence", lifetime_prefix="Ga")
    for _ in range(20):
        owner.step()
        if owner.finished:
            break
    assert owner.status == "handoff" and owner.phase == "planet_measurement_required"
    assert not owner.report["task_completed"]
    assert owner.report["artifact_paths"]["window_dir"] == "runtime/star/window"
    events = read_trace(owner.output / "events.jsonl")
    assert sum(e.event == "episode_summary" for e in events) == 1
    # BrowserStarSession accepts raw native envelopes and forwards its own
    # outer-style events. Both original layers remain replayable.
    observation = next(e for e in events if e.event == "observation")
    nested = observation.payload["component_event"]["payload"]["component_event"]
    assert nested["run_id"] == "numeric" and nested["sequence"] == 1
    assert not any(c[0] in {"save", "verify"} for c in rig.calls)


def test_v1_command_handoffs_are_explicit_and_no_replay_or_score(rig):
    owner = create(rig)
    owner.command({"command": "start", "payload": {}})
    owner.command(
        {
            "command": "step",
            "payload": {
                "class_source": {
                    "class_dir": str(rig.root / "class"),
                    "selected_class": "main_sequence",
                    "lifetime_prefix": "Ga",
                }
            },
        }
    )
    owner.command({"command": "step", "payload": {}})
    assert len(rig.children) == 1 and rig.children[0].steps == 0
    for command in (
        {"command": "replay", "payload": {"path": "private"}},
        {"command": "step", "payload": {"score": True}},
        {"command": "start", "payload": {"guess_class": True}},
    ):
        with pytest.raises(ValueError):
            owner.command(command)
    owner.command({"command": "abort", "payload": {}})


@pytest.mark.parametrize(
    "values",
    [
        {"max_advances": True},
        {"max_advances": 0},
        {"max_advances": 2049},
        {"max_seconds": float("nan")},
        {"max_seconds": float("inf")},
        {"max_seconds": True},
        {"max_seconds": 3601},
        {"interval": -1},
        {"interval": 11},
    ],
)
def test_invalid_limits_fail_before_creating_output(rig, values):
    with pytest.raises(BrowserSafetyStop):
        create(rig, **values)
    assert not (rig.root / "runtime").exists()


@pytest.mark.parametrize("kind", ["pending", "multiple", "symlink"])
def test_unsafe_attempt_journal_fails_before_component_creation(rig, kind):
    if kind == "pending":
        rig.journal.append(Collected(star_id="alpha", name="ALPHA", source_sha256="0" * 64))
        rig.journal.append(
            WriteReserved(
                action_id="save", write_kind="save_star", star_id="alpha", revision=1, before_sha256="0" * 64
            )
        )
    elif kind == "multiple":
        ProjectJournal(rig.root, project_id="fixture", attempt_id="other").create()
    else:
        real = rig.root / "original.jsonl"
        rig.journal.path.rename(real)
        rig.journal.path.symlink_to(real)
    with pytest.raises(BrowserSafetyStop):
        create(rig)
    assert not rig.children
