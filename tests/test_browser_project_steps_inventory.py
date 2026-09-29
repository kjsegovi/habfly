"""Owner/automatic-inventory integration with injected offline adapters only."""
# ruff: noqa: F811

from types import SimpleNamespace

import pytest
from test_browser_project_steps import create, ready, rig, sha, write  # noqa: F401
from test_project_progress import complete_star

import habfly.browser_project_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.project_progress import Collected
from habfly.runtime import read_trace


@pytest.fixture
def inventory_rig(rig, monkeypatch):
    children = []
    flags = SimpleNamespace(failure=None, finish_hook=None, lying_import=False, construct_hook=None)

    class Inventory:
        def __init__(self, page, config, output, *, emit, cancelled, **options):
            self.output, self.emit, self.cancelled, self.options = output, emit, cancelled, options
            self.finished, self.aborted, self.steps, self.report = False, False, 0, None
            self.phase, self.status, self.failure_reason = "to_list", "ready", None
            self.inventory_dir = self.inventory_sha256 = None
            self.output.mkdir()
            children.append(self)
            rig.calls.append(("inventory_construct", None))
            if flags.construct_hook:
                flags.construct_hook()
            self.emit("hello", {"star": "ALPHA", "task_completed": False})
            self.emit("state", self.state())

        def state(self):
            return {
                "star": "ALPHA",
                "phase": self.phase,
                "status": self.status,
                "finished": self.finished,
                "expected_stars": list(self.options["expected_stars"]),
                "workflow_sha256": self.options["workflow_sha256"],
                "task_completed": False,
                "project_completed": False,
                "collection_count_verified": self.status == "completed",
                "inventory_dir": self.inventory_dir,
                "inventory_sha256": self.inventory_sha256,
                "failure_reason": self.failure_reason,
            }

        def advance(self):
            self.emit("state", {**self.state(), "scheduled_call": self.phase})
            if self.cancelled() or self.aborted:
                return
            rig.calls.append(("inventory_dispatch", self.phase))
            self.steps += 1
            if self.steps < 3:
                self.phase = ("to_stellar", "verify_inventory")[self.steps - 1]
                self.emit("state", self.state())
                return
            if flags.failure:
                self.finished, self.status, self.phase, self.failure_reason = (
                    True,
                    "stopped",
                    "stopped",
                    flags.failure,
                )
                self.report = self.state()
                write(self.output / "stopped.json", self.report)
                self.emit("episode_summary", self.report)
                return
            path = self.output / "inventory/confirmed.json"
            write(path, {"fixture_only": True, "collection_count_verified": True, "task_completed": False})
            upstream = self.output / "pinned-navigation.json"
            write(upstream, {"fixture_navigation": "source"})
            self.inventory_dir = str(path.parent.relative_to(rig.root))
            self.inventory_sha256 = sha(path)
            self.finished, self.status, self.phase = True, "completed", "inventory_verified"
            workflow = self.options["workflow_dir"] / "confirmed.json"
            self.report = {
                **self.state(),
                "source_sha256": {
                    str(path.relative_to(rig.root)): sha(path),
                    str(workflow.relative_to(rig.root)): sha(workflow),
                    str(upstream.relative_to(rig.root)): sha(upstream),
                },
            }
            write(self.output / "report.json", self.report)
            if flags.finish_hook:
                flags.finish_hook(self)
            self.emit("episode_summary", self.report)

        def abort(self):
            rig.calls.append(("inventory_abort", None))
            self.finished = self.aborted = True
            self.phase = self.status = "aborted"

    def importer(journal, history, inventory_dir, workflow_dir):
        rig.calls.append(("import", inventory_dir))
        assert inventory_dir == rig.root / "runtime/collection/inventory"
        assert workflow_dir == rig.root / "runtime/star/workflow"
        if flags.lying_import:
            return {"task_completed": True, "fixture_only": True}
        previous = journal.load()
        completed = complete_star(previous, "ALPHA")
        for record in completed.records[len(previous.records) :]:
            journal.append(record.payload)
        return {"task_completed": True, "fixture_only": True}

    monkeypatch.setattr(module, "ProjectInventorySteps", Inventory)
    monkeypatch.setattr(module, "import_verified_no_planet", importer)
    rig.outcome.update(phase="verified_no_planet", task_completed=True)
    return SimpleNamespace(rig=rig, flags=flags, children=children)


def to_inventory(rig, **kwargs):
    owner = ready(rig, automatic_inventory=True, **kwargs)
    for _ in range(3):
        owner.step()
    assert owner.phase == "inventory_initializing"
    return owner


def to_import(owner):
    for _ in range(4):
        owner.step()
        if owner.finished:
            return
    assert owner.phase == "inventory_import"


def test_automatic_inventory_construct_navigate_verify_import_are_separate_boundaries(inventory_rig):
    item, rig = inventory_rig, inventory_rig.rig
    owner = to_inventory(rig)
    assert not item.children and not owner.state()["task_completed"]
    assert owner.status == "paused" and owner.state()["inventory_handoff"] is None
    owner.step()
    assert len(item.children) == 1 and item.children[0].steps == 0
    assert owner.phase == "inventory_active" and owner.status == "paused"
    assert owner.state()["star_component"]["phase"] == "verified_no_planet"
    assert owner.state()["inventory_component"]["phase"] == "to_list"
    for count in range(1, 4):
        old = len(rig.calls)
        owner.step()
        assert len([c for c in rig.calls[old:] if c[0] == "inventory_dispatch"]) == 1
        assert item.children[0].steps == count
        assert not owner.state()["task_completed"]
    assert owner.phase == "inventory_import" and rig.journal.load().reduce().report()["verified"] == 0
    assert not any(c[0] == "import" for c in rig.calls)
    assert (owner.output / "inventory-handoff.json").is_file()
    owner.step()
    assert owner.finished and owner.status == "completed" and owner.report["task_completed"]
    assert not owner.report["project_completed"] and owner.report["project_progress"]["verified"] == 1
    assert len([c for c in rig.calls if c[0] == "import"]) == 1
    before = list(rig.calls), rig.journal.path.read_bytes()
    owner.step()
    owner.tick()
    owner.close()
    assert (rig.calls, rig.journal.path.read_bytes()) == before
    events = read_trace(owner.output / "events.jsonl")
    assert [(e.event, e.payload) for e in events] == rig.events
    nested = [e for e in events if e.payload.get("component") == "project.inventory"]
    assert nested and all(e.event != "episode_summary" for e in nested)
    assert not any(e.payload["task_completed"] for e in nested if e.event == "state")
    assert sum(e.event == "episode_summary" for e in events) == 1


@pytest.mark.parametrize("already_collected", [False, True])
def test_expected_names_come_only_from_pinned_journal_plus_active_star(inventory_rig, already_collected):
    item, rig = inventory_rig, inventory_rig.rig
    rig.journal.append(Collected(star_id="prior", name="Previous", source_sha256="1" * 64))
    if already_collected:
        rig.journal.append(Collected(star_id="alpha", name="Alpha", source_sha256="2" * 64))
    owner = to_inventory(rig)
    owner.step()
    assert item.children[0].options["expected_stars"] == (
        "Previous",
        "Alpha" if already_collected else "ALPHA",
    )
    assert owner._inventory_expected == item.children[0].options["expected_stars"]
    owner.abort()


@pytest.mark.parametrize("phase", ["inventory_initializing", "inventory_active", "inventory_import"])
def test_pause_resume_and_abort_are_preserved_at_each_inventory_phase(inventory_rig, phase):
    item, rig = inventory_rig, inventory_rig.rig
    owner = to_inventory(rig, interval=0)
    if phase != "inventory_initializing":
        owner.step()
    if phase == "inventory_import":
        for _ in range(3):
            owner.step()
    assert owner.phase == phase
    previous = list(rig.calls)
    owner.tick()
    owner.advance_if_due()
    assert rig.calls == previous
    owner.resume()
    owner.pause()
    assert owner.status == "paused" and owner.phase == phase
    owner.abort()
    before = list(rig.calls)
    owner.step()
    owner.tick()
    assert rig.calls == before and owner.status == "aborted"
    assert not any(c[0] == "import" for c in rig.calls)
    if phase == "inventory_active":
        assert item.children[0].aborted


@pytest.mark.parametrize("effect", ["abort", "reentry", "failure", "pause", "journal", "class"])
def test_inventory_callback_controls_and_changed_sources_cannot_dispatch_unchecked(inventory_rig, effect):
    rig = inventory_rig.rig
    holder = {}

    def callback(kind, payload):
        rig.events.append((kind, payload))
        if payload.get("component") == "project.inventory" and payload.get("component_state", {}).get(
            "scheduled_call"
        ):
            if effect == "abort":
                holder["owner"].abort()
            elif effect == "reentry":
                holder["owner"].step()
            elif effect == "failure":
                raise RuntimeError("private credentials")
            elif effect == "pause":
                holder["owner"].pause()
            elif effect == "journal":
                rig.journal.append(Collected(star_id="unexpected", name="Unexpected", source_sha256="3" * 64))
            else:
                rig.source["source_hashes"]["class"] = "9" * 64

    owner = to_inventory(rig, emit=callback)
    holder["owner"] = owner
    owner.step()
    owner.resume()
    owner.tick()
    assert sum(c[0] == "inventory_dispatch" for c in rig.calls) == int(effect == "pause")
    assert owner.status == ("paused" if effect == "pause" else "aborted" if effect == "abort" else "stopped")
    owner.close()
    assert "private" not in (owner.output / "events.jsonl").read_text()


@pytest.mark.parametrize(
    "change",
    ["receipt", "component_report", "upstream", "handoff", "workflow", "invalidated", "journal", "class"],
)
def test_changed_pins_before_import_stop_without_journal_completion(inventory_rig, change):
    rig = inventory_rig.rig
    owner = to_inventory(rig)
    to_import(owner)
    path = {
        "receipt": owner.output / "collection/inventory/confirmed.json",
        "component_report": owner.output / "collection/report.json",
        "upstream": owner.output / "collection/pinned-navigation.json",
        "handoff": owner.output / "inventory-handoff.json",
        "workflow": owner.output / "star/workflow/confirmed.json",
        "invalidated": owner.output / "collection/invalidated.json",
    }.get(change)
    if path:
        write(path, {"changed": True})
    elif change == "journal":
        rig.journal.append(Collected(star_id="unexpected", name="Unexpected", source_sha256="3" * 64))
    else:
        rig.source["source_hashes"]["class"] = "9" * 64
    owner.step()
    assert owner.status == "stopped" and not owner.report["task_completed"]
    assert not any(c[0] == "import" for c in rig.calls)
    assert rig.journal.load().reduce().report()["verified"] == 0


@pytest.mark.parametrize(
    "kind", ["partial", "lying_import", "bad_child_hash", "missing_pin", "bad_expected_names", "wrong_star"]
)
def test_inventory_failure_and_forged_completion_never_complete_owner(inventory_rig, kind):
    item, rig = inventory_rig, inventory_rig.rig
    if kind == "partial":
        item.flags.failure = "project_inventory_steps_unsupported_pagination"
    elif kind == "lying_import":
        item.flags.lying_import = True
    else:

        def corrupt(child):
            if kind == "bad_child_hash":
                child.inventory_sha256 = "0" * 64
            elif kind == "missing_pin":
                child.report["source_sha256"].pop("runtime/star/workflow/confirmed.json")
                write(child.output / "report.json", child.report)
            elif kind == "bad_expected_names":
                child.report["expected_stars"].append("Invented")
                write(child.output / "report.json", child.report)
            else:
                child.state = lambda: {"star": "Wrong"}

        item.flags.finish_hook = corrupt
    owner = to_inventory(rig)
    to_import(owner)
    owner.step()
    assert owner.status == "stopped" and not owner.report["task_completed"]
    assert rig.journal.load().reduce().report()["verified"] == 0
    if kind == "partial":
        assert owner.failure == "project_inventory_steps_unsupported_pagination"


@pytest.mark.parametrize("at", ["construct", "advance", "import"])
def test_original_owner_deadline_is_not_reset_by_inventory(inventory_rig, at):
    item, rig = inventory_rig, inventory_rig.rig
    owner = to_inventory(rig, max_seconds=10)
    rig.clock.now = 7
    if at != "construct":
        owner.step()
        assert item.children[0].options["max_seconds"] == 3
    if at == "import":
        for _ in range(3):
            owner.step()
    rig.clock.now = 10
    before = list(rig.calls)
    owner.step()
    assert owner.status == "stopped" and owner.failure == "project_steps_time_limit"
    assert [c for c in rig.calls[len(before) :] if c[0] != "preflight"] in [[], [("inventory_abort", None)]]
    assert not any(c[0] == "import" for c in rig.calls)


def test_false_default_keeps_manual_handoff_even_when_component_is_available(inventory_rig):
    rig = inventory_rig.rig
    owner = ready(rig)
    for _ in range(3):
        owner.step()
    assert owner.phase == "awaiting_inventory" and not inventory_rig.children
    assert owner.state()["automatic_inventory"] is False
    owner.close()


def test_running_auto_owner_does_not_force_manual_inventory_pause(inventory_rig):
    rig = inventory_rig.rig
    owner = ready(rig, automatic_inventory=True, interval=0)
    owner.resume()
    phases = []
    for _ in range(8):
        owner.tick()
        phases.append(owner.phase)
        if owner.finished:
            break
        assert owner.status == "running"
    assert phases == [
        "active",
        "active",
        "inventory_initializing",
        "inventory_active",
        "inventory_active",
        "inventory_active",
        "inventory_import",
        "verified_no_planet",
    ]
    assert owner.status == "completed" and rig.journal.load().reduce().report()["verified"] == 1


def test_inventory_completion_callback_is_retired_before_import(inventory_rig):
    item, rig = inventory_rig, inventory_rig.rig
    owner = to_inventory(rig)
    to_import(owner)
    with pytest.raises(ValueError, match="Stale"):
        item.children[0].emit("state", {"star": "ALPHA", "phase": "to_list"})
    assert owner.phase == "inventory_import" and not owner.finished
    assert not any(c[0] == "import" for c in rig.calls)
    owner.close()


def test_no_external_inventory_payload_can_skip_automatic_component(inventory_rig):
    rig = inventory_rig.rig
    owner = to_inventory(rig)
    with pytest.raises(BrowserSafetyStop, match="inventory_handoff_not_ready"):
        owner.provide_inventory(inventory_dir=rig.root, inventory_sha256="0" * 64)
    assert owner.phase == "inventory_initializing" and not any(c[0] == "import" for c in rig.calls)
    owner.close()


def test_inventory_advance_limit_is_shared_with_stellar_work(inventory_rig):
    rig = inventory_rig.rig
    owner = to_inventory(rig, max_advances=4)
    owner.step()
    assert owner.phase == "inventory_active" and owner.advances == 4
    owner.step()
    assert owner.failure == "project_steps_advance_limit"
    assert not any(c[0] in {"inventory_dispatch", "import"} for c in rig.calls)


@pytest.mark.parametrize("value", [1, None, "true", []])
def test_automatic_inventory_opt_in_requires_actual_boolean(inventory_rig, value):
    with pytest.raises(BrowserSafetyStop, match="invalid_automatic_inventory"):
        create(inventory_rig.rig, automatic_inventory=value)
    assert not (inventory_rig.rig.root / "runtime").exists()
