"""Injected coordinator tests; genuine native terrestrial E2E has its own idle gate."""
# ruff: noqa: F811

import json
from types import SimpleNamespace

import pytest
from test_browser_project_steps import create, rig, sha, write  # noqa: F401
from test_browser_project_steps_inventory import inventory_rig  # noqa: F401
from test_browser_project_steps_positive import awaiting, positive_rig, workflow  # noqa: F401
from test_project_progress import complete_star

import habfly.browser_project_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.contracts import RuntimeEvent
from habfly.runtime import read_trace

RATIONALE = "Explicit fixture-only reference judgment, not learned identity or scientific truth."
STAGES = (
    "navigate_habitability",
    "compare_gases",
    "awaiting_gases",
    "select_gases",
    "temperature",
    "greenhouse",
    "chamber",
    "water_phase",
    "awaiting_habitability",
    "select_habitability",
    "save",
    "verify",
)


@pytest.fixture
def terrestrial_rig(positive_rig, monkeypatch):
    item, rig = positive_rig, positive_rig.rig
    rig.outcome.update(phase="planet_classification_required", task_completed=False)
    flags = SimpleNamespace(hook=None, final_hook=None, constructor_hook=None)
    children = []
    graph = SimpleNamespace(checksum="g" * 64)
    monkeypatch.setattr(module, "graph_fingerprint", lambda value: value.checksum)
    pilot, final = rig.root / "frozen-pilot", rig.root / "frozen-final"
    for path in (
        pilot / "training/checkpoint.pt",
        pilot / "training/checkpoint.pt.json",
        pilot / "report.json",
        final / "report.json",
    ):
        write(path, {"injected": True})
    options = {"pilot": pilot, "final_evaluation": final, "graph": graph, "candidates": ["CO2"]}

    class Child:
        def __init__(self, page, config, output, *, emit, **options):
            self.output, self.emit, self.options = output, emit, options
            self.phase, self.finished, self.aborted = STAGES[0], False, False
            self.advances = self.sequence = self.saves = 0
            self.report = self.gas_decision = self.habitability_decision = None
            self.output.mkdir()
            self.events = self.output / "events.jsonl"
            self.events.touch()
            children.append(self)
            rig.calls.append(("terrestrial_construct", options))
            if flags.constructor_hook:
                flags.constructor_hook()
            self.send("hello", {"protocol_version": 1})
            self.send("state", self.state())

        def send(self, kind, payload):
            event = RuntimeEvent(event=kind, payload=payload, sequence=self.sequence, run_id="terra")
            self.sequence += 1
            with self.events.open("a") as stream:
                stream.write(event.model_dump_json() + "\n")
            self.emit(event.model_dump(mode="json"))

        def state(self):
            good = self.finished and not self.aborted
            return {
                "component": module.TERRESTRIAL_MODE,
                "star": "ALPHA",
                "planet_class": "terrestrial",
                "phase": self.phase,
                "finished": self.finished,
                "task_completed": good,
                "failure_reason": "fixture_aborted" if self.aborted else None,
                "gas_decision": self.gas_decision,
                "habitability_decision": self.habitability_decision,
                "phase_reference": {"phase": "Gas", "temperature": "330", "pressure": "9"}
                if self.phase in STAGES[8:] or good
                else None,
                "save_acknowledgement_verified": self.saves == 1,
                "save_dispatch_attempts_recorded": self.saves,
                "project_completed": False,
                "scientific_verified": False,
                "gas_identification_learned": False,
                "habitability_decision_learned": False,
                "cleanup_failed": False,
                "workflow_dir": str((self.output / "workflow").relative_to(rig.root)) if good else None,
                "workflow_sha256": sha(self.output / "workflow/confirmed.json") if good else None,
            }

        def advance(self):
            assert self.phase not in {"awaiting_gases", "awaiting_habitability"}
            self.send("action_proposed", {"fixture": self.phase})
            if self.aborted:
                return
            if flags.hook:
                flags.hook(self)
            rig.calls.append(("terrestrial_advance", self.phase))
            self.advances += 1
            if self.phase == "save":
                self.saves += 1
            if self.phase != "verify":
                self.phase = STAGES[STAGES.index(self.phase) + 1]
                self.send("state", self.state())
                return
            path = self.output / "workflow/confirmed.json"
            write(path, workflow(mode="terrestrial_visible_workflow_readback"))
            self.finished, self.phase = True, "finished"
            self.send("episode_summary", self.state())
            paths = [self.options["planet_class_dir"] / "confirmed.json"] + [
                self.output / (kind + "-decision.json") for kind in ("gas", "habitability")
            ]
            self.report = {
                "schema_version": 1,
                "mode": module.TERRESTRIAL_MODE,
                **self.state(),
                "outcome": "verified_terrestrial",
                "events_sha256": sha(self.events),
                "source_sha256": {str(p): sha(p) for p in paths},
            }
            write(self.output / "report.json", self.report)
            if flags.final_hook:
                flags.final_hook(self)

        def provide_gases(self, *, gases, rationale, supplied_greenhouse_increment):
            assert self.phase == "awaiting_gases"
            if gases != ["CO2"] or not rationale or supplied_greenhouse_increment not in {0, 10, 30, 100}:
                raise BrowserSafetyStop("terrestrial_steps_invalid_supplied_gases")
            self.gas_decision = {
                "gases": gases,
                "rationale": rationale,
                "supplied_greenhouse_increment": supplied_greenhouse_increment,
            }
            write(self.output / "gas-decision.json", self.gas_decision)
            self.phase = "select_gases"
            self.send("state", self.state())

        def provide_habitability(self, *, choice, rationale):
            assert self.phase == "awaiting_habitability"
            if choice not in {"habitable", "not_habitable"} or not rationale:
                raise BrowserSafetyStop("terrestrial_steps_invalid_supplied_habitability")
            self.habitability_decision = {"choice": choice, "rationale": rationale}
            write(self.output / "habitability-decision.json", self.habitability_decision)
            self.phase = "select_habitability"
            self.send("state", self.state())

        def abort(self):
            self.aborted = self.finished = True
            self.phase = "aborted"
            rig.calls.append(("terrestrial_abort", None))

    monkeypatch.setattr(module, "TerrestrialSteps", Child)
    return SimpleNamespace(item=item, rig=rig, flags=flags, children=children, options=options)


def starting(item, **options):
    owner = awaiting(item.item, terrestrial_options=item.options, **options)
    assert owner.state()["planet_class_handoff"]["terrestrial_workflow_available"]
    owner.provide_planet_class("terrestrial", RATIONALE)
    owner.step()
    assert owner.phase == "terrestrial_initializing" and not item.children
    return owner


def initialized(item, **options):
    owner = starting(item, **options)
    owner.step()
    assert owner.phase == "terrestrial_active" and item.children[0].advances == 0
    return owner


def advance_until(owner, phase):
    for _ in range(20):
        if owner.phase == phase or owner.finished:
            break
        owner.step()
    assert owner.phase == phase, owner.state()


def provide_gases(owner):
    return owner.command(
        {
            "command": "step",
            "payload": {
                "gases": {
                    "gases": ["CO2"],
                    "rationale": RATIONALE,
                    "supplied_greenhouse_increment": 10,
                }
            },
        }
    )


def provide_habitability(owner, choice="not_habitable"):
    return owner.command(
        {"command": "step", "payload": {"habitability": {"choice": choice, "rationale": RATIONALE}}}
    )


def completed_child(item, **options):
    owner = initialized(item, **options)
    advance_until(owner, "awaiting_gases")
    provide_gases(owner)
    advance_until(owner, "awaiting_habitability")
    provide_habitability(owner)
    for _ in range(3):
        owner.step()
    return owner


@pytest.mark.parametrize("choice", ["habitable", "not_habitable"])
def test_all_stages_and_explicit_offline_decisions_are_separate(terrestrial_rig, choice):
    item = terrestrial_rig
    owner = initialized(item)
    child = item.children[0]
    journal = item.rig.journal.path.read_bytes()
    assert child.options["graph"] is item.options["graph"]
    assert child.options["max_seconds"] == 1800
    assert child.options["settle_reserved_notice"] is True
    advance_until(owner, "awaiting_gases")
    assert owner.status == "paused" and not owner.finished and not owner.state()["task_completed"]
    assert owner.state()["gas_handoff"]["candidates"] == ["CO2"]
    assert owner.state()["star_component"]["phase"] == "planet_classification_required"
    with pytest.raises(BrowserSafetyStop):
        owner.resume()
    advances = child.advances
    provide_gases(owner)
    assert child.advances == advances and owner.phase == "terrestrial_active"
    assert child.gas_decision["gases"] == ["CO2"]
    advance_until(owner, "awaiting_habitability")
    assert owner.state()["habitability_handoff"]["phase_reference"]["phase"] == "Gas"
    advances = child.advances
    provide_habitability(owner, choice)
    assert child.advances == advances and child.habitability_decision["choice"] == choice
    for _ in range(3):
        owner.step()
    assert owner.phase == "awaiting_inventory" and not owner.finished
    assert child.saves == 1 and not item.item.finalizers
    assert owner.state()["terrestrial_component"]["task_completed"]
    assert owner.state()["planet_class_decision"] == {"name": "terrestrial", "classification_learned": False}
    assert item.rig.journal.path.read_bytes() == journal
    original = read_trace(child.events)
    nested = [
        event.payload["component_event"]
        for event in read_trace(owner.output / "events.jsonl")
        if event.payload.get("component") == "planet.terrestrial"
    ]
    assert nested == [event.model_dump(mode="json") for event in original]
    assert all(not event.payload.get("task_completed") for event in read_trace(owner.output / "events.jsonl"))
    owner.close()


def test_automatic_inventory_uses_only_existing_strict_terrestrial_importer(
    inventory_rig, terrestrial_rig, monkeypatch
):
    item = terrestrial_rig
    calls = []

    def imported(journal, history, inventory, workflow_dir):
        calls.append((inventory, workflow_dir))
        prior = journal.load()
        changed = complete_star(prior, "ALPHA")
        for record in changed.records[len(prior.records) :]:
            journal.append(record.payload)
        return {"injected_strict_importer": True}

    monkeypatch.setattr(module, "import_verified_terrestrial", imported)
    for name in ("import_verified_no_planet", "import_verified_positive_planet"):
        monkeypatch.setattr(module, name, lambda *args: pytest.fail("wrong importer"))
    owner = completed_child(item, automatic_inventory=True)
    assert owner.phase == "inventory_initializing" and not calls
    for _ in range(5):
        owner.step()
    assert owner.phase == "verified_terrestrial" and owner.status == "completed"
    assert calls == [(owner.output / "collection/inventory", owner.output / "terrestrial/workflow")]
    assert not owner.state()["project_completed"] and item.children[0].saves == 1


@pytest.mark.parametrize("phase", STAGES)
def test_abort_is_sticky_at_every_terrestrial_boundary(terrestrial_rig, phase):
    item = terrestrial_rig
    owner = initialized(item)
    child = item.children[0]
    for _ in range(20):
        if child.phase == phase:
            break
        if owner.phase == "awaiting_gases":
            provide_gases(owner)
        elif owner.phase == "awaiting_habitability":
            provide_habitability(owner)
        else:
            owner.step()
    assert child.phase == phase
    owner.abort()
    calls = list(item.rig.calls)
    owner.tick()
    owner.step()
    owner.close()
    assert item.rig.calls == calls and child.aborted and not owner.state()["task_completed"]


@pytest.mark.parametrize("kind", ["model", "graph", "options", "source", "class", "decision"])
def test_changed_pins_stop_before_any_next_stage(terrestrial_rig, kind):
    item = terrestrial_rig
    owner = initialized(item)
    advance_until(owner, "awaiting_gases")
    provide_gases(owner)
    child = item.children[0]
    before = child.advances
    if kind == "model":
        write(item.options["pilot"] / "training/checkpoint.pt", {"changed": True})
    elif kind == "graph":
        item.options["graph"].checksum = "changed"
    elif kind == "options":
        owner.terrestrial_options["candidates"] = ("O2",)
    else:
        path = {
            "source": "star/positive/raw/report.json",
            "class": "planet-class/confirmed.json",
            "decision": "terrestrial/gas-decision.json",
        }[kind]
        write(owner.output / path, {"changed": True})
    owner.step()
    assert owner.status == "stopped" and child.aborted and child.advances == before
    assert item.rig.journal.load().reduce().report()["verified"] == 0


@pytest.mark.parametrize("handoff", ["awaiting_gases", "awaiting_habitability"])
def test_pause_time_counts_toward_original_deadline(terrestrial_rig, handoff):
    item = terrestrial_rig
    owner = initialized(item, max_seconds=10)
    assert item.children[0].options["max_seconds"] == 10
    advance_until(owner, "awaiting_gases")
    if handoff == "awaiting_habitability":
        provide_gases(owner)
        advance_until(owner, handoff)
    item.rig.clock.now = 10
    (provide_gases if handoff == "awaiting_gases" else provide_habitability)(owner)
    assert owner.status == "stopped" and owner.failure == "project_steps_time_limit"
    assert item.children[0].saves == 0


@pytest.mark.parametrize("mode", ["abort", "source", "raise", "reentrant"])
def test_callback_cannot_dispatch_after_abort_or_drift(terrestrial_rig, mode):
    item = terrestrial_rig
    owner = initialized(item)

    def callback(kind, payload):
        if kind != "action_proposed":
            return
        if mode == "abort":
            owner.abort()
        elif mode == "source":
            write(owner.output / "star/positive/raw/report.json", {})
        elif mode == "reentrant":
            owner.step()
        else:
            raise ValueError("untrusted/private callback detail")

    owner._callback = callback
    owner.step()
    assert owner.finished and item.children[0].advances == 0
    assert "untrusted/private callback detail" not in str(owner.report)


@pytest.mark.parametrize(
    "failure",
    [
        "report",
        "hash",
        "star",
        "mode",
        "task",
        "saves",
        "source",
        "stream",
        "stop",
        "finalization_failed",
        "stage_stopped",
        "newfile",
    ],
)
def test_persisted_workflow_and_full_source_tree_required(terrestrial_rig, failure):
    item = terrestrial_rig

    def mutate(child):
        if failure in {"stop", "newfile", "finalization_failed", "stage_stopped"}:
            write(
                child.output
                / {
                    "stop": "stopped.json",
                    "newfile": "after-pin.json",
                    "finalization_failed": "finalization_failed.json",
                    "stage_stopped": "gas-stopped.json",
                }[failure],
                {},
            )
        elif failure == "report":
            write(child.output / "report.json", {})
        elif failure == "source":
            write(child.options["planet_class_dir"] / "confirmed.json", {})
        elif failure == "stream":
            with child.events.open("a") as stream:
                stream.write("\n")
        elif failure == "saves":
            child.saves = 0
            child.report.update(child.state())
            write(child.output / "report.json", child.report)
        else:
            path = child.output / "workflow/confirmed.json"
            proof = json.loads(path.read_bytes())
            proof.update(
                {"star": "BETA"}
                if failure == "star"
                else {"mode": "positive_planet_visible_workflow_readback"}
                if failure == "mode"
                else {"task_completed": 1}
                if failure == "task"
                else {"mutated": True}
            )
            write(path, proof)
            if failure != "hash":
                child.report["workflow_sha256"] = sha(path)
                write(child.output / "report.json", child.report)

    if failure != "newfile":
        item.flags.final_hook = mutate
    owner = completed_child(item)
    if failure == "newfile":
        assert owner.phase == "awaiting_inventory"
        mutate(item.children[0])
        write(item.rig.root / "inventory/confirmed.json", {})
        owner.provide_inventory(
            inventory_dir=item.rig.root / "inventory",
            inventory_sha256=sha(item.rig.root / "inventory/confirmed.json"),
        )
    assert owner.status == "stopped" and not owner.state()["task_completed"]
    assert item.rig.journal.load().reduce().report()["verified"] == 0


@pytest.mark.parametrize("change", ["disabled", "unknown", "candidates", "seed", "budget", "nan", "missing"])
def test_bad_terrestrial_options_fail_before_output(terrestrial_rig, change):
    item = terrestrial_rig
    options = dict(item.options)
    if change == "unknown":
        options["auto_habitable"] = True
    elif change == "candidates":
        options["candidates"] = ["unknown"]
    elif change == "seed":
        options["seed"] = True
    elif change == "budget":
        options["max_seconds"] = 1801
    elif change == "nan":
        options["timeout_seconds"] = float("nan")
    elif change == "missing":
        del options["graph"]
    with pytest.raises(BrowserSafetyStop):
        create(item.rig, reference_planet_continuation=change != "disabled", terrestrial_options=options)
    assert not (item.rig.root / "runtime").exists() and not item.children


def test_constructor_abort_and_default_disabled_leave_no_next_action(terrestrial_rig):
    item = terrestrial_rig
    owner = starting(item)
    item.flags.constructor_hook = owner.abort
    owner.step()
    assert owner.status == "aborted" and item.children[0].advances == 0
    assert not any(name == "terrestrial_advance" for name, _ in item.rig.calls)


def test_invalid_explicit_decision_is_not_replaced(terrestrial_rig):
    item = terrestrial_rig
    owner = initialized(item)
    advance_until(owner, "awaiting_gases")
    before = item.children[0].advances
    owner.provide_gases(gases=["O2"], rationale=RATIONALE, supplied_greenhouse_increment=10)
    assert owner.status == "stopped" and item.children[0].advances == before
    assert item.children[0].gas_decision is None


def test_running_child_pauses_at_both_reference_handoffs(terrestrial_rig):
    owner = initialized(terrestrial_rig)
    owner.resume()
    owner.tick()
    owner.tick()
    assert owner.status == "paused" and owner.phase == "awaiting_gases"
    count = terrestrial_rig.children[0].advances
    owner.tick()
    assert terrestrial_rig.children[0].advances == count
    provide_gases(owner)
    assert owner.status == "paused"
    owner.resume()
    for _ in range(8):
        owner.tick()
    assert owner.status == "paused" and owner.phase == "awaiting_habitability"
    assert terrestrial_rig.children[0].saves == 0
    owner.close()


def test_second_output_cannot_resume_an_uncertain_star(terrestrial_rig):
    item = terrestrial_rig
    owner = initialized(item)
    owner.abort()
    second = create(
        item.rig, output="second", reference_planet_continuation=True, terrestrial_options=item.options
    )
    second.start()
    second.provide_class(
        class_dir=item.rig.root / "class", selected_class="main_sequence", lifetime_prefix="Ga"
    )
    second.step()
    assert second.status == "stopped" and second.failure == "project_steps_star_already_reserved"
    assert len(item.children) == 1
