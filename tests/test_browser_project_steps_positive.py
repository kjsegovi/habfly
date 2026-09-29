"""Offline owner orchestration; native transports are separately tested adapters."""
# ruff: noqa: F811

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_project_steps import create, ready, rig, sha, write  # noqa: F401
from test_browser_project_steps_inventory import inventory_rig  # noqa: F401
from test_project_progress import complete_star

import habfly.browser_project_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.contracts import RuntimeEvent
from habfly.runtime import read_trace


def workflow(star="ALPHA", mode="positive_planet_visible_workflow_readback"):
    return {
        "schema_version": 1,
        "star": star,
        "mode": mode,
        "authority": "visible_workflow_readback",
        "task_completed": True,
    }


@pytest.fixture
def positive_rig(rig, monkeypatch):
    flags = SimpleNamespace(
        current_star="ALPHA",
        current_view="complete-derived",
        paint=None,
        class_hook=None,
        final_hook=None,
        outcome="positive_workflow_verified",
        constructor_hook=None,
    )
    finalizers = []

    def capture(path):
        write(
            path / "observation.json", {"star": "ALPHA", "view": "complete-derived", "ignored_frame_urls": []}
        )
        write(path / "manifest.json", {"observation_sha256": sha(path / "observation.json")})

    class PositiveChild(rig.factory):
        def __init__(self, *args, **kwargs):
            self.positive = None
            super().__init__(*args, **kwargs)

        def state(self):
            state = super().state()
            state["artifact_paths"] = {
                **{
                    key: str((self.output / path).relative_to(rig.root))
                    for key, path in (
                        ("numeric_dir", "numeric"),
                        ("color_dir", "color"),
                        ("positive_dir", "positive"),
                        ("raw_dir", "positive/raw"),
                        ("derived_dir", "positive/derived"),
                    )
                },
                "class_dir": "class",
            }
            state["component"] = deepcopy(self.positive)
            return state

        def advance(self):
            super().advance()
            if not self.finished or self.aborted:
                return
            for name in ("numeric", "color"):
                write(self.output / name / "manifest.json", {"star": "ALPHA", "fixture": name})
            self.positive = {
                "star": "ALPHA",
                "mode": "positive_planet_reference_to_frozen_derived",
                "phase": "planet_classification_required",
                "finished": True,
                "derived_transport_verified": True,
                "task_completed": False,
                "failure_reason": None,
                "event_forwarding_failed": False,
                "child_artifacts": {},
            }
            capture(self.output / "positive/derived/native-copies/copy-04-after")
            for kind in ("presence", "raw", "derived"):
                path = (
                    self.output
                    / "positive"
                    / kind
                    / ("confirmed.json" if kind == "presence" else "report.json")
                )
                write(path, {"star": "ALPHA", "fixture": kind})
                hashes = {
                    str(p.relative_to(rig.root)): sha(p)
                    for p in sorted(path.parent.rglob("*"))
                    if p.is_file()
                }
                report = {"path": str(path.relative_to(rig.root)), "sha256": sha(path), "files": len(hashes)}
                self.positive["child_artifacts"][kind] = report
                write(
                    self.output / "positive" / (kind + "-completed.json"), {"report": report, "files": hashes}
                )
            write(self.output / "positive/report.json", self.positive)
            write(self.output / "report.json", self.state())

    def stellar_sources(book, *directories):
        for directory in directories:
            book.json(directory / ("confirmed.json" if directory.name == "class" else "manifest.json"))
        return {"star": "ALPHA"}

    def planet_sources(book, raw, derived, directory):
        for path in (raw / "report.json", derived / "report.json", directory / "confirmed.json"):
            book.json(path)
        receipt = book.json(directory / "confirmed.json")
        assert receipt["action_source"] == "reference_diagnostic" and "previous" not in receipt
        return {"star": receipt["star"], "planet_class": receipt["value"]}

    def class_adapter(page, config, output, name, **options):
        assert options.keys() == {"source", "before_dispatch"}
        assert options["source"] == "reference_diagnostic"
        rig.calls.append(("class_enter", name))
        if flags.class_hook:
            flags.class_hook()
        source = {"star": flags.current_star, "view": flags.current_view}
        options["before_dispatch"](source, {"star_name": flags.current_star}, {"selected": flags.paint})
        rig.calls.append(("class_dispatch", name))
        receipt = {
            "star": flags.current_star,
            "value": name,
            "action_source": "reference_diagnostic",
            "readback_verified": True,
        }
        write(output / "confirmed.json", receipt)
        write(output / "reserved.json", {k: v for k, v in receipt.items() if k != "readback_verified"})
        return receipt

    class Finalizer:
        def __init__(self, page, config, output, *, emit, **options):
            self.output, self.emit, self.options = output, emit, options
            self.finished = self.aborted = False
            self.steps = self.sequence = 0
            self.phase, self.report = "readback", None
            self.output.mkdir()
            finalizers.append(self)
            rig.calls.append(("final_construct", None))
            if flags.constructor_hook:
                flags.constructor_hook()
            self.send("hello", {"star": "ALPHA", "task_completed": False})

        def send(self, kind, payload):
            item = RuntimeEvent(
                event=kind, sequence=self.sequence, run_id="positive-final", payload=payload
            ).model_dump(mode="json")
            self.sequence += 1
            self.emit(item)

        def state(self):
            return {
                "star": "ALPHA",
                "phase": self.phase,
                "finished": self.finished,
                "task_completed": self.finished
                and not self.aborted
                and flags.outcome == "positive_workflow_verified",
            }

        def advance(self):
            self.send("action_proposed", {"fixture": self.phase})
            if self.aborted:
                return
            rig.calls.append(("final_advance", self.phase))
            self.steps += 1
            if self.steps < 3:
                self.phase = ("save", "verify")[self.steps - 1]
                self.send("state", self.state())
                return
            self.finished, self.phase = True, "finished"
            path = self.output / "workflow/confirmed.json"
            write(path, workflow())
            self.report = {
                "schema_version": 1,
                "mode": "cooperative_positive_planet_finalization",
                "star": "ALPHA",
                "planet_class": json.loads(
                    (self.options["planet_class_dir"] / "confirmed.json").read_bytes()
                )["value"],
                "planet_class_sha256": self.options["planet_class_sha256"],
                "outcome": flags.outcome,
                "task_completed": flags.outcome == "positive_workflow_verified",
                "save_acknowledgement_verified": flags.outcome == "positive_workflow_verified",
                "project_completed": False,
                "workflow_dir": str(path.parent.relative_to(rig.root)),
                "workflow_sha256": sha(path),
            }
            write(self.output / "report.json", self.report)
            if flags.final_hook:
                flags.final_hook(self)
            self.send("episode_summary", self.report)

        def abort(self):
            self.finished = self.aborted = True
            self.phase = "aborted"
            rig.calls.append(("final_abort", None))

    monkeypatch.setattr(module, "_stellar_sources", stellar_sources)
    monkeypatch.setattr(module, "_planet_sources", planet_sources)
    monkeypatch.setattr(module, "_planet", lambda report: {"star_name": report["star"]})
    monkeypatch.setattr(
        module, "planet_projection", lambda capture, mapping: (capture["view"], mapping["star_name"])
    )
    monkeypatch.setattr(module, "select_planet_class", class_adapter)
    monkeypatch.setattr(module, "PositiveFinalizationSteps", Finalizer)
    rig.factory = PositiveChild
    return SimpleNamespace(rig=rig, flags=flags, finalizers=finalizers)


def awaiting(item, **options):
    owner = ready(item.rig, reference_planet_continuation=True, **options)
    for _ in range(3):
        owner.step()
    assert owner.phase == "awaiting_planet_class" and owner.status == "paused" and not owner.finished
    return owner


def finalizing(item, **options):
    owner = awaiting(item, **options)
    owner.provide_planet_class("gas_giant", "Supplied course reference diagnostic")
    owner.step()
    assert owner.phase == "positive_initializing"
    owner.step()
    assert owner.phase == "positive_active"
    return owner


def test_explicit_handoff_and_every_native_boundary_are_separate(positive_rig):
    item, rig = positive_rig, positive_rig.rig
    owner = awaiting(item, positive_save_settle_seconds=30)
    journal = rig.journal.path.read_bytes()
    assert owner.report is None and owner.state()["planet_class_handoff"]["rationale_required"]
    assert owner.state()["star_component"]["phase"] == "planet_classification_required"
    with pytest.raises(BrowserSafetyStop, match="handoff"):
        owner.resume()
    calls = list(rig.calls)
    owner.command(
        {
            "command": "step",
            "payload": {"planet_class": {"name": "gas_giant", "rationale": "Reference source"}},
        }
    )
    assert rig.calls[len(calls) :] == [("preflight", "main_sequence")]
    assert not any(c[0] == "class_dispatch" for c in rig.calls)
    assert owner.phase == "selecting_planet_class"
    assert owner.state()["planet_class_decision"] == {
        "name": "gas_giant",
        "classification_learned": False,
    }
    assert "rationale" not in owner.state()["planet_class_decision"]
    owner.step()
    assert [c for c in rig.calls if c[0] == "class_dispatch"] == [("class_dispatch", "gas_giant")]
    assert not item.finalizers and owner.phase == "positive_initializing"
    owner.step()
    child = item.finalizers[0]
    assert child.steps == 0 and child.options["settle_timeout_seconds"] == 30
    assert child.options["settle_reserved_notice"] is True
    for count in (1, 2, 3):
        owner.step()
        assert child.steps == count and not owner.state()["task_completed"]
    assert owner.phase == "awaiting_inventory" and not owner.finished
    assert owner.state()["positive_component"]["task_completed"]
    assert owner.state()["star_component"]["task_completed"] is False
    assert rig.journal.path.read_bytes() == journal
    trace = read_trace(owner.output / "events.jsonl")
    nested = [e for e in trace if e.payload.get("component") == "planet.finalization"]
    assert nested and all(not e.payload.get("task_completed", False) for e in nested)
    assert all(e.event != "episode_summary" for e in trace)
    owner.close()


@pytest.mark.parametrize("name", ["gas_giant", "ice_giant", "terrestrial"])
def test_no_inferred_class_and_terrestrial_is_only_an_explicit_handoff(positive_rig, name):
    owner = awaiting(positive_rig)
    owner.provide_planet_class(name, "Explicit source-backed selection")
    owner.step()
    assert (owner.output / "planet-class/confirmed.json").is_file()
    assert not owner.state()["task_completed"]
    if name == "terrestrial":
        assert owner.finished and owner.phase == "terrestrial_workflow_required" and owner.status == "handoff"
        assert not positive_rig.finalizers and not owner._workflow
        assert owner.report["artifact_paths"]["planet_class_dir"].endswith("planet-class")
    else:
        assert owner.phase == "positive_initializing" and not owner.finished
    owner.close()


@pytest.mark.parametrize(
    "name,rationale",
    [
        ("invented", "ref"),
        (None, "ref"),
        ("gas_giant", ""),
        ("gas_giant", " "),
        ("gas_giant", None),
        ("gas_giant", "a" * 1001),
    ],
)
def test_invalid_explicit_choices_fail_without_native_dispatch(positive_rig, name, rationale):
    owner = awaiting(positive_rig)
    owner.provide_planet_class(name, rationale)
    assert owner.status == "stopped" and not any(c[0] == "class_dispatch" for c in positive_rig.rig.calls)


@pytest.mark.parametrize(
    "key,value", [("current_star", "BETA"), ("current_view", "changed answers"), ("paint", "ice_giant")]
)
def test_current_star_answers_and_inherited_paint_never_authorize_class_click(positive_rig, key, value):
    owner = awaiting(positive_rig)
    owner.provide_planet_class("gas_giant", "ref")
    setattr(positive_rig.flags, key, value)
    owner.step()
    assert owner.status == "stopped" and "planet_class_current" in owner.failure
    assert not any(c[0] == "class_dispatch" for c in positive_rig.rig.calls)
    assert not positive_rig.finalizers


@pytest.mark.parametrize(
    "relative",
    [
        "star/numeric/manifest.json",
        "star/color/manifest.json",
        "star/positive/raw/report.json",
        "star/positive/derived/report.json",
        "star/positive/presence/confirmed.json",
        "positive-handoff.json",
        "planet-class-handoff.json",
    ],
)
def test_source_changes_during_handoff_prevent_class_dispatch(positive_rig, relative):
    owner = awaiting(positive_rig)
    owner.provide_planet_class("gas_giant", "ref")
    write(owner.output / relative, {"changed": True})
    owner.step()
    assert owner.status == "stopped" and not any(c[0] == "class_dispatch" for c in positive_rig.rig.calls)


def test_added_stop_in_completed_child_is_not_hidden_by_old_hashes(positive_rig):
    owner = awaiting(positive_rig)
    write(owner.output / "star/positive/raw/stopped.json", {"reason": "late_failure"})
    owner.provide_planet_class("gas_giant", "ref")
    assert owner.status == "stopped" and "failed_positive_source" in owner.failure


@pytest.mark.parametrize("boundary", ["handoff", "class", "constructor", "readback", "save", "verify"])
def test_abort_is_sticky_at_positive_boundaries(positive_rig, boundary):
    owner = awaiting(positive_rig)
    if boundary != "handoff":
        owner.provide_planet_class("gas_giant", "ref")
    if boundary in {"constructor", "readback", "save", "verify"}:
        owner.step()
    if boundary in {"readback", "save", "verify"}:
        owner.step()
    for _ in range({"save": 1, "verify": 2}.get(boundary, 0)):
        owner.step()
    owner.abort()
    calls = list(positive_rig.rig.calls)
    owner.step()
    owner.tick()
    owner.close()
    assert positive_rig.rig.calls == calls and owner.status == "aborted"
    assert not owner.state()["task_completed"]


def test_callback_abort_before_native_class_write_and_fixed_deadline(positive_rig):
    owner = awaiting(positive_rig, max_seconds=10)
    owner.provide_planet_class("gas_giant", "ref")
    positive_rig.flags.class_hook = lambda: setattr(positive_rig.rig.clock, "now", 10)
    owner.step()
    assert owner.failure == "project_steps_time_limit"
    assert not any(c[0] == "class_dispatch" for c in positive_rig.rig.calls)


def test_abort_from_class_proposal_callback_prevents_adapter_entry(positive_rig):
    owner = awaiting(positive_rig)
    owner.provide_planet_class("gas_giant", "ref")
    owner._callback = lambda kind, payload: owner.abort() if kind == "action_proposed" else None
    owner.step()
    assert owner.status == "aborted" and not any(c[0] == "class_enter" for c in positive_rig.rig.calls)


@pytest.mark.parametrize("boundary", [0, 1, 2])
def test_finalizer_callback_abort_prevents_next_native_stage(positive_rig, boundary):
    owner = finalizing(positive_rig)
    for _ in range(boundary):
        owner.step()
    child = positive_rig.finalizers[0]
    owner._callback = lambda kind, payload: owner.abort() if kind == "action_proposed" else None
    owner.step()
    assert owner.status == "aborted" and child.aborted and child.steps == boundary
    assert not owner._workflow


def test_source_mutated_by_class_proposal_callback_stops_before_adapter(positive_rig):
    owner = awaiting(positive_rig)
    owner.provide_planet_class("gas_giant", "ref")

    def mutate(kind, payload):
        if kind == "action_proposed":
            write(owner.output / "star/positive/raw/report.json", {"changed": True})

    owner._callback = mutate
    owner.step()
    assert owner.status == "stopped" and not any(c[0] == "class_enter" for c in positive_rig.rig.calls)


def test_original_owner_deadline_applies_to_later_finalizer_steps(positive_rig):
    owner = finalizing(positive_rig, max_seconds=10)
    owner.step()
    positive_rig.rig.clock.now = 10
    owner.step()
    assert owner.failure == "project_steps_time_limit" and positive_rig.finalizers[0].steps == 1


def test_changed_class_receipt_after_readback_prevents_finalizer_save(positive_rig):
    owner = finalizing(positive_rig)
    owner.step()
    write(owner.output / "planet-class/confirmed.json", {"changed": True})
    owner.step()
    assert owner.status == "stopped" and positive_rig.finalizers[0].steps == 1


def test_finalizer_callback_cannot_change_owner_sources_before_save(positive_rig):
    owner = finalizing(positive_rig)
    owner.step()

    def mutate(kind, payload):
        if kind == "action_proposed":
            write(owner.output / "star/positive/derived/report.json", {"changed": True})

    owner._callback = mutate
    owner.step()
    assert owner.status == "stopped" and positive_rig.finalizers[0].steps == 1


def test_constructor_abort_closes_returned_finalizer_without_advancing(positive_rig):
    owner = awaiting(positive_rig)
    owner.provide_planet_class("gas_giant", "ref")
    owner.step()
    positive_rig.flags.constructor_hook = owner.abort
    owner.step()
    assert owner.status == "aborted" and positive_rig.finalizers[0].steps == 0
    # A constructor may reject its callback after the owner stops; no dispatch follows.
    assert not any(c[0] == "final_advance" for c in positive_rig.rig.calls)


@pytest.mark.parametrize("failure", ["outcome", "hash", "star", "mode", "receipt", "report", "stop"])
def test_finalization_requires_persisted_matching_workflow_not_termination(positive_rig, failure):
    owner = finalizing(positive_rig)
    if failure == "outcome":
        positive_rig.flags.outcome = "source_changed"
    else:

        def mutate(child):
            if failure == "stop":
                write(child.output / "stopped.json", {"failure": True})
            elif failure == "report":
                write(child.output / "report.json", {})
            else:
                receipt = workflow(
                    star="BETA" if failure == "star" else "ALPHA",
                    mode="terrestrial_visible_workflow_readback"
                    if failure == "mode"
                    else "positive_planet_visible_workflow_readback",
                )
                if failure == "receipt":
                    receipt["task_completed"] = 1
                if failure == "hash":
                    receipt["unexpected"] = True
                write(child.output / "workflow/confirmed.json", receipt)
                if failure != "hash":
                    child.report["workflow_sha256"] = sha(child.output / "workflow/confirmed.json")
                    write(child.output / "report.json", child.report)

        positive_rig.flags.final_hook = mutate
    for _ in range(3):
        owner.step()
    assert owner.status == "stopped" and not owner.state()["task_completed"]
    assert positive_rig.rig.journal.load().reduce().report()["verified"] == 0


@pytest.mark.parametrize(
    "mode,importer,phase",
    [
        ("no_planet_visible_workflow_readback", "import_verified_no_planet", "verified_no_planet"),
        (
            "positive_planet_visible_workflow_readback",
            "import_verified_positive_planet",
            "verified_positive_planet",
        ),
        ("terrestrial_visible_workflow_readback", "import_verified_terrestrial", "verified_terrestrial"),
    ],
)
def test_only_declared_verified_mode_selects_strict_importer(rig, monkeypatch, mode, importer, phase):
    owner = ready(rig)
    path = owner.output / "external-verified/confirmed.json"
    write(path, workflow(mode=mode))
    owner._workflow_ready(path)
    inventory = rig.root / "inventory/confirmed.json"
    write(inventory, {"fixture": "inventory"})
    called = []

    def verified(journal, *args):
        called.append(importer)
        previous = journal.load()
        completed = complete_star(previous, "ALPHA")
        for record in completed.records[len(previous.records) :]:
            journal.append(record.payload)
        return {"fixture_importer": importer}

    for name in (
        "import_verified_no_planet",
        "import_verified_positive_planet",
        "import_verified_terrestrial",
    ):
        monkeypatch.setattr(
            module, name, verified if name == importer else lambda *args: pytest.fail("wrong importer")
        )
    owner.provide_inventory(inventory_dir=inventory.parent, inventory_sha256=sha(inventory))
    assert called == [importer] and owner.status == "completed" and owner.phase == phase
    assert not owner.state()["project_completed"]


def test_positive_workflow_uses_existing_automatic_inventory_owner(inventory_rig, positive_rig, monkeypatch):
    rig = positive_rig.rig
    rig.outcome.update(phase="planet_classification_required", task_completed=False)
    called = []

    def verified(journal, history, inventory, directory):
        called.append(directory)
        assert directory == rig.root / "runtime/positive-finalization/workflow"
        previous = journal.load()
        completed = complete_star(previous, "ALPHA")
        for record in completed.records[len(previous.records) :]:
            journal.append(record.payload)
        return {"fixture_positive_import": True}

    monkeypatch.setattr(module, "import_verified_positive_planet", verified)
    owner = finalizing(positive_rig, automatic_inventory=True)
    for _ in range(3):
        owner.step()
    assert owner.phase == "inventory_initializing"
    for _ in range(5):
        owner.step()
    assert owner.phase == "verified_positive_planet" and owner.state()["task_completed"]
    assert len(called) == 1 and not owner.state()["project_completed"]


@pytest.mark.parametrize(
    "options",
    [
        {"reference_planet_continuation": 1},
        {"positive_save_settle_seconds": True},
        {"positive_save_settle_seconds": 0},
        {"positive_save_settle_seconds": 31},
        {"positive_save_settle_seconds": float("nan")},
    ],
)
def test_invalid_opt_in_budgets_fail_before_output(rig, options):
    with pytest.raises(BrowserSafetyStop):
        create(rig, **options)
    assert not (rig.root / "runtime").exists()
