"""Offline source proofs and injected cooperative steps; never launches a browser."""

import io
import json
import socket
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from test_browser_project_inventory_steps import Config as BaseConfig
from test_browser_project_inventory_steps import Events
from test_browser_project_inventory_steps import capture as view_capture
from test_browser_stellar import capture as stellar_capture
from test_project_evidence import ingest, inventory, read, sha, sources, write  # noqa: F401 - pytest fixture

import habfly.browser_project_next_star_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_probe import save_probe
from habfly.contracts import RuntimeEvent
from habfly.project_progress import Active, Collected, ProjectJournal, WriteReserved


def starfield_png():
    image = Image.new("RGB", (600, 400), (12, 12, 12))
    draw = ImageDraw.Draw(image)
    for x in (240, 280, 320):
        draw.rectangle((x, 220, x + 1, 221), fill=(180, 180, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def fresh_capture(star):
    report = stellar_capture()
    report["ignored_frame_urls"] = []
    frame = report["frames"][0]
    frame["text"] = frame["text"].replace("ALTHINAGON", star.upper())
    frame["accessibility"] = frame["accessibility"].replace("Althinagon", star)
    for control in frame["controls"][:3]:
        control["value"] = ""
    return report


def fresh_receipt(directory, star, *, names=(), excluded=(), initial=False, anchor=(0.4, 0.55)):
    directory.mkdir(parents=True, exist_ok=True)
    png = starfield_png()
    point = module.visible_star_point(png, excluded=excluded, anchor=anchor)
    scope = {
        "action_source": "deterministic_navigation",
        "visited_stars": list(names),
        "excluded_points": list(excluded),
        "starfield_anchor": list(anchor),
        "selection_basis": "rendered_dot_geometry_only_not_scientific_class",
        "max_star_clicks": 1,
        "max_view_clicks": 1,
        "max_seconds": 90,
        "answer_writes": 0,
        "task_completed": False,
        "collection_count_verified": False,
    }
    if initial:
        scope = {
            "mode": "read_only_initial_setup_handoff",
            "selection_source": "completed_initial_setup_rendered_starfield",
            "starfield_sha256": module._sha(png),
            "credentials_recorded": False,
            "browser_actions": 0,
            "automatic_retry": False,
            "ready_screen_sha256": module.screen_identity(fresh_capture(star)),
        }
    write(directory / "scope.json", scope)
    (directory / "setup-starfield.png").write_bytes(png)
    save_probe(fresh_capture(star), directory / "stellar")
    save_probe(
        fresh_capture(star) if initial else view_capture("starfield", None, None), directory / "before"
    )
    receipt = {
        "star": star,
        "selected_point": point,
        "stellar_observations_verified": True,
        "fresh_blank_numeric_answers_verified": True,
        "painted_stellar_class": None,
        "class_selection_verified": False,
        "action_source": "deterministic_navigation",
        "answer_writes": 0,
        "collection_count_verified": False,
        "task_completed": False,
    }
    write(directory / "confirmed.json", receipt)
    entry = {"path": str(directory / "confirmed.json"), "sha256": sha(directory / "confirmed.json")}
    if initial:
        entry["starfield_path"] = str(directory / "setup-starfield.png")
    return entry, receipt


@pytest.fixture
def proof(sources):  # noqa: F811 - imported pytest fixture
    journal, history, inventory, workflow, _, _ = sources
    result = ingest(sources)
    owner = history / "owner"
    owner.mkdir()
    report = {
        "mode": "bounded_single_star_project_runtime",
        "status": "completed",
        "phase": "verified_no_planet",
        "star": "Althinagon",
        "finished": True,
        "task_completed": True,
        "project_completed": False,
        "failure_reason": None,
        "event_forwarding_failed": False,
        "project_progress": result["progress"],
    }
    write(owner / "report.json", report)
    event = RuntimeEvent(
        event="episode_summary", sequence=0, run_id="offline-owner", payload={**report, "completed": True}
    )
    (owner / "events.jsonl").write_text(event.model_dump_json() + "\n")
    write(owner / "workflow-import.json", result)
    first, first_receipt = fresh_receipt(history / "initial", "Althinagon", initial=True)
    second, _ = fresh_receipt(
        history / "second", "Beta", names=["Althinagon"], excluded=[first_receipt["selected_point"]]
    )
    return SimpleNamespace(
        journal=journal,
        root=history,
        owner=owner,
        inventory=inventory,
        workflow=workflow,
        expected=["Althinagon", "Beta"],
        entries=[first, second],
        sources=sources,
    )


class Config(BaseConfig):
    def allows(self, url):
        return url == "http://localhost/owned-preview"


@pytest.fixture
def rig(proof, monkeypatch):
    flags = SimpleNamespace(
        calls=[],
        now=0,
        cancel=False,
        inspect_hook=None,
        nav_hook=None,
        picker_hook=None,
        fail_picker=False,
        fail_navigation=False,
    )
    page = Events()
    page.frames = [object()]
    page.context = Events()
    page.context.pages = [page]
    page.url = "http://localhost/owned-preview"
    page.capture = read(proof.inventory / "after/observation.json")
    config = Config()
    actual_view = module.project_view
    monkeypatch.setattr(module, "project_view", lambda r: r.get("fixture_view") or actual_view(r))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("offline only"))

    def inspect(page, config):
        flags.calls.append("read")
        if flags.inspect_hook:
            flags.inspect_hook()
        return deepcopy(page.capture)

    def navigate(page, config, output, destination, expected_star=None):
        flags.calls.append("navigate")
        assert destination == "starfield" and expected_star is None
        before = deepcopy(page.capture)
        save_probe(before, output / "before")
        save_probe(before, output / "pre-click")
        intent = {
            "mode": "bounded_project_navigation",
            "destination": destination,
            "from": module.project_view(before),
            "max_clicks": 1,
            "automatic_retry": False,
            "task_completed": False,
            "answer_writes": 0,
            "collection_clicks": 0,
            "save_clicks": 0,
            "assessment_clicks": 0,
            "submission_clicks": 0,
        }
        write(output / "reserved.json", intent)
        if flags.nav_hook:
            flags.nav_hook(output)
        if flags.fail_navigation:
            write(output / "stopped.json", {"click_may_have_occurred": True})
            raise BrowserSafetyStop("project_navigation_target_or_screen_changed")
        page.capture = view_capture("starfield", None, None)
        save_probe(page.capture, output / "after")
        receipt = {
            **intent,
            "to": module.project_view(page.capture),
            "destination_verified": True,
            "navigation_clicks": 1,
            "config_mutated": False,
        }
        write(output / "confirmed.json", receipt)
        return receipt

    class Picker:
        MAX_SECONDS = 90

        def __init__(self, page, config, output, *, visited_stars, excluded_points, anchor, emit):
            flags.calls.append("picker_init")
            self.page, self.directory, self.emit = page, output, emit
            self.names, self.points, self.anchor = visited_stars, excluded_points, anchor
            self.initial = deepcopy(page.capture)
            self.closed = self.star_clicked = self.view_clicked = self.stellar_tab_clicked = False
            self.stage, self.receipt, self.step = "starfield", None, 0

        def advance(self):
            flags.calls.append("picker_advance")
            self.step += 1
            if flags.picker_hook:
                flags.picker_hook(self)
            if self.step in {1, 2}:
                self.stage = "selecting_visible_star" if self.step == 1 else "opening_star_data"
                self.emit("state", {"setup_stage": self.stage})
                if flags.fail_picker:
                    raise BrowserSafetyStop("next_star_uncertain_click")
                flags.calls.append("dot" if self.step == 1 else "view")
                if self.step == 1:
                    self.star_clicked = True
                else:
                    self.view_clicked = True
                return "waiting"
            _, self.receipt = fresh_receipt(
                self.directory, "Gamma", names=self.names, excluded=self.points, anchor=self.anchor
            )
            self.page.capture = fresh_capture("Gamma")
            self.stage, self.closed = "stellar_screen_ready", True
            self.emit("state", {"setup_stage": self.stage})
            return "stellar"

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "inspect_page", inspect)
    monkeypatch.setattr(module, "navigate_project", navigate)
    monkeypatch.setattr(module, "NextStarPicker", Picker)

    def create(name="transition", **overrides):
        options = {
            "run_history": proof.root,
            "journal": proof.journal,
            "completed_owner_dir": proof.owner,
            "expected_stars": proof.expected,
            "visited_receipts": proof.entries,
            "cancelled": lambda: flags.cancel,
            "_clock": lambda: flags.now,
        }
        options.update(overrides)
        return module.ProjectNextStarSteps(page, config, proof.root / name, **options)

    return SimpleNamespace(proof=proof, root=proof.root, flags=flags, page=page, config=config, create=create)


def complete(component):
    for _ in range(10):
        if component.finished:
            break
        component.advance()
    return component.state()


def test_strict_previous_workflow_and_latest_task_allow_older_unresolved(proof):
    book = module._Evidence(proof.root)
    state, star, capture = module._completed_owner(book, proof.journal, proof.owner, proof.expected)
    assert star.name == "Althinagon" and star.task_completed
    assert state.report()["verified"] == 1 and state.report()["unresolved"] == 1
    assert capture == read(proof.inventory / "after/observation.json")
    assert str(proof.workflow.relative_to(proof.root) / "confirmed.json") in book.hashes
    before = proof.journal.path.read_bytes()
    book.unchanged()
    assert proof.journal.path.read_bytes() == before


@pytest.mark.parametrize("mode", ["positive", "terrestrial"])
def test_positive_and_terrestrial_source_planners_are_revalidated_offline(tmp_path, monkeypatch, mode):
    # Import only source builders, not their optional Chromium fixtures.
    from test_project_positive_evidence import workflow as positive_workflow
    from test_project_terrestrial_evidence import workflow as terrestrial_workflow

    registry = {}
    helper = module.project_positive_evidence if mode == "positive" else module.project_terrestrial_evidence
    builder = positive_workflow if mode == "positive" else terrestrial_workflow
    importer = (
        helper.import_verified_positive_planet if mode == "positive" else helper.import_verified_terrestrial
    )

    def validated(book, **directories):
        hashes, bundle = registry[str(directories["numeric_dir"])]
        for path in hashes:
            book.clean((book.history / path).parent)
            book.read(book.history / path)
        return deepcopy(bundle)

    monkeypatch.setattr(helper, "_load_sources", validated)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("offline only"))
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="offline-" + mode).create()
    inv = inventory(tmp_path, ["Dulat", "Beta"], mode)
    work = builder(tmp_path, "Dulat", mode, registry)
    result = importer(journal, tmp_path, inv, work)
    report = {
        "mode": "bounded_single_star_project_runtime",
        "status": "completed",
        "phase": "verified_positive_planet" if mode == "positive" else "verified_terrestrial",
        "star": "Dulat",
        "finished": True,
        "task_completed": True,
        "project_completed": False,
        "failure_reason": None,
        "event_forwarding_failed": False,
        "project_progress": result["progress"],
    }
    owner = tmp_path / "owner"
    write(owner / "report.json", report)
    write(owner / "workflow-import.json", result)
    event = RuntimeEvent(
        event="episode_summary", sequence=0, run_id="offline", payload={**report, "completed": True}
    )
    (owner / "events.jsonl").write_text(event.model_dump_json() + "\n")
    book = module._Evidence(tmp_path)
    state, previous, _ = module._completed_owner(book, journal, owner, ["Dulat", "Beta"])
    assert previous.task_completed and state.report()["verified"] == 1
    assert state.report()["unresolved"] == 1
    assert any("class/confirmed.json" in path for path in book.hashes)
    assert not state.report()["project_completed"]


def test_thirtieth_collected_star_is_an_absolute_transition_gate(rig):
    names = list(rig.proof.expected)
    for index in range(28):
        name = "Extra" + str(index)
        rig.proof.journal.append(Collected(star_id=name, name=name, source_sha256="f" * 64))
        names.append(name)
    with pytest.raises(BrowserSafetyStop, match="collection_limit_or_names_mismatch"):
        rig.create(expected_stars=names)
    assert not rig.flags.calls


def test_an_older_completed_owner_cannot_skip_a_newer_verified_task(rig):
    from test_project_evidence import workflow

    registry = rig.proof.sources[4]
    work = workflow(rig.root, "Beta", "newer", registry)
    result = module.project_evidence.import_verified_no_planet(
        rig.proof.journal, rig.root, rig.proof.inventory, work
    )
    assert result["progress"]["verified"] == 2
    with pytest.raises(BrowserSafetyStop, match="previous_owner_is_not_latest_active_task"):
        rig.create()
    assert not rig.flags.calls


def test_offline_constructor_and_bounded_steps_preserve_journal_and_receipts(rig):
    emitted = []

    def emit(kind, payload):
        event = json.loads((rig.root / "transition/events.jsonl").read_text().splitlines()[-1])
        assert (kind, payload) == (event["event"], event["payload"])
        emitted.append(event)

    original = rig.proof.journal.path.read_bytes()
    config = rig.config.model_dump(mode="json")
    component = rig.create(emit=emit)
    assert not rig.flags.calls and not rig.page.listeners
    assert component.state()["prior_unresolved_stars"] == ["Beta"]
    assert not component.claim_path.exists()
    phases = [
        "picker_initializing",
        "picking",
        "picking",
        "picking",
        "validating_receipt",
        "fresh_star_handoff",
    ]
    for phase in phases:
        previous = len(rig.flags.calls)
        component.advance()
        assert component.phase == phase, component.state()
        calls = rig.flags.calls[previous:]
        assert len([c for c in calls if c in {"navigate", "picker_init", "picker_advance"}]) <= 1
    assert component.status == "completed" and component.finished
    assert component.state()["star"] == "Gamma"
    assert component.state()["fresh_blank_numeric_answers_verified"]
    assert component.state()["navigation_clicks"] == 1
    assert rig.flags.calls.count("dot") == rig.flags.calls.count("view") == 1
    assert component.state()["project_progress"]["collected"] == 2  # No inferred collection of Gamma.
    for key in (
        "task_completed",
        "project_completed",
        "class_selection_verified",
        "collection_count_verified",
    ):
        assert component.state()[key] is False
    assert read(rig.root / "transition/picker/confirmed.json")["class_selection_verified"] is False
    assert rig.config.model_dump(mode="json") == config
    assert rig.proof.journal.path.read_bytes() == original
    assert all(not values for values in rig.page.listeners.values())
    assert all(not values for values in rig.page.context.listeners.values())
    for path, checksum in component.report["source_sha256"].items():
        assert sha(rig.root / path) == checksum
    assert [e["sequence"] for e in emitted] == list(range(len(emitted)))
    child = [e for e in emitted if "component_event" in e["payload"]]
    assert child and all(e["payload"]["component"] == "next_star.picker" for e in child)
    frozen = list(rig.flags.calls), (rig.root / "transition/events.jsonl").read_bytes()
    component.advance()
    component.abort()
    component.close()
    assert (rig.flags.calls, (rig.root / "transition/events.jsonl").read_bytes()) == frozen


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_completed", False),
        ("finished", False),
        ("status", "stopped"),
        ("phase", "planet_classification_required"),
        ("star", "Beta"),
        ("event_forwarding_failed", True),
    ],
)
def test_unresolved_owner_never_skipped(rig, field, value):
    report = read(rig.proof.owner / "report.json")
    report[field] = value
    write(rig.proof.owner / "report.json", report)
    with pytest.raises(BrowserSafetyStop):
        rig.create()
    assert not rig.flags.calls and not (rig.root / "transition").exists()


@pytest.mark.parametrize("source", ["workflow", "inventory", "binding", "events", "import", "capture"])
def test_changed_source_chain_rejected_before_actions(rig, source):
    paths = {
        "workflow": rig.proof.workflow / "confirmed.json",
        "inventory": rig.proof.inventory / "confirmed.json",
        "binding": next(rig.root.glob("project-evidence-*.json")),
        "events": rig.proof.owner / "events.jsonl",
        "import": rig.proof.owner / "workflow-import.json",
        "capture": rig.proof.workflow / "stellar-readback/verified/observation.json",
    }
    path = paths[source]
    if source == "events":
        path.write_text(RuntimeEvent(event="state", sequence=0).model_dump_json() + "\n")
    else:
        value = read(path)
        if source in {"workflow", "capture"}:
            value["tampered"] = True
        elif source == "inventory":
            value["total_collected"] = 3
        elif source == "binding":
            value["workflow_sha256"] = "0" * 64
        else:
            value["star_id"] = "other"
        write(path, value)
    with pytest.raises((BrowserSafetyStop, ValueError)):
        rig.create()
    assert not rig.flags.calls


@pytest.mark.parametrize("change", ["pending", "active_other", "second_journal", "extra_collected"])
def test_canonical_attempt_gates(rig, change):
    progress = rig.proof.journal.load().reduce()
    beta = next(s for s in progress.stars.values() if s.name == "Beta")
    if change == "pending":
        rig.proof.journal.append(
            WriteReserved(
                action_id="pending-save",
                write_kind="save_star",
                star_id=beta.id,
                revision=progress.revision,
                before_sha256="f" * 64,
            )
        )
    elif change == "active_other":
        rig.proof.journal.append(Active(star_id=beta.id, stage="stellar_numeric"))
    elif change == "second_journal":
        ProjectJournal(rig.root, project_id="habworlds", attempt_id="other-attempt").create()
    else:
        rig.proof.journal.append(Collected(star_id="extra", name="Extra", source_sha256="f" * 64))
    with pytest.raises((BrowserSafetyStop, ValueError)):
        rig.create()
    assert not rig.flags.calls


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "initial_png_missing",
        "hash",
        "point",
        "class_claim",
        "legacy",
        "blank",
        "outside",
    ],
)
def test_exact_visited_proofs_required(rig, change, tmp_path):
    entries = deepcopy(rig.proof.entries)
    if change == "missing":
        entries.pop()
    elif change == "duplicate":
        entries[1] = entries[0]
    elif change == "initial_png_missing":
        entries[0].pop("starfield_path")
    elif change == "hash":
        entries[0]["sha256"] = "0" * 64
    elif change in {"point", "class_claim", "legacy"}:
        path = rig.root / "second/confirmed.json"
        receipt = read(path)
        if change == "point":
            receipt["selected_point"]["x"] += 20
        elif change == "class_claim":
            receipt["class_selection_verified"] = True
        else:
            receipt["failed_run"] = "old-consumed-picker"
        write(path, receipt)
        entries[1]["sha256"] = sha(path)
    elif change == "blank":
        report = fresh_capture("Beta")
        report["frames"][0]["controls"][0]["value"] = "0"
        path = rig.root / "second/stellar/observation.json"
        write(path, report)
        manifest = read(path.parent / "manifest.json")
        manifest["observation_sha256"] = sha(path)
        write(path.parent / "manifest.json", manifest)
    elif change == "outside":
        entries[0]["path"] = str(tmp_path.parent / "outside-confirmed.json")
    with pytest.raises((BrowserSafetyStop, ValueError)):
        rig.create(visited_receipts=entries)
    assert not rig.flags.calls


@pytest.mark.parametrize("boundary", range(7))
def test_abort_is_sticky_at_each_boundary(rig, boundary):
    component = rig.create()
    for _ in range(boundary):
        component.advance()
    before = list(rig.flags.calls)
    component.abort()
    complete(component)
    assert rig.flags.calls == before
    assert component.status == ("completed" if boundary == 6 else "aborted")
    assert not component.state()["task_completed"]


@pytest.mark.parametrize("phase", ["constructor", "dot", "view", "summary"])
def test_callback_cancellation_stops_before_next_dispatch(rig, phase):
    def emit(kind, payload):
        setup = payload.get("component_state", {}).get("setup_stage")
        if (
            (phase == "constructor" and kind == "hello")
            or (phase == "dot" and setup == "selecting_visible_star")
            or (phase == "view" and setup == "opening_star_data")
            or (phase == "summary" and kind == "episode_summary")
        ):
            rig.flags.cancel = True

    component = rig.create(emit=emit)
    complete(component)
    assert component.status == "aborted"
    assert not (rig.root / "transition/confirmed.json").exists()
    assert rig.flags.calls.count("dot") == (1 if phase in {"view", "summary"} else 0)
    assert rig.flags.calls.count("view") == (1 if phase == "summary" else 0)


@pytest.mark.parametrize("phase", ["before_navigation", "before_picker", "before_dot", "before_handoff"])
def test_source_mutation_between_steps_never_succeeds(rig, phase):
    component = rig.create()
    for _ in range({"before_navigation": 0, "before_picker": 1, "before_dot": 2, "before_handoff": 5}[phase]):
        component.advance()
    path = rig.root / "second/confirmed.json"
    path.write_text(path.read_text() + " ")
    before = list(rig.flags.calls)
    complete(component)
    assert component.status == "stopped" and not component.state()["fresh_blank_numeric_answers_verified"]
    assert rig.flags.calls == before


@pytest.mark.parametrize("change", ["screen", "frames", "page", "url", "dialog", "popup", "config"])
def test_current_context_change_stops_before_navigation(rig, change):
    component = rig.create()
    if change == "screen":
        rig.page.capture["frames"][0]["accessibility"] += "\n- text: changed row"
    elif change == "config":
        component.config.frames[0].count = 2
    else:

        def inspect_hook():
            if change == "frames":
                rig.page.frames = [object()]
            elif change == "page":
                rig.page.context.pages.append(object())
            elif change == "url":
                rig.page.url = "http://localhost/outside"
            elif change == "dialog":
                component._dialog(object())
            else:
                component._popup(object())

        rig.flags.inspect_hook = inspect_hook
    complete(component)
    assert component.status == "stopped" and "navigate" not in rig.flags.calls
    assert not component.claim_path.exists()


@pytest.mark.parametrize(
    "failure", ["navigation", "dot", "view", "expired", "advance_limit", "handoff", "event", "summary_source"]
)
def test_post_claim_failure_is_consumed_and_never_retries(rig, failure):
    def emit(kind, payload):
        if (
            failure == "event"
            and payload.get("component_state", {}).get("setup_stage") == "opening_star_data"
        ):
            raise RuntimeError("secret detail must not escape")
        if failure == "summary_source" and kind == "episode_summary":
            path = rig.root / "second/confirmed.json"
            path.write_text(path.read_text() + " ")

    component = rig.create(emit=emit, max_advances=4 if failure == "advance_limit" else 128)
    if failure == "navigation":
        rig.flags.fail_navigation = True
    elif failure in {"dot", "view"}:

        def hook(picker):
            rig.flags.fail_picker = picker.step == (1 if failure == "dot" else 2)

        rig.flags.picker_hook = hook
    elif failure == "expired":
        component.advance()
        rig.flags.now = 180
    elif failure == "handoff":
        for _ in range(5):
            component.advance()
        rig.page.capture["frames"][0]["controls"][0]["value"] = "99"
    complete(component)
    assert component.status == "stopped", component.state()
    assert component.claim_path.exists()
    assert not (rig.root / "transition/confirmed.json").exists()
    assert "secret" not in json.dumps(component.report)
    assert rig.flags.calls.count("dot") <= 1 and rig.flags.calls.count("view") <= 1
    before = list(rig.flags.calls)
    with pytest.raises(BrowserSafetyStop):
        rig.create("retry")
    assert rig.flags.calls == before


def test_callbacks_cannot_reenter_dispatch_or_abort_success_summary(rig):
    holder = {}

    def emit(kind, payload):
        if kind == "episode_summary" and payload["status"] == "completed":
            holder["component"].abort()

    holder["component"] = rig.create(emit=emit)
    component = holder["component"]
    complete(component)
    assert component.status == "aborted" and not (rig.root / "transition/confirmed.json").exists()


def test_revision_claim_prevents_two_preconstructed_owners(rig):
    first, second = rig.create("first"), rig.create("second-run")
    first.advance()
    before = list(rig.flags.calls)
    second.advance()
    assert second.status == "stopped"
    assert rig.flags.calls.count("navigate") == 1
    assert "picker_init" not in rig.flags.calls[len(before) :]
    first.abort()


def test_canonical_claim_survives_a_changed_output_name_and_matching_screen(rig):
    first, second = rig.create("first"), rig.create("second-run")
    initial = deepcopy(rig.page.capture)
    first.advance()
    claim = first.claim_path.read_bytes()
    rig.page.capture = initial  # Even restored visible state cannot erase the durable claim.
    second.advance()
    assert second.status == "stopped" and rig.flags.calls.count("navigate") == 1
    assert first.claim_path.read_bytes() == claim
    first.abort()


@pytest.mark.parametrize(
    "target",
    [
        "initial_screen",
        "initial_png",
        "picker_scope",
        "picker_before",
        "exclusions",
        "legacy_error",
        "symlink",
    ],
)
def test_visited_capture_and_selection_sources_are_immutable(rig, target):
    if target == "initial_screen":
        path = rig.root / "initial/scope.json"
        scope = read(path)
        scope["ready_screen_sha256"] = "f" * 64
        write(path, scope)
    elif target == "initial_png":
        (rig.root / "initial/setup-starfield.png").write_bytes(b"not the captured PNG")
    elif target in {"picker_scope", "exclusions"}:
        path = rig.root / "second/scope.json"
        scope = read(path)
        if target == "picker_scope":
            scope["answer_writes"] = True
        else:
            scope["visited_stars"] = ["Beta"]
        write(path, scope)
    elif target == "picker_before":
        path = rig.root / "second/before/observation.json"
        report = read(path)
        report["fixture_view"]["surface"] = "detail"
        write(path, report)
        manifest = read(path.parent / "manifest.json")
        manifest["observation_sha256"] = sha(path)
        write(path.parent / "manifest.json", manifest)
    elif target == "legacy_error":
        write(rig.root / "second/stopped.json", {"reason": "old uncertain action"})
    else:
        path = rig.root / "alias.json"
        path.symlink_to(rig.root / "second/confirmed.json")
        rig.proof.entries[1]["path"] = str(path)
    with pytest.raises((BrowserSafetyStop, ValueError)):
        rig.create()
    assert not rig.flags.calls


def test_reentrant_callback_cannot_dispatch_another_step(rig):
    holder = {}

    def emit(kind, payload):
        if payload.get("component_state", {}).get("setup_stage") == "selecting_visible_star":
            holder["component"].advance()

    holder["component"] = rig.create(emit=emit)
    complete(holder["component"])
    assert holder["component"].status == "stopped"
    assert "dot" not in rig.flags.calls and "view" not in rig.flags.calls


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("anchor", (0.7, 0.5)),
        ("max_seconds", 999),
        ("max_advances", 999),
        ("expected", ("Different",)),
        ("points", []),
        ("scope", {"changed": True}),
    ],
)
def test_frozen_constructor_choices_cannot_change_between_advances(rig, attribute, value):
    component = rig.create()
    setattr(component, attribute, value)
    complete(component)
    assert component.status == "stopped" and not rig.flags.calls


def test_state_payload_is_detached_from_scope_and_sources(rig):
    component = rig.create()
    state = component.state()
    state["visited_points"][0]["x"] += 50
    state["source_sha256"].clear()
    state["expected_stars"].append("New")
    state["project_progress"]["collected"] = 30
    complete(component)
    assert component.status == "completed" and component.state()["project_progress"]["collected"] == 2


@pytest.mark.parametrize(
    "option,value",
    [
        ("max_seconds", 181),
        ("max_seconds", float("nan")),
        ("max_advances", True),
        ("max_advances", 3),
        ("anchor", (0, 1)),
    ],
)
def test_fixed_budgets_are_not_automatically_increased(rig, option, value):
    with pytest.raises(BrowserSafetyStop):
        rig.create(**{option: value})
    assert not rig.flags.calls
