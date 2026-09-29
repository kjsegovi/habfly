"""Injected offline inventory scheduling: no browser or network is opened."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

import habfly.browser_project_inventory_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_probe import save_probe
from habfly.browser_stellar import SIMULATION_URL


class Config:
    def __init__(self):
        self.frames = [SimpleNamespace(url=SIMULATION_URL, count=1, required_text=["original labels"])]

    def model_copy(self, deep):
        assert deep
        return deepcopy(self)

    def model_dump(self, mode):
        assert mode == "json"
        return [vars(rule).copy() for rule in self.frames]


class Events:
    def __init__(self):
        self.listeners = {}

    def on(self, key, handler):
        self.listeners.setdefault(key, []).append(handler)

    def remove_listener(self, key, handler):
        self.listeners[key].remove(handler)


def capture(surface="detail", section="planet", star="Fixture"):
    return {
        "mode": "injected_offline_fixture",
        "ignored_frame_urls": [],
        "outer_controls": [],
        "fixture_view": {
            "surface": surface,
            "section": section,
            "star": star if surface == "detail" else None,
        },
        "frames": [
            {
                "url": SIMULATION_URL,
                "text": "fixture unchanged visible fields",
                "controls": [],
                "accessibility": "",
            }
        ],
    }


def sha(path):
    return module._sha(path.read_bytes())


@pytest.fixture
def rig(tmp_path, monkeypatch):
    page = Events()
    page.frames = [object()]
    page.context = Events()
    page.context.pages = [page]
    settings = Config()
    flags = SimpleNamespace(
        calls=[],
        now=0,
        cancel=False,
        navigation_hook=None,
        read_hook=None,
        verify_hook=None,
        list_section="planet",
        fail_navigation=False,
        inventory_failure=None,
    )
    monkeypatch.setattr(module.time, "monotonic", lambda: flags.now)
    monkeypatch.setattr(module, "project_view", lambda report: deepcopy(report["fixture_view"]))

    def source(name="workflow", mode="no_planet_visible_workflow_readback"):
        directory = tmp_path / name
        directory.mkdir()
        upstream = directory / "upstream.json"
        upstream.write_text(json.dumps({"fixture_only": True, "never_a_live_workflow": True}))
        screens = {}
        for section in module.MODES[mode]:
            value = capture(section=section)
            save_probe(value, directory / (section + "-readback") / "verified")
            screens[section] = module.screen_identity(value)
        receipt = {
            "schema_version": 1,
            "mode": mode,
            "authority": "visible_workflow_readback",
            "star": "Fixture",
            "task_completed": True,
            "save_acknowledgement_verified": True,
            "current_screen_sha256": screens,
            "source_sha256": {str(upstream.relative_to(tmp_path)): sha(upstream)},
            **{k: 0 for k in module.ZERO},
            **{k: False for k in module.FALSE},
        }
        module.persist_json(directory / "confirmed.json", receipt)
        page.capture = capture(section=module.MODES[mode][-1])
        return directory

    directory = source()

    def inspect(page, config):
        flags.calls.append("read")
        if flags.read_hook:
            flags.read_hook()
        return deepcopy(page.capture)

    def navigate(page, config, output, destination, expected_star=None):
        flags.calls.append(destination)
        assert destination in {"list", "stellar"}
        assert expected_star == ("FIXTURE" if destination == "list" else None)
        output.mkdir()
        before = deepcopy(page.capture)
        save_probe(before, output / "before")
        save_probe(before, output / "pre-click")
        intent = {"mode": "bounded_project_navigation", "max_clicks": 1, "automatic_retry": False}
        module.persist_json(output / "reserved.json", intent)
        if flags.navigation_hook:
            flags.navigation_hook()
        if flags.fail_navigation:
            module.persist_json(output / "stopped.json", {"click_may_have_occurred": True})
            raise BrowserSafetyStop("project_navigation_target_or_screen_changed")
        page.capture = capture("list", flags.list_section if destination == "list" else "stellar")
        save_probe(page.capture, output / "after")
        receipt = {
            **intent,
            "destination": destination,
            "from": before["fixture_view"],
            "to": page.capture["fixture_view"],
            "destination_verified": True,
            "navigation_clicks": 1,
            "task_completed": False,
            "answer_writes": 0,
            "collection_clicks": 0,
            "save_clicks": 0,
            "assessment_clicks": 0,
            "submission_clicks": 0,
        }
        module.persist_json(output / "confirmed.json", receipt)
        return receipt

    def verify(page, config, output, expected_stars):
        flags.calls.append("verify")
        assert expected_stars == ("FIXTURE", "Other")
        output.mkdir()
        if flags.verify_hook:
            flags.verify_hook()
        if flags.inventory_failure:
            module.persist_json(output / "stopped.json", {"reason": flags.inventory_failure})
            raise BrowserSafetyStop(flags.inventory_failure)
        hashes = {
            name + "/observation.json": save_probe(page.capture, output / name)["observation_sha256"]
            for name in ("before", "after")
        }
        receipt = {
            "schema_version": 1,
            "mode": "read_only_collected_star_inventory",
            "authority": "visible_collected_list_readback",
            "section": "stellar",
            "collection_count_verified": True,
            "complete_visible_list_verified": True,
            "expected_stars_verified": True,
            "task_completed": False,
            "project_completed": False,
            "rows": [{"name": name} for name in expected_stars],
            "total_collected": 2,
            "source_sha256": hashes,
            "fixture_only": True,
        }
        module.persist_json(output / "confirmed.json", receipt)
        return receipt

    monkeypatch.setattr(module, "inspect_page", inspect)
    monkeypatch.setattr(module, "navigate_project", navigate)
    monkeypatch.setattr(module, "verify_project_inventory", verify)

    def create(name="run", **overrides):
        options = {
            "run_history": tmp_path,
            "workflow_dir": directory,
            "workflow_sha256": sha(directory / "confirmed.json"),
            "expected_star": "FIXTURE",
            "expected_stars": ["FIXTURE", "Other"],
            "cancelled": lambda: flags.cancel,
        }
        options.update(overrides)
        return module.ProjectInventorySteps(page, settings, tmp_path / name, **options)

    return SimpleNamespace(
        create=create,
        source=source,
        directory=directory,
        root=tmp_path,
        page=page,
        flags=flags,
        config=settings,
    )


def events(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def test_offline_constructor_and_one_bounded_adapter_per_advance(rig):
    emitted = []

    def emit(kind, payload):
        event = events(rig.root / "run")[-1]
        assert (kind, payload) == (event["event"], event["payload"])
        emitted.append(event)

    before_config = rig.config.model_dump(mode="json")
    component = rig.create(emit=emit)
    assert rig.flags.calls == [] and not rig.page.listeners
    assert component.state()["phase"] == "to_list"
    for expected in ["to_stellar", "verify_inventory", "inventory_verified"]:
        old = list(rig.flags.calls)
        component.advance()
        assert component.phase == expected
        assert len([call for call in rig.flags.calls[len(old) :] if call != "read"]) == 1
        assert not component.state()["task_completed"]
    assert component.finished and component.status == "completed"
    assert component.navigation_clicks == component.navigation_attempts == 2
    assert component.advances == 3 and component.state()["collection_count_verified"]
    assert rig.config.model_dump(mode="json") == before_config
    child = json.loads((rig.root / "run/inventory/confirmed.json").read_text())
    assert component._inventory == child and component.state()["inventory_sha256"] == sha(
        rig.root / "run/inventory/confirmed.json"
    )
    assert component.report["journal_writes"] == 0 and not component.report["project_completed"]
    assert emitted == events(rig.root / "run")
    assert [e["sequence"] for e in emitted] == list(range(len(emitted)))
    for relative, checksum in component.report["source_sha256"].items():
        assert sha(rig.root / relative) == checksum
    previous = (rig.root / "run/events.jsonl").read_bytes(), list(rig.flags.calls)
    component.advance()
    assert component.close() is component.report and component.abort() is component.report
    assert ((rig.root / "run/events.jsonl").read_bytes(), rig.flags.calls) == previous
    assert all(not callbacks for callbacks in rig.page.listeners.values())


@pytest.mark.parametrize("mode", list(module.MODES))
def test_actual_workflow_modes_and_already_stellar_list_skip_extra_navigation(rig, mode):
    directory = rig.source("other-workflow", mode)
    rig.flags.list_section = "stellar"
    component = rig.create(workflow_dir=directory, workflow_sha256=sha(directory / "confirmed.json"))
    component.advance()
    assert component.phase == "verify_inventory"
    component.advance()
    assert component.status == "completed" and component.advances == 2
    assert [c for c in rig.flags.calls if c != "read"] == ["list", "verify"]


@pytest.mark.parametrize("where", [0, 1, 2])
def test_abort_is_sticky_and_performs_no_later_page_calls(rig, where):
    component = rig.create()
    for _ in range(where):
        component.advance()
    old = list(rig.flags.calls)
    component.abort()
    component.advance()
    component.close()
    assert rig.flags.calls == old and component.status == "aborted"
    assert component.state()["inventory_dir"] is None
    assert not component.state()["collection_count_verified"]


@pytest.mark.parametrize("where", ["callback", "after_read", "after_navigation", "after_inventory"])
def test_cooperative_cancellation_never_calls_next_adapter(rig, where):
    def emit(kind, payload):
        if where == "callback" and payload.get("scheduled_call"):
            rig.flags.cancel = True

    component = rig.create(emit=emit)
    if where == "after_read":
        rig.flags.read_hook = lambda: setattr(rig.flags, "cancel", True)
    elif where == "after_navigation":
        rig.flags.navigation_hook = lambda: setattr(rig.flags, "cancel", True)
    elif where == "after_inventory":
        rig.flags.verify_hook = lambda: setattr(rig.flags, "cancel", True)
    for _ in range(4):
        component.advance()
    assert component.status == "aborted" and not component.state()["collection_count_verified"]
    assert component.state()["inventory_dir"] is None
    assert (
        len([c for c in rig.flags.calls if c != "read"])
        == {"callback": 0, "after_read": 0, "after_navigation": 1, "after_inventory": 3}[where]
    )


@pytest.mark.parametrize(
    "failure",
    [
        "source",
        "capture",
        "manifest",
        "failed_workflow",
        "config",
        "current_answer",
        "current_star",
        "frame",
        "popup",
        "dialog",
        "page",
    ],
)
def test_changed_sources_or_current_context_stop_before_next_navigation(rig, failure):
    component = rig.create()
    if failure in {"frame", "popup", "dialog", "page"}:
        component.advance()
    before = len([c for c in rig.flags.calls if c != "read"])
    if failure == "source":
        (rig.directory / "upstream.json").write_text("changed")
    elif failure == "capture":
        (rig.directory / "planet-readback/verified/observation.json").write_text("changed")
    elif failure == "manifest":
        (rig.directory / "planet-readback/verified/manifest.json").write_text("changed")
    elif failure == "failed_workflow":
        (rig.directory / "stopped.json").write_text("{}")
    elif failure == "config":
        component.config.frames[0].count = 2
    elif failure == "current_answer":
        rig.page.capture["frames"][0]["text"] = "changed answer"
    elif failure == "current_star":
        rig.page.capture["fixture_view"]["star"] = "Other"
    elif failure == "frame":
        rig.page.frames.append(object())
    elif failure == "popup":
        rig.page.context.listeners["page"][0](None)
    elif failure == "dialog":
        rig.page.listeners["dialog"][0](None)
    elif failure == "page":
        rig.page.context.pages.append(object())
    component.advance()
    assert component.status == "stopped" and not component.state()["collection_count_verified"]
    assert len([c for c in rig.flags.calls if c != "read"]) == before


@pytest.mark.parametrize(
    "reason",
    [
        "project_inventory_incomplete_visible_list",
        "project_inventory_duplicate_or_unexposed_name",
        "project_inventory_expected_stars_mismatch",
    ],
)
def test_partial_or_unverified_list_never_claims_collection_or_paginates(rig, reason):
    component = rig.create()
    rig.flags.inventory_failure = reason
    for _ in range(5):
        component.advance()
    assert component.status == "stopped" and component.state()["inventory_dir"] is None
    assert not component.state()["collection_count_verified"]
    assert rig.flags.calls.count("verify") == 1
    assert not (rig.root / "run/inventory/confirmed.json").exists()
    assert component.failure == (
        "project_inventory_steps_unsupported_pagination" if "incomplete_visible" in reason else reason
    )


def test_uncertain_navigation_is_not_retried_and_is_not_reported_as_verified(rig):
    component = rig.create()
    rig.flags.fail_navigation = True
    component.advance()
    component.advance()
    assert rig.flags.calls == ["read", "list"]
    assert component.state()["navigation_may_have_occurred"]
    assert component.navigation_clicks == 0 and component.navigation_attempts == 1


@pytest.mark.parametrize("when", ["paused", "guard", "adapter"])
def test_fixed_deadline_counts_pauses_and_stops_without_budget_extension(rig, when):
    component = rig.create(max_seconds=10)
    if when == "paused":
        rig.flags.now = 10
    elif when == "guard":
        rig.flags.read_hook = lambda: setattr(rig.flags, "now", 10)
    else:
        rig.flags.navigation_hook = lambda: setattr(rig.flags, "now", 10)
    component.advance()
    assert component.status == "stopped" and component.failure.endswith("time_limit")
    assert component.report["max_seconds"] == 10
    assert len([c for c in rig.flags.calls if c != "read"]) == int(when == "adapter")


@pytest.mark.parametrize(
    "override",
    [
        {"expected_stars": None},
        {"expected_stars": []},
        {"expected_stars": ["Fixture", "fixture"]},
        {"expected_stars": ["Other"]},
        {"expected_stars": ["Fixture", " bad"]},
        {"expected_star": "Unknown"},
        {"expected_star": 4},
        {"max_seconds": True},
        {"max_seconds": float("nan")},
        {"max_seconds": 181},
        {"max_seconds": 0},
        {"workflow_sha256": "0" * 64},
        {"workflow_sha256": "bad"},
    ],
)
def test_invalid_inputs_fail_offline_without_output_or_page_actions(rig, override):
    with pytest.raises(BrowserSafetyStop):
        rig.create(**override)
    assert not rig.flags.calls and not (rig.root / "run").exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "flag",
        "mode",
        "star",
        "sources",
        "source_hash",
        "missing_capture",
        "capture_hash",
        "symlink",
        "escape",
    ],
)
def test_task_completed_flag_without_owned_full_hash_and_capture_proofs_is_rejected(rig, corruption):
    path = rig.directory / "confirmed.json"
    receipt = json.loads(path.read_text())
    if corruption == "flag":
        receipt["scientific_verified"] = True
    elif corruption == "mode":
        receipt["mode"] = "unsupported"
    elif corruption == "star":
        receipt["star"] = "Wrong"
    elif corruption == "sources":
        receipt["source_sha256"] = {}
    elif corruption == "source_hash":
        receipt["source_sha256"]["workflow/upstream.json"] = "0" * 64
    elif corruption == "missing_capture":
        receipt["current_screen_sha256"].pop("planet")
    elif corruption == "capture_hash":
        receipt["current_screen_sha256"]["planet"] = "0" * 64
    elif corruption == "symlink":
        link = rig.directory / "symlink.json"
        link.symlink_to(rig.directory / "upstream.json")
        receipt["source_sha256"] = {"workflow/symlink.json": sha(link)}
    elif corruption == "escape":
        receipt["source_sha256"] = {"../outside.json": "0" * 64}
    path.write_text(json.dumps(receipt))
    with pytest.raises(BrowserSafetyStop):
        rig.create()
    assert not rig.flags.calls and not (rig.root / "run").exists()


def test_callback_failure_redacted_and_reentrant_advance_stops(rig):
    holder = {}

    def emit(kind, payload):
        if payload.get("scheduled_call"):
            holder["component"].advance()

    component = rig.create(emit=emit)
    holder["component"] = component
    component.advance()
    assert component.finished and not rig.flags.calls
    assert component.failure == "project_inventory_steps_reentrant_advance"


def test_callback_failure_never_exposes_private_text_or_runs_navigation(rig):
    def emit(kind, payload):
        if payload.get("scheduled_call"):
            raise RuntimeError("secret private credentials")

    component = rig.create(emit=emit)
    component.advance()
    assert component.finished and component._forward_failed and not rig.flags.calls
    assert "secret" not in (rig.root / "run/events.jsonl").read_text()


def test_final_callback_failure_invalidates_handoff_without_changing_inventory_child(rig):
    def emit(kind, payload):
        if kind == "episode_summary":
            raise RuntimeError("private exception details")

    component = rig.create(emit=emit)
    for _ in range(3):
        component.advance()
    assert component.status == "stopped" and component.state()["inventory_dir"] is None
    assert not component.state()["collection_count_verified"]
    child = json.loads((rig.root / "run/inventory/confirmed.json").read_text())
    assert child["collection_count_verified"]
    assert (rig.root / "run/invalidated.json").is_file()
    assert "private" not in (rig.root / "run/stopped.json").read_text()


def test_state_reads_are_detached_and_guard_and_navigation_proofs_remain_pinned(rig):
    component = rig.create()
    detached = component.state()
    detached["expected_stars"].append("Never Invent A Star")
    assert component.state()["expected_stars"] == ["FIXTURE", "Other"]
    component.advance()
    assert "run/guard-1/observation.json" in component.book.hashes
    assert "run/to-list/pre-click/observation.json" in component.book.hashes
    assert "run/to-list/reserved.json" in component.book.hashes
    (rig.root / "run/guard-1/observation.json").write_text("changed evidence")
    old = list(rig.flags.calls)
    component.advance()
    assert component.status == "stopped" and rig.flags.calls == old


def test_source_change_during_inventory_preserves_child_but_does_not_authorize_handoff(rig):
    component = rig.create()
    rig.flags.verify_hook = lambda: (rig.directory / "upstream.json").write_text("changed source")
    for _ in range(3):
        component.advance()
    assert component.status == "stopped" and component.state()["inventory_sha256"] is None
    assert not component.state()["collection_count_verified"]
    assert (rig.root / "run/inventory/confirmed.json").is_file()
