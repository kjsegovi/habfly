"""Offline public transcriptions and injected controls; no browser is launched."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_browser_numeric import config
from test_browser_project_inventory_steps import Events

import habfly.browser_assessment_panel_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_probe import save_probe
from habfly.browser_project_navigation import LIST_LABELS
from habfly.browser_stellar import SIMULATION_URL
from habfly.runtime import read_trace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def capture(mode="data_quality", *, funding=50000, quality=0, found=0, ack=False, extra_row="", row="Alpha"):
    header = f"Funding ${funding} Data Quality {quality}% Scavenger Hunt {found}/8"
    panel = {
        None: "",
        "data_quality": f"STARS 0% PLANETS 0% HABITABILITY 0% OVERALL {quality}% COST $100",
        "scavenger_hunt": "MAIN SEQUENCE WHITE DWARF RED GIANT SUPERGIANT GAS GIANT ICE GIANT TERRESTRIAL HABITABLE WORLD COST $100",
        "automation": "ASSESSMENT FIELD STARS SELECTED",
    }[mode]
    if ack:
        panel += " DATA QUALITY UPDATED"
    header += " " + panel
    buttons = ["EDIT DATA", "ASSESSMENT", "Save"]
    if mode is not None:
        buttons[:0] = ["Assess", "AUTOMATION", "SCAVENGER HUNT", "DATA QUALITY"]
    labels = "Observations Analyzed Data Star " + " ".join(LIST_LABELS["stellar"])
    row_text = f"{row} 0.045 370 5.68E-10 1 1 UV 1 Main Sequence 1 1 1 Ga" + extra_row
    footer = "viewing 1-1 of 1 total collected 1"
    atoms = [
        {"text": header},
        *[f'button "{b}"' for b in buttons[:-1]],
        {"text": labels},
        "img",
        {"text": row_text},
        {"text": footer},
        'button "Save"',
    ]
    text = "\n".join([header, *buttons[:-1], labels, row_text, footer, "Save"])
    return {
        "schema_version": 1,
        "mode": "read_only_browser_preflight",
        "actions_executed": 0,
        "allow_submission": False,
        "ignored_frame_urls": [],
        "outer_url": config().url,
        "outer_controls": [],
        "frames": [
            {
                "url": SIMULATION_URL,
                "id": "simulation-0",
                "text": text,
                "accessibility": yaml.safe_dump(atoms, sort_keys=False),
                "controls": [
                    {
                        "id": f"simulation-0:c{i}",
                        "role": "button",
                        "accessibility": f'- button "{b}"',
                        "enabled": True,
                        "actions": [],
                    }
                    for i, b in enumerate(buttons)
                ],
            }
        ],
    }


def source(history, report):
    directory = history / "inventory"
    hashes = {
        name + "/observation.json": save_probe(report, directory / name)["observation_sha256"]
        for name in ("before", "after")
    }
    counts, rows = module._capture_inventory(report)
    receipt = {
        "schema_version": 1,
        "mode": "read_only_collected_star_inventory",
        "authority": "visible_collected_list_readback",
        "section": "stellar",
        "viewing": counts,
        "total_collected": 1,
        "visible_row_count": 1,
        "collection_count_verified": True,
        "complete_visible_list_verified": True,
        "source_sha256": hashes,
        "rows": [
            {
                **r,
                "row_accessibility_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
                "source_sha256": hashes["after/observation.json"],
                "visible_name_box": {"x": 10, "y": 10, "width": 20, "height": 15},
            }
            for r in rows
        ],
        **dict.fromkeys(
            (
                "hidden_catalog_read",
                "learned_perception",
                "scientific_verified",
                "task_completed",
                "project_completed",
                "cross_session_persistence_verified",
                "config_mutated",
                "automatic_retry",
            ),
            False,
        ),
        **dict.fromkeys(
            (
                "browser_actions",
                "navigation_clicks",
                "scrolls",
                "answer_writes",
                "save_clicks",
                "deletion_clicks",
                "assessment_clicks",
                "submission_clicks",
            ),
            0,
        ),
    }
    write(directory / "confirmed.json", receipt)
    return directory


@pytest.fixture
def rig(tmp_path, monkeypatch):
    flags = SimpleNamespace(
        mode="data_quality",
        funding=50000,
        quality=0,
        found=0,
        ack=False,
        extra_row="",
        row="Alpha",
        reads=0,
        clicks=[],
        exposed=True,
        enabled=True,
        native_panel=None,
        no_change=False,
        open_mode="data_quality",
        click_hook=None,
        read_hook=None,
        now=0,
        cancel=False,
        outer=False,
        duplicate=False,
        wrong_tag=False,
        url_changed=False,
    )
    page = Events()
    page.context = Events()
    page.context.pages = [page]
    page.main_frame = object()
    frame = SimpleNamespace(url=SIMULATION_URL, parent_frame=page.main_frame)
    page.frames = [frame]
    buttons = {}

    def current():
        report = capture(
            flags.mode,
            funding=flags.funding,
            quality=flags.quality,
            found=flags.found,
            ack=flags.ack,
            extra_row=flags.extra_row,
            row=flags.row,
        )
        if flags.outer:
            report["outer_controls"] = [{"changed": True}]
        if flags.url_changed:
            report["outer_url"] += "/different-course"
        return report

    class Button:
        def __init__(self, label):
            self.label = label

        def is_visible(self):
            return True

        def is_enabled(self):
            return flags.enabled

        def element_handle(self, timeout=None):
            return self

        def evaluate(self, code, other=None):
            if "isConnected" in code:
                return self is other and buttons[self.label] is self
            if "tagName" in code:
                return not flags.wrong_tag
            return {"x": 10, "y": 10, "width": 30, "height": 20}

        def click(self, timeout):
            assert self.label in module.ALLOW
            root = tmp_path / "run"
            assert list(root.glob("*-dispatch.json")) and list(root.glob("*-reserved.json"))
            flags.clicks.append(self.label)
            if flags.click_hook:
                flags.click_hook()
            if not flags.no_change:
                flags.mode = (
                    flags.open_mode
                    if self.label == "ASSESSMENT"
                    else next(k for k, v in module.LABELS.items() if v == self.label)
                )

    for label in module.ALLOW:
        buttons[label] = Button(label)

    def roles(role, name, exact):
        assert role == "button" and exact and name in module.ALLOW
        return SimpleNamespace(all=lambda: [buttons[name]] * (2 if flags.duplicate else 1))

    frame.get_by_role = roles

    def native(code):
        report = current() if flags.native_panel is None else capture(flags.native_panel)
        return [{"text": report["frames"][0]["text"], "boxes": [{"x": 1, "y": 1, "width": 1, "height": 1}]}]

    frame.locator = lambda name: SimpleNamespace(evaluate=native)

    def inspect(page, settings):
        flags.reads += 1
        if flags.read_hook:
            flags.read_hook()
        return current()

    monkeypatch.setattr(module, "inspect_page", inspect)
    monkeypatch.setattr(module, "_visible_rows", lambda *args: None)
    monkeypatch.setattr(module, "_outer_exposed", lambda *args: flags.exposed)
    monkeypatch.setattr(module, "row_icon_exposed", lambda *args: flags.exposed)
    monkeypatch.setattr(module.time, "monotonic", lambda: flags.now)
    directory = source(tmp_path, current())
    events = []

    def create(**options):
        return module.AssessmentPanelSteps(
            page,
            config(),
            tmp_path / options.pop("output", "run"),
            run_history=tmp_path,
            inventory_dir=directory,
            inventory_sha256=sha(directory / "confirmed.json"),
            kind=options.pop("kind", "scavenger_hunt"),
            emit=options.pop("emit", events.append),
            cancelled=lambda: flags.cancel,
            **options,
        )

    return SimpleNamespace(
        root=tmp_path,
        page=page,
        frame=frame,
        flags=flags,
        buttons=buttons,
        source=directory,
        events=events,
        create=create,
    )


@pytest.mark.parametrize(
    "initial,requested,labels",
    [
        ("data_quality", "data_quality", []),
        ("data_quality", "scavenger_hunt", ["SCAVENGER HUNT"]),
        ("scavenger_hunt", "data_quality", ["DATA QUALITY"]),
        (None, "scavenger_hunt", ["ASSESSMENT", "SCAVENGER HUNT"]),
    ],
)
def test_offline_constructor_and_one_native_menu_click_per_advance(rig, initial, requested, labels):
    rig.flags.mode = initial
    owner = rig.create(kind=requested)
    assert rig.flags.reads == 0 and not rig.flags.clicks and not rig.page.listeners
    for _ in range(3):
        before = len(rig.flags.clicks)
        owner.advance()
        assert len(rig.flags.clicks) - before <= 1
        if owner.finished:
            break
    assert owner.finished and owner.report["panel_verified"], owner.report
    assert rig.flags.clicks == labels
    assert owner.report["assessment"]["mode"] == requested
    assert all(owner.report[k] == 0 for k in module.ZERO)
    assert not owner.report["assessment_confirmed"] and not owner.report["task_completed"]
    assert not any(rig.page.listeners.values()) and not any(rig.page.context.listeners.values())
    assert [
        event.model_dump(mode="json") for event in read_trace(owner.output / "events.jsonl")
    ] == rig.events
    calls = list(rig.flags.clicks)
    owner.advance()
    owner.close()
    assert rig.flags.clicks == calls


def test_funding_is_current_read_baseline_not_old_inventory(rig):
    rig.flags.funding = 49900
    rig.flags.quality = 25
    owner = rig.create()
    owner.advance()
    owner.advance()
    assert owner.report["panel_verified"] and owner.report["assessment"]["funding"] == 49900


@pytest.mark.parametrize(
    "change",
    [
        "funding",
        "quality",
        "found",
        "row",
        "outer",
        "source",
        "unknown_frame",
        "url",
        "popup",
        "frame",
        "ack",
        "hidden",
        "disabled",
        "duplicate",
        "tag",
        "replacement",
    ],
)
def test_drift_and_unexposed_controls_stop_before_click(rig, change):
    owner = rig.create()
    owner.advance()
    if change == "funding":
        rig.flags.funding -= 100
    elif change == "quality":
        rig.flags.quality = 25
    elif change == "found":
        rig.flags.found = 1
    elif change == "row":
        rig.flags.row = "Beta"
    elif change == "outer":
        rig.flags.outer = True
    elif change == "url":
        rig.flags.url_changed = True
    elif change == "source":
        write(rig.source / "confirmed.json", {})
    elif change == "popup":
        rig.page.context.pages.append(object())
    elif change == "frame":
        rig.page.frames.append(object())
    elif change == "ack":
        rig.flags.ack = True
    elif change == "hidden":
        rig.flags.exposed = False
    elif change == "disabled":
        rig.flags.enabled = False
    elif change == "duplicate":
        rig.flags.duplicate = True
    elif change == "tag":
        rig.flags.wrong_tag = True
    elif change == "replacement":
        old = rig.buttons["SCAVENGER HUNT"]
        owner._callback = lambda event: (
            rig.buttons.__setitem__("SCAVENGER HUNT", type(old)(old.label))
            if event["event"] == "action_proposed"
            else None
        )
    else:
        owner._config["frames"] = []
    owner.advance()
    assert owner.finished and owner.phase == "stopped" and not rig.flags.clicks


@pytest.mark.parametrize("change", ["funding", "row", "none", "raise"])
def test_postclick_uncertainty_is_recorded_without_retry(rig, change):
    owner = rig.create()
    owner.advance()

    def hook():
        if change == "funding":
            rig.flags.funding -= 100
        elif change == "row":
            rig.flags.row = "Beta"
        elif change == "none":
            rig.flags.no_change = True
        else:
            raise RuntimeError("untrusted driver contents")

    rig.flags.click_hook = hook
    owner.advance()
    assert owner.phase == "stopped" and owner.dispatch_attempts == 1
    assert len(rig.flags.clicks) == 1 and not (owner.output / "confirmed.json").exists()
    owner.advance()
    assert len(rig.flags.clicks) == 1


@pytest.mark.parametrize("boundary", ["initial", "proposal", "result", "summary"])
def test_callback_abort_never_allows_later_navigation(rig, boundary):
    owner = rig.create()
    if boundary == "initial":
        owner.abort()
    else:
        events = {"proposal": "action_proposed", "result": "action_result", "summary": "episode_summary"}
        owner._callback = lambda event: owner.abort() if event["event"] == events[boundary] else None
        owner.advance()
        owner.advance()
    assert owner.phase == "stopped" and not (owner.output / "confirmed.json").exists()
    assert len(rig.flags.clicks) == (0 if boundary in {"initial", "proposal"} else 1)


def test_expired_paused_deadline_never_clicks(rig):
    owner = rig.create(max_seconds=5)
    owner.advance()
    rig.flags.now = 5
    owner.advance()
    assert owner.failure == "assessment_panel_time_limit" and not rig.flags.clicks


def test_visible_scavenger_categories_in_rows_never_define_panel():
    report = capture(
        None,
        extra_row=" MAIN SEQUENCE WHITE DWARF RED GIANT SUPERGIANT GAS GIANT ICE GIANT TERRESTRIAL HABITABLE WORLD",
    )
    assert module._public_panel(report)[1] is None
    with pytest.raises(BrowserSafetyStop):
        module._public_panel(capture("automation"))
    with pytest.raises(BrowserSafetyStop):
        module._public_panel(capture(ack=True))


def test_native_exposed_panel_must_match_accessibility(rig):
    rig.flags.native_panel = "scavenger_hunt"
    owner = rig.create()
    owner.advance()
    assert owner.phase == "stopped" and not rig.flags.clicks


def test_only_three_grounded_buttons_can_ever_be_bound(rig):
    owner = rig.create()
    for name in ("ASSESS", "Assess", "Save", "Update Score", "Submit Project", "AUTOMATION", "EDIT DATA"):
        with pytest.raises(BrowserSafetyStop, match="forbidden_button"):
            owner._button(name)
    assert not rig.flags.clicks
    owner.close()


@pytest.mark.parametrize(
    "event,path",
    [
        ("action_proposed", "navigation-00-reserved.json"),
        ("action_result", "navigation-00-after/observation.json"),
        ("episode_summary", "verified/observation.json"),
    ],
)
def test_callback_cannot_change_recorded_navigation_evidence(rig, event, path):
    owner = rig.create()
    owner._callback = lambda value: write(owner.output / path, {}) if value["event"] == event else None
    owner.advance()
    owner.advance()
    assert owner.phase == "stopped" and not (owner.output / "confirmed.json").exists()
    assert len(rig.flags.clicks) == (0 if event == "action_proposed" else 1)


def test_callback_failure_is_sanitized_and_source_scope_is_pinned(rig):
    def fail(event):
        raise ValueError("private callback detail")

    owner = rig.create(emit=fail)
    assert owner.finished and owner.failure == "assessment_panel_event_forwarding_failed"
    assert "private callback detail" not in str(owner.report)
    assert rig.flags.reads == 0
    next_owner = rig.create(output="next")
    write(next_owner.output / "scope.json", {})
    next_owner.advance()
    assert next_owner.phase == "stopped" and rig.flags.reads == 0


@pytest.mark.parametrize(
    "relative,count",
    [
        ("inventory-251", 7),
        ("project-owner-260/collection/inventory", 8),
    ],
)
def test_actual_public_inventory_panel_maps_offline_without_modifying_sources(relative, count):
    root = Path(__file__).resolve().parents[1] / "experiments/full-stellar-probe/20260926-005"
    path = root / relative / "after/observation.json"
    if not path.exists():
        pytest.skip("Local public capture is not distributed")
    before = path.read_bytes()
    baseline, panel, counts, rows, buttons = module._public_panel(json.loads(before))
    assert panel.mode == "data_quality" and panel.acknowledgement is None
    assert baseline["funding"] == 49800 and counts["total"] == len(rows) == count
    assert module.ALLOW <= set(buttons)
    assert path.read_bytes() == before
