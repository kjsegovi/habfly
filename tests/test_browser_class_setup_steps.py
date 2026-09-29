"""Injected cooperative class transport; no browser launch, network or inference."""

import json
import socket
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_star_preflight import save_capture, stellar, write

import habfly.browser_class_setup_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_numeric import digest, screen_identity
from habfly.browser_probe import BrowserProbeConfig
from habfly.browser_star_preflight import validate_star_class_source
from habfly.browser_stellar import CLASSES, map_stellar_capture
from habfly.runtime import read_trace

RATIONALE = "The caller explicitly identifies this visible star using reviewed course reference evidence; this setup transports that supplied decision without claiming learned or scientifically verified classification."


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("No network in cooperative class fixture")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    history = tmp_path / "run"
    fresh = history / "initial-star"
    page = SimpleNamespace(report=stellar(), selected=None, generation=0, writes=[], calls=[], fail=None)
    page.main_frame = object()
    frame = object()
    page.frames, page.context = [frame], SimpleNamespace(pages=[page])
    clock = SimpleNamespace(now=0)
    config = BrowserProbeConfig(
        url="http://localhost/activity", frames=[{"name": "simulation", "url": "https://fixture.invalid"}]
    )
    source_hash = save_capture(fresh / "stellar", page.report)

    def fresh_receipt(paint=None):
        page.selected = paint
        write(
            fresh / "confirmed.json",
            {
                "star": "Althinagon",
                "fresh_blank_numeric_answers_verified": True,
                "class_selection_verified": False,
                "answer_writes": 0,
                "action_source": "deterministic_navigation",
                "painted_stellar_class": paint,
            },
        )

    fresh_receipt()

    def choices():
        return {
            "selected": page.selected,
            "rendering": {c: {"selected": page.selected == c} for c in CLASSES},
        }

    class Handle:
        def __init__(self):
            self.generation = page.generation

        def evaluate(self, script, other):
            return self.generation == page.generation == other.generation

    class Session:
        def __init__(self, p, conf, emit, *, fresh_star=None):
            page.calls.append("construct")
            self.emit, self.fresh_star = emit, fresh_star
            self.frame, self.stopped = frame, False
            self._refresh()

        def _refresh(self):
            self.report = deepcopy(page.report)
            self.mapping = map_stellar_capture(
                self.report,
                capture_sha256=screen_identity(self.report),
                allow_color_selection=True,
                allow_main_sequence_fields=True,
            )
            self.choices = choices()
            self.handles = {c: Handle() for c in CLASSES}
            self.generation = page.generation

        def current(self):
            if self.stopped or page.generation != self.generation:
                raise BrowserSafetyStop("class_control_replaced")
            if screen_identity(page.report) != screen_identity(self.report) or choices() != self.choices:
                raise BrowserSafetyStop("stale_class_observation")

        def select_class(self, name, *, source, expected_previous=None, revision_reason=None):
            self.current()
            previous = page.selected
            if self.fresh_star:
                module.persist_json(
                    fresh / "class-selection-reserved.json",
                    {
                        "star": "Althinagon",
                        "source_capture_sha256": source_hash,
                        "previous_unconfirmed_paint": previous,
                        "selected_class": name,
                        "action_source": source,
                        "max_clicks": 1,
                        "automatic_retry": False,
                    },
                )
            extra = (
                {"previous": expected_previous, "revision_reason": revision_reason}
                if expected_previous
                else {}
            )
            self.emit(
                "action_proposed",
                {
                    "kind": "SELECT",
                    "target": "stellar_class",
                    "value": name,
                    "action_source": source,
                    "correctness_verified": False,
                    **extra,
                },
            )
            page.writes.append(("class", name))
            if page.fail == "class":
                raise RuntimeError("private-driver-secret http://localhost?session=secret")
            page.selected, page.report = name, stellar(conditional=name == "main_sequence")
            self._refresh()
            result = {
                "selected_class": name,
                "selection_source": source,
                "readback_verified": True,
                "correctness_verified": False,
                "rendering_sha256": digest(self.choices),
                "task_completed": False,
                **extra,
            }
            if self.fresh_star:
                result.update(fresh_star_capture_sha256=source_hash, previous_unconfirmed_paint=previous)
            self.emit("action_result", result)
            return result

        def select_prefix(self, prefix):
            self.current()
            self.emit(
                "action_proposed",
                {
                    "kind": "SELECT",
                    "target": "lifetime_prefix",
                    "value": prefix,
                    "action_source": "explicit_unit_transport",
                },
            )
            page.writes.append(("prefix", prefix))
            if page.fail == "prefix":
                raise RuntimeError("private-driver-secret http://localhost?session=secret")
            page.report = deepcopy(page.report)
            visible = page.report["frames"][0]
            combo = next(c for c in visible["controls"] if c["id"] == "simulation-0:c10")
            old = combo["accessibility"]
            new = old.replace("option [selected]", "option").replace(
                f'option "{prefix}"', f'option "{prefix}" [selected]'
            )
            combo["accessibility"], combo["value"] = new, prefix
            visible["accessibility"] = visible["accessibility"].replace(old, new)
            lifetime = next(c for c in visible["controls"] if c["id"] == "simulation-0:c9")
            lifetime["value"] = "0.000"
            lifetime["accessibility"] = '- textbox "0.00000": "0.000"'
            visible["accessibility"] = visible["accessibility"].replace(
                '- textbox "0.00000"', lifetime["accessibility"]
            )
            self._refresh()
            result = {
                "lifetime_prefix": prefix,
                "readback_verified": True,
                "task_completed": False,
                "blank_lifetime_initialized_to_zero": True,
            }
            self.emit("action_result", result)
            return result

    monkeypatch.setattr(module, "StellarSelectionSession", Session)
    monkeypatch.setattr(module, "inspect_page", lambda *_: deepcopy(page.report))
    monkeypatch.setattr(module, "save_probe", lambda report, path: save_capture(path, report))
    monkeypatch.setattr(module, "_visible_frame", lambda *_: True)
    monkeypatch.setattr(module, "read_class_choices", lambda _: (choices(), {c: Handle() for c in CLASSES}))
    return SimpleNamespace(
        history=history,
        fresh=fresh,
        page=page,
        config=config,
        clock=clock,
        fresh_receipt=fresh_receipt,
        events=[],
    )


def create(rig, selected="main_sequence", **kwargs):
    return module.FreshStarClassSteps(
        rig.page,
        rig.config,
        rig.history / "setup",
        run_history=rig.history,
        fresh_star=rig.fresh,
        selected_class=selected,
        reference_rationale=kwargs.pop("rationale", RATIONALE),
        lifetime_prefix=kwargs.pop("prefix", "Ga" if selected == "main_sequence" else None),
        emit=kwargs.pop("emit", rig.events.append),
        _clock=lambda: rig.clock.now,
        **kwargs,
    )


@pytest.mark.parametrize("selected", CLASSES)
@pytest.mark.parametrize("inherited", [False, True])
def test_all_classes_preserve_legacy_preflight_and_one_write_per_advance(rig, selected, inherited):
    rig.fresh_receipt(selected if inherited else None)
    steps = create(rig, selected)
    assert not rig.page.calls and not rig.page.writes
    while not steps.finished:
        before = len(rig.page.writes)
        steps.advance()
        assert len(rig.page.writes) - before <= 1
    assert steps.report["setup_verified"], steps.report
    assert not steps.report["task_completed"] and not steps.report["scientific_verified"]
    proof = validate_star_class_source(rig.history, steps.output / "class", "Althinagon", selected)
    assert proof["class_clicks"] == 1 + inherited
    assert proof == steps.class_source
    assert len(rig.page.writes) == 1 + inherited + (selected == "main_sequence")
    saved = read_trace(steps.output / "events.jsonl")
    assert [e.model_dump(mode="json") for e in saved] == rig.events
    assert all(e.version == 1 for e in saved)
    steps.advance()
    steps.close()
    assert len(rig.page.writes) == 1 + inherited + (selected == "main_sequence")
    with pytest.raises(BrowserSafetyStop):
        module.FreshStarClassSteps(
            rig.page,
            rig.config,
            rig.history / "retry",
            run_history=rig.history,
            fresh_star=rig.fresh,
            selected_class=selected,
            reference_rationale=RATIONALE,
            lifetime_prefix="Ga" if selected == "main_sequence" else None,
        )


@pytest.mark.parametrize("at", [0, 1, 2])
def test_abort_after_proposal_prevents_each_class_or_prefix_write(rig, at):
    rig.fresh_receipt("main_sequence")
    holder, count = {}, []

    def callback(event):
        rig.events.append(event)
        if event["event"] == "action_proposed":
            count.append(event)
            if len(count) == at + 1:
                holder["steps"].abort()

    steps = holder["steps"] = create(rig, emit=callback)
    for _ in range(4):
        steps.advance()
    assert steps.report["status"] == "aborted"
    assert len(rig.page.writes) == at
    assert not steps.report["setup_verified"]
    assert (
        steps.report["class_clicks_may_have_occurred"] + steps.report["prefix_selections_may_have_occurred"]
        == at
    )


@pytest.mark.parametrize(
    "change", ["source", "decision", "handle", "answer", "class", "time", "callback", "reentry"]
)
def test_post_callback_change_cannot_dispatch(rig, change):
    holder = {}

    def callback(event):
        if event["event"] != "action_proposed":
            return
        if change == "source":
            (rig.fresh / "stellar/observation.json").write_text("{}")
        elif change == "decision":
            (holder["steps"].output / "decision.json").write_text("{}")
        elif change == "handle":
            rig.page.generation += 1
        elif change == "answer":
            rig.page.report["frames"][0]["text"] += " changed"
        elif change == "class":
            rig.page.selected = "red_giant"
        elif change == "time":
            rig.clock.now = 180
        elif change == "callback":
            raise RuntimeError("private-driver-secret")
        else:
            holder["steps"].advance()

    steps = holder["steps"] = create(rig, emit=callback)
    steps.advance()
    assert steps.finished and not steps.report["setup_verified"]
    assert not rig.page.writes
    assert "private-driver-secret" not in (steps.output / "report.json").read_text()


@pytest.mark.parametrize("change", ["source", "handle", "frame", "paint"])
def test_inherited_main_exact_predispatch_guard(rig, change):
    rig.fresh_receipt("main_sequence")

    def callback(event):
        if event["event"] != "action_proposed":
            return
        if change == "source":
            rig.page.report["frames"][0]["text"] += " changed"
        elif change == "handle":
            rig.page.generation += 1
        elif change == "frame":
            rig.page.frames.clear()
        else:
            rig.page.selected = "red_giant"

    steps = create(rig, emit=callback)
    steps.advance()
    assert steps.finished and not rig.page.writes


@pytest.mark.parametrize("failure", ["class", "prefix"])
def test_uncertain_dispatch_is_terminal_and_prefix_does_not_invalidate_class_receipt(rig, failure):
    steps = create(rig)
    if failure == "prefix":
        steps.advance()
        assert steps.phase == "prefix_pending"
        validate_star_class_source(rig.history, steps.output / "class", "Althinagon", "main_sequence")
    rig.page.fail = failure
    steps.advance()
    count = len(rig.page.writes)
    steps.advance()
    assert steps.finished and not steps.report["setup_verified"] and len(rig.page.writes) == count
    assert "private-driver-secret" not in (steps.output / "report.json").read_text()
    if failure == "prefix":
        assert steps.prefix_claim.exists() and not (steps.output / "prefix/confirmed.json").exists()
        assert (
            validate_star_class_source(rig.history, steps.output / "class", "Althinagon", "main_sequence")
            == steps.class_source
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rationale": "Because main"},
        {"rationale": "x" * 60},
        {"rationale": RATIONALE + "\n"},
        {"prefix": None},
        {"prefix": "years"},
        {"max_advances": 0},
        {"max_advances": True},
        {"max_seconds": float("nan")},
        {"max_seconds": 601},
    ],
)
def test_invalid_explicit_decision_or_budget_has_no_page_actions(rig, kwargs):
    with pytest.raises(BrowserSafetyStop):
        create(rig, **kwargs)
    assert not rig.page.calls and not (rig.history / "setup").exists()


def test_non_main_rejects_prefix_and_no_class_is_inferred(rig):
    with pytest.raises(BrowserSafetyStop):
        create(rig, "white_dwarf", prefix="Ga")
    with pytest.raises(BrowserSafetyStop):
        create(rig, None)
    assert not rig.page.calls


def test_fixed_advance_limit_and_abort_before_start_never_retry(rig):
    rig.fresh_receipt("main_sequence")
    steps = create(rig, max_advances=1)
    steps.advance()
    steps.advance()
    assert steps.failure == "fresh_class_steps_advance_limit" and len(rig.page.writes) == 1
    assert not steps.report["setup_verified"]


def test_abort_before_any_native_read_is_durable(rig):
    steps = create(rig)
    steps.abort()
    steps.advance()
    assert not rig.page.calls and not rig.page.writes
    assert steps.report["status"] == "aborted" and (steps.output / "stopped.json").exists()
    with pytest.raises(BrowserSafetyStop, match="star_already_reserved"):
        module.FreshStarClassSteps(
            rig.page,
            rig.config,
            rig.history / "retry",
            run_history=rig.history,
            fresh_star=rig.fresh,
            selected_class="main_sequence",
            reference_rationale=RATIONALE,
            lifetime_prefix="Ga",
        )


def test_completed_prefix_has_separate_exact_immutable_receipt(rig):
    steps = create(rig)
    steps.advance()
    class_bytes = (steps.output / "class/after/observation.json").read_bytes()
    assert steps.phase == "prefix_pending" and not steps.finished
    steps.advance()
    assert (steps.output / "class/after/observation.json").read_bytes() == class_bytes
    prefix = json.loads((steps.output / "prefix/confirmed.json").read_text())
    assert prefix["lifetime_prefix"] == "Ga" and prefix["blank_lifetime_initialized_to_zero"]
    assert prefix["numeric_answer_writes"] == 0 and not prefix["task_completed"]
    assert len(list((steps.output / "prefix").glob("event-*.json"))) == 2


@pytest.mark.parametrize("at", ["class_result", "terminal"])
def test_callback_cannot_mutate_evidence_into_a_ready_report(rig, at):
    holder = {}

    def callback(event):
        if (at == "class_result" and event["event"] == "action_result") or (
            at == "terminal" and event["event"] == "episode_summary"
        ):
            (holder["steps"].output / "decision.json").write_text("{}")

    steps = holder["steps"] = create(rig, "white_dwarf", emit=callback)
    steps.advance()
    assert steps.report["status"] == "stopped" and not steps.report["setup_verified"]
    assert (steps.output / "stopped.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("star", "Different"),
        ("answer_writes", True),
        ("fresh_blank_numeric_answers_verified", False),
        ("class_selection_verified", True),
        ("painted_stellar_class", "unknown"),
    ],
)
def test_wrong_fresh_receipt_never_reads_native_page(rig, field, value):
    source = rig.fresh / "confirmed.json"
    receipt = json.loads(source.read_text())
    receipt[field] = value
    source.write_text(json.dumps(receipt))
    with pytest.raises(BrowserSafetyStop):
        create(rig)
    assert not rig.page.calls and not rig.page.writes


def test_owned_history_and_existing_prefix_claim_are_required(rig, tmp_path):
    outside = tmp_path / "outside"
    with pytest.raises(BrowserSafetyStop, match="invalid_owned_path"):
        module.FreshStarClassSteps(
            rig.page,
            rig.config,
            outside,
            run_history=rig.history,
            fresh_star=rig.fresh,
            selected_class="white_dwarf",
            reference_rationale=RATIONALE,
        )
    (rig.fresh / "lifetime-prefix-reserved.json").write_text("{}")
    with pytest.raises(BrowserSafetyStop, match="prefix_already_reserved"):
        create(rig)
    assert not rig.page.calls


def test_pinned_intermediate_capture_is_checked_before_second_class_click(rig):
    rig.fresh_receipt("white_dwarf")
    steps = create(rig, "white_dwarf")
    steps.advance()
    (steps.output / "class/intermediate/observation.json").write_text("{}")
    steps.advance()
    assert steps.finished and len(rig.page.writes) == 1 and not steps.report["setup_verified"]
