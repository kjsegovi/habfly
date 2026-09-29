"""Offline timer readiness must not spend browser/learned-work advances."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_planet_window import make as make_window
from test_browser_planet_window import scheduled  # noqa: F401
from test_browser_project_steps import ready as make_owner
from test_browser_project_steps import rig as owner_rig  # noqa: F401
from test_browser_star_session import create as make_star
from test_browser_star_session import rig as star_rig  # noqa: F401
from test_planet_window_policy import crop, labels

import habfly.browser_planet_window as window_module
import habfly.browser_star_session as star_module
from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_progress
from habfly.browser_planet_window import PlanetWindowSession
from habfly.browser_star_session import BrowserStarSession


def test_window_hint_is_read_only_and_due_at_original_poll(scheduled):  # noqa: F811
    window = make_window(scheduled)
    assert window.ready_to_advance() is True
    window.advance()
    deadline, next_poll = window.deadline, window.next_poll
    events, calls = list(scheduled.events), list(scheduled.calls)
    files = {p: p.read_bytes() for p in window.output.rglob("*") if p.is_file()}
    for i in range(5000):
        scheduled.clock[0] = 100 + i / 1000
        assert window.ready_to_advance() is False
    assert (window.deadline, window.next_poll, window.polls) == (deadline, next_poll, 0)
    assert scheduled.events == events and scheduled.calls == calls
    assert files == {p: p.read_bytes() for p in window.output.rglob("*") if p.is_file()}
    scheduled.clock[0] = next_poll
    assert window.ready_to_advance() is True
    window.advance()
    assert window.polls == 1 and not window.ready_to_advance()


def test_window_deadline_is_due_even_when_next_poll_is_later(scheduled):  # noqa: F811
    window = make_window(scheduled, max_seconds=30)
    window.advance()
    window.next_poll = window.deadline + 5
    scheduled.clock[0] = window.deadline - 0.001
    assert not window.ready_to_advance()
    scheduled.clock[0] = window.deadline
    assert window.ready_to_advance()
    assert window.advance()["failure_reason"] == "planet_window_time_limit"
    assert scheduled.calls == [("start", 5000)] and window.polls == 0
    assert window.ready_to_advance()  # terminal state can be consumed by its parent


@pytest.mark.parametrize("phase", ["stellar_numeric", "stellar_color", "navigate_planet", "planet_window"])
def test_star_hint_defaults_to_due_without_a_timer_contract(star_rig, phase):  # noqa: F811
    star = make_star(star_rig)
    star.phase = phase
    star.component = SimpleNamespace()
    assert star.ready_to_advance() is True
    assert not star_rig.calls


def test_star_propagates_only_window_timer_and_checks_sources(star_rig, monkeypatch):  # noqa: F811
    star = make_star(star_rig)
    star.component = SimpleNamespace(ready_to_advance=lambda: False)
    assert star.ready_to_advance() is True  # numeric/color work cannot be skipped
    star.phase = "planet_window"
    assert star.ready_to_advance() is False
    monkeypatch.setattr(star_module, "validate_star_class_source", lambda *_: {"star": "SWAPPED"})
    with pytest.raises(BrowserSafetyStop, match="star_session_class_source_changed"):
        star.ready_to_advance()
    assert not star_rig.calls


@pytest.mark.parametrize("ready", [False, 0, lambda: None, lambda: 1, lambda: "false"])
def test_star_rejects_malformed_timer_contract(star_rig, ready):  # noqa: F811
    star = make_star(star_rig)
    star.phase = "planet_window"
    star.component = SimpleNamespace(ready_to_advance=ready)
    with pytest.raises(BrowserSafetyStop, match="star_session_invalid_readiness"):
        star.ready_to_advance()


@pytest.fixture
def nested(owner_rig, monkeypatch):  # noqa: F811
    """Real owner → real star → real window; browser work is explicitly injected."""
    rig = owner_rig
    calls, stars, windows = [], [], []
    monkeypatch.setattr(window_module.time, "monotonic", lambda: rig.clock.now)
    monkeypatch.setattr(star_module, "validate_star_class_source", lambda *_: deepcopy(rig.source))

    def start(*args, **kwargs):
        calls.append("start")
        assert kwargs["days"] == 5000
        return {"star": "ALPHA", "observation_completed": False}

    def capture(_page, _config, output, **kwargs):
        calls.append("capture")
        assert kwargs["requested_days"] == 5000
        output.mkdir()
        # Only the native capture seam is injected. Use a genuine synthetic
        # partial PNG and exposed axes so both production readiness and the
        # frozen analysis run normally, including their typed hash contract.
        png = crop(end=130)
        time, flux = labels(5000)
        report = {
            **trace_progress(png, time, requested_days=5000),
            "star": "ALPHA",
            "time_axis_labels": time,
            "flux_axis_labels": flux,
            "chart_sha256": hashlib.sha256(png).hexdigest(),
            "browser_actions": 0,
            "answer_writes": 0,
        }
        (output / "report.json").write_text(json.dumps(report))
        (output / "chart.png").write_bytes(png)
        return report

    monkeypatch.setattr(window_module, "start_planet_observation", start)
    monkeypatch.setattr(window_module, "capture_observation_progress", capture)

    def factory(*args, **kwargs):
        star = BrowserStarSession(*args, **kwargs)
        # Start at the observation boundary; no numerical policy or UI is run.
        star.phase = "planet_window"
        star.component = PlanetWindowSession(
            star.page,
            star.config,
            star.output / "window",
            run_history=rig.root,
            emit=star._bind_events("planet.window"),
        )
        stars.append(star)
        windows.append(star.component)
        return star

    owner = make_owner(rig, component_factory=factory)
    owner.resume()
    owner.tick()  # one component construction
    owner.tick()  # one start; next poll at 5, deadline at 600
    assert owner.advances == 2 and calls == ["start"]
    return SimpleNamespace(rig=rig, owner=owner, star=stars[0], window=windows[0], calls=calls)


def test_thousands_of_idle_ticks_cost_zero_work_then_due_poll_costs_one(nested):
    rig, owner, window = nested.rig, nested.owner, nested.window
    deadline, started = window.deadline, owner._started_at
    before_checks = len([item for item in rig.calls if item[0] == "preflight"])
    for i in range(2000):
        rig.clock.now = i / 500
        owner.tick()
    assert owner.advances == 2 and window.polls == 0 and nested.calls == ["start"]
    assert len([item for item in rig.calls if item[0] == "preflight"]) >= before_checks + 2000
    assert window.deadline == deadline == 600 and owner._started_at == started == 0
    assert owner.max_advances == 512 and owner.max_seconds == 1800
    rig.clock.now = 5
    owner.tick()
    assert owner.advances == 3 and window.polls == 1
    assert nested.calls == ["start", "capture"]
    for _ in range(1000):
        owner.tick()
    assert owner.advances == 3 and window.polls == 1
    owner.abort()


def test_fixed_window_deadline_stops_without_capture_or_second_play(nested):
    owner, window = nested.owner, nested.window
    # Model a preceding long capture leaving the next timer beyond the deadline.
    window.next_poll = 605
    nested.rig.clock.now = 599.999
    owner.tick()
    assert owner.advances == 2 and not owner.finished
    nested.rig.clock.now = 600
    owner.tick()
    assert owner.advances == 3 and owner.status == "stopped"
    # The outer owner preserves the first verified child cause instead of
    # replacing it with a generic component-unverified wrapper.
    assert owner.failure == nested.star.failure == window.failure == "planet_window_time_limit"
    assert window.deadline == 600 and nested.calls == ["start"]
    for _ in range(10):
        owner.tick()
        owner.step()
    assert owner.advances == 3 and nested.calls == ["start"]


def test_production_point_two_second_cadence_counts_only_due_polls(nested):
    for tick in range(1001):
        nested.rig.clock.now = tick / 5
        nested.owner.tick()
    assert nested.window.polls == 40
    assert nested.owner.advances == 42  # construction + one Play + 40 actual captures
    assert nested.calls == ["start"] + ["capture"] * 40
    assert nested.window.deadline == 600 and nested.owner.max_advances == 512
    assert not nested.owner.finished and not nested.owner.state()["task_completed"]
    nested.owner.abort()


def test_owner_wall_deadline_is_checked_even_if_child_hint_stays_false(owner_rig):  # noqa: F811
    owner = make_owner(owner_rig, max_seconds=5)
    owner.step()
    child = owner_rig.children[0]
    child.ready_to_advance = lambda: False
    owner_rig.clock.now = 5
    owner.step()
    assert owner.failure == "project_steps_time_limit" and owner.advances == 1
    assert child.steps == 0 and child.aborted


@pytest.mark.parametrize("change", ["class", "journal", "reservation", "star_planet_source"])
def test_waiting_does_not_skip_source_guards(nested, change):
    owner, rig = nested.owner, nested.rig
    if change == "class":
        rig.source["source_hashes"]["class"] = "f" * 64
    elif change == "journal":
        rig.journal.path.write_text(rig.journal.path.read_text() + "\n")
    elif change == "reservation":
        owner._claim_path.write_text("{}")
    else:
        nested.star.window_evidence = {
            "progress_path": "missing-report.json",
            "progress_sha256": "0" * 64,
        }
    owner.tick()
    assert owner.finished and owner.advances == 2 and nested.calls == ["start"]
    assert nested.window.phase == "aborted"


def test_pause_single_step_and_abort_are_responsive_during_wait(nested):
    owner, rig = nested.owner, nested.rig
    owner.pause()
    rig.clock.now = 5
    owner.tick()
    owner.advance_if_due()
    assert nested.window.polls == 0 and owner.advances == 2
    owner.step()
    assert nested.window.polls == 1 and owner.advances == 3 and owner.status == "paused"
    owner.step()  # not yet due; manual step does not force a premature capture
    assert nested.window.polls == 1 and owner.advances == 3
    owner.resume()
    owner.abort()
    rig.clock.now = 15
    owner.tick()
    owner.step()
    assert nested.calls == ["start", "capture"] and owner.status == "aborted"


@pytest.mark.parametrize("effect", ["abort", "pause", "fail"])
def test_idle_state_callback_keeps_cancellation_and_failure_boundaries(nested, effect):
    owner = nested.owner

    def callback(kind, _payload):
        if kind == "state":
            if effect == "fail":
                raise RuntimeError("PRIVATE DRIVER DETAILS")
            getattr(owner, effect)()

    owner._callback = callback
    owner.tick()
    expected = {"abort": "aborted", "pause": "paused", "fail": "stopped"}[effect]
    assert owner.status == expected and owner.advances == 2 and nested.calls == ["start"]
    assert "PRIVATE" not in (owner.output / "events.jsonl").read_text()
    owner.abort()


@pytest.mark.parametrize("hint", [None, "always_due", "unchanged_state"])
def test_default_512_actual_work_cap_and_legacy_components(owner_rig, hint):  # noqa: F811
    rig = owner_rig
    rig.outcome["steps"] = 2000
    owner = make_owner(rig)
    owner.resume()
    owner.tick()
    child = rig.children[0]
    if hint is not None:
        child.ready_to_advance = lambda: True
    if hint == "unchanged_state":
        child.advance = lambda: rig.calls.append(("same_state_work", None))
    for _ in range(511):
        owner.tick()
    assert owner.advances == 512 and not owner.finished
    expected_calls = 511 if hint == "unchanged_state" else 0
    assert sum(c[0] == "same_state_work" for c in rig.calls) == expected_calls
    assert child.steps == (0 if hint == "unchanged_state" else 511)
    owner.tick()
    assert owner.failure == "project_steps_advance_limit" and owner.advances == 512
    assert child.steps == (0 if hint == "unchanged_state" else 511)


@pytest.mark.parametrize("ready", [False, 0, lambda: None, lambda: 1, lambda: "false"])
def test_owner_fails_closed_on_invalid_readiness(owner_rig, ready):  # noqa: F811
    owner = make_owner(owner_rig)
    owner.step()
    owner_rig.children[0].ready_to_advance = ready
    owner.step()
    assert owner.failure == "project_steps_invalid_component_readiness" and owner.advances == 1
    assert owner_rig.children[0].steps == 0


def test_idle_ticks_do_not_spend_the_last_work_slot(owner_rig):  # noqa: F811
    owner = make_owner(owner_rig, max_advances=2)
    owner.step()
    child = owner_rig.children[0]
    child.ready_to_advance = lambda: owner_rig.clock.now >= 5
    for _ in range(1000):
        owner.step()
    assert owner.advances == 1 and not owner.finished
    owner_rig.clock.now = 5
    owner.step()
    assert owner.advances == 2 and child.steps == 1 and not owner.finished
    owner.step()
    assert owner.failure == "project_steps_advance_limit" and child.steps == 1
