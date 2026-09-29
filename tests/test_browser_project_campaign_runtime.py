"""Injected campaign lifecycle only; real public fresh-source validation.

No browser, network, model loading or training. Completed tasks below are
synthetic journal fixtures, not live workflow or course/scientific acceptance.
"""

import io
import json
import socket
from copy import deepcopy

import pytest
from test_browser_project_campaign_steps import fresh
from test_browser_project_reference_class import RATIONALE, stop_fixture_class
from test_browser_project_runtime import EMAIL, PASSWORD, create, sha, text_artifacts, write
from test_browser_project_runtime import rig as base_rig  # noqa: F401
from test_project_progress import complete_star
from test_runtime_browser_project import options

import habfly.browser_project_runtime as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_stellar import SIMULATION_URL
from habfly.contracts import RuntimeEvent
from habfly.project_progress import Active, _json
from habfly.runtime import RunOptions, Runtime, parse_run_options, read_trace


@pytest.fixture
def rig(base_rig, monkeypatch):  # noqa: F811
    rig = base_rig
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("Network forbidden"))
    rig.options = RunOptions(
        task="browser_project",
        environment="browser",
        browser_setup="automatic",
        stars=2,
        project_campaign=True,
        project_campaign_max_seconds=600,
        paused=True,
    )
    rig.config.frames[0].url = SIMULATION_URL
    rig.classes, rig.campaigns = [], []
    rig.adopt_hook = rig.step_hook = None

    def initial(setup, output):
        assert setup.closed
        rig.calls.append(("capture_initial",))
        _, receipt = fresh(output, "Alpha", initial=True)
        (setup.output / "setup-starfield.png").write_bytes((output / "setup-starfield.png").read_bytes())
        return receipt

    rig.capture = initial

    class Campaign:
        def __init__(
            self,
            page,
            config,
            output,
            *,
            journal,
            run_history,
            initial_star_dir,
            initial_star_sha256,
            target_stars,
            max_seconds,
            emit,
            cancelled,
            _clock,
            **values,
        ):
            rig.calls.append(("campaign_construct",))
            assert target_stars in {2, 3, 30} and (
                max_seconds == "uncapped"
                and target_stars == 30
                or isinstance(max_seconds, (int, float))
                and max_seconds <= 10800
            )
            assert initial_star_dir == run_history / "initial-star"
            assert sha(initial_star_dir / "confirmed.json") == initial_star_sha256
            assert not journal.load().records
            self.output, self.history, self.journal = output, run_history, journal
            self.config = config.model_copy(deep=True)
            self.emit, self.cancelled, self.clock = emit, cancelled, _clock
            self.target, self.seconds, self.values = target_stars, max_seconds, values
            self.status, self.phase, self.report = "idle", "not_started", None
            self.owner_state, self.failure, self.verified = None, None, 0
            self.context = self.context_for(1, "Alpha", initial_star_dir)
            self.points = [json.loads((initial_star_dir / "confirmed.json").read_bytes())["selected_point"]]
            self.next_context = None
            self.started = None
            write(output / "star-001.json", self.context)
            rig.campaigns.append(self)

        @property
        def finished(self):
            return self.status in {"handoff", "stopped", "aborted"}

        def context_for(self, ordinal, name, source):
            return {
                "ordinal": ordinal,
                "star": name,
                "fresh_dir": str(source.relative_to(self.history)),
                "fresh_sha256": sha(source / "confirmed.json"),
                "class_output": f"campaign/stars/{ordinal:03d}/reference-class",
                "owner_output": f"campaign/stars/{ordinal:03d}/owner",
            }

        def current_star_context(self):
            return deepcopy(self.context)

        def state(self):
            return {
                "mode": "fresh_cooperative_project_campaign",
                "status": self.status,
                "phase": self.phase,
                "star": self.context["star"],
                "current_star": self.current_star_context(),
                "project_owner": deepcopy(self.owner_state),
                "target_stars": self.target,
                "owner_limits": {"max_seconds": 1800, "max_advances": 512},
                "max_seconds": self.seconds,
                "task_completed": False,
                "project_completed": False,
                "finished": self.finished,
                "target_workflows_verified": self.phase == "awaiting_assessment",
                "project_progress": self.journal.load().reduce().report(),
                "source_sha256": {
                    str(
                        (self.output / f"star-{self.context['ordinal']:03d}.json").relative_to(self.history)
                    ): sha(self.output / f"star-{self.context['ordinal']:03d}.json"),
                },
                "failure_reason": self.failure,
                "verified_stars": self.verified,
            }

        def publish(self):
            self.emit("state", self.state())

        def start(self, *, paused):
            self.started = self.clock()
            self.status, self.phase = "paused" if paused else "running", "initializing_star"
            self.publish()

        def step(self):
            assert self.status == "paused"
            return self.advance()

        def tick(self):
            if self.status == "running":
                self.advance()

        def advance(self):
            rig.calls.append(("campaign_step", self.phase, self.context["star"]))
            if self.seconds != "uncapped" and self.clock() - self.started >= self.seconds:
                self.status, self.phase, self.failure = "stopped", "stopped", "project_campaign_time_limit"
                self.report = self.state()
                write(self.output / "report.json", self.report)
                return
            if self.cancelled():
                return self.abort()
            if rig.step_hook:
                rig.step_hook(self)
            if self.phase == "initializing_star":
                self.phase = "awaiting_class_source"
                self.owner_state = {
                    "star": self.context["star"],
                    "phase": self.phase,
                    "status": "paused",
                    "task_completed": False,
                    "policy_stage": "awaiting_reference_class",
                    "artifact_paths": {},
                }
                self.status = "paused"
            elif self.phase == "ready":
                self.phase = "verifying_star"
                self.owner_state.update(phase="verified_fixture", status="completed", task_completed=True)
            elif self.phase == "verifying_star":
                result = complete_star(self.journal.load(), self.context["star"])
                self.journal.path.write_text(
                    _json(result.header())
                    + "\n"
                    + "".join(_json(r.model_dump(mode="json")) + "\n" for r in result.records)
                )
                self.verified += 1
                if self.verified == self.target:
                    self.status, self.phase = "handoff", "awaiting_assessment"
                    self.report = self.state()
                    write(self.output / "report.json", self.report)
                    self.emit("episode_summary", self.report)
                    return
                self.phase = "next_star_initializing"
            elif self.phase == "next_star_initializing":
                self.phase = "next_star_active"
            elif self.phase == "next_star_active":
                ordinal = self.context["ordinal"] + 1
                name = ("Alpha", "Beta", "Gamma")[ordinal - 1]
                source = self.output / f"transitions/{ordinal - 1:03d}-to-{ordinal:03d}/picker"
                _, receipt = fresh(
                    source,
                    name,
                    names=[s.name for s in self.journal.load().reduce().stars.values()],
                    excluded=self.points,
                )
                self.points.append(receipt["selected_point"])
                self.next_context = self.context_for(ordinal, name, source)
                self.phase = "adopting_fresh_star"
            elif self.phase == "adopting_fresh_star":
                self.context = self.next_context
                if rig.adopt_hook:
                    rig.adopt_hook(self)
                write(self.output / f"star-{self.context['ordinal']:03d}.json", self.context)
                self.owner_state = None
                self.phase = "initializing_star"
            else:
                pytest.fail("unexpected scheduler action: " + self.phase)
            self.publish()

        def provide_class(self, **values):
            rig.calls.append(("campaign_class", self.context["star"], values))
            self.phase = "ready"
            self.owner_state.update(
                phase="ready",
                policy_stage="learned_stellar_numeric",
                stellar_class_decision=values["selected_class"],
            )
            self.publish()

        def provide_planet_class(self, **values):
            rig.calls.append(("campaign_planet", self.context["star"], values))
            self.phase = "awaiting_gases"
            self.owner_state["phase"] = self.phase
            self.publish()

        def provide_gases(self, **values):
            rig.calls.append(("campaign_gases", self.context["star"], values))
            self.phase = "awaiting_habitability"
            self.owner_state["phase"] = self.phase
            self.publish()

        def provide_habitability(self, **values):
            rig.calls.append(("campaign_habitability", self.context["star"], values))
            self.phase = "ready"
            self.owner_state["phase"] = self.phase
            self.publish()

        def pause(self):
            self.status = "paused"

        def resume(self):
            self.status = "running"

        def abort(self):
            if not self.finished:
                self.status, self.phase = "aborted", "aborted"
                rig.calls.append(("campaign_abort",))

        close = abort

    class Class:
        def __init__(
            self,
            page,
            config,
            output,
            *,
            run_history,
            fresh_star,
            selected_class,
            reference_rationale,
            lifetime_prefix,
            emit,
            **limits,
        ):
            self.output, self.history, self.source, self.emit = output, run_history, fresh_star, emit
            self.config = config.model_copy(deep=True)
            self.star = json.loads((fresh_star / "confirmed.json").read_bytes())["star"]
            self.selected, self.prefix = selected_class, lifetime_prefix
            self.status, self.phase, self.finished, self.report = "ready", "class_pending", False, None
            self.source_hashes, self.sequence, self.writes = {}, 0, []
            rig.calls.append(
                (
                    "class_construct",
                    self.star,
                    str(output.relative_to(run_history)),
                    str(fresh_star.relative_to(run_history)),
                )
            )
            output.mkdir(parents=True)
            write(
                output / "decision.json", {"selected_class": selected_class, "rationale": reference_rationale}
            )
            self.source_hashes[str((output / "decision.json").relative_to(run_history))] = sha(
                output / "decision.json"
            )
            rig.classes.append(self)

        def state(self):
            return {
                "star": self.star,
                "status": self.status,
                "phase": self.phase,
                "finished": self.finished,
                "setup_verified": self.status == "completed",
                "task_completed": False,
                "scientific_verified": False,
                "training_label": False,
                "classification_learned": False,
                "selected_class": self.selected,
                "lifetime_prefix": self.prefix,
                "source_hashes": deepcopy(self.source_hashes),
            }

        def event(self, kind, payload):
            value = RuntimeEvent(event=kind, sequence=self.sequence, run_id=self.star, payload=payload)
            self.sequence += 1
            self.emit(value.model_dump(mode="json"))

        def advance(self):
            self.event(
                "action_proposed", {"kind": "CLICK", "target": "stellar_class", "value": self.selected}
            )
            if self.finished:
                return
            self.writes.append(self.selected)
            write(
                self.output / "class/confirmed.json",
                {"star": self.star, "fresh_star": str(self.source.relative_to(self.history))},
            )
            self.source_hashes[str((self.output / "class/confirmed.json").relative_to(self.history))] = sha(
                self.output / "class/confirmed.json"
            )
            self.status, self.phase, self.finished = "completed", "verified", True
            self.report = self.state()
            write(self.output / "report.json", self.report)
            self.event("episode_summary", self.report)

        def abort(self):
            self.finished = True

        close = abort

    def preflight(history, directory, star, selected):
        source = json.loads((directory / "confirmed.json").read_bytes())
        assert source["star"].casefold() == star.casefold()
        fresh_dir = source["fresh_star"]
        return {
            "fresh_star": fresh_dir,
            "source_capture_sha256": sha(history / fresh_dir / "stellar/observation.json"),
        }

    monkeypatch.setattr(module, "validate_star_class_source", preflight)
    rig.campaign_factory, rig.class_factory = Campaign, Class
    return rig


def make(rig, **options):
    return create(rig, _campaign_factory=rig.campaign_factory, _class_factory=rig.class_factory, **options)


def reach(bridge, phase):
    for _ in range(25):
        if bridge.phase == phase or bridge.finished:
            break
        bridge.step()
    assert bridge.phase == phase, bridge.state()


def ready(rig, **options):
    bridge = make(rig, **options)
    bridge.start()
    reach(bridge, "awaiting_class_source")
    return bridge


def classify(bridge, selected="white_dwarf"):
    bridge.command(
        {
            "command": "step",
            "payload": {
                "reference_class": {
                    "selected_class": selected,
                    "reference_rationale": RATIONALE,
                    "lifetime_prefix": None,
                }
            },
        }
    )
    assert bridge.phase == "setting_reference_class"
    bridge.step()
    assert bridge.phase == "class_setup_handoff", bridge.state()
    bridge.step()
    assert bridge.phase == "ready", bridge.state()


def second(rig):
    bridge = ready(rig)
    classify(bridge)
    reach(bridge, "adopting_fresh_star")
    bridge.step()
    assert bridge.phase == "initializing_star", bridge.state()
    reach(bridge, "awaiting_class_source")
    return bridge


def test_second_star_class_stop_keeps_cause_and_completed_first_star(rig, monkeypatch):
    bridge = second(rig)
    previous_journal = bridge.journal.path.read_bytes()
    bridge.provide_reference_class(
        selected_class="main_sequence", reference_rationale=RATIONALE, lifetime_prefix="Ga"
    )
    stop_fixture_class(bridge.class_setup, monkeypatch, "unsupported_class_circle_rendering")
    bridge.step()
    assert bridge.status == "stopped" and bridge.failure == "unsupported_class_circle_rendering"
    assert bridge.state()["current_star"]["ordinal"] == 2
    assert bridge.state()["current_star"]["star"] == "Beta"
    assert bridge.report["event_forwarding_failed"] is False
    assert not bridge.report["target_workflows_verified"]
    assert bridge.report["task_completed"] is bridge.report["project_completed"] is False
    assert bridge.journal.path.read_bytes() == previous_journal
    before = list(rig.calls)
    bridge.step()
    bridge.tick()
    assert rig.calls == before and not bridge.class_setup.writes
    assert not any(c[0] == "campaign_class" and c[1] == "Beta" for c in rig.calls)
    bridge.close()


@pytest.mark.parametrize("target", [2, 3])
def test_complete_campaign_handoff_is_not_task_or_project_success(rig, target):
    rig.options.stars = target
    bridge = ready(rig)
    for ordinal in range(1, target + 1):
        assert bridge.state()["current_star"]["ordinal"] == ordinal
        classify(bridge)
        if ordinal < target:
            reach(bridge, "adopting_fresh_star")
            bridge.step()
            reach(bridge, "awaiting_class_source")
    reach(bridge, "awaiting_assessment")
    state = bridge.state()
    assert state["status"] == "handoff" and state["finished"] and state["target_workflows_verified"]
    assert not state["task_completed"] and not state["project_completed"]
    assert state["initial_star"]["star"] == "Alpha" and state["current_star"]["star"] != "Alpha"
    assert len(rig.campaigns) == 1 and len(rig.classes) == target
    assert len({c.output for c in rig.classes}) == target
    assert [c[0] for c in rig.calls].count("launch") == 1
    assert [c[0] for c in rig.calls].count("setup_construct") == 1
    assert not any(c[0] == "steps_construct" for c in rig.calls)
    events = read_trace(bridge.trace_path)
    assert [(e.event, e.payload) for e in events] == rig.events
    assert sum(e.event == "episode_summary" for e in events) == 1
    assert events[-1].event == "state" and events[-2].event == "episode_summary"
    assert events[-1].payload == bridge.report == state
    assert events[-1].payload["project_progress"]["verified"] == target
    assert events[-1].payload["project_progress"]["unresolved"] == 0
    assert events[-1].payload["project_progress"]["submitted"] is False
    assert all(
        not e.payload.get("task_completed", False) and not e.payload.get("project_completed", False)
        for e in events
        if e.event in {"state", "episode_summary"}
    )
    previous = list(rig.calls)
    bridge.step()
    bridge.tick()
    assert rig.calls == previous
    bridge.close()


def test_start_never_launches_and_credentials_consumed_once(rig, monkeypatch):
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", PASSWORD)
    bridge = make(rig, credentials=None)
    bridge.start()
    bridge.advance_if_due()
    assert not rig.calls and not bridge.credentials_consumed
    reach(bridge, "awaiting_class_source")
    classify(bridge)
    reach(bridge, "adopting_fresh_star")
    bridge.step()
    reach(bridge, "awaiting_class_source")
    assert bridge.credentials_consumed and [c[0] for c in rig.calls].count("launch") == 1
    bridge.close()
    for private in (EMAIL, PASSWORD, rig.config.url):
        assert private not in text_artifacts(bridge.output)


def test_autonomous_campaign_reaches_each_explicit_class_handoff_without_stalling(rig):
    bridge = make(rig)
    bridge.start(paused=False)
    for _ in range(15):
        rig.clock.now += 1
        bridge.advance_if_due()
        if bridge.phase == "awaiting_class_source":
            break
    assert bridge.phase == "awaiting_class_source" and bridge.status == "paused", bridge.state()
    assert bridge.project.status == "paused"
    bridge.provide_reference_class(
        selected_class="white_dwarf", reference_rationale=RATIONALE, lifetime_prefix=None
    )
    bridge.resume()
    for _ in range(15):
        rig.clock.now += 1
        bridge.advance_if_due()
        if bridge.phase == "awaiting_class_source":
            break
    assert bridge.phase == "awaiting_class_source" and bridge.status == "paused", bridge.state()
    assert bridge.state()["current_star"]["star"] == "Beta" and bridge.class_setup is None
    assert bridge.project.status == "paused"
    assert sum(c[0] == "campaign_class" for c in rig.calls) == 1
    bridge.close()


def test_stale_class_state_clears_only_on_verified_new_star_adoption(rig):
    bridge = ready(rig)
    classify(bridge)
    old_class = bridge.class_setup
    reach(bridge, "adopting_fresh_star")
    assert bridge.state()["class_setup"]["star"] == "Alpha"
    assert bridge.state()["current_star"]["star"] == "Alpha"
    bridge.step()
    state = bridge.state()
    assert state["current_star"]["star"] == "Beta" and state["initial_star"]["star"] == "Alpha"
    assert state["class_setup"] is None and state["project_owner"] is None and state["policy_stage"] is None
    assert state["artifact_paths"]["reference_class_setup"] is None
    assert bridge._class_decision is None and bridge._class_setup_report is None
    with pytest.raises(ValueError, match="Stale"):
        old_class.event("state", old_class.state())
    reach(bridge, "awaiting_class_source")
    assert bridge.state()["project_owner"]["star"] == "Beta"
    before = list(rig.calls)
    bridge.command({"command": "step", "payload": {}})
    assert rig.calls == before
    with pytest.raises(BrowserSafetyStop, match="handoff"):
        bridge.resume()
    classify(bridge, "red_giant")
    assert bridge.class_setup.output != old_class.output
    assert bridge.class_setup.source != old_class.source
    bridge.close()


@pytest.mark.parametrize("source", ["initial", "current", "prior_class", "context"])
def test_each_current_handoff_checks_original_and_current_immutable_sources(rig, source):
    bridge = second(rig)
    if source == "initial":
        path = bridge.output / "initial-star/stellar/observation.json"
    elif source == "current":
        path = bridge.output / bridge.state()["current_star"]["fresh_dir"] / "stellar/observation.json"
    elif source == "prior_class":
        path = rig.classes[0].output / "decision.json"
    else:
        path = bridge.output / "campaign/star-002.json"
    write(path, {"tampered": True})
    before = len(rig.classes)
    bridge.provide_reference_class(
        selected_class="red_giant", reference_rationale=RATIONALE, lifetime_prefix=None
    )
    assert bridge.status == "stopped" and len(rig.classes) == before
    bridge.close()


@pytest.mark.parametrize("change", ["star", "ordinal", "fresh_hash", "fresh_path", "class_path", "nonblank"])
def test_invalid_adoption_never_clears_or_relabels_old_class(rig, change):
    bridge = ready(rig)
    classify(bridge)
    previous = bridge.class_setup
    reach(bridge, "adopting_fresh_star")

    def mutate(campaign):
        if change == "star":
            campaign.context["star"] = "Foreign"
        elif change == "ordinal":
            campaign.context["ordinal"] = 3
        elif change == "fresh_hash":
            campaign.context["fresh_sha256"] = "f" * 64
        elif change == "fresh_path":
            campaign.context["fresh_dir"] = "initial-star"
        elif change == "class_path":
            campaign.context["class_output"] = "campaign/stars/001/reference-class"
        else:
            write(campaign.history / campaign.context["fresh_dir"] / "stellar/observation.json", {})

    rig.adopt_hook = mutate
    bridge.step()
    assert bridge.status == "stopped" and bridge.class_setup is previous
    assert bridge.state()["current_star"]["star"] == "Alpha"
    assert not bridge.state()["target_workflows_verified"]
    bridge.close()


def test_old_class_receipt_cannot_bind_a_different_current_star(rig):
    bridge = second(rig)
    bridge.provide_class(
        class_dir=rig.classes[0].output / "class", selected_class="white_dwarf", lifetime_prefix=None
    )
    assert bridge.status == "stopped"
    assert len([c for c in rig.calls if c[0] == "campaign_class"]) == 1
    bridge.close()


def test_reference_planet_gas_and_habitability_delegate_only_to_active_star(rig):
    bridge = ready(rig, reference_planet_continuation=True, terrestrial_options={"explicit": True})
    classify(bridge)
    reach(bridge, "adopting_fresh_star")
    bridge.step()
    reach(bridge, "awaiting_class_source")
    classify(bridge, "red_giant")
    bridge.project.phase = "awaiting_planet_class"
    bridge.project.owner_state["phase"] = "awaiting_planet_class"
    bridge._sync_project()
    bridge.command(
        {"command": "step", "payload": {"planet_class": {"name": "terrestrial", "rationale": RATIONALE}}}
    )
    bridge.command(
        {
            "command": "step",
            "payload": {
                "gases": {"gases": ["CO2"], "rationale": RATIONALE, "supplied_greenhouse_increment": 10}
            },
        }
    )
    bridge.command(
        {"command": "step", "payload": {"habitability": {"choice": "not_habitable", "rationale": RATIONALE}}}
    )
    assert bridge.phase == "ready"
    choices = [c for c in rig.calls if c[0] in {"campaign_planet", "campaign_gases", "campaign_habitability"}]
    assert len(choices) == 3 and all(c[1] == "Beta" for c in choices)
    assert not any(c[0] == "campaign_step" and c[1] == "awaiting_gases" for c in rig.calls)
    bridge.close()


@pytest.mark.parametrize(
    "phase", ["initializing_star", "next_star_initializing", "next_star_active", "adopting_fresh_star"]
)
def test_pause_abort_and_new_star_transitions_remain_one_scheduled_call(rig, phase):
    bridge = make(rig)
    bridge.start()
    if phase == "initializing_star":
        reach(bridge, phase)
    else:
        reach(bridge, "awaiting_class_source")
        classify(bridge)
        reach(bridge, phase)
    before = list(rig.calls)
    bridge.advance_if_due()
    bridge.tick()
    assert rig.calls == before
    bridge.abort()
    bridge.step()
    bridge.tick()
    assert bridge.status == "aborted" and not bridge.state()["target_workflows_verified"]
    assert not any(c[0] == "campaign_step" for c in rig.calls[len(before) :])
    bridge.close()


def test_campaign_deadline_does_not_restart_for_next_star(rig):
    bridge = second(rig)
    classify(bridge)
    rig.clock.now = 601
    bridge.step()
    assert bridge.status == "stopped" and not bridge.state()["task_completed"]
    bridge.close()


def test_campaign_deadline_also_guards_separately_scheduled_class_setup(rig):
    bridge = second(rig)
    bridge.provide_reference_class(
        selected_class="red_giant", reference_rationale=RATIONALE, lifetime_prefix=None
    )
    rig.clock.now = 600
    bridge.step()
    assert bridge.status == "stopped" and not bridge.class_setup.writes
    assert bridge.failure == "project_runtime_campaign_time_limit"
    bridge.close()


@pytest.mark.parametrize("effect", ["abort", "source_change", "deadline", "callback_error"])
def test_current_class_preaction_callback_never_allows_a_stale_write(rig, effect):
    holder = []

    def emit(kind, payload):
        if kind != "action_proposed" or payload.get("component") != "browser.reference_class":
            return
        bridge = holder[0]
        if effect == "abort":
            bridge.abort()
        elif effect == "source_change":
            write(
                bridge.output / bridge.state()["current_star"]["fresh_dir"] / "stellar/observation.json", {}
            )
        elif effect == "deadline":
            rig.clock.now = 600
        else:
            raise RuntimeError(PASSWORD)

    bridge = ready(rig, emit=emit)
    holder.append(bridge)
    bridge.provide_reference_class(
        selected_class="white_dwarf", reference_rationale=RATIONALE, lifetime_prefix=None
    )
    bridge.step()
    assert bridge.finished and not bridge.class_setup.writes
    assert not any(c[0] == "campaign_class" for c in rig.calls)
    bridge.close()
    assert PASSWORD not in text_artifacts(bridge.output)


@pytest.mark.parametrize("change", ["labels", "url", "frame_url"])
def test_current_class_uses_refreshed_public_labels_but_never_expands_boundary(rig, change):
    bridge = second(rig)
    if change == "labels":
        bridge.project.config.frames[0].required_text = ["Observations", "Reconstruction", "parallax"]
    elif change == "url":
        bridge.project.config.url = "http://localhost/other-project"
    else:
        bridge.project.config.frames[0].url = "https://foreign.invalid/hidden"
    before = len(rig.classes)
    bridge.provide_reference_class(
        selected_class="red_giant", reference_rationale=RATIONALE, lifetime_prefix=None
    )
    if change == "labels":
        assert bridge.phase == "setting_reference_class"
        assert (
            bridge.class_setup.config.frames[0].required_text == bridge.project.config.frames[0].required_text
        )
        assert bridge.config.frames[0].required_text != bridge.project.config.frames[0].required_text
    else:
        assert bridge.status == "stopped" and len(rig.classes) == before
    bridge.close()


def test_source_changed_during_explicit_class_validation_stops_before_handoff(rig, monkeypatch):
    bridge = second(rig)
    context = bridge.state()["current_star"]
    directory = bridge.output / "explicit-current-class"
    write(directory / "confirmed.json", {"star": "Beta", "fresh_star": context["fresh_dir"]})
    original = module.validate_star_class_source

    def changed(*args):
        source = original(*args)
        write(bridge.output / context["fresh_dir"] / "stellar/observation.json", {})
        return source

    monkeypatch.setattr(module, "validate_star_class_source", changed)
    before = len([c for c in rig.calls if c[0] == "campaign_class"])
    bridge.provide_class(class_dir=directory, selected_class="red_giant", lifetime_prefix=None)
    assert bridge.status == "stopped"
    assert len([c for c in rig.calls if c[0] == "campaign_class"]) == before
    bridge.close()


def test_unscheduled_current_context_change_stops_before_next_child_call(rig):
    bridge = second(rig)
    classify(bridge)
    bridge.project.context["star"] = "Foreign"
    before = len([c for c in rig.calls if c[0] == "campaign_step"])
    bridge.step()
    assert bridge.status == "stopped"
    assert len([c for c in rig.calls if c[0] == "campaign_step"]) == before
    bridge.close()


@pytest.mark.parametrize("publication", ["episode_summary", "state"])
@pytest.mark.parametrize(
    "source", ["initial", "current", "prior_class", "journal", "campaign_report", "context"]
)
def test_final_outer_callback_cannot_leave_stale_verified_target(rig, source, publication):
    holder = []

    def emit(kind, payload):
        rig.events.append((kind, deepcopy(payload)))
        if (
            kind != publication
            or payload.get("mode") != "fresh_browser_campaign_project_runtime"
            or payload.get("status") != "handoff"
        ):
            return
        bridge = holder[0]
        if source == "journal":
            bridge.journal.append(Active(star_id="Alpha", stage="stellar_numeric"))
        else:
            path = {
                "initial": bridge.output / "initial-star/stellar/observation.json",
                "current": bridge.output
                / bridge.state()["current_star"]["fresh_dir"]
                / "stellar/observation.json",
                "prior_class": rig.classes[0].output / "decision.json",
                "campaign_report": bridge.output / "campaign/report.json",
                "context": bridge.output / "campaign/star-002.json",
            }[source]
            write(path, {"changed_during_outer_callback": True})

    bridge = ready(rig, emit=emit)
    holder.append(bridge)
    classify(bridge)
    reach(bridge, "adopting_fresh_star")
    bridge.step()
    reach(bridge, "awaiting_class_source")
    classify(bridge)
    reach(bridge, "verifying_star")
    bridge.step()
    assert bridge.status == "stopped" and bridge.phase == "stopped"
    assert bridge.failure == "project_runtime_terminal_campaign_validation_failed"
    assert not bridge.report["target_workflows_verified"]
    assert not bridge.report["task_completed"] and not bridge.report["project_completed"]
    assert json.loads((bridge.output / "report.json").read_bytes()) == bridge.report
    assert read_trace(bridge.trace_path)[-1].payload["target_workflows_verified"] is False
    bridge.close()


@pytest.mark.parametrize("fail_at", ["handoff", "corrective_stop"])
def test_terminal_state_callback_failure_persists_corrective_stop(rig, fail_at):
    holder = []

    def emit(kind, payload):
        rig.events.append((kind, deepcopy(payload)))
        if kind == "state" and payload.get("status") == "handoff":
            if fail_at == "handoff":
                raise RuntimeError(PASSWORD)
            holder[0].journal.append(Active(star_id="Alpha", stage="stellar_numeric"))
        if kind == "state" and payload.get("status") == "stopped" and fail_at == "corrective_stop":
            raise RuntimeError(PASSWORD)

    bridge = ready(rig, emit=emit)
    holder.append(bridge)
    classify(bridge)
    reach(bridge, "adopting_fresh_star")
    bridge.step()
    reach(bridge, "awaiting_class_source")
    classify(bridge)
    reach(bridge, "verifying_star")
    bridge.step()
    assert bridge.status == bridge.phase == "stopped"
    assert bridge.failure == "project_runtime_event_forwarding_failed"
    assert bridge.report["event_forwarding_failed"] is True
    assert bridge.report["target_workflows_verified"] is False
    assert bridge.report["task_completed"] is bridge.report["project_completed"] is False
    assert read_trace(bridge.trace_path)[-1].payload == bridge.report
    assert json.loads((bridge.output / "report.json").read_bytes()) == bridge.report
    assert PASSWORD not in text_artifacts(bridge.output)
    before = list(rig.calls)
    bridge.step()
    bridge.tick()
    assert rig.calls == before
    bridge.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"stars": 1},
        {"stars": 29},
        {"project_campaign_max_seconds": None},
        {"project_max_advances": 511},
        {"project_max_seconds": 1801},
        {"project_campaign": False, "stars": 2},
        {"task": "stellar"},
    ],
)
def test_parser_rejects_unapproved_campaign_scope_without_artifacts(rig, changes):
    value = options(rig.root, project_campaign=True, project_campaign_max_seconds=600, stars=2)
    value.update(changes)
    with pytest.raises(ValueError):
        parse_run_options(value)
    assert not rig.calls


def test_outer_jsonl_runtime_preserves_campaign_target_and_protocol(rig, monkeypatch):
    real = module.BrowserProjectRuntime

    def injected(settings, output, **values):
        return real(
            settings,
            output,
            config=rig.config,
            credentials=(EMAIL, PASSWORD),
            _driver_factory=rig.driver,
            _setup_factory=rig.setup,
            _capture_initial=rig.capture,
            _campaign_factory=rig.campaign_factory,
            _class_factory=rig.class_factory,
            _clock=lambda: rig.clock.now,
            **values,
        )

    monkeypatch.setattr(module, "BrowserProjectRuntime", injected)
    runtime = Runtime(io.StringIO())
    runtime.command(
        {
            "command": "start",
            "payload": options(rig.root, project_campaign=True, project_campaign_max_seconds=600, stars=2),
        }
    )
    assert not rig.calls and runtime.options.stars == 2
    bridge = runtime.env
    for _ in range(10):
        runtime.command({"command": "step", "payload": {}})
        if bridge.phase == "awaiting_class_source":
            break
    assert bridge.phase == "awaiting_class_source"
    outer, inner = read_trace(runtime.trace_path), read_trace(bridge.trace_path)
    assert all(e.version == 1 for e in outer)
    assert all(
        e.payload.get("stars") == 2 and e.payload.get("project_campaign") is True
        for e in outer
        if e.event in {"hello", "state"}
    )
    assert [(e.event, e.payload.get("current_star")) for e in outer] == [
        (e.event, e.payload.get("current_star")) for e in inner
    ]
    runtime.close()
    assert PASSWORD not in runtime.output.getvalue()
