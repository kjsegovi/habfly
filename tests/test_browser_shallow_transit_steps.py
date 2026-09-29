"""Injected chart/generator boundaries, actual source and reference validators.

Synthetic native event templates are fixture data, not browser execution. No
page, trained model, grading oracle, real attempt, or network is accessed here.
"""

import hashlib
import io
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image
from test_browser_numeric import config
from test_browser_shallow_transit_probe import png
from test_planet_tooltip_reference import fixture as diagnostic_fixture

import habfly.browser_shallow_transit_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_progress
from habfly.browser_shallow_transit_probe import (
    EXACT_TWO_HINT_POLICY,
    candidate_columns,
    overview_hint_metadata,
)
from habfly.browser_shallow_transit_probe import (
    FIRST_THREE_TWO_ROW_HINT_POLICY as FIRST_THREE_HINT_POLICY,
)
from habfly.planet_tooltip_reference import DIAGNOSTIC_FILES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def load(path):
    return json.loads(path.read_bytes())


@pytest.fixture
def rig(tmp_path, request):
    count = getattr(request, "param", 3)
    period = 100 if count == 48 else 1000
    columns = [int(30.5 + period * (index + 1) * 0.046) for index in range(count)]
    data = png(baseline=20, features=tuple((column, column, 22) for column in columns), size=(280, 196))
    times = [{"value": str(day), "center_x": 30.5 + day * 0.046} for day in range(0, 5001, 500)]
    flux = [{"value": str(value), "center_y": 20.5 + (100 - value) * 1.3} for value in range(10, 101, 10)]

    def report(image, labels=None):
        labels = times if labels is None else labels
        return {
            **trace_progress(image, labels, requested_days=5000),
            "star": "EXAMPLE",
            "time_axis_labels": labels,
            "flux_axis_labels": deepcopy(flux),
            "chart_sha256": hashlib.sha256(image).hexdigest(),
            "browser_actions": 0,
            "answer_writes": 0,
        }

    source = tmp_path / "original"
    source.mkdir()
    (source / "chart.png").write_bytes(data)
    save(source / "report.json", report(data))
    checksum = sha(source / "report.json")
    state = SimpleNamespace(
        calls=[],
        clock=0,
        star="EXAMPLE",
        signature=[["5000", "", "", ""]],
        image=data,
        labels=times,
        capture_hook=None,
        probe_hook=None,
        probe_scope_hook=None,
        fail_probe=None,
        before_next=None,
        native=[],
        probe_next=0,
        closed=0,
        events=[],
    )

    class Session:
        def __init__(self, page, boundary, emit, **options):
            state.calls.append(("session", options))
            self.star, self.signature = state.star, deepcopy(state.signature)
            self.emit, self.actions, self.stopped = emit, 0, False
            self.handle = SimpleNamespace(evaluate=lambda _: True)

            def screenshot(**options):
                from habfly.presentation_capture import EVIDENCE_SCREENSHOT_STYLE

                assert options == {"style": EVIDENCE_SCREENSHOT_STYLE}
                return state.image

            self.chart = SimpleNamespace(screenshot=screenshot)
            emit("observation", {"chart": {"source": "visible_tooltips", "star": self.star}})

        def _guard(self):
            if self.stopped or self.star != state.star or self.signature != state.signature:
                raise BrowserSafetyStop("chart_context_or_answers_changed")

        def time_axis_labels(self):
            return deepcopy(state.labels)

        def flux_axis_labels(self):
            return deepcopy(flux)

        def action(self, kind, payload):
            self._guard()
            self.actions += 1
            self.emit(
                "action_proposed", {"kind": kind, "surface": "chart", "sequence": self.actions, **payload}
            )
            state.native.append((kind, payload))
            self.emit("action_result", {"sequence": self.actions, "task_completed": False})

        def zoom(self, x, y, delta):
            assert (x, y, delta) == (0.5, 0.4, 1500)
            self.action("SCROLL", {"x_fraction": x, "y_fraction": y, "delta_y": delta})

        def clear_pointer(self):
            self.action("HOVER", {"purpose": "remove_pointer_overlay"})

        def close(self):
            self.stopped = True
            state.closed += 1

    def capture(page, boundary, output, **options):
        state.calls.append(("capture", options))
        output.mkdir(parents=True)
        if state.capture_hook:
            state.capture_hook(output)
        value = report(state.image, state.labels)
        (output / "chart.png").write_bytes(state.image)
        save(output / "report.json", value)
        return value

    def probe(
        page,
        boundary,
        output,
        *,
        run_history,
        source_dir,
        source_report_sha256,
        candidate_index,
        overview_hint_policy=None,
    ):
        # Explicit synthetic native seam: generate a valid diagnostic template,
        # then bind its declared overview/candidate to the owner's actual input.
        templates = tmp_path / "templates"
        templates.mkdir(exist_ok=True)
        assert overview_hint_policy == FIRST_THREE_HINT_POLICY or (
            count == 2 and overview_hint_policy == EXACT_TWO_HINT_POLICY
        )
        spec = diagnostic_fixture(templates, output.name, period * (candidate_index + 1), linked=True)
        template = templates / spec["directory"]
        output.mkdir()
        raw_events = [json.loads(line) for line in (template / "events.jsonl").read_text().splitlines()]
        current = load(source_dir / "report.json")
        source_hashes = {"report.json": source_report_sha256, "chart.png": sha(source_dir / "chart.png")}
        scope, result = load(template / "scope.json"), load(template / "report.json")
        group = candidate_columns(
            (source_dir / "chart.png").read_bytes(), flux, hint_policy=overview_hint_policy
        )[candidate_index]
        ticks = sorted((float(row["center_x"]), float(row["value"])) for row in current["time_axis_labels"])
        per_pixel = (ticks[-1][1] - ticks[0][1]) / (ticks[-1][0] - ticks[0][0])
        hint_interval = [
            max(0.0, ticks[0][1] + (group[0] - 0.5 - ticks[0][0]) * per_pixel),
            min(5000.0, ticks[0][1] + (group[-1] + 1.5 - ticks[0][0]) * per_pixel),
        ]
        scope.update(
            source_dir=str(source_dir.relative_to(tmp_path)),
            source_hashes=source_hashes,
            candidate_index=candidate_index,
            candidate_columns=group,
            overview_hint_policy=overview_hint_policy,
            source_hint_day_interval=hint_interval,
        )
        scope.update(overview_hint_metadata(overview_hint_policy))
        result.update(
            source_hashes=source_hashes,
            overview_hint_policy=overview_hint_policy,
            source_hint_day_interval=hint_interval,
        )
        result.update(overview_hint_metadata(overview_hint_policy))
        for name in DIAGNOSTIC_FILES - {"scope.json", "report.json", "events.jsonl"}:
            (output / name).write_bytes((template / name).read_bytes())
        before = load(output / "before.json")
        before.update(
            time_axis_labels=current["time_axis_labels"],
            flux_axis_labels=current["flux_axis_labels"],
            chart_sha256=current["chart_sha256"],
        )
        save(output / "before.json", before)
        (output / "before.png").write_bytes((source_dir / "chart.png").read_bytes())
        if state.probe_scope_hook:
            state.probe_scope_hook(scope)
        save(output / "scope.json", scope)
        raw_events[-1]["payload"] = result
        stream = (output / "events.jsonl").open("x")

        def emit(event):
            event["run_id"] = output.name
            stream.write(json.dumps(event) + "\n")
            stream.flush()

        try:
            emit(raw_events[0])
            state.probe_next += 1
            yield {"stage": "source_verified", "browser_actions": 0}
            for index in range(1, len(raw_events) - 1, 2):
                if state.before_next:
                    state.before_next(output, index)
                state.probe_next += 1
                emit(raw_events[index])
                state.native.append((raw_events[index]["payload"]["kind"], {"probe": candidate_index}))
                if state.fail_probe == candidate_index:
                    raise BrowserSafetyStop("fixture_probe_failed_after_dispatch")
                emit(raw_events[index + 1])
                yield {"stage": "sample", "browser_actions": (index + 1) // 2}
            if state.probe_hook:
                state.probe_hook(result, scope)
            save(output / "report.json", result)
            emit(raw_events[-1])
            state.probe_next += 1
            return result
        except BaseException:
            save(output / "stopped.json", {"reason": "fixture_stopped", "answer_writes": 0})
            raise
        finally:
            stream.close()

    def make(name="sensor", emit=None, **changes):
        options = {
            "run_history": tmp_path,
            "star": "Example",
            "source_report": source / "report.json",
            "source_report_sha256": checksum,
            "emit": emit or (lambda kind, payload: state.events.append((kind, payload))),
            "_clock": lambda: state.clock,
            "_session_factory": Session,
            "_probe_factory": probe,
            "_capture_progress": capture,
        }
        options.update(changes)
        return module.ShallowTransitSteps(None, config(), tmp_path / name, **options)

    return SimpleNamespace(root=tmp_path, source=source, state=state, make=make, report=report, data=data)


def drive(owner, until=None):
    for _ in range(module.MAX_ADVANCES + 1):
        if owner.finished or owner.phase == until:
            return owner.state()
        owner.advance()
    pytest.fail("Owner did not reach a bounded terminal/handoff")


def test_offline_constructor_full_bounded_measurements_and_exact_provenance(rig):
    owner = rig.make()
    assert not rig.state.calls and not rig.state.native
    while not owner.finished:
        before = rig.state.probe_next
        owner.advance()
        assert rig.state.probe_next - before <= 1
    assert owner.phase == "measurements_ready", owner.report
    assert owner.native_action_attempts == owner.native_actions_confirmed == 39
    assert owner.advances <= 80 and rig.state.closed == 1
    assert len(owner.report["tooltip_reference"]["diagnostics"]) == 3
    assert owner.report["measurements"]["brightness_drop_percent"]["value"] == "0.762"
    assert owner.report["measurements"]["brightness_drop_percent"]["physical_bounds"] is None
    assert owner.report["measurements"]["period_days"]["value"] == "1000"
    assert owner.report["window_evidence"]["progress_path"] == "sensor/restore-03/progress/report.json"
    assert not owner.report["task_completed"] and not owner.report["project_completed"]
    assert not owner.report["answer_authorized"]
    assert [item for kind, item in rig.state.native if kind == "SCROLL" and "delta_y" in item] == [
        {"x_fraction": 0.5, "y_fraction": 0.4, "delta_y": 1500}
    ] * 6
    assert owner.report["restorations"] == 3
    assert owner.report["restore_recipe_version"] == "post_probe_center_0_5_0_4_v2"
    assert owner.report["restore_pointer"] == {"x_fraction": 0.5, "y_fraction": 0.4}
    events = [json.loads(line) for line in (owner.output / "events.jsonl").read_text().splitlines()]
    assert [(e["event"], e["payload"]) for e in events] == rig.state.events
    child = [e["payload"]["component_event"] for e in events if "component_event" in e["payload"]]
    expected = [
        json.loads(line)
        for n in range(1, 4)
        for line in (owner.output / f"probe-{n:02d}/events.jsonl").read_text().splitlines()
    ]
    assert child == expected
    files = {str(p): p.read_bytes() for p in owner.output.rglob("*") if p.is_file()}
    owner.advance()
    owner.close()
    assert files == {str(p): p.read_bytes() for p in owner.output.rglob("*") if p.is_file()}


def test_initial_overview_is_freshly_captured_without_redundant_zoom_or_hover(rig):
    owner = rig.make()
    owner.advance()
    assert owner.phase == "initial_capture"
    assert [name for name, _ in rig.state.calls] == ["session"]
    assert not rig.state.native and owner.anchor is None
    owner.advance()
    assert owner.phase == "probe_initializing"
    assert [name for name, _ in rig.state.calls] == ["session", "capture"]
    assert owner.native_action_attempts == owner.native_actions_confirmed == 0
    assert not rig.state.native and not (owner.output / "restore-00").exists()
    assert owner.anchor["chart_sha256"] == sha(rig.source / "chart.png")
    assert owner.window_evidence["progress_path"] == "sensor/initial/progress/report.json"
    assert owner.anchor["fingerprint"] == owner.source["fingerprint"]
    owner.abort()


@pytest.mark.parametrize("change", ["plot", "gutter", "axis", "encoding"])
def test_initial_capture_must_exactly_match_original_not_adopt_new_plausible_overview(rig, change):
    owner = rig.make()
    owner.advance()

    def change_capture(_directory):
        if change == "axis":
            rig.state.labels = rig.state.labels[1:]  # Grounded only for later restores.
            return
        image = Image.open(io.BytesIO(rig.state.image)).convert("RGB")
        if change != "encoding":
            image.putpixel((40, 30) if change == "plot" else (30, 21), (1, 2, 3))
        output = io.BytesIO()
        image.save(output, format="PNG", compress_level=0)
        rig.state.image = output.getvalue()

    rig.state.capture_hook = change_capture
    owner.advance()
    assert owner.phase == "stopped"
    assert owner.failure == "shallow_steps_initial_current_source_changed"
    assert owner.anchor is None and owner.window_evidence is None
    assert not rig.state.native and not (owner.output / "probe-01").exists()
    assert owner.native_action_attempts == 0
    owner.advance()
    assert not rig.state.native
    with pytest.raises(BrowserSafetyStop, match="owner_already_claimed"):
        rig.make("no-retry")


def test_expired_initial_capture_never_publishes_anchor_or_starts_probe(rig):
    owner = rig.make()
    owner.advance()
    rig.state.capture_hook = lambda _: setattr(rig.state, "clock", 900)
    owner.advance()
    assert owner.failure == "shallow_steps_time_limit"
    assert not rig.state.native and owner.anchor is None
    assert not (owner.output / "probe-01").exists()


def test_fixed_restore_recipe_is_pinned_before_any_action(rig):
    owner = rig.make()
    owner.scope["restore_pointer"]["y_fraction"] = 0.1
    owner.advance()
    assert owner.failure == "shallow_steps_scope_changed"
    assert not rig.state.calls and not rig.state.native


@pytest.mark.parametrize("rig", [48], indirect=True)
def test_frequent_overview_uses_only_first_three_without_extra_actions(rig):
    owner = rig.make()
    drive(owner)
    assert owner.phase == "measurements_ready", owner.report
    assert owner.scope["overview_hint_policy"] == FIRST_THREE_HINT_POLICY
    assert owner.report["overview_hint_policy"] == FIRST_THREE_HINT_POLICY
    assert owner.report["measurements"]["period_days"]["value"] == "100"
    assert owner.native_action_attempts == owner.native_actions_confirmed == 39
    assert owner.advances <= 80 and len(owner.completed_probes) == 3
    assert owner.report["max_seconds"] == 900 and owner.report["max_native_actions"] == 64
    for index in range(3):
        directory = owner.output / f"probe-{index + 1:02d}"
        scope, result = load(directory / "scope.json"), load(directory / "report.json")
        assert scope["overview_hint_policy"] == result["overview_hint_policy"] == FIRST_THREE_HINT_POLICY
        assert scope["candidate_index"] == index
        assert scope["candidate_columns"] == owner.candidates[index]
        assert result["source_hint_linked"] is True
        assert (
            result["source_hint_day_interval"][0]
            <= 100 * (index + 1)
            <= result["source_hint_day_interval"][1]
        )
    assert not (owner.output / "probe-04").exists()
    assert owner.report["answer_writes"] == 0 and not owner.report["task_completed"]


def test_owner_hint_policy_cannot_change_after_claim(rig):
    owner = rig.make()
    owner.scope["overview_hint_policy"] = "different"
    owner.advance()
    assert owner.failure == "shallow_steps_scope_changed"
    assert not rig.state.native and not rig.state.calls


@pytest.mark.parametrize("stage", ["owner", "scope", "report"])
def test_recipe_boolean_alias_is_rejected_at_each_owner_boundary(rig, stage):
    def change(value):
        value["overview_hint_recipe"]["answer_authorized"] = 0

    if stage == "scope":
        rig.state.probe_scope_hook = change
    if stage == "report":
        rig.state.probe_hook = lambda report, _: change(report)
    owner = rig.make()
    if stage == "owner":
        change(owner.scope)
    drive(owner)
    assert owner.phase == "stopped" and not owner.completed_probes
    assert not (owner.output / "probe-02").exists()
    if stage != "report":
        assert not rig.state.native


@pytest.mark.parametrize("policy", [None, "different", True])
def test_child_scope_policy_must_match_before_any_probe_actions(rig, policy):
    def change(scope):
        if policy is None:
            scope.pop("overview_hint_policy")
        else:
            scope["overview_hint_policy"] = policy

    rig.state.probe_scope_hook = change
    owner = rig.make()
    drive(owner)
    assert owner.failure == "shallow_steps_candidate_selection_changed"
    assert not rig.state.native and not owner.completed_probes
    assert not (owner.output / "probe-02").exists()


@pytest.mark.parametrize("field", ["index", "columns"])
def test_child_scope_numeric_alias_rejected_before_probe_actions(rig, field):
    def change(scope):
        if field == "index":
            scope["candidate_index"] = False
        else:
            scope["candidate_columns"] = [float(value) for value in scope["candidate_columns"]]

    rig.state.probe_scope_hook = change
    owner = rig.make()
    drive(owner)
    assert owner.failure == "shallow_steps_candidate_selection_changed"
    assert not rig.state.native and not owner.completed_probes


@pytest.mark.parametrize("policy", [None, "different", True])
def test_child_result_policy_cannot_be_missing_or_replaced(rig, policy):
    def change(report, _scope):
        if policy is None:
            report.pop("overview_hint_policy")
        else:
            report["overview_hint_policy"] = policy

    rig.state.probe_hook = change
    owner = rig.make()
    drive(owner)
    assert owner.phase == "stopped" and not owner.completed_probes
    assert owner.failure in {"shallow_steps_probe_policy_changed", "shallow_steps_probe_not_verified"}
    assert not (owner.output / "probe-02").exists() and not (owner.output / "restore-01").exists()


@pytest.mark.parametrize(
    "phase", ["initializing", "initial_capture", "restore_zoom_2", "probe_active", "validating"]
)
def test_abort_is_terminal_no_new_actions_and_claim_prevents_output_retry(rig, phase):
    owner = rig.make()
    drive(owner, phase)
    before = len(rig.state.native)
    owner.abort()
    frozen = {str(p): p.read_bytes() for p in owner.output.rglob("*") if p.is_file()}
    owner.advance()
    owner.close()
    assert owner.phase == "aborted" and len(rig.state.native) == before
    assert frozen == {str(p): p.read_bytes() for p in owner.output.rglob("*") if p.is_file()}
    with pytest.raises(BrowserSafetyStop, match="owner_already_claimed"):
        rig.make("new-output")


@pytest.mark.parametrize("change", ["source", "native", "star", "config", "completed", "deadline"])
def test_mutation_is_caught_before_another_native_action(rig, change):
    owner = rig.make()
    drive(owner, "probe_initializing")
    if change == "source":
        (rig.source / "chart.png").write_bytes(b"changed")
    elif change == "native":
        rig.state.signature[0][1] = "wrong answer"
    elif change == "star":
        rig.state.star = "OTHER"
    elif change == "config":
        owner.config.url = "http://different.invalid/"
    elif change == "completed":
        (owner.output / "initial/extra").mkdir()
    else:
        rig.state.clock = 900
    before = len(rig.state.native)
    owner.advance()
    assert owner.finished and owner.phase == "stopped"
    assert len(rig.state.native) == before and not (owner.output / "report.json").exists()


def test_pause_does_not_refresh_budget_or_advance_generator(rig):
    owner = rig.make()
    owner.pause()
    owner.tick()
    assert not rig.state.calls
    owner.step()
    assert owner.phase == "initial_capture"
    rig.state.clock = 900
    owner.step()
    assert owner.phase == "stopped" and not rig.state.native


def test_inflight_probe_expiry_preserves_actions_but_never_advances_again(rig):
    owner = rig.make()
    drive(owner, "probe_active")
    owner.advance()  # Initial read-only generator yield.
    rig.state.clock = 899

    def expire(_output, _index):
        rig.state.clock = 901

    rig.state.before_next = expire
    owner.advance()
    assert owner.phase == "stopped" and owner.failure == "shallow_steps_time_limit"
    assert owner.native_action_attempts == owner.native_actions_confirmed == 1
    before = list(rig.state.native)
    owner.advance()
    assert rig.state.native == before and not (owner.output / "report.json").exists()


@pytest.mark.parametrize("boundary", ["restore", "probe", "summary"])
def test_callback_abort_never_resumes_or_publishes_measurements(rig, boundary):
    holder = {}

    def emit(kind, payload):
        if (
            (boundary == "restore" and kind == "action_proposed" and "component_event" not in payload)
            or (boundary == "probe" and kind == "action_proposed" and "component_event" in payload)
            or (boundary == "summary" and kind == "episode_summary")
        ):
            holder["owner"].abort()

    owner = rig.make(emit=emit)
    holder["owner"] = owner
    drive(owner)
    assert owner.phase == "aborted" and owner.report["measurements"] is None
    assert not (owner.output / "report.json").exists()
    before = list(rig.state.native)
    owner.advance()
    assert rig.state.native == before
    if boundary == "restore":
        assert len(rig.state.native) == 10 and owner.native_action_attempts == 11
        assert all("probe" in payload for _, payload in rig.state.native)
    elif boundary == "probe":
        assert len(rig.state.native) == 1


def test_failed_probe_retains_dispatched_action_count_and_never_skips_feature(rig):
    rig.state.fail_probe = 0
    owner = rig.make()
    drive(owner)
    assert owner.phase == "stopped" and not owner.completed_probes
    assert owner.native_action_attempts == 1 and owner.native_actions_confirmed == 0
    assert owner.state()["native_action_outcome_uncertain"]
    assert not (owner.output / "probe-02").exists()


def test_restore_accepts_only_grounded_zero_glyph_omission_not_plot_change(rig):
    def change(directory):
        if directory.parent.name == "restore-01":
            image = Image.open(io.BytesIO(rig.state.image)).convert("RGB")
            image.putpixel((30, 21), (1, 2, 3))
            result = io.BytesIO()
            image.save(result, format="PNG")
            rig.state.image, rig.state.labels = result.getvalue(), rig.state.labels[1:]

    rig.state.capture_hook = change
    owner = rig.make()
    drive(owner)
    assert owner.phase == "measurements_ready", owner.report
    assert owner.report["settled_anchor"]["chart_sha256"] != owner.report["window_evidence"]["chart_sha256"]


def test_changed_restored_plot_fails_without_new_probe(rig):
    def change(directory):
        if directory.parent.name == "restore-01":
            image = Image.open(io.BytesIO(rig.state.image)).convert("RGB")
            image.putpixel((40, 30), (1, 2, 3))
            result = io.BytesIO()
            image.save(result, format="PNG")
            rig.state.image = result.getvalue()

    rig.state.capture_hook = change
    owner = rig.make()
    drive(owner)
    assert owner.phase == "stopped" and len(owner.completed_probes) == 1
    assert not (owner.output / "probe-02").exists()


def test_unresolved_probe_does_not_fall_back_or_claim_no(rig):
    rig.state.probe_hook = lambda report, _: report.update(
        status="unresolved", sampled_decline_verified=False
    )
    owner = rig.make()
    drive(owner)
    assert owner.phase == "stopped" and owner.report["planet_decision"] is None
    assert not owner.completed_probes and not (owner.output / "probe-02").exists()
