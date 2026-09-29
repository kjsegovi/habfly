"""Explicit single-event assumption routing; native calls are injected, not proven.

Real raster analysis, policy sidecars, choice receipts, Save source reconstruction
and replay stay active. Nothing here establishes absence, science or completion.
"""

# ruff: noqa: F811 - imported pytest fixtures

import json
from pathlib import Path

import pytest
from test_baseline_edge_integration import choice_rig, sha, write  # noqa: F401
from test_browser_planet_window import make, scheduled  # noqa: F401
from test_browser_star_session import advance_to, create
from test_browser_star_session import rig as star_rig  # noqa: F401
from test_browser_window_replay import capture
from test_planet_window_policy import crop

import habfly.browser_no_planet_save as save
import habfly.browser_planet_window_choice as choice
import habfly.browser_star_session as star
import habfly.planet_window_baseline_band as band
import habfly.planet_window_baseline_edge as edge
import habfly.planet_window_single_event as single
from habfly.browser import BrowserSafetyStop
from habfly.browser_window_replay import load_planet_window_capture

FLAGS = {
    "allow_baseline_edge_reference": True,
    "allow_baseline_band_reference": True,
    "allow_single_event_reference": True,
}


def single_source(directory, *, dip=(80, 30)):
    directory.mkdir(exist_ok=True)
    capture(directory, dip=dip)
    write(directory / "policy.json", single.policy_manifest())
    return directory / "report.json"


def choose(rig, monkeypatch, *, fresh_dip=(80, 30)):
    single_source(rig.path.parent)

    def fresh(page, config, directory, **kwargs):
        # The native screenshot seam alone is synthetic. _fresh writes and
        # validates its own actual selected-policy sidecar afterward.
        directory.mkdir()
        capture(directory, dip=fresh_dip)
        return json.loads((directory / "report.json").read_bytes())

    monkeypatch.setattr(choice, "capture_observation_progress", fresh)
    return choice.select_no_planet_from_window(
        object(),
        object(),
        rig.root / "choice",
        run_history=rig.root,
        evidence_path=rig.path,
        evidence_sha256=sha(rig.path),
        policy=single.policy_manifest(),
    )


@pytest.mark.parametrize("dip", [None, (80, 30)])
def test_new_source_and_replay_preserve_explicit_assumption_only(tmp_path, dip):
    path = single_source(tmp_path, dip=dip)
    before = {p.name: sha(p) for p in tmp_path.iterdir()}
    evidence = choice._evidence(path, sha(path), policy=single.policy_manifest())
    replay = load_planet_window_capture(tmp_path)
    assert replay["policy"] == evidence["analysis"]["policy"] == single.policy_manifest()
    assert evidence["policy_sha256"] == sha(tmp_path / "policy.json")
    assert replay["status"] == "assume_no_planet" and replay["approximate_measurements"] is None
    assert (replay.get("possible_planet_ignored") is True) if dip else "possible_planet_ignored" not in replay
    if dip:
        assert replay["visible_candidate_events"] == 1
        assert replay["approximation"] == "user_approved_single_event_no_planet_shortcut"
    for key in (
        "write_authorized",
        "task_completed",
        "scientific_verified",
        "training_label",
        "absence_proven",
    ):
        assert replay[key] is False
    assert {p.name: sha(p) for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("replacement", ["missing", "frozen", "edge", "band", "hash", "alias", "symlink"])
def test_no_implicit_upgrade_or_sidecar_relabel(tmp_path, replacement):
    path = single_source(tmp_path)
    sidecar = tmp_path / "policy.json"
    if replacement == "missing":
        sidecar.unlink()
    elif replacement == "symlink":
        sidecar.rename(tmp_path / "actual-policy.json")
        sidecar.symlink_to(tmp_path / "actual-policy.json")
    else:
        value = {
            "frozen": choice.frozen_manifest(),
            "edge": edge.policy_manifest(),
            "band": band.policy_manifest(),
            "hash": {**single.policy_manifest(), "sha256": "0" * 64},
            "alias": {**single.policy_manifest(), "user_approved": 1},
        }[replacement]
        write(sidecar, value)
    with pytest.raises(BrowserSafetyStop):
        choice._evidence(path, sha(path), policy=single.policy_manifest())


def test_new_policy_requires_matching_recorded_analysis(tmp_path):
    single_source(tmp_path)
    write(tmp_path / "analysis.json", {"policy": band.policy_manifest()})
    with pytest.raises(BrowserSafetyStop, match="replay_policy_mismatch"):
        load_planet_window_capture(tmp_path)


def test_real_analyzer_saved_fresh_preselect_and_save_reconstruction(choice_rig, monkeypatch):
    receipt = choose(choice_rig, monkeypatch)
    assert choice_rig.calls == ["No"]
    for key in ("saved_evidence", "fresh_evidence", "preselect_evidence"):
        analysis = receipt[key]["analysis"]
        assert analysis["possible_planet_ignored"] is True and analysis["visible_candidate_events"] == 1
        assert analysis["policy"] == receipt["policy"] == single.policy_manifest()
        assert len(receipt[key]["policy_sha256"]) == 64
    path = choice_rig.root / "choice/confirmed.json"
    assert save._load_choice(path, sha(path), choice_rig.root)[0] == receipt
    for key in ("absence_proven", "scientific_verified", "training_label", "task_completed"):
        assert receipt[key] is False


@pytest.mark.parametrize("stage", ["source", "choice/fresh-progress", "choice/preselect-progress"])
@pytest.mark.parametrize("change", ["policy", "bytes", "chart"])
def test_save_source_rebuild_rejects_each_mutated_boundary(choice_rig, monkeypatch, stage, change):
    choose(choice_rig, monkeypatch)
    directory = choice_rig.root / stage
    if change == "policy":
        write(directory / "policy.json", band.policy_manifest())
    elif change == "bytes":
        path = directory / "policy.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        (directory / "chart.png").write_bytes(crop())
    path = choice_rig.root / "choice/confirmed.json"
    with pytest.raises(BrowserSafetyStop):
        save._load_choice(path, sha(path), choice_rig.root)
    assert choice_rig.calls == ["No"]


def test_fresh_invalid_feature_blocks_no_without_retry(choice_rig, monkeypatch):
    # Each fresh crop must independently pass the explicit policy; a newly
    # clipped edge feature cannot reuse the saved interior feature's result.
    with pytest.raises(BrowserSafetyStop):
        choose(choice_rig, monkeypatch, fresh_dip=(31, 30))
    assert choice_rig.calls == []


@pytest.mark.parametrize("mode", ["frozen", "edge", "band"])
def test_historical_replay_is_byte_equivalent_without_importing_new_engine(tmp_path, monkeypatch, mode):
    capture(tmp_path)
    policy = {
        "frozen": choice.frozen_manifest(),
        "edge": edge.policy_manifest(),
        "band": band.policy_manifest(),
    }[mode]
    if mode != "frozen":
        write(tmp_path / "policy.json", policy)
    before = json.dumps(load_planet_window_capture(tmp_path), sort_keys=True, allow_nan=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy replay must not depend on single-event engine")

    for name in ("policy_manifest", "recorded_policy_manifest", "analyze_recorded_planet_window"):
        monkeypatch.setattr(single, name, forbidden)
    assert json.dumps(load_planet_window_capture(tmp_path), sort_keys=True, allow_nan=False) == before


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_strict_boolean_checked_before_any_output_or_child(scheduled, star_rig, value):
    with pytest.raises(ValueError):
        make(scheduled, **{**FLAGS, "allow_single_event_reference": value})
    with pytest.raises(ValueError):
        create(star_rig, **{**FLAGS, "allow_single_event_reference": value})
    assert not scheduled.calls and not star_rig.calls
    assert not (scheduled.root / "window").exists() and not (star_rig.root / "run").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"allow_single_event_reference": True},
        {"allow_baseline_edge_reference": True, "allow_single_event_reference": True},
        {"allow_baseline_band_reference": True, "allow_single_event_reference": True},
    ],
)
def test_new_permission_requires_entire_explicit_dependency_chain(scheduled, star_rig, options):
    with pytest.raises(ValueError):
        make(scheduled, **options)
    with pytest.raises(ValueError):
        create(star_rig, **options)
    assert not scheduled.calls and not star_rig.calls


@pytest.mark.parametrize("enabled", [False, True])
def test_star_forwards_only_explicit_permission_and_source_pin(star_rig, monkeypatch, enabled):
    original, seen = star.PlanetWindowSession, []

    def window(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(star, "PlanetWindowSession", window)
    session = create(star_rig, **(FLAGS if enabled else {}))
    advance_to(session, "planet_window")
    session.advance()
    assert set(seen[0]) == {"run_history", "select_no", "emit"} | (FLAGS.keys() if enabled else set())
    if enabled:
        assert session.scope["single_event_reference_enabled"] is True
        assert session.scope["single_event_source_sha256"] == sha(Path(single.__file__))
        assert seen[0]["allow_single_event_reference"] is True
    else:
        assert "single_event_reference_enabled" not in session.scope
    session.abort()


def test_window_uses_new_policy_keeps_fixed_caps_and_exposes_actual_assumption(scheduled):
    scheduled.images[:] = [crop(dip=(80, 30))]
    session = make(scheduled, select_no=True, **FLAGS)
    assert session.scope["max_seconds"] == 600 and session.max_polls == 60
    assert session.scope["single_event_source_sha256"] == sha(Path(single.__file__))
    assert "analysis" not in session.state()
    session.advance()
    scheduled.clock[0] += 5
    state = session.advance()
    assert state["phase"] == "ready_to_select_no"
    assert state["analysis"]["possible_planet_ignored"] is True
    assert state["analysis"]["visible_candidate_events"] == 1
    assert load_planet_window_capture(session.output / "progress-000")["policy"] == single.policy_manifest()
    assert session.advance()["phase"] == "no_selected"
    assert scheduled.calls == [("start", 5000), ("capture", 5000), ("choice", "No")]
    assert not session.report["task_completed"]


def test_explicit_new_policy_does_not_erase_prior_detected_dip(scheduled):
    scheduled.images[:] = [crop(end=130, dip=(80, 30)), crop(dip=(80, 30))]
    session = make(scheduled, select_no=True, **FLAGS)
    session.advance()
    scheduled.clock[0] += 5
    assert session.advance()["phase"] == "observing" and session.state()["prior_dip_observed"]
    scheduled.clock[0] += 5
    state = session.advance()
    assert state["phase"] == "stopped" and state["failure_reason"] == "planet_window_prior_dip_conflict"
    assert state["analysis"]["possible_planet_ignored"] is True
    assert session.report["first_dip_evidence"] and not session.report["task_completed"]
    assert not any(call[0] == "choice" for call in scheduled.calls)


def mutate(session, change):
    if change == "flag":
        session.allow_single_event_reference = False
    elif change == "alias":
        session.allow_single_event_reference = 1
    elif change == "scope":
        session.scope["single_event_reference_enabled"] = False
    elif change == "band":
        session.allow_baseline_band_reference = False
    elif change == "disk":
        path = session.output / "scope.json"
        path.write_bytes(path.read_bytes() + b" ")


@pytest.mark.parametrize("change", ["flag", "alias", "scope", "band", "disk", "source", "callback"])
def test_window_revocation_stops_before_any_initial_native_action(scheduled, monkeypatch, change):
    session = make(scheduled, **FLAGS)
    if change == "source":
        original = Path.read_bytes
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda p: original(p) + b" " if p == session._single_event_source else original(p),
        )
    elif change == "callback":
        session.emit = lambda *_: mutate(session, "flag")
    else:
        mutate(session, change)
    session.advance()
    assert session.phase == "stopped" and not scheduled.calls
    assert session.failure == "planet_window_single_event_permission_changed"


def test_window_no_callback_revocation_blocks_selection(scheduled):
    scheduled.images[:] = [crop(dip=(80, 30))]
    session = make(scheduled, select_no=True, **FLAGS)
    session.advance()
    scheduled.clock[0] += 5
    session.advance()
    session.emit = lambda *_: mutate(session, "flag")
    assert session.advance()["phase"] == "stopped"
    assert scheduled.calls == [("start", 5000), ("capture", 5000)]


@pytest.mark.parametrize("change", ["flag", "alias", "scope", "band", "disk", "source"])
def test_star_action_callback_revocation_never_returns_to_dispatch(star_rig, monkeypatch, change):
    session = create(star_rig, **FLAGS)
    session.advance()
    child, native, altered = star_rig.components[0], [], [False]
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: original(p) + b" " if p == session._single_event_source and altered[0] else original(p),
    )

    def proposed():
        child.emit(
            {"event": "action_proposed", "run_id": "numeric", "sequence": 0, "payload": {"kind": "TYPE"}}
        )
        native.append("dispatch")

    def revoke(event, payload):
        if event == "action_proposed":
            if change == "source":
                altered[0] = True
            else:
                mutate(session, change)

    child.advance, session._callback = proposed, revoke
    session.advance()
    assert session.phase == "stopped" and not native and not session.state()["task_completed"]
