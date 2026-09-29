"""Pure explicit-third-policy routing; native transport seams are declared.

These tests do not establish planet absence or course completion. All analyzer,
recorded-policy, choice-source and Save-replay validation stays enabled.
"""

# ruff: noqa: F811 - imported pytest fixture dependencies

import hashlib
import json
from pathlib import Path

import pytest
from test_baseline_edge_integration import choice_rig, sha, source, write  # noqa: F401
from test_browser_planet_window import make, scheduled  # noqa: F401
from test_browser_window_replay import capture
from test_planet_window_policy import crop
from test_shallow_two_hint_stop import axes, representative_png

import habfly.browser_no_planet_save as save_module
import habfly.browser_planet_window_choice as choice
import habfly.planet_window_baseline_band as band
import habfly.planet_window_baseline_edge as edge
from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_progress
from habfly.browser_window_replay import load_planet_window_capture


def band_source(directory, *, localized=False):
    directory.mkdir(exist_ok=True)
    if localized:
        png = representative_png()
        times, flux = axes()
        report = {
            **trace_progress(png, times, requested_days=5000),
            "star": "FIXTURE",
            "time_axis_labels": times,
            "flux_axis_labels": flux,
            "chart_sha256": hashlib.sha256(png).hexdigest(),
            "browser_actions": 0,
            "answer_writes": 0,
        }
        (directory / "chart.png").write_bytes(png)
        write(directory / "report.json", report)
    else:
        capture(directory)
    write(directory / "policy.json", band.policy_manifest())
    return directory / "report.json"


def band_choose(rig):
    write(rig.path.parent / "policy.json", band.policy_manifest())
    return choice.select_no_planet_from_window(
        object(),
        object(),
        rig.root / "choice",
        run_history=rig.root,
        evidence_path=rig.path,
        evidence_sha256=sha(rig.path),
        policy=band.policy_manifest(),
    )


@pytest.mark.parametrize("localized", [False, True])
def test_choice_source_and_offline_replay_bind_explicit_third_policy(tmp_path, localized):
    path = band_source(tmp_path, localized=localized)
    original = (tmp_path / "chart.png").read_bytes()
    evidence = choice._evidence(path, sha(path), policy=band.policy_manifest())
    replay = load_planet_window_capture(tmp_path)
    assert replay["status"] == "assume_no_planet"
    assert evidence["analysis"]["policy"] == replay["policy"] == band.policy_manifest()
    assert evidence["policy_sha256"] == sha(tmp_path / "policy.json")
    assert (tmp_path / "chart.png").read_bytes() == original
    assert not any(
        replay[k] for k in ("write_authorized", "task_completed", "scientific_verified", "training_label")
    )


@pytest.mark.parametrize("replacement", [None, "edge", "frozen", "unknown", "alias"])
def test_third_policy_sidecar_cannot_silently_change_or_upgrade(tmp_path, replacement):
    path = band_source(tmp_path)
    if replacement is None:
        (tmp_path / "policy.json").unlink()
    else:
        selected = {
            "edge": edge.policy_manifest(),
            "frozen": choice.frozen_manifest(),
            "unknown": {**band.policy_manifest(), "sha256": "0" * 64},
            "alias": {**band.policy_manifest(), "user_approved": 1},
        }[replacement]
        write(tmp_path / "policy.json", selected)
    with pytest.raises(BrowserSafetyStop):
        choice._evidence(path, sha(path), policy=band.policy_manifest())


def test_saved_analysis_cannot_mix_new_and_old_policy(tmp_path):
    band_source(tmp_path)
    write(tmp_path / "analysis.json", {"policy": edge.policy_manifest()})
    with pytest.raises(BrowserSafetyStop, match="replay_policy_mismatch"):
        load_planet_window_capture(tmp_path)


def test_saved_fresh_preselect_and_save_replay_use_same_band_identity(choice_rig):
    receipt = band_choose(choice_rig)
    assert choice_rig.calls == ["No"]
    for label in ("saved_evidence", "fresh_evidence", "preselect_evidence"):
        assert receipt[label]["analysis"]["policy"] == receipt["policy"] == band.policy_manifest()
        assert len(receipt[label]["policy_sha256"]) == 64
    path = choice_rig.root / "choice/confirmed.json"
    assert save_module._load_choice(path, sha(path), choice_rig.root)[0] == receipt
    assert not receipt["task_completed"] and not receipt["absence_proven"]


@pytest.mark.parametrize("stage", ["source", "choice/fresh-progress", "choice/preselect-progress"])
def test_save_replay_rejects_cross_policy_at_each_boundary(choice_rig, stage):
    band_choose(choice_rig)
    write(choice_rig.root / stage / "policy.json", edge.policy_manifest())
    path = choice_rig.root / "choice/confirmed.json"
    with pytest.raises(BrowserSafetyStop):
        save_module._load_choice(path, sha(path), choice_rig.root)
    assert choice_rig.calls == ["No"]


def test_preselect_policy_switch_is_not_dispatched_or_retried(choice_rig, monkeypatch):
    original = choice._fresh

    def changed(page, config, directory, **kwargs):
        result = original(page, config, directory, **kwargs)
        if directory.name == "preselect-progress":
            result["analysis"]["policy"] = edge.policy_manifest()
        return result

    monkeypatch.setattr(choice, "_fresh", changed)
    with pytest.raises(BrowserSafetyStop, match="context_changed"):
        band_choose(choice_rig)
    stopped = json.loads((choice_rig.root / "choice/stopped.json").read_bytes())
    assert stopped["reservation_created"] and not stopped["write_may_have_occurred"]
    assert choice_rig.calls == []


@pytest.mark.parametrize("new_edge", [False, True])
def test_old_choice_save_replay_outputs_do_not_depend_on_new_band_module(choice_rig, monkeypatch, new_edge):
    selected = edge.policy_manifest() if new_edge else choice.frozen_manifest()
    if not new_edge:
        (choice_rig.path.parent / "policy.json").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Old policy must not depend on optional band engine")

    for name in ("policy_manifest", "recorded_policy_manifest", "analyze_recorded_planet_window"):
        monkeypatch.setattr(band, name, forbidden)
    receipt = choice.select_no_planet_from_window(
        object(),
        object(),
        choice_rig.root / "choice",
        run_history=choice_rig.root,
        evidence_path=choice_rig.path,
        evidence_sha256=sha(choice_rig.path),
        **({"policy": selected} if new_edge else {}),
    )
    path = choice_rig.root / "choice/confirmed.json"
    assert save_module._load_choice(path, sha(path), choice_rig.root)[0] == receipt
    assert load_planet_window_capture(choice_rig.path.parent)["policy"] == selected
    assert ("policy_sha256" in receipt["saved_evidence"]) is new_edge


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_sensor_flag_is_strict_and_validated_before_output(scheduled, value):
    with pytest.raises(ValueError):
        make(scheduled, allow_baseline_edge_reference=True, allow_baseline_band_reference=value)
    assert not (scheduled.root / "window").exists() and not scheduled.calls


def test_sensor_new_policy_requires_existing_edge_permission(scheduled):
    with pytest.raises(ValueError):
        make(scheduled, allow_baseline_band_reference=True)
    assert not scheduled.calls


def test_sensor_new_policy_sidecar_and_fixed_budgets(scheduled):
    scheduled.images[:] = [crop()]
    session = make(
        scheduled, select_no=True, allow_baseline_edge_reference=True, allow_baseline_band_reference=True
    )
    assert session.scope["max_seconds"] == 600 and session.max_polls == 60
    assert session.scope["baseline_band_reference_enabled"] is True
    assert session.scope["policy"] == band.policy_manifest()
    assert session.scope["baseline_band_source_sha256"] == sha(Path(band.__file__))
    session.advance()
    scheduled.clock[0] += 5
    assert session.advance()["phase"] == "ready_to_select_no"
    directory = session.output / "progress-000"
    assert json.loads((directory / "policy.json").read_bytes()) == band.policy_manifest()
    assert load_planet_window_capture(directory)["policy"] == band.policy_manifest()
    assert session.advance()["phase"] == "no_selected"
    assert scheduled.calls == [("start", 5000), ("capture", 5000), ("choice", "No")]
    assert not session.report["task_completed"]


@pytest.mark.parametrize("change", ["flag", "alias", "scope", "policy", "source", "callback"])
def test_sensor_permission_and_source_drift_stop_before_start(scheduled, monkeypatch, change):
    session = make(scheduled, allow_baseline_edge_reference=True, allow_baseline_band_reference=True)
    if change in {"flag", "alias"}:
        session.allow_baseline_band_reference = False if change == "flag" else 1
    elif change == "scope":
        session.scope["baseline_band_reference_enabled"] = False
    elif change == "policy":
        session.scope["policy"] = edge.policy_manifest()
    elif change == "source":
        read = Path.read_bytes
        monkeypatch.setattr(
            Path, "read_bytes", lambda p: read(p) + b" " if p == session._baseline_band_source else read(p)
        )
    else:
        session.emit = lambda *_: setattr(session, "allow_baseline_band_reference", False)
    session.advance()
    assert session.failure == "planet_window_baseline_band_permission_changed"
    assert session.phase == "stopped" and not scheduled.calls


def test_band_permission_change_at_no_callback_prevents_selection(scheduled):
    scheduled.images[:] = [crop()]
    session = make(
        scheduled, select_no=True, allow_baseline_edge_reference=True, allow_baseline_band_reference=True
    )
    session.advance()
    scheduled.clock[0] += 5
    session.advance()
    session.emit = lambda *_: setattr(session, "allow_baseline_band_reference", False)
    session.advance()
    assert session.phase == "stopped" and session.failure == "planet_window_baseline_band_permission_changed"
    assert scheduled.calls == [("start", 5000), ("capture", 5000)]
