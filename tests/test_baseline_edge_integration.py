"""Offline policy/owner integration; native controls are explicitly injected."""
# ruff: noqa: F401,F811

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_planet_window import make, scheduled
from test_browser_project_runtime import rig as bridge_rig
from test_browser_star_session import advance_to, create
from test_browser_star_session import rig as star_rig
from test_browser_window_replay import capture
from test_runtime_browser_project import integrated, options, start

import habfly.browser_no_planet_save as save_module
import habfly.browser_planet_window as window_module
import habfly.browser_planet_window_choice as choice_module
import habfly.browser_star_session as star_module
from habfly.browser import BrowserSafetyStop
from habfly.browser_window_replay import load_planet_window_capture
from habfly.planet_window_baseline_edge import policy_manifest, recorded_policy_manifest
from habfly.runtime import RunOptions, parse_run_options


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def source(tmp_path, *, new=True):
    tmp_path.mkdir(exist_ok=True)
    capture(tmp_path)
    if new:
        write(tmp_path / "policy.json", policy_manifest())
    return tmp_path / "report.json"


@pytest.mark.parametrize("new", [False, True])
def test_offline_source_and_replay_dispatch_exact_recorded_policy(tmp_path, new):
    path = source(tmp_path, new=new)
    expected = policy_manifest() if new else recorded_policy_manifest()
    kwargs = {"policy": expected} if new else {}
    evidence = choice_module._evidence(path, sha(path), **kwargs)
    replay = load_planet_window_capture(tmp_path)
    assert evidence["analysis"]["policy"] == replay["policy"] == expected
    assert ("policy_sha256" in evidence) is new
    if new:
        assert evidence["policy_sha256"] == sha(tmp_path / "policy.json")
    assert not replay["write_authorized"] and not replay["task_completed"]
    assert not replay["scientific_verified"] and not replay["training_label"]


@pytest.mark.parametrize("kind", ["missing", "implicit", "legacy", "unknown", "boolean", "mixed", "symlink"])
def test_sidecar_cannot_upgrade_downgrade_or_change_policy(tmp_path, kind):
    path = source(tmp_path)
    sidecar = tmp_path / "policy.json"
    selected = policy_manifest()
    if kind == "missing":
        sidecar.unlink()
    elif kind == "implicit":
        selected = None
    elif kind == "legacy":
        write(sidecar, recorded_policy_manifest())
    elif kind in {"unknown", "boolean", "mixed"}:
        changed = policy_manifest()
        if kind == "unknown":
            changed["sha256"] = "0" * 64
        elif kind == "boolean":
            changed["user_approved"] = 1
        else:
            changed["version"] = recorded_policy_manifest()["version"]
        write(sidecar, changed)
    else:
        sidecar.rename(tmp_path / "policy-source.json")
        sidecar.symlink_to(tmp_path / "policy-source.json")
    with pytest.raises(BrowserSafetyStop):
        choice_module._evidence(path, sha(path), policy=selected)


def test_sidecar_hash_is_same_read_as_validated_manifest(tmp_path, monkeypatch):
    path = source(tmp_path)
    sidecar, original_read = tmp_path / "policy.json", Path.read_bytes
    expected, reads = sha(sidecar), []

    def changing_read(p):
        if p == sidecar:
            reads.append(1)
            if len(reads) > 1:
                return b"{}"
        return original_read(p)

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    evidence = choice_module._evidence(path, sha(path), policy=policy_manifest())
    assert evidence["policy_sha256"] == expected and len(reads) == 1


@pytest.mark.parametrize("new", [False, True])
def test_replay_rejects_mixed_saved_analysis(tmp_path, new):
    source(tmp_path, new=new)
    write(tmp_path / "analysis.json", {"policy": recorded_policy_manifest() if new else policy_manifest()})
    with pytest.raises(BrowserSafetyStop, match="replay_policy_mismatch"):
        load_planet_window_capture(tmp_path)


@pytest.fixture
def choice_rig(tmp_path, monkeypatch):
    """Only native transport/capture seams fake; policy and receipt loaders real."""
    calls = []
    mapping = {
        "star_name": "FIXTURE",
        "observation": {
            "values": {
                "has_planet": None,
                "browser_field_map": {
                    name: {"current_value": "5000" if name == "observation_days" else ""}
                    for name in save_module.BASE_FIELDS
                },
            }
        },
    }

    class Control:
        def evaluate(self, script, other):
            return other is self

        def select_option(self, *, label, timeout):
            calls.append(label)
            mapping["observation"]["values"]["has_planet"] = label

        def count(self):
            return 1

        def element_handle(self, **kwargs):
            return self

    control = Control()

    class Session:
        frame = SimpleNamespace(get_by_role=lambda *_: control)

        def __init__(self, *args, **kwargs):
            pass

        def current(self):
            return {}, deepcopy(mapping), {"selected": None}, None

        read = current

        def close(self):
            pass

    def save_probe(value, directory):
        write(directory / "observation.json", value)

    def fresh_capture(page, config, directory, **kwargs):
        directory.mkdir()
        capture(directory)
        return json.loads((directory / "report.json").read_bytes())

    monkeypatch.setattr(choice_module, "PlanetNumericSession", Session)
    monkeypatch.setattr(choice_module, "_control", lambda _: control)
    monkeypatch.setattr(choice_module, "save_probe", save_probe)
    monkeypatch.setattr(choice_module, "capture_observation_progress", fresh_capture)
    monkeypatch.setattr(choice_module, "presence_projection", lambda *args: "injected-native-projection")
    monkeypatch.setattr(choice_module, "preserved_paint_transition", lambda *args: True)
    monkeypatch.setattr(choice_module, "rendered_control", lambda _: True)
    monkeypatch.setattr(save_module, "load_planet_capture", lambda _: deepcopy(mapping))
    path = source(tmp_path / "source")
    return SimpleNamespace(root=tmp_path, calls=calls, path=path, mapping=mapping)


def choose(rig, **kwargs):
    return choice_module.select_no_planet_from_window(
        object(),
        object(),
        rig.root / "choice",
        run_history=rig.root,
        evidence_path=rig.path,
        evidence_sha256=sha(rig.path),
        policy=policy_manifest(),
        **kwargs,
    )


def test_saved_fresh_preselect_and_save_reader_all_recompute_same_policy(choice_rig):
    receipt = choose(choice_rig)
    assert choice_rig.calls == ["No"]
    for label in ("saved_evidence", "fresh_evidence", "preselect_evidence"):
        assert receipt[label]["analysis"]["policy"] == receipt["policy"] == policy_manifest()
        assert len(receipt[label]["policy_sha256"]) == 64
    path = choice_rig.root / "choice/confirmed.json"
    actual, _, _ = save_module._load_choice(path, sha(path), choice_rig.root)
    assert actual == receipt
    assert actual["task_completed"] is False and actual["absence_proven"] is False


@pytest.mark.parametrize("which", ["source", "choice/fresh-progress", "choice/preselect-progress"])
@pytest.mark.parametrize("kind", ["remove", "change", "relabel"])
def test_save_reader_rejects_changed_or_mixed_sidecar(choice_rig, which, kind):
    choose(choice_rig)
    sidecar = choice_rig.root / which / "policy.json"
    if kind == "remove":
        sidecar.unlink()
    elif kind == "change":
        sidecar.write_text(sidecar.read_text() + " ")
    else:
        write(sidecar, recorded_policy_manifest())
    path = choice_rig.root / "choice/confirmed.json"
    with pytest.raises(BrowserSafetyStop):
        save_module._load_choice(path, sha(path), choice_rig.root)
    assert choice_rig.calls == ["No"]


def test_changed_policy_at_preselect_stops_before_dispatch_and_retains_claim(choice_rig, monkeypatch):
    fresh = choice_module._fresh

    def mismatch(page, config, directory, **kwargs):
        result = fresh(page, config, directory, **kwargs)
        if directory.name == "preselect-progress":
            result["analysis"]["policy"] = recorded_policy_manifest()
        return result

    monkeypatch.setattr(choice_module, "_fresh", mismatch)
    with pytest.raises(BrowserSafetyStop, match="context_changed"):
        choose(choice_rig)
    stopped = json.loads((choice_rig.root / "choice/stopped.json").read_bytes())
    assert stopped["reservation_created"] and not stopped["write_may_have_occurred"]
    assert choice_rig.calls == []


def test_window_records_new_policy_before_choice_without_new_budget(scheduled):
    from test_planet_window_policy import crop

    scheduled.images[:] = [crop()]
    session = make(scheduled, select_no=True, allow_baseline_edge_reference=True)
    assert session.scope["max_seconds"] == 600 and session.scope["policy"] == policy_manifest()
    session.advance()
    scheduled.clock[0] += 5
    assert session.advance()["phase"] == "ready_to_select_no"
    path = session.output / "progress-000/policy.json"
    assert json.loads(path.read_bytes()) == policy_manifest()
    assert load_planet_window_capture(path.parent)["policy"] == policy_manifest()
    assert session.advance()["phase"] == "no_selected"
    assert scheduled.calls[-1] == ("choice", "No")


@pytest.mark.parametrize("change", ["flag", "bool_alias", "scope", "policy", "source", "callback"])
def test_window_mutation_cannot_start_observation(scheduled, monkeypatch, change):
    session = make(scheduled, allow_baseline_edge_reference=True)
    if change in {"flag", "bool_alias"}:
        session.allow_baseline_edge_reference = False if change == "flag" else 1
    elif change == "scope":
        session.scope["baseline_edge_reference_enabled"] = False
    elif change == "policy":
        session.scope["policy"] = recorded_policy_manifest()
    elif change == "source":
        read = Path.read_bytes
        monkeypatch.setattr(
            Path, "read_bytes", lambda p: read(p) + b" " if p == session._baseline_source else read(p)
        )
    else:
        session.emit = lambda *_: setattr(session, "allow_baseline_edge_reference", False)
    session.advance()
    assert session.phase == "stopped" and scheduled.calls == []
    assert session.failure == "planet_window_baseline_edge_permission_changed"


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_strict_runtime_flag_rejects_aliases_before_launch(tmp_path, value):
    with pytest.raises(ValueError):
        parse_run_options(options(tmp_path, project_baseline_edge_reference=value))


def test_nonproject_flag_is_rejected_and_legacy_default_false():
    assert RunOptions().project_baseline_edge_reference is False
    with pytest.raises(ValueError, match="requires_project_task"):
        parse_run_options({"project_baseline_edge_reference": True})


def test_runtime_forwards_optin_and_mutation_stops_before_browser(integrated):
    bridge = start(integrated, project_baseline_edge_reference=True)
    assert bridge.model_options["allow_baseline_edge_reference"] is True
    assert bridge.scope["baseline_edge_reference_enabled"] is True
    assert not integrated.calls
    integrated.runtime.options.project_baseline_edge_reference = False
    with pytest.raises(ValueError, match="authorization_changed"):
        integrated.runtime.advance_if_due()
    assert not integrated.calls


def test_legacy_choice_save_and_replay_never_call_optional_engine(choice_rig, monkeypatch):
    import habfly.planet_window_baseline_edge as optional

    (choice_rig.path.parent / "policy.json").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy execution must not depend on optional engine")

    monkeypatch.setattr(optional, "recorded_policy_manifest", forbidden)
    monkeypatch.setattr(optional, "analyze_recorded_planet_window", forbidden)
    monkeypatch.setattr(optional, "policy_manifest", forbidden)
    receipt = choice_module.select_no_planet_from_window(
        object(),
        object(),
        choice_rig.root / "choice",
        run_history=choice_rig.root,
        evidence_path=choice_rig.path,
        evidence_sha256=sha(choice_rig.path),
    )
    path = choice_rig.root / "choice/confirmed.json"
    assert save_module._load_choice(path, sha(path), choice_rig.root)[0] == receipt
    assert "policy_sha256" not in receipt["saved_evidence"]
    replay = load_planet_window_capture(choice_rig.path.parent)
    assert replay["policy"] == choice_module.frozen_manifest()
    assert not list(choice_rig.root.rglob("policy.json"))


def test_star_forwards_explicit_policy_without_defaulting(star_rig, monkeypatch):
    factory, seen = star_module.PlanetWindowSession, []

    def window(*args, **kwargs):
        seen.append(kwargs)
        return factory(*args, **kwargs)

    monkeypatch.setattr(star_module, "PlanetWindowSession", window)
    session = create(star_rig, allow_baseline_edge_reference=True)
    advance_to(session, "planet_window")
    session.advance()
    assert seen[0]["allow_baseline_edge_reference"] is True
    assert session.scope["baseline_edge_reference_enabled"] is True


def test_only_capped_supplied_three_and_held_thirty_profiles_opt_in():
    selected = {
        Path("configs/browser_project_supplied_three_star.json"): 3,
        Path("configs/browser_project_two_event_three_star.json"): 3,
        Path("configs/browser_project_baseline_band_three_star.json"): 3,
        Path("configs/browser_project_single_event_three_star.json"): 3,
        Path("configs/browser_project_thirty_star.json"): 30,
    }
    for path in Path("configs").glob("browser_project*.json"):
        data = json.loads(path.read_bytes())
        assert data.get("project_baseline_edge_reference", False) is (path in selected)
        if path in selected:
            assert data["stars"] == selected[path] and data["paused"] is True
