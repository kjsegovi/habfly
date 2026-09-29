"""Autonomous scheduling uses local decisions, never fixture browser authorities."""

# ruff: noqa: F811 - imported pytest fixtures
import json
from copy import deepcopy

import pytest
from test_browser_project_reference_class import (  # noqa: F401  # noqa: F401
    RATIONALE,
    bridge_rig,
    fake_class_factory,
    ready,
    rig,
)
from test_browser_project_runtime import sha

from habfly.browser import BrowserSafetyStop
from habfly.project_wire import compact_project_event
from habfly.runtime import parse_run_options


def automatic(rig, monkeypatch, **kwargs):
    rig.options.project_autonomous_decisions = True
    bridge = ready(rig, reference_planet_continuation=True, terrestrial_options={}, **kwargs)
    result = {
        "decision_kind": "stellar_class",
        "status": "reference_decision",
        "payload": {
            "selected_class": "main_sequence",
            "lifetime_prefix": "Ga",
            "reference_rationale": RATIONALE,
        },
        "source_sha256": {
            "initial-star/stellar/observation.json": sha(
                bridge.output / "initial-star/stellar/observation.json"
            )
        },
        "classification_learned": False,
    }
    monkeypatch.setattr(bridge, "_make_autonomous_decision", lambda: deepcopy(result))
    return bridge, result


def test_preparation_application_and_native_dispatch_are_separate_ticks(rig, monkeypatch):
    bridge, _ = automatic(rig, monkeypatch)
    bridge.resume()
    assert bridge.status == "running" and bridge.project.status == "paused"
    bridge.tick()
    assert bridge.phase == "awaiting_class_source" and bridge.autonomous_decision["status"] == "prepared"
    assert bridge.class_setup is None
    bridge.tick()
    assert bridge.phase == "setting_reference_class" and not bridge.class_setup.writes
    bridge.tick()
    assert bridge.class_setup.writes == ["class"]
    bridge.tick()
    bridge.tick()
    assert bridge.phase == "ready" and bridge.status == bridge.project.status == "running"
    assert bridge.autonomous_decision["learned"] is False
    assert len(list((bridge.output / "autonomous-decisions").rglob("*.json"))) == 1
    assert not bridge.state()["task_completed"] and not bridge.state()["project_completed"]
    bridge.close()


def test_single_step_and_pause_do_not_run_later_actions(rig, monkeypatch):
    bridge, _ = automatic(rig, monkeypatch)
    bridge.step()
    bridge.tick()
    assert bridge.class_setup is None and bridge.status == "paused"
    bridge.step()
    assert bridge.class_setup is not None and not bridge.class_setup.writes
    bridge.advance_if_due()
    assert not bridge.class_setup.writes
    bridge.abort()
    bridge.tick()
    bridge.step()
    assert not bridge.class_setup.writes
    bridge.close()


@pytest.mark.parametrize("change", ["source", "record", "decision", "star", "authority"])
def test_changed_source_or_decision_cannot_dispatch(rig, monkeypatch, change):
    bridge, result = automatic(rig, monkeypatch)
    bridge.step()
    if change == "source":
        (bridge.output / "initial-star/stellar/observation.json").write_text("{}")
    elif change == "record":
        (bridge.output / bridge._autonomous_pending["path"]).write_text("{}")
    elif change == "decision":
        result["payload"]["lifetime_prefix"] = "Ma"
    elif change == "star":
        bridge._autonomous_pending["record"]["current_star"]["star"] = "FOREIGN"
    else:
        bridge.autonomous_decisions = False
    bridge.step()
    assert bridge.status == "stopped" and bridge.class_setup is None
    assert not bridge.report["task_completed"]
    bridge.close()


@pytest.mark.parametrize("effect", ["pause", "abort", "reentrant", "throw"])
def test_prepare_callback_never_dispatches_a_reference_choice(rig, monkeypatch, effect):
    bridge, _ = automatic(rig, monkeypatch)
    original = bridge._callback

    def callback(kind, payload):
        original(kind, payload)
        if (payload.get("autonomous_decision") or {}).get("status") != "prepared":
            return
        if effect == "pause":
            bridge.pause()
        elif effect == "abort":
            bridge.abort()
        elif effect == "reentrant":
            bridge.step()
        else:
            raise RuntimeError("private callback text")

    bridge._callback = callback
    bridge.resume()
    bridge.tick()
    assert bridge.class_setup is None
    assert bridge.status == ("paused" if effect == "pause" else "aborted" if effect == "abort" else "stopped")
    assert "private callback text" not in bridge.trace_path.read_text()
    bridge.close()


def test_manual_command_cannot_override_autonomous_choice(rig, monkeypatch):
    bridge, _ = automatic(rig, monkeypatch)
    with pytest.raises(BrowserSafetyStop, match="autonomous_mode_rejects_supplied_decisions"):
        bridge.command(
            {
                "command": "step",
                "payload": {
                    "reference_class": {
                        "selected_class": "white_dwarf",
                        "reference_rationale": RATIONALE,
                        "lifetime_prefix": None,
                    }
                },
            }
        )
    assert bridge.class_setup is None
    bridge.close()


def test_compact_wire_preserves_autonomous_provenance_not_learned_confidence(rig, monkeypatch):
    bridge, _ = automatic(rig, monkeypatch)
    bridge.step()
    result = compact_project_event("state", bridge.state(), source_trace="events.jsonl")
    assert result["autonomous_decisions_enabled"] is True
    assert result["experimental_hr_matcher_enabled"] is True
    assert result["decision_source"] == "local_reference_tools_not_learned"
    assert result["autonomous_decision"] == bridge.autonomous_decision
    assert "confidence" not in json.dumps(result["autonomous_decision"])
    bridge.close()


def test_abstention_preserves_evidence_without_native_choice_or_default(rig, monkeypatch):
    bridge, result = automatic(rig, monkeypatch)
    result.update(status="abstained", payload=None, reason="ambiguous_reference")
    bridge.step()
    assert bridge.finished and bridge.failure == "project_runtime_autonomous_reference_abstained"
    assert bridge.class_setup is None
    saved = json.loads((bridge.output / bridge.autonomous_decision["artifact"]).read_bytes())
    assert saved["decision"] == result and saved["browser_actions"] == 0
    assert bridge.autonomous_decision["status"] == "abstained"
    assert bridge.autonomous_decision["reason"] == "ambiguous_reference"
    bridge.step()
    bridge.tick()
    assert bridge.class_setup is None
    bridge.close()


def test_autonomous_profile_requires_all_seven_gas_candidates(tmp_path):
    from test_browser_project_terrestrial_runtime import configured

    payload = configured(tmp_path)
    payload["project_autonomous_decisions"] = True
    with pytest.raises(ValueError, match="requires_complete_planet_options"):
        parse_run_options(payload)
    payload["browser_habitability_gas_candidates"] = ["CH4", "CO2", "H2O", "H2S", "N2O", "NH3", "O3"]
    assert parse_run_options(payload).project_autonomous_decisions
