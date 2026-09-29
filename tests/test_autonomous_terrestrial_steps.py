"""Injected stage scheduling; native/physical acceptance remains separate."""
# ruff: noqa: F811

import hashlib
import json

import pytest
from test_browser_terrestrial_steps import reach, rig  # noqa: F401

import habfly.autonomous_planet as decisions
from habfly.browser import BrowserSafetyStop
from habfly.browser_habitability import GASES

RATIONALE = "Explicit approximate pixel-template comparison; not learned chemistry or course truth."


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(rig, *, status="decided", hook=None):
    def decide(history, comparison, comparison_sha, selection, selection_sha, *, expected_star):
        assert history == rig.root and expected_star == "Terra"
        assert comparison_sha == digest(comparison / "report.json")
        assert selection_sha == digest(selection / "report.json")
        assert rig.values()["selected_gases"] == ["CO2"]
        assert rig.values()["equilibrium_temp"]["value"] == ""
        if hook:
            hook()
        return {
            "status": status,
            "payload": {"supplied_greenhouse_increment": 30} if status == "decided" else None,
            "source_sha256": {str((selection / "report.json").relative_to(history)): selection_sha},
            "evidence": {"absorption_percent": "50"},
            "provenance": "reference_prediction",
            "task_completed": False,
            "scientific_verified": False,
        }

    return decide


def test_autonomous_gas_decision_is_offline_and_original_record_remains_immutable(rig, monkeypatch):
    owner = rig.make(candidates=list(GASES))
    reach(owner, "awaiting_gases")
    calls = list(rig.state.calls)
    sources = owner.state()["decision_sources"]
    assert sources["gas_comparison"]["report_sha256"] == digest(owner.output / "gas-comparison/report.json")
    assert sources["planet_class"]["confirmed_sha256"] == digest(rig.pin)
    owner.provide_autonomous_gases(gases=["CO2"], rationale=RATIONALE)
    assert rig.state.calls == calls
    path = owner.output / "gas-decision.json"
    original = path.read_bytes()
    assert "supplied_greenhouse_increment" not in json.loads(original)
    assert json.loads(original)["greenhouse_mode"] == "derive_after_selection"
    monkeypatch.setattr(decisions, "decide_greenhouse", reference(rig))
    owner.advance()
    assert owner.phase == "temperature_initializing"
    assert path.read_bytes() == original
    assert owner.gas_decision["supplied_greenhouse_increment"] == 30
    derived = owner.output / "greenhouse-derived.json"
    assert derived.is_file() and owner.book.hashes[str(derived.relative_to(rig.root))] == digest(derived)
    assert "temperature_init" not in rig.state.calls
    owner.advance()
    assert owner.component.increment == 30
    assert path.read_bytes() == original
    owner.abort()


@pytest.mark.parametrize("failure", ["unresolved", "callback_abort", "tampered_original", "changed_source"])
def test_autonomous_readback_failure_never_initializes_temperature_or_retries(rig, monkeypatch, failure):
    owner = rig.make(candidates=list(GASES))
    reach(owner, "awaiting_gases")
    owner.provide_autonomous_gases(gases=["CO2"], rationale=RATIONALE)
    if failure == "tampered_original":
        (owner.output / "gas-decision.json").write_text("{}")
    hook = None
    if failure == "callback_abort":
        hook = owner.abort
    elif failure == "changed_source":
        hook = lambda: (owner.output / "gas-selection/report.json").write_text("{}")
    monkeypatch.setattr(
        decisions,
        "decide_greenhouse",
        reference(rig, status="abstained" if failure == "unresolved" else "decided", hook=hook),
    )
    owner.advance()
    assert owner.finished and not owner.state()["task_completed"]
    assert "temperature_init" not in rig.state.calls
    calls = list(rig.state.calls)
    owner.advance()
    assert rig.state.calls == calls
    assert not (owner.output / "greenhouse-derived.json").exists()


def test_legacy_supplied_increment_remains_strict_and_never_uses_new_decider(rig, monkeypatch):
    owner = rig.make()
    reach(owner, "awaiting_gases")
    with pytest.raises(BrowserSafetyStop, match="invalid_supplied_gases"):
        owner.provide_gases(gases=["CO2"], rationale=RATIONALE, supplied_greenhouse_increment=None)
    monkeypatch.setattr(decisions, "decide_greenhouse", lambda *a, **k: pytest.fail("Legacy path changed"))
    owner.provide_gases(gases=["CO2"], rationale=RATIONALE, supplied_greenhouse_increment=30)
    owner.advance()
    assert owner.phase == "temperature_initializing"
    assert not (owner.output / "greenhouse-derived.json").exists()
    owner.abort()


def test_autonomous_path_refuses_incomplete_candidate_set_before_selection(rig):
    owner = rig.make()
    reach(owner, "awaiting_gases")
    calls = list(rig.state.calls)
    with pytest.raises(BrowserSafetyStop, match="invalid_autonomous_gases"):
        owner.provide_autonomous_gases(gases=["CO2"], rationale=RATIONALE)
    assert calls == rig.state.calls and not (owner.output / "gas-decision.json").exists()
    owner.abort()


def test_autonomous_decision_cannot_be_repeated(rig):
    owner = rig.make(candidates=list(GASES))
    reach(owner, "awaiting_gases")
    owner.provide_autonomous_gases(gases=["CO2"], rationale=RATIONALE)
    original = (owner.output / "gas-decision.json").read_bytes()
    with pytest.raises(BrowserSafetyStop, match="gas_handoff_not_ready"):
        owner.provide_autonomous_gases(gases=["H2O"], rationale=RATIONALE)
    assert (owner.output / "gas-decision.json").read_bytes() == original
    owner.abort()
