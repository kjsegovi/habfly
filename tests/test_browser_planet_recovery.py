"""Frozen-prefix recovery must not repeat either completed native write."""
# ruff: noqa: F811

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_numeric import planet_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_policy import run_planet_derived
from habfly.browser_planet_recovery import ReconciledPlanetSession, load_mass_orbit_recovery
from habfly.contracts import Action


def test_replay_cannot_skip_change_or_repeat_retained_copy():
    # Pure protocol check, not browser provenance or a learned result.
    session = object.__new__(ReconciledPlanetSession)
    session.native = SimpleNamespace(current=lambda: None, copy=lambda *a, **k: pytest.fail("write"))
    session.emit = lambda *args: None
    session.verified = {}
    session.replay_index = -1
    session.recovery = {
        "receipts": {"orbital_radius": {"value": "2", "unit": "au"}},
        "actions": [Action(kind="CLICK", target="0:copy")],
    }
    for name, text, unit in (
        ("planet_mass", "2", "MEarth"),
        ("orbital_radius", "3", "au"),
        ("orbital_radius", "2", "m"),
    ):
        with pytest.raises(BrowserSafetyStop):
            session.copy(name, text, unit, source="checkpoint")
    with pytest.raises(BrowserSafetyStop, match="prefix_diverged"):
        session.require_prefix(Action(kind="CLICK", target="0:execute"), 0)
    session.require_prefix(Action(kind="CLICK", target="0:copy"), 0)
    session.copy("orbital_radius", "2", "au", source="checkpoint")
    with pytest.raises(BrowserSafetyStop, match="copy_diverged"):
        session.copy("orbital_radius", "2", "au", source="checkpoint")


def test_native_recovery_reconstructs_same_checkpoint_prefix_without_rewrites(
    planet_page, tmp_path, monkeypatch
):
    import habfly.browser_planet_numeric as numeric
    from habfly.data import load_graph
    from habfly.training.planet_sequence import file_hash

    page, frame = planet_page
    for key, value in {"period_days": "108", "line_shift": "0.000000629", "brightness_drop": "0.008"}.items():
        frame.locator("#" + key).fill(value)
    frame.locator("#orbital_radius").evaluate(
        "e=>e.onblur=()=>document.querySelector('#derived').innerText='STAR MASS (Ms) 2.368 STAR RADIUS (Rs) 1.961 ORBIT (years) 0.2959'"
    )
    frame.locator("#planet_mass").evaluate(
        "e=>e.onblur=()=>document.querySelector('#derived').innerText='STAR MASS (Ms) 2.368 STAR RADIUS (Rs) 1.961 ORBIT (years) 0.2958'"
    )
    original = numeric.validate_orbit_readout_change

    def old_guard(before, after, destination):
        if destination == "planet_mass":
            raise BrowserSafetyStop("unexpected_planet_copy_side_effect")
        return original(before, after, destination)

    monkeypatch.setattr(numeric, "validate_orbit_readout_change", old_guard)
    args = {
        "pilot": Path("experiments/planet-pilot-001"),
        "final_evaluation": Path("experiments/planet-final-001"),
        "graph": load_graph("data/processed/graphs-v2/graph-2000"),
        "supplied_star_class": "main_sequence",
    }
    failed = tmp_path / "failed"
    result = run_planet_derived(page, config(), failed, **args)
    assert result["outcome"] == "unexpected_planet_copy_side_effect"
    assert result["write_attempts"] == ["orbital_radius", "planet_mass"]
    monkeypatch.setattr(numeric, "validate_orbit_readout_change", original)
    checksum = file_hash(args["pilot"] / "training/checkpoint.pt")
    before_hash = file_hash(failed / "report.json")
    recovery = load_mass_orbit_recovery(failed, checksum)
    assert len(recovery["actions"]) == result["steps"] + 1
    with pytest.raises(BrowserSafetyStop):
        load_mass_orbit_recovery(failed, "wrong")

    # Tampered traces fail before any browser action. Restore only the fixture
    # file so the successful continuation can exercise the same failed prefix.
    raw = (failed / "events.jsonl").read_bytes()
    (failed / "events.jsonl").write_bytes(raw + b"\n")
    with pytest.raises(BrowserSafetyStop):
        load_mass_orbit_recovery(failed, checksum)
    (failed / "events.jsonl").write_bytes(raw)
    # A changed live raw input prevents even the first replayed action.
    frame.locator("#period_days").fill("109")
    blocked = run_planet_derived(page, config(), tmp_path / "blocked", resume_from=failed, **args)
    assert blocked["outcome"] == "planet_recovery_live_state_changed" and not blocked["write_attempts"]
    frame.locator("#period_days").fill("108")

    # Any native rewrite of an already-filled answer trips a visible error.
    for name in ("orbital_radius", "planet_mass"):
        frame.locator("#" + name).evaluate(
            "e=>e.oninput=()=>document.body.append('Unexpected repeated write')"
        )
    continued = run_planet_derived(page, config(), tmp_path / "continued", resume_from=failed, **args)
    assert continued["planet_transport_verified"], continued["outcome"]
    assert continued["steps"] == 62 and continued["write_attempts"] == ["planet_density", "planet_radius"]
    assert len(continued["verified_fields"]) == 4 and not continued["task_completed"]
    assert continued["checkpoint_unchanged"] and continued["optimizer_updates"] == 0
    assert file_hash(failed / "report.json") == before_hash
    journal = [json.loads(row) for row in (tmp_path / "continued/events.jsonl").read_text().splitlines()]
    retained = [r for r in journal if r["payload"].get("replayed_native_receipt")]
    assert len(retained) == 2 and all(not r["payload"]["browser_action_executed"] for r in retained)
    repeated = run_planet_derived(page, config(), tmp_path / "repeat", resume_from=failed, **args)
    assert repeated["outcome"] == "planet_recovery_live_state_changed" and not repeated["write_attempts"]
