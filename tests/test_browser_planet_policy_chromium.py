"""Native fixture transport: scripted reference only, no learned acceptance claim."""
# ruff: noqa: F811

import os
from pathlib import Path

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_numeric import planet_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_numeric import PlanetNumericSession
from habfly.browser_planet_policy import PlanetBrowserToolEnv
from habfly.environments.planet_calculations import FIELDS, planet_expert


def test_native_four_field_bridge_preserves_raw_inputs_and_external_controls(planet_page, tmp_path):
    page, frame = planet_page
    raw = {"period_days": "108", "line_shift": "0.000000629", "brightness_drop": "0.008"}
    for field, value in raw.items():
        frame.locator("#" + field).fill(value)
    session = PlanetNumericSession(page, config(), tmp_path / "copies")
    try:
        env = PlanetBrowserToolEnv(session, supplied_star_class="main_sequence")
        for _ in range(62):
            result = env.step(planet_expert(env.observe(), env.pack))
        assert result.terminated and result.failure_reason is None
        assert env.transport_verified and set(session.verified) == set(FIELDS)
        assert not result.observation.progress["task_completed"]
        for field in FIELDS:
            assert float(frame.locator("#" + field).input_value()) == env.answers[field]
        for field, value in raw.items():
            assert frame.locator("#" + field).input_value() == value
        assert not page.get_by_role("checkbox").is_checked()
    finally:
        session.close()


def test_promoted_checkpoint_through_native_fixture(planet_page, tmp_path):
    pilot = os.environ.get("HABFLY_PLANET_PILOT")
    final = os.environ.get("HABFLY_PLANET_FINAL")
    if not pilot or not final:
        pytest.skip("Opt-in real-graph checkpoint fixture requires pilot and final evaluation paths")
    from habfly.browser_planet_policy import run_planet_derived
    from habfly.data import load_graph
    from habfly.training.planet_sequence import read
    from habfly.training.stellar import write_json

    page, frame = planet_page
    for name, value in {
        "period_days": "108",
        "line_shift": "0.000000629",
        "brightness_drop": "0.008",
    }.items():
        frame.locator("#" + name).fill(value)
    artifact = Path(os.environ.get("HABFLY_PLANET_FIXTURE_OUTPUT", str(tmp_path / "learned")))
    report = run_planet_derived(
        page,
        config(),
        artifact,
        pilot=Path(pilot),
        final_evaluation=Path(final),
        graph=load_graph("data/processed/graphs-v2/graph-2000"),
        supplied_star_class="main_sequence",
    )
    write_json(
        artifact / "fixture-scope.json",
        {"network": "all_requests_intercepted", "live_course": False, "learned_policy": True},
    )
    assert report["planet_transport_verified"], report["outcome"]
    assert len(report["verified_fields"]) == 4 and report["steps"] == 62
    assert report["optimizer_updates"] == 0 and report["checkpoint_unchanged"]
    assert not report["task_completed"] and not report["submitted"]
    assert read(artifact / "fixture-scope.json")["live_course"] is False


@pytest.mark.parametrize("mutation", ["measurement", "modal", "navigation", "replaced_target"])
def test_changed_browser_during_local_reasoning_blocks_before_first_write(planet_page, tmp_path, mutation):
    page, frame = planet_page
    for field, value in {
        "period_days": "108",
        "line_shift": "0.000000629",
        "brightness_drop": "0.008",
    }.items():
        frame.locator("#" + field).fill(value)
    session = PlanetNumericSession(page, config(), tmp_path / "copies")
    try:
        env = PlanetBrowserToolEnv(session, supplied_star_class="main_sequence")
        calls = []
        original = session.current

        def counted():
            calls.append(True)
            return original()

        session.current = counted
        # Reach the first native copy by taking only ordinary local-tool actions.
        for _ in range(62):
            action = planet_expert(env.observe(), env.pack)
            if action.target.endswith(":copy"):
                break
            env.step(action)
        else:
            pytest.fail("Expert never proposed the first copy")
        assert not calls and not session.attempted
        if mutation == "measurement":
            frame.locator("#period_days").fill("109")
        elif mutation == "modal":
            frame.locator("body").evaluate(
                "e=>{const d=document.createElement('dialog');d.textContent='Unexpected modal';e.append(d);d.showModal()}"
            )
        elif mutation == "navigation":
            page.route("https://outside.invalid/**", lambda route: route.fulfill(body="Escaped fixture"))
            page.goto("https://outside.invalid/escaped")
        else:
            frame.locator("#orbital_radius").evaluate("e=>e.replaceWith(e.cloneNode(true))")
        with pytest.raises(BrowserSafetyStop):
            env.step(action)
        assert calls and not session.attempted and not session.verified
        if mutation != "navigation":
            assert all(frame.locator("#" + k).input_value() == "" for k in FIELDS)
        assert not list((tmp_path / "copies").glob("copy-*-reserved.json"))
    finally:
        session.close()


def test_final_check_revalidates_browser_after_all_copies(planet_page, tmp_path):
    page, frame = planet_page
    for field, value in {
        "period_days": "108",
        "line_shift": "0.000000629",
        "brightness_drop": "0.008",
    }.items():
        frame.locator("#" + field).fill(value)
    session = PlanetNumericSession(page, config(), tmp_path / "copies")
    try:
        env = PlanetBrowserToolEnv(session, supplied_star_class="main_sequence")
        for _ in range(62):
            action = planet_expert(env.observe(), env.pack)
            if action.target.endswith(":check"):
                break
            env.step(action)
        assert set(session.verified) == set(FIELDS)
        frame.locator("#brightness_drop").fill("0.009")
        with pytest.raises(BrowserSafetyStop, match="stale_planet_observation"):
            env.step(action)
        assert not env.transport_verified
    finally:
        session.close()
