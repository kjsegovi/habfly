"""Fully intercepted native color fixtures. No live HabWorlds writes."""

from pathlib import Path

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_color import BrowserColorEnv, ColorJournal, ColorSession, open_gated_color_session
from habfly.browser_numeric import screen_identity
from habfly.browser_probe import inspect_page
from habfly.browser_stellar import SIMULATION_URL, StellarMappingError, map_stellar_capture
from habfly.color_reference import ColorReferenceError, load_color_reference
from habfly.contracts import Action


@pytest.fixture
def color_session(page, tmp_path):  # noqa: F811
    reference = load_color_reference()
    journal = ColorJournal(
        tmp_path / "color",
        {"color_gate_passed": True, "color_reference_hash": reference.checksum, "fixture_only": True},
    )
    session = ColorSession(page, config(), journal, reference)
    session.start()
    yield session
    session.close()


def test_color_native_transport_leaves_numeric_and_protected_controls_untouched(color_session):
    session = color_session
    for frame in session.page.frames:
        for button in frame.get_by_role("button").all():
            button.evaluate("b => b.onclick=()=>document.body.replaceChildren()")
    assert session.select_color("Red", confirm=lambda _: True)
    # 370 nm is UV, but the transport MUST NOT correct the wrong Red selection.
    assert session.journal.receipt["selected_color"] == "Red"
    assert session.journal.receipt["correctness_verified"] is False
    assert all(
        not item["current_value"]
        for item in session.mapping["observation"]["values"]["browser_field_map"].values()
    )
    assert not session.page.get_by_role("checkbox").is_checked()
    assert session.attempts == 1
    with pytest.raises(BrowserSafetyStop, match="color_write_limit"):
        session.select_color("UV", confirm=lambda _: True)
    with pytest.raises(BrowserSafetyStop, match="numeric_actions_disabled"):
        session.copy("anything", "temperature")
    # Default mapper remains strict: the numeric workflow is not broadened.
    report = inspect_page(session.page, config())
    with pytest.raises(StellarMappingError, match="unsupported_color"):
        map_stellar_capture(report, capture_sha256=screen_identity(report))


def test_browser_color_uses_same_visible_interface_without_grading_answers(color_session):
    env = BrowserColorEnv(color_session)
    for key, value in (("source", "browser_wavelength"), ("color", "Red"), ("check", None)):
        obs = env.observe()
        assert "expected_color" not in obs.model_dump_json()
        target = next(c for c in obs.controls if c.id.endswith(":" + key))
        result = env.step(
            Action(
                kind="CLICK" if key == "check" else "SELECT",
                target=target.id,
                value=value,
                observation_revision=obs.revision,
            ),
            confirm=lambda _: True,
        )
    assert result.terminated and result.observation.progress["color_transport_verified"]
    assert not result.observation.progress["task_completed"] and result.reward == 0
    assert env.close()["color_transport_verified"]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("document.getElementById('temperature').value='12'", "stale_numeric_observation"),
        (
            "document.querySelector('select').outerHTML=document.querySelector('select').outerHTML",
            "color_control_replaced",
        ),
        ("document.querySelector('select').disabled=true", "stale_numeric_observation"),
    ],
)
def test_changed_page_stops_before_any_color_write(color_session, mutation, reason):
    session = color_session
    session.frame.evaluate(mutation)
    with pytest.raises((BrowserSafetyStop, StellarMappingError), match=reason):
        session.select_color("UV", confirm=lambda _: True)
    assert session.attempts == 0


def test_revalidate_after_confirmation_and_no_repair(color_session):
    session = color_session

    def confirm(_):
        session.frame.locator("#distance").fill("123")
        return True

    with pytest.raises(BrowserSafetyStop, match="stale_numeric_observation"):
        session.select_color("UV", confirm=confirm)
    assert session.attempts == 0


def test_unrelated_change_after_color_stops_without_retry(color_session):
    session = color_session
    session.frame.get_by_role("combobox").evaluate(
        "e=>e.onchange=()=>document.getElementById('distance').value='9'"
    )
    with pytest.raises(BrowserSafetyStop, match="unrelated_state_changed_by_color"):
        session.select_color("UV", confirm=lambda _: True)
    assert session.attempts == 1 and session.stopped and not session.journal.receipt
    assert session.frame.get_by_role("combobox").input_value() == "UV"


def test_missing_gate_cannot_write(color_session):
    color_session.journal.provenance["color_gate_passed"] = False
    with pytest.raises(BrowserSafetyStop, match="color_learning_gate_required"):
        color_session.select_color("UV", confirm=lambda _: True)
    assert color_session.attempts == 0


@pytest.mark.parametrize(
    "value,reason",
    [
        ("450", "ambiguous_color_boundary"),
        ("494.5", "uncovered_color_gap"),
        ("2000", "color_outside_training_domain"),
    ],
)
def test_ambiguous_or_out_of_training_domain_never_becomes_a_color(page, tmp_path, value, reason):  # noqa: F811
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.locator("body").evaluate("(e,v)=>e.innerHTML=e.innerHTML.replace('370 Flux',v+' Flux')", value)
    reference = load_color_reference()
    journal = ColorJournal(
        tmp_path / "boundary", {"color_gate_passed": True, "color_reference_hash": reference.checksum}
    )
    session = ColorSession(page, config(), journal, reference)
    try:
        with pytest.raises((ColorReferenceError, BrowserSafetyStop), match=reason):
            session.start()
        assert session.attempts == 0
    finally:
        session.close()


def test_failed_learning_gate_prevents_browser_session(monkeypatch, tmp_path):
    from habfly.training import color

    def failed(*args, **kwargs):
        raise ValueError("Color browser gate not passed")

    monkeypatch.setattr(color, "require_browser_color_gate", failed)
    # An opaque object cannot be used as a page; rejection must precede page access.
    with pytest.raises(ValueError, match="gate"):
        open_gated_color_session(object(), config(), tmp_path / "no-output", Path("unpromoted"))
    assert not (tmp_path / "no-output").exists()


def test_existing_color_is_not_overwritten(page, tmp_path):  # noqa: F811
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.get_by_role("combobox").select_option(label="Red")
    reference = load_color_reference()
    journal = ColorJournal(
        tmp_path / "existing", {"color_gate_passed": True, "color_reference_hash": reference.checksum}
    )
    session = ColorSession(page, config(), journal, reference)
    try:
        with pytest.raises(BrowserSafetyStop, match="preexisting_color_selection"):
            session.start()
        assert session.attempts == 0 and frame.get_by_role("combobox").input_value() == "Red"
    finally:
        session.close()
