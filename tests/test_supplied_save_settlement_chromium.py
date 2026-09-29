"""Opted-in supplied-input Save chains with intercepted native controls.

Reuses the explicitly synthetic checkpoint/private-gate and water-indicator
fixtures, not actual learned/course acceptance. All source readers, native
controls, settlement diagnostics, workflow verification and canonical imports
remain real. No validator is replaced beyond the declared original gate seam.
"""
# ruff: noqa: F811 - imported pytest fixtures

import json
import os

import pytest
import test_supplied_browser_workflows_chromium as positive_fixture
import test_supplied_terrestrial_native_chain as terrestrial_fixture
from test_browser_numeric import page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_raster_planet_evidence import raster_page  # noqa: F401
from test_browser_tooltip_reference_chromium import chromium  # noqa: F401 - launch idle gate

import habfly.browser_habitability_save as habitability_save
import habfly.browser_positive_finalize as positive

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must authorize an intercepted Chromium slot",
)


def pinned_configs(monkeypatch):
    original = positive_fixture.config

    def configured():
        return original().model_copy(update={"pinned_control_capture": True})

    monkeypatch.setattr(positive_fixture, "config", configured)
    monkeypatch.setattr(terrestrial_fixture, "config", configured)


def assert_settled_sources(root, save, workflow):
    receipt = json.loads((root / workflow / "confirmed.json").read_bytes())
    confirmed = json.loads((root / save / "confirmed.json").read_bytes())
    assert confirmed["settle_reserved_notice"] is True
    assert confirmed["reserved_phase_timeout_seconds"] == 20
    assert f"{save}/reserved-phase-confirmed.json" in receipt["source_sha256"]
    assert (root / save / "dispatch.json").is_file()
    assert (root / save / "acknowledgement.json").is_file()
    assert not (root / save / "dispatch-budget-rejected.json").exists()


def test_supplied_positive_settlement_pinned_capture_and_strict_import(raster_page, tmp_path, monkeypatch):
    pinned_configs(monkeypatch)
    original_owner, original_session = (
        positive_fixture.PositiveFinalizationSteps,
        positive.PlanetNumericSession,
    )
    sessions = []

    def owner(*args, **kwargs):
        assert args[1].pinned_control_capture is True
        return original_owner(*args, **kwargs, settle_reserved_notice=True)

    def session(*args, **kwargs):
        sessions.append((args[1].pinned_control_capture, kwargs.get("_pin_controls")))
        return original_session(*args, **kwargs)

    monkeypatch.setattr(positive_fixture, "PositiveFinalizationSteps", owner)
    monkeypatch.setattr(positive, "PlanetNumericSession", session)
    positive_fixture.test_white_dwarf_visible_inputs_native_save_and_new_mode_import(
        raster_page, tmp_path, monkeypatch
    )
    assert sessions == [(True, True)]
    assert_settled_sources(tmp_path, "finalize/save", "finalize/workflow")


def test_supplied_terrestrial_settlement_pinned_capture_and_strict_import(raster_page, tmp_path, monkeypatch):
    pinned_configs(monkeypatch)
    original_save = terrestrial_fixture.save_habitability_work
    original_session = habitability_save.HabitabilityMenuSession
    sessions = []

    def save(*args, **kwargs):
        assert args[1].pinned_control_capture is True
        return original_save(*args, **kwargs, settle_reserved_notice=True)

    def session(*args, **kwargs):
        sessions.append((args[1].pinned_control_capture, kwargs.get("_pin_controls")))
        return original_session(*args, **kwargs)

    monkeypatch.setattr(terrestrial_fixture, "save_habitability_work", save)
    monkeypatch.setattr(habitability_save, "HabitabilityMenuSession", session)
    terrestrial_fixture.test_white_dwarf_temperature_habitability_save_and_strict_import(
        raster_page, tmp_path, monkeypatch
    )
    assert sessions == [(True, True)]
    assert_settled_sources(tmp_path, "final-save", "terrestrial-workflow")
