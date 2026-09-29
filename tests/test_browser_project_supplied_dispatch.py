"""Offline dispatch contract; injected children do not prove course completion."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import habfly.browser_project_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.runtime import parse_run_options
from habfly.supplied_browser_modes import NON_MAIN_CLASSES


def test_supplied_three_star_profile_remains_bounded_and_separate():
    base = parse_run_options(
        json.loads(Path("configs/browser_project_autonomous_three_star.json").read_text())
    )
    enabled = parse_run_options(
        json.loads(Path("configs/browser_project_supplied_three_star.json").read_text())
    )
    assert enabled.stars == 3 and enabled.project_campaign and enabled.paused
    assert enabled.project_autonomous_decisions and enabled.project_supplied_stellar_inputs
    assert enabled.project_max_advances == 512 and enabled.project_max_seconds == 1800
    assert enabled.project_campaign_max_seconds == 5400
    assert not enabled.project_allow_scoring and not enabled.project_allow_submission
    assert enabled.artifact_dir == Path("experiments/browser-project-supplied-three-star")
    assert enabled.artifact_dir != base.artifact_dir
    assert enabled.browser_planet_supplied_evaluation == Path(
        "experiments/planet-supplied-input-transfer-001"
    )
    assert enabled.browser_habitability_supplied_evaluation == Path(
        "experiments/habitability-supplied-input-transfer-001"
    )
    assert not base.project_supplied_stellar_inputs
    for name in (
        "checkpoint",
        "graph",
        "dataset",
        "color_experiment",
        "browser_planet_pilot",
        "browser_planet_final_evaluation",
        "browser_habitability_pilot",
        "browser_habitability_final_evaluation",
    ):
        assert getattr(enabled, name) == getattr(base, name)


@pytest.mark.parametrize("selected", ["main_sequence", *NON_MAIN_CLASSES])
@pytest.mark.parametrize("save_strategy", ["explicit", "autosave"])
def test_temperature_gate_switch_is_per_actual_star_not_global(
    selected, save_strategy, tmp_path, monkeypatch
):
    owner = module.BrowserProjectSteps.__new__(module.BrowserProjectSteps)
    owner._save_strategy = save_strategy
    owner.model_options = {"planet_supplied_evaluation": tmp_path / "new-planet-gate"}
    owner._class_options = {"selected_class": selected}
    owner.terrestrial_options = {
        "pilot": tmp_path / "pilot",
        "final_evaluation": tmp_path / "legacy-temperature",
        "supplied_evaluation": tmp_path / "new-temperature",
        "graph": object(),
        "candidates": ("CO2",),
        "max_seconds": 1800,
    }
    owner.phase, owner.star, owner.status = "terrestrial_initializing", "Fixture", "paused"
    owner.history, owner.output, owner.page, owner.config = tmp_path, tmp_path / "owner", None, None
    owner._positive_source = {
        "directories": {
            key: tmp_path / key for key in ("numeric_dir", "color_dir", "class_dir", "raw_dir", "derived_dir")
        }
    }
    owner._planet_class_source = {"directory": tmp_path / "planet-class", "sha256": "a" * 64}
    owner.max_seconds, owner._started_at, owner._clock = 1800, 0, lambda: 1
    owner._continuation_check = lambda: None
    owner._owned = Path
    owner._relay = SimpleNamespace(bind=lambda *a, **k: lambda *_: None)
    owner._sync_terrestrial = lambda: None
    options = []
    monkeypatch.setattr(module, "TerrestrialSteps", lambda *a, **kwargs: options.append(kwargs) or object())
    owner._drive_terrestrial()
    assert ("save_strategy" in options[0]) is (save_strategy == "autosave")
    assert options[0].get("save_strategy", "explicit") == save_strategy
    assert owner.phase == "terrestrial_active" and len(options) == 1
    assert "supplied_evaluation" not in options[0]
    assert options[0]["final_evaluation"] == tmp_path / (
        "legacy-temperature" if selected == "main_sequence" else "new-temperature"
    )
    assert options[0].get("supplied_inputs", False) is (selected != "main_sequence")
    assert options[0]["max_seconds"] == 1799  # Existing remaining budget is not reset or increased.
    assert options[0]["settle_reserved_notice"] is True


@pytest.mark.parametrize("actual_class", NON_MAIN_CLASSES)
def test_non_main_temperature_gate_cannot_fall_back_to_legacy(actual_class, tmp_path):
    owner = module.BrowserProjectSteps.__new__(module.BrowserProjectSteps)
    owner.model_options = {"planet_supplied_evaluation": tmp_path}
    owner._class_options = {"selected_class": actual_class}
    owner.terrestrial_options = {"final_evaluation": tmp_path / "legacy"}
    owner.phase = "terrestrial_initializing"
    owner._positive_source = {"directories": {}}
    owner._continuation_check = lambda: None
    with pytest.raises(BrowserSafetyStop, match="supplied_temperature_gate_required"):
        owner._drive_terrestrial()


def test_old_and_new_workflow_phases_remain_distinct_identities():
    assert module.POSITIVE_SUPPLIED_MODE != "positive_planet_visible_workflow_readback"
    assert module.TERRESTRIAL_SUPPLIED_MODE != "terrestrial_visible_workflow_readback"
    assert module._WORKFLOW_PHASES[module.POSITIVE_SUPPLIED_MODE] == "verified_positive_planet"
    assert module._WORKFLOW_PHASES[module.TERRESTRIAL_SUPPLIED_MODE] == "verified_terrestrial"
