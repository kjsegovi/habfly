"""Intercepted native readback/workflow/import; no real preview or learned run.

Existing fixture-only learned event streams and tab-markup routing are reused.
No source validator is mocked. Save adapters are replaced with the actual new
read-only producers BEFORE the fixtures reach that stage, never relabelled.
"""
# ruff: noqa: F811

import json
import os
from types import SimpleNamespace

import pytest
import test_browser_positive_planet_workflow as positive_fixture
import test_browser_terrestrial_workflow as terrestrial_fixture
from test_browser_numeric import config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_raster_planet_evidence import raster_page  # noqa: F401
from test_browser_tooltip_reference_chromium import chromium  # noqa: F401
from test_project_evidence import inventory

from habfly.browser import BrowserSafetyStop
from habfly.browser_autosave import MODE
from habfly.browser_positive_finalize import _positive_autosave
from habfly.browser_terrestrial_steps import _terrestrial_autosave
from habfly.project_positive_evidence import import_verified_positive_planet
from habfly.project_progress import ProjectJournal
from habfly.project_terrestrial_evidence import import_verified_terrestrial

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must authorize an intercepted Chromium slot",
)


def read(path):
    return json.loads(path.read_bytes())


@pytest.fixture(params=[False, True], ids=["default", "pinned"])
def autosave_fixture(raster_page, tmp_path, monkeypatch, request):
    page, frame = raster_page
    boundary = config().model_copy(update={"pinned_control_capture": request.param})
    monkeypatch.setattr(positive_fixture, "config", lambda: boundary)
    monkeypatch.setattr(terrestrial_fixture, "config", lambda: boundary)

    def forbid_save():
        frame.get_by_role("button", name="Save", exact=True).evaluate("""e=>{
          window.autosaveFixtureClicks=0;
          e.onclick=()=>{window.autosaveFixtureClicks++;throw Error('Forbidden fixture Save')};
        }""")

    def planet_readback(page, config, output):
        forbid_save()
        return _positive_autosave(
            page,
            config,
            output,
            history=tmp_path,
            directories=positive_fixture.kwargs(tmp_path),
            supplied_inputs=False,
            check=lambda: None,
            timeout_seconds=20,
        )

    def habitat_readback(page, config, output, *, run_history, phase_dir, choice_dir):
        forbid_save()
        star = read(choice_dir / "confirmed.json")["star"]
        return _terrestrial_autosave(
            page,
            config,
            output,
            history=run_history,
            phase_dir=phase_dir,
            choice_dir=choice_dir,
            star=star,
            check=lambda: None,
            timeout_seconds=20,
        )

    monkeypatch.setattr(positive_fixture, "save_planet_work", planet_readback)
    monkeypatch.setattr(terrestrial_fixture, "save_habitability_work", habitat_readback)
    return page, frame, boundary


@pytest.mark.parametrize("branch", ["positive", "terrestrial"])
def test_native_autosave_readbacks_verify_and_import_without_save(
    autosave_fixture, tmp_path, monkeypatch, branch
):
    page, frame, boundary = autosave_fixture
    if branch == "positive":
        _, _, screens = positive_fixture.positive.__wrapped__(
            (page, frame), tmp_path, SimpleNamespace(param="gas_giant")
        )
        positive_fixture.seam(monkeypatch, frame, screens)
        receipt = positive_fixture.run(page, tmp_path)
        directory, save_dir = tmp_path / "workflow", tmp_path / "save"
        importer = import_verified_positive_planet
    else:
        _, _, screens = terrestrial_fixture.terrestrial.__wrapped__(
            (page, frame), tmp_path, SimpleNamespace(param=("gas", "not_habitable"))
        )
        terrestrial_fixture.seam(monkeypatch, frame, screens)
        receipt = terrestrial_fixture.run(page, tmp_path)
        directory, save_dir = tmp_path / "terrestrial-workflow", tmp_path / "final-save"
        importer = import_verified_terrestrial
    assert receipt["task_completed"] is True
    assert receipt["save_strategy"] == "autosave"
    assert receipt["source_save_click_delivered"] is False
    assert receipt["save_acknowledgement_verified"] is False
    assert receipt["persistence_verified"] is False
    assert read(save_dir / "confirmed.json")["mode"] == MODE
    assert frame.evaluate("window.autosaveFixtureClicks") == 0
    assert not any(
        (save_dir / name).exists()
        for name in ("reserved.json", "dispatch.json", "acknowledgement.json", "footer-probes")
    )
    inv = inventory(tmp_path, [receipt["star"]], "autosave-native")
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="intercepted-autosave").create()
    result = importer(journal, tmp_path, inv, directory)
    assert result["progress"]["verified"] == 1
    assert result["evidence_scopes"]["save_readback"]["persistence_verified"] is False
    assert importer(journal, tmp_path, inv, directory)["idempotent"] is True
    # A changed native scientific value still blocks a new readback, without
    # touching the prior confirmed receipt or clicking/waiting for Save.
    if branch == "positive":
        frame.get_by_role("textbox").nth(1).fill("9")
        with pytest.raises(BrowserSafetyStop):
            _positive_autosave(
                page,
                boundary,
                tmp_path / "changed-readback",
                history=tmp_path,
                directories=positive_fixture.kwargs(tmp_path),
                supplied_inputs=False,
                check=lambda: None,
                timeout_seconds=20,
            )
    else:
        frame.locator("#temperature").fill("999")
        with pytest.raises(BrowserSafetyStop):
            _terrestrial_autosave(
                page,
                boundary,
                tmp_path / "changed-readback",
                history=tmp_path,
                phase_dir=tmp_path / "habitability/phase",
                choice_dir=tmp_path / "habitability/choice",
                star=receipt["star"],
                check=lambda: None,
                timeout_seconds=20,
            )
    assert not (tmp_path / "changed-readback/confirmed.json").exists()
    assert frame.evaluate("window.autosaveFixtureClicks") == 0
