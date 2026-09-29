"""Offline guard ordering plus explicitly intercepted native opt-in fixtures.

The pure scheduling seam does not certify chart evidence. Native cases use the
unchanged real source/freshness/mapping primitives against fulfilled fixtures.
"""

# ruff: noqa: F811 - pytest fixture dependencies

import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_numeric import config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_raster_planet_evidence import raster_page, read, select, sources  # noqa: F401
from test_browser_tooltip_reference_chromium import chromium  # noqa: F401 - launch is idle-gated

import habfly.browser_raster_planet_evidence as module
from habfly.browser import BrowserSafetyStop


@pytest.fixture
def rig(tmp_path, monkeypatch):
    flags = SimpleNamespace(calls=[], now=0, deadline=600, changed=False, replaced=False, writes=0)
    mapping = {"star_name": "Fixture", "stellar_mass": 1, "stellar_radius": 0.01}
    evidence = {"star": "FIXTURE", "measurements": {}}

    class Handle:
        def evaluate(self, expression, other):
            flags.calls.append("identity")
            return self is other

        def select_option(self, **kwargs):
            assert kwargs == {"label": "Yes", "timeout": 3000}
            flags.calls.append("select")
            flags.writes += 1

    handle, replacement = Handle(), Handle()

    class Control:
        def count(self):
            return 1

        def is_enabled(self):
            return True

        def input_value(self):
            return "Yes"

        def element_handle(self, **kwargs):
            return replacement if flags.replaced else handle

    class Session:
        def __init__(self, page, config, output, *, max_seconds, _allow_unset_planet):
            assert _allow_unset_planet
            flags.calls.append("session")
            flags.deadline = flags.now + max_seconds
            self.frame = SimpleNamespace(get_by_role=lambda role: Control())

        def current(self):
            flags.calls.append("current")
            if flags.now >= flags.deadline:
                raise BrowserSafetyStop("planet_copy_time_limit")
            if flags.changed:
                raise BrowserSafetyStop("stale_planet_observation")
            return self.read()

        def read(self):
            return {"fixture": "public"}, deepcopy(mapping), {"selected": None}, {}

        def close(self):
            flags.calls.append("close")

    def fresh(page, config, owner, directory, source):
        flags.calls.append("fresh")
        directory.mkdir()
        return {"fixture": directory.name}

    def save(capture, directory):
        directory.mkdir()
        module.persist_json(directory / "observation.json", capture)

    monkeypatch.setattr(module, "PlanetNumericSession", Session)
    monkeypatch.setattr(module, "_evidence", lambda *args, **kwargs: deepcopy(evidence))
    monkeypatch.setattr(module, "_fresh", fresh)
    monkeypatch.setattr(module, "_reload", lambda *args: flags.calls.append("reload"))
    monkeypatch.setattr(module, "_blank_current", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_control", lambda session: replacement if flags.replaced else handle)
    monkeypatch.setattr(module, "save_probe", save)
    monkeypatch.setattr(module, "presence_projection", lambda report, mapping: report)
    monkeypatch.setattr(module, "preserved_paint_transition", lambda *args: True)
    monkeypatch.setattr(module, "rendered_control", lambda control: True)

    def run(name="presence", **kwargs):
        return module.select_raster_detected_planet(
            None,
            None,
            tmp_path / name,
            run_history=tmp_path,
            window_report="fixture",
            window_report_sha256="fixture",
            spectrum_path="fixture",
            spectrum_sha256="fixture",
            **kwargs,
        )

    return SimpleNamespace(run=run, root=tmp_path, flags=flags, mapping=mapping)


def stopped(rig):
    result = read(rig.root / "presence/stopped.json")
    assert result["reservation_created"] is True and result["write_may_have_occurred"] is False
    assert rig.flags.writes == 0 and rig.flags.calls[-1] == "close"
    assert not (rig.root / "presence/confirmed.json").exists()
    with pytest.raises(BrowserSafetyStop, match="presence_already_reserved"):
        rig.run("retry")
    assert rig.flags.writes == 0


@pytest.mark.parametrize("bad", [False, True, 0, 1, "guard", {}, []])
def test_guard_must_be_callable_before_output_or_browser(tmp_path, bad):
    with pytest.raises(BrowserSafetyStop, match="invalid_before_dispatch_guard"):
        module.select_raster_detected_planet(
            None,
            None,
            tmp_path / "presence",
            run_history=tmp_path,
            window_report=None,
            window_report_sha256=None,
            spectrum_path=None,
            spectrum_sha256=None,
            before_dispatch=bad,
        )
    assert list(tmp_path.iterdir()) == []


def test_default_none_retains_receipt_and_read_sequence(rig):
    receipt = rig.run(before_dispatch=None)
    assert rig.flags.calls == [
        "session",
        "current",
        "fresh",
        "current",
        "fresh",
        "reload",
        "current",
        "identity",
        "select",
        "identity",
        "reload",
        "close",
    ]
    assert not any("guard" in key or "callback" in key for key in receipt)
    assert receipt == read(rig.root / "presence/confirmed.json")
    assert rig.flags.writes == 1 and receipt["max_seconds"] == 600


def test_successful_guard_is_once_after_reservation_and_before_final_native_check(rig):
    def guard(mapping):
        assert (rig.root / "presence/reserved.json").is_file()
        assert (rig.root / "presence/preselect.json").is_file()
        assert mapping == rig.mapping
        rig.flags.calls.append("guard")
        mapping["stellar_mass"] = 999  # detached view cannot rewrite the session/evidence
        return {"value": "No"}  # return values never select or repair an answer

    receipt = rig.run(before_dispatch=guard)
    assert rig.mapping["stellar_mass"] == 1
    assert rig.flags.calls.count("guard") == 1
    index = rig.flags.calls.index("guard")
    assert rig.flags.calls[index - 2 : index + 4] == [
        "reload",
        "current",
        "guard",
        "current",
        "identity",
        "select",
    ]
    assert receipt["value"] == "Yes" and rig.flags.writes == 1
    assert receipt["max_seconds"] == rig.flags.deadline == 600


@pytest.mark.parametrize("failure", ["stale_mass", "stale_radius", "abort", "source", "unknown"])
def test_callback_failure_retains_reservation_before_attempt(rig, failure):
    def guard(mapping):
        if failure.startswith("stale_"):
            name = "stellar_" + failure.removeprefix("stale_")
            if mapping[name] != 999:
                raise BrowserSafetyStop("current_supplied_stellar_inputs_changed")
        elif failure == "unknown":
            raise ValueError("private input or secret driver detail")
        else:
            raise BrowserSafetyStop("explicit_" + failure)

    with pytest.raises(BrowserSafetyStop):
        rig.run(before_dispatch=guard)
    assert "private" not in (rig.root / "presence/stopped.json").read_text()
    stopped(rig)


@pytest.mark.parametrize("change", ["ui", "replacement", "deadline"])
def test_callback_cannot_mutate_ui_replace_native_target_or_extend_deadline(rig, change):
    def guard(mapping):
        if change == "ui":
            rig.flags.changed = True
        elif change == "replacement":
            rig.flags.replaced = True
        else:
            rig.flags.now = 600

    with pytest.raises(BrowserSafetyStop):
        rig.run(before_dispatch=guard)
    assert rig.flags.deadline == 600
    stopped(rig)


@pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1", reason="Native idle window required"
)
@pytest.mark.parametrize("change", ["none", "stale_mass", "abort", "ui", "replace", "overlay"])
def test_intercepted_native_predispatch_guard(raster_page, tmp_path, change):
    _, frame = raster_page
    source = sources(raster_page, tmp_path)
    calls = []

    def guard(mapping):
        calls.append(deepcopy(mapping))
        assert (tmp_path / "presence/reserved.json").exists()
        if change == "stale_mass":
            mass = mapping["observation"]["values"]["stellar_inputs"]["stellar_mass"]["value"]
            assert mass != 999
            raise BrowserSafetyStop("current_supplied_stellar_inputs_changed")
        if change == "abort":
            raise BrowserSafetyStop("parent_aborted")
        if change == "ui":
            frame.locator("#period_days").fill("123")
        elif change == "replace":
            frame.get_by_role("combobox").evaluate("e=>e.replaceWith(e.cloneNode(true))")
        elif change == "overlay":
            frame.get_by_role(
                "combobox"
            ).evaluate("""e=>{const r=e.getBoundingClientRect(),d=document.createElement('div');
              d.style=`position:fixed;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;background:black;z-index:999`;
              document.body.append(d)}""")

    if change == "none":
        result = select(raster_page, tmp_path, source, before_dispatch=guard)
        assert result["readback_verified"] and result["value"] == "Yes"
        assert frame.evaluate("window.fixtureSelections") == 1
    else:
        with pytest.raises(BrowserSafetyStop):
            select(raster_page, tmp_path, source, before_dispatch=guard)
        result = read(tmp_path / "presence/stopped.json")
        assert result["reservation_created"] is True and result["write_may_have_occurred"] is False
        assert frame.evaluate("window.fixtureSelections") == 0
        assert not (tmp_path / "presence/confirmed.json").exists()
    assert len(calls) == 1
    assert all(frame.locator("#" + name).input_value() == "" for name in module.DERIVED)
