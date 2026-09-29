"""Public page parsing and injected panel scheduling; no native browser proof."""

import re
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml
from test_browser_assessment_panel_steps import rig as base_rig  # noqa: F401
from test_browser_project_paginated_inventory_steps import list_footer
from test_project_paginated_inventory import capture

import habfly.browser_assessment_panel_steps as module
import habfly.browser_project_paginated_inventory_steps as pages
from habfly.browser import BrowserSafetyStop
from habfly.project_inventory_source import InventoryEvidence

NAMES = [f"Star{i:02d}" for i in range(11)]


def panel_capture(mode="data_quality", start=1, *, funding=50000):
    report = capture(NAMES[start - 1 : start + 9], start, 11)
    frame = report["frames"][0]
    stats = {
        "data_quality": "STARS 0% PLANETS 0% HABITABILITY 0% OVERALL 0% COST $100",
        "scavenger_hunt": "MAIN SEQUENCE WHITE DWARF RED GIANT SUPERGIANT GAS GIANT ICE GIANT TERRESTRIAL HABITABLE WORLD COST $100",
    }[mode]
    labels = ["Assess", "AUTOMATION", "SCAVENGER HUNT", "DATA QUALITY", "EDIT DATA", "ASSESSMENT"]
    atoms = yaml.safe_load(frame["accessibility"])
    old = atoms[0]["text"]
    new = old.replace("$50000", f"${funding}").replace(" Observations", f" {stats} Observations")
    atoms[0]["text"] = new
    atoms[1:1] = [f'button "{label}"' for label in labels]
    frame["text"] = frame["text"].replace(old, new)
    frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)
    frame["controls"].extend(
        {
            "id": f"simulation-0:c{i + 2}",
            "role": "button",
            "accessibility": f'- button "{label}"',
            "enabled": True,
            "actions": [],
        }
        for i, label in enumerate(labels)
    )
    return report


@pytest.mark.parametrize("mode", ["data_quality", "scavenger_hunt"])
@pytest.mark.parametrize("start", [1, 11])
def test_explicit_page_parser_maps_actual_range_without_loosening_legacy(mode, start):
    report = panel_capture(mode, start)
    baseline, panel, counts, rows, _ = module._public_panel(report, inventory_kind="live_paginated")
    assert panel.mode == mode and baseline["collected"] == 11
    assert counts == {"start": start, "end": 10 if start == 1 else 11, "total": 11}
    assert len(rows) == (10 if start == 1 else 1)
    with pytest.raises(BrowserSafetyStop, match="incomplete_visible_list"):
        module._public_panel(report)


@pytest.fixture
def rig(base_rig, monkeypatch):  # noqa: F811
    """Only source eligibility and native controls are explicit injected seams."""
    rig = base_rig
    rig.flags.start = 1
    anchor = panel_capture()
    _, first = pages.parse_inventory_page(anchor)
    _, last = pages.parse_inventory_page(panel_capture(start=11))
    source = InventoryEvidence(
        receipt={"total_collected": 11},
        rows=first + last,
        anchor=anchor,
        anchor_dir=rig.source / "after",
        whole_collection_sha256="a" * 64,
        kind="live_paginated",
        anchor_evidence={"capture_dir": "injected/anchor"},
    )
    monkeypatch.setattr(module, "load_inventory_source", lambda *a, **k: source)

    def current():
        return panel_capture(rig.flags.mode, rig.flags.start, funding=rig.flags.funding)

    monkeypatch.setattr(module, "inspect_page", lambda *a: current())
    rig.frame.locator = lambda _: SimpleNamespace(
        evaluate=lambda _code: [
            {
                "text": current()["frames"][0]["text"],
                "boxes": [{"x": 1, "y": 1, "width": 1, "height": 1}],
            }
        ]
    )
    rig.visible_pages = []
    monkeypatch.setattr(pages, "visible_inventory_page", lambda _f, c, r: rig.visible_pages.append((c, r)))
    rig.inventory_evidence = source
    return rig


def test_source_keeps_all_rows_but_panel_compares_only_actual_first_anchor(rig):
    owner = rig.create(kind="scavenger_hunt")
    assert len(owner.inventory_source.rows) == 11 and len(owner.rows) == 10
    assert owner.source == rig.inventory_evidence.anchor
    assert not rig.flags.clicks
    owner.advance()
    owner.advance()
    assert owner.finished and owner.report["panel_verified"]
    assert rig.flags.clicks == ["SCAVENGER HUNT"]
    assert owner.report["inventory_kind"] == "live_paginated"
    assert owner.report["whole_collection_sha256"] == "a" * 64
    assert all(counts["start"] == 1 and len(rows) == 10 for counts, rows in rig.visible_pages)
    assert all(owner.report[key] == 0 for key in module.ZERO)


def test_page_change_is_not_treated_as_a_new_anchor(rig):
    owner = rig.create()
    rig.flags.start = 11
    owner.advance()
    assert owner.finished and owner.failure == "assessment_panel_collection_changed"
    assert not rig.flags.clicks


def test_second_page_cannot_be_declared_as_the_assessment_anchor(rig, monkeypatch):
    source = deepcopy(rig.inventory_evidence)
    replacement = InventoryEvidence(
        source.receipt,
        source.rows,
        panel_capture(start=11),
        source.anchor_dir,
        source.whole_collection_sha256,
        source.kind,
        source.anchor_evidence,
    )
    monkeypatch.setattr(module, "load_inventory_source", lambda *a, **k: replacement)
    with pytest.raises(BrowserSafetyStop, match="invalid_paginated_anchor"):
        rig.create()
    assert not rig.flags.clicks


def test_unexposed_current_page_prevents_tab_dispatch(rig, monkeypatch):
    def covered(*_):
        raise BrowserSafetyStop("live_inventory_outer_frame_occluded")

    monkeypatch.setattr(pages, "visible_inventory_page", covered)
    owner = rig.create()
    owner.advance()
    assert owner.finished and "occluded" in owner.failure
    assert not rig.flags.clicks


@pytest.mark.parametrize("effect", ["fades", "persists", "cancel", "changed_rows"])
def test_panel_settles_known_footer_without_rewriting_evidence_or_extending_deadline(
    rig, monkeypatch, effect
):
    owner = rig.create(max_seconds=1)
    settled = list_footer(panel_capture(), notice=False)
    notice = deepcopy(settled)
    frame = notice["frames"][0]
    frame["text"] = frame["text"].replace("\n \n \nVIEWING", "\nData saved\n \n \nVIEWING")
    frame["accessibility"] = re.sub(
        r"(\n- button(?: \[disabled\])?\n- button(?: \[disabled\])?\n- text: viewing)",
        r"\n- text: Data saved\1",
        frame["accessibility"],
        count=1,
    )
    assert pages.paired_footer_autosave(notice)
    raw_notice = deepcopy(notice)
    if effect == "changed_rows":
        for key in ("text", "accessibility"):
            settled["frames"][0][key] = settled["frames"][0][key].replace("0.045", "0.046")
    waits = []

    def wait(ms):
        waits.append(ms)
        rig.flags.now += ms / 1000
        if effect == "cancel":
            rig.flags.cancel = True

    rig.page.wait_for_timeout = wait
    monkeypatch.setattr(
        module, "inspect_page", lambda *args: notice if not waits or effect == "persists" else settled
    )
    if effect == "fades":
        report, _ = owner._read()
        assert report == settled and waits == [100]
    else:
        with pytest.raises(BrowserSafetyStop):
            owner._read()
        assert 0 < sum(waits) <= 1001
    assert notice == raw_notice and not rig.flags.clicks
    owner.abort()
