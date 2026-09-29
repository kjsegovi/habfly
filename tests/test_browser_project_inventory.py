"""Visible inventory fixtures; all requests intercepted, no live browser."""
# ruff: noqa: F811

import hashlib
import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401

import habfly.browser_project_inventory as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_project_inventory import verify_project_inventory
from habfly.browser_project_navigation import LIST_LABELS
from habfly.browser_stellar import SIMULATION_URL


def row(name, values="0.045 370 5.68E-10 - - - - - - - -"):
    return (
        '<div class="star-row" style="display:flex;height:33px;width:900px">'
        '<svg class="eye" role="img" style="display:block;width:22px;height:25px;flex:none"><rect width="22" height="25"/></svg>'
        f'<div style="display:flex"><div><div class="name" style="text-transform:uppercase">{name}</div></div>'
        f'<div> {values}</div></div><span role="img" style="display:block;width:22px;height:25px">Delete</span></div>'
    )


@pytest.fixture
def inventory_page(page):
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("iframe").first.evaluate("e=>e.style='width:1100px;height:700px;border:0'")
    frame = page.frame(url=SIMULATION_URL)
    frame.set_content(
        '<div id="headers">Funding $50000 Data Quality 0% Scavenger Hunt 0/8 Observations Analyzed Data Star '
        + " ".join(LIST_LABELS["stellar"])
        + '</div><div id="rows">'
        + row("Previous", "1 2 3 -99 2 UV 0 White Dwarf - - -")
        + row("Althinagon")
        + "</div>"
        '<div id="footer">viewing <span id="viewing">1-2 of 2</span> total collected <span id="total">2</span></div>'
        "<button>Save</button>"
    )
    frame.evaluate("""()=>{
      window.fixtureClicks=0;window.fixtureChanges=0;
      document.addEventListener('click',()=>window.fixtureClicks++);
      document.addEventListener('input',()=>window.fixtureChanges++);
    }""")
    return page, frame


def verify(fixture, tmp_path, **kwargs):
    return verify_project_inventory(fixture[0], config(), tmp_path / "inventory", **kwargs)


def test_complete_inventory_proves_collection_not_analysis_or_science(inventory_page, tmp_path):
    page, frame = inventory_page
    settings = config()
    settings.frames[0].required_text = ["obsolete detail-only text"]
    before_settings = settings.model_dump()
    receipt = verify_project_inventory(
        page, settings, tmp_path / "inventory", expected_stars=["ALTHINAGON", "Previous"]
    )
    assert receipt["collection_count_verified"] and receipt["complete_visible_list_verified"]
    assert receipt["total_collected"] == receipt["visible_row_count"] == 2
    assert [item["name"] for item in receipt["rows"]] == ["Previous", "Althinagon"]
    assert receipt["expected_stars_verified"] and settings.model_dump() == before_settings
    for path, expected in receipt["source_sha256"].items():
        assert hashlib.sha256((tmp_path / "inventory" / path).read_bytes()).hexdigest() == expected
    for item in receipt["rows"]:
        assert item["source_sha256"] == receipt["source_sha256"]["after/observation.json"]
    for key in (
        "browser_actions",
        "navigation_clicks",
        "scrolls",
        "answer_writes",
        "save_clicks",
        "deletion_clicks",
        "assessment_clicks",
        "submission_clicks",
    ):
        assert receipt[key] == 0
    for key in (
        "hidden_catalog_read",
        "learned_perception",
        "scientific_verified",
        "task_completed",
        "project_completed",
        "cross_session_persistence_verified",
        "config_mutated",
    ):
        assert receipt[key] is False
    assert frame.evaluate("window.fixtureClicks+window.fixtureChanges") == 0
    assert not page.get_by_role("checkbox").is_checked()


@pytest.mark.parametrize(
    "kind",
    [
        "missing_row",
        "duplicate_row",
        "duplicate_name",
        "count_mismatch",
        "page_partial",
        "name_covered",
        "name_partly_covered",
        "name_clipped",
        "row_offscreen",
        "name_hidden",
        "name_transparent",
        "eye_covered",
        "footer_covered",
        "outer_overlay",
        "footer_duplicate",
        "footer_missing",
        "wrong_headers",
        "wrong_eye",
        "row_structure",
        "modal",
        "auth",
        "unexpected_frame",
    ],
)
def test_incomplete_ambiguous_or_occluded_inventory_stops_without_actions(inventory_page, tmp_path, kind):
    page, frame = inventory_page
    if kind == "missing_row":
        frame.locator(".star-row").last.evaluate("e=>e.remove()")
    elif kind == "duplicate_row":
        frame.locator(".name").last.evaluate("e=>e.textContent='Previous'")
    elif kind == "duplicate_name":
        frame.locator("body").evaluate("e=>e.insertAdjacentHTML('beforeend','<div>Previous</div>')")
    elif kind == "count_mismatch":
        frame.locator("#total").evaluate("e=>e.textContent='3'")
    elif kind == "page_partial":
        frame.locator("#viewing").evaluate("e=>e.textContent='2-3 of 3'")
        frame.locator("#total").evaluate("e=>e.textContent='3'")
    elif kind in {"name_covered", "name_partly_covered", "eye_covered", "footer_covered"}:
        selector = ".eye" if kind == "eye_covered" else "#footer" if kind == "footer_covered" else ".name"
        frame.locator(selector).first.evaluate(
            """(e,partial)=>{const r=e.getBoundingClientRect(),d=document.createElement('div');
          d.style=`position:fixed;left:${r.x}px;top:${r.y}px;width:${partial?8:r.width}px;height:${r.height}px;background:red;z-index:10`;
          document.body.append(d)}""",
            kind == "name_partly_covered",
        )
    elif kind == "name_clipped":
        frame.locator(".name").first.evaluate("e=>e.parentElement.style='width:5px;overflow:hidden'")
    elif kind == "row_offscreen":
        frame.locator(".star-row").last.evaluate("e=>{e.style.position='absolute';e.style.top='1200px'}")
    elif kind == "name_hidden":
        frame.locator(".name").last.evaluate("e=>e.hidden=true")
    elif kind == "name_transparent":
        frame.locator(".name").last.evaluate("e=>e.style.opacity=0")
    elif kind == "outer_overlay":
        page.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<div style=\"position:fixed;inset:0;z-index:20\"></div>')"
        )
    elif kind == "footer_duplicate":
        frame.locator("#footer").evaluate("e=>e.after(e.cloneNode(true))")
    elif kind == "footer_missing":
        frame.locator("#footer").evaluate("e=>e.remove()")
    elif kind == "wrong_headers":
        frame.locator("#headers").evaluate("e=>e.textContent=e.textContent.replace('Parallax','Other')")
    elif kind == "wrong_eye":
        frame.locator(".eye").last.evaluate("e=>e.style.width='21px'")
    elif kind == "row_structure":
        frame.locator(".name").last.evaluate(
            "e=>{const w=document.createElement('div');e.before(w);w.append(e)}"
        )
    elif kind == "unexpected_frame":
        page.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<iframe src=\"https://unallowed.invalid/\"></iframe>')"
        )
        page.wait_for_timeout(20)
    else:
        frame.locator("body").evaluate(
            "(e,h)=>e.insertAdjacentHTML('beforeend',h)",
            '<div role="dialog">Unexpected modal</div>' if kind == "modal" else '<input type="password">',
        )
    with pytest.raises(BrowserSafetyStop):
        verify(inventory_page, tmp_path)
    assert not (tmp_path / "inventory/confirmed.json").exists()
    stopped = json.loads((tmp_path / "inventory/stopped.json").read_bytes())
    assert not stopped["collection_count_verified"] and not stopped["task_completed"]
    assert frame.evaluate("window.fixtureClicks+window.fixtureChanges") == 0


@pytest.mark.parametrize(
    "expected",
    [
        ["Missing"],
        ["Previous"],
        ["Previous", "Althinagon", "Extra"],
        ["Previous", "previous"],
        "Previous",
        [""],
        [None],
    ],
)
def test_expected_names_are_exact_not_navigation_instructions(inventory_page, tmp_path, expected):
    with pytest.raises(BrowserSafetyStop):
        verify(inventory_page, tmp_path, expected_stars=expected)
    assert inventory_page[1].evaluate("window.fixtureClicks") == 0


@pytest.mark.parametrize("kind", ["new_name", "reorder", "replace", "outer", "popup", "dialog"])
def test_changed_readback_never_certifies_old_inventory(inventory_page, tmp_path, monkeypatch, kind):
    page, frame = inventory_page
    original = module.save_probe

    def changing_save(report, output):
        result = original(report, output)
        if output.name == "before":
            if kind == "new_name":
                frame.locator(".name").last.evaluate("e=>e.textContent='Different'")
            elif kind == "reorder":
                frame.locator(".star-row").first.evaluate("e=>e.parentElement.append(e)")
            elif kind == "replace":
                frame.locator(".name").last.evaluate("e=>e.replaceWith(e.cloneNode(true))")
            elif kind == "outer":
                page.get_by_role("checkbox").check()
            elif kind == "popup":
                popup = page.context.new_page()
                popup.close()
            else:
                frame.evaluate("confirm('fixture')")
        return result

    monkeypatch.setattr(module, "save_probe", changing_save)
    with pytest.raises(BrowserSafetyStop):
        verify(inventory_page, tmp_path)
    assert not (tmp_path / "inventory/confirmed.json").exists()


def test_zero_inventory_requires_visible_zero_range_not_missing_rows(inventory_page, tmp_path):
    _, frame = inventory_page
    frame.locator("#rows").evaluate("e=>e.replaceChildren()")
    frame.locator("#viewing").evaluate("e=>e.textContent='0-0 of 0'")
    frame.locator("#total").evaluate("e=>e.textContent='0'")
    receipt = verify(inventory_page, tmp_path, expected_stars=[])
    assert receipt["total_collected"] == 0 and receipt["rows"] == [] and receipt["collection_count_verified"]


def test_capture_parser_does_not_count_detail_or_planet_lists(inventory_page, tmp_path):
    _, frame = inventory_page
    frame.locator("#headers").evaluate(
        "(e,h)=>e.textContent=h", "Funding Observations Analyzed Data Star " + " ".join(LIST_LABELS["planet"])
    )
    with pytest.raises(BrowserSafetyStop, match="stellar_list_required"):
        verify(inventory_page, tmp_path)
