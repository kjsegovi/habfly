"""Intercepted project-navigation fixtures; never opens the live preview."""
# ruff: noqa: F811

import json

import pytest
from test_browser_full_stellar import full_html
from test_browser_habitability_numeric import habitat_page  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_numeric import planet_html

from habfly.browser import BrowserSafetyStop
from habfly.browser_project_navigation import LIST_LABELS, navigate_project


@pytest.fixture
def navigation_page(habitat_page):
    page, frame = habitat_page
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("iframe").first.evaluate(
        "e=>e.style='position:absolute;left:0;top:200px;width:1200px;height:850px'"
    )
    habitat = frame.locator("body").inner_html().replace("JYREMIS", "Althinagon").replace('<img alt="1">', "")
    details = {
        "stellar": full_html().replace('<img alt="1">', ""),
        "planet": planet_html().replace("JYREMIS", "Althinagon").replace('<img alt="1">', ""),
        "habitability": habitat,
    }
    tabs = (
        '<nav style="display:flex;gap:10px">'
        + "".join(
            f'<span class="numbered" role="img" data-destination="{name}" style="display:inline-flex;width:24px;height:24px;align-items:center;justify-content:center;background:lightblue">{number}</span>'
            for number, name in enumerate(("stellar", "planet", "habitability"), 1)
        )
        + "</nav>"
    )
    details = {name: html.replace("</div>", "</div>" + tabs, 1) for name, html in details.items()}
    lists = {
        name: tabs
        + "<div>FUNDING $50000 DATA QUALITY 0% SCAVENGER HUNT 0/8</div><div>Observations Analyzed Data Star "
        + " ".join(labels)
        + "</div><div>viewing 1-1 of 1 total collected 1</div><button>Save</button>"
        for name, labels in LIST_LABELS.items()
    }
    frame.locator("body").evaluate(
        """(e,data)=>{
      e.innerHTML=`<style>body{margin:0}#content{padding-top:50px}.header{position:absolute;top:0;width:40px;height:40px}</style>
        <svg role='img' class='header' id='sky' style='left:0'><rect width='40' height='40' fill='gray'/></svg>
        <svg role='img' class='header' id='list' style='left:50px'><rect width='40' height='40' fill='gray'/></svg><div id='content'></div>`;
      window.fixtureSurface='detail';window.fixtureSection='stellar';window.fixtureClicks=0;
      window.fixtureRender=(surface,section)=>{
        window.fixtureSurface=surface;window.fixtureSection=section;
        document.querySelector('#content').innerHTML=surface==='starfield'
          ? '<div>FUNDING $50000 DATA QUALITY 0% SCAVENGER HUNT 0/8</div><button>Save</button>'
          : data[surface][section];
        for(const tab of document.querySelectorAll('.numbered'))tab.onclick=()=>window.fixtureNavigate(tab.dataset.destination);
      };
      window.fixtureNavigate=destination=>{
        window.fixtureClicks++;
        if(destination==='list')window.fixtureRender('list',window.fixtureSection||'stellar');
        else if(destination==='starfield')window.fixtureRender('starfield',null);
        else window.fixtureRender(window.fixtureSurface,destination);
      };
      document.querySelector('#sky').onclick=()=>window.fixtureNavigate('starfield');
      document.querySelector('#list').onclick=()=>window.fixtureNavigate('list');
      window.fixtureRender('detail','stellar');
    }""",
        {"detail": details, "list": lists},
    )
    return page, frame


def test_planet_list_navigation_accepts_only_transient_final_save_footer(
    navigation_page, tmp_path, monkeypatch
):
    import habfly.browser_project_navigation as module

    page, frame = navigation_page
    frame.evaluate("window.fixtureRender('detail','planet')")
    frame.locator("#derived").evaluate(
        """e=>{
          e.insertAdjacentHTML('beforebegin','<button>+</button>');
          e.innerHTML='<div>STAR MASS (Ms)</div><div>2.368</div>'+
          '<div>STAR RADIUS (Rs)</div><div>1.961</div><div>ORBIT (years)</div><div>0.000</div>';
        }"""
    )
    original = module.inspect_page
    calls = 0

    def capture(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            frame.locator("#derived").evaluate("e=>e.insertAdjacentHTML('afterend','<div>Data saved</div>')")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "inspect_page", capture)
    output = tmp_path / "footer-navigation"
    receipt = navigate_project(page, config(), output, "list", expected_star="Althinagon")
    assert receipt["navigation_clicks"] == 1 and receipt["destination_verified"]
    assert frame.evaluate("window.fixtureClicks") == 1
    before = json.loads((output / "before/observation.json").read_bytes())
    current = json.loads((output / "pre-click/observation.json").read_bytes())
    assert "Data saved" not in before["frames"][0]["text"]
    assert "Data saved" in current["frames"][0]["text"]


@pytest.mark.parametrize(
    "destination", ["stellar", "stellar-tab", "planet", "habitability", "list", "starfield"]
)
def test_one_navigation_and_destination_proof_without_config_mutation(navigation_page, tmp_path, destination):
    page, frame = navigation_page
    settings = config()
    settings.frames[0].required_text = ["old source-only label"]
    original = settings.model_dump()
    receipt = navigate_project(
        page, settings, tmp_path / "navigation", destination, expected_star="ALTHINAGON"
    )
    assert receipt["navigation_clicks"] == 1 and receipt["destination_verified"]
    assert frame.evaluate("window.fixtureClicks") == 1
    assert receipt["answer_writes"] == receipt["collection_clicks"] == receipt["save_clicks"] == 0
    assert receipt["assessment_clicks"] == receipt["submission_clicks"] == 0
    assert not receipt["task_completed"] and not receipt["automatic_retry"] and not receipt["config_mutated"]
    assert settings.model_dump() == original
    assert not page.get_by_role("checkbox").is_checked()
    if destination not in {"list", "starfield"}:
        assert receipt["same_star_verified"] and receipt["to"]["star"] == "Althinagon"
    text = json.loads((tmp_path / "navigation/after/observation.json").read_text())["frames"][0]["text"]
    assert all(label in text for label in receipt["suggested_required_text"])


@pytest.mark.parametrize("destination", ["stellar", "planet", "habitability"])
def test_numbered_tabs_preserve_list_context(navigation_page, tmp_path, destination):
    page, frame = navigation_page
    frame.evaluate("window.fixtureRender('list','stellar')")
    receipt = navigate_project(page, config(), tmp_path / "navigation", destination)
    assert receipt["from"]["surface"] == receipt["to"]["surface"] == "list"
    assert receipt["to"]["section"] == destination and not receipt["same_star_verified"]
    assert frame.evaluate("window.fixtureClicks") == 1


@pytest.mark.parametrize("change", ["known_footer", "row", "score", "unknown_notice"])
def test_list_footer_change_is_narrow_and_preserves_preclick_evidence(
    navigation_page, tmp_path, monkeypatch, change
):
    import habfly.browser_project_navigation as module

    page, frame = navigation_page
    frame.evaluate("window.fixtureRender('list','planet')")
    original, reads = module.inspect_page, []

    def inspect(*args, **kwargs):
        report = original(*args, **kwargs)
        reads.append(True)
        if len(reads) > 2:
            return report
        # Inject the already unit-tested, real capture footer spellings into
        # this fixture's readback. This isolates the navigator's comparison;
        # actual fixture targets and page context are still guarded normally.
        sim = report["frames"][0]
        sim["text"] = (
            sim["text"].rsplit("viewing", 1)[0] + "\n \nVIEWING\n1-1 OF 1\n \nTOTAL COLLECTED\n1\nSave"
        )
        sim["accessibility"] = (
            sim["accessibility"].rsplit("\n- text: viewing", 1)[0]
            + '\n- button [disabled]\n- button [disabled]\n- text: viewing 1-1 of 1 total collected 1\n- button "Save"'
        )
        if len(reads) == 2:
            notice = "Unexpected notice" if change == "unknown_notice" else "Data saved"
            sim["text"] = sim["text"].replace("\n \nVIEWING", f"\n{notice}\n \nVIEWING")
            sim["accessibility"] = sim["accessibility"].replace(
                "\n- button [disabled]", f"\n- text: {notice}\n- button [disabled]", 1
            )
            if change in {"row", "score"}:
                sim["text"] += "\nCHANGED ROW VALUE" if change == "row" else "\nCHANGED SCORE"
        return report

    monkeypatch.setattr(module, "inspect_page", inspect)
    if change == "known_footer":
        receipt = navigate_project(page, config(), tmp_path / "navigation", "stellar")
        assert receipt["destination_verified"] and frame.evaluate("window.fixtureClicks") == 1
    else:
        with pytest.raises(BrowserSafetyStop, match="target_or_screen_changed"):
            navigate_project(page, config(), tmp_path / "navigation", "stellar")
        assert frame.evaluate("window.fixtureClicks") == 0
    before = json.loads((tmp_path / "navigation/before/observation.json").read_text())
    current = json.loads((tmp_path / "navigation/pre-click/observation.json").read_text())
    assert before["frames"][0]["text"] != current["frames"][0]["text"]


@pytest.mark.parametrize(
    "kind",
    [
        "wrong_star",
        "duplicate_star",
        "duplicate_tab",
        "unknown_header",
        "third_header",
        "header_spacing",
        "header_low",
        "occluded",
        "modal",
        "auth",
        "unknown_frame",
    ],
)
def test_unknown_or_unsafe_targets_do_not_click(navigation_page, tmp_path, kind):
    page, frame = navigation_page
    destination, star = "planet", None
    if kind == "wrong_star":
        star = "Otherstar"
    elif kind == "duplicate_star":
        frame.locator("#content").evaluate("e=>e.insertAdjacentHTML('beforeend','<div>Althinagon</div>')")
    elif kind == "duplicate_tab":
        frame.locator(".numbered").nth(1).evaluate("e=>e.after(e.cloneNode(true))")
    elif kind.startswith("header") or kind in {"unknown_header", "third_header"}:
        destination = "list"
        script = {
            "unknown_header": "document.querySelector('#list').setAttribute('aria-label','unrelated')",
            "third_header": "document.querySelector('#list').after(document.querySelector('#list').cloneNode(true))",
            "header_spacing": "document.querySelector('#list').style.left='55px'",
            "header_low": "document.querySelectorAll('.header').forEach(e=>e.style.top='120px')",
        }[kind]
        frame.evaluate(script)
    elif kind == "occluded":
        frame.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<div style=\"position:fixed;inset:0;z-index:50;background:red\"></div>')"
        )
    elif kind == "unknown_frame":
        frame.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<iframe src=about:blank></iframe>')"
        )
    else:
        frame.locator("body").evaluate(
            "(e,h)=>e.insertAdjacentHTML('beforeend',h)",
            "<div role=dialog>stop</div>" if kind == "modal" else "<input type=password>",
        )
    with pytest.raises(BrowserSafetyStop):
        navigate_project(page, config(), tmp_path / "navigation", destination, expected_star=star)
    assert frame.evaluate("window.fixtureClicks") == 0
    assert not json.loads((tmp_path / "navigation/stopped.json").read_text())["click_may_have_occurred"]


@pytest.mark.parametrize("kind", ["replace", "answer", "outer", "frame", "move", "dialog"])
def test_post_reservation_changes_prevent_click(navigation_page, tmp_path, monkeypatch, kind):
    import habfly.browser_project_navigation as module

    page, frame = navigation_page
    original = module.persist_json

    def persist(path, value):
        original(path, value)
        if path.name == "reserved.json":
            if kind == "replace":
                frame.locator(".numbered").nth(1).evaluate("e=>e.replaceWith(e.cloneNode(true))")
            elif kind == "answer":
                frame.locator("#distance").fill("50")
            elif kind == "outer":
                page.get_by_role("checkbox").check()
            elif kind == "frame":
                page.locator("iframe").nth(1).evaluate("e=>e.replaceWith(e.cloneNode(true))")
            elif kind == "move":
                frame.locator("nav").evaluate("e=>e.style.marginLeft='10px'")
            else:
                frame.evaluate("confirm('private dialog')")

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        navigate_project(page, config(), tmp_path / "navigation", "planet")
    assert frame.evaluate("window.fixtureClicks") == 0
    raw = (tmp_path / "navigation/stopped.json").read_text()
    assert not json.loads(raw)["click_may_have_occurred"] and "private dialog" not in raw


@pytest.mark.parametrize(
    "kind",
    ["star", "wrong_destination", "outer", "frame", "modal", "auth", "escape", "popup", "fleeting_popup"],
)
def test_uncertain_dispatch_is_preserved_without_retry(navigation_page, tmp_path, kind):
    page, frame = navigation_page
    code = {
        "star": "document.querySelector('#content>div').textContent='OTHERSTAR'",
        "wrong_destination": "window.fixtureRender('detail','stellar')",
        "outer": "parent.postMessage('fixture-change','*')",
        "frame": "document.body.insertAdjacentHTML('beforeend','<iframe src=about:blank></iframe>')",
        "modal": "document.body.insertAdjacentHTML('beforeend','<div role=dialog>stop</div>')",
        "auth": "document.body.insertAdjacentHTML('beforeend','<input type=password>')",
        "escape": "parent.postMessage('fixture-escape','*')",
        "popup": "window.open('about:blank')",
        "fleeting_popup": "window.open('about:blank').close()",
    }[kind]
    page.evaluate(
        "addEventListener('message',e=>{if(e.data==='fixture-change')document.querySelector('input').checked=true;if(e.data==='fixture-escape')history.replaceState({},'', '/outside')})"
    )
    frame.locator(".numbered").nth(1).evaluate(
        "(e,code)=>e.onclick=()=>{window.fixtureNavigate('planet');new Function(code)()}", code
    )
    with pytest.raises(BrowserSafetyStop):
        navigate_project(page, config(), tmp_path / "navigation", "planet", expected_star="Althinagon")
    assert frame.evaluate("window.fixtureClicks") == 1
    assert json.loads((tmp_path / "navigation/stopped.json").read_text())["click_may_have_occurred"]
    assert not (tmp_path / "navigation/confirmed.json").exists()


def test_starfield_cannot_masquerade_as_a_numbered_detail_tab(navigation_page, tmp_path):
    page, frame = navigation_page
    frame.evaluate("window.fixtureRender('starfield',null)")
    with pytest.raises(BrowserSafetyStop, match="requires_detail_or_list"):
        navigate_project(page, config(), tmp_path / "navigation", "planet")
    assert frame.evaluate("window.fixtureClicks") == 0


def test_existing_output_is_not_overwritten(navigation_page, tmp_path):
    page, frame = navigation_page
    output = tmp_path / "navigation"
    output.mkdir()
    with pytest.raises(FileExistsError):
        navigate_project(page, config(), output, "planet")
    assert frame.evaluate("window.fixtureClicks") == 0


def test_detail_roundtrip_requires_same_star_for_each_tab(navigation_page, tmp_path):
    page, frame = navigation_page
    for index, destination in enumerate(("planet", "stellar", "habitability", "planet")):
        receipt = navigate_project(
            page, config(), tmp_path / f"navigation-{index}", destination, expected_star="Althinagon"
        )
        assert receipt["same_star_verified"] and receipt["to"]["section"] == destination
    assert frame.evaluate("window.fixtureClicks") == 4


def test_hidden_retained_tooltip_text_is_not_a_starfield_identity(navigation_page, tmp_path):
    from habfly.browser_probe import inspect_page
    from habfly.browser_project_navigation import project_view

    page, frame = navigation_page
    frame.evaluate("window.fixtureRender('starfield',null)")
    report = inspect_page(page, config())
    report["frames"][0]["text"] += "\nOLDSTAR\nSTAR COLLECTED\nVIEW STAR DATA"
    assert project_view(report) == {"surface": "starfield", "section": None, "star": None}
    receipt = navigate_project(page, config(), tmp_path / "navigation", "list")
    assert receipt["to"]["surface"] == "list" and not receipt["same_star_verified"]


def test_header_handle_replacement_after_reservation_is_rejected(navigation_page, tmp_path, monkeypatch):
    import habfly.browser_project_navigation as module

    page, frame = navigation_page
    original = module.persist_json

    def persist(path, value):
        original(path, value)
        if path.name == "reserved.json":
            frame.locator("#list").evaluate("e=>e.replaceWith(e.cloneNode(true))")

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop, match="target_or_screen_changed"):
        navigate_project(page, config(), tmp_path / "navigation", "list")
    assert frame.evaluate("window.fixtureClicks") == 0


@pytest.mark.parametrize("destination,star", [("submit", None), ("planet", "")])
def test_invalid_requests_never_dispatch(navigation_page, tmp_path, destination, star):
    page, frame = navigation_page
    with pytest.raises(BrowserSafetyStop):
        navigate_project(page, config(), tmp_path / "navigation", destination, expected_star=star)
    assert frame.evaluate("window.fixtureClicks") == 0
