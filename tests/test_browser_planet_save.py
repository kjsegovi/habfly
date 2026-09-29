"""Save receipt is not score transfer, submission or durable session state."""
# ruff: noqa: F811

import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_classification import classified_page  # noqa: F401
from test_browser_planet_numeric import planet_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_save import save_planet_work


@pytest.fixture
def save_page(classified_page):
    page, frame = classified_page
    frame.locator(".choice label").first.click()
    frame.get_by_role("button", name="Save", exact=True).evaluate("""e=>{
        const notice=document.createElement('div');notice.id='save-notice';
        const footer=document.createElement('div');e.before(footer);footer.append(notice,e);
        e.onclick=()=>notice.textContent='Data saved';
    }""")
    return page, frame


@pytest.mark.parametrize("already_present", [False, True])
def test_save_ack_is_distinct_from_persistence_and_completion(save_page, tmp_path, already_present):
    page, frame = save_page
    if already_present:
        frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
    receipt = save_planet_work(page, config(), tmp_path / "save")
    assert receipt["answers_unchanged"] and receipt["data_saved_notice_observed"]
    assert receipt["notice_was_already_present"] == already_present
    assert not any(
        receipt[k]
        for k in (
            "cross_session_persistence_verified",
            "task_completed",
            "assessed",
            "score_updated",
            "submitted",
        )
    )
    assert not page.get_by_role("checkbox").is_checked()
    with pytest.raises(FileExistsError):
        save_planet_work(page, config(), tmp_path / "save")


@pytest.mark.parametrize(
    "kind", ["blank", "class", "disabled", "dialog", "no_ack", "answer", "other_feedback", "navigation"]
)
def test_failed_save_has_no_completion_or_automatic_retry(save_page, tmp_path, kind):
    page, frame = save_page
    if kind == "blank":
        frame.locator("#period_days").fill("")
    elif kind == "class":
        frame.locator(".choice").first.evaluate("e=>e.classList.remove('selected')")
    elif kind == "disabled":
        frame.get_by_role("button", name="Save").evaluate("e=>e.disabled=true")
    else:
        body = {
            "dialog": "confirm('private confirmation')",
            "no_ack": "",
            "answer": "document.getElementById('period_days').value='999';document.getElementById('save-notice').textContent='Data saved'",
            "other_feedback": "document.getElementById('save-notice').textContent='Save failed'",
            "navigation": "history.replaceState(null,'','/escaped')",
        }[kind]
        frame.get_by_role("button", name="Save").evaluate("(e,code)=>e.onclick=new Function(code)", body)
    with pytest.raises(BrowserSafetyStop):
        save_planet_work(page, config(), tmp_path / "save", timeout_seconds=0.1)
    assert not (tmp_path / "save/confirmed.json").exists()
    raw = (tmp_path / "save/stopped.json").read_text()
    stopped = json.loads(raw)
    assert not stopped["automatic_retry"] and not stopped["task_completed"]
    assert stopped["save_may_have_occurred"] == (kind not in {"blank", "class", "disabled"})
    assert "private confirmation" not in raw


@pytest.mark.parametrize("overlay", [False, True])
@pytest.mark.parametrize("transparent_child", [False, True])
def test_save_child_caption_is_allowed_but_sibling_overlay_blocks(
    save_page, tmp_path, overlay, transparent_child
):
    page, frame = save_page
    button = frame.get_by_role("button", name="Save", exact=True)
    button.evaluate("e=>e.innerHTML='<span>Save</span>'")
    if transparent_child:
        button.evaluate("""e=>{
            e.style.position='relative';
            e.insertAdjacentHTML('beforeend','<span style="position:absolute;inset:0;opacity:0;pointer-events:auto"></span>');
        }""")
    if overlay:
        button.evaluate("""e=>{
            const r=e.getBoundingClientRect(),cover=document.createElement('div');
            cover.style=`position:fixed;left:${r.x}px;top:${r.y}px;width:${r.width}px;height:${r.height}px;background:black;z-index:999`;
            e.after(cover);
        }""")
        with pytest.raises(BrowserSafetyStop, match="save_control_unavailable"):
            save_planet_work(page, config(), tmp_path / "save")
        assert not json.loads((tmp_path / "save/stopped.json").read_text())["save_may_have_occurred"]
    else:
        assert save_planet_work(page, config(), tmp_path / "save")["data_saved_notice_observed"]


def test_brief_visible_ack_is_recorded_before_slow_full_validation(save_page, tmp_path, monkeypatch):
    from habfly.browser_planet_numeric import PlanetNumericSession

    page, frame = save_page
    frame.get_by_role("button", name="Save").evaluate("""e=>e.onclick=()=>{
        const n=document.getElementById('save-notice');n.textContent='Data saved';
        setTimeout(()=>n.textContent='',200);
    }""")
    original = PlanetNumericSession.current
    reads = 0

    def slow_current(self):
        nonlocal reads
        reads += 1
        if reads == 3:
            self.page.wait_for_timeout(350)
        return original(self)

    monkeypatch.setattr(PlanetNumericSession, "current", slow_current)
    result = save_planet_work(page, config(), tmp_path / "save")
    assert result["data_saved_notice_observed"] and not result["notice_was_already_present"]
    assert frame.locator("#save-notice").inner_text() == ""
    assert (tmp_path / "save/acknowledgement.json").exists()
