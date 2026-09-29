"""One explicit Save click, with a UI acknowledgement distinct from persistence.

No server/storage inspection, reload, score update, or submission. A Save banner
only acknowledges the UI; it does not prove this preview survives a new session.
"""

import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_numeric import committed_display, screen_identity
from .browser_planet_chart import VISIBLE_TOOLTIP
from .browser_planet_numeric import ANSWER_UNITS, PlanetNumericSession
from .browser_probe import save_probe
from .browser_setup import RENDER_VISIBILITY_JS
from .browser_stellar import SIMULATION_URL


def save_control_exposed(control):
    # The live button contains an opacity-zero loading span that still receives
    # pointer hits. It belongs to the visible native button's clickable surface;
    # it is not a source of readable text. A sibling overlay is never accepted.
    return control.evaluate(
        "e=>{"
        + RENDER_VISIBILITY_JS
        + """
        const r=e.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
        return e.tagName==='BUTTON' && styled(e) && hit && (hit===e||e.contains(hit)) &&
            exposed(r,hit);
    }"""
    )


def save_planet_work(page, config, output, *, timeout_seconds=20):
    if type(timeout_seconds) not in {int, float} or not 0.1 <= timeout_seconds <= 30:
        raise ValueError("Save acknowledgement deadline must be bounded")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    session = None
    attempted = False
    last_report = None
    try:
        session = PlanetNumericSession(page, config, directory / "read-guard", max_seconds=120)
        before, mapping, choices, _ = session.current()
        last_report = before
        fields = mapping["observation"]["values"]["browser_field_map"]
        for name in ANSWER_UNITS:
            committed_display(fields[name]["current_value"], fields[name]["current_value"])
        if choices["selected"] is None:
            raise BrowserSafetyStop("planet_class_required_before_save")
        button = session.frame.get_by_role("button", name="Save", exact=True)
        if button.count() != 1 or not button.is_enabled() or not save_control_exposed(button):
            raise BrowserSafetyStop("save_control_unavailable")
        handle = button.element_handle(timeout=2000)
        initial_notice = next(f["text"] for f in before["frames"] if f["url"] == SIMULATION_URL).endswith(
            "\nData saved\nSave"
        )
        intent = {
            "kind": "CLICK",
            "visible_label": "Save",
            "star": mapping["star_name"],
            "before_sha256": screen_identity(before),
            "max_save_clicks": 1,
            "numeric_writes": 0,
            "selection_source": "scripted_setup_not_learned",
            "task_completed": False,
            "automatic_retry": False,
        }
        persist_json(directory / "reserved.json", intent)
        session.current()
        if not handle.evaluate("(a,b)=>a.isConnected&&a===b", button.element_handle(timeout=2000)):
            raise BrowserSafetyStop("save_control_replaced")
        attempted = True
        handle.click(timeout=3000)
        deadline = time.monotonic() + timeout_seconds
        while True:
            # Observe a potentially brief painted acknowledgement before the
            # slower full-screen comparison. No application state is queried.
            notices = [
                e
                for e in session.frame.get_by_text("Data saved", exact=True).all()
                if e.is_visible()
                and e.evaluate(
                    "(e,b)=>b.parentElement.contains(e)&&b.parentElement.getBoundingClientRect().height<=64",
                    handle,
                )
                and e.evaluate(VISIBLE_TOOLTIP)
            ]
            if len(notices) == 1 and button.is_enabled():
                persist_json(
                    directory / "acknowledgement.json",
                    {
                        "visible_text": "Data saved",
                        "source": "fully_exposed_footer_text",
                        "notice_was_already_present": initial_notice,
                    },
                )
                after, _, _, _ = session.current()
                last_report = after
                break
            if time.monotonic() >= deadline:
                last_report, _, _, _ = session.current()
                raise BrowserSafetyStop("save_acknowledgement_timeout")
            page.wait_for_timeout(100)
        save_probe(after, directory / "after")
        receipt = {
            **intent,
            "save_click_delivered": True,
            "data_saved_notice_observed": True,
            "notice_was_already_present": initial_notice,
            "answers_unchanged": True,
            "cross_session_persistence_verified": False,
            "course_completion_verified": False,
            "assessed": False,
            "score_updated": False,
            "submitted": False,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        if last_report is not None:
            save_probe(last_report, directory / "last-verified-observation")
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "planet_save_operation_failed",
                "save_may_have_occurred": attempted,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        raise
    finally:
        if session is not None:
            session.close()
