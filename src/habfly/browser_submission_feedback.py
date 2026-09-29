"""Grounded negative feedback only; no successful-submission recognizer.

The exact public refusal and AX ordering were observed in disposable preview
browser-submission-shape-006/capture-02. That diagnostic is not a canonical
receipt. Runtime recognition additionally needs a post-dispatch transition and
currently exposed native feedback controls/text, with unchanged outer snapshots.
"""

import hashlib
import json
import math
import re

from .browser_setup import RENDER_VISIBILITY_JS

REFUSAL = "You need to analyze and submit at least 30 stars."
READY = "I am ready to submit project."
MODE = "grounded_negative_submission_feedback_v1"
_BLOCK = (
    '  - button "OK"',
    '  - button "Close feedback": Open/Close Feedback',
    "  - paragraph: " + REFUSAL,
    '  - button "Close feedback": \uf00d',
)
_BUTTON_AX = (
    '- button "OK"',
    '- button "Close feedback": Open/Close Feedback',
    '- button "Close feedback": \uf00d',
)
_PARAGRAPH = (
    "(e,text)=>{"
    + RENDER_VISIBILITY_JS
    + """
    if(e.tagName!=='P'||e.innerText.trim()!==text||!styled(e))return null;
    const r=e.getBoundingClientRect();
    if(r.width<=0||r.height<=0||r.left<0||r.top<0||r.right>innerWidth||r.bottom>innerHeight)return null;
    const range=document.createRange();range.selectNodeContents(e);
    const rects=[...range.getClientRects()];
    const textExposed=t=>{
      if(exposed(t,e,true))return true;
      // Observed production003: the exact paragraph is painted by its own
      // noninteractive SPAN. Do not admit sibling overlays or unrelated text.
      const h=document.elementFromPoint((t.left+t.right)/2,(t.top+t.bottom)/2);
      const interactive=h?.closest('button,a[href],input,select,textarea,[role],[tabindex],[contenteditable]');
      return h&&h.tagName==='SPAN'&&e.contains(h)&&h.innerText.trim()===text&&styled(h)&&
        !(interactive&&e.contains(interactive))&&exposed(t,h,true);
    };
    if(!rects.length||!rects.every(t=>t.width>0&&t.height>0&&t.left>=0&&t.top>=0&&
        t.right<=innerWidth&&t.bottom<=innerHeight&&textExposed(t)))return null;
    return {x:r.x,y:r.y,width:r.width,height:r.height};
    }"""
)


def _snapshot(value):
    return (
        isinstance(value, dict)
        and value.get("consistency_unverified") is not True
        and all(isinstance(value.get(k), str) and len(value[k]) <= 100_000 for k in ("text", "accessibility"))
    )


def _ready(ax, checked):
    line = f'    - checkbox "{READY}"' + (" [checked]" if checked else "")
    return (
        ax.splitlines().count(line) == 1 and ax.splitlines().count('  - group "Select all that apply":') == 1
    )


def submission_refusal_candidate(before, after):
    """A cheap shape predicate, expressly not an exposed-feedback decision."""
    if not _snapshot(before) or not _snapshot(after):
        return False
    old_ax, new_ax = before["accessibility"], after["accessibility"]
    if (
        old_ax.splitlines().count("- main:") != 1
        or new_ax.splitlines().count("- main:") != 1
        or not (_ready(old_ax, False) or _ready(old_ax, True))
        or not _ready(new_ax, True)
        or REFUSAL in before["text"]
        or REFUSAL in old_ax
        or 'button "Close feedback"' in old_ax
        or 'button "OK"' in old_ax
    ):
        return False
    lines = new_ax.splitlines()
    matched = (
        sum(line.strip() == REFUSAL for line in after["text"].splitlines()) == 1
        and new_ax.count(REFUSAL) == 1
        and after["text"].count(REFUSAL) == 1
        and lines.count(_BLOCK[0]) == 1
        and lines.count(_BLOCK[1]) == 1
        and lines.count(_BLOCK[2]) == 1
        and lines.count(_BLOCK[3]) == 1
        and sum('button "Close feedback"' in line for line in lines) == 2
        and sum('button "OK"' in line for line in lines) == 1
        and "\n".join(_BLOCK) in new_ax
    )
    if not matched:
        return False
    # Only the grounded feedback block and the already-verified readiness
    # check may differ. An unrelated/contradictory outcome remains unsupported.
    start = lines.index(_BLOCK[0])
    projected = lines[:start] + lines[start + len(_BLOCK) :]
    expected = [
        line + " [checked]" if line == f'    - checkbox "{READY}"' else line for line in old_ax.splitlines()
    ]
    return projected == expected


def _box(value):
    return (
        isinstance(value, dict)
        and set(value) == {"x", "y", "width", "height"}
        and all(type(v) in {int, float} and math.isfinite(v) for v in value.values())
        and value["x"] >= 0
        and value["y"] >= 0
        and value["width"] > 0
        and value["height"] > 0
    )


def classify_submission_feedback(before, after, *, submit_dispatched, exposure=None):
    """Return a negative diagnostic, never a positive or canonical receipt."""
    result = {
        "schema_version": 1,
        "mode": MODE,
        "status": "unrecognized",
        "reason": "unsupported_or_unverified_feedback",
        "refusal_visible": False,
        "submission_verified": False,
        "submitted": False,
        "task_completed": False,
        "project_completed": False,
        "canonical_receipt": False,
        "automatic_retry": False,
        "positive_acknowledgement_recognized": False,
    }
    if submit_dispatched is not True or not submission_refusal_candidate(before, after):
        return result
    if (
        not isinstance(exposure, dict)
        or exposure.get("same_outer_snapshot_verified") is not True
        or not _box(exposure.get("paragraph"))
        or not isinstance(exposure.get("buttons"), list)
        or len(exposure["buttons"]) != 3
        or [b.get("accessibility") for b in exposure["buttons"] if isinstance(b, dict)] != list(_BUTTON_AX)
        or any(not isinstance(b, dict) or not _box(b.get("box")) for b in exposure["buttons"])
    ):
        return result
    return {
        **result,
        "status": "course_refusal",
        "reason": "at_least_thirty_analyzed_submitted_stars_required",
        "refusal_visible": True,
        "visible_text": REFUSAL,
        "before_outer_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        "after_outer_sha256": hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
        "exposure": exposure,
        "interpretation": "The course displayed this refusal; canonical submission remains pending, with no retry authorized.",
    }


def read_submission_feedback_exposure(page, after, *, guard, timeout):
    """Read exposed native elements only, then recheck the exact outer snapshot.

    The caller supplies its existing auth/modal/frame/source/deadline guard.
    A missing or covered element returns no proof, never an action/recovery.
    """
    from .browser_submission_controls import SUBMIT_EXPOSED

    guard()
    # Bind the observed paragraph role, not the deepest text node: the visible
    # sentence may be wrapped in a span without changing its paragraph AX.
    paragraph = page.get_by_role("paragraph").filter(
        has_text=re.compile(r"^\s*" + re.escape(REFUSAL) + r"\s*$")
    )
    ok = page.get_by_role("button", name="OK", exact=True)
    closes = page.get_by_role("button", name="Close feedback", exact=True)
    if paragraph.count() != 1 or ok.count() != 1 or closes.count() != 2:
        return None
    controls = [ok, *closes.all()]
    handles = [paragraph.element_handle(timeout=timeout())]
    handles += [c.element_handle(timeout=timeout()) for c in controls]
    if any(handle is None for handle in handles):
        return None
    if paragraph.aria_snapshot(timeout=timeout()) != "- paragraph: " + REFUSAL:
        return None
    paragraph_box = handles[0].evaluate(_PARAGRAPH, REFUSAL)
    buttons = []
    button_proofs = []
    for control, handle, expected in zip(controls, handles[1:], _BUTTON_AX, strict=True):
        guard()
        if not control.is_enabled(timeout=timeout()) or control.aria_snapshot(timeout=timeout()) != expected:
            return None
        proof = handle.evaluate(SUBMIT_EXPOSED)
        if not isinstance(proof, dict):
            return None
        button_proofs.append(proof)
        buttons.append(
            {"accessibility": expected, "box": {k: proof[k] for k in ("x", "y", "width", "height")}}
        )
    guard()
    body = page.locator("body")
    text, ax = body.inner_text(timeout=timeout()), body.aria_snapshot(timeout=timeout())
    guard()
    if text != after["text"] or ax != after["accessibility"]:
        return None
    rebound = [paragraph.element_handle(timeout=timeout())]
    rebound += [c.element_handle(timeout=timeout()) for c in [ok, *closes.all()]]
    if len(rebound) != len(handles) or any(
        not old.evaluate("(a,b)=>a.isConnected&&a===b", new)
        for old, new in zip(handles, rebound, strict=True)
    ):
        return None
    if handles[0].evaluate(_PARAGRAPH, REFUSAL) != paragraph_box or any(
        handle.evaluate(SUBMIT_EXPOSED) != proof
        for handle, proof in zip(handles[1:], button_proofs, strict=True)
    ):
        return None
    guard()
    return {"paragraph": paragraph_box, "buttons": buttons, "same_outer_snapshot_verified": True}
