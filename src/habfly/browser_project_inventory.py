"""Read-only, complete visible stellar-list inventory; no pagination or catalog.

Counts collection only, never analyzed work, correctness, persistence or project
completion. Covered/clipped names and partial pages cannot certify an inventory.
"""

import hashlib
import re
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_collected import row_icon_exposed
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_project_navigation import project_view
from .browser_setup import RENDER_VISIBILITY_JS
from .browser_stellar import NUMBER, SIMULATION_URL, _atoms

NAME = r"[A-Za-z][A-Za-z0-9 '-]{1,79}"
VIEWING = re.compile(r"\bviewing\s+(\d+)\s*-\s*(\d+)\s+of\s+(\d+)\b", re.IGNORECASE)
TOTAL = re.compile(r"\btotal\s+collected\s+(\d+)\b", re.IGNORECASE)
ROW = re.compile(rf"^({NAME}?)\s+{NUMBER}\s+{NUMBER}\s+{NUMBER}\s+.+$", re.IGNORECASE)

# Reads painted text and geometry only. A text fragment must be wholly inside
# the viewport and clipping ancestors; a covered centre/edge is not accepted.
VISIBLE_TEXT = (
    """root=>{"""
    + RENDER_VISIBILITY_JS
    + """
  const out=[], walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
  while(walker.nextNode()) {
    const node=walker.currentNode,e=node.parentElement;
    if(!styled(e))continue;
    const text=node.textContent.replace(/\\s+/g,' ').trim();if(!text)continue;
    const range=document.createRange();
    range.setStart(node,node.textContent.search(/\\S/));
    range.setEnd(node,node.textContent.replace(/\\s+$/,'').length);
    const rects=[...range.getClientRects()];
    const boxes=[];let okay=rects.length>0;
    for(const r of rects){
      if(r.width<=0||r.height<=0||r.left<0||r.top<0||r.right>innerWidth||r.bottom>innerHeight){okay=false;break}
      for(let p=e.parentElement;p;p=p.parentElement){const s=getComputedStyle(p),b=p.getBoundingClientRect();
        if((['hidden','clip','scroll','auto'].includes(s.overflowX)&&(r.left<b.left||r.right>b.right))||
           (['hidden','clip','scroll','auto'].includes(s.overflowY)&&(r.top<b.top||r.bottom>b.bottom)))okay=false;
      }
      // Chromium hit-testing rounds fractional right edges by up to a pixel.
      const dx=Math.min(1.25,r.width/4),dy=Math.min(1.25,r.height/4);
      for(const x of [r.left+dx,(r.left+r.right)/2,r.right-dx])
        for(const y of [r.top+dy,(r.top+r.bottom)/2,r.bottom-dy]){
          const h=document.elementFromPoint(x,y);
          if(!(h===e||(getComputedStyle(e).pointerEvents==='none'&&h&&h.contains(e))))okay=false;
        }
      boxes.push({x:r.x,y:r.y,width:r.width,height:r.height});
    }
    if(okay)out.push({text,boxes});
  }return out;
}"""
)


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("project_inventory_" + reason)


def _counts(text):
    viewing, total = VIEWING.findall(text), TOTAL.findall(text)
    _require(len(viewing) == len(total) == 1, "ambiguous_or_missing_visible_counts")
    start, end, count = map(int, viewing[0])
    _require(
        int(total[0]) == count and 0 <= count <= 100 and (start, end) == ((1, count) if count else (0, 0)),
        "incomplete_visible_list",
    )
    return {"start": start, "end": end, "total": count}


def _capture_inventory(report):
    _require(
        project_view(report) == {"surface": "list", "section": "stellar", "star": None},
        "stellar_list_required",
    )
    capture = next(frame for frame in report["frames"] if frame["url"] == SIMULATION_URL)
    texts = [
        value for key, value in _atoms(capture["accessibility"]) if key == "text" and isinstance(value, str)
    ]
    counts = _counts(" ".join(texts))
    rows = []
    for text in texts:
        match = ROW.fullmatch(" ".join(text.split()))
        if match:
            rows.append({"name": match[1], "text": " ".join(text.split())})
    names = [row["name"].casefold() for row in rows]
    _require(
        len(names) == counts["total"] and len(set(names)) == len(names), "duplicate_or_missing_visible_rows"
    )
    return counts, rows


def _outer_exposed(frame, boxes):
    return frame.frame_element().evaluate(
        """(e,boxes)=>{
      if(getComputedStyle(e).opacity==='0')return false;
      const b=e.getBoundingClientRect();
      return boxes.every(r=>[r.x+0.5,r.x+r.width/2,r.x+r.width-0.5].every(x=>
        [r.y+0.5,r.y+r.height/2,r.y+r.height-0.5].every(y=>{
          const px=x+b.x+e.clientLeft,py=y+b.y+e.clientTop;
          return px>=0&&py>=0&&px<innerWidth&&py<innerHeight&&document.elementFromPoint(px,py)===e;
        })));
    }""",
        boxes,
    )


def _visible_rows(frame, counts, rows):
    nodes = frame.locator("body").evaluate(VISIBLE_TEXT)
    _require(_counts(" ".join(node["text"] for node in nodes)) == counts, "visible_count_mismatch")
    # The footer and every named row must also remain exposed through the outer
    # iframe, not merely visible to the simulation's own elementFromPoint.
    _require(_outer_exposed(frame, [box for node in nodes for box in node["boxes"]]), "outer_frame_occluded")
    verified, handles = [], []
    for row in rows:
        name = row["name"]
        _require(
            sum(node["text"].casefold() == name.casefold() for node in nodes) == 1,
            "duplicate_or_unexposed_name",
        )
        names = [
            e
            for e in frame.get_by_text(re.compile(r"^" + re.escape(name) + r"$", re.IGNORECASE)).all()
            if e.is_visible()
        ]
        _require(len(names) == 1, "ambiguous_name_locator")
        label = names[0]
        data, container = label.locator("../.."), label.locator("../../..")
        _require(
            " ".join(data.inner_text().split()).casefold() == row["text"].casefold(),
            "row_text_or_structure_changed",
        )
        nb, rb = label.bounding_box(), container.bounding_box()
        eyes = []
        for image in container.get_by_role("img").all():
            box = image.bounding_box() if image.is_visible() else None
            if (
                box
                and nb
                and rb
                and image.aria_snapshot() == "- img"
                and box["width"] == 22
                and box["height"] == 25
                and rb["x"] <= box["x"] < box["x"] + 22 <= nb["x"]
                and rb["y"] <= box["y"] < box["y"] + 25 <= rb["y"] + rb["height"]
                and row_icon_exposed(image)
            ):
                eyes.append(image)
        _require(len(eyes) == 1, "unverified_named_row_structure")
        handles.append(label.element_handle(timeout=2000))
        verified.append(
            {
                "name": name,
                "row_accessibility_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "visible_name_box": nb,
            }
        )
    return verified, handles


def verify_project_inventory(page, config, output, expected_stars=None):
    """Certify only a complete, stable, currently visible stellar collection.

    expected_stars, when supplied, is an exact case-insensitive set, not a hint
    for finding absent rows. No scrolling, navigation or browser writes occur.
    """
    if expected_stars is not None:
        _require(
            isinstance(expected_stars, (list, tuple))
            and len(expected_stars) <= 100
            and all(
                isinstance(name, str) and re.fullmatch(NAME, name) and name == " ".join(name.split())
                for name in expected_stars
            ),
            "invalid_expected_stars",
        )
        expected_stars = [name.casefold() for name in expected_stars]
        _require(len(set(expected_stars)) == len(expected_stars), "duplicate_expected_stars")
    boundary = config.model_copy(deep=True)
    rules = [rule for rule in boundary.frames if rule.url == SIMULATION_URL]
    _require(len(rules) == 1 and rules[0].count == 1, "unsupported_simulation_boundary")
    rules[0].required_text = []
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    frames, dialogs, popups = page.frames.copy(), [], []

    def dialog_handler(dialog):
        dialogs.append(True)
        dialog.dismiss()

    def popup_handler(_page):
        popups.append(True)

    page.on("dialog", dialog_handler)
    page.context.on("page", popup_handler)
    try:

        def read():
            _require(
                not dialogs and not popups and page.frames == frames and len(page.context.pages) == 1,
                "context_changed",
            )
            report = inspect_page(page, boundary)
            _require(not report["ignored_frame_urls"], "unknown_frame")
            simulation = [f for f in page.frames if f.url == SIMULATION_URL]
            _require(
                len(simulation) == 1 and simulation[0].parent_frame == page.main_frame,
                "unsupported_frame_embedding",
            )
            counts, rows = _capture_inventory(report)
            verified, handles = _visible_rows(simulation[0], counts, rows)
            if expected_stars is not None:
                _require(
                    set(expected_stars) == {row["name"].casefold() for row in verified},
                    "expected_stars_mismatch",
                )
            return report, counts, verified, handles

        before, counts, rows, handles = read()
        first = save_probe(before, directory / "before")
        after, newer_counts, newer_rows, newer_handles = read()
        _require(
            not dialogs
            and not popups
            and page.frames == frames
            and len(page.context.pages) == 1
            and screen_identity(before) == screen_identity(after)
            and counts == newer_counts
            and rows == newer_rows
            and all(a.evaluate("(a,b)=>a.isConnected&&a===b", b) for a, b in zip(handles, newer_handles)),
            "changed_during_readback",
        )
        last = save_probe(after, directory / "after")
        receipt = {
            "schema_version": 1,
            "mode": "read_only_collected_star_inventory",
            "authority": "visible_collected_list_readback",
            "section": "stellar",
            "total_collected": counts["total"],
            "visible_row_count": len(rows),
            "viewing": counts,
            "rows": [{**row, "source_sha256": last["observation_sha256"]} for row in rows],
            "collection_count_verified": True,
            "complete_visible_list_verified": True,
            "expected_stars_verified": expected_stars is not None,
            "source_sha256": {
                "before/observation.json": first["observation_sha256"],
                "after/observation.json": last["observation_sha256"],
            },
            "browser_actions": 0,
            "navigation_clicks": 0,
            "scrolls": 0,
            "answer_writes": 0,
            "save_clicks": 0,
            "deletion_clicks": 0,
            "assessment_clicks": 0,
            "submission_clicks": 0,
            "hidden_catalog_read": False,
            "learned_perception": False,
            "scientific_verified": False,
            "cross_session_persistence_verified": False,
            "task_completed": False,
            "project_completed": False,
            "config_mutated": False,
            "automatic_retry": False,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "project_inventory_read_failed",
                "collection_count_verified": False,
                "browser_actions": len(dialogs),
                "modal_dismissals": len(dialogs),
                "task_completed": False,
                "automatic_retry": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("project_inventory_read_failed") from None
    finally:
        page.remove_listener("dialog", dialog_handler)
        page.context.remove_listener("page", popup_handler)
