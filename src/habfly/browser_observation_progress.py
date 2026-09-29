"""Read-only raster progress of the course's plotted observation trace.

An axis ending at 10,000 is not proof that 10,000 days have been collected.
This diagnostic reads only a chart crop and fully exposed numeric axis labels.
It never reads SVG paths, chooses a planet answer, or certifies data completeness.
"""

import hashlib
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_planet_chart import FluxChartSession
from .browser_setup import RENDER_VISIBILITY_JS
from .presentation_capture import evidence_screenshot

PLOT_EXPOSED = (
    "e=>{"
    + RENDER_VISIBILITY_JS
    + """
    if(!e.isConnected||!styled(e))return false;
    const c=e.getBoundingClientRect();
    if(Math.abs(c.width-280)>.1||Math.abs(c.height-195)>.1)return false;
    const plot={left:c.left+30,right:c.left+261,top:c.top+20,bottom:c.top+151};
    let left=0,top=0,right=innerWidth,bottom=innerHeight;
    for(let p=e.parentElement;p;p=p.parentElement){
        const s=getComputedStyle(p),b=p.getBoundingClientRect();
        if(['hidden','clip','scroll','auto'].includes(s.overflowX)){left=Math.max(left,b.left);right=Math.min(right,b.right);}
        if(['hidden','clip','scroll','auto'].includes(s.overflowY)){top=Math.max(top,b.top);bottom=Math.min(bottom,b.bottom);}
    }
    if(plot.left<left||plot.right>right||plot.top<top||plot.bottom>bottom)return false;
    // Inspect actual hit-testing across the plotted region, not the geometry of
    // covered background screens. Raster analysis separately rejects gaps and
    // unknown colors (including non-interactive overlays).
    for(let x=31;x<=260;x+=4)for(let y=21;y<=150;y+=4){
        const hit=document.elementFromPoint(c.left+x,c.top+y);
        if(!hit||!(hit===e||e.contains(hit))||!styled(hit))return false;
    }
    return true;
    }"""
)


def trace_mask(png):
    """Observed plot palette, excluding UI icons/labels; no curve/data access."""
    with Image.open(io.BytesIO(png)) as image:
        # A fractional CSS top coordinate makes Playwright round the capture
        # height up by one pixel. Horizontal axis coordinates remain unchanged.
        if image.size not in {(280, 195), (280, 196)}:
            raise BrowserSafetyStop("unsupported_observation_crop_geometry")
        pixels = np.asarray(image.convert("RGB"), dtype=float)
    r, g, b = (pixels[:, :, index] for index in range(3))
    mask = (
        (r >= 10)
        & (r <= 110)
        & (g >= 1.35 * r)
        & (g <= 1.95 * r)
        & (b >= 1.65 * r)
        & (b <= 2.35 * r)
        & (b >= g + 5)
    )
    mask[:20] = False
    mask[152:] = False
    return mask


def trace_progress(png, labels, *, requested_days):
    if type(requested_days) is not int or not 1 <= requested_days <= 10000:
        raise ValueError("An explicit bounded observation duration is required")
    mask = trace_mask(png)
    if len(labels) < 3:
        raise BrowserSafetyStop("insufficient_observation_axis_labels")
    try:
        points = sorted((float(row["value"]), float(row["center_x"])) for row in labels)
        slope = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
        origin = points[0][1] - points[0][0] * slope
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        raise BrowserSafetyStop("invalid_observation_progress_axis") from None
    end = origin + requested_days * slope
    if (
        slope <= 0
        or not all(math.isfinite(v) for row in points for v in row)
        or not 29 <= origin <= 32
        or not 96 <= end <= 262
        or points[-1][0] < requested_days
        or len({row[0] for row in points}) != len(points)
        or any(abs(origin + day * slope - x) > 1 for day, x in points)
    ):
        raise BrowserSafetyStop("observation_range_not_fully_visible")
    mask[:, : math.floor(origin)] = False
    mask[:, math.ceil(end) + 1 :] = False
    columns = np.flatnonzero(mask.any(axis=0))
    frontier = float(columns[-1]) if len(columns) else None
    # A clipped transit or obscured trace may also give insufficient evidence.
    # Do not turn either case into a synthetic value or an absence conclusion.
    substantial = len(columns) >= 64
    endpoint_columns = int(sum(math.ceil(end) - 8 <= x <= math.ceil(end) for x in columns))
    reaches_end = substantial and frontier >= end - 1.5 and endpoint_columns >= 6
    return {
        "method": "rendered_blue_trace_frontier_v1",
        "source": "chart_crop_pixels_and_visible_axis_labels",
        "requested_days": requested_days,
        "status": "trace_reaches_requested_end" if reaches_end else "trace_partial_or_unresolved",
        "trace_columns": len(columns),
        "endpoint_trace_columns": endpoint_columns,
        "rightmost_trace_pixel": frontier,
        "approximate_rendered_day": None
        if frontier is None
        else round(max(0, (frontier - origin) / slope), 1),
        "endpoint_visible": reaches_end,
        "observation_completed": False,
        "planet_presence": None,
        "learned_perception": False,
        "task_completed": False,
        "limitation": "Raster readiness evidence only; clipping, occlusion and rendering can hide a trace. Not transit, period or planet-absence evidence.",
    }


def capture_observation_progress(page, config, output, *, requested_days, max_seconds=30):
    # A slower native chart may need an explicitly larger read budget. Never
    # extend a running capture, retry its actions, or change observation days.
    if type(max_seconds) not in {int, float} or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 120:
        raise ValueError("Chart capture budget must be between 1 and 120 seconds")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    events = []
    session = None
    try:
        session = FluxChartSession(
            page,
            config,
            lambda event, payload: events.append({"event": event, "payload": payload}),
            max_actions=1,
            max_seconds=max_seconds,
        )
        session._guard()
        labels = session.time_axis_labels()
        flux_labels = session.flux_axis_labels()
        if not session.handle.evaluate(PLOT_EXPOSED):
            raise BrowserSafetyStop("observation_plot_occluded")
        png = evidence_screenshot(session.chart)
        session._guard()
        if not session.handle.evaluate(PLOT_EXPOSED):
            raise BrowserSafetyStop("observation_plot_occluded")
        if session.time_axis_labels() != labels or session.flux_axis_labels() != flux_labels:
            raise BrowserSafetyStop("observation_axis_changed_during_crop")
        with (directory / "chart.png").open("xb") as stream:
            stream.write(png)
        report = {
            **trace_progress(png, labels, requested_days=requested_days),
            "star": session.star,
            "time_axis_labels": labels,
            "flux_axis_labels": flux_labels,
            "chart_sha256": hashlib.sha256(png).hexdigest(),
            "browser_actions": 0,
            "answer_writes": 0,
            "capture_max_seconds": max_seconds,
        }
        persist_json(directory / "report.json", report)
        return report
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "observation_progress_capture_failed",
                "browser_actions": 0,
                "answer_writes": 0,
            },
        )
        raise
    finally:
        if session is not None:
            session.close()
