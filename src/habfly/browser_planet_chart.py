"""Bounded chart-only UI transport. No form writes, scoring, or hidden data.

The caller chooses zoom, pan, and hover positions. This transport does not choose
measurements, infer a planet, restart observation, or repair policy decisions.
It reads ordinary rendered tooltip text; no SVG paths or simulation arrays.
"""

import math
import re
import time

from .browser import BrowserSafetyStop
from .browser_probe import _url_identity, _visible_frame, inspect_page
from .browser_setup import RENDER_VISIBILITY_JS
from .browser_stellar import SIMULATION_URL
from .planet_charts import parse_flux_tooltip

NATIVE_VALUES = (
    """(root,bound) => {"""
    + RENDER_VISIBILITY_JS
    + """
    if(!root.isConnected||root!==document.body) return {detached:true};
    const chart=bound?.chart;
    if(chart&&!chart.isConnected) return {chart_detached:true};
    if(chart&&chart.parentElement!==bound.parent) return {chart_reparented:true};
    const visible=e=>{const r=e.getBoundingClientRect();return styled(e)&&r.width>0&&r.height>0;};
    // All ordinary preview embeddings belong to this main document. Check
    // their exact retained handles in the same read as its native controls.
    if(bound?.embeddings?.some(e=>!e.isConnected||!visible(e)))
        return {embedding_changed:true};
    if(bound?.hiddenEmbeddings?.some(e=>{
        const r=e.getBoundingClientRect(),s=getComputedStyle(e);
        return s.visibility!=='hidden'&&s.visibility!=='collapse'&&r.width>0&&r.height>0;
    })) return {hidden_embedding_visible:true};
    const auth=[...root.querySelectorAll('input[type=password]')].some(visible);
    const modal=[...root.querySelectorAll('[role=dialog],dialog[open]')].some(visible);
    if(auth||modal) return {auth,modal};
    // The observed plot creates a second 27px icon-only reset button on its
    // first zoom. These two plot-local navigation buttons are not answers.
    // Do not exempt arbitrary buttons, inputs, a broad wrapper, or labels.
    let navigation=[];
    const scope=chart?.parentElement;
    if(scope){
        const c=chart.getBoundingClientRect(),s=scope.getBoundingClientRect();
        if(s.width>=c.width&&s.width<=c.width+16&&s.height>=c.height&&s.height<=c.height+16){
            navigation=[...scope.querySelectorAll('button')].filter(e=>{
                const r=e.getBoundingClientRect();
                return visible(e)&&r.width===27&&r.height===27&&!e.innerText.trim()
                    &&!e.getAttribute('aria-label')&&!e.getAttribute('title')
                    &&e.querySelectorAll('svg').length===1
                    &&r.left>=c.right-64&&r.right<=s.right&&r.top>=c.top-8&&r.bottom<=c.top+27;
            });
            if(navigation.length>2) navigation=[];
        }
    }
    const values=[...root.querySelectorAll('input,select,textarea,button')].filter(e=>{
        const r=e.getBoundingClientRect();return styled(e)&&r.width>0&&r.height>0&&!navigation.includes(e);
    }).map(e=>({tag:e.tagName,type:e.type,text:e.tagName==='BUTTON'?e.innerText:null,
        value:e.tagName==='BUTTON'?null:e.value,
        checked:e.tagName==='INPUT'&&['checkbox','radio'].includes(e.type)?e.checked:null,
        disabled:e.tagName==='BUTTON'&&e.innerText.trim()==='Save'?null:e.disabled}));
    const text=root.innerText;
    return {auth:false,modal:false,values,heading:text.split(String.fromCharCode(10)).find(line=>line.trim()),
        planet:['OBSERVATIONS','SPECTRUM','DOPPLER SHIFT','BRIGHTNESS DROP'].every(t=>text.toUpperCase().includes(t))};
}"""
)

# Unlike the generic center-point visibility helper, this rejects truncated
# tooltip text. Reading an off-chart day suffix would exceed visible evidence.
VISIBLE_TOOLTIP = (
    "e=>{"
    + RENDER_VISIBILITY_JS
    + """
    if(!styled(e)) return false;
    const r=e.getBoundingClientRect();
    let left=0,top=0,right=innerWidth,bottom=innerHeight;
    for(let p=e.parentElement;p;p=p.parentElement){
        const s=getComputedStyle(p),b=p.getBoundingClientRect();
        if(['hidden','clip','scroll','auto'].includes(s.overflowX)){left=Math.max(left,b.left);right=Math.min(right,b.right);}
        if(['hidden','clip','scroll','auto'].includes(s.overflowY)){top=Math.max(top,b.top);bottom=Math.min(bottom,b.bottom);}
    }
    if(r.left<left||r.right>right||r.top<top||r.bottom>bottom) return false;
    const walker=document.createTreeWalker(e,NodeFilter.SHOW_TEXT);
    let characters=0;
    while(walker.nextNode()){
        const n=walker.currentNode;
        for(let i=0;i<n.length;i++){
            if(!n.textContent[i].trim()) continue;
            characters++;
            const range=document.createRange();range.setStart(n,i);range.setEnd(n,i+1);
            const boxes=[...range.getClientRects()];
            if(!boxes.length||!boxes.every(b=>exposed(b,e,true))) return false;
        }
    }
    return characters>0||exposed(r,e);
    }"""
)

# Same full-glyph visibility predicate, one browser round trip. Reading only
# these painted tooltip strings avoids repeated Locator resolution per hover.
VISIBLE_FLUX_TOOLTIPS = (
    "chart=>{const visible=" + VISIBLE_TOOLTIP + ";"
    "return [...chart.querySelectorAll('text')].filter(e=>"
    "/^Brightness:/.test(e.textContent)&&visible(e)).map(e=>e.textContent);}"
)


# One atomic rendered-frame read; both outputs use full-glyph visibility.
VISIBLE_CHART_SAMPLE = (
    "chart=>{const visible="
    + VISIBLE_TOOLTIP
    + ";"
    + """
    const c=chart.getBoundingClientRect(), labels=[], tips=[];
    for(const e of chart.querySelectorAll('text')){
        const text=e.textContent,b=e.getBoundingClientRect();
        if(/^Brightness:/.test(text)&&visible(e))tips.push(text);
        if(Math.abs(c.width-280)<=1&&Math.abs(c.height-195)<=1&&
            /^-?[0-9]+(?:\\.[0-9]+)?$/.test(text)&&b.top-c.top>=150&&b.bottom-c.top<=176&&visible(e))
            labels.push({value:text,center_x:b.left+b.width/2-c.left});
    }
    return {tooltips:tips,labels,width:c.width};
    }"""
)

VISIBLE_FLUX_AXIS_LABELS = (
    "chart=>{const visible=" + VISIBLE_TOOLTIP + ";"
    "const c=chart.getBoundingClientRect(),labels=[];"
    "if(Math.abs(c.width-280)>1||Math.abs(c.height-195)>1)return null;"
    "for(const e of chart.querySelectorAll('text')){"
    "const b=e.getBoundingClientRect(),text=e.textContent;"
    "if(b.right-c.left<=29&&b.top-c.top>=5&&b.bottom-c.top<=153&&"
    "/^-?[0-9]+(?:\\.[0-9]+)?$/.test(text)&&visible(e))"
    "labels.push({value:text,center_y:b.top+b.height/2-c.top});}return labels;}"
)


class FluxChartSession:
    def __init__(self, page, config, emit, *, max_actions=1024, max_seconds=600, verify_day_axis=False):
        if not 1 <= max_actions <= 4096 or not 1 <= max_seconds <= 1800:
            raise ValueError("Chart action and time budgets must be bounded")
        self.page, self.config, self.emit = page, config, emit
        self.max_actions, self.max_seconds = max_actions, max_seconds
        self.verify_day_axis = verify_day_axis
        self.started, self.actions = time.monotonic(), 0
        self.stopped = self.unexpected_dialog = False
        self.frame = self.chart = None
        self.frames = []
        self.signature = None
        self.star = None
        page.on("dialog", self._dialog)
        try:
            report = inspect_page(page, config)
            if report["ignored_frame_urls"] or len(page.context.pages) != 1:
                raise BrowserSafetyStop("unexpected_chart_context")
            self.frames = [page.main_frame] + [
                f for f in page.frames if f != page.main_frame and _visible_frame(f, page.main_frame)
            ]
            frames = [f for f in self.frames if f.url == SIMULATION_URL]
            if len(frames) != 1:
                raise BrowserSafetyStop("ambiguous_simulation_frame")
            self.frame = frames[0]
            self.bodies = [f.locator("body").element_handle(timeout=2000) for f in self.frames]
            self.frame_urls = [f.url for f in self.frames]
            self.all_frames = page.frames.copy()
            self.embeddings = [f.frame_element() for f in self.frames if f != page.main_frame]
            self.hidden_embeddings = [f.frame_element() for f in self.all_frames if f not in self.frames]
            self.bundle_embeddings = all(
                f.parent_frame == page.main_frame for f in self.all_frames if f != page.main_frame
            )
            self.chart = self._find_chart()
            self.handle = self.chart.element_handle(timeout=2000)
            self.chart_parent = self.chart.locator("..").element_handle(timeout=2000)
            self.signature, self.star = self._values()
            self._guard()
            self.emit("observation", {"chart": {"source": "visible_tooltips", "star": self.star}})
        except BaseException:
            self.close()
            raise

    def _dialog(self, dialog):
        self.unexpected_dialog = True
        dialog.dismiss()

    def close(self):
        self.stopped = True
        self.page.remove_listener("dialog", self._dialog)

    def _find_chart(self):
        charts = [
            e
            for e in self.frame.get_by_role("img").all()
            if e.is_visible() and "Normalized Flux Days Observed" in e.aria_snapshot(timeout=2000)
        ]
        if len(charts) != 1:
            raise BrowserSafetyStop("ambiguous_flux_chart")
        return charts[0]

    def _values(self):
        states = []
        for f, body in zip(self.frames, self.bodies, strict=True):
            state = body.evaluate(
                NATIVE_VALUES,
                {"chart": self.handle, "parent": self.chart_parent}
                if f == self.frame
                else {"embeddings": self.embeddings, "hiddenEmbeddings": self.hidden_embeddings}
                if f == self.page.main_frame and self.bundle_embeddings
                else None,
            )
            # The main document is first. Reject newly hidden embeddings
            # before reading any child-frame controls, not after collecting
            # values from a frame that is no longer visible to the student.
            for flag, reason in (
                ("detached", "chart_document_body_replaced"),
                ("chart_detached", "chart_replaced"),
                ("chart_reparented", "chart_container_changed"),
                ("embedding_changed", "chart_frame_hidden_or_replaced"),
                ("hidden_embedding_visible", "previously_hidden_frame_became_visible"),
                ("auth", "authentication_required"),
                ("modal", "unexpected_modal"),
            ):
                if state.get(flag):
                    raise BrowserSafetyStop(reason)
            states.append(state)
        state = states[self.frames.index(self.frame)]
        if not state["planet"]:
            raise BrowserSafetyStop("not_planet_observations")
        heading = (state["heading"] or "").strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z -]{1,40}", heading):
            raise BrowserSafetyStop("ambiguous_visible_star_name")
        return [state["values"] for state in states], heading.upper()

    def _guard(self):
        if self.stopped:
            raise BrowserSafetyStop("chart_session_stopped")
        try:
            self.page.wait_for_timeout(0)
            if time.monotonic() - self.started > self.max_seconds:
                raise BrowserSafetyStop("chart_time_limit")
            if self.unexpected_dialog or len(self.page.context.pages) != 1:
                raise BrowserSafetyStop("unexpected_chart_context")
            if not self.config.allows(self.page.url):
                raise BrowserSafetyStop("navigation_outside_activity")
            if self.page.frames != self.all_frames or any(
                _url_identity(f.url, preview=f == self.page.main_frame)
                != _url_identity(url, preview=f == self.page.main_frame)
                for f, url in zip(self.frames, self.frame_urls, strict=True)
            ):
                raise BrowserSafetyStop("chart_frame_changed")
            if not self.bundle_embeddings:
                # Retain the generic path for nested frames. Cross-document
                # handles cannot be passed together to a main-frame read.
                for element in self.embeddings:
                    if not element.evaluate(
                        "e=>{"
                        + RENDER_VISIBILITY_JS
                        + "const r=e.getBoundingClientRect();return e.isConnected&&styled(e)&&r.width>0&&r.height>0;}"
                    ):
                        raise BrowserSafetyStop("chart_frame_hidden_or_replaced")
                if any(element.is_visible() for element in self.hidden_embeddings):
                    raise BrowserSafetyStop("previously_hidden_frame_became_visible")
            # Identity, parent and native values are read atomically in the
            # same frame evaluation. Both pre/post-action guards still run;
            # no observation or action is sampled less frequently.
            if self._values() != (self.signature, self.star):
                raise BrowserSafetyStop("chart_context_or_answers_changed")
        except BaseException:
            self.stopped = True
            raise

    def _point(self, x, y):
        if type(x) not in {int, float} or type(y) not in {int, float} or not 0 < x < 1 or not 0 < y < 1:
            raise ValueError("Chart positions are interior fractions")
        box = self.handle.bounding_box()
        if not box or not 100 <= box["width"] <= 1600 or not 100 <= box["height"] <= 1200:
            raise BrowserSafetyStop("unsupported_chart_geometry")
        exposed = self.handle.evaluate(
            """(e,p)=> {"""
            + RENDER_VISIBILITY_JS
            + """
            const r=e.getBoundingClientRect(), x=r.x+r.width*p[0],y=r.y+r.height*p[1];
            const hit=document.elementFromPoint(x,y);
            return styled(e)&&hit&&(hit===e||e.contains(hit))&&styled(hit);
            }""",
            [x, y],
        )
        if not exposed:
            raise BrowserSafetyStop("chart_point_occluded")
        return box["x"] + box["width"] * x, box["y"] + box["height"] * y

    def _begin(self, kind, payload):
        self._guard()
        if self.actions >= self.max_actions:
            self.stopped = True
            raise BrowserSafetyStop("chart_action_limit")
        self.actions += 1
        self.emit("action_proposed", {"kind": kind, "surface": "chart", "sequence": self.actions, **payload})

    def flux_axis_labels(self):
        """Fully exposed numeric y-axis glyphs and their rendered positions only.

        No plot paths, bound data, application arrays or hidden attributes. The
        verified chart's left gutter ends at x=29; x-axis labels sit below y=153.
        """
        self._guard()
        labels = self.handle.evaluate(VISIBLE_FLUX_AXIS_LABELS)
        if labels is None:
            raise BrowserSafetyStop("unsupported_flux_axis_geometry")
        self._guard()
        return labels

    def hover(self, x, y):
        return self._hover(x, y, allow_unexposed=False)

    def hover_day(self, x, y, requested_day):
        """Require the chosen integer day, not merely a nearby visible sample."""
        if not self.verify_day_axis or type(requested_day) is not int or not 0 <= requested_day <= 5000:
            raise ValueError("An axis-checked day in the observed window is required")
        return self._hover(x, y, allow_unexposed=False, requested_day=requested_day)

    def probe_hover(self, x, y):
        """Read-only exposure probe; None is missing evidence, never a flux value.

        Context, action, glyph and day-axis guards remain identical. Only a
        zero-tooltip result is recoverable for caller-selected viewport pans;
        ambiguous, stale or malformed readings still stop the session.
        """
        return self._hover(x, y, allow_unexposed=True)

    def _hover(self, x, y, *, allow_unexposed, requested_day=None):
        intent = {"x_fraction": x, "y_fraction": y}
        if requested_day is not None:
            intent["requested_day"] = requested_day
        self._begin("HOVER", intent)
        try:
            self.page.mouse.move(*self._point(x, y))
            self._guard()
            # Read only the SVG's exposed tooltip, not a hidden or page-wide
            # duplicate. No data-bound properties, path geometry, or attributes.
            reading = self.handle.evaluate(VISIBLE_CHART_SAMPLE) if self.verify_day_axis else None
            matches = reading["tooltips"] if reading else self.handle.evaluate(VISIBLE_FLUX_TOOLTIPS)
            if not matches and allow_unexposed:
                self.emit(
                    "action_result",
                    {
                        "sequence": self.actions,
                        "chart_readout_unavailable": True,
                        "reason": "tooltip_not_fully_exposed",
                        "task_completed": False,
                    },
                )
                return None
            if len(matches) != 1:
                raise BrowserSafetyStop("missing_or_ambiguous_visible_tooltip")
            sample = parse_flux_tooltip(matches[0])
            if self.verify_day_axis:
                verify_tooltip_day(reading["labels"], x * reading["width"], sample.day)
            if requested_day is not None and sample.day != requested_day:
                raise BrowserSafetyStop("tooltip_day_disagrees_with_requested_day")
            self.emit(
                "action_result",
                {"sequence": self.actions, "chart_sample": sample.model_dump(), "task_completed": False},
            )
            return sample
        except BaseException:
            self.stopped = True
            raise

    def time_axis_labels(self):
        """Only painted x-axis tick glyphs/positions, never scale/data properties."""
        self._guard()
        return self.handle.evaluate(VISIBLE_CHART_SAMPLE)["labels"]

    def clear_pointer(self):
        """Ordinary mouse leave so a hover disk cannot hide a raster feature.

        Only the exposed, noninteractive point immediately below this chart is
        eligible. This is an explicit budgeted pointer action, never a click or
        a programmatic deletion of tooltip DOM.
        """
        self._begin("HOVER", {"purpose": "remove_pointer_overlay", "outside_chart": True})
        try:
            offset = self.handle.evaluate(
                "e=>{"
                + RENDER_VISIBILITY_JS
                + """
                const r=e.getBoundingClientRect(),x=r.left+r.width/2,y=r.bottom+5;
                const hit=document.elementFromPoint(x,y);
                if(!hit||!styled(hit)||e.contains(hit)||x<0||y<0||x>=innerWidth||y>=innerHeight
                   ||hit.closest('button,input,select,textarea,a,[role=button],[role=link]'))return null;
                return [r.width/2,r.height+5];}
                """
            )
            box = self.handle.bounding_box()
            if offset is None or box is None:
                raise BrowserSafetyStop("chart_pointer_exit_not_exposed")
            self.page.mouse.move(box["x"] + offset[0], box["y"] + offset[1])
            self._guard()
            if self.handle.evaluate(VISIBLE_FLUX_TOOLTIPS):
                raise BrowserSafetyStop("chart_pointer_overlay_remains")
            self.emit(
                "action_result",
                {"sequence": self.actions, "pointer_outside_chart": True, "task_completed": False},
            )
        except BaseException:
            self.stopped = True
            raise

    def zoom(self, x, y, delta_y):
        if type(delta_y) not in {int, float} or not 1 <= abs(delta_y) <= 1500:
            raise ValueError("Bounded nonzero wheel delta required")
        self._begin("SCROLL", {"x_fraction": x, "y_fraction": y, "delta_y": delta_y})
        try:
            self.page.mouse.move(*self._point(x, y))
            self.page.mouse.wheel(0, delta_y)
            self.page.wait_for_timeout(250)  # Bounded ordinary zoom transition.
            self._guard()
            self.emit(
                "action_result",
                {
                    "sequence": self.actions,
                    "chart_accessibility": self.chart.aria_snapshot(),
                    "task_completed": False,
                },
            )
        except BaseException:
            self.stopped = True
            raise

    def pan(self, start, end):
        self._begin("DRAG", {"start_fraction": start, "end_fraction": end})
        try:
            a, b = self._point(*start), self._point(*end)
            self.page.mouse.move(*a)
            self.page.mouse.down()
            try:
                self.page.mouse.move(*b, steps=10)
            finally:
                self.page.mouse.up()
            self.page.wait_for_timeout(250)
            self._guard()
            self.emit(
                "action_result",
                {
                    "sequence": self.actions,
                    "chart_accessibility": self.chart.aria_snapshot(),
                    "task_completed": False,
                },
            )
        except BaseException:
            self.stopped = True
            raise


def verify_tooltip_day(labels, x, day):
    """Reject stale tooltips inconsistent with the displayed linear time axis."""
    if len(labels) < 3:
        raise BrowserSafetyStop("insufficient_visible_time_axis_labels")
    try:
        points = sorted((float(row["center_x"]), float(row["value"])) for row in labels)
        dx = points[-1][0] - points[0][0]
        slope = (points[-1][1] - points[0][1]) / dx
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise BrowserSafetyStop("invalid_visible_time_axis") from None
    if (
        dx <= 0
        or not math.isfinite(slope)
        or slope <= 0
        or any(not math.isfinite(v) for point in points for v in point)
    ):
        raise BrowserSafetyStop("invalid_visible_time_axis")
    estimate = lambda pixel: points[0][1] + (pixel - points[0][0]) * slope
    # Glyph-center and native mouse coordinates can differ by a subpixel.
    tolerance = max(1.1, slope * 1.1)
    if any(abs(estimate(pixel) - value) > tolerance for pixel, value in points):
        raise BrowserSafetyStop("nonlinear_visible_time_axis")
    # Zoomed axes can omit the first tick (e.g. first visible tick x=43.5,
    # first sampled pixel x=31). Permit at most one tick interval of linear
    # extrapolation, still strictly inside the observed 30..260 plot region.
    left_gap, right_gap = points[1][0] - points[0][0], points[-1][0] - points[-2][0]
    if (
        not 30 <= x <= 260
        or left_gap <= 0
        or right_gap <= 0
        or not points[0][0] - left_gap <= x <= points[-1][0] + right_gap
        or abs(estimate(x) - day) > tolerance
    ):
        raise BrowserSafetyStop("tooltip_day_disagrees_with_visible_axis")
