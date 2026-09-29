"""Native submission controls, including the observed Janus readiness label.

Only the fixed readiness checkbox may use its visibly painted associated label.
The native input remains the checked-state and identity source. These bindings do
not authorize submission, change state, or interpret any outcome.
"""

from .browser import BrowserSafetyStop
from .browser_setup import RENDER_VISIBILITY_JS

READY = "I am ready to submit project."
_IDENTITY = "(a,b)=>a.isConnected&&a===b"

# A submission-only variant: do not relax the legacy exposure predicate used by
# other controls (including the collected-list pager).
SUBMIT_EXPOSED = (
    "(e)=>{"
    + RENDER_VISIBILITY_JS
    + """
  if(e.tagName!=='BUTTON'||!e.isConnected||!styled(e))return null;
  const r=e.getBoundingClientRect(),s=getComputedStyle(e);
  if(r.width<=0||r.height<=0||r.left<0||r.top<0||r.right>innerWidth||r.bottom>innerHeight)return null;
  for(let p=e.parentElement;p;p=p.parentElement){const t=getComputedStyle(p),b=p.getBoundingClientRect();
    if((['hidden','clip','scroll','auto'].includes(t.overflowX)&&(r.left<b.left||r.right>b.right))||
       (['hidden','clip','scroll','auto'].includes(t.overflowY)&&(r.top<b.top||r.bottom>b.bottom)))return null;
  }
  const radii=[s.borderTopLeftRadius,s.borderTopRightRadius,s.borderBottomLeftRadius,s.borderBottomRightRadius];
  if(!radii.every(v=>/^\\d+(?:\\.\\d+)?px(?: \\d+(?:\\.\\d+)?px)?$/.test(v)))return null;
  const radius=Math.max(...radii.flatMap(v=>v.split(' ').map(parseFloat)));
  const inset=Math.max(2,radius+1);
  if(2*inset>=r.width||2*inset>=r.height)return null;
  for(const x of [r.left+inset,r.x+r.width/2,r.right-inset])
    for(const y of [r.top+inset,r.y+r.height/2,r.bottom-inset]){
      const h=document.elementFromPoint(x,y);if(!(h===e||e.contains(h)))return null;
    }
  return {x:r.x,y:r.y,width:r.width,height:r.height,hit_inset:inset};
}"""
)

_LABEL = """(e,ready)=>{
  if(e.tagName!=='INPUT'||e.type!=='checkbox'||getComputedStyle(e).opacity!=='0')return null;
  const labels=Array.from(e.labels||[]);
  return labels.length===1&&labels[0].control===e&&labels[0].innerText.trim()===ready?labels[0]:null;
}"""

_LABEL_PROOF = (
    "(label,a)=>{"
    + RENDER_VISIBILITY_JS
    + """
  const e=a.input,t=a.text,ready=a.ready;
  const norm=v=>v.replace(/\\s+/g,' ').trim();
  if(!label.isConnected||!e.isConnected||!t.isConnected||label.tagName!=='LABEL'||
     e.tagName!=='INPUT'||e.type!=='checkbox'||e.disabled||e.checked!==a.checked||
     label.control!==e||e.labels.length!==1||e.labels[0]!==label||
     !label.contains(t)||norm(label.innerText)!==ready||norm(t.innerText)!==ready||
     getComputedStyle(e).opacity!=='0'||!styled(label)||!styled(t))return null;
  if(e.closest('[hidden],[aria-hidden="true"]'))return null;
  if(Array.from(label.querySelectorAll('input,button,select,textarea,a[href],[role="button"],[role="checkbox"],[contenteditable]')).some(n=>n!==e))return null;
  const es=getComputedStyle(e);
  if(es.display==='none'||es.visibility!=='visible'||es.pointerEvents==='none')return null;
  for(let p=e.parentElement;p;p=p.parentElement)if(!styled(p))return null;
  const r=label.getBoundingClientRect(),ir=e.getBoundingClientRect(),tr=t.getBoundingClientRect();
  const inside=(b,o)=>b.width>0&&b.height>0&&b.left>=o.left&&b.top>=o.top&&b.right<=o.right&&b.bottom<=o.bottom;
  const viewport={left:0,top:0,right:innerWidth,bottom:innerHeight};
  if(!inside(r,viewport)||!inside(ir,r)||!inside(tr,r)||r.width<8||r.height<8)return null;
  for(const node of [label,e,t]){
    const b=node.getBoundingClientRect();
    for(let p=node.parentElement;p;p=p.parentElement){const s=getComputedStyle(p),c=p.getBoundingClientRect();
      if((['hidden','clip','scroll','auto'].includes(s.overflowX)&&(b.left<c.left||b.right>c.right))||
         (['hidden','clip','scroll','auto'].includes(s.overflowY)&&(b.top<c.top||b.bottom>c.bottom)))return null;
    }
  }
  // Janus paints the checkbox square with the associated label's ::before.
  const p=getComputedStyle(label,'::before');
  const dimension=v=>/^\\d+(?:\\.\\d+)?px$/.test(v)?parseFloat(v):NaN;
  const w=dimension(p.width),h=dimension(p.height);
  const color=p.backgroundColor.match(/^rgba?\\(([^)]+)\\)$/);
  const components=color?color[1].split(',').map(v=>Number(v.trim())):[];
  const painted=components.length===3||components.length===4&&components[3]>0;
  if(!['""',"''"].includes(p.content)||p.display==='none'||p.visibility!=='visible'||
     Number(p.opacity)<=0||!(w>=8&&w<=32&&h>=8&&h<=32)||!painted)return null;
  const allowed=h=>h===label||h===t||
    (h===e&&label.control===h)||
    (h&&label.contains(h)&&h!==e&&styled(h));
  for(const b of [r,tr,ir]){
    const ix=Math.min(2,b.width/4),iy=Math.min(2,b.height/4);
    for(const x of [b.left+ix,b.x+b.width/2,b.right-ix])
      for(const y of [b.top+iy,b.y+b.height/2,b.bottom-iy])
        if(!allowed(document.elementFromPoint(x,y)))return null;
  }
  return {x:r.x,y:r.y,width:r.width,height:r.height};
}"""
)


def outer_submit_box(box):
    """Translate the existing outer-frame .5px grid to this button's safe grid."""
    inset = box.get("hit_inset")
    if inset is None:  # Legacy injected controls do not carry browser geometry.
        return box
    return {
        "x": box["x"] + inset - 0.5,
        "y": box["y"] + inset - 0.5,
        "width": box["width"] - 2 * inset + 1,
        "height": box["height"] - 2 * inset + 1,
    }


def submit_outer_exposed(frame, boxes):
    """Same inset grid through the current, unclipped, painted outer iframe."""
    return frame.frame_element().evaluate(
        "(e,boxes)=>{"
        + RENDER_VISIBILITY_JS
        + """
      if(e.tagName!=='IFRAME'||!e.isConnected||!styled(e))return false;
      const b=e.getBoundingClientRect();
      if(b.width<=0||b.height<=0||b.left<0||b.top<0||b.right>innerWidth||b.bottom>innerHeight||
         b.width!==e.offsetWidth||b.height!==e.offsetHeight)return false;
      for(let p=e.parentElement;p;p=p.parentElement){const s=getComputedStyle(p),r=p.getBoundingClientRect();
        if((['hidden','clip','scroll','auto'].includes(s.overflowX)&&(b.left<r.left||b.right>r.right))||
           (['hidden','clip','scroll','auto'].includes(s.overflowY)&&(b.top<r.top||b.bottom>r.bottom)))return false;
      }
      return boxes.every(r=>[r.x+.5,r.x+r.width/2,r.x+r.width-.5].every(x=>
        [r.y+.5,r.y+r.height/2,r.y+r.height-.5].every(y=>{
          const px=x+b.x+e.clientLeft,py=y+b.y+e.clientTop;
          return px>=0&&py>=0&&px<innerWidth&&py<innerHeight&&document.elementFromPoint(px,py)===e;
        })));
    }""",
        boxes,
    )


class ReadinessBinding:
    """Element-handle-compatible pair; identity includes input AND click target."""

    def __init__(
        self, locator, native, target, box, *, mode, checked, prefix, exposed, text=None, text_locator=None
    ):
        self.locator, self.native, self.target, self.box = locator, native, target, box
        self.mode, self.checked, self.prefix = mode, checked, prefix
        self.exposed, self.text = exposed, text
        self.text_locator = text_locator

    def evaluate(self, script, other):
        if script != _IDENTITY or not isinstance(other, ReadinessBinding):
            return False
        return (
            self.mode == other.mode
            and self.checked is other.checked
            and self.native.evaluate(_IDENTITY, other.native)
            and self.target.evaluate(_IDENTITY, other.target)
            and (self.text is None or other.text is not None and self.text.evaluate(_IDENTITY, other.text))
        )

    def click(self, *, timeout):
        # The owner has already reserved its one dispatch before invoking this.
        # A failed final read stays uncertain; it never triggers a retry.
        if (
            self.locator.count() != 1
            or not self.locator.is_visible()
            or not self.locator.is_enabled()
            or self.locator.is_checked() is not self.checked
            or not self.native.evaluate(_IDENTITY, self.locator.element_handle(timeout=timeout))
        ):
            raise BrowserSafetyStop(self.prefix + "native_binding_changed")
        if self.mode == "native_input":
            box = self.native.evaluate(self.exposed, "checkbox")
        else:
            if (
                self.text_locator.count() != 1
                or not self.text_locator.is_visible()
                or not self.text.evaluate(_IDENTITY, self.text_locator.element_handle(timeout=timeout))
            ):
                raise BrowserSafetyStop(self.prefix + "native_binding_changed")
            box = self.target.evaluate(
                _LABEL_PROOF,
                {"input": self.native, "text": self.text, "ready": READY, "checked": self.checked},
            )
        if box != self.box:
            raise BrowserSafetyStop(self.prefix + "readiness_control_unexposed")
        self.target.click(timeout=timeout)


def bind_readiness(page, checkbox, *, checked, exposed, prefix, timeout=2000):
    """Read-only binding; callers retain their exact native checked/AX checks."""
    if (
        checkbox.count() != 1
        or not checkbox.is_visible()
        or not checkbox.is_enabled()
        or checkbox.is_checked() is not checked
    ):
        raise BrowserSafetyStop(prefix + "readiness_control_unexposed")
    box = checkbox.evaluate(exposed, "checkbox")
    native = checkbox.element_handle(timeout=timeout)
    if isinstance(box, dict) and native is not None:
        return ReadinessBinding(
            checkbox,
            native,
            native,
            box,
            mode="native_input",
            checked=checked,
            prefix=prefix,
            exposed=exposed,
        )
    try:
        label = checkbox.evaluate_handle(_LABEL, READY).as_element()
        text = page.get_by_text(READY, exact=True)
        if label is None or text.count() != 1 or not text.is_visible():
            raise ValueError("unavailable")
        text_handle = text.element_handle(timeout=timeout)
        proof = label.evaluate(
            _LABEL_PROOF,
            {"input": native, "text": text_handle, "ready": READY, "checked": checked},
        )
        if not isinstance(proof, dict):
            raise TypeError("unexposed")
    except Exception:  # noqa: BLE001 - driver errors must not expose page or private exception text.
        raise BrowserSafetyStop(prefix + "readiness_control_unexposed") from None
    return ReadinessBinding(
        checkbox,
        native,
        label,
        proof,
        mode="associated_painted_label",
        checked=checked,
        prefix=prefix,
        exposed=exposed,
        text=text_handle,
        text_locator=text,
    )
