"""Opt-in, disposable presentation. Never an observation, policy or evidence source.

The canvas lives in a closed shadow root outside the page body, has no text or
controls, and cannot receive input. Every evidence screenshot uses the separate
presentation_capture helper to temporarily hide its host without painting over
any science pixels. Only known runtime captions and native pointer geometry go
to this renderer; no page text, input values, model reasoning or credentials do.
"""

import base64
import json
import sys
import uuid
from pathlib import Path

from .mascot_status import MascotStatusReducer

ASSETS = Path(__file__).with_name("assets")
HOST_ID = "habfly-mascot-overlay-v1"


class BrowserMascot:
    def __init__(self):
        self.status = MascotStatusReducer()
        self.channel = "habfly-presentation-" + uuid.uuid4().hex
        self.page = None
        self.disabled = False
        self._frames = set()
        self._navigation_listener = None
        self._asset = "data:image/png;base64," + base64.b64encode(
            (ASSETS / "sporky-fly-v1.png").read_bytes()
        ).decode("ascii")
        self._script = (ASSETS / "mascot.js").read_text()
        self._pointer_script = (ASSETS / "mascot-pointer.js").read_text()

    def consume(self, event, *, page=None, allowed_frame_urls=()):
        """Best-effort display only; never change/retry/authorize a task action."""
        if self.disabled:
            return
        try:
            display = self.status.consume(event)
            if page is None:
                return
            if self.page is not page:
                self.close()
                self.page = page
                self._navigation_listener = lambda changed: self._frames.difference_update(
                    [identity for identity in self._frames if identity[0] == changed]
                )
                page.on("framenavigated", self._navigation_listener)
            installed = page.evaluate(
                "a => { const v=window[Symbol.for(a.channel)]; if(!v)return false; "
                "v.update(a.display); return true; }",
                {"channel": self.channel, "display": display},
            )
            if not installed:
                self._frames.clear()
                page.evaluate(
                    self._script,
                    {"host": HOST_ID, "asset": self._asset, "channel": self.channel, "display": display},
                )
            for frame in page.frames:
                identity = (frame, frame.url)
                if identity not in self._frames and (
                    frame == page.main_frame or frame.url in allowed_frame_urls
                ):
                    frame.evaluate(self._pointer_script, self.channel)
                    self._frames.add(identity)
        except Exception:  # noqa: BLE001 - never expose page/driver/credential errors
            self.disabled = True
            self.close()
            print("HabFly mascot display unavailable; task safety controls are unchanged.", file=sys.stderr)

    def close(self):
        for frame in {item[0] for item in self._frames}:
            try:
                frame.evaluate(
                    "key => { const tracker=window[Symbol.for(key+'-pointer')]; "
                    "if(tracker)tracker.remove(); }",
                    self.channel,
                )
            except Exception:  # noqa: BLE001, S110 - detached frames are already disposed
                pass
        self._frames.clear()
        if self.page is not None:
            try:
                if self._navigation_listener is not None:
                    self.page.remove_listener("framenavigated", self._navigation_listener)
                self.page.evaluate(
                    "key => { const view = window[Symbol.for(key)]; if(view) view.remove(); }",
                    self.channel,
                )
            except Exception:  # noqa: BLE001, S110 - driver text is private; cleanup is best-effort
                pass
        self.page = None
        self._navigation_listener = None


def preview_html():
    """Self-contained, offline design preview; deliberately not a live run."""
    mascot = BrowserMascot()
    script = mascot._script
    pointer = mascot._pointer_script
    arguments = json.dumps(
        {
            "host": HOST_ID,
            "asset": mascot._asset,
            "channel": mascot.channel,
            "display": {"caption": "Ready for a demo", "activity": "idle", "source_label": "Demo only"},
        }
    )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HabFly · Sporky flight deck</title><style>
body{margin:0;background:#121a25;color:#edf1f6;font:17px system-ui;min-height:100vh}
main{max-width:880px;margin:0 auto;padding:70px 35px}small{color:#f5c55a;letter-spacing:.12em}
h1{font-size:48px;line-height:1.1;max-width:650px;margin:20px 0}p{color:#b7c5d6;max-width:600px;line-height:1.7}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:38px 0}
button{padding:24px;text-align:left;color:#edf1f6;background:#202e40;border:1px solid #3a4b60;
border-radius:16px;font:inherit;cursor:pointer}button:hover,button:focus-visible{border-color:#f5c55a}
footer{border-top:1px solid #3a4b60;padding-top:25px;font-size:14px;color:#99aac0}
</style><main><small>HABFLY / PRESENTATION PREVIEW</small><h1>A little fly.<br>A clearer view of the work.</h1>
<p>Move your pointer, then try a status below. Sporky follows the pointer without intercepting clicks.
These are demonstration captions, not a running model or hidden thoughts.</p><div class="cards">
<button data-caption="Thinking…" data-activity="thinking">Thinking</button>
<button data-caption="Selected distance" data-activity="selecting">Select distance</button>
<button data-caption="Calculating distance" data-activity="calculating">Calculate distance</button>
<button data-caption="Copying distance" data-activity="copying">Copy a result</button>
<button data-caption="Watching the lightcurve" data-activity="observing">Observe the chart</button>
<button data-caption="Paused" data-activity="paused">Pause</button>
<button data-caption="Stopped — check the TUI" data-activity="stopped">Safety stop</button>
</div><footer>Offline design preview · No HabWorlds connection · No training · No answers or scoring</footer></main>
<script>const install=("""
        + script
        + "); const config="
        + arguments
        + "; install(config);("
        + pointer
        + """
)(config.channel);document.querySelectorAll('button').forEach(button=>button.onclick=()=>{
config.display={caption:button.dataset.caption,activity:button.dataset.activity,source_label:'Demo only'};
install(config);});</script></html>"""
    )
