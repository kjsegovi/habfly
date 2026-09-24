"""Deterministic setup against disposable, fully intercepted browser fixtures."""

import io
import json
import time
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from playwright.sync_api import Error as PlaywrightError
from test_browser_numeric import OUTER, WIDGET, chromium, config, stellar_html  # noqa: F401

from habfly.browser_setup import (
    BrowserSetup,
    SetupStop,
    consume_credentials,
    rendered_text,
    visible_star_link,
    visible_star_point,
)
from habfly.browser_stellar import SIMULATION_URL
from habfly.runtime import RunOptions

LOGIN = "http://localhost/authors/log_in"
EMAIL, PASSWORD = "fixture-user@example.invalid", "fixture-only-secret"


def png(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_pixels_are_deterministic_and_exclude_header_and_noise():
    image = Image.new("RGB", (800, 500), "#081020")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 20, 105, 25), fill="white")
    draw.point((150, 180), fill="white")
    draw.rectangle((200, 220, 205, 225), fill="white")
    draw.rectangle((700, 400, 705, 405), fill="white")
    assert visible_star_point(png(image)) == {"x": 202.5, "y": 222.5, "width": 800, "height": 500}
    with pytest.raises(SetupStop, match="setup_no_visible_star_candidate"):
        visible_star_point(png(Image.new("RGB", (800, 500))))
    with pytest.raises(SetupStop, match="setup_no_visible_star_candidate"):
        visible_star_point(png(Image.new("RGB", (800, 500), "white")))
    with pytest.raises(SetupStop, match="setup_unsupported_starfield_size"):
        visible_star_point(png(Image.new("RGB", (20, 20))))
    assert visible_star_link(png(image), visible_star_point(png(image))) is None


def test_css_scale_dim_two_pixel_stars_require_local_contrast():
    image = Image.new("RGB", (800, 500), (8, 15, 26))
    draw = ImageDraw.Draw(image)
    draw.rectangle((200, 220, 201, 220), fill=(139, 135, 119))
    draw.rectangle((300, 240, 350, 270), fill=(120, 120, 120))  # diffuse patch, not a star
    draw.rectangle((400, 300, 411, 311), fill=(90, 90, 90))
    draw.rectangle((405, 305, 406, 305), fill=(120, 120, 120))  # insufficient contrast
    draw.point((300, 280), fill="white")  # isolated noise
    assert visible_star_point(png(image)) == {"x": 200.5, "y": 220.0, "width": 800, "height": 500}


def test_credentials_consumed_without_exception_values(monkeypatch):
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", PASSWORD)
    assert consume_credentials() == (EMAIL, PASSWORD)
    with pytest.raises(SetupStop, match="^setup_credentials_missing$"):
        consume_credentials()


def test_automatic_setup_rejected_for_other_tasks():
    from habfly.runtime import Runtime

    with pytest.raises(ValueError, match="only for the supervised browser"):
        Runtime(output=io.StringIO()).start({"browser_setup": "automatic"})


def canvas_html(*, semantic=False, blank=False, no_response=False, detail=None):
    detail = stellar_html() if detail is None else detail
    return (
        """
    <body style='margin:0;background:#081020;color:white'>
    <div>FUNDING $50000 DATA QUALITY 0.0%</div>
    <canvas width=800 height=450></canvas>
    <button onclick='document.body.replaceChildren()'>Save</button>
    <script>
    const canvas=document.querySelector('canvas'), ctx=canvas.getContext('2d');
    ctx.fillStyle='#081020';ctx.fillRect(0,0,800,450);
    ctx.fillStyle='white';
    """
        + ("" if blank else "ctx.fillRect(200,200,5,5);")
        + """
    let selected=false;
    canvas.onclick=e=>{
      const x=e.offsetX, y=e.offsetY;
      if (!selected && x>=195 && x<=210 && y>=195 && y<=210) {
        selected=true;
    """
        + (
            "return;"
            if no_response
            else (
                "const b=document.createElement('button');b.textContent='VIEW STAR DATA';"
                + f"b.onclick=()=>document.body.innerHTML={json.dumps(detail)};document.body.append(b);"
                if semantic
                else """
        ctx.fillStyle='black';ctx.fillRect(198,110,240,75);
        ctx.fillStyle='white';ctx.font='16px sans-serif';ctx.fillText('FIXTURE STAR',210,130);
        ctx.fillStyle='#42ffff';ctx.fillText('VIEW STAR DATA',210,168);
        """
            )
        )
        + f"""
      }} else if (selected && x>=210 && x<=390 && y>=150 && y<=175) {{
        document.body.innerHTML={json.dumps(detail)};
      }}
    }};
    </script></body>
    """
    )


@pytest.fixture
def setup_page(chromium, monkeypatch):  # noqa: F811
    # Keep normal fixture transitions fast; the polling test below explicitly
    # restores the production interval and advances a module-local fake clock.
    monkeypatch.setattr(BrowserSetup, "STARFIELD_POLL_SECONDS", 0)
    context = chromium.new_context()
    state = {
        "authenticated": False,
        "posts": 0,
        "bad_login": False,
        "canvas": canvas_html(),
        "login_extra": "",
    }
    simulation = (
        f'<iframe style="width:800px;height:500px;border:0" src="{SIMULATION_URL}"></iframe>'
        f'<iframe src="{WIDGET}"></iframe><iframe src="{WIDGET}"></iframe>'
    )
    screens = [
        f"<p>{text}</p><button onclick='next()'>{label}</button>" for text, label in BrowserSetup.SCREENS
    ]
    intro = (
        "<body><main></main><script>const screens="
        + json.dumps(screens + [simulation])
        + ";let i=0;function next(){document.querySelector('main').innerHTML=screens[++i]};document.querySelector('main').innerHTML=screens[0];</script></body>"
    )

    def route(request_route):
        request = request_route.request
        if request.url == OUTER:
            if not state["authenticated"]:
                request_route.fulfill(
                    content_type="text/html", body=f"<script>location.replace({json.dumps(LOGIN)})</script>"
                )
            else:
                request_route.fulfill(content_type="text/html", body=intro)
        elif request.url == LOGIN:
            if request.method == "POST":
                state["posts"] += 1
                if not state["bad_login"]:
                    state["authenticated"] = True
                    request_route.fulfill(
                        content_type="text/html",
                        body=f"<script>location.replace({json.dumps(OUTER)})</script>",
                    )
                    return
            request_route.fulfill(
                content_type="text/html",
                body=f"""
            <section><h2>We use cookies</h2><button onclick='this.parentNode.remove()'>Close</button>
              <button onclick='document.body.replaceChildren()'>Accept all</button></section>
            <form action='{LOGIN}' method='post'>
              <input type=email name=email placeholder=Email>
              <input type=password name=password placeholder=Password>
              <button>Sign in</button></form>{state["login_extra"]}""",
            )
        elif request.url == SIMULATION_URL:
            request_route.fulfill(content_type="text/html", body=state["canvas"])
        elif request.url == WIDGET:
            request_route.fulfill(
                content_type="text/html", body="<button>Update Score</button><button>Submit Project</button>"
            )
        else:
            request_route.abort()

    context.route("**/*", route)
    page = context.new_page()
    yield page, state
    context.close()


def begin(page, tmp_path, events=None):
    tmp_path.mkdir(exist_ok=True)
    setup = BrowserSetup(
        page,
        config(),
        (EMAIL, PASSWORD),
        emit=lambda *e: events.append(e) if events is not None else None,
        output=tmp_path,
    )
    page.goto(OUTER)
    return setup


def finish(setup):
    for _ in range(30):
        if setup.advance() == "stellar":
            return
        setup.page.wait_for_timeout(20)
    pytest.fail("Fixture did not reach stellar screen")


@pytest.mark.parametrize("semantic", [False, True])
def test_login_intro_star_stellar_no_answers_or_protected_actions(setup_page, tmp_path, semantic):
    page, state = setup_page
    state["canvas"] = canvas_html(semantic=semantic)
    events = []
    setup = begin(page, tmp_path, events)
    finish(setup)
    assert setup.closed and setup.stage == "stellar_screen_ready"
    assert setup.visited == {0, 1, 2, 3} and state["posts"] == 1
    assert setup._email == setup._password == ""
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    assert all(
        frame.locator(f"#{name}").input_value() == "" for name in ("distance", "luminosity", "temperature")
    )
    assert "Your Reconstruction" in frame.locator("body").inner_text()
    assert PASSWORD not in json.dumps(events) and EMAIL not in json.dumps(events)
    assert {p.name for p in tmp_path.iterdir()} == (
        {"setup-starfield.png"} if semantic else {"setup-starfield.png", "setup-star-link.png"}
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "document.querySelector('form').action='https://fixture.invalid/steal'",
        "document.querySelector('form button').setAttribute('formaction','https://fixture.invalid/steal')",
        "document.querySelector('input[type=email]').oninput=()=>document.querySelector('form').action='https://fixture.invalid/steal'",
    ],
)
def test_reject_untrusted_or_changed_form_before_password(setup_page, tmp_path, mutation):
    page, state = setup_page
    state["login_extra"] = f"<script>{mutation}</script>"
    setup = begin(page, tmp_path)
    setup.advance()  # cookie dismissal
    with pytest.raises(SetupStop, match="setup_untrusted_login_form"):
        setup.advance()
    assert state["posts"] == 0 and page.get_by_placeholder("Password").input_value() == ""
    assert setup.closed and setup._password == ""
    assert not list(tmp_path.iterdir())


def test_wrong_login_only_submits_once_and_times_out(setup_page, tmp_path):
    page, state = setup_page
    state["bad_login"] = True
    setup = begin(page, tmp_path)
    for _ in range(5):
        setup.advance()
    assert state["posts"] == 1
    setup.started = time.monotonic() - setup.MAX_SECONDS - 1
    with pytest.raises(SetupStop, match="setup_time_limit"):
        setup.advance()
    assert setup._password == "" and not list(tmp_path.iterdir())


def test_named_cookie_preferences_closes_without_saving(setup_page, tmp_path):
    page, state = setup_page
    state["login_extra"] = """<div role=dialog aria-label='Cookie Preferences'>
      <button onclick='this.parentNode.remove()'>Close</button>
      <button onclick='document.body.replaceChildren()'>Save my preferences</button></div>"""
    setup = begin(page, tmp_path)
    try:
        setup.advance()
        assert page.get_by_role("dialog").count() == 0
        assert page.get_by_placeholder("Password").input_value() == ""
        assert state["posts"] == 0
    finally:
        setup.close()


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("popup", "unexpected_popup"),
        ("modal", "unexpected_modal"),
        ("dialog", "unexpected_dialog"),
        ("navigation", "navigation_outside_boundary"),
    ],
)
def test_setup_safety_stops(setup_page, tmp_path, problem, reason):
    page, _ = setup_page
    setup = begin(page, tmp_path)
    setup.advance()
    if problem == "popup":
        page.context.new_page()
    elif problem == "modal":
        page.evaluate("document.body.insertAdjacentHTML('beforeend','<div role=dialog>Unexpected</div>')")
    elif problem == "dialog":
        page.evaluate("alert('fixture dialog')")
    else:
        with pytest.raises(PlaywrightError):
            page.goto("http://localhost/outside")
    with pytest.raises(SetupStop, match=f"setup_{reason}"):
        setup.advance()
    assert setup.closed and not list(tmp_path.iterdir())


@pytest.mark.parametrize("blank", [False, True])
def test_no_candidate_or_unconfirmed_star_never_keeps_clicking(setup_page, tmp_path, blank):
    page, state = setup_page
    state["authenticated"] = True
    state["canvas"] = canvas_html(blank=blank, no_response=True)
    setup = begin(page, tmp_path)
    reason = "starfield_render_timeout" if blank else "star_selection_not_confirmed"
    with pytest.raises(SetupStop, match=f"setup_{reason}"):
        for _ in range(20):
            setup.advance()
            if setup.star_clicked:
                setup.star_click_time = time.monotonic() - 9
            elif blank and setup.starfield_loading_recorded:
                setup.starfield_started = time.monotonic() - setup.STARFIELD_WAIT_SECONDS - 1
    assert setup.closed
    assert len(list(tmp_path.iterdir())) == (2 if blank else 1)
    assert (tmp_path / ("setup-starfield-rejected.png" if blank else "setup-starfield.png")).is_file()
    if blank:
        assert (tmp_path / "setup-starfield-loading.png").is_file()


def test_delayed_starfield_waits_and_polls_without_clicking_then_requires_stability(
    setup_page, tmp_path, monkeypatch
):
    import habfly.browser_setup as module

    clock = [1000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(BrowserSetup, "STARFIELD_POLL_SECONDS", 0.5)
    reads = []
    original = module.visible_star_point

    def detect(pixels):
        reads.append(clock[0])
        return original(pixels)

    monkeypatch.setattr(module, "visible_star_point", detect)
    page, state = setup_page
    state["authenticated"] = True
    state["canvas"] = canvas_html(blank=True)
    events = []
    setup = begin(page, tmp_path, events)
    for _ in range(4):
        setup.advance()
    page.frame_locator(f'iframe[src="{SIMULATION_URL}"]').locator("canvas").wait_for()
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    assert setup.advance() == "waiting"
    assert setup.stage == "waiting_for_starfield_render"
    assert reads == [1000.0] and not setup.star_clicked
    clock[0] += 0.25
    assert setup.advance() == "waiting" and reads == [1000.0]
    clock[0] += 8.0  # Slow canvas render, not a setup failure.
    assert setup.advance() == "waiting" and not setup.closed
    frame.locator("canvas").evaluate(
        "e=>{const c=e.getContext('2d');c.fillStyle='rgb(139,135,119)';c.fillRect(200,200,2,1)}"
    )
    clock[0] += 0.5
    assert setup.advance() == "waiting"
    assert setup.stage == "waiting_for_stable_starfield" and not setup.star_clicked
    # A transient blank frame invalidates the pending candidate, never clicks it.
    frame.locator("canvas").evaluate("e=>e.getContext('2d').clearRect(200,200,2,1)")
    clock[0] += 0.5
    assert setup.advance() == "waiting" and setup.starfield_candidate is None
    assert not setup.star_clicked
    frame.locator("canvas").evaluate(
        "e=>{const c=e.getContext('2d');c.fillStyle='rgb(139,135,119)';c.fillRect(200,200,2,1)}"
    )
    clock[0] += 0.5
    assert setup.advance() == "waiting" and not setup.star_clicked
    clock[0] += 0.5
    assert setup.advance() == "waiting" and setup.star_clicked
    finish(setup)
    assert setup.closed and setup.view_clicked
    assert sum(e[1].get("setup_stage") == "selecting_visible_star" for e in events) == 1
    assert not (tmp_path / "setup-starfield-rejected.png").exists()


def test_abort_while_starfield_loading_stops_without_a_click(setup_page, tmp_path):
    page, state = setup_page
    state["authenticated"] = True
    state["canvas"] = canvas_html(blank=True)
    setup = begin(page, tmp_path)
    for _ in range(7):
        assert setup.advance() == "waiting"
    assert setup.stage == "waiting_for_starfield_render" and not setup.star_clicked
    setup.close()
    with pytest.raises(SetupStop, match="setup_closed"):
        setup.advance()
    assert not setup.star_clicked and not setup.view_clicked


def decorative_heading_html():
    # Transparent decoration intercepts pointer hit-testing, not visibility/AX.
    # No source changes or hidden application state are needed to read this label.
    return stellar_html().replace(
        "Your Reconstruction",
        '<span style="position:relative"><span>Your Reconstruction</span>'
        '<span aria-hidden="true" style="position:absolute;inset:0"></span></span>',
    )


@pytest.mark.parametrize("decorative_heading", [False, True])
def test_runtime_automatic_setup_captures_without_advancing_policy(
    setup_page, tmp_path, monkeypatch, decorative_heading
):
    from habfly.browser_policy import BrowserPolicyBridge
    from habfly.runtime import Runtime

    page, state = setup_page
    if decorative_heading:
        state["canvas"] = canvas_html(detail=decorative_heading_html())
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", PASSWORD)
    options = RunOptions(task="browser_numeric", browser_setup="automatic")
    bridge = BrowserPolicyBridge(options, tmp_path / "evidence", {}, page=page, config=config())
    page.goto(OUTER)
    events = io.StringIO()
    runtime = Runtime(output=events)
    runtime.env, runtime.options, runtime.status = bridge, options, "paused"
    try:
        for _ in range(30):
            runtime.setup_tick()
            if bridge.phase == "ready":
                break
        assert bridge.phase == "ready" and runtime.status == "paused"
        assert runtime.observation is not None
        assert bridge.state()["browser_guidance"].startswith("n = one learned tool action")
        if decorative_heading:
            frame = next(f for f in page.frames if f.url == SIMULATION_URL)
            assert "Your Reconstruction" not in rendered_text(frame)
            assert "Your Reconstruction" in frame.locator("body").aria_snapshot()
        assert not bridge.session.attempts
        assert bridge.pending is None
        assert '"action_proposed"' not in events.getvalue()
        assert PASSWORD not in events.getvalue()
    finally:
        bridge.close()


def test_driver_error_is_redacted_and_abort_clears_credentials(setup_page, tmp_path, monkeypatch):
    page, _ = setup_page
    setup = begin(page, tmp_path)

    def fail():
        raise RuntimeError(f"driver call log {EMAIL} {PASSWORD}")

    monkeypatch.setattr(setup, "_advance", fail)
    with pytest.raises(SetupStop, match="^setup_browser_operation_failed$"):
        setup.advance()
    assert setup.closed and setup._email == setup._password == ""
    assert not list(tmp_path.iterdir())


def test_runtime_abort_during_setup_does_not_act(setup_page, tmp_path, monkeypatch):
    from habfly.browser_policy import BrowserPolicyBridge
    from habfly.runtime import Runtime

    page, state = setup_page
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", PASSWORD)
    options = RunOptions(task="browser_numeric", browser_setup="automatic")
    bridge = BrowserPolicyBridge(options, tmp_path / "abort", {}, page=page, config=config())
    runtime = Runtime(output=io.StringIO())
    runtime.env, runtime.options, runtime.status = bridge, options, "paused"
    runtime.command({"command": "abort"})
    assert state["posts"] == 0 and bridge.setup.closed
    assert bridge.setup._email == bridge.setup._password == ""
    assert not bridge.summary["task_completed"]


@pytest.mark.parametrize(
    "style",
    [
        "opacity:0;position:absolute;inset:0;pointer-events:none",
        "position:absolute;left:-3000px;top:0;width:800px",
        "position:absolute;inset:0;z-index:-1",
    ],
)
def test_mounted_unseen_details_do_not_skip_star_selection(setup_page, tmp_path, style):
    page, state = setup_page
    state["authenticated"] = True
    state["canvas"] = canvas_html().replace(
        "</body>", f'<section style="{style}">{stellar_html()}</section></body>'
    )
    setup = begin(page, tmp_path)
    for _ in range(4):
        setup.advance()
    page.frame_locator(f'iframe[src="{SIMULATION_URL}"]').locator("canvas").wait_for()
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    # Reproduce the original false readiness signal: innerText includes this
    # mounted panel even though its labels cannot be seen on the star map.
    assert "Your Reconstruction" in frame.locator("body").inner_text()
    assert "Your Reconstruction" not in rendered_text(frame)
    finish(setup)
    assert setup.star_clicked and setup.view_clicked
    assert (tmp_path / "setup-starfield.png").is_file()
    assert (tmp_path / "setup-star-link.png").is_file()


def test_loading_detail_markup_does_not_trigger_capture_before_a_star_is_opened(setup_page, tmp_path):
    page, state = setup_page
    state["authenticated"] = True
    state["canvas"] = stellar_html()
    setup = begin(page, tmp_path)
    for _ in range(7):
        assert setup.advance() == "waiting"
    assert not setup.closed and not setup.star_clicked and not setup.view_clicked
    assert setup.stage == "waiting_for_starfield"
    assert not list(tmp_path.iterdir())
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.set_content(canvas_html())
    finish(setup)
    assert setup.star_clicked and setup.view_clicked


def test_rendered_labels_can_be_noninteractive_but_not_clipped_or_covered(setup_page):
    page, _ = setup_page
    page.set_content("""<body style='margin:0'>
      <span style='pointer-events:none'>Visible parallax label</span>
      <div style='width:0;height:0;overflow:hidden'>
        <span style='pointer-events:none'>Clipped reconstruction</span></div>
      <div style='position:relative;height:60px'>
        <span style='pointer-events:none'>Covered wavelength</span>
        <div style='position:absolute;inset:0;background:black'></div></div>
      </body>""")
    text = rendered_text(page.main_frame)
    assert "Visible parallax label" in text
    assert "Clipped reconstruction" not in text and "Covered wavelength" not in text


def test_ready_waits_for_auxiliary_frames_and_stable_editable_fields(setup_page, tmp_path):
    page, state = setup_page
    state["authenticated"] = True
    setup = begin(page, tmp_path)
    for _ in range(4):
        setup.advance()
    widget = page.locator(f'iframe[src="{WIDGET}"]').first
    widget.evaluate("e=>e.style.display='none'")
    for _ in range(5):
        assert setup.advance() == "waiting"
    assert setup.view_clicked and not setup.closed
    assert setup.stage == "waiting_for_stellar_frames"
    widget.evaluate("e=>e.style.display='block'")
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.locator("#distance").evaluate("e=>e.readOnly=true")
    assert setup.advance() == "waiting"
    assert setup.stage == "waiting_for_stellar_controls"
    frame.locator("#distance").evaluate("e=>e.readOnly=false")
    assert setup.advance() == "waiting"
    assert setup.advance() == "stellar"


def test_open_detail_without_measurements_waits_and_requires_two_stable_reads(setup_page, tmp_path):
    page, state = setup_page
    state["authenticated"] = True
    state["canvas"] = canvas_html(detail=decorative_heading_html().replace("0.045", "0"))
    setup = begin(page, tmp_path)
    for _ in range(10):
        assert setup.advance() == "waiting"
    assert setup.view_clicked and setup.stage == "waiting_for_stellar_measurements"
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.set_content(decorative_heading_html())
    assert setup.advance() == "waiting"
    assert setup.stage == "verifying_stellar_screen"
    frame.set_content(decorative_heading_html().replace("0.045", "0.046"))
    assert setup.advance() == "waiting"  # Changed measurement restarts stability check.
    assert setup.advance() == "stellar"


@pytest.mark.parametrize("style", ["opacity:0", "position:absolute;left:-3000px", "visibility:hidden"])
def test_open_action_does_not_make_unexposed_fields_ready(setup_page, tmp_path, style):
    page, state = setup_page
    state["authenticated"] = True
    state["canvas"] = canvas_html(detail=f'<section style="{style}">{stellar_html()}</section>')
    setup = begin(page, tmp_path)
    for _ in range(10):
        assert setup.advance() == "waiting"
    assert setup.view_clicked and not setup.closed and setup.ready_signature is None
    assert setup.stage == "waiting_for_stellar_controls"
    setup.close()


def test_open_fields_with_wrong_labels_cannot_become_ready(setup_page, tmp_path):
    page, state = setup_page
    state["authenticated"] = True
    state["canvas"] = canvas_html(detail=stellar_html().replace("distance (ly)", "other measurement"))
    setup = begin(page, tmp_path)
    for _ in range(10):
        assert setup.advance() == "waiting"
    assert setup.view_clicked and setup.ready_signature is None
    assert setup.stage == "waiting_for_stellar_measurements"
    setup.close()


@pytest.mark.parametrize(
    "message,expected",
    [
        ("frame_count_mismatch:widgets", "setup_capture_frame_count_mismatch:widgets"),
        ("unverified_visible_field_label", "setup_capture_unverified_visible_field_label"),
        (f"frame_not_ready:https://fixture.invalid/{PASSWORD}", "setup_stellar_capture_failed"),
    ],
)
def test_capture_reason_is_sanitized_and_browser_stays_open_until_quit(
    setup_page, tmp_path, monkeypatch, message, expected
):
    from habfly.browser import BrowserSafetyStop
    from habfly.browser_policy import BrowserPolicyBridge
    from habfly.runtime import Runtime

    page, state = setup_page
    state["authenticated"] = True
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", PASSWORD)
    options = RunOptions(task="browser_numeric", browser_setup="automatic")
    bridge = BrowserPolicyBridge(options, tmp_path / "held", {}, page=page, config=config())
    bridge.context = page.context  # This test gives ownership to the bridge.
    page.goto(OUTER)

    def fail_capture():
        raise BrowserSafetyStop(message)

    monkeypatch.setattr(bridge, "ready", fail_capture)
    output = io.StringIO()
    runtime = Runtime(output=output)
    runtime.env, runtime.options, runtime.status = bridge, options, "paused"
    for _ in range(20):
        runtime.setup_tick()
        if runtime.status == "stopped":
            break
    assert runtime.status == "stopped" and bridge.outcome == expected
    assert not page.is_closed() and bridge.browser_held
    assert bridge.pending is None and bridge.setup.closed
    assert not bridge.summary["numeric_transport_passed"]
    assert bridge.summary["write_attempts"] == 0
    assert PASSWORD not in output.getvalue()
    evidence = (tmp_path / "held" / "events.jsonl").read_bytes()
    with pytest.raises(ValueError, match=f"STOP: {expected}"):
        runtime.command({"command": "step"})
    runtime.command({"command": "abort"})
    assert page.is_closed()
    assert (tmp_path / "held" / "events.jsonl").read_bytes() == evidence
    runtime.close()  # Cleanup is idempotent after the held browser is closed.
