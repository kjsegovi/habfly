"""All URLs intercepted. The fixture phase is paint, never a scientific oracle."""
# ruff: noqa: F811

import json

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_water_chamber import CHAMBER_URL, HELP_BUTTON, WaterChamberSession, condition_number


def helper_html(phase="gas", behavior=""):
    icons = "".join(
        f'<div role="img" aria-label="{name}" style="position:absolute;width:100px;height:100px;'
        f"top:100px;left:500px;opacity:{int(name == phase)};background:blue;"
        f'pointer-events:{"auto" if name == phase else "none"}"></div>'
        for name in ("solid", "liquid", "gas")
    )
    return (
        '<button>Water (H2O)</button><label>atm<input id="pressure" aria-label="atm" value="1.000"></label>'
        '<label>K<input id="temperature" aria-label="K" value="273.2"></label>'
        + icons
        + '<script>pressure.onkeydown=e=>{if(e.key==="Enter") pressure.value=Number(pressure.value).toFixed(3)};'
        + behavior
        + "</script>"
    )


def install_helper(page, *, html=None, open_script="", close_script=""):
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.route(CHAMBER_URL, lambda route: route.fulfill(content_type="text/html", body=html or helper_html()))
    page.locator("body").evaluate(
        """(e,args)=>{
        const button=document.createElement('button');button.textContent=args.name;
        button.onclick=()=>{
            const d=document.createElement('div');d.setAttribute('role','dialog');
            d.style='position:fixed;inset:100px;z-index:99;background:white';
            const close=document.createElement('button');close.textContent='Close';
            close.onclick=()=>{new Function(args.close)();d.remove()};d.append(close);
            const iframe=document.createElement('iframe');iframe.src=args.url;
            iframe.style='display:block;width:900px;height:400px';d.append(iframe);e.append(d);
            new Function(args.open)();
        };e.prepend(button);
    }""",
        {"name": HELP_BUTTON, "url": CHAMBER_URL, "open": open_script, "close": close_script},
    )


@pytest.mark.parametrize("phase", ["solid", "liquid", "gas"])
def test_visible_phase_exact_conditions_and_unchanged_task(page, tmp_path, phase):
    install_helper(page, html=helper_html(phase))
    events = []
    session = WaterChamberSession(page, config(), tmp_path / "query", lambda *e: events.append(e))
    result = session.query("9", "789.4", source="checkpoint")
    assert result["phase"] == phase and result["conditions_verified"]
    assert result["visible_readback"] == {"atm": "9.000", "K": "789.4"}
    assert not result["task_completed"] and not result["task_answer_write"]
    assert (tmp_path / "query/confirmed.json").exists()
    assert len(events) == 2 and not page.get_by_role("dialog").count()
    assert not page.get_by_role("checkbox").is_checked()
    with pytest.raises(BrowserSafetyStop, match="already_used"):
        session.query("9", "789.4", source="checkpoint")


@pytest.mark.parametrize("value", [0, "0", "-1", "nan", "Infinity", "1+2", "1_0", " 1", "1 ", "", "1" * 65])
def test_invalid_conditions(value):
    with pytest.raises(BrowserSafetyStop, match="invalid_chamber_condition"):
        condition_number(value)


@pytest.mark.parametrize(
    "kwargs", [{"pressure_unit": "bar"}, {"temperature_unit": "C"}, {"source": "oracle"}]
)
def test_invalid_request_never_opens_help(page, tmp_path, kwargs):
    install_helper(page)
    session = WaterChamberSession(page, config(), tmp_path / "query")
    with pytest.raises(BrowserSafetyStop, match="invalid_chamber_request"):
        session.query("9", "789.4", **({"source": "checkpoint"} | kwargs))
    assert not page.get_by_role("dialog").count() and not session.attempted


@pytest.mark.parametrize(
    "behavior, reason",
    [
        ('pressure.onkeydown=e=>{if(e.key==="Enter")pressure.value="0.380"}', "readback_mismatch"),
        ('temperature.onblur=()=>temperature.value="300"', "readback_mismatch"),
        ("temperature.onblur=()=>temperature.replaceWith(temperature.cloneNode(true))", "input_replaced"),
        ('pressure.onkeydown=()=>alert("unexpected")', "unexpected_dialog"),
        (
            'temperature.onblur=()=>document.body.insertAdjacentHTML("beforeend","<input type=password>")',
            "unexpected_state",
        ),
    ],
)
def test_bad_or_uncertain_readback_closes_only_owned_help(page, tmp_path, behavior, reason):
    install_helper(page, html=helper_html(behavior=behavior))
    session = WaterChamberSession(page, config(), tmp_path / "query")
    with pytest.raises(BrowserSafetyStop, match=reason):
        session.query("9", "789.4", source="reference_diagnostic")
    assert session.closed and not page.get_by_role("dialog").count()
    stopped = json.loads((tmp_path / "query/stopped.json").read_text())
    assert not stopped["retry_allowed"] and not (tmp_path / "query/confirmed.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        'document.querySelector("input[type=checkbox]").checked=true',
        'document.querySelectorAll("iframe")[1].replaceWith(document.querySelectorAll("iframe")[1].cloneNode(true))',
    ],
)
def test_task_mutation_or_same_url_frame_replacement_is_not_success(page, tmp_path, change):
    install_helper(page, close_script=change)
    session = WaterChamberSession(page, config(), tmp_path / "query")
    with pytest.raises(BrowserSafetyStop):
        session.query("9", "789.4", source="checkpoint")
    assert not (tmp_path / "query/confirmed.json").exists()


@pytest.mark.parametrize(
    "html",
    [
        helper_html().replace("opacity:1", "opacity:0.5"),
        helper_html().replace("opacity:0", "opacity:1"),
        helper_html().replace("left:500px", "left:950px"),
        helper_html().replace("Water (H2O)", "Methane (CH4)"),
    ],
)
def test_no_phase_inference_from_faint_multiple_clipped_or_wrong_material(page, tmp_path, html):
    install_helper(page, html=html)
    session = WaterChamberSession(page, config(), tmp_path / "query", max_seconds=4)
    with pytest.raises(BrowserSafetyStop):
        session.query("9", "789.4", source="checkpoint")
    assert not (tmp_path / "query/confirmed.json").exists()


def test_unknown_help_frame_never_read(page, tmp_path):
    install_helper(
        page, open_script='document.querySelector("[role=dialog] iframe").src="https://unknown.invalid/"'
    )
    session = WaterChamberSession(page, config(), tmp_path / "query")
    with pytest.raises(BrowserSafetyStop, match="unknown_frame"):
        session.query("9", "789.4", source="checkpoint")
    assert not (tmp_path / "query/observed.json").exists()


def test_unexpected_replacement_dialog_is_not_dismissed(page, tmp_path):
    install_helper(
        page,
        open_script='const d=document.querySelector("[role=dialog]");d.insertAdjacentHTML("afterend","<div role=dialog>Unknown modal</div>")',
    )
    session = WaterChamberSession(page, config(), tmp_path / "query")
    with pytest.raises(BrowserSafetyStop):
        session.query("9", "789.4", source="checkpoint")
    assert page.get_by_text("Unknown modal").is_visible()


def test_phase_transition_waits_for_fully_opaque_indicator(page, tmp_path):
    html = helper_html(
        "solid",
        behavior="""temperature.onblur=()=>setTimeout(()=>{
      document.querySelectorAll('[role=img]').forEach(e=>{const on=e.getAttribute('aria-label')==='liquid';e.style.opacity=on?1:0;e.style.pointerEvents=on?'auto':'none'});
    },900)""",
    )
    install_helper(page, html=html)
    result = WaterChamberSession(page, config(), tmp_path / "query").query("9", "789.4", source="checkpoint")
    assert result["phase"] == "liquid"


def test_uncertain_close_is_not_clicked_twice(page, tmp_path):
    install_helper(page)
    page.get_by_role("button", name=HELP_BUTTON, exact=True).evaluate("""e=>{
        const open=e.onclick;e.onclick=()=>{open();
          const close=document.querySelector('[role=dialog] button');
          close.onclick=()=>{close.textContent='Close';document.body.setAttribute('data-close-attempts',String(Number(document.body.getAttribute('data-close-attempts')||0)+1))};
        };
    }""")
    session = WaterChamberSession(page, config(), tmp_path / "query")
    with pytest.raises(PlaywrightTimeoutError):
        session.query("9", "789.4", source="checkpoint")
    assert page.locator("body").get_attribute("data-close-attempts") == "1"
    assert (tmp_path / "query/close-reserved.json").exists()
    assert not (tmp_path / "query/confirmed.json").exists()


def test_dialog_container_need_not_be_a_pointer_target(page, tmp_path):
    install_helper(
        page,
        open_script="""const d=document.querySelector('[role=dialog]');
      d.style.pointerEvents='none';d.querySelectorAll('button,iframe').forEach(e=>e.style.pointerEvents='auto');
    """,
    )
    result = WaterChamberSession(page, config(), tmp_path / "query").query("9", "789.4", source="checkpoint")
    assert result["phase"] == "gas" and not page.get_by_role("dialog").count()


def test_nested_button_captions_are_exposed_but_sibling_overlay_is_not(page, tmp_path):
    from habfly.browser_water_chamber import chamber_control_exposed

    install_helper(
        page,
        html=helper_html().replace("Water (H2O)</button>", "<span>Water (H2O)</span></button>"),
        open_script='document.querySelector("[role=dialog] button").innerHTML="<span>Close</span>"',
    )
    result = WaterChamberSession(page, config(), tmp_path / "query").query("9", "789.4", source="checkpoint")
    assert result["phase"] == "gas"
    button = page.get_by_role("button", name=HELP_BUTTON, exact=True)
    button.evaluate("""e=>{const r=e.getBoundingClientRect(),cover=document.createElement('div');
       cover.style=`position:fixed;left:${r.x}px;top:${r.y}px;width:${r.width}px;height:${r.height}px;z-index:999;background:white`;
       e.after(cover);
    }""")
    assert not chamber_control_exposed(button)


def test_inputs_may_exist_before_helper_finishes_loading(page, tmp_path):
    html = helper_html(behavior="temperature.disabled=true;setTimeout(()=>temperature.disabled=false,500)")
    install_helper(page, html=html)
    result = WaterChamberSession(page, config(), tmp_path / "query").query("9", "789.4", source="checkpoint")
    assert result["conditions_verified"] and result["visible_readback"]["K"] == "789.4"
