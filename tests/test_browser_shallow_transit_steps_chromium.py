"""Real sensor/probes/validator over a fully intercepted synthetic chart UI.

No production validator, session, generator, capture, clock, or browser boundary
is mocked. The SVG is an explicit fixture, not a hook in the real application.
Only root may authorize Chromium while the actual preview is idle.
"""
# ruff: noqa: F811

import hashlib
import json
import os
import time

import pytest
from PIL import Image
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_chart import chart_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import capture_observation_progress
from habfly.browser_shallow_transit_probe import FIRST_THREE_QUANTIZED_HINT_POLICY, candidate_columns
from habfly.browser_shallow_transit_steps import FIRST_THREE_HINT_POLICY, MAX_ADVANCES, ShallowTransitSteps
from habfly.planet_tooltip_reference import load_tooltip_reference

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must explicitly keep real previews idle during native fixtures",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def shallow_sensor_page(chart_page, tmp_path, request):
    browser, frame = chart_page
    browser.locator("iframe").first.evaluate(
        "e=>e.style='position:absolute;left:300px;top:250px;width:950px;height:600px;border:0'"
    )
    frame.get_by_role("textbox").first.fill("5000")
    # The pointer-selected region controls which *rendered* daily chart appears.
    # Each overview hint contains the corresponding visible 1000/2000/3000 day
    # decline. The production sensor never reads these fixture variables.
    count = getattr(request, "param", 3)
    frame.locator("svg").evaluate(
        """(svg,count)=>{
      const NS='http://www.w3.org/2000/svg';
      const variant=count;
      const twoRow=typeof count==='string'&&count.startsWith('two_row');
      const dark=typeof count==='string',flat=['low_intensity_flat','quantized_flat','two_row_flat'].includes(count);
      const quantized=['quantized','quantized_flat'].includes(count);
      const period=count===48?100:1000;
      if(dark)count=3;
      const centers=Array.from({length:count},(_,i)=>period*(i+1));
      let depth=0,center=centers[0];
      const add=(name,attrs,text)=>{
        const e=document.createElementNS(NS,name);
        for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);
        if(text!==undefined)e.textContent=text;svg.append(e);return e;
      };
      const label=(x,y,text,anchor='middle')=>{
        const e=add('text',{x,y,fill:'white','font-size':10,'text-anchor':anchor},text);
        const b=e.getBoundingClientRect(),s=svg.getBoundingClientRect();
        e.setAttribute('y',y+(y-(b.top+b.height/2-s.top)));return e;
      };
      const draw=()=>{
        const focused=depth>0,base=focused?80:twoRow?20.45:20.95;
        svg.replaceChildren();add('title',{},'Normalized Flux Days Observed');
        add('rect',{width:280,height:195,fill:'black'});
        for(const v of (focused?[90,95,100,105]:[10,20,30,40,50,60,70,80,90,100]))
          label(25,base+(100-v)*(focused?2.6:1.3),''+v,'end');
        if(focused){
          for(const delta of [-20,0,20,40,60])label(89+delta*2.08,165,''+(center+delta));
        }else{
          for(let day=0;day<=5000;day+=500)label(30.5+day*.046,165,''+day);
        }
        const py=focused?80.5:twoRow?20.5:21.5;
        if(!focused&&dark)add('rect',{x:30,y:twoRow?21:22,width:231,height:1,fill:'rgb(26,26,26)'});
        add('path',{d:`M30 ${py}H260`,fill:'none',stroke:'rgb(80,132,154)','stroke-width':1});
        if(focused)add('rect',{x:88,y:py,width:2,height:5,fill:'rgb(80,132,154)'});
        else for(const [i,day] of centers.entries()){
          const x=Math.floor(30.5+day*.046);
          if(twoRow){
            // Explicit synthetic antialias support chain, not proof of how
            // Chromium or the course renderer produced a physical blend.
            if(i===0&&variant==='two_row_missing_original')
              add('rect',{x,y:20,width:1,height:1,fill:'rgb(26,26,26)'});
            add('rect',{x,y:21,width:1,height:1,fill:
              i===0&&variant==='two_row_broken_parent'?'rgb(26,26,26)':'rgb(32,39,42)'});
            add('rect',{x,y:22,width:1,height:1,fill:
              i===0&&variant==='two_row_invalid_child'?'rgb(5,8,11)':'rgb(10,17,19)'});
            continue;
          }
          if(quantized&&i===2)add('rect',{x,y:23,width:1,height:1,fill:'rgb(51,51,51)'});
          add('rect',{x,y:dark?22:21,width:dark?1:2,height:dark?1:2,
            fill:quantized?['rgb(5,8,10)','rgb(10,17,19)','rgb(53,56,58)'][i]:
              dark?['rgb(29,36,38)','rgb(27,31,32)','rgb(31,41,45)'][i]:'rgb(80,132,154)'});
        }
        add('text',{id:'tip',x:40,y:45,fill:'white','font-size':10,style:'display:none'});
        add('circle',{id:'hover-marker',r:4,fill:'white',style:'display:none'});
      };
      svg.onwheel=e=>{
        e.preventDefault();
        if(e.deltaY<0&&depth===0){
          const x=e.clientX-svg.getBoundingClientRect().left;
          center=centers.reduce((best,day)=>
            Math.abs(30.5+day*.046-x)<Math.abs(30.5+best*.046-x)?day:best,centers[0]);
        }
        depth=Math.max(0,Math.min(2750,depth-e.deltaY));
        draw();
      };
      svg.onmousemove=e=>{
        if(depth===0)return;
        const x=e.clientX-svg.getBoundingClientRect().left;
        const day=Math.round(center+(x-89)/2.08);
        const value=!flat&&day===center?'99.238':'100';
        const tip=svg.querySelector('#tip');tip.style.display='block';
        tip.textContent=`Brightness: ${value}% , Day: ${day}`;
        const marker=svg.querySelector('#hover-marker');marker.style.display='block';
        marker.setAttribute('cx',x);marker.setAttribute('cy',83);
      };
      svg.onmouseleave=()=>{
        svg.querySelector('#tip').style.display='none';
        svg.querySelector('#hover-marker').style.display='none';
      };
      draw();
    }""",
        count,
    )
    forbidden = []
    browser.expose_function("fixtureUnexpectedClick", lambda value: forbidden.append(value))
    for context in browser.frames:
        for button in context.get_by_role("button").all():
            button.evaluate("e=>e.addEventListener('click',()=>window.fixtureUnexpectedClick('button'))")
    source = tmp_path / "original-overview"
    capture_observation_progress(browser, config(), source, requested_days=5000)
    events = []
    owner = ShallowTransitSteps(
        browser,
        config(),
        tmp_path / "sensor",
        run_history=tmp_path,
        star="JYREMIS",
        source_report=source / "report.json",
        source_report_sha256=sha(source / "report.json"),
        emit=lambda kind, payload: events.append((kind, payload)),
    )
    try:
        yield browser, frame, owner, events, source, forbidden
    finally:
        owner.close()


def assert_answers_untouched(browser, frame):
    assert frame.get_by_role("textbox").first.input_value() == "5000"
    assert frame.get_by_role("textbox").nth(1).input_value() == ""
    assert frame.get_by_role("combobox").input_value() == ""
    assert not browser.get_by_role("checkbox").is_checked()


@pytest.mark.parametrize(
    "shallow_sensor_page", [3, 48, "low_intensity", "quantized", "two_row"], indirect=True
)
def test_native_three_probe_sensor_matches_independent_receipt_without_answer_writes(
    shallow_sensor_page, tmp_path, request
):
    browser, frame, owner, events, source, forbidden = shallow_sensor_page
    original = {name: sha(source / name) for name in ("report.json", "chart.png")}
    count = request.node.callspec.params["shallow_sensor_page"]
    source_value = json.loads((source / "report.json").read_bytes())
    groups = candidate_columns(
        (source / "chart.png").read_bytes(),
        source_value["flux_axis_labels"],
        hint_policy=FIRST_THREE_HINT_POLICY,
    )
    assert len(groups) == 3
    if count == 48:
        with pytest.raises(BrowserSafetyStop, match="too_many_candidate_groups"):
            candidate_columns((source / "chart.png").read_bytes(), source_value["flux_axis_labels"])
    if count == "low_intensity":
        assert (
            len(candidate_columns((source / "chart.png").read_bytes(), source_value["flux_axis_labels"])) < 3
        )
    if count == "two_row":
        assert groups == [[76], [122], [168]]
        with Image.open(source / "chart.png") as image:
            rgb = image.convert("RGB")
            for x in (76, 122, 168):
                assert rgb.getpixel((x, 20)) == (80, 132, 154)
                assert rgb.getpixel((x, 21)) == (32, 39, 42)
                assert rgb.getpixel((x, 22)) == (10, 17, 19)
        with pytest.raises(BrowserSafetyStop, match="unsupported_low_intensity_hint_context"):
            candidate_columns(
                (source / "chart.png").read_bytes(),
                source_value["flux_axis_labels"],
                hint_policy=FIRST_THREE_QUANTIZED_HINT_POLICY,
            )
    assert not events and owner.native_action_attempts == 0
    started = time.monotonic()
    for _ in range(MAX_ADVANCES):
        if owner.finished:
            break
        previous = owner.native_action_attempts
        owner.advance()
        # One existing bounded generator call may pan+clear; no whole probe can
        # hide inside a parent advance, and restoration actions remain separate.
        assert owner.native_action_attempts - previous <= 2
    assert owner.phase == "measurements_ready", owner.report
    result = json.loads((owner.output / "report.json").read_bytes())
    assert result == owner.report
    assert time.monotonic() - started < 900
    assert result["max_seconds"] == 900 and result["max_native_actions"] == 64
    assert result["max_advances"] == 80 and result["advances"] <= 80
    assert 0 < result["native_action_attempts"] == result["native_actions_confirmed"] <= 64
    assert result["native_action_outcome_uncertain"] is False
    assert len(result["completed_probes"]) == 3
    descriptor = result["tooltip_reference"]
    independent = load_tooltip_reference(
        tmp_path, descriptor["diagnostics"], expected_star=descriptor["expected_star"]
    )
    assert independent == result["measurements"]
    assert independent["period_days"]["value"] == ("100" if count == 48 else "1000")
    assert result["overview_hint_policy"] == FIRST_THREE_HINT_POLICY
    assert owner.candidates == groups
    for index in range(3):
        directory = owner.output / f"probe-{index + 1:02d}"
        scope = json.loads((directory / "scope.json").read_bytes())
        probe = json.loads((directory / "report.json").read_bytes())
        assert scope["overview_hint_policy"] == probe["overview_hint_policy"] == FIRST_THREE_HINT_POLICY
        assert scope["candidate_index"] == index and scope["candidate_columns"] == groups[index]
        assert probe["source_hint_linked"] is True
    assert not (owner.output / "probe-04").exists()
    assert independent["brightness_drop_percent"]["value"] == "0.762"
    assert independent["brightness_drop_percent"]["physical_bounds"] is None
    assert independent["period_days"]["compatibility_interval"]["endpoints"] == "open"
    for report in (result, independent):
        for key in (
            "scientific_verified",
            "learned_perception",
            "training_label",
            "task_completed",
            "answer_authorized",
        ):
            assert report[key] is False
    assert result["project_completed"] is False and result["answer_writes"] == 0
    assert {name: sha(source / name) for name in original} == original
    assert_answers_untouched(browser, frame)
    assert not forbidden
    native = [payload for kind, payload in events if kind == "action_proposed"]
    assert len(native) == owner.native_action_attempts
    assert all(item["kind"] in {"SCROLL", "HOVER", "DRAG"} for item in native)
    positive = [item for item in native if item.get("delta_y", 0) > 0]
    assert len(positive) == 6  # Two restorations after each of three probes; none initially.
    assert all(item["delta_y"] == 1500 for item in positive)
    print(
        json.dumps(
            {
                "fixture_overview_groups": count,
                "native_actions": owner.native_action_attempts,
                "advances": owner.advances,
                "elapsed_seconds": time.monotonic() - started,
                "max_seconds": 900,
                "max_actions": 64,
                "max_advances": 80,
            }
        )
    )


def test_native_sensor_stops_before_next_action_after_visible_answer_mutation(shallow_sensor_page, tmp_path):
    browser, frame, owner, _events, source, forbidden = shallow_sensor_page
    for _ in range(MAX_ADVANCES):
        if owner.finished or owner.completed_probes:
            break
        owner.advance()
    assert not owner.finished and len(owner.completed_probes) == 1, owner.report
    before = owner.native_action_attempts
    # An explicit test-owned visible change, not an application hook or sensor
    # repair. The real long-lived native signature must reject it.
    frame.get_by_role("textbox").nth(1).fill("7")
    owner.advance()
    assert owner.phase == "stopped" and owner.finished
    assert owner.native_action_attempts == before
    assert "context_or_answers_changed" in owner.failure
    assert not (owner.output / "report.json").exists()
    assert not (owner.output / "probe-02").exists()
    owner.advance()
    assert owner.native_action_attempts == before and not forbidden
    assert frame.get_by_role("textbox").nth(1).input_value() == "7"
    assert not browser.get_by_role("checkbox").is_checked()
    with pytest.raises(BrowserSafetyStop, match="owner_already_claimed"):
        ShallowTransitSteps(
            browser,
            config(),
            tmp_path / "cannot-restart",
            run_history=tmp_path,
            star="JYREMIS",
            source_report=source / "report.json",
            source_report_sha256=sha(source / "report.json"),
        )


@pytest.mark.parametrize(
    "shallow_sensor_page", ["low_intensity_flat", "quantized_flat", "two_row_flat"], indirect=True
)
def test_native_faint_hints_without_bracketed_tooltips_never_produce_measurements(shallow_sensor_page):
    browser, frame, owner, events, _source, forbidden = shallow_sensor_page
    for _ in range(MAX_ADVANCES):
        if owner.finished:
            break
        owner.advance()
    assert owner.finished and owner.phase == "stopped", owner.report
    assert owner.completed_probes == []
    assert not (owner.output / "report.json").exists()
    assert not (owner.output / "probe-02").exists()
    assert owner.native_action_attempts <= 16 and owner.advances <= 80
    assert owner.report["task_completed"] is False and owner.report["answer_writes"] == 0
    assert_answers_untouched(browser, frame)
    assert not forbidden
    actions = [payload for kind, payload in events if kind == "action_proposed"]
    assert all(item["kind"] in {"SCROLL", "HOVER", "DRAG"} for item in actions)
    previous = owner.native_action_attempts
    owner.advance()
    assert owner.native_action_attempts == previous


@pytest.mark.parametrize(
    "shallow_sensor_page",
    ["two_row_broken_parent", "two_row_missing_original", "two_row_invalid_child"],
    indirect=True,
)
def test_native_two_row_invalid_first_chain_never_skips_or_dispatches(shallow_sensor_page):
    browser, frame, owner, events, source, forbidden = shallow_sensor_page
    value = json.loads((source / "report.json").read_bytes())
    with pytest.raises(BrowserSafetyStop, match="unsupported_low_intensity_hint_context"):
        candidate_columns(
            (source / "chart.png").read_bytes(),
            value["flux_axis_labels"],
            hint_policy=FIRST_THREE_HINT_POLICY,
        )
    original = {name: sha(source / name) for name in ("report.json", "chart.png")}
    for _ in range(MAX_ADVANCES):
        if owner.finished:
            break
        owner.advance()
    assert owner.finished and owner.phase == "stopped", owner.report
    assert "unsupported_low_intensity_hint_context" in owner.failure
    assert owner.native_action_attempts == owner.native_actions_confirmed == 0
    assert owner.completed_probes == []
    assert not (owner.output / "probe-01").exists()
    assert not (owner.output / "report.json").exists()
    assert not any(kind == "action_proposed" for kind, _ in events)
    assert owner.report["task_completed"] is False and owner.report["answer_writes"] == 0
    assert {name: sha(source / name) for name in original} == original
    owner.advance()
    assert owner.native_action_attempts == 0 and not forbidden
    assert_answers_untouched(browser, frame)
