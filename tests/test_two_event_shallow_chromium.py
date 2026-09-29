"""Real native sensor/owner, synthetic intercepted SVG; no HabWorlds requests."""
# ruff: noqa: F811

import hashlib
import os

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_chart import chart_page  # noqa: F401

from habfly.browser_observation_progress import capture_observation_progress
from habfly.browser_shallow_transit_steps import ShallowTransitSteps
from habfly.planet_tooltip_reference import TWO_MODE, load_tooltip_reference

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must explicitly keep real previews idle during native fixtures",
)


@pytest.mark.parametrize("second_decline", [True, False])
def test_two_native_features_restore_exactly_or_stop_without_answer(chart_page, tmp_path, second_decline):
    page, frame = chart_page
    page.locator("iframe").first.evaluate(
        "e=>e.style='position:absolute;left:300px;top:250px;width:950px;height:600px;border:0'"
    )
    frame.get_by_role("textbox").first.fill("5000")
    # Authored fixture rendering only; production sees glyphs, PNGs, and native
    # tooltips. Neither sensor nor owner can read this closure's variables.
    frame.locator("svg").evaluate(
        """(svg,secondDecline)=>{
      const NS='http://www.w3.org/2000/svg';let focused=false, feature=0, origin=996;
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
        svg.replaceChildren();add('title',{},'Normalized Flux Days Observed');
        add('rect',{width:280,height:195,fill:'black'});
        const base=focused?80:20.5;
        for(const v of (focused?[90,95,100,105]:[10,20,30,40,50,60,70,80,90,100]))
          label(25,base+(100-v)*(focused?2.6:1.3),''+v,'end');
        if(focused){for(let v=origin-40;v<=origin+60;v+=20)label(79.5+(v-origin)*2.08,165,''+v);}
        else {for(let v=0;v<=5000;v+=500)label(30.5+v*.046,165,''+v);}
        const py=focused?80.5:20.5;
        add('path',{d:`M30 ${py}H260`,fill:'none',stroke:'rgb(80,132,154)','stroke-width':1});
        if(focused)add('rect',{x:88,y:py,width:2,height:5,fill:'rgb(80,132,154)'});
        else for(const x of [76,168])add('rect',{x,y:21,width:1,height:1,fill:'rgb(55,83,96)'});
        add('text',{id:'tip',x:40,y:45,fill:'white','font-size':10,style:'display:none'});
        add('circle',{id:'hover-marker',r:4,fill:'white',style:'display:none'});
      };
      svg.onwheel=e=>{
        e.preventDefault();
        if(e.deltaY>0)focused=false;
        else if(!focused){feature=e.clientX-svg.getBoundingClientRect().left<125?0:1;origin=996+2000*feature;focused=true;}
        draw();
      };
      svg.onmousemove=e=>{
        if(!focused)return;
        const x=e.clientX-svg.getBoundingClientRect().left;
        const day=Math.round(origin+(x-79.5)/2.08);
        const value=x>=88.5&&x<90&&(feature===0||secondDecline)?'99.238':'100';
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
        second_decline,
    )
    source = tmp_path / "original"
    initial = capture_observation_progress(page, config(), source, requested_days=5000)
    original_bytes = (source / "chart.png").read_bytes()
    checksum = hashlib.sha256((source / "report.json").read_bytes()).hexdigest()
    owner = ShallowTransitSteps(
        page,
        config(),
        tmp_path / "sensor",
        run_history=tmp_path,
        star=initial["star"],
        source_report=source / "report.json",
        source_report_sha256=checksum,
        allow_two_events=True,
    )
    try:
        for _ in range(80):
            state = owner.advance()
            if owner.finished:
                break
        assert owner.finished, state
        assert state["native_action_attempts"] == state["native_actions_confirmed"] <= 38
        assert state["max_seconds"] == 900 and state["probe_max_seconds"] == 180
        assert state["measurement_mode"] == TWO_MODE
        assert state["answer_writes"] == 0 and state["planet_decision"] is None
        assert state["task_completed"] is state["project_completed"] is False
        if second_decline:
            assert state["phase"] == "measurements_ready", state
            assert state["restorations"] == len(state["completed_probes"]) == 2
            result = state["measurements"]
            assert result == load_tooltip_reference(
                tmp_path,
                state["tooltip_reference"]["diagnostics"],
                expected_star=initial["star"],
                mode=TWO_MODE,
            )
            assert result["period_days"]["value"] == "2000"
            assert result["brightness_drop_percent"]["value"] == "0.762"
            assert result["recurrence_confirmed"] is result["answer_authorized"] is False
            assert result["confirmed_feature_count"] == 2 and result["consistency_redundancy"] == 0
            assert (tmp_path / "sensor/restore-02/progress/chart.png").read_bytes() == original_bytes
        else:
            assert state["phase"] == "stopped"
            assert state["failure_reason"] == "shallow_steps_probe_not_verified"
            assert state["measurements"] is state["tooltip_reference"] is None
            assert not (tmp_path / "sensor/report.json").exists()
        assert (source / "chart.png").read_bytes() == original_bytes
        assert frame.get_by_role("textbox").first.input_value() == "5000"
        assert frame.get_by_role("textbox").nth(1).input_value() == ""
        assert frame.get_by_role("combobox").input_value() == ""
        assert not page.get_by_role("checkbox").is_checked()
    finally:
        owner.close()
