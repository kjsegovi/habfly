"""Native diagnostic fixture; every request is intercepted, root idle opt-in."""
# ruff: noqa: F811

import hashlib
import os

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_chart import chart_page  # noqa: F401

from habfly.browser_observation_progress import capture_observation_progress
from habfly.browser_shallow_transit_probe import probe_shallow_feature

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must explicitly keep real previews idle during native fixtures",
)


@pytest.mark.parametrize("need_pan", [False, True])
@pytest.mark.parametrize("linked", [False, True])
def test_native_zoom_and_visible_tooltips_never_write_answers(chart_page, tmp_path, need_pan, linked):
    page, frame = chart_page
    page.locator("iframe").first.evaluate(
        "e=>e.style='position:absolute;left:300px;top:250px;width:950px;height:600px;border:0'"
    )
    frame.get_by_role("textbox").first.fill("5000")
    # This is a synthetic UI fixture, not an application hook in HabWorlds.
    # The adapter only sees rendered glyphs, pixels and ordinary pointer events.
    frame.locator("svg").evaluate(
        """(svg,{needPan,linked})=>{
      const NS='http://www.w3.org/2000/svg';let focused=false, centered=!needPan;
      const origin=linked?2564:2334;
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
        const base=focused&&centered?80:20.95;
        for(const v of (focused?[90,95,100,105]:[55,70,85,100]))
          label(25,base+(100-v)*2.6,''+v,'end');
        if(focused){for(let v=2300;v<=2640;v+=20)label(79.5+(v-origin)*2.08,165,''+v);}
        else {for(let v=0;v<=5000;v+=500)label(30.5+v*.046,165,''+v);}
        const py=focused&&centered?80.5:21.5;
        add('path',{d:`M30 ${py}H260`,fill:'none',stroke:'rgb(80,132,154)','stroke-width':1});
        if(focused)add('rect',{x:88,y:py,width:2,height:5,fill:'rgb(80,132,154)'});
        else add('rect',{x:148,y:21,width:2,height:2,fill:'rgb(80,132,154)'});
        add('text',{id:'tip',x:40,y:45,fill:'white','font-size':10,style:'display:none'});
        add('circle',{id:'hover-marker',r:4,fill:'white',style:'display:none'});
      };
      svg.onwheel=e=>{e.preventDefault();focused=true;draw()};
      svg.onmousemove=e=>{
        if(e.buttons===1&&focused&&!centered){centered=true;draw();}
        if(!focused)return;
        const x=e.clientX-svg.getBoundingClientRect().left;
        const day=Math.round(origin+(x-79.5)/2.08);
        const value=x>=88.5&&x<90?'99.238':'100';
        const tip=svg.querySelector('#tip');tip.style.display='block';
        tip.textContent=`Brightness: ${value}% , Day: ${day}`;
        const marker=svg.querySelector('#hover-marker');marker.style.display='block';
        marker.setAttribute('cx',x);marker.setAttribute('cy',centered?83:24);
      };
      svg.onmouseleave=()=>{
        svg.querySelector('#tip').style.display='none';
        svg.querySelector('#hover-marker').style.display='none';
      };
      draw();
    }""",
        {"needPan": need_pan, "linked": linked},
    )
    overview = tmp_path / "overview"
    capture_observation_progress(page, config(), overview, requested_days=5000)
    checksum = hashlib.sha256((overview / "report.json").read_bytes()).hexdigest()
    result = probe_shallow_feature(
        page,
        config(),
        tmp_path / "probe",
        run_history=tmp_path,
        source_dir=overview,
        source_report_sha256=checksum,
        candidate_index=0,
    )
    assert result["status"] == ("visible_bracketed_dip" if linked else "visible_decline_not_source_linked")
    assert result["source_hint_linked"] is linked
    assert result["browser_actions"] <= 16
    assert result["daily_coverage_verified"]
    assert result["planet_decision"] is None and result["answer_writes"] == 0
    assert not result["task_completed"] and not result["period_evidence_verified"]
    assert frame.get_by_role("textbox").first.input_value() == "5000"
    assert frame.get_by_role("textbox").nth(1).input_value() == ""
    assert frame.get_by_role("combobox").input_value() == ""
    assert not page.get_by_role("checkbox").is_checked()
