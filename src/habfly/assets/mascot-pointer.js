// Native pointer coordinates only. No targets, labels, values or application state.
channel => {
  const key=Symbol.for(channel+'-pointer');
  if(window[key])return;
  function send(x,y) {
    const data={channel,type:'habfly-pointer',x,y};
    if(window===window.top)window.postMessage(data,'*');
    else window.parent.postMessage(data,'*');
  }
  const move=event=>send(event.clientX,event.clientY);
  const message=event=>{
    const data=event.data;
    if(event.source===window || !data || data.channel!==channel || data.type!=='habfly-pointer'
       || !Number.isFinite(data.x) || !Number.isFinite(data.y))return;
    for(const frame of document.querySelectorAll('iframe,frame')){
      if(frame.contentWindow!==event.source)continue;
      const rect=frame.getBoundingClientRect();
      if(!frame.offsetWidth || !frame.offsetHeight)return;
      const sx=rect.width/frame.offsetWidth,sy=rect.height/frame.offsetHeight;
      send(rect.left+(frame.clientLeft+data.x)*sx,rect.top+(frame.clientTop+data.y)*sy);
      return;
    }
  };
  window.addEventListener('pointermove',move,{passive:true});
  window.addEventListener('message',message);
  window[key]={remove(){
    window.removeEventListener('pointermove',move);
    window.removeEventListener('message',message);
    delete window[key];
  }};
}
