// This renders only a known HabFly presentation payload. It never reads app data.
args => {
  const key = Symbol.for(args.channel);
  if (window[key]) { window[key].update(args.display); return; }
  if (document.getElementById(args.host)) throw new Error('presentation_host_collision');
  const host = document.createElement('div');
  host.id = args.host;
  host.setAttribute('data-habfly-presentation', 'mascot-v1');
  host.setAttribute('aria-hidden', 'true');
  host.setAttribute('role', 'presentation');
  host.inert = true;
  // Never inline visibility (including an !important `all` reset): the evidence
  // screenshot stylesheet must be able to hide the host and its shadow canvas.
  host.style.cssText = 'all:initial;position:fixed!important;inset:0!important;'
    + 'display:block!important;pointer-events:none!important;z-index:2147483647!important;'
    + 'overflow:hidden!important;contain:strict!important;';
  const root = host.attachShadow({mode:'closed'});
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;height:100%;pointer-events:none;';
  root.append(canvas);
  document.documentElement.append(host);
  const image = new Image();
  let display = args.display, point = null;
  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
  function draw() {
    if (!host.isConnected) return;
    const w = innerWidth, h = innerHeight, dpr = devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
    // The ring marks the real last observed pointer coordinate, not a predicted target.
    const x = point ? point.x : 40, y = point ? point.y : h - 120;
    if (point) {
      ctx.strokeStyle = '#f7cb62'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2); ctx.stroke();
    }
    const left = clamp(x + 13, 6, w - 274), top = clamp(y - 95, 6, h - 102);
    if (image.complete && image.naturalWidth) ctx.drawImage(image,left,top+15,82,82);
    const bx = left + 82, by = top, bw = 184, bh = 70;
    ctx.fillStyle = '#fff9e9'; ctx.strokeStyle = '#d5aa4c'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(bx,by,bw,bh,16); ctx.fill(); ctx.stroke();
    for (const [cx,cy,r] of [[bx-5,by+55,4],[bx-12,by+62,2.5]]) {
      ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.fill(); ctx.stroke();
    }
    ctx.fillStyle = display.activity === 'stopped' ? '#a02c37' : '#302737';
    ctx.font = '600 14px system-ui';
    const words = String(display.caption).slice(0,72).split(' ');
    const lines = []; let line = '';
    for (const word of words) {
      const next = line ? line+' '+word : word;
      if (ctx.measureText(next).width > bw-22 && line) {lines.push(line);line=word;} else line=next;
    }
    if(line)lines.push(line);
    lines.slice(0,2).forEach((text,i)=>ctx.fillText(text,bx+11,by+22+i*18,bw-22));
    ctx.font = '10px system-ui'; ctx.fillStyle = '#806f57';
    ctx.fillText(String(display.source_label).slice(0,40),bx+11,by+58,bw-22);
  }
  const message = event => {
    const data = event.data;
    if (!data || data.channel !== args.channel || data.type !== 'habfly-pointer'
        || !Number.isFinite(data.x) || !Number.isFinite(data.y)) return;
    // Forwarded coordinates have already been transformed by the direct parent.
    if (event.source !== window) return;
    point = {x:clamp(data.x,0,innerWidth),y:clamp(data.y,0,innerHeight)}; draw();
  };
  window.addEventListener('message',message);
  window.addEventListener('resize',draw);
  image.onload = draw; image.src = args.asset;
  window[key] = {
    update(next) { display=next; draw(); },
    remove() {
      host.remove(); window.removeEventListener('message',message);
      window.removeEventListener('resize',draw); delete window[key];
    }
  };
  draw();
}
