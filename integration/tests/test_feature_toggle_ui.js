// Regression tests against the actual inline UI functions, using controlled HTTP replies.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../../web/static/index.html'), 'utf8');
function extract(name, nextName) {
  const start = html.indexOf('    async function ' + name + '(');
  const end = html.indexOf('    ' + nextName, start);
  assert(start >= 0 && end > start);
  return html.slice(start, end);
}
const functions = extract('fetchStatus', 'async function fetchStats') +
                  extract('toggleFeatureState', 'function toggleSettingsModal');
function fixture(fetch) {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, {checked:false,disabled:false,textContent:'',classes,
        classList:{add(...names){names.forEach(n=>classes.add(n));},remove(...names){names.forEach(n=>classes.delete(n));}},src:''});
    }
    return elements.get(id);
  };
  const messages=[];
  const ctx=vm.createContext({fetch, document:{getElementById:element},console,window:{},
    setApiHealth(){},showToast:msg=>messages.push(msg),feedImg:element('feed'),
    placeholder:element('placeholder'),btnPlay:element('btn-play'),isUserDraggingSlider:false,
    refreshFeedImage(){},seekSlider:element('slider'),timeCurrent:element('time-current'),
    timeDuration:element('time-duration')});
  vm.runInContext(functions,ctx);
  return {ctx,element,messages};
}
const response = body => ({ok:true,json:async()=>body});
const status = enabled_features => ({enabled_features,is_running:false,is_paused:false,
  device:'cpu',fps:0,active_tracks_count:0,weights_file:'yolo26n-pose.pt',last_error:null});
(async()=>{
  let oldResponse,first=true;
  const a=fixture((url,options)=>{
    if(options?.method==='POST') return Promise.resolve(response({enabled_features:['fall']}));
    if(first){first=false;return new Promise(resolve=>oldResponse=resolve);}
    return Promise.resolve(response(status(['fall'])));
  });
  const pending=a.ctx.fetchStatus();
  await a.ctx.toggleFeatureState('fall',true);
  oldResponse(response(status([]))); await pending;
  assert.equal(a.element('toggle-fall').checked,true,'stale poll must not undo successful toggle');
  const b=fixture((url,options)=>Promise.resolve(response(options?.method==='POST'
    ? {enabled_features:[]} : status([]))));
  await b.ctx.toggleFeatureState('fall',true);
  assert.equal(b.element('toggle-fall').checked,false);
  assert(b.messages.some(msg=>msg.includes('รีสตาร์ต')),'200 with unchanged runtime must show error');
  assert(!b.messages.some(msg=>msg.startsWith('ตั้งค่าฟีเจอร์')),'must not show a false success');
  let cameraStatus = {...status([]), is_running:true, frame_ready:false};
  const camera = fixture(()=>Promise.resolve(response(cameraStatus)));
  await camera.ctx.fetchStatus();
  assert(camera.element('feed').classes.has('hidden'), 'no frame must show a waiting message');
  assert(camera.element('stream-placeholder-message').textContent.includes('กำลังเปิด'));
  cameraStatus = {...cameraStatus, frame_ready:true, camera_warning:'ภาพจากกล้องมืดมาก'};
  await camera.ctx.fetchStatus();
  assert(!camera.element('feed').classes.has('hidden'));
  assert.equal(camera.element('camera-warning').textContent,'ภาพจากกล้องมืดมาก');
  cameraStatus = {...cameraStatus, is_running:false, frame_ready:false, last_error:'เปิดเว็บแคมไม่ได้', camera_warning:null};
  await camera.ctx.fetchStatus();
  assert.equal(camera.element('stream-placeholder-message').textContent,'เปิดเว็บแคมไม่ได้');
  assert(camera.element('camera-warning').classes.has('hidden'));
  assert(html.includes('<option value="1">'), 'a second webcam must be selectable');
  assert(!html.includes('toggle-seizure') && !html.includes('อาการชัก'), 'archived feature must be absent from the page');
  console.log('PASS: stale status response cannot undo checkbox; unchanged runtime cannot report success');
  console.log('PASS: camera selection, waiting frame, dark advisory and capture error states');
})().catch(error=>{console.error(error);process.exitCode=1;});
