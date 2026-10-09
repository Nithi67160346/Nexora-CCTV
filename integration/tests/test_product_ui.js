const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict'), path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../../web/static/product.js'),'utf8');
const html=fs.readFileSync(path.join(__dirname,'../../web/static/index.html'),'utf8');
const priority=html.slice(html.indexOf('    function updatePriorityAlert('),html.indexOf('    function renderEvents('));
function fixture(fetch) {
  const nodes=new Map(); let serial=0;
  function element(id) {
    if (!nodes.has(id)) {
      const node={value:'',textContent:'',checked:false,disabled:false,children:[],className:'',
        classList:{add(){},remove(){},toggle(){}},replaceChildren(){this.children=[];},
        append(...children){this.children.push(...children);}};
      Object.defineProperty(node,'options',{get(){return this.children;}});
      Object.defineProperty(node,'innerHTML',{set(){throw Error('Unsafe HTML');}});
      nodes.set(id,node);
    }
    return nodes.get(id);
  }
  const window={addEventListener(){}};
  const ctx=vm.createContext({window,fetch,console,confirm:()=>true,
    document:{getElementById:element,createElement:()=>element('new'+serial++)},
    currentSource:'test.mp4',analyzedSource:'',feedImg:{src:'old'},cancelZoneDrawing(){},showToast(){},
    fetchStatus:async()=>{},refreshFeedImage(){}});
  vm.runInContext(source+priority,ctx);
  return {ctx,element,window};
}
const reply=(body,ok=true)=>({ok,json:async()=>body});
const cameraData={active_id:'cam_01',profiles:[{id:'cam_01',name:'ห้อง A',source:'test.mp4'},
  {id:'cam_02',name:'<img onerror=bad>',source:''}]};
for (const profile of cameraData.profiles) profile.config={features:{location:{config:{zones:{}}},wandering:{zones_relative:{}}}};
(async()=>{
  const calls=[];
  const a=fixture(async(url,options)=>{calls.push([url,options]);return reply({...cameraData,active_id:'cam_02'});});
  a.ctx.renderCameras(cameraData);
  assert.equal(a.element('camera-profile-selector').children[1].textContent,'<img onerror=bad>');
  await a.ctx.selectCameraProfile('cam_02');
  assert.equal(a.window.activeCameraId,'cam_02'); assert.equal(a.ctx.currentSource,'');
  assert.equal(a.element('camera-profile-selector').disabled,false);
  assert.equal(calls.length,1,'room selection must not auto-start video');
  assert(calls[0][0].endsWith('/select'));
  a.ctx.updatePriorityAlert([{source_id:'cam_01',review_status:'needs_review',event_type:'fall_detected',severity:'high'}]);
  assert.equal(a.element('priority-alert-text').textContent,'','another room must not raise the selected room banner');
  a.ctx.updatePriorityAlert([{event_id:'fall2',source_id:'cam_02',review_status:'needs_review',event_type:'fall_detected',severity:'high'}]);
  assert(a.element('priority-alert-text').textContent.includes('<img onerror=bad>'));
  let replayed=null;a.ctx.playEvent=id=>replayed=id;a.element('priority-playback-button').onclick();
  assert.equal(replayed,'fall2','priority banner must replay the actual event rather than just scrolling');
  a.ctx.updatePriorityAlert([]);assert.equal(a.element('priority-playback-button').onclick,null);
  a.ctx.updateProductStatus({health:{capture:{state:'paused'},detector:{state:'ready'},features:{fall:{state:'no_observation'}}}});
  assert.equal(a.element('health-fall').textContent,'ไม่มีคนให้ประเมิน');
  a.ctx.updateProductStatus({product_revision:'qa-fall-lstm-main-20261007-fall54fix1',is_running:true,current_source:'uploads/current.mp4',
    interaction_diagnostics:{model_kind:'resnet18_lstm',result:{window_start_ms:2000,window_end_ms:3900,probability_fighting:.998,fighting:true}}});
  assert.match(a.element('main-violence-result').textContent,/current.mp4/);
  assert.match(a.element('main-violence-result').textContent,/2.0–3.9s/);
  assert.match(a.element('main-violence-result').textContent,/99.8%/);
  a.ctx.updateProductStatus({product_revision:'qa-fall-lstm-main-20261007-fall54fix1',is_running:false});
  assert.equal(a.element('main-violence-result').textContent,'','closing/switching the source must remove the previous scene score');
  a.ctx.updateProductStatus({is_running:true});
  assert.match(a.element('main-violence-result').textContent,/เซิร์ฟเวอร์เก่า/,'old native/Docker servers must not masquerade as the new main web');
  a.ctx.updateProductStatus({fall_backend:'yolo_pose_rf_v2',fall_model:{ml_loaded:true},fall_statuses:{7:{phase:'FALL_DETECTED',fall_score:.8}}});
  assert.match(a.element('main-fall-result').textContent,/คน #7/);
  assert.match(a.element('main-fall-result').textContent,/80.0%/);
  a.ctx.updateProductStatus({fall_backend:'yolo_pose_rf_v2',fall_model:{ml_loaded:true,feature_count:54},fall_statuses:{7:{phase:'FALL_DETECTED',fall_score:.1}}});
  assert.match(a.element('main-fall-result').textContent,/ตรวจล้ม 54 ค่า/);
  assert.match(a.element('main-fall-result').textContent,/คะแนน RF หน้าต่าง 10.0%/);
  a.ctx.updateProductStatus({fall_backend:'yolo_pose_rf_v2',fall_model:null});
  assert.equal(a.element('main-fall-result').textContent,'','disabled or unavailable fall module must not display old ML result');
  a.element('qa-event-type').value='fall_detected'; a.ctx.syncQaLabelScope();
  a.element('qa-start').value='1'; a.element('qa-end').value='2';a.ctx.addQaLabel();
  assert.equal(a.element('qa-labels').children.length,1);
  a.element('qa-negative').checked=true; a.window.activeCameraId='cam_01';a.ctx.syncQaLabelScope();
  assert.equal(a.element('qa-labels').children.length,0);assert.equal(a.element('qa-negative').checked,false);
  const draft={enabled:true,event_types:['fall_detected'],cooldown_sec:30,suppressed:0,drafts:[{text:'<script>not HTML</script>'}]};
  a.ctx.renderTelegram(draft);
  assert(a.element('telegram-drafts').children[0].textContent.includes('<script>'));
  const b=fixture(async()=>reply({mode:'draft_only',sent:false,text:'example'}));
  await b.ctx.previewTelegram(); assert(b.element('telegram-preview').textContent.includes('ยังไม่ได้ส่ง'));
  const c=fixture(async()=>reply({detail:'invalid QA'},false));
  await c.ctx.startQaReview();assert.equal(c.element('qa-start-button').disabled,false);
  assert.equal(c.element('qa-message').textContent,'invalid QA');
  console.log('PASS: camera selection, safe names, health, label scope, Telegram draft, QA failure');
})().catch(error=>{console.error(error);process.exitCode=1;});
