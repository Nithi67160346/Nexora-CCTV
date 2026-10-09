const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict'), path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../../web/static/violence.js'),'utf8');
function fixture(fetch) {
  const nodes=new Map(), timers=[];
  function element(id) {
    if(!nodes.has(id)) nodes.set(id,{value:'cpu',textContent:'',disabled:false,classList:{add(){},remove(){},toggle(){}}});
    return nodes.get(id);
  }
  const ctx=vm.createContext({fetch,currentSource:'original.mp4',encodeURIComponent,
    document:{getElementById:element},setTimeout:fn=>timers.push(fn)});
  vm.runInContext(source,ctx); return {ctx,element,timers};
}
const reply=(value,ok=true)=>({ok,json:async()=>value});
const done={id:'original',filename:'<img src=x onerror=bad>.mp4',state:'completed',device:'cpu',
  result:{fighting:true,probability_fighting:.8,sampled_frames:16,duration_s:3.67}};
(async()=>{
  const a=fixture(async()=>reply({available:true}));
  await a.ctx.openViolenceTest(); a.ctx.renderViolenceJob(done);
  assert.equal(a.element('violence-source').textContent,'คลิปที่ทดสอบ: '+done.filename);
  assert.match(a.element('violence-result').textContent,/80.00%/);
  a.ctx.currentSource='new.mp4';
  assert.match(a.element('violence-source').textContent,/<img/,'result must retain original clip label');
  await a.ctx.openViolenceTest();
  assert.equal(a.element('violence-result').textContent,'','opening the test for a new clip must discard the old result');
  assert.match(a.element('violence-source').textContent,/new.mp4/);
  let resolve;
  const b=fixture(()=>new Promise(r=>resolve=r));
  const pending=b.ctx.pollViolenceJob('original',0);
  b.ctx.closeViolenceTest(); resolve(reply(done)); await pending;
  assert.equal(b.element('violence-result').textContent,'','closed/stale poll must not overwrite UI');
  const c=fixture(async()=>reply({detail:'GPU unavailable'},false));
  await c.ctx.runViolenceClip();
  assert.equal(c.element('violence-message').textContent,'GPU unavailable');
  assert.equal(c.element('violence-result').textContent,'');
  console.log('PASS: LSTM result scope, stale polling and errors');
})().catch(error=>{console.error(error);process.exitCode=1;});
