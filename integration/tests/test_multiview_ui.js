const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(require('node:path').join(__dirname,'../../web/static/multiview.js'),'utf8');
let clock=0,serial=0,posts=0;const timers=new Map();
class Recorder{
  static isTypeSupported(){return true;}
  constructor(){this.state='inactive';}
  start(){this.state='recording';}
  stop(){this.state='inactive';this.ondataavailable({data:new Blob(['data'])});queueMicrotask(()=>this.onstop());}
}
const context=vm.createContext({console,Map,Blob,FormData,MediaRecorder:Recorder,
  performance:{now:()=>clock},window:{MediaRecorder:Recorder,addEventListener(){}},
  setInterval(){},requestAnimationFrame(){},setTimeout(fn){const id=++serial;timers.set(id,fn);return id;},clearTimeout:id=>timers.delete(id),
  productRequest:async()=>[{id:'cam',session_id:'one'}],fetch:async()=>{posts++;return {ok:true};},
  document:{getElementById:()=>({textContent:''})}});
vm.runInContext(source,context);
vm.runInContext("multiCards.set('cam',{session:null,status:{textContent:''}})",context);
(async()=>{
  const session={media:{},started:0,closed:false};await context.beginRecordingSegment('cam',session);
  for(let segment=1;segment<=3;segment++){
    const timer=session.recordTimer;clock=segment*10000;
    assert(timers.has(timer),'next recording segment must keep its own timer');
    timers.get(timer)();await Promise.resolve();await Promise.resolve();
    assert(timers.has(session.recordTimer),'previous onstop must not cancel the next segment');
  }
  assert.equal(posts,3);
  session.closed=true;timers.get(session.recordTimer)();await Promise.resolve();
  assert.equal(session.recorder.state,'inactive');
  console.log('PASS: continuous independent 10s webcam segments across multiple recorder restarts');
})().catch(e=>{console.error(e);process.exitCode=1;});
