const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const nodes=new Map();
function element(id){
  if(!nodes.has(id)){
    const classes=new Set();
    nodes.set(id,{src:'',currentTime:0,duration:10,paused:true,focus(){},
      classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c)},
      getContext:()=>({clearRect(){}}),play(){this.paused=false;return Promise.resolve();},
      pause(){this.paused=true;},load(){},removeAttribute(){this.src='';}});
  }
  return nodes.get(id);
}
const requests=[],painted=[];let mainSync=0;
const main={source:'main.mp4',session:'main-run',frame:12};
const context=vm.createContext({console,document:{getElementById:element,activeElement:{focus(){}},addEventListener(){}},
  window:{addEventListener(){},framePlayerActive:true,syncReview:()=>{mainSync++;main.frame++;}},
  requestAnimationFrame(){},setInterval(){},
  paintEvidence:(canvas,video,rows,seconds)=>painted.push({rows,seconds}),
  productRequest:async(url,body)=>{requests.push({url,body});return url.endsWith('/playback')?
    {url:'event.mp4',time_s:2,evidence_url:'/api/events/one/annotations'}:
    {frames:[{time_s:2,persons:[{track_id:7,alert:true}]}]};}
});
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../../web/static/event-replay.js'),'utf8'),context);
(async()=>{
  await context.playEvent('one');
  assert.equal(element('event-replay-video').src,'event.mp4');
  await element('event-replay-video').onloadedmetadata();
  context.paintEventReplay();
  assert.equal(painted.at(-1).rows[0].persons[0].track_id,7);
  context.window.syncReview({is_running:true,frame_by_frame:true,current_frame:13});
  assert.equal(mainSync,1);assert.equal(main.frame,13);assert.equal(context.window.framePlayerActive,true);
  assert.equal(main.source,'main.mp4');assert.equal(main.session,'main-run');
  assert(element('event-replay-main-status').textContent.includes('กำลังประมวลผล'));
  assert(requests.every(call=>!call.body && call.url.startsWith('/api/events/')),'replay must not start/stop/pause/seek/ACK the main worker');
  context.closeEventReplay();assert(element('event-replay-video').paused);
  assert.equal(context.window.framePlayerActive,true);
  assert.equal(element('event-replay-video').src,'');
  let finish;context.productRequest=()=>new Promise(resolve=>finish=resolve);
  const delayed=context.playEvent('old');context.closeEventReplay();finish({url:'old.mp4'});await delayed;
  assert.equal(element('event-replay-video').src,'','closed modal must reject a delayed replay reply');
  context.productRequest=async()=>({url:'broken.avi'});await context.playEvent('bad');element('event-replay-video').onerror();
  assert(element('event-replay-status').textContent.includes('ตัวเล่นหลักยังใช้ได้'));
  assert.equal(context.window.framePlayerActive,true);
  console.log('PASS: independent event replay, saved evidence, main processing continuity, close/error/stale-response safety');
})().catch(error=>{console.error(error);process.exitCode=1;});
