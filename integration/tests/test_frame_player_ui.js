const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const nodes=new Map();
function node(id){if(!nodes.has(id))nodes.set(id,{src:'',duration:3,currentTime:0,paused:true,
  focus(){},pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();},load(){},removeAttribute(){this.src='';},
  getContext:()=>({clearRect(){}}),classList:{add(){},remove(){}},textContent:''});return nodes.get(id);}
let imageId=0,frames=[],acks=[],animation=[],pendingBlob=null;
const context=vm.createContext({console,crypto:{randomUUID:()=> 'client-123'},AbortController,
  localStorage:{getItem(){return null;},setItem(){}},window:{addEventListener(){}},
  document:{getElementById:node,addEventListener(){}},feedImg:node('feed'),placeholder:node('placeholder'),paintEvidence(){},
  URL:{createObjectURL:()=> 'blob:'+ (++imageId),revokeObjectURL(){}},
  requestAnimationFrame:callback=>animation.push(callback),setInterval(){},
  showToast(){},loadingStatus(){},closeReview(){},reviewView:{archive:false},
  refreshFeedImage(){node('feed').src='mjpeg';},startStreamWithSource(){},stopStream(){},togglePlay(){},pauseStream(){},fetchStatus:async()=>{},
  fetch:async url=>{frames.push(url);return {ok:true,status:200,headers:{get:key=>({'X-Frame-Sequence':'1','X-Frame-Id':'1','X-Frame-Time':'.04'})[key]},blob:()=>pendingBlob?new Promise(resolve=>pendingBlob=resolve):Promise.resolve({})};},
  productRequest:async(url,body)=>{if(url.endsWith('/ack'))acks.push(body);return {status:'resumed'};}
});
context.window.syncReview=()=>{};
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../../web/static/frame-player.js'),'utf8'),context);
async function settle(){for(let i=0;i<12;i++)await Promise.resolve();}
async function paintUntil(done){
  for(let i=0;i<10 && !done();i++){assert(animation.length,'expected a pending paint callback');animation.shift()();await settle();}
  assert(done(),'frame rendering did not complete');
}
const watchdog=setTimeout(()=>{console.error('FAIL: frame player test did not complete');process.exitCode=1;},3000);
(async()=>{
  context.startFramePlayer({session_id:'run'});await settle();
  assert.equal(frames.length,1);assert.equal(acks.length,0,'must not acknowledge a downloaded image before it loads');
  context.refreshFeedImage();assert(context.feedImg.src.startsWith('blob:'),'status polling must not replace the frame with MJPEG');
  context.feedImg.onload();await settle();assert.equal(acks.length,0,'must wait for a render opportunity after decode');
  animation.shift()();await settle();assert.equal(acks.length,0,'must allow the first paint before advancing');
  animation.shift()();await settle();assert.equal(acks.length,1);assert.equal(acks[0].sequence,1);
  assert(node('playback-progress').textContent.includes('เฟรม 1'));
  // Run both production modules together: event replay must keep the real
  // frame receiver alive and ACK its next image while the modal is open.
  vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../../web/static/event-replay.js'),'utf8'),context);
  const originalRequest=context.productRequest;
  context.productRequest=async(url,body)=>url.startsWith('/api/events/')?
    (url.endsWith('/playback')?{url:'event.mp4',time_s:1,evidence_url:'/api/events/one/annotations'}:{frames:[{time_s:1,persons:[]}]}):originalRequest(url,body);
  await context.playEvent('one');await node('event-replay-video').onloadedmetadata();
  assert.equal(context.window.framePlayerActive,true);assert(context.feedImg.src.startsWith('blob:'));
  context.fetch=async()=>({ok:true,status:200,headers:{get:key=>({'X-Frame-Sequence':'2','X-Frame-Id':'2','X-Frame-Time':'.08'})[key]},blob:async()=>({})});
  const next=context.receiveFrame(vm.runInContext('framePlayer',context));await settle();
  context.feedImg.onload();await settle();await paintUntil(()=>acks.at(-1).sequence===2);await next;
  assert.equal(acks.at(-1).sequence,2,'main inference must advance while event replay is open');
  context.closeEventReplay();assert.equal(context.window.framePlayerActive,true,'closing replay cannot close the frame receiver');
  const countBeforePause=acks.length;
  context.closeFramePlayer();
  context.startFramePlayer({session_id:'paused'});await settle();
  vm.runInContext('framePlayer.paused=true',context);
  context.feedImg.onload();await settle();await paintUntil(()=>!vm.runInContext('framePlayer.busy',context));
  assert.equal(acks.length,countBeforePause,'manual pause must hold the displayed frame without advancing inference');
  context.fetch=async()=>({ok:true,status:200,headers:{get:key=>({'X-Frame-Sequence':'3','X-Frame-Id':'3','X-Frame-Time':'.12'})[key]},blob:()=>new Promise(resolve=>pendingBlob=resolve)});
  pendingBlob=true;await context.pauseStream();await settle();assert.equal(acks.length,countBeforePause+1,'resume must release the held frame');
  context.closeFramePlayer();pendingBlob({});await settle();assert.equal(acks.length,countBeforePause+1,'stale downloads must not acknowledge a replacement run');
  context.window.syncReview({frame_by_frame:true,is_running:true,session_id:'other',playback_client:'other-client'});
  assert(node('playback-progress').textContent.includes('อีก') || node('playback-progress').textContent.includes('แท็บ'));
  console.log('PASS: frame display/decode/render acknowledgement, pause/resume, feed protection and stale-session cleanup');
  clearTimeout(watchdog);
})().catch(error=>{clearTimeout(watchdog);console.error(error);process.exitCode=1;});
