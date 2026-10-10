const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.join(__dirname,'../../web/static');
function element(){const names=new Set();return {value:'cpu',textContent:'',src:'',style:{},children:[],readyState:2,videoWidth:640,videoHeight:480,
  classList:{add:n=>names.add(n),remove:n=>names.delete(n),contains:n=>names.has(n),toggle(n,on){if(on)names.add(n);else names.delete(n);}},
  listeners:{},addEventListener(name,fn){this.listeners[name]=fn;},setAttribute(){},removeAttribute(name){this[name]='';},
  append(...items){this.children.push(...items);},replaceChildren(){this.children=[];},play:async()=>{},pause(){this.paused=true;},remove(){this.removed=true;}};}
function fixture(){
  const nodes=new Map(),posts=[],tracks=[],starts=[];let running=false,current='',sid='';
  const node=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  const feedImg=node('stream-feed-img'),placeholder=node('video-placeholder');
  const status=()=>({is_running:running,current_source:current,session_id:sid,device:'cpu',analysis_sec:2,frame_ready:false});
  const context=vm.createContext({console,Map,Set,Blob,FormData,performance:{now:()=>Date.now()},setTimeout,clearTimeout,setInterval(){},requestAnimationFrame(){},
    window:{activeCameraId:'room',addEventListener(){}},navigator:{mediaDevices:{getUserMedia:async constraints=>{const track={stop(){this.stopped=true;}};tracks.push(track);return {getTracks:()=>[track]};},
      enumerateDevices:async()=>[{kind:'videoinput',deviceId:'first',label:'one'},{kind:'videoinput',deviceId:'second',label:'two'}]}},
    document:{getElementById:node,body:{append(el){el.attached=true;}},createElement(type){if(type==='canvas')return {getContext:()=>({drawImage(){}}),toBlob:fn=>fn(new Blob(['jpeg']))};return element();}},
    feedImg,placeholder,reviewView:{active:false},currentSource:'',closeReview(){context.reviewView.active=false;},loadingStatus(){},refreshFeedImage(){feedImg.src='/api/stream/feed';},
    fetchStatus:async()=>context.syncMainBrowserCamera?.(status()),
    startStreamWithSource:async source=>{starts.push(source);return true;},stopStream:async()=>{running=false;},
    productRequest:async(url,body)=>{
      posts.push({url,body});if(url==='/api/status')return status();
      if(url==='/api/stream/start'){running=true;current=body.source;sid='session-'+posts.length;return {session_id:sid,camera_id:'room'};}
      if(url==='/api/stream/stop'){running=false;return {};}
      return [];
    },fetch:async(url,options)=>{posts.push({url,options});return {ok:true};}});
  vm.runInContext(fs.readFileSync(path.join(root,'multiview.js'),'utf8'),context);
  vm.runInContext(fs.readFileSync(path.join(root,'main-webcam.js'),'utf8'),context);
  return {context,node,posts,tracks,starts,status};
}
async function flush(){for(let i=0;i<16;i++)await Promise.resolve();}
(async()=>{
  const f=fixture();await f.context.listMainBrowserCameras();
  const select=f.node('main-camera-selector');assert.equal(select.children.length,2);select.value='second';
  await f.context.openSelectedMainCamera();await flush();
  const session=vm.runInContext('mainCameraSession',f.context);
  assert.equal(session.device,'second');assert.equal(f.context.currentSource,'browser://second');
  assert(f.posts.some(p=>p.url==='/api/stream/start' && p.body.decode_device==='cpu' && p.body.review_mode===false));
  assert(f.posts.some(p=>p.url==='/api/stream/frame'),'primary camera must upload to primary worker');
  assert(!f.posts.some(p=>p.url.includes('/api/multistream/')),'primary camera must not create a secondary worker');
  assert(session.card.recordingError && !session.closed,'unsupported recording must not stop primary frame transport');
  assert(!f.node('main-camera-preview').classList.contains('hidden'),'show webcam while AI feed loads');
  f.node('stream-feed-img').listeners.load();f.context.syncMainBrowserCamera(f.status());
  assert(f.node('main-camera-preview').classList.contains('hidden'));assert(!f.node('stream-feed-img').classList.contains('hidden'));
  f.node('stream-feed-img').listeners.error();assert(!f.node('main-camera-preview').classList.contains('hidden'));
  f.context.reviewView.active=true;f.context.syncMainBrowserCamera(f.status());assert(f.node('main-camera-preview').classList.contains('hidden'),'archive must stay above live camera');
  f.context.reviewView.active=false;
  await f.context.startStreamWithSource('clip.mp4');await session.frameTask;
  assert(session.closed && session.raw.removed);assert(f.tracks.every(track=>track.stopped));assert.deepEqual(f.starts,['clip.mp4']);

  const g=fixture();await g.context.listMainBrowserCameras();await g.context.openSelectedMainCamera();await flush();
  const first=vm.runInContext('mainCameraSession',g.context);
  // Applying CPU/model settings restarts the primary camera with a new session.
  await g.context.startStreamWithSource(g.context.currentSource);await first.frameTask;
  const second=vm.runInContext('mainCameraSession',g.context);assert(second && second!==first);assert(first.closed);
  await g.context.stopStream();await second.frameTask;assert(second.closed && g.tracks.every(track=>track.stopped));

  const h=fixture();await h.context.listMainBrowserCameras();
  h.context.navigator.mediaDevices.getUserMedia=async()=>{const error=new Error();error.name='NotAllowedError';throw error;};
  assert.equal(await h.context.openSelectedMainCamera(),false);assert.match(h.node('main-camera-message').textContent,/สิทธิ์กล้อง/);
  assert(!h.posts.some(p=>p.url==='/api/stream/start'),'permission failure must not start primary worker');
  console.log('PASS: main webcam discovery, second device, primary frame transport, AI/preview, recording failure, source/restart/stop cleanup and permission errors');
})().catch(error=>{console.error(error);process.exitCode=1;});
