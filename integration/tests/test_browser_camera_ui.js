const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict'), path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../../web/static/multiview.js'),'utf8');
const html=fs.readFileSync(path.join(__dirname,'../../web/static/index.html'),'utf8');
function classes(){const names=new Set();return {add:n=>names.add(n),remove:n=>names.delete(n),contains:n=>names.has(n)};}
function video(){return {muted:false,playsInline:false,readyState:2,videoWidth:640,videoHeight:480,style:{},classList:classes(),
  setAttribute(){},play:async()=>{},pause(){this.paused=true;},remove(){this.removed=true;}};}
function fixture(){
  const messages=[],cards=new Map(),posts=[],nodes=new Map(),tracks=[{stop(){this.stopped=true;}}];let raw;
  const context=vm.createContext({console,Map,Set,Blob,FormData,performance:{now:()=>Date.now()},
    setTimeout,clearTimeout,setInterval(){},requestAnimationFrame(){},
    window:{addEventListener(){}},navigator:{mediaDevices:{getUserMedia:async()=>({getTracks:()=>tracks}),
      enumerateDevices:async()=>[{kind:'videoinput',deviceId:'first',label:'one'},{kind:'videoinput',deviceId:'second',label:'two'}]}},
    document:{body:{append(v){v.attached=true;}},getElementById(id){if(!nodes.has(id))nodes.set(id,{value:'cpu',textContent:'',children:[],classList:classes(),replaceChildren(){this.children=[];},append(...items){this.children.push(...items);}});return nodes.get(id);},
      createElement(type){if(type==='video'){raw=video();return raw;}if(type==='canvas')return {getContext:()=>({drawImage(){assert(raw.readyState>=2);}}),toBlob:fn=>fn(new Blob(['jpeg']))};return {value:'',children:[],setAttribute(){},append(item){this.children.push(item);}};}},
    productRequest:async url=>url==='/api/runtime/options'?{max_streams:4}:url==='/api/multistream'?[{id:'cam',session_id:'sid',is_running:true,frame_ready:true,analysis_sec:1,device:'cpu'}]:{},
    fetch:async(url,options)=>{posts.push({url,options});return {ok:true};},loadCameras:async()=>{}});
  vm.runInContext(source,context);
  context.multiMessage=text=>messages.push(text);
  context.newStreamProfile=async()=>({id:'cam',name:'camera'});
  context.addMultiCard=(profile,source,live)=>{
    const card={profile,source,live,video:video(),img:{src:'',classList:classes(),removeAttribute(){this.src='';}},
      status:{textContent:''},card:{remove(){}}};cards.set(profile.id,card);context.card=card;
    vm.runInContext("multiCards.set('cam',card)",context);return card;
  };
  return {context,posts,messages,tracks,cards,get raw(){return raw;}};
}
async function flush(){for(let i=0;i<12;i++)await Promise.resolve();}
(async()=>{
  assert(html.includes('id="main-camera-selector"'),'primary camera discovery has a dedicated control');
  assert(!html.includes('<option value="0">'),'default selector must use the client webcam, not container camera 0');
  const a=fixture();
  await a.context.startBrowserCamera({deviceId:'device',label:'camera'},0);await flush();
  const session=vm.runInContext("browserSessions.get('cam')",a.context);
  assert(session && !session.closed,'unsupported recording must not close the camera');
  assert(a.posts.some(p=>p.url.endsWith('/frame')),'frames must upload even without MediaRecorder');
  assert(a.cards.get('cam').recordingError,'recording failure must remain visible');
  assert(a.raw.attached,'frame source must be attached to the page');
  await a.context.pollMulti();const card=a.cards.get('cam');
  assert(!card.video.classList.contains('hidden'),'keep local preview until an AI image actually loads');
  card.img.onload();assert(card.video.classList.contains('hidden'));
  card.img.onerror();assert(!card.video.classList.contains('hidden'),'failed AI feed must restore local preview');
  await a.context.stopBrowserSession('cam');await session.frameTask;
  assert(a.tracks[0].stopped && a.raw.removed);

  const b=fixture(), delayed=video();delayed.readyState=1;delayed.videoWidth=0;delayed.videoHeight=0;
  setTimeout(()=>{delayed.readyState=2;delayed.videoWidth=640;delayed.videoHeight=480;},5);
  await b.context.waitBrowserVideo(delayed,1000);assert.equal(delayed.videoWidth,640);
  const empty=video();empty.readyState=1;empty.videoWidth=0;
  await assert.rejects(b.context.waitBrowserVideo(empty,1),/webcam ไม่ส่งภาพ/);
  assert.match(b.context.cameraErrorMessage({name:'NotAllowedError'}),/สิทธิ์กล้อง/);
  assert.match(b.context.cameraErrorMessage({name:'NotReadableError'}),/แอปอื่น/);
  const d=fixture();let chosen;
  d.context.startBrowserCamera=(device,index)=>chosen={device,index};
  await d.context.listBrowserCameras();
  const [select,button]=d.context.document.getElementById('browser-camera-list').children;
  assert.equal(select.children.length,2);select.value='1';button.onclick();
  assert.equal(chosen.device.deviceId,'second');assert.equal(chosen.index,1);
  console.log('PASS: browser webcam routing, frame readiness, recording-independent transport, preview fallback and cleanup');
})().catch(error=>{console.error(error);process.exitCode=1;});
