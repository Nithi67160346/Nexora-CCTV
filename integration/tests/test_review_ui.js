const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const code=fs.readFileSync(require('node:path').join(__dirname,'../../web/static/review.js'),'utf8');
const nodes=new Map();let createdElements=0;
function element(id){
  if(!nodes.has(id)){
    const classes=new Set(),draw=[],bones=[],joints=[],texts=[];
    nodes.set(id,{value:'',textContent:'',children:[],listeners:{},addEventListener(name,handler){this.listeners[name]=handler;},paused:true,currentTime:0,duration:10,readyState:4,videoWidth:640,videoHeight:480,clientWidth:640,clientHeight:480,
      classList:{add:(...c)=>c.forEach(v=>classes.add(v)),remove:(...c)=>c.forEach(v=>classes.delete(v)),toggle:(c,on)=>on?classes.add(c):classes.delete(c),contains:c=>classes.has(c)},
      pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();},load(){},removeAttribute(){},querySelector(){return {};},
      replaceChildren(...c){this.children=c;},append(...c){this.children.push(...c);},
      getContext(){return {clearRect(){},strokeRect(){draw.push(this.strokeStyle);},fillText(text){texts.push(text);},
        beginPath(){this.path=[];},moveTo(x,y){this.path.push([x,y]);},lineTo(x,y){this.path.push([x,y]);},
        stroke(){bones.push({color:this.strokeStyle,points:this.path});},arc(x,y){this.point=[x,y];},fill(){joints.push(this.point);}};},draw,bones,joints,texts});
  }return nodes.get(id);
}
let calls=[],startSuccess=true,timerIndex=0;const networkTimers=new Map();
const context=vm.createContext({console,performance:{now:()=>1000},setInterval(){},requestAnimationFrame(){},window:{},
  setTimeout(callback){const id=++timerIndex;networkTimers.set(id,callback);return id;},clearTimeout(id){networkTimers.delete(id);},
  document:{getElementById:element,createElement:tag=>element('new-'+tag+'-'+(++createdElements))},
  feedImg:element('feed'),placeholder:element('placeholder'),seekSlider:element('slider'),timeCurrent:element('current'),timeDuration:element('duration'),btnPlay:element('play'),isUserDraggingSlider:false,
  formatTime:n=>String(n),showToast(){},refreshFeedImage(){},startStreamWithSource:async()=>startSuccess,
  togglePlay(){calls.push('oldPlay');},pauseStream(){},onSeekSliderChange(){calls.push('oldSeek');},seekDelta(){},setSpeed(){calls.push('oldSpeed');},stopStream(){calls.push('oldStop');},fetchEvents:async()=>{},fetchStatus:async()=>{},
  productRequest:async url=>{
    calls.push(url);
    if(url==='/api/runtime/options')return {gpu_available:false,ffmpeg_available:false};
    if(url==='/api/status')return {is_running:true,is_live:false,current_source:'clip.mp4'};
    if(url.includes('/playback'))return {source:'original.mp4',url:'/api/playback/media?source=original.mp4',time_s:3};
    return {};
  }});
vm.runInContext(code,context);
(async()=>{
  context.openReview('clip.mp4');
  await context.setSpeed(2);assert.equal(element('native-video').playbackRate,2);assert(!calls.includes('oldSpeed'));
  context.openReview('next.mp4');assert.equal(element('native-video').playbackRate,1,'new clips must not inherit a previous 2x rate');
  assert(element('speed-10').classList.contains('bg-brand-500'),'1x button must match actual playback rate');
  element('native-video').paused=true;
  await context.onSeekSliderChange(60);assert.equal(element('native-video').currentTime,6);assert(element('native-video').paused);assert(!calls.includes('oldSeek'));
  context.beginUiOperation();context.loadingStatus(true,'อัปโหลด',50);
  context.closeReview();context.window.syncReview({is_running:true,frame_ready:true});
  assert(!element('loading-status').classList.contains('hidden'),'status polling must not erase upload progress');
  context.endUiOperation();assert(element('loading-status').classList.contains('hidden'));
  startSuccess=false;calls=[];await context.startStreamWithSource('broken.mp4');
  assert(!calls.includes('/api/status'),'failed start must not open another existing stream');
  await context.playEvent('event1');assert.equal(element('native-video').src,'/api/playback/media?source=original.mp4');
  element('native-video').onloadedmetadata();assert.equal(element('native-video').currentTime,3);
  calls=[];await context.stopStream();assert(!calls.includes('oldStop'),'closing an event replay must keep live monitoring running');
  context.openReview('segment.avi','/api/recordings/id/media',0,true);element('native-video').onerror();
  assert(element('feed').classList.contains('hidden'),'failed archive must not show unrelated live feed');
  assert.equal(element('recording-list').children[0].href,'/api/recordings/id/media');
  calls=[];context.openReview('unsupported.avi');await element('native-video').onerror();
  assert(calls.includes('/api/playback/paced'),'AI fallback must switch to a paced source clock instead of full-speed analysis');calls=[];
  await context.setSpeed(2);assert(!calls.includes('oldSpeed'));
  assert(element('speed-20').disabled,'unsupported codecs must not advertise fake 2x');
  const canvas=element('evidence'),video=element('native-video');
  context.paintEvidence(canvas,video,[{time_s:1,persons:[{track_id:7,bbox_xyxy:[1,1,40,50],fall:{backend:'yolo_pose_rf_v2',phase:'FALL_DETECTED',fall_score:.8}}]}],1);
  assert(canvas.texts.some(text=>text.includes('คน #7') && text.includes('80.0%')),'per-person Fall V2 score must be visible on original playback frame');
  canvas.draw.length=0;
  context.paintEvidence(canvas,video,[{time_s:1,persons:[],violence:{result:{fighting:true,probability_fighting:.998}}}],1);
  assert(canvas.texts.some(text=>text.includes('99.8%')),'scene LSTM score must appear even without person IDs');
  const textCount=canvas.texts.length;
  context.paintEvidence(canvas,video,[{time_s:1,persons:[],violence:{result:{fighting:true,probability_fighting:.998}}}],5);
  assert.equal(canvas.texts.length,textCount,'scene result must not appear at another playback time');
  context.paintEvidence(canvas,video,[{time_s:1,persons:[{track_id:1,bbox_xyxy:[1,1,40,50],alert:true}]}],1);
  assert.deepEqual(canvas.draw,['#ff3333']);
  context.paintEvidence(canvas,video,[{time_s:1,persons:[{track_id:1,bbox_xyxy:[1,1,40,50],alert:true}]}],5);
  assert.equal(canvas.draw.length,1,'never draw evidence from a different playback time');
  let finishOptions;context.productRequest=()=>new Promise(resolve=>finishOptions=resolve);
  const pending=context.initializeRuntimeChoices();element('inference-device').value='cpu';element('inference-device').listeners.change();
  finishOptions({gpu_available:true,ffmpeg_available:false,current_device:'cuda',models:[{name:'yolo26n-pose.pt',label:'n',available:true}]});await pending;
  assert.equal(element('inference-device').value,'cpu','late startup options must not overwrite the user CPU choice');
  context.productRequest=async()=>({gpu_available:true,current_model:'yolo26n-pose.pt',models:[
    {name:'yolo26n-pose.pt',label:'n',available:true},
    {name:'yolo26s-pose.pt',label:'s',available:false},
    {name:'yolo26m-pose.pt',label:'m',available:true}]});
  await context.initializeRuntimeChoices();
  assert.deepEqual(element('pose-model').children.map(o=>[o.value,o.disabled]),[
    ['yolo26n-pose.pt',false],['yolo26s-pose.pt',true],['yolo26m-pose.pt',false]]);
  assert(element('runtime-choice-message').textContent.includes('YOLO26s-pose'));
  context.productRequest=async()=>({gpu_available:true,current_model:'yolo26s-pose.pt',models:[
    {name:'yolo26s-pose.pt',label:'26s',available:true},
    {name:'yolov8s-pose.pt',label:'v8s',available:false}]});
  await context.initializeRuntimeChoices();
  assert.equal(element('pose-model').children[1].disabled,true,'missing v8 weights cannot be selected');
  context.productRequest=async()=>({gpu_available:true,current_model:'yolov8s-pose.pt',models:[
    {name:'yolo26s-pose.pt',label:'26s',available:true},
    {name:'yolov8s-pose.pt',label:'v8s',available:true}]});
  await context.initializeRuntimeChoices();
  assert.equal(element('pose-model').children[1].disabled,false,'downloaded v8 weights become available on reload');
  assert.equal(element('pose-model').value,'yolov8s-pose.pt');
  const supported=['yolo26n','yolo26s','yolo26m','yolov8n','yolov8s','yolov8m'];
  context.productRequest=async()=>({gpu_available:true,current_model:'yolo26s-pose.pt',models:
    supported.map(name=>({name:name+'-pose.pt',label:'คำเปรียบเทียบเดิม',available:true}))});
  await context.initializeRuntimeChoices();
  assert(element('pose-model').children.every(option=>!option.disabled),'all six installed models must be selectable');
  assert.deepEqual(element('pose-model').children.map(o=>o.textContent),
    ['YOLO26n-pose','YOLO26s-pose','YOLO26m-pose','YOLOv8n-pose','YOLOv8s-pose','YOLOv8m-pose']);
  const skeleton=element('skeleton');skeleton.clientWidth=400;skeleton.clientHeight=400;
  const pose=Array.from({length:17},()=>[0,0,0]);
  pose[0]=[150,50,.8];pose[5]=[100,100,.9];pose[6]=[200,100,.9];
  pose[7]=[300,300,.2];pose[8]=[NaN,100,.9];pose[16]=[600,100,.9];
  const evidence=[{time_s:2,persons:[{track_id:1,bbox_xyxy:[0,0,400,400],pose,alert:true}]}];
  context.paintEvidence(skeleton,video,evidence,2);
  assert.equal(skeleton.bones.length,3,'draw only confident in-bounds bone endpoints');
  assert.equal(skeleton.joints.length,3,'masked, weak, nonfinite and out-of-box joints must be hidden');
  assert(skeleton.bones.some(b=>JSON.stringify(b.points)===JSON.stringify([[62.5,112.5],[125,112.5]])),
    'skeleton must align with the video including letterbox offsets');
  assert.equal(skeleton.draw[0],'#ff3333','adding skeletons must preserve red alert boxes');
  context.paintEvidence(skeleton,video,evidence,4);assert.equal(skeleton.bones.length,3,'never draw stale pose');
  await context.setClipLoop(true);context.openReview('loop.mp4');assert.equal(element('native-video').loop,true);
  await context.setClipLoop(false);assert.equal(element('native-video').loop,false);
  context.window.syncReview({is_running:true,is_live:true});assert.equal(element('clip-loop').disabled,true);
  context.window.syncReview({is_running:true,is_live:false,qa:{state:'running'}});assert.equal(element('clip-loop').disabled,true);
  // First play waits for actual timestamped evidence, including valid empty-person results.
  video.duration=120; // Long clips buffer the initial section, not the entire file.
  video.ended=false;
  context.productRequest=async()=>({source:'cctv.mp4',session_id:'one',analysis_sec:.5,
    frames:[{time_s:1/30,persons:[]},{time_s:.5,persons:[]}]});
  context.openReview('cctv.mp4',undefined,0,false,'one');video.onloadedmetadata();
  assert(video.paused,'do not play before the first AI result');
  assert(!element('loading-status').classList.contains('hidden'));
  context.beginUiOperation();context.endUiOperation();
  assert(!element('loading-status').classList.contains('hidden'),'ending upload must retain the AI waiting indicator');
  await context.refreshReviewEvidence();assert(!video.paused,'processed empty frames are valid evidence, not a stuck loading state');
  video.readyState=1;video.onwaiting();assert(element('loading-status').classList.contains('hidden'),'brief loop/decoder buffering must not flash a loading overlay');
  video.onplaying();assert.equal(networkTimers.size,0,'resuming playback must cancel delayed network loading');
  video.onwaiting();const networkTimer=[...networkTimers.entries()][0];networkTimers.delete(networkTimer[0]);networkTimer[1]();
  assert(!element('loading-status').classList.contains('hidden'),'sustained video buffering must remain visible');
  video.readyState=4;video.oncanplay();assert(element('loading-status').classList.contains('hidden'));
  const initialEvidenceRequest=context.productRequest;
  calls=[];context.productRequest=async url=>{calls.push(url);return {source:'cctv.mp4',session_id:'one',analysis_sec:3.2,frames:[]};};
  await context.setSpeed(2);video.currentTime=1.5;
  context.window.syncReview({is_running:true,is_live:false,review_mode:true,current_source:'cctv.mp4',session_id:'one',analysis_sec:3.2,stage:'analysis_complete'});
  for(let i=0;i<10;i++){video.currentTime+=.05;await context.refreshReviewEvidence();assert(!video.paused,'delayed evidence must not repeatedly stop completed-clip playback');}
  assert.equal(video.playbackRate,2,'continuous playback must preserve the chosen speed');
  video.loop=true;video.currentTime=0;await context.refreshReviewEvidence();
  assert(!video.paused,'loop wrap must not restart the AI waiting cycle');video.loop=false;
  context.window.syncReview({is_running:true,is_live:false,review_mode:true,current_source:'cctv.mp4',session_id:'one',analysis_sec:.1,stage:'running'});
  await context.refreshReviewEvidence();assert(!video.paused,'ongoing slow AI must not pause a playing video');
  assert(!calls.includes('/api/playback/analyze-near'),'overlay delay must never rerun inference automatically');
  context.productRequest=initialEvidenceRequest;video.currentTime=0;await context.setSpeed(1);
  context.togglePlay();await context.refreshReviewEvidence();assert(video.paused,'polling must preserve a manual pause');
  context.openReview('cctv.mp4',undefined,0,false,'one');video.onloadedmetadata();
  context.setWaitForAI(false);assert(!video.paused,'user can watch immediately without waiting for inference');
  context.setWaitForAI(true);assert(!video.paused,'enabling startup wait during playback must not stop the current video');
  await context.refreshReviewEvidence();assert(!video.paused);
  video.ended=true;video.onended();video.pause();await context.refreshReviewEvidence();
  assert(video.paused,'finished playback must not restart with loop disabled');video.ended=false;
  // A delayed response from the old run must never become evidence for a new model/session.
  let finishEvidence;context.productRequest=()=>new Promise(resolve=>finishEvidence=resolve);
  const stale=context.refreshReviewEvidence();context.openReview('cctv.mp4',undefined,0,false,'two');video.onloadedmetadata();
  finishEvidence({source:'cctv.mp4',session_id:'one',analysis_sec:10,frames:[{time_s:.033,persons:[]},{time_s:.5,persons:[]}]});
  await stale;assert(video.paused);
  // Seeking into cached evidence does not restart inference; missing evidence requests preceding context.
  context.window.syncReview({is_running:true,is_live:false,review_mode:true,current_source:'cctv.mp4',session_id:'two',analysis_sec:10});
  calls=[];context.productRequest=async(url)=>{
    calls.push(url);const sec=Number(url.split('seconds=')[1]);
    return {source:'cctv.mp4',session_id:'two',analysis_sec:10,frames:[{time_s:sec,persons:[]},{time_s:sec+.5,persons:[]}]};
  };
  await context.seekReview(6);assert(!calls.includes('/api/playback/analyze-near'));assert(!video.paused,'cached seek must resume directly');
  calls=[];context.productRequest=async(url)=>{calls.push(url);return {source:'cctv.mp4',session_id:'two',frames:[]};};
  await context.seekReview(8);assert(calls.includes('/api/playback/analyze-near'));assert(video.paused,'uncached explicit seek must buffer once at its destination');
  context.openReview('cctv.mp4',undefined,0,false,'two','new-config');video.onloadedmetadata();
  context.productRequest=async()=>({source:'cctv.mp4',session_id:'two',evidence_revision:'old-config',
    analysis_sec:10,frames:[{time_s:.033,persons:[]},{time_s:.5,persons:[]}]});
  await context.refreshReviewEvidence();assert(video.paused,'old feature settings must not become current evidence');
  // Browsers report ended=false as soon as looping is enabled, even at the last frame.
  Object.defineProperty(video,'ended',{configurable:true,get(){return this.currentTime>=this.duration && !this.loop;}});
  video.loop=false;video.currentTime=video.duration;video.pause();video.onended();
  context.window.syncReview({is_running:true,is_live:false,review_mode:true,current_source:'cctv.mp4',session_id:'two',evidence_revision:'new-config',analysis_sec:10,stage:'analysis_complete'});
  context.productRequest=async()=>({source:'cctv.mp4',session_id:'two',evidence_revision:'new-config',
    analysis_sec:10,frames:[{time_s:.033,persons:[]},{time_s:.5,persons:[]}]});
  await context.setClipLoop(true);assert.equal(video.currentTime,0,'enabling loop at EOF must rewind before ended becomes false');
  await context.refreshReviewEvidence();assert(!video.paused,'enabling loop after EOF must restore play intent');
  await context.setClipLoop(false);
  // Short clips finish one chronological analysis before playback; repeated
  // native viewing must not start another inference pass.
  video.duration=5;context.openReview('short.mp4',undefined,0,false,'short-run');video.onloadedmetadata();
  calls=[];context.productRequest=async url=>{calls.push(url);return {source:'short.mp4',session_id:'short-run',stage:'running',analysis_sec:1,
    frames:[{time_s:.033,persons:[]},{time_s:.5,persons:[]}]};};
  await context.refreshReviewEvidence();assert(video.paused,'short first play must wait for complete analysis, not only .3s ahead');
  context.productRequest=async url=>{calls.push(url);return {source:'short.mp4',session_id:'short-run',stage:'analysis_complete',analysis_sec:5,
    frames:[{time_s:.033,persons:[]},{time_s:.5,persons:[]}]};};
  await context.refreshReviewEvidence();assert(!video.paused,'short clip starts after the first complete analysis');
  assert(!calls.includes('/api/playback/analyze-near'));
  // Event replay reads the original event evidence without starting/stopping AI.
  context.productRequest=async url=>{calls.push(url);return url.endsWith('/playback')?
    {source:'event.mp4',url:'/api/playback/media?source=event.mp4',time_s:2,evidence_url:'/api/events/event2/annotations'}:
    {event_id:'event2',frames:[{time_s:2,persons:[{track_id:1,bbox_xyxy:[1,1,40,50],pose,alert:true}]}]};};
  calls=[];await context.playEvent('event2');video.onloadedmetadata();await context.refreshReviewEvidence();
  assert(!element('ai-playback-overlay').classList.contains('hidden'),'event replay must show original timestamped overlays');
  assert.equal(context.matchingReviewRow().persons[0].alert,true);
  assert(calls.includes('/api/events/event2/annotations?seconds=2'));
  assert(!calls.some(url=>url.includes('analyze-near') || url==='/api/start'),'event replay must never run inference again');
  // Old polling replies near a new seek position must not replace fresher rows.
  let finishBeforeSeek;context.productRequest=()=>new Promise(resolve=>finishBeforeSeek=resolve);
  const beforeSeek=context.refreshReviewEvidence();
  context.productRequest=async()=>({event_id:'event2',frames:[{time_s:2.2,persons:[]}]});
  await context.seekReview(2.2);
  finishBeforeSeek({event_id:'event2',frames:[{time_s:2,persons:[]}]});await beforeSeek;
  assert.equal(context.matchingReviewRow().time_s,2.2,'pre-seek reply cannot clear the new overlay');
  await context.stopStream();
  console.log('PASS: native 2x, paused scrub, loading race, failed start, event replay source, codec fallback and timestamped red evidence');
})().catch(error=>{console.error(error);process.exitCode=1;});
