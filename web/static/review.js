/* Native playback has its own clock; AI evidence always keeps source timestamps. */
const reviewVideo=document.getElementById('native-video');
const reviewCanvas=document.getElementById('ai-playback-overlay');
let reviewView={active:false,archive:false,source:'',rows:[],analysis:0}, loadingSince=0;
let reviewFetchBusy=false;
let viewStartBusy=false;
let uiOperations=0;
let reviewNetworkTimer=null;
function clearReviewNetworkWait(){if(reviewNetworkTimer!==null){clearTimeout(reviewNetworkTimer);reviewNetworkTimer=null;}}
let waitForAI=true;
try{waitForAI=localStorage.getItem('nexora.waitForAI.v1')!=='false';}catch(_){}
document.getElementById('wait-for-ai').checked=waitForAI;
function matchingReviewRow(){
  const seconds=reviewVideo.currentTime || 0;
  const row=reviewView.rows.reduce((best,r)=>!best || Math.abs(r.time_s-seconds)<Math.abs(best.time_s-seconds)?r:best,null);
  return row && Math.abs(row.time_s-seconds)<=.1 ? row : null;
}
function updateAIPlayback(){
  if(!reviewView.active || reviewView.archive || !reviewView.metadataReady || !reviewView.userWantsPlay)return;
  if(reviewVideo.ended && !reviewVideo.loop){reviewView.userWantsPlay=false;return;}
  const seconds=reviewVideo.currentTime || 0;
  const complete=(reviewView.analysisComplete ||
    (lastReviewStatus.current_source===reviewView.source && lastReviewStatus.stage==='analysis_complete')) && reviewView.analysis>=seconds-.1;
  const ahead=reviewView.rows.some(row=>row.time_s>=seconds+.3);
  const fullClip=reviewView.initialOpen && Number.isFinite(reviewVideo.duration) && reviewVideo.duration>0 && reviewVideo.duration<=60;
  // Buffer again when playback catches the chronological analysis watermark.
  // Missing overlay responses alone must not stall an already analysed section.
  const caughtAI=!complete && seconds>=Math.max(0,reviewView.analysis-.05);
  if(waitForAI && (caughtAI || (reviewView.needsAIBuffer && (!matchingReviewRow() || (!complete && (fullClip || !ahead)))))){
    reviewView.needsAIBuffer=true;
    reviewView.waitingAI=true;reviewVideo.pause();
    loadingStatus(true,fullClip?`กำลังวิเคราะห์คลิปก่อนเล่น ${Math.min(100,Math.round(reviewView.analysis/reviewVideo.duration*100))}% • ดูผลได้ในรอบแรก`:`กำลังรอผล AI ช่วง ${formatTime(seconds)} • เมื่อพร้อมจะเล่นอัตโนมัติ`);
  }else{
    reviewView.needsAIBuffer=false;
    reviewView.initialOpen=false;
    if(reviewView.waitingAI || reviewVideo.paused){
      reviewView.waitingAI=false;loadingStatus(false);
      reviewVideo.play().catch(()=>{reviewView.userWantsPlay=false;showToast('กดเล่นเพื่อเริ่มคลิป');});
    }
  }
}
function requestReviewPlay(){
  if(reviewVideo.ended){reviewVideo.currentTime=0;reviewView.needsAIBuffer=true;}
  reviewView.userWantsPlay=true;
  if(reviewView.archive || !waitForAI){reviewView.waitingAI=false;reviewView.needsAIBuffer=false;loadingStatus(false);reviewVideo.play().catch(()=>showToast('กดเล่นเพื่อเริ่มคลิป'));return;}
  updateAIPlayback();
}
function setWaitForAI(enabled){
  waitForAI=enabled;document.getElementById('wait-for-ai').checked=enabled;
  try{localStorage.setItem('nexora.waitForAI.v1',String(enabled));}catch(_){}
  updateAIPlayback();
}
let clipLoopEnabled=false,lastReviewStatus={};
try{clipLoopEnabled=localStorage.getItem('nexora.clipLoop.v1')==='true';}catch(_){}
function refreshClipLoop(){
  const locked=!reviewView.archive && (lastReviewStatus.is_live || lastReviewStatus.qa?.state==='running');
  const box=document.getElementById('clip-loop');box.disabled=!!locked;box.checked=!locked && clipLoopEnabled;
  box.title=locked?'วนได้เฉพาะคลิปนอกการทดสอบ QA':'MP4/WebM วนภาพและดูผล AI เดิม • ภาพ AI จาก AVI วนการวิเคราะห์ด้วย';
  if(reviewView.active)reviewVideo.loop=box.checked;
}
async function setClipLoop(enabled){
  const box=document.getElementById('clip-loop');
  if(box.disabled){showToast('วนได้เฉพาะคลิปนอกการทดสอบ QA');return;}
  // Setting loop=true can immediately make HTMLMediaElement.ended return false.
  const restartAtEnd=enabled && reviewView.active && (reviewVideo.ended ||
    (Number.isFinite(reviewVideo.duration) && reviewVideo.duration>0 && reviewVideo.currentTime>=reviewVideo.duration-.01));
  try{
    if(!reviewView.active && lastReviewStatus.is_running){
      await productRequest('/api/playback/loop',{source:lastReviewStatus.current_source,enabled});
    }
    clipLoopEnabled=enabled;refreshClipLoop();
    try{localStorage.setItem('nexora.clipLoop.v1',String(enabled));}catch(_){}
    if(restartAtEnd){reviewVideo.currentTime=0;reviewView.needsAIBuffer=true;requestReviewPlay();}
    showToast(enabled?'เปิดวนคลิปแล้ว':'ปิดวนคลิปแล้ว');
  }catch(e){refreshClipLoop();showToast(e.message);}
}
refreshClipLoop();
function beginUiOperation(){uiOperations++;}
function endUiOperation(){uiOperations=Math.max(0,uiOperations-1);if(!uiOperations && !reviewView.waitingAI && (!reviewView.active || reviewVideo.readyState>=3))loadingStatus(false);}
function loadingStatus(on,message='กำลังเปิดแหล่งภาพ กรุณารอสักครู่...',percent=null) {
  if(!on && uiOperations)return;
  const box=document.getElementById('loading-status');
  box.classList.toggle('hidden',!on);
  document.getElementById('loading-message').textContent=message+(percent==null?'':` ${Math.round(percent)}%`);
  if (on && !loadingSince) loadingSince=performance.now();
  if (!on) loadingSince=0;
}
setInterval(()=>{
  document.getElementById('loading-elapsed').textContent=loadingSince ? `เว็บยังทำงานอยู่ • รอ ${Math.floor((performance.now()-loadingSince)/1000)} วินาที` : '';
},500);
function nativeSource(source) { return '/api/playback/media?source='+encodeURIComponent(source); }
function openReview(source,url=nativeSource(source),seconds=0,archive=false,sessionId=null,evidenceRevision=null,evidenceUrl=null) {
  reviewView={active:true,archive,source,rows:[],analysis:0,metadataReady:false,userWantsPlay:true,waitingAI:false,needsAIBuffer:true,initialOpen:true,sessionId,evidenceRevision,evidenceUrl};
  reviewVideo.pause();
  for(const id of ['speed-05','speed-10','speed-20'])document.getElementById(id).disabled=false;
  reviewVideo.src=url;reviewVideo.currentTime=0;refreshClipLoop();
  reviewVideo.playbackRate=1;
  for(const [id,value] of [['speed-05',.5],['speed-10',1],['speed-20',2]])document.getElementById(id).classList.toggle('bg-brand-500',value===1);
  reviewVideo.classList.remove('hidden');reviewCanvas.classList.toggle('hidden',archive && !evidenceUrl);
  reviewCanvas.getContext('2d').clearRect(0,0,reviewCanvas.width,reviewCanvas.height);
  loadingStatus(true,'กำลังโหลดวิดีโอ...');
  reviewVideo.onloadedmetadata=()=>{reviewVideo.currentTime=Math.min(seconds,Math.max(0,reviewVideo.duration-.01));reviewView.metadataReady=true;requestReviewPlay();};
  reviewVideo.oncanplay=()=>{clearReviewNetworkWait();if(!reviewView.waitingAI)loadingStatus(false);};
  reviewVideo.onwaiting=()=>{
    clearReviewNetworkWait();const view=reviewView;
    reviewNetworkTimer=setTimeout(()=>{
      reviewNetworkTimer=null;
      if(reviewView===view && view.userWantsPlay && !view.waitingAI && reviewVideo.readyState<3)
        loadingStatus(true,'กำลังรับข้อมูลวิดีโอ...');
    },700);
  };
  reviewVideo.onplaying=()=>{clearReviewNetworkWait();if(!reviewView.waitingAI)loadingStatus(false);};
  reviewVideo.onended=()=>{if(!reviewVideo.loop){reviewView.userWantsPlay=false;reviewView.waitingAI=false;loadingStatus(false);}};
  reviewVideo.onerror=async()=>{
    loadingStatus(false);reviewView.active=false;reviewVideo.classList.add('hidden');reviewCanvas.classList.add('hidden');
    if(archive){feedImg.classList.add('hidden');placeholder.classList.remove('hidden');
      document.getElementById('recording-browser').open=true;
      const link=document.createElement('a');link.href=url;link.download='';link.className='text-cyan-300';link.textContent='ดาวน์โหลดวิดีโอเพื่อเปิดด้วยโปรแกรมเล่นวิดีโอ';
      const box=document.getElementById('recording-list');box.replaceChildren(link);
      showToast('เบราว์เซอร์เล่นไฟล์นี้ไม่ได้ ดาวน์โหลดเพื่อดูย้อนหลังได้');
    }else{reviewView.codecFallback=true;
      for(const id of ['speed-05','speed-20'])document.getElementById(id).disabled=true;
      try{await productRequest('/api/playback/paced',{source});if(reviewView.source!==source)return;
        if(clipLoopEnabled)await productRequest('/api/playback/loop',{source,enabled:true});refreshFeedImage();
        document.getElementById('playback-progress').textContent='เบราว์เซอร์เล่น codec นี้ไม่ได้ • แสดงภาพ AI ที่ 1x (อาจช้าลงเมื่อ AI ตามไม่ทัน) • ใช้ H.264 MP4 / WebM เพื่อกรอและเล่น 2x';
        showToast('เปลี่ยนเป็นภาพ AI ที่ 1x และเริ่มคลิปใหม่แล้ว');
      }catch(e){showToast(e.message);document.getElementById('playback-progress').textContent=e.message;}}
  };
}
function closeReview() {
  clearReviewNetworkWait();
  reviewVideo.pause();reviewVideo.removeAttribute('src');reviewVideo.load();
  reviewVideo.classList.add('hidden');reviewCanvas.classList.add('hidden');reviewView.active=false;reviewView.archive=false;reviewView.codecFallback=false;
  for(const id of ['speed-05','speed-10','speed-20'])document.getElementById(id).disabled=false;
  reviewCanvas.getContext('2d').clearRect(0,0,reviewCanvas.width,reviewCanvas.height);
}
window.syncReview=function(data) {
  lastReviewStatus=data;refreshClipLoop();
  document.getElementById('runtime-active-message').textContent=data.is_running ? `ตัวเล่นหลักกำลังใช้: ${data.weights_file || 'YOLO'} • ${data.device==='cuda'?'GPU (NVIDIA)':'CPU'}` : 'ตัวเล่นหลักยังไม่รันโมเดล';
  const timing=data.processing_ms;
  document.getElementById('runtime-performance-message').textContent=timing?.total ? `${data.is_paused?'พักภาพ • ค่าจากเฟรมล่าสุด':'เวลาประมวลผลต่อเฟรม'}: YOLO ${timing.pose.toFixed(0)} ms • ฟีเจอร์ ${timing.features.toFixed(0)} ms • วาดภาพ/ส่งภาพ ${(timing.render+timing.jpeg).toFixed(0)} ms • อุปกรณ์จริง ${timing.actual_device} • คลิปต้นฉบับ ${Number(data.source_fps || 0).toFixed(0)} FPS` : '';
  if (!reviewView.active) {
    if(reviewView.source!==data.current_source)reviewView.codecFallback=false;
    if(data.paced_fallback && data.is_running && !reviewView.archive){reviewView.codecFallback=true;reviewView.source=data.current_source;
      for(const id of ['speed-05','speed-20'])document.getElementById(id).disabled=true;
      document.getElementById('playback-progress').textContent='ภาพ AI ที่ 1x (อาจช้าลงเมื่อ AI ตามไม่ทัน) • ใช้ H.264 MP4 / WebM เพื่อกรอและเล่น 2x';}
    if(!reviewView.archive && !reviewView.codecFallback && !uiOperations && data.review_mode && data.is_running && !data.is_live){openReview(data.current_source,undefined,0,false,data.session_id,data.evidence_revision);return;}
    if (data.is_running && !data.frame_ready) loadingStatus(true,'กำลังเปิดกล้อง / เตรียม AI...');
    else if (!uiOperations && (!data.is_running || data.frame_ready)) loadingStatus(false);
    return;
  }
  if (!reviewView.archive && data.current_source!==reviewView.source) {closeReview();return;}
  if(!reviewView.archive && reviewView.sessionId && data.session_id && reviewView.sessionId!==data.session_id){
    openReview(data.current_source,undefined,0,false,data.session_id,data.evidence_revision);return;
  }
  if(!reviewView.archive && data.evidence_revision){
    const changed=reviewView.evidenceRevision && reviewView.evidenceRevision!==data.evidence_revision;
    reviewView.evidenceRevision=data.evidence_revision;
    if(changed){reviewView.rows=[];if(data.review_mode && data.is_running && data.qa?.state!=='running')seekReview(reviewVideo.currentTime);}
  }
  if(!reviewView.archive){reviewView.analysis=data.analysis_sec || data.current_sec || 0;reviewView.analysisComplete=data.stage==='analysis_complete';}
  feedImg.classList.add('hidden');placeholder.classList.add('hidden');
  seekSlider.disabled=false;seekSlider.classList.remove('control-disabled');
  const sec=reviewVideo.currentTime || 0,duration=Number.isFinite(reviewVideo.duration)?reviewVideo.duration:0;
  timeCurrent.textContent=formatTime(sec);timeDuration.textContent=formatTime(duration);
  if (!isUserDraggingSlider && duration>0) seekSlider.value=sec/duration*100;
  btnPlay.innerHTML=reviewView.waitingAI?'❚❚ หยุดรอ':reviewVideo.paused?'▶ เล่นต่อ':'❚❚ พัก';
  document.getElementById('playback-progress').textContent=reviewView.archive ?
    (reviewView.evidenceUrl?'ดูย้อนหลังพร้อมผล AI ของรอบที่แจ้งเตือน • '+(reviewView.evidenceMessage || (!matchingReviewRow()?'ไม่มีผล AI ตรงเวลานี้':'ไม่วิเคราะห์ซ้ำ')):'ดูวิดีโอย้อนหลัง • ไม่สร้างผล AI ใหม่')
    : `คลิป ${sec.toFixed(3)}s • AI วิเคราะห์ถึง ${reviewView.analysis.toFixed(1)}s`+(!matchingReviewRow()?' • ยังไม่มีผล AI ตรงเวลานี้':matchingReviewRow().persons?.length===0?' • AI ประมวลผลแล้ว ยังไม่พบคนผ่านเกณฑ์':'')+(reviewView.cacheError?' • เก็บผลย้อนหลังไม่พร้อม: '+reviewView.cacheError:'')+(reviewVideo.ended && !reviewView.analysisComplete && data.is_running?' • ภาพเล่นจบแล้ว แต่ AI ยังวิเคราะห์ต่อ ไม่ต้องเริ่มคลิปใหม่':'');
  if(!reviewView.archive && (data.last_error || !data.is_running)){
    reviewView.userWantsPlay=false;reviewView.waitingAI=false;reviewVideo.pause();loadingStatus(false);
    document.getElementById('playback-progress').textContent=data.last_error || 'การวิเคราะห์หยุดแล้ว กรุณาเปิดคลิปอีกครั้ง';return;
  }
  updateAIPlayback();
};
const previousStart=startStreamWithSource;
startStreamWithSource=async function(source) {
  if(viewStartBusy){showToast('กำลังเตรียมแหล่งภาพ กรุณารอให้เปิดเสร็จ');return;}
  viewStartBusy=true;
  beginUiOperation();closeReview();loadingStatus(true,'กำลังเตรียมแหล่งภาพและ AI...');
  try{
    if(!await previousStart(source)){loadingStatus(false);return;}
    const status=await productRequest('/api/status');
    if (status.last_error || !status.is_running) {loadingStatus(false);return;}
    if (!status.is_live && !status.frame_by_frame) openReview(source,undefined,0,false,status.session_id,status.evidence_revision);
  }catch(e){loadingStatus(false);showToast(e.message);}finally{viewStartBusy=false;endUiOperation();}
};
const previousPlay=togglePlay,previousPause=pauseStream,previousSeek=onSeekSliderChange,previousDelta=seekDelta,previousSpeed=setSpeed,previousStop=stopStream;
togglePlay=function(){if(!reviewView.active)return previousPlay();if(reviewView.waitingAI || !reviewVideo.paused){reviewView.userWantsPlay=false;reviewView.waitingAI=false;reviewVideo.pause();loadingStatus(false);}else requestReviewPlay();};
pauseStream=async function(){if(!reviewView.active)return previousPause();togglePlay();};
onSeekSliderInput=function(value){isUserDraggingSlider=true;if(reviewView.active && Number.isFinite(reviewVideo.duration)){reviewVideo.currentTime=Number(value)*reviewVideo.duration/100;timeCurrent.textContent=formatTime(reviewVideo.currentTime);}};
let reviewSeekSerial=0;
async function seekReview(seconds){
  const serial=++reviewSeekSerial;
  reviewView.needsAIBuffer=true;
  reviewView.initialOpen=false;
  reviewView.rows=[];
  reviewVideo.currentTime=Math.max(0,Math.min(reviewVideo.duration-.01,seconds));
  updateAIPlayback();
  if(reviewView.archive && reviewView.evidenceUrl){await refreshReviewEvidence(true);}
  else if(!reviewView.archive && lastReviewStatus.is_running && lastReviewStatus.review_mode && lastReviewStatus.qa?.state!=='running'){
    const view=reviewView;await refreshReviewEvidence(true);
    if(reviewView===view && serial===reviewSeekSerial && !matchingReviewRow()){
      try{await productRequest('/api/playback/analyze-near',{source:view.source,time_s:reviewVideo.currentTime,session_id:view.sessionId,evidence_revision:view.evidenceRevision});}
      catch(e){showToast(e.message);}
    }
  }
  updateAIPlayback();
}
onSeekSliderChange=async function(value){isUserDraggingSlider=false;if(!reviewView.active)return previousSeek(value);await seekReview(Number(value)*reviewVideo.duration/100);};
seekDelta=async function(delta){if(!reviewView.active)return previousDelta(delta);await seekReview(reviewVideo.currentTime+delta);};
setSpeed=async function(speed){
  if(reviewView.codecFallback){showToast('แปลงคลิปเป็น H.264 MP4 / WebM ก่อนใช้ความเร็วดูคลิป');return;}
  if(!reviewView.active)return previousSpeed(speed);
  reviewVideo.playbackRate=speed;
  for(const [id,value] of [['speed-05',.5],['speed-10',1],['speed-20',2]])document.getElementById(id).classList.toggle('bg-brand-500',value===speed);
  showToast('ความเร็วดูคลิป '+speed+'x • AI วิเคราะห์แยกจากตัวเล่น');
};
stopStream=async function(){const archive=reviewView.archive;closeReview();loadingStatus(false);if(archive){document.getElementById('active-source-title').textContent=window.activeCameraName || 'ตัวเล่นหลัก';await fetchStatus();refreshFeedImage();return;}return previousStop();};
async function setLastSeenNotifications(enabled) {try{await productRequest('/api/notifications/settings',{last_seen_update:enabled});await fetchEvents();}catch(e){showToast(e.message);}}
async function loadRecordingList(){
  const list=document.getElementById('recording-list');list.replaceChildren();
  try{
    const data=await productRequest('/api/recordings');
    if(!data.recordings.length){list.textContent='ยังไม่มีวิดีโอที่บันทึกครบช่วง (แบ่งช่วงละ 10 วินาที)';return;}
    for(const row of data.recordings){const button=document.createElement('button');button.className='block text-cyan-300';const name=typeof cameraProfiles!=='undefined'?cameraProfiles.find(p=>p.id===row.source_id)?.name:null;button.textContent=`▶ ${new Date(row.created_at*1000).toLocaleString('th-TH')} • ${name || 'กล้อง '+row.source_id} • ${(row.end_s-row.start_s).toFixed(1)} วินาที`;button.onclick=()=>openReview(row.filename,'/api/recordings/'+row.id+'/media',0,true);list.append(button);}
  }catch(e){list.textContent=e.message;}
}
// COCO-17 edges and RGB colors match the server's OpenCV overlay.
const playbackSkeleton=[[0,1],[0,2],[1,3],[2,4],[0,5],[0,6],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
const playbackBoneColors=['#0080ff','#0080ff','#0080ff','#0080ff','#ffff00','#ffff00','#00ff00','#ffc800','#ffc800','#8000ff','#8000ff','#00ff00','#00ff00','#00ff00','#00ffff','#00ffff','#ff00ff','#ff00ff'];
function paintPlaybackPose(c,person,video,scale,ox,oy){
  if(!Array.isArray(person.pose) || person.pose.length!==17)return;
  const [x1,y1,x2,y2]=person.bbox_xyxy;
  const valid=person.pose.map(p=>Array.isArray(p) && p.length>=3 && p.slice(0,3).every(Number.isFinite)
    && p[2]>=.5 && p[2]<=1 && (p[0]!==0 || p[1]!==0)
    && p[0]>=Math.max(0,x1) && p[0]<Math.min(video.videoWidth,x2+1)
    && p[1]>=Math.max(0,y1) && p[1]<Math.min(video.videoHeight,y2+1));
  c.lineWidth=2;
  playbackSkeleton.forEach(([a,b],i)=>{
    if(!valid[a] || !valid[b])return;
    const p=person.pose[a],q=person.pose[b];c.strokeStyle=playbackBoneColors[i];
    c.beginPath();c.moveTo(ox+p[0]*scale,oy+p[1]*scale);c.lineTo(ox+q[0]*scale,oy+q[1]*scale);c.stroke();
  });
  c.fillStyle='#ffff00';
  person.pose.forEach((p,i)=>{if(valid[i]){c.beginPath();c.arc(ox+p[0]*scale,oy+p[1]*scale,4,0,2*Math.PI);c.fill();}});
}
function paintEvidence(canvas,video,rows,seconds) {
  canvas.width=canvas.clientWidth;canvas.height=canvas.clientHeight;const c=canvas.getContext('2d');c.clearRect(0,0,canvas.width,canvas.height);
  const row=rows.reduce((best,r)=>!best || Math.abs(r.time_s-seconds)<Math.abs(best.time_s-seconds)?r:best,null);
  if(!row || Math.abs(row.time_s-seconds)>.1 || !video.videoWidth)return;
  const scale=Math.min(canvas.width/video.videoWidth,canvas.height/video.videoHeight),ox=(canvas.width-video.videoWidth*scale)/2,oy=(canvas.height-video.videoHeight*scale)/2;
  c.font='14px sans-serif';
  const scene=row.violence?.result;
  if(scene){
    c.fillStyle=scene.fighting?'#ffb020':'#22dd66';
    c.fillText(`LSTM คะแนนทำร้าย ${(scene.probability_fighting*100).toFixed(1)}%${scene.fighting?' • กรุณาตรวจสอบ':''}`,Math.max(8,ox+8),Math.max(22,oy+22));
  }
  for(const p of row.persons){const [x1,y1,x2,y2]=p.bbox_xyxy,phase=p.fall?.phase;
    const red=p.alert || phase==='FALL_DETECTED',warn=['WARMING_UP','CONFIRMING','POSTURE_REVIEW','POSTURE_CHECK','UNAVAILABLE','ABNORMAL_MOVEMENT'].includes(phase);
    paintPlaybackPose(c,p,video,scale,ox,oy);
    c.strokeStyle=red?'#ff3333':warn?'#ffb020':'#22dd66';c.lineWidth=3;c.strokeRect(ox+x1*scale,oy+y1*scale,(x2-x1)*scale,(y2-y1)*scale);
    const fallScore=p.fall?.backend==='yolo_pose_rf_v2' && p.fall.fall_score!=null?` • ล้ม V2 ${(p.fall.fall_score*100).toFixed(1)}%`:'';
    c.fillStyle=c.strokeStyle;c.fillText(`คน #${p.track_id}${fallScore}${red?' • มีเหตุเตือน':p.observation_kind==='bbox_only'?' • ข้อต่อไม่พร้อม':warn?' • กำลังตรวจสอบ':''}`,Math.max(0,Math.min(canvas.width-260,ox+x1*scale)),Math.max(16,oy+y1*scale-5));
  }
}
function paintReview(){if(reviewView.active && (!reviewView.archive || reviewView.evidenceUrl)){updateAIPlayback();paintEvidence(reviewCanvas,reviewVideo,reviewView.rows,reviewVideo.currentTime);}requestAnimationFrame(paintReview);}
requestAnimationFrame(paintReview);
async function refreshReviewEvidence(force=false){
  if(!reviewView.active || (reviewView.archive && !reviewView.evidenceUrl) || (reviewFetchBusy && !force))return;
  if(!force)reviewFetchBusy=true;const view=reviewView,seconds=reviewVideo.currentTime;
  const seekSerial=reviewSeekSerial,requestId=view.evidenceRequestId=(view.evidenceRequestId || 0)+1;
  try{const data=await productRequest((view.evidenceUrl || '/api/playback/annotations')+'?seconds='+seconds);
    if(reviewView===view && seekSerial===reviewSeekSerial && requestId===view.evidenceRequestId && Math.abs(reviewVideo.currentTime-seconds)<=.4 &&
      (view.archive || (data.source===view.source && (!view.sessionId || data.session_id===view.sessionId) && (!view.evidenceRevision || data.evidence_revision===view.evidenceRevision)))){
      view.rows=Array.isArray(data.frames)?data.frames:[];view.analysis=data.analysis_sec || 0;
      view.evidenceMessage=data.message;view.analysisComplete=data.stage==='analysis_complete';
      view.sourceFps=data.source_fps || 30;view.cacheError=data.cache?.error;
      view.sessionId=data.session_id || view.sessionId;updateAIPlayback();
      view.evidenceRevision=data.evidence_revision || view.evidenceRevision;
    }
  }catch(e){if(reviewView===view)document.getElementById('playback-progress').textContent='ยังอ่านผล AI ไม่ได้: '+e.message;}
  finally{if(!force)reviewFetchBusy=false;}
}
setInterval(refreshReviewEvidence,200);
let runtimeDeviceEdited=false,runtimeModelEdited=false;
document.getElementById('inference-device').addEventListener?.('change',()=>runtimeDeviceEdited=true);
document.getElementById('pose-model').addEventListener?.('change',()=>runtimeModelEdited=true);
async function initializeRuntimeChoices(){
  try{const data=await productRequest('/api/runtime/options');
    if(!runtimeDeviceEdited)document.getElementById('inference-device').value=data.current_device || (data.gpu_available?'cuda':'cpu');
    document.getElementById('inference-device').querySelector('[value="cuda"]').disabled=!data.gpu_available;
    document.getElementById('decode-device').querySelector('[value="cuda"]').disabled=!data.gpu_available || !data.ffmpeg_available;
    const select=document.getElementById('pose-model');
    if(!data.models){document.getElementById('runtime-choice-message').textContent='เซิร์ฟเวอร์ยังใช้โค้ดเดิม: บันทึกผลตรวจ แล้วเปิด NEXORA ใหม่เพื่อใช้การเลือกโมเดลและแก้ความเร็ว AVI';for(const name of ['yolo26s-pose.pt','yolo26m-pose.pt','yolov8n-pose.pt','yolov8s-pose.pt','yolov8m-pose.pt']){const option=select.querySelector(`[value="${name}"]`);if(option)option.disabled=true;}return;}
    const selectedModel=select.value;select.replaceChildren();
    const modelName=name=>name.replace(/^yolo26/,'YOLO26').replace(/^yolov8/,'YOLOv8').replace(/\.pt$/,'');
    for(const model of data.models){const option=document.createElement('option');option.value=model.name;option.textContent=modelName(model.name);option.disabled=!model.available;select.append(option);}
    select.value=runtimeModelEdited && data.models.some(m=>m.name===selectedModel && m.available)?selectedModel:data.current_model || 'yolo26n-pose.pt';
    const ready=data.models.filter(m=>m.available).map(m=>modelName(m.name)).join(', ');
    const missing=data.models.filter(m=>!m.available).map(m=>modelName(m.name)).join(', ');
    document.getElementById('runtime-choice-message').textContent='เลือก CPU/GPU และโมเดล แล้วกดใช้กับตัวเล่นหลัก คลิปจะเริ่มใหม่ • กล้อง/คลิปที่จะเปิดเพิ่มใช้ค่าที่เลือกนี้ • พร้อมใช้: '+ready+(missing?' • ต้องดาวน์โหลดก่อน: '+missing:'');
  }catch(e){document.getElementById('runtime-choice-message').textContent=e.message;}
}
async function applyRuntimeChoices(){
  const button=document.getElementById('apply-runtime-button');button.disabled=true;
  try{const source=currentSource || analyzedSource;if(!source){showToast('เลือกคลิปหรือกล้องก่อน');return;}await startStreamWithSource(source);}
  finally{button.disabled=false;}
}
initializeRuntimeChoices();
