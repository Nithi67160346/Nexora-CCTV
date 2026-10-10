/* Read original event evidence in a separate player. Never mutate main playback. */
const eventReplayModal=document.getElementById('event-replay-modal');
const eventReplayVideo=document.getElementById('event-replay-video');
const eventReplayCanvas=document.getElementById('event-replay-overlay');
const eventReplayStatus=document.getElementById('event-replay-status');
let eventReplay=null;
function closeEventReplay(){
  const previous=eventReplay;eventReplay=null;
  eventReplayModal.classList.add('hidden');
  eventReplayVideo.onloadedmetadata=eventReplayVideo.onseeked=eventReplayVideo.onerror=null;
  eventReplayVideo.pause();eventReplayVideo.removeAttribute('src');eventReplayVideo.load();
  eventReplayCanvas.getContext('2d').clearRect(0,0,eventReplayCanvas.width,eventReplayCanvas.height);
  previous?.focus?.focus?.();
}
async function playEvent(id){
  closeEventReplay();
  const view={id,rows:[],serial:0,busy:false,ready:false,focus:document.activeElement};eventReplay=view;
  eventReplayModal.classList.remove('hidden');document.getElementById('event-replay-close').focus?.();
  eventReplayStatus.textContent='กำลังโหลดช่วงแจ้งเตือนและผลที่บันทึกไว้...';
  document.getElementById('event-replay-download').classList.add('hidden');
  try{
    const result=await productRequest('/api/events/'+encodeURIComponent(id)+'/playback');
    if(eventReplay!==view)return;
    view.evidence=result.evidence_url;view.url=result.url;
    eventReplayVideo.onloadedmetadata=async()=>{
      if(eventReplay!==view)return;
      eventReplayVideo.currentTime=Math.min(result.time_s || 0,Math.max(0,eventReplayVideo.duration-.01));
      view.ready=true;await refreshEventReplay(true);
      if(eventReplay===view)eventReplayVideo.play().catch(()=>{eventReplayStatus.textContent+=' • กดเล่นเพื่อดูวิดีโอ';});
    };
    eventReplayVideo.onseeked=()=>{if(eventReplay===view){view.rows=[];refreshEventReplay(true);}};
    eventReplayVideo.onerror=()=>{
      if(eventReplay!==view)return;
      view.ready=false;view.rows=[];
      eventReplayStatus.textContent='เบราว์เซอร์เล่นไฟล์นี้ไม่ได้ ตัวเล่นหลักยังใช้ได้ตามเดิม';
      const link=document.getElementById('event-replay-download');link.href=result.url;link.classList.remove('hidden');
    };
    eventReplayVideo.src=result.url;
  }catch(error){if(eventReplay===view)eventReplayStatus.textContent=error.message;}
}
async function refreshEventReplay(force=false){
  const view=eventReplay;if(!view?.ready || (view.busy && !force))return;
  if(!view.evidence){eventReplayStatus.textContent='ดูย้อนหลัง • เหตุนี้ไม่มีผล AI ของรอบเดิม';return;}
  const serial=++view.serial,seconds=eventReplayVideo.currentTime;
  view.busy=true;
  try{
    const result=await productRequest(view.evidence+'?seconds='+seconds);
    if(eventReplay!==view || view.serial!==serial || Math.abs(eventReplayVideo.currentTime-seconds)>.4)return;
    view.rows=Array.isArray(result.frames)?result.frames:[];
    const matching=view.rows.some(row=>Math.abs(row.time_s-eventReplayVideo.currentTime)<=.1);
    eventReplayStatus.textContent=result.message || (matching?'ดูย้อนหลัง • ผล AI ของรอบที่แจ้งเตือนตรงเวลา':'ดูย้อนหลัง • ไม่มีผล AI ที่บันทึกไว้ตรงเวลานี้');
  }catch(error){if(eventReplay===view)eventReplayStatus.textContent='อ่านผลย้อนหลังไม่ได้: '+error.message;}
  finally{if(view.serial===serial)view.busy=false;}
}
function paintEventReplay(){
  if(eventReplay?.ready)paintEvidence(eventReplayCanvas,eventReplayVideo,eventReplay.rows,eventReplayVideo.currentTime);
  requestAnimationFrame(paintEventReplay);
}
requestAnimationFrame(paintEventReplay);setInterval(refreshEventReplay,200);
const eventReplayPreviousSync=window.syncReview;
window.syncReview=function(data){
  eventReplayPreviousSync(data);
  document.getElementById('event-replay-main-status').textContent='ตัวเล่นหลัก: '+
    (!data.is_running?'หยุดอยู่':data.last_error?'มีข้อผิดพลาด':data.stage==='analysis_complete'?'ประมวลผลจบแล้ว':data.is_paused?'พักอยู่':`กำลังประมวลผล • เฟรม ${data.current_frame || 0}`);
};
document.addEventListener('keydown',event=>{if(event.key==='Escape' && eventReplay)closeEventReplay();});
window.addEventListener('pagehide',closeEventReplay);
