/* The producer waits until this tab has decoded and displayed each AI image. */
window.framePlaybackClient=globalThis.crypto?.randomUUID?.() || 'player-'+Date.now()+'-'+Math.random().toString(36).slice(2);
window.clipFramePlayback=true;
try{window.clipFramePlayback=localStorage.getItem('nexora.framePlayback.v1')!=='false';}catch(_){}
document.getElementById('frame-by-frame').checked=window.clipFramePlayback;
function setFramePlaybackMode(enabled){
  window.clipFramePlayback=enabled;
  try{localStorage.setItem('nexora.framePlayback.v1',String(enabled));}catch(_){}
  showToast('ใช้โหมดที่เลือกเมื่อเปิดคลิปครั้งถัดไป');
}
let framePlayer=null;
function closeFramePlayer(){
  const player=framePlayer;framePlayer=null;window.framePlayerActive=false;
  if(player){player.abort.abort();player.loaded?.();if(player.url)URL.revokeObjectURL(player.url);}
  feedImg.onload=feedImg.onerror=null;
  document.getElementById('wait-for-ai').disabled=false;
}
async function acknowledgeFrame(player){
  if(framePlayer!==player || !player.sequence || player.paused)return;
  await productRequest('/api/playback/frame/ack',{session_id:player.session,client_id:window.framePlaybackClient,sequence:player.sequence});
}
async function receiveFrame(player){
  if(player.busy || framePlayer!==player || player.complete)return;
  player.busy=true;
  try{
    const response=await fetch('/api/playback/frame?session_id='+encodeURIComponent(player.session)+'&client_id='+encodeURIComponent(window.framePlaybackClient)+'&after='+player.sequence,{signal:player.abort.signal,cache:'no-store'});
    if(framePlayer!==player)return;
    if(response.status===204)return;
    if(!response.ok)throw new Error((await response.json()).detail || 'อ่านภาพ AI ไม่สำเร็จ');
    const sequence=Number(response.headers.get('X-Frame-Sequence'));
    const frame=Number(response.headers.get('X-Frame-Id'));
    const seconds=Number(response.headers.get('X-Frame-Time'));
    const blob=await response.blob();if(framePlayer!==player)return;
    const url=URL.createObjectURL(blob);
    try{
      await new Promise((resolve,reject)=>{
        player.loaded=resolve;
        feedImg.onload=resolve;feedImg.onerror=()=>reject(new Error('แสดงภาพ AI ไม่สำเร็จ'));
        feedImg.src=url;feedImg.classList.remove('hidden');placeholder.classList.add('hidden');
      });
      await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
      if(framePlayer!==player)return;
      if(player.url)URL.revokeObjectURL(player.url);player.url=url;
      player.sequence=sequence;player.frame=frame;player.seconds=seconds;
      loadingStatus(false);
      document.getElementById('playback-progress').textContent=`ประมวลผลทีละเฟรม • เฟรม ${frame} • ${seconds.toFixed(3)}s • ภาพพร้อมผล AI`;
      await acknowledgeFrame(player);
    }finally{if(framePlayer!==player || player.url!==url)URL.revokeObjectURL(url);}
  }catch(error){
    if(framePlayer===player && error.name!=='AbortError'){
      player.paused=true;loadingStatus(false);showToast(error.message);
      document.getElementById('playback-progress').textContent=error.message+' • เปิดคลิปอีกครั้งเพื่อลองใหม่';
    }
  }finally{player.busy=false;}
}
function startFramePlayer(data){
  if(framePlayer?.session===data.session_id)return;
  closeFramePlayer();closeReview();
  framePlayer={session:data.session_id,sequence:0,paused:false,busy:false,complete:false,abort:new AbortController()};
  window.framePlayerActive=true;feedImg.src='';
  document.getElementById('wait-for-ai').disabled=true;
  loadingStatus(true,'กำลังประมวลผลเฟรมแรก...');
  receiveFrame(framePlayer);
}
const framePreviousSync=window.syncReview;
window.syncReview=function(data){
  if(reviewView.archive){closeFramePlayer();return framePreviousSync(data);}
  if(data.frame_by_frame && data.is_running && data.playback_client===window.framePlaybackClient){
    startFramePlayer(data);
    framePreviousSync(data);
    framePlayer.paused=data.is_paused;
    framePlayer.complete=data.stage==='analysis_complete';
    if(framePlayer.complete){loadingStatus(false);document.getElementById('playback-progress').textContent=`จบคลิป • ประมวลผลครบ ${data.current_frame} เฟรม`;}
    else if(data.last_error){framePlayer.paused=true;loadingStatus(false);}
    else receiveFrame(framePlayer);
    return;
  }
  closeFramePlayer();framePreviousSync(data);
  if(data.frame_by_frame && data.is_running)document.getElementById('playback-progress').textContent='คลิปกำลังประมวลผลทีละเฟรมจากแท็บที่เปิดคลิป • กดเปิดคลิปเพื่อเริ่มจากแท็บนี้';
};
const framePreviousFeed=refreshFeedImage;
refreshFeedImage=function(){if(!window.framePlayerActive)return framePreviousFeed();};
const framePreviousStart=startStreamWithSource;
startStreamWithSource=async function(source){closeFramePlayer();return framePreviousStart(source);};
const framePreviousStop=stopStream;
stopStream=async function(){closeFramePlayer();return framePreviousStop();};
const framePreviousPlay=togglePlay;
togglePlay=function(){if(!framePlayer)return framePreviousPlay();pauseStream();};
const framePreviousPause=pauseStream;
pauseStream=async function(){
  if(!framePlayer)return framePreviousPause();
  const player=framePlayer;if(player.pauseBusy)return;player.pauseBusy=true;
  try{
    if(player.complete){
      await productRequest('/api/stream/seek',{time_sec:0});
      player.complete=false;
    }
    const result=await productRequest('/api/stream/pause',{});
    if(framePlayer!==player)return;
    player.paused=result.status==='paused';
    if(!player.paused){await acknowledgeFrame(player);receiveFrame(player);}
    await fetchStatus();
  }catch(error){showToast(error.message);}finally{player.pauseBusy=false;}
};
setInterval(()=>{if(framePlayer)receiveFrame(framePlayer);},50);
window.addEventListener('pagehide',()=>{closeFramePlayer();});
