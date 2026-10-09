/* Each source owns its tracker, clock and recording session. */
let clipQueue=[], multiCards=new Map(), browserSessions=new Map(), multiPolling=false;
function openMultiView(){document.getElementById('multi-view').classList.remove('hidden');}
function multiMessage(text){document.getElementById('multi-message').textContent=text;}
function uploadWithProgress(file,onProgress){
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest(),form=new FormData();form.append('file',file);
    xhr.open('POST','/api/upload');xhr.upload.onprogress=e=>onProgress(e.lengthComputable?e.loaded/e.total*100:null);
    xhr.onerror=()=>reject(new Error('การเชื่อมต่อขาดระหว่างอัปโหลด'));
    xhr.onload=()=>{let body;try{body=JSON.parse(xhr.responseText);}catch(_){reject(new Error('อ่านผลอัปโหลดไม่สำเร็จ'));return;}
      xhr.status>=200 && xhr.status<300?resolve(body):reject(new Error(body.detail || 'อัปโหลดไม่สำเร็จ'));};
    xhr.send(form);
  });
}
handleFileUpload=async function(files){
  if(!files?.length)return;const selected=Array.from(files);if(selected.length>1)openMultiView();
  beginUiOperation();
  try{
    for(let i=0;i<selected.length;i++){
      const file=selected[i];loadingStatus(true,`อัปโหลด ${i+1}/${selected.length}: ${file.name}`);
      const uploaded=await uploadWithProgress(file,pct=>loadingStatus(true,`อัปโหลด ${i+1}/${selected.length}: ${file.name}`,pct));
      clipQueue.push({...uploaded,selected:true});
    }
    renderClipQueue();loadingStatus(false);
    if(selected.length===1){const row=clipQueue.at(-1);currentSource=row.path;document.getElementById('active-source-title').textContent=row.filename;await startStreamWithSource(row.path);}
    else multiMessage(`อัปโหลดครบ ${selected.length} คลิปแล้ว เลือกคลิปและกดเปิดพร้อมกัน`);
  }catch(e){loadingStatus(false);showToast(e.message);multiMessage(e.message);renderClipQueue();}finally{endUiOperation();}
};
function renderClipQueue(){const list=document.getElementById('clip-queue');list.replaceChildren();
  for(const row of clipQueue){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=row.selected;input.onchange=()=>row.selected=input.checked;label.className='block';label.append(input,document.createTextNode(' '+row.filename));list.append(label);}}
async function newStreamProfile(name,source){return productRequest('/api/cameras',{name,source});}
async function startSelectedClips(){
  const selected=clipQueue.filter(c=>c.selected);if(!selected.length){multiMessage('เลือกคลิปก่อน');return;}
  openMultiView();multiMessage('กำลังเปิดคลิปที่เลือก...');
  for(const row of selected){
    try{const profile=row.profile || await newStreamProfile(row.filename,row.path);row.profile=profile;
      await productRequest('/api/multistream/start',{camera_id:profile.id,source:row.path,model:document.getElementById('pose-model').value || 'yolo26n-pose.pt',device:document.getElementById('inference-device').value,decode_device:document.getElementById('decode-device').value});
      addMultiCard(profile,row.path,false);row.selected=false;
    }catch(e){multiMessage(e.message);break;}
  }
  renderClipQueue();await loadCameras();await pollMulti();
}
function addMultiCard(profile,source,live){
  const card=document.createElement('article');card.className='bg-slate-800 rounded-xl p-3 space-y-2';
  const title=document.createElement('strong');title.textContent=profile.name;
  const status=document.createElement('p');status.className='text-xs text-cyan-300';status.textContent='กำลังเปิดแหล่งภาพและเตรียม AI...';
  const area=document.createElement('div');area.className='relative aspect-video bg-black';
  const video=document.createElement('video');video.muted=true;video.playsInline=true;video.className='w-full h-full object-contain';
  const overlay=document.createElement('canvas');overlay.className='absolute inset-0 w-full h-full pointer-events-none';
  const img=document.createElement('img');img.className='hidden w-full h-full object-contain';img.alt='ภาพตรวจจับ '+profile.name;
  area.append(video,overlay,img);
  const stop=document.createElement('button');stop.textContent='หยุดแหล่งภาพ';stop.className='text-red-300 text-xs';stop.onclick=()=>stopMulti(profile.id);
  const rate=document.createElement('select');rate.setAttribute('aria-label','ความเร็ว '+profile.name);rate.className='bg-slate-700 text-xs rounded p-1';
  for(const speed of [.5,1,2]){const op=document.createElement('option');op.value=speed;op.textContent=speed+'x';op.selected=speed===1;rate.append(op);}rate.onchange=()=>video.playbackRate=Number(rate.value);
  video.controls=!live;
  if(!live){video.src=nativeSource(source);video.play().catch(()=>{});}else rate.classList.add('hidden');
  card.append(title,status,area,rate,stop);document.getElementById('multi-grid').append(card);
  const entry={profile,source,live,video,img,overlay,status,rows:[],card,fetchBusy:false,session:null};multiCards.set(profile.id,entry);
  video.onerror=async()=>{entry.fallback=true;rate.disabled=true;video.classList.add('hidden');
    try{await productRequest('/api/playback/paced',{camera_id:profile.id,source});img.classList.remove('hidden');img.src='/api/stream/feed?camera_id='+profile.id;status.textContent='แสดงภาพ AI ที่ 1x • ใช้ H.264 / WebM เพื่อกรอและเล่น 2x';}
    catch(e){entry.playbackError=e.message;status.textContent=e.message;}};
  return entry;
}
async function pollMulti(){
  if(multiPolling)return;multiPolling=true;
  try{const data=await productRequest('/api/multistream');
    const ids=new Set(data.map(row=>row.id));
    for(const [id,card] of multiCards)if(!ids.has(id)){card.missing=true;card.running=false;card.status.textContent='AI ของแหล่งภาพนี้หยุดแล้ว / server เริ่มรอบใหม่ • คลิปที่โหลดไว้ยังดูได้';if(browserSessions.has(id))await stopBrowserSession(id);}
    for(const row of data){let card=multiCards.get(row.id);
      if(!card && (row.state==='starting' || row.is_running)){openMultiView();card=addMultiCard({id:row.id,name:row.camera?.name || row.id},row.current_source,row.is_live);}
      if(!card)continue;card.session=row.session_id;card.missing=false;card.running=row.is_running || row.state==='starting';
      const err=row.error || row.last_error;
      card.status.textContent=err ? 'เปิดไม่สำเร็จ: '+err : row.state==='starting' ? 'กำลังโหลดโมเดล / เปิดแหล่งภาพ...' : !row.is_running?'AI หยุดแล้ว • คลิปที่โหลดไว้ยังดูได้':`AI ${row.device?.toUpperCase()} • วิเคราะห์ถึง ${row.analysis_sec.toFixed(1)}s`+(card.live?' • บันทึกภาพ (24 ชั่วโมง)':'');
      if(row.recording?.error)card.status.textContent+=' • บันทึกไม่สำเร็จ: '+row.recording.error;
      if(card.recordingError)card.status.textContent+=' • บันทึกไม่สำเร็จ: '+card.recordingError;
      card.status.textContent+=' • เปิด: '+(row.configured_features?.map(f=>productNames[f] || f).join(' / ') || 'ยังไม่เปิดฟีเจอร์');
      if(card.fallback)card.status.textContent+=' • '+(card.playbackError || 'ภาพ AI ที่ 1x • ใช้ H.264 / WebM เพื่อกรอและเล่น 2x');
      if(row.weights_file)card.status.textContent+=' • '+row.weights_file;
      if(card.live && row.frame_ready && !card.img.src){card.img.src='/api/stream/feed?camera_id='+row.id;card.img.classList.remove('hidden');card.video.classList.add('hidden');}
      if((err || (!row.is_running && row.state!=='starting')) && browserSessions.has(row.id))await stopBrowserSession(row.id);
    }
  }catch(e){multiMessage(e.message);}finally{multiPolling=false;}
}
setInterval(pollMulti,1000);
setInterval(async()=>{for(const [id,card] of multiCards){if(card.live || card.fallback || card.fetchBusy)continue;card.fetchBusy=true;
  try{const rows=await productRequest('/api/playback/annotations?camera_id='+id+'&seconds='+card.video.currentTime);if(rows.source===card.source)card.rows=rows.frames;}catch(_){}finally{card.fetchBusy=false;}}},250);
function paintMulti(){for(const card of multiCards.values())if(!card.live && !card.fallback)paintEvidence(card.overlay,card.video,card.rows,card.video.currentTime);requestAnimationFrame(paintMulti);}requestAnimationFrame(paintMulti);
async function listBrowserCameras(){
  openMultiView();multiMessage('กำลังขอสิทธิ์ใช้กล้อง...');
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('เปิดผ่าน localhost หรือ HTTPS เพื่อใช้ webcam');
    const permission=await navigator.mediaDevices.getUserMedia({video:true,audio:false});permission.getTracks().forEach(t=>t.stop());
    const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
    const list=document.getElementById('browser-camera-list');list.replaceChildren();
    const select=document.createElement('select');select.className='bg-slate-800 rounded p-2';select.setAttribute('aria-label','เลือก webcam ที่ต้องการเปิด');
    devices.forEach((device,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=device.label || 'webcam '+(i+1);select.append(option);});select.value='0';
    const button=document.createElement('button');button.className='bg-cyan-700 rounded px-3 py-2';button.textContent='เปิด webcam ที่เลือก';button.disabled=!devices.length;button.onclick=()=>startBrowserCamera(devices[Number(select.value)],Number(select.value));list.append(select,button);
    multiMessage(devices.length?'เลือก webcam แต่ละตัวเพื่อเปิดพร้อมกัน':'ไม่พบ webcam');
  }catch(e){multiMessage(e.message);}
}
async function startBrowserCamera(device,index){
  multiMessage('กำลังเปิด '+(device.label || 'webcam')+'...');let media,profile;
  try{
    if([...browserSessions.values()].some(s=>s.device===device.deviceId))throw new Error('กล้องนี้เปิดอยู่แล้ว');
    const options=await productRequest('/api/runtime/options');if([...multiCards.values()].filter(c=>c.running!==false && !c.missing).length>=options.max_streams)throw new Error('หยุดแหล่งภาพก่อนเปิดกล้องเพิ่ม');
    media=await navigator.mediaDevices.getUserMedia({video:{deviceId:{exact:device.deviceId},width:{ideal:960},height:{ideal:540}},audio:false});
    const raw=document.createElement('video');raw.muted=true;raw.playsInline=true;raw.srcObject=media;await raw.play();
    const width=Math.min(960,raw.videoWidth),height=Math.round(raw.videoHeight*width/raw.videoWidth);
    profile=await newStreamProfile(device.label || 'webcam '+(index+1),'browser://'+device.deviceId);
    await productRequest('/api/multistream/start',{camera_id:profile.id,source:'browser://'+profile.id,model:document.getElementById('pose-model').value || 'yolo26n-pose.pt',device:document.getElementById('inference-device').value,decode_device:'cpu',browser_width:width,browser_height:height});
    const card=addMultiCard(profile,'browser://'+profile.id,true);card.video.srcObject=media;card.video.play().catch(()=>{});
    const session={device:device.deviceId,media,raw,width,height,started:performance.now(),closed:false,recorder:null,recordTimer:null};browserSessions.set(profile.id,session);
    await beginRecordingSegment(profile.id,session);
    sendCameraFrames(profile.id,session);multiMessage('กล้องเปิดแล้ว • บันทึกภาพ ไม่มีเสียง • เก็บ 24 ชั่วโมง');await loadCameras();
  }catch(e){media?.getTracks().forEach(t=>t.stop());multiMessage(e.message);if(profile){await stopBrowserSession(profile.id);const card=multiCards.get(profile.id);if(card){card.card.remove();multiCards.delete(profile.id);}try{await productRequest('/api/multistream/'+profile.id+'/stop',{});}catch(_){}}}
}
async function waitCameraSession(id,session){
  const deadline=performance.now()+45000;
  while(!session.closed && performance.now()<deadline){const data=await productRequest('/api/multistream');const row=data.find(r=>r.id===id);if(row?.error)throw new Error(row.error);if(row?.session_id){multiCards.get(id).session=row.session_id;return row.session_id;}await new Promise(resolve=>setTimeout(resolve,200));}
  throw new Error('เปิดกล้องใช้เวลานานเกินไป ลองเปิดใหม่');
}
async function beginRecordingSegment(id,session){
  const sid=await waitCameraSession(id,session);
  const mime=['video/webm;codecs=vp8','video/webm'].find(t=>window.MediaRecorder?.isTypeSupported(t));
  if(!mime)throw new Error('เบราว์เซอร์นี้ยังบันทึก WebM ไม่ได้');
  function next(){
    if(session.closed)return;
    const chunks=[],start=(performance.now()-session.started)/1000;let segmentTimer,stoppedAt=null;
    const recorder=new MediaRecorder(session.media,{mimeType:mime,videoBitsPerSecond:1500000});session.recorder=recorder;
    recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
    recorder.onstop=async()=>{
      const end=stoppedAt ?? (performance.now()-session.started)/1000;clearTimeout(segmentTimer);
      const form=new FormData();form.append('file',new Blob(chunks,{type:mime}),'segment.webm');form.append('session_id',sid);form.append('start_s',start);form.append('end_s',end);
      try{const res=await fetch('/api/multistream/'+id+'/recording',{method:'POST',body:form});if(!res.ok)throw new Error((await res.json()).detail || 'บันทึกไม่สำเร็จ');}
      catch(e){const card=multiCards.get(id);if(card){card.recordingError=e.message;card.status.textContent='บันทึกไม่สำเร็จ: '+e.message;}multiMessage('ภาพยังเปิดอยู่ แต่บันทึกไม่สำเร็จ: '+e.message);}
    };
    recorder.onerror=e=>{const card=multiCards.get(id);if(card)card.recordingError=e.error?.message || 'ตัวบันทึกกล้องหยุด';multiMessage('บันทึกภาพไม่สำเร็จ กรุณาเปิดกล้องใหม่');};
    recorder.start();segmentTimer=setTimeout(()=>{stoppedAt=(performance.now()-session.started)/1000;if(recorder.state!=='inactive')recorder.stop();next();},10000);session.recordTimer=segmentTimer;
  }
  next();
}
async function sendCameraFrames(id,session){
  const canvas=document.createElement('canvas');canvas.width=session.width;canvas.height=session.height;const c=canvas.getContext('2d');
  while(!session.closed){
    try{const sid=multiCards.get(id)?.session;if(sid){c.drawImage(session.raw,0,0,canvas.width,canvas.height);const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.75));const form=new FormData();form.append('file',blob,'frame.jpg');form.append('session_id',sid);form.append('timestamp_ms',performance.now()-session.started);const res=await fetch('/api/multistream/'+id+'/frame',{method:'POST',body:form});if(!res.ok)throw new Error((await res.json()).detail);}}
    catch(e){multiMessage('ส่งภาพกล้องไม่สำเร็จ: '+e.message);}
    await new Promise(resolve=>setTimeout(resolve,100));
  }
}
async function stopBrowserSession(id){const session=browserSessions.get(id);if(!session)return;session.closed=true;clearTimeout(session.recordTimer);if(session.recorder?.state==='recording')session.recorder.stop();session.media.getTracks().forEach(t=>t.stop());session.raw.pause();browserSessions.delete(id);}
async function stopMulti(id){try{await stopBrowserSession(id);const card=multiCards.get(id);if(!card.missing)await productRequest('/api/multistream/'+id+'/stop',{});card.video.pause();card.img.src='';card.card.remove();multiCards.delete(id);await pollMulti();}catch(e){multiMessage(e.message);}}
window.addEventListener('pagehide',()=>{for(const id of browserSessions.keys()){const session=browserSessions.get(id);session.closed=true;session.media.getTracks().forEach(t=>t.stop());fetch('/api/multistream/'+id+'/stop',{method:'POST',keepalive:true}).catch(()=>{});}});
