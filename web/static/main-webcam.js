/* The primary browser camera uses the primary worker, dashboard and event clock. */
let mainCameraDevices=[],mainCameraSession=null,mainCameraOpening=false,mainCameraSerial=0;
const mainCameraPreview=document.getElementById('main-camera-preview');
function mainCameraMessage(text){document.getElementById('main-camera-message').textContent=text;}
async function listMainBrowserCameras(){
  const button=document.getElementById('main-camera-open'),select=document.getElementById('main-camera-selector');
  mainCameraMessage('กำลังค้นหา webcam และขอสิทธิ์กล้อง...');
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('เปิดผ่าน localhost หรือ HTTPS เพื่อใช้ webcam');
    // Do not reacquire a busy default camera while a browser camera is already open.
    if(!mainCameraSession && !browserSessions.size){
      const permission=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      permission.getTracks().forEach(track=>track.stop());
    }
    const previous=select.value;
    mainCameraDevices=(await navigator.mediaDevices.enumerateDevices()).filter(device=>device.kind==='videoinput');
    select.replaceChildren();
    for(const [index,device] of mainCameraDevices.entries()){
      const option=document.createElement('option');option.value=device.deviceId;option.textContent=device.label || 'webcam '+(index+1);select.append(option);
    }
    select.value=mainCameraDevices.some(device=>device.deviceId===previous)?previous:(mainCameraDevices[0]?.deviceId || '');
    select.disabled=!mainCameraDevices.length;button.disabled=!mainCameraDevices.length || mainCameraOpening;
    mainCameraMessage(mainCameraDevices.length?'เลือกกล้อง แล้วกดเปิด webcam ในตัวเล่นหลัก':'ไม่พบ webcam');
  }catch(error){mainCameraMessage(cameraErrorMessage(error));}
}
async function openSelectedMainCamera(){
  const device=mainCameraDevices.find(device=>device.deviceId===document.getElementById('main-camera-selector').value);
  if(!device){await listMainBrowserCameras();return;}
  return startMainBrowserCamera(device);
}
async function stopMainBrowserCamera(stopServer=false){
  mainCameraSerial++;
  const session=mainCameraSession;mainCameraSession=null;window.mainCameraDeviceId=null;
  if(session)releaseBrowserCamera(session);
  mainCameraPreview.pause();mainCameraPreview.srcObject=null;mainCameraPreview.classList.add('hidden');
  if(stopServer && session)await productRequest('/api/stream/stop',{session_id:session.card.session});
  if(session)mainCameraMessage('หยุด webcam ในตัวเล่นหลักแล้ว');
}
async function startMainBrowserCamera(device){
  if(mainCameraOpening){mainCameraMessage('กำลังเปิดกล้อง กรุณารอ');return false;}
  mainCameraOpening=true;document.getElementById('main-camera-open').disabled=true;
  let media,raw,session,ownsOpening=false,serial;
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('เปิดผ่าน localhost หรือ HTTPS เพื่อใช้ webcam');
    if(openingBrowserDevices.has(device.deviceId) || [...browserSessions.values()].some(s=>s.device===device.deviceId))throw new Error('กล้องนี้เปิดอยู่ในหลายแหล่งภาพ กดหยุดก่อนเปิดในตัวเล่นหลัก');
    const status=await productRequest('/api/status');
    if(status.qa?.state==='running')throw new Error('หยุดรอบ QA ก่อนเปลี่ยนแหล่งภาพ');
    await stopMainBrowserCamera(true);serial=++mainCameraSerial;
    openingBrowserDevices.add(device.deviceId);ownsOpening=true;
    mainCameraMessage('กำลังเปิด '+(device.label || 'webcam')+' ในตัวเล่นหลัก...');
    media=await navigator.mediaDevices.getUserMedia({video:{deviceId:{exact:device.deviceId},width:{ideal:960},height:{ideal:540}},audio:false});
    raw=document.createElement('video');raw.muted=true;raw.playsInline=true;raw.srcObject=media;
    raw.setAttribute('aria-hidden','true');raw.style.cssText='position:fixed;left:0;top:0;width:1px;height:1px;opacity:0;pointer-events:none';
    document.body.append(raw);await raw.play();await waitBrowserVideo(raw);
    if(serial!==mainCameraSerial)throw new Error('ยกเลิกการเปิดกล้องแล้ว');
    closeReview();loadingStatus(false);feedImg.src='';feedImg.classList.add('hidden');
    placeholder.classList.add('hidden');mainCameraPreview.srcObject=media;mainCameraPreview.classList.remove('hidden');await mainCameraPreview.play();
    const width=Math.min(960,raw.videoWidth),height=Math.round(raw.videoHeight*width/raw.videoWidth),source='browser://'+device.deviceId;
    const result=await productRequest('/api/stream/start',{source,loop:false,review_mode:false,
      camera_id:window.activeCameraId || null,device:document.getElementById('inference-device').value || 'cpu',
      model:document.getElementById('pose-model').value || 'yolo26n-pose.pt',decode_device:'cpu',browser_width:width,browser_height:height});
    session={main:true,device:device.deviceId,deviceInfo:device,source,media,raw,width,height,
      started:performance.now(),closed:false,recorder:null,recordTimer:null,aiReady:false,
      message:mainCameraMessage,card:{session:result.session_id,status:document.getElementById('main-camera-message')}};
    if(serial!==mainCameraSerial){releaseBrowserCamera(session);await productRequest('/api/stream/stop',{session_id:result.session_id});return false;}
    mainCameraSession=session;window.mainCameraDeviceId=device.deviceId;currentSource=source;
    document.getElementById('sample-selector').value='';
    if(typeof updateClipDeleteButton==='function')updateClipDeleteButton();
    document.getElementById('active-source-title').textContent=device.label || 'webcam ตัวเล่นหลัก';
    session.frameTask=sendCameraFrames(result.camera_id,session);
    beginRecordingSegment(result.camera_id,session).catch(error=>{if(!session.closed){session.card.recordingError=error.message;mainCameraMessage('ภาพกล้องยังเปิดอยู่ • บันทึกไม่ได้: '+error.message);}});
    mainCameraMessage('กล้องเปิดในตัวเล่นหลักแล้ว • แสดงภาพต้นฉบับระหว่างเตรียม AI');
    refreshFeedImage();await fetchStatus();return true;
  }catch(error){
    if(session && !session.closed){releaseBrowserCamera(session);if(mainCameraSession===session)await stopMainBrowserCamera(true);}
    else{media?.getTracks().forEach(track=>track.stop());raw?.pause();raw?.remove();}
    mainCameraPreview.pause();mainCameraPreview.srcObject=null;mainCameraPreview.classList.add('hidden');
    mainCameraMessage(cameraErrorMessage(error));await fetchStatus();return false;
  }finally{if(ownsOpening)openingBrowserDevices.delete(device.deviceId);mainCameraOpening=false;document.getElementById('main-camera-open').disabled=!mainCameraDevices.length;}
}
function syncMainBrowserCamera(data){
  const session=mainCameraSession;if(!session)return;
  if(data.current_source!==session.source || data.session_id!==session.card.session || !data.is_running){stopMainBrowserCamera(false);return;}
  if(reviewView.active){mainCameraPreview.classList.add('hidden');return;}
  placeholder.classList.add('hidden');loadingStatus(false);
  mainCameraPreview.classList.toggle('hidden',session.aiReady);feedImg.classList.toggle('hidden',!session.aiReady);
  const warning=[session.card.frameError && 'ส่งภาพไม่ได้: '+session.card.frameError,
    session.card.recordingError && 'บันทึกไม่ได้: '+session.card.recordingError,data.camera_warning].filter(Boolean).join(' • ');
  mainCameraMessage((session.aiReady?'กล้องตัวเล่นหลัก • AI '+data.device.toUpperCase()+' • วิเคราะห์ถึง '+data.analysis_sec.toFixed(1)+'s':'กล้องเปิดแล้ว • แสดงภาพต้นฉบับระหว่างเตรียม AI')+(warning?' • '+warning:''));
}
feedImg.addEventListener('load',()=>{if(mainCameraSession && !mainCameraSession.closed){mainCameraSession.aiReady=true;if(!reviewView.active){mainCameraPreview.classList.add('hidden');feedImg.classList.remove('hidden');placeholder.classList.add('hidden');}}});
feedImg.addEventListener('error',()=>{if(mainCameraSession){mainCameraSession.aiReady=false;if(!reviewView.active){mainCameraPreview.classList.remove('hidden');feedImg.classList.add('hidden');}mainCameraMessage('กำลังรับภาพ AI ใหม่ • แสดงภาพ webcam ต้นฉบับ');}});
const mainCameraPreviousStart=startStreamWithSource;
startStreamWithSource=async function(source){
  if(source==='browser-webcam' || /^\d+$/.test(source) || source?.startsWith('browser://')){
    const device=mainCameraSession?.deviceInfo || mainCameraDevices.find(device=>'browser://'+device.deviceId===source);
    if(device)return startMainBrowserCamera(device);
    await listMainBrowserCameras();return false;
  }
  await stopMainBrowserCamera(true);return mainCameraPreviousStart(source);
};
const mainCameraPreviousStop=stopStream;
stopStream=async function(){await stopMainBrowserCamera(false);return mainCameraPreviousStop();};
window.addEventListener('pagehide',()=>{if(mainCameraSession){releaseBrowserCamera(mainCameraSession);fetch('/api/stream/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:mainCameraSession.card.session}),keepalive:true}).catch(()=>{});}});
