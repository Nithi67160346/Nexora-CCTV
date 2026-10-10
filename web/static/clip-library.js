/* Uploading adds to the library; only selecting a clip starts main inference. */
let uploadedClips=[],clipUploadBusy=false,clipDeleteBusy=false;
const deletedClipPaths=new Set();
let pendingClipDeletion=null;
let clipLibraryOpen=false;
function updateClipDeleteButton(){
  const select=document.getElementById('sample-selector');
  const selected=Array.from(select.options).find(option=>option.value===select.value);
  document.getElementById('clip-library-selected').textContent=selected?.textContent || '-- เลือกคลิปที่อัปโหลด --';
  const list=document.getElementById('clip-library-list');list.replaceChildren();
  for(const option of Array.from(select.options).filter(option=>option.value)){
    const row=document.createElement('div');row.className='flex items-center gap-1 rounded hover:bg-slate-800';
    const play=document.createElement('button');play.className='min-w-0 flex-1 text-left text-xs text-slate-200 truncate px-2 py-2';
    play.textContent=option.textContent;play.title=option.textContent;
    play.onclick=()=>{select.value=option.value;closeClipLibrary();onSelectSample(option.value);};
    row.append(play);
    if(uploadedClips.some(clip=>clip.path===option.value)){
      const remove=document.createElement('button');remove.textContent='ลบ';
      remove.className='shrink-0 rounded text-red-300 hover:bg-red-900 px-2 py-2 text-xs disabled:opacity-40';
      remove.setAttribute('aria-label','ลบ '+option.textContent);remove.disabled=clipUploadBusy || clipDeleteBusy;
      remove.onclick=()=>deleteSelectedClip(option.value);row.append(remove);
    }
    list.append(row);
  }
  if(!list.children.length){const empty=document.createElement('p');empty.className='text-xs text-slate-400 p-3';empty.textContent='ยังไม่มีคลิป • กดอัปโหลดคลิปเพื่อเพิ่มรายการ';list.append(empty);}
}
function closeClipLibrary(){
  clipLibraryOpen=false;document.getElementById('clip-library-list').classList.add('hidden');
  document.getElementById('clip-library-toggle').setAttribute('aria-expanded','false');
}
function toggleClipLibrary(){
  if(clipLibraryOpen){closeClipLibrary();return;}
  updateClipDeleteButton();const toggle=document.getElementById('clip-library-toggle'),list=document.getElementById('clip-library-list');
  const rect=toggle.getBoundingClientRect(),width=Math.min(Math.max(rect.width,240),window.innerWidth-16),below=window.innerHeight-rect.bottom;
  list.style.width=width+'px';list.style.left=Math.max(8,Math.min(rect.left,window.innerWidth-width-8))+'px';
  const openBelow=below>=180 || below>=rect.top;
  list.style.top=openBelow?(rect.bottom+4)+'px':'auto';list.style.bottom=openBelow?'auto':(window.innerHeight-rect.top+4)+'px';
  list.style.maxHeight=Math.max(100,Math.min(320,(openBelow?below:rect.top)-12))+'px';
  list.classList.remove('hidden');clipLibraryOpen=true;toggle.setAttribute('aria-expanded','true');
}
function renderUploadedClips(){
  const select=document.getElementById('sample-selector');
  const existing=Array.from(select.options).find(option=>option.value===currentSource);
  const placeholder=document.createElement('option');placeholder.value='';
  placeholder.textContent=uploadedClips.length?'-- เลือกคลิปที่อัปโหลดเพื่อเริ่ม --':'-- อัปโหลดคลิปเพื่อเพิ่มรายการ --';
  select.replaceChildren(placeholder);
  const copies=new Map();
  for(const clip of uploadedClips){
    const option=document.createElement('option'),copy=(copies.get(clip.filename) || 0)+1;
    copies.set(clip.filename,copy);option.value=clip.path;
    option.textContent=clip.filename+(copy>1?` (สำเนา ${copy})`:'');select.append(option);
  }
  if(existing?.value && !deletedClipPaths.has(existing.value) && !/^(browser|rtsp|https?):/.test(existing.value) && !/^\d+$/.test(existing.value) &&
    !uploadedClips.some(clip=>clip.path===existing.value))select.append(existing);
  select.value=Array.from(select.options).some(option=>option.value===currentSource)?currentSource:'';
  updateClipDeleteButton();
}
async function loadUploadedClips(){
  try{
    const rows=await productRequest('/api/uploads');
    const clips=new Map(rows.filter(row=>!deletedClipPaths.has(row.path)).map(row=>[row.path,row]));
    for(const row of uploadedClips)if(!clips.has(row.path))clips.set(row.path,row);
    uploadedClips=Array.from(clips.values());renderUploadedClips();
  }catch(error){document.getElementById('clip-library-message').textContent='โหลดรายการคลิปไม่ได้: '+error.message;}
}
async function handleFileUpload(files){
  const batch=Array.from(files || []);if(!batch.length || clipUploadBusy || clipDeleteBusy)return;
  clipUploadBusy=true;
  updateClipDeleteButton();
  const input=document.getElementById('video-upload-input'),message=document.getElementById('clip-library-message');
  input.disabled=true;let uploaded=0;const failures=[];
  try{
    for(const [index,file] of batch.entries()){
      message.textContent=`กำลังอัปโหลด ${index+1}/${batch.length}: ${file.name}`;
      try{
        if(!/\.(mp4|webm|avi|mov|mkv)$/i.test(file.name))throw new Error('รองรับ MP4, WebM, AVI, MOV และ MKV');
        if(file.size>2*1024**3)throw new Error('ไฟล์เกิน 2 GB');
        const body=new FormData();body.append('file',file);
        const response=await fetch('/api/upload',{method:'POST',body});
        const result=await response.json();if(!response.ok)throw new Error(result.detail || 'อัปโหลดไม่สำเร็จ');
        uploadedClips.unshift({path:result.path,filename:result.filename,size:file.size});uploaded++;renderUploadedClips();
      }catch(error){failures.push(file.name+': '+error.message);}
    }
    message.textContent=`อัปโหลดสำเร็จ ${uploaded}/${batch.length} คลิป • เลือกชื่อจากรายการเพื่อเริ่มประมวลผล`+
      (failures.length?' • ไม่สำเร็จ: '+failures.join(' • '):'');
  }finally{clipUploadBusy=false;input.disabled=false;input.value='';updateClipDeleteButton();}
}
function cancelClipDeletion(){
  pendingClipDeletion=null;document.getElementById('clip-delete-modal').classList.add('hidden');
}
function deleteSelectedClip(source=document.getElementById('sample-selector').value){
  if(clipUploadBusy || clipDeleteBusy)return;
  const selected=source;
  const clip=uploadedClips.find(row=>row.path===selected);if(!clip)return;
  pendingClipDeletion=clip;
  closeClipLibrary();
  document.getElementById('clip-delete-name').textContent=clip.filename;
  document.getElementById('clip-delete-modal').classList.remove('hidden');
  document.getElementById('clip-delete-cancel').focus?.();
}
async function confirmClipDeletion(){
  const clip=pendingClipDeletion;if(!clip || clipUploadBusy || clipDeleteBusy)return;
  const selected=clip.path;cancelClipDeletion();
  const message=document.getElementById('clip-library-message'),input=document.getElementById('video-upload-input');
  clipDeleteBusy=true;input.disabled=true;updateClipDeleteButton();
  try{
    await productRequest('/api/uploads',{source:selected},'DELETE');
    deletedClipPaths.add(selected);uploadedClips=uploadedClips.filter(row=>row.path!==selected);
    if(currentSource===selected){currentSource='';document.getElementById('active-source-title').textContent=window.activeCameraName || 'เลือกคลิปเพื่อเริ่ม';}
    renderUploadedClips();message.textContent='ลบคลิป '+clip.filename+' แล้ว';
  }catch(error){message.textContent=error.message;}
  finally{clipDeleteBusy=false;input.disabled=false;updateClipDeleteButton();}
}
document.addEventListener('click',event=>{
  if(clipLibraryOpen && !document.getElementById('clip-library-container').contains(event.target) && !document.getElementById('clip-library-list').contains(event.target))closeClipLibrary();
});
document.addEventListener('keydown',event=>{if(event.key==='Escape'){closeClipLibrary();cancelClipDeletion();}});
window.addEventListener('resize',closeClipLibrary);
