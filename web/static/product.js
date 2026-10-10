/* Camera profiles, runtime health, and user-triggered recorded-clip QA. */
const productNames = {fall:'ล้ม', violence:'ทำร้าย (LSTM)', location:'ตำแหน่ง/เตียง', wandering:'พลัดหลง'};
const healthNames = {ready:'พร้อม', running:'กำลังรับภาพ', disabled:'ปิด', error:'ผิดพลาด',
  unavailable:'ไม่พร้อม', waiting:'รอภาพใหม่', no_observation:'ไม่มีคนให้ประเมิน', no_pair:'รอคู่คน', warming:'เก็บการเคลื่อนไหว',
  needs_configuration:'ต้องตั้งโซน', stalled:'ภาพค้าง', connecting:'กำลังเปิดภาพ', idle:'ยังไม่เปิดภาพ', paused:'พักภาพ'};
let cameraProfiles = [], cameraBusy = false, qaLabels = [], qaReportKey = '';
let qaLabelScope = '';
window.cameraRevision = 0;
async function productRequest(url, body, method = 'POST') {
  const options = body === undefined ? {} : {method, headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)};
  const response = await fetch(url, options), result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'ข้อมูลไม่ถูกต้อง กรุณาตรวจรายการที่กรอก');
  return result;
}
function productMessage(kind, text) { document.getElementById(kind+'-message').textContent = text || ''; }
function closeProductModal(kind) { document.getElementById(kind+'-modal').classList.add('hidden'); }
function setProfileSource(source) {
  currentSource = source || '';
  const select = document.getElementById('sample-selector');
  const isClip=source && !/^\d+$/.test(source) && !/^(browser|rtsp|https?):/.test(source);
  if (isClip && !Array.from(select.options).some(option => option.value === source)) {
    const option = document.createElement('option'); option.value = source; option.textContent = 'คลิปของห้องที่เลือก'; select.append(option);
  }
  select.value = isClip ? source : '';
  if(typeof updateClipDeleteButton==='function')updateClipDeleteButton();
  document.getElementById('active-source-title').textContent = window.activeCameraName + (source ? ' • พร้อมเปิดแหล่งภาพ' : ' • เลือกคลิปเพื่อทดลอง');
}
function renderCameras(data) {
  cameraProfiles = data.profiles;
  window.activeCameraId = data.active_id;
  const active = cameraProfiles.find(profile => profile.id === data.active_id);
  window.activeCameraName = active.name;
  const select = document.getElementById('camera-profile-selector'); select.replaceChildren();
  const list = document.getElementById('camera-list'); list.replaceChildren();
  for (const profile of cameraProfiles) {
    const option = document.createElement('option'); option.value = profile.id; option.textContent = profile.name; select.append(option);
    const row = document.createElement('div'); row.className = 'flex flex-wrap items-center gap-3 bg-slate-800 rounded-lg p-3';
    const name = document.createElement('span');
    const features = profile.config.features;
    name.textContent = `${profile.name} • เตียง/พื้นที่ ${Object.keys(features.location?.config?.zones || {}).length} • เขตพลัดหลง ${Object.keys(features.wandering?.zones_relative || {}).length}`;
    const choose = document.createElement('button'); choose.textContent = profile.id === data.active_id ? 'เลือกอยู่' : 'เลือก'; choose.className = 'text-brand-400'; choose.disabled = profile.id === data.active_id;
    choose.onclick = () => selectCameraProfile(profile.id);
    const remove = document.createElement('button'); remove.textContent = 'ลบ'; remove.className = 'text-red-300'; remove.disabled = profile.id === data.active_id;
    remove.onclick = async () => {
      if (!confirm('ลบโปรไฟล์ '+profile.name+' หรือไม่?')) return;
      try { renderCameras(await productRequest('/api/cameras/'+encodeURIComponent(profile.id), {}, 'DELETE')); }
      catch (error) { productMessage('camera', error.message); }
    };
    row.append(name, choose, remove); list.append(row);
  }
  select.value = data.active_id;
  document.getElementById('camera-name').value = active.name;
  document.getElementById('camera-source').value = active.source;
}
async function loadCameras() { renderCameras(await productRequest('/api/cameras')); }
async function openCameraManager() {
  document.getElementById('camera-modal').classList.remove('hidden'); productMessage('camera', '');
  try { await loadCameras(); } catch (error) { productMessage('camera', error.message); }
}
async function selectCameraProfile(id) {
  if (cameraBusy) return;
  cameraBusy = true; window.cameraRevision++;
  const select = document.getElementById('camera-profile-selector'); select.disabled = true;
  try {
    const data = await productRequest('/api/cameras/'+encodeURIComponent(id)+'/select', {});
    renderCameras(data); cancelZoneDrawing(); setProfileSource(cameraProfiles.find(profile => profile.id === id).source);
    feedImg.src = ''; showToast('เลือก '+window.activeCameraName+' แล้ว • กดเล่นหรือเลือกคลิปเพื่อเริ่ม');
  } catch (error) { productMessage('camera', error.message); showToast(error.message); await loadCameras(); }
  finally { cameraBusy = false; select.disabled = false; window.cameraRevision++; await fetchStatus(); }
}
async function createCameraProfile() {
  try {
    const profile = await productRequest('/api/cameras', {name:document.getElementById('camera-name').value, source:document.getElementById('camera-source').value});
    await selectCameraProfile(profile.id); productMessage('camera', 'สร้างและเลือกห้องใหม่แล้ว วาดโซนของห้องนี้หลังเปิดคลิป');
  } catch (error) { productMessage('camera', error.message); }
}
async function editCameraProfile() {
  try {
    const data = await productRequest('/api/cameras/'+encodeURIComponent(window.activeCameraId), {name:document.getElementById('camera-name').value, source:document.getElementById('camera-source').value}, 'PUT');
    renderCameras(data); setProfileSource(cameraProfiles.find(profile => profile.id === data.active_id).source);
    await fetchStatus();
    productMessage('camera', 'บันทึกชื่อ/แหล่งภาพแล้ว กดเล่นเพื่อเปิดแหล่งภาพใหม่');
  } catch (error) { productMessage('camera', error.message); }
}
async function importCameraProfiles(files) {
  if (!files?.length) return;
  if (files[0].size > 1024*1024) { productMessage('camera', 'ไฟล์โปรไฟล์ต้องไม่เกิน 1 MB'); return; }
  if (!confirm('นำเข้าจะหยุดภาพและแทนโปรไฟล์ในรอบนี้ ต้องการนำเข้าหรือไม่?')) return;
  cameraBusy = true; window.cameraRevision++;
  try {
    const data = await productRequest('/api/cameras/import', JSON.parse(await files[0].text()));
    renderCameras(data); cancelZoneDrawing(); setProfileSource(cameraProfiles.find(profile => profile.id === data.active_id).source);
    productMessage('camera', 'นำเข้าโปรไฟล์แล้ว');
  } catch (error) { productMessage('camera', error.message); }
  finally { cameraBusy = false; window.cameraRevision++; await fetchStatus(); }
}
function updateProductStatus(data) {
  const fallNode=document.getElementById('main-fall-result');
  if(fallNode){
    const entries=Object.entries(data.fall_statuses || {});
    fallNode.textContent=data.fall_backend==='yolo_pose_rf_v2' && data.fall_model?
      (data.fall_model.feature_count===54?'ตรวจล้ม 54 ค่า • ':'ตรวจล้ม V2 • ')+(entries.length?entries.map(([id,s])=>`คน #${id} • ${s.phase==='FALL_DETECTED'?'อาจล้ม • ตรวจสอบภาพ':s.phase==='ABNORMAL_MOVEMENT'?'ท่าผิดปกติ':s.phase==='WARMING_UP'?`เก็บภาพ ${s.sampled_frames}/${s.required_frames}`:s.phase==='UNAVAILABLE'?'ข้อต่อไม่พร้อม':'ท่าปกติ'}${s.fall_score!=null?` • คะแนน RF หน้าต่าง ${(s.fall_score*100).toFixed(1)}%`:''}`).join(' | '):'โมเดลพร้อม • รอคนที่มีข้อต่อใช้งานได้'):'';
  }
  const sceneNode=document.getElementById('main-violence-result'), diagnostic=data.interaction_diagnostics;
  if(sceneNode){
    const result=diagnostic?.model_kind==='resnet18_lstm'?diagnostic.result:null;
    sceneNode.textContent=data.product_revision!=='qa-upload-delete-20261010'?'หน้าเว็บกับเซิร์ฟเวอร์คนละรุ่น กรุณารีเฟรชหน้าเว็บ หากยังพบข้อความนี้ให้อัปเดตระบบเป็นรุ่นล่าสุด':!data.is_running || !diagnostic || diagnostic.model_kind!=='resnet18_lstm'?'':result?
      `ผลล่าสุด LSTM${result.model_version==='v3'?' v3 (2 ชั้น)':''} • ${String(data.current_source || '').split(/[\\/]/).pop()} • ช่วง ${(result.window_start_ms/1000).toFixed(1)}–${(result.window_end_ms/1000).toFixed(1)}s • คะแนนทำร้าย ${(result.probability_fighting*100).toFixed(1)}%${result.fighting?' • กรุณาตรวจสอบภาพ':''}`:
      `LSTM กำลังเก็บภาพ ${diagnostic.sampled_frames}/16 เฟรม`;
  }
  const enabled=data.configured_features || data.enabled_features || Object.entries(data.health?.features || {}).filter(([,s])=>s.state!=='disabled').map(([name])=>name);
  for (const name of Object.keys(productNames)) document.getElementById('feature-card-'+name)?.classList.toggle('hidden',!enabled.includes(name));
  const summary=document.getElementById('enabled-feature-summary');
  if (summary) summary.textContent=enabled.length ? 'ฟีเจอร์ที่เปิด: '+enabled.map(name=>productNames[name] || name).join(' • ') : 'ยังไม่ได้เปิดฟีเจอร์ AI — เปิดได้จากปุ่มตั้งค่า';
  if (data.camera && !cameraBusy) {
    window.activeCameraId = data.camera.id; window.activeCameraName = data.camera.name;
    document.getElementById('camera-profile-selector').value = data.camera.id;
  }
  if (data.health) {
    const health = data.health;
    document.getElementById('health-capture').textContent = 'แหล่งภาพ: '+(healthNames[health.capture.state] || health.capture.state)+(health.capture.message ? ' • '+health.capture.message : '');
    document.getElementById('health-detector').textContent = 'ตัวตรวจจับคน: '+(healthNames[health.detector.state] || health.detector.state)+(health.detector.message ? ' • '+health.detector.message : '');
    const details = document.getElementById('health-details'); details.replaceChildren();
    for (const [feature, status] of Object.entries(health.features)) {
      const label = document.getElementById('health-'+feature); if (!label) continue;
      label.textContent = healthNames[status.state] || status.state;
      label.className = 'text-[10px] '+(['error','unavailable'].includes(status.state) ? 'text-red-400' : status.state === 'ready' ? 'text-emerald-400' : 'text-amber-300');
      if (status.message) { const row = document.createElement('p'); row.textContent = productNames[feature]+': '+status.message; details.append(row); }
    }
  }
  updateQaStatus(data.qa);
}
function updateQaStatus(run) {
  const status = document.getElementById('qa-status'), result = document.getElementById('qa-result'), download = document.getElementById('qa-download');
  download.disabled = !run || run.state === 'running';
  if (!run) { status.textContent = 'ยังไม่มีรอบทดสอบ • เลือกคลิป แล้วกดตั้งการทดสอบเพื่อระบุช่วงเหตุจริง'; return; }
  status.textContent = run.state === 'running' ? `กำลังทดสอบ ${run.frames}/${run.expected_frames} ภาพ • ${Math.round(run.progress*100)}%`
    : run.state === 'completed' ? 'จบรอบ ครบคลิป' : 'รอบไม่สมบูรณ์ • '+(run.reason || 'หยุดก่อนจบ');
  const key = run.id+':'+run.state;
  if (run.state === 'running') { result.replaceChildren(); qaReportKey = ''; return; }
  if (qaReportKey === key) return;
  qaReportKey = key;
  productRequest('/api/qa').then(report => {
    if (qaReportKey !== key || report?.run.id !== run.id) return;
    result.replaceChildren();
    const lines = report.metrics ? [
      `ห้อง: ${report.provenance.camera_name || report.run.source_id}`,
      `เหตุจริง ${report.labels.length} • เจอ ${report.metrics.true_positives} • พลาด ${report.metrics.missed_events} • เตือนผิด ${report.metrics.false_positives}`,
      `การเตือนตรงกับเหตุจริง: ${report.metrics.precision == null ? 'ไม่มีการเตือน' : (report.metrics.precision*100).toFixed(1)+'%'}`,
      `ตรวจเจอจากเหตุจริงทั้งหมด: ${report.metrics.event_recall == null ? 'คลิปนี้ไม่มีเหตุจริง' : (report.metrics.event_recall*100).toFixed(1)+'%'}`,
      `เตือนช้าเฉลี่ย: ${report.metrics.mean_alert_delay_s == null ? 'ไม่มีเหตุที่จับคู่ได้' : report.metrics.mean_alert_delay_s.toFixed(2)+' วินาที'}`,
      `เตือนผิดต่อชั่วโมง: ${report.metrics.false_alerts_per_hour.toFixed(1)} (คำนวณจากคลิปนี้)`
    ] : ['ยังไม่สรุปความแม่นยำ เพราะรอบทดสอบไม่สมบูรณ์'];
    lines.push(`ประมวลผล ${report.coverage.processed_frames}/${report.coverage.expected_frames} ภาพ • มีคนผ่านเกณฑ์ ${report.coverage.frames_with_usable_person} ภาพ`, 'ผลจากคลิปที่ตรวจนี้เท่านั้น คลิปสั้นอาจทำให้อัตราต่อชั่วโมงแกว่งมาก');
    for (const text of lines) { const row = document.createElement('p'); row.textContent = text; result.append(row); }
  }).catch(error => { qaReportKey = ''; status.textContent = error.message; });
}
async function openQaManager() {
  document.getElementById('qa-modal').classList.remove('hidden'); productMessage('qa', '');
  document.getElementById('qa-source-label').textContent = 'คลิป: '+(currentSource || analyzedSource || 'ยังไม่ได้เลือกคลิป');
  await updateQaZones(); renderQaLabels();
}
async function updateQaZones() {
  const type = document.getElementById('qa-event-type').value, needsZone = type.startsWith('wandering_') || type === 'location_update';
  const wrapper = document.getElementById('qa-zone-label'); wrapper.classList.toggle('hidden', !needsZone);
  const select = document.getElementById('qa-zone'); select.replaceChildren();
  if (!needsZone) { syncQaLabelScope(); return; }
  try {
    const config = await productRequest('/api/config');
    const zones = type.startsWith('wandering_') ? config.features.wandering.zones_relative : config.features.location.config.zones;
    for (const name of Object.keys(zones || {})) { const option = document.createElement('option'); option.value = name; option.textContent = name; select.append(option); }
    if (!select.options.length) productMessage('qa', 'วาดโซนของห้องนี้ก่อนทดสอบชนิดเหตุนี้');
    syncQaLabelScope();
  } catch (error) { productMessage('qa', error.message); }
}
function syncQaLabelScope() {
  const key = JSON.stringify([window.activeCameraId, currentSource || analyzedSource,
    document.getElementById('qa-event-type').value, document.getElementById('qa-zone').value]);
  if (qaLabelScope && qaLabelScope !== key) {
    qaLabels = []; document.getElementById('qa-negative').checked = false; renderQaLabels();
  }
  qaLabelScope = key;
}
function renderQaLabels() {
  const list = document.getElementById('qa-labels'); list.replaceChildren();
  qaLabels.forEach((label, index) => {
    const row = document.createElement('div'), text = document.createElement('span'), remove = document.createElement('button');
    text.textContent = `ช่วง ${index+1}: ${label.start_s}–${label.end_s} วินาที `;
    remove.textContent = 'ลบ'; remove.className = 'text-red-300'; remove.onclick = () => { qaLabels.splice(index, 1); renderQaLabels(); };
    row.append(text, remove); list.append(row);
  });
}
function addQaLabel() {
  const start = Number(document.getElementById('qa-start').value), end = Number(document.getElementById('qa-end').value);
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start) { productMessage('qa', 'ตรวจเวลาเริ่ม/จบช่วงเหตุ'); return; }
  qaLabels.push({start_s:start, end_s:end}); document.getElementById('qa-negative').checked = false; renderQaLabels(); productMessage('qa', '');
}
async function startQaReview() {
  syncQaLabelScope();
  const button = document.getElementById('qa-start-button'); button.disabled = true;
  try {
    const run = await productRequest('/api/qa/start', {source:currentSource || analyzedSource, camera_id:window.activeCameraId,
      event_type:document.getElementById('qa-event-type').value, zone:document.getElementById('qa-zone').value,
      labels:qaLabels, negative_confirmed:document.getElementById('qa-negative').checked,
      tolerance_sec:Number(document.getElementById('qa-tolerance').value)});
    updateQaStatus(run); closeProductModal('qa'); refreshFeedImage(); await fetchStatus();
  } catch (error) { productMessage('qa', error.message); }
  finally { button.disabled = false; }
}
window.addEventListener('DOMContentLoaded', async () => {
  try { await loadCameras(); if (!currentSource) setProfileSource(cameraProfiles.find(profile => profile.id === window.activeCameraId).source); }
  catch (error) { showToast(error.message); }
});

const telegramTypes = ['fall_detected','fall_posture_review','high_risk_interaction','wandering_entered_zone','wandering_exited_zone'];
function renderTelegram(data) {
  document.getElementById('telegram-enabled').checked = data.enabled;
  document.getElementById('telegram-cooldown').value = data.cooldown_sec;
  for (const type of telegramTypes) document.getElementById('telegram-'+type).checked = data.event_types.includes(type);
  const list = document.getElementById('telegram-drafts'); list.replaceChildren();
  for (const draft of data.drafts) {
    const item = document.createElement('pre'); item.className = 'whitespace-pre-wrap bg-slate-800 rounded-lg p-3';
    item.textContent = 'ร่าง • ยังไม่ได้ส่ง\n'+draft.text; list.append(item);
  }
  if (!data.drafts.length) { const text = document.createElement('p'); text.textContent = 'ยังไม่มีร่างจากเหตุการณ์'; list.append(text); }
  productMessage('telegram', `โหมดเตรียมระบบ • ไม่ส่งจริง • เว้นร่างที่ถี่เกินไป ${data.suppressed} รายการ`);
}
async function openTelegramManager() {
  document.getElementById('telegram-modal').classList.remove('hidden');
  try { renderTelegram(await productRequest('/api/telegram')); }
  catch (error) { productMessage('telegram', error.message); }
}
async function saveTelegramSettings() {
  try {
    const data = await productRequest('/api/telegram/settings', {enabled:document.getElementById('telegram-enabled').checked,
      event_types:telegramTypes.filter(type=>document.getElementById('telegram-'+type).checked),
      cooldown_sec:Number(document.getElementById('telegram-cooldown').value)});
    renderTelegram(data);
  } catch (error) { productMessage('telegram', error.message); }
}
async function previewTelegram() {
  try {
    const result = await productRequest('/api/telegram/preview', {});
    document.getElementById('telegram-preview').textContent = 'ตัวอย่าง • ยังไม่ได้ส่ง\n'+result.text;
  } catch (error) { productMessage('telegram', error.message); }
}
