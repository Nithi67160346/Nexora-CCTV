/* Local enrollment UI. Names are rendered as text, never HTML. */
let faceBusy = false;
function faceMessage(message) { document.getElementById('face-message').textContent = message || ''; }
async function faceRequest(url, options) {
  const response = await fetch(url, options), body = await response.json();
  if (!response.ok) throw new Error(body.detail || 'จัดการใบหน้าไม่สำเร็จ');
  return body;
}
function updateFaceStatus(face, people = []) {
  if (!face) return;
  const mode = face.recognition_active ? 'ระบุชื่อ' : face.detection_active ? 'ตรวจพบใบหน้า' : 'ปิด';
  document.getElementById('face-runtime-status').textContent = `ใบหน้า: ${mode} • ลงทะเบียน ${face.registered_count} คน`;
  document.getElementById('face-readiness').textContent = face.warning ||
    (!face.detector_ready || !face.embedder_ready ? 'โมเดลยังไม่ครบ อ่านวิธีติดตั้งในคู่มือใบหน้า' : 'โมเดลพร้อมใช้งาน');
  if (!faceBusy) document.getElementById('face-mode').value = face.mode;
  const active = document.getElementById('face-active-people'); active.replaceChildren();
  for (const person of people) {
    const row = document.createElement('p');
    const role = {resident:'ผู้พัก', caregiver:'เจ้าหน้าที่', visitor:'ผู้มาเยี่ยม'}[person.person_role] || 'ไม่ทราบบทบาท';
    row.textContent = person.resident_id ? `${person.identity_name} • ${role} • Track #${person.track_id}` : `ไม่ทราบชื่อ • Track #${person.track_id}`;
    active.append(row);
  }
}
function renderRegisteredPeople(people) {
  const container = document.getElementById('face-registered-people'); container.replaceChildren();
  if (!people.length) { container.textContent = 'ยังไม่มีคนลงทะเบียน'; return; }
  for (const person of people) {
    const row = document.createElement('div'); row.className = 'flex items-center justify-between gap-3 border-b border-slate-800 py-2';
    const label = document.createElement('span');
    const role = {resident:'ผู้พัก', caregiver:'เจ้าหน้าที่', visitor:'ผู้มาเยี่ยม'}[person.role] || 'ไม่ทราบบทบาท';
    label.textContent = `${person.display_name} • ${role} • ${person.samples} รูป` + (person.assigned_bed ? ` • ${person.assigned_bed}` : '');
    const button = document.createElement('button'); button.type = 'button'; button.className = 'text-red-300 px-2 py-1'; button.textContent = 'ลบ';
    button.onclick = () => deleteRegisteredPerson(person.identity_id, person.display_name);
    row.append(label, button); container.append(row);
  }
}
async function loadFacePanel() {
  const face = await faceRequest('/api/face'); updateFaceStatus(face); renderRegisteredPeople(face.people || []);
}
async function openFaceManager() {
  document.getElementById('face-modal').classList.remove('hidden'); faceMessage('');
  try { await loadFacePanel(); } catch (error) { faceMessage(error.message); }
}
function closeFaceManager() { document.getElementById('face-modal').classList.add('hidden'); }
async function changeFaceMode() {
  const select = document.getElementById('face-mode'); faceBusy = true; select.disabled = true;
  try {
    const face = await faceRequest('/api/face/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({mode:select.value})});
    faceMessage('บันทึกโหมดใบหน้าแล้ว'); renderRegisteredPeople(face.people || []);
  } catch (error) { faceMessage(error.message); }
  finally { faceBusy = false; select.disabled = false; await loadFacePanel().catch(error => faceMessage(error.message)); }
}
async function enrollRegisteredPerson(event) {
  event.preventDefault(); const form = event.currentTarget, submit = document.getElementById('face-enroll-submit');
  const files = document.getElementById('face-images').files;
  if (files.length < 3 || files.length > 12) { faceMessage('เลือกภาพ 3–12 รูปก่อน'); return; }
  if ([...files].some(file => file.size > 8 * 1024 * 1024)) { faceMessage('ภาพแต่ละรูปต้องไม่เกิน 8 MB'); return; }
  submit.disabled = true; faceMessage('กำลังตรวจคุณภาพและลงทะเบียนใบหน้า…');
  try {
    const result = await faceRequest('/api/face/enroll', {method:'POST', body:new FormData(form)}); form.reset();
    faceMessage(`ลงทะเบียนสำเร็จ ใช้ ${result.accepted} รูป` + (result.rejected.length ? ` • ข้าม ${result.rejected.length} รูป` : '') + ' — เลือกโหมดระบุชื่อเพื่อเริ่มใช้งาน');
    await loadFacePanel();
  } catch (error) { faceMessage(error.message); }
  finally { submit.disabled = false; }
}
async function deleteRegisteredPerson(identityId, displayName) {
  if (!confirm(`ลบข้อมูลใบหน้าของ ${displayName} หรือไม่?`)) return;
  try {
    await faceRequest('/api/face/people/' + encodeURIComponent(identityId), {method:'DELETE'});
    await loadFacePanel(); faceMessage('ลบข้อมูลใบหน้าแล้ว');
  } catch (error) { faceMessage(error.message); }
}
document.getElementById('face-enroll-form').addEventListener('submit', enrollRegisteredPerson);
