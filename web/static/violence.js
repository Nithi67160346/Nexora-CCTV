/* Whole-clip classifier results stay attached to their own clip/job. */
let violenceJob = null, violencePollRevision = 0, violenceAvailable = false;
const violenceNode = id => document.getElementById('violence-'+id);
async function violenceRequest(url, body) {
  const options = body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)};
  const response = await fetch(url, options), data = await response.json();
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'ทดสอบคลิปไม่สำเร็จ');
  return data;
}
function renderViolenceJob(job) {
  violenceJob = job;
  violenceNode('source').textContent = 'คลิปที่ทดสอบ: '+job.filename;
  violenceNode('start').disabled = job.state === 'running' || !violenceAvailable;
  violenceNode('export').classList.toggle('hidden', job.state !== 'completed');
  violenceNode('result').textContent = '';
  if (job.state === 'running') {
    const stage = {loading:'กำลังโหลดโมเดล', sampling:'กำลังอ่าน 16 เฟรม', inference:'กำลังวิเคราะห์'};
    violenceNode('message').textContent = (stage[job.stage] || 'กำลังวิเคราะห์')+' • '+job.progress+'% • '+job.device.toUpperCase();
  } else if (job.state === 'error') {
    violenceNode('message').textContent = job.error || 'ทดสอบไม่สำเร็จ';
  } else {
    const result = job.result;
    violenceNode('result').className = 'text-sm whitespace-pre-line '+(result.fighting ? 'text-amber-300' : 'text-emerald-300');
    violenceNode('message').textContent = 'วิเคราะห์เสร็จ • '+job.device.toUpperCase();
    violenceNode('result').textContent = (result.fighting ? 'โมเดลจัดคลิปนี้เป็นการทำร้ายร่างกาย' : 'โมเดลจัดคลิปนี้เป็นปกติ')+
      '\nคะแนนคลาสทำร้าย: '+(result.probability_fighting*100).toFixed(2)+'% (เกณฑ์มากกว่า 50%)'+
      '\nใช้ '+result.sampled_frames+' เฟรม จากคลิป '+result.duration_s.toFixed(2)+' วินาที';
  }
}
function closeViolenceTest() {
  violencePollRevision++;
  violenceNode('modal').classList.add('hidden');
}
async function pollViolenceJob(id, revision) {
  try {
    const job = await violenceRequest('/api/violence-clip/'+encodeURIComponent(id));
    if (revision !== violencePollRevision) return;
    renderViolenceJob(job);
    if (job.state === 'running') setTimeout(() => pollViolenceJob(id, revision), 800);
  } catch (error) {
    if (revision !== violencePollRevision) return;
    violenceNode('message').textContent = error.message;
    violenceNode('start').disabled = !violenceAvailable;
  }
}
async function openViolenceTest() {
  const revision = ++violencePollRevision;
  violenceNode('modal').classList.remove('hidden');
  violenceNode('device').value = document.getElementById('inference-device').value;
  violenceNode('start').disabled = true;
  violenceNode('message').textContent = 'กำลังตรวจโมเดล';
  if(violenceJob && currentSource && currentSource.split(/[\\/]/).pop()!==violenceJob.filename){
    violenceJob=null;violenceNode('result').textContent='';violenceNode('export').classList.add('hidden');
  }
  if (!violenceJob) violenceNode('source').textContent = currentSource ? 'คลิปที่เลือก: '+currentSource.split(/[\\/]/).pop() : 'ยังไม่ได้เลือกคลิป';
  try {
    const options = await violenceRequest('/api/violence-clip/options');
    if (revision !== violencePollRevision) return;
    violenceAvailable = options.available;
    violenceNode('start').disabled = !violenceAvailable;
    violenceNode('message').textContent = violenceAvailable ? 'โมเดลพร้อม • ทดสอบคลิปที่เลือกได้' : 'ไม่พบ weights โมเดล LSTM กรุณาใช้ source/QA ZIP ที่มี best_lstm_model.pth ครบ';
    if (violenceJob) await pollViolenceJob(violenceJob.id, revision);
  } catch (error) {
    if (revision === violencePollRevision) violenceNode('message').textContent = error.message;
  }
}
async function runViolenceClip() {
  const source = currentSource, revision = ++violencePollRevision;
  violenceNode('start').disabled = true;
  violenceNode('result').textContent = '';
  violenceNode('export').classList.add('hidden');
  violenceNode('message').textContent = 'กำลังเริ่มทดสอบ';
  try {
    const job = await violenceRequest('/api/violence-clip', {source, device:violenceNode('device').value});
    if (revision !== violencePollRevision) { violenceJob = job; return; }
    renderViolenceJob(job);
    await pollViolenceJob(job.id, revision);
  } catch (error) {
    if (revision !== violencePollRevision) return;
    violenceNode('message').textContent = error.message;
    violenceNode('start').disabled = !violenceAvailable;
  }
}
function exportViolenceResult() {
  if (violenceJob?.state !== 'completed') return;
  const anchor = document.createElement('a');
  anchor.href = '/api/violence-clip/'+encodeURIComponent(violenceJob.id)+'/report';
  anchor.download = 'nexora-lstm-'+violenceJob.id+'.json';
  document.body.append(anchor); anchor.click(); anchor.remove();
}
