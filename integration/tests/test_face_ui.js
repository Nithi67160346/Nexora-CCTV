const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../../web/static/face.js'), 'utf8');
function fixture(fetch) {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) {
      const node = {textContent:'', value:'off', disabled:false, children:[], files:[],
        classList:{add(){}, remove(){}}, replaceChildren(){this.children=[];},
        append(...children){this.children.push(...children);}, addEventListener(){}};
      Object.defineProperty(node, 'innerHTML', {set(){throw new Error('Names must not use innerHTML');}});
      elements.set(id, node);
    }
    return elements.get(id);
  }
  const ctx = vm.createContext({fetch, console, confirm:()=>true, FormData:class {},
    document:{getElementById:element, createElement:()=>element('new-'+elements.size)}});
  vm.runInContext(source, ctx);
  return {ctx,element};
}
const ready = {mode:'off', detector_ready:true, embedder_ready:true, registered_count:0, people:[]};
const reply = (body, ok=true) => ({ok, json:async()=>body});
(async()=>{
  const a = fixture(()=>Promise.resolve(reply(ready)));
  a.ctx.updateFaceStatus({...ready, recognition_active:true}, [{resident_id:'a', identity_name:'ผู้ทดสอบ', track_id:7, person_role:'resident'}, {resident_id:null, track_id:8}]);
  assert(a.element('face-runtime-status').textContent.includes('ระบุชื่อ'));
  assert(a.element('face-active-people').children[0].textContent.includes('ผู้ทดสอบ'));
  assert(a.element('face-active-people').children[1].textContent.includes('ไม่ทราบชื่อ'));
  a.ctx.renderRegisteredPeople([{identity_id:'a', display_name:'<img src=x onerror=alert(1)>', role:'visitor', samples:3}]);
  assert(a.element('face-registered-people').children[0].children[0].textContent.includes('<img'));
  let posts=0;
  const b=fixture((url, options)=>{if(options?.method==='POST'){posts++;return Promise.resolve(reply({detail:'ลงทะเบียนก่อน'},false));}return Promise.resolve(reply(ready));});
  b.element('face-mode').value='recognize'; await b.ctx.changeFaceMode();
  assert.equal(posts,1); assert.equal(b.element('face-mode').value,'off'); assert.equal(b.element('face-mode').disabled,false);
  assert.equal(b.element('face-message').textContent,'ลงทะเบียนก่อน');
  b.element('face-images').files=[{size:10}];
  await b.ctx.enrollRegisteredPerson({preventDefault(){},currentTarget:{}});
  assert.equal(posts,1,'invalid file count must not submit');
  console.log('PASS: face modes, unknown identities, safe names and failed enrollment/mode states');
})().catch(error=>{console.error(error);process.exitCode=1;});
