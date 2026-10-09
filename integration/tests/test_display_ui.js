const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const code=fs.readFileSync(path.join(__dirname,'../../web/static/display.js'),'utf8');
function fixture(saved){
  const nodes=new Map(),store=new Map(saved===undefined?[]:[['nexora.eventCategories.v1',saved]]);
  const events=[{event_type:'fall_detected'},{event_type:'wandering_exit'},{event_type:'last_seen_update'},{event_type:'high_risk_interaction'},{event_type:'custom'}];
  let priority=[];
  const ctx=vm.createContext({window:{},localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,v)},
    document:{getElementById:id=>{if(!nodes.has(id))nodes.set(id,{});return nodes.get(id);}},lastEvents:events,
    updatePriorityAlert:events=>priority=events,renderEvents(){}});
  vm.runInContext(code,ctx);return {ctx,store,nodes,events,priority:()=>priority};
}
const a=fixture();assert.equal(a.events.filter(a.ctx.window.eventCategoryVisible).length,5);
a.ctx.window.setAllEventCategories(false);a.ctx.window.setEventCategory('fall',true);a.ctx.window.setEventCategory('wandering',true);
assert.deepEqual(a.events.filter(a.ctx.window.eventCategoryVisible).map(e=>e.event_type),['fall_detected','wandering_exit']);
assert.equal(a.priority().length,2,'hidden categories must disappear from the priority banner too');
const b=fixture(a.store.get('nexora.eventCategories.v1'));
assert.equal(b.nodes.get('show-category-location').checked,false,'preferences survive reload');
b.ctx.window.setAllEventCategories(false);assert.equal(b.priority().length,0);
assert.equal(fixture(b.store.get('nexora.eventCategories.v1')).priority().length,0,'empty selection is intentional');
assert.equal(fixture('bad JSON').priority().length,5,'broken storage falls back to all categories');
assert.equal(fixture('["fall","unknown"]').priority().length,1,'unknown categories are ignored');
console.log('PASS: multiple event categories, persistence, empty selection and priority banner');
