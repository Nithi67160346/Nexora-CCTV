/* Display preferences affect the event list, not AI collection or saved reviews. */
const eventCategoryNames={fall:'การล้ม',wandering:'พลัดหลง / ข้ามเขต',violence:'การปะทะ',location:'ตำแหน่ง / เตียง',other:'อื่น ๆ'};
const eventDisplayKey='nexora.eventCategories.v1';
let eventDisplayCategories=new Set(Object.keys(eventCategoryNames));
try{
  const saved=JSON.parse(localStorage.getItem(eventDisplayKey));
  if(Array.isArray(saved))eventDisplayCategories=new Set(saved.filter(name=>Object.hasOwn(eventCategoryNames,name)));
}catch(_){}
window.eventCategoryVisible=function(event){
  const type=String(event.event_type || '');
  const category=event.category || (type.startsWith('fall')?'fall':type.startsWith('wandering')?'wandering':
    ['high_risk_interaction','violence_detected'].includes(type)?'violence':
    ['location_update','identity_update','last_seen_update'].includes(type)?'location':'other');
  return eventDisplayCategories.has(category);
};
function refreshEventDisplay(){
  for(const name of Object.keys(eventCategoryNames))document.getElementById('show-category-'+name).checked=eventDisplayCategories.has(name);
  document.getElementById('event-category-summary').textContent=eventDisplayCategories.size===5?'หมวดที่แสดง: ทุกหมวด':
    eventDisplayCategories.size===0?'หมวดที่แสดง: ไม่เลือก':`หมวดที่แสดง: ${eventDisplayCategories.size}/5`;
  try{localStorage.setItem(eventDisplayKey,JSON.stringify([...eventDisplayCategories]));}catch(_){}
  if(typeof lastEvents!=='undefined'){
    updatePriorityAlert(lastEvents.filter(window.eventCategoryVisible));
    renderEvents(lastEvents);
  }
}
window.setEventCategory=function(name,enabled){
  if(!Object.hasOwn(eventCategoryNames,name))return;
  if(enabled)eventDisplayCategories.add(name);else eventDisplayCategories.delete(name);
  refreshEventDisplay();
};
window.setAllEventCategories=function(enabled){
  eventDisplayCategories=new Set(enabled?Object.keys(eventCategoryNames):[]);refreshEventDisplay();
};
refreshEventDisplay();
