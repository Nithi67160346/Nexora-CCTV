/* Remove individual alerts from the current round, with an undo action. */
const deletedEvents=new Set(),eventDeleteUndo=[];
window.eventDeletedLocally=id=>deletedEvents.has(id);
function updateEventDeleteNotice(){
  document.getElementById('event-delete-notice').classList.toggle('hidden',!eventDeleteUndo.length);
}
async function eventActionRequest(id,action){
  const response=await fetch('/api/events/'+encodeURIComponent(id)+(action==='restore'?'/restore':''),
    {method:action==='restore'?'POST':'DELETE'});
  const result=await response.json();
  if(!response.ok)throw new Error(result.detail || 'จัดการเหตุการณ์ไม่สำเร็จ');
  return result;
}
async function deleteEvent(id,button){
  if(deletedEvents.has(id) || button?.disabled)return;
  if(button)button.disabled=true;
  // Hide pending deletions from stale polling replies, but restore on failure.
  deletedEvents.add(id);
  try{
    await eventActionRequest(id,'delete');
    eventDeleteUndo.push(id);updateEventDeleteNotice();
    lastEvents=lastEvents.filter(e=>e.event_id!==id);
    updatePriorityAlert(window.eventCategoryVisible?lastEvents.filter(window.eventCategoryVisible):lastEvents);
    renderEvents(lastEvents);
    if(typeof eventReplay!=='undefined' && eventReplay?.id===id)closeEventReplay();
    await Promise.all([fetchEvents(),fetchStats()]);
  }catch(error){deletedEvents.delete(id);showToast(error.message);}
  finally{if(button)button.disabled=false;}
}
async function undoDeleteEvent(button){
  if(!eventDeleteUndo.length || button?.disabled)return;
  if(button)button.disabled=true;
  const id=eventDeleteUndo.at(-1);
  try{
    await eventActionRequest(id,'restore');
    deletedEvents.delete(id);
    eventDeleteUndo.splice(eventDeleteUndo.lastIndexOf(id),1);updateEventDeleteNotice();
    await Promise.all([fetchEvents(),fetchStats()]);
  }catch(error){showToast(error.message);}
  finally{if(button)button.disabled=false;}
}
