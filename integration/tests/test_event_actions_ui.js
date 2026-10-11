const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../../web/static/index.html'),'utf8');
assert(html.includes('/static/event-actions.js?v=event-delete-20261012'));
const renderer=html.slice(html.indexOf('    function renderEvents('),html.indexOf('    async function reviewEvent('));
const nodes=new Map();function element(id){if(!nodes.has(id)){
 const classes=new Set();nodes.set(id,{innerHTML:'',textContent:'',classList:{toggle(c,on){on?classes.add(c):classes.delete(c);},contains:c=>classes.has(c)}});
}return nodes.get(id);}
let requests=[],messages=[],refreshes=0,stats=0,closed=0,priority=[];
const a={event_id:'one',event_type:'fall_detected',source_id:'cam',metadata:{},review_status:'confirmed',severity:'high'},b={...a,event_id:'two'};
const ctx=vm.createContext({console,window:{},document:{getElementById:element},lastEvents:[a,b],eventFilter:'all',eventSearch:'',eventReplay:{id:'one'},
 fetch:async(url,options)=>{requests.push([url,options.method]);return {ok:true,json:async()=>({})};},
 showToast:text=>messages.push(text),fetchEvents:async()=>refreshes++,fetchStats:async()=>stats++,
 updatePriorityAlert:events=>priority=events,closeEventReplay:()=>closed++});
vm.runInContext(renderer+fs.readFileSync(path.join(__dirname,'../../web/static/event-actions.js'),'utf8'),ctx);
(async()=>{
 ctx.renderEvents(ctx.lastEvents);assert(element('events-container').innerHTML.includes('ลบเหตุการณ์'));
 const button={disabled:false};await ctx.deleteEvent('one',button);
 assert.deepEqual(requests[0],['/api/events/one','DELETE']);assert.equal(ctx.lastEvents.length,1);
 assert.equal(priority.length,1);assert.equal(closed,1);assert.equal(stats,1);assert.equal(refreshes,1);assert.equal(button.disabled,false);
 assert(!element('event-delete-notice').classList.contains('hidden'));
 ctx.renderEvents([a,b]);assert(!element('events-container').innerHTML.includes('data-event-id="one"'),'stale polling must not resurrect a deleted card');
 await ctx.undoDeleteEvent(button);assert.deepEqual(requests.at(-1),['/api/events/one/restore','POST']);
 assert.equal(ctx.window.eventDeletedLocally('one'),false);assert(element('event-delete-notice').classList.contains('hidden'));
 let finish;ctx.fetch=()=>new Promise(resolve=>finish=resolve);
 const pending=ctx.deleteEvent('two',button);assert(button.disabled);assert(ctx.window.eventDeletedLocally('two'));
 await ctx.deleteEvent('two',button);finish({ok:false,json:async()=>({detail:'not found'})});await pending;
 assert(!ctx.window.eventDeletedLocally('two'));assert.equal(messages.at(-1),'not found');assert.equal(button.disabled,false);
 console.log('PASS: delete buttons, undo, stats refresh, replay close, stale polling and failed deletion');
})().catch(error=>{console.error(error);process.exitCode=1;});
