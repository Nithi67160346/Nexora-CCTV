const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const nodes=new Map();let next=0;
function node(id){if(!nodes.has(id)){
  const classes=new Set();
  const item={value:'',textContent:'',disabled:false,children:[],attributes:{},style:{},setAttribute(key,value){this.attributes[key]=value;},getBoundingClientRect(){return {width:400,left:20,top:500,bottom:530};},classList:{add:c=>classes.add(c),remove:c=>classes.delete(c)},append(...items){this.children.push(...items);},replaceChildren(...items){this.children=[...items];}};
  Object.defineProperty(item,'options',{get(){return this.children;}});
  Object.defineProperty(item,'innerHTML',{set(){throw Error('unsafe filename rendering');}});nodes.set(id,item);
}return nodes.get(id);}
const uploads=[];
const context=vm.createContext({console,currentSource:'playing.mp4',
  window:{activeCameraName:'ห้องทดสอบ',innerWidth:567,innerHeight:650,addEventListener(){}},
  document:{getElementById:node,createElement:()=>node('option'+next++),addEventListener(){}},
  FormData:class{append(key,file){this.file=file;}},
  productRequest:async()=>[{path:'existing.mp4',filename:'existing.mp4'}],
  fetch:async(url,options)=>{
    assert.equal(url,'/api/upload');uploads.push(options.body.file.name);
    const name=options.body.file.name;
    return {ok:name!=='broken.mp4',json:async()=>name==='broken.mp4'?{detail:'bad file'}:{path:'uploads/'+name,filename:name}};
  }});
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../../web/static/clip-library.js'),'utf8'),context);
(async()=>{
  await context.loadUploadedClips();assert(node('sample-selector').options.some(row=>row.value==='existing.mp4'));
  await context.handleFileUpload([{name:'one.mp4',size:10},{name:'broken.mp4',size:10},{name:'<unsafe>.avi',size:10},{name:'skip.txt',size:10}]);
  assert.deepEqual(uploads,['one.mp4','broken.mp4','<unsafe>.avi'],'upload every supported file and continue after a failed upload');
  assert.equal(context.currentSource,'playing.mp4','uploading must not replace or start the main source');
  assert(node('sample-selector').options.some(row=>row.textContent==='<unsafe>.avi'));
  assert(node('clip-library-message').textContent.includes('2/4'));
  assert(node('clip-library-message').textContent.includes('bad file'));
  assert.equal(node('video-upload-input').disabled,false);assert.equal(node('video-upload-input').value,'');
  context.currentSource='uploads/one.mp4';await context.loadUploadedClips();
  assert.equal(node('sample-selector').value,'uploads/one.mp4','library refresh must keep current clip selection');
  assert(node('sample-selector').options.some(row=>row.value==='uploads/<unsafe>.avi'),'startup-list reply cannot erase new uploads');
  let deletes=0;
  context.toggleClipLibrary();assert.equal(node('clip-library-toggle').attributes['aria-expanded'],'true');
  const otherRow=node('clip-library-list').children.find(row=>row.children[0]?.textContent==='<unsafe>.avi');
  assert.equal(otherRow.children[1].textContent,'ลบ','each uploaded row has a delete button');
  context.productRequest=async(url,body,method)=>{assert.equal(method,'DELETE');assert.equal(body.source,'uploads/<unsafe>.avi');deletes++;};
  otherRow.children[1].onclick();assert.equal(deletes,0);assert.equal(node('clip-delete-name').textContent,'<unsafe>.avi');
  assert.equal(context.currentSource,'uploads/one.mp4');assert.equal(node('sample-selector').value,'uploads/one.mp4','opening another row confirmation must preserve main selection');
  context.cancelClipDeletion();await context.confirmClipDeletion();assert.equal(deletes,0);
  otherRow.children[1].onclick();await context.confirmClipDeletion();assert.equal(deletes,1);
  assert.equal(context.currentSource,'uploads/one.mp4');assert.equal(node('sample-selector').value,'uploads/one.mp4','deleting another clip must preserve the playing clip');
  deletes=0;
  context.productRequest=async(url,body,method)=>{assert.equal(method,'DELETE');assert.equal(body.source,'uploads/one.mp4');deletes++;};
  context.deleteSelectedClip();assert.equal(deletes,0,'opening confirmation must never send a delete request');
  context.cancelClipDeletion();await context.confirmClipDeletion();assert.equal(deletes,0,'cancel must not delete');
  context.productRequest=async()=>{throw Error('stop first');};
  context.deleteSelectedClip();await context.confirmClipDeletion();assert.equal(context.currentSource,'uploads/one.mp4');assert(node('clip-library-message').textContent.includes('stop first'));
  context.productRequest=async(url,body,method)=>{assert.equal(method,'DELETE');assert.equal(body.source,'uploads/one.mp4');deletes++;};
  context.deleteSelectedClip();await context.confirmClipDeletion();assert.equal(deletes,1);assert.equal(context.currentSource,'');
  assert(!node('sample-selector').options.some(row=>row.value==='uploads/one.mp4'));
  assert(!node('clip-library-list').children.some(row=>row.children[0]?.textContent==='one.mp4'));
  context.productRequest=async()=>[{path:'uploads/one.mp4',filename:'one.mp4'}];
  await context.loadUploadedClips();assert(!node('sample-selector').options.some(row=>row.value==='uploads/one.mp4'),'old list replies cannot restore deleted clips');
  console.log('PASS: multi-upload, selection, safe names, deletion confirmation/cancel/errors, persisted removal and stale-list safety');
})().catch(error=>{console.error(error);process.exitCode=1;});
