"""Independent trackers and pipelines for simultaneous sources."""
from copy import deepcopy
import threading


class StreamRegistry:
    def __init__(self, factory, parent, maximum=4):
        self.factory,self.parent,self.maximum=factory,parent,maximum
        self.entries={}; self.lock=threading.RLock()

    def start(self, profile, source, **options):
        cid=profile['id']
        with self.lock:
            previous=self.entries.get(cid)
            if previous and (previous['state']=='starting' or previous['worker'].is_running):
                raise ValueError('แหล่งภาพนี้เปิดอยู่แล้ว กดหยุดก่อนเปลี่ยน')
            occupied=sum(e['state']=='starting' or e['worker'].is_running for e in self.entries.values())+int(self.parent.is_running)
            if occupied>=self.maximum: raise ValueError(f'เปิดพร้อมกันได้สูงสุด {self.maximum} แหล่งภาพ')
            if self.parent.is_running and self.parent.source_id==cid:
                raise ValueError('ห้องนี้กำลังเปิดในตัวเล่นหลัก กดหยุดตัวเล่นหลักก่อน')
            # One physical webcam cannot be opened by two workers.
            if source.isdigit() and ((self.parent.is_running and self.parent.source==source) or any((e['state']=='starting' or e['worker'].is_running) and e['worker'].source==source for e in self.entries.values())):
                raise ValueError('webcam หมายเลขนี้เปิดอยู่แล้ว')
            child=self.factory()
            child.source=source
            child.source_id=cid
            child.cameras.profiles={cid:deepcopy(profile)}; child.cameras.active_id=cid
            child.config=child._product_config(profile['config'],cid)
            child.event_sink=self.parent._record_updates
            entry=dict(worker=child,state='starting',error=None)
            self.entries[cid]=entry
        def launch():
            try:
                child.start_stream(source,**options)
                entry['state']='running'
            except Exception as error:
                entry['state']='error'; entry['error']=str(error)
                child.last_error=str(error); child.stage='error'
        thread=threading.Thread(target=launch,daemon=True);entry['thread']=thread;thread.start()
        return dict(id=cid,state='starting')

    def get(self,cid):
        with self.lock:
            if cid not in self.entries: raise KeyError('ไม่พบแหล่งภาพ')
            return self.entries[cid]['worker']

    def stop(self,cid):
        with self.lock: entry=self.entries.get(cid)
        if not entry: raise KeyError('ไม่พบแหล่งภาพ')
        if entry['state']=='starting': raise ValueError('กำลังเปิดแหล่งภาพ กรุณารอให้เปิดเสร็จก่อนหยุด')
        child=entry['worker']
        child.stop_stream()
        # Keep recording session/evidence for the final segment upload and
        # event replay; release model references once the processing thread ends.
        child.core=child.yolo_model=child.pipeline=None
        entry['state']='stopped'

    def stop_all(self):
        for cid in list(self.entries): self.stop(cid)

    def active(self, cid=None):
        with self.lock:
            return any((cid is None or key==cid) and (e['state']=='starting' or e['worker'].is_running) for key,e in self.entries.items())
