"""Recorded video segments, with a 24-hour retention policy and disk budget."""
from pathlib import Path
import json
import math
import shutil
import threading
import time
import re
from uuid import uuid4


class MediaStore:
    def __init__(self, directory, retention_hours=24, max_bytes=10*1024**3):
        self.directory = Path(directory)
        self.retention_hours, self.max_bytes = retention_hours, max_bytes
        self.lock = threading.RLock()
        self.directory.mkdir(parents=True, exist_ok=True)

    def list(self, source_id=None):
        with self.lock:
            rows=[]
            for p in self.directory.glob('*.json'):
                try:
                    row=json.loads(p.read_text(encoding='utf-8'))
                    if not isinstance(row,dict) or not row.get('complete') or not isinstance(row.get('created_at'),(int,float)) or not math.isfinite(row['created_at']): continue
                    if not all(isinstance(row.get(k),str) for k in ('source_id','session_id')):continue
                    if not re.fullmatch(r'[0-9a-f]{32}',row.get('id','')) or row.get('filename') not in (row['id']+'.webm',row['id']+'.mp4',row['id']+'.avi'): continue
                    if not all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) for k in ('start_s','end_s')) or row['end_s']<=row['start_s']: continue
                    media=(self.directory/row['filename']).resolve()
                    if media.parent!=self.directory.resolve() or not media.is_file(): continue
                    if source_id is None or row['source_id']==source_id: rows.append(row)
                except (OSError, ValueError, KeyError, TypeError): continue
            return sorted(rows,key=lambda r:r['created_at'],reverse=True)

    def cleanup(self):
        # UUID files only; expire interrupted segments after a crash too.
        cutoff=time.time()-self.retention_hours*3600
        with self.lock:
            for row in self.list():
                if row['created_at'] < cutoff:
                    self.path(row['id']).unlink(missing_ok=True)
                    (self.directory/(row['id']+'.json')).unlink(missing_ok=True)
            for p in self.directory.iterdir():
                if re.fullmatch(r'[0-9a-f]{32}\.(?:mp4|webm|avi|json|tmp)',p.name) and not p.is_symlink() and p.is_file() and p.stat().st_mtime<cutoff:
                    p.unlink(missing_ok=True)

    def begin(self, source_id, session_id, start_s, suffix):
        if suffix not in ('.webm','.mp4','.avi') or not math.isfinite(start_s) or start_s<0:
            raise ValueError('ข้อมูลวิดีโอไม่ถูกต้อง')
        self.cleanup()
        used=sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())
        if used>=self.max_bytes or shutil.disk_usage(self.directory).free < 256*1024**2:
            raise ValueError('พื้นที่บันทึกเต็ม กรุณาส่งออกวิดีโอหรือเพิ่มพื้นที่ดิสก์')
        rid=uuid4().hex
        return dict(id=rid,source_id=source_id,session_id=session_id,start_s=float(start_s),
                    end_s=None,created_at=time.time(),filename=rid+suffix,complete=False)

    def finish(self, row, end_s):
        if not math.isfinite(end_s) or end_s<=row['start_s']:
            raise ValueError('ช่วงเวลาบันทึกไม่ถูกต้อง')
        path=self.directory/row['filename']
        if not path.is_file() or path.stat().st_size==0: raise ValueError('วิดีโอไม่มีข้อมูล')
        row=dict(row,end_s=float(end_s),complete=True,bytes=path.stat().st_size)
        with self.lock:
            target=self.directory/(row['id']+'.json')
            temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(row,ensure_ascii=False),encoding='utf-8')
            temporary.replace(target)
        return row

    def path(self, recording_id):
        if not isinstance(recording_id,str) or len(recording_id)!=32 or any(c not in '0123456789abcdef' for c in recording_id):
            raise ValueError('รหัสวิดีโอไม่ถูกต้อง')
        row=json.loads((self.directory/(recording_id+'.json')).read_text())
        path=(self.directory/row['filename']).resolve()
        if path.parent != self.directory.resolve() or not path.is_file(): raise ValueError('ที่อยู่วิดีโอไม่ถูกต้อง')
        return path

    def for_event(self, event):
        md=event.get('metadata',{}); session=md.get('recording_session')
        when=float(event.get('end_timestamp_ms',0))/1000
        rows=[r for r in self.list(event.get('source_id')) if r['session_id']==session and r['start_s']<=when<=r['end_s']]
        if not rows: return None
        row=rows[0]
        scale=row.get('media_duration_s',row['end_s']-row['start_s'])/(row['end_s']-row['start_s'])
        return dict(recording=row,time_s=(when-row['start_s'])*scale)
