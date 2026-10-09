"""Newest-frame camera transport. Recording is independent in the browser."""
from queue import Queue, Empty, Full
import time
import threading
import math
import cv2


class BrowserCapture:
    def __init__(self, width, height, fps=10):
        self.width,self.height,self.fps=width,height,fps
        self.queue=Queue(maxsize=1)
        self.opened=True
        self.timestamp_ms=0
        self.last_submitted_ms=-1
        self.lock=threading.Lock()

    def submit(self,frame,timestamp_ms):
        with self.lock: self._submit(frame,timestamp_ms)

    def _submit(self,frame,timestamp_ms):
        if frame.shape[1]!=self.width or frame.shape[0]!=self.height:
            raise ValueError('ขนาดภาพจากกล้องเปลี่ยน กรุณาเปิดกล้องใหม่')
        if not self.opened: raise ValueError('กล้องนี้หยุดแล้ว')
        if not math.isfinite(timestamp_ms) or timestamp_ms<=self.last_submitted_ms: raise ValueError('เวลาภาพต้องเพิ่มตามลำดับ')
        self.last_submitted_ms=timestamp_ms
        try: self.queue.put_nowait((frame,float(timestamp_ms)))
        except Full:
            try: self.queue.get_nowait()
            except Empty: pass
            self.queue.put_nowait((frame,float(timestamp_ms)))
    def read(self):
        try:
            frame,self.timestamp_ms=self.queue.get(timeout=.5)
            return True,frame
        except Empty: return False,None
    def get(self,key):
        return {cv2.CAP_PROP_FPS:self.fps,cv2.CAP_PROP_FRAME_WIDTH:self.width,
                cv2.CAP_PROP_FRAME_HEIGHT:self.height}.get(key,0)
    def isOpened(self): return self.opened
    def getBackendName(self): return 'BROWSER_CAMERA'
    def release(self): self.opened=False
