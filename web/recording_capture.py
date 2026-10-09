"""Read and record local webcams continuously while AI consumes newest frames."""
from queue import Queue, Empty, Full
import threading
import time
import subprocess
import tempfile
import math
import cv2
from web.capture import ffmpeg_binary


class H264Writer:
    def __init__(self,path,fps,width,height):
        self.error_file=tempfile.TemporaryFile()
        self.process=subprocess.Popen([ffmpeg_binary(),'-nostdin','-v','error','-y','-f','rawvideo',
            '-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an',
            '-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],
            stdin=subprocess.PIPE,stderr=self.error_file,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def isOpened(self): return self.process.poll() is None
    def write(self,frame): self.process.stdin.write(frame.tobytes())
    def release(self):
        self.process.stdin.close()
        try: code=self.process.wait(timeout=10)
        except subprocess.TimeoutExpired: self.process.kill();self.process.wait();code=-1
        self.error_file.close()
        if code: raise ValueError('เข้ารหัสวิดีโอไม่สำเร็จ ตรวจ FFmpeg / พื้นที่ดิสก์')


class RecordingCapture:
    def __init__(self, capture, store, source_id, session_id, segment_seconds=10):
        self.capture,self.store=capture,store
        self.source_id,self.session_id=source_id,session_id
        self.segment_seconds=segment_seconds
        self.queue=Queue(maxsize=1); self.opened=capture.isOpened()
        self.timestamp_ms=0; self.error=None; self.recording_error=None
        self.closed=threading.Event(); self.writer=None; self.row=None
        self.thread=threading.Thread(target=self._pump,daemon=True)
        if self.opened: self.thread.start()

    def _close_segment(self, end_s):
        if self.writer:
            try:
                self.writer.release()
                self.row['media_duration_s']=self.recorded_frames/self.record_fps
                self.store.finish(self.row,max(end_s,self.row['start_s']+.001))
            except Exception as error: self.recording_error=str(error)
            self.writer=None
            self.row=None

    def _pump(self):
        started=time.monotonic()
        failed_reads=0
        try:
            while not self.closed.is_set():
                ok,frame=self.capture.read()
                if not ok or frame is None or not frame.size:
                    failed_reads+=1
                    if failed_reads>=20: break
                    self.closed.wait(.1)
                    continue
                failed_reads=0
                now=time.monotonic()-started
                if self.row and now-self.row['start_s']>=self.segment_seconds: self._close_segment(now)
                if not self.writer and not self.recording_error:
                    try:
                        encoder=ffmpeg_binary()
                        self.row=self.store.begin(self.source_id,self.session_id,now,'.mp4' if encoder else '.avi')
                        fps=self.capture.get(cv2.CAP_PROP_FPS)
                        fps=fps if math.isfinite(fps) and 1<=fps<=120 else 30
                        self.record_fps=fps;self.recorded_frames=0
                        path=self.store.directory/self.row['filename']
                        self.writer=H264Writer(path,fps,frame.shape[1],frame.shape[0]) if encoder else cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),fps,(frame.shape[1],frame.shape[0]))
                        if not self.writer.isOpened(): raise ValueError('เปิดตัวบันทึก webcam ไม่สำเร็จ')
                    except Exception as error:
                        self.recording_error=str(error)
                        if self.writer: self.writer.release(); self.writer=None
                if self.writer:
                    try: self.writer.write(frame); self.recorded_frames+=1
                    except Exception as error:
                        self.recording_error=str(error); self._close_segment(now)
                try: self.queue.put_nowait((frame,now*1000))
                except Full:
                    try: self.queue.get_nowait()
                    except Empty: pass
                    self.queue.put_nowait((frame,now*1000))
        except Exception as error: self.error=str(error)
        finally:
            self._close_segment(time.monotonic()-started)
            self.capture.release(); self.opened=False

    def read(self):
        try:
            frame,self.timestamp_ms=self.queue.get(timeout=.5)
            return True,frame
        except Empty:
            if self.error: raise RuntimeError(self.error)
            return False,None
    def get(self,key): return self.capture.get(key)
    def getBackendName(self): return self.capture.getBackendName()+' + RECORDING'
    def isOpened(self): return self.opened or not self.queue.empty()
    def release(self):
        self.closed.set()
        if self.thread.is_alive(): self.thread.join(timeout=1)
