"""Open the selected camera; never switch to another camera automatically."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import math
import json
import re
import cv2
import numpy as np


def ffmpeg_binary():
    selected=os.environ.get('NEXORA_FFMPEG') or shutil.which('ffmpeg')
    if selected: return selected
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError, RuntimeError): return None


class CudaCapture:
    """NVDEC decode with explicit failure; no silent CPU fallback."""
    def __init__(self, source):
        self.binary=ffmpeg_binary()
        if not self.binary: raise ValueError('ยังไม่มี FFmpeg สำหรับถอดรหัส GPU ติดตั้ง dependencies หรือใช้ Docker GPU')
        metadata=cv2.VideoCapture(source)
        self.values={k:metadata.get(k) for k in (cv2.CAP_PROP_FPS,cv2.CAP_PROP_FRAME_COUNT,cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT)}
        opened=metadata.isOpened(); metadata.release()
        if not opened: raise ValueError('อ่านข้อมูลคลิปไม่สำเร็จ')
        if any(not math.isfinite(self.values[k]) or self.values[k]<=0 for k in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT)):
            raise ValueError('ขนาดคลิปไม่ถูกต้อง')
        fps=self.values[cv2.CAP_PROP_FPS]
        if not math.isfinite(fps) or fps<=0:self.values[cv2.CAP_PROP_FPS]=30
        self.source=source; self.position=0; self.process=None
        self.error_file=tempfile.TemporaryFile()
        self._start(0)

    def _start(self, frame):
        if self.process: self.release_process()
        self.error_file.seek(0); self.error_file.truncate()
        fps=self.values[cv2.CAP_PROP_FPS] or 30
        command=[self.binary,'-nostdin','-v','error','-hwaccel','cuda','-hwaccel_output_format','cuda',
                 '-ss',str(frame/fps),'-i',self.source,'-an','-vf','hwdownload,format=nv12',
                 '-pix_fmt','bgr24','-f','rawvideo','pipe:1']
        self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=self.error_file,
                                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.position=frame

    def isOpened(self): return self.process is not None
    def getBackendName(self): return 'FFMPEG_NVDEC'
    def get(self,key):
        return self.position if key==cv2.CAP_PROP_POS_FRAMES else self.values.get(key,0)
    def set(self,key,value):
        if key!=cv2.CAP_PROP_POS_FRAMES: return False
        self._start(int(value)); return True
    def read(self):
        w,h=int(self.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size=w*h*3; data=bytearray()
        while len(data)<size:
            chunk=self.process.stdout.read(size-len(data))
            if not chunk: break
            data.extend(chunk)
        if len(data)!=size:
            code=self.process.poll()
            if code is None:
                try: code=self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('ตัวถอดรหัสหยุดส่งภาพแต่ยังไม่ปิด กรุณาเปิดคลิปใหม่')
            if code not in (None,0):
                self.error_file.seek(0); detail=self.error_file.read(1500).decode(errors='replace')
                raise RuntimeError(getattr(self, 'failure_message', 'ถอดรหัส GPU ไม่สำเร็จ กรุณาเลือก CPU หรือตรวจ driver/codec: ')+detail)
            if data: raise RuntimeError('ตัวถอดรหัสส่งเฟรมไม่ครบ กรุณาใช้คลิปที่สมบูรณ์')
            return False,None
        self.position+=1
        return True,np.frombuffer(data,dtype=np.uint8).reshape(h,w,3).copy()
    def release_process(self):
        if self.process.poll() is None: self.process.terminate()
        try: self.process.wait(timeout=2)
        except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=2)
        self.process.stdout.close()
    def release(self):
        if self.process: self.release_process(); self.process=None
        self.error_file.close()


def _avi_metadata(source, binary):
    """Probe outside Python: even metadata must not open the unsafe AVI decoder."""
    sibling = Path(binary).with_name('ffprobe.exe' if os.name == 'nt' else 'ffprobe')
    probe = str(sibling) if sibling.is_file() else shutil.which('ffprobe')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    if probe:
        result = subprocess.run([probe, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,duration:format=duration',
            '-of', 'json', str(source)], capture_output=True, timeout=20, creationflags=flags)
        if result.returncode: raise ValueError('อ่านข้อมูล AVI ไม่สำเร็จ: '+result.stderr.decode(errors='replace')[-500:])
        info = json.loads(result.stdout)
        stream = info['streams'][0]
        numerator, denominator = str(stream['avg_frame_rate']).split('/')
        fps = float(numerator)/float(denominator)
        width, height = int(stream['width']), int(stream['height'])
        if str(stream.get('nb_frames', '')).isdigit():
            total = int(stream['nb_frames'])
        else:
            duration_text = stream.get('duration')
            if duration_text in (None, 'N/A'): duration_text = info.get('format', {}).get('duration', 0)
            total = round(float(duration_text)*fps)
    else:
        # imageio-ffmpeg on Windows bundles FFmpeg without FFprobe.
        result = subprocess.run([binary, '-hide_banner', '-nostdin', '-i', str(source),
            '-map', '0:v:0', '-frames:v', '0', '-an', '-f', 'null', '-'],
            capture_output=True, timeout=20, creationflags=flags)
        text = result.stderr.decode(errors='replace')
        video = next((line for line in text.splitlines() if 'Stream #' in line and 'Video:' in line), '')
        dimensions = re.search(r'\b(\d+)x(\d+)\b', video)
        rate = re.search(r'([\d.]+) fps', video)
        timing = re.search(r'Duration: (\d+):(\d+):([\d.]+)', text)
        if result.returncode or not (dimensions and rate and timing):
            raise ValueError('อ่านข้อมูล AVI ไม่สำเร็จ: '+text[-500:])
        width, height = map(int, dimensions.groups()); fps = float(rate.group(1))
        duration = int(timing[1])*3600+int(timing[2])*60+float(timing[3])
        total = round(duration*fps)
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0 or width*height > 67108864 or total <= 0:
        raise ValueError('ข้อมูลขนาดหรือเวลาของ AVI ไม่ถูกต้อง')
    return {cv2.CAP_PROP_FPS:fps, cv2.CAP_PROP_FRAME_COUNT:total,
            cv2.CAP_PROP_FRAME_WIDTH:width, cv2.CAP_PROP_FRAME_HEIGHT:height}


class AviCapture(CudaCapture):
    """Stream AVI through an isolated FFmpeg process; a decoder abort cannot kill the API."""
    failure_message = 'ถอดรหัส AVI ไม่สำเร็จ: '

    def __init__(self, source):
        self.binary = ffmpeg_binary()
        if not self.binary: raise ValueError('การอ่าน AVI ต้องใช้ FFmpeg กรุณาใช้ Docker หรือ MP4')
        self.values = _avi_metadata(source, self.binary)
        self.source = str(source); self.position = 0; self.process = None
        self.error_file = tempfile.TemporaryFile()
        try: self._start(0)
        except Exception:
            self.error_file.close(); raise

    def _start(self, frame):
        if self.process: self.release_process()
        self.error_file.seek(0); self.error_file.truncate()
        command = [self.binary, '-hide_banner', '-nostdin', '-v', 'error', '-ss',
            str(frame/self.values[cv2.CAP_PROP_FPS]), '-i', self.source,
            '-map', '0:v:0', '-an', '-sn', '-dn', '-fps_mode', 'passthrough', '-pix_fmt', 'bgr24',
            '-f', 'rawvideo', 'pipe:1']
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.error_file,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.position = frame

    def getBackendName(self): return 'FFMPEG_AVI_CPU'


def open_selected_capture(source, decode_device='cpu'):
    if decode_device=='cpu': return open_capture(source)
    if decode_device!='cuda': raise ValueError('อุปกรณ์ถอดรหัสไม่ถูกต้อง')
    if isinstance(source,int): raise ValueError('เว็บแคมรับภาพผ่านระบบกล้อง ใช้ GPU decode เฉพาะไฟล์คลิป')
    if Path(str(source)).suffix.lower() == '.avi':
        raise ValueError('AVI ใช้ CPU สำหรับอ่านภาพ เลือก GPU สำหรับรัน AI ได้ตามเดิม')
    return CudaCapture(source)


def open_capture(source):
    if not isinstance(source, int) and Path(str(source)).suffix.lower() == '.avi':
        return AviCapture(source)
    if not isinstance(source, int) or os.name != 'nt':
        return cv2.VideoCapture(source)
    capture = None
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        capture = cv2.VideoCapture(source, backend)
        if capture.isOpened():
            return capture
        capture.release()
    return capture


def valid_frame(ok, frame):
    return bool(ok and frame is not None and frame.size > 0)


def dark_frame(frame):
    # Advisory only: a dark room can also produce a valid, very dark frame.
    return float(frame.mean()) < 2.0
