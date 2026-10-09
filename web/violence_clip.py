"""User-triggered whole-clip ResNet18/LSTM testing; no per-person events."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import threading
import time
from uuid import uuid4

import cv2
import numpy as np
from .capture import open_capture

MODEL_NAME = 'ResNet18 + LSTM'
FRAME_COUNT = 16


def build_model():
    # All CNN parameters are in the supplied checkpoint. Never fetch ImageNet.
    import torch.nn as nn
    from torchvision import models

    class ResNetLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            resnet = models.resnet18(weights=None)
            self.cnn = nn.Sequential(*list(resnet.children())[:-1])
            self.lstm = nn.LSTM(512, 256, num_layers=1, batch_first=True)
            self.fc = nn.Linear(256, 2)

        def forward(self, x):
            batch, frames, channels, height, width = x.size()
            features = self.cnn(x.reshape(batch * frames, channels, height, width))
            sequence, _ = self.lstm(features.reshape(batch, frames, -1))
            return self.fc(sequence[:, -1, :])

    return ResNetLSTM()


def sample_frames(path, progress=lambda stage, value: None, capture=open_capture):
    cap = capture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('เปิดคลิปไม่ได้ กรุณาอัปโหลด MP4/WebM ที่อ่านได้')
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(total) or total < FRAME_COUNT or not math.isfinite(fps) or fps <= 0:
            raise ValueError('คลิปต้องมีอย่างน้อย 16 เฟรมและมีข้อมูลเวลา อ่านคลิปนี้ไม่ได้')
        indices = np.linspace(0, int(total) - 1, FRAME_COUNT, dtype=int).tolist()
        frames = []
        for index in indices:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError('อ่านตำแหน่งเฟรมของคลิปนี้ไม่ได้ กรุณาใช้ MP4/WebM ที่สมบูรณ์')
            ok, frame = cap.read()
            if not ok or frame is None or not frame.size:
                raise ValueError(f'อ่านเฟรม {index} ไม่ได้ กรุณาใช้คลิปที่สมบูรณ์')
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            progress('sampling', 10 + round(40 * len(frames) / FRAME_COUNT))
        return frames, dict(total_frames=int(total), fps=float(fps),
            duration_s=float(total / fps), sampled_frames=FRAME_COUNT,
            sampled_frame_indices=indices, sampled_times_s=[index / fps for index in indices])
    finally:
        cap.release()


class ViolenceClipService:
    def __init__(self, root, predictor=None):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.cached_model = None
        self.inference_lock = threading.RLock()
        self.predictor = predictor or self.predict

    def weights_path(self):
        return self.root / 'best_lstm_model.pth'

    def options(self):
        path = self.weights_path()
        return dict(model=MODEL_NAME, available=path.is_file() and path.stat().st_size > 0,
            scope='whole_clip', sampled_frames=FRAME_COUNT, threshold=0.5,
            class_mapping={'0': 'normal', '1': 'fighting'})

    def start(self, source, device='cpu'):
        if device not in ('cpu', 'cuda'):
            raise ValueError('เลือก CPU หรือ GPU (NVIDIA) เท่านั้น')
        if not self.options()['available']:
            raise ValueError('ไม่พบ best_lstm_model.pth กรุณาใช้ source/QA ZIP ที่มี weights ครบ')
        source = Path(source).resolve()
        with self.lock:
            if self.active:
                raise RuntimeError('โมเดล LSTM กำลังวิเคราะห์คลิปอยู่ กรุณารอให้เสร็จ')
            while len(self.jobs) >= 8:
                del self.jobs[next(iter(self.jobs))]
            job_id = uuid4().hex
            self.jobs[job_id] = dict(id=job_id, state='running', stage='loading', progress=0,
                filename=source.name, device=device, model=MODEL_NAME, scope='whole_clip',
                created_at=time.time(), result=None, error=None)
            self.active = job_id
            threading.Thread(target=self._run, args=(job_id, source, device), daemon=True).start()
            return self.snapshot(job_id)

    def snapshot(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError('ไม่พบผลทดสอบนี้ กรุณาเริ่มทดสอบใหม่')
            return deepcopy(self.jobs[job_id])

    def _run(self, job_id, source, device):
        def progress(stage, value):
            with self.lock:
                self.jobs[job_id].update(stage=stage, progress=value)
        try:
            before = source.stat()
            result = self.predictor(source, device, progress)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('ไฟล์คลิปเปลี่ยนระหว่างทดสอบ กรุณาทดสอบใหม่')
            with self.lock:
                self.jobs[job_id].update(state='completed', stage='completed', progress=100,
                    result=result, completed_at=time.time())
        except Exception as error:
            with self.lock:
                self.jobs[job_id].update(state='error', stage='error', error=str(error))
        finally:
            with self.lock:
                self.active = None

    def predict(self, source, device, progress):
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise ValueError('GPU ไม่พร้อมใน server/container นี้ เลือก CPU เพื่อทดสอบ')
        progress('sampling', 10)
        frames, metadata = sample_frames(source, progress)
        progress('inference', 65)
        result = self.predict_frames(frames, device)
        return dict(**metadata, **result, scope='whole_clip', person_ids=[], event_intervals=[])

    def predict_frames(self, frames, device='cpu'):
        # All cameras and the manual clip test share one serialized model cache.
        with self.inference_lock:
            return self._predict_frames(frames, device)

    def _predict_frames(self, frames, device):
        import torch
        from torchvision import transforms
        if device not in ('cpu', 'cuda'):
            raise ValueError('เลือก CPU หรือ GPU (NVIDIA) เท่านั้น')
        if len(frames) != FRAME_COUNT:
            raise ValueError('LSTM ต้องใช้ภาพจริงครบ 16 เฟรม')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('GPU ไม่พร้อมใน server/container นี้ เลือก CPU เพื่อทดสอบ')
        path = self.weights_path()
        weight_version = (path.stat().st_size, path.stat().st_mtime_ns)
        if self.cached_model is None or self.cached_model[:2] != (device, weight_version):
            model = build_model()
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            model.eval().to(device)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            self.cached_model = (device, weight_version, model, checksum)
        _, _, model, checksum = self.cached_model
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        tensor = torch.stack([transform(frame) for frame in frames]).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
        if len(probabilities) != 2 or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
            raise ValueError('โมเดลให้คะแนนไม่ถูกต้อง ยังสรุปผลไม่ได้')
        return dict(fighting=probabilities[1] > 0.5,
            probability_fighting=probabilities[1], probability_normal=probabilities[0],
            threshold=0.5, weights_sha256=checksum, model=MODEL_NAME,
            class_mapping={'0': 'normal', '1': 'fighting'})


_shared_services = {}
_services_lock = threading.Lock()


def shared_service(root):
    key = str(Path(root).resolve())
    with _services_lock:
        if key not in _shared_services:
            _shared_services[key] = ViolenceClipService(root)
        return _shared_services[key]


def resolve_clip(source, allowed_directories):
    if not isinstance(source, str) or not source or '://' in source:
        raise ValueError('เลือกหรืออัปโหลดไฟล์คลิปก่อนทดสอบ LSTM ใช้กับกล้องสดไม่ได้')
    path = Path(source).resolve()
    if not any(path.is_relative_to(Path(folder).resolve()) for folder in allowed_directories):
        raise ValueError('กรุณาอัปโหลดคลิปผ่านเว็บก่อนทดสอบ')
    if path.suffix.lower() not in ('.mp4', '.webm', '.avi', '.mov', '.mkv', '.m4v') or not path.is_file():
        raise ValueError('ไม่พบไฟล์คลิปที่อ่านได้ กรุณาอัปโหลดใหม่')
    return path
