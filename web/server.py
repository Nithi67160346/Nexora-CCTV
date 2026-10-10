"""NEXORA AI Monitoring - Web Application Server
Provides REST APIs, real-time MJPEG video streaming with AI overlays,
and video scrubbing / seeking capabilities.
"""
import asyncio
from copy import deepcopy
import json
import hashlib
import logging
import math
import os
import re
import sqlite3
from pathlib import Path
import threading
import time
import zlib
from typing import Dict, List, Optional
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel

def _preferred_device() -> str:
    if os.environ.get('NEXORA_DEVICE')=='cpu': return 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

# Project paths
ROOT = Path(__file__).resolve().parents[1]
SETTINGS_DIR = Path(os.environ.get('NEXORA_DATA_DIR', str(ROOT / "local_only")))
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(SETTINGS_DIR.resolve()))

from integration.core import SharedYoloCore
from integration.person_filter import clean_pose, DRAW_JOINT_CONFIDENCE
from integration.pipeline import ProgressPipeline, load_config
from integration.feature_policy import ARCHIVED_FEATURES, apply_web_feature_policy, is_archived_event
from integration.runtime_identity import launcher_info
from web.presentation import fall_display
from web.capture import open_capture, valid_frame, dark_frame
from web.capture import open_selected_capture, ffmpeg_binary
from web.browser_capture import BrowserCapture
from web.event_view import localized_event, event_category
from web.playback import AnnotationBuffer
from web.frame_playback import FramePlayback
from web.media_store import MediaStore
from web.recording_capture import RecordingCapture
from web.multistream import StreamRegistry
from web.model_catalog import POSE_MODELS, resolve_pose_weights, model_options
from web.face_service import FaceService
from web.face_overlay import draw_person_label
from web.camera_profiles import CameraProfiles, apply_zones, clean_text
from web.runtime_health import snapshot as health_snapshot
from web.qa_service import QaRun, EVENT_FEATURES
from web.telegram_service import TelegramDrafts
from web.violence_clip import shared_service, resolve_clip

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nexora_web")
media_store = MediaStore(SETTINGS_DIR/'recordings', retention_hours=24)
violence_clip_service = shared_service(ROOT)
PRODUCT_REVISION = json.loads((ROOT/'web/runtime_version.json').read_text(encoding='utf-8'))['revision']

# Skeleton definition for COCO-17
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Facial keypoints
    (0, 5), (0, 6),                  # Nose to shoulders
    (5, 6),                          # Shoulder to shoulder
    (5, 7), (7, 9),                  # Left arm
    (6, 8), (8, 10),                 # Right arm
    (5, 11), (6, 12), (11, 12),      # Torso
    (11, 13), (13, 15),              # Left leg
    (12, 14), (14, 16),              # Right leg
]

SKELETON_COLORS = [
    (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),
    (0, 255, 255), (0, 255, 255),
    (0, 255, 0),
    (0, 200, 255), (0, 200, 255),
    (255, 0, 128), (255, 0, 128),
    (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (255, 255, 0), (255, 255, 0),
    (255, 0, 255), (255, 0, 255),
]


class StreamWorker:
    def __init__(self):
        self.lock = threading.RLock()
        self.lifecycle_lock = threading.RLock()
        self.frame_playback = FramePlayback()
        self.frame_by_frame = False
        self.playback_client = None
        self.is_running = False
        self.is_paused = False
        self.generation_id = 0
        self.thread: Optional[threading.Thread] = None

        self.source = ""
        self.source_id = "cam_01"
        self.device = _preferred_device()
        self.loop_video = True
        self.is_live = False
        self.playback_speed = 1.0
        self.decode_device = 'cpu'
        self.review_mode = False
        self.paced_fallback = False
        self.annotations = AnnotationBuffer()
        self.evidence_revision = uuid4().hex
        self.pending_seek_preserve_cache = False
        self.stage = 'idle'
        self.session_id = None
        self.browser_capture = None
        self.event_sink = None
        self.capture = None
        self.alert_tracks = {}
        self.fall_diagnostics = {}
        self.analysis_frames = 0
        self.empty_person_frames = 0
        self.processing_ms = {}
        self.source_dimensions = None

        # Video metadata & seeking
        self.total_frames = 0
        self.duration_sec = 0.0
        self.current_sec = 0.0
        self.current_frame_id = 0
        self.video_fps = 30.0
        self.video_width = 1920
        self.video_height = 1080
        self.pending_seek_frame: Optional[int] = None

        # Shared pose extraction remains unchanged for Fall and Interaction.
        self.weights_path = ROOT / "models/yolo26n-pose.pt"
        if not self.weights_path.is_file():
            alt = ROOT.parent / "seizure-detection-starter/yolo26n-pose.pt"
            if alt.is_file():
                self.weights_path = alt

        self.latest_frame_jpeg: Optional[bytes] = None
        self.fps = 0.0
        self.active_tracks: List[dict] = []
        self.person_filter_report = None
        self.current_alerts: List[dict] = []
        self.last_error = None
        self.camera_warning = None
        self.capture_backend = None
        self.capture_error = self.detector_error = None
        self.last_frame_at = None
        self.qa_run = None
        self.telegram = TelegramDrafts()

        self.events: List[dict] = []
        self.notify_last_seen = False
        self.review_stats = {"total": 0, "confirmed": 0, "false_alarm": 0, "needs_review": 0}

        # Pipeline and Models
        self.yolo_model = None
        self.core: Optional[SharedYoloCore] = None
        self.pipeline: Optional[ProgressPipeline] = None
        self.face_service = FaceService(ROOT)
        self.config = self._product_config(load_config(ROOT / "integration/progress.yaml"))
        self.cameras = CameraProfiles(self.config)

        # Respect the shared configuration, including Fall disabled by default.
        if "features" in self.config:
            self.config["features"].setdefault("fall", {
                "enabled": False,
                "window_size": 10,
                "fall_ratio_threshold": 0.4,
                "abnormal_ratio_threshold": 0.3,
                "alert_cooldown_sec": 15,
            })

    def _product_config(self, config, source_id=None):
        candidate = apply_web_feature_policy(config)
        candidate.setdefault('features', {}).setdefault('violence', {})['backend'] = 'resnet18_lstm'
        fall = candidate['features'].setdefault('fall', {'enabled':False})
        if fall.get('backend') != 'yolo_pose_rf_v2':
            fall.update(backend='yolo_pose_rf_v2', window_size=6, fall_ratio_threshold=.3,
                        abnormal_ratio_threshold=.3, fall_proba_threshold=.4, alert_cooldown_sec=30)
        location = candidate.get('features', {}).get('location')
        if location is not None:
            location['config'] = self.face_service.location_config(location.get('config', {}))
            for zone in location['config'].get('zones', {}).values():
                if isinstance(zone, dict):
                    zone['source_ids'] = [source_id or self.source_id]
        return candidate

    def initialize_models(self, device=None):
        from ultralytics import YOLO
        device = device or _preferred_device()
        if device == 'cuda' and _preferred_device() != 'cuda':
            device = 'cpu'
        self.weights_path = resolve_pose_weights(ROOT, self.weights_path.name, self.weights_path)
        logger.info(f"Loading YOLO model on {device} from {self.weights_path}")
        self.yolo_model = YOLO(str(self.weights_path.resolve()))
        if self.yolo_model.task != 'pose' or list(self.yolo_model.model.yaml.get('kpt_shape', [])) != [17, 3]:
            self.yolo_model = None
            raise ValueError('โมเดลต้องเป็น pose ของคน 17 ข้อต่อ ห้ามใช้โมเดล detection แทน')
        self.device = device
        self.core = SharedYoloCore(self.yolo_model, device=self.device)
        self.config = self._product_config(self.config)
        self.pipeline = ProgressPipeline.from_config(self.config)
        self.cameras.save_active(self.config)
        logger.info(f"ProgressPipeline initialized with modules: {list(self.pipeline.modules.keys())}")

    def update_config(self, new_config: dict, camera_id=None, preserve_qa=False):
        with self.lock:
            candidate = self._product_config(new_config, camera_id)
            replacement = ProgressPipeline.from_config(candidate)
            mode = self.face_service.read()[2]
            location = replacement.modules.get('location') if hasattr(replacement, 'modules') else None
            if mode != 'off' and location is not None:
                if location.face is None or (mode == 'recognize' and not location.face.recognition_enabled):
                    raise ValueError('โมเดลใบหน้าเปิดไม่สำเร็จ การตั้งค่าเดิมยังคงอยู่')
            if self.pipeline:
                self._record_updates(self.pipeline.flush())
            if not preserve_qa and self.qa_run and self.qa_run.state == 'running':
                self.qa_run.finish(False, 'การตั้งค่าหรือโมเดลเปลี่ยนระหว่าง QA')
            self.config, self.pipeline = candidate, replacement
            self.annotations.clear()
            self.alert_tracks.clear()
            self.evidence_revision = uuid4().hex
            if camera_id:
                self.cameras.active_id = self.source_id = camera_id
            self.cameras.save_active(self.config)
            logger.info("Pipeline reloaded with updated configuration")

    def _record_updates(self, updates):
        with self.lock:
            forwarded = []
            for ev in updates:
                if ev.get('event_type')=='last_seen_update' and not self.notify_last_seen:
                    continue
                ev = localized_event(ev, self.source, self.session_id if self.is_live else None)
                if ev.get('source_id') == self.source_id and ev['metadata'].get('playback_source') == self.source:
                    ev['metadata'].setdefault('playback_evidence', dict(session_id=self.session_id,
                        revision=self.evidence_revision, source_fps=self.video_fps,
                        cache=self.annotations.cache_path.name if self.annotations.cache_path else None))
                if ev.get('source_id')==self.source_id and ev.get('severity') in ('high','critical','medium','warning'):
                    expiry=float(ev.get('end_timestamp_ms',0))+5000
                    for tid in ev.get('track_ids',[]): self.alert_tracks[tid]=expiry
                location = self.healthy_module('location')
                if location and ev.get('event_type') in ('fall_detected', 'fall_abnormal_movement', 'high_risk_interaction'):
                    ev.setdefault('metadata', {})['people'] = [
                        dict(track_id=tid, **identity) for tid in ev.get('track_ids', [])
                        if (identity := location.get_identity(tid, ev['source_id'])) and identity.get('identity_id')]
                existing = next((e for e in self.events if e['event_id'] == ev['event_id'] and e['source_id'] == ev['source_id']), None)
                if existing:
                    if existing.get('metadata', {}).get('people') and not ev.get('metadata', {}).get('people'):
                        ev.setdefault('metadata', {})['people'] = deepcopy(existing['metadata']['people'])
                    existing.update(ev)
                else:
                    ev = deepcopy(ev)
                    ev.update(review_status='needs_review', human_verdict=None, reviewed_at=None)
                    self.events.insert(0, ev)
                    self.events = self.events[:500]
                camera_name = self.cameras.profiles.get(ev['source_id'], {}).get('name', ev['source_id'])
                self.telegram.record(ev, camera_name, self.current_sec)
                forwarded.append(deepcopy(ev))
            self._recalc_review_stats()
            if self.event_sink and updates:
                self.event_sink(forwarded)

    def healthy_module(self, name):
        if not self.pipeline or getattr(self.pipeline, 'health', {}).get(name, {}).get('state') == 'error':
            return None
        return getattr(self.pipeline, 'modules', {}).get(name)

    def start_stream(self, source: str, loop: bool = True, device: Optional[str] = None, qa_run=None,
                     decode_device='cpu', review_mode=False, browser_capture=None, model=None,
                     frame_by_frame=False, playback_client=None):
        with self.lifecycle_lock:
            return self._start_stream(source, loop, device, qa_run, decode_device, review_mode, browser_capture, model,
                                      frame_by_frame, playback_client)

    def _start_stream(self, source, loop, device, qa_run, decode_device, review_mode, browser_capture, model,
                      frame_by_frame, playback_client):
        selected = resolve_pose_weights(ROOT, model or self.weights_path.name, self.weights_path)
        if device == 'cuda' and _preferred_device() != 'cuda':
            raise ValueError('เครื่องนี้ยังใช้ GPU สำหรับ AI ไม่ได้ กรุณาเลือก CPU')
        self.stop_stream()
        with self.lock:
            self.generation_id += 1
            gen_id = self.generation_id
            self.source = source
            if selected != self.weights_path.resolve() or (device and device != self.device):
                self.core = self.yolo_model = self.pipeline = None
            self.weights_path = selected
            self.playback_speed = 1.0
            self.stage = 'loading_model'
            self.decode_device, self.review_mode = decode_device, bool(review_mode)
            self.paced_fallback = False
            self.session_id = uuid4().hex
            self.frame_by_frame = bool(frame_by_frame)
            self.playback_client = playback_client
            if self.frame_by_frame:
                self.review_mode = False
                self.frame_playback.reset(self.session_id, playback_client)
            self.evidence_revision = uuid4().hex
            self.browser_capture = browser_capture
            self.annotations.clear()
            self.alert_tracks.clear()
            self.fall_diagnostics.clear()
            self.analysis_frames=self.empty_person_frames=0
            self.processing_ms = {}
            self.cameras.save_active(self.config, source)
            if qa_run is not None:
                self.qa_run = qa_run
            self.loop_video = loop
            self.is_live = str(source).isdigit() or str(source).startswith(("rtsp://", "http://", "https://", "browser://"))
            self.annotations.enable_disk(None if self.is_live else ROOT/'local_only/playback_cache')
            self.pending_seek_frame = None
            self.pending_seek_preserve_cache = False

            if device:
                if device not in ('cpu','cuda'): raise ValueError('เลือกอุปกรณ์ AI เป็น CPU หรือ GPU')
                if device == 'cuda' and _preferred_device() != 'cuda': raise ValueError('เครื่องนี้ยังใช้ GPU สำหรับ AI ไม่ได้ กรุณาเลือก CPU')
                self.device = device
                if self.core:
                    self.core.device = self.device
            if self.core is None or self.pipeline is None:
                self.initialize_models(self.device)
            if self.core:
                self.core.reset()
            if self.pipeline:
                self.pipeline.reset(self.source_id)
            self.last_error = None
            self.camera_warning = None
            self.capture_backend = None
            self.capture_error = self.detector_error = None
            self.last_frame_at = None
            self.current_frame_id = 0
            self.current_sec = 0.0
            self.fps = 0.0
            self.person_filter_report = None

            self.is_running = True
            self.is_paused = False
            self.stage = 'opening_source'
            self.thread = threading.Thread(target=self._run_loop, args=(gen_id,), daemon=True)
            self.thread.start()
            logger.info(f"Stream started [gen {gen_id}] for source: {source}")

    def _record_frame_evidence(self, context, updates):
        # The first annotation creates the disk cache. Record events afterward
        # so even first-frame events can replay their original evidence later.
        timestamp_ms = context['timestamp_ms']
        violence = self.healthy_module('violence')
        if violence and getattr(violence, 'uses_scene_pixels', False):
            context['violence'] = violence.diagnostics(self.source_id)
        self.alert_tracks = {tid: expiry for tid, expiry in self.alert_tracks.items() if expiry >= timestamp_ms}
        self.annotations.add(context, self.healthy_module('fall'), updates, self.alert_tracks)
        self._record_updates(updates)

    def stop_stream(self):
        with self.lifecycle_lock:
            return self._stop_stream()

    def _stop_stream(self):
        with self.lock:
            if self.qa_run and self.qa_run.state == 'running':
                self.qa_run.finish(False, 'หยุดหรือเปลี่ยนแหล่งภาพก่อน QA ครบคลิป')
            self.generation_id += 1
            self.is_running = False
            self.pending_seek_frame = None
            self.stage = 'stopping'
            self.frame_playback.cancel()
            if self.browser_capture: self.browser_capture.release()
            if self.capture and self.decode_device=='cuda': self.capture.release()

        if self.thread and self.thread.is_alive():
            # A CPU forward pass cannot be interrupted safely. Wait for it to
            # finish without holding the processing lock or reusing its models.
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError('รอบก่อนยังประมวลผลเฟรมไม่เสร็จ กรุณารอแล้วเปิดคลิปอีกครั้ง')
        self.thread = None

        with self.lock:
            if self.pipeline:
                self._record_updates(self.pipeline.flush())
            self.latest_frame_jpeg = None
            self.active_tracks = []
            self.person_filter_report = None
            self.current_alerts = []
            self.stage = 'idle'
            self.annotations.close()
        logger.info("Stream stopped cleanly")

    def toggle_pause(self):
        with self.lock:
            self.is_paused = not self.is_paused
            return self.is_paused

    def seek(self, time_sec: Optional[float] = None, ratio: Optional[float] = None, delta_sec: Optional[float] = None):
        with self.lock:
            if self.is_live or self.total_frames <= 0 or self.video_fps <= 0:
                return {"status": "error", "message": "Cannot seek live stream or uninitialized video"}
            if self.qa_run and self.qa_run.state == 'running':
                return {"status": "error", "message": "QA ต้องอ่านคลิปตามลำดับ กดหยุด QA ก่อนกรอคลิป"}

            target_sec = self.current_sec
            if delta_sec is not None:
                target_sec += delta_sec
            elif ratio is not None:
                target_sec = ratio * self.duration_sec
            elif time_sec is not None:
                target_sec = time_sec

            target_sec = max(0.0, min(target_sec, self.duration_sec))
            target_frame = int(target_sec * self.video_fps)
            target_frame = max(0, min(target_frame, self.total_frames - 1))

            self.pending_seek_frame = target_frame
            self.frame_playback.discard()
            self.pending_seek_preserve_cache = False
            self.current_sec = target_sec
            self.current_frame_id = target_frame

            return {
                "status": "ok",
                "target_sec": round(target_sec, 2),
                "target_frame": target_frame,
                "duration_sec": round(self.duration_sec, 2),
            }

    def set_speed(self, speed: float):
        with self.lock:
            self.playback_speed = max(0.25, min(float(speed), 4.0))
            return {"status": "ok", "speed": self.playback_speed}

    def _run_loop(self, gen_id: int):
        cap = None
        try:
            src = int(self.source) if self.source.isdigit() else self.source
            cap = self.browser_capture if self.browser_capture else (open_capture(src) if self.decode_device=='cpu' else open_selected_capture(src,self.decode_device))
            if self.is_live and not isinstance(cap,BrowserCapture) and cap.isOpened():
                cap=RecordingCapture(cap,media_store,self.source_id,self.session_id)
            self.capture = cap
            self._run_capture(gen_id, cap)
        except Exception as error:
            logger.exception('Video worker failed')
            with self.lock:
                if self.generation_id == gen_id:
                    self.is_running = False
                    self.capture_error = self.last_error = 'ประมวลผลภาพหยุด: '+str(error)
                    self.stage = 'error'
                    if self.qa_run:
                        self.qa_run.finish(False, self.last_error)
        finally:
            if cap is not None:
                cap.release()
            if self.capture is cap: self.capture = None

    def _run_capture(self, gen_id, cap):
        src = int(self.source) if self.source.isdigit() else self.source
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.source}")
            with self.lock:
                if self.generation_id == gen_id:
                    self.last_error = (f'เปิดเว็บแคมหมายเลข {src} ไม่ได้ ลองเลือกกล้องหมายเลขอื่น '
                        'ปิดแอปที่ใช้กล้อง และตรวจสิทธิ์กล้องของ Windows'
                        if isinstance(src, int) else 'Cannot open video source')
                    self.is_running = False
                    self.capture_error = self.last_error
                    if self.qa_run:
                        self.qa_run.finish(False, self.last_error)
            return

        try:
            self.capture_backend = cap.getBackendName()
        except cv2.error:
            self.capture_backend = None

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = native_fps if math.isfinite(native_fps) and native_fps > 0 else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 and fps > 0 else 0.0
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with self.lock:
            self.video_fps = fps
            self.total_frames = max(0, total_frames)
            self.duration_sec = duration
            self.video_width = max(1, vw) if vw > 0 else 1920
            self.video_height = max(1, vh) if vh > 0 else 1080

            # Sync LocationModule pixel zones if relative points were saved
            loc_zones = self.config.get("features", {}).get("location", {}).get("config", {}).get("zones", {})
            loc_updated = False
            for zval in loc_zones.values():
                if isinstance(zval, dict) and "points_relative" in zval:
                    zval["points"] = [[round(p[0] * self.video_width), round(p[1] * self.video_height)] for p in zval["points_relative"]]
                    loc_updated = True
            if loc_updated and self.pipeline:
                self.update_config(self.config, preserve_qa=True)

        frame_count = 0
        live_start_time = time.monotonic()
        fps_timer = time.monotonic()
        fps_counter = 0
        failed_reads = 0
        dark_since = None

        logger.info(f"Loop started: total_frames={total_frames}, fps={fps:.1f}, duration={duration:.1f}s")

        while self.is_running and self.generation_id == gen_id:
            # Handle seek request
            seek_target = None
            preserve_annotations = False
            with self.lock:
                if self.pending_seek_frame is not None:
                    seek_target = self.pending_seek_frame
                    preserve_annotations = self.pending_seek_preserve_cache
                    self.pending_seek_preserve_cache = False
                    self.pending_seek_frame = None

            if seek_target is not None:
                fps_counter=0;fps_timer=time.monotonic()
                if not preserve_annotations:self.annotations.clear()
                self.alert_tracks.clear()
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                frame_count = seek_target
                with self.lock:
                    if self.pipeline:
                        self._record_updates(self.pipeline.flush())
                        self.pipeline.reset(self.source_id)
                    if self.core:
                        self.core.reset()

            if self.is_paused and seek_target is None:
                fps_counter=0;fps_timer=time.monotonic()
                time.sleep(0.05)
                continue

            loop_start = time.monotonic()
            frame_epoch = self.frame_playback.epoch
            ret, frame = cap.read()
            read_done = time.monotonic()
            if not valid_frame(ret, frame):
                if self.is_live:
                    failed_reads += 1
                    if failed_reads < 10:
                        time.sleep(.1)
                        continue
                    with self.lock:
                        if self.generation_id == gen_id:
                            self.last_error = 'กล้องหยุดส่งภาพ ลองเปิดใหม่หรือเลือกกล้องหมายเลขอื่น และปิดแอปที่ใช้กล้องอยู่'
                            self.capture_error = self.last_error
                    break
                if self.loop_video and not self.is_live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    fps_counter=0;fps_timer=time.monotonic()
                    with self.lock:
                        if self.pipeline:
                            self._record_updates(self.pipeline.flush())
                            self.pipeline.reset(self.source_id)
                        if self.core:
                            self.core.reset()
                        self.annotations.clear()
                        self.alert_tracks.clear()
                        self.current_sec,self.current_frame_id=0,0
                    continue
                else:
                    with self.lock:
                        if self.pipeline:
                            final_updates = self.pipeline.flush()
                            self._record_updates(final_updates)
                        else:
                            final_updates = []
                        if self.qa_run:
                            if getattr(self.pipeline, 'health', {}).get(self.qa_run.feature, {}).get('state') == 'error':
                                self.qa_run.finish(False, 'ฟีเจอร์ผิดพลาดขณะปิดเหตุท้ายคลิป')
                            self.qa_run.finish(True, final_updates=final_updates)
                        self.is_paused = True
                        self.stage = 'analysis_complete'
                    time.sleep(0.1)
                    continue

            failed_reads = 0
            if isinstance(src, int) or isinstance(cap, BrowserCapture):
                if dark_frame(frame):
                    dark_since = dark_since or time.monotonic()
                else:
                    dark_since = None
                with self.lock:
                    self.camera_warning = ('ภาพจากกล้องมืดมาก ตรวจฝาปิดเลนส์หรือกล้องเสมือน และลองเลือกกล้องหมายเลขอื่น'
                        if dark_since is not None and time.monotonic()-dark_since >= 2 else None)
            frame_count += 1
            fps_counter += 1

            if self.is_live:
                ts_ms = cap.timestamp_ms if isinstance(cap,(BrowserCapture,RecordingCapture)) else (time.monotonic() - live_start_time) * 1000.0
            else:
                ts_ms = frame_count * 1000.0 / fps

            if time.monotonic() - fps_timer >= 1.0:
                self.fps = fps_counter / (time.monotonic() - fps_timer)
                fps_counter = 0
                fps_timer = time.monotonic()

            # Process AI frame
            try:
                context = self.core.process(
                    frame,
                    source_id=self.source_id,
                    frame_id=frame_count,
                    timestamp_ms=ts_ms,
                    fps=fps,
                )
                pose_done = time.monotonic()
                with self.lock:
                    if self.generation_id != gen_id:
                        break
                    context['device'] = self.device
                    context['pose_model'] = self.weights_path.name
                    updates = self.pipeline.process(frame, context) if self.pipeline else []
                    self.last_error = None
                    self.detector_error = None
                    self._record_frame_evidence(context, updates)
                    self.analysis_frames+=1
                    self.empty_person_frames+=not bool(context.get('persons'))
                    fall=self.healthy_module('fall')
                    if fall:
                        for person in context.get('persons',[]):
                            reason=fall.status(self.source_id,person['track_id'])['reason']
                            self.fall_diagnostics[reason]=self.fall_diagnostics.get(reason,0)+1
            except Exception as e:
                logger.error(f"Error in frame processing: {e}")
                if self.generation_id != gen_id:
                    break
                self.last_error = str(e)
                self.detector_error = str(e)
                updates = []
                context = {"persons": []}
                pose_done = time.monotonic()

            # Handle new/updated events
            features_done = time.monotonic()
            # Draw AI Visualizations
            with self.lock:
                if self.generation_id != gen_id:
                    break
                if self.qa_run and self.generation_id == gen_id:
                    feature_health = getattr(self.pipeline, 'health', {}).get(self.qa_run.feature, {})
                    self.qa_run.observe(frame_count, ts_ms, context, updates, self.detector_error,
                                        feature_health.get('state') == 'error')
                annotated_frame = self._render_overlay(frame.copy(), context, updates)
            render_done = time.monotonic()

            # Encode frame to JPEG
            _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            encode_done = time.monotonic()
            with self.lock:
                if self.generation_id != gen_id:
                    break
                self.processing_ms = dict(read=round((read_done-loop_start)*1000,2),
                    pose=round((pose_done-read_done)*1000,2),features=round((features_done-pose_done)*1000,2),
                    render=round((render_done-features_done)*1000,2),jpeg=round((encode_done-render_done)*1000,2),
                    total=round((encode_done-loop_start)*1000,2),
                    actual_device=str(getattr(getattr(self.yolo_model,'predictor',None),'device',self.device)))
                self.latest_frame_jpeg = buffer.tobytes()
                self.last_frame_at = time.monotonic()
                self.stage = 'running'
                self.current_frame_id = frame_count
                self.current_sec = ts_ms/1000
                self.active_tracks = [
                    {
                        "track_id": p["track_id"],
                        "confidence": round(p["confidence"], 2),
                        "bbox": [round(v, 1) for v in p["bbox_xyxy"]],
                        "observation_kind": p.get('observation_kind', 'body_pose'),
                    }
                    for p in context.get("persons", [])
                ]
                self.person_filter_report = deepcopy(context.get('person_filter'))

            if self.frame_by_frame and not self.is_live:
                if not self.frame_playback.publish(self.session_id, self.latest_frame_jpeg, frame_count, ts_ms/1000, frame_epoch):
                    break

            # Playback speed pacing
            elapsed = time.monotonic() - loop_start
            target_delay = (1.0 / max(1.0, fps)) / max(0.25, self.playback_speed)
            sleep_time = max(0.001, target_delay - elapsed)
            if not self.review_mode or self.is_live or (self.qa_run and self.qa_run.state=='running'):
                time.sleep(sleep_time)

        with self.lock:
            if self.pipeline and self.generation_id == gen_id:
                self._record_updates(self.pipeline.flush())
                self.is_running = False
        logger.info(f"Loop finished [gen {gen_id}]")

    def _render_overlay(self, frame: np.ndarray, context: dict, recent_updates: list) -> np.ndarray:
        h, w = frame.shape[:2]

        # 1. Draw Configured Zones
        # A. Wandering relative zones
        wandering_zones = self.config.get("features", {}).get("wandering", {}).get("zones_relative", {})
        for name, poly in wandering_zones.items():
            pts = np.array([[int(px * w), int(py * h)] for px, py in poly], np.int32)
            if len(pts) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 180))
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                draw_person_label(frame, f"ZONE: {name}", cx - 40, cy, (0, 0, 180))

        # B. Location zones (Bed, Room, etc.)
        loc_zones = self.config.get("features", {}).get("location", {}).get("config", {}).get("zones", {})
        for name, val in loc_zones.items():
            pts_list = val.get("points_relative") if isinstance(val, dict) and "points_relative" in val else (val.get("points", val) if isinstance(val, dict) else val)
            if isinstance(pts_list, list) and len(pts_list) >= 3:
                is_normalized = all(
                    isinstance(p, (list, tuple)) and len(p) == 2 and 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0
                    for p in pts_list
                )
                if is_normalized:
                    pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in pts_list], np.int32)
                else:
                    pts = np.array(pts_list, np.int32)

                if len(pts) >= 3:
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (180, 100, 0))
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.polylines(frame, [pts], True, (255, 200, 0), 2)
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    draw_person_label(frame, f"BED: {name}", cx - 30, cy, (180, 100, 0))

        scene = context.get('violence', {}).get('result')
        if scene:
            text = f"LSTM {'ภาพเสี่ยง • ตรวจสอบ' if scene['fighting'] else 'คะแนนทำร้าย'} {scene['probability_fighting']*100:.1f}%"
            draw_person_label(frame, text, 10, 36, (0, 100, 200) if scene['fighting'] else (0, 120, 0))

        # 2. Draw People, Skeletons, and Alerts
        active_alert_labels = []
        location = self.healthy_module('location')
        people = {p['track_id']: p for p in location.get_snapshot(self.source_id)} if location else {}
        for person in context.get("persons", []):
            tid = person["track_id"]
            bbox = [int(v) for v in person["bbox_xyxy"]]
            x1, y1, x2, y2 = bbox

            status_text = "NORMAL"
            color = (0, 220, 0)  # Green

            # A fall event is emitted once. Render its retained track status on
            # following frames, rather than blinking red for one frame only.
            fall_module = self.healthy_module('fall')
            if self.config.get('features', {}).get('fall', {}).get('enabled') and fall_module is None:
                status_text, color = 'ตรวจล้มไม่พร้อม', (0, 165, 255)
            if fall_module:
                fall_st = fall_module.status(self.source_id, tid)
                fall_text, fall_level = fall_display(fall_st)
                if fall_level == 'alert':
                    status_text, color = fall_text, (0, 0, 255)
                    active_alert_labels.append(f'คน #{tid}: อาจล้ม')
                elif color != (0, 0, 255):
                    if fall_level in ('observing', 'warning'):
                        status_text, color = fall_text, (0, 165, 255)
                    elif status_text == 'NORMAL':
                        status_text = fall_text

            # Fall status from recent_updates
            for ev in recent_updates:
                if tid in ev.get("track_ids", []):
                    if ev.get("event_type") == "fall_detected":
                        status_text = 'อาจล้ม • กรุณาตรวจสอบ'
                        color = (0, 0, 255)
                        active_alert_labels.append(f'คน #{tid}: อาจล้ม')
                    elif ev.get("event_type") == "fall_abnormal_movement":
                        status_text = 'พบการเคลื่อนไหวผิดปกติ'
                        color = (0, 165, 255)

            # Draw COCO Skeleton
            if tid in self.alert_tracks:
                color=(0,0,255)
                status_text='มีเหตุแจ้งเตือน • กรุณาตรวจภาพ'
            elif person.get('observation_kind') == 'bbox_only':
                status_text='พบกรอบคน • ข้อต่อไม่พร้อม'
            pose = clean_pose(person.get("pose"), person['bbox_xyxy'], frame.shape)
            if pose is not None and len(pose) >= 17:
                for idx, (p1, p2) in enumerate(COCO_SKELETON):
                    if p1 < len(pose) and p2 < len(pose):
                        pt1, pt2 = pose[p1], pose[p2]
                        if pt1[2] >= DRAW_JOINT_CONFIDENCE and pt2[2] >= DRAW_JOINT_CONFIDENCE:
                            c = SKELETON_COLORS[idx % len(SKELETON_COLORS)]
                            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), c, 2)
                for pt in pose:
                    if pt[2] >= DRAW_JOINT_CONFIDENCE:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)

            # Draw Bounding Box & Label
            if status_text=='NORMAL':status_text='กำลังติดตาม'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid} [{status_text}]"
            identity = people.get(tid, {})
            if identity.get('resident_id'):
                label = f"{identity['identity_name']} / ID:{tid} [{status_text}]"
            if location:
                state = location.state.get(self.source_id, tid)
                observation = state.last_face if state else None
                if observation and observation.face_detected and observation.face_bbox:
                    fx1, fy1, fx2, fy2 = map(int, observation.face_bbox)
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 2)
            draw_person_label(frame, label, x1, y1, color)

        # 3. Top HUD Banner
        hud = frame.copy()
        cv2.rectangle(hud, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.addWeighted(hud, 0.8, frame, 0.2, 0, frame)

        time_str = time.strftime("%H:%M:%S")
        curr_m = int(self.current_sec // 60)
        curr_s = int(self.current_sec % 60)
        dur_m = int(self.duration_sec // 60)
        dur_s = int(self.duration_sec % 60)
        pos_str = f"{curr_m:02d}:{curr_s:02d}/{dur_m:02d}:{dur_s:02d}" if not self.is_live else "LIVE"

        hud_text = f"NEXORA AI | SRC: {self.source_id} | FPS: {self.fps:.1f} | TRACKS: {len(context.get('persons', []))} | POS: {pos_str} | {time_str}"
        cv2.putText(frame, hud_text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

        # Flashing Alert Banner
        if active_alert_labels:
            alert_banner = " | ".join(set(active_alert_labels))
            banner_y = 70
            cv2.rectangle(frame, (0, 36), (w, banner_y), (0, 0, 220), -1)
            draw_person_label(frame,alert_banner+' • กรุณาตรวจสอบ',15,68,(0,0,220))

        return frame

    def _recalc_review_stats(self):
        current_events = [e for e in self.events if not is_archived_event(e)]
        total = len(current_events)
        confirmed = sum(1 for e in current_events if e.get("review_status") == "confirmed")
        false_alarm = sum(1 for e in current_events if e.get("review_status") == "false_alarm")
        needs_review = sum(1 for e in current_events if e.get("review_status") == "needs_review")
        self.review_stats = {
            "total": total,
            "confirmed": confirmed,
            "false_alarm": false_alarm,
            "needs_review": needs_review,
            "false_alarm_rate": round(false_alarm / max(1, (confirmed + false_alarm)) * 100, 1),
        }


# Global worker instance
worker = StreamWorker()
streams = StreamRegistry(StreamWorker,worker)

# FastAPI Application
app = FastAPI(title="NEXORA CCTV CORE AI", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    retention_stop.clear()
    threading.Thread(target=_retention_loop,daemon=True).start()
    try:
        worker.initialize_models(device=worker.device)
    except Exception as e:
        worker.last_error = str(e)
        worker.detector_error = str(e)
        logger.warning(f"Error during startup model initialization: {e}")


retention_stop=threading.Event()


def _retention_loop():
    while not retention_stop.is_set():
        try: media_store.cleanup()
        except OSError as error: logger.warning('Recording retention: %s',error)
        retention_stop.wait(300)


@app.on_event('shutdown')
def shutdown_event():
    retention_stop.set()
    for cid,entry in list(streams.entries.items()):
        entry['thread'].join(timeout=1)
        if entry['state']!='starting':
            try: streams.stop(cid)
            except RuntimeError: logger.warning('Source %s still stopping',cid)
    try: worker.stop_stream()
    except RuntimeError: logger.warning('Main source still stopping')


@app.get("/api/status")
def get_status():
    with worker.lock:
        return _status_snapshot()


@app.get('/api/launcher-info')
def get_launcher_info():
    return launcher_info(ROOT, PRODUCT_REVISION)


@app.get('/api/runtime/options')
def runtime_options():
    return dict(gpu_available=_preferred_device()=='cuda',ffmpeg_available=bool(ffmpeg_binary()),
                models=model_options(ROOT, worker.weights_path), current_model=worker.weights_path.name,
                current_device=worker.device,
                max_streams=streams.maximum,retention_hours=media_store.retention_hours,
                disk_limit_gb=media_store.max_bytes/1024**3)


def media_path(source):
    path=Path(source).resolve()
    allowed=(SETTINGS_DIR/'uploads',SETTINGS_DIR/'recordings',ROOT.parent/'incoming_cctv')
    if path.suffix.lower() not in ('.mp4','.webm','.avi','.mov','.mkv') or not any(path.is_relative_to(p.resolve()) for p in allowed):
        raise HTTPException(400,'ดูไฟล์ที่อัปโหลด วิดีโอที่บันทึก หรือคลิปตัวอย่างเท่านั้น')
    if not path.is_file(): raise HTTPException(404,'ไม่พบวิดีโอ')
    return path


@app.get('/api/playback/media')
def playback_media(source: str):
    return FileResponse(media_path(source))


@app.get('/api/playback/annotations')
def playback_annotations(seconds: float, camera_id: Optional[str] = None):
    if not math.isfinite(seconds) or seconds<0: raise HTTPException(400,'เวลาคลิปไม่ถูกต้อง')
    try: target=streams.get(camera_id) if camera_id else worker
    except KeyError: raise HTTPException(404,'ไม่พบแหล่งภาพ')
    with target.lock:
        return dict(source=target.source,frames=target.annotations.near(seconds),analysis_sec=target.current_sec,
                    source_fps=target.video_fps,cache=target.annotations.info(),session_id=target.session_id,
                    evidence_revision=target.evidence_revision,stage=target.stage)


class PlaybackModeRequest(BaseModel):
    camera_id: Optional[str] = None
    source: str


class AnalyzeNearRequest(BaseModel):
    source: str
    time_s: float
    session_id: Optional[str] = None
    evidence_revision: Optional[str] = None


@app.post('/api/playback/analyze-near')
def analyze_near(req: AnalyzeNearRequest):
    with worker.lock:
        if worker.source != req.source or not worker.is_running or worker.is_live or not worker.review_mode:
            raise HTTPException(409, 'แหล่งภาพเปลี่ยนหรือไม่ใช่คลิปในตัวเล่น กรุณาเปิดคลิปใหม่')
        if req.session_id and req.session_id != worker.session_id:
            raise HTTPException(409, 'รอบวิเคราะห์เปลี่ยนแล้ว กรุณารีโหลดสถานะ')
        if req.evidence_revision and req.evidence_revision != worker.evidence_revision:
            raise HTTPException(409, 'การตั้งค่า AI เปลี่ยนแล้ว กรุณารีโหลดสถานะ')
        if worker.qa_run and worker.qa_run.state == 'running':
            raise HTTPException(409, 'QA ต้องอ่านตามลำดับ ไม่กรอการวิเคราะห์ระหว่างทดสอบ')
        if not math.isfinite(req.time_s) or not 0 <= req.time_s <= worker.duration_sec:
            raise HTTPException(400, 'เวลาคลิปอยู่นอกขอบเขต')
        # Four seconds of preceding context for temporal interaction/fall rules.
        result = worker.seek(time_sec=max(0,req.time_s-4))
        if result.get('status') == 'error':raise HTTPException(409,result['message'])
        worker.pending_seek_preserve_cache = True
        worker.is_paused = False
        return dict(status='analyzing',context_start_s=max(0,req.time_s-4))


@app.post('/api/playback/paced')
def paced_playback(req: PlaybackModeRequest):
    try: target = streams.get(req.camera_id) if req.camera_id else worker
    except KeyError: raise HTTPException(404, 'ไม่พบแหล่งภาพ')
    with target.lock:
        if target.source != req.source or not target.is_running or target.is_live:
            raise HTTPException(409, 'แหล่งภาพเปลี่ยนแล้ว กรุณาเปิดคลิปอีกครั้ง')
        if target.qa_run and target.qa_run.state == 'running':
            raise HTTPException(409, 'QA กำลังวิเคราะห์ตามลำดับ ภาพ AI ไม่ใช่ตัวเล่นคลิป กรุณาหยุด QA ก่อนดูแบบ 1x')
        result = target.seek(time_sec=0)
        if result.get('status') == 'error': raise HTTPException(409, result['message'])
        target.review_mode = False
        target.paced_fallback = True
        target.playback_speed = 1.0
        target.is_paused = False
        return dict(status='paced', speed=1.0)


class ClipLoopRequest(BaseModel):
    source: str
    enabled: bool


@app.post('/api/playback/loop')
def clip_loop(req: ClipLoopRequest):
    with worker.lock:
        if worker.qa_run and worker.qa_run.state == 'running':
            raise HTTPException(409, 'รอบ QA ต้องอ่านคลิปครั้งเดียว ปิดการวนระหว่างทดสอบ')
        if worker.source != req.source or not worker.is_running or worker.is_live:
            raise HTTPException(409, 'วนได้เฉพาะคลิปที่เปิดอยู่ กรุณาเลือกคลิปอีกครั้ง')
        if worker.review_mode:
            raise HTTPException(409, 'ตัวเล่นวิดีโอจริงวนภาพในเบราว์เซอร์ ไม่ต้องวน AI ซ้ำ')
        if req.enabled and worker.is_paused and worker.total_frames > 0 and worker.current_frame_id >= worker.total_frames:
            result = worker.seek(time_sec=0)
            if result.get('status') == 'error': raise HTTPException(409, result['message'])
            worker.is_paused = False
        worker.loop_video = req.enabled
        return dict(enabled=worker.loop_video)


@app.get('/api/recordings')
def recordings(source_id: Optional[str] = None):
    media_store.cleanup()
    return dict(retention_hours=24,recordings=media_store.list(source_id))


@app.get('/api/recordings/{recording_id}/media')
def recording_media(recording_id: str):
    try: return FileResponse(media_store.path(recording_id))
    except (ValueError,FileNotFoundError): raise HTTPException(404,'ไม่พบวิดีโอ หรือเกินระยะเก็บ 24 ชั่วโมง')


@app.get('/api/events/{event_id}/playback')
def event_playback(event_id: str):
    with worker.lock:
        event=next((e for e in worker.events if e['event_id']==event_id),None)
        if not event: raise HTTPException(404,'ไม่พบเหตุการณ์')
        recorded=media_store.for_event(event)
        if recorded:
            row=recorded['recording']
            return dict(url='/api/recordings/'+row['id']+'/media',source=str(media_store.path(row['id'])),
                        time_s=max(0,recorded['time_s']-2),camera_id=event['source_id'],recorded=True,
                        evidence_url='/api/events/'+event_id+'/annotations')
        source=event.get('metadata',{}).get('playback_source')
        if not source:
            raise HTTPException(409,'เหตุการณ์เก่านี้ไม่มีข้อมูลคลิปต้นฉบับ จึงย้อนดูไม่ได้ กรุณาเปิดคลิปเพื่อทดสอบใหม่')
        if source.isdigit() or source.startswith(('browser://','rtsp://','http://','https://')):
            raise HTTPException(409,'ช่วงวิดีโอนี้ยังบันทึกไม่เสร็จ รอจบช่วงบันทึก หรือวิดีโอเกิน 24 ชั่วโมงแล้ว')
        media_path(source)
        from urllib.parse import urlencode
        return dict(url='/api/playback/media?'+urlencode(dict(source=source)),source=source,
                    time_s=max(0,float(event.get('end_timestamp_ms',0))/1000-2),camera_id=event['source_id'],recorded=False,
                    evidence_url='/api/events/'+event_id+'/annotations')


@app.get('/api/events/{event_id}/annotations')
def event_annotations(event_id: str, seconds: float):
    if not math.isfinite(seconds) or seconds < 0:
        raise HTTPException(400, 'เวลาคลิปไม่ถูกต้อง')
    with worker.lock:
        event = deepcopy(next((e for e in worker.events if e['event_id'] == event_id), None))
    if not event: raise HTTPException(404, 'ไม่พบเหตุการณ์')
    reference = event.get('metadata', {}).get('playback_evidence')
    result = dict(event_id=event_id, frames=[], message=None)
    if not reference:
        return dict(result, message='เหตุการณ์เก่านี้ไม่ได้เก็บผล AI ของรอบเดิม จึงแสดงเฉพาะวิดีโอ')
    offset, scale = 0., 1.
    recorded = media_store.for_event(event)
    if recorded:
        row = recorded['recording']
        offset = row['start_s']
        scale = row.get('media_duration_s', row['end_s']-offset)/(row['end_s']-offset)
    source_seconds = offset+seconds/scale
    target = worker if event['source_id'] == worker.source_id else None
    if target is None:
        try: target = streams.get(event['source_id'])
        except KeyError: pass
    try:
        if target:
            with target.lock:
                same_run = (target.source == event.get('metadata', {}).get('playback_source')
                    and target.session_id == reference.get('session_id')
                    and target.evidence_revision == reference.get('revision'))
                rows = target.annotations.near(source_seconds, .6/scale) if same_run else None
        else: rows = None
        if rows is None:
            rows = AnnotationBuffer.read_cached(ROOT/'local_only/playback_cache', reference.get('cache'), source_seconds, .6/scale)
        if recorded:
            rows = [r for r in rows if offset <= r['time_s'] <= recorded['recording']['end_s']]
        for row in rows: row['time_s'] = (row['time_s']-offset)*scale
        result['frames'] = rows
    except (OSError, ValueError, sqlite3.Error, zlib.error) as error:
        result['message'] = 'ผล AI ของรอบเดิมไม่พร้อมหรือหมดอายุ แสดงเฉพาะวิดีโอ'
    return result


@app.post('/api/notifications/settings')
def notification_settings(payload: dict):
    if not isinstance(payload.get('last_seen_update'),bool): raise HTTPException(400,'เลือกเปิดหรือปิดการแจ้งตำแหน่งล่าสุด')
    worker.notify_last_seen=payload['last_seen_update']
    return dict(last_seen_update=worker.notify_last_seen)


class MultiStartRequest(BaseModel):
    source: str
    camera_id: str
    device: str = 'cpu'
    decode_device: str = 'cpu'
    model: str = 'yolo26n-pose.pt'
    browser_width: int = 960
    browser_height: int = 540


def check_server_webcam(source):
    if source.isdigit() and Path('/.dockerenv').exists() and not Path('/dev/video'+source).exists():
        raise HTTPException(400, 'กล้องหมายเลขนี้เป็นกล้องของ server/container และยังไม่ได้เชื่อมอุปกรณ์ '
                            'เลือก เปิด webcam ของเครื่องนี้ หรือ ค้นหา webcam เพื่อใช้กล้องผ่านเบราว์เซอร์')


@app.post('/api/multistream/start')
def multi_start(req: MultiStartRequest):
    with streams.lock:
        return _multi_start_request(req)


def _multi_start_request(req: MultiStartRequest):
    if req.camera_id not in worker.cameras.profiles: raise HTTPException(404,'ไม่พบห้อง / กล้อง')
    browser=None
    source=req.source
    check_server_webcam(source)
    if source.startswith('browser://'):
        if not 64<=req.browser_width<=3840 or not 64<=req.browser_height<=2160: raise HTTPException(400,'ขนาดภาพไม่ถูกต้อง')
        browser=BrowserCapture(req.browser_width,req.browser_height)
    elif not source.isdigit() and not source.startswith(('rtsp://','http://','https://')):
        if not Path(source).is_file(): raise HTTPException(404,'ไม่พบคลิป')
    if req.device not in ('cpu','cuda') or req.decode_device not in ('cpu','cuda'): raise HTTPException(400,'อุปกรณ์ไม่ถูกต้อง')
    if req.device=='cuda' and _preferred_device()!='cuda': raise HTTPException(400,'GPU ยังไม่พร้อม กรุณาเลือก CPU')
    if req.decode_device=='cuda' and (browser or source.isdigit()): raise HTTPException(400,'webcam ไม่ใช้ GPU decode เลือก GPU สำหรับ AI ได้')
    if req.decode_device=='cuda' and not ffmpeg_binary(): raise HTTPException(400,'GPU decode ต้องติดตั้ง FFmpeg ก่อน')
    try: resolve_pose_weights(ROOT, req.model, worker.weights_path)
    except ValueError as error: raise HTTPException(400, str(error))
    try:
        return streams.start(deepcopy(worker.cameras.profiles[req.camera_id]),source,
                             device=req.device,decode_device=req.decode_device,model=req.model,loop=False,
                             review_mode=browser is None and Path(source).is_file(),browser_capture=browser)
    except ValueError as error: raise HTTPException(409,str(error))


@app.get('/api/multistream')
def multi_status():
    with streams.lock: entries=list(streams.entries.items())
    return [dict(id=cid,state=e['state'],error=e['error'],**_status_snapshot(e['worker'])) for cid,e in entries]


@app.post('/api/multistream/{camera_id}/stop')
def multi_stop(camera_id: str):
    try: streams.stop(camera_id)
    except (KeyError,ValueError,RuntimeError) as error: raise HTTPException(409,str(error))
    return dict(status='stopped')


@app.post('/api/stream/frame')
@app.post('/api/multistream/{camera_id}/frame')
async def browser_frame(camera_id: Optional[str] = None, session_id: str = Form(...), timestamp_ms: float = Form(...),file: UploadFile = File(...)):
    try: target=streams.get(camera_id) if camera_id else worker
    except KeyError: raise HTTPException(404,'ไม่พบกล้อง')
    if target.session_id!=session_id or not target.browser_capture: raise HTTPException(409,'กล้องเปลี่ยนแล้ว')
    if not math.isfinite(timestamp_ms) or timestamp_ms<0: raise HTTPException(400,'เวลาภาพไม่ถูกต้อง')
    raw=await file.read(2*1024**2+1)
    if len(raw)>2*1024**2: raise HTTPException(413,'ภาพใหญ่เกินไป')
    frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if frame is None: raise HTTPException(400,'ภาพไม่ถูกต้อง')
    try: target.browser_capture.submit(frame,timestamp_ms)
    except ValueError as error: raise HTTPException(400,str(error))
    return dict(status='received')


@app.post('/api/stream/recording')
@app.post('/api/multistream/{camera_id}/recording')
async def browser_recording(camera_id: Optional[str] = None, session_id: str = Form(...), start_s: float = Form(...),end_s: float = Form(...),file: UploadFile = File(...)):
    try: target=streams.get(camera_id) if camera_id else worker
    except KeyError: raise HTTPException(404,'ไม่พบกล้อง')
    if target.session_id!=session_id: raise HTTPException(409,'กล้องเปลี่ยนแล้ว')
    if not math.isfinite(end_s) or end_s<=start_s or end_s-start_s>120: raise HTTPException(400,'ช่วงบันทึกไม่ถูกต้อง')
    try: row=media_store.begin(camera_id or target.source_id,session_id,start_s,'.webm')
    except ValueError as error: raise HTTPException(409,str(error))
    path=media_store.directory/row['filename'];size=0
    try:
        with path.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>100*1024**2: raise HTTPException(413,'ช่วงวิดีโอใหญ่เกินไป')
                out.write(chunk)
        with path.open('rb') as check: signature=check.read(4)
        if size<4 or signature!=b'\x1aE\xdf\xa3': raise HTTPException(400,'รูปแบบวิดีโอ WebM ไม่ถูกต้อง')
        row=media_store.finish(row,end_s)
    except ValueError as error:
        path.unlink(missing_ok=True);raise HTTPException(400,str(error))
    except Exception:
        path.unlink(missing_ok=True); raise
    return row


def _status_snapshot(target=None):
    worker = target or globals()['worker']
    location = worker.healthy_module('location')
    track_ids = {t['track_id'] for t in worker.active_tracks}
    return {
        "is_running": worker.is_running,
        "product_revision": PRODUCT_REVISION,
        "runtime": 'docker' if Path('/.dockerenv').exists() else 'native',
        "is_paused": worker.is_paused,
        "is_live": worker.is_live,
        "current_source": worker.source,
        "frame_ready": worker.latest_frame_jpeg is not None,
        "camera_warning": worker.camera_warning,
        "capture_backend": worker.capture_backend,
        "webcam_revision": "browser-webcam-main-v3",
        "fps": round(worker.fps, 1),
        "source_fps": worker.video_fps,
        "processing_ms": deepcopy(worker.processing_ms),
        "device": worker.device,
        "decode_device": worker.decode_device,
        "stage": worker.stage,
        "session_id": worker.session_id,
        "evidence_revision": worker.evidence_revision,
        "review_mode": worker.review_mode,
        "frame_by_frame": worker.frame_by_frame,
        "playback_client": worker.playback_client,
        "paced_fallback": worker.paced_fallback,
        "analysis_sec": round(worker.current_sec,2),
        "playback_cache": worker.annotations.info(),
        "recording": {"retention_hours":24,"active":worker.is_live and worker.is_running,
                      "error":getattr(worker.capture,'recording_error',None)},
        "active_tracks_count": len(worker.active_tracks),
        "active_tracks": worker.active_tracks,
        "person_filter": deepcopy(worker.person_filter_report),
        "interaction_diagnostics": worker.healthy_module('violence').diagnostics(worker.source_id)
            if worker.healthy_module('violence') else None,
        "people": [p for p in location.get_snapshot(worker.source_id) if p['track_id'] in track_ids] if location else [],
        "health": health_snapshot(worker),
        "camera": dict(id=worker.cameras.active_id, name=worker.cameras.profiles[worker.cameras.active_id]['name']),
        "qa": worker.qa_run.snapshot() if worker.qa_run else None,
        "face": worker.face_service.status(location),
        "enabled_features": list(worker.pipeline.modules.keys()) if worker.pipeline else [],
        "configured_features": [name for name, cfg in worker.config.get('features', {}).items() if cfg.get('enabled')],
        "feature_scope_revision": "four-features-v1",
        "release": "qa-v1",
        "experimental": True,
        "model_quality_gate_passed": False,
        "fall_backend": getattr(worker.healthy_module('fall'), 'backend', worker.config.get('features', {}).get('fall', {}).get('backend')),
        "fall_model": worker.healthy_module('fall').diagnostics(worker.source_id)
            if worker.healthy_module('fall') and getattr(worker.healthy_module('fall'), 'ml_loaded', False) else None,
        "fall_revision": getattr(worker.healthy_module('fall'), 'revision', None),
        "fall_statuses": {
            str(t["track_id"]): worker.healthy_module('fall').status(worker.source_id, t["track_id"])
            for t in worker.active_tracks
        } if worker.healthy_module('fall') else {},
        "last_error": worker.last_error,
        "weights_file": worker.weights_path.name,
        "current_sec": round(worker.current_sec, 2),
        "duration_sec": round(worker.duration_sec, 2),
        "current_frame": worker.current_frame_id,
        "total_frames": worker.total_frames,
        "video_width": worker.video_width,
        "video_height": worker.video_height,
        "playback_speed": worker.playback_speed,
        "loop": worker.loop_video,
    }


@app.get("/api/stats")
def get_stats():
    return worker.review_stats


@app.get('/api/diagnostics/export')
def export_diagnostics():
    with streams.lock:targets=[worker]+[e['worker'] for e in streams.entries.values()]
    results=[]
    for target in targets:
        with target.lock:
            results.append(dict(camera_id=target.source_id,source=target.source,
                device=target.device,decode_device=target.decode_device,stage=target.stage,
                processed_frames=target.analysis_frames,frames_without_person=target.empty_person_frames,
                fall_reason_person_frames=dict(target.fall_diagnostics),configuration=deepcopy(target.config),
                health=_status_snapshot(target)['health'],fall_revision=getattr(target.healthy_module('fall'),'revision',None)))
    return JSONResponse(dict(version=1,sources=results,independent_accuracy_validated=False),
        headers={'Content-Disposition':'attachment; filename="nexora_diagnostics.json"'})


class StartStreamRequest(BaseModel):
    source: str
    loop: bool = True
    device: Optional[str] = "cuda"
    camera_id: Optional[str] = None
    decode_device: str = 'cpu'
    review_mode: bool = False
    frame_by_frame: bool = False
    playback_client: Optional[str] = None
    model: Optional[str] = None
    browser_width: int = 960
    browser_height: int = 540


@app.post("/api/stream/start")
def start_stream(req: StartStreamRequest):
    with worker.lifecycle_lock:
        return _start_stream_request(req)


def _start_stream_request(req: StartStreamRequest):
    if worker.qa_run and worker.qa_run.state == 'running':
        raise HTTPException(409, 'หยุดรอบ QA ก่อนเปลี่ยนโมเดลหรือเริ่มคลิปใหม่')
    check_server_webcam(req.source)
    if req.frame_by_frame and (req.source.isdigit() or req.source.startswith(('browser://','rtsp://','http://','https://'))):
        raise HTTPException(400, 'ประมวลผลทีละเฟรมใช้กับไฟล์คลิปเท่านั้น')
    if req.frame_by_frame and (not req.playback_client or not re.fullmatch(r'[a-zA-Z0-9_-]{8,80}', req.playback_client)):
        raise HTTPException(400, 'ต้องระบุแท็บที่เปิดคลิป')
    browser = None
    if req.source.startswith('browser://'):
        if not 64 <= req.browser_width <= 3840 or not 64 <= req.browser_height <= 2160:
            raise HTTPException(400, 'ขนาดภาพไม่ถูกต้อง')
        browser = BrowserCapture(req.browser_width, req.browser_height)
    if req.decode_device not in ('cpu','cuda'): raise HTTPException(400,'เลือกตัวถอดรหัสเป็น CPU หรือ GPU')
    if (browser or req.source.isdigit()) and req.decode_device=='cuda': raise HTTPException(400,'webcam ใช้ระบบรับภาพกล้อง เลือก CPU decode และเลือก GPU สำหรับ AI ได้')
    if req.decode_device=='cuda' and not ffmpeg_binary(): raise HTTPException(400,'GPU decode ต้องติดตั้ง FFmpeg ก่อน')
    if streams.active(worker.source_id): raise HTTPException(409,'ห้องนี้เปิดในมุมมองหลายกล้องแล้ว')
    if req.camera_id and req.camera_id != worker.cameras.active_id:
        raise HTTPException(status_code=409, detail='เลือกโปรไฟล์กล้องนี้ก่อนเริ่มแหล่งภาพ')
    resolved = req.source
    if not req.source.isdigit() and not req.source.startswith(("rtsp://", "http://", "https://", "browser://")):
        p = Path(req.source)
        if not p.is_file():
            cand = ROOT.parent / req.source
            if cand.is_file():
                resolved = str(cand.resolve())
            else:
                raise HTTPException(status_code=404, detail=f"Video file not found: {req.source}")
    try:
        worker.start_stream(resolved, loop=req.loop, device=req.device or _preferred_device(),
                            decode_device=req.decode_device,review_mode=req.review_mode and browser is None,
                            browser_capture=browser,model=req.model,
                            frame_by_frame=req.frame_by_frame,playback_client=req.playback_client)
    except RuntimeError as e:
        if worker.stage == 'stopping':
            raise HTTPException(409, str(e)) from e
        worker.last_error = str(e)
        raise HTTPException(400, f'Cannot start stream: {e}') from e
    except Exception as e:
        worker.last_error = str(e)
        raise HTTPException(status_code=400, detail=f'Cannot start stream: {e}') from e
    return {"status": "started", "source": resolved, "session_id": worker.session_id,
            "camera_id": worker.source_id}


@app.post("/api/stream/stop")
def stop_stream(payload: Optional[dict] = None):
    with worker.lifecycle_lock:
        if payload and payload.get('session_id') and payload['session_id'] != worker.session_id:
            raise HTTPException(409, 'ตัวเล่นหลักเปลี่ยนรอบแล้ว')
        try:
            worker.stop_stream()
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
    return {"status": "stopped"}


@app.post("/api/stream/pause")
def pause_stream():
    paused = worker.toggle_pause()
    return {"status": "paused" if paused else "resumed"}


class SeekRequest(BaseModel):
    time_sec: Optional[float] = None
    ratio: Optional[float] = None
    delta_sec: Optional[float] = None


@app.post("/api/stream/seek")
def seek_stream(req: SeekRequest):
    res = worker.seek(time_sec=req.time_sec, ratio=req.ratio, delta_sec=req.delta_sec)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


class SpeedRequest(BaseModel):
    speed: float


@app.post("/api/stream/speed")
def set_speed(req: SpeedRequest):
    return worker.set_speed(req.speed)


@app.get("/api/stream/feed")
async def stream_feed(request: Request, camera_id: Optional[str] = None):
    try: worker=streams.get(camera_id) if camera_id else globals()['worker']
    except KeyError: raise HTTPException(404,'ไม่พบแหล่งภาพ')
    async def frame_generator():
        last_frame_id = -1
        consecutive_empty = 0
        while True:
            if await request.is_disconnected():
                break

            frame_data = None
            if worker.is_running and worker.latest_frame_jpeg is not None:
                if worker.current_frame_id != last_frame_id or worker.is_paused:
                    last_frame_id = worker.current_frame_id
                    frame_data = worker.latest_frame_jpeg
                    consecutive_empty = 0

            if frame_data:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
                )
            else:
                consecutive_empty += 1
                if consecutive_empty > 300 and not worker.is_running:
                    break

            await asyncio.sleep(0.015)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get('/api/playback/frame')
def playback_frame(session_id: str, client_id: str, after: int = 0):
    if not worker.frame_by_frame:
        raise HTTPException(409, 'คลิปนี้ไม่ได้ใช้การเล่นตาม AI ทีละเฟรม')
    try:
        frame = worker.frame_playback.read(session_id, client_id, after)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    if frame is None:
        return Response(status_code=204, headers={'Cache-Control':'no-store'})
    return Response(frame['jpeg'], media_type='image/jpeg', headers={
        'Cache-Control':'no-store', 'X-Frame-Sequence':str(frame['sequence']),
        'X-Frame-Id':str(frame['frame_id']), 'X-Frame-Time':str(frame['seconds'])})


class FrameAckRequest(BaseModel):
    session_id: str
    client_id: str
    sequence: int


@app.post('/api/playback/frame/ack')
def playback_frame_ack(req: FrameAckRequest):
    try:
        worker.frame_playback.acknowledge(req.session_id, req.client_id, req.sequence)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {'status':'displayed'}


@app.get("/api/events")
def get_events(limit: int = 100, category: str = 'all', source_id: Optional[str] = None, include_last_seen: bool = False):
    with worker.lock:
        return [localized_event(e,e.get('metadata',{}).get('playback_source','')) for e in worker.events if not is_archived_event(e)
                and (include_last_seen or e.get('event_type')!='last_seen_update')
                and (category=='all' or event_category(e.get('event_type',''))==category)
                and (source_id is None or e.get('source_id')==source_id)][:max(1,min(limit,500))]


class ReviewEventRequest(BaseModel):
    verdict: str
    notes: Optional[str] = None


@app.post("/api/events/{event_id}/review")
def review_event(event_id: str, req: ReviewEventRequest):
    if req.verdict not in ('confirmed', 'false_alarm', 'needs_review'):
        raise HTTPException(status_code=400, detail='Unsupported review verdict')
    with worker.lock:
        ev = next((e for e in worker.events if e["event_id"] == event_id), None)
        if not ev:
            raise HTTPException(status_code=404, detail="Event not found")
        ev["review_status"] = req.verdict
        ev["human_verdict"] = req.verdict
        ev["notes"] = req.notes
        ev["reviewed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        worker._recalc_review_stats()
        return {"status": "updated", "event": ev}


@app.get("/api/samples")
def get_sample_videos():
    # 2. UR Fall Detection videos
    ur_falls = []
    ur_adls = []
    ur_dir = ROOT.parent / "incoming_cctv/UR_Fall_Detection"
    if ur_dir.is_dir():
        for p in ur_dir.rglob("*.mp4"):
            if "fall" in p.name.lower():
                ur_falls.append({
                    "category": "UR Fall (ตรวจจับการหกล้ม)",
                    "name": p.name,
                    "path": str(p.resolve()),
                })
            else:
                ur_adls.append({
                    "category": "UR ADL (การใช้ชีวิตปกติ)",
                    "name": p.name,
                    "path": str(p.resolve()),
                })

    return ur_falls[:15] + ur_adls[:5]


@app.get('/api/uploads')
def list_uploaded_clips():
    uploads = (SETTINGS_DIR / 'uploads').resolve()
    if not uploads.is_dir():
        return []
    rows = []
    for path in uploads.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in ('.mp4', '.webm', '.avi', '.mov', '.mkv'):
            continue
        try:
            info = path.stat()
        except OSError:
            continue
        if not info.st_size:
            continue
        rows.append(dict(path=str(path.resolve()), filename=re.sub(r'^[0-9a-f]{8}_', '', path.name),
                         size=info.st_size, uploaded_at=info.st_mtime))
    return sorted(rows, key=lambda row: (row['uploaded_at'], row['path']), reverse=True)


class DeleteUploadedClipRequest(BaseModel):
    source: str


@app.delete('/api/uploads')
def delete_uploaded_clip(req: DeleteUploadedClipRequest):
    uploads = (SETTINGS_DIR / 'uploads').resolve()
    try:
        candidate = Path(req.source)
        path = candidate.resolve()
    except (OSError, ValueError) as error:
        raise HTTPException(400, 'ที่อยู่คลิปไม่ถูกต้อง') from error
    if candidate.is_symlink() or path.parent != uploads or path.suffix.lower() not in ('.mp4', '.webm', '.avi', '.mov', '.mkv'):
        raise HTTPException(400, 'ลบได้เฉพาะคลิปในรายการอัปโหลด')
    # Use the same lock order as primary start; registry/clip-test starts
    # validate their file while holding their respective locks too.
    with worker.lifecycle_lock, streams.lock, violence_clip_service.lock:
        targets = [(worker, False)] + [(entry['worker'], entry['state'] == 'starting') for entry in streams.entries.values()]
        for target, starting in targets:
            source = getattr(target, 'source', '')
            thread = getattr(target, 'thread', None)
            active = starting or target.is_running or (thread is not None and thread.is_alive())
            if active and source and not source.isdigit() and '://' not in source and Path(source).resolve() == path:
                raise HTTPException(409, 'คลิปนี้ยังใช้งานอยู่ กรุณากดหยุดตัวเล่นหลักหรือมุมมองหลายกล้องก่อนลบ')
        job = violence_clip_service.jobs.get(violence_clip_service.active, {})
        if job.get('filename') == path.name:
            raise HTTPException(409, 'คลิปนี้กำลังทดสอบ LSTM กรุณารอให้เสร็จก่อนลบ')
        if not path.is_file():
            raise HTTPException(404, 'ไม่พบคลิปนี้ อาจถูกลบแล้ว')
        try:
            path.unlink()
        except OSError as error:
            raise HTTPException(409, 'ลบไม่ได้ ไฟล์อาจยังเปิดอยู่ในโปรแกรมอื่น กรุณาปิดแล้วลองใหม่') from error
        if worker.source and '://' not in worker.source and not worker.source.isdigit() and Path(worker.source).resolve() == path:
            worker.source = ''
        for profile in worker.cameras.profiles.values():
            source = profile.get('source', '')
            if source and '://' not in source and not source.isdigit() and Path(source).resolve() == path:
                profile['source'] = ''
    return dict(status='deleted', source=str(path))


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    uploads_dir = SETTINGS_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    name=Path((file.filename or 'video.mp4').replace('\\','/')).name
    if Path(name).suffix.lower() not in ('.mp4','.webm','.avi','.mov','.mkv'):
        raise HTTPException(400,'รองรับไฟล์ MP4, WebM, AVI, MOV และ MKV')
    target_path = uploads_dir / f"{uuid4().hex[:8]}_{name}"
    pending_path = target_path.with_suffix(target_path.suffix + '.part')
    size=0
    try:
        with pending_path.open('wb') as f:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>2*1024**3: raise HTTPException(413,'ไฟล์ต้องไม่เกิน 2 GB')
                f.write(chunk)
        if not size: raise HTTPException(400,'ไฟล์วิดีโอไม่มีข้อมูล')
        pending_path.replace(target_path)
    except Exception:
        pending_path.unlink(missing_ok=True);raise
    return {"status": "uploaded", "path": str(target_path.resolve()), "filename": name}


class ViolenceClipRequest(BaseModel):
    source: str
    device: str = 'cpu'


@app.get('/api/violence-clip/options')
def violence_clip_options():
    return violence_clip_service.options()


@app.post('/api/violence-clip')
def start_violence_clip(req: ViolenceClipRequest):
    with violence_clip_service.lock:
        return _start_violence_clip_request(req)


def _start_violence_clip_request(req: ViolenceClipRequest):
    try:
        path = resolve_clip(req.source, [SETTINGS_DIR/'uploads', SETTINGS_DIR/'recordings',
            ROOT.parent/'incoming_cctv', ROOT/'videos'])
        return violence_clip_service.start(path, req.device)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error


@app.get('/api/violence-clip/{job_id}')
def get_violence_clip(job_id: str):
    try:
        return violence_clip_service.snapshot(job_id)
    except KeyError as error:
        raise HTTPException(404, str(error.args[0])) from error


@app.get('/api/violence-clip/{job_id}/report')
def export_violence_clip(job_id: str):
    job = get_violence_clip(job_id)
    if job['state'] != 'completed':
        raise HTTPException(409, 'รอให้วิเคราะห์สำเร็จก่อนดาวน์โหลดผล')
    return JSONResponse(job, headers={'Content-Disposition': f'attachment; filename="nexora-lstm-{job_id}.json"'})


@app.get("/api/config")
def get_configuration():
    return worker.config


@app.post("/api/config/zones")
def update_zones(zones_payload: dict):
    if "patient_roi" in zones_payload:
        raise HTTPException(status_code=410, detail="This feature is no longer part of the web system")
    with worker.lock:
        if zones_payload.get('camera_id', worker.source_id) != worker.source_id:
            raise HTTPException(status_code=409, detail='โปรไฟล์เปลี่ยนแล้ว กรุณาวาดโซนใหม่ในห้องที่เลือก')
        vw = worker.video_width if worker.video_width > 0 else 1920
        vh = worker.video_height if worker.video_height > 0 else 1080
        try:
            candidate = apply_zones(worker.config, zones_payload, vw, vh, worker.source_id)
            worker.update_config(candidate)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Configuration rejected: {e}') from e
    return {"status": "zones_updated", "config": worker.config}


@app.post("/api/config/features")
def toggle_features(features_payload: dict):
    with worker.lock:
        candidate = deepcopy(worker.config)
        for feat, state in features_payload.items():
            if feat in ARCHIVED_FEATURES:
                raise HTTPException(status_code=410, detail='This feature is no longer part of the web system')
            if feat not in candidate.get('features', {}) or not isinstance(state, bool):
                raise HTTPException(status_code=400, detail='Expected known feature names and boolean states')
            candidate['features'][feat]['enabled'] = state
        try:
            worker.update_config(candidate)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Configuration rejected: {e}') from e
    return {"status": "features_updated", "enabled_features": list(worker.pipeline.modules.keys()) if worker.pipeline else []}


@app.get('/api/cameras')
def get_cameras():
    with worker.lock:
        return worker.cameras.snapshot()


class CameraRequest(BaseModel):
    name: str
    source: str = ''


@app.post('/api/cameras')
def create_camera(req: CameraRequest):
    with worker.lock:
        try:
            return worker.cameras.create(req.name, req.source)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error


@app.put('/api/cameras/{camera_id}')
def edit_camera(camera_id: str, req: CameraRequest):
    if streams.active(camera_id): raise HTTPException(409,'หยุดแหล่งภาพในมุมมองหลายกล้องก่อนแก้โปรไฟล์')
    with worker.lock:
        if camera_id not in worker.cameras.profiles:
            raise HTTPException(status_code=404, detail='ไม่พบโปรไฟล์')
        try:
            name = clean_text(req.name, 'ชื่อห้อง')
            source = clean_text(req.source, 'แหล่งภาพ', 2048) if req.source else ''
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        stop_active = camera_id == worker.cameras.active_id and source != worker.cameras.profiles[camera_id]['source']
    # Avoid relabelling an old stream as the newly edited source.
    if stop_active:
        try:
            worker.stop_stream()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
    with worker.lock:
        worker.cameras.profiles[camera_id].update(name=name, source=source)
        if stop_active:
            worker.source = source
        return worker.cameras.snapshot()


def select_camera(camera_id):
    if streams.active(camera_id): raise HTTPException(409,'หยุดแหล่งภาพในมุมมองหลายกล้องก่อนเลือกห้องนี้ในตัวเล่นหลัก')
    if camera_id not in worker.cameras.profiles:
        raise HTTPException(status_code=404, detail='ไม่พบโปรไฟล์กล้อง/ห้อง')
    worker.stop_stream()
    with worker.lock:
        try:
            profile = worker.cameras.profiles[camera_id]
            worker.update_config(profile['config'], camera_id=camera_id)
            worker.source = profile['source']
            worker.is_paused = False
            if worker.core:
                worker.core.reset()
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    return get_cameras()


@app.post('/api/cameras/{camera_id}/select')
def select_camera_endpoint(camera_id: str):
    return select_camera(camera_id)


@app.delete('/api/cameras/{camera_id}')
def delete_camera(camera_id: str):
    if streams.active(camera_id): raise HTTPException(409,'หยุดแหล่งภาพก่อนลบโปรไฟล์')
    with worker.lock:
        if camera_id == worker.cameras.active_id:
            raise HTTPException(status_code=409, detail='เลือกห้องอื่นก่อนลบโปรไฟล์นี้')
        if camera_id not in worker.cameras.profiles:
            raise HTTPException(status_code=404, detail='ไม่พบโปรไฟล์')
        del worker.cameras.profiles[camera_id]
        return worker.cameras.snapshot()


@app.get('/api/cameras/export')
def export_cameras():
    with worker.lock:
        return JSONResponse(dict(version=1, **worker.cameras.snapshot()),
                            headers={'Content-Disposition': 'attachment; filename="camera_profiles.json"'})


@app.post('/api/cameras/import')
def import_cameras(payload: dict):
    if streams.active(): raise HTTPException(409,'หยุดแหล่งภาพทั้งหมดในมุมมองหลายกล้องก่อนนำเข้าโปรไฟล์')
    with worker.lock:
        try:
            profiles, active = worker.cameras.imported(payload)
            # Validate every runtime config before replacing any existing profile.
            for cid, profile in profiles.items():
                ProgressPipeline.from_config(worker._product_config(profile['config'], cid))
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    worker.stop_stream()
    with worker.lock:
        old_profiles, old_active = worker.cameras.profiles, worker.cameras.active_id
        try:
            worker.cameras.profiles = profiles
            worker.update_config(profiles[active]['config'], camera_id=active)
            worker.source = profiles[active]['source']
        except Exception as error:
            worker.cameras.profiles, worker.cameras.active_id = old_profiles, old_active
            raise HTTPException(status_code=400, detail=str(error)) from error
    return get_cameras()


class QaRequest(BaseModel):
    source: str
    event_type: str = 'fall_detected'
    zone: str = ''
    labels: List[dict] = []
    negative_confirmed: bool = False
    tolerance_sec: float = 5
    camera_id: Optional[str] = None


@app.post('/api/qa/start')
def start_qa(req: QaRequest):
    with worker.lifecycle_lock:
        return _start_qa_request(req)


def _start_qa_request(req: QaRequest):
    with worker.lock:
        try:
            if req.camera_id and req.camera_id != worker.cameras.active_id:
                raise ValueError('เลือกโปรไฟล์กล้องก่อนเริ่ม QA')
            path = Path(req.source).resolve()
            if req.source.isdigit() or not path.is_file():
                raise ValueError('QA ใช้ไฟล์คลิปในเครื่อง ไม่ใช้กล้องสดหรือ RTSP')
            feature = EVENT_FEATURES.get(req.event_type)
            if feature is None or not worker.config['features'].get(feature, {}).get('enabled'):
                raise ValueError('เปิดฟีเจอร์ที่ต้องการทดสอบก่อน')
            zones = (worker.config['features']['wandering'].get('zones_relative', {}) if feature == 'wandering'
                     else worker.config['features']['location'].get('config', {}).get('zones', {}) if feature == 'location' else {})
            if feature in ('location', 'wandering') and req.zone not in zones:
                raise ValueError('เลือกโซนที่ตั้งไว้ในโปรไฟล์นี้ก่อนทดสอบ')
            cap = open_capture(str(path))
            try:
                if not cap.isOpened():
                    raise ValueError('เปิดคลิป QA ไม่ได้')
                fps, count = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError('คลิปไม่มี FPS ที่ชัดเจน')
            finally:
                cap.release()
            files = ['integration/core.py', 'integration/person_filter.py', 'integration/fall_adapter.py',
                     'integration/pipeline.py', 'app/features/violence/module.py', 'web/qa_service.py',
                     'Location/location/module.py', 'Location/location/zone_engine.py',
                     'web/server.py']
            files += [str(path.relative_to(ROOT).as_posix())
                      for directory in ('Location/location', 'Wandering/ai_camera_system')
                      for path in (ROOT/directory).rglob('*.py')
                      if not any(part in ('tests', 'eval', '__pycache__') for part in path.relative_to(ROOT).parts)]
            files = sorted(set(files))
            stat = path.stat()
            provenance = dict(weights_sha256=hashlib.sha256(worker.weights_path.read_bytes()).hexdigest(),
                              weights_file=worker.weights_path.name, device=worker.device,
                              code_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
                              config=deepcopy(worker.cameras.profiles[worker.cameras.active_id]['config']),
                              video=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, fps=fps, frames=count),
                              camera_name=worker.cameras.profiles[worker.cameras.active_id]['name'])
            run = QaRun(source=str(path), source_id=worker.source_id, event_type=req.event_type, zone=req.zone,
                        labels=req.labels, negative_confirmed=req.negative_confirmed, duration=count/fps,
                        expected_frames=count, fps=fps, tolerance=req.tolerance_sec, provenance=provenance)
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    # Joining the previous stream must occur outside its processing lock.
    try:
        worker.start_stream(str(path), loop=False, qa_run=run)
    except Exception as error:
        run.finish(False, 'เริ่ม QA ไม่สำเร็จ')
        raise HTTPException(status_code=400, detail='เริ่ม QA ไม่สำเร็จ: '+str(error)) from error
    return run.snapshot()


@app.get('/api/qa')
def get_qa():
    with worker.lock:
        return worker.qa_run.report() if worker.qa_run else None


class TelegramSettings(BaseModel):
    enabled: bool = False
    event_types: List[str] = []
    cooldown_sec: float = 30


@app.get('/api/telegram')
def get_telegram():
    with worker.lock:
        return worker.telegram.snapshot()


@app.post('/api/telegram/settings')
def configure_telegram(req: TelegramSettings):
    with worker.lock:
        try:
            return worker.telegram.settings(req.enabled, req.event_types, req.cooldown_sec)
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error


@app.post('/api/telegram/preview')
def preview_telegram():
    with worker.lock:
        return dict(mode='draft_only', sent=False,
                    text=worker.telegram.preview(worker.cameras.profiles[worker.cameras.active_id]['name']))


@app.get('/api/qa/report')
def export_qa():
    with worker.lock:
        if not worker.qa_run or worker.qa_run.state == 'running':
            raise HTTPException(status_code=409, detail='ยังไม่มีผล QA ที่จบรอบ')
        return JSONResponse(worker.qa_run.report(), headers={
            'Content-Disposition': f'attachment; filename="qa_{worker.qa_run.id[:8]}.json"'})


def change_face_data(mutate):
    """Commit local enrollment/settings and runtime together, or restore the old file."""
    with worker.lock:
        service = worker.face_service
        original = service.snapshot()
        db, profiles, mode = service.read()
        try:
            new_mode = mutate(db, profiles, mode)
            service.save(db, profiles, new_mode)
            worker.update_config(worker.config)
        except Exception:
            service.restore(original)
            raise
        location = worker.pipeline.modules.get('location') if worker.pipeline else None
        return service.status(location)


@app.get('/api/face')
def get_face_status():
    with worker.lock:
        location = worker.pipeline.modules.get('location') if worker.pipeline else None
        return worker.face_service.status(location)


class FaceModeRequest(BaseModel):
    mode: str


@app.post('/api/face/settings')
def set_face_mode(req: FaceModeRequest):
    with worker.lock:
        try:
            worker.face_service.validate_mode(req.mode)
            if req.mode != 'off' and not worker.config.get('features', {}).get('location', {}).get('enabled'):
                raise ValueError('เปิดฟีเจอร์ Location ก่อนใช้ใบหน้า')
            return change_face_data(lambda db, profiles, mode: req.mode)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error


@app.post('/api/face/enroll')
async def enroll_face(name: str = Form(...), role: str = Form(...),
        assigned_bed: str = Form(''), consent: bool = Form(False),
        images: List[UploadFile] = File(...)):
    try:
        if not consent:
            raise ValueError('โปรดยืนยันว่าเจ้าของภาพยินยอมก่อนลงทะเบียน')
        if not 3 <= len(images) <= 12:
            raise ValueError('เลือกภาพ 3–12 รูป แต่ละรูปมีใบหน้าเดียว')
        raw_images = []
        for image in images:
            raw = await image.read(FaceService.MAX_IMAGE_BYTES + 1)
            if len(raw) > FaceService.MAX_IMAGE_BYTES:
                raise ValueError('ภาพแต่ละรูปต้องไม่เกิน 8 MB')
            raw_images.append(raw)
        identity, profile, embeddings, rejected = await run_in_threadpool(
            worker.face_service.prepare_enrollment, name, role, assigned_bed, consent, raw_images)
        def add(db, profiles, mode):
            db.add_identity(identity, embeddings, profile['name'])
            profiles[identity] = profile
            return mode
        status = change_face_data(add)
        return dict(identity_id=identity, accepted=len(embeddings), rejected=rejected, face=status)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        for image in images:
            await image.close()


@app.delete('/api/face/people/{identity_id}')
def remove_face(identity_id: str):
    def remove(db, profiles, mode):
        if identity_id not in db.identities():
            raise HTTPException(status_code=404, detail='ไม่พบคนที่ลงทะเบียน')
        db.remove_identity(identity_id)
        profiles.pop(identity_id, None)
        return 'detect' if mode == 'recognize' and not len(db) else mode
    return change_face_data(remove)


# Mount static assets
static_path = ROOT / "web/static"
if static_path.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = ROOT / "web/static/index.html"
    if index_file.is_file():
        content=index_file.read_text(encoding='utf-8')
        if (static_path/'vendor/tailwind.css').is_file():
            content=re.sub(r'<script src="https://cdn.tailwindcss.com"></script>\s*<script>.*?</script>', '<link rel="stylesheet" href="/static/vendor/tailwind.css">',content,count=1,flags=re.S)
        if (static_path/'vendor/fontawesome/css/all.min.css').is_file():
            content=content.replace('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css','/static/vendor/fontawesome/css/all.min.css')
        return HTMLResponse(content=content)
    return HTMLResponse("<h3>NEXORA Web Application - UI not found</h3>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=False)
