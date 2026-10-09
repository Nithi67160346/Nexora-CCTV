"""
Optional Identity Layer — Face

Person Track → Face Detection → Quality Check → (Alignment) → Embedding
             → Database Search → Candidate → Temporal Confirmation → Identity

แยกจาก Location Engine โดยสิ้นเชิง: ส่งออกแค่ Identity ให้ IdentityResolver
ถ้าไม่มีโมเดล/ปิดใน config → build_face_pipeline() คืน None แล้ว Location ทำงานต่อได้ปกติ
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from ..models import Identity, unknown_identity
from .face_database import FaceDatabase, InMemoryFaceDatabase, NpzFaceDatabase
from .face_detector import FaceDetection, FaceDetector, create_face_detector, crop_face
from .face_embedder import FaceEmbedder, create_face_embedder
from .face_quality import FaceQualityChecker
from .identity_matcher import IdentityMatcher, IdentityStateManager, MatchResult

log = logging.getLogger(__name__)

__all__ = [
    "FaceDetection", "FaceDetector", "FaceEmbedder", "FaceDatabase", "InMemoryFaceDatabase",
    "NpzFaceDatabase", "FaceQualityChecker", "IdentityMatcher", "IdentityStateManager",
    "MatchResult", "FaceIdentityPipeline", "FaceObservation", "build_face_pipeline",
]


@dataclass
class FaceObservation:
    face_detected: bool = False
    face_bbox: tuple | None = None
    face_conf: float = 0.0
    quality_ok: bool = False
    quality_score: float = 0.0
    match: MatchResult | None = None
    identity: Identity | None = None      # ส่งให้ IdentityResolver
    newly_confirmed: bool = False

    def to_dict(self):
        return {
            "face_detected": self.face_detected,
            "face_bbox": [round(v, 1) for v in self.face_bbox] if self.face_bbox else None,
            "face_conf": round(self.face_conf, 4),
            "face_quality": round(self.quality_score, 4),
            "match_similarity": round(self.match.similarity, 4) if self.match else None,
        }


class FaceIdentityPipeline:
    def __init__(self, detector: FaceDetector, quality: FaceQualityChecker,
                 embedder: FaceEmbedder | None = None, matcher: IdentityMatcher | None = None,
                 state: IdentityStateManager | None = None, every_n_frames: int = 3):
        self.detector = detector
        self.quality = quality
        self.embedder = embedder
        self.matcher = matcher
        self.state = state
        self.every_n = max(1, int(every_n_frames))
        self._counter: dict[tuple, int] = {}

    @property
    def recognition_enabled(self):
        return self.embedder is not None and self.matcher is not None and self.state is not None

    def process(self, frame, person_bbox, track_key) -> FaceObservation | None:
        """None = ข้ามเฟรมนี้ (ประหยัดเวลา) ; ตัวตนที่ยืนยันแล้วยังคงอยู่ใน IdentityStateManager"""
        c = self._counter.get(track_key, 0)
        self._counter[track_key] = c + 1
        if c % self.every_n:
            return None

        obs = FaceObservation()
        face = self.detector.best(self.detector.detect_in_person(frame, person_bbox))
        if face is None:
            return obs
        obs.face_detected, obs.face_bbox, obs.face_conf = True, face.bbox, face.conf

        q = self.quality.check(crop_face(frame, face.bbox), face.conf)
        obs.quality_ok, obs.quality_score = q.ok, q.score
        if not q.ok or not self.recognition_enabled:
            return obs                           # หน้าคุณภาพต่ำ → ไม่จับคู่

        emb = self.embedder.embed(frame, face)
        if emb is None:
            return obs
        obs.match = self.matcher.match(emb)
        confirmed, new = self.state.observe(track_key, obs.match)
        obs.newly_confirmed = new
        obs.identity = confirmed if confirmed is not None \
            else unknown_identity("face_recognition", obs.match.similarity)
        return obs

    def transfer(self, old_key, new_key):
        if self.state:
            self.state.transfer(old_key, new_key)

    def forget(self, track_key):
        self._counter.pop(track_key, None)
        if self.state:
            self.state.forget(track_key)

    def reset(self, source_id=None):
        if source_id is None:
            self._counter.clear()
        else:
            self._counter = {k: v for k, v in self._counter.items() if k[0] != source_id}
        if self.state:
            self.state.reset(source_id)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(p, must_exist=True):
    """path สัมพัทธ์: ลอง cwd → รากโปรเจกต์ → local_only/models/"""
    if not p or os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    alt = os.path.join(PROJECT_ROOT, p)
    if os.path.exists(alt) or not must_exist:
        return alt
    # โมเดลที่ไม่ขึ้น git เก็บไว้ที่ local_only/models/
    for base in (PROJECT_ROOT, os.path.dirname(PROJECT_ROOT)):
        alt2 = os.path.join(base, "local_only", "models", os.path.basename(p))
        if os.path.exists(alt2):
            return alt2
    return p


def build_face_pipeline(cfg: dict, residents: dict | None = None):
    """
    สร้าง pipeline ตาม config — คืน None ถ้าไม่ได้เปิด face_detection
    Face Recognition จะเปิดได้ก็ต่อเมื่อ **ทั้ง** face_recognition.enabled และ
    privacy.face_recognition_enabled เป็น true (ห้ามเปิดโดยไม่ตั้งใจ)
    """
    fd_cfg = dict(cfg.get("face_detection", {}) or {})
    fr_cfg = dict(cfg.get("face_recognition", {}) or {})
    if "model_path" in fd_cfg:
        fd_cfg["model_path"] = resolve_path(fd_cfg["model_path"])
    if "embedder_model_path" in fr_cfg:
        fr_cfg["embedder_model_path"] = resolve_path(fr_cfg["embedder_model_path"])
    fr_cfg["database_path"] = resolve_path(fr_cfg.get("database_path", "face_db.npz"), must_exist=False)
    pv_cfg = cfg.get("privacy", {}) or {}

    if not fd_cfg.get("enabled", False):
        return None

    try:
        detector = create_face_detector(fd_cfg)
    except Exception as e:                       # ไม่มีโมเดล / ไม่มี ultralytics
        log.warning("Face detection ถูกปิด: %s — Location ยังทำงานต่อด้วย Track ID", e)
        return None

    quality = FaceQualityChecker(fd_cfg.get("minimum_face_size", 40),
                                 fd_cfg.get("quality_threshold", 0.5))

    embedder = matcher = state = None
    want_fr = bool(fr_cfg.get("enabled", False)) and bool(pv_cfg.get("face_recognition_enabled", False))
    if fr_cfg.get("enabled") and not pv_cfg.get("face_recognition_enabled"):
        log.warning("face_recognition.enabled=true แต่ privacy.face_recognition_enabled=false → ปิด")
    if want_fr:
        try:
            embedder = create_face_embedder(fr_cfg)
            db = NpzFaceDatabase(fr_cfg["database_path"], autosave=False)
            if len(db) == 0:
                log.warning("Face database ว่าง — ทุกคนจะเป็น unknown")
            names = {k: (v or {}).get("display_name") for k, v in (residents or {}).items()}
            matcher = IdentityMatcher(db, fr_cfg.get("similarity_threshold", 0.70),
                                      fr_cfg.get("ambiguity_margin", 0.03))
            state = IdentityStateManager(
                fr_cfg.get("confirmation_matches", 3), fr_cfg.get("confirmation_window", 5),
                fr_cfg.get("switch_matches"),
                name_lookup=lambda rid: names.get(rid) or db.display_name(rid))
        except Exception as e:
            log.warning("Face recognition ถูกปิด: %s", e)
            embedder = matcher = state = None

    return FaceIdentityPipeline(detector, quality, embedder, matcher, state,
                                fd_cfg.get("every_n_frames", 3))
