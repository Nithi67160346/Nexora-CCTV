"""Local face settings and enrollment; raw uploads are never stored here."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import cv2
import numpy as np

from Location.location.face.face_database import NpzFaceDatabase
from Location.location.face.face_detector import create_face_detector, crop_face
from Location.location.face.face_embedder import create_face_embedder
from Location.location.face.face_quality import FaceQualityChecker


class FaceService:
    MODES = ('off', 'detect', 'recognize')
    MAX_IMAGE_BYTES = 8 * 1024 * 1024

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'local_only/face/face_db.npz'
        self.detector_path = self.root / 'models/face/face_detection_yunet_2023mar.onnx'
        self.embedder_path = self.root / 'models/face/face_recognition_sface_2021dec.onnx'

    def read(self):
        db = NpzFaceDatabase(str(self.path), autosave=False)
        profiles, mode = {}, 'off'
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                if 'profiles_json' in data:
                    profiles = json.loads(str(data['profiles_json'].item()))
                if 'mode' in data:
                    mode = str(data['mode'].item())
        if mode not in self.MODES:
            raise ValueError('Invalid local face mode')
        return db, profiles, mode

    def save(self, db, profiles, mode):
        owners, embeddings = [], []
        ids = db.identities()
        for identity in ids:
            for embedding in db._emb[identity]:
                owners.append(identity); embeddings.append(embedding)
        output = io.BytesIO()
        np.savez(output, ids=np.array(ids, dtype=str),
            names=np.array([db.display_name(identity) or '' for identity in ids], dtype=str),
            owner=np.array(owners, dtype=str),
            embeddings=np.array(embeddings, dtype=np.float32) if embeddings else np.zeros((0, 128), dtype=np.float32),
            profiles_json=np.array(json.dumps(profiles, ensure_ascii=False)), mode=np.array(mode))
        self.restore(output.getvalue())

    def restore(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        if data is None:
            self.path.unlink(missing_ok=True)
            return
        try:
            temporary.write_bytes(data)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def snapshot(self):
        return self.path.read_bytes() if self.path.exists() else None

    def status(self, location=None):
        db, profiles, mode = self.read()
        detection_active = bool(location and location.face)
        recognition_active = bool(detection_active and location.face.recognition_enabled)
        warning = None
        if mode != 'off' and not detection_active:
            warning = 'ยังไม่ได้ตรวจใบหน้า: ตรวจว่าเปิด Location และมีโมเดลใบหน้าครบ'
        elif mode == 'recognize' and not recognition_active:
            warning = 'ยังระบุชื่อไม่ได้: ตรวจโมเดลและข้อมูลลงทะเบียน'
        return dict(mode=mode, detector_ready=self.detector_path.is_file(),
            embedder_ready=self.embedder_path.is_file(), detection_active=detection_active,
            recognition_active=recognition_active, registered_count=len(db), warning=warning,
            people=[dict(identity_id=identity, display_name=db.display_name(identity),
                samples=len(db._emb[identity]), **profiles.get(identity, {})) for identity in db.identities()])

    def location_config(self, base):
        cfg = deepcopy(base)
        _, profiles, mode = self.read()
        cfg['identity'] = dict(enabled=mode != 'off')
        cfg['face_detection'] = dict(enabled=mode != 'off', backend='yunet',
            model_path=str(self.detector_path), conf=.8, minimum_face_size=40,
            quality_threshold=.5, every_n_frames=3)
        cfg['face_recognition'] = dict(enabled=mode == 'recognize', embedder='sface',
            embedder_model_path=str(self.embedder_path), database_path=str(self.path),
            similarity_threshold=.70, ambiguity_margin=.03, confirmation_matches=3, confirmation_window=5)
        cfg['privacy'] = dict(face_recognition_enabled=mode == 'recognize', store_raw_face_images=False)
        if mode != 'off':
            # A nearby new track must confirm its own face, rather than inherit a name.
            cfg.setdefault('location', {}).setdefault('reid', {})['enabled'] = False
        cfg['residents'] = {}; cfg['staff'] = {}
        for identity, profile in profiles.items():
            entry = dict(display_name=profile['name'], role=profile['role'])
            if profile.get('assigned_bed'):
                entry['assigned_bed'] = profile['assigned_bed']
            cfg['residents' if profile['role'] == 'resident' else 'staff'][identity] = entry
        return cfg

    def validate_mode(self, mode):
        if mode not in self.MODES:
            raise ValueError('โหมดใบหน้าไม่ถูกต้อง')
        if mode != 'off' and not self.detector_path.is_file():
            raise ValueError('ไม่พบโมเดลตรวจใบหน้า กรุณาติดตั้งโมเดลก่อน')
        if mode == 'recognize':
            if not self.embedder_path.is_file():
                raise ValueError('ไม่พบโมเดลจดจำใบหน้า')
            if not len(self.read()[0]):
                raise ValueError('ลงทะเบียนอย่างน้อยหนึ่งคนก่อนเปิดระบุชื่อ')

    def prepare_enrollment(self, name, role, assigned_bed, consent, images):
        name = name.strip(); assigned_bed = assigned_bed.strip()
        if not consent:
            raise ValueError('โปรดยืนยันว่าเจ้าของภาพยินยอมก่อนลงทะเบียน')
        if not 1 <= len(name) <= 60 or any(ord(c) < 32 for c in name):
            raise ValueError('กรอกชื่อ 1–60 ตัวอักษร')
        if role not in ('resident', 'caregiver', 'visitor') or len(assigned_bed) > 64:
            raise ValueError('ข้อมูลบทบาทหรือเตียงไม่ถูกต้อง')
        if not 3 <= len(images) <= 12:
            raise ValueError('เลือกภาพ 3–12 รูป แต่ละรูปมีใบหน้าเดียว')
        if not self.detector_path.is_file() or not self.embedder_path.is_file():
            raise ValueError('โมเดลใบหน้ายังไม่พร้อม')
        detector = create_face_detector(dict(backend='yunet', model_path=str(self.detector_path), conf=.8))
        embedder = create_face_embedder(dict(embedder='sface', embedder_model_path=str(self.embedder_path)))
        quality = FaceQualityChecker(40, .5)
        embeddings, rejected, seen = [], [], set()
        for i, raw in enumerate(images, 1):
            if len(raw) > self.MAX_IMAGE_BYTES:
                raise ValueError('ภาพแต่ละรูปต้องไม่เกิน 8 MB')
            from PIL import Image, UnidentifiedImageError
            try:
                with Image.open(io.BytesIO(raw)) as header:
                    if header.width * header.height > 16_000_000:
                        rejected.append(dict(image=i, reason='รูปใหญ่เกิน 16 ล้านพิกเซล')); continue
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                rejected.append(dict(image=i, reason='อ่านรูปไม่ได้')); continue
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                rejected.append(dict(image=i, reason='อ่านรูปไม่ได้')); continue
            if image.shape[0] * image.shape[1] > 16_000_000:
                rejected.append(dict(image=i, reason='รูปใหญ่เกิน 16 ล้านพิกเซล')); continue
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen:
                rejected.append(dict(image=i, reason='รูปซ้ำ')); continue
            seen.add(digest)
            h, w = image.shape[:2]
            faces = detector.detect_in_person(image, (0, 0, w, h))
            if len(faces) != 1:
                rejected.append(dict(image=i, reason=f'พบ {len(faces)} หน้า ต้องมีหนึ่งหน้า')); continue
            if not quality.check(crop_face(image, faces[0].bbox), faces[0].conf).ok:
                rejected.append(dict(image=i, reason='ใบหน้าเล็ก เบลอ หรือแสงไม่เหมาะ')); continue
            embedding = embedder.embed(image, faces[0])
            if embedding is None or not np.isfinite(embedding).all() or np.linalg.norm(embedding) <= 0:
                rejected.append(dict(image=i, reason='อ่านลักษณะใบหน้าไม่ได้')); continue
            embeddings.append(embedding)
        if len(embeddings) < 3:
            raise ValueError('ได้รูปที่ผ่านเกณฑ์เพียง ' + str(len(embeddings)) + '/3: ' +
                '; '.join(f"รูป {r['image']} {r['reason']}" for r in rejected))
        identity = 'person_' + uuid4().hex[:12]
        profile = dict(name=name, role=role, assigned_bed=assigned_bed,
            consent_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        return identity, profile, embeddings, rejected
