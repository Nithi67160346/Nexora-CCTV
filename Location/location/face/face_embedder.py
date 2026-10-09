"""
Face Embedder — แปลงใบหน้าเป็นเวกเตอร์ (L2-normalized) สำหรับเทียบ cosine similarity

Backend สำหรับ Prototype: OpenCV SFace
  - ไฟล์: face_recognition_sface_2021dec.onnx (~37 MB)
  - แหล่ง: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
  - License: Apache-2.0 | รันด้วย cv2 ล้วน ไม่ต้องใช้ GPU
  - ถ้ามี landmark (YuNet) → alignCrop ; ถ้าไม่มี (yolo_face) → resize กรอบเป็น 112×112
    (แม่นน้อยกว่า align ควรใช้ yunet เมื่อต้องการ recognition จริงจัง)

⚠ Embedding ยังนับเป็นข้อมูล biometric (PDPA) — ต้องป้องกันเหมือนภาพใบหน้า
"""

from __future__ import annotations

import os

import numpy as np


def l2_normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class FaceEmbedder:
    """Interface — backend ใหม่ (ArcFace, service ภายนอก ฯลฯ) ให้ override embed()"""
    dim: int = 0

    def embed(self, frame, face) -> np.ndarray:
        raise NotImplementedError


class SFaceEmbedder(FaceEmbedder):
    dim = 128

    def __init__(self, model_path="face_recognition_sface_2021dec.onnx"):
        import cv2
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ไม่พบ {model_path} — โหลดจาก https://media.githubusercontent.com/media/opencv/"
                f"opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
        self.cv2 = cv2
        self.rec = cv2.FaceRecognizerSF.create(model_path, "")

    def embed(self, frame, face):
        cv2 = self.cv2
        if face.landmarks is not None:
            aligned = self.rec.alignCrop(frame, np.asarray(face.landmarks, dtype=np.float32))
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                return None
            aligned = cv2.resize(crop, (112, 112))
        return l2_normalize(self.rec.feature(aligned))


def create_face_embedder(cfg: dict) -> FaceEmbedder:
    backend = cfg.get("embedder", "sface")
    if backend == "sface":
        return SFaceEmbedder(cfg.get("embedder_model_path", "face_recognition_sface_2021dec.onnx"))
    raise ValueError(f"face_recognition.embedder ไม่รองรับ: {backend}")
