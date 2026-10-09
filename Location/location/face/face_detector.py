"""
Face Detector — ค้นหาใบหน้า "ภายใน Person Bounding Box" เท่านั้น

Person bbox → crop (+margin) → face detector → face bbox (พิกัดเฟรมเต็ม)

Backend
  yolo_face : yolov8n-face.pt (akanametov/yolo-face release 1.0.0) — ตัวเดียวกับที่วัด
              Precision ไว้ใน F4 เดิม (91.38% ที่ conf 0.25) ไม่มี landmark
  yunet     : OpenCV YuNet (face_detection_yunet_2023mar.onnx, MIT) — ให้ landmark 5 จุด
              ทำให้ SFace align หน้าได้แม่นขึ้น

หมายเหตุ: ไม่ใช่ person detector — ไม่ขัดกฎ "ห้ามสร้าง Person Detector ใหม่"
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class FaceDetection:
    bbox: tuple                      # (x1, y1, x2, y2) พิกัดเฟรมเต็ม
    conf: float
    landmarks: np.ndarray | None = None   # YuNet row (15 ค่า) พิกัดเฟรมเต็ม

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def crop_person(frame, bbox, margin: float = 0.05):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    X1, Y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    X2, Y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if X2 <= X1 or Y2 <= Y1:
        return None, (0, 0)
    return frame[Y1:Y2, X1:X2], (X1, Y1)


def crop_face(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class FaceDetector:
    """Interface — backend ใหม่ให้ override detect_in_crop()"""

    def detect_in_crop(self, crop) -> list[FaceDetection]:
        raise NotImplementedError

    def detect_in_person(self, frame, person_bbox) -> list[FaceDetection]:
        crop, (ox, oy) = crop_person(frame, person_bbox)
        if crop is None:
            return []
        out = []
        for f in self.detect_in_crop(crop):
            x1, y1, x2, y2 = f.bbox
            lm = None
            if f.landmarks is not None:
                lm = np.array(f.landmarks, dtype=np.float32).copy()
                lm[0:14:2] += ox
                lm[1:14:2] += oy
            out.append(FaceDetection((x1 + ox, y1 + oy, x2 + ox, y2 + oy), f.conf, lm))
        return out

    @staticmethod
    def best(faces: list[FaceDetection]):
        return max(faces, key=lambda f: f.area * f.conf) if faces else None


class YoloFaceDetector(FaceDetector):
    def __init__(self, model_path="yolov8n-face.pt", conf=0.25, imgsz=320, device=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ไม่พบ {model_path} — โหลดจาก https://github.com/akanametov/yolo-face/"
                f"releases/download/1.0.0/yolov8n-face.pt")
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect_in_crop(self, crop):
        r = self.model.predict(crop, conf=self.conf, imgsz=self.imgsz,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        return [FaceDetection(tuple(float(v) for v in xyxy[i]), float(cf[i]))
                for i in range(len(xyxy))]


class YuNetFaceDetector(FaceDetector):
    def __init__(self, model_path="face_detection_yunet_2023mar.onnx", conf=0.6):
        import cv2
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ไม่พบ {model_path} — โหลดจาก https://media.githubusercontent.com/media/opencv/"
                f"opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), conf, 0.3, 50)

    def detect_in_crop(self, crop):
        h, w = crop.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(crop)
        if faces is None:
            return []
        out = []
        for row in faces:
            x, y, fw, fh = row[:4]
            out.append(FaceDetection((float(x), float(y), float(x + fw), float(y + fh)),
                                     float(row[14]), row[:15].copy()))
        return out


def create_face_detector(cfg: dict) -> FaceDetector:
    backend = cfg.get("backend", "yolo_face")
    if backend == "yolo_face":
        return YoloFaceDetector(cfg.get("model_path", "yolov8n-face.pt"),
                                conf=cfg.get("conf", 0.25), imgsz=cfg.get("imgsz", 320),
                                device=cfg.get("device"))
    if backend == "yunet":
        return YuNetFaceDetector(cfg.get("model_path", "face_detection_yunet_2023mar.onnx"),
                                 conf=cfg.get("conf", 0.6))
    raise ValueError(f"face_detection.backend ไม่รองรับ: {backend}")
