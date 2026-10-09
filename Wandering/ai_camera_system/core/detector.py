"""
core/detector.py
=================
ครอบ YOLOv11 (ultralytics) ไว้ชั้นหนึ่ง เพื่อไม่ให้ส่วนอื่นของระบบ "รู้จัก" ultralytics
โดยตรง — ถ้าวันหนึ่งอยากเปลี่ยนไปใช้โมเดลตัวอื่น (เช่น YOLOv12 หรือโมเดลกำหนดเอง)
แก้ไฟล์นี้ไฟล์เดียวพอ ส่วนอื่นไม่ต้องเปลี่ยน เพราะ output เป็น Detection (schema กลาง) เสมอ
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from core.schema import BoundingBox, Detection

# ตาม COCO class index: person = 0
PERSON_CLASS_ID = 0


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",  # "cuda:0" ถ้ามี GPU
        target_classes: tuple[int, ...] = (PERSON_CLASS_ID,),
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.target_classes = set(target_classes)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """รัน YOLOv11 บนเฟรมเดียว แล้วแปลงผลลัพธ์เป็น Detection (schema กลาง)"""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            classes=list(self.target_classes) or None,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=result.names.get(cls_id, str(cls_id)),
                )
            )
        return detections
