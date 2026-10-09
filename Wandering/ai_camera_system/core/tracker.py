"""
core/tracker.py
================
ครอบ ByteTrack ไว้ชั้นหนึ่ง รับ list[Detection] (จาก detector.py) -> คืน list[TrackedObject]
(schema กลาง) พร้อม track_id ที่คงที่ในแต่ละ object ตลอดการปรากฏในเฟรม

ใช้ implementation จากไลบรารี `supervision` (ByteTrack) เพราะติดตั้งง่ายและ
คืนค่าที่แปลงเป็น numpy array ตรงไปตรงมา ถ้าต้องการสลับไปใช้ ByteTrack เวอร์ชันอื่น
แก้ไฟล์นี้ไฟล์เดียว ส่วนอื่นของระบบไม่กระทบ เพราะรับ-ส่งผ่าน schema กลางเท่านั้น
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import supervision as sv

from core.schema import BoundingBox, Detection, TrackedObject

MAX_TRAIL_LENGTH = 30  # เก็บ trail ย้อนหลังกี่เฟรม สำหรับ feature อื่น (fall/seizure) ใช้วิเคราะห์


class PersonTracker:
    def __init__(
        self,
        track_activation_threshold: float = 0.5,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        # เก็บ trail ของแต่ละ track_id ไว้ที่ tracker (อายุยืนกว่า 1 เฟรม)
        self._trails: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=MAX_TRAIL_LENGTH)
        )

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        if not detections:
            sv_detections = sv.Detections.empty()
        else:
            xyxy = np.array([d.bbox.as_xyxy() for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id = np.array([d.class_id for d in detections], dtype=int)
            sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

        tracked = self._tracker.update_with_detections(sv_detections)

        class_names = {d.class_id: d.class_name for d in detections}
        tracked_objects: list[TrackedObject] = []
        for xyxy, _, conf, cls_id, track_id, _ in tracked:
            bbox = BoundingBox(*xyxy.tolist())
            trail = self._trails[int(track_id)]
            trail.append(bbox.bottom_center)

            tracked_objects.append(
                TrackedObject(
                    track_id=int(track_id),
                    bbox=bbox,
                    confidence=float(conf) if conf is not None else 0.0,
                    class_name=class_names.get(int(cls_id), "person"),
                    trail=list(trail),
                )
            )
        return tracked_objects
