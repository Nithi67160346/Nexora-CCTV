"""
core/schema.py
================
Data Schema กลาง (Central Data Contract)

ทุกโมดูล (Detector, Tracker, Feature ต่าง ๆ) จะสื่อสารกันผ่าน Data Class เหล่านี้เท่านั้น
เพื่อให้ทุกฟีเจอร์ในอนาคต (Violence / Fall / Seizure / ...) ใช้ผลลัพธ์จากการ Detect + Track
ชุดเดียวกันได้ โดยไม่ต้องรันโมเดล YOLO หรือ ByteTrack ซ้ำ
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any


class ZoneEventType(str, Enum):
    """สถานะที่เกี่ยวกับขอบเขตพื้นที่ / เส้นกั้น"""

    NONE = "none"
    ENTERED_ZONE = "entered_zone"
    EXITED_ZONE = "exited_zone"
    LINE_CROSSED_IN = "line_crossed_in"
    LINE_CROSSED_OUT = "line_crossed_out"


@dataclass(slots=True)
class BoundingBox:
    """พิกัดกรอบวัตถุ รูปแบบ xyxy (พิกเซล, origin บนซ้าย)"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        # จุดเท้า ใช้เช็คการข้ามเส้น/เข้าออกโซนแม่นกว่าจุดกึ่งกลาง
        return ((self.x1 + self.x2) / 2, self.y2)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(slots=True)
class Detection:
    """ผลลัพธ์ดิบจาก Detector (ก่อน Track ID)"""

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(slots=True)
class TrackedObject:
    """
    วัตถุ 1 ชิ้นหลังผ่าน Tracker (มี track_id คงที่ตลอดอายุการปรากฏในเฟรม)
    นี่คือ "หน่วยข้อมูลกลาง" ที่ทุก Feature module จะรับไปประมวลผลต่อ
    """

    track_id: int
    bbox: BoundingBox
    confidence: float
    class_name: str
    # ประวัติจุดเท้าไม่กี่เฟรมล่าสุด ไว้ให้ feature อื่น (fall/seizure) ใช้วิเคราะห์ท่าทาง/ความเร็ว
    trail: list[tuple[float, float]] = field(default_factory=list)
    # ที่เก็บสถานะเฉพาะของแต่ละ feature เช่น {"boundary": "inside", "fall": "standing"}
    # ทำให้ feature ใหม่เพิ่ม state ของตัวเองได้โดยไม่ต้องแก้ schema กลาง
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureEvent:
    """Event ที่ feature module ใดก็ตามสร้างขึ้นเพื่อแจ้งเตือน/บันทึก"""

    feature_name: str
    track_id: int | None
    event_type: str
    message: str
    severity: str = "info"  # info | warning | critical
    timestamp: float = field(default_factory=time)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrameResult:
    """
    Payload มาตรฐานที่ Pipeline ส่งให้ทุก Feature module ในแต่ละเฟรม
    เป็น "สัญญา" เดียวที่ feature ทุกตัวต้อง consume
    """

    frame_id: int
    timestamp: float
    frame_width: int
    frame_height: int
    tracked_objects: list[TrackedObject] = field(default_factory=list)
    events: list[FeatureEvent] = field(default_factory=list)

    def add_event(self, event: FeatureEvent) -> None:
        self.events.append(event)
