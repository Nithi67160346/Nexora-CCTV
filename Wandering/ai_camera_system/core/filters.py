"""
core/filters.py
================
กรองผล Detection ตั้งแต่ก่อนเข้า Tracker — ถ้าจุดเท้า (bottom_center) ของกรอบ
ไม่อยู่ในโซนที่กำหนดเลยสักโซน จะถูกตัดทิ้ง ไม่ถูกส่งต่อไป track/วาดกรอบ/เช็ค event ใด ๆ
ทำเป็นไฟล์แยกต่างหาก (ไม่ยัดใส่ boundary.py) เพราะเป็นคนละหน้าที่:
  - filters.py  = ตัดสินใจว่า "จะสนใจคนนี้ไหม" (ก่อน tracking)
  - boundary.py = ตัดสินใจว่า "คนที่สนใจอยู่แล้ว เข้า/ออก/ข้ามหรือยัง" (หลัง tracking)
"""

from __future__ import annotations

from core.schema import Detection
from utils.geometry import Point, point_in_polygon


def filter_detections_in_zones(
    detections: list[Detection],
    zones: dict[str, list[Point]],
) -> list[Detection]:
    """
    คืนเฉพาะ Detection ที่จุดเท้าอยู่ในโซนใดโซนหนึ่ง (union ของทุกโซนที่ส่งเข้ามา)
    ถ้า zones ว่าง (ไม่ได้ตั้งค่าไว้) จะคืนค่าเดิมทั้งหมด — ไม่กรองอะไร
    """
    if not zones:
        return detections

    kept: list[Detection] = []
    for detection in detections:
        point = detection.bbox.bottom_center
        if any(point_in_polygon(point, polygon) for polygon in zones.values()):
            kept.append(detection)
    return kept
