"""
config.py
=========
จุดเดียวที่ต้องแก้เวลาจะ "เสียบ/ถอด" ฟีเจอร์ หรือเปลี่ยนค่ากล้อง/โมเดล
ไม่ต้องแตะ main.py หรือ core/* เลย
"""

from __future__ import annotations

import json
from pathlib import Path

from features.base import BaseFeature
from features.boundary import BoundaryFeature
from utils.geometry import Point, polygon_edges, scale_polygon

BASE_DIR = Path(__file__).resolve().parent

# ---- Camera / Video source ----
# ใช้เลข 0 สำหรับ webcam, หรือ "rtsp://user:pass@ip:port/stream" สำหรับกล้อง IP
CAMERA_SOURCE: int | str = 0

# ---- Detection / Tracking ----
# เปลี่ยนจาก yolo11n -> yolo11s (แม่นขึ้นชัดเจน แต่ไฟล์โมเดลใหญ่ขึ้น/ช้าลงเล็กน้อยบน CPU)
# และลด confidence จาก 0.5 -> 0.25 เพราะพบว่าเฟรมที่คนอยู่ในท่าทางไม่ปกติ (ล้ม/ประชิดตัว/ถูกบัง)
# โมเดลมักให้ค่าความมั่นใจต่ำกว่า 0.5 ทั้งที่ตรวจถูก การตั้งไว้ 0.5 เลยพลาดไปเยอะ
YOLO_MODEL_PATH = "yolo11s.pt"
DETECTION_CONFIDENCE = 0.25
DEVICE = "cpu"  # ถ้าเครื่องมี GPU (NVIDIA) และ torch รุ่น CUDA ใช้ได้ ให้เปลี่ยนเป็น "cuda:0" จะเร็วขึ้นมาก

# ---- Zones (เก็บเป็น "สัดส่วน" 0.0-1.0 ไม่ใช่พิกเซลตายตัว) ----
_zone_config = json.loads((BASE_DIR / "zones" / "zone_config.json").read_text(encoding="utf-8"))
ZONES_RELATIVE: dict[str, list[Point]] = {
    name: [tuple(p) for p in poly] for name, poly in _zone_config["zones"].items()
}

# True  -> ตรวจจับ/แสดงผลเฉพาะคนที่อยู่ในกรอบโซนเท่านั้น คนนอกกรอบจะไม่ถูกตรวจจับเลย
# False -> ตรวจจับทั้งเฟรมตามเดิม (โซนใช้แค่เช็ค event เข้า/ออก)
DETECT_ONLY_INSIDE_ZONE = True


def build_zones_and_lines(
    width: int, height: int
) -> tuple[dict[str, list[Point]], dict[str, tuple[Point, Point]]]:
    """
    แปลงโซนแบบสัดส่วน (ZONES_RELATIVE) ให้เป็นพิกัดพิกเซลจริง ตาม width/height ของ
    วิดีโอ/กล้องตัวที่กำลังรันอยู่ -> "ปรับพิกัดอัตโนมัติ" ทุกครั้งที่เปลี่ยนแหล่งวิดีโอ
    ไม่ต้องมานั่งวัดพิกัดพิกเซลเองทีละคลิป

    เส้นข้าม (lines) ยังคงรวมเป็นขอบของกรอบโซนเหมือนเดิม (ดูคำตอบเรื่อง "รวมเส้นกับกรอบ")
    """
    zones_px = {
        name: scale_polygon(relative_polygon, width, height)
        for name, relative_polygon in ZONES_RELATIVE.items()
    }
    lines_px = {
        f"{zone_name}_edge_{i}": edge
        for zone_name, polygon in zones_px.items()
        for i, edge in enumerate(polygon_edges(polygon))
    }
    return zones_px, lines_px


def build_feature_pipeline(width: int, height: int) -> list[BaseFeature]:
    """
    รายการฟีเจอร์ที่ pipeline จะรันเรียงตามลำดับในลิสต์นี้ทุกเฟรม
    ต้องส่ง width/height ของวิดีโอ/กล้องเข้ามา เพื่อ scale โซนแบบสัดส่วนให้เป็นพิกเซลจริงก่อน
    ตอนนี้เปิดใช้เฉพาะ Boundary (ออกนอกสถานที่ / ข้ามเส้น)
    ฟีเจอร์อื่นในอนาคต (violence/fall/seizure) แค่สร้างคลาสใหม่ที่ inherit BaseFeature
    แล้วเติมต่อท้ายลิสต์นี้ ไม่ต้องแก้ main.py
    """
    zones_px, lines_px = build_zones_and_lines(width, height)
    return [
        BoundaryFeature(zones=zones_px, lines=lines_px, enabled=True),
    ]