"""
utils/geometry.py
==================
ฟังก์ชันคณิตศาสตร์ล้วน ๆ (pure functions) ไม่ผูกกับ schema หรือ opencv โดยตรง
เพื่อให้ feature module ไหนก็เรียกใช้ได้ และเทสต์แยกได้ง่าย
"""

from __future__ import annotations

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray casting algorithm — คืนค่า True ถ้าจุดอยู่ในโพลิกอน"""
    x, y = point
    inside = False
    n = len(polygon)
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    x_intersect = x1 if y1 == y2 else (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or x <= x_intersect:
                        inside = not inside
        x1, y1 = x2, y2
    return inside


def which_side_of_line(point: Point, line_start: Point, line_end: Point) -> float:
    """
    คืนค่า sign ว่าจุดอยู่ฝั่งไหนของเส้นตรง (ลากจาก line_start -> line_end)
    > 0 = ฝั่งหนึ่ง, < 0 = อีกฝั่ง, 0 = อยู่บนเส้นพอดี
    ใช้เช็คว่ามีการ "ข้ามเส้น" เกิดขึ้นระหว่าง 2 เฟรม (สลับ sign) หรือไม่
    """
    (x, y), (x1, y1), (x2, y2) = point, line_start, line_end
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """เช็คว่าเส้นตรง p1-p2 (เส้นทางการเดินของคน) ตัดกับเส้น p3-p4 (เส้นกั้น) หรือไม่"""

    def orientation(a: Point, b: Point, c: Point) -> int:
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    o1, o2 = orientation(p1, p2, p3), orientation(p1, p2, p4)
    o3, o4 = orientation(p3, p4, p1), orientation(p3, p4, p2)
    return o1 != o2 and o3 != o4


def polygon_edges(polygon: list[Point]) -> list[tuple[Point, Point]]:
    """
    แปลงโพลิกอน (จุดมุมกรอบ) ให้เป็นรายการ "เส้นขอบ" แต่ละด้าน
    ใช้รวมเส้นกั้น (line-crossing) เข้ากับขอบของโซนกรอบเดียวกัน แทนที่จะตั้งเส้นแยกลอย ๆ
    เช่น กรอบสี่เหลี่ยม 4 จุด -> คืนค่าเส้นขอบ 4 ด้าน (บน/ล่าง/ซ้าย/ขวา)
    """
    n = len(polygon)
    return [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]


def scale_polygon(relative_polygon: list[Point], width: int, height: int) -> list[Point]:
    """
    แปลงพิกัดโซนแบบ "สัดส่วน" (0.0-1.0 เทียบกับความกว้าง/สูงของภาพ) ให้เป็นพิกัดพิกเซลจริง
    ตาม resolution ของวิดีโอ/กล้องแต่ละตัว — นี่คือกลไกที่ทำให้ "ปรับพิกัดอัตโนมัติ"
    เวลาสลับไปรันกับคลิปที่ความละเอียดไม่เท่ากัน (เช่น 320x240, 1280x720, แนวตั้ง ฯลฯ)
    โดยไม่ต้องตั้งพิกัดพิกเซลใหม่ทุกครั้ง — แค่กำหนดว่า "อยู่ตรงไหนของเฟรมเป็นเปอร์เซ็นต์" ครั้งเดียว
    """
    return [(x * width, y * height) for x, y in relative_polygon]
