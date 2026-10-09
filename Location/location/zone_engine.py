"""
Zone Engine — ตรรกะเรขาคณิตล้วน (ไม่รู้จักโมเดล / กล้อง / ใบหน้า)

รับจุด (x, y) แล้วบอกว่าอยู่ใน Zone ใดบ้าง
- โหลด Polygon จาก config (YAML/JSON) — ห้าม hard-code ใน Python
- รองรับ Zone ซ้อนกัน (room ⊃ bed) โดยเรียงจาก "เจาะจงที่สุด" ก่อน
- รองรับ boundary tolerance สำหรับ hysteresis (ดู state_manager.py)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

ZONE_TYPES = {
    "room", "bed", "bedside", "hallway", "dining", "bathroom_entrance",
    "garden", "nurse_station", "exit_corridor", "exit", "restricted_area",
    "safe_zone", "transition_zone", "unknown",
}

# ยิ่งเลขสูง ยิ่งเจาะจง → ถูกเลือกเป็น current_zone ก่อน
TYPE_SPECIFICITY = {
    "bed": 30,
    "bedside": 20, "bathroom_entrance": 20, "restricted_area": 20,
    "nurse_station": 20, "transition_zone": 20,
    "safe_zone": 15,
    "hallway": 10, "dining": 10, "garden": 10, "exit_corridor": 10, "exit": 10,
    "room": 5,
    "unknown": 0,
}

POINT_MODES = ("bottom_center", "center", "top_center")


def reference_point(bbox_xyxy, mode: str = "bottom_center"):
    """
    จุดอ้างอิงของ bounding box
      bottom_center (default) : ((x1+x2)/2, y2)  — ตำแหน่งเท้า เหมาะกับ floor mapping
      center                  : จุดกึ่งกลาง      — ใช้กับกรอบใบหน้า
      top_center              : ((x1+x2)/2, y1)
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    cx = (x1 + x2) / 2.0
    if mode == "bottom_center":
        return (cx, y2)
    if mode == "center":
        return (cx, (y1 + y2) / 2.0)
    if mode == "top_center":
        return (cx, y1)
    raise ValueError(f"point_mode ต้องเป็นหนึ่งใน {POINT_MODES}")


@dataclass
class Zone:
    zone_id: str
    type: str
    points: list
    room_id: str | None = None
    bed_id: str | None = None
    floor_id: str | None = None
    site_id: str | None = None
    source_ids: list | None = None      # None = ใช้กับทุกกล้อง
    priority: int | None = None         # override specificity
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.points) < 3:
            raise ValueError(f"zone '{self.zone_id}' ต้องมีอย่างน้อย 3 จุด")
        self.points = [(float(x), float(y)) for x, y in self.points]
        if self.type not in ZONE_TYPES:
            self.extra.setdefault("unrecognized_type", self.type)

    # ── geometry ────────────────────────────────────────────────────────
    @property
    def area(self) -> float:
        s = 0.0
        pts = self.points
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0

    @property
    def specificity(self) -> int:
        if self.priority is not None:
            return int(self.priority)
        return TYPE_SPECIFICITY.get(self.type, 10)

    def signed_distance(self, pt) -> float:
        """> 0 อยู่ใน, < 0 อยู่นอก, 0 อยู่บนเส้น (หน่วย px)"""
        d = _distance_to_edges(pt, self.points)
        return d if _point_in_polygon(pt, self.points) else -d

    def contains(self, pt, tolerance: float = 0.0) -> bool:
        """tolerance > 0 = ขยายขอบออกไป (ใช้ตอนตรวจว่ายัง 'อยู่ใน zone เดิม' ไหม)"""
        if _point_in_polygon(pt, self.points):
            return True
        return tolerance > 0 and _distance_to_edges(pt, self.points) <= tolerance

    def applies_to(self, source_id) -> bool:
        return self.source_ids is None or source_id in self.source_ids


class ZoneEngine:
    def __init__(self, zones: list[Zone]):
        ids = [z.zone_id for z in zones]
        if len(ids) != len(set(ids)):
            raise ValueError("zone_id ซ้ำกัน")
        self.zones = zones
        self._by_id = {z.zone_id: z for z in zones}

    @classmethod
    def from_config(cls, zones_cfg: dict | None) -> "ZoneEngine":
        zones = []
        for zid, spec in (zones_cfg or {}).items():
            spec = dict(spec)
            sids = spec.pop("source_ids", None) or spec.pop("source_id", None)
            if isinstance(sids, str):
                sids = [sids]
            zones.append(Zone(
                zone_id=str(zid),
                type=spec.pop("type", "unknown"),
                points=spec.pop("points"),
                room_id=spec.pop("room_id", None),
                bed_id=spec.pop("bed_id", None),
                floor_id=spec.pop("floor_id", None),
                site_id=spec.pop("site_id", None),
                source_ids=sids,
                priority=spec.pop("priority", None),
                extra=spec,
            ))
        return cls(zones)

    def get(self, zone_id):
        return self._by_id.get(zone_id)

    def match(self, pt, source_id=None, tolerance: float = 0.0) -> list[Zone]:
        """ทุก zone ที่มีจุดนี้อยู่ เรียงจากเจาะจงที่สุด → กว้างที่สุด"""
        hits = [z for z in self.zones
                if z.applies_to(source_id) and z.contains(pt, tolerance)]
        # เจาะจงกว่าก่อน; ถ้าเท่ากัน zone เล็กกว่าก่อน; ถ้าเท่ากันอีก อยู่ลึกกว่าก่อน
        hits.sort(key=lambda z: (-z.specificity, z.area, -z.signed_distance(pt)))
        return hits


# ── geometry helpers (pure python ไม่ต้องใช้ OpenCV) ───────────────────────

def _point_in_polygon(pt, poly) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if _on_segment(x, y, x1, y1, x2, y2):
            return True                       # บนเส้น = นับว่าอยู่ใน
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
    return inside


def _on_segment(x, y, x1, y1, x2, y2, eps=1e-9) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps * max(1.0, abs(x2 - x1) + abs(y2 - y1)):
        return False
    return min(x1, x2) - eps <= x <= max(x1, x2) + eps and \
        min(y1, y2) - eps <= y <= max(y1, y2) + eps


def _distance_to_edges(pt, poly) -> float:
    px, py = pt
    best = math.inf
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
        cx, cy = x1 + t * dx, y1 + t * dy
        best = min(best, math.hypot(px - cx, py - cy))
    return best
