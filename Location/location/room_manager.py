"""
Room Manager — หาว่าอยู่ห้องไหน / ชั้นไหน / site ไหน

ลำดับการตัดสิน room_id
  1. Polygon ชนิด room ที่จุดตกอยู่            (กล้องครอบหลายห้อง)
  2. room_id ที่ zone ย่อยระบุไว้ (เช่น bed_203_A → room_203)
  3. Zone แบบพื้นที่ส่วนกลาง (hallway, dining, ...) → ใช้ zone_id เป็น room
  4. sources.<source_id>.room_id ใน config      (กล้อง 1 ตัว = 1 ห้อง)
  5. None
"""

from __future__ import annotations

from .models import Location
from .zone_engine import Zone

STANDALONE_AREA_TYPES = {
    "hallway", "dining", "garden", "nurse_station", "exit_corridor", "exit",
}


class RoomManager:
    def __init__(self, sources_cfg: dict | None = None,
                 default_site: str | None = None, default_floor: str | None = None):
        self.sources = sources_cfg or {}
        self.default_site = default_site
        self.default_floor = default_floor

    def source_info(self, source_id) -> dict:
        return self.sources.get(source_id, {}) or {}

    def room_for(self, zones: list[Zone], source_id) -> str | None:
        for z in zones:
            if z.type == "room":
                return z.room_id or z.zone_id
        for z in zones:
            if z.room_id:
                return z.room_id
        for z in zones:
            if z.type in STANDALONE_AREA_TYPES:
                return z.zone_id
        return self.source_info(source_id).get("room_id")

    def floor_for(self, zones: list[Zone], source_id) -> str | None:
        for z in zones:
            if z.floor_id:
                return z.floor_id
        return self.source_info(source_id).get("floor_id", self.default_floor)

    def site_for(self, zones: list[Zone], source_id) -> str | None:
        for z in zones:
            if z.site_id:
                return z.site_id
        return self.source_info(source_id).get("site_id", self.default_site)

    def build_location(self, zones: list[Zone], source_id, bed_id) -> Location:
        """zones = ผลจาก ZoneEngine.match() (เรียงเจาะจงสุดก่อนแล้ว)"""
        top = zones[0] if zones else None
        return Location(
            site_id=self.site_for(zones, source_id),
            floor_id=self.floor_for(zones, source_id),
            room_id=self.room_for(zones, source_id),
            zone_id=top.zone_id if top else "unknown",
            zone_type=top.type if top else "unknown",
            bed_id=bed_id,
            matched_zones=tuple(z.zone_id for z in zones),
        )
