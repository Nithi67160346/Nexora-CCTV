"""
Bed Manager — รองรับ 2 แบบ ใช้คู่กันได้ และ "เป็นคนละค่า"

MODE 1  Geometric Bed Zone      : จุดของ track อยู่ใน polygon ชนิด bed ใด → current_detected_bed
MODE 2  Resident Bed Assignment : profile ผู้พักระบุ assigned_bed          → assigned_bed

Feature นี้ "ไม่ตัดสิน" ว่า assigned ≠ detected เป็นเรื่องผิดปกติ
แค่รายงานทั้งสองค่า ให้ Rule Engine / Feature อื่นตัดสินเอง
"""

from __future__ import annotations

from .zone_engine import Zone


class BedManager:
    def __init__(self, residents_cfg: dict | None = None):
        self.residents = residents_cfg or {}

    @staticmethod
    def detected_bed(zones: list[Zone]) -> str | None:
        for z in zones:
            if z.type == "bed":
                return z.bed_id or z.zone_id
        return None

    def assigned_bed(self, resident_id) -> str | None:
        if resident_id is None:
            return None
        return (self.residents.get(resident_id) or {}).get("assigned_bed")

    def bed_info(self, zones: list[Zone], resident_id) -> dict:
        detected = self.detected_bed(zones)
        assigned = self.assigned_bed(resident_id)
        return {
            "current_detected_bed": detected,
            "assigned_bed": assigned,
            # ข้อมูลประกอบเท่านั้น ไม่ใช่ alert
            "bed_matches_assignment": None if (detected is None or assigned is None)
            else detected == assigned,
        }

    def display_name(self, resident_id) -> str | None:
        return (self.residents.get(resident_id) or {}).get("display_name")
