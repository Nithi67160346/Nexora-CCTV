"""
features/boundary.py
=====================
ฟีเจอร์ที่ 2 ของเฟสนี้: "ออกนอกสถานที่" (Boundary & Line Crossing)
เช็คว่าคน (TrackedObject) เดินเข้า/ออกโพลิกอนโซน หรือข้ามเส้นกั้นที่กำหนดหรือไม่
ไม่รัน YOLO/ByteTrack เอง — ใช้ tracked_objects ที่ pipeline ส่งมาให้เท่านั้น
"""

from __future__ import annotations

import numpy as np

if __package__ and __package__.count('.') >= 2:
    from ..core.schema import FeatureEvent, FrameResult, ZoneEventType
    from .base import BaseFeature
    from ..utils.geometry import Point, point_in_polygon, segments_intersect
else:
    from core.schema import FeatureEvent, FrameResult, ZoneEventType
    from features.base import BaseFeature
    from utils.geometry import Point, point_in_polygon, segments_intersect


class BoundaryFeature(BaseFeature):
    name = "boundary"

    def __init__(
        self,
        zones: dict[str, list[Point]] | None = None,
        lines: dict[str, tuple[Point, Point]] | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        # โซนต้องห้าม: {"restricted_area": [(x1,y1), (x2,y2), ...]}
        self.zones = zones or {}
        # เส้นกั้น: {"gate_line": ((x1,y1), (x2,y2))}
        self.lines = lines or {}
        # จำตำแหน่งจุดเท้าเฟรมก่อนหน้าของแต่ละ track ไว้เช็คว่า "ข้ามเส้น" เมื่อไร
        self._prev_position: dict[int, Point] = {}
        # จำว่าตอนนี้แต่ละ track อยู่ในโซนไหนบ้าง ไว้เช็ค enter/exit โดยไม่แจ้งซ้ำทุกเฟรม
        self._inside_zones: dict[int, set[str]] = {}

    def reset(self) -> None:
        self._prev_position.clear()
        self._inside_zones.clear()

    def process(self, frame: np.ndarray, frame_result: FrameResult) -> FrameResult:
        if not self.enabled:
            return frame_result

        for obj in frame_result.tracked_objects:
            current_point = obj.bbox.bottom_center
            self._check_zones(obj.track_id, current_point, frame_result)
            self._check_lines(obj.track_id, current_point, frame_result)
            self._prev_position[obj.track_id] = current_point

        return frame_result

    def _check_zones(self, track_id: int, point: Point, frame_result: FrameResult) -> None:
        currently_inside = self._inside_zones.setdefault(track_id, set())
        for zone_name, polygon in self.zones.items():
            is_inside = point_in_polygon(point, polygon)
            was_inside = zone_name in currently_inside

            if is_inside and not was_inside:
                currently_inside.add(zone_name)
                frame_result.add_event(
                    FeatureEvent(
                        feature_name=self.name,
                        track_id=track_id,
                        event_type=ZoneEventType.ENTERED_ZONE.value,
                        message=f"Track {track_id} เข้าพื้นที่ '{zone_name}'",
                        severity="warning",
                        extra={"zone": zone_name},
                    )
                )
            elif not is_inside and was_inside:
                currently_inside.discard(zone_name)
                frame_result.add_event(
                    FeatureEvent(
                        feature_name=self.name,
                        track_id=track_id,
                        event_type=ZoneEventType.EXITED_ZONE.value,
                        message=f"Track {track_id} ออกจากพื้นที่ '{zone_name}'",
                        severity="warning",
                        extra={"zone": zone_name},
                    )
                )

    def _check_lines(self, track_id: int, point: Point, frame_result: FrameResult) -> None:
        prev_point = self._prev_position.get(track_id)
        if prev_point is None:
            return  # เฟรมแรกที่เจอ track นี้ ยังไม่มีเส้นทางให้เทียบ

        for line_name, (line_start, line_end) in self.lines.items():
            if segments_intersect(prev_point, point, line_start, line_end):
                frame_result.add_event(
                    FeatureEvent(
                        feature_name=self.name,
                        track_id=track_id,
                        event_type=ZoneEventType.LINE_CROSSED_IN.value,
                        message=f"Track {track_id} ข้ามเส้น '{line_name}'",
                        severity="critical",
                        extra={"line": line_name},
                    )
                )
