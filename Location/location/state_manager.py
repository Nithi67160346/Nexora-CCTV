"""
State Manager — สถานะต่อ Track ID + Zone Hysteresis + Tracking Loss

Hysteresis (กันสลับ Room A → Hallway → Room A ตรงขอบ polygon) ใช้ 3 ชั้น
  1. boundary_tolerance_px : หลุดขอบ zone ปัจจุบันไม่เกิน N px → ถือว่ายังอยู่ zone เดิม
  2. switch_confirmation_frames : ตำแหน่งใหม่ต้องคงที่ติดกัน N เฟรม
  3. minimum_dwell_sec : และต้องอยู่ต่อเนื่องอย่างน้อย N วินาที
ตำแหน่งแรกของ track ใหม่ commit ทันที (ไม่อย่างนั้นจะ unknown หลายเฟรม)

Tracking loss
  active   → เห็นในเฟรมนี้
  occluded → หายไปไม่เกิน lost_track_timeout_sec (ยังไม่ประกาศว่าหาย)
  lost     → หายเกิน timeout → ส่ง last_seen_update หนึ่งครั้ง
  (ลบออกจากหน่วยความจำเมื่อเกิน last_seen_retention_sec)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import Identity, Location, unknown_identity

ACTIVE, OCCLUDED, LOST = "active", "occluded", "lost"


@dataclass
class TrackState:
    source_id: str
    track_id: int
    first_seen_timestamp: int
    status: str = ACTIVE

    location: Location | None = None
    previous_location: Location | None = None
    zone_enter_timestamp: int | None = None
    room_enter_timestamp: int | None = None

    last_seen_timestamp: int | None = None
    last_seen_source_id: str | None = None
    last_point: tuple | None = None
    last_bbox: tuple | None = None
    tracking_confidence: float = 0.0

    identity: Identity = field(default_factory=unknown_identity)
    reid_from: tuple | None = None
    last_face: object = None           # FaceObservation ล่าสุด (ถ้าเปิด face layer)

    # hysteresis
    candidate: Location | None = None
    candidate_count: int = 0
    candidate_since: int | None = None

    # metrics
    zone_switches: int = 0
    recoveries: int = 0

    @property
    def key(self):
        return (self.source_id, self.track_id)

    # ── alias ตามชื่อฟิลด์ในสเปก ──────────────────────────────────────────
    @property
    def current_zone(self):
        return self.location.zone_id if self.location else None

    @property
    def previous_zone(self):
        return self.previous_location.zone_id if self.previous_location else None

    @property
    def current_room(self):
        return self.location.room_id if self.location else None

    @property
    def current_bed(self):
        return self.location.bed_id if self.location else None

    @property
    def identity_id(self):
        return self.identity.identity_id

    @property
    def identity_name(self):
        return self.identity.display_name

    @property
    def identity_confidence(self):
        return self.identity.confidence

    @property
    def identity_source(self):
        return self.identity.source


class StateManager:
    def __init__(self, switch_confirmation_frames: int = 5, minimum_dwell_sec: float = 0.0,
                 boundary_tolerance_px: float = 15.0, lost_track_timeout_sec: float = 2.0,
                 last_seen_retention_sec: float = 3600.0,
                 reid_enabled: bool = True, reid_window_sec: float = 3.0,
                 reid_max_distance_px: float = 120.0):
        self.confirm_frames = max(1, int(switch_confirmation_frames))
        self.min_dwell_ms = float(minimum_dwell_sec) * 1000.0
        self.tolerance = float(boundary_tolerance_px)
        self.lost_timeout_ms = float(lost_track_timeout_sec) * 1000.0
        self.retention_ms = float(last_seen_retention_sec) * 1000.0
        self.reid_enabled = reid_enabled
        self.reid_window_ms = float(reid_window_sec) * 1000.0
        self.reid_max_dist = float(reid_max_distance_px)
        self.tracks: dict[tuple, TrackState] = {}

    # ── lifecycle ────────────────────────────────────────────────────────
    def get(self, source_id, track_id):
        return self.tracks.get((source_id, track_id))

    def get_or_create(self, source_id, track_id, ts) -> tuple[TrackState, bool]:
        key = (source_id, track_id)
        st = self.tracks.get(key)
        if st is None:
            st = TrackState(source_id, track_id, first_seen_timestamp=ts)
            self.tracks[key] = st
            return st, True
        return st, False

    def mark_seen(self, st: TrackState, ts, point, bbox, conf):
        if st.status in (OCCLUDED, LOST) and st.last_seen_timestamp is not None:
            st.recoveries += 1
        st.status = ACTIVE
        st.last_seen_timestamp = ts
        st.last_seen_source_id = st.source_id
        st.last_point = tuple(point)
        st.last_bbox = tuple(bbox)
        st.tracking_confidence = float(conf)

    def find_reid_candidate(self, source_id, point, ts, exclude: set) -> TrackState | None:
        """หา track ที่เพิ่งหายในกล้องเดียวกัน ใกล้จุดเดิม → น่าจะเป็นคนเดิมที่ได้ track_id ใหม่"""
        if not self.reid_enabled:
            return None
        best, best_d = None, math.inf
        for st in self.tracks.values():
            if st.source_id != source_id or st.key in exclude:
                continue
            # ยังเห็นอยู่ในเฟรมนี้ = คนละคน (track เดิมที่เพิ่งหายเฟรมก่อน ยังมี status ACTIVE ได้)
            if st.last_seen_timestamp is not None and st.last_seen_timestamp >= ts:
                continue
            if st.last_seen_timestamp is None or ts - st.last_seen_timestamp > self.reid_window_ms:
                continue
            if st.last_point is None:
                continue
            d = math.dist(st.last_point, point)
            if d <= self.reid_max_dist and d < best_d:
                best, best_d = st, d
        return best

    # ── hysteresis ───────────────────────────────────────────────────────
    def update_location(self, st: TrackState, observed: Location, point, ts, zone_engine):
        """คืน True ถ้าตำแหน่งที่ยืนยันแล้ว 'เปลี่ยน' ในเฟรมนี้"""
        if st.location is None:
            self._commit(st, observed, ts)
            return True

        cur = st.location
        if observed.key == cur.key:
            st.candidate, st.candidate_count, st.candidate_since = None, 0, None
            return False

        # boundary tolerance: หลุดขอบ zone ปัจจุบันนิดเดียว (และไม่ได้เข้า zone ย่อยข้างใน) → อยู่ที่เดิม
        if cur.zone_id != "unknown" and cur.zone_id not in observed.matched_zones:
            z = zone_engine.get(cur.zone_id)
            if z is not None and z.contains(point, self.tolerance):
                st.candidate, st.candidate_count, st.candidate_since = None, 0, None
                return False

        if st.candidate is not None and st.candidate.key == observed.key:
            st.candidate_count += 1
        else:
            st.candidate, st.candidate_count, st.candidate_since = observed, 1, ts

        dwell_ok = (ts - st.candidate_since) >= self.min_dwell_ms
        if st.candidate_count >= self.confirm_frames and dwell_ok:
            self._commit(st, observed, ts)
            st.zone_switches += 1
            return True
        return False

    @staticmethod
    def _commit(st: TrackState, loc: Location, ts):
        prev = st.location
        st.previous_location = prev
        st.location = loc
        if prev is None or prev.zone_id != loc.zone_id:
            st.zone_enter_timestamp = ts
        if prev is None or prev.room_id != loc.room_id:
            st.room_enter_timestamp = ts
        st.candidate, st.candidate_count, st.candidate_since = None, 0, None

    # ── tracking loss ────────────────────────────────────────────────────
    def update_missing(self, source_id, seen_keys: set, ts) -> list[TrackState]:
        """คืน track ที่เพิ่งกลายเป็น LOST ในเฟรมนี้"""
        newly_lost = []
        for st in self.tracks.values():
            if st.source_id != source_id or st.key in seen_keys or st.status == LOST:
                continue
            gap = ts - (st.last_seen_timestamp or ts)
            if gap > self.lost_timeout_ms:
                st.status = LOST
                newly_lost.append(st)
            else:
                st.status = OCCLUDED
        return newly_lost

    def expire_all(self, source_id=None) -> list[TrackState]:
        """ใช้ตอน flush(): ปิด track ที่ยังไม่ LOST ทั้งหมด"""
        out = []
        for st in self.tracks.values():
            if (source_id is None or st.source_id == source_id) and st.status != LOST:
                st.status = LOST
                out.append(st)
        return out

    def purge(self, ts):
        dead = [k for k, st in self.tracks.items()
                if st.status == LOST and st.last_seen_timestamp is not None
                and ts - st.last_seen_timestamp > self.retention_ms]
        for k in dead:
            del self.tracks[k]

    def reset(self, source_id=None):
        if source_id is None:
            self.tracks.clear()
        else:
            for k in [k for k in self.tracks if k[0] == source_id]:
                del self.tracks[k]
