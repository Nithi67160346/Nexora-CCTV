"""
Last-Seen Manager — จำตำแหน่งสุดท้ายที่เห็น แม้ track จะหายไปแล้ว

เก็บ 2 ระดับ
  ต่อ track    : (source_id, track_id) → last seen
  ต่อ resident : resident_id → last seen ล่าสุดจากทุกกล้อง (รองรับ multi-camera)

ทุก record มี source_id เสมอ เพื่อรองรับหลายกล้องในอนาคต
"""

from __future__ import annotations

from collections import deque

from .models import Location


def _record(ts, source_id, loc: Location | None, track_id=None, resident_id=None):
    loc = loc or Location()
    return {
        "resident_id": resident_id,
        "track_id": track_id,
        "timestamp_ms": int(ts),
        "source_id": source_id,
        "site_id": loc.site_id,
        "floor_id": loc.floor_id,
        "room_id": loc.room_id,
        "bed_id": loc.bed_id,
        "zone_id": loc.zone_id,
    }


class LastSeenManager:
    def __init__(self, retention_sec: float = 3600.0, history_len: int = 200):
        self.retention_ms = float(retention_sec) * 1000.0
        self._by_track: dict[tuple, dict] = {}
        self._by_resident: dict[str, dict] = {}
        self._history: dict[str, deque] = {}
        self._history_len = history_len

    def update(self, source_id, track_id, ts, location: Location | None, resident_id=None):
        rec = _record(ts, source_id, location, track_id, resident_id)
        self._by_track[(source_id, track_id)] = rec
        if resident_id is not None:
            prev = self._by_resident.get(resident_id)
            if prev is None or rec["timestamp_ms"] >= prev["timestamp_ms"] \
                    or prev["source_id"] != source_id:
                self._by_resident[resident_id] = rec
            hist = self._history.setdefault(resident_id, deque(maxlen=self._history_len))
            if not hist or (hist[-1]["source_id"], hist[-1]["room_id"], hist[-1]["zone_id"]) != \
                    (rec["source_id"], rec["room_id"], rec["zone_id"]):
                hist.append(rec)
        return rec

    def bind_resident(self, source_id, track_id, resident_id):
        """เมื่อเพิ่งรู้ตัวตน → ย้อนผูก last seen ของ track นั้นเข้ากับ resident"""
        rec = self._by_track.get((source_id, track_id))
        if rec is not None and resident_id is not None:
            rec = dict(rec, resident_id=resident_id)
            self._by_track[(source_id, track_id)] = rec
            self._by_resident[resident_id] = rec

    def get_track(self, source_id, track_id):
        r = self._by_track.get((source_id, track_id))
        return dict(r) if r else None

    def get_resident(self, resident_id):
        r = self._by_resident.get(resident_id)
        return dict(r) if r else None

    def get_history(self, resident_id):
        return [dict(r) for r in self._history.get(resident_id, [])]

    def purge(self, now_ms):
        cutoff = now_ms - self.retention_ms
        self._by_track = {k: v for k, v in self._by_track.items() if v["timestamp_ms"] >= cutoff}
        self._by_resident = {k: v for k, v in self._by_resident.items() if v["timestamp_ms"] >= cutoff}

    def reset(self, source_id=None):
        if source_id is None:
            self._by_track.clear()
            self._by_resident.clear()
            self._history.clear()
        else:
            self._by_track = {k: v for k, v in self._by_track.items() if k[0] != source_id}
