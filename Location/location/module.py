"""
LocationModule — Location / Bed / Zone / Last-Seen + Optional Face Recognition

Integration contract (ห้ามเปลี่ยน):
    location = LocationModule()
    location.setup(config)
    events += location.process(frame, context)     # ทุกเฟรม
    events += location.flush()                     # ตอนจบ

- ไม่เปิดกล้อง / ไม่โหลด person detector / ไม่สร้าง tracker — ใช้ persons + track_id จาก SharedContext
- frame ใช้เฉพาะ Face layer (ถ้าเปิด) ; ส่ง frame=None ได้ถ้าใช้ track-only
- event_type: location_update (หลัก), identity_update, last_seen_update
"""

from __future__ import annotations

import copy
import logging
import os
import time
import uuid
from collections import deque

from .bed_manager import BedManager
from .identity_resolver import IdentityResolver, identity_from_external
from .last_seen_manager import LastSeenManager
from .models import PERSON_ROLES, Identity, Location
from .room_manager import RoomManager
from .state_manager import LOST, StateManager, TrackState
from .zone_engine import ZoneEngine, reference_point

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(HERE, "config.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    import json
    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f)


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _is_person(p: dict) -> bool:
    name = p.get("class_name")
    if name is not None:
        return name == "person"
    return p.get("class_id", 0) == 0


class LocationModule:
    name = "location"

    def __init__(self):
        self.cfg: dict = {}
        self._ready = False

    # ======================================================================
    # Contract
    # ======================================================================
    def setup(self, config: dict):
        """
        config = dict (โครงเดียวกับ config.yaml)
        ถ้ามีคีย์ "config_path" จะโหลดไฟล์นั้นก่อน แล้วเอาค่าอื่นใน dict ทับ
        """
        config = dict(config or {})
        path = config.pop("config_path", None)
        cfg = _deep_merge(load_config(path), config) if path else config
        self.cfg = cfg

        loc = cfg.get("location", {}) or {}
        self.point_mode = loc.get("point_mode", "bottom_center")
        reference_point((0, 0, 1, 1), self.point_mode)         # validate
        self.emit_on_first_seen = loc.get("emit_on_first_seen", True)
        # กรอง detection ซ้อน (เช่น กรอบครึ่งตัวบนซ้อนในกรอบเต็มตัวของคนเดียวกัน) และกรอบเล็กผิดปกติ
        self.dedupe_containment = float(loc.get("dedupe_containment", 0.85))   # 0 = ปิด
        self.min_box_height_px = float(loc.get("min_box_height_px", 0))       # 0 = ปิด
        self.dedupe_window_ms = float(loc.get("dedupe_window_sec", 0.5)) * 1000.0
        self._held: set = set()

        self.zones = ZoneEngine.from_config(cfg.get("zones"))
        self.rooms = RoomManager(cfg.get("sources"), loc.get("site_id"), loc.get("floor_id"))
        self.beds = BedManager(cfg.get("residents"))
        self.residents_cfg = cfg.get("residents") or {}
        self.staff_cfg = cfg.get("staff") or {}          # ผู้ดูแล / พยาบาล / พนักงาน
        reid = loc.get("reid", {}) or {}
        self.state = StateManager(
            switch_confirmation_frames=loc.get("switch_confirmation_frames", 5),
            minimum_dwell_sec=loc.get("minimum_dwell_sec", 0.0),
            boundary_tolerance_px=loc.get("boundary_tolerance_px", 15),
            lost_track_timeout_sec=loc.get("lost_track_timeout_sec", 2.0),
            last_seen_retention_sec=loc.get("last_seen_retention_sec", 3600),
            reid_enabled=reid.get("enabled", True),
            reid_window_sec=reid.get("window_sec", 3.0),
            reid_max_distance_px=reid.get("max_distance_px", 120),
        )
        self.last_seen = LastSeenManager(loc.get("last_seen_retention_sec", 3600))

        ident_cfg = cfg.get("identity", {}) or {}
        self.identity_enabled = ident_cfg.get("enabled", True)
        self.resolver = IdentityResolver(cfg.get("identity_priority"))

        self.face = None
        if self.identity_enabled:
            from .face import build_face_pipeline         # import เฉพาะเมื่อใช้
            self.face = build_face_pipeline(cfg, {**self.staff_cfg, **self.residents_cfg})

        self._latency_ms = deque(maxlen=10000)
        self._frames = 0
        self._skipped_no_track = 0
        self._suppressed_dup = 0
        self._suppressed_small = 0
        self._last_ts = 0
        self._ready = True
        return self

    def reset(self, source_id: str):
        self._require()
        self.state.reset(source_id)
        self.resolver.reset(source_id)
        self.last_seen.reset(source_id)
        if self.face:
            self.face.reset(source_id)

    def process(self, frame, context: dict) -> list[dict]:
        self._require()
        t0 = time.perf_counter()
        events: list[dict] = []

        source_id = context.get("source_id", "unknown_source")
        ts = self._timestamp(context)
        self._last_ts = max(self._last_ts, ts)
        external = context.get("external_identities") or {}

        persons = self._filter_persons([p for p in (context.get("persons") or []) if _is_person(p)],
                                       source_id, ts)
        seen: set = set(self._held)          # track ที่ถูกยืนยันว่ายังอยู่ผ่านกรอบครึ่งตัว
        for key in self._held:
            st = self.state.get(*key)
            if st is not None:
                st.status = "active"
        # track ทุกตัวที่อยู่ในเฟรมนี้ — ห้ามถูกใช้เป็นต้นทาง re-id ของคนอื่น
        present = {(source_id, p.get("track_id")) for p in persons if p.get("track_id") is not None}

        for p in persons:
            tid = p.get("track_id")
            if tid is None:                     # ไม่มี track → ติดตามสถานะไม่ได้
                self._skipped_no_track += 1
                continue
            bbox = p.get("bbox_xyxy")
            conf = float(p.get("confidence", 0.0))
            point = reference_point(bbox, self.point_mode)

            st, is_new = self.state.get_or_create(source_id, tid, ts)
            if is_new:
                self._try_reid(st, point, ts, seen | present)
            self.state.mark_seen(st, ts, point, bbox, conf)
            seen.add(st.key)

            # ── Identity (แยกจาก Location) ─────────────────────────────
            face_obs = self._identity_step(frame, st, bbox, external.get(tid) or external.get(str(tid)))
            ident = self.resolver.resolve(st.key) if self.identity_enabled \
                else Identity(None, "unknown", "track_only")
            prev_ident = st.identity
            st.identity = ident
            if ident.is_known and ident.identity_id != prev_ident.identity_id:
                self.last_seen.bind_resident(source_id, tid, ident.identity_id)
                events.append(self._identity_event(st, ts, prev_ident))

            # ── Location ────────────────────────────────────────────────
            observed = self._observe_location(point, source_id)
            changed = self.state.update_location(st, observed, point, ts, self.zones)
            self.last_seen.update(source_id, tid, ts, st.location, ident.identity_id)
            if changed and (st.previous_location is not None or self.emit_on_first_seen):
                events.append(self._location_event(st, ts, point, face_obs))

        # ── Tracking loss ──────────────────────────────────────────────
        for st in self.state.update_missing(source_id, seen, ts):
            events.append(self._last_seen_event(st, "Track lost; last-seen location retained"))
        self.state.purge(ts)
        self.last_seen.purge(self._last_ts)

        self._frames += 1
        self._latency_ms.append((time.perf_counter() - t0) * 1000.0)
        return events

    def flush(self) -> list[dict]:
        self._require()
        return [self._last_seen_event(st, "Stream ended; last-seen location retained")
                for st in self.state.expire_all()]

    # ======================================================================
    # Last-Seen / Query API
    # ======================================================================
    def get_current_location(self, track_id, source_id=None):
        st = self._find(track_id, source_id)
        if st is None or st.location is None:
            return None
        d = st.location.to_dict()
        d.update(source_id=st.source_id, track_id=st.track_id, status=st.status,
                 zone_enter_timestamp_ms=st.zone_enter_timestamp,
                 room_enter_timestamp_ms=st.room_enter_timestamp)
        return d

    def get_last_seen(self, track_id, source_id=None):
        st = self._find(track_id, source_id)
        return self.last_seen.get_track(st.source_id, st.track_id) if st else \
            (self.last_seen.get_track(source_id, track_id) if source_id else None)

    def get_identity(self, track_id, source_id=None):
        st = self._find(track_id, source_id)
        return st.identity.to_dict() if st else None

    def get_resident_last_seen(self, resident_id):
        return self.last_seen.get_resident(resident_id)

    def get_resident_history(self, resident_id):
        return self.last_seen.get_history(resident_id)

    def get_snapshot(self, source_id=None, role=None) -> list[dict]:
        """Location State Snapshot สำหรับระบบกลาง (1 แถวต่อ track ที่ยังไม่ถูก purge)
        role="resident" / "caregiver" / ... = กรองเฉพาะบทบาทนั้น"""
        out = []
        for st in self.state.tracks.values():
            if source_id is not None and st.source_id != source_id:
                continue
            if role is not None and self.role_of(st.identity) != role:
                continue
            loc = st.location or Location()
            out.append({
                "track_id": st.track_id,
                "temporary_id": f"{st.source_id}:track_{st.track_id}",
                "resident_id": st.identity.identity_id,
                "identity_name": st.identity.display_name,
                "identity_source": st.identity.source,
                "identity_confidence": round(st.identity.confidence, 4),
                "person_role": self.role_of(st.identity),
                "source_id": st.source_id,
                "site_id": loc.site_id,
                "floor_id": loc.floor_id,
                "room_id": loc.room_id,
                "bed_id": loc.bed_id,
                "zone_id": loc.zone_id,
                "assigned_bed": self.beds.assigned_bed(st.identity.identity_id),
                "status": st.status,
                "last_seen_timestamp_ms": st.last_seen_timestamp,
            })
        return out

    def assign_identity(self, source_id, track_id, identity_id, source="manual",
                        confidence=1.0, display_name=None, role=None):
        """ช่องทางให้ RFID / BLE / Wristband / Manual / Service ภายนอก ผูกตัวตนกับ track
        role (optional): "resident" | "caregiver" | "visitor" — ถ้าไม่ระบุ ดูจาก config residents/staff"""
        self._require()
        if role is not None and role not in PERSON_ROLES:
            raise ValueError(f"role ต้องเป็นหนึ่งใน {PERSON_ROLES}")
        name = display_name or self._display_name(identity_id) or identity_id
        extra = {"role": role} if role else {}
        self.resolver.submit((source_id, track_id), Identity(identity_id, name, source, confidence, extra))

    # ── บทบาท: ผู้สูงอายุ / ผู้ดูแล ─────────────────────────────────────────
    def role_of(self, identity: Identity) -> str:
        """
        ลำดับการตัดสินบทบาท (ไม่เดาจากท่าทาง/ตำแหน่ง — ไม่มีหลักฐานพอ):
          1. แหล่งตัวตนระบุ role มาเอง (เช่น ป้าย RFID พนักงาน)
          2. identity_id อยู่ใน config.staff → role ของรายการนั้น (ค่าเริ่มต้น caregiver)
          3. identity_id อยู่ใน config.residents → resident
          4. ไม่รู้ตัวตน → unknown
        """
        r = (identity.extra or {}).get("role")
        if r in PERSON_ROLES:
            return r
        iid = identity.identity_id
        if iid is None:
            return "unknown"
        if iid in self.staff_cfg:
            r = (self.staff_cfg[iid] or {}).get("role", "caregiver")
            return r if r in PERSON_ROLES else "caregiver"
        if iid in self.residents_cfg:
            return "resident"
        return "unknown"

    def get_people_by_role(self, role: str, source_id=None) -> list[dict]:
        """ตำแหน่งปัจจุบันของทุกคนในบทบาทนั้น เช่น get_people_by_role("caregiver")"""
        return [s for s in self.get_snapshot(source_id, role=role) if s["status"] != "lost"]

    def _display_name(self, identity_id):
        return ((self.staff_cfg.get(identity_id) or {}).get("display_name")
                or self.beds.display_name(identity_id))

    def get_metrics(self) -> dict:
        lat = sorted(self._latency_ms)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None
        return {
            "frames": self._frames,
            "avg_latency_ms": round(sum(lat) / len(lat), 3) if lat else None,
            "p95_latency_ms": round(p95, 3) if p95 is not None else None,
            "zone_switches": sum(s.zone_switches for s in self.state.tracks.values()),
            "tracking_loss_recoveries": sum(s.recoveries for s in self.state.tracks.values()),
            "persons_skipped_no_track_id": self._skipped_no_track,
            "persons_suppressed_duplicate": self._suppressed_dup,
            "persons_suppressed_small": self._suppressed_small,
            "face_recognition_active": bool(self.face and self.face.recognition_enabled),
        }

    # ======================================================================
    # internals
    # ======================================================================
    def _require(self):
        if not self._ready:
            raise RuntimeError("เรียก setup(config) ก่อน")

    def _filter_persons(self, persons, source_id=None, ts=None):
        """
        Adapter ภายใน feature (ไม่แก้ shared contract):
        - ตัดกรอบที่เตี้ยกว่า min_box_height_px (false positive เล็ก ๆ)
        - ตัดกรอบที่ถูกกรอบใหญ่กว่าครอบไว้ >= dedupe_containment ของพื้นที่ตัวเอง
          (detector ตีกรอบครึ่งตัว + เต็มตัวให้คนเดียวกัน → ได้ 2 track ซ้อนกัน)
          ตรวจทั้งกรอบในเฟรมเดียวกัน และกรอบของ track อื่นที่เพิ่งเห็นภายใน dedupe_window_sec
          (detector สลับให้กรอบเต็มตัว/ครึ่งตัวคนละเฟรม → track_id กระพริบสลับกัน)
        ข้อจำกัด: คนสองคนที่ซ้อนกันเกือบทั้งตัว (เช่น ผู้ดูแลยืนบังผู้ป่วย) อาจถูกนับเป็นคนเดียว
        """
        self._held = set()
        out = []
        for p in persons:
            x1, y1, x2, y2 = p["bbox_xyxy"]
            if self.min_box_height_px and (y2 - y1) < self.min_box_height_px:
                self._suppressed_small += 1
                continue
            out.append(p)
        if self.dedupe_containment <= 0 or not out:
            return out

        def area(b):
            return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

        def contained(b, K):
            iw = max(0.0, min(b[2], K[2]) - max(b[0], K[0]))
            ih = max(0.0, min(b[3], K[3]) - max(b[1], K[1]))
            a = area(b)
            return a > 0 and iw * ih / a >= self.dedupe_containment

        # 1) กรอบซ้อนในเฟรมเดียวกัน
        out.sort(key=lambda p: -area(p["bbox_xyxy"]))
        kept = []
        for p in out:
            if any(contained(p["bbox_xyxy"], k["bbox_xyxy"]) for k in kept):
                self._suppressed_dup += 1
            else:
                kept.append(p)

        # 2) กรอบเล็กที่อยู่ในกรอบใหญ่ของ track อื่นที่เพิ่งเห็น (ไม่ได้อยู่ในเฟรมนี้)
        if self.dedupe_window_ms <= 0 or ts is None:
            return kept
        present = {p.get("track_id") for p in kept}
        recent = [st for st in self.state.tracks.values()
                  if st.source_id == source_id and st.track_id not in present
                  and st.last_bbox is not None and st.last_seen_timestamp is not None
                  and 0 < ts - st.last_seen_timestamp <= self.dedupe_window_ms]
        final = []
        for p in kept:
            b = p["bbox_xyxy"]
            own = self.state.get(source_id, p.get("track_id"))
            established = own is not None and own.last_seen_timestamp is not None \
                and (own.last_seen_timestamp - own.first_seen_timestamp) > self.dedupe_window_ms
            host = None
            if not established:
                for st in recent:
                    if area(b) <= 0.7 * area(st.last_bbox) and contained(b, st.last_bbox):
                        host = st
                        break
            if host is not None:
                self._suppressed_dup += 1
                host.last_seen_timestamp = ts          # คนเดิมยังอยู่ (detector แค่ให้กรอบครึ่งตัว)
                self._held.add(host.key)
            else:
                final.append(p)
        return final

    @staticmethod
    def _timestamp(ctx) -> int:
        if ctx.get("timestamp_ms") is not None:
            return int(ctx["timestamp_ms"])
        fps = ctx.get("fps") or 30.0
        return int(ctx.get("frame_id", 0) * 1000.0 / fps)

    def _find(self, track_id, source_id=None) -> TrackState | None:
        if source_id is not None:
            return self.state.get(source_id, track_id)
        cands = [s for s in self.state.tracks.values() if s.track_id == track_id]
        return max(cands, key=lambda s: s.last_seen_timestamp or 0) if cands else None

    def _try_reid(self, st: TrackState, point, ts, seen):
        old = self.state.find_reid_candidate(st.source_id, point, ts, seen | {st.key})
        if old is None:
            return
        # คนเดิมได้ track_id ใหม่ → รับตัวตน+ตำแหน่งเดิม ไม่สร้าง resident ใหม่
        st.location, st.previous_location = old.location, old.previous_location
        st.zone_enter_timestamp, st.room_enter_timestamp = old.zone_enter_timestamp, old.room_enter_timestamp
        st.identity = old.identity
        st.reid_from = old.key
        st.recoveries = old.recoveries + 1
        self.resolver.transfer(old.key, st.key)
        if self.face:
            self.face.transfer(old.key, st.key)
        self.resolver.forget(old.key)
        del self.state.tracks[old.key]

    def _identity_step(self, frame, st, bbox, external):
        if not self.identity_enabled:
            return None
        if external:
            try:
                self.resolver.submit(st.key, identity_from_external(external))
            except ValueError as e:
                log.warning("external identity ถูกข้าม: %s", e)
        if self.face is None or frame is None:
            return None
        try:
            obs = self.face.process(frame, bbox, st.key)
        except Exception as e:                   # face ล่มต้องไม่ทำให้ location ล่ม
            log.warning("face layer error: %s", e)
            return None
        if obs is not None and obs.identity is not None:
            self.resolver.submit(st.key, obs.identity)
        if obs is not None:
            st.last_face = obs
        return st.last_face

    def _observe_location(self, point, source_id) -> Location:
        zones = self.zones.match(point, source_id)
        bed = self.beds.detected_bed(zones)
        return self.rooms.build_location(zones, source_id, bed)

    def _location_confidence(self, st: TrackState) -> float:
        loc = st.location
        if loc is None or loc.zone_id == "unknown":
            return 0.5
        z = self.zones.get(loc.zone_id)
        tol = max(self.state.tolerance, 1.0)
        d = z.signed_distance(st.last_point) if z else 0.0
        return round(max(0.5, min(1.0, 0.75 + 0.25 * d / tol)), 4)

    # ── events ──────────────────────────────────────────────────────────
    @staticmethod
    def _event(event_type, source_id, track_ids, ts_start, ts_end, confidence, message, metadata):
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "source_id": source_id,
            "track_ids": list(track_ids),
            "start_timestamp_ms": int(ts_start),
            "end_timestamp_ms": int(ts_end),
            "confidence": round(float(confidence), 4),
            "severity": "info",
            "message": message,
            "status": "active",
            "metadata": metadata,
        }

    def _identity_meta(self, st: TrackState):
        i = st.identity
        return {
            "resident_id": i.identity_id,
            "identity_name": i.display_name,
            "identity_source": i.source,
            "identity_confidence": round(i.confidence, 4),
            "person_role": self.role_of(i),
            "temporary_id": f"{st.source_id}:track_{st.track_id}",
        }

    def _location_event(self, st: TrackState, ts, point, face_obs):
        cur, prev = st.location, st.previous_location
        bed = self.beds.bed_info([z for z in (self.zones.get(zid) for zid in cur.matched_zones) if z],
                                 st.identity.identity_id)
        md = self._identity_meta(st)
        md.update({
            "site_id": cur.site_id,
            "floor_id": cur.floor_id,
            "previous_room": prev.room_id if prev else None,
            "current_room": cur.room_id,
            "previous_bed": prev.bed_id if prev else None,
            "current_bed": cur.bed_id,
            "previous_zone": prev.zone_id if prev else None,
            "current_zone": cur.zone_id,
            "zone_type": cur.zone_type,
            "matched_zones": list(cur.matched_zones),
            "assigned_bed": bed["assigned_bed"],
            "current_detected_bed": bed["current_detected_bed"],
            "bed_matches_assignment": bed["bed_matches_assignment"],
            "location_confidence": self._location_confidence(st),
            "point_mode": self.point_mode,
            "ref_point": [round(point[0], 1), round(point[1], 1)],
            "zone_enter_timestamp_ms": st.zone_enter_timestamp,
            "room_enter_timestamp_ms": st.room_enter_timestamp,
            "last_seen_timestamp_ms": st.last_seen_timestamp,
            "last_seen_location": {"room_id": cur.room_id, "bed_id": cur.bed_id, "zone_id": cur.zone_id},
            "reid_from_track": st.reid_from[1] if st.reid_from else None,
        })
        if face_obs is not None:
            md["face"] = face_obs.to_dict()
        return self._event("location_update", st.source_id, [st.track_id], ts, ts,
                           st.tracking_confidence, "Location updated", md)

    def _identity_event(self, st: TrackState, ts, previous: Identity):
        md = {
            "resident_id": st.identity.identity_id,
            "identity_name": st.identity.display_name,
            "identity_source": st.identity.source,
            "person_role": self.role_of(st.identity),
            "previous_resident_id": previous.identity_id,
            "assigned_bed": self.beds.assigned_bed(st.identity.identity_id),
        }
        return self._event("identity_update", st.source_id, [st.track_id], ts, ts,
                           st.identity.confidence, "Identity associated with active track", md)

    def _last_seen_event(self, st: TrackState, message):
        loc = st.location or Location()
        md = self._identity_meta(st)
        md.update({
            "track_status": LOST,
            "last_seen": {
                "timestamp_ms": st.last_seen_timestamp,
                "source_id": st.source_id,
                "site_id": loc.site_id,
                "floor_id": loc.floor_id,
                "room_id": loc.room_id,
                "bed_id": loc.bed_id,
                "zone_id": loc.zone_id,
            },
        })
        ts = st.last_seen_timestamp or 0
        return self._event("last_seen_update", st.source_id, [st.track_id],
                           st.first_seen_timestamp, ts, st.tracking_confidence, message, md)
