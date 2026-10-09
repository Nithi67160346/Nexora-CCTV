"""
Identity Resolver — รวมตัวตนจากหลายแหล่ง แล้วเลือกตาม priority ใน config

Location Engine ไม่รู้ว่าตัวตนมาจากไหน รู้แค่ Identity(id, name, source, confidence)

แหล่งที่รองรับ: track_only, face_recognition, rfid, ble, wristband, manual, external_service
ลำดับเริ่มต้น (ปรับได้ใน config.identity_priority):
    wristband > rfid > face_recognition > track_only

ไม่มีแหล่งไหนรู้จักตัวตน → identity_id = None, name = "unknown"
  - ถ้าเห็นหน้าแต่จับคู่ไม่ได้ → source = "face_recognition"
  - ถ้าไม่มีข้อมูลเลย          → source = "track_only" (ใช้ Track ID เป็นตัวตนชั่วคราว)

Provider ภายนอก (RFID/BLE/Wristband/Service) ส่งเข้ามาได้ 2 ทาง โดยไม่แก้ Shared Contract
  1. เรียก LocationModule.assign_identity(...) จากระบบกลาง
  2. ใส่คีย์ optional ใน context: context["external_identities"] = {track_id: {...}}
"""

from __future__ import annotations

from .models import IDENTITY_SOURCES, Identity, unknown_identity

DEFAULT_PRIORITY = ["wristband", "rfid", "ble", "manual", "external_service",
                    "face_recognition", "track_only"]


class IdentityResolver:
    def __init__(self, priority: list[str] | None = None):
        priority = list(priority or DEFAULT_PRIORITY)
        bad = [p for p in priority if p not in IDENTITY_SOURCES]
        if bad:
            raise ValueError(f"identity_priority มีค่าที่ไม่รองรับ: {bad}")
        # แหล่งที่ไม่ได้ระบุใน priority → ต่อท้าย (ต่ำสุด) แต่ track_only ต่ำสุดเสมอ
        for s in IDENTITY_SOURCES:
            if s not in priority and s != "track_only":
                priority.append(s)
        if "track_only" in priority:
            priority.remove("track_only")
        priority.append("track_only")
        self.priority = priority
        self._rank = {s: i for i, s in enumerate(priority)}
        self._claims: dict[tuple, dict[str, Identity]] = {}

    def submit(self, track_key, identity: Identity | None):
        """แหล่งใดแหล่งหนึ่งรายงานตัวตนของ track นี้ (ค่าล่าสุดของแหล่งนั้นแทนค่าเดิม)"""
        if identity is None:
            return
        claims = self._claims.setdefault(track_key, {})
        old = claims.get(identity.source)
        # unknown จากแหล่งเดิม ไม่ลบตัวตนที่แหล่งนั้นเคยยืนยันแล้ว
        if old is not None and old.is_known and not identity.is_known:
            return
        claims[identity.source] = identity

    def revoke(self, track_key, source: str):
        self._claims.get(track_key, {}).pop(source, None)

    def resolve(self, track_key) -> Identity:
        claims = self._claims.get(track_key, {})
        known = [c for c in claims.values() if c.is_known]
        if known:
            return min(known, key=lambda c: (self._rank.get(c.source, 99), -c.confidence))
        if "face_recognition" in claims:
            return unknown_identity("face_recognition", claims["face_recognition"].confidence)
        return unknown_identity("track_only")

    def transfer(self, old_key, new_key):
        """re-identification: track_id ใหม่รับตัวตนจาก track เดิม"""
        if old_key in self._claims:
            merged = dict(self._claims[old_key])
            merged.update(self._claims.get(new_key, {}))
            self._claims[new_key] = merged

    def forget(self, track_key):
        self._claims.pop(track_key, None)

    def reset(self, source_id=None):
        if source_id is None:
            self._claims.clear()
        else:
            self._claims = {k: v for k, v in self._claims.items() if k[0] != source_id}


def identity_from_external(d: dict, default_source="external_service") -> Identity:
    """แปลง dict จากระบบภายนอก → Identity (ใช้กับ context['external_identities'])"""
    src = d.get("source", default_source)
    iid = d.get("identity_id") or d.get("resident_id")
    extra = {"role": d["role"]} if d.get("role") else {}
    return Identity(
        identity_id=iid,
        display_name=d.get("display_name") or (iid if iid else "unknown"),
        source=src,
        confidence=float(d.get("confidence", 1.0)),
        extra=extra,
    )
