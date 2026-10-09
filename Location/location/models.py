"""
Data model กลางของ Location Feature (ไม่มี logic)

Location hierarchy:  Site → Floor → Room → Zone → Bed
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# บทบาทของบุคคล (BMC: "ระบุตำแหน่งผู้สูงอายุและผู้ดูแล")
PERSON_ROLES = ("resident", "caregiver", "visitor", "unknown")

IDENTITY_SOURCES = (
    "track_only", "face_recognition", "rfid", "ble",
    "wristband", "manual", "external_service",
)


@dataclass(frozen=True)
class Location:
    site_id: str | None = None
    floor_id: str | None = None
    room_id: str | None = None
    zone_id: str = "unknown"
    zone_type: str = "unknown"
    bed_id: str | None = None
    matched_zones: tuple = ()          # zone_id ทั้งหมดที่จุดตกอยู่ (ซ้อนกันได้)

    @property
    def key(self):
        """ใช้เทียบว่า 'ตำแหน่งเปลี่ยน' หรือไม่"""
        return (self.room_id, self.zone_id, self.bed_id)

    def to_dict(self):
        d = asdict(self)
        d["matched_zones"] = list(self.matched_zones)
        return d


UNKNOWN_LOCATION = Location()


@dataclass
class Identity:
    identity_id: str | None
    display_name: str
    source: str
    confidence: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in IDENTITY_SOURCES:
            raise ValueError(f"identity source '{self.source}' ไม่รองรับ: {IDENTITY_SOURCES}")

    @property
    def is_known(self) -> bool:
        return self.identity_id is not None

    def to_dict(self):
        return {
            "identity_id": self.identity_id,
            "display_name": self.display_name,
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
        }


def unknown_identity(source: str = "track_only", confidence: float = 0.0) -> Identity:
    return Identity(None, "unknown", source, confidence)
