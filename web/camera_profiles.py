"""Session-only camera configuration. Users may explicitly export/import JSON."""
from copy import deepcopy
import math
from uuid import uuid4

from integration.feature_policy import apply_web_feature_policy
from Location.location.zone_engine import ZONE_TYPES


def clean_text(value, field, maximum=100):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError(f'{field}: กรุณาระบุข้อความ 1–{maximum} ตัวอักษร')
    return value.strip()


def polygon(points):
    if not isinstance(points, list) or not 3 <= len(points) <= 100:
        raise ValueError('โซนต้องมี 3–100 จุด')
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError('จุดโซนต้องเป็น [x, y]')
        pair = [float(v) for v in point]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in pair):
            raise ValueError('จุดโซนต้องอยู่ในช่วง 0–1 และไม่เป็น NaN')
        result.append(pair)
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(result, result[1:]+result[:1]))) / 2
    if len(set(map(tuple, result))) < 3 or area < .00001:
        raise ValueError('โซนต้องมีพื้นที่ ไม่ใช่เส้นตรงหรือจุดซ้ำ')
    return result


def profile_config(config, source_id):
    candidate = apply_web_feature_policy(config)
    location = candidate.get('features', {}).get('location', {}).get('config', {})
    # Identity registry is global and belongs to FaceService, not camera exports.
    for key in ('identity', 'residents', 'staff'):
        location.pop(key, None)
    for zone in location.get('zones', {}).values():
        if isinstance(zone, dict):
            zone['source_ids'] = [source_id]
    return candidate


def apply_zones(config, payload, width, height, source_id):
    candidate = deepcopy(config)
    if 'wandering_zones' in payload:
        zones = payload['wandering_zones']
        if not isinstance(zones, dict) or len(zones) > 64:
            raise ValueError('เขตพลัดหลงต้องเป็นรายการไม่เกิน 64 โซน')
        candidate['features'].setdefault('wandering', {})['zones_relative'] = {
            clean_text(name, 'ชื่อเขต'): polygon(points) for name, points in zones.items()}
    if 'location_zones' in payload:
        from Location.location.zone_engine import ZONE_TYPES
        zones = payload['location_zones']
        if not isinstance(zones, dict) or len(zones) > 64:
            raise ValueError('เตียง/พื้นที่ต้องเป็นรายการไม่เกิน 64 โซน')
        clean = {}
        for name, value in zones.items():
            name = clean_text(name, 'ชื่อโซน')
            value = value if isinstance(value, dict) else dict(points=value, type='bed')
            points = value.get('points_relative', value.get('points'))
            if not isinstance(points, list):
                raise ValueError('โซนต้องมีจุดขอบเขต')
            # Support the original API's pixel coordinates as well as normalized UI coordinates.
            if any(any(float(v) > 1 for v in point) for point in points):
                points = [[float(p[0])/width, float(p[1])/height] for p in points]
            points = polygon(points)
            kind = value.get('type', 'bed')
            if kind not in ZONE_TYPES:
                raise ValueError('ประเภทโซนไม่รองรับ')
            clean[name] = dict(type=kind, points_relative=points,
                               points=[[round(x*width), round(y*height)] for x, y in points],
                               source_ids=[source_id], room_id=clean_text(value.get('room_id') or source_id, 'ชื่อห้อง'))
            if kind == 'bed':
                clean[name]['bed_id'] = clean_text(value.get('bed_id') or name, 'ชื่อเตียง')
        candidate['features'].setdefault('location', {}).setdefault('config', {})['zones'] = clean
    return candidate


class CameraProfiles:
    def __init__(self, config):
        self.active_id = 'cam_01'
        self.profiles = {'cam_01': dict(id='cam_01', name='ห้องทดลอง A', source='',
                                      config=profile_config(config, 'cam_01'))}

    def snapshot(self):
        return dict(active_id=self.active_id, profiles=[deepcopy(p) for p in self.profiles.values()],
                    storage='session', simultaneous_streams=4)

    def save_active(self, config, source=None):
        profile = self.profiles[self.active_id]
        profile['config'] = profile_config(config, self.active_id)
        if source is not None:
            profile['source'] = source

    def create(self, name, source=''):
        if len(self.profiles) >= 32:
            raise ValueError('รองรับโปรไฟล์ทดลองสูงสุด 32 รายการ')
        name = clean_text(name, 'ชื่อกล้อง/ห้อง')
        source = clean_text(source, 'แหล่งภาพ', 2048) if source else ''
        cid = 'cam_' + uuid4().hex[:8]
        cfg = deepcopy(self.profiles[self.active_id]['config'])
        cfg['features'].get('wandering', {})['zones_relative'] = {}
        cfg['features'].get('location', {}).setdefault('config', {})['zones'] = {}
        self.profiles[cid] = dict(id=cid, name=name, source=source, config=profile_config(cfg, cid))
        return deepcopy(self.profiles[cid])

    def imported(self, payload):
        # Return a validated replacement without mutating the live registry.
        if payload.get('version') != 1 or not isinstance(payload.get('profiles'), list) or not 1 <= len(payload['profiles']) <= 32:
            raise ValueError('ไฟล์โปรไฟล์ไม่ถูกต้อง')
        result = {}
        for raw in payload['profiles']:
            cid = clean_text(raw['id'], 'รหัสกล้อง', 64)
            if not all(c.isalnum() or c in '_-' for c in cid) or cid in result:
                raise ValueError('รหัสกล้องซ้ำหรือไม่ถูกต้อง')
            cfg = profile_config(raw['config'], cid)
            if not isinstance(cfg.get('features'), dict) or set(cfg['features']) != {'fall', 'violence', 'location', 'wandering'}:
                raise ValueError('ฟีเจอร์ในโปรไฟล์ไม่ถูกต้อง')
            for feature in cfg['features'].values():
                if not isinstance(feature, dict) or not isinstance(feature.get('enabled'), bool):
                    raise ValueError('สถานะฟีเจอร์ต้องเป็น true/false')
            for points in cfg['features'].get('wandering', {}).get('zones_relative', {}).values():
                polygon(points)
            for zone in cfg['features'].get('location', {}).get('config', {}).get('zones', {}).values():
                if not isinstance(zone, dict) or 'points_relative' not in zone:
                    raise ValueError('โซนที่นำเข้าต้องมี points_relative')
                if zone.get('type', 'area') not in ZONE_TYPES:
                    raise ValueError('ชนิดโซนที่นำเข้าไม่ถูกต้อง')
                polygon(zone['points_relative'])
            result[cid] = dict(id=cid, name=clean_text(raw['name'], 'ชื่อกล้อง/ห้อง'),
                               source=clean_text(raw['source'], 'แหล่งภาพ', 2048) if raw.get('source') else '', config=cfg)
        active = payload.get('active_id', next(iter(result)))
        if active not in result:
            raise ValueError('ไม่พบโปรไฟล์ที่เลือก')
        return result, active
