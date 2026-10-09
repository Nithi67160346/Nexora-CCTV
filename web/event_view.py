"""Thai event text and recording references for human review."""
from copy import deepcopy

EVENT_NAMES = {
    'fall_detected': 'อาจเกิดการล้ม — กรุณาตรวจสอบภาพ',
    'fall_posture_review': 'ท่าทางเสี่ยง — ยังยืนยันจังหวะล้มไม่ได้',
    'high_risk_interaction': 'พบความเสี่ยงการปะทะ — กรุณาตรวจสอบภาพ',
    'violence_detected': 'LSTM พบภาพเสี่ยงทำร้าย — กรุณาตรวจสอบ',
    'location_update': 'อัปเดตตำแหน่งบุคคล',
    'identity_update': 'อัปเดตข้อมูลบุคคล',
    'last_seen_update': 'บันทึกตำแหน่งที่พบล่าสุด',
}


def event_category(event_type):
    if event_type.startswith('fall'): return 'fall'
    if event_type.startswith('wandering'): return 'wandering'
    if event_type in ('high_risk_interaction', 'violence_detected'): return 'violence'
    if event_type in ('location_update', 'identity_update', 'last_seen_update'): return 'location'
    return 'other'


def localized_event(event, source, session_id=None):
    result = deepcopy(event)
    result['category'] = event_category(result.get('event_type', ''))
    kind = result.get('event_type', '')
    if kind in EVENT_NAMES:
        result['message'] = EVENT_NAMES[kind]
        if kind=='fall_detected' and result.get('metadata',{}).get('backend')=='yolo_pose_rf_v2':
            result['message']='โมเดล V2 พบท่าล้ม — กรุณาตรวจสอบภาพ'
    elif kind.startswith('wandering'):
        zone = result.get('metadata', {}).get('zone', '')
        result['message'] = ('ออกจากเขตที่กำหนด' if 'exit' in kind else 'เข้า / ข้ามเขตที่กำหนด') + (f' {zone}' if zone else '')
    md = result.setdefault('metadata', {})
    # Keep the source that produced the event. A room may later select another clip.
    md.setdefault('playback_source', source)
    md.setdefault('recording_session', session_id)
    return result
