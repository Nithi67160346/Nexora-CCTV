"""Session-only, event-level QA for user-labelled recorded clips."""
from copy import deepcopy
import math
import time
from uuid import uuid4


EVENT_FEATURES = {'fall_detected': 'fall', 'fall_posture_review': 'fall', 'high_risk_interaction': 'violence', 'violence_detected': 'violence',
                  'wandering_entered_zone': 'wandering', 'wandering_exited_zone': 'wandering',
                  'location_update': 'location'}


def validate_labels(labels, duration, negative_confirmed):
    if not isinstance(labels, list) or len(labels) > 500:
        raise ValueError('ระบุช่วงเหตุไม่เกิน 500 ช่วง')
    if not labels and not negative_confirmed:
        raise ValueError('เพิ่มช่วงเหตุจริง หรือยืนยันว่าคลิปไม่มีเหตุชนิดที่เลือก')
    if labels and negative_confirmed:
        raise ValueError('คลิปมีช่วงเหตุจริงแล้ว กรุณาเอาติ๊กคลิปไม่มีเหตุออก')
    result = []
    for label in labels:
        start, end = float(label['start_s']), float(label['end_s'])
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start <= end <= duration:
            raise ValueError('เวลาเหตุจริงต้องอยู่ภายในคลิป และเวลาเริ่มไม่เกินเวลาสิ้นสุด')
        result.append(dict(start_s=start, end_s=end))
    return sorted(result, key=lambda item: (item['start_s'], item['end_s']))


def event_metrics(labels, predictions, duration, tolerance):
    """Maximum one-to-one matching; repeated alerts cannot improve recall."""
    edges = [[i for i, prediction in enumerate(predictions)
              if label['start_s'] <= prediction['alert_s'] <= label['end_s']+tolerance]
             for label in labels]
    owners = {}
    def assign(label_index, visited):
        for prediction_index in edges[label_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            if prediction_index not in owners or assign(owners[prediction_index], visited):
                owners[prediction_index] = label_index
                return True
        return False
    for i in range(len(labels)):
        assign(i, set())
    matches = [dict(label_index=li, prediction_index=pi,
                    delay_s=round(predictions[pi]['alert_s']-labels[li]['start_s'], 3))
               for pi, li in sorted(owners.items())]
    tp, fp, fn = len(matches), len(predictions)-len(matches), len(labels)-len(matches)
    delays = [match['delay_s'] for match in matches]
    return dict(true_positives=tp, false_positives=fp, missed_events=fn,
                precision=tp/len(predictions) if predictions else None,
                event_recall=tp/len(labels) if labels else None,
                false_alerts_per_hour=fp*3600/duration if duration > 0 else None,
                mean_alert_delay_s=sum(delays)/len(delays) if delays else None,
                max_alert_delay_s=max(delays) if delays else None, matches=matches)


class QaRun:
    def __init__(self, *, source, source_id, event_type, zone, labels, negative_confirmed,
                 duration, expected_frames, fps, tolerance=5, provenance=None):
        if event_type not in EVENT_FEATURES:
            raise ValueError('ชนิดเหตุ QA ไม่รองรับ')
        if not math.isfinite(float(tolerance)) or not 0 <= tolerance <= 30:
            raise ValueError('ระยะเผื่อการเตือนต้องอยู่ระหว่าง 0–30 วินาที')
        if not all(math.isfinite(v) and v > 0 for v in (duration, expected_frames, fps)):
            raise ValueError('QA ต้องใช้คลิปที่มีจำนวนภาพ/FPS/ระยะเวลาชัดเจน')
        if duration > 7200:
            raise ValueError('QA ผ่านเว็บรองรับคลิปไม่เกิน 2 ชั่วโมงต่อรอบ')
        self.id = uuid4().hex
        self.source, self.source_id = source, source_id
        self.event_type, self.feature, self.zone = event_type, EVENT_FEATURES[event_type], zone
        self.labels = validate_labels(labels, duration, negative_confirmed)
        self.duration, self.expected_frames, self.fps = duration, int(expected_frames), fps
        self.tolerance, self.provenance = float(tolerance), deepcopy(provenance or {})
        self.state, self.reason = 'running', None
        self.frames = self.errors = self.frames_with_person = self.feature_errors = 0
        self.last_frame = 0
        self.last_timestamp = 0
        self.predictions, self.seen = [], set()
        self.started_at = time.time()

    def record_events(self, events, timestamp_ms):
        if self.state != 'running':
            return
        for event in events:
            if event.get('source_id') != self.source_id or event.get('event_type') != self.event_type:
                continue
            metadata = event.get('metadata', {})
            actual_zone = metadata.get('zone') if self.feature == 'wandering' else metadata.get('current_zone')
            if self.zone and actual_zone != self.zone:
                continue
            key = (event['source_id'], event['event_id'])
            if key in self.seen:
                continue
            if len(self.predictions) >= 10000:
                self.finish(False, 'เหตุเกินขีดจำกัดของ QA รอบนี้')
                return
            self.seen.add(key)
            self.predictions.append(dict(event_id=event['event_id'], track_ids=event.get('track_ids', []),
                                         alert_s=timestamp_ms/1000, event_type=self.event_type, zone=actual_zone))

    def observe(self, frame_id, timestamp_ms, context, events, detector_error=None, module_error=False):
        if self.state != 'running':
            return
        if frame_id != self.last_frame+1 or timestamp_ms <= self.last_timestamp:
            self.finish(False, 'ลำดับภาพไม่ต่อเนื่องหรือมีการกรอคลิป')
            return
        self.last_frame, self.last_timestamp = frame_id, timestamp_ms
        self.frames += 1
        self.errors += bool(detector_error)
        self.feature_errors += bool(module_error)
        self.frames_with_person += bool(context.get('persons'))
        self.record_events(events, timestamp_ms)

    def finish(self, completed, reason=None, final_updates=()):
        if self.state != 'running':
            return
        self.record_events(final_updates, self.last_timestamp)
        if self.state != 'running':
            return
        # Early decoder EOF, errors or dropped frames invalidate full-clip claims.
        valid = completed and self.frames == self.expected_frames and not self.errors and not self.feature_errors
        self.state = 'completed' if valid else 'incomplete'
        self.reason = reason or (None if valid else 'ประมวลผลไม่ครบคลิป หรือมีช่วงตรวจจับ/ฟีเจอร์ผิดพลาด')

    def snapshot(self):
        return dict(id=self.id, state=self.state, reason=self.reason, event_type=self.event_type,
                    feature=self.feature, source_id=self.source_id, frames=self.frames,
                    expected_frames=self.expected_frames, progress=min(1, self.frames/self.expected_frames),
                    ground_truth_count=len(self.labels), prediction_count=len(self.predictions))

    def report(self):
        return dict(version=1, run=self.snapshot(), source=self.source,
                    provenance=deepcopy(self.provenance), labels=deepcopy(self.labels),
                    predictions=deepcopy(self.predictions), tolerance_sec=self.tolerance, scope_zone=self.zone,
                    coverage=dict(processed_frames=self.frames, expected_frames=self.expected_frames,
                                  processed_fraction=self.frames/self.expected_frames,
                                  detector_error_frames=self.errors, feature_error_frames=self.feature_errors,
                                  frames_with_usable_person=self.frames_with_person),
                    metrics=event_metrics(self.labels, self.predictions, self.duration, self.tolerance)
                    if self.state == 'completed' else None,
                    evaluation_scope='user_reviewed_clip_events', independent_accuracy_validated=False,
                    note='ผลตรงกับชนิดเหตุ/โซนและช่วงเวลาที่ผู้ใช้ระบุ ไม่ประเมินตัวตนใบหน้าหรือความแม่นยำในสถานที่จริง')
