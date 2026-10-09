"""Runtime readiness is distinct from absence of alerts and model accuracy."""
import time


def snapshot(worker):
    age = None if worker.last_frame_at is None else max(0, time.monotonic()-worker.last_frame_at)
    capture = ('error' if worker.capture_error else 'paused' if worker.is_running and worker.is_paused
               else 'stalled' if worker.is_running and age is not None and age > 5
               else 'connecting' if worker.is_running and age is None
               else 'running' if worker.is_running else 'idle')
    detector = 'error' if worker.detector_error else 'ready' if worker.core is not None else 'unavailable'
    module_health = getattr(worker.pipeline, 'health', {})
    features = {}
    for name, cfg in worker.config.get('features', {}).items():
        raw = module_health.get(name, {})
        state = 'disabled' if not cfg.get('enabled') else 'unavailable' if not worker.pipeline or name not in worker.pipeline.modules else raw.get('state', 'ready')
        message = raw.get('error')
        if state == 'ready':
            if detector != 'ready' or capture in ('error', 'stalled'):
                state, message = 'unavailable', 'แหล่งภาพหรือตัวตรวจจับไม่พร้อม'
            elif capture in ('idle', 'connecting', 'paused'):
                state, message = 'waiting', 'ยังไม่มีภาพใหม่ให้ประเมิน'
            elif name == 'wandering' and not cfg.get('zones_relative'):
                state, message = 'needs_configuration', 'ยังไม่ได้วาดเขตพลัดหลงของกล้องนี้'
            elif name == 'location' and not cfg.get('config', {}).get('zones'):
                state, message = 'needs_configuration', 'ติดตามคนได้ แต่ยังไม่ได้ตั้งเตียง/โซน'
            elif name == 'violence' and getattr(worker.pipeline.modules[name], 'uses_scene_pixels', False):
                diagnostic = worker.pipeline.modules[name].diagnostics(worker.source_id)
                result = diagnostic.get('result')
                state = 'ready' if result else 'warming'
                message = ('LSTM คะแนนทำร้าย '+format(result['probability_fighting']*100, '.1f')+'% • ตรวจทั้งภาพ'
                    if result else f"LSTM กำลังเก็บภาพ {diagnostic['sampled_frames']}/16 เฟรม")
            elif not worker.active_tracks:
                state, message = 'no_observation', 'ไม่มีคนที่ผ่านเกณฑ์ให้ประเมิน'
            elif name in ('fall', 'violence') and all(
                    p.get('observation_kind') == 'bbox_only' for p in worker.active_tracks):
                state, message = 'no_observation', 'พบกรอบคน แต่ข้อต่อไม่พร้อมสำหรับประเมินเหตุการณ์'
            elif name=='fall' and getattr(worker.pipeline.modules[name],'ml_loaded',False):
                states=[worker.pipeline.modules[name].status(worker.source_id,p['track_id']) for p in worker.active_tracks]
                state='warming' if states and all(s['phase']=='WARMING_UP' for s in states) else 'ready'
                message='ตรวจล้ม V2 • IsolationForest + RandomForest • CPU'
            elif name == 'violence':
                diagnostic = worker.pipeline.modules[name].diagnostics(worker.source_id)
                if not diagnostic['candidate_pairs']:
                    state, message = 'no_pair', 'ยังไม่มีคนสองคนอยู่ใกล้พอให้ประเมินการปะทะ'
                elif not diagnostic['evaluated_pairs']:
                    state, message = 'warming', 'กำลังเก็บการเคลื่อนไหวของคู่คน'
                else:
                    message = 'กำลังประเมินความเสี่ยงการปะทะ'
                if diagnostic['model_kind'] == 'weighted_rule':
                    message += ' • ใช้กฎการเคลื่อนไหว ยังไม่ยืนยันการทำร้าย'
        features[name] = dict(state=state, message=message, failures=raw.get('failures', 0),
                              last_success_ms=raw.get('last_success_ms'))
    return dict(capture=dict(state=capture, message=worker.capture_error or worker.camera_warning,
                             frame_age_sec=round(age, 2) if age is not None else None),
                detector=dict(state=detector, message=worker.detector_error), features=features,
                accuracy_validated=False)
