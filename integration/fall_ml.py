"""YOLO V2 fall classification using the host's existing accepted poses."""
from collections import Counter, deque
from copy import deepcopy
import math
from pathlib import Path
from uuid import uuid4

import numpy as np

from integration.fall_model import DECISION_POLICY, FEATURE_NAMES, load_bundle, pose_features


class FallMLAdapter:
    backend = 'yolo_pose_rf_v2'
    revision = 'qa-fall-project54-20261007'

    def __init__(self, bundle=None):
        self.bundle = bundle

    def setup(self, config):
        self.config = deepcopy(config)
        self.window_size = int(config.get('window_size', 6))
        self.fall_ratio = float(config.get('fall_ratio_threshold', .3))
        self.abnormal_ratio = float(config.get('abnormal_ratio_threshold', .3))
        self.proba_threshold = float(config.get('fall_proba_threshold', .4))
        self.cooldown_ms = float(config.get('alert_cooldown_sec', 30))*1000
        self.gap_ms = float(config.get('maximum_observation_gap_sec', .35))*1000
        if not 2 <= self.window_size <= 30:
            raise ValueError('Fall V2 window_size ต้องอยู่ระหว่าง 2–30 เฟรม')
        if any(not math.isfinite(value) or not 0<value<=1 for value in (self.fall_ratio, self.abnormal_ratio, self.proba_threshold)):
            raise ValueError('Fall V2 thresholds ต้องมากกว่า0และไม่เกิน1')
        if any(not math.isfinite(value) or value<=0 for value in (self.cooldown_ms, self.gap_ms)):
            raise ValueError('Fall V2 timing ต้องมากกว่า0')
        if self.bundle is None:
            self.bundle = load_bundle(Path(__file__).resolve().parents[1]/'Fall/models_yolo')
        self.ml_loaded = True
        self.sources, self.clocks = {}, {}
        return self

    def reset(self, source_id):
        self.sources.pop(source_id, None)
        self.clocks.pop(source_id, None)

    @staticmethod
    def _unavailable(state, reason):
        state['samples'].clear()
        state.update(phase='UNAVAILABLE', reason=reason, observed=False, result=None,
                     fall_ratio=0., abnormal_ratio=0., fall_score=None)

    def status(self, source_id, track_id):
        state = self.sources.get(source_id, {}).get(track_id)
        base = dict(phase='UNAVAILABLE', reason='no_observation', observed=False,
            backend=self.backend, revision=self.revision, ml_loaded=self.ml_loaded,
            sampled_frames=0, required_frames=self.window_size, probability_fall=None, fall_score=None,
            threshold=self.proba_threshold, fall_ratio=0., abnormal_ratio=0., raw_status=None,
            ai1_score=None, decision_policy=DECISION_POLICY, feature_count=len(FEATURE_NAMES),
            ai1_gate_used=False, weights_sha256=self.bundle.checksum)
        if state:
            base.update({key:state[key] for key in ('phase','reason','observed','fall_ratio','abnormal_ratio','fall_score')})
            base['sampled_frames'] = len(state['samples'])
            if state['result']:base.update(state['result'])
        return base

    def diagnostics(self, source_id):
        return dict(backend=self.backend, revision=self.revision, ml_loaded=self.ml_loaded,
            model='RandomForest + Project.py geometry', feature_count=len(FEATURE_NAMES),
            decision_policy=DECISION_POLICY, ai1_gate_used=False, weights_sha256=self.bundle.checksum,
            fall_proba_threshold=self.proba_threshold, window_size=self.window_size,
            fall_ratio_threshold=self.fall_ratio, abnormal_ratio_threshold=self.abnormal_ratio,
            people={str(tid):self.status(source_id,tid) for tid in self.sources.get(source_id,{})})

    @staticmethod
    def _vector(person, width, height):
        try:
            if person.get('track_id') is None or person.get('observation_kind') == 'bbox_only':return None
            pose=np.asarray(person['pose'],dtype=float)
            if pose.shape!=(17,3) or not np.isfinite(pose).all():return None
            if np.any((pose[:,2]<0)|(pose[:,2]>1)) or np.count_nonzero(pose[5:,2]>=.25)<6:return None
            if width<=0 or height<=0 or np.any(pose[:,:2]<0) or np.any(pose[:,0]>width) or np.any(pose[:,1]>height):return None
            # Geometry must use observed shoulders and hips, not missing anchors.
            if np.any(pose[[5,6,11,12],2]<.25):return None
            shoulder=(pose[5,:2]+pose[6,:2])/2.;hip=(pose[11,:2]+pose[12,:2])/2.
            if np.linalg.norm(hip-shoulder)<1e-6:return None
            bbox=np.asarray(person['bbox_xyxy'],dtype=float)
            if bbox.shape!=(4,) or not np.isfinite(bbox).all():return None
            if bbox[0]<0 or bbox[1]<0 or bbox[2]>width or bbox[3]>height:return None
            return pose_features(pose,bbox)
        except (KeyError,TypeError,ValueError):return None

    def process(self, frame, context):
        source,ts=context['source_id'],float(context['timestamp_ms'])
        previous=self.clocks.get(source)
        if previous is not None and ts<=previous:self.reset(source)
        self.clocks[source]=ts
        height,width=frame.shape[:2]
        tracks=self.sources.setdefault(source,{})
        accepted=[]
        for person in context.get('persons',[]):
            vector=self._vector(person,width,height)
            if vector is not None:accepted.append((person['track_id'],vector))
        counts=Counter(tid for tid,_ in accepted)
        accepted=[(tid,vector) for tid,vector in accepted if counts[tid]==1]
        # Failed extraction cannot contribute a normal or positive observation.
        present={tid for tid,_ in accepted}
        for tid,state in list(tracks.items()):
            if tid not in present:
                self._unavailable(state,'track_missing_or_pose_unusable')
                if ts-state['last_seen']>1000:del tracks[tid]
        if not accepted:return []
        results=self.bundle.predict([vector for _,vector in accepted],self.proba_threshold)
        if len(results)!=len(accepted):raise ValueError('Fall V2 result count ไม่ตรงจำนวนคน')
        events=[]
        for (tid,_),result in zip(accepted,results):
            state=tracks.get(tid)
            if state is None:
                state=dict(samples=deque(maxlen=self.window_size),last_seen=ts,last_alert=None,
                    phase='WARMING_UP',reason='collecting_model_observations',observed=True,
                    fall_ratio=0.,abnormal_ratio=0.,fall_score=None,result=None)
                tracks[tid]=state
            if ts-state['last_seen']>self.gap_ms:self._unavailable(state,'observation_gap')
            state.update(last_seen=ts, observed=True, result=result)
            state['samples'].append((ts,result))
            samples=state['samples'];n=len(samples)
            state['fall_ratio']=sum(r['raw_status']=='ALERT_FALL' for _,r in samples)/n
            state['abnormal_ratio']=sum(r['raw_status']!='NORMAL' for _,r in samples)/n
            probabilities=[r['probability_fall'] for _,r in samples if r['probability_fall'] is not None]
            state['fall_score']=max(probabilities) if probabilities else None
            if n<self.window_size:
                state.update(phase='WARMING_UP',reason='collecting_model_observations')
            elif state['fall_ratio']>=self.fall_ratio:
                state.update(phase='FALL_DETECTED',reason='model_possible_fall')
            elif state['abnormal_ratio']>=self.abnormal_ratio:
                state.update(phase='ABNORMAL_MOVEMENT',reason='model_abnormal_pose')
            else:
                state.update(phase='NORMAL',reason='model_normal_pose')
            if state['phase']=='FALL_DETECTED' and (state['last_alert'] is None or ts-state['last_alert']>=self.cooldown_ms):
                md=self.status(source,tid)
                md.update(feature='fall', confidence_kind='model_score', experimental=True,
                    onset_observed=False, model_quality_gate_passed=False, lifecycle='detected',
                    sample_times_ms=[time for time,_ in samples],
                    pose_model=context.get('pose_model'),
                    score_kind='maximum_rf_score_in_observation_window', model='RandomForest + Project.py geometry',
                    window_decision_bases=dict(Counter(r.get('decision_basis','unknown') for _,r in samples)))
                events.append(dict(schema_version='1.0',event_id=str(uuid4()),event_type='fall_detected',
                    source_id=source,track_ids=[tid],start_timestamp_ms=samples[0][0],end_timestamp_ms=ts,
                    confidence=state['fall_score'],severity='high',status='needs_review',
                    message='โมเดล V2 พบท่าล้ม — กรุณาตรวจสอบภาพ',metadata=md))
                state['last_alert']=ts
        return events

    def flush(self):
        self.sources.clear();self.clocks.clear()
        return []
