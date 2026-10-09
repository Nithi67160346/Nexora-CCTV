"""Sliding scene classification. Scores never identify a perpetrator or victim."""
from collections import deque
from copy import deepcopy
import math
from pathlib import Path
from uuid import uuid4

import cv2
from PIL import Image
import numpy as np

from web.violence_clip import FRAME_COUNT, shared_service


class ViolenceLSTM:
    uses_scene_pixels = True
    backend = 'resnet18_lstm'

    def __init__(self, predictor=None):
        self.predictor = predictor
        self.sources = {}

    def setup(self, config):
        self.interval_ms = float(config.get('sample_interval_ms', 125))
        self.stride_ms = float(config.get('inference_interval_ms', 1000))
        self.max_gap_ms = float(config.get('max_gap_ms', 500))
        for value in (self.interval_ms, self.stride_ms, self.max_gap_ms):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('LSTM timing must be positive and finite')
        self.service = shared_service(Path(__file__).resolve().parents[1])
        return self

    def reset(self, source_id):
        self.sources.pop(source_id, None)

    def _state(self, source):
        return self.sources.setdefault(source, dict(samples=deque(maxlen=FRAME_COUNT),
            last_frame=None, next_sample=None, last_inference=None, result=None, active=None))

    def diagnostics(self, source_id):
        state = self.sources.get(source_id)
        return dict(model_kind='resnet18_lstm', model='ResNet18 + LSTM', scope='scene_window',
            sampled_frames=len(state['samples']) if state else 0, required_frames=FRAME_COUNT,
            result=deepcopy(state['result']) if state else None)

    @staticmethod
    def _end(state):
        event = state['active']
        state['active'] = None
        if event:
            event['status'] = 'ended'
            return [deepcopy(event)]
        return []

    def process(self, frame, context):
        source, ts = context['source_id'], float(context['timestamp_ms'])
        state = self._state(source)
        updates = []
        previous = state['last_frame']
        if previous is not None and (ts <= previous or ts - previous > self.max_gap_ms):
            updates.extend(self._end(state))
            self.reset(source)
            state = self._state(source)
        state['last_frame'] = ts
        samples = state['samples']
        if state['next_sample'] is not None and ts < state['next_sample'] - .001:
            return updates
        due = state['next_sample'] if state['next_sample'] is not None else ts
        state['next_sample'] = due + (int(max(0, ts-due)/self.interval_ms)+1)*self.interval_ms
        # PIL uses the same bilinear resize as the supplied model's transform.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR))
        samples.append((ts, context['frame_id'], rgb))
        if len(samples) < FRAME_COUNT:
            return updates
        if state['last_inference'] is not None and ts - state['last_inference'] < self.stride_ms:
            return updates
        predictor = self.predictor or self.service.predict_frames
        result = predictor([sample[2] for sample in samples], context.get('device', 'cpu'))
        score = float(result['probability_fighting'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('LSTM returned an invalid score')
        result = dict(result, fighting=score > .5, threshold=.5, scope='scene_window',
            window_start_ms=samples[0][0], window_end_ms=ts,
            sampled_frame_ids=[sample[1] for sample in samples],
            sampled_times_ms=[sample[0] for sample in samples], sampled_frames=FRAME_COUNT)
        state['result'], state['last_inference'] = result, ts
        if score <= .5:
            return updates + self._end(state)
        event = state['active']
        if event is None:
            event = dict(schema_version='1.0', event_id=str(uuid4()),
                event_type='violence_detected', source_id=source, track_ids=[],
                start_timestamp_ms=samples[0][0], end_timestamp_ms=ts,
                confidence=score, severity='high', status='active',
                message='LSTM พบภาพเสี่ยงทำร้าย — กรุณาตรวจสอบ', metadata={})
            state['active'] = event
        event.update(end_timestamp_ms=ts, confidence=score)
        event['metadata'] = dict(result, feature='violence', confidence_kind='model_score',
            person_attribution=False, experimental=True)
        return updates + [deepcopy(event)]

    def flush(self):
        updates = []
        for state in self.sources.values():
            updates.extend(self._end(state))
            state['samples'].clear()
            state['next_sample'] = None
            state['last_frame'] = state['last_inference'] = None
        return updates
