"""QA v1: temporal fall rule. Scores are evidence scores, not probabilities."""
from copy import deepcopy
import math
from uuid import uuid4


class FallAdapter:
    # Start only after a substantial drop. During confirmation, a narrower
    # release threshold prevents small bbox shifts from erasing that onset.
    START_DROP_FRACTION = .2
    HOLD_DROP_FRACTION = .15
    START_LYING_RATIO = 1.35
    HOLD_LYING_RATIO = 1.30
    STRICT_LYING_OBSERVATIONS = 3

    def setup(self, config):
        self.config = deepcopy(config or {})
        self.sources, self.clocks = {}, {}
        self.ml_loaded = False
        self.backend = 'temporal_geometry_rule_v1'
        self.revision = 'qa-fall-fix8'
        self.posture_review_enabled = self.config.get('posture_review_enabled', True)
        self.baseline_ms = 1000 * float(self.config.get('upright_baseline_sec', 1.0))
        self.transition_ms = 1000 * float(self.config.get('transition_max_sec', 1.5))
        self.confirm_ms = 1000 * float(self.config.get('lying_confirmation_sec', .6))
        self.cooldown_ms = 1000 * float(self.config.get('alert_cooldown_sec', 30))
        self.timeout_ms = 1000 * float(self.config.get('track_timeout_sec', 1))
        self.observation_gap_ms = 1000 * float(self.config.get('maximum_observation_gap_sec', .35))
        self.reference_ms = 1000 * float(self.config.get('pose_reference_sec', .3))
        self.pose_person_confidence = float(self.config.get('pose_backed_person_confidence', .35))
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.baseline_ms, self.transition_ms, self.confirm_ms, self.cooldown_ms,
                self.timeout_ms, self.observation_gap_ms, self.reference_ms)):
            raise ValueError('Fall temporal settings must be finite and positive')
        if not math.isfinite(self.pose_person_confidence) or not .25 <= self.pose_person_confidence <= .5:
            raise ValueError('Pose-backed person confidence must be between .25 and .5')
        return self

    @staticmethod
    def _pause_evidence(state, reason, keep_lying=False):
        # A missing observation cannot count toward continuous posture evidence.
        # Keep a candidate through brief dropouts, but stop its observed-time
        # clock. Long gaps reset its evidence; unseen time never counts.
        state['upright_start'], state['upright_count'] = None, 0
        state['initial_reference'] = None
        state['last_lying_ms'] = None
        if not keep_lying:
            state['lying_start'], state['lying_count'] = None, 0
            state['lying_evidence_ms'] = 0
        state['reason'] = reason

    def status(self, source_id, track_id):
        state = self.sources.get(source_id, {}).get(track_id)
        reason = state['reason'] if state else 'no_observation'
        unavailable = reason in ('no_observation', 'track_missing', 'unusable_detection')
        phase = ('UNAVAILABLE' if unavailable else 'FALL_DETECTED' if reason == 'possible_fall'
                 else 'POSTURE_REVIEW' if state and state.get('posture_review_emitted')
                 else 'POSTURE_CHECK' if reason == 'down_pose_without_onset'
                 else 'CONFIRMING' if reason == 'confirming_lying'
                 else 'UPRIGHT_READY' if reason == 'upright_ready'
                 else 'WARMING_UP' if reason == 'upright_warming_up' else 'NO_ALERT')
        return dict(phase=phase, observed=not unavailable, reason=reason,
                    backend=self.backend, revision=self.revision,
                    bbox_ratio=state.get('bbox_ratio') if state else None,
                    drop_fraction=state.get('drop_fraction') if state else None,
                    reference_kind=state.get('reference_kind') if state else None,
                    lying_cue=state.get('lying_cue') if state else None,
                    posture_review_ms=state.get('posture_review_ms', 0) if state else 0,
                    observed_lying_ms=state.get('lying_evidence_ms', 0) if state else 0)

    @staticmethod
    def _has_body_pose(person):
        """Six confident body joints inside this detection, not face points."""
        try:
            box = list(map(float, person['bbox_xyxy']))
            pose = person['pose']
            if len(pose) != 17 or len(box) != 4:
                return False
            visible = 0
            for joint in pose[5:]:
                x, y, confidence = map(float, joint)
                if (all(math.isfinite(v) for v in (x, y, confidence)) and confidence >= .25
                        and box[0] <= x <= box[2] and box[1] <= y <= box[3]):
                    visible += 1
            return visible >= 6
        except (KeyError, TypeError, ValueError):
            return False

    def reset(self, source_id):
        self.sources.pop(source_id, None)
        self.clocks.pop(source_id, None)

    @staticmethod
    def _horizontal_pose(person):
        """Both torso and legs extended horizontally; bending alone is insufficient."""
        try:
            box=list(map(float,person['bbox_xyxy']));pose=person['pose']
            if len(pose)!=17 or len(box)!=4:return False
            for i in (5,6,11,12,15,16):
                x,y,c=map(float,pose[i])
                if not all(math.isfinite(v) for v in (x,y,c)) or c<.5 or not (box[0]<=x<=box[2] and box[1]<=y<=box[3]):return False
            shoulder=[(pose[5][i]+pose[6][i])/2 for i in (0,1)]
            hip=[(pose[11][i]+pose[12][i])/2 for i in (0,1)]
            ankle=[(pose[15][i]+pose[16][i])/2 for i in (0,1)]
            torso=[hip[i]-shoulder[i] for i in (0,1)]
            legs=[ankle[i]-hip[i] for i in (0,1)]
            span=max(box[2]-box[0],box[3]-box[1])
            return (span>0 and abs(torso[0])>=.18*span and abs(legs[0])>=.2*span
                    and abs(torso[1])<=.45*abs(torso[0]) and abs(legs[1])<=.45*abs(legs[0])
                    and torso[0]*legs[0]>0)
        except (KeyError,TypeError,ValueError,IndexError):return False

    @staticmethod
    def _down_pose(person):
        """A narrow bbox is not upright when reliable torso/head/legs are inverted.

        This is a view-dependent review cue, not proof of a fall or floor contact.
        All required joints must belong to the current accepted detection.
        """
        try:
            box = list(map(float, person['bbox_xyxy']))
            pose = person['pose']
            if len(pose) != 17 or box[3] <= box[1]:
                return False
            for i in (0, 5, 6, 11, 12, 15, 16):
                x, y, conf = map(float, pose[i])
                if not all(math.isfinite(v) for v in (x,y,conf)) or conf < .5 or not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
                    return False
            height = box[3]-box[1]
            shoulder = (pose[5][1]+pose[6][1])/2
            hip = (pose[11][1]+pose[12][1])/2
            ankle = (pose[15][1]+pose[16][1])/2
            return shoulder-hip >= .1*height and pose[0][1]-hip >= .1*height and hip-ankle >= .1*height
        except (KeyError, TypeError, ValueError):
            return False

    def _in_bed(self, person, frame, context):
        """Suppress the posture-only cue in configured bed zones, even if Location is off."""
        zones = self.config.get('posture_review_exclusion_zones', [])
        if not zones:
            return False
        pose = person['pose']
        x, y = (pose[11][0]+pose[12][0])/2, (pose[11][1]+pose[12][1])/2
        height, width = frame.shape[:2] if frame is not None else (context.get('frame_height',0),context.get('frame_width',0))
        for zone in zones:
            if zone.get('source_ids') and context['source_id'] not in zone['source_ids']:
                continue
            points = zone.get('points', [])
            if 'points_relative' in zone:
                if not width or not height:
                    continue
                points = [[a*width,b*height] for a,b in zone['points_relative']]
            inside = False
            for a,b in zip(points,points[1:]+points[:1]):
                if (a[1]>y) != (b[1]>y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
                    inside = not inside
            if inside:
                return True
        return False

    def _review_posture(self, state, person, frame, context, down_pose):
        ts = float(context['timestamp_ms'])
        samples = state.setdefault('posture_samples', [])
        samples[:] = [sample for sample in samples if ts-sample[0] <= 1500]
        if down_pose and self._in_bed(person,frame,context):
            samples.clear()
            state['posture_review_ms'], state['last_posture_ms'], state['posture_review_emitted'] = 0, None, False
            return []
        if down_pose and self.posture_review_enabled:
            previous = state.get('last_posture_ms')
            fps = float(context.get('fps',10))
            frame_ms = 1000/fps if math.isfinite(fps) and fps>0 else 100
            evidence = min(ts-previous,frame_ms) if previous is not None and 0<ts-previous<=self.observation_gap_ms else 0
            samples.append((ts,evidence))
            state['last_posture_ms'] = ts
        else:
            state['last_posture_ms'] = None
        state['posture_review_ms'] = sum(sample[1] for sample in samples)
        if (not down_pose or not self.posture_review_enabled or state.get('posture_review_emitted') or state.get('emitted')
                or state.get('baseline') is not None or state['posture_review_ms']+1e-6<self.confirm_ms):
            return []
        state['posture_review_emitted'] = True
        return [dict(schema_version='1.0',event_id=str(uuid4()),event_type='fall_posture_review',
                     source_id=context['source_id'],track_ids=[person['track_id']],
                     start_timestamp_ms=samples[0][0],end_timestamp_ms=ts,confidence=.5,severity='medium',
                     message='ท่าทางเสี่ยงที่ควรตรวจภาพ — ยังยืนยันจังหวะล้มไม่ได้',status='needs_review',
                     metadata=dict(feature='fall',backend=self.backend,adapter_revision=self.revision,
                                   confidence_kind='uncalibrated_rule_evidence',experimental=True,
                                   model_quality_gate_passed=False,event_policy_validated=False,lifecycle='detected',
                                   onset_observed=False,reference_kind='pose_only_review',
                                   observed_posture_ms=state['posture_review_ms']))]

    @staticmethod
    def _overlap(a, b):
        intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
        return intersection / union if union > 0 else 0

    def _recover_candidate(self, tracks, persons, ts):
        # A tracker can split one falling person into two IDs. Transfer only
        # an existing fall candidate, after the old ID disappears, with one
        # unique overlapping body-pose match in both directions. Never borrow
        # another visible person's baseline or start from an already lying ID.
        present = {p.get('track_id') for p in persons}
        matches = {}
        for person in persons:
            tid = person.get('track_id')
            geometry = self._geometry(person)
            target = tracks.get(tid, {})
            if (tid is None or target.get('baseline') is not None or target.get('lying_start') is not None
                    or geometry is None or not geometry[1] or not self._has_body_pose(person)):
                continue
            box = person['bbox_xyxy']
            for old_tid, state in tracks.items():
                old_box = state.get('last_bbox')
                if (old_tid in present or state.get('lying_start') is None or state.get('baseline') is None
                        or state.get('emitted') or not state.get('last_body_pose') or old_box is None
                        or not 0 < ts-state['last_seen'] <= self.observation_gap_ms):
                    continue
                old_center = ((old_box[0]+old_box[2])/2, (old_box[1]+old_box[3])/2)
                center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                distance = math.hypot(center[0]-old_center[0], center[1]-old_center[1])
                if self._overlap(box, old_box) >= .5 and distance <= .5*(old_box[3]-old_box[1]):
                    matches.setdefault(tid, set()).add(old_tid)
        for tid, candidates in matches.items():
            if len(candidates) != 1:
                continue
            old_tid = next(iter(candidates))
            if sum(old_tid in other for other in matches.values()) != 1:
                continue
            state = tracks.pop(old_tid)
            self._pause_evidence(state, 'track_reassociated', keep_lying=True)
            state.setdefault('previous_track_ids', []).append(old_tid)
            tracks[tid] = state

    def _geometry(self, person):
        try:
            x1, y1, x2, y2 = map(float, person['bbox_xyxy'])
            confidence = float(person.get('confidence', 0))
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, confidence)) or x2 <= x1 or y2 <= y1:
            return None
        if confidence < .5 and (confidence < self.pose_person_confidence or not self._has_body_pose(person)):
            return None
        height = y2 - y1
        ratio = (x2 - x1) / height
        return ratio < .8, ratio > self.START_LYING_RATIO, (y1 + y2) / 2, height

    def process(self, frame, context):
        source, ts = context['source_id'], float(context['timestamp_ms'])
        if not math.isfinite(ts) or ts < 0:
            raise ValueError('Invalid fall timestamp')
        previous = self.clocks.get(source)
        if previous is not None and ts < previous:
            raise ValueError('Timestamp moved backwards; reset source after seeking')
        if previous == ts:
            return []
        self.clocks[source] = ts
        tracks = self.sources.setdefault(source, {})
        for tid, state in list(tracks.items()):
            if ts - state['last_seen'] > self.timeout_ms:
                del tracks[tid]
        events = []
        self._recover_candidate(tracks, context.get('persons', []), ts)
        present = set()
        for person in context.get('persons', []):
            tid = person.get('track_id')
            if tid is None or tid in present:
                continue
            present.add(tid)
            geometry = self._geometry(person)
            if geometry is None:
                if tid in tracks:
                    self._pause_evidence(tracks[tid], 'unusable_detection',
                                         ts - tracks[tid]['last_seen'] <= self.observation_gap_ms)
                    tracks[tid]['last_posture_ms'] = None
                continue
            state = tracks.setdefault(tid, dict(last_seen=ts, upright_start=None, upright_count=0,
                baseline=None, lying_start=None, lying_count=0, emitted=False, cooldown_until=0,
                reason='no_observation', initial_reference=None, last_lying_ms=None, lying_evidence_ms=0,
                reference_kind=None, upright_reference=None))
            if ts <= state['last_seen'] and state.get('processed'):
                continue
            if state.get('processed') and ts - state['last_seen'] > self.observation_gap_ms:
                self._pause_evidence(state, 'observation_gap')
                state['posture_samples'], state['last_posture_ms'] = [], None
            state['last_seen'], state['processed'] = ts, True
            state['last_bbox'] = list(person['bbox_xyxy'])
            state['last_body_pose'] = self._has_body_pose(person)
            upright, lying, center, height = geometry
            down_pose = self._down_pose(person) or self._horizontal_pose(person)
            if down_pose:
                upright = False
                # When a real prior upright anchor exists, pose can supply the
                # second posture cue even if perspective keeps the bbox narrow.
                lying = True
            events.extend(self._review_posture(state,person,frame,context,down_pose))
            box = person['bbox_xyxy']
            state['bbox_ratio'] = (float(box[2]) - float(box[0])) / height
            state['drop_fraction'] = ((center - state['baseline'][1]) / state['baseline'][2]
                                      if state['baseline'] else None)
            strict_lying = lying
            state['lying_cue'] = 'pose' if down_pose else 'bbox_start' if lying else None
            # Keep an established onset through small bbox shape changes.
            # This margin cannot create an onset: require three strict lying
            # observations, a current body pose and a still-lowered center.
            if (not lying and state['lying_start'] is not None
                    and state.get('strict_lying_count', 0) >= self.STRICT_LYING_OBSERVATIONS
                    and state['bbox_ratio'] >= self.HOLD_LYING_RATIO
                    and state['baseline'] is not None
                    and center-state['baseline'][1] >= self.HOLD_DROP_FRACTION*state['baseline'][2]
                    and self._has_body_pose(person)):
                lying = True
                state['lying_cue'] = 'bbox_hold'
            if upright:
                state['reason'] = 'upright_warming_up'
                state['lying_start'], state['lying_count'] = None, 0
                state['lying_evidence_ms'], state['last_lying_ms'] = 0, None
                if self._has_body_pose(person):
                    # Short chair-fall clips can start seated and fall before
                    # the old 1s baseline. Require a stable pose-backed anchor.
                    reference = state['initial_reference']
                    if (reference is None or abs(center-reference[1]) > .08 * reference[2]
                            or abs(height-reference[2]) > .15 * reference[2]):
                        reference = (ts, center, height, 0)
                    reference = (*reference[:3], reference[3] + 1)
                    state['initial_reference'] = reference
                    if ts-reference[0] >= self.reference_ms and reference[3] >= 3:
                        state['baseline'] = (ts, reference[1], reference[2])
                        state['reference_kind'] = 'stable_pose_start'
                        state['emitted'] = False
                        state['reason'] = 'upright_ready'
                        state['posture_review_emitted'] = False
                        state['posture_samples'] = []
                else:
                    state['initial_reference'] = None
                reference = state.get('upright_reference')
                if (reference is None or abs(center-reference[0]) > .08*reference[1]
                        or abs(height-reference[1]) > .15*reference[1]):
                    state['upright_start'], state['upright_count'] = None, 0
                    state['upright_reference'] = (center, height)
                if state['upright_start'] is None:
                    state['upright_start'], state['upright_count'] = ts, 0
                state['upright_count'] += 1
                if ts - state['upright_start'] >= self.baseline_ms and state['upright_count'] >= 3:
                    # Keep a stable reference while the person starts falling
                    # but their bounding box is still narrow. Updating the
                    # baseline every frame erases the actual downward drop.
                    state['baseline'] = (ts, *state['upright_reference'])
                    state['reference_kind'] = 'upright_1s'
                    state['emitted'] = False
                    state['reason'] = 'upright_ready'
                    state['posture_review_emitted'] = False
                    state['posture_samples'] = []
                continue
            state['upright_start'], state['upright_count'] = None, 0
            state['upright_reference'] = None
            state['initial_reference'] = None
            baseline = state['baseline']
            if not lying:
                state['reason'] = 'posture_not_horizontal'
                state['lying_start'], state['lying_count'] = None, 0
                state['lying_evidence_ms'], state['last_lying_ms'] = 0, None
                if baseline and ts - baseline[0] > self.transition_ms:
                    state['baseline'] = None
                continue
            if state['lying_start'] is None:
                if baseline is None:
                    state['reason'] = 'down_pose_without_onset' if down_pose else 'no_upright_baseline'
                    continue
                if ts - baseline[0] > self.transition_ms:
                    state['reason'] = 'transition_too_slow'
                    state['baseline'] = None
                    continue
                if center - baseline[1] < self.START_DROP_FRACTION * baseline[2]:
                    state['reason'] = 'insufficient_downward_drop'
                    # An early/noisy horizontal box is not evidence that the
                    # verified upright anchor never existed. Keep it only for
                    # the existing transition window; it cannot count as lying.
                    continue
                state['lying_start'], state['lying_count'] = ts, 0
                state['strict_lying_count'] = 0
                state['lying_evidence_ms'], state['last_lying_ms'] = 0, None
            # Require continued lying and a lowered center after a qualifying
            # onset. Genuine recovery above the release threshold still cancels
            # all accumulated evidence; missing frames never add evidence.
            if baseline is None or center - baseline[1] < self.HOLD_DROP_FRACTION * baseline[2]:
                state['reason'] = 'insufficient_downward_drop'
                state['lying_start'], state['lying_count'] = None, 0
                state['lying_evidence_ms'], state['last_lying_ms'] = 0, None
                continue
            state['strict_lying_count'] = state.get('strict_lying_count', 0) + int(strict_lying)
            state['lying_count'] += 1
            if state['last_lying_ms'] is not None:
                fps = float(context.get('fps', 10))
                frame_ms = 1000 / fps if math.isfinite(fps) and fps > 0 else 100
                state['lying_evidence_ms'] += min(ts - state['last_lying_ms'], frame_ms)
            state['last_lying_ms'] = ts
            state['reason'] = 'possible_fall' if state['emitted'] else 'confirming_lying'
            if (ts - state['lying_start'] >= self.confirm_ms and state['lying_evidence_ms'] + 1e-6 >= self.confirm_ms
                    and state['lying_count'] >= 3 and not state['emitted'] and ts >= state['cooldown_until']):
                events.append(dict(schema_version='1.0', event_id=str(uuid4()), event_type='fall_detected',
                    source_id=source, track_ids=[tid], start_timestamp_ms=state['lying_start'],
                    end_timestamp_ms=ts, confidence=.5, severity='high',
                    message='Possible fall after rapid posture transition; human review required', status='needs_review',
                    metadata=dict(feature='fall', backend=self.backend, adapter_revision=self.revision,
                        confidence_kind='uncalibrated_rule_evidence',
                        experimental=True, model_quality_gate_passed=False, event_policy_validated=False,
                        lifecycle='detected', upright_baseline_ms=self.baseline_ms,
                        reference_kind=state['reference_kind'],
                        previous_track_ids=state.get('previous_track_ids', []),
                        observed_lying_ms=state['lying_evidence_ms'],
                        start_drop_fraction=self.START_DROP_FRACTION,
                        hold_drop_fraction=self.HOLD_DROP_FRACTION,
                        start_lying_ratio=self.START_LYING_RATIO,
                        hold_lying_ratio=self.HOLD_LYING_RATIO,
                        strict_lying_observations_required=self.STRICT_LYING_OBSERVATIONS,
                        lying_confirmation_ms=ts-state['lying_start'])))
                state['emitted'], state['cooldown_until'] = True, ts + self.cooldown_ms
                state['reason'] = 'possible_fall'
        for tid, state in tracks.items():
            if tid not in present:
                self._pause_evidence(state, 'track_missing', ts - state['last_seen'] <= self.observation_gap_ms)
                state['last_posture_ms'] = None
        return events

    def flush(self):
        self.sources.clear()
        self.clocks.clear()
        return []
