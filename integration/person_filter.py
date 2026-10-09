"""Conservative observation checks, not a trained human/non-human classifier."""
from collections import Counter
from copy import deepcopy
import math

import numpy as np


REVISION = 'person-observation-v2'
DRAW_JOINT_CONFIDENCE = .5


def clean_pose(pose, bbox, frame_shape):
    """Keep COCO indices; mask invalid/outside joints without inventing points."""
    if pose is None:
        return None
    try:
        joints = np.asarray(pose, dtype=float)
        if joints.shape != (17, 3):
            return None
        joints = joints.copy()
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        valid = (np.isfinite(joints).all(axis=1)
                 & (joints[:, 2] >= 0) & (joints[:, 2] <= 1)
                 & (joints[:, 0] >= max(0, x1)) & (joints[:, 0] < min(width, x2 + 1))
                 & (joints[:, 1] >= max(0, y1)) & (joints[:, 1] < min(height, y2 + 1))
                 & ((joints[:, 0] != 0) | (joints[:, 1] != 0)))
        joints[~valid] = 0
        return joints.tolist()
    except (TypeError, ValueError, IndexError):
        return None


class PersonObservationFilter:
    """Confirm person boxes separately from usable body-pose observations.

    Strong person + body-pose evidence passes immediately. Other plausible
    detections need two consecutive observations of the same track. Horizontal
    poses, missing faces and partial occlusion are allowed; no standing rule.
    A high-score box with unusable pose needs three consecutive observations.
    It carries no skeleton and is excluded from fall/interaction evidence.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.source = None
        self.last_timestamp = None
        self.tracks = {}
        self.last_report = dict(revision=REVISION, candidates=0, accepted=0, reasons={})

    @staticmethod
    def _evidence(person, frame_shape, require_pose):
        try:
            box = list(map(float, person['bbox_xyxy']))
            score = float(person['confidence'])
            if len(box) != 4 or not all(math.isfinite(v) for v in (*box, score)):
                return None, 'invalid_detection', False
            h, w = frame_shape[:2]
            box = [max(0., box[0]), max(0., box[1]), min(float(w), box[2]), min(float(h), box[3])]
            if box[2] - box[0] < 8 or box[3] - box[1] < 8 or not 0 <= score <= 1:
                return None, 'invalid_detection', False
            pose = clean_pose(person.get('pose'), box, frame_shape)
            body = [] if pose is None else [p for p in pose[5:] if p[2] >= .25]
            spread = bool(body) and max(
                (max(p[0] for p in body)-min(p[0] for p in body))/(box[2]-box[0]),
                (max(p[1] for p in body)-min(p[1] for p in body))/(box[3]-box[1])) >= .15
            if score < .35 or (score < .5 and (len(body) < 6 or not spread)):
                return None, 'weak_person_evidence', False
            bbox_only = require_pose and (len(body) < 3 or not spread)
            if bbox_only and score < .65:
                return None, 'weak_body_pose', False
            clean = dict(person, bbox_xyxy=box, confidence=score, pose=pose,
                         center_xy=[(box[0]+box[2])/2, (box[1]+box[3])/2])
            if bbox_only:
                clean.update(pose=None, observation_kind='bbox_only')
            else:
                clean['observation_kind'] = 'body_pose' if require_pose else 'detection'
            strong = score >= .75 and len(body) >= 6 and spread
            return clean, None, strong
        except (KeyError, TypeError, ValueError, IndexError):
            return None, 'invalid_detection', False

    @staticmethod
    def _same_observation(old, person, ts):
        if old is None or not 0 <= ts-old['timestamp'] <= 350:
            return False
        a, b = old['bbox'], person['bbox_xyxy']
        area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
        return union > 0 and area / union >= .1

    def process(self, context, frame_shape, *, require_pose=True):
        source, ts = context['source_id'], float(context['timestamp_ms'])
        if not math.isfinite(ts) or ts < 0:
            raise ValueError('Invalid person observation timestamp')
        if self.source != source or (self.last_timestamp is not None and ts < self.last_timestamp):
            self.reset()
        # Duplicate timestamps cannot supply a second confirmation observation.
        duplicate = self.last_timestamp == ts
        self.source, self.last_timestamp = source, ts
        reasons, accepted, current, seen = Counter(), [], {}, set()
        candidates = context.get('persons', [])
        for person in candidates:
            tid = person.get('track_id')
            if tid is None or tid in seen:
                reasons['invalid_track'] += 1
                continue
            seen.add(tid)
            clean, reason, strong = self._evidence(person, frame_shape, require_pose)
            if clean is None:
                reasons[reason] += 1
                continue
            old = self.tracks.get(tid)
            continuous = (self._same_observation(old, clean, ts)
                          and old['kind'] == clean['observation_kind'])
            # A duplicate/reused ID or a switch from pose to box evidence cannot
            # borrow old confirmation observations.
            count = old['count'] if duplicate and continuous else old['count']+1 if continuous else 1
            needed = 3 if clean['observation_kind'] == 'bbox_only' else 2
            confirmed = strong or count >= needed
            current[tid] = dict(timestamp=ts, bbox=clean['bbox_xyxy'], count=count,
                                kind=clean['observation_kind'])
            if confirmed:
                accepted.append(clean)
            else:
                reasons['confirming_person'] += 1
        # No stale track or rejected observation can confirm a later detection.
        self.tracks = current
        self.last_report = dict(revision=REVISION, candidates=len(candidates),
                                accepted=len(accepted), reasons=dict(reasons),
                                bbox_only=sum(p['observation_kind'] == 'bbox_only' for p in accepted))
        result = dict(context, persons=accepted, person_filter=deepcopy(self.last_report))
        return result
