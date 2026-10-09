"""
High-Risk Interaction Detection module.

event_type = "high_risk_interaction"

Design summary (see README.md for the full rationale):
  - Baseline is bbox/track kinematics only (no pixel/CNN features) so it cannot
    learn the "large person in frame == risk" shortcut that the whole-clip
    violence classifier in this project (see ../../../Improvement_Report.md)
    had to be corrected for. Every distance/velocity feature is normalized by
    bbox height, so it is scale- and camera-distance-invariant by construction.
  - Pose is used opportunistically when context["persons"][i]["pose"] is not
    None (either Shared Pose from the main system, or a fallback the demo/dev
    harness supplies) but is never required.
  - This module does exactly one thing per contract: frame + context in,
    list[dict] events out. It never opens a camera, never loads a person
    detector, and never talks to a UI, DB, or notification system.
"""
from __future__ import annotations

import itertools
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "1.0"
EVENT_TYPE = "high_risk_interaction"


# ---------------------------------------------------------------------------
# Small geometry helpers (bbox/track kinematics only — no pixels)
# ---------------------------------------------------------------------------

def _bbox_height(bbox_xyxy) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return max(float(y2) - float(y1), 1e-6)


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _pair_key(id_a: int, id_b: int):
    return tuple(sorted((id_a, id_b)))


# ---------------------------------------------------------------------------
# Per-frame feature extraction for a candidate pair
# ---------------------------------------------------------------------------

@dataclass
class FrameSample:
    timestamp_ms: float
    norm_distance: float          # center-to-center distance / avg bbox height
    iou: float                    # bbox overlap
    speed_a: float                # |velocity of person A| normalized by bbox height, per second
    speed_b: float
    closing_speed: float          # d(norm_distance)/dt ; negative = approaching
    min_hand_distance: Optional[float] = None  # normalized, only if pose available


class PairFeatureExtractor:
    """
    Stateless-ish extractor: needs the previous snapshot of each person to
    compute velocity, which the caller (PairState) keeps between calls.
    """

    def __init__(self, use_pose: bool):
        self.use_pose = use_pose

    def extract(
        self,
        timestamp_ms: float,
        person_a: dict,
        person_b: dict,
        prev_a: Optional[dict],
        prev_b: Optional[dict],
        prev_timestamp_ms: Optional[float],
        prev_norm_distance: Optional[float],
    ) -> FrameSample:
        bbox_a, bbox_b = person_a["bbox_xyxy"], person_b["bbox_xyxy"]
        center_a, center_b = person_a["center_xy"], person_b["center_xy"]
        avg_h = (_bbox_height(bbox_a) + _bbox_height(bbox_b)) / 2.0

        norm_distance = _dist(center_a, center_b) / avg_h
        iou = _bbox_iou(bbox_a, bbox_b)

        dt_sec = None
        if prev_timestamp_ms is not None:
            dt_sec = max((timestamp_ms - prev_timestamp_ms) / 1000.0, 1e-3)

        speed_a = 0.0
        speed_b = 0.0
        if prev_a is not None and dt_sec:
            speed_a = _dist(center_a, prev_a["center_xy"]) / avg_h / dt_sec
        if prev_b is not None and dt_sec:
            speed_b = _dist(center_b, prev_b["center_xy"]) / avg_h / dt_sec

        closing_speed = 0.0
        if prev_norm_distance is not None and dt_sec:
            closing_speed = (norm_distance - prev_norm_distance) / dt_sec

        min_hand_distance = None
        if self.use_pose:
            min_hand_distance = self._min_hand_distance(person_a, person_b, avg_h)

        return FrameSample(
            timestamp_ms=timestamp_ms,
            norm_distance=norm_distance,
            iou=iou,
            speed_a=speed_a,
            speed_b=speed_b,
            closing_speed=closing_speed,
            min_hand_distance=min_hand_distance,
        )

    @staticmethod
    def _min_hand_distance(person_a: dict, person_b: dict, avg_h: float) -> Optional[float]:
        """
        Bonus feature when pose keypoints are available. Expects COCO-style
        17-keypoint pose as a list of [x, y, score] (or None) with wrists at
        indices 9 (left) and 10 (right) -- the common convention for both
        Shared Pose output and typical pose fallbacks (e.g. YOLO-pose,
        OpenPose-COCO, MediaPipe remapped to COCO order). If the pose format
        doesn't match, this simply returns None and the module falls back to
        bbox-only features -- pose is a bonus signal, never a requirement.
        """
        pose_a, pose_b = person_a.get("pose"), person_b.get("pose")
        if not pose_a or not pose_b:
            return None
        try:
            def reliable_hands(pose, person):
                x1, y1, x2, y2 = person['bbox_xyxy']
                return [pose[i] for i in (9, 10) if pose[i] is not None and len(pose[i]) >= 3
                        and all(math.isfinite(float(v)) for v in pose[i][:3])
                        and float(pose[i][2]) >= .5
                        and x1 <= pose[i][0] <= x2 and y1 <= pose[i][1] <= y2]
            hands_a = reliable_hands(pose_a, person_a)
            hands_b = reliable_hands(pose_b, person_b)
        except (IndexError, TypeError, ValueError):
            return None
        if not hands_a or not hands_b:
            return None
        best = min(
            _dist(h[:2], person_b["center_xy"]) for h in hands_a
        )
        best = min(best, min(_dist(h[:2], person_a["center_xy"]) for h in hands_b))
        return best / avg_h


# ---------------------------------------------------------------------------
# Per-pair temporal state: ring buffer + event state machine
# ---------------------------------------------------------------------------

class PairState:
    def __init__(self, window_seconds: float, max_len_hint: int = 300):
        self.window_seconds = window_seconds
        self.samples: deque[FrameSample] = deque(maxlen=max_len_hint)
        self.last_seen_ts: float = 0.0
        self.prev_person_a: Optional[dict] = None
        self.prev_person_b: Optional[dict] = None
        self.prev_timestamp_ms: Optional[float] = None
        self.prev_norm_distance: Optional[float] = None

        # event state machine
        self.active: bool = False
        self.active_start_ts: Optional[float] = None
        self.last_event_end_ts: Optional[float] = None
        self.track_ids: tuple[int, int] = (-1, -1)
        self.diagnostic = dict(state='warming', rule_score=None, decision_score=None, reasons=[])

    def push(self, sample: FrameSample, person_a: dict, person_b: dict):
        self.samples.append(sample)
        self.last_seen_ts = sample.timestamp_ms
        self.prev_person_a = person_a
        self.prev_person_b = person_b
        self.prev_timestamp_ms = sample.timestamp_ms
        self.prev_norm_distance = sample.norm_distance
        self._trim(sample.timestamp_ms)

    def _trim(self, now_ms: float):
        window_ms = self.window_seconds * 1000.0
        while self.samples and (now_ms - self.samples[0].timestamp_ms) > window_ms:
            self.samples.popleft()

    def fill_ratio(self, now_ms: float) -> float:
        if not self.samples:
            return 0.0
        span_ms = now_ms - self.samples[0].timestamp_ms
        return min(span_ms / (self.window_seconds * 1000.0), 1.0)


# ---------------------------------------------------------------------------
# Window aggregation + scoring
# ---------------------------------------------------------------------------

@dataclass
class WindowAggregate:
    mean_closing_speed_abs: float
    max_closing_speed_abs: float
    impact_spike_count: int
    contact_frame_ratio: float
    direction_change_count: int
    max_individual_speed: float
    min_hand_distance: Optional[float]


def aggregate_window(samples: list[FrameSample], impact_velocity_threshold: float,
                     contact_iou_threshold: float = 0.02) -> WindowAggregate:
    closing = [s.closing_speed for s in samples]
    abs_closing = [abs(c) for c in closing]
    contact = sum(1 for s in samples if s.iou >= contact_iou_threshold and s.iou > 0.0)
    hand_dists = [s.min_hand_distance for s in samples if s.min_hand_distance is not None]

    direction_changes = 0
    for prev, cur in zip(closing, closing[1:]):
        if prev == 0 or cur == 0:
            continue
        if (prev > 0) != (cur > 0):
            direction_changes += 1

    return WindowAggregate(
        mean_closing_speed_abs=sum(abs_closing) / len(abs_closing) if abs_closing else 0.0,
        max_closing_speed_abs=max(abs_closing) if abs_closing else 0.0,
        impact_spike_count=sum(1 for c in abs_closing if c >= impact_velocity_threshold),
        contact_frame_ratio=contact / len(samples) if samples else 0.0,
        direction_change_count=direction_changes,
        max_individual_speed=max((max(s.speed_a, s.speed_b) for s in samples), default=0.0),
        min_hand_distance=min(hand_dists) if hand_dists else None,
    )


class RuleBasedScorer:
    """
    Weighted-rule baseline. Each term is in [0, 1]; weights sum to 1 so the
    final score stays in [0, 1] without a sigmoid. Kept deliberately simple
    and inspectable -- this is the "build a baseline first" model, not the
    final word. `reasons` is returned for logging / human review, never
    exposed as a verdict like "abuse".
    """

    def __init__(self, contact_iou_threshold: float, impact_velocity_threshold: float):
        self.contact_iou_threshold = contact_iou_threshold
        self.impact_velocity_threshold = impact_velocity_threshold

    def score(self, agg: WindowAggregate) -> tuple[float, list[str]]:
        reasons = []

        impact_term = min(agg.impact_spike_count / 3.0, 1.0)
        if agg.impact_spike_count > 0:
            reasons.append(f"impact spikes: {agg.impact_spike_count}")

        contact_term = min(agg.contact_frame_ratio / 0.3, 1.0)
        if agg.contact_frame_ratio > 0:
            reasons.append(f"contact frame ratio: {agg.contact_frame_ratio:.2f}")

        repetition_term = min(agg.direction_change_count / 4.0, 1.0)
        if agg.direction_change_count >= 2:
            reasons.append(f"direction changes (push/pull pattern): {agg.direction_change_count}")

        speed_term = min(agg.max_individual_speed / (self.impact_velocity_threshold * 1.5), 1.0)

        hand_term = 0.0
        if agg.min_hand_distance is not None and agg.min_hand_distance < 0.5:
            hand_term = min((0.5 - agg.min_hand_distance) / 0.5, 1.0)
            reasons.append(f"hand-to-body proximity: {agg.min_hand_distance:.2f}x bbox height")

        score = (
            0.30 * impact_term
            + 0.25 * contact_term
            + 0.20 * repetition_term
            + 0.15 * speed_term
            + 0.10 * hand_term
        )
        return min(max(score, 0.0), 1.0), reasons


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------

class HighRiskInteractionModule:
    """
    Contract-compliant feature module. See ../../../README.md at the repo
    root for how this plugs into the shared pipeline, and this folder's
    README.md for the architecture rationale.
    """

    def __init__(self):
        self._config: dict = {}
        self._sources: dict[str, dict[tuple[int, int], PairState]] = {}
        self._extractor: Optional[PairFeatureExtractor] = None
        self._scorer: Optional[RuleBasedScorer] = None
        self._calibrator = None  # optional sklearn model, lazy-loaded
        self._pose_gru = None    # optional torch model, lazy-loaded
        self._diagnostics = {}

    # -- required API -------------------------------------------------

    def setup(self, config: dict):
        self._config = config or {}
        f = self._config.get("features", {})
        self._extractor = PairFeatureExtractor(
            use_pose=bool(f.get("use_pose_if_available", True))
        )
        self._scorer = RuleBasedScorer(
            contact_iou_threshold=float(f.get("contact_iou_threshold", 0.02)),
            impact_velocity_threshold=float(f.get("impact_velocity_threshold", 4.0)),
        )
        model_type = self._config.get("feature", {}).get("model_type", "rule_based")
        if model_type == "calibrated":
            self._load_calibrator()
        elif model_type == "pose_gru":
            self._load_pose_gru()

    def reset(self, source_id: str):
        self._sources[source_id] = {}
        self._diagnostics.pop(source_id, None)

    def diagnostics(self, source_id: str) -> dict:
        from copy import deepcopy
        return deepcopy(self._diagnostics.get(source_id, dict(
            candidate_pairs=0, evaluated_pairs=0, max_rule_score=None,
            threshold=float(self._config.get('detection', {}).get('confidence_threshold', .70)),
            model_kind='weighted_rule', pairs=[])))

    def process(self, frame, context: dict) -> list[dict]:
        if not self._config.get("feature", {}).get("enabled", True):
            return []
        if self._extractor is None or self._scorer is None:
            raise RuntimeError("HighRiskInteractionModule.setup() must be called before process()")

        source_id = context["source_id"]
        timestamp_ms = float(context["timestamp_ms"])
        persons = context.get("persons") or []
        pairs_state = self._sources.setdefault(source_id, {})

        events: list[dict] = []

        candidate_keys = set()
        pair_cfg = self._config.get("pair", {})
        max_pair_distance = float(pair_cfg.get("max_pair_distance_bbox_heights", 1.5))

        for person_a, person_b in itertools.combinations(persons, 2):
            # Tracker output order can change between frames; keep velocities on the same ID.
            if person_a['track_id'] > person_b['track_id']:
                person_a, person_b = person_b, person_a
            if person_a.get("class_name", "person") != "person":
                continue
            if person_b.get("class_name", "person") != "person":
                continue
            avg_h = (_bbox_height(person_a["bbox_xyxy"]) + _bbox_height(person_b["bbox_xyxy"])) / 2.0
            d = _dist(person_a["center_xy"], person_b["center_xy"]) / avg_h
            if d > max_pair_distance:
                continue

            key = _pair_key(person_a["track_id"], person_b["track_id"])
            candidate_keys.add(key)
            state = pairs_state.get(key)
            if state is None:
                state = PairState(window_seconds=float(self._config.get("temporal", {}).get("window_seconds", 3.0)))
                state.track_ids = key
                pairs_state[key] = state

            sample = self._extractor.extract(
                timestamp_ms, person_a, person_b,
                state.prev_person_a, state.prev_person_b,
                state.prev_timestamp_ms, state.prev_norm_distance,
            )
            state.push(sample, person_a, person_b)

            event = self._evaluate_pair(source_id, key, state, timestamp_ms)
            if event is not None:
                events.append(event)

        self._expire_stale_pairs(pairs_state, timestamp_ms, candidate_keys)
        current = [dict(track_ids=list(key), fill_ratio=pairs_state[key].fill_ratio(timestamp_ms),
                        **pairs_state[key].diagnostic) for key in sorted(candidate_keys)]
        scores = [p['rule_score'] for p in current if p['rule_score'] is not None]
        self._diagnostics[source_id] = dict(candidate_pairs=len(current), evaluated_pairs=len(scores),
            max_rule_score=max(scores) if scores else None,
            threshold=float(self._config.get('detection', {}).get('confidence_threshold', .70)),
            model_kind='calibrated' if self._calibrator is not None else 'pose_gru' if self._pose_gru is not None else 'weighted_rule',
            pairs=current)
        return events

    def flush(self) -> list[dict]:
        """
        Finalize any pair currently mid-event (e.g. stream/video ended while
        a high-risk interaction was still ongoing) so it isn't silently lost.
        """
        events: list[dict] = []
        for source_id, pairs_state in self._sources.items():
            for key, state in pairs_state.items():
                if state.active and state.active_start_ts is not None:
                    events.append(
                        self._build_event(
                            source_id, key,
                            start_ts=state.active_start_ts,
                            end_ts=state.last_seen_ts,
                            confidence=0.70,  # conservative: window was cut short
                        )
                    )
                    state.active = False
        return events

    # -- internals ------------------------------------------------------

    def _evaluate_pair(self, source_id: str, key, state: PairState, now_ms: float) -> Optional[dict]:
        temporal_cfg = self._config.get("temporal", {})
        min_fill = float(temporal_cfg.get("min_window_fill_ratio", 0.5))
        if state.fill_ratio(now_ms) < min_fill:
            state.diagnostic = dict(state='warming', rule_score=None, decision_score=None, reasons=[])
            return None

        agg = aggregate_window(list(state.samples), self._scorer.impact_velocity_threshold,
                               self._scorer.contact_iou_threshold)
        confidence, reasons = self._scorer.score(agg)
        rule_score = confidence

        # optional ML layers refine (never replace) the rule-based confidence
        if self._calibrator is not None:
            confidence = self._apply_calibrator(agg, confidence)
        elif self._pose_gru is not None:
            confidence = self._apply_pose_gru(state, confidence)

        det_cfg = self._config.get("detection", {})
        threshold = float(det_cfg.get("confidence_threshold", 0.70))
        event_cfg = self._config.get("event", {})
        cooldown_sec = float(event_cfg.get("cooldown_sec", 10.0))
        min_duration_sec = float(event_cfg.get("minimum_duration_sec", 0.5))
        state.diagnostic = dict(state='below_threshold', rule_score=rule_score,
                                decision_score=confidence, reasons=reasons)

        if state.last_event_end_ts is not None:
            if (now_ms - state.last_event_end_ts) / 1000.0 < cooldown_sec:
                state.diagnostic['state'] = 'cooldown'
                return None

        if confidence >= threshold:
            state.diagnostic['state'] = 'confirming'
            if not state.active:
                state.active = True
                state.active_start_ts = now_ms
            duration_sec = (now_ms - state.active_start_ts) / 1000.0
            if duration_sec >= min_duration_sec:
                event = self._build_event(source_id, key, state.active_start_ts, now_ms, confidence, reasons)
                state.active = False
                state.last_event_end_ts = now_ms
                state.diagnostic['state'] = 'alert'
                return event
            return None
        else:
            state.active = False
            return None

    def _build_event(self, source_id, key, start_ts, end_ts, confidence, reasons=None) -> dict:
        det_cfg = self._config.get("detection", {})
        severity_high = float(det_cfg.get("severity_high_threshold", 0.85))
        severity = "high" if confidence >= severity_high else "medium"
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "event_type": EVENT_TYPE,
            "source_id": source_id,
            "track_ids": list(key),
            "start_timestamp_ms": start_ts,
            "end_timestamp_ms": end_ts,
            "confidence": round(float(confidence), 3),
            "severity": severity,
            "message": "High-risk interaction detected. Human review required.",
            "status": "needs_review",
            "metadata": {
                "duration_sec": round((end_ts - start_ts) / 1000.0, 2),
                "reasons": reasons or [],
            },
        }

    def _expire_stale_pairs(self, pairs_state: dict, now_ms: float, candidate_keys: set):
        timeout_ms = float(self._config.get("pair", {}).get("pair_timeout_sec", 2.0)) * 1000.0
        stale = [
            k for k, s in pairs_state.items()
            if k not in candidate_keys and (now_ms - s.last_seen_ts) > timeout_ms
        ]
        for k in stale:
            del pairs_state[k]

    # -- optional ML layers (lazy import so numpy-only baseline has zero
    #    hard dependency on sklearn/torch) ------------------------------

    def _load_calibrator(self):
        import pickle
        from pathlib import Path

        path = Path(__file__).parent / "models" / "calibrator.pkl"
        if not path.exists():
            self._calibrator = None
            return
        with open(path, "rb") as fh:
            self._calibrator = pickle.load(fh)

    def _apply_calibrator(self, agg: WindowAggregate, rule_score: float) -> float:
        import numpy as np

        x = np.array([[
            agg.mean_closing_speed_abs, agg.max_closing_speed_abs,
            agg.impact_spike_count, agg.contact_frame_ratio,
            agg.direction_change_count, agg.max_individual_speed,
            agg.min_hand_distance if agg.min_hand_distance is not None else -1.0,
            rule_score,
        ]])
        return float(self._calibrator.predict_proba(x)[0, 1])

    def _load_pose_gru(self):
        from pathlib import Path

        path = Path(__file__).parent / "models" / "pose_gru.pt"
        if not path.exists():
            self._pose_gru = None
            return
        import torch

        from training.model_pose_gru import PoseGRU  # local import, optional dep

        device = "cuda" if torch.cuda.is_available() and self._config.get("device", {}).get("prefer", "auto") != "cpu" else "cpu"
        model = PoseGRU()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        self._pose_gru = (model, device)

    def _apply_pose_gru(self, state: PairState, rule_score: float) -> float:
        # Falls back to the rule score whenever pose sequences aren't usable
        # for this pair (e.g. no pose supplied this run) -- pose_gru is a
        # strict upgrade path, never a hard requirement.
        if self._pose_gru is None:
            return rule_score
        model, device = self._pose_gru
        import torch

        seq = [s.min_hand_distance for s in state.samples if s.min_hand_distance is not None]
        if len(seq) < 4:
            return rule_score
        x = torch.tensor(seq, dtype=torch.float32, device=device).view(1, -1, 1)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        return float(prob)
