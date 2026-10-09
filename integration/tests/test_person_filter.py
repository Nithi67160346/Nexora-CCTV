import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from integration.core import SharedYoloCore
from integration.fall_adapter import FallAdapter
from integration.person_filter import PersonObservationFilter, clean_pose
from integration.pipeline import ProgressPipeline
from web import server


def person(score=.95, tid=1, box=(20, 40, 100, 220)):
    x1, y1, x2, y2 = box
    return dict(track_id=tid, confidence=score, bbox_xyxy=list(box),
                pose=[[x1+(x2-x1)*(.3 if j%2 else .7),
                       y1+(y2-y1)*(.1+j*.045), .9] for j in range(17)])


def context(ts, people, source='A'):
    return dict(schema_version='1.0', source_id=source, frame_id=int(ts),
                timestamp_ms=ts, fps=30, persons=people)


class PersonFilterTests(unittest.TestCase):
    def setUp(self):
        self.gate = PersonObservationFilter()
        self.shape = (300, 400, 3)

    def run_frame(self, ts, people, source='A', **kwargs):
        return self.gate.process(context(ts, people, source), self.shape, **kwargs)['persons']

    def test_low_score_is_rejected_and_bad_pose_never_supplies_skeleton(self):
        for mode in ('low_score', 'face_only', 'collapsed_body', 'outside_body'):
            with self.subTest(mode=mode):
                self.gate.reset()
                p = person()
                if mode == 'low_score': p['confidence'] = .3
                if mode == 'face_only': p['pose'][5:] = [[40, 80, .1]]*12
                if mode == 'collapsed_body': p['pose'][5:] = [[40, 80, .99]]*12
                if mode == 'outside_body': p['pose'][5:] = [[350, 290, .99]]*12
                for ts in range(0, 600, 33):
                    result = self.run_frame(ts, [p])
                    if mode == 'low_score' or ts < 66:
                        self.assertEqual(result, [])
                    else:
                        self.assertEqual(result[0]['observation_kind'], 'bbox_only')
                        self.assertIsNone(result[0]['pose'])

    def test_single_frame_moderate_detection_is_withheld(self):
        self.assertEqual(self.run_frame(0, [person(.6)]), [])
        self.assertEqual(self.run_frame(33, []), [])
        self.assertEqual(self.run_frame(66, [person(.6)]), [])

    def test_continuous_moderate_person_is_admitted_without_standing_constraint(self):
        lying = person(.4, box=(20, 180, 260, 240))
        self.assertEqual(self.run_frame(0, [lying]), [])
        result = self.run_frame(33, [lying])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['bbox_xyxy'], lying['bbox_xyxy'])

    def test_strong_body_pose_passes_immediately_without_visible_face(self):
        p = person()
        p['pose'][:5] = [[0, 0, 0]]*5
        self.assertEqual(self.run_frame(0, [p])[0]['track_id'], 1)

    def test_partially_occluded_person_does_not_need_full_body(self):
        p = person(.65)
        p['pose'][9:] = [[0, 0, 0]]*8
        self.assertEqual(self.run_frame(0, [p]), [])
        self.assertEqual(len(self.run_frame(33, [p])), 1)

    def test_nonfinite_and_outside_joints_are_masked_without_reindexing(self):
        p = person()
        p['pose'][0] = [float('nan'), 60, .99]
        p['pose'][1] = [350, 250, .99]
        p['pose'][2] = [0, 0, .99]
        cleaned = clean_pose(p['pose'], p['bbox_xyxy'], self.shape)
        self.assertEqual(cleaned[:3], [[0, 0, 0]]*3)
        self.assertEqual(cleaned[5:], p['pose'][5:])
        self.assertEqual(len(self.run_frame(0, [p])), 1)

    def test_invalid_or_tiny_boxes_and_nonfinite_score_are_rejected(self):
        for box, score in (((20, 40, 19, 220), .95), ((0, 0, 5, 5), .95),
                           ((500, 40, 600, 220), .95), ((20, 40, 100, 220), float('nan'))):
            with self.subTest(box=box, score=score):
                self.assertEqual(self.run_frame(0, [person(score, box=box)]), [])

    def test_high_score_box_without_body_needs_three_real_observations(self):
        p = person(); p['pose'] = None
        for ts in (0, 33, 33):
            self.assertEqual(self.run_frame(ts, [p]), [])
        result = self.run_frame(66, [p])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['observation_kind'], 'bbox_only')
        self.assertIsNone(result[0]['pose'])
        self.assertEqual(self.gate.last_report['bbox_only'], 1)

    def test_low_score_missing_pose_cannot_use_box_fallback(self):
        for score in (.3, .4, .6):
            self.gate.reset()
            p = person(score); p['pose'] = None
            for ts in (0, 33, 66, 99):
                self.assertEqual(self.run_frame(ts, [p]), [])

    def test_pose_history_cannot_confirm_new_box_only_mode(self):
        self.run_frame(0, [person()])
        p = person(); p['pose'] = None
        self.assertEqual(self.run_frame(33, [p]), [])
        self.assertEqual(self.run_frame(66, [p]), [])
        self.assertEqual(len(self.run_frame(99, [p])), 1)

    def test_box_confirmation_resets_at_gap_source_id_or_distant_box(self):
        p = person(); p['pose'] = None
        self.run_frame(0, [p]); self.run_frame(33, [p])
        self.assertEqual(self.run_frame(500, [p]), [])
        self.assertEqual(self.run_frame(533, [p], source='B'), [])
        p['track_id'] = 2
        self.assertEqual(self.run_frame(566, [p], source='B'), [])
        self.run_frame(599, [p], source='B')
        p['bbox_xyxy'] = [220, 40, 300, 220]
        self.assertEqual(self.run_frame(632, [p], source='B'), [])

    def test_bbox_only_reaches_location_but_not_fall_or_interaction(self):
        received = {}
        def module(name):
            return SimpleNamespace(process=lambda frame, ctx: received.update({name: ctx}) or [])
        pipeline = ProgressPipeline({name: module(name) for name in ('fall', 'violence', 'location', 'wandering')})
        p = person(); p['pose'] = None
        for ts in (0, 33, 66):
            ctx = self.gate.process(context(ts, [p]), self.shape)
            pipeline.process(np.zeros(self.shape, np.uint8), ctx)
        self.assertEqual(len(received['location']['persons']), 1)
        self.assertEqual(len(received['wandering']['persons']), 1)
        self.assertEqual(received['fall']['persons'], [])
        self.assertEqual(received['violence']['persons'], [])
        self.assertEqual(received['fall']['timestamp_ms'], 66)
        self.assertEqual(len(ctx['persons']), 1)
        # Fresh usable pose is still passed on to action modules.
        ctx = self.gate.process(context(99, [person()]), self.shape)
        pipeline.process(np.zeros(self.shape, np.uint8), ctx)
        self.assertEqual(len(received['fall']['persons']), 1)
        self.assertEqual(len(received['violence']['persons']), 1)

    def test_detection_only_model_keeps_supported_bbox_tracking(self):
        p = person(); p['pose'] = None
        self.assertEqual(self.run_frame(0, [p], require_pose=False), [])
        self.assertEqual(len(self.run_frame(33, [p], require_pose=False)), 1)

    def test_different_sources_ids_gaps_and_duplicate_frames_do_not_confirm(self):
        self.assertEqual(self.run_frame(0, [person(.6)]), [])
        self.assertEqual(self.run_frame(0, [person(.6)]), [])
        self.assertEqual(self.run_frame(33, [person(.6, tid=2)]), [])
        self.assertEqual(self.run_frame(66, [person(.6, tid=2)], source='B'), [])
        self.assertEqual(self.run_frame(1000, [person(.6, tid=2)], source='B'), [])
        self.assertEqual(self.run_frame(0, [person(.6, tid=2)], source='B'), [])
        self.assertEqual(len(self.run_frame(33, [person(.6, tid=2)], source='B')), 1)
        self.gate.reset()
        self.assertEqual(self.run_frame(66, [person(.6, tid=2)], source='B'), [])

    def test_reused_id_at_distant_object_requires_confirmation_again(self):
        self.run_frame(0, [person(.6)])
        self.run_frame(33, [person(.6)])
        self.assertEqual(self.run_frame(66, [person(.6, box=(220, 40, 300, 220))]), [])

    def test_weak_current_observation_is_not_filled_from_confirmed_history(self):
        self.run_frame(0, [person()])
        weak = person(.4); weak['pose'][5:] = [[40, 80, .1]]*12
        self.assertEqual(self.run_frame(33, [weak]), [])
        self.assertEqual(self.gate.last_report['accepted'], 0)

    def test_moderate_horizontal_person_still_reaches_fall_confirmation(self):
        fall = FallAdapter().setup({})
        events = []
        for ts in range(0, 1200, 100):
            filtered = self.gate.process(context(ts, [person()]), self.shape)
            filtered['fps'] = 10
            events.extend(fall.process(None, filtered))
        for ts in range(1200, 2400, 100):
            filtered = self.gate.process(context(ts, [person(.4, box=(20, 200, 260, 260))]), self.shape)
            filtered['fps'] = 10
            events.extend(fall.process(None, filtered))
        self.assertEqual(len(events), 1)

    def test_core_filters_before_return_and_reset_discards_confirmation(self):
        p = person(.6)
        boxes = SimpleNamespace(xyxy=torch.tensor([p['bbox_xyxy']]),
                                conf=torch.tensor([p['confidence']]), id=torch.tensor([1.]))
        joints = torch.tensor([p['pose']])
        pose = SimpleNamespace(xy=joints[..., :2], conf=joints[..., 2])
        tracker = SimpleNamespace(reset=lambda: None)
        model = SimpleNamespace(task='pose', predictor=SimpleNamespace(trackers=[tracker]),
                                track=lambda *args, **kwargs: [SimpleNamespace(boxes=boxes, keypoints=pose)])
        core = SharedYoloCore(model)
        frame = np.zeros(self.shape, np.uint8)
        def run(ts):
            return core.process(frame, source_id='A', frame_id=ts, timestamp_ms=ts, fps=30)
        self.assertEqual(run(0)['persons'], [])
        confirmed = run(33)
        self.assertEqual(len(confirmed['persons']), 1)
        self.assertEqual(confirmed['person_filter']['raw_detections'], 1)
        self.assertEqual(confirmed['person_filter']['tracked_detections'], 1)
        with patch.object(tracker, 'reset', wraps=tracker.reset) as reset:
            core.reset()
            reset.assert_called_once()
        self.assertEqual(run(66)['persons'], [])

    def test_overlay_does_not_draw_uncertain_or_invalid_joints(self):
        worker = server.StreamWorker()
        p = person()
        p['pose'][7] = [60, 140, .3]
        p['pose'][8] = [float('nan'), 100, .99]
        worker.pipeline = SimpleNamespace(modules={})
        with patch.object(server.cv2, 'line') as line, patch.object(server.cv2, 'circle') as circle:
            worker._render_overlay(np.zeros(self.shape, np.uint8), context(0, [p]), [])
        points = [call.args[1] for call in circle.call_args_list]
        self.assertNotIn((60, 140), points)
        self.assertEqual(len(points), 15)
        for call in line.call_args_list:
            self.assertNotEqual(call.args[1], (60, 140))
            self.assertNotEqual(call.args[2], (60, 140))


if __name__ == '__main__':
    unittest.main()
