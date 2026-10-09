import unittest
from integration.fall_adapter import FallAdapter
from integration.pipeline import ProgressPipeline


def context(ts, bbox=(10, 10, 50, 150), source='A', track=1, confidence=.95):
    return dict(schema_version='1.0', source_id=source, timestamp_ms=ts, persons=[
        dict(track_id=track, confidence=confidence, bbox_xyxy=list(bbox), pose=None)])


class FallAdapterTests(unittest.TestCase):
    def start_pose_candidate(self):
        for ts in range(0, 1200, 100):
            self.module.process(None, self.pose_context(ts))
        for ts in range(1200, 1500, 100):
            self.module.process(None, self.pose_context(ts, (10, 140, 170, 180)))

    def test_narrow_box_during_descent_does_not_move_stable_baseline(self):
        self.feed(0, 1200)
        for ts in range(1200, 1600, 100):
            offset = (ts-1100)*.08
            self.module.process(None, context(ts, (10, 10+offset, 50, 150+offset)))
        baseline = self.module.sources['A'][1]['baseline']
        self.assertEqual(baseline[1:], (80, 140))
        self.assertEqual(len(self.feed(1600, 2400, (10, 140, 170, 180))), 1)

    def test_candidate_survives_one_unique_overlapping_body_pose_id_change(self):
        self.start_pose_candidate()
        events = []
        for ts in range(1500, 2100, 100):
            ctx = self.pose_context(ts, (12, 140, 172, 180))
            ctx['persons'][0]['track_id'] = 2
            events.extend(self.module.process(None, ctx))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['track_ids'], [2])
        self.assertEqual(events[0]['metadata']['previous_track_ids'], [1])
        self.assertEqual(events[0]['metadata']['observed_lying_ms'], 600)

    def test_shape_margin_holds_established_onset_with_current_body_pose(self):
        self.feed(0,1200)
        events=[]
        for ts in range(1200,2200,100):
            box=(10,65,146,165) if ts<1500 else (10,70,140,170)
            events.extend(self.module.process(None,self.pose_context(ts,box)))
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['metadata']['observed_lying_ms'],600)
        self.assertEqual(self.module.status('A',1)['lying_cue'],'bbox_hold')

    def test_shape_margin_cannot_start_a_fall_or_extend_a_single_horizontal_frame(self):
        for strict_frames in (0,1,2):
            with self.subTest(strict_frames=strict_frames):
                self.module=FallAdapter().setup({});self.feed(0,1200)
                events=[]
                for i,ts in enumerate(range(1200,2500,100)):
                    box=(10,65,146,165) if i<strict_frames else (10,70,140,170)
                    events.extend(self.module.process(None,self.pose_context(ts,box)))
                self.assertEqual(events,[])

    def test_shape_margin_still_rejects_recovery_narrow_pose_and_missing_body(self):
        for box,pose in (((10,49.6,140,149.6),True),((10,70,139,170),True),((10,70,140,170),False)):
            with self.subTest(box=box,pose=pose):
                self.module=FallAdapter().setup({});self.feed(0,1200)
                for ts in range(1200,1500,100):self.module.process(None,self.pose_context(ts,(10,65,146,165)))
                events=[]
                for ts in range(1500,2600,100):
                    ctx=self.pose_context(ts,box) if pose else context(ts,box)
                    events.extend(self.module.process(None,ctx))
                self.assertEqual(events,[])

    def test_shape_margin_does_not_count_missing_frames_as_lying_time(self):
        self.feed(0,1200)
        for ts in range(1200,1500,100):self.module.process(None,self.pose_context(ts,(10,65,146,165)))
        missing=context(1500);missing['persons']=[];self.module.process(None,missing)
        events=[]
        for ts in range(1600,2000,100):events.extend(self.module.process(None,self.pose_context(ts,(10,70,140,170))))
        self.assertEqual(events,[])
        self.assertEqual(self.module.status('A',1)['observed_lying_ms'],500)
        self.assertEqual(len(self.module.process(None,self.pose_context(2000,(10,70,140,170)))),1)

    def test_reassociation_cannot_borrow_a_visible_person_candidate(self):
        self.start_pose_candidate()
        ctx = self.pose_context(1500, (10, 140, 170, 180))
        other = self.pose_context(1500, (12, 140, 172, 180))['persons'][0]
        other['track_id'] = 2
        ctx['persons'].append(other)
        self.module.process(None, ctx)
        self.assertIsNone(self.module.sources['A'][2]['baseline'])

    def test_ambiguous_reassociation_targets_are_rejected(self):
        self.start_pose_candidate()
        ctx = self.pose_context(1500, (10, 140, 170, 180))
        ctx['persons'][0]['track_id'] = 2
        other = self.pose_context(1500, (12, 140, 172, 180))['persons'][0]
        other['track_id'] = 3
        ctx['persons'].append(other)
        self.module.process(None, ctx)
        self.assertIsNone(self.module.sources['A'][2]['baseline'])
        self.assertIsNone(self.module.sources['A'][3]['baseline'])

    def test_distant_or_bodyless_or_late_id_cannot_inherit_candidate(self):
        for bbox, has_pose, ts in (((300, 140, 460, 180), True, 1500),
                                   ((10, 140, 170, 180), False, 1500),
                                   ((10, 140, 170, 180), True, 1900)):
            with self.subTest(bbox=bbox, has_pose=has_pose, ts=ts):
                self.module = FallAdapter().setup({})
                self.start_pose_candidate()
                ctx = self.pose_context(ts, bbox)
                ctx['persons'][0]['track_id'] = 2
                if not has_pose: ctx['persons'][0]['pose'] = None
                self.module.process(None, ctx)
                self.assertIsNone(self.module.sources['A'][2]['baseline'])

    def pose_context(self, ts, bbox=(10, 10, 50, 150), confidence=.95):
        ctx = context(ts, bbox, confidence=confidence)
        x1, y1, x2, y2 = bbox
        ctx['persons'][0]['pose'] = [[x1+(x2-x1)*(.3 if j%2 else .7),
                                     y1+(y2-y1)*(.1+j*.045), .9] for j in range(17)]
        return ctx

    def test_pose_reference_supports_short_seated_to_floor_clip(self):
        for ts in range(0, 500, 100):self.module.process(None, self.pose_context(ts))
        events = []
        for ts in range(500, 1500, 100):
            events.extend(self.module.process(None, self.pose_context(ts, (10, 140, 170, 180), .4)))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['metadata']['reference_kind'], 'stable_pose_start')

    def test_pose_reference_does_not_alert_an_already_lying_person(self):
        for ts in range(0, 3000, 100):
            self.assertEqual(self.module.process(None, self.pose_context(ts, (10, 140, 170, 180), .4)), [])

    def test_face_only_or_outside_joints_do_not_admit_low_confidence_detection(self):
        person = self.pose_context(0, confidence=.4)['persons'][0]
        person['pose'][5:] = [[30, 40, .1] for _ in range(12)]
        self.assertIsNone(self.module._geometry(person))
        person['pose'] = [[500, 500, .9] for _ in range(17)]
        self.assertIsNone(self.module._geometry(person))

    def test_unstable_pose_does_not_create_a_short_reference(self):
        for ts in range(0, 500, 100):
            self.module.process(None, self.pose_context(ts, (10, 10+ts*.2, 50, 150+ts*.2)))
        self.assertIsNone(self.module.sources['A'][1]['baseline'])

    def test_long_dropout_resets_accumulated_confirmation_time(self):
        self.feed(0, 1200)
        self.feed(1200, 1500, (10, 140, 170, 180))
        for ts in range(1500, 2200, 100):
            ctx = context(ts);ctx['persons'] = [];self.module.process(None, ctx)
        self.assertEqual(self.feed(2200, 2800, (10, 140, 170, 180)), [])
        self.assertEqual(len(self.feed(2800, 2900, (10, 140, 170, 180))), 1)

    def test_sparse_frames_cannot_supply_unobserved_confirmation_time(self):
        self.feed(0, 1200)
        events = []
        for ts in range(1200, 1900, 100):
            ctx = context(ts, (10, 140, 170, 180));ctx['fps'] = 30
            events.extend(self.module.process(None, ctx))
        self.assertEqual(events, [])
        self.assertAlmostEqual(self.module.status('A', 1)['observed_lying_ms'], 200)

    def setUp(self):
        self.module = FallAdapter().setup({})

    def feed(self, start, end, bbox=(10, 10, 50, 150), **kwargs):
        events = []
        for ts in range(start, end, 100):
            events.extend(self.module.process(None, context(ts, bbox, **kwargs)))
        return events

    def test_already_lying_is_not_a_fall_even_after_cooldown(self):
        self.assertEqual(self.feed(0, 40000, (10, 140, 170, 180)), [])

    def test_standing_is_not_a_fall(self):
        self.assertEqual(self.feed(0, 3000), [])

    def test_one_horizontal_frame_does_not_alert(self):
        self.feed(0, 1200)
        self.assertEqual(self.feed(1200, 1300, (10, 140, 170, 180)), [])
        self.assertEqual(self.feed(1300, 2500), [])

    def test_bbox_jitter_after_qualifying_drop_keeps_observed_confirmation(self):
        self.feed(0, 1200)
        # Reference center=80, height=140. Begin at .25, then settle at .18.
        self.assertEqual(self.feed(1200, 1400, (10, 95, 170, 135)), [])
        events = self.feed(1400, 2400, (10, 85.2, 170, 125.2))
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]['metadata']['observed_lying_ms'], 600)
        self.assertEqual(events[0]['metadata']['start_drop_fraction'], .2)
        self.assertEqual(events[0]['metadata']['hold_drop_fraction'], .15)

    def test_release_margin_cannot_start_a_fall_on_its_own(self):
        self.feed(0, 1200)
        self.assertEqual(self.feed(1200, 4000, (10, 85.2, 170, 125.2)), [])
        self.assertEqual(self.module.status('A', 1)['observed_lying_ms'], 0)

    def test_early_horizontal_box_does_not_erase_anchor_before_real_drop(self):
        self.feed(0, 1200)
        self.assertEqual(self.feed(1200, 1300, (10, 85.2, 170, 125.2)), [])
        self.assertIsNotNone(self.module.sources['A'][1]['baseline'])
        self.assertEqual(len(self.feed(1300, 2300, (10, 140, 170, 180))), 1)

    def test_recovery_above_release_threshold_cancels_candidate(self):
        self.feed(0, 1200)
        self.feed(1200, 1500, (10, 95, 170, 135))
        self.assertEqual(self.feed(1500, 3000, (10, 79.6, 170, 119.6)), [])
        self.assertEqual(self.module.status('A', 1)['observed_lying_ms'], 0)

    def test_jitter_tolerance_does_not_fill_missing_confirmation_time(self):
        self.feed(0, 1200)
        self.feed(1200, 1500, (10, 95, 170, 135))
        missing = context(1500); missing['persons'] = []
        self.module.process(None, missing)
        self.assertEqual(self.feed(1600, 2000, (10, 85.2, 170, 125.2)), [])
        self.assertEqual(len(self.feed(2000, 2400, (10, 85.2, 170, 125.2))), 1)

    def test_transition_emits_one_review_event_and_no_repeat_while_lying(self):
        self.feed(0, 1200)
        events = self.feed(1200, 40000, (10, 140, 170, 180))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['status'], 'needs_review')
        self.assertFalse(events[0]['metadata']['event_policy_validated'])

    def test_slow_transition_is_not_a_fall(self):
        self.feed(0, 1200)
        self.feed(1200, 3100, (10, 30, 110, 140))
        self.assertEqual(self.feed(3100, 5000, (10, 140, 170, 180)), [])

    def test_lost_track_and_low_confidence_cannot_supply_baseline(self):
        self.feed(0, 1200)
        self.assertEqual(self.feed(5000, 7000, (10, 140, 170, 180)), [])
        self.module.flush()
        self.feed(0, 1200, confidence=.1)
        self.assertEqual(self.feed(1200, 3000, (10, 140, 170, 180)), [])

    def test_sources_and_track_ids_do_not_share_baseline(self):
        self.feed(0, 1200)
        self.assertEqual(self.feed(1200, 3000, (10, 140, 170, 180), source='B'), [])
        self.assertEqual(self.feed(1200, 3000, (10, 140, 170, 180), track=2), [])

    def test_brief_low_confidence_during_transition_preserves_verified_baseline(self):
        self.feed(0, 1200)
        self.module.process(None, context(1200, confidence=.3))
        self.assertEqual(self.module.status('A', 1)['phase'], 'UNAVAILABLE')
        events = self.feed(1300, 2300, (10, 140, 170, 180))
        self.assertEqual(len(events), 1)
        self.assertEqual(self.module.status('A', 1)['phase'], 'FALL_DETECTED')

    def test_missing_frame_cannot_count_as_lying_confirmation(self):
        self.feed(0, 1200)
        self.feed(1200, 1500, (10, 140, 170, 180))
        missing = context(1500); missing['persons'] = []
        self.module.process(None, missing)
        self.assertEqual(self.module.status('A', 1)['reason'], 'track_missing')
        self.assertEqual(self.feed(1600, 2000, (10, 140, 170, 180)), [])
        events = self.feed(2000, 2400, (10, 140, 170, 180))
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]['metadata']['observed_lying_ms'], 600)

    def test_unsampled_time_does_not_supply_an_upright_baseline(self):
        self.feed(0, 300)
        self.feed(900, 1200)
        self.assertEqual(self.feed(1200, 2500, (10, 140, 170, 180)), [])

    def test_confirmed_fall_status_persists_without_duplicate_events(self):
        self.feed(0, 1200)
        self.assertEqual(len(self.feed(1200, 2000, (10, 140, 170, 180))), 1)
        self.assertEqual(self.feed(2000, 4000, (10, 140, 170, 180)), [])
        self.assertEqual(self.module.status('A', 1)['phase'], 'FALL_DETECTED')
        self.module.reset('A')
        self.assertEqual(self.module.status('A', 1)['phase'], 'UNAVAILABLE')

    def test_pipeline_can_enable_fall_without_other_models(self):
        cfg = dict(features={name: dict(enabled=name=='fall') for name in ('fall','seizure','location','violence','wandering')})
        pipeline = ProgressPipeline.from_config(cfg)
        self.assertEqual(list(pipeline.modules), ['fall'])

    def test_timestamp_guard_and_reset(self):
        self.module.process(None, context(100))
        self.assertEqual(self.module.process(None, context(100)), [])
        with self.assertRaises(ValueError): self.module.process(None, context(0))
        self.module.reset('A')
        self.assertEqual(self.module.process(None, context(0)), [])
