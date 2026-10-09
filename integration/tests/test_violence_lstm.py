import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import time
from unittest.mock import Mock

import numpy as np

from integration.violence_lstm import ViolenceLSTM
from integration.pipeline import ProgressPipeline
from web.playback import AnnotationBuffer
from web.runtime_health import snapshot
from web import server


def context(ts, source='A'):
    return dict(source_id=source, timestamp_ms=ts, frame_id=round(ts*30/1000)+1,
                persons=[], device='cpu')


class ViolenceLSTMTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((32, 48, 3), np.uint8)
        self.predict = Mock(return_value=dict(probability_fighting=.998, model='ResNet18 + LSTM'))
        self.module = ViolenceLSTM(self.predict).setup({})

    def feed(self, timestamps, source='A', module=None):
        updates=[]
        for ts in timestamps:
            updates += (module or self.module).process(self.frame, context(ts, source))
        return updates

    def test_main_web_selects_lstm_even_for_an_old_camera_profile(self):
        config = server.StreamWorker()._product_config(dict(features={'violence':{'enabled':True}}))
        pipeline = ProgressPipeline.from_config(config)
        self.assertIsInstance(pipeline.modules['violence'], ViolenceLSTM)
        self.assertNotIn('violence', ProgressPipeline.from_config(dict(features={'violence':{'enabled':False}})).modules)

    def test_complete_short_window_creates_scene_event_without_inventing_people(self):
        updates=self.feed([i*1000/30 for i in range(60)])
        self.assertEqual(self.predict.call_count, 1)
        self.assertEqual(len(self.predict.call_args.args[0]), 16)
        event=updates[0]
        self.assertEqual(event['event_type'], 'violence_detected')
        self.assertEqual(event['track_ids'], [])
        self.assertEqual(event['confidence'], .998)
        self.assertEqual(event['metadata']['scope'], 'scene_window')
        self.assertFalse(event['metadata']['person_attribution'])
        self.assertLess(event['end_timestamp_ms'], 2000)
        self.assertEqual(len(event['metadata']['sampled_frame_ids']), 16)

    def test_no_prediction_before_real_frames_and_cameras_never_share_samples(self):
        self.feed([i*125 for i in range(15)], 'A')
        self.feed([i*125 for i in range(15)], 'B')
        self.predict.assert_not_called()
        self.module.process(self.frame, context(1875, 'B'))
        self.assertEqual(self.predict.call_count, 1)
        self.assertIsNone(self.module.diagnostics('A')['result'])

    def test_seek_dropout_and_explicit_source_reset_discard_old_results(self):
        self.feed([i*125 for i in range(16)])
        for ts in (50, 1000):
            updates=self.module.process(self.frame, context(ts))
            self.assertIsNone(self.module.diagnostics('A')['result'])
        self.module.reset('A')
        self.assertEqual(self.module.diagnostics('A')['sampled_frames'], 0)
        self.assertEqual(self.predict.call_count, 1)

    def test_continuous_alert_updates_one_event_and_normal_closes_it(self):
        first=self.feed([i*125 for i in range(16)])[0]
        positive=self.feed([i*125 for i in range(16,24)])[0]
        self.assertEqual(first['event_id'], positive['event_id'])
        self.predict.return_value={'probability_fighting':.2}
        closed=self.feed([i*125 for i in range(24,32)])[0]
        self.assertEqual(closed['status'], 'ended')
        self.assertEqual(closed['event_id'], first['event_id'])
        self.assertFalse(self.module.diagnostics('A')['result']['fighting'])

    def test_failure_is_visible_and_cannot_turn_into_normal_or_keep_old_window(self):
        self.predict.side_effect=ValueError('bad weights')
        pipeline=ProgressPipeline({'violence':self.module})
        for i in range(16):pipeline.process(self.frame, context(i*125))
        self.assertEqual(pipeline.health['violence']['state'], 'error')
        self.assertIn('bad weights',pipeline.health['violence']['error'])
        self.assertEqual(pipeline.aggregator.snapshot(), [])
        self.assertIsNone(self.module.diagnostics('A')['result'])

    def test_score_is_persisted_with_its_frame_and_replay_survives_switch(self):
        self.feed([i*125 for i in range(16)])
        worker=server.StreamWorker();worker.pipeline=ProgressPipeline({'violence':self.module})
        worker.source_id='A';worker.source='original.mp4';worker.session_id='original'
        with tempfile.TemporaryDirectory() as directory:
            worker.annotations.enable_disk(directory)
            worker._record_frame_evidence(context(1875), [])
            filename=worker.annotations.cache_path.name
            worker.annotations.clear(); self.module.reset('A')
            row=AnnotationBuffer.read_cached(directory,filename,1.875)[0]
            self.assertEqual(row['violence']['result']['probability_fighting'], .998)
            self.assertEqual(row['persons'], [])

    def test_frame_classifier_health_does_not_require_pose_or_a_pair(self):
        self.feed([i*125 for i in range(16)])
        worker=server.StreamWorker(); worker.pipeline=ProgressPipeline({'violence':self.module})
        worker.config={'features':{'violence':{'enabled':True}}};worker.source_id='A'
        worker.core=object();worker.is_running=True;worker.last_frame_at=time.monotonic()
        worker.active_tracks=[]
        features=snapshot(worker)['features']
        self.assertEqual(features['violence']['state'], 'ready')
        self.assertIn('99.8%',features['violence']['message'])

    def test_score_half_is_normal_and_invalid_score_never_creates_event(self):
        self.predict.return_value={'probability_fighting':.5}
        self.assertEqual(self.feed([i*125 for i in range(16)]), [])
        self.module.reset('A'); self.predict.return_value={'probability_fighting':float('nan')}
        with self.assertRaises(ValueError):self.feed([i*125 for i in range(16)])

    def test_end_of_clip_keeps_result_for_replay_but_next_run_requires_new_samples(self):
        self.feed([i*125 for i in range(16)])
        self.assertEqual(self.module.flush()[0]['status'], 'ended')
        self.assertEqual(self.module.diagnostics('A')['result']['probability_fighting'], .998)
        self.module.reset('A'); self.feed([i*125 for i in range(15)])
        self.assertEqual(self.predict.call_count,1)
