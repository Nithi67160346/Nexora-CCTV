import unittest
from types import SimpleNamespace
from unittest.mock import patch
from web import server
from web.presentation import fall_display


class FallWebFixTests(unittest.TestCase):
    def test_fall_alert_remains_on_overlay_after_the_event_frame(self):
        import numpy as np
        from integration.fall_adapter import FallAdapter
        worker = server.StreamWorker()
        module = FallAdapter().setup({})
        worker.pipeline = SimpleNamespace(modules={'fall': module})
        ctx = dict(schema_version='1.0', source_id=worker.source_id, persons=[
            dict(track_id=1, confidence=.95, bbox_xyxy=[10, 10, 50, 150], pose=None)])
        for ts in range(0, 1200, 100):
            ctx['timestamp_ms'] = ts
            module.process(None, ctx)
        ctx['persons'][0]['bbox_xyxy'] = [10, 140, 170, 180]
        updates = []
        for ts in range(1200, 2100, 100):
            ctx['timestamp_ms'] = ts
            updates.extend(module.process(None, ctx))
        self.assertEqual(len(updates), 1)
        # No event update on this later frame; the confirmed state must render.
        overlay = worker._render_overlay(np.zeros((240, 320, 3), np.uint8), ctx, [])
        self.assertEqual(overlay[40, 5].tolist(), [0, 0, 220])
        worker.active_tracks = [dict(track_id=1)]
        with patch.object(server, 'worker', worker):
            status = server.get_status()
        self.assertEqual(status['fall_statuses']['1']['phase'], 'FALL_DETECTED')
        self.assertEqual(status['fall_revision'], 'qa-fall-fix8')
        module.reset(worker.source_id)
        overlay = worker._render_overlay(np.zeros((240, 320, 3), np.uint8), ctx, [])
        self.assertNotEqual(overlay[40, 5].tolist(), [0, 0, 220])


    def test_unavailable_fall_observation_does_not_display_normal_or_alert(self):
        self.assertEqual(fall_display(dict(phase='UNAVAILABLE'))[1], 'observing')
        self.assertEqual(fall_display(dict(phase='CONFIRMING'))[1], 'observing')
