import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
from fastapi import HTTPException
from web import server
from web.model_catalog import resolve_pose_weights, model_options


class ModelPlaybackTests(unittest.TestCase):
    def test_legacy_alert_without_media_has_specific_error(self):
        worker = server.StreamWorker()
        worker.events = [dict(event_id='old', source_id='cam', metadata={})]
        with patch.object(server, 'worker', worker), patch.object(server.media_store, 'for_event', return_value=None):
            with self.assertRaises(HTTPException) as raised: server.event_playback('old')
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn('ไม่มีข้อมูลคลิปต้นฉบับ', raised.exception.detail)

    def test_only_local_supported_pose_models_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'models').mkdir()
            (root/'models/yolo26m-pose.pt').write_bytes(b'test weights')
            self.assertEqual(resolve_pose_weights(root, 'yolo26m-pose.pt').name, 'yolo26m-pose.pt')
            for name in ('yolov8s.pt', 'yolov8s-seg.pt', 'yolo26s.pt', 'yolo26m.pt', '../outside.pt', 'https://example.com/model.pt'):
                with self.assertRaises(ValueError): resolve_pose_weights(root, name)
            self.assertEqual([row['available'] for row in model_options(root)], [False, False, True, False, False, False])
            (root/'models/yolo26s-pose.pt').write_bytes(b'test weights')
            self.assertEqual(resolve_pose_weights(root, 'yolo26s-pose.pt').name, 'yolo26s-pose.pt')
            self.assertEqual([row['available'] for row in model_options(root)], [False, True, True, False, False, False])
            for name in ('yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt'):
                (root/'models'/name).write_bytes(b'test weights')
                self.assertEqual(resolve_pose_weights(root, name).name, name)
            self.assertEqual([row['available'] for row in model_options(root)], [False, True, True, True, True, True])

    def test_missing_model_does_not_stop_current_stream(self):
        worker = server.StreamWorker()
        with patch.object(worker, 'stop_stream') as stop, patch.object(server, 'resolve_pose_weights', side_effect=ValueError('missing weights')):
            with self.assertRaises(ValueError): worker.start_stream('clip.avi', model='yolo26m-pose.pt')
            stop.assert_not_called()

    def test_model_change_rebuilds_core_and_preserves_selected_cpu(self):
        worker = server.StreamWorker()
        original_core = SimpleNamespace(reset=Mock(), device='cpu')
        worker.core = original_core
        worker.pipeline = SimpleNamespace(reset=Mock())
        selected = worker.weights_path.with_name('yolo26m-pose.pt').resolve()
        def initialize(device):
            worker.core = SimpleNamespace(reset=Mock(), device=device)
            worker.pipeline = SimpleNamespace(reset=Mock())
        with patch.object(server, 'resolve_pose_weights', return_value=selected), patch.object(worker, 'stop_stream'), \
             patch.object(worker, 'initialize_models', side_effect=initialize) as init, \
             patch.object(server.threading, 'Thread'), patch.object(worker.cameras, 'save_active'):
            worker.start_stream('clip.avi', device='cpu', model=selected.name)
        init.assert_called_once_with('cpu')
        self.assertIsNot(worker.core, original_core)
        self.assertEqual(worker.weights_path, selected)
        self.assertEqual(worker.playback_speed, 1)

    def test_fallback_restarts_at_real_1x_and_rejects_qa_live_or_stale_source(self):
        worker = server.StreamWorker()
        worker.source='clip.avi';worker.is_running=True;worker.total_frames=180
        worker.current_frame_id=180;worker.current_sec=6;worker.is_paused=True;worker.review_mode=True
        worker.playback_speed=2
        request=server.PlaybackModeRequest(source='clip.avi')
        with patch.object(server, 'worker', worker):
            result=server.paced_playback(request)
            self.assertEqual(result['speed'], 1)
            self.assertEqual(worker.pending_seek_frame, 0)
            self.assertFalse(worker.review_mode);self.assertFalse(worker.is_paused)
            for kind in ('qa', 'live', 'stale'):
                worker.qa_run=SimpleNamespace(state='running') if kind=='qa' else None
                worker.is_live=kind=='live'
                worker.source='other.avi' if kind=='stale' else 'clip.avi'
                with self.assertRaises(HTTPException): server.paced_playback(request)

    def test_same_model_device_change_creates_a_fresh_backend(self):
        worker=server.StreamWorker();worker.device='cpu'
        old=SimpleNamespace(reset=Mock(),device='cpu');worker.core=old
        worker.pipeline=SimpleNamespace(reset=Mock())
        def initialize(device):
            worker.device=device;worker.core=SimpleNamespace(reset=Mock(),device=device)
            worker.pipeline=SimpleNamespace(reset=Mock())
        with patch.object(server,'_preferred_device',return_value='cuda'), \
             patch.object(server,'resolve_pose_weights',return_value=worker.weights_path.resolve()), \
             patch.object(worker,'stop_stream'),patch.object(worker,'initialize_models',side_effect=initialize) as init, \
             patch.object(server.threading,'Thread'),patch.object(worker.cameras,'save_active'):
            worker.start_stream('clip.mp4',device='cuda')
        init.assert_called_once_with('cuda');self.assertIsNot(worker.core,old)

    def test_fallback_loop_can_resume_eof_and_refuses_other_sources_live_qa_and_native(self):
        worker=server.StreamWorker();worker.source='clip.avi';worker.is_running=True
        worker.is_paused=True;worker.total_frames=10;worker.current_frame_id=10
        with patch.object(server,'worker',worker):
            result=server.clip_loop(server.ClipLoopRequest(source='clip.avi',enabled=True))
            self.assertTrue(result['enabled']);self.assertFalse(worker.is_paused)
            self.assertEqual(worker.pending_seek_frame,0)
            self.assertFalse(server.clip_loop(server.ClipLoopRequest(source='clip.avi',enabled=False))['enabled'])
            for case in ('stale','live','qa','native'):
                worker.is_live=case=='live';worker.review_mode=case=='native'
                worker.qa_run=SimpleNamespace(state='running') if case=='qa' else None
                with self.assertRaises(HTTPException):
                    server.clip_loop(server.ClipLoopRequest(source='other.avi' if case=='stale' else 'clip.avi',enabled=True))
                self.assertFalse(worker.loop_video)


if __name__ == '__main__': unittest.main()
