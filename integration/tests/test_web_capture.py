import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
from web import capture, server


class PassthroughRecording:
    """These tests isolate camera error handling; recording has its own suite."""
    def __new__(cls, cap, *_): return cap


class WebcamTests(unittest.TestCase):
    def test_windows_fallback_keeps_selected_camera_and_releases_failed_backend(self):
        failed, working = Mock(), Mock()
        failed.isOpened.return_value = False
        working.isOpened.return_value = True
        with patch.object(capture.os, 'name', 'nt'), patch.object(capture.cv2, 'VideoCapture', side_effect=[failed, working]) as factory:
            self.assertIs(capture.open_capture(1), working)
        self.assertEqual(factory.call_args_list[0].args, (1, capture.cv2.CAP_DSHOW))
        self.assertEqual(factory.call_args_list[1].args, (1, capture.cv2.CAP_MSMF))
        failed.release.assert_called_once()
        working.release.assert_not_called()

    def test_file_source_keeps_default_capture(self):
        with patch.object(capture.cv2, 'VideoCapture') as factory:
            capture.open_capture('movie.mp4')
        factory.assert_called_once_with('movie.mp4')

    def test_empty_frame_is_not_published_and_dark_frame_is_only_an_advisory(self):
        self.assertFalse(capture.valid_frame(True, None))
        self.assertFalse(capture.valid_frame(True, np.empty((0, 0, 3))))
        black = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertTrue(capture.valid_frame(True, black))
        self.assertTrue(capture.dark_frame(black))
        self.assertFalse(capture.dark_frame(black+100))

    def worker_and_camera(self):
        worker = server.StreamWorker()
        worker.source, worker.is_live, worker.is_running = '1', True, True
        worker.pipeline = SimpleNamespace(modules={}, flush=lambda:[], process=lambda frame,context:[])
        worker.core = Mock()
        cam = Mock()
        cam.isOpened.return_value = True
        cam.get.return_value = 30
        cam.getBackendName.return_value = 'DSHOW'
        return worker, cam

    def test_permanent_live_read_failure_stops_with_error_instead_of_pausing_blank(self):
        worker, cam = self.worker_and_camera()
        cam.read.return_value = (False, None)
        with patch.object(server, 'open_capture', return_value=cam), patch.object(server, 'RecordingCapture', new=PassthroughRecording), patch.object(server.time, 'sleep'):
            worker._run_loop(worker.generation_id)
        self.assertFalse(worker.is_running)
        self.assertFalse(worker.is_paused)
        self.assertIn('กล้องหยุดส่งภาพ', worker.last_error)
        self.assertIsNone(worker.latest_frame_jpeg)
        worker.core.process.assert_not_called()
        cam.release.assert_called_once()

    def test_camera_warmup_can_recover_and_publish_first_frame(self):
        worker, cam = self.worker_and_camera()
        frame = np.full((80, 120, 3), 120, dtype=np.uint8)
        cam.read.side_effect = [(False, None), (True, None), (True, frame)]
        def process(*args, **kwargs):
            worker.is_running = False
            return dict(persons=[],source_id=kwargs['source_id'],frame_id=kwargs['frame_id'],timestamp_ms=kwargs['timestamp_ms'])
        worker.core.process.side_effect = process
        with patch.object(server, 'open_capture', return_value=cam), patch.object(server, 'RecordingCapture', new=PassthroughRecording), patch.object(server.time, 'sleep'):
            worker._run_loop(worker.generation_id)
        self.assertIsNotNone(worker.latest_frame_jpeg)
        self.assertEqual(worker.current_frame_id, 1)
        self.assertIsNone(worker.last_error)
        self.assertEqual(worker.capture_backend, 'DSHOW')
        cam.release.assert_called_once()

    def test_file_end_still_pauses_without_camera_error(self):
        worker, cam = self.worker_and_camera()
        worker.source, worker.is_live, worker.loop_video = 'movie.mp4', False, False
        cam.read.return_value = (False, None)
        with patch.object(server, 'open_capture', return_value=cam), patch.object(server.time, 'sleep', side_effect=lambda _:setattr(worker,'is_running',False)):
            worker._run_loop(worker.generation_id)
        self.assertTrue(worker.is_paused)
        self.assertIsNone(worker.last_error)
        cam.release.assert_called_once()

    def test_loop_rewinds_source_time_and_discards_old_tracking_and_alerts(self):
        worker,cam=self.worker_and_camera()
        worker.source,worker.is_live,worker.loop_video='clip.avi',False,True
        worker.alert_tracks={99:5000}
        worker.pipeline=SimpleNamespace(modules={},flush=Mock(return_value=[]),reset=Mock(),process=Mock(return_value=[]))
        frame=np.full((80,120,3),120,dtype=np.uint8)
        cam.get.return_value=25
        cam.read.side_effect=[(True,frame),(False,None),(True,frame)]
        timestamps=[]
        def process(*args,**kwargs):
            timestamps.append(kwargs['timestamp_ms'])
            if len(timestamps)==2:worker.is_running=False
            return dict(persons=[],source_id=kwargs['source_id'],frame_id=kwargs['frame_id'],timestamp_ms=kwargs['timestamp_ms'])
        worker.core.process.side_effect=process
        with patch.object(server,'open_capture',return_value=cam),patch.object(server.time,'sleep'):
            worker._run_loop(worker.generation_id)
        self.assertEqual(timestamps,[40,40]);worker.core.reset.assert_called_once()
        worker.pipeline.reset.assert_called_once_with(worker.source_id)
        self.assertEqual(worker.alert_tracks,{})
        self.assertEqual(len(worker.annotations.frames),1)
