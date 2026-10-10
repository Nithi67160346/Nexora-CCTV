import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import cv2
import numpy as np
from fastapi import HTTPException

from web.frame_playback import FramePlayback
from web import server


class FramePlaybackTests(unittest.TestCase):
    def setUp(self):
        self.player = FramePlayback()
        self.player.reset('session', 'client')
        self.producer = None

    def tearDown(self):
        self.player.cancel()
        if self.producer:
            self.producer.join(2)
            self.assertFalse(self.producer.is_alive())

    def wait_frame(self, after=0):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            frame = self.player.read('session', 'client', after)
            if frame:
                return frame
            time.sleep(.005)
        self.fail('producer did not publish')

    def test_every_frame_waits_for_display_before_next_inference(self):
        inferred = []

        def produce():
            for frame_id in range(1, 5):
                inferred.append(frame_id)
                if not self.player.publish('session', b'jpeg', frame_id, frame_id / 25):
                    break

        self.producer = threading.Thread(target=produce)
        self.producer.start()
        for frame_id in range(1, 5):
            frame = self.wait_frame(frame_id - 1)
            self.assertEqual(frame['frame_id'], frame_id)
            self.assertEqual(frame['seconds'], frame_id / 25)
            self.assertEqual(inferred, list(range(1, frame_id + 1)))
            self.assertEqual(self.player.read('session', 'client', frame_id), None)
            self.player.acknowledge('session', 'client', frame['sequence'])
        self.producer.join(2)
        self.assertFalse(self.producer.is_alive())

    def test_other_tab_and_stale_session_cannot_advance_clip(self):
        self.producer = threading.Thread(target=lambda: self.player.publish('session', b'jpeg', 1, .04))
        self.producer.start()
        self.wait_frame()
        for session, client, sequence in [('old', 'client', 1), ('session', 'other', 1), ('session', 'client', 2)]:
            with self.assertRaises(ValueError):
                self.player.acknowledge(session, client, sequence)
        self.assertEqual(self.player.acknowledged, 0)
        self.assertTrue(self.producer.is_alive())

    def test_stop_and_explicit_seek_release_blocked_producer(self):
        for action in (self.player.discard, self.player.cancel):
            self.player.reset('session', 'client')
            self.producer = threading.Thread(target=lambda: self.player.publish('session', b'jpeg', 1, .04))
            self.producer.start()
            self.wait_frame()
            action()
            self.producer.join(2)
            self.assertFalse(self.producer.is_alive())
            if action == self.player.cancel:
                self.assertFalse(self.player.publish('session', b'old', 2, .08))

    def test_seek_during_inference_discards_old_position(self):
        epoch = self.player.epoch
        self.player.discard()
        self.assertTrue(self.player.publish('session', b'old', 1, .04, epoch))
        self.assertIsNone(self.player.read('session', 'client', 0))


class WorkerFramePlaybackTests(unittest.TestCase):
    def test_worker_checks_every_frame_and_pause_stop_do_not_deadlock(self):
        worker = server.StreamWorker()
        worker.source = 'fixture.avi'
        worker.session_id = 'session'
        worker.frame_by_frame = True
        worker.frame_playback.reset('session', 'client')
        worker.is_running = True
        worker.loop_video = False
        worker.pipeline = SimpleNamespace(modules={}, health={}, process=lambda *args: [], flush=lambda: [])
        worker.core = Mock()
        worker.core.process.side_effect = lambda frame, **kwargs: dict(kwargs, persons=[])
        cap = Mock()
        cap.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS:25, cv2.CAP_PROP_FRAME_COUNT:3,
                                          cv2.CAP_PROP_FRAME_WIDTH:64, cv2.CAP_PROP_FRAME_HEIGHT:48}.get(prop, 0)
        cap.read.side_effect = [(True, np.full((48, 64, 3), i*50, np.uint8)) for i in range(1,4)] + [(False, None)]
        worker.thread = threading.Thread(target=worker._run_capture, args=(0, cap))
        worker.thread.start()
        try:
            for index in range(1, 4):
                deadline = time.monotonic() + 2
                frame = None
                while time.monotonic() < deadline and frame is None:
                    frame = worker.frame_playback.read('session', 'client', index-1)
                    time.sleep(.005)
                self.assertIsNotNone(frame)
                self.assertEqual(frame['frame_id'], index)
                self.assertEqual(worker.core.process.call_count, index)
                decoded = cv2.imdecode(np.frombuffer(frame['jpeg'], np.uint8), cv2.IMREAD_COLOR)
                self.assertIsNotNone(decoded)
                if index == 1:
                    worker.is_paused = True
                    worker.frame_playback.acknowledge('session', 'client', index)
                    time.sleep(.08)
                    self.assertEqual(worker.core.process.call_count, 1)
                    worker.is_paused = False
                else:
                    worker.frame_playback.acknowledge('session', 'client', index)
            deadline = time.monotonic() + 2
            while worker.stage != 'analysis_complete' and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(worker.stage, 'analysis_complete')
            self.assertEqual(worker.analysis_frames, 3)
        finally:
            worker.stop_stream()
        self.assertEqual(worker.stage, 'idle')

    def test_frame_endpoints_return_same_jpeg_and_reject_stale_ack(self):
        worker = server.StreamWorker()
        worker.frame_by_frame = True
        worker.frame_playback.reset('session', 'client')
        thread = threading.Thread(target=lambda: worker.frame_playback.publish('session', b'jpeg', 7, .28))
        thread.start()
        try:
            with patch.object(server, 'worker', worker):
                deadline = time.monotonic() + 2
                while worker.frame_playback.frame is None and time.monotonic() < deadline:
                    time.sleep(.005)
                response = server.playback_frame('session', 'client')
                self.assertEqual(response.body, b'jpeg')
                self.assertEqual(response.headers['x-frame-id'], '7')
                self.assertEqual(server.playback_frame('session', 'client', 1).status_code, 204)
                with self.assertRaises(HTTPException):
                    server.playback_frame_ack(server.FrameAckRequest(session_id='old', client_id='client', sequence=1))
                server.playback_frame_ack(server.FrameAckRequest(session_id='session', client_id='client', sequence=1))
                thread.join(2)
                self.assertFalse(thread.is_alive())
        finally:
            worker.frame_playback.cancel()
            thread.join(2)

    def test_live_camera_cannot_enable_file_frame_playback(self):
        for source in ('browser://camera', 'rtsp://camera'):
            with self.assertRaises(HTTPException) as error:
                server.start_stream(server.StartStreamRequest(source=source, frame_by_frame=True, playback_client='client-123'))
            self.assertEqual(error.exception.status_code, 400)
