from unittest.mock import patch
import unittest
import tempfile
import cv2
import numpy as np
from types import SimpleNamespace
from fastapi.testclient import TestClient

from fastapi import HTTPException
from web import server
from web.browser_capture import BrowserCapture
from web.media_store import MediaStore


class BrowserWebcamRoutingTests(unittest.TestCase):
    def test_stale_primary_stop_cannot_stop_a_new_source(self):
        with patch.object(server.worker, 'session_id', 'new'), patch.object(server.worker, 'stop_stream') as stop:
            with self.assertRaises(HTTPException) as raised:
                server.stop_stream({'session_id':'old'})
            self.assertEqual(raised.exception.status_code, 409)
            stop.assert_not_called()
            server.stop_stream({'session_id':'new'})
            stop.assert_called_once()

    def test_primary_final_recording_uses_primary_room_and_rejects_old_session(self):
        target = SimpleNamespace(session_id='current', source_id='room')
        client = TestClient(server.app)
        with tempfile.TemporaryDirectory() as directory:
            store = MediaStore(directory)
            with patch.object(server, 'worker', target), patch.object(server, 'media_store', store):
                data = {'session_id':'current', 'start_s':0, 'end_s':2}
                # A synthetic EBML fixture tests routing/storage, not playable video quality.
                files = {'file':('segment.webm', b'\x1aE\xdf\xa3fixture', 'video/webm')}
                response = client.post('/api/stream/recording', data=data, files=files)
                self.assertEqual(response.status_code, 200, response.text)
                row = response.json()
                self.assertEqual(row['source_id'], 'room')
                self.assertEqual(row['session_id'], 'current')
                self.assertEqual(store.path(row['id']).read_bytes(), files['file'][1])
                data['session_id'] = 'old'
                self.assertEqual(client.post('/api/stream/recording', data=data, files=files).status_code, 409)
                self.assertEqual(len(store.list()), 1)

    def test_main_start_creates_browser_capture_without_opening_server_camera(self):
        with patch.object(server.worker, 'start_stream') as start, \
             patch.object(server.streams, 'active', return_value=False):
            result = server.start_stream(server.StartStreamRequest(source='browser://device', device='cpu',
                        review_mode=True, browser_width=120, browser_height=80))
        self.assertEqual(result['source'], 'browser://device')
        capture = start.call_args.kwargs['browser_capture']
        self.assertIsInstance(capture, BrowserCapture)
        self.assertEqual((capture.width, capture.height), (120, 80))
        self.assertFalse(start.call_args.kwargs['review_mode'])

    def test_main_browser_capture_rejects_bad_dimensions_and_gpu_decode_before_start(self):
        for options in ({'browser_width':0}, {'browser_height':5000}, {'decode_device':'cuda'}):
            with patch.object(server.worker, 'start_stream') as start:
                with self.assertRaises(HTTPException) as raised:
                    server.start_stream(server.StartStreamRequest(source='browser://device', device='cpu', **options))
                self.assertEqual(raised.exception.status_code, 400)
                start.assert_not_called()

    def test_main_frames_reach_primary_worker_and_reject_stale_or_stopped_session(self):
        capture = BrowserCapture(120, 80)
        target = SimpleNamespace(session_id='main-session', browser_capture=capture)
        ok, encoded = cv2.imencode('.jpg', np.full((80, 120, 3), 120, dtype=np.uint8))
        self.assertTrue(ok)
        client = TestClient(server.app)
        with patch.object(server, 'worker', target), patch.object(server.streams, 'get') as child:
            data = {'session_id':'main-session', 'timestamp_ms':1}
            files = {'file':('frame.jpg', encoded.tobytes(), 'image/jpeg')}
            response = client.post('/api/stream/frame', data=data, files=files)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(capture.read()[0])
            data['session_id'] = 'old'
            self.assertEqual(client.post('/api/stream/frame', data=data, files=files).status_code, 409)
            data['session_id'] = 'main-session'; capture.release()
            self.assertEqual(client.post('/api/stream/frame', data=data, files=files).status_code, 400)
            child.assert_not_called()

    def test_docker_numeric_camera_without_device_is_rejected_before_stream_changes(self):
        with patch.object(server.Path, 'exists', side_effect=[True, False]), \
             patch.object(server.worker, 'start_stream') as start:
            with self.assertRaises(HTTPException) as raised:
                server.start_stream(server.StartStreamRequest(source='0', device='cpu'))
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn('ค้นหา webcam', raised.exception.detail)
        start.assert_not_called()

    def test_browser_source_does_not_require_container_usb_device(self):
        with patch.object(server.Path, 'exists') as exists:
            server.check_server_webcam('browser://cam')
        exists.assert_not_called()

    def test_native_or_explicitly_mounted_server_camera_is_allowed(self):
        for present in ([False], [True, True]):
            with patch.object(server.Path, 'exists', side_effect=present):
                server.check_server_webcam('0')

    def test_uploaded_jpeg_reaches_the_browser_capture_and_old_sessions_are_rejected(self):
        capture = BrowserCapture(120, 80)
        target = SimpleNamespace(session_id='current', browser_capture=capture)
        image = np.full((80, 120, 3), 120, dtype=np.uint8)
        ok, encoded = cv2.imencode('.jpg', image)
        self.assertTrue(ok)
        client = TestClient(server.app)
        with patch.object(server.streams, 'get', return_value=target):
            response = client.post('/api/multistream/cam/frame', data={'session_id':'current', 'timestamp_ms':1},
                                   files={'file':('frame.jpg', encoded.tobytes(), 'image/jpeg')})
            self.assertEqual(response.status_code, 200, response.text)
            ready, frame = capture.read()
            self.assertTrue(ready)
            self.assertEqual(frame.shape, image.shape)
            response = client.post('/api/multistream/cam/frame', data={'session_id':'old', 'timestamp_ms':2},
                                   files={'file':('frame.jpg', encoded.tobytes(), 'image/jpeg')})
            self.assertEqual(response.status_code, 409)
            response = client.post('/api/multistream/cam/frame', data={'session_id':'current', 'timestamp_ms':2},
                                   files={'file':('frame.jpg', b'bad', 'image/jpeg')})
            self.assertEqual(response.status_code, 400)
