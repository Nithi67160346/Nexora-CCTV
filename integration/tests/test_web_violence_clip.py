from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi import HTTPException

from web import server
from web.violence_clip import ViolenceClipService, resolve_clip, sample_frames


def wait(service, job_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service.snapshot(job_id)
        if job['state'] != 'running':
            return job
        time.sleep(.005)
    raise AssertionError('Job did not finish')


class ViolenceClipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root/'best_lstm_model.pth').write_bytes(b'fixture')
        self.source = self.root/'uploads/test.mp4'
        self.source.parent.mkdir()
        self.source.write_bytes(b'fixture clip')

    def test_samples_sixteen_known_positions_without_decoding_whole_clip(self):
        cap = Mock()
        cap.get.side_effect = lambda field: 301 if field == cv2.CAP_PROP_FRAME_COUNT else 30
        cap.read.return_value = (True, np.ones((10, 10, 3), dtype=np.uint8))
        frames, metadata = sample_frames(self.source, capture=lambda path: cap)
        self.assertEqual(len(frames), 16)
        self.assertEqual(cap.read.call_count, 16)
        self.assertEqual(metadata['sampled_frame_indices'], np.linspace(0, 300, 16, dtype=int).tolist())
        cap.release.assert_called_once()

    def test_bad_decode_is_error_instead_of_black_frames_or_normal_result(self):
        cap = Mock()
        cap.get.side_effect = lambda field: 100 if field == cv2.CAP_PROP_FRAME_COUNT else 30
        cap.read.return_value = (False, None)
        with self.assertRaises(ValueError):
            sample_frames(self.source, capture=lambda path: cap)
        cap.release.assert_called_once()

    def test_invalid_timing_or_too_short_clip_is_rejected(self):
        for total, fps in [(5, 30), (100, 0), (100, float('nan')), (float('inf'), 30)]:
            cap = Mock()
            cap.get.side_effect = lambda field: total if field == cv2.CAP_PROP_FRAME_COUNT else fps
            with self.assertRaises(ValueError):
                sample_frames(self.source, capture=lambda path: cap)
            self.assertFalse(cap.read.called)
            cap.release.assert_called_once()

    def test_clip_paths_reject_live_sources_and_files_outside_upload_directory(self):
        self.assertEqual(resolve_clip(str(self.source), [self.source.parent]), self.source)
        for value in ['0', 'https://example.com/video.mp4', str(self.root/'best_lstm_model.pth')]:
            with self.assertRaises(ValueError):
                resolve_clip(value, [self.source.parent])

    def test_background_result_keeps_original_clip_and_does_not_add_events(self):
        entered, release = threading.Event(), threading.Event()
        def predict(source, device, progress):
            entered.set()
            release.wait(2)
            return {'probability_fighting': .7, 'fighting': True, 'person_ids': [], 'event_intervals': []}
        service = ViolenceClipService(self.root, predictor=predict)
        event_count = len(server.worker.events)
        job = service.start(self.source)
        self.assertTrue(entered.wait(2))
        try:
            with self.assertRaises(RuntimeError):
                service.start(self.source)
        finally:
            release.set()
        complete = wait(service, job['id'])
        self.assertEqual(complete['filename'], 'test.mp4')
        self.assertTrue(complete['result']['fighting'])
        self.assertEqual(len(server.worker.events), event_count)
        complete['result']['fighting'] = False
        self.assertTrue(service.snapshot(job['id'])['result']['fighting'])

    def test_failed_model_or_changed_clip_has_no_success_result(self):
        def changed(source, device, progress):
            source.write_bytes(b'changed clip contents')
            return {'fighting': False}
        service = ViolenceClipService(self.root, predictor=changed)
        job = wait(service, service.start(self.source)['id'])
        self.assertEqual(job['state'], 'error')
        self.assertIsNone(job['result'])
        service = ViolenceClipService(self.root, predictor=Mock(side_effect=ValueError('weights mismatch')))
        job = wait(service, service.start(self.source)['id'])
        self.assertEqual(job['error'], 'weights mismatch')
        self.assertIsNone(job['result'])

    def test_cuda_unavailable_never_falls_back_or_downloads(self):
        import torch
        service = ViolenceClipService(self.root)
        with patch.object(torch.cuda, 'is_available', return_value=False), patch('web.violence_clip.build_model') as build:
            with self.assertRaisesRegex(ValueError, 'GPU'):
                service.predict(self.source, 'cuda', lambda *args: None)
        build.assert_not_called()

    def test_api_binds_job_and_rejects_unknown_job_or_invalid_source(self):
        service = ViolenceClipService(self.root, predictor=lambda *args: {'fighting': False})
        with patch.object(server, 'violence_clip_service', service), patch.object(server, 'SETTINGS_DIR', self.root):
            job = server.start_violence_clip(server.ViolenceClipRequest(source=str(self.source)))
            self.assertEqual(server.get_violence_clip(job['id'])['filename'], 'test.mp4')
            with self.assertRaises(HTTPException) as error:
                server.get_violence_clip('other-clip')
            self.assertEqual(error.exception.status_code, 404)
            with self.assertRaises(HTTPException) as error:
                server.start_violence_clip(server.ViolenceClipRequest(source='0'))
            self.assertEqual(error.exception.status_code, 400)
        wait(service, job['id'])

    def test_missing_weights_and_unsupported_device_are_actionable(self):
        service = ViolenceClipService(self.root)
        with self.assertRaises(ValueError):
            service.start(self.source, 'other')
        (self.root/'best_lstm_model.pth').unlink()
        self.assertFalse(service.options()['available'])
        with self.assertRaisesRegex(ValueError, 'best_lstm_model'):
            service.start(self.source)

    def test_report_is_downloadable_and_never_exports_an_unfinished_job(self):
        import json
        service = Mock()
        job = dict(id='a'*32, state='completed', result={'fighting':False}, filename='original.mp4')
        service.snapshot.return_value = job
        with patch.object(server, 'violence_clip_service', service):
            response = server.export_violence_clip(job['id'])
            self.assertEqual(json.loads(response.body)['filename'], 'original.mp4')
            self.assertIn('attachment;', response.headers['content-disposition'])
            service.snapshot.return_value = dict(job, state='running', result=None)
            with self.assertRaises(HTTPException) as error:
                server.export_violence_clip(job['id'])
            self.assertEqual(error.exception.status_code, 409)
