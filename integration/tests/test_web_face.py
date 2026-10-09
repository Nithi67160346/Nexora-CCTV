import asyncio
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
from fastapi import HTTPException, UploadFile
import numpy as np

from web import server
from web.face_service import FaceService
from Location.location.face.face_detector import FaceDetection, create_face_detector
from Location.location.face.face_embedder import create_face_embedder
from Location.location.face.identity_matcher import IdentityMatcher
from Location.location.face.face_database import InMemoryFaceDatabase


class WebFaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.service = FaceService(self.temp.name)
        self.worker = server.StreamWorker()
        self.worker.face_service = self.service
        self.worker.config = self.worker._product_config(self.worker.config)
        self.service.detector_path.parent.mkdir(parents=True)
        self.service.detector_path.touch(); self.service.embedder_path.touch()

    def save_person(self, mode='off'):
        db, profiles, _ = self.service.read()
        db.add_identity('test_person', [np.eye(128, dtype=np.float32)[0]] * 3, 'ผู้ทดสอบ')
        profiles['test_person'] = dict(name='ผู้ทดสอบ', role='resident', assigned_bed='bed_01', consent_at='test')
        self.service.save(db, profiles, mode)

    def test_atomic_local_store_preserves_settings_and_names_after_restart(self):
        self.save_person('recognize')
        restarted = FaceService(self.temp.name)
        db, profiles, mode = restarted.read()
        self.assertEqual(mode, 'recognize')
        self.assertEqual(db.display_name('test_person'), 'ผู้ทดสอบ')
        cfg = restarted.location_config(dict(zones={'bed_01': {}}))
        self.assertEqual(cfg['residents']['test_person']['assigned_bed'], 'bed_01')
        self.assertIn('bed_01', cfg['zones'])
        self.assertTrue(cfg['privacy']['face_recognition_enabled'])
        self.assertEqual([p.name for p in self.service.path.parent.iterdir()], ['face_db.npz'])

    def test_unregistered_and_ambiguous_faces_are_not_forced_to_a_name(self):
        db = InMemoryFaceDatabase()
        db.add_identity('a', [np.eye(128)[0]])
        matcher = IdentityMatcher(db, .7, .03)
        self.assertIsNone(matcher.match(np.eye(128)[1]).resident_id)
        db.add_identity('b', [np.eye(128)[0]])
        self.assertIsNone(matcher.match(np.eye(128)[0]).resident_id)

    def test_consent_and_image_count_rejected_before_models_or_store(self):
        with patch('web.face_service.create_face_detector') as detector:
            with self.assertRaises(ValueError):
                self.service.prepare_enrollment('A', 'resident', '', False, [b'x'] * 3)
            with self.assertRaises(ValueError):
                self.service.prepare_enrollment('A', 'resident', '', True, [b'x'])
        detector.assert_not_called(); self.assertFalse(self.service.path.exists())

    def encoded_images(self, count=3):
        return [cv2.imencode('.png', np.random.default_rng(i).integers(0, 255, (100, 100, 3), dtype=np.uint8))[1].tobytes() for i in range(count)]

    def mocks(self):
        detector = SimpleNamespace(detect_in_person=lambda frame, box: [FaceDetection((0, 0, 90, 90), .99)], best=lambda faces: faces[0] if faces else None)
        embedder = SimpleNamespace(embed=lambda frame, face: np.eye(128, dtype=np.float32)[0])
        return detector, embedder

    def test_duplicate_photos_do_not_count_as_three_samples(self):
        detector, embedder = self.mocks(); image = self.encoded_images(1)[0]
        with patch('web.face_service.create_face_detector', return_value=detector), patch('web.face_service.create_face_embedder', return_value=embedder):
            with self.assertRaisesRegex(ValueError, 'รูปซ้ำ'):
                self.service.prepare_enrollment('A', 'resident', '', True, [image] * 3)
        self.assertFalse(self.service.path.exists())

    def test_multi_face_photos_cannot_register_a_mixed_identity(self):
        detector, embedder = self.mocks()
        detector.detect_in_person = lambda frame, box: [FaceDetection((0, 0, 40, 40), .99)] * 2
        with patch('web.face_service.create_face_detector', return_value=detector), patch('web.face_service.create_face_embedder', return_value=embedder):
            with self.assertRaisesRegex(ValueError, 'พบ 2 หน้า'):
                self.service.prepare_enrollment('A', 'resident', '', True, self.encoded_images())
        self.assertFalse(self.service.path.exists())

    def test_web_enrollment_stores_only_embeddings_and_profile(self):
        detector, embedder = self.mocks()
        files = [UploadFile(file=io.BytesIO(image), filename=f'{i}.png') for i, image in enumerate(self.encoded_images())]
        with patch.object(server, 'worker', self.worker), patch('web.face_service.create_face_detector', return_value=detector), patch('web.face_service.create_face_embedder', return_value=embedder):
            result = asyncio.run(server.enroll_face('ผู้ทดสอบ', 'caregiver', '', True, files))
        self.assertEqual(result['accepted'], 3)
        self.assertTrue(all(file.file.closed for file in files))
        self.assertEqual(self.service.read()[1][result['identity_id']]['role'], 'caregiver')
        self.assertEqual([p.name for p in self.service.path.parent.iterdir()], ['face_db.npz'])

    def test_failed_runtime_reload_rolls_back_local_enrollment_file_and_config(self):
        self.save_person(); old_file = self.service.snapshot(); old_config = deepcopy(self.worker.config)
        with patch.object(server, 'worker', self.worker), patch.object(server.ProgressPipeline, 'from_config', side_effect=ValueError('model failed')):
            with self.assertRaises(HTTPException):
                server.set_face_mode(server.FaceModeRequest(mode='detect'))
        self.assertEqual(self.service.snapshot(), old_file)
        self.assertEqual(self.worker.config, old_config)

    def test_missing_models_or_empty_database_cannot_report_recognition_success(self):
        with patch.object(server, 'worker', self.worker):
            with self.assertRaises(HTTPException): server.set_face_mode(server.FaceModeRequest(mode='recognize'))
            self.service.detector_path.unlink()
            with self.assertRaises(HTTPException): server.set_face_mode(server.FaceModeRequest(mode='detect'))
        self.assertFalse(self.service.path.exists())

    def test_delete_last_person_exits_recognition_and_clears_runtime_identity(self):
        self.save_person('recognize')
        location = SimpleNamespace(face=SimpleNamespace(recognition_enabled=False))
        replacement = SimpleNamespace(modules={'location': location})
        with patch.object(server, 'worker', self.worker), patch.object(server.ProgressPipeline, 'from_config', return_value=replacement):
            status = server.remove_face('test_person')
        self.assertEqual(status['registered_count'], 0)
        self.assertEqual(status['mode'], 'detect')
        self.assertFalse(status['recognition_active'])

    def test_source_change_resets_identity_state_even_when_track_id_is_reused(self):
        reset_calls = []
        self.worker.pipeline = SimpleNamespace(flush=lambda: [], reset=lambda source: reset_calls.append(source))
        self.worker.core = SimpleNamespace(reset=lambda: None)
        with patch.object(server.threading, 'Thread', return_value=SimpleNamespace(start=lambda: None)):
            self.worker.start_stream('0')
        self.assertEqual(reset_calls, ['cam_01'])

    def test_event_name_evidence_survives_ended_update(self):
        event = dict(event_id='a', source_id='cam_01', event_type='fall_detected', track_ids=[7],
            metadata=dict(people=[dict(track_id=7, display_name='ผู้ทดสอบ')], lifecycle='started'))
        self.worker._record_updates([event])
        self.worker._record_updates([dict(event, metadata=dict(lifecycle='ended'))])
        self.assertEqual(self.worker.events[0]['metadata']['people'][0]['display_name'], 'ผู้ทดสอบ')

    def test_real_opencv_models_load_and_process_synthetic_input(self):
        service = FaceService(server.ROOT)
        if not service.detector_path.is_file() or not service.embedder_path.is_file():
            self.skipTest('Optional pretrained face models not installed')
        detector = create_face_detector(dict(backend='yunet', model_path=str(service.detector_path)))
        self.assertEqual(detector.detect_in_person(np.zeros((320, 320, 3), np.uint8), (0, 0, 320, 320)), [])
        embedder = create_face_embedder(dict(embedder='sface', embedder_model_path=str(service.embedder_path)))
        vector = embedder.embed(np.zeros((112, 112, 3), np.uint8), FaceDetection((0, 0, 112, 112), .99))
        self.assertEqual(vector.shape, (128,)); self.assertTrue(np.isfinite(vector).all())

    def test_registered_identity_reaches_live_status_overlay_and_fall_evidence(self):
        self.save_person('recognize')
        detector, embedder = self.mocks()
        with patch('Location.location.face.create_face_detector', return_value=detector), patch('Location.location.face.create_face_embedder', return_value=embedder):
            self.worker.update_config(self.worker.config)
        frame = self.encoded_images(1)[0]
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        for i in range(7):
            ctx = dict(source_id='cam_01', frame_id=i, timestamp_ms=i * 100, fps=10,
                persons=[dict(track_id=7, confidence=.99, bbox_xyxy=[0, 0, 100, 100])])
            self.worker._record_updates(self.worker.pipeline.process(frame, ctx))
        self.worker.active_tracks = [dict(track_id=7)]
        with patch.object(server, 'worker', self.worker):
            status = server.get_status()
        self.assertTrue(status['face']['recognition_active'])
        self.assertEqual(status['people'][0]['identity_name'], 'ผู้ทดสอบ')
        self.assertEqual(status['people'][0]['person_role'], 'resident')
        event = dict(event_id='fall', event_type='fall_detected', source_id='cam_01', track_ids=[7], metadata={})
        self.worker._record_updates([event])
        self.assertEqual(self.worker.events[0]['metadata']['people'][0]['display_name'], 'ผู้ทดสอบ')
        self.worker._render_overlay(frame.copy(), ctx, [])
        self.worker.pipeline.reset('cam_01')
        self.assertIsNone(self.worker.pipeline.modules['location'].get_identity(7, 'cam_01'))
