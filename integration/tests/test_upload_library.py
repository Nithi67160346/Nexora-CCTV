import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import threading
from fastapi.testclient import TestClient
from web import server


class UploadLibraryTests(unittest.TestCase):
    def test_delete_only_selected_upload_and_clear_stale_profile_source(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'SETTINGS_DIR', Path(directory)):
            uploads = Path(directory) / 'uploads'
            uploads.mkdir()
            clip, keep = uploads / 'selected.mp4', uploads / 'keep.mp4'
            clip.write_bytes(b'fixture'); keep.write_bytes(b'other fixture')
            target = server.StreamWorker()
            target.source = str(clip)
            target.cameras.profiles[target.cameras.active_id]['source'] = str(clip)
            with patch.object(server, 'worker', target), patch.object(server.streams, 'entries', {}):
                response = TestClient(server.app).request('DELETE', '/api/uploads', json={'source': str(clip)})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse(clip.exists()); self.assertTrue(keep.exists())
                self.assertEqual(target.source, '')
                self.assertEqual(target.cameras.profiles[target.cameras.active_id]['source'], '')
                self.assertEqual([row['filename'] for row in server.list_uploaded_clips()], ['keep.mp4'])
                self.assertEqual(TestClient(server.app).request('DELETE', '/api/uploads', json={'source': str(clip)}).status_code, 404)

    def test_reject_outside_nested_unrelated_and_live_sources(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'SETTINGS_DIR', Path(directory)):
            root = Path(directory); uploads = root / 'uploads'; uploads.mkdir()
            outside = root / 'outside.mp4'; outside.write_bytes(b'keep')
            notes = uploads / 'notes.md'; notes.write_bytes(b'keep')
            nested = uploads / 'nested'; nested.mkdir(); clip = nested / 'clip.mp4'; clip.write_bytes(b'keep')
            for source in (str(outside), str(notes), str(clip), 'browser://device'):
                reply = TestClient(server.app).request('DELETE', '/api/uploads', json={'source': source})
                self.assertEqual(reply.status_code, 400, reply.text)
            self.assertTrue(outside.exists()); self.assertTrue(notes.exists()); self.assertTrue(clip.exists())

    def test_active_main_multi_starting_and_clip_test_cannot_be_deleted(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'SETTINGS_DIR', Path(directory)):
            uploads = Path(directory) / 'uploads'; uploads.mkdir()
            clip = uploads / 'active.mp4'; clip.write_bytes(b'keep')
            target = server.StreamWorker(); target.source = str(clip); target.is_running = True
            client = TestClient(server.app)
            with patch.object(server, 'worker', target), patch.object(server.streams, 'entries', {}):
                self.assertEqual(client.request('DELETE', '/api/uploads', json={'source': str(clip)}).status_code, 409)
                target.is_running = False
                child = SimpleNamespace(source=str(clip), is_running=False, thread=None)
                with patch.object(server.streams, 'entries', {'room': dict(worker=child, state='starting')}):
                    self.assertEqual(client.request('DELETE', '/api/uploads', json={'source': str(clip)}).status_code, 409)
                with patch.object(server.violence_clip_service, 'active', 'job'), patch.object(server.violence_clip_service, 'jobs', {'job': {'filename': clip.name}}):
                    self.assertEqual(client.request('DELETE', '/api/uploads', json={'source': str(clip)}).status_code, 409)
                self.assertTrue(clip.exists())

    def test_delete_waits_for_primary_start_lifecycle_and_then_rejects_active_clip(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'SETTINGS_DIR', Path(directory)):
            uploads = Path(directory) / 'uploads'; uploads.mkdir()
            clip = uploads / 'starting.mp4'; clip.write_bytes(b'keep')
            target = server.StreamWorker()
            reply = []; entered = threading.Event(); finished = threading.Event()
            def attempt():
                entered.set()
                reply.append(TestClient(server.app).request('DELETE', '/api/uploads', json={'source': str(clip)}))
                finished.set()
            with patch.object(server, 'worker', target), patch.object(server.streams, 'entries', {}):
                with target.lifecycle_lock:
                    request = threading.Thread(target=attempt)
                    request.start(); self.assertTrue(entered.wait(1))
                    self.assertFalse(finished.wait(.05), 'delete must wait until starting source becomes visible')
                    target.source = str(clip); target.is_running = True
                request.join(timeout=3)
                self.assertTrue(finished.is_set())
                self.assertEqual(reply[0].status_code, 409)
                self.assertTrue(clip.exists())

    def test_uploaded_clips_remain_selectable_with_duplicate_names(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'SETTINGS_DIR', Path(directory)):
            client = TestClient(server.app)
            for name in ('clip.mp4', 'clip.mp4', 'คลิป ทดสอบ.avi'):
                reply = client.post('/api/upload', files={'file': (name, b'short synthetic fixture', 'video/mp4')})
                self.assertEqual(reply.status_code, 200, reply.text)
            first = client.get('/api/uploads').json()
            second = TestClient(server.app).get('/api/uploads').json()
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)
            self.assertEqual(len({row['path'] for row in first}), 3)
            self.assertEqual(sorted(row['filename'] for row in first), ['clip.mp4', 'clip.mp4', 'คลิป ทดสอบ.avi'])
            for row in first:
                self.assertTrue(Path(row['path']).is_relative_to(Path(directory) / 'uploads'))

    def test_list_excludes_empty_unrelated_and_nested_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'SETTINGS_DIR', Path(directory)):
            self.assertEqual(server.list_uploaded_clips(), [])
            uploads = Path(directory) / 'uploads'
            uploads.mkdir()
            (uploads / '1234abcd_kept.MP4').write_bytes(b'fixture')
            (uploads / 'empty.avi').touch()
            (uploads / 'in-progress.mp4.part').write_bytes(b'partial upload')
            (uploads / 'notes.md').write_text('not a clip')
            (uploads / 'nested').mkdir()
            (uploads / 'nested' / 'other.mp4').write_bytes(b'fixture')
            self.assertEqual([row['filename'] for row in server.list_uploaded_clips()], ['kept.MP4'])

    def test_rejected_upload_does_not_add_library_item(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'SETTINGS_DIR', Path(directory)):
            client = TestClient(server.app)
            self.assertEqual(client.post('/api/upload', files={'file': ('empty.mp4', b'', 'video/mp4')}).status_code, 400)
            self.assertEqual(client.post('/api/upload', files={'file': ('notes.txt', b'text', 'text/plain')}).status_code, 400)
            self.assertEqual(server.list_uploaded_clips(), [])
