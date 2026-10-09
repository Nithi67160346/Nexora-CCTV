"""QA feedback contracts, without weights, camera hardware or long inference."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
from web import server
from web.browser_capture import BrowserCapture
from web.event_view import localized_event
from web.media_store import MediaStore
from web.multistream import StreamRegistry
from web.playback import AnnotationBuffer
from web.recording_capture import RecordingCapture


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=MediaStore(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def segment(self,source='cam1',session='a',start=0,end=10):
        row=self.store.begin(source,session,start,'.webm')
        (self.store.directory/row['filename']).write_bytes(b'\x1aE\xdf\xa3payload')
        return self.store.finish(row,end)
    def test_event_uses_same_camera_session_and_time(self):
        row=self.segment()
        self.segment(session='new')
        event=dict(source_id='cam1',end_timestamp_ms=4500,metadata={'recording_session':'a'})
        self.assertEqual(self.store.for_event(event)['recording']['id'],row['id'])
        self.assertEqual(self.store.for_event(event)['time_s'],4.5)
        event['source_id']='cam2'
        self.assertIsNone(self.store.for_event(event))
    def test_retention_expires_24h_not_23h_and_orphan_but_not_user_file(self):
        old=self.segment(); recent=self.segment(session='new')
        p=self.store.directory/(old['id']+'.json')
        data=json.loads(p.read_text());data['created_at']=time.time()-25*3600;p.write_text(json.dumps(data))
        p=self.store.directory/(recent['id']+'.json')
        data=json.loads(p.read_text());data['created_at']=time.time()-23*3600;p.write_text(json.dumps(data))
        orphan=self.store.directory/('a'*32+'.avi');orphan.write_bytes(b'orphan');os.utime(orphan,(0,0))
        user=self.store.directory/'holiday.mp4';user.write_bytes(b'keep');os.utime(user,(0,0))
        self.store.cleanup()
        self.assertFalse((self.store.directory/old['filename']).exists())
        self.assertTrue((self.store.directory/recent['filename']).exists())
        self.assertFalse(orphan.exists());self.assertTrue(user.exists())
    def test_budget_rejects_new_recording_without_deleting_recent(self):
        row=self.segment();self.store.max_bytes=1
        with self.assertRaisesRegex(ValueError,'พื้นที่'): self.store.begin('cam','session',0,'.webm')
        self.assertTrue(self.store.path(row['id']).is_file())
    def test_path_traversal_and_corrupt_manifests_are_rejected(self):
        with self.assertRaises(ValueError): self.store.path('../secret')
        (self.store.directory/'broken.json').write_text('{')
        (self.store.directory/'wrong.json').write_text('[]')
        self.assertEqual(self.store.list(),[])
    def test_camera_recording_survives_warmup_and_does_not_wait_for_ai(self):
        class Camera:
            count=0; releases=0
            def isOpened(self): return True
            def get(self,key): return 30
            def getBackendName(self): return 'TEST'
            def read(self):
                self.count+=1
                time.sleep(.002)
                if self.count<=2:return False,None
                if self.count<=32:return True,np.full((48,64,3),self.count*3,np.uint8)
                return False,None
            def release(self): self.releases+=1
        camera=Camera()
        with patch('web.recording_capture.ffmpeg_binary',return_value=None):
            cap=RecordingCapture(camera,self.store,'cam','a')
            cap.thread.join(timeout=4)
            self.assertFalse(cap.thread.is_alive())
            self.assertEqual(cap.queue.qsize(),1) # no AI consumer was needed
            self.assertIsNone(cap.recording_error)
            rows=self.store.list();self.assertEqual(len(rows),1)
            recorded=cv2.VideoCapture(str(self.store.path(rows[0]['id'])))
            try:self.assertEqual(int(recorded.get(cv2.CAP_PROP_FRAME_COUNT)),30)
            finally:recorded.release()
            self.assertEqual(camera.releases,1)
            self.assertTrue(cap.read()[0]) # final frame remains readable
            cap.release()


class EvidenceTests(unittest.TestCase):
    def test_browser_drops_old_frames_but_keeps_monotonic_source_clock(self):
        cap=BrowserCapture(64,48)
        frame=np.zeros((48,64,3),np.uint8)
        cap.submit(frame,100);cap.submit(frame+1,200)
        ok,current=cap.read();self.assertTrue(ok);self.assertEqual(current[0,0,0],1)
        self.assertEqual(cap.timestamp_ms,200)
        with self.assertRaises(ValueError): cap.submit(frame,150)
        cap.release()
        with self.assertRaises(ValueError): cap.submit(frame,300)
    def test_evidence_is_bounded_retained_alerts_are_red_and_never_relabel_time(self):
        buffer=AnnotationBuffer(2)
        for i in range(3):
            buffer.add(dict(source_id='cam',frame_id=i,timestamp_ms=i*1000,persons=[dict(track_id=1,bbox_xyxy=[0,0,20,30])]),None,[],{1})
        self.assertEqual(buffer.near(0),[])
        row=buffer.near(2)[0];self.assertTrue(row['persons'][0]['alert'])
        row['persons'][0]['alert']=False
        self.assertTrue(buffer.near(2)[0]['persons'][0]['alert'])
    def test_event_reference_is_immutable_when_room_changes_clip(self):
        event=dict(event_id='x',event_type='fall_detected',source_id='cam',metadata={})
        first=localized_event(event,'first.mp4','a')
        second=localized_event(first,'second.mp4','b')
        self.assertEqual(second['metadata']['playback_source'],'first.mp4')
        self.assertEqual(second['metadata']['recording_session'],'a')
        self.assertIn('ล้ม',second['message'])
        self.assertEqual(event['metadata'],{})


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.parent=SimpleNamespace(is_running=False,source_id='main',source='0',_record_updates=lambda _:None)
        class Child:
            def __init__(self):
                self.is_running=False;self.pipeline=object();self.core=object()
                self.cameras=SimpleNamespace(profiles={},active_id=None)
            def _product_config(self,config,cid):return dict(config)
            def start_stream(self,source,**options):self.source=source;self.is_running=True
            def stop_stream(self):self.is_running=False
        self.registry=StreamRegistry(Child,self.parent,2)
    def start(self,cid,source):
        result=self.registry.start(dict(id=cid,config={}),source)
        self.registry.entries[cid]['thread'].join(timeout=1)
        return result
    def test_independent_trackers_and_bounded_sources(self):
        self.start('one','first.mp4');self.start('two','second.mp4')
        self.assertIsNot(self.registry.get('one').core,self.registry.get('two').core)
        self.assertIsNot(self.registry.get('one').pipeline,self.registry.get('two').pipeline)
        with self.assertRaises(ValueError):self.start('three','third.mp4')
        self.registry.stop('one');self.assertTrue(self.registry.get('two').is_running)
        self.assertIsNone(self.registry.get('one').core)
        self.assertIsNone(self.registry.get('one').pipeline)
        self.start('three','third.mp4');self.registry.stop_all()
    def test_same_physical_webcam_cannot_be_opened_twice(self):
        self.parent.is_running=True
        with self.assertRaisesRegex(ValueError,'webcam'):self.start('two','0')


class FeedbackApiTests(unittest.TestCase):
    def setUp(self):self.client=TestClient(server.app)
    def test_event_filters_hide_last_seen_by_default_and_thai_text(self):
        events=[dict(event_id='a',source_id='cam',event_type='last_seen_update',metadata={}),dict(event_id='b',source_id='cam',event_type='fall_detected',metadata={})]
        with patch.object(server.worker,'events',events):
            response=self.client.get('/api/events');self.assertEqual(response.status_code,200)
            self.assertEqual([e['event_id'] for e in response.json()],['b'])
            self.assertIn('ล้ม',response.json()[0]['message'])
            self.assertEqual(len(self.client.get('/api/events?include_last_seen=true').json()),2)
            self.assertEqual(self.client.get('/api/events?category=location').json(),[])
    def test_webm_recording_rejects_other_formats_and_accepts_same_session(self):
        with tempfile.TemporaryDirectory() as temp:
            store=MediaStore(temp);target=SimpleNamespace(session_id='same')
            with patch.object(server,'media_store',store),patch.object(server.streams,'get',return_value=target):
                fields=dict(session_id='same',start_s='0',end_s='10')
                url='/api/multistream/cam/recording'
                self.assertEqual(self.client.post(url,data=fields,files={'file':('clip.webm',b'invalid')}).status_code,400)
                self.assertEqual(store.list(),[])
                fields['session_id']='old'
                self.assertEqual(self.client.post(url,data=fields,files={'file':('clip.webm',b'\x1aE\xdf\xa3payload')}).status_code,409)
                fields['session_id']='same'
                response=self.client.post(url,data=fields,files={'file':('clip.webm',b'\x1aE\xdf\xa3payload')})
                self.assertEqual(response.status_code,200)
                media=self.client.get('/api/recordings/'+response.json()['id']+'/media',headers={'Range':'bytes=0-3'})
                self.assertEqual(media.status_code,206);self.assertEqual(media.content,b'\x1aE\xdf\xa3')
    def test_active_multistream_profiles_cannot_be_deleted_or_replaced(self):
        with patch.object(server.streams,'active',return_value=True):
            self.assertEqual(self.client.delete('/api/cameras/cam').status_code,409)
            self.assertEqual(self.client.post('/api/cameras/import',json={}).status_code,409)
    def test_gpu_request_without_gpu_has_actionable_error(self):
        with patch.object(server,'_preferred_device',return_value='cpu'):
            response=self.client.post('/api/stream/start',json={'source':'0','device':'cuda'})
            self.assertEqual(response.status_code,400)
            self.assertIn('GPU',response.json()['detail'])


if __name__=='__main__':unittest.main()
