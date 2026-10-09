from copy import deepcopy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi import HTTPException

from integration.pipeline import ProgressPipeline
from web import server
from web.camera_profiles import CameraProfiles, apply_zones, polygon
from web.runtime_health import snapshot
from web.qa_service import QaRun, event_metrics


def run(**overrides):
    values = dict(source='test.mp4', source_id='cam_01', event_type='fall_detected', zone='',
                  labels=[dict(start_s=.1, end_s=.2)], negative_confirmed=False,
                  duration=.3, expected_frames=3, fps=10)
    return QaRun(**dict(values, **overrides))


def event(event_id='a', event_type='fall_detected', zone=None):
    return dict(event_id=event_id, source_id='cam_01', event_type=event_type, track_ids=[1],
                start_timestamp_ms=0, end_timestamp_ms=0, metadata={'zone':zone, 'current_zone':zone})


class ProductHealthTests(unittest.TestCase):
    def test_module_error_does_not_stop_other_features_or_keep_old_evidence(self):
        bad, good = Mock(), Mock()
        bad.process.side_effect = [ValueError('bad pose'), []]
        good.process.return_value = [event()]
        pipeline = ProgressPipeline({'bad':bad, 'good':good})
        context = dict(source_id='cam_01', timestamp_ms=100)
        self.assertEqual(len(pipeline.process(None, context)), 1)
        self.assertEqual(pipeline.health['bad']['state'], 'error')
        bad.reset.assert_called_once_with('cam_01')
        context['timestamp_ms'] = 200
        pipeline.process(None, context)
        self.assertEqual(pipeline.health['bad']['state'], 'ready')
        self.assertEqual(pipeline.health['bad']['failures'], 1)

    def test_flush_and_reset_errors_are_isolated(self):
        bad, good = Mock(), Mock()
        bad.flush.side_effect = ValueError('flush failed')
        bad.reset.side_effect = ValueError('reset failed')
        good.flush.return_value = [event()]
        pipeline = ProgressPipeline({'bad':bad, 'good':good})
        self.assertEqual(len(pipeline.flush()), 1)
        pipeline.reset('cam_01')
        good.reset.assert_called_once_with('cam_01')
        self.assertEqual(pipeline.health['bad']['state'], 'error')

    def test_empty_frame_is_no_observation_not_a_normal_verdict(self):
        worker = server.StreamWorker()
        worker.core = object(); worker.is_running = True; worker.last_frame_at = time.monotonic()
        worker.pipeline = SimpleNamespace(modules={'fall':object()}, health={'fall':dict(state='ready')})
        worker.config['features']['fall']['enabled'] = True
        self.assertEqual(snapshot(worker)['features']['fall']['state'], 'no_observation')

    def test_pause_stall_error_disabled_and_missing_zone_are_distinct(self):
        worker = server.StreamWorker(); worker.core = object()
        worker.is_running = True; worker.last_frame_at = time.monotonic()-10
        worker.pipeline = SimpleNamespace(modules={'wandering':object()}, health={})
        self.assertEqual(snapshot(worker)['capture']['state'], 'stalled')
        worker.is_paused = True
        self.assertEqual(snapshot(worker)['capture']['state'], 'paused')
        worker.capture_error = 'camera disconnected'
        self.assertEqual(snapshot(worker)['capture']['state'], 'error')
        worker.capture_error = None; worker.is_paused = False; worker.last_frame_at = time.monotonic()
        worker.active_tracks = [dict(track_id=1)]
        self.assertEqual(snapshot(worker)['features']['wandering']['state'], 'needs_configuration')
        self.assertEqual(snapshot(worker)['features']['fall']['state'], 'disabled')


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.worker = server.StreamWorker()
        self.points = [[.1,.1],[.9,.1],[.9,.9],[.1,.9]]

    def test_editing_active_source_stops_old_stream_before_replacing_profile(self):
        with patch.object(server,'worker',self.worker),patch.object(self.worker,'stop_stream') as stop:
            server.edit_camera('cam_01',server.CameraRequest(name='ห้อง A',source='new.mp4'))
            stop.assert_called_once()
            self.assertEqual(self.worker.source,'new.mp4')
            self.assertEqual(self.worker.cameras.profiles['cam_01']['source'],'new.mp4')

    def test_new_room_has_separate_empty_zones_and_export_has_no_identity_registry(self):
        cfg = deepcopy(self.worker.config)
        cfg['features']['location']['config']['zones'] = {'เตียง A':dict(points_relative=self.points, points=self.points, source_ids=['cam_01'])}
        cfg['features']['location']['config']['residents'] = {'private':{'name':'private'}}
        profiles = CameraProfiles(cfg)
        created = profiles.create('ห้อง B')
        self.assertEqual(created['config']['features']['location']['config']['zones'], {})
        self.assertEqual(len(profiles.profiles['cam_01']['config']['features']['location']['config']['zones']), 1)
        self.assertNotIn('residents', profiles.profiles['cam_01']['config']['features']['location']['config'])

    def test_named_bed_and_room_metadata_are_kept_on_the_correct_camera(self):
        cfg = apply_zones(self.worker.config, {'location_zones':{'เตียง A':dict(type='bed',points=self.points,bed_id='A',room_id='ห้อง A',source_ids=['wrong'])}}, 640, 480, 'camera_B')
        bed = cfg['features']['location']['config']['zones']['เตียง A']
        self.assertEqual(bed['source_ids'], ['camera_B'])
        self.assertEqual(bed['bed_id'], 'A'); self.assertEqual(bed['room_id'], 'ห้อง A')
        self.assertEqual(bed['points'][0], [64,48])

    def test_switch_back_restores_old_zones_and_feature_settings(self):
        self.worker.pipeline = None
        self.worker.config['features']['fall']['enabled'] = True
        self.worker.config['features']['wandering']['zones_relative'] = {'A':self.points}
        self.worker.cameras.save_active(self.worker.config)
        created = self.worker.cameras.create('ห้อง B')
        factory = lambda cfg: SimpleNamespace(modules={},flush=lambda:[])
        with patch.object(server, 'worker', self.worker), patch.object(server.ProgressPipeline, 'from_config', side_effect=factory):
            server.select_camera(created['id'])
            self.worker.config['features']['fall']['enabled'] = False
            self.worker.cameras.save_active(self.worker.config)
            server.select_camera('cam_01')
        self.assertTrue(self.worker.config['features']['fall']['enabled'])
        self.assertEqual(self.worker.config['features']['wandering']['zones_relative'], {'A':self.points})
        self.assertFalse(self.worker.cameras.profiles[created['id']]['config']['features']['fall']['enabled'])

    def test_invalid_import_does_not_replace_profiles_or_stop_stream(self):
        original = self.worker.cameras.snapshot()
        payload = dict(version=1, **deepcopy(original))
        payload['profiles'][0]['config']['features']['wandering']['zones_relative'] = {'bad':[[0,0],[1,1],[1,1]]}
        with patch.object(server, 'worker', self.worker), patch.object(self.worker, 'stop_stream') as stop:
            with self.assertRaises(HTTPException): server.import_cameras(payload)
        stop.assert_not_called()
        self.assertEqual(self.worker.cameras.snapshot(), original)

    def test_export_import_round_trip_restores_profiles_without_files_or_models(self):
        profiles = self.worker.cameras
        created = profiles.create('ห้อง B','0')
        payload = dict(version=1, **profiles.snapshot())
        restored, active = profiles.imported(payload)
        self.assertEqual(active, 'cam_01')
        self.assertEqual(restored[created['id']]['source'], '0')
        self.assertEqual(restored[created['id']]['name'], 'ห้อง B')

    def test_invalid_polygon_and_stale_camera_zone_request_are_rejected(self):
        for points in ([[0,0],[1,1],[.5,.5]], [[0,0],[float('nan'),1],[1,0]], [[0,0],[2,0],[0,1]]):
            with self.assertRaises(ValueError): polygon(points)
        with patch.object(server, 'worker', self.worker):
            with self.assertRaises(HTTPException) as raised:
                server.update_zones({'camera_id':'other','wandering_zones':{}})
        self.assertEqual(raised.exception.status_code, 409)


class QaTests(unittest.TestCase):
    def test_maximum_one_to_one_match_and_repeated_alerts(self):
        labels = [dict(start_s=0,end_s=10),dict(start_s=0,end_s=1)]
        predictions = [dict(alert_s=.5),dict(alert_s=9),dict(alert_s=20)]
        metrics = event_metrics(labels,predictions,30,0)
        self.assertEqual(metrics['true_positives'],2)
        self.assertEqual(metrics['false_positives'],1)
        self.assertEqual(metrics['missed_events'],0)

    def test_negative_clip_and_zero_predictions_do_not_claim_perfect_recall(self):
        metrics = event_metrics([],[],10,0)
        self.assertIsNone(metrics['event_recall']); self.assertIsNone(metrics['precision'])
        metrics = event_metrics([dict(start_s=1,end_s=2)],[],10,0)
        self.assertEqual(metrics['missed_events'],1); self.assertEqual(metrics['event_recall'],0)

    def test_live_emission_time_is_used_and_lifecycle_updates_deduplicated(self):
        qa = run()
        for i in range(1,4): qa.observe(i,i*100,dict(persons=[]),[event()] if i >= 2 else [])
        qa.finish(True,final_updates=[event()])
        report = qa.report()
        self.assertEqual(report['predictions'][0]['alert_s'],.2)
        self.assertEqual(report['metrics']['true_positives'],1)
        self.assertEqual(report['metrics']['mean_alert_delay_s'],.1)
        self.assertEqual(len(report['predictions']),1)

    def test_out_of_scope_zone_and_source_are_not_counted(self):
        qa = run(event_type='location_update',zone='bed_A')
        wrong_source = event(event_type='location_update',zone='bed_A'); wrong_source['source_id']='other'
        qa.record_events([event(event_type='location_update',zone='bed_B'),wrong_source],100)
        self.assertEqual(qa.predictions,[])
        qa.record_events([event(event_type='location_update',zone='bed_A')],200)
        self.assertEqual(len(qa.predictions),1)

    def test_abort_early_eof_and_processing_error_never_publish_accuracy_metrics(self):
        for scenario in ('abort','early_eof','detector','feature','gap'):
            with self.subTest(scenario=scenario):
                qa = run()
                for i in range(1,4 if scenario not in ('abort','early_eof') else 2):
                    qa.observe(i if scenario != 'gap' else i+1,i*100,dict(persons=[]),[],
                               detector_error='bad' if scenario == 'detector' else None,
                               module_error=scenario == 'feature')
                qa.finish(scenario != 'abort')
                self.assertEqual(qa.state,'incomplete'); self.assertIsNone(qa.report()['metrics'])

    def test_labels_require_explicit_negative_review_and_valid_times(self):
        for kwargs in (dict(labels=[]),dict(labels=[dict(start_s=-1,end_s=0)]),
                       dict(labels=[dict(start_s=0,end_s=1)]),dict(tolerance=float('nan')),
                       dict(negative_confirmed=True)):
            with self.assertRaises(ValueError): run(**kwargs)
        self.assertEqual(run(labels=[],negative_confirmed=True).labels,[])

    def test_config_reload_aborts_active_qa_and_seek_is_rejected(self):
        worker = server.StreamWorker(); worker.qa_run = run()
        worker.total_frames = 3; worker.video_fps = 10
        self.assertEqual(worker.seek(time_sec=0)['status'],'error')
        with patch.object(server.ProgressPipeline,'from_config',return_value=SimpleNamespace(modules={})):
            worker.update_config(worker.config)
        self.assertEqual(worker.qa_run.state,'incomplete')

    def test_full_worker_loop_finishes_qa_from_eof_without_model_inference(self):
        worker = server.StreamWorker(); worker.source = 'test.mp4'; worker.is_running = True; worker.loop_video = False
        worker.qa_run = run(); worker.pipeline = SimpleNamespace(modules={},health={},flush=lambda:[],
            process=lambda frame,ctx:[event()] if ctx['frame_id'] == 2 else [])
        worker.core = Mock()
        worker.core.process.side_effect = lambda frame,**kw: dict(kw,persons=[])
        cap = Mock(); cap.isOpened.return_value = True
        cap.get.side_effect = lambda key: {server.cv2.CAP_PROP_FPS:10,server.cv2.CAP_PROP_FRAME_COUNT:3,
            server.cv2.CAP_PROP_FRAME_WIDTH:64,server.cv2.CAP_PROP_FRAME_HEIGHT:64}.get(key,0)
        cap.getBackendName.return_value = 'TEST'
        cap.read.side_effect = [(True,np.zeros((64,64,3),np.uint8))]*3+[(False,None)]
        def sleep(_):
            if worker.is_paused: worker.is_running = False
        with patch.object(server,'open_capture',return_value=cap),patch.object(server.time,'sleep',side_effect=sleep):
            worker._run_loop(worker.generation_id)
        self.assertEqual(worker.qa_run.state,'completed')
        self.assertEqual(worker.qa_run.report()['metrics']['true_positives'],1)
        self.assertEqual(worker.qa_run.report()['coverage']['processed_fraction'],1)
        cap.release.assert_called_once()

    def test_worker_setup_failure_stops_qa_and_releases_capture(self):
        worker = server.StreamWorker(); worker.source='test.mp4'; worker.is_running=True; worker.qa_run=run()
        cap = Mock(); cap.isOpened.return_value=True; cap.get.side_effect=RuntimeError('bad metadata')
        with patch.object(server,'open_capture',return_value=cap): worker._run_loop(worker.generation_id)
        self.assertFalse(worker.is_running)
        self.assertEqual(worker.qa_run.state,'incomplete')
        self.assertIsNone(worker.qa_run.report()['metrics'])
        self.assertIn('bad metadata',worker.capture_error)
        cap.release.assert_called_once()

    def test_qa_start_prepares_provenance_without_long_inference(self):
        worker=server.StreamWorker(); worker.config['features']['fall']['enabled']=True
        worker.cameras.save_active(worker.config)
        with tempfile.TemporaryDirectory() as directory:
            video=Path(directory)/'test.mp4'; video.write_bytes(b'metadata test')
            cap=Mock(); cap.isOpened.return_value=True
            cap.get.side_effect=lambda key: 10 if key == server.cv2.CAP_PROP_FPS else 3
            with patch.object(server,'worker',worker),patch.object(server.cv2,'VideoCapture',return_value=cap),patch.object(worker,'start_stream') as start:
                state=server.start_qa(server.QaRequest(source=str(video),labels=[dict(start_s=.1,end_s=.2)]))
                self.assertEqual(state['state'],'running')
                self.assertEqual(start.call_args.kwargs['qa_run'].provenance['video']['frames'],3)
                self.assertIn('Location/location/module.py',start.call_args.kwargs['qa_run'].provenance['code_sha256'])
                start.side_effect=RuntimeError('not stopped')
                with self.assertRaises(HTTPException): server.start_qa(server.QaRequest(source=str(video),labels=[],negative_confirmed=True))

    def test_get_report_cannot_start_inference_and_start_requires_user_request(self):
        worker = server.StreamWorker()
        with patch.object(server,'worker',worker),patch.object(worker,'start_stream') as start:
            self.assertIsNone(server.get_qa())
            with self.assertRaises(HTTPException): server.export_qa()
            start.assert_not_called()


if __name__ == '__main__': unittest.main()
