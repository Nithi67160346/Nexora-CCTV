from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from integration.fall_ml import FallMLAdapter
from integration.fall_model import FEATURE_NAMES, MODEL_FILES, SKLEARN_VERSION, FallModelBundle, load_bundle, pose_features, project_status, validate_models
from integration.pipeline import ProgressPipeline
from web import server
from web.playback import AnnotationBuffer
from web.presentation import fall_display


def person(tid=1):
    return dict(track_id=tid,confidence=.9,bbox_xyxy=[10,10,90,95],
        pose=[[20+i*2,20+i*3,.9] for i in range(17)],observation_kind='body_pose')


def context(ts=0, people=None, source='A'):
    return dict(source_id=source,timestamp_ms=ts,frame_id=int(ts/33)+1,fps=30,
                persons=[person()] if people is None else people)


def result(kind='ALERT_FALL', probability=.8):
    return dict(raw_status=kind,probability_fall=probability,ai1_score=-.6,ai1_threshold=-.56)


class FallMLTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.zeros((100,100,3),np.uint8)
        self.bundle=SimpleNamespace(checksum='fixture',predict=Mock(side_effect=lambda vectors,threshold:[result() for _ in vectors]))
        self.module=FallMLAdapter(self.bundle).setup({})

    def feed(self,count=6,source='A',people=None,start=0,module=None):
        events=[]
        for i in range(count):events += (module or self.module).process(self.frame,context(start+i*33,people,source))
        return events

    def test_web_migrates_old_profile_to_model_defaults_without_turning_feature_on(self):
        cfg=server.StreamWorker()._product_config({'features':{'fall':{'enabled':False,'window_size':10}}})
        self.assertFalse(cfg['features']['fall']['enabled'])
        self.assertEqual(cfg['features']['fall']['backend'],'yolo_pose_rf_v2')
        self.assertEqual(cfg['features']['fall']['window_size'],6)
        cfg['features']['fall']['enabled']=True
        with patch('integration.fall_ml.load_bundle',return_value=self.bundle):pipeline=ProgressPipeline.from_config(cfg)
        self.assertIsInstance(pipeline.modules['fall'],FallMLAdapter)
        self.assertTrue(pipeline.modules['fall'].ml_loaded)

    def test_features_use_bbox_relative_normalization_in_the_saved_order(self):
        self.module.process(self.frame,context())
        vector=self.bundle.predict.call_args.args[0][0]
        self.assertEqual(len(vector),54)
        np.testing.assert_allclose(vector[:3],[-.375,-32.5/85,.9])
        np.testing.assert_allclose(vector[48:51],[.025,15.5/85,.9])
        self.assertAlmostEqual(vector[-3],85/80)
        self.assertGreater(vector[-2],45)

    def test_missing_torso_anchors_or_bad_bbox_cannot_trigger_fall(self):
        for joint in (5,6,11,12):
            p=person();p['pose'][joint][2]=.24
            self.assertEqual(self.feed(people=[p]),[])
        for box in ([10,10,10,95],[-1,10,90,95],[10,10,101,95],[10,10,float('nan'),95]):
            p=person();p['bbox_xyxy']=box
            self.assertEqual(self.feed(people=[p]),[])
        self.bundle.predict.assert_not_called()

    def test_six_real_observations_emit_track_bound_event_with_provenance(self):
        self.assertEqual(self.feed(5),[])
        self.assertEqual(self.module.status('A',1)['phase'],'WARMING_UP')
        event=self.feed(1,start=165)[0]
        self.assertEqual(event['track_ids'],[1])
        self.assertEqual(event['event_type'],'fall_detected')
        self.assertEqual(event['metadata']['confidence_kind'],'model_score')
        self.assertFalse(event['metadata']['onset_observed'])
        self.assertEqual(event['metadata']['sample_times_ms'],[0.,33.,66.,99.,132.,165.])
        self.assertEqual(fall_display(self.module.status('A',1))[1],'alert')

    def test_ratio_requires_two_positive_predictions_in_six_observations(self):
        values=iter([result()]+[result('NORMAL',None)]*5)
        self.bundle.predict.side_effect=lambda *args:[next(values)]
        self.assertEqual(self.feed(),[])
        self.assertEqual(self.module.status('A',1)['phase'],'NORMAL')
        self.module.reset('A');values=iter([result()]*2+[result('NORMAL',None)]*4)
        self.assertEqual(len(self.feed()),1)

    def test_normal_and_abnormal_frames_are_distinct_from_fall_alerts(self):
        for kind,probability,phase in [('NORMAL',None,'NORMAL'),('ABNORMAL_NOT_FALL',.2,'ABNORMAL_MOVEMENT')]:
            self.module.reset('A')
            self.bundle.predict.side_effect=lambda *args:[result(kind,probability)]
            self.assertEqual(self.feed(),[])
            self.assertEqual(self.module.status('A',1)['phase'],phase)

    def test_missing_bad_or_bbox_only_pose_does_not_become_normal_or_fall(self):
        for pose in [None,[[0,0,0]]*17,[[1,1,float('nan')]]*17,[[20,20,.9]]*16]:
            p=person();p['pose']=pose
            self.assertEqual(self.feed(people=[p]),[])
        p=person();p['observation_kind']='bbox_only';self.feed(people=[p])
        self.bundle.predict.assert_not_called()
        self.assertFalse(self.module.status('A',1)['observed'])

    def test_missing_people_and_long_gaps_cannot_complete_old_window(self):
        self.feed(5);self.feed(1,people=[],start=165)
        self.assertEqual(self.module.status('A',1)['phase'],'UNAVAILABLE')
        self.assertEqual(self.feed(5,start=198),[])
        self.module.process(self.frame,context(1500))
        self.assertEqual(self.module.status('A',1)['sampled_frames'],1)

    def test_duplicate_track_ids_cannot_count_as_multiple_frames_or_share_pose(self):
        self.assertEqual(self.feed(6,people=[person(1),person(1)]),[])
        self.bundle.predict.assert_not_called()
        self.assertEqual(self.module.status('A',1)['sampled_frames'],0)

    def test_tracks_sources_and_resets_do_not_borrow_evidence(self):
        self.feed(5)
        self.assertEqual(self.feed(1,people=[person(2)],start=165),[])
        self.assertEqual(self.feed(5,source='B'),[])
        self.module.reset('B');self.assertEqual(self.feed(5,source='B'),[])
        self.assertEqual(self.module.status('A',2)['sampled_frames'],1)

    def test_cooldown_prevents_spam_and_restarts_after_source_reset(self):
        self.assertEqual(len(self.feed(90)),1)
        self.module.reset('A');self.assertEqual(len(self.feed()),1)

    def test_inference_failure_is_visible_and_discards_interrupted_evidence(self):
        pipeline=ProgressPipeline({'fall':self.module})
        for i in range(5):pipeline.process(self.frame,context(i*33))
        self.bundle.predict.side_effect=RuntimeError('broken fall model')
        self.assertEqual(pipeline.process(self.frame,context(165)),[])
        self.assertEqual(pipeline.health['fall']['state'],'error')
        self.assertEqual(self.module.status('A',1)['sampled_frames'],0)

    def test_ml_status_is_stored_with_original_frame_and_replay_survives_switch(self):
        self.feed()
        with tempfile.TemporaryDirectory() as folder:
            cache=AnnotationBuffer();cache.enable_disk(folder)
            cache.add(context(165),self.module,[])
            name=cache.cache_path.name;cache.clear();self.module.reset('A')
            row=AnnotationBuffer.read_cached(folder,name,.165)[0]
            self.assertEqual(row['persons'][0]['fall']['fall_score'],.8)
            self.assertEqual(row['persons'][0]['fall']['backend'],'yolo_pose_rf_v2')
            self.assertEqual(row['persons'][0]['fall']['phase'],'FALL_DETECTED')

    def test_invalid_settings_and_missing_models_fail_explicitly(self):
        for config in [{'window_size':1},{'fall_proba_threshold':float('nan')},{'fall_ratio_threshold':0}]:
            with self.assertRaises(ValueError):FallMLAdapter(self.bundle).setup(config)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'โมเดลตรวจล้มไม่ครบ'):validate_models(folder)


class FallVersionTests(unittest.TestCase):
    def test_54_feature_schema_rejects_reordered_columns_and_old_estimators(self):
        from sklearn.ensemble import RandomForestClassifier
        for wrong_columns,wrong_size in ((True,False),(False,True)):
            ai2=RandomForestClassifier()
            ai2.n_features_in_=54
            ai2.feature_names_in_=np.asarray(FEATURE_NAMES)
            ai2.classes_=np.asarray([0,1])
            if wrong_size:ai2.n_features_in_=51
            columns=FEATURE_NAMES[::-1] if wrong_columns else FEATURE_NAMES
            with patch('sklearn.__version__',SKLEARN_VERSION), patch('joblib.load',side_effect=[ai2,columns]):
                with self.assertRaises(ValueError):FallModelBundle([Path(name) for name in MODEL_FILES])

    def test_legacy_ai1_artifacts_are_not_deserialized_in_rf_only_policy(self):
        from sklearn.ensemble import RandomForestClassifier
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/name for name in MODEL_FILES]
            for path in paths:path.write_bytes(b'legacy is not a loadable pickle')
            ai2=RandomForestClassifier();ai2.n_features_in_=54;ai2.classes_=np.asarray([0,1])
            with patch('sklearn.__version__',SKLEARN_VERSION), patch('joblib.load',side_effect=[ai2,FEATURE_NAMES]) as loader:
                bundle=FallModelBundle(paths)
            self.assertEqual([c.args[0] for c in loader.call_args_list],paths[3:])
            self.assertIsNone(bundle.ai1);self.assertIsNone(bundle.scaler)
            self.assertEqual(bundle.inactive_artifacts,list(MODEL_FILES[:3]))

    def test_wrong_runtime_version_rejected_before_deserializing_models(self):
        with patch('sklearn.__version__', '0.0.0'), patch('joblib.load') as loader:
            with self.assertRaisesRegex(ValueError, f'scikit-learn=={SKLEARN_VERSION}'):
                FallModelBundle([Path(name) for name in MODEL_FILES])
            loader.assert_not_called()

    def test_saved_estimator_version_mismatch_is_not_silenced(self):
        from sklearn.exceptions import InconsistentVersionWarning
        import warnings

        def mismatch(path):
            warnings.warn(InconsistentVersionWarning(
                estimator_name='DecisionTreeClassifier',
                current_sklearn_version=SKLEARN_VERSION,
                original_sklearn_version='1.6.1'))

        with patch('sklearn.__version__', SKLEARN_VERSION), patch('joblib.load', side_effect=mismatch):
            with self.assertRaisesRegex(ValueError, f'ai2_fall_detector_yolo.pkl.*1.6.1.*{SKLEARN_VERSION}'):
                FallModelBundle([Path(name) for name in MODEL_FILES])


class FallArtifactTests(unittest.TestCase):
    def test_real_supplied_artifacts_load_and_match_project54_predictions(self):
        import joblib,pandas as pd
        folder=Path(__file__).resolve().parents[2]/'Fall/models_yolo'
        bundle=load_bundle(folder)
        columns=joblib.load(folder/MODEL_FILES[-1])
        self.assertEqual(columns,FEATURE_NAMES)
        ai2=joblib.load(folder/MODEL_FILES[3]);ai2.n_jobs=1
        vectors=[pose_features(person()['pose'],person()['bbox_xyxy'])]
        vectors += [vectors[0][:51]+[.5,10.,.1]]
        matrix=pd.DataFrame(vectors,columns=columns)
        probabilities=ai2.predict_proba(matrix)[:,ai2.classes_.tolist().index(1)]
        results=bundle.predict(vectors)
        for vector,probability,actual in zip(vectors,probabilities,results):
            self.assertIsNone(actual['ai1_score'])
            self.assertFalse(actual['ai1_gate_used'])
            self.assertAlmostEqual(actual['probability_fall'],probability)
            expected=project_status(probability,vector[-3],vector[-2])[0]
            self.assertEqual(actual['raw_status'],expected)
        self.assertEqual(len(bundle.checksum),64)


class Project54PolicyTests(unittest.TestCase):
    def bundle(self, probabilities):
        bundle=object.__new__(FallModelBundle)
        bundle.ai2=Mock();bundle.ai2.predict_proba.return_value=[[1-p,p] for p in probabilities]
        bundle.ai1=Mock();bundle.scaler=Mock();bundle.threshold=-.57;bundle.fall_index=1;bundle.checksum='fixture'
        return bundle

    def test_upright_person_is_not_fall_even_at_high_rf_score_over_six_frames(self):
        bundle=self.bundle([.99]);module=FallMLAdapter(bundle).setup({})
        frame=np.zeros((100,100,3),np.uint8)
        events=[]
        for i in range(6):events.extend(module.process(frame,context(i*33)))
        self.assertEqual(events,[])
        self.assertEqual(module.status('A',1)['phase'],'ABNORMAL_MOVEMENT')
        bundle.ai1.score_samples.assert_not_called();bundle.scaler.transform.assert_not_called()

    def test_horizontal_pose_follows_rf_and_project_geometry_override(self):
        normal=pose_features(person()['pose'],person()['bbox_xyxy'])
        vectors=[normal[:51]+[.8,20.,.2],normal[:51]+[.8,20.,.2],normal[:51]+[.6,20.,.2]]
        result=self.bundle([.4,.39,.1]).predict(vectors)
        self.assertEqual([r['raw_status'] for r in result],['ALERT_FALL','NORMAL','ALERT_FALL'])
        self.assertEqual(result[-1]['decision_basis'],'lying_pose_geometry_override')

    def test_project_boundary_angles_and_aspect_do_not_count_as_lying(self):
        self.assertEqual(project_status(.99,.8,45.)[0],'ABNORMAL_NOT_FALL')
        self.assertEqual(project_status(.99,.9,20.)[0],'ABNORMAL_NOT_FALL')
        self.assertEqual(project_status(.49,2.,89.)[0],'NORMAL')

    def test_bbox_features_are_invariant_to_translation_and_scaling(self):
        p=person();pose=np.asarray(p['pose']);box=np.asarray(p['bbox_xyxy'])
        baseline=pose_features(pose,box)
        moved=pose.copy();moved[:,:2]=moved[:,:2]*2+[30,40]
        scaled=pose_features(moved,box*2+[30,40,30,40])
        np.testing.assert_allclose(baseline[:51],scaled[:51])
        self.assertAlmostEqual(baseline[-3],scaled[-3])
        self.assertLess(baseline[0],0)  # Bbox-centered coordinates may be negative.

    def test_feature_arithmetic_matches_supplied_project_source_without_importing_it(self):
        import ast
        path=Path(__file__).resolve().parents[2]/'Fall/Project.py'
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict_and_track')
        loop=next(n for n in function.body if isinstance(n,ast.For))
        def assignment(node,name):
            return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
        start=next(i for i,n in enumerate(loop.body) if assignment(n,'bw'))
        end=next(i for i,n in enumerate(loop.body) if assignment(n,'feat_df'))
        code=compile(ast.Module(body=loop.body[start:end],type_ignores=[]),str(path),'exec')
        p=person();pose=np.asarray(p['pose'])
        scope=dict(np=np,boxes_xyxy=np.asarray([p['bbox_xyxy']]),kp_xy=pose[:,:2],kp_conf=pose[:,2],NUM_KEYPOINTS=17,i=0)
        exec(code,scope)
        np.testing.assert_allclose(pose_features(pose,p['bbox_xyxy']),scope['features'])

    def test_bad_rf_output_and_old_51_vectors_fail_without_events(self):
        bundle=self.bundle([.5]);vector=pose_features(person()['pose'],person()['bbox_xyxy'])
        for values in ([vector[:51]],[vector[:53]+[float('nan')]]):
            with self.assertRaises(ValueError):bundle.predict(values)
        for output in ([[float('nan'),.5]],[[.1,1.1]],[[.5]],[]):
            bundle.ai2.predict_proba.return_value=output
            with self.assertRaises(ValueError):bundle.predict([vector])
