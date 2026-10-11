import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from web.feature_settings import FeatureSettings
from web import server

class FeatureSettingsTests(unittest.TestCase):
    def config(self):return {'features':{'fall':{'enabled':True,'backend':'yolo_pose_rf_v2'},'location':{'enabled':True}}}
    def test_disabled_fall_survives_new_instance_and_can_be_reenabled(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'toggles.json';settings=FeatureSettings(path)
            config=self.config();config['features']['fall']['enabled']=False;settings.save(config)
            restored=FeatureSettings(path).load(self.config())
            self.assertFalse(restored['features']['fall']['enabled'])
            self.assertTrue(restored['features']['location']['enabled'])
            self.assertEqual(restored['features']['fall']['backend'],'yolo_pose_rf_v2')
            restored['features']['fall']['enabled']=True;settings.save(restored)
            self.assertTrue(FeatureSettings(path).load(self.config())['features']['fall']['enabled'])
    def test_atomic_write_failure_keeps_previous_switches(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'toggles.json';settings=FeatureSettings(path);settings.save(self.config())
            candidate=self.config();candidate['features']['fall']['enabled']=False
            with patch('web.feature_settings.os.replace',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):settings.save(candidate)
            self.assertTrue(settings.load(self.config())['features']['fall']['enabled'])
            self.assertEqual(len(list(Path(folder).iterdir())),1)
    def test_invalid_saved_switches_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'toggles.json';path.write_text('{"fall":"false"}')
            with self.assertRaises(ValueError):FeatureSettings(path).load(self.config())
    def test_toggle_api_saves_after_pipeline_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'toggles.json';config=self.config()
            with patch.object(server.worker,'config',config),patch.object(server.worker,'pipeline',None), patch.object(server.worker,'feature_settings',FeatureSettings(path)), patch.object(server.ProgressPipeline,'from_config',return_value=SimpleNamespace(modules={})), patch.object(server.worker.cameras,'save_active'):
                response=TestClient(server.app).post('/api/config/features',json={'fall':False})
                self.assertEqual(response.status_code,200)
                self.assertFalse(FeatureSettings(path).load(self.config())['features']['fall']['enabled'])
                with patch.object(server.ProgressPipeline,'from_config',side_effect=ValueError('bad model')):
                    response=TestClient(server.app).post('/api/config/features',json={'fall':True})
                self.assertEqual(response.status_code,400)
                self.assertFalse(FeatureSettings(path).load(self.config())['features']['fall']['enabled'])

if __name__=='__main__':unittest.main()
