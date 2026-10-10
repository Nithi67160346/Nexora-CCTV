"""Docker startup failures and session preservation without building an image."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


def local_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'docker' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DockerReadinessTests(unittest.TestCase):
    def test_incompatible_fall_bundle_fails_before_web_start(self):
        entry=local_module('entrypoint')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'models').mkdir();(root/'Fall/models_yolo').mkdir(parents=True)
            (root/'models/yolo26n-pose.pt').write_bytes(b'pose')
            (root/'best_lstm_model.pth').write_bytes(b'lstm')
            from integration.fall_model import MODEL_FILES
            for name in MODEL_FILES:(root/'Fall/models_yolo'/name).write_bytes(b'fixture')
            fall=Mock();fall.load_bundle.side_effect=ValueError('wrong version or schema')
            importer=Mock(side_effect=lambda name:fall if name=='integration.fall_model' else Mock())
            with self.assertRaisesRegex(ValueError,'wrong version or schema'):
                entry.check_environment(root,{'NEXORA_DEVICE':'cpu'},importer)
            fall.load_bundle.assert_called_once_with(root/'Fall/models_yolo')

    def test_fall_weights_are_required_without_downloading_or_training(self):
        entry=local_module('entrypoint')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'models').mkdir()
            (root/'models/yolo26n-pose.pt').write_bytes(b'pose')
            (root/'best_lstm_model.pth').write_bytes(b'lstm')
            with self.assertRaisesRegex(RuntimeError,'Fall/models_yolo'):
                entry.check_environment(root, {'NEXORA_DEVICE':'cpu'}, Mock())

    def test_main_lstm_checkpoint_is_required_before_opening_web(self):
        entry = local_module('entrypoint')
        with tempfile.TemporaryDirectory() as directory:
            weights=Path(directory)/'models/yolo26n-pose.pt'
            weights.parent.mkdir();weights.write_bytes(b'pose')
            with self.assertRaisesRegex(RuntimeError,'best_lstm_model.pth'):
                entry.check_environment(directory,{'NEXORA_DEVICE':'cpu'},Mock())

    def test_missing_or_empty_weights_fail_without_network_download(self):
        entry = local_module('entrypoint')
        importer = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Missing models'):
                entry.check_environment(directory, {}, importer)
            weights = Path(directory) / 'models/yolo26n-pose.pt'
            weights.parent.mkdir(); weights.touch()
            with self.assertRaisesRegex(RuntimeError, 'Missing models'):
                entry.check_environment(directory, {}, importer)
        importer.assert_not_called()

    def test_gpu_unavailable_fails_instead_of_using_cpu(self):
        entry = local_module('entrypoint')
        torch = Mock(); torch.cuda.is_available.return_value = False
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'models/yolo26n-pose.pt'
            weights.parent.mkdir(); weights.write_bytes(b'test')
            with self.assertRaisesRegex(RuntimeError, 'GPU was requested'):
                entry.check_environment(directory, {'NEXORA_DEVICE': 'cuda'}, Mock(return_value=torch))

    def test_idle_web_does_not_require_a_camera_but_requires_detector(self):
        health = local_module('healthcheck')
        self.assertTrue(health.ready({'health': {'capture': {'state': 'idle'}, 'detector': {'state': 'ready'}}}))
        for value in (None, [], {}, {'health': {}}, {'health': {'detector': {'state': 'error'}}}):
            self.assertFalse(health.ready(value))


class DockerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.smoke = local_module('smoke')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'models').mkdir()
        for name in self.smoke.MODELS:
            (self.root / 'models' / name).write_bytes(b'test')
        (self.root/'Fall/models_yolo').mkdir(parents=True)
        for name in self.smoke.FALL_MODELS:(self.root/'Fall/models_yolo'/name).write_bytes(b'test')
        self.responses = {
            '/api/status': json.dumps({'health': {'detector': {'state': 'ready'}}, 'is_running': False}).encode(),
            '/api/runtime/options': json.dumps({'ffmpeg_available': True, 'gpu_available': True,
                'models': [dict(name=name, available=True) for name in self.smoke.MODELS]}).encode(),
            '/api/multistream': b'[]',
            '/': b'/static/vendor/tailwind.css /static/vendor/fontawesome/css/all.min.css',
            '/static/vendor/tailwind.css': b'x' * 101,
            '/static/vendor/fontawesome/css/all.min.css': b'x' * 101,
            '/static/vendor/fontawesome/webfonts/fa-solid-900.woff2': b'x' * 101,
        }
        self.predict = Mock(return_value='cpu')
        self.command = Mock(return_value=Mock(stdout='ffmpeg version test\n'))

    def probe(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {'NEXORA_DEVICE': 'cpu'}):
            return self.smoke.run_probe(self.root, self.responses.__getitem__, self.predict, self.command)

    def test_idle_container_runs_only_one_synthetic_prediction(self):
        report = self.probe()
        self.assertTrue(report['passed'], report)
        self.assertEqual(report['synthetic_frames'], 1)
        self.assertEqual(report['inference'], 'passed')
        self.predict.assert_called_once_with(self.root / 'models/yolo26n-pose.pt', 'cpu')

    def test_bad_weights_or_assets_never_trigger_inference(self):
        (self.root / 'models/yolov8m-pose.pt').unlink()
        report = self.probe()
        self.assertFalse(report['passed'])
        self.predict.assert_not_called()
        (self.root / 'models/yolov8m-pose.pt').write_bytes(b'test')
        self.responses['/'] = b'https://cdn.tailwindcss.com'
        self.assertFalse(self.probe()['passed'])
        self.predict.assert_not_called()

    def test_paused_main_or_starting_child_session_skips_inference(self):
        self.responses['/api/status'] = json.dumps({'health': {'detector': {'state': 'ready'}},
                                                   'is_running': True, 'is_paused': True}).encode()
        report = self.probe()
        self.assertEqual(report['inference'], 'skipped_active_session')
        self.assertTrue(report['passed'])
        self.predict.assert_not_called()
        self.responses['/api/status'] = b'{"health":{"detector":{"state":"ready"}},"is_running":false}'
        self.responses['/api/multistream'] = b'[{"state":"starting","is_running":false}]'
        self.assertEqual(self.probe()['inference'], 'skipped_active_session')
        self.predict.assert_not_called()

    def test_wrong_device_fails_runtime_report(self):
        self.predict.return_value = 'cuda:0'
        report = self.probe()
        self.assertFalse(report['passed'])
        self.assertIn('device mismatch', report['checks'][-1]['error'])

    def test_missing_fall_model_prevents_success_report_and_synthetic_inference(self):
        (self.root/'Fall/models_yolo/ai2_fall_detector_yolo.pkl').unlink()
        self.assertFalse(self.probe()['passed'])
        self.predict.assert_not_called()


FAKE_DOCKER = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
with pathlib.Path(os.environ['QA_DOCKER_CALLS']).open('a') as log:
    log.write(json.dumps(args) + '\n')
if args[0] == 'version':
    if os.environ.get('QA_ENGINE') == 'off':
        print('Engine is stopped', file=sys.stderr); sys.exit(1)
    print(os.environ.get('QA_ARCH', 'amd64') if 'Server.Arch' in args[-1] else os.environ.get('QA_ENGINE', 'linux'))
elif args[0] == 'compose':
    if 'version' in args: print('Docker Compose version v2.40.0')
    elif 'ps' in args and '-q' in args:
        if os.environ.get('QA_EXISTING'): print('a' * 64)
    elif 'build' in args:
        if os.environ.get('QA_BUILD_FAIL'): sys.exit(1)
    elif 'up' in args:
        if os.environ.get('QA_UP_FAIL'): sys.exit(1)
elif args[0] == 'inspect':
    if '--format' not in args:
        print(json.dumps([dict(State=dict(Running=os.environ.get('QA_EXISTING') == 'running'),
            Config=dict(Image=os.environ.get('QA_RUNNING_IMAGE', 'nexora-qa:cpu-20261005')),
            Image='sha256:test', NetworkSettings=dict(Ports={'8080/tcp': [dict(HostPort=os.environ.get('QA_RUNNING_PORT', '18245'))]}),
            Mounts=[dict(Destination='/app/models', RW=bool(os.environ.get('QA_WRITABLE_MODELS'))),
                    dict(Destination='/app/Fall/models_yolo', RW=bool(os.environ.get('QA_WRITABLE_FALL_MODELS'))),
                    dict(Destination='/app/local_only', Type='volume', RW=True, Name='test_volume')])]))
    else:
        template = args[args.index('--format') + 1]
        if 'State.Running' in template: print('true' if os.environ.get('QA_EXISTING') == 'running' else 'false')
        elif 'Config.Image' in template: print(os.environ.get('QA_RUNNING_IMAGE', 'nexora-qa:cpu-20261005'))
        elif 'Config.Labels' in template:
            print(json.dumps({'nexora.web.revision': os.environ.get('QA_WEB_REVISION', 'qa-upload-delete-20261010'),
                'nexora.source.fingerprint': os.environ.get('QA_IMAGE_FINGERPRINT',os.environ.get('NEXORA_SOURCE_FINGERPRINT',''))}))
        elif 'json .NetworkSettings.Ports' in template:
            print(json.dumps({'8080/tcp': None if os.environ.get('QA_MISSING_PORT') else [dict(HostIp='127.0.0.1', HostPort='18245')]}))
elif args[0] == 'exec':
    sys.stdin.read()
    failed = bool(os.environ.get('QA_PROBE_FAIL'))
    print(json.dumps(dict(passed=not failed, checks=[], inference='passed')))
    sys.exit(1 if failed else 0)
elif args[0] == 'run':
    if os.environ.get('QA_GPU_FAIL'): print('Docker GPU unavailable',file=sys.stderr);sys.exit(1)
    print('NVIDIA Test GPU')
elif args[:2] == ['image', 'ls']:
    if (os.environ.get('QA_IMAGE') or pathlib.Path(os.environ['QA_DOCKER_CALLS']+'.loaded').exists()) and not ('reference=nexora-qa:gpu-20261005' in args and os.environ.get('QA_NO_GPU_IMAGE')): print('image123')
elif args[:2] == ['image', 'inspect']:
    print(os.environ.get('QA_READY_IMAGE_ID','sha256:'+'1'*64) if any(a.startswith('nexora-qa:ready-') for a in args) else 'image123')
elif args[0] == 'load':
    pathlib.Path(os.environ['QA_DOCKER_CALLS']+'.loaded').touch();print('Loaded image')
elif args[0] == 'save': pathlib.Path(args[args.index('--output') + 1]).write_bytes(b'exported image data')
sys.exit(0)
'''


@unittest.skipUnless(os.name == 'nt' and shutil.which('powershell.exe'), 'Windows PowerShell launcher')
class DockerLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='NEXORA QA ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.script = self.root / 'scripts/windows/run_docker.ps1'
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'scripts/windows/run_docker.ps1', self.script)
        for name in ('compose.yaml', 'compose.gpu.yaml'):
            shutil.copyfile(ROOT / name, self.root / name)
        (self.root / 'docker').mkdir()
        shutil.copyfile(ROOT / 'docker/smoke.py', self.root / 'docker/smoke.py')
        weights = self.root / 'models/yolo26n-pose.pt'
        weights.parent.mkdir(); weights.write_bytes(b'test')
        (self.root / 'best_lstm_model.pth').write_bytes(b'test checkpoint')
        (self.root / 'web').mkdir()
        shutil.copyfile(ROOT/'web/runtime_version.json', self.root/'web/runtime_version.json')
        fall_models=self.root/'Fall/models_yolo';fall_models.mkdir(parents=True)
        for name in ('ai1_isolation_forest_yolo.pkl','ai1_scaler_yolo.pkl','ai1_threshold_yolo.pkl',
                     'ai2_fall_detector_yolo.pkl','feature_names_yolo.pkl'):
            (fall_models/name).write_bytes(b'fake fall model')
        fake = self.root / 'fake-bin'; fake.mkdir()
        (fake / 'fake_docker.py').write_text(FAKE_DOCKER, encoding='utf-8')
        (fake / 'docker.cmd').write_text(
            f'@echo off\r\n"{sys.executable}" "%~dp0fake_docker.py" %*\r\nexit /b %ERRORLEVEL%\r\n', encoding='utf-8')
        (fake / 'nvidia-smi.cmd').write_text(
            '@echo off\r\nif "%QA_NVIDIA%"=="present" (\r\n echo NVIDIA Test GPU\r\n exit /b 0\r\n)\r\nexit /b 1\r\n',encoding='utf-8')
        self.calls = self.root / 'calls.jsonl'
        self.environment = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'],
                                QA_DOCKER_CALLS=str(self.calls))

    def run_launcher(self, arguments=(), **environment):
        result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
            '-File', str(self.script), '-NoBrowser', *arguments],
            env=dict(self.environment, **environment), capture_output=True, text=True, timeout=30)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []
        return result, calls

    def test_stopped_engine_gives_actionable_error_and_no_build(self):
        result, calls = self.run_launcher(QA_ENGINE='off')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Open Docker Desktop', result.stdout)
        self.assertEqual(len(calls), 1)

    def test_double_click_selects_gpu_only_after_local_cuda_probe_passes(self):
        result,calls=self.run_launcher(QA_NVIDIA='present',QA_IMAGE='present')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Automatic device selection: gpu',result.stdout)
        self.assertTrue(any(any(arg.endswith('compose.gpu.yaml') for arg in call) for call in calls))
        self.assertTrue(any(call[0]=='run' for call in calls))
        self.assertFalse(any('build' in call or 'stop' in call for call in calls))

    def test_nvidia_with_broken_docker_gpu_falls_back_to_cpu(self):
        result,calls=self.run_launcher(QA_NVIDIA='present',QA_IMAGE='present',QA_GPU_FAIL='true')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Automatic device selection: cpu',result.stdout)
        up=next(call for call in calls if 'up' in call)
        self.assertFalse(any(arg.endswith('compose.gpu.yaml') for arg in up))
        probe=next(call for call in calls if call[0]=='run')
        self.assertIn('--pull',probe);self.assertIn('never',probe);self.assertIn('--rm',probe)
        self.assertFalse(any('volume' in call or 'down' in call or 'stop' in call for call in calls))

    def test_gpu_without_local_probe_image_uses_cpu_without_downloading_gpu_image(self):
        result,calls=self.run_launcher(QA_NVIDIA='present',QA_IMAGE='present',QA_NO_GPU_IMAGE='true')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Automatic device selection: cpu',result.stdout)
        self.assertFalse(any(call[0]=='run' or call[0]=='pull' for call in calls))

    def test_check_only_does_not_run_gpu_probe_or_modify_containers(self):
        result,calls=self.run_launcher(['-CheckOnly'],QA_NVIDIA='present',QA_IMAGE='present')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertFalse(any(call[0]=='run' or 'up' in call or 'build' in call for call in calls))

    def test_reopening_gpu_keeps_device_even_when_host_probe_fails(self):
        result,calls=self.run_launcher(QA_EXISTING='running',QA_RUNNING_IMAGE='nexora-qa:gpu-20261005',QA_GPU_FAIL='true')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Automatic device selection: gpu',result.stdout)
        self.assertFalse(any(call[0]=='run' or 'build' in call or 'stop' in call for call in calls))

    def test_arm_engine_fails_clearly_before_build(self):
        result,calls=self.run_launcher(QA_ARCH='arm64')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires a Linux amd64 engine',result.stdout)
        self.assertFalse(any('build' in call or 'up' in call for call in calls))

    def test_changed_source_with_same_revision_rebuilds_stopped_image(self):
        result,calls=self.run_launcher(QA_IMAGE='present',QA_IMAGE_FINGERPRINT='old')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertTrue(any('build' in call for call in calls))

    def test_fingerprint_tracks_web_changes_but_excludes_models_and_user_data(self):
        result,_=self.run_launcher(QA_IMAGE='present')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        state=self.root/'local_only/docker_launcher.json'
        fingerprint=json.loads(state.read_text(encoding='utf-8-sig'))['source_fingerprint']
        (self.root/'models/yolo26n-pose.pt').write_bytes(b'changed weights')
        (self.root/'web/ignored-weights.pt').write_bytes(b'model ignored by Docker')
        (self.root/'local_only/user.txt').write_text('private QA data')
        result,calls=self.run_launcher(QA_IMAGE='present',QA_IMAGE_FINGERPRINT=fingerprint)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(json.loads(state.read_text(encoding='utf-8-sig'))['source_fingerprint'],fingerprint)
        (self.root/'web/new-ui.js').write_text('new UI')
        self.calls.unlink()
        result,calls=self.run_launcher(QA_IMAGE='present',QA_IMAGE_FINGERPRINT=fingerprint)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertNotEqual(json.loads(state.read_text(encoding='utf-8-sig'))['source_fingerprint'],fingerprint)
        self.assertTrue(any('build' in call for call in calls))

    def test_changed_source_never_recreates_active_session(self):
        result,calls=self.run_launcher(QA_EXISTING='running',QA_IMAGE_FINGERPRINT='old')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(any('build' in call or 'stop' in call or 'up' in call for call in calls))

    def test_corrupt_saved_settings_still_reopens_actual_container(self):
        state=self.root/'local_only/docker_launcher.json';state.parent.mkdir();state.write_text('{broken')
        result,calls=self.run_launcher(QA_EXISTING='running')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('settings are unreadable',result.stdout)
        self.assertIn('--no-recreate',next(call for call in calls if 'up' in call))

    def test_default_cpu_when_nvidia_probe_is_unavailable(self):
        result,calls=self.run_launcher(['-CheckOnly'])
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Automatic device selection: cpu',result.stdout)
        self.assertFalse(any(any(arg.endswith('compose.gpu.yaml') for arg in call) for call in calls))

    def test_explicit_cpu_overrides_automatic_nvidia_detection(self):
        result,calls=self.run_launcher(['-CheckOnly','-Device','cpu'],QA_NVIDIA='present')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('device=cpu',result.stdout)
        self.assertFalse(any(any(arg.endswith('compose.gpu.yaml') for arg in call) for call in calls))

    def test_no_build_never_builds_loads_or_starts_without_local_image(self):
        result, calls = self.run_launcher(['-NoBuild'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No local QA image', result.stdout)
        self.assertFalse(any('build' in c or 'up' in c or c[0] == 'load' for c in calls))
        self.assertTrue(list((self.root / 'local_only/qa').glob('docker_start_*.log')))

    def test_no_build_also_refuses_tar_loading(self):
        image = self.root / 'docker_images/nexora-qa-cpu-20261005.tar'
        image.parent.mkdir(); image.write_bytes(b'fake image data')
        result, calls = self.run_launcher(['-NoBuild'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Loading the TAR', result.stdout)
        self.assertFalse(any(c[0] == 'load' or 'build' in c or 'up' in c for c in calls))

    def test_no_build_can_start_an_existing_local_image(self):
        result, calls = self.run_launcher(['-NoBuild'], QA_IMAGE='present')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(any('up' in c for c in calls))
        self.assertFalse(any(c[0] == 'load' or 'build' in c for c in calls))

    def test_verify_without_container_never_builds_or_starts(self):
        result, calls = self.run_launcher(['-Action', 'verify'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No QA container', result.stdout)
        self.assertFalse(any('build' in c or 'up' in c or 'stop' in c for c in calls))

    def verify_running(self, **environment):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"health":{"detector":{"state":"ready"}}}')
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            return self.run_launcher(['-Action', 'verify'], QA_EXISTING='running',
                                     QA_RUNNING_PORT=str(server.server_port), **environment)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_verify_reports_existing_container_and_never_reconfigures_it(self):
        result, calls = self.verify_running()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report_file = next((self.root / 'local_only/qa').glob('docker_runtime_*.json'))
        report = json.loads(report_file.read_text(encoding='utf-8-sig'))
        self.assertTrue(report['passed'])
        self.assertEqual(report['container_image_id'], 'sha256:test')
        self.assertTrue(report['models_read_only'])
        self.assertTrue(any(c[0] == 'exec' for c in calls))
        self.assertFalse(any('build' in c or 'up' in c or 'stop' in c for c in calls))

    def test_failed_probe_saves_failure_report_and_returns_nonzero(self):
        result, calls = self.verify_running(QA_PROBE_FAIL='true')
        self.assertNotEqual(result.returncode, 0)
        report_file = next((self.root / 'local_only/qa').glob('docker_runtime_*.json'))
        self.assertFalse(json.loads(report_file.read_text(encoding='utf-8-sig'))['passed'])

    def test_writable_models_are_rejected_before_probe(self):
        result, calls = self.verify_running(QA_WRITABLE_MODELS='true')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('read-only model bind', result.stdout)
        self.assertFalse(any(c[0] == 'exec' for c in calls))

    def test_writable_fall_models_are_rejected_before_probe(self):
        result,calls=self.verify_running(QA_WRITABLE_FALL_MODELS='true')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('read-only fall model bind',result.stdout)
        self.assertFalse(any(c[0]=='exec' for c in calls))

    def test_windows_container_mode_is_rejected_before_build(self):
        result, calls = self.run_launcher(QA_ENGINE='windows')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Select Linux containers', result.stdout)
        self.assertFalse(any('build' in call for call in calls))

    def test_check_only_does_not_build_start_or_write_state(self):
        result, calls = self.run_launcher(['-CheckOnly'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any('build' in call or 'up' in call for call in calls))
        self.assertFalse((self.root / 'local_only').exists())

    def test_busy_port_is_skipped_and_no_other_service_is_stopped(self):
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0)); occupied.listen()
            result, calls = self.run_launcher(['-Port', str(occupied.getsockname()[1])], QA_IMAGE='present')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = json.loads((self.root / 'local_only/docker_launcher.json').read_text(encoding='utf-8-sig'))
            self.assertNotEqual(state['port'], occupied.getsockname()[1])
        self.assertFalse(any('stop' in c or 'down' in c or 'rm' in c for c in calls))

    def test_build_failure_never_attempts_to_start(self):
        result, calls = self.run_launcher(QA_BUILD_FAIL='true')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(any('build' in c for c in calls))
        self.assertFalse(any('up' in c for c in calls))
        log_file = next((self.root / 'local_only/qa').glob('docker_start_*.log'))
        self.assertIn('Docker command failed', log_file.read_text(encoding='utf-8-sig'))

    def test_reopening_running_session_does_not_build_or_recreate(self):
        result, calls = self.run_launcher(QA_EXISTING='running')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('18245', result.stdout)
        self.assertFalse(any('build' in c or 'stop' in c for c in calls))
        up = next(c for c in calls if 'up' in c)
        self.assertIn('--no-recreate', up)
        port_call = next(c for c in calls if any('NetworkSettings.Ports' in arg for arg in c))
        self.assertIn('{{json .NetworkSettings.Ports}}', port_call)

    def test_reopen_refuses_missing_localhost_port_without_recreating_session(self):
        result, calls = self.run_launcher(QA_EXISTING='running', QA_MISSING_PORT='true')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Could not read the current QA localhost port', result.stdout)
        self.assertFalse(any('build' in c or 'up' in c or 'stop' in c for c in calls))

    def test_old_running_web_is_rejected_without_recreating_or_stopping_reviews(self):
        result, calls = self.run_launcher(QA_EXISTING='running', QA_WEB_REVISION='old')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('older than the source', result.stdout)
        self.assertFalse(any('build' in c or 'up' in c or 'stop' in c for c in calls))

    def test_plain_start_updates_stopped_old_web_using_build_cache(self):
        result, calls = self.run_launcher(QA_IMAGE='present', QA_WEB_REVISION='old')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertTrue(any('build' in c for c in calls))
        self.assertTrue(any('up' in c for c in calls))
        self.assertFalse(any('--no-cache' in c or 'down' in c for c in calls))

    def test_no_build_rejects_stale_web_instead_of_opening_old_ui(self):
        result, calls = self.run_launcher(['-NoBuild'], QA_IMAGE='present', QA_WEB_REVISION='old')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('older web', result.stdout)
        self.assertFalse(any('build' in c or 'up' in c for c in calls))

    def test_changing_device_while_session_runs_requires_explicit_stop(self):
        result, calls = self.run_launcher(['-Device', 'gpu'], QA_EXISTING='running')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Export reviews', result.stdout)
        self.assertFalse(any('build' in c or 'stop' in c or 'up' in c for c in calls))

    def test_stopped_container_can_be_recreated_for_selected_device(self):
        result, calls = self.run_launcher(['-Device', 'gpu'], QA_EXISTING='stopped', QA_IMAGE='present')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        up = next(c for c in calls if 'up' in c)
        self.assertNotIn('--no-recreate', up)
        self.assertTrue(any(any('compose.gpu.yaml' in arg for arg in c) for c in calls))

    def test_failed_health_wait_shows_logs_and_does_not_open_or_save_ready_state(self):
        result, calls = self.run_launcher(QA_IMAGE='present', QA_UP_FAIL='true')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(any('logs' in c for c in calls))
        self.assertFalse((self.root / 'local_only/docker_launcher.json').exists())

    def test_supplied_image_with_matching_checksum_loads_without_build(self):
        image = self.root / 'docker_images/nexora-qa-cpu-20261005.tar'
        image.parent.mkdir(); image.write_bytes(b'fake image data')
        image.with_suffix('.tar.sha256').write_text(hashlib.sha256(image.read_bytes()).hexdigest())
        result, calls = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(any(c[0] == 'load' for c in calls))
        self.assertFalse(any('build' in c for c in calls))

    def test_image_checksum_failure_does_not_load_build_or_start(self):
        image = self.root / 'docker_images/nexora-qa-cpu-20261005.tar'
        image.parent.mkdir(); image.write_bytes(b'damaged image')
        image.with_suffix('.tar.sha256').write_text('0' * 64)
        result, calls = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checksum mismatch', result.stdout)
        self.assertFalse(any(c[0] == 'load' or 'build' in c or 'up' in c for c in calls))

    def test_export_creates_tar_with_checksum_and_never_exports_volumes(self):
        result, calls = self.run_launcher(['-Action', 'export'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        archive = self.root / 'local_only/qa/docker_images/nexora-qa-cpu-20261005.tar'
        self.assertEqual(archive.with_suffix('.tar.sha256').read_text().split()[0],
                         hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertFalse(any('build' in c or 'up' in c or 'volume' in c for c in calls))

    def test_explicit_gpu_export_is_not_replaced_by_saved_cpu_mode(self):
        state = self.root / 'local_only/docker_launcher.json'
        state.parent.mkdir(); state.write_text(json.dumps({'device': 'cpu', 'port': 18245}))
        result, calls = self.run_launcher(['-Action', 'export', '-Device', 'gpu'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        save = next(c for c in calls if c[0] == 'save')
        self.assertIn('nexora-qa:gpu-20261005', save)

    def test_export_rejects_stale_image_instead_of_shipping_old_source(self):
        result,calls=self.run_launcher(['-Action','export'],QA_IMAGE_FINGERPRINT='old')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('before exporting for QA',result.stdout)
        self.assertFalse(any(call[0]=='save' or 'build' in call for call in calls))

    def test_verify_rejects_stale_container_without_running_inference(self):
        result,calls=self.run_launcher(['-Action','verify'],QA_EXISTING='running',QA_IMAGE_FINGERPRINT='old')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(any(call[0]=='exec' or 'up' in call or 'build' in call for call in calls))

    def prepare_ready_bundle(self):
        archive=self.root/'docker_images/nexora-qa-ready-test.tar';archive.parent.mkdir()
        archive.write_bytes(b'prebuilt QA image')
        files=[dict(path='best_lstm_model.pth',sha256=hashlib.sha256((self.root/'best_lstm_model.pth').read_bytes()).hexdigest())]
        ready=dict(image='nexora-qa:ready-test',image_id='sha256:'+'1'*64,archive=archive.name,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),source_fingerprint='2'*64,
            web_revision=json.loads((self.root/'web/runtime_version.json').read_text())['revision'],files=files)
        (self.root/'QA_READY.json').write_text(json.dumps(ready))
        return archive

    def test_ready_bundle_imports_and_starts_cpu_without_build_or_pull(self):
        self.prepare_ready_bundle()
        result,calls=self.run_launcher()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertTrue(any(call[0]=='load' for call in calls))
        self.assertTrue(any('up' in call for call in calls))
        self.assertIn('Automatic device selection: cpu',result.stdout)
        self.assertFalse(any('build' in call or call[0]=='pull' for call in calls))

    def prepare_split_bundle(self):
        archive = self.prepare_ready_bundle()
        data = archive.read_bytes()
        ready_path = self.root / 'QA_READY.json'
        ready = json.loads(ready_path.read_text())
        ready['archive_parts'] = []
        parts = []
        for index, block in enumerate((data[:7], data[7:]), 1):
            part = self.root / (archive.name + f'.part{index:03d}')
            part.write_bytes(block)
            parts.append(part)
            ready['archive_parts'].append(dict(name=part.name, bytes=len(block), sha256=hashlib.sha256(block).hexdigest()))
        ready_path.write_text(json.dumps(ready))
        archive.unlink()
        return archive, parts

    def test_split_ready_image_combines_in_order_without_build(self):
        archive, parts = self.prepare_split_bundle()
        result, calls = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(archive.read_bytes(), b'prebuilt QA image')
        self.assertTrue(all(p.exists() for p in parts))
        self.assertTrue(any(c[0] == 'load' for c in calls))
        self.assertFalse(any('build' in c or c[0] == 'pull' for c in calls))

    def test_corrupt_or_missing_split_image_never_loads_or_starts(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                if missing:
                    parts[0].unlink()
                else:
                    archive, parts = self.prepare_split_bundle()
                    parts[0].write_bytes(b'damaged')
                result, calls = self.run_launcher()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(c[0] == 'load' or 'up' in c or 'build' in c for c in calls))
                self.assertFalse(archive.exists())

    def test_split_check_only_and_no_build_never_join_or_import(self):
        archive, _ = self.prepare_split_bundle()
        for args, expected in ((['-CheckOnly'], 0), (['-NoBuild'], 1)):
            result, calls = self.run_launcher(args)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            self.assertFalse(archive.exists())
            self.assertFalse(any(c[0] == 'load' or 'up' in c or 'build' in c for c in calls))

    def test_ready_bundle_imports_gpu_before_automatic_cuda_probe(self):
        self.prepare_ready_bundle()
        result,calls=self.run_launcher(QA_NVIDIA='present')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertLess(next(i for i,c in enumerate(calls) if c[0]=='load'),next(i for i,c in enumerate(calls) if c[0]=='run'))
        self.assertIn('nexora-qa:ready-test',next(c for c in calls if c[0]=='run'))
        self.assertIn('Automatic device selection: gpu',result.stdout)
        self.assertFalse(any('build' in call for call in calls))

    def test_ready_bundle_gpu_failure_uses_same_prebuilt_image_on_cpu(self):
        self.prepare_ready_bundle()
        result,calls=self.run_launcher(QA_NVIDIA='present',QA_GPU_FAIL='yes')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Automatic device selection: cpu',result.stdout)
        self.assertFalse(any('build' in call or call[0]=='pull' for call in calls))

    def test_ready_bundle_bad_archive_never_imports_or_starts(self):
        archive=self.prepare_ready_bundle();archive.write_bytes(b'damaged')
        result,calls=self.run_launcher()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('checksum mismatch',result.stdout)
        self.assertFalse(any(c[0]=='load' or 'up' in c or 'build' in c for c in calls))

    def test_ready_bundle_bad_model_never_imports_or_starts(self):
        self.prepare_ready_bundle();(self.root/'best_lstm_model.pth').write_bytes(b'damaged model')
        result,calls=self.run_launcher()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('file checksum mismatch',result.stdout)
        self.assertFalse(any(c[0]=='load' or 'up' in c or 'build' in c for c in calls))

    def test_ready_bundle_image_identity_mismatch_does_not_start(self):
        self.prepare_ready_bundle()
        result,calls=self.run_launcher(QA_IMAGE='present',QA_READY_IMAGE_ID='sha256:'+'3'*64)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(any('up' in c or 'build' in c for c in calls))

    def test_ready_bundle_stale_label_never_builds(self):
        self.prepare_ready_bundle()
        result,calls=self.run_launcher(QA_IMAGE='present',QA_IMAGE_FINGERPRINT='old')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('No build was attempted',result.stdout)
        self.assertFalse(any('build' in c or 'up' in c for c in calls))

    def test_ready_bundle_check_only_never_imports_or_probes(self):
        self.prepare_ready_bundle()
        result,calls=self.run_launcher(['-CheckOnly'],QA_NVIDIA='present')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertFalse(any(c[0] in ('run','load') or 'build' in c or 'up' in c for c in calls))

    def test_ready_bundle_refuses_rebuild(self):
        self.prepare_ready_bundle()
        result,calls=self.run_launcher(['-Rebuild'])
        self.assertNotEqual(result.returncode,0)
        self.assertIn('never builds',result.stdout)
        self.assertFalse(any('build' in c or 'up' in c for c in calls))


if __name__ == '__main__':
    unittest.main()
