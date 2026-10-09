"""Bounded container probe: no stream changes, downloads, training, or full videos."""
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import ProxyHandler, build_opener

MODELS = ('yolo26n-pose.pt', 'yolo26s-pose.pt', 'yolo26m-pose.pt',
          'yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt')
FALL_MODELS = ('ai1_isolation_forest_yolo.pkl','ai1_scaler_yolo.pkl','ai1_threshold_yolo.pkl',
               'ai2_fall_detector_yolo.pkl','feature_names_yolo.pkl')


def fetch(path):
    with build_opener(ProxyHandler({})).open('http://127.0.0.1:8080' + path, timeout=10) as response:
        return response.read()


def run_probe(root=Path('/app'), get=fetch, predict=None, command=subprocess.run):
    report = dict(passed=False, checked_at=datetime.now(timezone.utc).isoformat(),
                  scope='Container runtime smoke; no detection accuracy or manual browser QA claim',
                  checks=[], inference='not_run', synthetic_frames=0, full_video_runs=0)

    def check(name, operation):
        try:
            detail = operation()
            report['checks'].append(dict(name=name, passed=True, detail=detail))
            return detail
        except Exception as error:
            report['checks'].append(dict(name=name, passed=False, error=str(error)))
            return None

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def api():
        status = json.loads(get('/api/status'))
        options = json.loads(get('/api/runtime/options'))
        require(status.get('health', {}).get('detector', {}).get('state') == 'ready', 'Detector not ready')
        expected = os.environ.get('NEXORA_DEVICE', 'cpu')
        require(options.get('gpu_available') if expected == 'cuda' else True, 'Requested CUDA unavailable')
        available = {m['name'] for m in options.get('models', []) if m.get('available')}
        require(set(MODELS) <= available, 'Not all six QA pose models are available')
        require(options.get('ffmpeg_available'), 'Web runtime cannot find FFmpeg')
        return dict(detector='ready', configured_device=expected, current_device=options.get('current_device'),
                    available_models=sorted(available))

    check('web_api', api)

    def assets():
        html = get('/').decode('utf-8')
        for path in ('/static/vendor/tailwind.css', '/static/vendor/fontawesome/css/all.min.css'):
            require(path in html, 'Page does not use bundled asset: ' + path)
            require(len(get(path)) > 100, 'Empty bundled asset: ' + path)
        require(len(get('/static/vendor/fontawesome/webfonts/fa-solid-900.woff2')) > 100, 'Missing icon font')
        require('https://cdn.tailwindcss.com' not in html and 'https://cdnjs.cloudflare.com' not in html,
                'Page still depends on CSS/icon CDN')
        return 'Bundled CSS/icons/font served over HTTP'

    check('offline_assets', assets)

    def weights():
        sizes = {}
        for name in MODELS:
            path = root / 'models' / name
            require(path.is_file() and path.stat().st_size > 0, 'Missing/empty weight: ' + name)
            sizes[name] = path.stat().st_size
        for name in FALL_MODELS:
            path=root/'Fall/models_yolo'/name
            require(path.is_file() and path.stat().st_size>0,'Missing/empty fall weight: '+name)
            sizes['Fall/models_yolo/'+name]=path.stat().st_size
        return sizes

    check('local_weights', weights)

    def ffmpeg():
        result = command([os.environ.get('NEXORA_FFMPEG', '/usr/bin/ffmpeg'), '-version'],
                         capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.splitlines()[0]

    check('ffmpeg_binary', ffmpeg)

    def inference():
        status = json.loads(get('/api/status'))
        children = json.loads(get('/api/multistream'))
        require(isinstance(children, list) and all(isinstance(row, dict) for row in children), 'Unexpected multistream response')
        if status.get('is_running') or any(row.get('is_running') or row.get('state') == 'starting' for row in children):
            report['inference'] = 'skipped_active_session'
            return 'Existing sources preserved; run again while all QA sources are stopped for inference verification'
        device = os.environ.get('NEXORA_DEVICE', 'cpu')
        actual = (predict or predict_one)(root / 'models/yolo26n-pose.pt', device)
        require(str(actual).split(':')[0] == device, 'Inference device mismatch: ' + str(actual))
        report['inference'] = 'passed'
        report['synthetic_frames'] = 1
        return dict(actual_device=str(actual), frames=1, image='synthetic blank 64x64; no accuracy claim')

    # Failed preflight must not load weights or attempt inference.
    if all(row['passed'] for row in report['checks']):
        check('single_frame_inference', inference)
    report['passed'] = all(row['passed'] for row in report['checks'])
    return report


def predict_one(weights, device):
    import numpy as np
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if model.task != 'pose' or list(model.model.yaml.get('kpt_shape', [])) != [17, 3]:
        raise RuntimeError('Expected COCO17 pose weights')
    model.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64, device=device,
                  verbose=False, save=False)
    return model.predictor.device


if __name__ == '__main__':
    with contextlib.redirect_stdout(sys.stderr):
        report = run_probe()
    print(json.dumps(report, ensure_ascii=True))
    raise SystemExit(0 if report['passed'] else 1)
