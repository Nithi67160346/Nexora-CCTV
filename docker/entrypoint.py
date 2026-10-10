"""Fail clearly before starting the QA container; never download missing weights."""
import importlib
import os
from pathlib import Path
import sys


def check_environment(root, environment, importer=importlib.import_module):
    weights = Path(root) / 'models/yolo26n-pose.pt'
    if not weights.is_file() or weights.stat().st_size == 0:
        raise RuntimeError('Missing models/yolo26n-pose.pt. Extract the complete QA ZIP; '
                           'models must be mounted at /app/models. No weights were downloaded.')
    device = environment.get('NEXORA_DEVICE', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise RuntimeError('NEXORA_DEVICE must be cpu or cuda')
    torch = importer('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU was requested but Torch CUDA is unavailable. Check Docker GPU '
                           'support and the NVIDIA driver, or launch the CPU container explicitly.')
    action_weights = Path(root)/'best_lstm_model.pth'
    if not action_weights.is_file() or action_weights.stat().st_size == 0:
        raise RuntimeError('Missing best_lstm_model.pth. Extract the current QA ZIP with the LSTM '
                           'checkpoint mounted at /app/best_lstm_model.pth. No weights were downloaded.')
    for name in ('ai1_isolation_forest_yolo.pkl','ai1_scaler_yolo.pkl','ai1_threshold_yolo.pkl',
                 'ai2_fall_detector_yolo.pkl','feature_names_yolo.pkl'):
        path=Path(root)/'Fall/models_yolo'/name
        if not path.is_file() or path.stat().st_size==0:
            raise RuntimeError('Missing Fall/models_yolo/'+name+'. Extract the complete current QA ZIP.')
    for module in ('numpy', 'yaml', 'cv2', 'torchvision', 'ultralytics', 'scipy', 'lap',
                   'fastapi', 'uvicorn', 'multipart', 'PIL', 'httpx', 'pandas', 'sklearn', 'joblib', 'threadpoolctl'):
        importer(module)
    # Fail before the web starts even when the saved profile disables Fall.
    # Presence of five files alone does not establish version/schema compatibility.
    bundle = importer('integration.fall_model').load_bundle(Path(root)/'Fall/models_yolo')
    if not bundle.loaded:
        raise RuntimeError('Fall model bundle could not be loaded.')
    print(f'NEXORA container checks passed: {device}; Fall54 loaded {bundle.checksum}.', flush=True)


def main():
    try:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        check_environment(root, os.environ)
    except Exception as error:
        print(f'NEXORA container cannot start: {error}', file=sys.stderr, flush=True)
        return 1
    command = sys.argv[1:]
    if not command:
        print('NEXORA container has no server command.', file=sys.stderr)
        return 1
    os.execvp(command[0], command)


if __name__ == '__main__':
    raise SystemExit(main())
