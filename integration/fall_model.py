"""The supplied Project.py 54-feature schema and RF/geometry decision policy."""
import hashlib
from pathlib import Path
import threading
import warnings

import numpy as np

MODEL_FILES = ('ai1_isolation_forest_yolo.pkl', 'ai1_scaler_yolo.pkl',
    'ai1_threshold_yolo.pkl', 'ai2_fall_detector_yolo.pkl', 'feature_names_yolo.pkl')
FEATURE_NAMES = [name for i in range(17) for name in
                 (f'norm_kp_{i}_x', f'norm_kp_{i}_y', f'kp_{i}_conf')]
FEATURE_NAMES += ['aspect_ratio', 'torso_angle', 'torso_dy_dx']
SKLEARN_VERSION = '1.9.1'
DECISION_POLICY = 'project54-rf-geometry-v1'
_cache = {}
_cache_lock = threading.Lock()


def pose_features(pose, bbox):
    """Exact bbox-relative arithmetic supplied in Fall/Project.py (16:03)."""
    pose = np.asarray(pose, dtype=float)
    box = np.asarray(bbox, dtype=float)
    if pose.shape != (17, 3) or box.shape != (4,) or not np.isfinite(pose).all() or not np.isfinite(box).all():
        raise ValueError('ข้อต่อหรือกรอบคนไม่ถูกต้อง')
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError('กรอบคนต้องมีความกว้างและสูง')
    bw, bh = max(1., float(x2-x1)), max(1., float(y2-y1))
    cx, cy = (x1+x2)/2., (y1+y2)/2.
    shoulder = (pose[5, :2]+pose[6, :2])/2.
    hip = (pose[11, :2]+pose[12, :2])/2.
    dx, dy = np.abs(hip-shoulder)
    normalized = pose.copy()
    normalized[:, 0] = (pose[:, 0]-cx)/bw
    normalized[:, 1] = (pose[:, 1]-cy)/bh
    return normalized.flatten().tolist()+[bh/bw, float(np.degrees(np.arctan2(dy, dx+1e-6))), float(dy/(dx+1e-6))]


def project_status(probability, aspect_ratio, torso_angle, fall_threshold=.4):
    lying = torso_angle < 45. and aspect_ratio < .90
    if lying and probability >= fall_threshold:
        return 'ALERT_FALL', 'rf_and_lying_pose'
    if lying and aspect_ratio < .65:
        return 'ALERT_FALL', 'lying_pose_geometry_override'
    if probability >= .50:
        return 'ABNORMAL_NOT_FALL', 'rf_abnormal_pose'
    return 'NORMAL', 'project_normal_pose'


def validate_models(directory):
    paths = [Path(directory)/name for name in MODEL_FILES]
    missing = [path.name for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError('โมเดลตรวจล้มไม่ครบใน Fall/models_yolo: '+', '.join(missing))
    return paths


class FallModelBundle:
    def __init__(self, paths):
        try:
            import joblib
            import sklearn
            from sklearn.exceptions import InconsistentVersionWarning
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as error:
            raise ValueError('ยังไม่มี dependency ตรวจล้ม ติดตั้ง integration/requirements.txt หรืออัปเดต Docker image') from error
        if sklearn.__version__ != SKLEARN_VERSION:
            raise ValueError(f'โมเดลตรวจล้มชุดนี้ต้องใช้ scikit-learn=={SKLEARN_VERSION}; พบ {sklearn.__version__}')
        # This delivery mixes legacy AI1/scaler (1.6.1, 51 features) with the
        # active RF (1.9.1, 54 features). Project.py no longer uses AI1/scaler.
        # Keep their files intact for provenance but never deserialize them.
        # Only the active RF and its column schema must load in this runtime.
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            artifacts = []
            for path in paths[3:]:
                try:
                    artifacts.append(joblib.load(path))
                except InconsistentVersionWarning as error:
                    raise ValueError(
                        f'{path.name} บันทึกด้วย scikit-learn {error.original_sklearn_version}; '
                        f'runtime ใช้ {error.current_sklearn_version} ต้องใช้เวอร์ชันเดียวกัน'
                    ) from error
            self.ai2, columns = artifacts
        self.ai1 = self.scaler = self.threshold = None
        self.inactive_artifacts = list(MODEL_FILES[:3])
        self.active_artifacts = list(MODEL_FILES[3:])
        if not isinstance(self.ai2, RandomForestClassifier):
            raise ValueError('ชนิดโมเดลตรวจล้มต้องเป็น RandomForestClassifier')
        if list(columns) != FEATURE_NAMES:
            raise ValueError('schema ตรวจล้มต้องเป็น Project.py 54 ค่าในลำดับที่กำหนด')
        if getattr(self.ai2, 'n_features_in_', None) != len(FEATURE_NAMES):
            raise ValueError('โมเดลตรวจล้มรับข้อมูลไม่ใช่ 54 ค่า')
        if hasattr(self.ai2, 'feature_names_in_') and list(self.ai2.feature_names_in_) != FEATURE_NAMES:
            raise ValueError('ลำดับ feature ของโมเดลตรวจล้มไม่ตรง')
        classes = self.ai2.classes_.tolist()
        if len(classes) != 2 or set(classes) != {0, 1}:
            raise ValueError('คลาสตรวจล้มต้องเป็น 0=ไม่ล้ม และ 1=ล้ม')
        self.fall_index = classes.index(1)
        # One batched call per frame; no per-person pools competing with YOLO.
        self.ai2.n_jobs = 1
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode()); digest.update(path.read_bytes())
        self.checksum = digest.hexdigest()
        self.loaded = True

    def predict(self, vectors, fall_threshold=.4):
        import pandas as pd
        if not len(vectors):return []
        values = np.asarray(vectors, dtype=float)
        if values.shape != (len(vectors), len(FEATURE_NAMES)) or not np.isfinite(values).all():
            raise ValueError('ข้อมูลตรวจล้มต้องมี 54 ค่าที่เป็นจำนวนจริง')
        matrix = pd.DataFrame(values, columns=FEATURE_NAMES)
        # Project.py predicts RF directly; legacy AI1/scaler are not loaded.
        predicted = np.asarray(self.ai2.predict_proba(matrix), dtype=float)
        if predicted.shape != (len(values), 2) or not np.isfinite(predicted).all() or np.any((predicted<0)|(predicted>1)):
            raise ValueError('RandomForest ให้คะแนนไม่ถูกต้อง')
        results = []
        for vector, probability in zip(values, predicted[:, self.fall_index]):
            aspect, angle, ratio = vector[-3:]
            status, basis = project_status(float(probability), float(aspect), float(angle), fall_threshold)
            results.append(dict(ai1_score=None, ai1_threshold=self.threshold, ai1_gate_used=False,
                probability_fall=float(probability), raw_status=status, decision_basis=basis,
                decision_policy=DECISION_POLICY, aspect_ratio=float(aspect), torso_angle=float(angle),
                torso_dy_dx=float(ratio), is_lying_down=bool(angle<45. and aspect<.90)))
        return results


def load_bundle(directory):
    paths = validate_models(directory)
    key = tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
    with _cache_lock:
        if key not in _cache:
            bundle = FallModelBundle(paths)
            _cache.clear()
            _cache[key] = bundle
        return _cache[key]
