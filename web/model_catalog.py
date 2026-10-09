"""Explicit, local pose weights supported by the shared 17-joint pipeline."""
from pathlib import Path

POSE_MODELS = {'yolo26n-pose.pt': 'YOLO26n-pose',
               'yolo26s-pose.pt': 'YOLO26s-pose',
               'yolo26m-pose.pt': 'YOLO26m-pose',
               'yolov8n-pose.pt': 'YOLOv8n-pose',
               'yolov8s-pose.pt': 'YOLOv8s-pose',
               'yolov8m-pose.pt': 'YOLOv8m-pose'}


def resolve_pose_weights(root, name, current=None):
    if name not in POSE_MODELS:
        raise ValueError('เลือกโมเดล pose ในรายการ YOLO26 หรือ YOLOv8 ขนาด n/s/m เท่านั้น ระบบล้มต้องใช้จุดข้อต่อ')
    candidate = Path(root) / 'models' / name
    if not candidate.is_file() and current and Path(current).name == name:
        candidate = Path(current)
    if not candidate.is_file():
        raise ValueError(f'ยังไม่มี {name} ใน models กรุณาติดตั้งโมเดลจาก Release ด้วย scripts/install_models.py ก่อน')
    return candidate.resolve()


def model_options(root, current=None):
    result = []
    for name, label in POSE_MODELS.items():
        try:
            resolve_pose_weights(root, name, current)
            available = True
        except ValueError:
            available = False
        result.append(dict(name=name, label=label, available=available))
    return result
