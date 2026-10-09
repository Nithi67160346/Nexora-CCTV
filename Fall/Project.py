import os
import time
from collections import deque

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from ultralytics import YOLO

# ==========================================
# 1. ตั้งค่า Path บนเครื่องคอมพิวเตอร์
# ==========================================
PROJECT_ROOT = "."  # ใช้โฟลเดอร์ปัจจุบันเพราะ POSE_OUTPUT อยู่ในโฟลเดอร์เดียวกับ Project.py แล้ว

RAW_PATH = os.path.join(PROJECT_ROOT, "UR_FALL_RAW")
ADL_PATH = os.path.join(RAW_PATH, "ADL")
FALL_PATH = os.path.join(RAW_PATH, "Fall")

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "POSE_OUTPUT")
MODEL_DIR = os.path.join(OUTPUT_PATH, "models_yolo")
VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_PATH, "demo_videos")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. ตั้งค่า YOLO & อุปกรณ์ประมวลผล
# ==========================================
YOLO_WEIGHTS = "yolov8n-pose.pt"
DEVICE = "cpu"  # หากเครื่องมี GPU Nvidia ให้เปลี่ยนเป็น "cuda"
INFER_IMGSZ = 480
NUM_KEYPOINTS = 17

yolo_pose_model = YOLO(YOLO_WEIGHTS)

feature_names = []
for i in range(NUM_KEYPOINTS):
  feature_names.extend([f"kp_{i}_x", f"kp_{i}_y", f"kp_{i}_conf"])

# ==========================================
# 3. สถานะและระบบ Smoothing (Multi-Person)
# ==========================================
STATUS_COLOR = {
    "NORMAL": (0, 200, 0),
    "ABNORMAL_NOT_FALL": (0, 165, 255),
    "ALERT_FALL": (0, 0, 255),
}

STATUS_TEXT_TH = {
    "NORMAL": "Normal",
    "ABNORMAL_NOT_FALL": "Abnormal Movement",
    "ALERT_FALL": "ALERT: FALL!",
}

FALL_PROBA_THRESHOLD = 0.4

YOLO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


class FallEventSmoother:

  def __init__(
      self,
      window_size=10,
      fall_ratio_threshold=0.4,
      abnormal_ratio_threshold=0.3,
  ):
    self.window = deque(maxlen=window_size)
    self.fall_ratio_threshold = fall_ratio_threshold
    self.abnormal_ratio_threshold = abnormal_ratio_threshold

  def update(self, frame_status):
    self.window.append(frame_status)
    return self.decide()

  def decide(self):
    if len(self.window) == 0:
      return "NORMAL"
    n = len(self.window)
    fall_ratio = sum(1 for s in self.window if s == "ALERT_FALL") / n
    abnormal_ratio = (
        sum(1 for s in self.window if s in ("ALERT_FALL", "ABNORMAL_NOT_FALL"))
        / n
    )
    if fall_ratio >= self.fall_ratio_threshold:
      return "ALERT_FALL"
    elif abnormal_ratio >= self.abnormal_ratio_threshold:
      return "ABNORMAL_MOVEMENT"
    return "NORMAL"


smoothers = {}


def get_smoother(track_id):
  if track_id not in smoothers:
    smoothers[track_id] = FallEventSmoother(
        window_size=6, fall_ratio_threshold=0.3, abnormal_ratio_threshold=0.3
    )
  return smoothers[track_id]


def draw_yolo_skeleton(image_bgr, kp_xy, kp_conf, conf_threshold=0.3):
  for a, b in YOLO_SKELETON:
    if kp_conf[a] > conf_threshold and kp_conf[b] > conf_threshold:
      x1, y1 = int(kp_xy[a][0]), int(kp_xy[a][1])
      x2, y2 = int(kp_xy[b][0]), int(kp_xy[b][1])
      cv2.line(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
  for i in range(len(kp_xy)):
    if kp_conf[i] > conf_threshold:
      x, y = int(kp_xy[i][0]), int(kp_xy[i][1])
      cv2.circle(image_bgr, (x, y), 4, (255, 0, 0), -1)
  return image_bgr


def predict_and_track(
    image_bgr, model, scaler, threshold, ai2_model, feature_cols, persist=True
):
  h, w = image_bgr.shape[:2]
  results = yolo_pose_model.track(
      image_bgr,
      persist=persist,
      tracker="bytetrack.yaml",
      verbose=False,
      device=DEVICE,
      imgsz=INFER_IMGSZ,
  )
  r = results[0]

  display_frame = image_bgr.copy()
  people_results = []

  if r.keypoints is None or r.boxes is None or r.boxes.id is None:
    return display_frame, people_results

  track_ids = r.boxes.id.int().cpu().tolist()
  boxes_xyxy = r.boxes.xyxy.cpu().numpy()

  for i, track_id in enumerate(track_ids):
    kp_xy = r.keypoints.xy[i].cpu().numpy()
    kp_conf = (
        r.keypoints.conf[i].cpu().numpy()
        if r.keypoints.conf is not None
        else np.ones(NUM_KEYPOINTS)
    )

    # สัดส่วนขนาดกล่องของบุคคล
    bw = max(1.0, float(boxes_xyxy[i][2] - boxes_xyxy[i][0]))
    bh = max(1.0, float(boxes_xyxy[i][3] - boxes_xyxy[i][1]))
    aspect_ratio = bh / bw  # ท่ายืน: > 1.2, ท่านอน/ล้ม: < 0.90
    cx = (boxes_xyxy[i][0] + boxes_xyxy[i][2]) / 2.0
    cy = (boxes_xyxy[i][1] + boxes_xyxy[i][3]) / 2.0

    # มุมแกนลำตัว (เส้นระหว่างจุดกึ่งกลางไหล่ 5,6 กับสะโพก 11,12)
    sh_mid = (kp_xy[5] + kp_xy[6]) / 2.0
    hip_mid = (kp_xy[11] + kp_xy[12]) / 2.0
    dx = abs(hip_mid[0] - sh_mid[0])
    dy = abs(hip_mid[1] - sh_mid[1])
    torso_angle = np.degrees(np.arctan2(dy, dx + 1e-6))  # ยืน: ~70-90 องศา, นอน/ล้ม: < 45 องศา
    torso_dy_dx = dy / (dx + 1e-6)

    # สกัด Features แบบ Scale-Invariant (พิกัดสัมพัทธ์ต่อขนาดกล่องตัวเอง)
    # ทำให้โมเดลไม่ขึ้นกับตำแหน่งใกล้-ไกลกล้อง และไม่จำท่ายืนปนกับท่าล้ม
    features = []
    for k in range(NUM_KEYPOINTS):
      features.extend([
          (float(kp_xy[k][0]) - cx) / bw,
          (float(kp_xy[k][1]) - cy) / bh,
          float(kp_conf[k]),
      ])
    features.extend([aspect_ratio, torso_angle, torso_dy_dx])
    feat_df = pd.DataFrame([features], columns=feature_cols)

    # ค่าความน่าจะเป็นการล้มจากโมเดลใหม่
    ai2_proba = ai2_model.predict_proba(feat_df)[0, 1]

    # เงื่อนไขการตัดสิน:
    # 1. ล้มจริง: โมเดล AI ระบุความเสี่ยงสูง ร่วมกับสรีระเปลี่ยนสภาพเป็นแนวนอน (torso_angle < 45° และ aspect_ratio < 0.90)
    is_lying_down = (torso_angle < 45.0) and (aspect_ratio < 0.90)

    if is_lying_down and ai2_proba >= 0.40:
      raw_status = "ALERT_FALL"
    elif is_lying_down and aspect_ratio < 0.65:
      raw_status = "ALERT_FALL"
    elif ai2_proba >= 0.50:
      raw_status = "ABNORMAL_NOT_FALL"
    else:
      raw_status = "NORMAL"

    smoother = get_smoother(track_id)
    event_status = smoother.update(raw_status)

    people_results.append({
        "track_id": track_id,
        "raw_status": raw_status,
        "event_status": event_status,
        "aspect_ratio": aspect_ratio,
        "torso_angle": torso_angle,
        "bbox": boxes_xyxy[i],
    })

    display_frame = draw_yolo_skeleton(display_frame, kp_xy, kp_conf)
    color = STATUS_COLOR.get(event_status, (255, 255, 255))
    x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id}: {STATUS_TEXT_TH.get(event_status, event_status)}"
    cv2.rectangle(
        display_frame,
        (x1, max(0, y1 - 22)),
        (x1 + len(label) * 10, y1),
        color,
        -1,
    )
    cv2.putText(
        display_frame,
        label,
        (x1 + 3, max(15, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

  return display_frame, people_results


# ==========================================
# 4. ฟังก์ชันเปิด WebCam หรือรันไฟล์ Video
# ==========================================
def load_all_models():
  model_path = os.path.join(MODEL_DIR, "ai1_isolation_forest_yolo.pkl")
  if not os.path.exists(model_path):
    print("❌ ไม่พบไฟล์โมเดล กรุณาตรวจสอบโฟลเดอร์ models_yolo")
    return None
  model = joblib.load(os.path.join(MODEL_DIR, "ai1_isolation_forest_yolo.pkl"))
  scaler = joblib.load(os.path.join(MODEL_DIR, "ai1_scaler_yolo.pkl"))
  threshold = joblib.load(os.path.join(MODEL_DIR, "ai1_threshold_yolo.pkl"))
  ai2_model = joblib.load(os.path.join(MODEL_DIR, "ai2_fall_detector_yolo.pkl"))
  feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_names_yolo.pkl"))
  return model, scaler, threshold, ai2_model, feature_cols


def run_video(video_source=0, save_output=False, output_filename="result.mp4"):
  models = load_all_models()
  if models is None:
    return
  model, scaler, threshold, ai2_model, feature_cols = models

  cap = cv2.VideoCapture(video_source)
  if not cap.isOpened():
    print(f"❌ ไม่สามารถเปิดวิดีโอได้: {video_source}")
    return

  fps = int(cap.get(cv2.CAP_PROP_FPS))
  if fps <= 0:
    fps = 25
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  writer = None
  if save_output:
    out_path = os.path.join(VIDEO_OUTPUT_DIR, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    print(f"💾 จะบันทึกผลลัพธ์ไปที่: {out_path}")

  print(f"✅ กำลังเล่นวิดีโอ: {video_source} (กด 'q' เพื่อออก, 'space' เพื่อหยุด/เล่นต่อ)...")
  smoothers.clear()
  delay = max(1, int(1000 / fps))

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    display_frame, _ = predict_and_track(
        frame, model, scaler, threshold, ai2_model, feature_cols, persist=True
    )

    if writer:
      writer.write(display_frame)

    cv2.imshow("Elderly CCTV Fall Detection", display_frame)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
      break
    elif key == ord(" "):
      cv2.waitKey(0)

  cap.release()
  if writer:
    writer.release()
  cv2.destroyAllWindows()


def run_webcam():
  run_video(0)


if __name__ == "__main__":
  import sys

  if len(sys.argv) > 1:
    target_video = sys.argv[1]
    save = "--save" in sys.argv
    run_video(target_video, save_output=save)
  else:
    # หากไม่ได้ใส่พารามิเตอร์ ให้เปิดกล้อง WebCam ตามเดิม
    run_webcam()
