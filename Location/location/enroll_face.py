"""
Face Enrollment — สำหรับ Prototype เท่านั้น (แยกจาก Main Runtime)

    python enroll_face.py --resident-id resident_001 --input resident_001_images/ --consent
    python enroll_face.py --remove resident_001
    python enroll_face.py --list

- เก็บเฉพาะ embedding ลงไฟล์ .npz — ไม่คัดลอก/ไม่บันทึกภาพใบหน้า
- ต้องใส่ --consent เพื่อยืนยันว่าเจ้าของใบหน้ายินยอมแล้ว
- ใช้ภาพของสมาชิกทีม/นักแสดงที่ยินยอมเท่านั้น ห้ามใช้ภาพผู้สูงอายุ/ผู้ป่วยจริง (PDPA)
- ภาพ 1 รูปต้องมีหน้าเดียว (ถ้าเจอหลายหน้าจะข้ามรูปนั้น กันผูกผิดคน)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from location.face import FaceQualityChecker, NpzFaceDatabase, resolve_path  # noqa: E402
from location.face.face_detector import create_face_detector, crop_face  # noqa: E402
from location.face.face_embedder import create_face_embedder  # noqa: E402
from location.module import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser(description="Enroll face embeddings (prototype only)")
    ap.add_argument("--resident-id")
    ap.add_argument("--input", help="โฟลเดอร์ภาพของ resident คนนี้")
    ap.add_argument("--name", help="display name (ถ้าไม่ใส่ใช้จาก config.residents)")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--db", help="override face_recognition.database_path")
    ap.add_argument("--consent", action="store_true", help="ยืนยันว่าได้รับความยินยอมแล้ว")
    ap.add_argument("--remove", metavar="RESIDENT_ID", help="ลบ embedding ของคนนี้ (ถอนความยินยอม)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    import cv2
    cfg = load_config(args.config)
    fr = cfg.get("face_recognition", {}) or {}
    db = NpzFaceDatabase(args.db or resolve_path(fr.get("database_path", "face_db.npz"), must_exist=False))

    if args.list:
        for rid in db.identities():
            print(f"{rid:20} {db.display_name(rid) or '':20} {len(db._emb[rid])} embeddings")
        return
    if args.remove:
        db.remove_identity(args.remove)
        print(f"[ลบแล้ว] {args.remove} → {db.path}")
        return

    if not (args.resident_id and args.input):
        ap.error("ต้องใส่ --resident-id และ --input")
    if not args.consent:
        sys.exit("[หยุด] ต้องใส่ --consent เพื่อยืนยันว่าเจ้าของใบหน้ายินยอมให้เก็บข้อมูล biometric")

    fd_cfg = dict(cfg.get("face_detection", {}) or {})
    fd_cfg["model_path"] = resolve_path(fd_cfg.get("model_path", "yolov8n-face.pt"))
    fr = dict(fr, embedder_model_path=resolve_path(fr.get("embedder_model_path",
                                                             "face_recognition_sface_2021dec.onnx")))
    detector = create_face_detector(fd_cfg)
    embedder = create_face_embedder(fr)
    quality = FaceQualityChecker(fd_cfg.get("minimum_face_size", 40), fd_cfg.get("quality_threshold", 0.5))

    files = sorted(f for f in glob.glob(os.path.join(args.input, "*")) if f.lower().endswith(IMG_EXT))
    if not files:
        sys.exit(f"ไม่พบภาพใน {args.input}")

    embeddings = []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"  ข้าม {os.path.basename(path)}: อ่านไม่ได้")
            continue
        h, w = img.shape[:2]
        faces = detector.detect_in_person(img, (0, 0, w, h))
        if len(faces) != 1:
            print(f"  ข้าม {os.path.basename(path)}: พบ {len(faces)} หน้า (ต้องมี 1)")
            continue
        q = quality.check(crop_face(img, faces[0].bbox), faces[0].conf)
        if not q.ok:
            print(f"  ข้าม {os.path.basename(path)}: คุณภาพต่ำ {q.reasons}")
            continue
        emb = embedder.embed(img, faces[0])
        if emb is not None:
            embeddings.append(emb)
            print(f"  ใช้ {os.path.basename(path)} (quality {q.score:.2f})")
        del img                                   # ไม่เก็บภาพไว้ที่ใด

    if len(embeddings) < 3:
        sys.exit(f"[หยุด] ได้ embedding แค่ {len(embeddings)} รูป — ควรมีอย่างน้อย 3 รูป (หลายมุม/แสง)")

    name = args.name or ((cfg.get("residents") or {}).get(args.resident_id) or {}).get("display_name")
    db.add_identity(args.resident_id, embeddings, name)
    print(f"[สำเร็จ] {args.resident_id}: {len(embeddings)} embeddings → {db.path}")
    print("  ⚠ ไฟล์นี้เป็นข้อมูล biometric — ห้าม commit ขึ้น GitHub / ห้ามอัปโหลด cloud")


if __name__ == "__main__":
    main()
