"""
Demo — Location / Bed / Zone / Last-Seen (+ Optional Face)

    python demo.py --input location_test.mp4
    python demo.py --input location_test.mp4 --face                       # แสดงกรอบหน้า
    python demo.py --input location_test.mp4 --face-recognition           # + ระบุตัวตน
    python demo.py --input location_test.mp4 --save out.mp4 --no-show
    python demo.py --input location_test.mp4 --pick-zone bed_area_203_A   # คลิกหาพิกัด zone
    python demo.py --input 0                                              # webcam

⚠ ไฟล์นี้เท่านั้นที่เปิดวิดีโอ + รัน person detector/tracker (จำลอง SHARED AI CORE)
   Core module (module.py) ไม่เปิดกล้องเองและไม่โหลด person detector
⚠ ใช้ stock video ที่ถ่ายด้วยนักแสดงเท่านั้น — ห้ามใช้ CCTV จริงจากสถานดูแล (PDPA)

ปุ่ม: p = หยุด/เล่นต่อ   s = บันทึกภาพ   q / Esc = ออก
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from location import DEFAULT_CONFIG_PATH, LocationModule, load_config  # noqa: E402

ZONE_COLORS = {                     # BGR
    "room": (200, 200, 200), "bed": (255, 170, 60), "bedside": (255, 220, 120),
    "hallway": (120, 220, 120), "exit_corridor": (60, 60, 230), "exit": (60, 60, 230),
    "restricted_area": (40, 40, 200), "safe_zone": (80, 200, 80),
}


# ============================================================================
# SHARED AI CORE (stand-in) — ในระบบจริง main.py เป็นคนทำส่วนนี้ครั้งเดียวต่อเฟรม
# ============================================================================

class SharedCoreStandIn:
    def __init__(self, model="yolo26m.pt", tracker="bytetrack.yaml", conf=0.4, device=None):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.tracker, self.conf, self.device = tracker, conf, device

    def context(self, frame, source_id, frame_id, ts_ms, fps):
        r = self.model.track(frame, persist=True, classes=[0], conf=self.conf,
                             tracker=self.tracker, device=self.device, verbose=False)[0]
        persons = []
        b = r.boxes
        if b is not None and len(b) and b.id is not None:
            for xyxy, c, tid in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.id.cpu().numpy()):
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                persons.append({"track_id": int(tid), "class_id": 0, "class_name": "person",
                                "confidence": float(c), "bbox_xyxy": [x1, y1, x2, y2],
                                "center_xy": [(x1 + x2) / 2, (y1 + y2) / 2], "pose": None})
        h, w = frame.shape[:2]
        return {"schema_version": "1.0", "source_id": source_id, "frame_id": frame_id,
                "timestamp_ms": ts_ms, "fps": fps, "frame_width": w, "frame_height": h,
                "persons": persons}


# ============================================================================
# Overlay (ข้อความภาษาอังกฤษเท่านั้น — OpenCV วาดฟอนต์ไทยไม่ได้)
# ============================================================================

def draw_zones(img, zones_cfg, source_id):
    ov = img.copy()
    for zid, z in (zones_cfg or {}).items():
        sids = z.get("source_ids")
        if sids and source_id not in sids:
            continue
        pts = np.array(z["points"], np.int32)
        col = ZONE_COLORS.get(z.get("type"), (180, 180, 180))
        if z.get("type") != "room":
            cv2.fillPoly(ov, [pts], col)
        cv2.polylines(img, [pts], True, col, 2)
        cv2.putText(img, zid, tuple(pts[0] + [6, 22]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)


def fmt_clock(start: dt.datetime, ts_ms):
    return (start + dt.timedelta(milliseconds=ts_ms or 0)).strftime("%H:%M:%S") if ts_ms is not None else "-"


def short(v, prefix):
    return str(v).replace(prefix, "") if v else "-"


def draw_person(img, st, start_clock, face_info, role_fn=None):
    x1, y1, x2, y2 = [int(v) for v in st.last_bbox]
    known = st.identity.is_known
    role = role_fn(st.identity) if role_fn else "unknown"
    col = (255, 160, 60) if role == "caregiver" else ((60, 200, 60) if known else (0, 200, 255))
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    px, py = [int(v) for v in st.last_point]
    cv2.circle(img, (px, py), 5, col, -1)

    loc = st.location
    face_line = "Face: -"
    if face_info is not None and face_info.face_detected:
        fx1, fy1, fx2, fy2 = [int(v) for v in face_info.face_bbox]
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (255, 0, 255), 1)
        sim = face_info.match.similarity if face_info.match else None
        face_line = f"Face q={face_info.quality_score:.2f}" + (f" sim={sim:.2f}" if sim is not None else "")
    lines = [
        f"ID: {st.identity.identity_id or 'Unknown'} ({st.identity.source})",
        f"Role: {role_fn(st.identity) if role_fn else '-'}",
        f"Track: {st.track_id}",
        face_line,
        f"Room: {short(loc.room_id if loc else None, 'room_')}",
        f"Bed: {short(loc.bed_id if loc else None, 'bed_')}",
        f"Zone: {loc.zone_id if loc else '-'}",
        f"Last Seen: {fmt_clock(start_clock, st.last_seen_timestamp)}",
    ]
    tx, ty = x1, max(y1 - 8 - 18 * len(lines), 5)
    cv2.rectangle(img, (tx - 2, ty - 14), (tx + 250, ty + 18 * len(lines) - 8), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (tx, ty + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)


def draw_last_seen_panel(img, module, start_clock):
    lost = [s for s in module.state.tracks.values() if s.status != "active"][-6:]
    if not lost:
        return
    h = 24 + 20 * len(lost)
    cv2.rectangle(img, (10, 10), (470, 10 + h), (0, 0, 0), -1)
    cv2.putText(img, "Last seen (not in view)", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for i, s in enumerate(lost):
        loc = s.location
        who = s.identity.identity_id or f"track {s.track_id}"
        t = f"{who}: {loc.room_id if loc else '-'} / {loc.zone_id if loc else '-'} @ " \
            f"{fmt_clock(start_clock, s.last_seen_timestamp)} [{s.status}]"
        cv2.putText(img, t, (18, 52 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def blur_faces(img, module):
    for s in module.state.tracks.values():
        f = s.last_face
        if s.status == "active" and f is not None and f.face_detected:
            x1, y1, x2, y2 = [max(0, int(v)) for v in f.face_bbox]
            roi = img[y1:y2, x1:x2]
            if roi.size:
                img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), 12)


# ============================================================================
# เครื่องมือหาพิกัด zone
# ============================================================================

def pick_zone(frame, zone_id):
    pts, win = [], "Pick zone: L-click add | R-click undo | Enter done"

    def redraw():
        img = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(img, p, 5, (0, 0, 255), -1)
            if i:
                cv2.line(img, pts[i - 1], p, (0, 0, 255), 2)
        cv2.imshow(win, img)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
        redraw()

    redraw()
    cv2.setMouseCallback(win, on_mouse)
    while cv2.waitKey(20) & 0xFF not in (13, ord("q")):
        pass
    cv2.destroyAllWindows()
    print(f"\n# วางใน config.yaml ใต้ zones:\n  {zone_id}:\n    type: <room|bed|hallway|...>")
    print(f"    points: {[list(p) for p in pts]}\n")


# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Location feature demo")
    ap.add_argument("--input", required=True, help="ไฟล์วิดีโอ (.mp4/.mov/.avi) | 0 = webcam | rtsp://...")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--source-id", default="cam_room_203_01")
    ap.add_argument("--model", default="yolo26m.pt", help="person detector ของ shared core (เช่น yolov8n.pt ถ้าเครื่องช้า)")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--face", action="store_true", help="เปิด face detection (ไม่ระบุตัวตน)")
    ap.add_argument("--face-recognition", action="store_true", help="เปิด face recognition (ต้อง enroll ก่อน)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--pick-zone", metavar="ZONE_ID")
    ap.add_argument("--events-out", default=None, help="บันทึก event ทั้งหมดเป็น .jsonl")
    ap.add_argument("--frames-out", default=None, help="บันทึกสถานะทุกเฟรมเป็น .jsonl (ใช้กับ evaluate.py)")
    ap.add_argument("--metrics-out", default=os.path.join(HERE, "metrics", "metrics.json"))
    args = ap.parse_args()

    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"เปิด {args.input} ไม่ได้")
    ok, frame = cap.read()
    if not ok:
        sys.exit("อ่านเฟรมแรกไม่ได้")

    if args.pick_zone:
        pick_zone(frame, args.pick_zone)
        return

    cfg = load_config(args.config)
    over = {}
    if args.face or args.face_recognition:
        over["face_detection"] = {"enabled": True}
    if args.face_recognition:
        over["face_recognition"] = {"enabled": True}
        over["privacy"] = {"face_recognition_enabled": True}
    module = LocationModule().setup({"config_path": args.config, **over})
    print(f"[demo] face layer: {'on' if module.face else 'off'} | "
          f"recognition: {module.get_metrics()['face_recognition_active']}")

    model_path = args.model
    local_model = os.path.join(HERE, "..", "..", "local_only", "models", os.path.basename(model_path))
    if not os.path.exists(model_path) and os.path.exists(local_model):
        model_path = local_model                      # ใช้ไฟล์ใน local_only/models/ ถ้ามี
    core = SharedCoreStandIn(model_path, args.tracker, args.conf, args.device)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    live = isinstance(src, int) or str(src).startswith("rtsp")
    start_clock = dt.datetime.now()
    t_live0 = time.time()

    writer = None
    if args.save:
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    events_f = open(args.events_out, "w", encoding="utf-8") if args.events_out else None
    frames_f = open(args.frames_out, "w", encoding="utf-8") if args.frames_out else None

    frame_id, paused, all_events, core_ms = 0, False, [], []
    while True:
        if not paused:
            if frame_id > 0:
                ok, frame = cap.read()
                if not ok:
                    break
            ts = int((time.time() - t_live0) * 1000) if live else int(frame_id * 1000 / fps)

            t0 = time.perf_counter()
            context = core.context(frame, args.source_id, frame_id, ts, fps)
            core_ms.append((time.perf_counter() - t0) * 1000)
            events = module.process(frame, context)          # ← integration contract
            all_events += events
            if frames_f:
                tracks = []
                for st in module.state.tracks.values():
                    if st.source_id != args.source_id or st.status != "active" or st.last_seen_timestamp != ts:
                        continue
                    loc = st.location
                    tracks.append({"track_id": st.track_id, "bbox": [round(v, 1) for v in st.last_bbox],
                                   "room": loc.room_id if loc else None, "bed": loc.bed_id if loc else None,
                                   "zone": loc.zone_id if loc else None, "role": module.role_of(st.identity),
                                   "identity": st.identity.identity_id})
                frames_f.write(json.dumps({"frame_id": frame_id, "t": round(ts / 1000.0, 3),
                                           "tracks": tracks}, ensure_ascii=False) + "\n")
            for e in events:
                md = e["metadata"]
                loc = md.get("last_seen") or {"room_id": md.get("current_room"),
                                               "bed_id": md.get("current_bed"),
                                               "zone_id": md.get("current_zone")}
                extra = f" reid_from={md['reid_from_track']}" if md.get("reid_from_track") is not None else ""
                print(f"[{e['event_type']}] track={e['track_ids']} id={md.get('resident_id')} "
                      f"room={loc.get('room_id')} bed={loc.get('bed_id')} zone={loc.get('zone_id')}{extra}")
                if events_f:
                    events_f.write(json.dumps(e, ensure_ascii=False) + "\n")

            vis = frame.copy()
            draw_zones(vis, cfg.get("zones"), args.source_id)
            for st in module.state.tracks.values():
                if st.status == "active" and st.source_id == args.source_id:
                    draw_person(vis, st, start_clock, st.last_face, module.role_of)
            draw_last_seen_panel(vis, module, start_clock)
            if writer:
                out = vis.copy()
                if (cfg.get("privacy") or {}).get("mask_face_in_saved_evidence"):
                    blur_faces(out, module)
                writer.write(out)
            frame_id += 1

        if args.no_show:
            continue
        cv2.imshow("Location demo", vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("p"):
            paused = not paused
        if k == ord("s"):
            cv2.imwrite(f"location_{frame_id}.png", vis)

    for e in module.flush():
        all_events.append(e)
        if events_f:
            events_f.write(json.dumps(e, ensure_ascii=False) + "\n")
    cap.release()
    if writer:
        writer.release()
    if events_f:
        events_f.close()
    if frames_f:
        last = [module.get_last_seen(st.track_id, st.source_id) for st in module.state.tracks.values()]
        last = [r for r in last if r]
        final = max(last, key=lambda r: r["timestamp_ms"]) if last else None
        frames_f.write(json.dumps({"final_last_seen": final}, ensure_ascii=False) + "\n")
        frames_f.close()
    if not args.no_show:
        cv2.destroyAllWindows()
    write_perf_metrics(args.metrics_out, module, core_ms, frame_id, all_events)


def write_perf_metrics(path, module, core_ms, frames, events):
    m = module.get_metrics()
    core = sorted(core_ms)
    total_s = sum(core_ms) / 1000.0 + (m["avg_latency_ms"] or 0) * frames / 1000.0
    perf = {
        "frames": frames,
        "fps_end_to_end": round(frames / total_s, 2) if total_s else None,
        "location_module_avg_latency_ms": m["avg_latency_ms"],
        "location_module_p95_latency_ms": m["p95_latency_ms"],
        "shared_core_avg_latency_ms": round(sum(core) / len(core), 2) if core else None,
        "shared_core_p95_latency_ms": round(core[int(0.95 * (len(core) - 1))], 2) if core else None,
        "gpu_usage": None, "vram_usage_mb": None,
        "measured_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        import torch
        if torch.cuda.is_available():
            perf["vram_usage_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            perf["gpu_usage"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data.setdefault("performance", {}).update(perf)
    data.setdefault("location_runtime", {}).update({
        "zone_switches": m["zone_switches"],
        "tracking_loss_recoveries": m["tracking_loss_recoveries"],
        "events": {t: sum(e["event_type"] == t for e in events)
                   for t in ("location_update", "identity_update", "last_seen_update")},
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[demo] {frames} เฟรม | FPS ~{perf['fps_end_to_end']} | metrics → {path}")


if __name__ == "__main__":
    main()
