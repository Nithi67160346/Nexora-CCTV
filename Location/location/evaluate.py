"""
วัดความแม่นยำของ Location Feature เทียบกับคำตอบที่ติดป้ายไว้ (ground truth)

    # 1) รัน demo แล้วเก็บสถานะทุกเฟรม
    python demo.py --input clip.mp4 --config config_clip.yaml --source-id cam_x \
                   --model yolov8n.pt --no-show --frames-out frames.jsonl
    # 2) เทียบกับ ground truth
    python evaluate.py --frames frames.jsonl --gt eval/gt_clip.yaml [--update-metrics]

รูปแบบ ground truth (YAML) — บอกตำแหน่งที่ถูกต้องของ "บุคคลหลัก" เป็นช่วงเวลา (วินาที)
    clip: static2_elderly_enter_bed.mp4
    subject: ผู้สูงอายุในคลิป
    segments:
      - {start: 0.0,  end: 9.0,  zone: bed_area_A, room: bedroom_01, bed: bed_A}
      - {start: 9.0,  end: 41.9, zone: bedside_A,  room: bedroom_01, bed: null}
      - {start: 41.9, end: 45.0, zone: null}        # zone: null = ไม่อยู่ในภาพ
    tolerance_sec: 1.0     # ช่วงรอบจุดเปลี่ยนตำแหน่งที่ไม่นับคะแนน (ความคลาดเคลื่อนของการติดป้าย)

ตัวชี้วัด (ตรงกับหัวข้อ Metrics ในสเปก)
    detection_rate        สัดส่วนเฟรมที่คนอยู่ในภาพ และระบบติดตามได้
    zone/room/bed_accuracy  สัดส่วนเฟรม (ที่ติดตามได้) ที่ตำแหน่งตรงคำตอบ
    transition_accuracy   จุดเปลี่ยนตำแหน่งในคำตอบ ที่ระบบเปลี่ยนตามภายใน tolerance
    false_zone_switches   ครั้งที่ zone ของ track เดียวกันเปลี่ยน เกินจากจุดเปลี่ยนในคำตอบ (hysteresis ทำงานดีแค่ไหน)
    primary_flips         ครั้งที่ zone ของ "บุคคลหลัก" เปลี่ยนรวมกรณีสลับ track (มุมมองผู้ใช้ปลายทาง)
    id_switches           จำนวน track_id ที่เป็นบุคคลหลัก ≥ 5 เฟรม - 1 (หลัง re-id / dedupe ของ feature)
    last_seen_correct     ตำแหน่ง Last-Seen ตอนจบคลิป ตรงกับตำแหน่งสุดท้ายในคำตอบหรือไม่

บุคคลหลักในแต่ละเฟรม = track ที่กรอบใหญ่ที่สุด (เปลี่ยนได้ด้วย select: leftmost | rightmost)
คลิปหลายคน ใช้ subjects: [{name, select, segments}] และ people: [{start, end, count}]
    people_count_accuracy  สัดส่วนเฟรมที่จำนวนคนที่ติดตามได้ = จำนวนจริง
    frames_overcount       เฟรมที่นับคนเกิน (เงาสะท้อน / กรอบซ้อน)
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_frames(path):
    frames, final = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "final_last_seen" in d:
                final = d["final_last_seen"]
            else:
                frames.append(d)
    return frames, final


def gt_at(segments, t):
    for s in segments:
        if s["start"] <= t < s["end"]:
            return s
    return None


def near_boundary(segments, t, tol):
    return any(abs(t - s["start"]) < tol for s in segments[1:])


def primary(tracks, select="largest"):
    """เลือก track ของบุคคลที่กำลังให้คะแนน: largest | leftmost | rightmost"""
    if not tracks:
        return None

    def area(tr):
        x1, y1, x2, y2 = tr["bbox"]
        return (x2 - x1) * (y2 - y1)

    def cx(tr):
        return (tr["bbox"][0] + tr["bbox"][2]) / 2
    if select == "leftmost":
        return min(tracks, key=cx)
    if select == "rightmost":
        return max(tracks, key=cx)
    return max(tracks, key=area)


def evaluate(frames, final, gt):
    """GT แบบคนเดียว (segments) หรือหลายคน (subjects: [{name, select, segments}])"""
    if "subjects" in gt:
        out = {"clip": gt.get("clip"), "subjects": {}}
        for sub in gt["subjects"]:
            g = {**gt, "segments": sub["segments"], "select": sub.get("select", "largest")}
            g.pop("subjects")
            g.pop("people", None)
            r = evaluate(frames, final if sub.get("last_seen", False) else None, g)
            r.pop("clip", None)
            if not sub.get("last_seen", False):
                r.pop("last_seen_correct"), r.pop("last_seen_reported")
            out["subjects"][sub["name"]] = r
        out.update(count_people(frames, gt))
        return out
    return {**_evaluate_one(frames, final, gt), **count_people(frames, gt)}


def count_people(frames, gt):
    """people: [{start, end, count}] → สัดส่วนเฟรมที่จำนวนคนที่ระบบติดตามได้ตรงกับจริง"""
    segs = gt.get("people")
    if not segs:
        return {}
    ok = n = over = 0
    tol = float(gt.get("tolerance_sec", 1.0))
    for fr in frames:
        g = next((s for s in segs if s["start"] <= fr["t"] < s["end"]), None)
        if g is None or any(abs(fr["t"] - s["start"]) < tol for s in segs[1:]):
            continue
        n += 1
        k = len(fr["tracks"])
        ok += k == g["count"]
        over += k > g["count"]
    return {"people_count_accuracy": round(ok / n, 4) if n else None,
            "frames_overcount": over}


def _evaluate_one(frames, final, gt):
    segs = gt["segments"]
    select = gt.get("select", "largest")
    tol = float(gt.get("tolerance_sec", 1.0))
    present = tracked = 0
    hit = {"zone": 0, "room": 0, "bed": 0}
    counted = 0
    seq, ids = [], []

    for fr in frames:
        g = gt_at(segs, fr["t"])
        p = primary(fr["tracks"], select)
        if p is not None:
            seq.append((fr["t"], p["zone"]))
            ids.append(p["track_id"])
        if g is None or g.get("zone") is None:
            continue
        present += 1
        if p is None:
            continue
        tracked += 1
        if near_boundary(segs, fr["t"], tol):
            continue
        counted += 1
        for k in hit:
            if p.get(k) == g.get(k):
                hit[k] += 1

    # transitions ที่ควรเกิด (ข้ามช่วง "ไม่อยู่ในภาพ")
    visible = [s for s in segs if s.get("zone") is not None]
    expected = [(b["start"], b["zone"]) for a, b in zip(visible, visible[1:]) if a["zone"] != b["zone"]]
    reported = [(t, z) for (t, z), (_, zp) in zip(seq[1:], seq) if z != zp]
    ok_tr = sum(any(abs(t - te) <= max(tol, 2.0) and z == ze for t, z in reported) for te, ze in expected)

    from collections import Counter
    id_counts = Counter(ids)
    main_ids = [i for i, c in id_counts.items() if c >= 5]

    # การเปลี่ยน zone ภายใน track เดียวกัน
    per_track, track_switches = {}, 0
    for fr in frames:
        for tr in fr["tracks"]:
            z = tr["zone"]
            if tr["track_id"] in per_track and per_track[tr["track_id"]] != z:
                track_switches += 1
            per_track[tr["track_id"]] = z

    last_gt = visible[-1] if visible else None
    last_ok = None
    if final and last_gt:
        last_ok = final.get("zone_id") == last_gt["zone"]

    r = lambda a, b: round(a / b, 4) if b else None     # noqa: E731
    return {
        "clip": gt.get("clip"),
        "frames_person_present": present,
        "detection_rate": r(tracked, present),
        "frames_scored": counted,
        "zone_accuracy": r(hit["zone"], counted),
        "room_accuracy": r(hit["room"], counted),
        "bed_accuracy": r(hit["bed"], counted),
        "transitions_expected": len(expected),
        "transition_accuracy": r(ok_tr, len(expected)),
        "false_zone_switches": max(0, track_switches - ok_tr),
        "primary_flips": len(reported),
        "id_switches": max(0, len(main_ids) - 1),
        "last_seen_correct": last_ok,
        "last_seen_reported": final.get("zone_id") if final else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate location accuracy against ground truth")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--update-metrics", action="store_true",
                    help="บันทึกผลลง metrics/metrics.json (หัวข้อ evaluation)")
    args = ap.parse_args()

    frames, final = load_frames(args.frames)
    with open(args.gt, encoding="utf-8") as f:
        gt = yaml.safe_load(f)
    res = evaluate(frames, final, gt)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.update_metrics:
        path = os.path.join(HERE, "metrics", "metrics.json")
        data = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        data.setdefault("evaluation", {})[res["clip"]] = res
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[evaluate] บันทึกลง {path}")


if __name__ == "__main__":
    main()
