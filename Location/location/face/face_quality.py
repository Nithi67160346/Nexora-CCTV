"""
Face Quality Check — ไม่ส่งใบหน้าคุณภาพต่ำไปจับคู่ (ลด false match)

เกณฑ์
  - ขนาด     : ด้านสั้นของกรอบต้อง >= minimum_face_size px (ต่ำกว่านี้ reject ทันที)
  - ความคม   : variance ของ Laplacian (ภาพเบลอ = ต่ำ)
  - ความสว่าง : มืดหรือสว่างจ้าเกินไป = คะแนนลด
  - ความมั่นใจของ detector
เบลอมาก (sharpness < 0.15) หรือแสงแย่มาก (brightness < 0.3) → reject ทันที
score รวม 0–1 ต้อง >= quality_threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QualityResult:
    ok: bool
    score: float
    reasons: list = field(default_factory=list)
    parts: dict = field(default_factory=dict)


class FaceQualityChecker:
    def __init__(self, minimum_face_size: int = 40, quality_threshold: float = 0.5,
                 sharpness_ref: float = 100.0):
        self.min_size = int(minimum_face_size)
        self.threshold = float(quality_threshold)
        self.sharpness_ref = float(sharpness_ref)

    def check(self, face_crop, det_conf: float = 1.0) -> QualityResult:
        if face_crop is None or face_crop.size == 0:
            return QualityResult(False, 0.0, ["empty_crop"])

        h, w = face_crop.shape[:2]
        if min(h, w) < self.min_size:
            return QualityResult(False, 0.0, [f"too_small({w}x{h}<{self.min_size})"],
                                 {"size": min(h, w)})

        gray = face_crop.astype(np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)

        size_s = min(1.0, min(h, w) / (2.0 * self.min_size))
        sharp_s = min(1.0, _laplacian_var(gray) / self.sharpness_ref)
        mean = float(gray.mean())
        bright_s = 1.0 if 50 <= mean <= 210 else max(0.0, 1.0 - min(abs(mean - 50), abs(mean - 210)) / 50.0)
        conf_s = float(max(0.0, min(1.0, det_conf)))

        score = 0.3 * size_s + 0.3 * sharp_s + 0.2 * bright_s + 0.2 * conf_s
        reasons = []
        if sharp_s < 0.3:
            reasons.append("blurry")
        if bright_s < 0.5:
            reasons.append("bad_lighting")
        # hard gate: เบลอมากหรือมืด/จ้ามาก reject เลย ไม่ให้ขนาด/conf มาชดเชย
        hard_fail = sharp_s < 0.15 or bright_s < 0.3
        ok = score >= self.threshold and not hard_fail
        if score < self.threshold:
            reasons.append(f"score {score:.2f} < {self.threshold}")
        return QualityResult(ok, round(score, 4), reasons,
                             {"size": size_s, "sharpness": sharp_s,
                              "brightness": bright_s, "det_conf": conf_s})


def _laplacian_var(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())
