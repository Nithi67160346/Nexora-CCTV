"""
features/base.py
=================
"สัญญา" กลางที่ทุกฟีเจอร์ (ปัจจุบันและอนาคต) ต้อง implement
Pipeline หลักไม่จำเป็นต้องรู้จัก logic เฉพาะของแต่ละฟีเจอร์เลย
รู้แค่ว่าทุกตัวมีเมธอด .process(frame_result) -> FrameResult เท่านั้น

การเพิ่มฟีเจอร์ใหม่ (Violence / Fall / Seizure ฯลฯ) = สร้างคลาสใหม่ที่ inherit จากนี่
แล้วไปลงทะเบียนใน config.py บรรทัดเดียว ไม่ต้องแก้ pipeline หลัก (Open/Closed Principle)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

if __package__ and __package__.count('.') >= 2:
    from ..core.schema import FrameResult
else:
    from core.schema import FrameResult


class BaseFeature(ABC):
    #: ชื่อฟีเจอร์ ใช้ tag ใน FeatureEvent.feature_name และ track.attributes
    name: str = "base_feature"

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @abstractmethod
    def process(self, frame: np.ndarray, frame_result: FrameResult) -> FrameResult:
        """
        รับเฟรมภาพดิบ (เผื่อฟีเจอร์ไหนต้องวิเคราะห์ pixel เพิ่ม เช่น pose estimation)
        และ FrameResult ที่มี tracked_objects จาก Core อยู่แล้ว (ไม่ต้อง detect/track ซ้ำ)

        ให้แก้ไข frame_result ในที่ (in-place) โดย:
          - เพิ่ม event ผ่าน frame_result.add_event(...)
          - เพิ่ม/แก้ state เฉพาะฟีเจอร์ผ่าน tracked_object.attributes[self.name] = ...
        แล้ว return frame_result กลับ เพื่อส่งต่อให้ feature ตัวถัดไปใน pipeline
        """
        raise NotImplementedError

    def reset(self) -> None:
        """เรียกตอนเริ่มสตรีมใหม่ / รีสตาร์ทกล้อง เผื่อฟีเจอร์มี state ภายใน เช่น ตำแหน่งเดิมของ track"""
        return None
