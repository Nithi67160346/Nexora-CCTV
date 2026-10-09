"""
core/capture.py
================
แยกส่วน "การดึงภาพจากกล้อง" ออกจาก Logic การประมวลผลอย่างเด็ดขาด
รองรับทั้งกล้อง USB, RTSP (IP Camera) และไฟล์วิดีโอ โดยอ่านภาพใน thread แยก
เพื่อไม่ให้การอ่านเฟรม (I/O bound) ไปบล็อกการประมวลผล AI (CPU/GPU bound)
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue

import cv2
import numpy as np


class VideoStream:
    """
    Threaded video capture. เรียก .start() แล้วดึงเฟรมล่าสุดด้วย .read()

    source: int (webcam index) หรือ str (RTSP URL / video file path)
    """

    def __init__(self, source: int | str, queue_size: int = 2, reconnect_delay: float = 2.0) -> None:
        self.source = source
        self.reconnect_delay = reconnect_delay
        self._cap: cv2.VideoCapture | None = None
        self._queue: Queue[np.ndarray] = Queue(maxsize=queue_size)
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "VideoStream":
        self._open()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        return self

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise ConnectionError(f"ไม่สามารถเปิดแหล่งวิดีโอได้: {self.source}")

    def _update_loop(self) -> None:
        while not self._stopped.is_set():
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok:
                # กล้อง IP หลุด -> พยายาม reconnect แทนที่จะตายทั้งโปรเซส
                self._cap.release()
                time.sleep(self.reconnect_delay)
                try:
                    self._open()
                except ConnectionError:
                    continue
                continue

            if self._queue.full():
                try:
                    self._queue.get_nowait()  # ทิ้งเฟรมเก่า ให้ pipeline ใช้เฟรมล่าสุดเสมอ (low-latency)
                except Empty:
                    pass
            try:
                self._queue.put_nowait(frame)
            except Full:
                pass

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "VideoStream":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
