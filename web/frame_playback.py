"""One rendered frame in flight; advance only after its browser displays it."""
from threading import Condition


class FramePlayback:
    def __init__(self):
        self.condition = Condition()
        self.session = self.client = None
        self.sequence = self.acknowledged = 0
        self.frame = None
        self.cancelled = True
        self.epoch = 0

    def reset(self, session, client):
        with self.condition:
            self.session, self.client = session, client
            self.sequence = self.acknowledged = 0
            self.frame = None
            self.cancelled = False
            self.epoch += 1
            self.condition.notify_all()

    def cancel(self):
        with self.condition:
            self.cancelled = True
            self.frame = None
            self.condition.notify_all()

    def discard(self):
        """Explicit seeking may abandon the frame at the old position."""
        with self.condition:
            self.acknowledged = self.sequence
            self.epoch += 1
            self.frame = None
            self.condition.notify_all()

    def publish(self, session, jpeg, frame_id, seconds, epoch=None):
        with self.condition:
            if self.cancelled or session != self.session:
                return False
            if epoch is not None and epoch != self.epoch:
                return True
            self.sequence += 1
            sequence = self.sequence
            self.frame = dict(jpeg=jpeg, sequence=sequence, frame_id=frame_id, seconds=seconds)
            self.condition.notify_all()
            while not self.cancelled and session == self.session and self.acknowledged < sequence:
                self.condition.wait()
            return not self.cancelled and session == self.session

    def _validate(self, session, client):
        if self.cancelled or session != self.session or client != self.client:
            raise ValueError('รอบคลิปเปลี่ยนแล้ว หรือคลิปนี้เปิดจากอีกแท็บ')

    def read(self, session, client, after):
        with self.condition:
            self._validate(session, client)
            return dict(self.frame) if self.frame and self.sequence > after else None

    def acknowledge(self, session, client, sequence):
        with self.condition:
            self._validate(session, client)
            if sequence <= 0 or sequence > self.sequence:
                raise ValueError('หมายเลขเฟรมไม่ตรงกับภาพที่ส่ง')
            if sequence == self.sequence:
                self.acknowledged = sequence
                self.condition.notify_all()
