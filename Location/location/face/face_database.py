"""
Face Database — เก็บ "embedding" ของผู้ที่ได้รับอนุญาต (ไม่เก็บภาพใบหน้าดิบ)

Prototype: ไฟล์ .npz ในเครื่อง (ไม่มีการอัปโหลดขึ้น cloud ใด ๆ)

⚠ ข้อจำกัดด้าน Privacy ของ Prototype
  - ไฟล์ .npz ไม่ได้เข้ารหัส ใครเข้าถึงไฟล์ได้ = เข้าถึงข้อมูล biometric ได้
  - ระบบจริงต้องเข้ารหัส at-rest, จำกัดสิทธิ์, มี audit log และมีขั้นตอนลบข้อมูลเมื่อถอนความยินยอม
  - ห้าม commit ไฟล์นี้ขึ้น GitHub (ใส่ใน .gitignore)
"""

from __future__ import annotations

import os

import numpy as np

from .face_embedder import l2_normalize


class FaceDatabase:
    """Interface ตามสเปก"""

    def add_identity(self, resident_id, embeddings, display_name=None):
        raise NotImplementedError

    def search(self, embedding, top_k: int = 2) -> list[tuple[str, float]]:
        """คืน [(resident_id, cosine_similarity), ...] เรียงมากไปน้อย (หนึ่งแถวต่อคน)"""
        raise NotImplementedError

    def remove_identity(self, resident_id):
        raise NotImplementedError

    def display_name(self, resident_id):
        return None


class InMemoryFaceDatabase(FaceDatabase):
    def __init__(self):
        self._emb: dict[str, np.ndarray] = {}
        self._names: dict[str, str] = {}

    def __len__(self):
        return len(self._emb)

    def identities(self):
        return list(self._emb)

    def add_identity(self, resident_id, embeddings, display_name=None):
        arr = np.stack([l2_normalize(e) for e in np.atleast_2d(np.asarray(embeddings, np.float32))])
        if resident_id in self._emb:
            arr = np.vstack([self._emb[resident_id], arr])
        self._emb[resident_id] = arr
        if display_name:
            self._names[resident_id] = display_name

    def remove_identity(self, resident_id):
        self._emb.pop(resident_id, None)
        self._names.pop(resident_id, None)

    def display_name(self, resident_id):
        return self._names.get(resident_id)

    def search(self, embedding, top_k: int = 2):
        if not self._emb or embedding is None:
            return []
        q = l2_normalize(embedding)
        scores = []
        for rid, arr in self._emb.items():
            sims = arr @ q
            # ใช้ค่าเฉลี่ยของ top-3 ตัวอย่าง → ทนต่อตัวอย่างแย่ 1–2 รูปมากกว่า max
            k = min(3, len(sims))
            scores.append((rid, float(np.sort(sims)[-k:].mean())))
        scores.sort(key=lambda t: -t[1])
        return scores[:top_k]


class NpzFaceDatabase(InMemoryFaceDatabase):
    def __init__(self, path: str, autosave: bool = True):
        super().__init__()
        self.path = path
        self.autosave = autosave
        if os.path.exists(path):
            self.load()

    def load(self):
        data = np.load(self.path, allow_pickle=False)
        ids = [str(x) for x in data["ids"]]
        names = [str(x) for x in data["names"]]
        emb = data["embeddings"]
        owner = [str(x) for x in data["owner"]]
        self._emb, self._names = {}, {}
        for rid, name in zip(ids, names):
            rows = emb[[o == rid for o in owner]]
            if len(rows):
                self._emb[rid] = rows.astype(np.float32)
            if name:
                self._names[rid] = name

    def save(self):
        ids = list(self._emb)
        owner, rows = [], []
        for rid in ids:
            for r in self._emb[rid]:
                owner.append(rid)
                rows.append(r)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        np.savez(self.path,
                 ids=np.array(ids, dtype=str),
                 names=np.array([self._names.get(i, "") for i in ids], dtype=str),
                 owner=np.array(owner, dtype=str),
                 embeddings=(np.array(rows, dtype=np.float32) if rows
                             else np.zeros((0, 0), np.float32)))

    def add_identity(self, resident_id, embeddings, display_name=None):
        super().add_identity(resident_id, embeddings, display_name)
        if self.autosave:
            self.save()

    def remove_identity(self, resident_id):
        super().remove_identity(resident_id)
        if self.autosave:
            self.save()
