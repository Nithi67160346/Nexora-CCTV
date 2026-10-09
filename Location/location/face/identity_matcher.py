"""
IdentityMatcher      : embedding หนึ่งตัว → ผู้สมัคร (หรือ unknown) — ห้าม force match
IdentityStateManager : temporal voting ต่อ track — ต้องชนะ N จาก M ครั้งก่อนยืนยัน
                       และเปลี่ยนตัวตนที่ยืนยันแล้วยากกว่ายืนยันครั้งแรก (กันสลับไปมา)
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from ..models import Identity


@dataclass
class MatchResult:
    resident_id: str | None     # None = unknown
    similarity: float
    runner_up: float = 0.0
    reason: str = ""


class IdentityMatcher:
    def __init__(self, database, similarity_threshold: float = 0.70, ambiguity_margin: float = 0.03):
        self.db = database
        self.threshold = float(similarity_threshold)
        self.margin = float(ambiguity_margin)

    def match(self, embedding) -> MatchResult:
        hits = self.db.search(embedding, top_k=2)
        if not hits:
            return MatchResult(None, 0.0, reason="empty_database")
        best_id, best = hits[0]
        second = hits[1][1] if len(hits) > 1 else 0.0
        if best < self.threshold:
            return MatchResult(None, best, second, "below_threshold")
        if len(hits) > 1 and best - second < self.margin:
            return MatchResult(None, best, second, "ambiguous")
        return MatchResult(best_id, best, second, "match")


class IdentityStateManager:
    def __init__(self, confirmation_matches: int = 3, confirmation_window: int = 5,
                 switch_matches: int | None = None, name_lookup=None):
        self.n = int(confirmation_matches)
        self.window = int(confirmation_window)
        if self.n > self.window:
            raise ValueError("confirmation_matches ต้อง <= confirmation_window")
        # เปลี่ยนจากคนที่ยืนยันแล้วไปเป็นอีกคน ต้องได้คะแนนมากกว่าการยืนยันครั้งแรก
        self.switch_n = min(self.window, switch_matches or self.n + 1)
        self.name_lookup = name_lookup or (lambda rid: None)
        self._votes: dict[tuple, deque] = {}
        self._confirmed: dict[tuple, Identity] = {}

    def observe(self, track_key, match: MatchResult) -> tuple[Identity | None, bool]:
        """คืน (ตัวตนที่ยืนยันแล้วหรือ None, เพิ่งยืนยัน/เปลี่ยนในรอบนี้หรือไม่)"""
        votes = self._votes.setdefault(track_key, deque(maxlen=self.window))
        votes.append((match.resident_id, match.similarity))

        counts = Counter(r for r, _ in votes if r is not None)
        current = self._confirmed.get(track_key)
        if not counts:
            return current, False

        top_id, top_n = counts.most_common(1)[0]
        need = self.n if current is None else self.switch_n
        if (current is None or top_id != current.identity_id) and top_n >= need:
            sims = [s for r, s in votes if r == top_id]
            ident = Identity(top_id, self.name_lookup(top_id) or top_id,
                             "face_recognition", sum(sims) / len(sims))
            self._confirmed[track_key] = ident
            return ident, True

        if current is not None and top_id == current.identity_id:
            sims = [s for r, s in votes if r == top_id]
            current.confidence = sum(sims) / len(sims)
        return current, False

    def get(self, track_key):
        return self._confirmed.get(track_key)

    def transfer(self, old_key, new_key):
        if old_key in self._confirmed and new_key not in self._confirmed:
            self._confirmed[new_key] = self._confirmed[old_key]

    def forget(self, track_key):
        self._votes.pop(track_key, None)
        self._confirmed.pop(track_key, None)

    def reset(self, source_id=None):
        if source_id is None:
            self._votes.clear()
            self._confirmed.clear()
        else:
            self._votes = {k: v for k, v in self._votes.items() if k[0] != source_id}
            self._confirmed = {k: v for k, v in self._confirmed.items() if k[0] != source_id}
