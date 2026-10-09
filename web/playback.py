"""Bounded per-frame evidence for smooth browser playback; never invent AI results."""
from collections import OrderedDict
from contextlib import closing
from copy import deepcopy
import threading
import json
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4
import zlib


class AnnotationBuffer:
    def __init__(self, capacity=1800):
        self.capacity = capacity
        self.frames = OrderedDict()
        self.lock = threading.RLock()
        self.cache_dir = None
        self.connection = None
        self.cache_path = None
        self.cache_error = None

    def enable_disk(self, directory=None):
        """Optional disposable clip evidence; live streams keep only the RAM ring."""
        with self.lock:
            self.clear()
            self.cache_dir = Path(directory) if directory else None

    def _open_disk(self):
        if self.cache_dir is None or self.connection or self.cache_error:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Only our generated, expired SQLite files; never source video or reviews.
        for old in self.cache_dir.glob('*.sqlite'):
            try:
                if re.fullmatch(r'[0-9a-f]{32}\.sqlite', old.name) and time.time()-old.stat().st_mtime > 86400:
                    old.unlink()
            except OSError: pass  # An active Windows connection can hold the file.
        files = list(self.cache_dir.glob('*.sqlite'))
        if sum(p.stat().st_size for p in files) > 2*1024**3:
            raise OSError('พื้นที่เก็บผล AI เกิน 2 GB กรุณาตรวจ local_only/playback_cache')
        self.cache_path = self.cache_dir/(uuid4().hex+'.sqlite')
        self.connection = sqlite3.connect(self.cache_path, check_same_thread=False)
        self.connection.execute('CREATE TABLE frames (id INTEGER PRIMARY KEY, seconds REAL, payload BLOB)')
        self.connection.execute('CREATE INDEX frame_time ON frames(seconds)')

    def close(self):
        with self.lock:
            if self.connection:
                try:
                    self.connection.commit()
                except sqlite3.Error as error:
                    self.cache_error = str(error)
                finally:
                    self.connection.close()
                    self.connection = None

    def clear(self):
        with self.lock:
            self.close()
            self.frames.clear()
            self.cache_path = None
            self.cache_error = None

    def add(self, context, fall_module, alerts, retained_ids=()):
        people = []
        for person in context.get('persons', []):
            p = deepcopy(person)
            p['fall'] = fall_module.status(context['source_id'], p['track_id']) if fall_module else None
            p['alert'] = p['track_id'] in retained_ids or any(p['track_id'] in e.get('track_ids', []) and e.get('severity') in ('high','critical','medium','warning') for e in alerts)
            people.append(p)
        row = dict(frame_id=context['frame_id'], time_s=context['timestamp_ms']/1000, persons=people)
        if 'violence' in context:
            row['violence'] = deepcopy(context['violence'])
        with self.lock:
            self.frames[row['frame_id']] = row
            while len(self.frames) > self.capacity: self.frames.popitem(last=False)
            if self.cache_dir is not None and not self.cache_error:
                try:
                    self._open_disk()
                    if self.cache_path.stat().st_size > 512*1024**2:
                        raise OSError('ผล AI ของรอบนี้เกิน 512 MB เก็บผลเก่าได้ถึงขอบเขตนี้เท่านั้น')
                    blob = zlib.compress(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode(), 3)
                    self.connection.execute('INSERT OR REPLACE INTO frames VALUES (?,?,?)', (row['frame_id'],row['time_s'],blob))
                    # Readers use the same locked connection; commit in batches.
                    if row['frame_id'] % 100 == 0:self.connection.commit()
                except (OSError,sqlite3.Error,ValueError) as error:
                    self.cache_error = str(error)

    def near(self, seconds, radius=.6):
        with self.lock:
            if self.connection:
                try:
                    rows = self.connection.execute('SELECT payload FROM frames WHERE seconds BETWEEN ? AND ? ORDER BY seconds LIMIT 120', (seconds-radius, seconds+radius)).fetchall()
                    found = {r['frame_id']:r for r in (json.loads(zlib.decompress(row[0])) for row in rows)}
                    found.update({r['frame_id']:deepcopy(r) for r in self.frames.values() if abs(r['time_s']-seconds)<=radius})
                    return sorted(found.values(),key=lambda r:r['time_s'])
                except (sqlite3.Error,ValueError,zlib.error) as error:
                    self.cache_error = str(error)
            return [deepcopy(r) for r in self.frames.values() if abs(r['time_s']-seconds) <= radius]

    def info(self):
        with self.lock:
            return dict(disk_enabled=self.cache_dir is not None, error=self.cache_error,
                        first_memory_s=next(iter(self.frames.values()))['time_s'] if self.frames else None)

    @staticmethod
    def read_cached(directory, filename, seconds, radius=.6):
        """Read the original event's closed clip cache, never another run's rows."""
        if not isinstance(filename, str) or not re.fullmatch(r'[0-9a-f]{32}\.sqlite', filename):
            raise ValueError('ไม่มีข้อมูลผล AI ของรอบเดิม')
        root = Path(directory).resolve()
        path = (root/filename).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError('ผล AI ของรอบเดิมหมดอายุหรือไม่พร้อม')
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as connection:
            rows = connection.execute('SELECT payload FROM frames WHERE seconds BETWEEN ? AND ? ORDER BY seconds LIMIT 120',
                                      (seconds-radius, seconds+radius)).fetchall()
            return [json.loads(zlib.decompress(row[0])) for row in rows]
