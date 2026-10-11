"""Persist only the four feature switches in the existing Docker data volume."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

FEATURES={'fall','violence','location','wandering'}

class FeatureSettings:
    def __init__(self, path=None):
        self.path=Path(path) if path else None

    def load(self, config):
        result=deepcopy(config)
        if self.path is None or not self.path.exists():return result
        if self.path.stat().st_size>4096:raise ValueError('ไฟล์สถานะฟีเจอร์มีขนาดผิดปกติ')
        saved=json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(saved,dict) or set(saved)-FEATURES or any(not isinstance(v,bool) for v in saved.values()):
            raise ValueError('ไฟล์สถานะฟีเจอร์ไม่ถูกต้อง')
        for name,state in saved.items():
            if name in result.get('features',{}):result['features'][name]['enabled']=state
        return result

    def save(self, config):
        if self.path is None:return
        values={name:feature['enabled'] for name,feature in config.get('features',{}).items() if name in FEATURES and 'enabled' in feature}
        if any(not isinstance(v,bool) for v in values.values()):raise ValueError('สถานะฟีเจอร์ต้องเป็น true/false')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=self.path.parent,delete=False) as out:
                temporary=Path(out.name);json.dump(values,out);out.flush();os.fsync(out.fileno())
            os.replace(temporary,self.path)
        finally:
            if temporary is not None and temporary.exists():temporary.unlink()
