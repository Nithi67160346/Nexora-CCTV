"""Identify this checkout for launcher reuse without exposing its filesystem path."""
import hashlib
import os
import json
from pathlib import Path


def launcher_info(root, revision=None):
    identity = os.path.normcase(str(Path(root).resolve()))
    version_path = Path(root)/'web/runtime_version.json'
    if revision is None and version_path.is_file():
        revision = json.loads(version_path.read_text(encoding='utf-8'))['revision']
    return dict(application='nexora-cctv', launcher_protocol=1,
                checkout_id=hashlib.sha256(identity.encode('utf-8')).hexdigest(), product_revision=revision)
