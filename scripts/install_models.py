"""Verify trusted release weights and copy only model files; no deserialization."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError(f'Invalid model path: {name}')
    if name != 'best_lstm_model.pth' and not name.startswith(('models/', 'Fall/models_yolo/')):
        raise ValueError(f'Unexpected model path: {name}')
    base = Path(root).resolve()
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f'Model path escapes directory: {name}')
    return candidate


def verify(root, entries):
    present = []
    for entry in entries:
        path = model_path(root, entry['path'])
        if not path.is_file():
            if entry['required']:
                raise ValueError(f"Missing required model: {entry['path']}")
            continue
        if path.stat().st_size != entry['bytes'] or checksum(path) != entry['sha256']:
            raise ValueError(f"Model checksum mismatch: {entry['path']}")
        present.append(entry)
    return present


def install(source, destination, entries, replace=False):
    # Validate the complete source and all destination conflicts before writing.
    present = verify(source, entries)
    pending = []
    for entry in present:
        target = model_path(destination, entry['path'])
        if target.exists():
            if target.is_file() and checksum(target) == entry['sha256']:
                continue
            if not replace or not target.is_file():
                raise ValueError(f"Existing model differs: {entry['path']}; inspect it before using --replace")
        pending.append((entry, target))
    for entry, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.nexora-model-', delete=False) as stream:
                staging = Path(stream.name)
                with model_path(source, entry['path']).open('rb') as original:
                    shutil.copyfileobj(original, stream)
            if staging.stat().st_size != entry['bytes'] or checksum(staging) != entry['sha256']:
                raise ValueError(f"Source changed while copying: {entry['path']}")
            staging.replace(target)
        finally:
            if staging is not None:
                staging.unlink(missing_ok=True)
    verify(destination, entries)
    return len(pending), len(present)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-dir', type=Path, help='Trusted extracted Release directory')
    action.add_argument('--check', action='store_true', help='Verify installed required and present optional models')
    parser.add_argument('--replace', action='store_true', help='Explicitly replace differing existing model files')
    args = parser.parse_args()
    entries = json.loads((ROOT / 'deploy/models.manifest.json').read_text(encoding='utf-8'))['files']
    try:
        if args.check:
            present = verify(ROOT, entries)
            print(f'Verified {len(present)} models; all required models present.')
        else:
            copied, total = install(args.from_dir, ROOT, entries, args.replace)
            print(f'Copied {copied} models; verified {total} models. No data, reviews or documentation copied.')
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
