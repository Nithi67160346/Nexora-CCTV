"""Reuse a verified ready bundle for GitHub Release assets; never build/export an image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = args.bundle.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory.')
    ready = json.loads((original / 'QA_READY.json').read_text(encoding='utf-8'))
    archive_name = ready['archive']
    if Path(archive_name).name != archive_name:
        raise ValueError('Invalid image archive name.')
    archive = original / 'docker_images' / archive_name
    output.mkdir(parents=True)
    target = output / output.name
    target.mkdir()
    for entry in ready['files']:
        source = (original / entry['path']).resolve()
        if not source.is_relative_to(original) or checksum(source) != entry['sha256']:
            raise ValueError(f"Invalid original file: {entry['path']}")
        destination = target / entry['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Host launcher can change without rebuilding the immutable application image.
        if entry['path'] == 'scripts/windows/run_docker.ps1':
            source = ROOT / entry['path']
        shutil.copyfile(source, destination)
        entry.update(bytes=destination.stat().st_size, sha256=checksum(destination))
    (target / 'videos').mkdir()
    (target / 'docker_images').mkdir()
    digest = hashlib.sha256()
    parts = []
    print('Splitting the existing image, with checksum verification; no build/export.', flush=True)
    with archive.open('rb') as source:
        for index in range(1, 1000):
            remaining = 1800 * 1024 * 1024
            first = source.read(min(4 * 1024 * 1024, remaining))
            if not first:
                break
            part = output / (archive_name + f'.part{index:03d}')
            part_digest = hashlib.sha256()
            with part.open('xb') as destination:
                block = first
                while block:
                    destination.write(block)
                    digest.update(block)
                    part_digest.update(block)
                    remaining -= len(block)
                    if not remaining:
                        break
                    block = source.read(min(4 * 1024 * 1024, remaining))
            parts.append(dict(name=part.name, bytes=part.stat().st_size, sha256=part_digest.hexdigest()))
            print(f'Prepared {part.name}: {part.stat().st_size} bytes', flush=True)
    if digest.hexdigest() != ready['archive_sha256']:
        raise ValueError('Original image TAR checksum mismatch; do not publish these assets.')
    ready['archive_parts'] = parts
    (target / 'QA_READY.json').write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding='utf-8')
    guide = (
        'NEXORA QA — ชุดพร้อมรัน ไม่ต้อง build\n\n'
        'ดาวน์โหลด QA ZIP และ image .part001 / .part002 / .part003 ให้ครบจาก Release เดียวกัน\n'
        'วางไฟล์ทั้งสี่ไว้ในโฟลเดอร์เดียวกัน แล้วแตก ZIP ตรงนั้น\n'
        'เข้าโฟลเดอร์ที่แตก เปิด Docker Desktop รอ Engine running แล้วดับเบิลคลิก start_docker.cmd\n'
        'ตัวรันจะตรวจไฟล์ รวม image และนำเข้าเองครั้งแรก จากนั้นเปิดเว็บให้\n'
        'ครั้งต่อไปไม่ต้องรวม/นำเข้าซ้ำ ไม่ต้องเปิด PowerShell\n'
        'ใช้ Windows Intel/AMD x64 + Docker Desktop Linux containers / WSL2\n'
        'NVIDIA ที่ Docker รองรับจะใช้ GPU อัตโนมัติ ถ้าไม่พร้อมใช้ CPU\n'
        'เผื่อพื้นที่อย่างน้อย 35 GB สำหรับไฟล์ดาวน์โหลด image และข้อมูลทดสอบ\n'
        'หยุดด้วย stop_docker.cmd โดยข้อมูลและรีวิวยังคงอยู่\n'
        'ถ้าเปิดไม่ได้ ส่ง log ใน local_only/qa ให้ทีม\n'
        'Linux server: ใช้โค้ด main และคู่มือ deploy/README_TH.md; โมเดลอยู่ใน QA ZIP\n'
    )
    (target / 'เปิดอ่านก่อน_TH.txt').write_text(guide, encoding='utf-8-sig')
    zip_path = output / (output.name + '.zip')
    with zipfile.ZipFile(zip_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(target.rglob('*')):
            if path.is_file():
                bundle.write(path, target.name + '/' + path.relative_to(target).as_posix())
    (output / 'QA_README_TH.txt').write_text(guide, encoding='utf-8-sig')
    assets = [zip_path, *(output / p['name'] for p in parts), output / 'QA_README_TH.txt']
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{checksum(p)}  {p.name}\n' for p in assets), encoding='utf-8')
    print('Release assets ready: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
