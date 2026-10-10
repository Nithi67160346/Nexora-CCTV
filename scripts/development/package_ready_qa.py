"""Package an existing image and models for double-click QA; never build or pull."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='nexora-qa:gpu-20261005')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists() or target.with_suffix('.zip').exists():
        raise FileExistsError('Use a new output path; existing QA packages are never overwritten.')
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', args.image], text=True))[0]
    labels = image['Config'].get('Labels') or {}
    revision = json.loads((ROOT / 'web/runtime_version.json').read_text())['revision']
    fingerprint = labels.get('nexora.source.fingerprint', '')
    if labels.get('nexora.web.revision') != revision or len(fingerprint) != 64:
        raise ValueError('Build the current image yourself first; no build is performed by this packager.')
    command = '''$projectRoot=$env:NEXORA_PACKAGE_ROOT; $tokens=$null; $errors=$null;
    $ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $projectRoot 'scripts/windows/run_docker.ps1'),[ref]$tokens,[ref]$errors);
    if($errors){throw $errors};
    $ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -in @('Get-QaFileHash','Get-QaSourceFingerprint')},$true) | ForEach-Object {Invoke-Expression $_.Extent.Text};
    Get-QaSourceFingerprint'''
    current = subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],
        env=dict(os.environ, NEXORA_PACKAGE_ROOT=str(ROOT)), text=True).strip()
    if current != fingerprint:
        raise ValueError('Image code differs from current source; rebuild it yourself before packaging.')
    if image['Architecture'] != 'amd64' or image['Os'] != 'linux':
        raise ValueError('The QA Windows package requires a Linux amd64 image.')
    files = ['start_docker.cmd', 'stop_docker.cmd', 'scripts/windows/run_docker.ps1',
             'compose.yaml', 'compose.gpu.yaml', 'web/runtime_version.json', 'docker/smoke.py',
             'best_lstm_model.pth']
    files += [f'models/{name}-pose.pt' for name in ('yolo26n','yolo26s','yolo26m','yolov8n','yolov8s','yolov8m')]
    files += [f'Fall/models_yolo/{name}' for name in (
        'ai1_isolation_forest_yolo.pkl','ai1_scaler_yolo.pkl','ai1_threshold_yolo.pkl',
        'ai2_fall_detector_yolo.pkl','feature_names_yolo.pkl')]
    files += [path.relative_to(ROOT).as_posix() for path in (ROOT/'models/face').glob('*')
              if path.is_file() and (path.suffix == '.onnx' or path.name.endswith('_LICENSE.txt'))]
    for name in files:
        if not (ROOT/name).is_file() or not (ROOT/name).stat().st_size:
            raise FileNotFoundError(name)
    target.mkdir(parents=True)
    manifest_files = []
    for name in files:
        destination = target/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
        manifest_files.append(dict(path=name, sha256=sha256(destination), bytes=destination.stat().st_size))
    (target/'videos').mkdir()
    (target/'docker_images').mkdir()
    suffix = '20261010-' + image['Id'].split(':')[1][:12]
    ready_tag = 'nexora-qa:ready-' + suffix
    archive = target/'docker_images'/('nexora-qa-ready-' + suffix + '.tar')
    subprocess.run(['docker', 'tag', image['Id'], ready_tag], check=True)
    print('Exporting existing image; no build/download. Keep this window open.', flush=True)
    subprocess.run(['docker', 'save', '--output', str(archive), ready_tag], check=True)
    archive_hash = sha256(archive)
    archive.with_suffix('.tar.sha256').write_text(f'{archive_hash}  {archive.name}\n', encoding='utf-8')
    ready = dict(format_version=1, image=ready_tag, image_id=image['Id'],
                 web_revision=revision, source_fingerprint=fingerprint,
                 archive=archive.name, archive_sha256=archive_hash,
                 supports=['cpu','nvidia-gpu'], installation='prebuilt-no-build', files=manifest_files)
    (target/'QA_READY.json').write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding='utf-8')
    (target/'เปิดอ่านก่อน_TH.txt').write_text(
        'NEXORA — ชุดพร้อมรันสำหรับ QA\n\n'
        '1. ใช้ Windows Intel/AMD x64 และติดตั้ง Docker Desktop แบบ Linux containers / WSL2\n'
        '2. แตก ZIP ทั้งหมดลงโฟลเดอร์ใหม่ ห้ามรันจากภายใน ZIP\n'
        '3. เปิด Docker Desktop รอ Engine running\n'
        '4. ดับเบิลคลิก start_docker.cmd แล้วรอเว็บเปิดเอง ไม่ต้องเปิด PowerShell หรือ build\n'
        '   ครั้งแรกตรวจ checksum และนำเข้า image ขนาดใหญ่ อาจใช้เวลาหลายนาที\n'
        '   ครั้งต่อไปใช้ image เดิม ไม่ต้องนำเข้าซ้ำ\n'
        '5. มี NVIDIA และ Docker CUDA พร้อม จะใช้ GPU อัตโนมัติ ไม่พร้อมใช้ CPU\n'
        '6. อัปโหลดคลิปและเลือกชื่อในรายการ หรือค้นหา webcam แล้วเปิดกล้อง\n'
        '7. ส่งออกรีวิว/โปรไฟล์ที่ต้องเก็บ แล้วใช้ stop_docker.cmd เพื่อหยุด\n\n'
        'ถ้าเปิดไม่ได้ ส่งไฟล์ log จาก local_only/qa กลับมาให้ทีม\n'
        'ชุดนี้ไม่ build หรือดาวน์โหลด image หากไฟล์หาย/ไม่ตรง ให้แตก ZIP ใหม่\n'
        'ตรวจสอบการทำงานได้ แต่คะแนนโมเดลไม่ใช่ความแม่นยำ ต้องให้คนรีวิวเหตุ\n', encoding='utf-8-sig')
    print('Creating one ZIP with image + models + launcher.', flush=True)
    zip_path = target.with_suffix('.zip')
    partial = zip_path.with_suffix('.building.zip')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as bundle:
        for path in sorted(target.rglob('*')):
            if path.is_file():
                bundle.write(path, target.name+'/'+path.relative_to(target).as_posix())
    partial.replace(zip_path)
    zip_path.with_suffix('.zip.sha256').write_text(f'{sha256(zip_path)}  {zip_path.name}\n', encoding='utf-8')
    print(f'READY: {zip_path}\n{zip_path.stat().st_size / 1024**3:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
