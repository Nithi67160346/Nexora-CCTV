"""Download pinned OpenCV models only; no video, enrollment, or training."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = '47534e27c9851bb1128ccc0102f1145e27f23f98'
MODELS = (
    ('face_detection_yunet', 'face_detection_yunet_2023mar.onnx',
     '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4', 232589),
    ('face_recognition_sface', 'face_recognition_sface_2021dec.onnx',
     '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79', 38696353),
)


def main():
    destination = ROOT / 'models/face'
    destination.mkdir(parents=True, exist_ok=True)
    for folder, name, expected, size in MODELS:
        target = destination / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f'{name}: existing file checksum differs; leave it intact')
            print(f'{name}: verified existing model', flush=True)
        else:
            partial = target.with_suffix('.download')
            digest = hashlib.sha256(); count = 0
            url = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/{folder}/{name}'
            try:
                with urllib.request.urlopen(url, timeout=30) as source, partial.open('wb') as output:
                    while chunk := source.read(1024 * 1024):
                        count += len(chunk)
                        if count > size:
                            raise ValueError('Model exceeds pinned size')
                        output.write(chunk); digest.update(chunk)
                if count != size or digest.hexdigest() != expected:
                    raise ValueError(f'{name}: model integrity check failed')
                partial.replace(target)
                print(f'{name}: downloaded and verified SHA256', flush=True)
            finally:
                partial.unlink(missing_ok=True)
        license_path = destination / (folder + '_LICENSE.txt')
        if not license_path.exists():
            url = f'https://raw.githubusercontent.com/opencv/opencv_zoo/{REVISION}/models/{folder}/LICENSE'
            with urllib.request.urlopen(url, timeout=20) as response:
                license_path.write_bytes(response.read())
    print('Face models ready; no faces registered and no training performed.', flush=True)


if __name__ == '__main__':
    main()
