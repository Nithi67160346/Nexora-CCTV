import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('model_install', Path(__file__).resolve().parents[1] / 'scripts/install_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ModelInstallTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.source = self.root / 'source'
        self.destination = self.root / 'destination'
        self.source.mkdir()
        self.destination.mkdir()
        self.entries = []
        for name in ('models/yolo26n-pose.pt', 'best_lstm_model.pth'):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = name.encode()
            path.write_bytes(data)
            self.entries.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), required=True))

    def test_install_and_repeat_verify_only_weights(self):
        (self.source / 'private.txt').write_text('not a model')
        self.assertEqual(module.install(self.source, self.destination, self.entries), (2, 2))
        self.assertEqual(module.install(self.source, self.destination, self.entries), (0, 2))
        self.assertFalse((self.destination / 'private.txt').exists())

    def test_invalid_last_file_prevents_any_copy(self):
        (self.source / 'best_lstm_model.pth').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            module.install(self.source, self.destination, self.entries)
        self.assertEqual(list(self.destination.rglob('*')), [])

    def test_existing_model_requires_explicit_replace_before_any_copy(self):
        (self.destination / 'best_lstm_model.pth').write_bytes(b'old')
        with self.assertRaisesRegex(ValueError, 'Existing model differs'):
            module.install(self.source, self.destination, self.entries)
        self.assertFalse((self.destination / 'models').exists())
        module.install(self.source, self.destination, self.entries, replace=True)
        self.assertEqual(module.verify(self.destination, self.entries), self.entries)

    def test_required_missing_and_optional_missing(self):
        optional = dict(self.entries[0], path='models/optional.pt', required=False)
        self.assertEqual(len(module.verify(self.source, self.entries + [optional])), 2)
        (self.source / 'best_lstm_model.pth').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing required model'):
            module.verify(self.source, self.entries)

    def test_manifest_cannot_copy_outside_root_or_nonmodel_documents(self):
        for name in ('../private.pth', '/models/absolute.pt', 'models/../../private.pt',
                     'README.md', 'models\\escape.pt', 'C:/private.pt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.model_path(self.source, name)


if __name__ == '__main__':
    unittest.main()
