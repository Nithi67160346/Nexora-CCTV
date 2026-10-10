# Runtime models

Model binaries are supplied in the QA ZIP attached to Release v0.2.0-rc.1. Verify exact bytes using deploy/models.manifest.json and scripts/install_models.py. They are not stored in Git.

The primary player uses Ultralytics YOLO pose, the supplied RandomForest Fall detector with 54 features and scikit-learn 1.9.1, and the supplied ResNet18 + LSTM v3 violence checkpoint. The v3 checkpoint uses two LSTM layers (512 to 256); the loader constructs the matching architecture and loads state strictly. Live temporal windows can differ from whole-clip uniform sampling. This model scores the image sequence and does not identify a perpetrator.

Legacy AI1 pickle files are included to retain the supplied bundle; they are optional and are not deserialized by the Fall runtime. No models were retrained for this release. Scores are model outputs, not validated accuracy. See deploy/MODEL_NOTICES.md for distribution notices.
