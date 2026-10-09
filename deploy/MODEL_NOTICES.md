# Model distribution

Model binaries are distributed with this project's Release, separately from Git source. `models.manifest.json` identifies their exact bytes; it does not measure accuracy or establish training-data provenance.

- YOLO pose checkpoints are upstream Ultralytics models. Follow the applicable upstream licensing terms: https://www.ultralytics.com/license
- YuNet / SFace files retain their supplied license notices in `models/face/`.
- `best_lstm_model.pth` and `Fall/models_yolo/*.pkl` are the team's supplied checkpoints. This repository does not supply their training dataset or grant a new third-party dataset license. Class mapping and operational limitations are documented in the project README.

The bundle contains no training videos, patient information, face enrollment database, user reviews, or credentials. Only load checkpoints obtained from a trusted source. The Release is experimental and model scores are not a validated accuracy claim.
