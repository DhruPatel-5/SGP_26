# TRUTHGUARD – Unified Deepfake Detection Platform

TRUTHGUARD is a unified Streamlit platform that combines **image** and **video** deepfake detection into one professional, presentation-ready forensic dashboard.

## Core Features

- Single Streamlit application (`app.py`)
- Login/authentication (SQLite-backed user store)
- Unified dashboard with:
  - Image Deepfake Detection
  - Video Deepfake Detection
  - Detection History
  - Upload Reports
  - User Profile
  - ML Pipeline visibility
  - Admin panel (role-gated)
- Prediction logging in SQLite (`truthguard/data/truthguard.db`)
- CSV export for prediction history
- CPU-compatible inference path
- Preserved visibility of legacy training & preprocessing pipelines in `image_detector/` and `video_detector/`

## Project Structure

```text
SGP_26/
├── app.py
├── truthguard/
│   ├── auth.py
│   ├── config.py
│   ├── db.py
│   ├── data/
│   └── inference/
│       ├── image_service.py
│       └── video_service.py
├── image_detector/
└── video_detector/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Default login:
- username: `admin`
- password: `admin123`

## ML Engineering Visibility

### Image pipeline
- Training: `python image_detector/main_trainer.py --config image_detector/config.yaml`
- Dataset prep: `python image_detector/tools/split_dataset.py`
- Train/val split: `python image_detector/tools/split_train_val.py`
- Evaluation: `python image_detector/realeval.py`

### Video pipeline
- Face detection/extraction: `python video_detector/preprocessing/detect_faces.py`
- Crop extraction: `python video_detector/preprocessing/extract_crops.py`
- Training: `python video_detector/cross-efficient-vit/train.py`
- Evaluation: `python video_detector/cross-efficient-vit/test.py --config <config_path> --model_path <checkpoint>`

## Notes

- The unified frontend keeps image/video detectors as separate services internally.
- If pretrained checkpoints are unavailable, fallback CPU-safe inference logic is used so the platform remains demoable.
- For production, point `truthguard/config.py` to the trained checkpoints.
