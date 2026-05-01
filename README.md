# TRUTHGUARD - Unified Deepfake Detection Platform

TRUTHGUARD is a final-year project for image and video deepfake detection. It provides a single Streamlit application with role-based dashboards, CPU-safe inference, prediction logging, and clear model explanation sections for presentation and viva.

## Features

- Unified app entry point: `app.py`
- Image deepfake detection using an EfficientNet-B0 based CNN classifier
- Video deepfake detection using frame sampling with an EfficientNet + Vision Transformer checkpoint when dependencies are available
- CPU-only model loading with safe fallback inference for demo stability
- Demo login users:
  - `admin` / `123`
  - `celebrity` / `123`
  - `analyst` / `123`
- Admin dashboard with detections, history, report upload, and user view
- Celebrity dashboard focused on media authenticity verification
- Analyst dashboard with confidence values, logs, and model details
- SQLite prediction history and CSV export

## Project Structure

```text
SGP/
|-- app.py
|-- README.md
|-- requirements.txt
|-- image_detector/
|   |-- models/
|   |-- datasets/
|   |-- inference/
|   |-- lightning_modules/
|   `-- tools/
|-- video_detector/
|   |-- efficient-vit/
|   |-- preprocessing/
|   |-- deep_fakes/
|   `-- data/
`-- truthguard/
    |-- auth.py
    |-- config.py
    |-- db.py
    |-- inference/
    `-- data/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app runs on CPU and accepts local image/video uploads from the Streamlit interface.

## Model Engineering Summary

Image detection uses CNN-based classification to identify visual manipulation patterns. Video detection extracts representative frames and evaluates spatial artifacts using an EfficientNet feature backbone with a Vision Transformer classifier when the local checkpoint and optional dependencies are available. Prediction results are stored with confidence, processing time, user role, and file name for demo review.
