from pathlib import Path
import shutil
import subprocess
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from truthguard.auth import ensure_default_users, require_auth
from truthguard.config import REPORTS_DIR, SUPPORTED_IMAGE_TYPES, SUPPORTED_VIDEO_TYPES, UPLOAD_DIR
from truthguard.db import (
    fetch_predictions,
    fetch_reports,
    fetch_users,
    init_db,
    record_prediction,
    record_report,
)
from truthguard.inference.image_service import ImageDetectorService
from truthguard.inference.video_service import VideoDetectorService

st.set_page_config(page_title="TRUTHGUARD", page_icon="🛡️", layout="wide")

init_db()
ensure_default_users()
user = require_auth()

if "img_service" not in st.session_state:
    st.session_state["img_service"] = ImageDetectorService()
if "vid_service" not in st.session_state:
    st.session_state["vid_service"] = VideoDetectorService()

st.title("🛡️ TRUTHGUARD – Unified Deepfake Detection Platform")
st.caption("Image + Video Forensics | Audit Logging | ML Engineering Transparency")

with st.sidebar:
    st.markdown(f"### Welcome, {user['full_name']}")
    menu = st.radio(
        "Navigation",
        ["Dashboard", "Image Detection", "Video Detection", "Detection History", "Upload Reports", "User Profile", "ML Pipeline", "Admin"],
    )
    if st.button("Logout"):
        st.session_state.pop("user", None)
        st.rerun()


if menu == "Dashboard":
    rows = fetch_predictions(None if user["role"] == "admin" else user["username"])
    df = pd.DataFrame([dict(r) for r in rows])
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Detections", len(df))
    col2.metric("Deepfake Flags", int((df.get("predicted_label") == "Deepfake").sum()) if not df.empty else 0)
    avg_conf = round(float(df["confidence"].mean() * 100), 2) if not df.empty else 0
    col3.metric("Avg Confidence (%)", avg_conf)
    if not df.empty:
        st.dataframe(df[["detector_type", "source_file", "predicted_label", "confidence", "created_at"]], use_container_width=True)

elif menu == "Image Detection":
    st.subheader("Image Deepfake Detection")
    uploaded = st.file_uploader("Upload image", type=SUPPORTED_IMAGE_TYPES)
    if uploaded and st.button("Run image detector"):
        save_path = Path(UPLOAD_DIR) / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uploaded.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded.read())

        img = Image.open(save_path).convert("RGB")
        st.image(img, caption="Input image", use_container_width=True)
        result = st.session_state["img_service"].predict(img)
        st.success(f"Prediction: {result['label']} ({result['confidence']*100:.2f}%)")
        st.caption(f"Model source: {result['model_source']}")
        record_prediction(user["username"], "image", uploaded.name, result["label"], result["confidence"], result["processing_seconds"])

elif menu == "Video Detection":
    st.subheader("Video Deepfake Detection")
    uploaded = st.file_uploader("Upload video", type=SUPPORTED_VIDEO_TYPES)
    if uploaded and st.button("Run video detector"):
        save_path = Path(UPLOAD_DIR) / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uploaded.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded.read())

        st.video(str(save_path))
        result = st.session_state["vid_service"].predict(str(save_path))
        st.success(f"Prediction: {result['label']} ({result['confidence']*100:.2f}%)")
        st.caption(f"Frames analyzed: {result['frames_analyzed']} | Model source: {result['model_source']}")
        record_prediction(user["username"], "video", uploaded.name, result["label"], result["confidence"], result["processing_seconds"])

elif menu == "Detection History":
    st.subheader("Detection History")
    rows = fetch_predictions(None if user["role"] == "admin" else user["username"])
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        st.info("No predictions yet.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button("Export as CSV", df.to_csv(index=False).encode("utf-8"), file_name="truthguard_predictions.csv")

elif menu == "Upload Reports":
    st.subheader("Upload Reports")
    report = st.file_uploader("Upload report/document", type=["pdf", "docx", "csv", "txt"])
    if report and st.button("Store report"):
        out = Path(REPORTS_DIR) / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{report.name}"
        with open(out, "wb") as f:
            f.write(report.read())
        record_report(user["username"], report.name, str(out))
        st.success("Report uploaded.")

    reports = pd.DataFrame([dict(r) for r in fetch_reports(user["username"])])
    if not reports.empty:
        st.dataframe(reports[["report_name", "file_path", "created_at"]], use_container_width=True)

elif menu == "User Profile":
    st.subheader("User Profile")
    st.json(user)

elif menu == "ML Pipeline":
    st.subheader("ML Engineering Pipeline Visibility")
    st.markdown("""
    **Image model training**
    - `python image_detector/main_trainer.py --config image_detector/config.yaml`

    **Image dataset preprocessing**
    - `python image_detector/tools/split_dataset.py`
    - `python image_detector/tools/split_train_val.py`

    **Video model training**
    - `python video_detector/cross-efficient-vit/train.py`

    **Video preprocessing**
    - `python video_detector/preprocessing/detect_faces.py`
    - `python video_detector/preprocessing/extract_crops.py`

    **Evaluation**
    - `python image_detector/realeval.py`
    - `python video_detector/cross-efficient-vit/test.py --config <config_path> --model_path <checkpoint>`
    """)
    if st.button("Run quick pipeline proof (dry-run commands)"):
        commands = [
            ["python", "image_detector/main_trainer.py", "--help"],
            ["python", "video_detector/cross-efficient-vit/train.py", "--help"],
        ]
        outputs = []
        for cmd in commands:
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
                outputs.append(f"$ {' '.join(cmd)}\nexit={res.returncode}\n{res.stdout[:300]}\n{res.stderr[:200]}")
            except Exception as e:
                outputs.append(f"$ {' '.join(cmd)}\nerror={e}")
        st.code("\n\n".join(outputs))

elif menu == "Admin":
    if user["role"] != "admin":
        st.warning("Admin only view")
    else:
        st.subheader("Admin Panel")
        users_df = pd.DataFrame([dict(r) for r in fetch_users()])
        st.dataframe(users_df, use_container_width=True)
        if st.button("Archive uploads folder snapshot"):
            archive = Path(REPORTS_DIR) / f"uploads_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            shutil.make_archive(str(archive), "zip", UPLOAD_DIR)
            st.success(f"Created archive: {archive}.zip")
