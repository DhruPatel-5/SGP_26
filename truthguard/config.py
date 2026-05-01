from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_MODEL_PATH = BASE_DIR / "image_detector" / "models" / "best_model-v3.pt"
VIDEO_MODEL_PATH = BASE_DIR / "video_detector" / "efficient-vit" / "pretrained_models" / "efficient_vit.pth"
VIDEO_CONFIG_PATH = BASE_DIR / "video_detector" / "efficient-vit" / "configs" / "architecture.yaml"
DB_PATH = BASE_DIR / "truthguard" / "data" / "truthguard.db"
UPLOAD_DIR = BASE_DIR / "truthguard" / "data" / "uploads"
REPORTS_DIR = BASE_DIR / "truthguard" / "data" / "reports"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]

DEFAULT_USERS = [
    {
        "username": "admin",
        "full_name": "TruthGuard Admin",
        "role": "admin",
        "password": "123",
        "email": "admin@truthguard.local",
    },
    {
        "username": "celebrity",
        "full_name": "Celebrity Demo User",
        "role": "celebrity",
        "password": "123",
        "email": "celebrity@truthguard.local",
    },
    {
        "username": "analyst",
        "full_name": "Forensic Analyst",
        "role": "analyst",
        "password": "123",
        "email": "analyst@truthguard.local",
    }
]
