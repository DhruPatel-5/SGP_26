from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_MODEL_PATH = BASE_DIR / "image_detector" / "models" / "best_model-v3.pt"
VIDEO_MODEL_PATH = BASE_DIR / "video_detector" / "cross-efficient-vit" / "models" / "final_model.pth"
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
        "password": "admin123",
        "email": "admin@truthguard.local",
    }
]
