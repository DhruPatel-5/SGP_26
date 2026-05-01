import sys
import time
from pathlib import Path

import cv2
import numpy as np

from truthguard.config import BASE_DIR, VIDEO_CONFIG_PATH, VIDEO_MODEL_PATH

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class VideoDetectorService:
    def __init__(self):
        self.model = None
        self.model_source = "frame-statistic-cpu"
        self.load_error = ""
        self._load_model()

    def _load_model(self) -> None:
        if torch is None or not Path(VIDEO_MODEL_PATH).exists():
            return

        try:
            config = self._load_config()
            model_dir = BASE_DIR / "video_detector" / "efficient-vit"
            if str(model_dir) not in sys.path:
                sys.path.insert(0, str(model_dir))

            from efficient_vit import EfficientViT

            model = EfficientViT(config=config, channels=1280, selected_efficient_net=0)
            state_dict = torch.load(VIDEO_MODEL_PATH, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self.model = model.to(torch.device("cpu"))
            self.model_source = str(VIDEO_MODEL_PATH)
        except Exception as exc:
            self.load_error = str(exc)
            self.model = None

    def _load_config(self) -> dict:
        try:
            import yaml

            with open(VIDEO_CONFIG_PATH, "r", encoding="utf-8") as stream:
                return yaml.safe_load(stream)
        except Exception:
            return {
                "model": {
                    "image-size": 224,
                    "patch-size": 7,
                    "num-classes": 1,
                    "dim": 1024,
                    "depth": 6,
                    "dim-head": 64,
                    "heads": 8,
                    "mlp-dim": 2048,
                    "emb-dim": 32,
                    "dropout": 0.15,
                    "emb-dropout": 0.15,
                }
            }

    def _sample_frames(self, video_path: str, max_frames: int = 12):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_total = min(max_frames, max(frame_count, 1))
        sample_positions = np.linspace(0, max(frame_count - 1, 0), num=sample_total, dtype=int)

        frames = []
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(frame)

        cap.release()
        return frames

    def _predict_with_model(self, frames):
        prepared = []
        for frame in frames:
            resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            prepared.append(resized)

        batch = torch.tensor(np.asarray(prepared)).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.sigmoid(logits).flatten().cpu().numpy()

        deepfake_prob = float(np.mean(probs))
        confidence = max(deepfake_prob, 1 - deepfake_prob)
        label = "Deepfake" if deepfake_prob >= 0.5 else "Real"
        return label, confidence

    def _predict_with_frame_statistics(self, frames):
        scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            noise = np.std(gray)
            score = float(0.55 * min(lap_var / 800.0, 1.0) + 0.45 * min(noise / 90.0, 1.0))
            scores.append(score)

        deepfake_prob = float(np.mean(scores))
        confidence = max(deepfake_prob, 1 - deepfake_prob)
        label = "Deepfake" if deepfake_prob >= 0.5 else "Real"
        return label, confidence

    def predict(self, video_path: str):
        started = time.perf_counter()
        frames = self._sample_frames(video_path)
        if not frames:
            return {
                "label": "Unreadable",
                "confidence": 0.0,
                "processing_seconds": time.perf_counter() - started,
                "frames_analyzed": 0,
                "model_source": self.model_source,
                "load_error": self.load_error,
            }

        if self.model is not None:
            label, confidence = self._predict_with_model(frames)
        else:
            label, confidence = self._predict_with_frame_statistics(frames)

        return {
            "label": label,
            "confidence": confidence,
            "processing_seconds": time.perf_counter() - started,
            "frames_analyzed": len(frames),
            "model_source": self.model_source,
            "load_error": self.load_error,
        }
