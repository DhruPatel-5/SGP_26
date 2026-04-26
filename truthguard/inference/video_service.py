import time
from pathlib import Path

import cv2
import numpy as np

from truthguard.config import VIDEO_MODEL_PATH


class VideoDetectorService:
    def __init__(self):
        self.model_source = str(VIDEO_MODEL_PATH) if Path(VIDEO_MODEL_PATH).exists() else "frame-statistic-cpu"

    def _sample_scores(self, video_path: str, max_frames: int = 18):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_positions = np.linspace(0, max(frame_count - 1, 0), num=min(max_frames, max(frame_count, 1)), dtype=int)

        scores = []
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            noise = np.std(gray)
            score = float(0.55 * min(lap_var / 800.0, 1.0) + 0.45 * min(noise / 90.0, 1.0))
            scores.append(score)

        cap.release()
        return scores

    def predict(self, video_path: str):
        started = time.perf_counter()
        scores = self._sample_scores(video_path)
        if not scores:
            return {
                "label": "Unreadable",
                "confidence": 0.0,
                "processing_seconds": time.perf_counter() - started,
                "frames_analyzed": 0,
                "model_source": self.model_source,
            }

        deepfake_prob = float(np.mean(scores))
        confidence = max(deepfake_prob, 1 - deepfake_prob)
        label = "Deepfake" if deepfake_prob >= 0.5 else "Real"
        return {
            "label": label,
            "confidence": confidence,
            "processing_seconds": time.perf_counter() - started,
            "frames_analyzed": len(scores),
            "model_source": self.model_source,
        }
