import time
from pathlib import Path

import numpy as np
from PIL import Image

from truthguard.config import IMAGE_MODEL_PATH

try:
    import torch
    from torchvision.models import efficientnet_b0
    from torchvision import transforms
except Exception:  # pragma: no cover
    torch = None


class ImageDetectorService:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.model_source = "heuristic-cpu"
        self._load()

    def _load(self):
        if torch is None:
            return
        if not Path(IMAGE_MODEL_PATH).exists():
            return

        model = efficientnet_b0()
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location="cpu"))
        model.eval()
        self.model = model
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model_source = str(IMAGE_MODEL_PATH)

    def predict(self, image: Image.Image):
        started = time.perf_counter()
        if self.model is not None:
            tensor = self.preprocess(image.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                out = self.model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                conf, pred = torch.max(probs, dim=0)
            label = "Real" if pred.item() == 0 else "Deepfake"
            confidence = float(conf.item())
        else:
            arr = np.asarray(image.convert("L"), dtype=np.float32)
            score = float(np.std(arr) / 128.0)
            confidence = min(max(score, 0.51), 0.93)
            label = "Deepfake" if score > 0.58 else "Real"

        return {
            "label": label,
            "confidence": confidence,
            "processing_seconds": time.perf_counter() - started,
            "model_source": self.model_source,
        }
