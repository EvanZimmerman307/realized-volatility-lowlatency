# src/pipeline/serve.py
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI
import onnxruntime as ort
import yaml

class Normalizer:
    """Apply train-split mean/std normalization before inference."""
    def __init__(self, manifest_path: str):
        m = json.loads(Path(manifest_path).read_text())
        self.mean = np.array(m["norm"]["mean"], dtype=np.float32)  # [F]
        self.std  = np.array(m["norm"]["std"], dtype=np.float32)   # [F]
        self.F = len(self.mean) # num_features

    # normalize
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-12)

def create_app(onnx_path: str, manifest_path: str, providers=None) -> FastAPI:
    """Return a FastAPI app that serves predictions."""
    sess = ort.InferenceSession(
        onnx_path,
        providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    norm = Normalizer(manifest_path)
    app = FastAPI()

    @app.post("/predict")
    def predict(payload: dict):
        # Expect payload["x"] as list of shape [B, 600, F]
        x = np.array(payload["x"], dtype=np.float32)
        assert x.ndim == 3 and x.shape[2] == norm.F, f"Bad shape {x.shape}"
        x = norm(x)
        y_log = sess.run(None, {"x": x})[0].reshape(-1)  # [B]
        y = np.exp(y_log) - 1e-8                        # back-transform
        return {"y": y.tolist()}

    return app

def serve_main(config_path: str) -> FastAPI:
    """Return a FastAPI app given a config file."""
    cfg = yaml.safe_load(open(config_path))
    return create_app(cfg["onnx_path"], cfg["manifest_path"])
