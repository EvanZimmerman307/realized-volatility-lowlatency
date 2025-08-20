# src/pipeline/serve.py
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from fastapi import Request
from fastapi.responses import Response
import onnxruntime as ort
import yaml
import io

class Normalizer:
    def __init__(self, manifest_path: str):
        m = json.loads(Path(manifest_path).read_text())
        self.mean = np.array(m["norm"]["mean"], dtype=np.float32)
        self.std  = np.array(m["norm"]["std"],  dtype=np.float32)
        self.F = len(self.mean)
        self.eps = np.float32(1e-12)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)  # ensure float32 input
        out = (x - self.mean[None, None, :]) / (self.std[None, None, :] + self.eps)
        return out.astype(np.float32, copy=False)       # keep float32

def create_app(onnx_path: str, manifest_path: str, providers=None) -> FastAPI:
    """Return a FastAPI app that serves predictions."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        onnx_path, sess_options=so,
        providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    norm = Normalizer(manifest_path)
    app = FastAPI(default_response_class=ORJSONResponse)
    app.add_middleware(GZipMiddleware, minimum_size=1024)


    @app.post("/predict")
    def predict(payload: dict):
        # Expect payload["x"] as list of shape [B, 600, F]
        x = np.array(payload["x"], dtype=np.float32)
        assert x.ndim == 3 and x.shape[2] == norm.F, f"Bad shape {x.shape}"
        x = norm(x)
        y_log = sess.run(None, {"x": x})[0].reshape(-1)  # [B]

        # Make outputs JSON-safe and numerically sane for benchmarking
        # 1) replace NaN/±Inf in y_log
        y_log = np.nan_to_num(y_log, nan=0.0, posinf=50.0, neginf=-50.0)
        # 2) safe exp: clip to avoid overflow (exp(20) ≈ 4.85e8)
        y = np.exp(np.clip(y_log, -20.0, 20.0)) - 1e-8
        return {"y": y.tolist()}

    @app.post("/predict_np")
    async def predict_np(request: Request):
        # 1) raw bytes in
        buf = await request.body()                      # <- no dict/JSON parsing
        x = np.load(io.BytesIO(buf), allow_pickle=False)  # [B,600,F] float32

        # 2) normalize + run
        x = norm(x)
        y_log = sess.run(None, {"x": x})[0].astype(np.float32).reshape(-1)

        # 3) bytes out (return y_log to avoid exp overflow)
        out = io.BytesIO()
        np.save(out, y_log, allow_pickle=False)
        return Response(content=out.getvalue(), media_type="application/x-npy")

    return app

def serve_main(config_path: str) -> FastAPI:
    """Return a FastAPI app given a config file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return create_app(cfg["onnx_path"], cfg["manifest_path"])
