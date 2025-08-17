# Optiver Realized Volatility — Ultra-Low-Latency ML

**Goal:** Predict realized volatility on Optiver's book/trade data and deploy a **sub-ms to a few ms** inference service.
You’ll demonstrate: large-data handling, GPU optimization (ONNX Runtime / TensorRT), a custom GPU feature kernel,
and a low-latency FastAPI service with observability.

## System Diagram
Data → Windowing → Features (CPU/Triton) → Compact 1D-CNN → Export (ONNX → TensorRT) → Service (FastAPI) → Metrics

## Repo Layout
```text
data/                     # (gitignored) raw and intermediate files
notebooks/                # profiling/EDA if needed
src/
  load_optiver.py         # fast readers
  make_windows.py         # windowization + target RV
  features_cpu.py         # CPU feature engineering
  features_triton.py      # Triton/CUDA kernel(s) for rolling stats
  model_cnn.py            # compact 1D-CNN for sequence features
  train.py                # training loop with AMP
  export_onnx.py          # torch -> onnx export
  build_trt.py            # onnx -> TensorRT engine (fp16)
  infer_bench.py          # model-only + end-to-end latency benchmarks
  server.py               # FastAPI service + /metrics
  client_bench.py         # request generator & latency histograms
  trace.py                # percentile utilities
deploy/
  Dockerfile
scripts/
  repro.sh                # end-to-end reproducibility
requirements.txt          # Python deps
```

## Quickstart
```bash
# 1) Create and activate virtual env (example)
python -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Reproduce end-to-end (edit paths in scripts/repro.sh first)
bash scripts/repro.sh
```

## Targets
- **Accuracy:** CNN competitive with GBM baseline on validation.
- **Latency:** TensorRT fp16 p50 ≤ ~1–2 ms; p99 ≤ ~3–5 ms (GPU, batch=1; hardware-dependent).
- **GPU Kernel:** ≥3× speedup vs CPU for rolling feature.

## Notes
- TensorRT installation varies by environment; if unavailable, use ONNX Runtime (CUDA EP) and still report metrics.
- Keep models small (≤ ~2M params) for exportability and fast inference.
