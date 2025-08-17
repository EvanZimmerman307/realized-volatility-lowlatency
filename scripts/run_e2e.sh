#!/usr/bin/env bash
set -euo pipefail

# Adjust these paths/env as needed
export PYTHONPATH=./

echo "[1/6] Windowizing Optiver data..."
python src/make_windows.py --input-dir data --out data/windows.parquet --max-samples 5000

echo "[2/6] Building CPU features..."
python src/features_cpu.py --in data/windows.parquet --out data/features_cpu.parquet

echo "[3/6] Training models (GBM + CNN)..."
python src/train.py --features data/features_cpu.parquet --out-dir data/models

echo "[4/6] Exporting to ONNX..."
python src/export_onnx.py --ckpt data/models/cnn.pt --out data/models/model.onnx

echo "[5/6] Building TensorRT engine (if available)..."
python src/build_trt.py --onnx data/models/model.onnx --engine data/models/model_fp16.engine || echo "TensorRT build skipped."

echo "[6/6] Running inference benchmarks..."
python src/infer_bench.py --onnx data/models/model.onnx --engine data/models/model_fp16.engine --features data/features_cpu.parquet

echo "Done."
