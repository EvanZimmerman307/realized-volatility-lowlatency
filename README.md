# Realized Volatility Low-Latency

This project implements an **end-to-end pipeline** for training, exporting, serving, and benchmarking models that predict **realized volatility** in financial time series. The focus is on **low-latency inference**, efficient data handling, and demonstrating GPU acceleration. The premise of this project is based on the [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview) Kaggle challenge

## Overview

The pipeline covers the full lifecycle:

- **Data ingestion & sharding**  
  Converts raw financial data into sharded datasets optimized for streaming training.

- **Model training**  
  Uses a lightweight Transformer (`TinyRVTransformer`) to predict log-volatility. Training supports standard MSE loss as well as custom **RMSPE (Root Mean Squared Percentage Error)**. A custom **CUDA extension** is provided for fast RMSPE computation.

- **Model export**  
  Exports trained models to **ONNX** format for deployment.

- **Serving**  
  Provides a FastAPI-based inference server with `/predict` (JSON) and `/predict_np` (binary NumPy) endpoints, backed by **ONNX Runtime with GPU acceleration**. The binary endpoint offers 11x faster throughput for high-frequency trading scenarios.

- **Benchmarking**  
  Includes benchmarking utilities for both **serial and concurrent requests**, enabling evaluation of inference latency and throughput under load.

- **Evaluation**  
  Tools for measuring validation performance (RMSPE) on held-out data.

## Key Features

- **Low-latency inference** using ONNX Runtime (GPU-backed)
- **Dual API endpoints**: JSON for integration, binary NumPy for maximum throughput
- **CUDA custom op** for RMSPE loss, demonstrating GPU kernel programming for custom metrics
- **Flexible pipeline CLI** built with Typer (`rvpipe_cli.py`) to manage all stages: `index`, `build`, `train`, `export`, `serve`, `evaluate`, `bench`
- **Extensible design**: models, losses, and serving logic can be easily swapped or extended

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- [Optiver Realized Volatility Prediction dataset](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data)

### Installation

```bash
git clone <repository-url>
cd realized-volatility-lowlatency
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### Data Setup

Download the Optiver dataset and place the files in `data/raw/`:
```
data/raw/
├── train.csv
├── book_train.parquet/
└── trade_train.parquet/
```

### Running the Pipeline

Execute the complete pipeline using the CLI:

```bash
# 0. Test the CUDA custom op can build and run (should see ~2x speed up over PyTorch)
PYTHONPATH=. pytest -vv

# 1. Create train/validation/test splits
python src/rvpipe_cli.py index

# 2. Build feature shards for training
python src/rvpipe_cli.py build

# 3. Train the model
python src/rvpipe_cli.py train

# 4. Evaluate on test set
python src/rvpipe_cli.py evaluate

# 5. Export to ONNX
python src/rvpipe_cli.py export

# 6. Start inference server
python src/rvpipe_cli.py serve

# 7. Benchmark inference performance
python src/rvpipe_cli.py bench
```

Each step uses configuration files in `configs/` that can be customized.

### Inference Server Usage

The server provides two endpoints optimized for different use cases:

**JSON Endpoint** (`/predict`):
```python
import requests
import numpy as np

data = {"x": np.random.randn(32, 600, 14).tolist()}
response = requests.post("http://localhost:8000/predict", json=data)
predictions = response.json()["y"]
```

**Binary NumPy Endpoint** (`/predict_np`):
```python
import requests
import numpy as np
import io

# Send binary NumPy array
x = np.random.randn(32, 600, 14).astype(np.float32)
buf = io.BytesIO()
np.save(buf, x, allow_pickle=False)

response = requests.post(
    "http://localhost:8000/predict_np",
    data=buf.getvalue(),
    headers={"Content-Type": "application/x-npy"}
)

# Receive binary response
y_log = np.load(io.BytesIO(response.content))
```

## Results

### Model Performance
- **Validation RMSPE**: 0.249 (training)
- **Test RMSPE**: 0.267 (evaluation on held-out data)
- **Model size**: ~400K parameters (TinyRVTransformer with d_model=128, 3 layers, 4 heads)

### Inference Throughput (batch size = 32)
- **JSON endpoint**: 85.7 QPS
- **Binary endpoint**: 948.3 QPS (11x faster)
- **Latency**: 33.7ms p50 and 34.38ms p99 (binary), 373ms p50 and 471ms p99 (JSON)
  - With a batch size of 32 we get ~1ms/sample for the binary endpoint

The binary NumPy endpoint demonstrates the performance gains possible when optimizing for low-latency financial applications.

## Why This Matters

Applications such as high frequency trading require both **accuracy and speed**. This project demonstrates how to build a complete system that:

- Trains a volatility forecasting model
- Optimizes it for GPU execution  
- Deploys it in a serving stack with multiple API formats
- Benchmarks inference latency under realistic conditions

The dual endpoint approach shows how to balance integration flexibility (JSON) with maximum performance (binary) for different use cases.

## Configuration

All pipeline stages are configured via YAML files in `configs/`:

- `index.yaml`: Data splitting parameters
- `build.yaml`: Feature engineering and sharding  
- `train.yaml`: Model architecture and training hyperparameters
- `export.yaml`: ONNX export settings
- `serve.yaml`: Inference server configuration
- `eval.yaml`: Evaluation parameters  
- `bench.yaml`: Benchmarking settings

Modify these files to customize the pipeline for your data and requirements.