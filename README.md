# Realized Volatility Low-Latency

This project implements an **end-to-end pipeline** for training, exporting, serving, and benchmarking models that predict **realized volatility** in financial time series. The focus is on **low-latency inference**, efficient data handling, and demonstrating GPU acceleration.

## Overview

The pipeline covers the full lifecycle:

- **Data ingestion & sharding**  
  Converts raw financial data into sharded datasets optimized for streaming training.

- **Model training**  
  Uses a lightweight Transformer (`TinyRVTransformer`) to predict log-volatility. Training supports standard MSE loss as well as custom **RMSPE (Root Mean Squared Percentage Error)**. A custom **CUDA extension** is provided for fast RMSPE computation.

- **Model export**  
  Exports trained models to **ONNX** format for deployment.

- **Serving**  
  Provides a FastAPI-based inference server with an `/predict` endpoint, backed by **ONNX Runtime with GPU acceleration**.

- **Benchmarking**  
  Includes benchmarking utilities for both **serial and concurrent requests**, enabling evaluation of inference latency and throughput under load.

- **Evaluation**  
  Tools for measuring validation performance (RMSPE) on held-out data.

## Key Features

- **Low-latency inference** using ONNX Runtime (GPU-backed).  
- **CUDA custom op** for RMSPE loss, demonstrating GPU kernel programming for finance-specific metrics.  
- **Flexible pipeline CLI** built with Typer (`rvpipe_cli.py`) to manage all stages: `index`, `build`, `train`, `export`, `serve`, `evaluate`, `bench`.  
- **Extensible design**: models, losses, and serving logic can be easily swapped or extended.

## Why This Matters

Financial applications such as trading and risk management require both **accuracy and speed**. This project demonstrates how to build a complete system that:  

- Trains a volatility forecasting model,  
- Optimizes it for GPU execution,  
- Deploys it in a serving stack, and  
- Benchmarks inference latency under realistic conditions.
