# tests/test_rmspe_cuda.py
import pytest
import torch
import time

"""PYTHONPATH=. pytest -vv"""

# --- reference Python implementation ---
def mspe_from_log(pred_log, y_log, eps=1e-8):
    y_true = torch.exp(y_log) - eps
    y_pred = torch.exp(pred_log) - eps
    return torch.mean(((y_pred - y_true) / (y_true + eps))**2)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for this test")
def test_mspe_cuda_matches_python_forward_backward():
    """Always run: correctness check (forward & backward)."""
    try:
        from src.ops.rmspe_cuda import mspe_from_log_cuda
    except Exception as e:
        pytest.skip(f"CUDA extension not available/importable: {e}")
        print(e)

    torch.manual_seed(0)
    N = 4096
    eps = 1e-8

    pred_log = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
    y_log    = torch.randn(N, device="cuda", dtype=torch.float32)

    # Forward parity
    mspe_py = mspe_from_log(pred_log, y_log, eps=eps)
    mspe_cu = mspe_from_log_cuda(pred_log, y_log, eps)
    print(mspe_cu)
    print(mspe_cu.item())
    assert torch.allclose(mspe_cu, mspe_py, rtol=1e-5, atol=1e-6)

    # Backward parity (grad w.r.t. pred_log)
    pred_log.grad = None
    mspe_cu.backward(retain_graph=True)
    grad_cu = pred_log.grad.detach().clone()

    pred_log.grad = None
    mspe_py.backward()
    grad_py = pred_log.grad.detach().clone()

    assert torch.allclose(grad_cu, grad_py, rtol=1e-4, atol=1e-5)

@pytest.mark.benchmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for this test")
def test_mspe_cuda_benchmark():
    """Optional: performance benchmark (enable with `-m benchmark`)."""
    try:
        from src.ops.rmspe_cuda import mspe_from_log_cuda
    except Exception as e:
        pytest.skip(f"CUDA extension not available/importable: {e}")
        print(e)

    torch.manual_seed(0)
    N = 200000 # int(torch.getenv("RV_BENCH_N", "200000"))  # allow override via env var
    iters = 50 # int(torch.getenv("RV_BENCH_ITERS", "50"))
    eps = 1e-8

    pred_log = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=False)
    y_log    = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=False)

    # Warmup
    torch.cuda.synchronize()
    for _ in range(5):
        _ = mspe_from_log(pred_log, y_log, eps=eps)
        _ = mspe_from_log_cuda(pred_log, y_log, eps)
    torch.cuda.synchronize()

    # Time Python (Torch ops)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = mspe_from_log(pred_log, y_log, eps=eps)
    torch.cuda.synchronize()
    t_py = time.time() - t0

    # Time CUDA op
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = mspe_from_log_cuda(pred_log, y_log, eps)
    torch.cuda.synchronize()
    t_cu = time.time() - t0

    speedup = t_py / t_cu if t_cu > 0 else float("inf")
    print(f"\n[Benchmark] Python={t_py:.4f}s, CUDA={t_cu:.4f}s → CUDA is {speedup:.2f}× faster "
          f"(iters={iters}, N={N})")
