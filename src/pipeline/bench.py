# src/pipeline/bench.py
import time, json, statistics, yaml
import numpy as np, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import io

def _one_call(session, url, x):
    """make one inference request and time the latency"""
    t0 = time.perf_counter()
    r = session.post(url, json={"x": x})
    r.raise_for_status()
    _ = r.json()
    return (time.perf_counter() - t0) * 1000.0  # ms

def _one_call_np(session, url_np, x_np):
    # x_np: np.ndarray shape [B,600,F], dtype float32
    buf = io.BytesIO()
    np.save(buf, x_np, allow_pickle=False)
    buf.seek(0)

    t0 = time.perf_counter()
    r = session.post(
        url_np,
        data=buf.getvalue(),  # <- raw bytes, not json=
        headers={
            "Content-Type": "application/x-npy",
            "Accept": "application/x-npy",
        },
    )
    r.raise_for_status()
    y_log = np.load(io.BytesIO(r.content), allow_pickle=False)
    return (time.perf_counter() - t0) * 1000.0  # ms

def _percentiles(times):
    """Given a distribution of request times calculate statistics for the latency distribution"""
    times = sorted(times)
    n = len(times)
    def pct(p): 
        i = max(0, min(n-1, int(p*n) - 1))
        return times[i]
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "count": n,
        "total_s": sum(times) / 1000.0,
    }

def bench_main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    url = cfg.get("url", "http://127.0.0.1:8000/predict_np")
    if "_np" in url:
        call_endpoint = _one_call_np
    else:
        call_endpoint = _one_call

    iters = int(cfg.get("iters", 200))
    batch_size = int(cfg.get("batch_size", 32))
    nfeat = int(cfg["nfeat"])
    seqlen = int(cfg.get("seqlen", 600))
    warmup = int(cfg.get("warmup", 20))
    concurrency = int(cfg.get("concurrency", 8))
    out_json = cfg.get("out_json", "artifacts/bench_report.json")

    # synthetic
    x = (np.random.randn(batch_size, seqlen, nfeat).astype("float32")).tolist()

    with requests.Session() as session:
        # Warmup (serial)
        for _ in range(warmup):
            call_endpoint(session, url, x)

        # ---- Serial run ----
        serial_times = [call_endpoint(session, url, x) for _ in range(iters)]
        serial_stats = _percentiles(serial_times)
        serial_qps = (iters * batch_size) / max(1e-8, serial_stats["total_s"]) # queries per second

        # ---- Concurrent run ----
        concurrent_times = []
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(call_endpoint, session, url, x) for _ in range(iters)]
            for f in as_completed(futures):
                concurrent_times.append(f.result())
        concurrent_stats = _percentiles(concurrent_times)
        conc_qps = (iters * batch_size) / max(1e-8, concurrent_stats["total_s"]) # queries per second

    report = {
        "config": {
            "url": url,
            "iters": iters,
            "batch_size": batch_size,
            "nfeat": nfeat,
            "seqlen": seqlen,
            "warmup": warmup,
            "concurrency": concurrency,
        },
        "serial": {**serial_stats, "throughput_qps": serial_qps},
        "concurrent": {**concurrent_stats, "throughput_qps": conc_qps},
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report
