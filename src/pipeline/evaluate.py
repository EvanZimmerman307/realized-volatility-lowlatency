# src/pipeline/evaluate.py
import json, math, numpy as np, torch
from pathlib import Path
from train import mspe_from_log  # reuse
from data.sharded_dataset import make_loader
from models.tiny_transformer import TinyRVTransformer
import yaml

EPS = 1e-8

@torch.inference_mode()
def evaluate_main(config_path):
    cfg = yaml.safe_load(open(config_path))
    shards_dir = Path(cfg["eval_shards_dir"])
    manifest = json.loads((shards_dir / "manifest.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = make_loader(
        shards_dir=str(shards_dir),
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        prefetch=2,
        shuffle_shards=False
    )

    model = TinyRVTransformer(in_dim=cfg["in_dim"],
                              d_model=cfg.get("d_model", 64),
                              nhead=cfg.get("nhead", 2),
                              nlayers=cfg.get("nlayers", 2),
                              dim_ff=cfg.get("dim_ff", 128),
                              dropout=0.1)
    state = torch.load(cfg["ckpt_path"], map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.to(device).eval()

    n = 0
    se_sum = 0.0           # for RMSE (real scale)
    mspe_sum = 0.0         # for RMSPE (real scale inside, derived from logs)

    # update errors by batches instead of storing all predictions in memory and calculating error once
    for xbatch, ybatch_log in loader:
        xbatch = xbatch.to(device, non_blocking=True)
        ybatch_log = ybatch_log.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=device.type=='cuda'):
            pred_batch_log = model(xbatch)
            if pred_batch_log.ndim > 1:
                pred_batch_log = pred_batch_log.squeeze(-1)

        # back-transform
        y = torch.exp(ybatch_log) - EPS
        pred = torch.exp(pred_batch_log) - EPS

        # computing error on gpu
        se_sum += ((pred - y) ** 2).sum().item()
        mspe_sum += (((pred - y) / (y + EPS)) ** 2).sum().item()
        n += y.shape[0]

    out = {
        "rmse": float(np.sqrt(se_sum / max(n, 1))),
        "rmspe": float(np.sqrt(mspe_sum / max(n, 1))),
        "count": int(n)
    }

    Path(cfg["out_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["out_json"]).write_text(json.dumps(out, indent=2))
    return out

