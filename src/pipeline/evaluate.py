# src/pipeline/evaluate.py
import json, math, numpy as np, torch, logging, time
from pathlib import Path
from pipeline.train import mspe_from_log, _assert_finite, sanitize_inputs
from dataset.sharded_dataset import make_loader
from models.tiny_transformer import TinyRVTransformer
import yaml

EPS = 1e-8

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def _strip_prefixes(sd, prefixes=("_orig_mod.", "module.")):
    for p in prefixes:
        if all(k.startswith(p) for k in sd.keys()):
            return {k[len(p):]: v for k, v in sd.items()}
    return sd

@torch.inference_mode()
def evaluate_main(config_path):
    logger.info(f"Loading eval config from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    shards_dir = Path(cfg["eval_shards_dir"])
    manifest_path = shards_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    logger.info(f"Eval shards: {shards_dir} | manifest: {manifest_path}")
    logger.info(f"Feature cols: {len(manifest.get('feature_cols', []))} | num_shards: {manifest.get('num_shards')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")

    bs = cfg["batch_size"]
    nw = cfg["num_workers"]
    logger.info(f"Dataloader: batch_size={bs}, num_workers={nw}, prefetch=2, shuffle_shards=False")

    loader = make_loader(
        shards_dir=str(shards_dir),
        batch_size=bs,
        num_workers=nw,
        prefetch=2,
        shuffle_shards=False
    )
    try:
        logger.info(f"Eval steps (batches): {len(loader)}")
    except Exception:
        pass
    
    ckpt_path = cfg["ckpt_path"]
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model", state)
    sd = _strip_prefixes(sd)
    # TODO: save clean checkpoints without prefixes
    
    model = TinyRVTransformer(
        in_dim=cfg["in_dim"],
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        nlayers=cfg.get("nlayers", 3),
        dim_ff=cfg.get("dim_ff", 256),
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(sd, strict=False)
    model.eval()
    logger.info(f"Loaded model checkpoint: {ckpt_path}")

    log_interval = int(cfg.get("log_interval", 200))
    use_amp = (device.type == 'cuda')
    logger.info(f"AMP enabled: {use_amp}")

    n = 0
    se_sum = 0.0   # for RMSE (on real scale)
    mspe_sum = 0.0 # for RMSPE (on real scale inside, derived from logs)

    t0 = time.time()
    for step, (xbatch, ybatch_log) in enumerate(loader, start=1):
        xbatch = xbatch.to(device, non_blocking=True)
        ybatch_log = ybatch_log.to(device, non_blocking=True)

        # sanitize + guards (matches training)
        xbatch = sanitize_inputs(xbatch, already_normalized=True)
        _assert_finite(xbatch, "X")
        _assert_finite(ybatch_log, "y_log")

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            pred_batch_log = model(xbatch)
            if pred_batch_log.ndim > 1:
                pred_batch_log = pred_batch_log.squeeze(-1)
        _assert_finite(pred_batch_log, "pred_log")

        # back-transform to real scale
        y = torch.exp(ybatch_log) - EPS
        pred = torch.exp(pred_batch_log) - EPS

        # accumulate errors
        se_sum += ((pred - y) ** 2).sum().item()
        mspe_sum += (((pred - y) / (y + EPS)) ** 2).sum().item()
        n += y.shape[0]

        if step % log_interval == 0:
            rmse_so_far = math.sqrt(se_sum / max(n, 1))
            rmspe_so_far = math.sqrt(mspe_sum / max(n, 1))
            logger.info(f"step {step}: rmse≈{rmse_so_far:.5f}, rmspe≈{rmspe_so_far:.5f}, count={n}")

    out = {
        "rmse": float(np.sqrt(se_sum / max(n, 1))),
        "rmspe": float(np.sqrt(mspe_sum / max(n, 1))),
        "count": int(n)
    }
    dt = time.time() - t0
    logger.info(f"Evaluation complete in {dt:.1f}s → rmse={out['rmse']:.5f}, rmspe={out['rmspe']:.5f}, count={n}")

    out_path = Path(cfg["out_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Wrote eval metrics to {out_path}")

    return out
