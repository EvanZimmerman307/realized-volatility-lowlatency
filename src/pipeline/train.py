# src/pipeline/train.py
import time, logging
import torch, torch.nn as nn
from dataset.sharded_dataset import make_loader
from models.tiny_transformer import TinyRVTransformer
import yaml, math
from pathlib import Path
try:
    from ops.rmspe_cuda import mspe_from_log_cuda
    _has_cuda_ext = True
except Exception:
    _has_cuda_ext = False

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def _assert_finite(t: torch.Tensor, name: str):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        raise ValueError(f"Non-finite values in {name}: {bad}/{t.numel()}")

def sanitize_inputs(X: torch.Tensor, already_normalized: bool) -> torch.Tensor:
    # Replace NaN/Inf first, in fp32
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Clamp to keep numerics stable (esp. with fp16)
    if already_normalized:
        X = X.clamp_(-10, 10)      # features ~N(0,1) after norm
    else:
        X = X.clamp_(-1e3, 1e3)    # conservative if not normalized
    return X

def mspe_from_log(pred_log, y_log, eps=1e-8):
    y_true = torch.exp(y_log) - eps
    y_pred = torch.exp(pred_log) - eps
    return torch.mean(((y_pred - y_true) / (y_true + eps))**2)

def _eval_rmspe(model, loader, device, cuda_loss=False):
    """For validating the model after each epoch"""
    model.eval()
    mspe_sum, n = 0.0, 0
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
        for X, y_log in loader:
            X = X.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)
            X = sanitize_inputs(X, already_normalized=True)
            _assert_finite(X, "X")
            _assert_finite(y_log, "y_log")
            pred_log = model(X)
            if pred_log.ndim > 1:  # handle [B,1]
                pred_log = pred_log.squeeze(-1)
            pred32 = pred_log.float()
            y32 = y_log.float()
            mspe = mspe_from_log_cuda(pred32, y32) if cuda_loss else mspe_from_log(pred32, y32)
            mspe_sum += mspe.item() * y_log.shape[0]
            n += y_log.shape[0]
    model.train()
    return float((mspe_sum / max(n, 1)) ** 0.5)  # RMSPE

def _count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_main(config_path):
    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {name}")
    else:
        logger.info("Using CPU")

    bs = cfg["batch_size"]
    nw = cfg.get("num_workers", 4)
    prefetch = cfg.get("prefetch", 4)
    logger.info(f"Dataloader: batch_size={bs}, num_workers={nw}, prefetch={prefetch}")

    train_loader = make_loader(
        cfg["train_shards_dir"],
        batch_size=bs,
        num_workers=nw,
        pin_memory=True,
        prefetch=prefetch,
    )
    try:
        logger.info(f"Train steps/epoch (batches): {len(train_loader)}")
    except Exception:
        pass

    val_loader = None
    if "eval_shards_dir" in cfg and cfg["eval_shards_dir"]:
        val_loader = make_loader(
            cfg["eval_shards_dir"],
            batch_size=bs,
            num_workers=nw,
            pin_memory=True,
            prefetch=2,
            shuffle_shards=False,  # val: deterministic
        )
        try:
            logger.info(f"Validation steps/epoch (batches): {len(val_loader)}")
        except Exception:
            pass

    model = TinyRVTransformer(
        in_dim=cfg["in_dim"],
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 2),
        nlayers=cfg.get("nlayers", 2),
        dim_ff=cfg.get("dim_ff", 128),
        dropout=0.1
    ).to(device)
    logger.info(
        f"Model: TinyRVTransformer(d_model={cfg.get('d_model', 64)}, "
        f"nhead={cfg.get('nhead', 2)}, nlayers={cfg.get('nlayers', 2)}, "
        f"dim_ff={cfg.get('dim_ff', 128)}) | params={_count_trainable_params(model):,}"
    )

    # Optional compile
    if device.type == "cuda":
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled")
        except Exception as e:
            logger.warning(f"torch.compile unavailable/failed, continuing without it: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
    loss_kind = cfg.get("loss", "log_mse")  # "log_mse" or "rmspe"
    cuda_loss = bool(cfg.get("cuda_rmspe", False)) and _has_cuda_ext and (device.type == "cuda")
    if loss_kind == "rmspe" and bool(cfg.get("cuda_rmspe", False)) and not _has_cuda_ext:
        logger.warning("cuda_rmspe requested but CUDA extension not found; falling back to Python loss.")
    logger.info(f"Loss: {loss_kind}{' (CUDA ext)' if cuda_loss else ''}; LR={cfg['lr']}")

    mse = nn.MSELoss()
    best_rmspe = float("inf")
    log_interval = int(cfg.get("log_interval", 200))

    epochs = cfg["epochs"]
    logger.info(f"Starting training for {epochs} epochs; checkpoint -> {cfg['out_ckpt']}")

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        running_mspe_sum, n = 0.0, 0
        steps = 0

        for X, y_log in train_loader:
            X = X.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)
            X = sanitize_inputs(X, already_normalized=True)

            _assert_finite(X, "X")
            _assert_finite(y_log, "y_log")

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                pred_log = model(X)
                if pred_log.ndim > 1:
                    pred_log = pred_log.squeeze(-1)
                pred32 = pred_log.float()
                y32 = y_log.float()
                if loss_kind == "log_mse":
                    loss = mse(pred32, y32)
                else:  # "rmspe" (actually MSPE then sqrt at the end)
                    loss = mspe_from_log_cuda(pred32, y32) if cuda_loss else mspe_from_log(pred32, y32)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_mspe_sum += loss.item() * y_log.shape[0]
            steps += 1
            n += y_log.shape[0]

            if steps % log_interval == 0:
                # Note: we log sqrt(mean(loss)) for parity with final print
                partial = float((running_mspe_sum / max(n, 1)) ** 0.5)
                logger.info(f"epoch {epoch+1}/{epochs} step {steps}: train_{loss_kind}≈{partial:.5f}")

        avg = float((running_mspe_sum / max(n, 1)) ** 0.5)
        msg = f"epoch {epoch+1}/{epochs} done in {time.time()-epoch_start:.1f}s: train_{loss_kind}={avg:.5f}"

        # ---- validation & best checkpoint ----
        Path(cfg["out_ckpt"]).parent.mkdir(parents=True, exist_ok=True)
        if val_loader is not None:
            rmspe = _eval_rmspe(model, val_loader, device, cuda_loss)
            msg += f", val_rmspe={rmspe:.5f}"
            if rmspe < best_rmspe:
                best_rmspe = rmspe
                torch.save({"model": model.state_dict(), "best_rmspe": best_rmspe}, cfg["out_ckpt"])
                logger.info(f"{msg}  |  new best; saved checkpoint -> {cfg['out_ckpt']}")
            else:
                logger.info(msg)
        else:
            torch.save(model.state_dict(), cfg["out_ckpt"])
            logger.info(f"{msg}  |  saved (no val set) -> {cfg['out_ckpt']}")

    if val_loader is not None and best_rmspe < float('inf'):
        logger.info(f"Training complete. Best val_rmspe={best_rmspe:.5f} @ {cfg['out_ckpt']}")
    else:
        logger.info("Training complete.")
