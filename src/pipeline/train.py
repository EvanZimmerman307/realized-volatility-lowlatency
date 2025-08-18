# src/pipeline/train.py
import torch, torch.nn as nn
from data.sharded_dataset import make_loader
from models.tiny_transformer import TinyRVTransformer
import yaml, math

def mspe_from_log(pred_log, y_log, eps=1e-8):
    y_true = torch.exp(y_log) - eps
    y_pred = torch.exp(pred_log) - eps
    return torch.mean(((y_pred - y_true) / (y_true + eps))**2)  # no sqrt (monotone), still same argmin, simpler

def _eval_rmspe(model, loader, device, cuda_loss = False):
    """For validating the model after each epoch"""
    model.eval()
    mspe_sum, n = 0.0, 0
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
        for X, y_log in loader:
            X = X.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)
            pred_log = model(X)
            if pred_log.ndim > 1:  # handle [B,1]
                pred_log = pred_log.squeeze(-1)
            pred32 = pred_log.float()
            y32 = y_log.float()
            mspe = mspe_from_log_cuda(pred32, y32) if cuda_loss else mspe_from_log(pred32, y32)
            mspe_sum += mspe.item() * y_log.shape[0] # sum error across samples
            n += y_log.shape[0]
    model.train()
    return float((mspe_sum / max(n, 1)) ** 0.5)  # RMSPE

def train_main(config_path):
    cfg = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(cfg["train_shards_dir"],
                         batch_size=cfg["batch_size"],
                         num_workers=cfg.get("num_workers", 4),
                         pin_memory=True, 
                         prefetch=cfg.get("prefetch", 4))
    
    val_loader = None
    if "eval_shards_dir" in cfg and cfg["eval_shards_dir"]:
        val_loader = make_loader(
            cfg["eval_shards_dir"],
            batch_size=cfg["batch_size"],
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
            prefetch=2,
            shuffle_shards=False,  # val: deterministic
        )

    model = TinyRVTransformer(in_dim=cfg["in_dim"],
                              d_model=cfg.get("d_model", 64),
                              nhead=cfg.get("nhead", 2),
                              nlayers=cfg.get("nlayers", 2),
                              dim_ff=cfg.get("dim_ff", 128),
                              dropout=0.1).to(device)

    if device.type == "cuda":
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
    loss_kind = cfg.get("loss", "log_mse")  # "log_mse" or "rmspe"
    
    try:
        from ops.rmspe_cuda import mspe_from_log_cuda
        _has_cuda_ext = True
    except Exception:
        _has_cuda_ext = False
    cuda_loss = bool(cfg.get("cuda_rmspe", False)) and _has_cuda_ext and (device.type == "cuda")
    
    mse = nn.MSELoss()

    best_rmspe = float("inf")
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        steps = 0
        for X, y_log in train_loader:
            X = X.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                pred_log = model(X)
                if pred_log.ndim > 1:  # handle [B,1]
                    pred_log = pred_log.squeeze(-1)
                pred32 = pred_log.float()
                y32 = y_log.float()
                if loss_kind == "log_mse":
                    loss = mse(pred32, y32)
                else:  # "rmspe"
                    loss = mspe_from_log_cuda(pred32, y32) if cuda_loss else mspe_from_log(pred32, y32)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            steps += 1
        avg = running / max(1, steps)
        msg = f"epoch {epoch}: train_{loss_kind}={avg:.5f}"
    
        # ---- validation & best checkpoint ----
        if val_loader is not None:
            rmspe = _eval_rmspe(model, val_loader, device, cuda_loss)
            msg += f", val_rmspe={rmspe:.5f}"
            if rmspe < best_rmspe:
                best_rmspe = rmspe
                torch.save({"model": model.state_dict(), "best_rmspe": best_rmspe}, cfg["out_ckpt"])
        else:
            # no val set → still save last model
            torch.save(model.state_dict(), cfg["out_ckpt"])

        print(msg)