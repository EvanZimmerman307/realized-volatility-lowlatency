# src/pipeline/export_onnx.py
import torch, yaml
from pathlib import Path
from models.tiny_transformer import TinyRVTransformer
from pipeline.evaluate import _strip_prefixes

def export_onnx(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(cfg["ckpt_path"])
    onnx_out  = Path(cfg["onnx_out"])
    in_dim    = cfg["in_dim"] # num features
    seq_len   = cfg.get("seq_len", 600)  # default 600 seconds
    opset     = cfg.get("opset", 17)

    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model", state)
    sd = _strip_prefixes(sd)
    
    model = TinyRVTransformer(
        in_dim=cfg["in_dim"],
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        nlayers=cfg.get("nlayers", 3),
        dim_ff=cfg.get("dim_ff", 256),
        dropout=0.1
    )
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Example input (batch=1)
    dummy = torch.randn(1, seq_len, in_dim, dtype=torch.float32)

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_out),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["y_log"],
        dynamic_axes={"x": {0: "batch"}, "y_log": {0: "batch"}},
    )
    print(f"Exported ONNX model to {onnx_out}")
