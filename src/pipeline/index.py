# src/pipeline/index.py
import polars as pl, yaml
from pathlib import Path

def index_main(config_path: str):
      with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
      raw_dir = Path(cfg["raw_dir"])
      out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

      eval_split = float(cfg.get("eval_split", 0.2))
      val_split = float(cfg.get("val_split", 0.1))
      seed = int(cfg.get("seed", 42))

      # train.csv has stock_id, time_id, target
      train = pl.read_csv(raw_dir / "train.csv").select(["stock_id","time_id","target"])

      # Deterministic hash-based split on (stock_id, time_id)
      # Note: pl.hash returns UInt64; divide by 2**64 to get [0,1).
      # Train: _hash in [0, train_threshold)
      # Validation: _hash in [train_threshold, train_threshold + val_split)
      # Eval/Test: _hash in [train_threshold + val_split, 1)
      train_threshold = 1 - eval_split - val_split
      tagged = (train.group_by("stock_id", maintain_order=True)
            .map_groups(lambda g: g.with_columns([
                  (pl.hash(["stock_id", "time_id"], seed=seed).cast(pl.Float64) / (2**64)).alias("_hash")
            ])
            .with_columns([
                  pl.when(pl.col("_hash") < train_threshold)
                  .then("train")
                  .when((pl.col("_hash") >= train_threshold) & 
                        (pl.col("_hash") < train_threshold + val_split))
                  .then("validation")
                  .otherwise("eval")
                  .alias("split"),
                  (pl.lit(str(raw_dir / "book_train.parquet/stock_id=")) + pl.col("stock_id").cast(pl.Utf8)).alias("book_path"),
                  (pl.lit(str(raw_dir / "trade_train.parquet/stock_id=")) + pl.col("stock_id").cast(pl.Utf8)).alias("trade_path")
            ])
            .drop("_hash"))
      )
      
      # Why not sample randomly instead?
      # Hash-based split is reproducible across runs/machines, doesn't depend on RNG state
      # we get about 80/20 for all stocks b/c we have 3820 - 3830 samples for each stock

      tagged.filter(pl.col("split") == "train").drop("split") \
            .write_parquet(out_dir / "train_index.parquet")
      tagged.filter(pl.col("split") == "validation").drop("split") \
            .write_parquet(out_dir / "validation_index.parquet")
      tagged.filter(pl.col("split") == "eval").drop("split") \
            .write_parquet(out_dir / "eval_index.parquet")

      print("wrote:", out_dir / "train_index.parquet", ",", out_dir / "validation_index.parquet", "and", out_dir / "eval_index.parquet")


