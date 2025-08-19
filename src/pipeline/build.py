# src/pipeline/build.py
import multiprocessing as mp
import os, json, numpy as np, polars as pl, logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List
from features.FeatureGenerator import FeatureGenerator
from features.DataLoader import DataLoader
import yaml

EPS = 1e-8

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _fg_one(args):
    """Generate features for one (stock_id, time_id) pair"""
    (book_path, trade_path, stock_id, time_id, feature_cols) = args
    book_path = "~/realized-volatility-lowlatency/" + book_path
    trade_path = "~/realized-volatility-lowlatency/" + trade_path
    book = DataLoader.load_data_for_time_id(book_path, time_id, logger)
    trade = DataLoader.load_data_for_time_id(trade_path, time_id, logger)
    fg = FeatureGenerator(book, trade).training_sample
    X = fg.select(feature_cols).cast(pl.Float32).to_numpy()
    return X  # y comes from labels table


def build_shards(config_path: str):
    """Make shards of samples"""
    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    feature_cols: List[str] = cfg["feature_cols"]
    shard_size = int(cfg.get("shard_size", 2048))
    workers = int(cfg.get("workers", max(1, (os.cpu_count() or 2) // 2)))
    train_mean = train_std = None  # cache train normalization stats to reuse for val/test

    order = ["train", "validation", "eval"]  # always present in build.yaml
    for split_name in order:
        logger.info(f"Starting split: {split_name}")
        paths = cfg["splits"][split_name]
        index_path = Path(paths["index"])
        index = pl.read_parquet(index_path)
        logger.info(f"Loaded index with {len(index)} rows from {index_path}")

        out_dir = Path(paths["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # normalizer (compute on-the-fly, streaming mean/var)
        running_count = 0
        running_sum = None
        running_sumsq = None

        buf_X, buf_y, num_shards = [], [], 0

        def _accum_stats(X):  # X: [600, F], numpy
            nonlocal running_sum, running_sumsq, running_count
            s = X.sum(axis=0).astype(np.float64)
            s2 = (X * X).sum(axis=0).astype(np.float64)
            if running_sum is None:
                running_sum = np.zeros_like(s)
                running_sumsq = np.zeros_like(s)
            running_sum += s
            running_sumsq += s2
            running_count += X.shape[0]

        def _flush():
            nonlocal buf_X, buf_y, num_shards
            _flush_shard(out_dir, num_shards, buf_X, buf_y)
            logger.info(f"Flushed shard {num_shards:05d} with {len(buf_X)} samples")
            buf_X, buf_y = [], []
            num_shards += 1

        fmap = {}
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for row in index.iter_rows(named=True):
                args = (
                    row["book_path"],
                    row["trade_path"],
                    row["stock_id"],
                    row["time_id"],
                    feature_cols,
                )
                fut = ex.submit(_fg_one, args)
                fmap[fut] = float(row["target"])

                # backpressure
                if len(fmap) >= 4 * workers:
                    for done in as_completed(list(fmap)):
                        X = done.result()
                        y = fmap.pop(done)
                        buf_X.append(X)
                        buf_y.append(np.log(y + EPS).astype(np.float32))
                        if split_name == "train":
                            _accum_stats(X)
                        if len(buf_X) >= shard_size:
                            _flush()
                            break

            # drain remaining
            for done in as_completed(list(fmap)):
                X = done.result()
                y = fmap.pop(done)
                buf_X.append(X)
                buf_y.append(np.log(y + EPS).astype(np.float32))
                if split_name == "train":
                    _accum_stats(X)
                if len(buf_X) >= shard_size:
                    _flush()

        if buf_X:
            _flush()

        if split_name == "train":
            mean64 = running_sum / max(1, running_count)
            var64 = running_sumsq / max(1, running_count) - mean64**2
            train_mean = mean64.astype(np.float32)
            train_std = np.sqrt(np.clip(var64, 1e-12, None)).astype(np.float32)
            logger.info(
                f"Computed train stats over {running_count} rows: mean/std shapes {train_mean.shape}/{train_std.shape}"
            )

        if split_name != "train":
            assert train_mean is not None and train_std is not None

        manifest = {
            "feature_cols": feature_cols,
            "num_shards": num_shards,
            "shard_size": shard_size,
            "norm": {
                "mean": (train_mean.tolist() if train_mean is not None else None),
                "std": (train_std.tolist() if train_std is not None else None),
                "count_seconds": int(running_count if split_name == "train" else 0),
                "source": "train",
            },
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info(f"Wrote manifest to {out_dir/'manifest.json'} for split {split_name}")

    logger.info("All splits complete.")


def _flush_shard(out_dir: Path, idx: int, X_list, y_list):
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    np.savez_compressed(out_dir / f"shard_{idx:05d}.npz", X=X, y=y)
