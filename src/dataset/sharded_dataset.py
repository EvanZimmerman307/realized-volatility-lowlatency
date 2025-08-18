# src/data/shard_dataset.py
import numpy as np, json
from pathlib import Path
import torch
from torch.utils.data import IterableDataset, get_worker_info

class ShardDataset(IterableDataset):
    def __init__(self, shards_dir: str, shuffle_shards: bool = True, normalize: bool = True, seed: int = 23):
        self.dir = Path(shards_dir)
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.shards = sorted(self.dir.glob("shard_*.npz"))
        self.shuffle_shards = shuffle_shards
        self.normalize = normalize
        self.seed = seed
        if normalize:
            self.mean = np.array(manifest["norm"]["mean"], dtype=np.float32)  # size [F]
            self.std  = np.array(manifest["norm"]["std"],  dtype=np.float32)  # size [F]
            assert self.mean is not None and self.std is not None, "Normalization stats missing in manifest."
        else:
            self.mean, self.std = None, None

        # Precompute a single global order of shards shared by all workers
        self.global_order = np.arange(len(self.shards)) # [0,1,...,len(shards) - 1]
        if self.shuffle_shards:
            np.random.default_rng(self.seed).shuffle(self.global_order)

    def __iter__(self):
        worker_info = get_worker_info()
        
        # TODO: vary shuffles across epochs
        
        order = self.global_order
        # split shards among workers
        # Gives each worker a contiguous slice of shard indices, so workers process disjoint subsets of shards (no duplicates)
        if worker_info is not None:
            n = len(order)
            shards_per_worker = int(np.ceil(n / worker_info.num_workers)) # 
            start = worker_info.id * shards_per_worker
            stop  = min(start + shards_per_worker, n)
            order = order[start:stop] # chunk of shards this worker handles

            # per-worker RNG for *within-shard* permutations only
            rng = np.random.default_rng(worker_info.seed ^ 0xBB491B)
        else:
            rng = np.random.default_rng(0xBB491B)

        for i in order:
            with np.load(self.shards[i], allow_pickle=False, mmap_mode="r") as data:
                X, y = data["X"], data["y"]     # X: [Ns,600,F]
                idx = rng.permutation(len(y))   # shuffle samples inside shard
                for j in idx:
                    x = X[j]                    # [600,F]
                    if self.normalize:
                        x = (x - self.mean) / (self.std + 1e-12)
                    yield torch.from_numpy(x), torch.tensor(y[j], dtype=torch.float32)

def make_loader(shards_dir, batch_size=64, num_workers=4, pin_memory=True, prefetch=4, shuffle_shards=True, normalize=True):
    ds = ShardDataset(shards_dir, shuffle_shards=shuffle_shards, normalize=normalize)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    if num_workers > 0:
        kwargs.update(pin_memory=pin_memory, prefetch_factor=prefetch, persistent_workers=True)
    return torch.utils.data.DataLoader(ds, **kwargs)
