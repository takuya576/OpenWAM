"""Streaming statistics accumulator shared by benchmark statistics tools."""

import operator

import numpy as np

RESERVOIR_CAP = 1_000_000


class Accumulator:
    """Online mean/std/min/max accumulator + reservoir for q01/q99."""

    def __init__(self, dim: int = 20, reservoir_cap: int = RESERVOIR_CAP, seed: int = 0):
        try:
            dim = operator.index(dim)
            reservoir_cap = operator.index(reservoir_cap)
        except TypeError as error:
            raise ValueError("dim and reservoir_cap must be positive integers") from error
        if dim < 1 or reservoir_cap < 1:
            raise ValueError("dim and reservoir_cap must be positive integers")
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min_val = np.full(dim, np.inf, dtype=np.float64)
        self.max_val = np.full(dim, -np.inf, dtype=np.float64)

        self.cap = int(reservoir_cap)
        self.rng = np.random.RandomState(seed)
        self._res = np.empty((self.cap, dim), dtype=np.float32)
        self._res_n = 0
        self._res_seen = 0

    def _validate_batch(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.dim:
            raise ValueError(f"expected a (N, {self.dim}) statistics batch, got {batch.shape}")
        if not np.isfinite(batch).all() or np.any(np.abs(batch) > np.finfo(np.float32).max):
            raise ValueError("statistics batches must contain finite float32-representable values")
        return batch

    def update(self, batch: np.ndarray):
        """Update with (N, dim) array using Welford's online algorithm."""
        batch = self._validate_batch(batch)
        for i in range(len(batch)):
            x = batch[i].astype(np.float64)
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2
            self.min_val = np.minimum(self.min_val, x)
            self.max_val = np.maximum(self.max_val, x)
        self._reservoir_add(np.asarray(batch, dtype=np.float32))

    def update_batch(self, batch: np.ndarray):
        """Batch update (more efficient for large arrays)."""
        batch = self._validate_batch(batch)
        n = len(batch)
        if n == 0:
            return
        self._reservoir_add(np.asarray(batch, dtype=np.float32))
        batch = batch.astype(np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_min = batch.min(axis=0)
        batch_max = batch.max(axis=0)

        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_var * n
            self.min_val = batch_min
            self.max_val = batch_max
            self.count = n
        else:
            total = self.count + n
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * n / total
            self.m2 = self.m2 + batch_var * n + delta**2 * self.count * n / total
            self.mean = new_mean
            self.count = total
            self.min_val = np.minimum(self.min_val, batch_min)
            self.max_val = np.maximum(self.max_val, batch_max)

    def _reservoir_add(self, batch: np.ndarray):
        """Feed (N, dim) rows into the reservoir (vectorized Algorithm R)."""
        n = len(batch)
        if n == 0:
            return

        if self._res_n < self.cap:
            take = min(self.cap - self._res_n, n)
            self._res[self._res_n : self._res_n + take] = batch[:take]
            self._res_n += take
            self._res_seen += take
            batch = batch[take:]
            if len(batch) == 0:
                return

        m = len(batch)
        t = self._res_seen + np.arange(m)
        p = self.rng.randint(0, t + 1)
        keep = p < self.cap
        self._res[p[keep]] = batch[keep]
        self._res_seen += m

    def finalize(self):
        if self.count == 0:
            raise ValueError("cannot finalize statistics without any rows")
        std = np.sqrt(self.m2 / max(self.count, 1))
        std = np.where(std < 1e-8, 1.0, std)
        if self._res_n > 0:
            res = self._res[: self._res_n]
            q01 = np.quantile(res, 0.01, axis=0)
            q99 = np.quantile(res, 0.99, axis=0)
        else:
            q01 = self.min_val
            q99 = self.max_val
        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "min": self.min_val.astype(np.float32).tolist(),
            "max": self.max_val.astype(np.float32).tolist(),
            "q01": q01.astype(np.float32).tolist(),
            "q99": q99.astype(np.float32).tolist(),
        }
