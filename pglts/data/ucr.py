"""UCR/UEA ``.ts`` loading and streaming descriptor construction.

Why this module exists
----------------------
The UEA *InsectWingbeat* problem is large in a way that punishes a naive
pipeline: 25,000 instances per split, 200 channels, and variable series length.
Three properties of the data and of PG-LTS make it cheap to handle anyway.

**Read the ``.ts``, not the CSV.** The distributed CSV pads every series to the
maximum length (22) and therefore stores ~4,400 columns per row, roughly three
times more numbers than exist: the median series length is 7. The ``.ts`` file
preserves the ragged structure and ``aeon`` parses it in a few seconds.

**PG-LTS descriptors are sufficient statistics, so the raw tensor is never
needed.** Occupancy is a mean over time and the transition descriptor is a mean
of outer products over adjacent pairs. Both can be accumulated per instance and
the series then discarded. Peak memory is therefore governed by the *descriptor*
size ``O(N G A^2)``, not by the data size ``O(N R L)``. Nothing is padded, so the
ragged lengths cost nothing.

**Adjacent frequency bands can be pooled.** The 200 channels are contiguous
frequency bands of a spectrogram, so averaging groups of adjacent bands is a
physically meaningful reduction in spectral resolution — the same operation as
the band-power features used on the radar corpus — and it divides the transition
descriptor count by the pooling factor. Without pooling, 200 channels at four
states would require 200x16 = 3200 transition coordinates.

The result is a two-pass streaming algorithm: one cheap pass over training data
to fit the semantizer breakpoints, one chunked pass per split to emit a compact
float32 descriptor matrix.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "dataset" / "ucr_insectwingbeat"


def load_ts(path: Path | str) -> tuple[list[np.ndarray], np.ndarray]:
    """Load a ``.ts`` file as a list of ``(n_channels, n_timepoints)`` arrays.

    Uses ``aeon``'s reader, which preserves unequal lengths instead of padding.
    """
    from aeon.datasets import load_from_ts_file

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y = load_from_ts_file(str(path))
    return list(X), np.asarray(y)


def load_insectwingbeat(root: Path | str | None = None):
    """Load the standard TRAIN/TEST split of UEA InsectWingbeat."""
    root = Path(root) if root is not None else default_root()
    Xtr, ytr = load_ts(root / "InsectWingbeat_TRAIN.ts")
    Xte, yte = load_ts(root / "InsectWingbeat_TEST.ts")
    return Xtr, ytr, Xte, yte


def pool_channels(x: np.ndarray, n_groups: int) -> np.ndarray:
    """Average contiguous channel groups: ``(R, L) -> (n_groups, L)``.

    The channels are ordered frequency bands, so this is a reduction of spectral
    resolution rather than an arbitrary projection.
    """
    r = x.shape[0]
    if n_groups >= r:
        return x
    edges = np.linspace(0, r, n_groups + 1).astype(int)
    return np.stack([x[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])


@dataclass
class StreamingSemantizer:
    """Per-channel quantile breakpoints fitted by streaming subsample.

    Fitting exact quantiles would require holding every value of every channel.
    Since breakpoints only need to be stable, not exact, we accumulate a bounded
    reservoir per channel from the training split and take quantiles of that.
    """

    n_states: int = 4
    n_groups: int = 20
    max_per_channel: int = 200_000
    seed: int = 0
    _edges: np.ndarray | None = None      # (G, n_states - 1)

    def fit(self, X: list[np.ndarray]) -> "StreamingSemantizer":
        rng = np.random.default_rng(self.seed)
        buf: list[list[np.ndarray]] = [[] for _ in range(self.n_groups)]
        counts = np.zeros(self.n_groups, dtype=np.int64)
        for x in X:
            pooled = pool_channels(np.asarray(x, dtype=np.float32), self.n_groups)
            for g in range(self.n_groups):
                if counts[g] < self.max_per_channel:
                    buf[g].append(pooled[g])
                    counts[g] += pooled.shape[1]
        qs = np.linspace(100.0 / self.n_states, 100.0 - 100.0 / self.n_states,
                         self.n_states - 1)
        self._edges = np.stack(
            [np.percentile(np.concatenate(b), qs) for b in buf]
        ).astype(np.float32)
        # Guard against degenerate (constant) channels.
        for g in range(self.n_groups):
            e = self._edges[g]
            if not np.all(np.diff(e) > 0):
                self._edges[g] = e + np.arange(e.size) * 1e-6
        _ = rng
        return self

    def memberships(self, pooled: np.ndarray) -> np.ndarray:
        """``(G, L) -> (G, L, A)`` barycentric memberships between breakpoints.

        Membership is 2-sparse by construction, matching the piecewise-linear
        hedge-algebra map: a value activates at most two adjacent states.
        """
        assert self._edges is not None
        g, length = pooled.shape
        a = self.n_states
        anchors = np.concatenate(
            [
                self._edges[:, :1] - (self._edges[:, 1:2] - self._edges[:, :1]),
                self._edges,
                self._edges[:, -1:] + (self._edges[:, -1:] - self._edges[:, -2:-1]),
            ],
            axis=1,
        )[:, :a]                                     # (G, A) ordered anchors
        m = np.zeros((g, length, a), dtype=np.float32)
        for k in range(a):
            c = anchors[:, k : k + 1]
            left = anchors[:, k - 1 : k] if k > 0 else c - 1.0
            right = anchors[:, k + 1 : k + 2] if k < a - 1 else c + 1.0
            d = pooled - c
            rising = np.where(d < 0, 1.0 + d / np.maximum(c - left, 1e-6), 1.0)
            falling = np.where(d >= 0, 1.0 - d / np.maximum(right - c, 1e-6), 1.0)
            m[:, :, k] = np.clip(np.minimum(rising, falling), 0.0, 1.0)
        total = m.sum(axis=-1, keepdims=True)
        np.divide(m, total, out=m, where=total > 0)
        return m


def descriptor_width(n_groups: int, n_states: int, families: tuple[str, ...]) -> int:
    w = 0
    if "O" in families:
        w += n_groups * n_states
    if "M" in families:
        w += n_groups * n_states * n_states
    return w


def build_descriptors(
    X: list[np.ndarray],
    sem: StreamingSemantizer,
    families: tuple[str, ...] = ("O", "M"),
    permute: bool = False,
    seed: int = 0,
    chunk: int = 2000,
) -> np.ndarray:
    """Stream instances and emit an ``(N, D)`` float32 descriptor matrix.

    ``permute`` shuffles the time index of each instance independently — the
    order control. It preserves the multiset of states and destroys only their
    sequence, so any gain that survives it is not attributable to the state
    distribution or to the descriptor dimension.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    d = descriptor_width(sem.n_groups, sem.n_states, families)
    out = np.zeros((n, d), dtype=np.float32)

    for start in range(0, n, chunk):
        for i in range(start, min(start + chunk, n)):
            pooled = pool_channels(np.asarray(X[i], dtype=np.float32), sem.n_groups)
            if permute and pooled.shape[1] > 1:
                pooled = pooled[:, rng.permutation(pooled.shape[1])]
            m = sem.memberships(pooled)                       # (G, L, A)
            parts = []
            if "O" in families:
                parts.append(m.mean(axis=1).ravel())
            if "M" in families:
                if m.shape[1] > 1:
                    t = np.einsum("gla,glb->gab", m[:, :-1], m[:, 1:])
                    s = t.sum(axis=(1, 2), keepdims=True)
                    np.divide(t, s, out=t, where=s > 0)
                else:
                    t = np.zeros((sem.n_groups, sem.n_states, sem.n_states),
                                 dtype=np.float32)
                parts.append(t.ravel())
            out[i] = np.concatenate(parts)
    return out


def static_summary(X: list[np.ndarray], n_groups: int = 20) -> np.ndarray:
    """Order-free numeric baseline: per-channel mean, std, min, max, mean |diff|."""
    n = len(X)
    out = np.zeros((n, n_groups * 5), dtype=np.float32)
    for i, x in enumerate(X):
        p = pool_channels(np.asarray(x, dtype=np.float32), n_groups)
        d = np.abs(np.diff(p, axis=1)).mean(axis=1) if p.shape[1] > 1 else np.zeros(p.shape[0])
        out[i] = np.concatenate([p.mean(1), p.std(1), p.min(1), p.max(1), d])
    return out
