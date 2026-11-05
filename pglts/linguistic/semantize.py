"""Semantisation — mapping numeric feature trajectories to linguistic states.

The primary scheme is **hedge-algebra semantisation**, reusing the algebraic core
of pyLTS (``HedgeAlgebra``, ``FuzzinessIntervals``). For a level-``k`` partition,
the fuzziness intervals ``ℑ(w)`` tile ``[0, 1]`` with ``2^k`` adjacent intervals,
each labelled by a word built from the generators ``c⁻``/``c⁺`` (*Small*/*Large*)
and the hedges ``L`` (*Rather*) and ``V`` (*Very*):

===========  ==============================================
level        vocabulary
===========  ==============================================
``k = 1``    ``-``, ``+``                       (2 words)
``k = 2``    ``V-``, ``L-``, ``L+``, ``V+``     (4 words)
``k = 3``    ``VV-`` … ``VV+``                  (8 words)
===========  ==============================================

Unlike an ad-hoc triangular partition, the breakpoints are *derived* from two
interpretable parameters per feature: ``θ`` (the semantic weight of the negative
generator) and ``α`` (the fuzziness measure of the hedge *Rather*). That gives a
principled, tunable granulation with a fixed algebraic meaning per word — which
is precisely what makes the downstream rules readable.

Two membership modes are supported:

``containment``
    Crisp assignment — the word whose interval contains the value. This mirrors
    the I-LTS semantisation of pyLTS.
``soft``
    Graded membership peaking at the word's SQM point ``ν(w)`` and decaying to the
    neighbouring words' SQM points, so values near an interval boundary retain
    their ambiguity instead of being forced into one label.

The alternative semantisers exist for the E3 ablation, and all share one
interface so they are interchangeable in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from lts.core.fuzziness_interval import FuzzinessIntervals
from lts.core.hedge_algebras import HAParams, HedgeAlgebra

#: Human-readable gloss for each hedge-algebra word, used when printing rules.
WORD_GLOSS: dict[str, str] = {
    "-": "Small",
    "+": "Large",
    "V-": "Very Small",
    "L-": "Rather Small",
    "L+": "Rather Large",
    "V+": "Very Large",
    "VV-": "Very Very Small",
    "LV-": "Rather Very Small",
    "LL-": "Rather Rather Small",
    "VL-": "Very Rather Small",
    "VL+": "Very Rather Large",
    "LL+": "Rather Rather Large",
    "LV+": "Rather Very Large",
    "VV+": "Very Very Large",
}


def gloss(word: str) -> str:
    """Natural-language reading of a hedge-algebra word."""
    return WORD_GLOSS.get(word, word)


class BaseSemantizer:
    """Interface shared by every semantisation scheme.

    ``fit`` must only ever see *training-fold* data: breakpoints estimated on the
    full corpus would leak test-subject statistics into the representation.
    """

    words: tuple[str, ...] = ()

    @property
    def n_states(self) -> int:
        return len(self.words)

    def fit(self, X: np.ndarray) -> "BaseSemantizer":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """``(N, L, R)`` trajectories -> ``(N, L, R, n_states)`` memberships."""
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _robust_range(X: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature universe of discourse, trimmed against outliers."""
    flat = X.reshape(-1, X.shape[-1])
    lb = np.percentile(flat, low, axis=0)
    ub = np.percentile(flat, high, axis=0)
    span = ub - lb
    degenerate = span <= 0
    ub = np.where(degenerate, lb + 1.0, ub)
    return lb.astype(float), ub.astype(float)


def _normalise_to_unit(X: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.clip((X - lb) / (ub - lb), 0.0, 1.0)


@dataclass
class HASemantizer(BaseSemantizer):
    """Hedge-algebra semantisation with per-feature ``(θ, α)``.

    Parameters
    ----------
    level
        Partition level; the vocabulary has ``2^level`` words.
    mode
        ``"containment"`` (crisp) or ``"soft"`` (graded).
    theta, alpha
        Either a scalar shared by all features, or one value per feature. Tuning
        these per feature is what ``models.tuning`` optimises.
    """

    level: int = 2
    mode: str = "soft"
    theta: float | np.ndarray = 0.5
    alpha: float | np.ndarray = 0.5
    words: tuple[str, ...] = field(default=(), init=False)
    _lb: np.ndarray | None = field(default=None, init=False, repr=False)
    _ub: np.ndarray | None = field(default=None, init=False, repr=False)
    _edges: np.ndarray | None = field(default=None, init=False, repr=False)
    _centres: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"containment", "soft"}:
            raise ValueError("mode must be 'containment' or 'soft'")
        if self.level < 1:
            raise ValueError("level must be >= 1")

    # -- per-feature algebra ------------------------------------------
    def _algebra(self, theta: float, alpha: float) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Return ``(words, edges, sqm_points)`` for one feature.

        ``edges`` has ``2^level + 1`` entries delimiting the adjacent intervals.
        """
        ha = HedgeAlgebra(HAParams(theta=float(theta), alpha=float(alpha)))
        fi = FuzzinessIntervals(ha)
        parts = fi.partition(self.level)
        words = [w for w, _ in parts]
        edges = np.array([parts[0][1].lo] + [iv.hi for _, iv in parts], dtype=float)
        centres = np.array([ha.sqm(w) for w in words], dtype=float)
        return words, edges, centres

    def _per_feature_params(self, n_features: int) -> tuple[np.ndarray, np.ndarray]:
        theta = np.broadcast_to(np.asarray(self.theta, dtype=float), (n_features,)).copy()
        alpha = np.broadcast_to(np.asarray(self.alpha, dtype=float), (n_features,)).copy()
        # Degenerate parameters collapse the partition; keep strictly inside (0, 1).
        return np.clip(theta, 0.05, 0.95), np.clip(alpha, 0.05, 0.95)

    # -- interface ------------------------------------------------------
    def fit(self, X: np.ndarray) -> "HASemantizer":
        X = np.asarray(X, dtype=float)
        n_features = X.shape[-1]
        self._lb, self._ub = _robust_range(X)
        theta, alpha = self._per_feature_params(n_features)

        edges, centres, words = [], [], None
        for r in range(n_features):
            w, e, c = self._algebra(theta[r], alpha[r])
            words = words or w
            edges.append(e)
            centres.append(c)
        self.words = tuple(words or ())
        self._edges = np.stack(edges, axis=0)      # (R, S+1)
        self._centres = np.stack(centres, axis=0)  # (R, S)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._edges is None or self._lb is None or self._ub is None:
            raise RuntimeError("HASemantizer must be fitted before transform()")
        u = _normalise_to_unit(np.asarray(X, dtype=float), self._lb, self._ub)

        if self.mode == "containment":
            return _containment_membership(u, self._edges)
        return _triangular_membership(u, self._centres)


def _containment_membership(u: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """One-hot membership: the interval that contains each value."""
    n_states = edges.shape[1] - 1
    out = np.zeros(u.shape + (n_states,), dtype=np.float32)
    for r in range(u.shape[-1]):
        idx = np.searchsorted(edges[r, 1:-1], u[..., r], side="right")
        np.put_along_axis(out[..., r, :], idx[..., None], 1.0, axis=-1)
    return out


def _triangular_membership(u: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Graded membership peaking at each word's SQM point.

    Each word's membership decays linearly to its neighbours' centres. The two
    extreme words carry *shoulders*: membership saturates at 1 below the first
    centre and above the last, so values in the tails stay fully assigned instead
    of falling outside every triangle. Memberships are then normalised to sum to
    one at every time step.
    """
    n_states = centres.shape[1]
    out = np.zeros(u.shape + (n_states,), dtype=np.float32)
    for r in range(u.shape[-1]):
        c = centres[r]
        v = u[..., r]
        for k in range(n_states):
            d = v - c[k]
            left = max(c[k] - c[k - 1], 1e-6) if k > 0 else None
            right = max(c[k + 1] - c[k], 1e-6) if k < n_states - 1 else None
            rising = 1.0 if left is None else 1.0 + d / left
            falling = 1.0 if right is None else 1.0 - d / right
            out[..., r, k] = np.clip(np.where(d < 0, rising, falling), 0.0, 1.0)
        total = out[..., r, :].sum(axis=-1, keepdims=True)
        out[..., r, :] = np.divide(
            out[..., r, :], total, out=np.zeros_like(out[..., r, :]), where=total > 0
        )
    return out


@dataclass
class QuantileSemantizer(BaseSemantizer):
    """Equal-frequency partition with triangular memberships (ablation baseline)."""

    n_states: int = 4  # type: ignore[assignment]
    soft: bool = True
    words: tuple[str, ...] = field(default=(), init=False)
    _centres: np.ndarray | None = field(default=None, init=False, repr=False)
    _edges: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray) -> "QuantileSemantizer":
        X = np.asarray(X, dtype=float)
        flat = X.reshape(-1, X.shape[-1])
        s = int(self.n_states)
        self.words = tuple(f"Q{i + 1}" for i in range(s))
        qs = np.linspace(0, 100, s + 1)
        self._edges = np.percentile(flat, qs, axis=0).T                     # (R, S+1)
        self._centres = np.percentile(flat, (qs[:-1] + qs[1:]) / 2, axis=0).T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._edges is None or self._centres is None:
            raise RuntimeError("QuantileSemantizer must be fitted before transform()")
        X = np.asarray(X, dtype=float)
        if not self.soft:
            return _containment_membership_raw(X, self._edges)
        lb, ub = self._centres[:, 0], self._centres[:, -1]
        span = np.where(ub - lb <= 0, 1.0, ub - lb)
        u = np.clip((X - lb) / span, 0.0, 1.0)
        centres = (self._centres - lb[:, None]) / span[:, None]
        return _triangular_membership(u, np.clip(centres, 0.0, 1.0))


@dataclass
class EqualWidthSemantizer(BaseSemantizer):
    """Crisp equal-width binning — the simplest possible granulation (ablation)."""

    n_states: int = 4  # type: ignore[assignment]
    words: tuple[str, ...] = field(default=(), init=False)
    _edges: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray) -> "EqualWidthSemantizer":
        lb, ub = _robust_range(np.asarray(X, dtype=float))
        s = int(self.n_states)
        self.words = tuple(f"B{i + 1}" for i in range(s))
        self._edges = np.stack([np.linspace(lb[r], ub[r], s + 1) for r in range(lb.size)])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._edges is None:
            raise RuntimeError("EqualWidthSemantizer must be fitted before transform()")
        return _containment_membership_raw(np.asarray(X, dtype=float), self._edges)


@dataclass
class FCMSemantizer(BaseSemantizer):
    """Fuzzy c-means memberships per feature (ablation for the granulation step)."""

    n_states: int = 4  # type: ignore[assignment]
    m: float = 2.0
    n_iter: int = 50
    seed: int = 0
    words: tuple[str, ...] = field(default=(), init=False)
    _centres: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray) -> "FCMSemantizer":
        X = np.asarray(X, dtype=float)
        flat = X.reshape(-1, X.shape[-1])
        rng = np.random.default_rng(self.seed)
        s = int(self.n_states)
        self.words = tuple(f"C{i + 1}" for i in range(s))
        centres = np.zeros((flat.shape[1], s))
        for r in range(flat.shape[1]):
            v = flat[:, r]
            c = np.percentile(v, np.linspace(10, 90, s)) + rng.normal(0, 1e-6, s)
            for _ in range(self.n_iter):
                d = np.abs(v[:, None] - c[None, :]) + 1e-12
                u = 1.0 / (d ** (2.0 / (self.m - 1.0)))
                u /= u.sum(axis=1, keepdims=True)
                w = u**self.m
                c = (w * v[:, None]).sum(0) / (w.sum(0) + 1e-12)
            centres[r] = np.sort(c)
        self._centres = centres
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._centres is None:
            raise RuntimeError("FCMSemantizer must be fitted before transform()")
        X = np.asarray(X, dtype=float)
        out = np.zeros(X.shape + (int(self.n_states),), dtype=np.float32)
        for r in range(X.shape[-1]):
            d = np.abs(X[..., r][..., None] - self._centres[r][None, :]) + 1e-12
            u = 1.0 / (d ** (2.0 / (self.m - 1.0)))
            out[..., r, :] = u / u.sum(axis=-1, keepdims=True)
        return out


def _containment_membership_raw(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """One-hot assignment against per-feature edges given in data units."""
    n_states = edges.shape[1] - 1
    out = np.zeros(X.shape + (n_states,), dtype=np.float32)
    for r in range(X.shape[-1]):
        idx = np.clip(np.searchsorted(edges[r, 1:-1], X[..., r], side="right"), 0, n_states - 1)
        np.put_along_axis(out[..., r, :], idx[..., None], 1.0, axis=-1)
    return out


def make_semantizer(name: str, n_states: int = 4, **kwargs) -> BaseSemantizer:
    """Factory used by the E3 ablation table.

    ``n_states`` is converted to a hedge-algebra ``level`` for the HA schemes,
    so every scheme in the comparison uses the same vocabulary size.
    """
    level = max(int(np.log2(max(n_states, 2))), 1)
    if name == "ha_soft":
        return HASemantizer(level=level, mode="soft", **kwargs)
    if name == "ha_containment":
        return HASemantizer(level=level, mode="containment", **kwargs)
    if name == "quantile":
        return QuantileSemantizer(n_states=n_states, soft=True)
    if name == "quantile_crisp":
        return QuantileSemantizer(n_states=n_states, soft=False)
    if name == "equal_width":
        return EqualWidthSemantizer(n_states=n_states)
    if name == "fcm":
        return FCMSemantizer(n_states=n_states)
    raise ValueError(f"unknown semantizer: {name!r}")
