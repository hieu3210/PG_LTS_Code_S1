"""Linguistic descriptors — compressing a word sequence into a feature vector.

Given membership tensors ``(N, L, R, S)`` produced by a semantiser, four families
of descriptor are computed:

``O`` occupancy
    Mean membership of each word — the fraction of the segment spent in a state
    such as *harmonicity is Very Large*. **Order-free.**
``P`` persistence
    Self-transition probability, separating a state that is held from one touched
    briefly by noise. **Order-dependent.**
``M`` transition
    First- (and optionally second-) order transitions between words, e.g.
    *Rather Large -> Very Large*. **Order-dependent** — this is the family that
    hypothesis H1' is about.
``C`` co-occurrence
    Joint membership of word pairs across two *different* features at the same
    instant, e.g. *F0 is Rather Large while harmonicity is Very Large*.
    **Order-free.** Restricted to a physically motivated pair list to keep the
    dimension down.

Every descriptor keeps a human-readable name so a non-zero classifier weight can
be turned straight back into a rule antecedent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pglts.linguistic.semantize import gloss

#: Feature pairs whose joint state is physically meaningful. Restricting
#: co-occurrence to these keeps the descriptor count linear rather than quadratic.
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("spec_f0", "harmonic_ratio"),
    ("spec_f0", "centroid"),
    ("harmonic_ratio", "hbr"),
    ("harmonic_ratio", "entropy"),
    ("h2_h1", "h3_h1"),
)


@dataclass
class DescriptorBank:
    """Builds ``ψ = [O, P, M, C]`` from membership tensors.

    Parameters
    ----------
    families
        Subset of ``{"O", "P", "M", "C"}`` to emit. The E1 ablation drives this.
    order
        Transition order: 1 or 2.
    pairs
        Feature-name pairs for co-occurrence; ignored when ``"C"`` is not emitted.
    """

    families: tuple[str, ...] = ("O", "P", "M", "C")
    order: int = 1
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS
    feature_names: tuple[str, ...] = ()
    words: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.families) - {"O", "P", "M", "C"}
        if unknown:
            raise ValueError(f"unknown descriptor families: {sorted(unknown)}")
        if self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")

    # -- descriptor families -------------------------------------------
    def occupancy(self, M: np.ndarray) -> np.ndarray:
        """``(N, R*S)`` — mean membership over time."""
        return M.mean(axis=1).reshape(M.shape[0], -1)

    def transition(self, M: np.ndarray, order: int = 1) -> np.ndarray:
        """``(N, R*S^(order+1))`` — normalised soft transition counts."""
        if M.shape[1] <= order:
            n = M.shape[0]
            width = M.shape[2] * M.shape[3] ** (order + 1)
            return np.zeros((n, width), dtype=np.float32)
        if order == 1:
            T = np.einsum("nlra,nlrb->nrab", M[:, :-1], M[:, 1:])
            axes = (-2, -1)
        else:
            T = np.einsum("nlra,nlrb,nlrc->nrabc", M[:, :-2], M[:, 1:-1], M[:, 2:])
            axes = (-3, -2, -1)
        total = T.sum(axis=axes, keepdims=True)
        T = np.divide(T, total, out=np.zeros_like(T), where=total > 0)
        return T.reshape(M.shape[0], -1)

    def persistence(self, M: np.ndarray) -> np.ndarray:
        """``(N, R*S)`` — diagonal of the first-order transition matrix."""
        if M.shape[1] < 2:
            return np.zeros((M.shape[0], M.shape[2] * M.shape[3]), dtype=np.float32)
        T = np.einsum("nlra,nlrb->nrab", M[:, :-1], M[:, 1:])
        total = T.sum(axis=(-2, -1), keepdims=True)
        T = np.divide(T, total, out=np.zeros_like(T), where=total > 0)
        return np.diagonal(T, axis1=-2, axis2=-1).reshape(M.shape[0], -1)

    def cooccurrence(self, M: np.ndarray, pair_idx: list[tuple[int, int]]) -> np.ndarray:
        """``(N, P*S*S)`` — mean joint membership of selected feature pairs."""
        if not pair_idx:
            return np.zeros((M.shape[0], 0), dtype=np.float32)
        blocks = [
            np.einsum("nla,nlb->nab", M[:, :, r], M[:, :, s]) / M.shape[1]
            for r, s in pair_idx
        ]
        return np.stack(blocks, axis=1).reshape(M.shape[0], -1)

    # -- assembly -------------------------------------------------------
    def _pair_indices(self) -> list[tuple[int, int]]:
        lookup = {n: i for i, n in enumerate(self.feature_names)}
        return [(lookup[a], lookup[b]) for a, b in self.pairs if a in lookup and b in lookup]

    def transform(self, M: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
        """Return ``(psi, names, groups)``.

        ``names`` are readable descriptor labels; ``groups`` assigns each column to
        a group for the group-lasso penalty (one group per feature per family).
        """
        M = np.asarray(M, dtype=np.float32)
        n, _, n_feat, n_states = M.shape
        feats = self.feature_names or tuple(f"f{i}" for i in range(n_feat))
        words = self.words or tuple(f"s{i}" for i in range(n_states))

        blocks: list[np.ndarray] = []
        names: list[str] = []
        groups: list[str] = []

        if "O" in self.families:
            blocks.append(self.occupancy(M))
            for r in feats:
                for w in words:
                    names.append(f"O[{r} is {gloss(w)}]")
                    groups.append(f"O:{r}")

        if "P" in self.families:
            blocks.append(self.persistence(M))
            for r in feats:
                for w in words:
                    names.append(f"P[{r} stays {gloss(w)}]")
                    groups.append(f"P:{r}")

        if "M" in self.families:
            blocks.append(self.transition(M, order=1))
            for r in feats:
                for a in words:
                    for b in words:
                        names.append(f"M1[{r}: {gloss(a)} -> {gloss(b)}]")
                        groups.append(f"M1:{r}")
            if self.order == 2:
                blocks.append(self.transition(M, order=2))
                for r in feats:
                    for a in words:
                        for b in words:
                            for c in words:
                                names.append(f"M2[{r}: {gloss(a)} -> {gloss(b)} -> {gloss(c)}]")
                                groups.append(f"M2:{r}")

        if "C" in self.families:
            pair_idx = self._pair_indices()
            blocks.append(self.cooccurrence(M, pair_idx))
            for r, s in pair_idx:
                for a in words:
                    for b in words:
                        names.append(f"C[{feats[r]} is {gloss(a)} & {feats[s]} is {gloss(b)}]")
                        groups.append(f"C:{feats[r]}|{feats[s]}")

        if not blocks:
            raise ValueError("no descriptor families selected")
        psi = np.concatenate(blocks, axis=1).astype(np.float32)
        return psi, names, groups


def static_statistics(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Order-free numeric baseline: per-feature mean, std, p10, p90 and mean |Δ|.

    This is the "aggregate the whole segment" representation that PG-LTS is
    contrasted against, and mirrors what the source paper computes.
    """
    X = np.asarray(X, dtype=float)
    deltas = np.abs(np.diff(X, axis=1)).mean(axis=1) if X.shape[1] > 1 else np.zeros_like(X[:, 0])
    parts = [
        X.mean(axis=1),
        X.std(axis=1),
        np.percentile(X, 10, axis=1),
        np.percentile(X, 90, axis=1),
        deltas,
    ]
    labels = ["mean", "std", "p10", "p90", "mean_abs_delta"]
    names = [f"{lab}[f{r}]" for lab in labels for r in range(X.shape[2])]
    return np.concatenate(parts, axis=1).astype(np.float32), names


def permute_time(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle the time axis of each trajectory independently.

    The control for H1': permutation destroys ordering while leaving the marginal
    distribution of states untouched. A transition-based representation that does
    not degrade under permutation is not actually using temporal order.
    """
    out = np.array(X, copy=True)
    for i in range(out.shape[0]):
        out[i] = out[i][rng.permutation(out.shape[1])]
    return out
