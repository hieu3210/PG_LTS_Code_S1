"""Multinomial logistic regression with a sparse-group-lasso penalty.

Objective::

    L(W, b) = CE(W, b) + lambda_1 * ||W||_1 + lambda_g * sum_g ||W_g||_2

The elementwise ``L1`` term removes individual descriptors; the group term acts on
whole blocks — one block per (feature, descriptor-family) pair, spanning every
class — so a feature group either participates in the decision or is dropped
entirely. That is what keeps the extracted rule set small enough to read.

scikit-learn has no group-lasso solver, so this is a compact FISTA (accelerated
proximal gradient) implementation. The proximal operator of the sparse group
lasso is the composition of elementwise soft-thresholding followed by block
soft-thresholding, which is exact for this penalty.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _prox(W: np.ndarray, step: float, l1: float, lg: float, blocks: list[np.ndarray]) -> np.ndarray:
    """Proximal operator: soft-threshold, then block soft-threshold."""
    V = np.sign(W) * np.maximum(np.abs(W) - step * l1, 0.0)
    if lg > 0:
        for idx in blocks:
            norm = np.linalg.norm(V[idx])
            if norm <= 0:
                continue
            V[idx] *= max(0.0, 1.0 - step * lg * np.sqrt(idx.size) / norm)
    return V


class SparseGroupLogistic(BaseEstimator, ClassifierMixin):
    """Sparse-group multinomial logistic regression.

    Parameters
    ----------
    l1, l_group
        Penalty strengths for the elementwise and group terms.
    groups
        Group id per column of ``X``. Columns sharing an id form one block.
    class_weight
        ``"balanced"`` reweights classes inversely to frequency — necessary here
        because the species are unevenly represented.
    """

    def __init__(
        self,
        l1: float = 1e-3,
        l_group: float = 1e-3,
        groups: list[str] | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        class_weight: str | None = "balanced",
    ) -> None:
        self.l1 = l1
        self.l_group = l_group
        self.groups = groups
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight

    # -- fitting --------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SparseGroupLogistic":
        from revision.solver import StableSparseGroup
        model = StableSparseGroup(self.l1, self.l_group, self.max_iter, self.tol,
                                  class_weight=self.class_weight).fit(X, y, self.groups)
        for name in ("classes_", "coef_", "intercept_", "n_iter_", "converged_", "objective_"):
            setattr(self, name, getattr(model, name))
        return self

    def _blocks(self, d: int) -> list[np.ndarray]:
        if not self.groups:
            return []
        groups = np.asarray(self.groups)
        if groups.size != d:
            raise ValueError(f"groups has length {groups.size}, expected {d}")
        return [np.flatnonzero(groups == g) for g in np.unique(groups)]

    def _objective(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_w: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        blocks: list[np.ndarray],
    ) -> float:
        P = _softmax(X @ W + b)
        ce = -float((sample_w[:, None] * Y * np.log(P + 1e-12)).sum()) / X.shape[0]
        pen = self.l1 * float(np.abs(W).sum())
        pen += self.l_group * sum(np.sqrt(idx.size) * np.linalg.norm(W[idx]) for idx in blocks)
        return ce + pen

    # -- prediction -----------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    # -- sparsity -------------------------------------------------------
    @property
    def n_nonzero(self) -> int:
        """Number of descriptors with a non-zero weight for at least one class."""
        return int((np.abs(self.coef_).max(axis=0) > 1e-10).sum())

    @property
    def active_groups(self) -> list[str]:
        if not self.groups:
            return []
        groups = np.asarray(self.groups)
        active = np.abs(self.coef_).max(axis=0) > 1e-10
        return sorted(set(groups[active].tolist()))
