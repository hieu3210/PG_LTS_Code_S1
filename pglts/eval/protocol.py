"""Evaluation protocol: subject-disjoint splitting, aggregation, uncertainty.

The independent unit is the **individual insect**, not the segment. One 60 s
recording yields tens of overlapping segments that share the same animal, the
same distance to the antenna and the same reflected power, so a random segment
split lets a model recognise the individual instead of the species. Everything
here exists to make that impossible — and to measure how much it was worth.

Two families of metric are reported:

*segment level*
    Every retained micro-Doppler segment scored independently.
*subject level*
    Segment predictions pooled per individual by majority vote. This is the
    operationally meaningful number: in deployment one classifies an insect, not
    a 2 s window.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold


@dataclass
class FoldResult:
    """Predictions of one fold, kept raw so metrics can be recomputed."""

    y_true: np.ndarray
    y_pred: np.ndarray
    groups: np.ndarray
    n_descriptors: int = 0
    n_active: int = 0


@dataclass
class EvalResult:
    """Aggregated outcome of a cross-validation run."""

    name: str
    folds: list[FoldResult] = field(default_factory=list)

    # -- pooled predictions --------------------------------------------
    def _pooled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate([f.y_true for f in self.folds]),
            np.concatenate([f.y_pred for f in self.folds]),
            np.concatenate([f.groups for f in self.folds]),
        )

    def segment_metrics(self) -> dict[str, float]:
        y, p, _ = self._pooled()
        return {
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        }

    def subject_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Majority-vote label per subject, with the subject's true label."""
        y, p, g = self._pooled()
        subjects = np.unique(g)
        true, pred = [], []
        for s in subjects:
            m = g == s
            true.append(Counter(y[m]).most_common(1)[0][0])
            pred.append(Counter(p[m]).most_common(1)[0][0])
        return np.asarray(true), np.asarray(pred)

    def subject_metrics(self) -> dict[str, float]:
        y, p = self.subject_predictions()
        return {
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        }

    def per_species_recall(self) -> dict[str, float]:
        y, p, _ = self._pooled()
        labels = np.unique(y)
        rec = recall_score(y, p, labels=labels, average=None, zero_division=0)
        return {str(l): float(r) for l, r in zip(labels, rec)}

    def confusion(self) -> tuple[np.ndarray, list[str]]:
        y, p, _ = self._pooled()
        labels = sorted(set(y.tolist()) | set(p.tolist()))
        return confusion_matrix(y, p, labels=labels), labels

    @property
    def n_descriptors(self) -> float:
        return float(np.mean([f.n_descriptors for f in self.folds])) if self.folds else 0.0

    @property
    def n_active(self) -> float:
        return float(np.mean([f.n_active for f in self.folds])) if self.folds else 0.0

    def summary(self) -> dict[str, float | str]:
        seg = self.segment_metrics()
        sub = self.subject_metrics()
        return {
            "model": self.name,
            "seg_accuracy": seg["accuracy"],
            "seg_macro_f1": seg["macro_f1"],
            "seg_balanced_acc": seg["balanced_accuracy"],
            "subj_accuracy": sub["accuracy"],
            "subj_macro_f1": sub["macro_f1"],
            "n_descriptors": self.n_descriptors,
            "n_active": self.n_active,
        }


def subject_bootstrap_ci(
    result: EvalResult,
    metric: str = "macro_f1",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a metric by resampling **subjects**, not segments.

    Resampling segments would treat dozens of correlated windows from one animal
    as independent evidence and produce implausibly tight intervals.

    Returns ``(point_estimate, lo, hi)``.
    """
    y, p, g = result._pooled()
    subjects = np.unique(g)
    rng = np.random.default_rng(seed)
    # Row indices per subject, so a subject drawn twice contributes twice. Building
    # a boolean mask instead would collapse duplicate draws and silently turn the
    # bootstrap into ~63% subsampling.
    rows_of = {s: np.flatnonzero(g == s) for s in subjects}

    def score(idx: np.ndarray) -> float:
        if idx.size == 0 or np.unique(y[idx]).size < 2:
            return np.nan
        if metric == "macro_f1":
            return float(f1_score(y[idx], p[idx], average="macro", zero_division=0))
        if metric == "accuracy":
            return float(accuracy_score(y[idx], p[idx]))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y[idx], p[idx]))
        raise ValueError(f"unknown metric {metric!r}")

    point = score(np.arange(y.size))
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(subjects, size=subjects.size, replace=True)
        idx = np.concatenate([rows_of[s] for s in picked])
        v = score(idx)
        if not np.isnan(v):
            draws.append(v)
    if not draws:
        return point, np.nan, np.nan
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_difference_ci(
    a: EvalResult,
    b: EvalResult,
    metric: str = "macro_f1",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI of ``a - b`` over a shared subject resample.

    This is the decision rule for H1': the improvement counts only if the
    interval excludes zero. Both results must cover the same subjects.
    """
    ya, pa, ga = a._pooled()
    yb, pb, gb = b._pooled()
    subjects = np.intersect1d(np.unique(ga), np.unique(gb))
    rng = np.random.default_rng(seed)
    # As in subject_bootstrap_ci, index concatenation preserves duplicate draws.
    rows_a = {s: np.flatnonzero(ga == s) for s in subjects}
    rows_b = {s: np.flatnonzero(gb == s) for s in subjects}

    def score(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> float:
        if idx.size == 0 or np.unique(y[idx]).size < 2:
            return np.nan
        if metric == "macro_f1":
            return float(f1_score(y[idx], p[idx], average="macro", zero_division=0))
        return float(accuracy_score(y[idx], p[idx]))

    point = score(ya, pa, np.arange(ya.size)) - score(yb, pb, np.arange(yb.size))
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(subjects, size=subjects.size, replace=True)
        ia = np.concatenate([rows_a[s] for s in picked])
        ib = np.concatenate([rows_b[s] for s in picked])
        va, vb = score(ya, pa, ia), score(yb, pb, ib)
        if not (np.isnan(va) or np.isnan(vb)):
            draws.append(va - vb)
    if not draws:
        return point, np.nan, np.nan
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def make_splitter(scheme: str, n_splits: int = 5, seed: int = 0):
    """Cross-validation splitter.

    ``"group"``  subject-disjoint stratified k-fold (the default protocol);
    ``"loso"``   leave-one-subject-out, 33 folds;
    ``"random"`` plain stratified k-fold that **ignores** subjects — included
    only to quantify the leakage it produces, never as a headline result.
    """
    if scheme == "group":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if scheme == "loso":
        return LeaveOneGroupOut()
    if scheme == "random":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"unknown split scheme: {scheme!r}")


def assert_disjoint(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Guard against subject leakage; raises if a subject straddles the split."""
    overlap = set(groups[train_idx].tolist()) & set(groups[test_idx].tolist())
    if overlap:
        raise AssertionError(f"subject leakage across fold: {sorted(overlap)}")


def cross_validate(
    fit_predict,
    X,
    y: np.ndarray,
    groups: np.ndarray,
    scheme: str = "group",
    n_splits: int = 5,
    seed: int = 0,
    name: str = "model",
    check_disjoint: bool = True,
) -> EvalResult:
    """Run cross-validation, enforcing subject-disjointness by default.

    ``fit_predict(X_train, y_train, X_test) -> (y_pred, info)`` where ``info`` is a
    dict that may carry ``n_descriptors`` / ``n_active``.
    """
    splitter = make_splitter(scheme, n_splits=n_splits, seed=seed)
    result = EvalResult(name=name)

    splits = (
        splitter.split(np.zeros(len(y)), y, groups)
        if scheme != "random"
        else splitter.split(np.zeros(len(y)), y)
    )
    for train_idx, test_idx in splits:
        if check_disjoint and scheme != "random":
            assert_disjoint(groups, train_idx, test_idx)
        if np.unique(y[train_idx]).size < 2:
            continue
        X_train = X[train_idx] if not isinstance(X, list) else [X[i] for i in train_idx]
        X_test = X[test_idx] if not isinstance(X, list) else [X[i] for i in test_idx]
        y_pred, info = fit_predict(X_train, y[train_idx], X_test)
        # CatBoost returns a column vector for string targets; flatten so every
        # model contributes a plain 1-D label array.
        result.folds.append(
            FoldResult(
                y_true=y[test_idx],
                y_pred=np.asarray(y_pred).reshape(-1),
                groups=groups[test_idx],
                n_descriptors=int(info.get("n_descriptors", 0)),
                n_active=int(info.get("n_active", 0)),
            )
        )
    return result
