"""PSO tuning of the hedge-algebra fuzziness parameters.

Continues the LTS-PSO / CO-LTS line of pyLTS: instead of hand-picking ``(θ, α)``,
they are optimised against a task objective. Here the objective is subject-disjoint
classification macro-F1 on an **inner** split of the training fold, so the outer
test subjects never influence the granulation.

Two search spaces are offered:

``shared``
    One ``(θ, α)`` pair for every feature — 2 dimensions, cheap and robust with
    only 33 subjects.
``per_feature``
    One pair per feature — ``2R`` dimensions, more expressive but easy to overfit
    at this sample size. Reported as an ablation rather than the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lts.optimization.pso import PSO, PSOConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from pglts.linguistic.semantize import HASemantizer
from pglts.models.pglts_classifier import PGLTSClassifier

#: Keep parameters strictly inside (0, 1); the partition degenerates at the ends.
BOUNDS = (0.15, 0.85)


@dataclass
class TuningResult:
    theta: np.ndarray | float
    alpha: np.ndarray | float
    score: float
    n_evaluations: int


def tune_ha_params(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: tuple[str, ...],
    space: str = "shared",
    level: int = 2,
    families: tuple[str, ...] = ("O", "P", "M", "C"),
    n_particles: int = 12,
    max_iter: int = 8,
    n_inner_splits: int = 3,
    seed: int = 0,
) -> TuningResult:
    """Optimise ``(θ, α)`` by PSO against inner subject-disjoint macro-F1."""
    X = np.asarray(X, dtype=float)
    n_features = X.shape[-1]
    dim = 2 if space == "shared" else 2 * n_features
    if space not in {"shared", "per_feature"}:
        raise ValueError("space must be 'shared' or 'per_feature'")

    n_inner = min(n_inner_splits, int(np.unique(groups).size), int(np.unique(y).size))
    if n_inner < 2:
        return TuningResult(theta=0.5, alpha=0.5, score=float("nan"), n_evaluations=0)
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    splits = list(inner.split(np.zeros(len(y)), y, groups))
    calls = {"n": 0}

    def unpack(position: list[float]):
        p = np.clip(np.asarray(position, dtype=float), *BOUNDS)
        if space == "shared":
            return float(p[0]), float(p[1])
        return p[:n_features], p[n_features:]

    def objective(position: list[float]) -> float:
        calls["n"] += 1
        theta, alpha = unpack(position)
        scores = []
        for tr, te in splits:
            if np.unique(y[tr]).size < 2:
                continue
            clf = PGLTSClassifier(
                semantizer=HASemantizer(level=level, mode="soft", theta=theta, alpha=alpha),
                families=families,
                feature_names=feature_names,
            )
            try:
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                return 1.0
            scores.append(f1_score(y[te], pred, average="macro", zero_division=0))
        return 1.0 - float(np.mean(scores)) if scores else 1.0

    pso = PSO(
        objective,
        PSOConfig(
            n_particles=n_particles,
            max_iter=max_iter,
            bounds=[BOUNDS] * dim,
            seed=seed,
        ),
    )
    best, value = pso.run()
    theta, alpha = unpack(best)
    return TuningResult(
        theta=theta, alpha=alpha, score=1.0 - float(value), n_evaluations=calls["n"]
    )
