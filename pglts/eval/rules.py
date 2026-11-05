"""Interpretability metrics: rule stability, subject support, faithfulness.

Three failure modes of a rule-based claim are tested here.

*Instability.* A rule set that changes completely when the test individual changes
describes the training subjects, not the species — measured by the mean pairwise
Jaccard overlap of the rule sets recovered in each subject-disjoint fold.

*Spurious support.* A rule backed by many segments of one talkative individual is
not well supported — support is therefore counted over subjects, never segments.

*Unfaithfulness.* A rule that is printed but not relied upon is decoration. This
is measured by erasure (comprehensiveness and sufficiency, DeYoung et al. 2020)
rather than by checking whether a perturbation moves the score in the direction
the weights predict — for a linear model the latter is tautological, since the
score *is* the weighted sum of the descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pglts.models.pglts_classifier import PGLTSClassifier, Rule


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def rule_stability(rule_sets: list[set[tuple[str, str, str]]]) -> float:
    """Mean pairwise Jaccard overlap of rule sets across folds.

    1.0 means every fold recovered the same rules; 0.0 means they share none.
    """
    if len(rule_sets) < 2:
        return float("nan")
    scores = [
        jaccard(rule_sets[i], rule_sets[j])
        for i in range(len(rule_sets))
        for j in range(i + 1, len(rule_sets))
    ]
    return float(np.mean(scores))


def subject_support(
    clf: PGLTSClassifier,
    X: np.ndarray,
    groups: np.ndarray,
    rules: list[Rule],
    quantile: float = 0.75,
) -> dict[str, float]:
    """Fraction of **subjects** in which each rule fires.

    Counting segments instead would let one individual with many recordings
    inflate a rule's apparent frequency, so support is averaged over subjects.
    A rule "fires" for a subject when that subject's median descriptor value
    exceeds the corpus ``quantile``.
    """
    psi = clf.transform(X)
    names = {n: i for i, n in enumerate(clf._names)}
    subjects = np.unique(groups)

    out: dict[str, float] = {}
    for rule in rules:
        col = names.get(rule.descriptor)
        if col is None:
            continue
        threshold = np.quantile(psi[:, col], quantile)
        fired = sum(
            1 for s in subjects if np.median(psi[groups == s, col]) > threshold
        )
        out[rule.descriptor] = fired / max(subjects.size, 1)
    return out


@dataclass
class FaithfulnessResult:
    """Sufficiency and comprehensiveness of a reported rule set.

    Attributes
    ----------
    comprehensiveness
        Accuracy lost when the rule descriptors are removed. High means the rules
        carry the decision.
    sufficiency
        Accuracy lost when *only* the rule descriptors are kept. Near zero means
        the rules alone reproduce the model.
    """

    n_rules: int
    n_descriptors_used: int
    n_descriptors_total: int
    accuracy_full: float
    accuracy_without_rules: float
    accuracy_rules_only: float
    comprehensiveness: float
    sufficiency: float


@dataclass
class InterventionResult:
    """Outcome of one controlled signal-level perturbation."""

    name: str
    accuracy_before: float
    accuracy_after: float
    accuracy_drop: float
    mean_abs_score_shift: float
    n_samples: int


def rule_faithfulness(
    clf: PGLTSClassifier,
    X: np.ndarray,
    y: np.ndarray,
    top_k: int = 5,
    rank_by: str = "contribution",
) -> FaithfulnessResult:
    """Do the *reported* rules carry the model's decision?

    Uses the standard erasure metrics of DeYoung et al. (2020):

    *comprehensiveness*
        accuracy(full) − accuracy(rule descriptors erased). Large means the model
        genuinely depends on the descriptors we print.
    *sufficiency*
        accuracy(full) − accuracy(only rule descriptors kept). Near zero means the
        printed rules alone reproduce the model.

    Erasure sets a descriptor to its **training mean**, which is exactly 0 in the
    standardised space the classifier operates in, so no distribution shift is
    introduced beyond the erasure itself.

    Why not test whether the weights predict the direction of a score change: for
    a linear model that is tautological, since the score *is* the weighted sum.
    Erasure asks the non-trivial question — whether a small, human-readable subset
    of the descriptors accounts for the behaviour of the whole model.
    """
    psi = clf.transform(X)                    # standardised; column mean is 0
    names = {n: i for i, n in enumerate(clf._names)}
    selected = clf.rules(top_k=top_k, rank_by=rank_by)
    rule_cols = sorted({names[r.descriptor] for r in selected if r.descriptor in names})

    def accuracy(matrix: np.ndarray) -> float:
        pred = clf._model.classes_[np.argmax(matrix @ clf._model.coef_.T + clf._model.intercept_, axis=1)]
        return float((pred == y).mean())

    full = accuracy(psi)

    erased = psi.copy()
    erased[:, rule_cols] = 0.0
    without = accuracy(erased)

    only = np.zeros_like(psi)
    only[:, rule_cols] = psi[:, rule_cols]
    rules_only = accuracy(only)

    return FaithfulnessResult(
        n_rules=len(selected),
        n_descriptors_used=len(rule_cols),
        n_descriptors_total=psi.shape[1],
        accuracy_full=full,
        accuracy_without_rules=without,
        accuracy_rules_only=rules_only,
        comprehensiveness=full - without,
        sufficiency=full - rules_only,
    )


def signal_intervention(
    clf: PGLTSClassifier,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    target_feature: str,
    delta: float = 1.0,
    n_samples: int = 400,
    seed: int = 0,
) -> InterventionResult:
    """Sensitivity of the model to a shift in one physical feature.

    Reported as an accuracy drop and a mean absolute score shift, **not** as a
    direction-agreement "fidelity". Predicting the sign of the score change from
    the weights is not well posed here: raising a feature redistributes membership
    across linguistic states non-monotonically (mass can leave both the lowest and
    a middle state at once), and the classifier acts on standardised descriptors,
    so an ordinal argument about the raw feature does not determine the sign.

    What this does measure is whether the decision depends on the physical quantity
    at all — a necessary condition for a rule that names it to be meaningful.
    """
    rng = np.random.default_rng(seed)
    if target_feature not in feature_names:
        raise ValueError(f"{target_feature!r} not in feature set")
    r = feature_names.index(target_feature)

    n = min(n_samples, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    base = np.asarray(X[idx], dtype=float)
    labels = y[idx]

    scale = base[..., r].std() + 1e-12
    shifted = base.copy()
    shifted[..., r] += delta * scale

    before = clf.decision_function(base)
    after = clf.decision_function(shifted)
    acc_before = float((clf.classes_[np.argmax(before, axis=1)] == labels).mean())
    acc_after = float((clf.classes_[np.argmax(after, axis=1)] == labels).mean())

    return InterventionResult(
        name=f"{target_feature} +{delta} sd",
        accuracy_before=acc_before,
        accuracy_after=acc_after,
        accuracy_drop=acc_before - acc_after,
        mean_abs_score_shift=float(np.mean(np.abs(after - before))),
        n_samples=int(n),
    )


def summarise_rules(rules: list[Rule]) -> dict[str, float]:
    """Compactness statistics for the transparency table."""
    if not rules:
        return {"n_rules": 0, "n_species_covered": 0, "mean_abs_weight": 0.0, "n_groups": 0}
    return {
        "n_rules": len(rules),
        "n_species_covered": len({r.species for r in rules}),
        "mean_abs_weight": float(np.mean([abs(r.weight) for r in rules])),
        "n_groups": len({r.group for r in rules}),
    }
