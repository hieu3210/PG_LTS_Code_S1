"""Direct measurement of the first-order temporal-order residual.

The analysis defines the residual of a trajectory as

.. math::
    \\Gamma = M - \\mathbb{E}_\\pi[M^\\pi],

where the permutation expectation has the exact finite-sample closed form

.. math::
    \\mathbb{E}_\\pi[M^\\pi]
      = \\frac{L^2 O O^{\\mathsf T} - \\sum_t \\mu_t \\mu_t^{\\mathsf T}}{L(L-1)}.

Both quantities are computable, so the condition "transitions carry information
beyond occupancy iff Gamma != 0" can be tested by *measuring* Gamma rather than
by inferring it from a downstream classification proxy. This module does that.

Two quantities are reported per feature/channel:

``residual_norm``
    :math:`\\|\\Gamma\\|_1`, the magnitude of the departure from the order-free
    expectation. Near zero means the adjacency structure is indistinguishable at
    first order from random ordering of the same states.
``class_separation``
    the between-class spread of the mean residual relative to its within-class
    spread. A nonzero residual establishes order *sensitivity*; taxonomic
    usefulness additionally requires that the class-conditional residuals differ,
    which is what this second quantity measures.
"""

from __future__ import annotations

import numpy as np


def transition_matrix(M: np.ndarray) -> np.ndarray:
    """Observed first-order transition ``(N, R, A, A)`` from memberships.

    ``M`` has shape ``(N, L, R, A)``. Rows are normalised to sum to one, matching
    the descriptor definition used by the classifier.
    """
    if M.shape[1] < 2:
        n, _, r, a = M.shape
        return np.zeros((n, r, a, a), dtype=np.float64)
    T = np.einsum("nlra,nlrb->nrab", M[:, :-1], M[:, 1:]).astype(np.float64)
    s = T.sum(axis=(-2, -1), keepdims=True)
    return np.divide(T, s, out=np.zeros_like(T), where=s > 0)


def permutation_expectation(M: np.ndarray) -> np.ndarray:
    """Exact ``E_pi[M^pi]`` for a uniformly random permutation of the time index.

    This is the closed form of the analysis, not a Monte-Carlo estimate, and it
    includes the finite-sample self-product correction that an independence
    approximation would omit.
    """
    n, L, r, a = M.shape
    if L < 2:
        return np.zeros((n, r, a, a), dtype=np.float64)
    O = M.mean(axis=1).astype(np.float64)                      # (N, R, A)
    outer_O = np.einsum("nra,nrb->nrab", O, O) * (L**2)
    self_terms = np.einsum("nlra,nlrb->nrab", M, M).astype(np.float64)
    E = (outer_O - self_terms) / (L * (L - 1))
    s = E.sum(axis=(-2, -1), keepdims=True)
    return np.divide(E, s, out=np.zeros_like(E), where=s > 0)


def order_residual(M: np.ndarray) -> np.ndarray:
    """``Gamma = M - E_pi[M^pi]`` with shape ``(N, R, A, A)``."""
    return transition_matrix(M) - permutation_expectation(M)


def residual_norm(M: np.ndarray) -> np.ndarray:
    """``||Gamma||_1`` per instance and feature, shape ``(N, R)``.

    This raw magnitude is **not** on its own evidence of temporal order. A single
    randomly ordered trajectory also departs from the permutation *expectation*,
    because the expectation averages over permutations while any one realisation
    fluctuates around it. At the short lengths used here (``L`` of 7 to 19) that
    fluctuation is large. Use :func:`standardised_residual` for the test.
    """
    return np.abs(order_residual(M)).sum(axis=(-2, -1))


def standardised_residual(
    M: np.ndarray, n_perm: int = 20, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Residual magnitude standardised by its own permutation null.

    For each instance we compute :math:`\\|\\Gamma_{\\mathrm{obs}}\\|_1` and the
    same quantity for ``n_perm`` random reorderings of the identical membership
    multiset. The reported statistic is

    .. math::
        z = \\frac{\\|\\Gamma_{\\mathrm{obs}}\\|_1 - \\mathrm{mean}_\\pi
             \\|\\Gamma_\\pi\\|_1}{\\mathrm{std}_\\pi \\|\\Gamma_\\pi\\|_1}.

    ``z`` near zero means the observed adjacency structure is indistinguishable
    from a random ordering of the same states — the residual is consistent with
    finite-sample fluctuation alone. Positive ``z`` means genuine order structure.

    Returns ``(z, ratio)`` per instance, where ``ratio`` is the observed norm
    divided by the permutation mean.
    """
    rng = np.random.default_rng(seed)
    obs = residual_norm(M).mean(axis=1)                      # (N,)
    L = M.shape[1]
    null = np.empty((n_perm, M.shape[0]))
    for b in range(n_perm):
        idx = np.argsort(rng.random((M.shape[0], L)), axis=1)
        permuted = np.take_along_axis(M, idx[:, :, None, None], axis=1)
        null[b] = residual_norm(permuted).mean(axis=1)
    mu, sd = null.mean(axis=0), null.std(axis=0)
    z = (obs - mu) / (sd + 1e-12)
    ratio = obs / (mu + 1e-12)
    return z, ratio


def class_separation(gamma_norm: np.ndarray, y: np.ndarray) -> float:
    """Between-class over within-class spread of the residual magnitude.

    Computed on the per-instance residual norm averaged over features. Values
    near zero indicate that, although a residual may exist, it does not differ
    across classes and is therefore not taxonomically useful.
    """
    v = gamma_norm.mean(axis=1) if gamma_norm.ndim > 1 else gamma_norm
    classes = np.unique(y)
    means = np.array([v[y == c].mean() for c in classes])
    within = np.mean([v[y == c].std() for c in classes])
    return float(means.std() / (within + 1e-12))


def summarise(M: np.ndarray, y: np.ndarray, label: str, n_perm: int = 20) -> dict:
    """Residual summary for one representation.

    ``mean_z`` is the headline statistic: the residual standardised by its own
    permutation null. ``frac_z_gt_2`` is the fraction of instances whose observed
    residual exceeds the null by more than two standard deviations.
    """
    g = residual_norm(M)
    z, ratio = standardised_residual(M, n_perm=n_perm)
    return {
        "representation": label,
        "n_instances": int(M.shape[0]),
        "steps_L": int(M.shape[1]),
        "mean_residual_l1": float(g.mean()),
        "mean_z": float(z.mean()),
        "median_z": float(np.median(z)),
        "frac_z_gt_2": float((z > 2.0).mean()),
        "obs_over_null": float(ratio.mean()),
        "class_separation": class_separation(g, y),
    }
