"""The PG-LTS classifier: semantisation -> descriptors -> sparse linear model.

The whole pipeline is fitted inside a training fold. That matters: membership
breakpoints, descriptor scaling and the sparse weights all depend on data, so
fitting any of them on the full corpus would leak test-subject statistics into
the representation and inflate subject-disjoint scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import StandardScaler

from pglts.linguistic.descriptors import DEFAULT_PAIRS, DescriptorBank
from pglts.linguistic.semantize import BaseSemantizer, HASemantizer
from pglts.models.sparse_group import SparseGroupLogistic


@dataclass
class Rule:
    """One IF-THEN rule read off a non-zero classifier weight."""

    species: str
    descriptor: str
    weight: float
    group: str

    @property
    def direction(self) -> str:
        return "increases" if self.weight > 0 else "decreases"

    def as_text(self) -> str:
        return (
            f"IF {self.descriptor} is high THEN evidence for {self.species} "
            f"{self.direction} (w={self.weight:+.3f})"
        )

    def key(self) -> tuple[str, str, str]:
        """Identity used for cross-fold stability: class, descriptor, sign."""
        return (self.species, self.descriptor, "+" if self.weight > 0 else "-")


@dataclass
class PGLTSClassifier:
    """Physically-Grounded Linguistic Time Series classifier.

    Parameters
    ----------
    semantizer
        Numeric -> linguistic mapping. Defaults to a soft level-2 hedge algebra.
    families
        Descriptor families to use: any subset of ``{"O", "P", "M", "C"}``.
    order
        Transition order (1 or 2).
    l1, l_group
        Sparsity penalties passed to :class:`SparseGroupLogistic`.
    """

    semantizer: BaseSemantizer = field(default_factory=lambda: HASemantizer(level=2, mode="soft"))
    families: tuple[str, ...] = ("O", "P", "M", "C")
    order: int = 1
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS
    feature_names: tuple[str, ...] = ()
    l1: float = 1e-3
    l_group: float = 5e-4
    max_iter: int = 400

    _bank: DescriptorBank | None = field(default=None, init=False, repr=False)
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _model: SparseGroupLogistic | None = field(default=None, init=False, repr=False)
    _names: list[str] = field(default_factory=list, init=False, repr=False)
    _groups: list[str] = field(default_factory=list, init=False, repr=False)
    _contribution: np.ndarray | None = field(default=None, init=False, repr=False)

    # -- representation -------------------------------------------------
    def _describe(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.semantizer.fit(X)
            self._bank = DescriptorBank(
                families=self.families,
                order=self.order,
                pairs=self.pairs,
                feature_names=self.feature_names,
                words=tuple(self.semantizer.words),
            )
        assert self._bank is not None
        M = self.semantizer.transform(X)
        psi, names, groups = self._bank.transform(M)
        if fit:
            self._names, self._groups = names, groups
        return psi

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Descriptor matrix for already-fitted components."""
        if self._bank is None or self._scaler is None:
            raise RuntimeError("PGLTSClassifier must be fitted first")
        return self._scaler.transform(self._describe(X, fit=False))

    # -- sklearn-ish API ------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PGLTSClassifier":
        psi = self._describe(np.asarray(X, dtype=float), fit=True)
        self._scaler = StandardScaler().fit(psi)
        scaled = self._scaler.transform(psi)
        self._model = SparseGroupLogistic(
            l1=self.l1,
            l_group=self.l_group,
            groups=self._groups,
            max_iter=self.max_iter,
        ).fit(scaled, y)

        # Mean absolute contribution |w_kd * psi_nd| of each descriptor to each
        # class score, over the training set. For a linear model this — not |w|
        # alone — is what determines how much a descriptor moves the decision:
        # two descriptors with equal weight contribute differently if their
        # standardised values differ in typical magnitude.
        self._contribution = np.abs(self._model.coef_) * np.abs(scaled).mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(self.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict_proba(self.transform(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.decision_function(self.transform(X))

    @property
    def classes_(self) -> np.ndarray:
        assert self._model is not None
        return self._model.classes_

    # -- interpretability -----------------------------------------------
    @property
    def n_descriptors(self) -> int:
        """Total descriptor count before sparsification."""
        return len(self._names)

    @property
    def n_active(self) -> int:
        """Descriptors surviving the sparsity penalties."""
        assert self._model is not None
        return self._model.n_nonzero

    def rules(
        self,
        top_k: int | None = None,
        min_weight: float = 1e-6,
        rank_by: str = "contribution",
    ) -> list[Rule]:
        r"""Extract rules, most important first.

        ``top_k`` limits the number *per class*, which is how the rule-set size is
        reported in the transparency table.

        ``rank_by`` selects the importance measure:

        ``"contribution"`` (default)
            mean :math:`|w_{kd}\,\psi_{nd}|` over the training set — how much the
            descriptor actually moves the class score.
        ``"weight"``
            :math:`|w_{kd}|` alone. This is the conventional choice and is retained
            for comparison, but erasure tests show it does **not** identify the
            descriptors the model relies on (see ``eval/rules.py``).
        """
        assert self._model is not None
        coef = self._model.coef_
        if rank_by not in {"contribution", "weight"}:
            raise ValueError("rank_by must be 'contribution' or 'weight'")
        importance = (
            np.abs(coef) if rank_by == "weight" or self._contribution is None
            else self._contribution
        )
        out: list[Rule] = []
        for j, species in enumerate(self._model.classes_):
            weights = coef[j]
            order = np.argsort(-importance[j])
            kept = 0
            for i in order:
                if abs(weights[i]) <= min_weight:
                    break
                out.append(
                    Rule(
                        species=str(species),
                        descriptor=self._names[i],
                        weight=float(weights[i]),
                        group=self._groups[i],
                    )
                )
                kept += 1
                if top_k is not None and kept >= top_k:
                    break
        return out

    def rule_keys(
        self, top_k: int | None = None, rank_by: str = "contribution"
    ) -> set[tuple[str, str, str]]:
        """Rule identities, for the cross-fold stability metric."""
        return {r.key() for r in self.rules(top_k=top_k, rank_by=rank_by)}
