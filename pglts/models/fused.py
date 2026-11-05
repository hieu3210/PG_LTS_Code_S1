"""Fused PG-LTS — spectral linguistic states plus cycle-synchronous dynamics.

E1 and E2 together identify what each view contributes:

* free-running windows over physical spectral features carry the discriminative
  *level* information (which frequency band, how harmonic), but their transitions
  are order-free;
* cycle-synchronous phase trajectories carry genuine *ordering* information about
  stroke kinematics, but little absolute spectral detail.

The fused model concatenates the linguistic descriptors of both views, so a rule
can combine a spectral state with a stroke-phase transition, and lets the
sparse-group penalty decide how much of each survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import StandardScaler

from pglts.linguistic.descriptors import DEFAULT_PAIRS, DescriptorBank
from pglts.linguistic.semantize import HASemantizer
from pglts.models.pglts_classifier import Rule
from pglts.models.sparse_group import SparseGroupLogistic
from pglts.signal.cycle import CYCLE_CHANNELS


def stack_cycles(cycles: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-segment cycle blocks into ``(total_cycles, n_phase, C)`` + ids."""
    seg_id = np.concatenate([np.full(len(c), i) for i, c in enumerate(cycles)])
    return np.concatenate(list(cycles), axis=0).astype(float), seg_id


def pool_by_segment(values: np.ndarray, seg_id: np.ndarray, n_segments: int) -> np.ndarray:
    """Mean of per-cycle descriptors within each segment."""
    out = np.zeros((n_segments, values.shape[1]), dtype=np.float32)
    counts = np.bincount(seg_id, minlength=n_segments).astype(float)
    np.add.at(out, seg_id, values)
    return out / np.maximum(counts, 1.0)[:, None]


@dataclass
class FusedPGLTS:
    """PG-LTS over both the spectral-window view and the wingbeat-phase view.

    ``fit`` takes ``(X_window, cycles)``: the rectangular window trajectories and
    the ragged list of cycle blocks, aligned segment by segment.
    """

    feature_names: tuple[str, ...] = ()
    window_families: tuple[str, ...] = ("O", "P", "C")
    cycle_families: tuple[str, ...] = ("O", "P", "M")
    level: int = 2
    mode: str = "soft"
    theta: float = 0.5
    alpha: float = 0.5
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS
    l1: float = 1e-3
    l_group: float = 5e-4

    _win_sem: HASemantizer | None = field(default=None, init=False, repr=False)
    _cyc_sem: HASemantizer | None = field(default=None, init=False, repr=False)
    _win_bank: DescriptorBank | None = field(default=None, init=False, repr=False)
    _cyc_bank: DescriptorBank | None = field(default=None, init=False, repr=False)
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _model: SparseGroupLogistic | None = field(default=None, init=False, repr=False)
    _names: list[str] = field(default_factory=list, init=False, repr=False)
    _groups: list[str] = field(default_factory=list, init=False, repr=False)

    # -- representation -------------------------------------------------
    def _build(self, X_window, cycles, fit: bool) -> np.ndarray:
        X_window = np.asarray(X_window, dtype=float)

        if fit:
            self._win_sem = HASemantizer(
                level=self.level, mode=self.mode, theta=self.theta, alpha=self.alpha
            ).fit(X_window)
            self._win_bank = DescriptorBank(
                families=self.window_families,
                pairs=self.pairs,
                feature_names=self.feature_names,
                words=tuple(self._win_sem.words),
            )
        assert self._win_sem is not None and self._win_bank is not None
        win_desc, win_names, win_groups = self._win_bank.transform(
            self._win_sem.transform(X_window)
        )

        flat, seg_id = stack_cycles(cycles)
        if fit:
            self._cyc_sem = HASemantizer(level=self.level, mode=self.mode).fit(flat)
            self._cyc_bank = DescriptorBank(
                families=self.cycle_families,
                feature_names=CYCLE_CHANNELS,
                words=tuple(self._cyc_sem.words),
                pairs=(),
            )
        assert self._cyc_sem is not None and self._cyc_bank is not None
        cyc_desc, cyc_names, cyc_groups = self._cyc_bank.transform(
            self._cyc_sem.transform(flat)
        )
        cyc_pooled = pool_by_segment(cyc_desc, seg_id, len(cycles))

        if fit:
            self._names = [f"spec::{n}" for n in win_names] + [f"cyc::{n}" for n in cyc_names]
            self._groups = [f"spec:{g}" for g in win_groups] + [f"cyc:{g}" for g in cyc_groups]
        return np.hstack([win_desc, cyc_pooled]).astype(np.float32)

    # -- API -------------------------------------------------------------
    def fit(self, X_window, cycles, y: np.ndarray) -> "FusedPGLTS":
        psi = self._build(X_window, cycles, fit=True)
        self._scaler = StandardScaler().fit(psi)
        self._model = SparseGroupLogistic(
            l1=self.l1, l_group=self.l_group, groups=self._groups
        ).fit(self._scaler.transform(psi), y)
        return self

    def transform(self, X_window, cycles) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("FusedPGLTS must be fitted first")
        return self._scaler.transform(self._build(X_window, cycles, fit=False))

    def predict(self, X_window, cycles) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(self.transform(X_window, cycles))

    def decision_function(self, X_window, cycles) -> np.ndarray:
        assert self._model is not None
        return self._model.decision_function(self.transform(X_window, cycles))

    @property
    def classes_(self) -> np.ndarray:
        assert self._model is not None
        return self._model.classes_

    @property
    def n_descriptors(self) -> int:
        return len(self._names)

    @property
    def n_active(self) -> int:
        assert self._model is not None
        return self._model.n_nonzero

    def rules(self, top_k: int | None = None, min_weight: float = 1e-6) -> list[Rule]:
        """Rules from non-zero weights, strongest first, at most ``top_k`` per class."""
        assert self._model is not None
        out: list[Rule] = []
        for j, species in enumerate(self._model.classes_):
            weights = self._model.coef_[j]
            kept = 0
            for i in np.argsort(-np.abs(weights)):
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

    def rule_keys(self, top_k: int | None = None) -> set[tuple[str, str, str]]:
        return {r.key() for r in self.rules(top_k=top_k)}


def fused_fit_predict(feature_names: tuple[str, ...], **kwargs):
    """``fit_predict`` closure over paired ``(window, cycles)`` inputs.

    ``X`` passed to ``cross_validate`` must be a list of ``(traj, cycle_block)``
    tuples so that both views are indexed consistently by the splitter.
    """

    def fit_predict(X_train, y_train, X_test):
        Aw = np.asarray([t for t, _ in X_train], dtype=float)
        Ac = [c for _, c in X_train]
        Bw = np.asarray([t for t, _ in X_test], dtype=float)
        Bc = [c for _, c in X_test]
        model = FusedPGLTS(feature_names=feature_names, **kwargs).fit(Aw, Ac, y_train)
        return model.predict(Bw, Bc), {
            "n_descriptors": model.n_descriptors,
            "n_active": model.n_active,
        }

    return fit_predict
