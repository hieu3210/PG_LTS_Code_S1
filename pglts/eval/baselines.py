"""Comparison models, all consuming the same trajectories and the same folds.

Three tiers:

*numeric static*
    Order-free aggregate statistics + a conventional classifier. Includes the
    hierarchical CatBoost/ExtraTrees cascade of the source paper, so that its
    reported behaviour can be checked under a correct protocol.
*numeric temporal*
    ROCKET, MultiRocket, DrCIF and RDST from ``aeon``, operating on the raw
    multivariate trajectory. These are published implementations of strong modern
    time-series classifiers, and are genuine comparisons. (InceptionTime requires
    TensorFlow, which has no wheel for Python 3.14; E4 skips it gracefully.)
*fuzzy*
    Order-free fuzzy occupancy (a stand-in for fuzzy static schemes) and a hybrid
    of aggregate features with fuzzy memberships. Both are **our own
    constructions** over our feature set, reported as ablations; neither is an
    implementation of a published method, and they must not be read as
    comparisons against published results.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pglts.data.polliradar import TAXONOMY
from pglts.linguistic.descriptors import DescriptorBank, static_statistics
from pglts.linguistic.semantize import make_semantizer


def _static(X: np.ndarray) -> np.ndarray:
    return static_statistics(X)[0]


# ---------------------------------------------------------------- numeric static
def numeric_static(model: str = "extratrees", seed: int = 0):
    """``fit_predict`` for an order-free numeric baseline."""

    def build():
        if model == "extratrees":
            return ExtraTreesClassifier(
                n_estimators=300, random_state=seed, n_jobs=4, class_weight="balanced"
            )
        if model == "randomforest":
            return RandomForestClassifier(
                n_estimators=300, random_state=seed, n_jobs=4, class_weight="balanced"
            )
        if model == "logreg":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced"),
            )
        if model == "svm":
            return make_pipeline(
                StandardScaler(), SVC(C=2.0, gamma="scale", class_weight="balanced")
            )
        if model == "catboost":
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=300, depth=6, verbose=0, random_seed=seed, auto_class_weights="Balanced"
            )
        raise ValueError(f"unknown numeric model: {model!r}")

    def fit_predict(X_train, y_train, X_test):
        clf = build()
        A, B = _static(X_train), _static(X_test)
        clf.fit(A, y_train)
        return clf.predict(B), {"n_descriptors": A.shape[1], "n_active": A.shape[1]}

    return fit_predict


def hierarchical_catboost(seed: int = 0):
    """The source paper's cascade: family -> genus -> species.

    Stage 1 (CatBoost) separates Apidae from Vespidae; Vespidae is finalised as
    *V. vulgaris*. Stage 2 (CatBoost) splits Apis from Bombus inside Apidae, and
    Apis is finalised as *A. mellifera*. Stage 3 (ExtraTrees) resolves the three
    *Bombus* species.
    """
    from catboost import CatBoostClassifier

    def _cat():
        return CatBoostClassifier(
            iterations=300, depth=6, verbose=0, random_seed=seed, auto_class_weights="Balanced"
        )

    def fit_predict(X_train, y_train, X_test):
        A, B = _static(X_train), _static(X_test)
        fam = np.array([TAXONOMY[s]["family"] for s in y_train])
        gen = np.array([TAXONOMY[s]["genus"] for s in y_train])

        stage1 = _cat().fit(A, fam)
        pred_fam = stage1.predict(B).ravel()
        out = np.array(["vespula_vulgaris"] * len(B), dtype=object)

        apidae = pred_fam == "Apidae"
        if apidae.any() and (fam == "Apidae").sum() > 1:
            m = fam == "Apidae"
            stage2 = _cat().fit(A[m], gen[m])
            pred_gen = stage2.predict(B[apidae]).ravel()
            sub = np.array(["apis_mellifera"] * apidae.sum(), dtype=object)

            bombus = pred_gen == "Bombus"
            mb = gen == "Bombus"
            if bombus.any() and np.unique(y_train[mb]).size > 1:
                stage3 = ExtraTreesClassifier(
                    n_estimators=300, random_state=seed, n_jobs=4, class_weight="balanced"
                ).fit(A[mb], y_train[mb])
                sub[bombus] = stage3.predict(B[apidae][bombus])
            out[apidae] = sub

        return out.astype(str), {"n_descriptors": A.shape[1], "n_active": A.shape[1]}

    return fit_predict


# -------------------------------------------------------------- numeric temporal
def aeon_temporal(model: str = "rocket", seed: int = 0):
    """ROCKET or InceptionTime on the raw multivariate trajectory.

    ``aeon`` expects ``(n_cases, n_channels, n_timepoints)``, the transpose of our
    ``(n, time, features)`` layout.
    """

    def build():
        if model == "rocket":
            from aeon.classification.convolution_based import RocketClassifier

            return RocketClassifier(n_kernels=2000, random_state=seed)
        if model == "multirocket":
            from aeon.classification.convolution_based import MultiRocketClassifier

            return MultiRocketClassifier(n_kernels=2000, random_state=seed)
        if model == "drcif":
            from aeon.classification.interval_based import DrCIFClassifier

            return DrCIFClassifier(n_estimators=100, random_state=seed, n_jobs=4)
        if model == "rdst":
            from aeon.classification.shapelet_based import RDSTClassifier

            return RDSTClassifier(random_state=seed)
        if model == "inceptiontime":
            # Requires TensorFlow, which has no wheel for Python 3.14. Kept for
            # environments where it is installable; E4 skips it gracefully.
            from aeon.classification.deep_learning import InceptionTimeClassifier

            return InceptionTimeClassifier(n_epochs=40, batch_size=64, random_state=seed)
        raise ValueError(f"unknown aeon model: {model!r}")

    def fit_predict(X_train, y_train, X_test):
        A = np.asarray(X_train, dtype=float).transpose(0, 2, 1)
        B = np.asarray(X_test, dtype=float).transpose(0, 2, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = build().fit(A, y_train)
            pred = clf.predict(B)
        width = A.shape[1] * A.shape[2]
        return pred, {"n_descriptors": width, "n_active": width}

    return fit_predict


# ------------------------------------------------------------------------- fuzzy
def fuzzy_static(scheme: str = "quantile", n_states: int = 4, seed: int = 0):
    """Occupancy-only fuzzy representation — memberships without any ordering."""

    def fit_predict(X_train, y_train, X_test):
        sem = make_semantizer(scheme, n_states=n_states).fit(X_train)
        bank = DescriptorBank(families=("O",), feature_names=(), words=tuple(sem.words))
        A, _, _ = bank.transform(sem.transform(X_train))
        B, _, _ = bank.transform(sem.transform(X_test))
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ).fit(A, y_train)
        return clf.predict(B), {"n_descriptors": A.shape[1], "n_active": A.shape[1]}

    return fit_predict


def fuzzy_hybrid(n_states: int = 4, seed: int = 0):
    """Aggregate features concatenated with fuzzy memberships.

    Inspired by the *idea* of Baldán & Martínez (2024) — combine conventional time
    series features with fuzzy membership degrees — but this is **our own
    construction**, not their method: it uses our feature set and our granulation,
    not their feature catalogue. It is therefore an ablation of our pipeline and
    must not be read as a comparison against published PLAYTIME results.
    """

    def fit_predict(X_train, y_train, X_test):
        sem = make_semantizer("fcm", n_states=n_states).fit(X_train)
        bank = DescriptorBank(families=("O",), feature_names=(), words=tuple(sem.words))
        A = np.hstack([_static(X_train), bank.transform(sem.transform(X_train))[0]])
        B = np.hstack([_static(X_test), bank.transform(sem.transform(X_test))[0]])
        clf = ExtraTreesClassifier(
            n_estimators=300, random_state=seed, n_jobs=4, class_weight="balanced"
        ).fit(A, y_train)
        return clf.predict(B), {"n_descriptors": A.shape[1], "n_active": A.shape[1]}

    return fit_predict


#: Registry consumed by E4.
BASELINES: dict[str, callable] = {
    "static_logreg": lambda: numeric_static("logreg"),
    "static_svm": lambda: numeric_static("svm"),
    "static_randomforest": lambda: numeric_static("randomforest"),
    "static_extratrees": lambda: numeric_static("extratrees"),
    "static_catboost": lambda: numeric_static("catboost"),
    "hierarchical_catboost": hierarchical_catboost,
    "rocket": lambda: aeon_temporal("rocket"),
    "inceptiontime": lambda: aeon_temporal("inceptiontime"),
    "fuzzy_static": lambda: fuzzy_static("quantile"),
    "fuzzy_hybrid": fuzzy_hybrid,
}
