"""Các thuật toán phân cụm với giao diện thống nhất (hard + soft assignment).

Mọi clusterer expose chung một interface để CLTSModel có thể hoán đổi tự do:

    fit(X)            -> self
    predict(X)        -> nhãn cứng (n,)            int
    predict_proba(X)  -> membership mềm (n, K)     float, mỗi hàng tổng = 1
    n_clusters        -> K
    centers           -> tâm cụm (K, d) hoặc None

KMeans/GMM bọc sklearn. FuzzyCMeans hiện thực thuần numpy (fuzzy c-means của
Bezdek) — được khuyến nghị trong claude.md vì membership mềm phù hợp với bản
chất "mờ" của biểu diễn ngôn ngữ.

Soft membership cho KMeans được suy ra bằng softmax trên khoảng cách âm
(giúp chế độ mixture-of-experts hoạt động với mọi backend).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-300, None)


class ClusterModel(ABC):
    """Giao diện trừu tượng cho mọi thuật toán phân cụm."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "ClusterModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    @property
    @abstractmethod
    def n_clusters(self) -> int: ...

    @property
    def centers(self) -> np.ndarray | None:
        return None


class KMeansCluster(ClusterModel):
    """Bọc sklearn.cluster.KMeans với soft membership softmax.

    Parameters
    ----------
    n_clusters : int
        Số regime K.
    temperature : float
        Nhiệt độ cho softmax membership (nhỏ → gần hard). Mặc định 0.1
        (theo thang khoảng cách trong không gian [0,1]).
    seed : int
        Random seed.
    """

    def __init__(self, n_clusters: int = 3, temperature: float = 0.1, seed: int = 0):
        self._k = n_clusters
        self._temperature = temperature
        self._seed = seed
        self._km = None

    def fit(self, X: np.ndarray) -> "KMeansCluster":
        from sklearn.cluster import KMeans

        k = min(self._k, len(X)) if len(X) else self._k
        self._km = KMeans(n_clusters=max(k, 1), n_init=10, random_state=self._seed)
        self._km.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._km.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d = self._km.transform(X)  # khoảng cách Euclid tới mỗi tâm
        return _softmax(-d / max(self._temperature, 1e-9), axis=1)

    @property
    def n_clusters(self) -> int:
        return self._km.n_clusters if self._km is not None else self._k

    @property
    def centers(self) -> np.ndarray | None:
        return None if self._km is None else self._km.cluster_centers_


class GMMCluster(ClusterModel):
    """Bọc sklearn.mixture.GaussianMixture (soft = posterior responsibilities)."""

    def __init__(self, n_clusters: int = 3, seed: int = 0, covariance_type: str = "full"):
        self._k = n_clusters
        self._seed = seed
        self._cov = covariance_type
        self._gmm = None

    def fit(self, X: np.ndarray) -> "GMMCluster":
        from sklearn.mixture import GaussianMixture

        k = min(self._k, len(X)) if len(X) else self._k
        cov = self._cov if len(X) > X.shape[1] + 1 else "diag"
        self._gmm = GaussianMixture(
            n_components=max(k, 1),
            covariance_type=cov,
            random_state=self._seed,
            reg_covar=1e-4,
        )
        self._gmm.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._gmm.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._gmm.predict_proba(X)

    @property
    def n_clusters(self) -> int:
        return self._gmm.n_components if self._gmm is not None else self._k

    @property
    def centers(self) -> np.ndarray | None:
        return None if self._gmm is None else self._gmm.means_


class FuzzyCMeans(ClusterModel):
    """Fuzzy c-means (Bezdek) — hiện thực thuần numpy.

    Membership mềm tự nhiên:
        u_ik = 1 / Σ_j (||x_i - c_k|| / ||x_i - c_j||)^{2/(m-1)}

    Parameters
    ----------
    n_clusters : int
        Số regime K.
    m : float
        Hệ số mờ (fuzzifier), m > 1. Mặc định 2.0.
    max_iter : int
        Số vòng lặp tối đa.
    tol : float
        Ngưỡng hội tụ trên thay đổi của ma trận membership.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 200,
        tol: float = 1e-5,
        seed: int = 0,
    ):
        self._k = n_clusters
        self._m = m
        self._max_iter = max_iter
        self._tol = tol
        self._seed = seed
        self._centers: np.ndarray | None = None

    def _dist(self, X: np.ndarray) -> np.ndarray:
        # (n, K) khoảng cách Euclid tới mỗi tâm
        diff = X[:, None, :] - self._centers[None, :, :]
        return np.sqrt(np.clip((diff ** 2).sum(axis=2), 1e-300, None))

    def _memberships(self, d: np.ndarray) -> np.ndarray:
        power = 2.0 / (self._m - 1.0)
        # u_ik = 1 / Σ_j (d_ik / d_ij)^power
        ratio = d[:, :, None] / np.clip(d[:, None, :], 1e-300, None)
        u = 1.0 / np.clip((ratio ** power).sum(axis=2), 1e-300, None)
        return u / np.clip(u.sum(axis=1, keepdims=True), 1e-300, None)

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        rng = np.random.default_rng(self._seed)
        n = len(X)
        k = max(min(self._k, n), 1)
        # Khởi tạo tâm bằng cách lấy mẫu k điểm dữ liệu
        idx = rng.choice(n, size=k, replace=(n < k))
        self._centers = X[idx].astype(float).copy()
        u_prev = None
        for _ in range(self._max_iter):
            d = self._dist(X)
            u = self._memberships(d)
            um = u ** self._m
            self._centers = (um.T @ X) / np.clip(um.sum(axis=0)[:, None], 1e-300, None)
            if u_prev is not None and np.max(np.abs(u - u_prev)) < self._tol:
                break
            u_prev = u
        self._k = k
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._memberships(self._dist(X))

    @property
    def n_clusters(self) -> int:
        return self._k

    @property
    def centers(self) -> np.ndarray | None:
        return self._centers


def make_clusterer(method: str, n_clusters: int, seed: int = 0, **kwargs) -> ClusterModel:
    """Factory: tên thuật toán → ClusterModel.

    method ∈ {"kmeans", "gmm", "fcm"/"fuzzy"}.
    """
    method = method.lower()
    if method == "kmeans":
        return KMeansCluster(n_clusters=n_clusters, seed=seed, **kwargs)
    if method == "gmm":
        return GMMCluster(n_clusters=n_clusters, seed=seed, **kwargs)
    if method in {"fcm", "fuzzy", "fuzzy_cmeans"}:
        return FuzzyCMeans(n_clusters=n_clusters, seed=seed, **kwargs)
    raise ValueError(f"Thuật toán phân cụm không hỗ trợ: {method!r}")
