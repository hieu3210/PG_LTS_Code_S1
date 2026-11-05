"""Online/Adaptive EFRS-LTS — phát hiện regime trực tuyến cho dữ liệu trôi khái niệm.

EFRS-LTS chuẩn cố định regime (centroid, cơ sở luật) từ fold huấn luyện. Với
luồng phi dừng / trôi khái niệm (concept drift), bản thân regime thay đổi theo
thời gian. OnlineEFRSLTS duy trì regime TRỰC TUYẾN:

    predict → observe → update     (mỗi bước O(1), không refit toàn bộ)

Cơ chế thích ứng:
    1. Cập nhật centroid kiểu EMA về phía cửa sổ mới của regime trúng (hệ số η).
    2. Bồi đắp luật: thêm LLR (lhs→rhs) mới vào cơ sở luật của regime trúng,
       có thể kèm lãng quên (giới hạn số RHS gần nhất) cho dữ liệu phi dừng.
    3. Phát hiện trôi: nếu độ mới (khoảng cách min tới mọi centroid) vượt ngưỡng
       drift_thresh kéo dài → sinh regime mới (nếu còn < K_max) để bắt chế độ mới.

Warm-start bằng một FuzzyRegimeSwitchingLTS đã fit (lấy HA semantic points,
centroid, ma trận chuyển, tham số). Sau đó chạy streaming.

Lợi ích: (i) chi phí mỗi bước O(K·λ) thay vì refit O(n); (ii) bám theo regime
mới xuất hiện sau huấn luyện.
"""

from __future__ import annotations

import numpy as np

from lts.core.hedge_algebras import HAParams
from lts.clustering.regime_switching import FuzzyRegimeSwitchingLTS


class OnlineEFRSLTS:
    """Bản trực tuyến, thích ứng của EFRS-LTS.

    Parameters
    ----------
    params : HAParams
    n_clusters : int          — số regime khởi tạo K0.
    tau, rho, sigma, local, scheme, specificity, order, seed, method :
        như FuzzyRegimeSwitchingLTS.
    eta : float               — tốc độ cập nhật centroid EMA (0..1).
    drift_thresh : float      — ngưỡng độ mới (trong không gian [0,1]) để xét trôi.
    drift_patience : int      — số bước liên tiếp vượt ngưỡng trước khi sinh regime.
    k_max : int               — trần số regime.
    rule_memory : int | None  — số RHS gần nhất giữ lại mỗi LLR (None = vô hạn).
    """

    def __init__(
        self,
        params: HAParams,
        n_clusters: int = 3,
        tau: float = 0.15,
        rho: float = 0.85,
        sigma: float | None = None,
        local: str = "exact",
        scheme: str = "semantic",
        specificity: int = 2,
        order: int = 3,
        seed: int = 0,
        method: str = "kmeans",
        eta: float = 0.05,
        drift_thresh: float = 0.25,
        drift_patience: int = 5,
        k_max: int = 8,
        rule_memory: int | None = None,
    ) -> None:
        self.params = params
        self.n_clusters = n_clusters
        self.tau, self.rho, self.sigma = tau, rho, sigma
        self.local, self.scheme = local, scheme
        self.specificity, self.order, self.seed, self.method = (
            specificity, order, seed, method)
        self.eta = eta
        self.drift_thresh = drift_thresh
        self.drift_patience = drift_patience
        self.k_max = k_max
        self.rule_memory = rule_memory

        self._base: FuzzyRegimeSwitchingLTS | None = None
        self._centroids: np.ndarray | None = None
        self._rules: list[dict] = []
        self._u: np.ndarray | None = None
        self._history: list[float] = []
        self._lb = 0.0
        self._ub = 1.0
        self._drift_count = 0
        self.n_spawned = 0
        self.drift_events: list[int] = []

    # ------------------------------------------------------------------
    def warm_start(self, data: list[float], lb: float, ub: float) -> "OnlineEFRSLTS":
        """Khởi tạo regime/luật từ FRS-LTS đã fit trên `data`."""
        m = FuzzyRegimeSwitchingLTS(
            self.params, n_clusters=self.n_clusters, tau=self.tau, rho=self.rho,
            sigma=self.sigma, local=self.local, scheme=self.scheme,
            specificity=self.specificity, order=self.order, seed=self.seed,
            method=self.method)
        m.fit(list(data), lb, ub)
        self._base = m
        self._lb, self._ub = lb, ub
        self._history = list(data)
        K = m.effective_k
        if m._centroids is not None and K > 1:
            self._centroids = np.asarray(m._centroids, dtype=float).copy()
            self._rules = [dict(r) for r in m._regime_rules]
            self._u = (m.last_membership.copy() if m.last_membership is not None
                       else np.full(K, 1.0 / K))
        else:
            # K=1: regime duy nhất, luật toàn cục
            self._centroids = None
            self._rules = [dict(m._rules)]
            self._u = np.ones(1)
        return self

    # ------------------------------------------------------------------
    def _label(self, value: float) -> str:
        return self._base._fuzzify(value, self._base._semantic_points)

    def _window_feature(self) -> tuple[tuple[str, ...], np.ndarray | None]:
        if len(self._history) < self.order:
            return tuple(), None
        lhs = tuple(self._label(v) for v in self._history[-self.order:])
        z = (self._base._repr.transform_one(lhs)
             if self._base._repr is not None else None)
        return lhs, z

    def _transition(self, K: int) -> np.ndarray:
        if K <= 1:
            return np.ones((1, 1))
        off = (1.0 - self.rho) / (K - 1)
        A = np.full((K, K), off)
        np.fill_diagonal(A, self.rho)
        return A

    def _evidence(self, z: np.ndarray) -> np.ndarray:
        diff = self._centroids - z[None, :]
        sq = (diff ** 2).sum(axis=1)
        return np.clip(np.exp(-sq / (2.0 * max(self.tau, 1e-6) ** 2)), 1e-300, None)

    # ------------------------------------------------------------------
    def forecast_next(self) -> float:
        """Dự báo điểm kế tiếp theo trạng thái regime hiện tại."""
        lhs, z = self._window_feature()
        if not lhs:
            return self._history[-1] if self._history else 0.0
        if self._centroids is None or z is None:
            return self._base._forecast_in_rules(lhs, self._rules[0])
        ev = self._evidence(z)
        u = ev * (self._transition(len(self._rules)).T @ self._u)
        s = u.sum()
        u = u / s if s > 1e-300 else np.full(len(u), 1.0 / len(u))
        out, wtot = 0.0, 0.0
        for k, uk in enumerate(u):
            if uk < 1e-6:
                continue
            rk = self._rules[k] if len(self._rules[k]) >= 2 else self._base._rules
            out += uk * self._base._forecast_in_rules(lhs, rk)
            wtot += uk
        return out / wtot if wtot > 1e-300 else self._base._forecast_in_rules(lhs, self._rules[0])

    # ------------------------------------------------------------------
    def update(self, value: float) -> None:
        """Quan sát giá trị thực mới → cập nhật regime, luật, membership, trôi."""
        lhs, z = self._window_feature()  # cửa sổ TRƯỚC khi thêm value
        self._history.append(float(value))
        if not lhs:
            return

        rhs = self._label(value)
        if self._centroids is None or z is None:
            self._append_rule(0, lhs, rhs)
            return

        ev = self._evidence(z)
        u = ev * (self._transition(len(self._rules)).T @ self._u)
        s = u.sum()
        u = u / s if s > 1e-300 else np.full(len(u), 1.0 / len(u))
        self._u = u
        k = int(np.argmax(u))

        # (1) cập nhật centroid EMA
        self._centroids[k] = (1 - self.eta) * self._centroids[k] + self.eta * z
        # (2) bồi đắp luật
        self._append_rule(k, lhs, rhs)
        # (3) phát hiện trôi
        novelty = float(np.sqrt(((self._centroids - z[None, :]) ** 2).sum(axis=1)).min())
        if novelty > self.drift_thresh:
            self._drift_count += 1
            if (self._drift_count >= self.drift_patience
                    and len(self._rules) < self.k_max):
                self._spawn_regime(z, lhs, rhs)
                self._drift_count = 0
        else:
            self._drift_count = max(0, self._drift_count - 1)

    def _append_rule(self, k: int, lhs: tuple[str, ...], rhs: str) -> None:
        rules = self._rules[k]
        lst = rules.setdefault(lhs, [])
        lst.append(rhs)
        if self.rule_memory is not None and len(lst) > self.rule_memory:
            del lst[: len(lst) - self.rule_memory]

    def _spawn_regime(self, z: np.ndarray, lhs: tuple[str, ...], rhs: str) -> None:
        """Sinh regime mới lấy cửa sổ trôi làm hạt giống centroid."""
        self._centroids = np.vstack([self._centroids, z[None, :]])
        self._rules.append({lhs: [rhs]})
        K = len(self._rules)
        self._u = np.append(self._u * 0.9, 0.1)
        self._u /= self._u.sum()
        self.n_spawned += 1
        self.drift_events.append(len(self._history))

    # ------------------------------------------------------------------
    @property
    def n_regimes(self) -> int:
        return len(self._rules)

    @property
    def centroids(self) -> np.ndarray | None:
        return self._centroids
