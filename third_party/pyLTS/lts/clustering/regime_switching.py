"""Fuzzy Regime-Switching LTS (FRS-LTS) — regime ngữ nghĩa có động lực thời gian.

Hạn chế của CLTS tĩnh: mỗi cửa sổ được gán regime ĐỘC LẬP (memoryless), bỏ qua
tính liên tục thời gian của regime — dù chính số liệu chẩn đoán cho thấy regime
rất "dính" (persistence 0.84–0.94 trên đường chéo ma trận chuyển).

FRS-LTS mô hình hóa tường minh động lực này bằng một **bộ lọc tiến (forward
filter) kiểu HMM mềm** trên không gian hedge-algebra: membership regime tại thời
điểm t phụ thuộc cả (i) bằng chứng cục bộ của cửa sổ hiện tại và (ii) membership
quá khứ truyền qua một **ma trận chuyển regime mờ** A:

    u_t(k)  ∝  π_k(z_t) · Σ_j A[j,k] · u_{t-1}(j)            (lọc nhân quả)
    ŷ_t     =  Σ_k u_t(k) · f(lhs_t ; R_k)                    (mixture-of-experts)

trong đó:
    π_k(z)  = exp(-‖z - c_k‖² / 2τ²)   — bằng chứng cục bộ (centroid c_k, băng τ)
    A[j,k]  = ρ nếu j=k, ngược lại (1-ρ)/(K-1)  — ma trận "sticky", persistence ρ
    f(·;R_k)= bộ dự báo trong regime k (exact hoặc kernel), tái dùng từ CLTS.

Các expert (R_k) lấy từ phân cụm tĩnh ban đầu (CLTSModel); A và τ điều khiển cách
trộn expert theo thời gian. Tất cả tham số liên tục (σ, τ, ρ) + (θ, α) sẽ được
đồng tối ưu trong EFRS-LTS (xem efrs_lts.py).

Tính chất hợp nhất (mở rộng CLTS):
    ρ → 1/K  (A đều)     ⇒ FRS-LTS thoái biến về CLTS soft tĩnh (memoryless).
    K = 1                ⇒ thoái biến về LTS/SW-LTS toàn cục.
    ρ → 1                ⇒ regime bị "khóa" (một expert thống trị toàn chuỗi).
"""

from __future__ import annotations

import numpy as np

from lts.core.hedge_algebras import HAParams
from lts.clustering.clts_model import CLTSModel


class FuzzyRegimeSwitchingLTS(CLTSModel):
    """LTS với regime-switching mờ có động lực thời gian.

    Parameters
    ----------
    params : HAParams
        Tham số fuzziness (theta, alpha).
    n_clusters : int
        Số regime K (>=1). Phân cụm tĩnh khởi tạo các expert.
    tau : float
        Băng thông bằng-chứng-cục-bộ π_k(z) trong không gian đặc trưng [0,1].
    rho : float
        Persistence của ma trận chuyển (∈ (1/K, 1)). rho lớn → regime dính.
    local : str
        Bộ tổng hợp trong regime: "exact" | "kernel".
    sigma : float | None
        Băng kernel (khi local="kernel").
    method, scheme, specificity, order, use_repeat, normalize, seed,
    min_regime_rules : truyền xuống CLTSModel.
    """

    def __init__(
        self,
        params: HAParams,
        n_clusters: int = 3,
        tau: float = 0.15,
        rho: float = 0.8,
        local: str = "exact",
        sigma: float | None = None,
        method: str = "kmeans",
        scheme: str = "semantic",
        specificity: int = 2,
        order: int = 3,
        use_repeat: bool = False,
        normalize: bool = True,
        seed: int = 0,
        min_regime_rules: int = 2,
    ) -> None:
        super().__init__(
            params=params,
            n_clusters=n_clusters,
            method=method,
            assignment="soft",      # FRS dùng membership mềm theo thời gian
            local=local,
            sigma=sigma,
            scheme=scheme,
            specificity=specificity,
            order=order,
            use_repeat=use_repeat,
            normalize=normalize,
            seed=seed,
            min_regime_rules=min_regime_rules,
        )
        self.tau = tau
        self.rho = rho

        self._centroids: np.ndarray | None = None
        self._A: np.ndarray | None = None
        self._u_last: np.ndarray | None = None  # membership cuối chuỗi train

    # ------------------------------------------------------------------
    def _build_transition(self, K: int) -> np.ndarray:
        """Ma trận chuyển 'sticky' row-stochastic từ persistence rho."""
        if K <= 1:
            return np.ones((1, 1))
        off = (1.0 - self.rho) / (K - 1)
        A = np.full((K, K), off)
        np.fill_diagonal(A, self.rho)
        return A

    def _local_evidence(self, Z: np.ndarray) -> np.ndarray:
        """π_k(z) cho từng cửa sổ: (n, K) — Gaussian theo khoảng cách centroid."""
        if self._centroids is None:
            return np.ones((len(Z), 1))
        diff = Z[:, None, :] - self._centroids[None, :, :]
        sq = (diff ** 2).sum(axis=2)
        ev = np.exp(-sq / (2.0 * max(self.tau, 1e-6) ** 2))
        return np.clip(ev, 1e-300, None)

    def _forward_filter(self, evidence: np.ndarray) -> np.ndarray:
        """Lọc tiến nhân quả → membership (n, K) chuẩn hóa theo hàng."""
        n, K = evidence.shape
        A = self._A
        U = np.empty((n, K))
        # t=0: chỉ bằng chứng cục bộ
        u = evidence[0] / evidence[0].sum()
        U[0] = u
        for t in range(1, n):
            prior = A.T @ u                  # Σ_j A[j,k] u_{t-1}(j)
            u = evidence[t] * prior
            s = u.sum()
            u = u / s if s > 1e-300 else np.full(K, 1.0 / K)
            U[t] = u
        return U

    # ------------------------------------------------------------------
    def fit(self, data: list[float], lb: float, ub: float) -> None:
        # CLTSModel.fit: fuzzify, LLRG toàn cục, phân cụm, expert R_k.
        super().fit(data, lb, ub)

        windows = self._train_windows()
        Z = self._repr.transform(windows) if self._repr is not None else np.empty((0, 0))
        K = self.effective_k

        if self._clusterer is not None and self._clusterer.centers is not None and K > 1:
            self._centroids = np.asarray(self._clusterer.centers, dtype=float)
        else:
            self._centroids = None
            K = 1

        self._A = self._build_transition(K)

        if len(Z) and self._centroids is not None:
            evidence = self._local_evidence(Z)
            U = self._forward_filter(evidence)
            self._u_last = U[-1]
            # Dự báo lại in-sample bằng mixture có động lực thời gian
            self._forecasted = [
                self._mix_forecast(windows[t], U[t]) for t in range(len(windows))
            ]
        else:
            self._u_last = np.ones(1)
            # K=1: giữ nguyên dự báo toàn cục của CLTS/LTS

    def _mix_forecast(self, lhs: tuple[str, ...], u: np.ndarray) -> float:
        """Trộn expert theo membership u (đã chuẩn hóa)."""
        out, wtot = 0.0, 0.0
        for k, uk in enumerate(u):
            if uk < 1e-6:
                continue
            fk = self._forecast_in_rules(lhs, self._effective_rules(k))
            out += uk * fk
            wtot += uk
        if wtot < 1e-300:
            return self._forecast_in_rules(lhs, self._rules)
        return out / wtot

    def forecast_next(self) -> float:
        """Dự báo 1 bước: truyền membership cuối qua A rồi cập nhật bằng chứng."""
        lhs_next = tuple(self._labels[-self.order:])
        if self._centroids is None or self._repr is None or self._u_last is None or self.effective_k <= 1:
            return self._forecast_in_rules(lhs_next, self._regime_rules[0])
        z = self._repr.transform_one(lhs_next).reshape(1, -1)
        ev = self._local_evidence(z)[0]
        prior = self._A.T @ self._u_last
        u = ev * prior
        s = u.sum()
        u = u / s if s > 1e-300 else np.full(len(u), 1.0 / len(u))
        return self._mix_forecast(lhs_next, u)

    # ------------------------------------------------------------------
    # Định lượng bất định: phương sai hỗn hợp regime (cho prediction interval)
    # ------------------------------------------------------------------
    def _regime_mean_spread(
        self, lhs: tuple[str, ...], rules: dict[tuple[str, ...], list[str]]
    ) -> tuple[float, float]:
        """Dự báo expert f_k và độ tản s_k (std hệ quả) trong một regime."""
        sp = self._semantic_points
        if self.local == "exact" and lhs in rules:
            vals = [sp[w] for w in rules[lhs]]
            m = sum(vals) / len(vals)
            s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            return m, s
        # fallback / kernel: lấy điểm dự báo, độ tản = 0 (sẽ được conformal bù lại)
        return self._forecast_in_rules(lhs, rules), 0.0

    def _mix_mean_std(self, lhs: tuple[str, ...], u: np.ndarray) -> tuple[float, float]:
        """Trung bình & độ lệch chuẩn của phân phối dự báo hỗn hợp regime.

        Dùng luật phương sai toàn phần: Var = E[Var trong regime] + Var[E giữa regime].
        """
        means, spreads, weights = [], [], []
        for k, uk in enumerate(u):
            if uk < 1e-6:
                continue
            m_k, s_k = self._regime_mean_spread(lhs, self._effective_rules(k))
            means.append(m_k); spreads.append(s_k); weights.append(uk)
        if not weights:
            return self._forecast_in_rules(lhs, self._rules), 0.0
        w = np.asarray(weights); w = w / w.sum()
        means = np.asarray(means); spreads = np.asarray(spreads)
        mean = float((w * means).sum())
        var = float((w * (spreads ** 2 + means ** 2)).sum() - mean ** 2)
        return mean, float(np.sqrt(max(var, 0.0)))

    def forecast_next_sigma(self) -> tuple[float, float]:
        """Dự báo 1 bước kèm độ bất định (mean, std hỗn hợp regime)."""
        lhs_next = tuple(self._labels[-self.order:])
        if self._centroids is None or self._repr is None or self._u_last is None or self.effective_k <= 1:
            m, s = self._regime_mean_spread(lhs_next, self._regime_rules[0])
            return m, s
        z = self._repr.transform_one(lhs_next).reshape(1, -1)
        ev = self._local_evidence(z)[0]
        u = ev * (self._A.T @ self._u_last)
        ssum = u.sum()
        u = u / ssum if ssum > 1e-300 else np.full(len(u), 1.0 / len(u))
        return self._mix_mean_std(lhs_next, u)

    def forecast_horizon(self, h: int) -> list[float]:
        """Dự báo h bước iterated, TRUYỀN membership regime qua A mỗi bước.

        Đây là nơi switching phát huy: khi đã xác định regime, mô hình lan truyền
        động lực của regime đó về tương lai; mô hình memoryless (ρ=1/K) mất dấu
        regime ngay khi không còn bằng chứng tức thời.
        """
        if self._centroids is None or self._repr is None or self._u_last is None or self.effective_k <= 1:
            return super().forecast_horizon(h)
        labels = list(self._labels)
        u = self._u_last.copy()
        out = []
        for _ in range(h):
            lhs = tuple(labels[-self.order:])
            z = self._repr.transform_one(lhs)
            ev = self._local_evidence(z.reshape(1, -1))[0]
            u = ev * (self._A.T @ u)
            s = u.sum()
            u = u / s if s > 1e-300 else np.full(len(u), 1.0 / len(u))
            yhat = self._mix_forecast(lhs, u)
            out.append(yhat)
            labels.append(self._fuzzify(yhat, self._semantic_points))
        return out

    def forecast_external(self, values: list[float]) -> list[float]:
        """Dự báo 1-bước cho chuỗi ngoài bằng mô hình đã fit (lọc tiến cố định).

        Fuzzify `values`, dựng cửa sổ LHS, chạy forward-filter (centroid/A cố
        định) để lấy membership theo thời gian, rồi trộn expert. Không refit,
        không dùng nhãn của `values` để dựng lại luật → dùng được cho validation
        không rò rỉ.
        """
        labels = self._label_data(list(values), self._semantic_points)
        windows = [
            tuple(labels[i - self.order : i]) for i in range(self.order, len(values))
        ]
        if self._centroids is None or self._repr is None or self.effective_k <= 1:
            return [
                self._forecast_in_rules(w, self._regime_rules[0]) for w in windows
            ]
        Z = self._repr.transform(windows)
        U = self._forward_filter(self._local_evidence(Z))
        return [self._mix_forecast(windows[t], U[t]) for t in range(len(windows))]

    # ------------------------------------------------------------------
    @property
    def transition_matrix(self) -> np.ndarray | None:
        return self._A

    @property
    def last_membership(self) -> np.ndarray | None:
        return self._u_last
