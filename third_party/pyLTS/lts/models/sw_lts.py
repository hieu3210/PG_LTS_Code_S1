"""SW-LTS: Similarity-Weighted Linguistic Time Series Forecasting.

Hướng nghiên cứu 1 — xem plan.md.

Thay thế exact-match lookup bằng kernel-weighted aggregation trên toàn bộ LLRG,
sử dụng độ tương tự Gaussian trong không gian semantic của Hedge Algebras:

    forecast(lhs_t) = Σ_k sim(lhs_t, lhs_k) × mean_s(RHS_k)
                      ──────────────────────────────────────────
                              Σ_k sim(lhs_t, lhs_k)

    sim(u, v) = exp( -||sp_norm(u) - sp_norm(v)||² / (2σ²) )

Khi normalize=True (mặc định): sp_norm = (sp - lb) / (ub - lb) ∈ [0, 1]
→ σ có ý nghĩa nhất quán giữa các dataset, không phụ thuộc đơn vị đo.

Tính chất:
    σ → 0 : degenerates về LTS gốc (exact-match)
    σ → ∞ : degenerates về mean of all RHS (HO-LTS fallback toàn cục)

SW-PSO-LTS: mở rộng tối ưu (θ, α, σ) đồng thời bằng PSO.
"""

from __future__ import annotations

import math

from lts.core.hedge_algebras import HAParams
from lts.metrics.measures import ForecastMetrics
from lts.models.lts_model import LTSModel
from lts.optimization.pso import PSO, PSOConfig


def _lhs_vector(
    lhs: tuple[str, ...], semantic_points: dict[str, float]
) -> list[float]:
    """Chuyển LHS tuple → vector semantic points."""
    return [semantic_points[w] for w in lhs]


def _sq_dist(u: list[float], v: list[float]) -> float:
    """Bình phương khoảng cách Euclidean giữa hai vector."""
    return sum((a - b) ** 2 for a, b in zip(u, v))


class SWLTSModel(LTSModel):
    """Similarity-Weighted LTS (SW-LTS).

    Parameters
    ----------
    params : HAParams
        Tham số fuzziness (theta, alpha).
    sigma : float | None
        Bandwidth của kernel trong không gian chuẩn hóa (khi normalize=True)
        hoặc không gian gốc (khi normalize=False).
        Nếu None: tự động tính bằng median pairwise distance.
    specificity : int
        Mức từ vựng: 1→7, 2→15, 3→31, 4→63 từ.
    order : int
        Bậc mô hình.
    use_repeat : bool
        Cho phép RHS trùng lặp trong LLRG.
    words : list[str] | None
        Từ vựng tường minh.
    normalize : bool
        True (mặc định): chuẩn hóa semantic points về [0, 1] trước khi tính kernel.
    kernel : str
        Loại kernel: "gaussian" (mặc định) hoặc "triangular".
        - gaussian: K = exp(-d²/2σ²)
        - triangular: K = max(0, 1 - d/σ)
    use_ha_distance : bool
        True (mặc định): dùng HA semantic points làm không gian khoảng cách.
        False: dùng uniform rank-based spacing (ablation — loại bỏ cấu trúc HA).
    """

    def __init__(
        self,
        params: HAParams,
        sigma: float | None = None,
        specificity: int = 1,
        order: int = 1,
        use_repeat: bool = False,
        words: list[str] | None = None,
        normalize: bool = True,
        kernel: str = "gaussian",
        use_ha_distance: bool = True,
    ) -> None:
        super().__init__(
            params=params,
            specificity=specificity,
            order=order,
            use_repeat=use_repeat,
            words=words,
        )
        self._sigma_init = sigma
        self._sigma: float = sigma if sigma is not None else 0.1
        self._normalize = normalize
        self._kernel = kernel
        self._use_ha_distance = use_ha_distance

    def fit(self, data: list[float], lb: float, ub: float) -> None:
        super().fit(data, lb, ub)
        if self._sigma_init is None:
            if self._normalize:
                span = ub - lb
                if span > 0:
                    sp_norm_vals = [
                        (v - lb) / span for v in self._semantic_points.values()
                    ]
                else:
                    sp_norm_vals = list(self._semantic_points.values())
            else:
                sp_norm_vals = list(self._semantic_points.values())

            dists = sorted(
                abs(sp_norm_vals[i] - sp_norm_vals[j])
                for i in range(len(sp_norm_vals))
                for j in range(i + 1, len(sp_norm_vals))
            )
            if dists:
                self._sigma = dists[len(dists) // 2]
            elif self._normalize:
                self._sigma = 0.1
            else:
                self._sigma = (ub - lb) / 10.0

    def _build_distance_space(
        self, semantic_points: dict[str, float]
    ) -> dict[str, float]:
        """Trả về không gian khoảng cách để tính kernel."""
        if not self._use_ha_distance:
            # Uniform rank-based spacing: loại bỏ cấu trúc HA (ablation)
            sorted_words = sorted(semantic_points.keys(), key=lambda w: semantic_points[w])
            n = len(sorted_words)
            return {w: i / (n - 1) if n > 1 else 0.5 for i, w in enumerate(sorted_words)}

        if self._normalize:
            span = self._ub - self._lb
            if span > 0:
                return {w: (v - self._lb) / span for w, v in semantic_points.items()}
        return dict(semantic_points)

    def _kernel_weight(self, sq_d: float) -> float:
        """Tính trọng số kernel từ bình phương khoảng cách."""
        if self._kernel == "triangular":
            d = math.sqrt(sq_d)
            return max(0.0, 1.0 - d / self._sigma)
        # gaussian (default)
        two_sigma_sq = 2.0 * self._sigma * self._sigma
        return math.exp(-sq_d / two_sigma_sq)

    def _forecast_one(
        self,
        lhs: tuple[str, ...],
        rules: dict[tuple[str, ...], list[str]],
        semantic_points: dict[str, float],
    ) -> float:
        """Kernel-weighted aggregation trên toàn bộ LLRG."""
        if not rules:
            return semantic_points[lhs[-1]]

        sp_kernel = self._build_distance_space(semantic_points)
        q_vec = _lhs_vector(lhs, sp_kernel)

        weighted_sum = 0.0
        weight_total = 0.0

        for lhs_k, rhs_list in rules.items():
            k_vec = _lhs_vector(lhs_k, sp_kernel)
            sq_d = _sq_dist(q_vec, k_vec)
            w = self._kernel_weight(sq_d)

            # RHS mean trong không gian gốc (không chuẩn hóa)
            mean_rhs = sum(semantic_points[r] for r in rhs_list) / len(rhs_list)
            weighted_sum += w * mean_rhs
            weight_total += w

        if weight_total < 1e-300:
            return semantic_points[lhs[-1]]

        return weighted_sum / weight_total

    @property
    def sigma(self) -> float:
        """Bandwidth đang dùng (sau khi fit)."""
        return self._sigma


class SWPSOLTSModel:
    """SW-PSO-LTS: Tối ưu đồng thời (θ, α, σ) bằng PSO.

    Wrapper xung quanh SWLTSModel — không kế thừa BaseForecaster trực tiếp
    vì cần lưu cả model tốt nhất.

    Parameters
    ----------
    specificity : int
        Mức từ vựng.
    order : int
        Bậc mô hình.
    pso_config : PSOConfig | None
        Cấu hình PSO. Mặc định: N=100, G_max=500.
    n_runs : int
        Số lần chạy PSO; lấy MSE nhỏ nhất.
    sigma_bounds : tuple[float, float]
        Khoảng tìm kiếm cho σ. Khi normalize=True: nên dùng (0.01, 1.0).
    normalize : bool
        Truyền thẳng xuống SWLTSModel.normalize.
    """

    def __init__(
        self,
        specificity: int = 1,
        order: int = 1,
        pso_config: PSOConfig | None = None,
        n_runs: int = 3,
        sigma_bounds: tuple[float, float] = (0.01, 1.0),
        normalize: bool = True,
    ) -> None:
        self.specificity = specificity
        self.order = order
        self.n_runs = n_runs
        self.sigma_bounds = sigma_bounds
        self.normalize = normalize
        self._pso_config = pso_config

        self._best_model: SWLTSModel | None = None
        self._best_params: tuple[float, float, float] | None = None  # (θ, α, σ)
        self._best_mse: float = float("inf")
        self._data: list[float] = []
        self._lb: float = 0.0
        self._ub: float = 1.0

    def fit(self, data: list[float], lb: float, ub: float) -> None:
        """Chạy PSO tối ưu (θ, α, σ), sau đó fit model tốt nhất."""
        self._data = list(data)
        self._lb = lb
        self._ub = ub

        cfg = self._pso_config or PSOConfig(
            n_particles=100,
            max_iter=500,
            omega=0.4,
            c1=2.0,
            c2=2.0,
            bounds=[
                (0.3, 0.7),
                (0.3, 0.7),
                (self.sigma_bounds[0], self.sigma_bounds[1]),
            ],
        )

        def objective(pos: list[float]) -> float:
            theta, alpha, sigma = pos
            try:
                params = HAParams(theta=theta, alpha=alpha)
                m = SWLTSModel(
                    params, sigma=sigma,
                    specificity=self.specificity, order=self.order,
                    normalize=self.normalize,
                )
                m.fit(data, lb, ub)
                actual = data[self.order:]
                return ForecastMetrics.compute(actual, m.predict()).mse
            except Exception:
                return float("inf")

        self._best_mse = float("inf")
        for run in range(self.n_runs):
            run_cfg = PSOConfig(
                n_particles=cfg.n_particles,
                max_iter=cfg.max_iter,
                omega=cfg.omega,
                c1=cfg.c1,
                c2=cfg.c2,
                bounds=cfg.bounds,
                seed=run,
            )
            pos, val = PSO(objective, run_cfg).run()
            if val < self._best_mse:
                self._best_mse = val
                self._best_params = (pos[0], pos[1], pos[2])

        theta, alpha, sigma = self._best_params
        best_params = HAParams(theta=theta, alpha=alpha)
        self._best_model = SWLTSModel(
            best_params, sigma=sigma,
            specificity=self.specificity, order=self.order,
            normalize=self.normalize,
        )
        self._best_model.fit(data, lb, ub)

    def predict(self) -> list[float]:
        return self._best_model.predict()

    def get_result(self):
        return self._best_model.get_result()

    @property
    def best_params(self) -> tuple[float, float, float] | None:
        """(theta, alpha, sigma) tối ưu."""
        return self._best_params

    @property
    def best_mse(self) -> float:
        return self._best_mse

    @property
    def sigma(self) -> float | None:
        return self._best_model.sigma if self._best_model else None
