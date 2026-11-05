"""EFRS-LTS — Evolutionary Fuzzy Regime-Switching Linguistic Time Series.

Đồng tối ưu (co-optimization) TẤT CẢ tham số liên tục của FRS-LTS bằng một
metaheuristic (PSO/DE), kết nối di sản LTS-PSO (2022) và CO-LTS (2023):

    minimize_{θ, α, σ, τ, ρ}   MSE_insample( FRS-LTS(θ,α,σ,τ,ρ ; K) )

cho mỗi K trong tập ứng viên; chọn K cho MSE huấn luyện nhỏ nhất (kèm phạt độ
phức tạp tùy chọn). Khác với grid-search của CLTS-Auto, PSO dò không gian liên
tục → khai thác được vùng tham số mà lưới bỏ sót.

Biến quyết định (vector vị trí PSO):
    [theta, alpha, sigma, tau, rho]
    - theta, alpha ∈ (0.3, 0.7)   : fuzziness Hedge Algebra
    - sigma        ∈ (0.02, 1.0)  : băng kernel trong regime (khi local=kernel)
    - tau          ∈ (0.03, 0.6)  : băng bằng-chứng-cục-bộ π_k
    - rho          ∈ (1/K, 0.99)  : persistence ma trận chuyển

Reproducibility: mọi ngẫu nhiên đến từ `seed` (PSO + phân cụm).
"""

from __future__ import annotations

from dataclasses import dataclass

from lts.core.hedge_algebras import HAParams
from lts.metrics.measures import ForecastMetrics
from lts.optimization.pso import PSO, PSOConfig
from lts.clustering.regime_switching import FuzzyRegimeSwitchingLTS


@dataclass
class EFRSConfig:
    """Cấu hình đồng tối ưu EFRS-LTS."""

    k_candidates: tuple[int, ...] = (1, 2, 3, 4)
    local: str = "exact"
    method: str = "kmeans"
    scheme: str = "semantic"
    specificity: int = 2
    order: int = 3
    n_particles: int = 30
    max_iter: int = 40
    n_runs: int = 1
    complexity_penalty: float = 0.0   # phạt theo K (× MSE chuẩn hóa); 0 = tắt
    objective: str = "validation"     # "validation" (chống overfit) | "insample"
    val_frac: float = 0.75            # tỉ lệ inner-train; phần đuôi làm validation
    seed: int = 0


class EFRSLTSModel:
    """EFRS-LTS: FRS-LTS với đồng tối ưu tham số bằng PSO.

    Sau fit(): truy cập best_params, best_k, và uỷ quyền predict()/forecast_next()
    cho mô hình FRS-LTS tốt nhất.
    """

    def __init__(self, config: EFRSConfig | None = None) -> None:
        self.cfg = config or EFRSConfig()
        self._best_model: FuzzyRegimeSwitchingLTS | None = None
        self._best_k: int | None = None
        self._best_params: dict | None = None
        self._best_mse: float = float("inf")
        self._history: dict[int, float] = {}
        self._anchor_used: bool = False

    # ------------------------------------------------------------------
    def _make_model(self, theta, alpha, sigma, tau, rho, K) -> FuzzyRegimeSwitchingLTS:
        c = self.cfg
        return FuzzyRegimeSwitchingLTS(
            HAParams(theta=theta, alpha=alpha),
            n_clusters=K, tau=tau, rho=rho, local=c.local, sigma=sigma,
            method=c.method, scheme=c.scheme, specificity=c.specificity,
            order=c.order, seed=c.seed,
        )

    @staticmethod
    def _mse(actual: list[float], pred: list[float]) -> float:
        return sum((a - p) ** 2 for a, p in zip(actual, pred)) / len(actual)

    def _objective_mse(self, model, data, lb, ub) -> float:
        """Hàm mục tiêu PSO: validation (chống overfit) hoặc in-sample."""
        try:
            if self.cfg.objective == "insample":
                model.fit(data, lb, ub)
                return self._mse(data[self.cfg.order:], model.predict())
            # validation: fit inner-train, đánh giá trên inner-val (không rò rỉ)
            n = len(data)
            n_in = max(int(n * self.cfg.val_frac), self.cfg.order + 2)
            if n_in >= n - 1:
                model.fit(data, lb, ub)
                return self._mse(data[self.cfg.order:], model.predict())
            model.fit(data[:n_in], lb, ub)
            preds = model.forecast_external(data)          # 1-bước trên toàn chuỗi
            # chỉ tính lỗi trên đoạn validation (đuôi)
            val_idx = range(n_in - self.cfg.order, len(preds))
            ap = [(data[self.cfg.order + i], preds[i]) for i in val_idx if i >= 0]
            if not ap:
                return float("inf")
            return self._mse([a for a, _ in ap], [p for _, p in ap])
        except Exception:
            return float("inf")

    def fit(self, data: list[float], lb: float, ub: float) -> None:
        c = self.cfg
        # Chuẩn hóa MSE để phạt độ phức tạp có ý nghĩa giữa dataset
        scale = max((max(data) - min(data)) ** 2, 1e-9)

        for K in c.k_candidates:
            rho_lo = 1.0 / K + 1e-3 if K > 1 else 0.0
            bounds = [
                (0.3, 0.7),       # theta
                (0.3, 0.7),       # alpha
                (0.02, 1.0),      # sigma
                (0.03, 0.6),      # tau
                (max(rho_lo, 0.2), 0.99) if K > 1 else (0.0, 1.0),  # rho
            ]

            def objective(pos: list[float], K=K) -> float:
                theta, alpha, sigma, tau, rho = pos
                m = self._make_model(theta, alpha, sigma, tau, rho, K)
                mse = self._objective_mse(m, data, lb, ub)
                if c.complexity_penalty and mse < float("inf"):
                    mse += c.complexity_penalty * scale * (K - 1)
                return mse

            best_pos, best_val = None, float("inf")
            for run in range(c.n_runs):
                pso_cfg = PSOConfig(
                    n_particles=c.n_particles, max_iter=c.max_iter,
                    omega=0.4, c1=2.0, c2=2.0, bounds=bounds,
                    seed=c.seed + run,
                )
                pos, val = PSO(objective, pso_cfg).run()
                if val < best_val:
                    best_val, best_pos = val, pos

            # Cũng xét biến thể TĨNH (ρ=1/K, ma trận chuyển đều = hỗn hợp memoryless)
            # với các tham số liên tục đã tối ưu → đảm bảo chọn được cấu hình tĩnh
            # khi động lực chuyển không có lợi (sửa lỗi chọn sai của PSO trên fold nhỏ).
            if K > 1 and best_pos is not None:
                static_pos = list(best_pos)
                static_pos[4] = 1.0 / K
                static_val = objective(static_pos, K=K)
                if static_val < best_val:
                    best_val, best_pos = static_val, static_pos

            self._history[K] = best_val
            if best_val < self._best_mse:
                self._best_mse = best_val
                self._best_k = K
                self._best_params = dict(
                    theta=best_pos[0], alpha=best_pos[1], sigma=best_pos[2],
                    tau=best_pos[3], rho=best_pos[4],
                )

        # Neo an toàn: LTS tham số mặc định. Nếu trên validation nó không thua
        # cấu hình PSO tốt nhất, dùng nó → EFRS-Auto không bao giờ tệ hơn LTS gốc
        # vì lý do overfit của PSO trên fold nhỏ.
        anchor = self._make_model(0.527, 0.412, 0.1, 0.15, 0.8, 1)
        anchor_val = self._objective_mse(anchor, data, lb, ub)
        self._history[0] = anchor_val
        self._anchor_used = anchor_val <= self._best_mse
        if self._anchor_used:
            self._best_mse = anchor_val
            self._best_k = 1
            self._best_params = dict(theta=0.527, alpha=0.412, sigma=0.1,
                                     tau=0.15, rho=0.8)

        # Refit mô hình tốt nhất
        p = self._best_params
        self._best_model = self._make_model(
            p["theta"], p["alpha"], p["sigma"], p["tau"], p["rho"], self._best_k)
        self._best_model.fit(data, lb, ub)

    # ------------------------------------------------------------------
    def predict(self) -> list[float]:
        return self._best_model.predict()

    def forecast_next(self) -> float:
        return self._best_model.forecast_next()

    def get_result(self):
        return self._best_model.get_result()

    @property
    def best_k(self) -> int | None:
        return self._best_k

    @property
    def best_params(self) -> dict | None:
        return self._best_params

    @property
    def best_mse(self) -> float:
        return self._best_mse

    @property
    def history(self) -> dict[int, float]:
        """MSE huấn luyện tốt nhất theo từng K (cho phân tích chọn mô hình)."""
        return dict(self._history)
