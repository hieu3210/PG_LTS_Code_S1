"""CLTS — Clustered Linguistic Time Series Forecasting.

Mở rộng lõi LTS/Hedge-Algebra bằng *semantic regime discovery*: thay vì một
cơ sở luật (LLRG) toàn cục, ta phát hiện K "regime" ngữ nghĩa bằng cách gom cụm
các cửa sổ trạng thái ngôn ngữ (LHS) trong không gian semantic của Hedge Algebra,
rồi dự báo *có điều kiện theo regime*.

Kiến trúc kế thừa nguyên vẹn lõi LTSModel (fuzzification, semantic points, LLRG),
chỉ thêm hai bước:

    (a) Representation : LHS-window → Z ∈ R^d   (lts.clustering.representation)
    (b) Clustering     : Z → regime c ∈ {0..K-1} (lts.clustering.algorithms)

Sau đó mỗi regime có cơ sở luật riêng (regime-specific LLRG). Dự báo:

    hard : định tuyến query tới đúng 1 regime, dùng LLRG của regime đó.
    soft : mixture-of-experts — trộn dự báo của các regime theo membership.

Bộ tổng hợp trong từng regime (`local`):
    "exact"  : trung bình RHS của LLRG khớp đúng (kiểu LTS gốc).
    "kernel" : trọng số Gaussian theo khoảng cách semantic (kiểu SW-LTS).

Thống nhất công trình trước như trường hợp đặc biệt:
    K=1, local="exact"  → LTS (2020)
    K=1, local="kernel" → SW-LTS
    K>1                 → CLTS (đề xuất)

Lưu ý reproducibility: toàn bộ ngẫu nhiên đến từ `seed` (clustering). Bước HA và
LLRG là tất định.
"""

from __future__ import annotations

import math

import numpy as np

from lts.core.hedge_algebras import HAParams
from lts.models.base import ForecastResult
from lts.models.lts_model import LTSModel
from lts.clustering.representation import LinguisticRepresentation
from lts.clustering.algorithms import ClusterModel, make_clusterer


class CLTSModel(LTSModel):
    """Clustered Linguistic Time Series forecaster.

    Parameters
    ----------
    params : HAParams
        Tham số fuzziness (theta, alpha) của Hedge Algebra.
    n_clusters : int | str
        Số regime K. K=1 thoái biến về LTS/SW-LTS toàn cục.
        "auto" → chọn K bằng silhouette trên cửa sổ huấn luyện (xem
        `auto_k_range`, `silhouette_floor`): nếu không có cấu trúc regime rõ
        ràng (silhouette < ngưỡng) thì tự lui về mô hình toàn cục (K=1).
        Đây là cơ chế bảo vệ: CLTS không tệ hơn baseline khi dữ liệu không có
        regime, nhưng tận dụng được khi có.
    method : str
        Thuật toán phân cụm: "kmeans" | "gmm" | "fcm".
    assignment : str
        "hard" (định tuyến 1 regime) hoặc "soft" (mixture-of-experts).
    local : str
        Bộ tổng hợp trong regime: "exact" (LTS) hoặc "kernel" (SW-LTS).
    sigma : float | None
        Bandwidth kernel (chỉ dùng khi local="kernel"). None → tự ước lượng
        bằng median pairwise distance trong không gian chuẩn hóa.
    scheme : str
        Chiến lược biểu diễn LHS (xem LinguisticRepresentation).
    specificity, order, use_repeat, words :
        Truyền thẳng xuống LTSModel.
    normalize : bool
        Chuẩn hóa semantic points về [0,1] cho representation & kernel.
    seed : int
        Random seed cho clustering (reproducibility).
    min_regime_rules : int
        Nếu một regime có ít hơn ngần này LLRG, dùng cơ sở luật toàn cục làm
        backoff (tránh overfit khi regime quá nhỏ).
    """

    def __init__(
        self,
        params: HAParams,
        n_clusters: int = 3,
        method: str = "kmeans",
        assignment: str = "soft",
        local: str = "kernel",
        sigma: float | None = None,
        scheme: str = "semantic",
        specificity: int = 1,
        order: int = 1,
        use_repeat: bool = False,
        words: list[str] | None = None,
        normalize: bool = True,
        seed: int = 0,
        min_regime_rules: int = 2,
        auto_k_range: tuple[int, int] = (2, 5),
        silhouette_floor: float = 0.55,
        cluster_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            params=params,
            specificity=specificity,
            order=order,
            use_repeat=use_repeat,
            words=words,
        )
        self.n_clusters = n_clusters
        self.method = method
        self.assignment = assignment
        self.local = local
        self._sigma_init = sigma
        self._sigma = sigma if sigma is not None else 0.1
        self.scheme = scheme
        self.normalize = normalize
        self.seed = seed
        self.min_regime_rules = min_regime_rules
        self.auto_k_range = auto_k_range
        self.silhouette_floor = silhouette_floor
        self._cluster_kwargs = cluster_kwargs or {}
        self._selected_k: int | None = None
        self._selection_scores: dict[int, float] = {}

        # Trạng thái thiết lập bởi fit()
        self._repr: LinguisticRepresentation | None = None
        self._clusterer: ClusterModel | None = None
        self._regime_rules: list[dict[tuple[str, ...], list[str]]] = []
        self._window_labels: np.ndarray | None = None  # regime của mỗi cửa sổ train
        self._sp_kernel: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_kernel_space(self) -> dict[str, float]:
        """Không gian khoảng cách [0,1] cho kernel (giống SW-LTS normalize)."""
        span = self._ub - self._lb
        if self.normalize and span > 0:
            return {w: (v - self._lb) / span for w, v in self._semantic_points.items()}
        return dict(self._semantic_points)

    def _estimate_sigma(self) -> float:
        vals = list(self._sp_kernel.values())
        dists = sorted(
            abs(vals[i] - vals[j])
            for i in range(len(vals))
            for j in range(i + 1, len(vals))
        )
        if dists:
            return dists[len(dists) // 2]
        return 0.1

    def _train_windows(self) -> list[tuple[str, ...]]:
        return [
            tuple(self._labels[i - self.order : i])
            for i in range(self.order, len(self._labels))
        ]

    def _build_rules_subset(
        self, indices: list[int]
    ) -> dict[tuple[str, ...], list[str]]:
        """Xây LLRG chỉ từ các cửa sổ train có chỉ số trong `indices`.

        Cửa sổ thứ j (0-based trong danh sách train) tương ứng nhãn dữ liệu
        tại vị trí (order + j): LHS = labels[j:j+order], RHS = labels[order+j].
        """
        rules: dict[tuple[str, ...], list[str]] = {}
        for j in indices:
            lhs = tuple(self._labels[j : j + self.order])
            rhs = self._labels[self.order + j]
            if lhs not in rules:
                rules[lhs] = [rhs]
            elif self.use_repeat:
                rules[lhs].append(rhs)
            elif rhs not in rules[lhs]:
                rules[lhs].append(rhs)
        return rules

    # ------------------------------------------------------------------
    # Bộ tổng hợp trong một regime
    # ------------------------------------------------------------------
    def _kernel_weight(self, sq_d: float) -> float:
        two_sigma_sq = 2.0 * self._sigma * self._sigma
        return math.exp(-sq_d / max(two_sigma_sq, 1e-300))

    def _forecast_in_rules(
        self, lhs: tuple[str, ...], rules: dict[tuple[str, ...], list[str]]
    ) -> float:
        """Dự báo dùng một cơ sở luật cụ thể (exact hoặc kernel)."""
        sp = self._semantic_points
        if not rules:
            return sp[lhs[-1]]

        if self.local == "exact":
            if lhs in rules:
                cons = rules[lhs]
                return sum(sp[w] for w in cons) / len(cons)
            # backoff: semantic point của từ cuối — khớp y hệt LTSModel gốc,
            # đảm bảo K=1 ⇒ CLTS-exact ≡ LTS (2020).
            return sp[lhs[-1]]

        # local == "kernel" : trọng số Gaussian trên toàn bộ LLRG của regime
        q = [self._sp_kernel[w] for w in lhs]
        wsum = 0.0
        wtot = 0.0
        for lhs_k, rhs_list in rules.items():
            kvec = [self._sp_kernel[w] for w in lhs_k]
            sq_d = sum((a - b) ** 2 for a, b in zip(q, kvec))
            w = self._kernel_weight(sq_d)
            mean_rhs = sum(sp[r] for r in rhs_list) / len(rhs_list)
            wsum += w * mean_rhs
            wtot += w
        if wtot < 1e-300:
            return sp[lhs[-1]]
        return wsum / wtot

    def _effective_rules(self, k: int) -> dict[tuple[str, ...], list[str]]:
        """Cơ sở luật của regime k, có backoff về toàn cục nếu quá nhỏ."""
        rk = self._regime_rules[k]
        if len(rk) < self.min_regime_rules:
            return self._rules  # backoff toàn cục
        return rk

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def fit(self, data: list[float], lb: float, ub: float) -> None:
        # Bước 1–5: lõi LTS (fuzzification + LLRG toàn cục)
        super().fit(data, lb, ub)

        self._sp_kernel = self._build_kernel_space()
        if self._sigma_init is None:
            self._sigma = self._estimate_sigma()

        windows = self._train_windows()

        # Representation: LHS-window → Z
        self._repr = LinguisticRepresentation(
            scheme=self.scheme, normalize=self.normalize
        ).fit(self._vocab, self._semantic_points, lb, ub)
        Z = self._repr.transform(windows)

        # Chọn số regime K
        if self.n_clusters == "auto":
            k_req = self._select_k_auto(Z)
        else:
            k_req = int(self.n_clusters)
        self._selected_k = k_req

        # Clustering: Z → regime
        n_distinct = len(set(map(tuple, Z.tolist()))) if len(Z) else 1
        eff_k = max(1, min(k_req, n_distinct))
        if eff_k <= 1 or len(Z) < 2:
            # Thoái biến về toàn cục: một regime duy nhất
            self._clusterer = None
            self._window_labels = np.zeros(len(windows), dtype=int)
            self._regime_rules = [dict(self._rules)]
        else:
            self._clusterer = make_clusterer(
                self.method, n_clusters=eff_k, seed=self.seed, **self._cluster_kwargs
            ).fit(Z)
            self._window_labels = self._clusterer.predict(Z)
            K = self._clusterer.n_clusters
            self._regime_rules = [
                self._build_rules_subset(
                    [j for j in range(len(windows)) if self._window_labels[j] == k]
                )
                for k in range(K)
            ]

        # Bước 6: dự báo lại có điều kiện theo regime
        self._forecasted = [
            self._forecast_regime(windows[j], Z[j] if len(Z) else None)
            for j in range(len(windows))
        ]

    def forecast_next(self) -> float:
        """Dự báo 1 bước ngoài mẫu có điều kiện theo regime."""
        lhs_next = tuple(self._labels[-self.order:])
        z = self._repr.transform_one(lhs_next) if self._repr is not None else None
        return self._forecast_regime(lhs_next, z)

    def _select_k_auto(self, Z: np.ndarray) -> int:
        """Chọn K theo silhouette trên cửa sổ train; lui về 1 nếu < ngưỡng.

        Hoàn toàn dựa trên đặc trưng huấn luyện (không rò rỉ test). Nếu cấu trúc
        regime không đủ rõ (max silhouette < silhouette_floor) → K=1 (toàn cục).
        """
        from lts.clustering.metrics import select_k

        lo, hi = self.auto_k_range
        best_k, scores = select_k(
            Z, self.method, range(lo, hi + 1), seed=self.seed
        )
        self._selection_scores = scores
        if not scores or scores.get(best_k, 0.0) < self.silhouette_floor:
            return 1
        return best_k

    def _forecast_regime(self, lhs: tuple[str, ...], z: np.ndarray | None) -> float:
        """Dự báo có điều kiện theo regime cho một cửa sổ LHS."""
        if self._clusterer is None or z is None:
            return self._forecast_in_rules(lhs, self._regime_rules[0])

        zz = z.reshape(1, -1)
        if self.assignment == "hard":
            k = int(self._clusterer.predict(zz)[0])
            return self._forecast_in_rules(lhs, self._effective_rules(k))

        # soft = mixture-of-experts
        proba = self._clusterer.predict_proba(zz)[0]  # (K,)
        out = 0.0
        wtot = 0.0
        for k, pk in enumerate(proba):
            if pk < 1e-6:
                continue
            fk = self._forecast_in_rules(lhs, self._effective_rules(k))
            out += pk * fk
            wtot += pk
        if wtot < 1e-300:
            return self._forecast_in_rules(lhs, self._rules)
        return out / wtot

    # ------------------------------------------------------------------
    # Thông tin regime (cho phân tích/đánh giá)
    # ------------------------------------------------------------------
    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def effective_k(self) -> int:
        """Số regime thực sự sử dụng sau fit()."""
        return len(self._regime_rules)

    @property
    def selected_k(self) -> int | None:
        """K được chọn (hữu ích khi n_clusters='auto')."""
        return self._selected_k

    @property
    def selection_scores(self) -> dict[int, float]:
        """Silhouette theo từng K khi chọn tự động (cho ablation/plot)."""
        return dict(self._selection_scores)

    @property
    def window_regimes(self) -> np.ndarray:
        """Nhãn regime của từng cửa sổ train (n_windows,)."""
        return self._window_labels if self._window_labels is not None else np.array([])

    @property
    def representation(self) -> LinguisticRepresentation | None:
        return self._repr

    @property
    def clusterer(self) -> ClusterModel | None:
        return self._clusterer

    def regime_features(self) -> np.ndarray:
        """Ma trận đặc trưng Z của các cửa sổ train (cho silhouette/plot)."""
        if self._repr is None:
            return np.empty((0, 0))
        return self._repr.transform(self._train_windows())

    def regime_rule_counts(self) -> list[int]:
        """Số LLRG trong mỗi regime."""
        return [len(r) for r in self._regime_rules]

    def get_result(self) -> ForecastResult:
        res = super().get_result()
        res.rules = {lhs: list(rhs) for lhs, rhs in self._rules.items()}
        return res
