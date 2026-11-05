"""Biểu diễn cửa sổ trạng thái ngôn ngữ (LHS) thành vector đặc trưng.

Mỗi bước dự báo trong LTS dựa trên một cửa sổ LHS gồm `order` nhãn ngôn ngữ:

    lhs_t = (L_{t-order}, ..., L_{t-1})

Để gom cụm các trạng thái thành "regime", ta cần ánh xạ mỗi LHS thành một
vector số học Z ∈ R^d. Module này cung cấp nhiều chiến lược biểu diễn, tất cả
đều dựa trên cấu trúc ngữ nghĩa sẵn có của Hedge Algebra (semantic points),
không thêm tham số học ngoài lõi LTS.

Chiến lược (`scheme`):
    "semantic"   — vector = semantic points đã chuẩn hóa của các từ trong LHS
                   (giữ thứ tự thời gian). d = order.
    "delta"      — "semantic" + các sai phân liên tiếp (xu hướng cục bộ).
                   d = order + (order-1).
    "stats"      — thống kê bất biến thứ tự: [mean, std, first, last, slope].
                   d = 5. Hữu ích khi muốn regime không phụ thuộc vị trí.
    "momentum"   — CHỈ các sai phân liên tiếp (bất biến mức/level). d = order-1.
                   Tách regime theo XU HƯỚNG/động lực (tăng vs giảm) bất kể đang ở
                   mức cao hay thấp → cần cho regime momentum (bull/bear) mà biểu
                   diễn theo level không phân biệt được.
    "onehot"     — one-hot trung bình của các từ (bag-of-words ngôn ngữ).
                   d = |vocab|. Loại bỏ khoảng cách ngữ nghĩa (ablation).

Tất cả vector đặc trưng được tính trong không gian CHUẨN HÓA [0,1] để khoảng
cách nhất quán giữa các dataset (giống SW-LTS normalize=True).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinguisticRepresentation:
    """Ánh xạ LHS-window → feature vector trong không gian ngữ nghĩa HA.

    Parameters
    ----------
    scheme : str
        Chiến lược biểu diễn: "semantic" | "delta" | "stats" | "onehot".
    normalize : bool
        Chuẩn hóa semantic points về [0, 1] trước khi tạo đặc trưng.
    """

    scheme: str = "semantic"
    normalize: bool = True

    # Trạng thái được thiết lập bởi fit()
    _vocab: list[str] | None = None
    _sp_norm: dict[str, float] | None = None
    _index: dict[str, int] | None = None

    def fit(
        self,
        vocab: list[str],
        semantic_points: dict[str, float],
        lb: float,
        ub: float,
    ) -> "LinguisticRepresentation":
        """Khởi tạo không gian ngữ nghĩa từ vocab + semantic points đã có.

        Parameters
        ----------
        vocab : list[str]
            Danh sách từ vựng (đã sắp theo SQM).
        semantic_points : dict[str, float]
            Ánh xạ word → semantic point trong [lb, ub].
        lb, ub : float
            Universe of discourse để chuẩn hóa.
        """
        span = ub - lb
        if self.normalize and span > 0:
            self._sp_norm = {w: (semantic_points[w] - lb) / span for w in vocab}
        else:
            self._sp_norm = {w: semantic_points[w] for w in vocab}
        self._vocab = list(vocab)
        self._index = {w: i for i, w in enumerate(vocab)}
        return self

    @property
    def dim(self) -> int:
        """Số chiều của vector đặc trưng cho `order` cho trước (gọi sau fit)."""
        raise NotImplementedError("dim phụ thuộc order — dùng transform().shape[1]")

    def _vec_one(self, lhs: tuple[str, ...]) -> np.ndarray:
        assert self._sp_norm is not None and self._index is not None
        sp = [self._sp_norm[w] for w in lhs]

        if self.scheme == "semantic":
            return np.asarray(sp, dtype=float)

        if self.scheme == "delta":
            deltas = [sp[i + 1] - sp[i] for i in range(len(sp) - 1)]
            return np.asarray(sp + deltas, dtype=float)

        if self.scheme == "momentum":
            # Chỉ sai phân → bất biến mức, tách regime theo xu hướng.
            if len(sp) < 2:
                return np.asarray([0.0], dtype=float)
            deltas = [sp[i + 1] - sp[i] for i in range(len(sp) - 1)]
            return np.asarray(deltas, dtype=float)

        if self.scheme == "stats":
            arr = np.asarray(sp, dtype=float)
            slope = (sp[-1] - sp[0]) / max(len(sp) - 1, 1)
            return np.asarray(
                [arr.mean(), arr.std(), sp[0], sp[-1], slope], dtype=float
            )

        if self.scheme == "onehot":
            vec = np.zeros(len(self._vocab), dtype=float)
            for w in lhs:
                vec[self._index[w]] += 1.0
            return vec / len(lhs)

        raise ValueError(f"scheme không hợp lệ: {self.scheme!r}")

    def transform(self, windows: list[tuple[str, ...]]) -> np.ndarray:
        """Chuyển danh sách LHS-window → ma trận đặc trưng (n_windows × d)."""
        if not windows:
            return np.empty((0, 0), dtype=float)
        return np.vstack([self._vec_one(w) for w in windows])

    def transform_one(self, lhs: tuple[str, ...]) -> np.ndarray:
        """Chuyển một LHS-window → vector đặc trưng (1D)."""
        return self._vec_one(lhs)
