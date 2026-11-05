"""Hiệu chuẩn conformal cho điểm số nguyên của HACP-LTS.

- :func:`conformal_quantile` — lượng tử split conformal hữu hạn mẫu.
- :class:`AdaptiveQuantile`  — ACI (Gibbs & Candès, 2021): chỉnh mức ``α_t`` trực
  tuyến, bảo đảm độ phủ dài hạn không cần giả thiết hoán đổi được.
"""

from __future__ import annotations

import math


def conformal_quantile(scores: list[float], alpha: float) -> float:
    """Phần tử nhỏ thứ ``⌈(n+1)(1−α)⌉`` của ``scores``; ``inf`` nếu chỉ số vượt ``n``.

    Đây là ngưỡng ``q̂`` cho bảo đảm ``P(s_{n+1} ≤ q̂) ≥ 1 − α`` khi các điểm số
    hoán đổi được (cho phép trùng giá trị).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha phải thuộc (0, 1).")
    n = len(scores)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return math.inf
    return sorted(scores)[rank - 1]


FULL_RANGE = 1e9  # ngưỡng "nhận mọi điểm số hữu hạn" → dải phủ toàn lưới [lb, ub]


def randomized_conformal_quantile(scores: list[float], alpha: float, u: float) -> float:
    """Ngưỡng ``q̃`` từ p-value conformal làm trơn (tie-breaking ngẫu nhiên ``u ~ U(0,1)``).

    Với điểm số nguyên, ``p(s) = (#{s_i > s} + u·(#{s_i = s} + 1)) / (n + 1)`` không tăng
    theo ``s``; tập nhận ``{s : p(s) > α}`` là ``{s ≤ q̃}``. Độ phủ đúng ``1 − α`` dưới hoán
    đổi được; ở đây lấy ``max(q̃, 0)`` (không trả dải rỗng) nên độ phủ ``≥ 1 − α`` và
    sát danh định hơn hẳn :func:`conformal_quantile` khi điểm số trùng nhiều.

    Trả về ``inf`` nếu cả điểm số ``inf`` cũng được nhận, :data:`FULL_RANGE` nếu mọi
    điểm số hữu hạn được nhận.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha phải thuộc (0, 1).")
    if not 0.0 <= u <= 1.0:
        raise ValueError("u phải thuộc [0, 1].")
    n = len(scores)
    if n == 0:
        return math.inf
    n_inf = sum(1 for s in scores if math.isinf(s))
    finite = [s for s in scores if not math.isinf(s)]

    def p(s: float) -> float:
        greater = sum(1 for t in scores if t > s)
        equal = sum(1 for t in scores if t == s)
        return (greater + u * (equal + 1)) / (n + 1)

    if p(math.inf) > alpha:
        return math.inf
    top = max(finite) if finite else -1.0
    if (n_inf + u) / (n + 1) > alpha:  # p(s) với s hữu hạn > max điểm số quan sát
        return FULL_RANGE
    q = 0.0
    for s in range(0, int(top) + 1):
        if p(float(s)) > alpha:
            q = float(s)
        else:
            break
    return q


class AdaptiveQuantile:
    """Adaptive Conformal Inference trên điểm số đã quan sát.

    Mỗi bước: :meth:`threshold` trả về ``q̂_t`` (lượng tử mức ``1 − α_t`` của các
    điểm số quá khứ, hoặc của cửa sổ ``window`` điểm gần nhất); sau khi thấy giá trị
    thật, :meth:`update` nhận điểm số mới và cập nhật
    ``α_{t+1} = α_t + γ(α − err_t)``.

    Quy ước ở biên (cần cho cận của Gibbs & Candès): ``α_t ≤ 0`` → dải toàn miền
    (``q̂ = inf``, không bao giờ trượt); ``α_t ≥ 1`` → dải rỗng (``q̂ = −1``, luôn trượt).
    """

    def __init__(
        self,
        alpha: float,
        gamma: float = 0.01,
        init_scores: list[float] | None = None,
        window: int | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha phải thuộc (0, 1).")
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha
        self.window = window
        self.scores: list[float] = list(init_scores or [])
        self.errors: list[int] = []

    def threshold(self) -> float:
        if self.alpha_t <= 0.0:
            return math.inf
        if self.alpha_t >= 1.0:
            return -1.0
        past = self.scores[-self.window:] if self.window else self.scores
        if not past:
            return math.inf
        return conformal_quantile(past, self.alpha_t)

    def update(self, score: float, q_used: float) -> int:
        """Ghi nhận điểm số mới với ngưỡng đã dùng; trả về ``err_t``."""
        err = int(score > q_used)
        self.errors.append(err)
        self.scores.append(score)
        self.alpha_t += self.gamma * (self.alpha - err)
        return err

    @property
    def miscoverage(self) -> float:
        return sum(self.errors) / len(self.errors) if self.errors else 0.0

    def bound(self) -> float:
        """Cận ``(max(α_1, 1−α_1) + γ)/(γT)`` của Gibbs & Candès (Prop. 4.1)."""
        T = len(self.errors)
        if T == 0:
            return math.inf
        return (max(self.alpha, 1.0 - self.alpha) + self.gamma) / (self.gamma * T)
