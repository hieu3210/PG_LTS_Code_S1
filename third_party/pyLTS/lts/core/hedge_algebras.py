"""
Hedge Algebras với 2 generators và 2 hedges.

Encoding từ vựng:
  Boundary:   "0" (infimum), "W" (neutral/theta), "1" (supremum)
  Generators: "-" (c_minus / Small), "+" (c_plus / Large)
  Hedges:     "L" (Rather — hedge âm), "V" (Very — hedge dương)

Ví dụ từ: "V-" = Very Small, "L+" = Rather Large, "VV+" = Very Very Large
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class HAParams:
    """Tham số fuzziness cho Hedge Algebra 2 generators / 2 hedges.

    theta = fm(c_minus)  — trọng số ngữ nghĩa của generator âm
    alpha = mu(L)        — fuzziness measure của hedge "Rather" (L)
    beta  = 1 - alpha    — fuzziness measure của hedge "Very" (V)
    """

    theta: float
    alpha: float
    # Biến thể đại số cho SQM ν (cùng quy ước với haClustering/edgehac/sqm.py):
    #   "lts"    — mặc định, tái lập các bài LTS 2020–2026: hàm dấu của đại số Little nhưng
    #              ω(L·) = α (giá trị của Rather). Đúng với ≤ 1 gia tử; với ≥ 2 gia tử và α ≠ 0.5
    #              ν có thể nằm ngoài ℑ(w) và đảo thứ tự (paper/hacp/FINDINGS.md F8).
    #   "little" — bảng dấu Ho & Wechler (1990), Ví dụ 1 (V dương, L âm đối với cả V và L);
    #              ω ≡ β ⇒ ν(w) là điểm cắt của ℑ(w), bảo toàn thứ tự ở mọi độ sâu.
    # Khoảng tính mờ ℑ (FuzzinessIntervals) luôn theo bảng dấu Little. θ = α = 0.5 ⇒ hai biến thể trùng nhau.
    omega_rule: str = "lts"

    def __post_init__(self) -> None:
        if self.omega_rule not in ("lts", "little"):
            raise ValueError("omega_rule phải là 'lts' hoặc 'little'.")

    @property
    def beta(self) -> float:
        return 1.0 - self.alpha

    @classmethod
    def enrollment(cls) -> "HAParams":
        """theta=0.57, alpha=0.49 — thực nghiệm Table 4 trong bài báo."""
        return cls(theta=0.57, alpha=0.49)

    @classmethod
    def variations(cls) -> "HAParams":
        """theta=0.55, alpha=0.52 — thực nghiệm Table 8 (variation series)."""
        return cls(theta=0.55, alpha=0.52)


class HedgeAlgebra:
    """Hedge Algebra AX = (X, G, C, H, ≤) với 2 generators và 2 hedges.

    Parameters
    ----------
    params : HAParams
        Tham số fuzziness (theta, alpha).
    """

    def __init__(self, params: HAParams) -> None:
        self.params = params
        # Bind cached methods theo params cụ thể
        self._fm = lru_cache(maxsize=None)(self._fm_impl)
        self._sign = lru_cache(maxsize=None)(self._sign_impl)
        self._omega = lru_cache(maxsize=None)(self._omega_impl)
        self._sqm = lru_cache(maxsize=None)(self._sqm_impl)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_words(self, specificity: int) -> list[str]:
        """Trả về danh sách từ đã sắp xếp theo SQM ở mức specificity.

        specificity=0 → 3 từ: ['-', 'W', '+']
        specificity=1 → 7 từ: ['V-', '-', 'L-', 'W', 'L+', '+', 'V+']
        specificity=2 → 15 từ (thêm LL-, VL-, LV-, VV-, LL+, ...)

        Luật sinh từ: hedge chỉ áp dụng lên generators và các từ đã hedged trước
        (không áp dụng lên "W" — neutral element).
        """
        # Từ vựng cơ sở: generators + neutral (không có boundary "0", "1")
        base: set[str] = {"-", "W", "+"}
        # Frontier: chỉ generators, W không được hedged
        frontier: set[str] = {"-", "+"}
        all_words: set[str] = set(base)

        for _ in range(specificity):
            new_frontier: set[str] = set()
            for word in frontier:
                for hedge in ("L", "V"):
                    hw = hedge + word
                    all_words.add(hw)
                    new_frontier.add(hw)
            frontier = new_frontier

        return self.sort_words(list(all_words))

    def sqm(self, word: str) -> float:
        """Semantic Quantifying Measure: ánh xạ word → [0, 1]."""
        return self._sqm(word)

    def fm(self, word: str) -> float:
        """Fuzziness measure của word."""
        return self._fm(word)

    def sign(self, word: str) -> int:
        """Dấu đại số của word: +1 hoặc -1."""
        return self._sign(word)

    def omega(self, word: str) -> float:
        """Hệ số hiệu chỉnh omega(word) ∈ {alpha, beta}."""
        return self._omega(word)

    def sort_words(self, words: list[str]) -> list[str]:
        """Trả về danh sách mới đã sắp xếp tăng dần theo SQM."""
        return sorted(words, key=self._sqm)

    # ------------------------------------------------------------------
    # Internal implementations (wrapped with lru_cache in __init__)
    # ------------------------------------------------------------------

    def _fm_impl(self, x: str) -> float:
        if x in {"W", "0", "1"}:
            return 0.0
        if len(x) == 1:
            return self.params.theta if x == "-" else 1.0 - self.params.theta
        if x[0] == "L":
            return self.params.alpha * self._fm(x[1:])
        return self.params.beta * self._fm(x[1:])

    def _sign_impl(self, x: str) -> int:
        if len(x) == 1:
            return -1 if x == "-" else 1
        sign_rest = self._sign(x[1:])
        # "L" (Rather) đảo dấu, "V" (Very) giữ dấu
        return sign_rest if x[0] == "V" else -sign_rest

    def _omega_impl(self, x: str) -> float:
        if self.params.omega_rule == "little":
            return self.params.beta
        # lts: Hedge V (Very) → omega = beta; Hedge L → omega = alpha (giá trị của Rather).
        factor = 1 if x[0] == "V" else -1
        return (1 + factor * (self.params.beta - self.params.alpha)) / 2

    def _sqm_impl(self, x: str) -> float:
        if x == "0":
            return 0.0
        if x == "W":
            return self.params.theta
        if x == "1":
            return 1.0
        if x == "-":
            return self.params.theta - self.params.alpha * self._fm(x)
        if x == "+":
            return self.params.theta + self.params.alpha * self._fm(x)
        return self._sqm(x[1:]) + self._sign(x) * self._fm(x) * (1.0 - self._omega(x))
