"""Khoảng ngữ nghĩa (fuzziness interval) ℑ(x) của Đại số Gia tử.

Cơ sở lý thuyết
================
Trong lý thuyết Đại số Gia tử (Nguyen Cat Ho & Wechler; Ho, Lan, Viet), mỗi hạng
từ ngôn ngữ ``x`` được gán một **khoảng mờ / khoảng ngữ nghĩa** ``ℑ(x) ⊆ [0, 1]``
thoả:

1. **Độ dài = độ đo tính mờ:** ``|ℑ(x)| = fm(x)`` — độ dài khoảng đúng bằng
   fuzziness measure của từ.
2. **Phân hoạch theo gia tử:** ``{ℑ(h·x) : h ∈ H}`` là một phân hoạch của
   ``ℑ(x)``. Với hai gia tử ``L`` (Rather, fm = α) và ``V`` (Very, fm = β = 1−α):
   ``|ℑ(L·x)| = α·fm(x)``, ``|ℑ(V·x)| = β·fm(x)``.
3. **Bảo toàn thứ tự:** vị trí trái/phải của ``ℑ(L·x)`` và ``ℑ(V·x)`` bên trong
   ``ℑ(x)`` tuân theo thứ tự ngôn ngữ của ``L·x`` và ``V·x`` (phụ thuộc dấu của x).
4. **Điểm SQM nằm trong khoảng:** giá trị định lượng ngữ nghĩa ``ν(x)`` (SQM) luôn
   thuộc ``ℑ(x)``.

Với hai sinh phần ``c⁻`` (Small) và ``c⁺`` (Large), khoảng gốc là::

    ℑ(c⁻) = [0, θ],   ℑ(c⁺) = [θ, 1]     (θ = fm(c⁻))

Quy tắc chia đệ quy cho ``x = h·p`` với ``ℑ(p) = [a, b]``, ``fm(p) = b − a``:

* nếu ``sign(p) = +1`` (từ dương, thứ tự ``L·p < V·p``)::

      ℑ(L·p) = [a,          a + α(b−a)]
      ℑ(V·p) = [a + α(b−a), b         ]

* nếu ``sign(p) = −1`` (từ âm, thứ tự ``V·p < L·p``)::

      ℑ(V·p) = [a,          a + β(b−a)]
      ℑ(L·p) = [a + β(b−a), b         ]

**Phân hoạch mức k.** Tập các từ có độ dài đúng ``k`` (``k−1`` gia tử + 1 sinh
phần) sinh ra một phân hoạch của ``[0, 1]`` gồm ``2^k`` khoảng liền kề, sắp theo
thứ tự ngữ nghĩa tăng dần. Đây là nền tảng cho *ngữ nghĩa hoá theo khoảng*
(interval-membership fuzzification) trong mô hình I-LTS: một điểm dữ liệu được gán
cho nhãn ``w`` khi giá trị chuẩn hoá của nó rơi vào ``ℑ(w)``.

Phần tử trung hoà ``W`` có ``fm(W) = 0`` nên khoảng suy biến thành điểm ``{θ}`` —
đó chính là biên chung giữa nửa âm và nửa dương của phân hoạch; ``W`` không tham
gia phân hoạch mức k.
"""

from __future__ import annotations

from dataclasses import dataclass

from lts.core.hedge_algebras import HedgeAlgebra


@dataclass(frozen=True)
class Interval:
    """Khoảng đóng ``[lo, hi]``."""

    lo: float
    hi: float

    @property
    def length(self) -> float:
        return self.hi - self.lo

    @property
    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def contains(self, x: float) -> bool:
        """True nếu ``lo ≤ x ≤ hi``."""
        return self.lo <= x <= self.hi

    def scaled(self, lb: float, ub: float) -> "Interval":
        """Ánh xạ khoảng chuẩn hoá trong [0,1] về miền dữ liệu [lb, ub]."""
        span = ub - lb
        return Interval(lb + span * self.lo, lb + span * self.hi)

    def __repr__(self) -> str:  # pragma: no cover - hiển thị
        return f"[{self.lo:.4f}, {self.hi:.4f}]"


class FuzzinessIntervals:
    """Tính khoảng ngữ nghĩa ℑ(x) và phân hoạch [0,1] cho một Đại số Gia tử.

    Parameters
    ----------
    ha : HedgeAlgebra
        Đại số gia tử đã khởi tạo với (theta, alpha).

    Notes
    -----
    Kết quả được cache theo từ; mọi khoảng nằm trong [0, 1] (chuẩn hoá). Dùng
    :meth:`Interval.scaled` để đưa về miền dữ liệu.
    """

    # Bảng dấu tương đối Sgn(h', h) cho từ có ≥ 2 gia tử (cùng quy ước haClustering/edgehac/sqm.py):
    #   little — Ho & Wechler (1990), Ví dụ 1: V dương, L âm đối với cả V và L (mặc định; = hàm sign() của HA)
    #   rather — V làm yếu Rather: Sgn(V,L) = −1, Sgn(L,L) = +1, Sgn(L,V) = −1 (chỉ dùng cho phân tích độ nhạy)
    _TABLES = {"little": {("V", "V"): 1, ("V", "L"): 1, ("L", "V"): -1, ("L", "L"): -1},
               "rather": {("V", "V"): 1, ("V", "L"): -1, ("L", "V"): -1, ("L", "L"): 1}}

    def __init__(self, ha: HedgeAlgebra, variant: str = "little") -> None:
        if variant not in self._TABLES:
            raise ValueError("variant phải là 'little' hoặc 'rather'.")
        self.variant = variant
        self.ha = ha
        self._theta = ha.params.theta
        self._alpha = ha.params.alpha
        self._beta = ha.params.beta
        self._cache: dict[str, Interval] = {}

    # ------------------------------------------------------------------
    # ℑ(x) — khoảng ngữ nghĩa của một từ bất kỳ
    # ------------------------------------------------------------------
    def interval(self, word: str) -> Interval:
        """Trả về khoảng ngữ nghĩa ℑ(word) ⊆ [0, 1] (đệ quy, có cache).

        Raises
        ------
        ValueError
            Nếu ``word`` là "W"/"0"/"1" (không có khoảng thực sự) hoặc rỗng.
        """
        if word in {"W", "0", "1", ""}:
            raise ValueError(
                f"Từ '{word}' không có khoảng ngữ nghĩa không suy biến "
                f"(fm = 0). Chỉ dùng cho sinh phần và các từ đã gia tử hoá."
            )
        if word in self._cache:
            return self._cache[word]

        if word == "-":
            iv = Interval(0.0, self._theta)
        elif word == "+":
            iv = Interval(self._theta, 1.0)
        else:
            parent = word[1:]
            pi = self.interval(parent)
            a, length = pi.lo, pi.length
            # con h·p nằm ở phần dưới của ℑ(p) khi Sgn(h·p) < 0 (với Little: L·p < V·p khi p dương)
            width = (self._alpha if word[0] == "L" else self._beta) * length
            if self._sign(word) < 0:
                iv = Interval(a, a + width)
            else:
                iv = Interval(pi.hi - width, pi.hi)

        self._cache[word] = iv
        return iv

    def _sign(self, word: str) -> int:
        """Dấu của từ theo bảng của biến thể (Little trùng ``HedgeAlgebra.sign``)."""
        if len(word) == 1:
            return -1 if word == "-" else 1
        if len(word) == 2:
            return self._sign(word[1:]) if word[0] == "V" else -self._sign(word[1:])
        return self._TABLES[self.variant][(word[0], word[1])] * self._sign(word[1:])

    # ------------------------------------------------------------------
    # Phân hoạch mức k
    # ------------------------------------------------------------------
    def partition_words(self, level: int) -> list[str]:
        """Danh sách từ có độ dài đúng ``level`` (đã sắp theo SQM tăng dần).

        level=1 → ['-', '+'] (2 từ)
        level=2 → ['V-', 'L-', 'L+', 'V+'] (4 từ)
        level=3 → 8 từ, ...  Tổng quát: 2^level từ.
        """
        if level < 1:
            raise ValueError("level phải ≥ 1.")
        words = ["-", "+"]
        for _ in range(level - 1):
            words = [h + w for w in words for h in ("L", "V")]
        # Sắp theo vị trí khoảng (luôn đúng thứ tự ngôn ngữ do tính phủ kín). Không sắp theo
        # SQM: với α ≠ 0.5 và từ có ≥ 3 gia tử, SQM hiện tại có thể nằm ngoài ℑ(w) và đảo thứ tự
        # (xem paper/hacp/FINDINGS.md F8).
        return sorted(words, key=lambda w: self.interval(w).lo)

    def partition(self, level: int) -> list[tuple[str, Interval]]:
        """Phân hoạch [0,1] ở mức ``level``: list [(word, ℑ(word)), ...] đã sắp xếp.

        Các khoảng liền kề và phủ kín [0, 1].
        """
        return [(w, self.interval(w)) for w in self.partition_words(level)]

    def verify_partition(self, level: int, tol: float = 1e-9) -> bool:
        """Kiểm tra các khoảng ở mức ``level`` liền kề và phủ kín [0,1]."""
        parts = self.partition(level)
        if abs(parts[0][1].lo - 0.0) > tol or abs(parts[-1][1].hi - 1.0) > tol:
            return False
        for (_, left), (_, right) in zip(parts, parts[1:]):
            if abs(left.hi - right.lo) > tol:
                return False
            if left.length < -tol:
                return False
        return True
