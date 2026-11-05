"""Lưới ngôn ngữ của ĐSGT và các họ dải lồng nhau cho HACP-LTS.

Phân hoạch mức k gồm ``N = 2^k`` ô ``C_1 < … < C_N`` (ảnh của khoảng tính mờ
ℑ(w) lên miền dữ liệu ``[lb, ub]``). Biên chung được gán cho ô thấp hơn, khớp với
mã hoá theo tính chứa của I-LTS. Hai họ dải lồng nhau theo ``q`` (xem
``paper/hacp/theory.tex``):

- **cell**  : nới đoạn ô cơ sở ``[l, u]`` thêm ``q`` ô mỗi phía.
- **hedge** : thay mỗi từ cơ sở bằng tổ tiên ``anc_q(w)`` (bỏ ``q`` gia tử ngoài cùng),
  tức khái quát hoá ngôn ngữ; ``q = k`` cho toàn miền.

Điểm số ``s(x, y) = min{q : y ∈ F_q(x)}`` là số nguyên, hoặc ``inf`` khi ``y`` nằm
ngoài miền (không dải hữu hạn nào chứa được).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from lts.core.fuzziness_interval import FuzzinessIntervals, Interval

VARIANTS = ("cell", "hedge")


@dataclass(frozen=True)
class LinguisticBand:
    """Một dải dự báo ngôn ngữ: khoảng số + hai từ đầu mút + mức nới ``q``."""

    lo: float
    hi: float
    lo_word: str
    hi_word: str
    q: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, y: float) -> bool:
        return self.lo <= y <= self.hi

    @property
    def words(self) -> tuple[str, str]:
        return self.lo_word, self.hi_word


class LinguisticLattice:
    """Các ô mức ``level`` trên ``[lb, ub]`` và phép nới ô / khái quát gia tử.

    Parameters
    ----------
    fi : FuzzinessIntervals
        Bộ tính khoảng tính mờ của đại số gia tử.
    level : int
        Độ dài từ k (``2^k`` ô).
    lb, ub : float
        Miền dữ liệu.
    """

    # Ô biên mở (Prop. 5 trong theory.tex): hai ô ngoài cùng kéo dài tới −∞ / +∞, nên các ô phủ kín ℝ,
    # điểm số luôn hữu hạn và mọi dải chạm ô biên là nửa vô hạn ("… hoặc hơn nữa").
    open_ends: bool = False

    def __init__(
        self, fi: FuzzinessIntervals, level: int, lb: float, ub: float, open_ends: bool = False
    ) -> None:
        if ub <= lb:
            raise ValueError("ub phải lớn hơn lb.")
        self.open_ends = open_ends
        self.fi = fi
        self.level = level
        self.lb = lb
        self.ub = ub
        self.words: list[str] = fi.partition_words(level)
        self.cells: list[Interval] = [fi.interval(w).scaled(lb, ub) for w in self.words]
        # biên phải của từng ô — dùng cho tìm nhị phân O(log N)
        self._his: list[float] = [c.hi for c in self.cells]
        self._his[-1] = ub

    @property
    def n_cells(self) -> int:
        return len(self.words)

    # ------------------------------------------------------------------
    # Chỉ số ô theo tính chứa
    # ------------------------------------------------------------------
    def index_of(self, y: float) -> int | None:
        """Chỉ số (0-based) của ô chứa ``y``; ``None`` nếu ``y`` ngoài ``[lb, ub]``.

        Biên chung thuộc ô thấp hơn (giống ``ILTSModel._fuzzify``).
        """
        if y < self.lb or y > self.ub:
            if self.open_ends:
                return 0 if y < self.lb else self.n_cells - 1
            return None
        return bisect.bisect_left(self._his, y)

    def index_clamped(self, y: float) -> int:
        """Như :meth:`index_of` nhưng kẹp giá trị ngoài miền về ô biên."""
        i = self.index_of(y)
        if i is not None:
            return i
        return 0 if y < self.lb else self.n_cells - 1

    # ------------------------------------------------------------------
    # Hai họ dải lồng nhau F_q
    # ------------------------------------------------------------------
    def band(self, l: int, u: int, q: float, variant: str = "cell") -> LinguisticBand:
        """Dải ``F_q`` từ đoạn ô cơ sở ``[l, u]`` (0-based, bao gồm hai đầu)."""
        b = self._band(l, u, q, variant)
        if not self.open_ends or math.isinf(q):
            return b
        tol = 1e-12 * (self.ub - self.lb)
        lo = -math.inf if b.lo <= self.lb + tol else b.lo
        hi = math.inf if b.hi >= self.ub - tol else b.hi
        return LinguisticBand(lo, hi, b.lo_word, b.hi_word, b.q)

    def _band(self, l: int, u: int, q: float, variant: str) -> LinguisticBand:
        if variant not in VARIANTS:
            raise ValueError(f"variant phải thuộc {VARIANTS}.")
        if math.isinf(q):
            return LinguisticBand(-math.inf, math.inf, "⊥", "⊤", q)
        q = int(q)
        if variant == "cell":
            lo_i = max(0, l - q)
            hi_i = min(self.n_cells - 1, u + q)
            return LinguisticBand(self.cells[lo_i].lo, self.cells[hi_i].hi,
                                  self.words[lo_i], self.words[hi_i], q)
        if q >= self.level:
            return LinguisticBand(self.lb, self.ub, "⊤", "⊤", q)
        w_lo, w_hi = self.words[l][q:], self.words[u][q:]
        iv_lo = self.fi.interval(w_lo).scaled(self.lb, self.ub)
        iv_hi = self.fi.interval(w_hi).scaled(self.lb, self.ub)
        return LinguisticBand(iv_lo.lo, iv_hi.hi, w_lo, w_hi, q)

    def describe(self, lo_i: int, hi_i: int) -> list[str]:
        """Tập từ **thô nhất** có khoảng tính mờ phân hoạch đúng đoạn ô ``[lo_i, hi_i]``.

        Các ô mức k là lá của cây gia tử nhị phân, sắp theo thứ tự; con cháu của một từ
        mức ``k − q`` là một khối liền ``2^q`` ô, căn theo bội ``2^q``. Tham lam từ trái
        lấy khối căn lớn nhất nằm trọn trong đoạn cho phân rã ít từ nhất (như segment
        tree). ``["⊤"]`` nghĩa là toàn miền.
        """
        if not 0 <= lo_i <= hi_i < self.n_cells:
            raise ValueError("Đoạn ô không hợp lệ.")
        terms: list[str] = []
        i = lo_i
        while i <= hi_i:
            q = 0
            while (q < self.level and i % (2 ** (q + 1)) == 0
                   and i + 2 ** (q + 1) - 1 <= hi_i):
                q += 1
            terms.append("⊤" if q == self.level else self.words[i][q:])
            i += 2 ** q
        return terms

    def describe_band(self, band: LinguisticBand) -> list[str]:
        """:meth:`describe` cho một dải (kẹp về miền); dải rỗng → ``[]``."""
        if band.hi < band.lo or (band.hi == band.lo and band.lo_word == "∅"):
            return []
        lo_i = self.index_clamped(max(band.lo, self.lb) + 1e-12 * (self.ub - self.lb))
        hi_i = self.index_clamped(min(band.hi, self.ub))
        return self.describe(lo_i, hi_i)

    def score(self, l: int, u: int, y: float, variant: str = "cell") -> float:
        """Điểm số ``s = min{q : y ∈ F_q}``; ``inf`` nếu ``y`` ngoài miền."""
        j = self.index_of(y)
        if j is None:
            return math.inf
        if variant == "cell":
            return float(max(0, l - j, j - u))
        if variant != "hedge":
            raise ValueError(f"variant phải thuộc {VARIANTS}.")
        if l <= j <= u:
            return 0.0
        # tổ tiên của các từ liền kề vẫn liền kề → chỉ cần so tiền tố của từ biên
        wj = self.words[j]
        edge = self.words[l] if j < l else self.words[u]
        for q in range(1, self.level):
            if wj[q:] == edge[q:]:
                return float(q)
        return float(self.level)


class BinLattice(LinguisticLattice):
    """Lưới **đối chứng**: ``2^k`` bin với biên tuỳ ý, cùng cấu trúc cây nhị phân như ĐSGT.

    Dùng để tách vai trò hình học ô của ĐSGT (REVIEW M1): cùng số ô, cùng phép nới ô và cùng
    phân rã mô tả tối thiểu (khối căn ``2^q`` ô), chỉ khác vị trí biên. Nhãn là mã nhị phân của
    đường đi trên cây (``"0"``/``"1"``; ký tự cuối là nửa trái/phải), nên ``word[q:]`` là tổ tiên —
    độ sâu mô tả so sánh được 1–1 với từ ĐSGT. Chỉ hỗ trợ họ ``"cell"``.

    Parameters
    ----------
    edges : list[float]
        ``2^k + 1`` biên tăng dần; ``edges[0]`` và ``edges[-1]`` là miền.
    """

    def __init__(self, edges: list[float]) -> None:
        n = len(edges) - 1
        level = n.bit_length() - 1
        if n < 2 or 2 ** level != n:
            raise ValueError("Số bin phải là lũy thừa của 2 (≥ 2).")
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("Biên phải tăng ngặt.")
        self.fi = None
        self.level = level
        self.lb, self.ub = edges[0], edges[-1]
        # bit thấp nhất (lá) ở bên trái chuỗi → word[q:] bỏ q mức lá
        self.words = ["".join(str((i >> b) & 1) for b in range(level)) for i in range(n)]
        self.cells = [Interval(a, b) for a, b in zip(edges, edges[1:])]
        self._his = [c.hi for c in self.cells]

    @classmethod
    def uniform(cls, level: int, lb: float, ub: float) -> "BinLattice":
        n = 2 ** level
        return cls([lb + (ub - lb) * i / n for i in range(n + 1)])

    @classmethod
    def quantile(
        cls,
        level: int,
        lb: float,
        ub: float,
        train: list[float],
        mix: float = 0.0,
    ) -> "BinLattice":
        """Biên trong là phân vị của dữ liệu train (như QFTS), hai đầu là miền ``[lb, ub]``.

        ``mix > 0``: phân vị của hỗn hợp ``(1 − mix)·F_train + mix·Uniform[lb, ub]`` — phân bổ một
        phần ô cho vùng ngoài khoảng train (đối chứng công bằng hơn khi miền được nới rộng).
        Phân vị trùng nhau được tách tối thiểu để biên tăng ngặt.
        """
        n = 2 ** level
        xs = sorted(train)

        def emp_cdf(x: float) -> float:
            return bisect.bisect_right(xs, x) / len(xs)

        def emp_quantile(p: float) -> float:
            pos = (len(xs) - 1) * p
            lo = int(pos)
            frac = pos - lo
            hi = min(lo + 1, len(xs) - 1)
            return xs[lo] * (1 - frac) + xs[hi] * frac

        inner = []
        for i in range(1, n):
            p = i / n
            if mix <= 0.0:
                inner.append(emp_quantile(p))
                continue
            a, b = lb, ub                                  # giải F_mix(x) = p bằng chia đôi
            for _ in range(60):
                m = 0.5 * (a + b)
                f = (1 - mix) * emp_cdf(m) + mix * (m - lb) / (ub - lb)
                a, b = (m, b) if f < p else (a, m)
            inner.append(0.5 * (a + b))
        edges = [lb] + inner + [ub]
        eps = (ub - lb) * 1e-9
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + eps
        if edges[-1] > ub:
            raise ValueError("Miền quá hẹp so với phân vị train.")
        return cls(edges)

    def band(self, l: int, u: int, q: float, variant: str = "cell") -> LinguisticBand:
        if variant != "cell":
            raise ValueError("BinLattice chỉ hỗ trợ variant='cell'.")
        return super().band(l, u, q, "cell")

    def score(self, l: int, u: int, y: float, variant: str = "cell") -> float:
        if variant != "cell":
            raise ValueError("BinLattice chỉ hỗ trợ variant='cell'.")
        return super().score(l, u, y, "cell")
