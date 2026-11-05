"""Hiệu chuẩn ngôn ngữ cho dự báo điểm của **bất kỳ** mô hình nào.

Như dải ``base="point"`` của HACP-LTS nhưng ô cơ sở là ô chứa dự báo điểm ngoài
(ETS, ARIMA, foundation model…). Bảo đảm của Định lý 1 / Định lý 2 giữ nguyên vì
chúng không phụ thuộc cách sinh dải cơ sở. So với conformal số trên cùng dự báo
điểm, độ rộng chênh lệch chính là **cái giá của việc đọc được bằng ngôn ngữ**.
"""

from __future__ import annotations

import math
import random

from lts.conformal.calibration import (
    AdaptiveQuantile,
    conformal_quantile,
    randomized_conformal_quantile,
)
from lts.conformal.lattice import VARIANTS, LinguisticBand, LinguisticLattice


class HACP:
    """Dải ngôn ngữ quanh dự báo điểm ngoài trên một :class:`LinguisticLattice`.

    Parameters
    ----------
    lattice : LinguisticLattice
        Lưới ô của đại số gia tử trên miền biến đích.
    alpha : float
        Mức trượt mục tiêu.
    variant : {"cell", "hedge"}
        Họ dải lồng nhau.
    randomized : bool
        Dùng p-value làm trơn (rút ``u`` riêng cho mỗi điểm test).
    """

    def __init__(
        self,
        lattice: LinguisticLattice,
        alpha: float = 0.1,
        variant: str = "cell",
        randomized: bool = False,
        seed: int | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant phải thuộc {VARIANTS}.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha phải thuộc (0, 1).")
        self.lattice = lattice
        self.alpha = alpha
        self.variant = variant
        self.randomized = randomized
        self._rng = random.Random(seed)
        self._scores: list[float] = []
        self._q_hat: float = math.inf

    # Dự báo đầu vào là điểm ``f`` hoặc khoảng ``(lo, hi)`` của mô hình nền. Khoảng cho ô cơ sở
    # thích nghi ``[j(lo), j(hi)]`` (như CQR: hẹp khi mô hình chắc, rộng khi không chắc);
    # bảo đảm giữ nguyên vì họ dải vẫn lồng nhau theo q.
    def _range(self, forecast: float | tuple[float, float]) -> tuple[int, int]:
        return _range_on(self.lattice, forecast)

    def score(self, forecast: float | tuple[float, float], y: float) -> float:
        l, u = self._range(forecast)
        return self.lattice.score(l, u, y, self.variant)

    def calibrate(self, forecasts: list, ys: list[float]) -> float:
        """Điểm số trên tập hiệu chuẩn; ``forecasts`` là danh sách điểm hoặc cặp ``(lo, hi)``."""
        self._scores = [self.score(f, y) for f, y in zip(forecasts, ys)]
        self._q_hat = conformal_quantile(self._scores, self.alpha)
        return self._q_hat

    def threshold(self) -> float:
        if self.randomized:
            return randomized_conformal_quantile(self._scores, self.alpha, self._rng.random())
        return self._q_hat

    def band(self, forecast: float | tuple[float, float], q: float | None = None) -> LinguisticBand:
        l, u = self._range(forecast)
        return self.lattice.band(l, u, self.threshold() if q is None else q, self.variant)

    def bands(self, forecasts: list) -> list[LinguisticBand]:
        return [self.band(f) for f in forecasts]

    def online(
        self,
        forecasts: list,
        ys: list[float],
        gamma: float = 0.01,
        window: int | None = None,
    ) -> tuple[list[LinguisticBand], AdaptiveQuantile]:
        """ACI: dải một bước, cập nhật sau mỗi giá trị thật (khởi tạo bằng điểm số hiệu chuẩn)."""
        aci = AdaptiveQuantile(self.alpha, gamma, init_scores=self._scores, window=window)
        out: list[LinguisticBand] = []
        for f, y in zip(forecasts, ys):
            q = aci.threshold()
            l, u = self._range(f)
            if q < 0:
                c = self.lattice.cells[l]
                out.append(LinguisticBand(c.lo, c.lo, "∅", "∅", q))
            else:
                out.append(self.lattice.band(l, u, q, self.variant))
            aci.update(self.lattice.score(l, u, y, self.variant), q)
        return out, aci

    def online_adaptive(
        self,
        forecasts: list,
        ys: list[float],
        lattices: list[LinguisticLattice],
        init_scores: list[float],
        gamma: float = 0.01,
    ) -> tuple[list[LinguisticBand], AdaptiveQuantile]:
        """ACI trên **lưới thay đổi theo thời gian** (Theorem 2′ trong theory.tex).

        ``lattices[t]`` là lưới dùng ở bước t (ví dụ miền dựng lại từ mọi dữ liệu trước t); điểm số
        của bước t tính bằng chính lưới đó. Cận độ phủ dài hạn của ACI đúng với mọi dãy họ dải lồng
        nhau, nên không cần lưới cố định. ``init_scores``: điểm số hiệu chuẩn (mỗi điểm theo lưới của
        thời điểm của nó). Bỏ qua ``self.lattice``.
        """
        aci = AdaptiveQuantile(self.alpha, gamma, init_scores=list(init_scores))
        out: list[LinguisticBand] = []
        for f, y, lat in zip(forecasts, ys, lattices):
            q = aci.threshold()
            l, u = _range_on(lat, f)
            if q < 0:
                c = lat.cells[l]
                out.append(LinguisticBand(c.lo, c.lo, "∅", "∅", q))
            else:
                out.append(lat.band(l, u, q, self.variant))
            aci.update(lat.score(l, u, y, self.variant), q)
        return out, aci

    @property
    def q_hat(self) -> float:
        return self._q_hat

    @property
    def scores(self) -> list[float]:
        return list(self._scores)


def _range_on(lattice: LinguisticLattice, forecast) -> tuple[int, int]:
    """Đoạn ô cơ sở của một dự báo (điểm hoặc ``(lo, hi)``) trên ``lattice``."""
    if isinstance(forecast, tuple):
        lo, hi = forecast
        if hi < lo:
            lo, hi = hi, lo
        return lattice.index_clamped(lo), lattice.index_clamped(hi)
    i = lattice.index_clamped(forecast)
    return i, i
