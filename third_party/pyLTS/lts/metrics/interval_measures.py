"""Độ đo đánh giá dự báo KHOẢNG (prediction interval).

Bổ sung cho :mod:`lts.metrics.measures` (độ đo điểm) các chỉ số chuẩn của dự báo
khoảng dùng trong mô hình I-LTS:

- **PICP** (Prediction Interval Coverage Probability): tỉ lệ giá trị thực nằm
  trong khoảng dự báo. Càng gần độ tin cậy danh nghĩa (nominal) càng tốt.
- **MPIW** (Mean Prediction Interval Width): độ rộng trung bình của khoảng —
  càng nhỏ càng "sắc", với điều kiện PICP đủ.
- **NMPIW**: MPIW chuẩn hoá theo biên độ dữ liệu ``range(actual)``.
- **ACE** (Average Coverage Error): ``PICP − nominal``.
- **CWC** (Coverage Width-based Criterion): đánh đổi phủ/độ rộng; phạt nặng khi
  PICP < nominal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IntervalMetrics:
    """Container bất biến cho các độ đo dự báo khoảng.

    Attributes
    ----------
    picp : float
        Prediction Interval Coverage Probability ∈ [0, 1].
    mpiw : float
        Mean Prediction Interval Width.
    nmpiw : float
        Normalized MPIW (MPIW / range(actual)).
    ace : float
        Average Coverage Error = picp − nominal.
    cwc : float
        Coverage Width-based Criterion.
    nominal : float
        Độ tin cậy danh nghĩa dùng để tính ACE/CWC.
    n : int
        Số điểm.
    """

    picp: float
    mpiw: float
    nmpiw: float
    ace: float
    cwc: float
    winkler: float
    nominal: float
    n: int

    @classmethod
    def compute(
        cls,
        actual: list[float],
        lowers: list[float],
        uppers: list[float],
        nominal: float = 0.90,
        eta: float = 50.0,
    ) -> "IntervalMetrics":
        """Tính các độ đo khoảng.

        Parameters
        ----------
        actual : list[float]
            Giá trị thực tế.
        lowers, uppers : list[float]
            Cận dưới / cận trên của khoảng dự báo (cùng độ dài với actual).
        nominal : float
            Độ tin cậy danh nghĩa (mặc định 0.90) cho ACE/CWC.
        eta : float
            Hệ số phạt trong CWC (độ dốc hàm mũ).

        Raises
        ------
        ValueError
            Nếu độ dài không khớp hoặc rỗng.
        """
        n = len(actual)
        if not (n == len(lowers) == len(uppers)):
            raise ValueError(
                f"Độ dài không khớp: actual={n}, lowers={len(lowers)}, "
                f"uppers={len(uppers)}"
            )
        if n == 0:
            raise ValueError("Danh sách không được rỗng.")

        covered = sum(1 for a, lo, hi in zip(actual, lowers, uppers) if lo <= a <= hi)
        picp = covered / n
        mpiw = sum(hi - lo for lo, hi in zip(lowers, uppers)) / n
        winkler = sum(cls.winkler_score(a, lo, hi, nominal)
                      for a, lo, hi in zip(actual, lowers, uppers)) / n

        rng = max(actual) - min(actual)
        nmpiw = mpiw / rng if rng > 0 else float("nan")

        ace = picp - nominal
        # CWC: nếu PICP ≥ nominal, phạt = 0 (chỉ tính bề rộng); ngược lại phạt mũ.
        if picp >= nominal:
            cwc = nmpiw if not math.isnan(nmpiw) else mpiw
        else:
            base = nmpiw if not math.isnan(nmpiw) else mpiw
            cwc = base * (1.0 + math.exp(-eta * (picp - nominal)))

        return cls(
            picp=picp,
            mpiw=mpiw,
            nmpiw=nmpiw,
            ace=ace,
            cwc=cwc,
            winkler=winkler,
            nominal=nominal,
            n=n,
        )

    @staticmethod
    def winkler_score(actual: float, lo: float, hi: float,
                      nominal: float = 0.90) -> float:
        """Winkler (interval) score cho một điểm — thang proper, càng nhỏ càng tốt.

        Phạt bề rộng (hi−lo) cộng phạt tuyến tính khi actual nằm ngoài khoảng,
        hệ số 2/alpha với alpha = 1 − nominal.
        """
        alpha = 1.0 - nominal
        width = hi - lo
        if actual < lo:
            return width + (2.0 / alpha) * (lo - actual)
        if actual > hi:
            return width + (2.0 / alpha) * (actual - hi)
        return width

    def __str__(self) -> str:
        return (
            f"IntervalMetrics(n={self.n}, PICP={self.picp:.3f}, "
            f"MPIW={self.mpiw:.2f}, NMPIW={self.nmpiw:.3f}, "
            f"ACE={self.ace:+.3f}, CWC={self.cwc:.3f} @nominal={self.nominal})"
        )
