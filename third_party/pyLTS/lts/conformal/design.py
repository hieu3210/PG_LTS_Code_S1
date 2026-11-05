"""Quy tắc thiết kế lưới cho HACP: miền và độ phân giải, chỉ dùng dữ liệu train.

Vì miền ``[lb, ub]`` và mức ``k`` được chọn **chỉ từ dữ liệu train**, chúng cố định
khi điều kiện trên train — Định lý 1 (split) và Định lý 2 (ACI) giữ nguyên.

- :func:`universe` : khoảng biến thiên train nới ``margin`` lần mỗi phía. Giá trị
  ngoài miền có điểm số ``inf`` (không phủ được), nên miền rộng nâng trần độ phủ.
- :func:`resolution_level` : k nhỏ nhất để ô **trung bình** hẹp hơn ``eps`` lần độ
  lệch chuẩn của chuỗi train — miền rộng được bù bằng lưới mịn hơn.

Mặc định (``margin=1.0``, ``eps=0.25``) được chọn từ khảo sát trên M4
(``paper/hacp/FINDINGS.md`` F6).
"""

from __future__ import annotations

import math
import statistics

DEFAULT_MARGIN = 1.0
DEFAULT_EPS = 0.25


def universe(
    train: list[float],
    margin: float = DEFAULT_MARGIN,
    symmetric: bool = False,
) -> tuple[float, float]:
    """``[min − margin·r, max + margin·r]`` với ``r`` là biên độ train.

    ``symmetric=True`` (dùng cho sai phân): miền ``[−M, M]`` với ``M = max(|lb|, |ub|)`` của
    miền thường — chứa miền thường và đối xứng quanh 0. Khi
    đó với ``θ = 0.5`` phần tử trung hoà W ứng đúng "không đổi", ``c⁻`` = *giảm*,
    ``c⁺`` = *tăng* — dải đọc được như "từ *giảm nhẹ* đến *tăng nhẹ*".
    """
    lo, hi = min(train), max(train)
    r = (hi - lo) or 1.0
    lb, ub = lo - margin * r, hi + margin * r
    if symmetric:
        m = max(abs(lb), abs(ub))
        return -m, m
    return lb, ub


def resolution_level(
    lb: float,
    ub: float,
    train: list[float],
    eps: float = DEFAULT_EPS,
    k_min: int = 3,
    k_max: int = 9,
) -> int:
    """``k = ⌈log₂((ub − lb) / (eps·σ̂))⌉`` kẹp trong ``[k_min, k_max]``.

    Với ``θ = α = 0.5`` các ô mức k đều nhau (rộng ``(ub − lb)/2^k``); với tham số
    khác, đây là độ rộng ô trung bình.
    """
    sd = statistics.pstdev(train) if len(train) > 1 else 0.0
    if sd <= 0.0:
        return k_min
    k = math.ceil(math.log2((ub - lb) / (eps * sd)))
    return max(k_min, min(k_max, k))


def select_alpha(
    forecasts: list,
    ys: list[float],
    level: int,
    lb: float,
    ub: float,
    delta: float,
    alphas: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5),
    theta: float = 0.5,
) -> float:
    """Chọn α (hình học lưới ĐSGT) **chỉ từ dữ liệu train** — giữ nguyên bảo đảm của Định lý 1.

    ``forecasts``/``ys``: dự báo một bước (điểm hoặc khoảng) và giá trị thật trên phần cuối của đoạn train
    (cùng không gian với lưới, ví dụ sai phân). Nửa đầu hiệu chuẩn thử, nửa sau đo độ rộng trung bình;
    trả α có dải hẹp nhất (hoà → α lớn hơn, gần lưới đều). Không dùng tập hiệu chuẩn thật nên lựa chọn
    độc lập với nó (Prop. 4 (i) với phần chọn là dữ liệu train).
    """
    from lts.conformal.lattice import LinguisticLattice
    from lts.conformal.wrapper import HACP
    from lts.core.fuzziness_interval import FuzzinessIntervals
    from lts.core.hedge_algebras import HAParams, HedgeAlgebra

    half = len(ys) // 2
    if half < 4:
        return 0.5
    best, best_w = 0.5, math.inf
    for a in sorted(alphas, reverse=True):
        lat = LinguisticLattice(FuzzinessIntervals(HedgeAlgebra(HAParams(theta, a))), level, lb, ub)
        w = HACP(lat, alpha=delta)
        w.calibrate(forecasts[:half], ys[:half])
        bands = w.bands(forecasts[half:])
        width = sum(min(b.hi, ub) - max(b.lo, lb) for b in bands) / len(bands)
        if width < best_w - 1e-12:
            best, best_w = a, width
    return best


def quantile_cell_bound(c: float, level: int, lb: float, ub: float, alpha: float,
                        theta: float = 0.5) -> float:
    """Phần dư của cận trên Prop. 6: ``2·s(c) + s⁺(c)`` — độ dài ô chứa ``c`` (phía dương, tính từ tâm
    miền) cộng ô kế tiếp ra ngoài. Độ rộng quần thể W thoả ``2c ≤ W < 2c + quantile_cell_bound``."""
    from lts.conformal.lattice import LinguisticLattice
    from lts.core.fuzziness_interval import FuzzinessIntervals
    from lts.core.hedge_algebras import HAParams, HedgeAlgebra

    lat = LinguisticLattice(FuzzinessIntervals(HedgeAlgebra(HAParams(theta, alpha))), level, lb, ub)
    centre = 0.5 * (lb + ub)
    i = lat.index_clamped(centre + c)
    nxt = min(i + 1, lat.n_cells - 1)
    return 2 * lat.cells[i].length + lat.cells[nxt].length


def plugin_alpha(
    abs_residuals: list[float],
    level: int,
    lb: float,
    ub: float,
    delta: float,
    alphas: tuple[float, ...] = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
) -> float:
    """Quy tắc plug-in cho α (Prop. 6): đặt ô mịn tại phân vị dự kiến của sai số.

    ``ĉ`` = phân vị (1 − δ/2) thực nghiệm của ``abs_residuals`` (phần dư một bước trên **train** — giữ nguyên
    bảo đảm); trả α nhỏ nhất hoá ``2·s_α(ĉ) + s⁺_α(ĉ)``. Hoà → α gần 0.5 hơn (lưới đều).
    """
    xs = sorted(abs(r) for r in abs_residuals)
    if not xs:
        return 0.5
    pos = min(len(xs) - 1, max(0, math.ceil((1 - delta / 2) * len(xs)) - 1))
    c = xs[pos]
    best, best_v = 0.5, math.inf
    for a in sorted(alphas, key=lambda v: abs(v - 0.5)):
        v = quantile_cell_bound(c, level, lb, ub, a)
        if v < best_v - 1e-12:
            best, best_v = a, v
    return best
