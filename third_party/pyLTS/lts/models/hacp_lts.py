"""HACP-LTS — HACP (Hedge-Algebraic Conformal Prediction) với I-LTS làm dải cơ sở.

Hiệu chuẩn conformal **trên lưới ngôn ngữ của ĐSGT**: dải dự báo được nới theo số
ô khoảng tính mờ (``variant="cell"``) hoặc theo số mức gia tử bị bỏ bớt
(``variant="hedge"``), nên đầu ra luôn là một phát biểu ngôn ngữ ("từ *A* đến *B*")
đồng thời có bảo đảm độ phủ biên ``≥ 1 − α`` (split) hoặc độ phủ dài hạn (ACI).

Dải cơ sở lấy từ I-LTS (mã hoá theo tính chứa — điều kiện để phủ nhãn ⇔ phủ giá
trị, xem ``paper/hacp/theory.tex`` Lemma 2):

- ``base="union"``: đoạn ô của tập vế phải ``R`` (dải hợp I-LTS; ``q̂ = 0`` ⇒ trùng I-LTS).
- ``base="point"``: ô chứa dự báo điểm LTS (dải cơ sở hẹp nhất).

``soft_match=True``: LHS chưa gặp được khớp với (các) LHS gần nhất theo khoảng cách
giá trị định lượng ngữ nghĩa — thay cho fallback ``ℑ(ℓ_{t−1})`` của I-LTS (ý tưởng
khớp mềm của SW-LTS).

Quy trình dùng::

    m = HACPLTSModel(params, level=3, alpha=0.1)
    m.fit(train, lb, ub)                    # dựng cơ sở luật (đóng băng)
    m.calibrate(cal_with_context)           # điểm số + q̂ (split)
    bands = m.forecast_external_bands(test_with_context)
    online = m.forecast_online(test_with_context, gamma=0.01)   # ACI
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
from lts.core.hedge_algebras import HAParams
from lts.models.ilts_model import ILTSModel


class HACPLTSModel:
    """Hedge-Algebraic Conformal Prediction (HACP).

    Parameters
    ----------
    params : HAParams
        Tham số (theta, alpha) của đại số gia tử.
    level : int
        Độ dài từ k (``2^k`` ô).
    order : int
        Bậc luật λ.
    alpha : float
        Mức trượt mục tiêu (độ phủ danh định ``1 − alpha``).
    variant : {"cell", "hedge"}
        Họ dải lồng nhau.
    base : {"union", "point"}
        Dải cơ sở.
    soft_match : bool
        Khớp LHS chưa gặp với LHS gần nhất.
    randomized : bool
        Dùng p-value làm trơn (rút ``u ~ U(0,1)`` riêng cho mỗi điểm test) để độ phủ
        sát ``1 − α`` thay vì bảo thủ do điểm số nguyên trùng nhau.
    seed : int | None
        Hạt giống cho ``u`` khi ``randomized=True``.
    """

    def __init__(
        self,
        params: HAParams,
        level: int = 3,
        order: int = 1,
        alpha: float = 0.1,
        variant: str = "cell",
        base: str = "union",
        soft_match: bool = True,
        randomized: bool = False,
        seed: int | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant phải thuộc {VARIANTS}.")
        if base not in {"union", "point"}:
            raise ValueError("base phải là 'union' hoặc 'point'.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha phải thuộc (0, 1).")
        self.params = params
        self.level = level
        self.order = order
        self.alpha = alpha
        self.variant = variant
        self.base = base
        self.soft_match = soft_match
        self.randomized = randomized
        self._rng = random.Random(seed)

        self._ilts = ILTSModel(params, level=level, order=order, band="union",
                               encoding="containment")
        self._lattice: LinguisticLattice | None = None
        self._index: dict[str, int] = {}
        self._nu: dict[str, float] = {}
        self._scores: list[float] = []
        self._q_hat: float = math.inf

    # ------------------------------------------------------------------
    # Fit dải cơ sở
    # ------------------------------------------------------------------
    def fit(self, data: list[float], lb: float, ub: float) -> None:
        self._ilts.fit(data, lb, ub)
        self._lattice = LinguisticLattice(self._ilts._fi, self.level, lb, ub)
        self._index = {w: i for i, w in enumerate(self._lattice.words)}
        span = ub - lb
        self._nu = {w: (p - lb) / span for w, p in self._ilts._semantic_points.items()}
        self._scores = []
        self._q_hat = math.inf

    def _lhs(self, context: list[float]) -> tuple[str, ...]:
        sp = self._ilts._semantic_points
        return tuple(self._ilts._fuzzify(v, sp) for v in context)

    def _consequents(self, context: list[float], rules: dict | None = None) -> list[str]:
        lhs = self._lhs(context)
        if rules is None:
            rules = self._ilts._rules
        if lhs in rules:
            return rules[lhs]
        if not self.soft_match or not rules:
            return [lhs[-1]]
        nu = self._nu
        best, rhs = math.inf, []
        for key, cons in rules.items():
            d = sum((nu[a] - nu[b]) ** 2 for a, b in zip(lhs, key))
            if d < best - 1e-12:
                best, rhs = d, list(cons)
            elif abs(d - best) <= 1e-12:
                rhs.extend(cons)
        return rhs

    def base_range(self, context: list[float], rules: dict | None = None) -> tuple[int, int]:
        """Đoạn ô cơ sở ``[l, u]`` (0-based) cho một ngữ cảnh ``order`` giá trị."""
        if self._lattice is None:
            raise RuntimeError("Gọi fit() trước.")
        cons = self._consequents(context, rules)
        if self.base == "point":
            sp = self._ilts._semantic_points
            y_hat = sum(sp[w] for w in cons) / len(cons)
            i = self._lattice.index_clamped(y_hat)
            return i, i
        idx = [self._index[w] for w in cons]
        return min(idx), max(idx)

    def point_forecast(self, context: list[float]) -> float:
        sp = self._ilts._semantic_points
        cons = self._consequents(context)
        return sum(sp[w] for w in cons) / len(cons)

    # ------------------------------------------------------------------
    # Điểm số & hiệu chuẩn
    # ------------------------------------------------------------------
    def score(self, context: list[float], y: float) -> float:
        l, u = self.base_range(context)
        return self._lattice.score(l, u, y, self.variant)

    def _pairs(self, values: list[float]):
        for i in range(self.order, len(values)):
            yield values[i - self.order:i], values[i]

    def calibrate(self, values: list[float]) -> float:
        """Tính điểm số trên ``values`` (có ``order`` điểm ngữ cảnh ở đầu) và ``q̂``."""
        self._scores = [self.score(ctx, y) for ctx, y in self._pairs(values)]
        self._q_hat = conformal_quantile(self._scores, self.alpha)
        return self._q_hat

    def calibrate_pairs(self, contexts: list[list[float]], ys: list[float]) -> float:
        """Như :meth:`calibrate` nhưng nhận các cặp (ngữ cảnh, giá trị) độc lập."""
        self._scores = [self.score(c, y) for c, y in zip(contexts, ys)]
        self._q_hat = conformal_quantile(self._scores, self.alpha)
        return self._q_hat

    # ------------------------------------------------------------------
    # Dự báo dải
    # ------------------------------------------------------------------
    def threshold(self) -> float:
        """Ngưỡng dùng cho một điểm test: ``q̂`` cố định, hoặc rút ngẫu nhiên nếu ``randomized``."""
        if self.randomized:
            return randomized_conformal_quantile(self._scores, self.alpha, self._rng.random())
        return self._q_hat

    def predict_band(self, context: list[float], q: float | None = None) -> LinguisticBand:
        l, u = self.base_range(context)
        return self._lattice.band(l, u, self.threshold() if q is None else q, self.variant)

    def describe(self, band: LinguisticBand) -> list[str]:
        """Tập từ thô nhất phủ đúng dải (độ dài mô tả ngôn ngữ)."""
        return self._lattice.describe_band(band)

    def forecast_external_bands(self, values: list[float]) -> list[LinguisticBand]:
        """Dải split-conformal cho ``values[order:]`` (q̂ cố định từ :meth:`calibrate`)."""
        return [self.predict_band(ctx) for ctx, _ in self._pairs(values)]

    def jackknife_plus(
        self,
        contexts: list[list[float]],
        ys: list[float],
        test_contexts: list[list[float]],
    ) -> list[LinguisticBand]:
        """Dải jackknife+ (nested, Gupta et al. 2022) cho chuỗi ngắn — không cần tập hiệu chuẩn.

        Cơ sở luật được dựng lại **từ các cặp** ``(contexts[i], ys[i])`` (đa tập chuyển trạng
        thái); mô hình bỏ cặp ``i`` chỉ khác đúng một chuyển trạng thái nên mỗi lần "refit"
        là O(1). Với điểm test ``x``, ô ``j`` được nhận nếu
        ``#{i : S₋ᵢ(x, j) > R_i} < (1 − α)(n + 1)`` với ``R_i`` là điểm số LOO của cặp ``i``;
        dải trả về là bao liền kề của các ô được nhận (chứa tập jackknife+ nên vẫn hợp lệ).
        Bảo đảm ``≥ 1 − 2α`` khi các cặp hoán đổi được (Barber et al. 2021). Cần gọi
        :meth:`fit` trước để cố định miền, ngữ nghĩa và lưới.
        """
        if self._lattice is None:
            raise RuntimeError("Gọi fit() trước.")
        lat = self._lattice
        lhss = [self._lhs(c) for c in contexts]
        labels = [self._ilts._fuzzify(y, self._ilts._semantic_points) for y in ys]
        rules: dict[tuple[str, ...], list[str]] = {}
        for lhs, rhs in zip(lhss, labels):
            rules.setdefault(lhs, []).append(rhs)

        def loo_rules(i: int) -> dict:
            out = dict(rules)
            rest = list(rules[lhss[i]])
            rest.remove(labels[i])
            if rest:
                out[lhss[i]] = rest
            else:
                del out[lhss[i]]
            return out

        n = len(ys)
        loo = [loo_rules(i) for i in range(n)]
        r = []
        for i in range(n):
            l, u = self.base_range(contexts[i], loo[i])
            r.append(lat.score(l, u, ys[i], self.variant))
        mids = [c.mid for c in lat.cells]
        limit = (1.0 - self.alpha) * (n + 1)
        bands: list[LinguisticBand] = []
        full_cache: dict[tuple[str, ...], tuple[int, int]] = {}
        for x in test_contexts:
            lhs_x = self._lhs(x)
            if lhs_x not in full_cache:
                full_cache[lhs_x] = self.base_range(x, rules)
            ranges = [self.base_range(x, loo[i]) if lhss[i] == lhs_x else full_cache[lhs_x]
                      for i in range(n)]
            accepted = []
            for j, mid in enumerate(mids):
                exceed = sum(1 for (l, u), ri in zip(ranges, r)
                             if lat.score(l, u, mid, self.variant) > ri)
                if exceed < limit:
                    accepted.append(j)
            if not accepted:
                l, u = full_cache[lhs_x]
                accepted = [l, u]
            lo_i, hi_i = min(accepted), max(accepted)
            bands.append(LinguisticBand(lat.cells[lo_i].lo, lat.cells[hi_i].hi,
                                        lat.words[lo_i], lat.words[hi_i], float(hi_i - lo_i)))
        return bands

    def forecast_online(
        self,
        values: list[float],
        gamma: float = 0.01,
        window: int | None = None,
    ) -> tuple[list[LinguisticBand], AdaptiveQuantile]:
        """Dải ACI một bước, cập nhật sau mỗi giá trị thật.

        Khởi tạo bằng các điểm số hiệu chuẩn (nếu có). Khi ``α_t ≥ 1`` ACI trả
        ``q̂ = −1`` (dải rỗng, tính là trượt); dải trả về khi đó có độ rộng 0.
        """
        aci = AdaptiveQuantile(self.alpha, gamma, init_scores=self._scores, window=window)
        bands: list[LinguisticBand] = []
        for ctx, y in self._pairs(values):
            q = aci.threshold()
            l, u = self.base_range(ctx)
            if q < 0:
                c = self._lattice.cells[l]
                band = LinguisticBand(c.lo, c.lo, "∅", "∅", q)
            else:
                band = self._lattice.band(l, u, q, self.variant)
            bands.append(band)
            aci.update(self._lattice.score(l, u, y, self.variant), q)
        return bands, aci

    # ------------------------------------------------------------------
    # Truy cập
    # ------------------------------------------------------------------
    @property
    def q_hat(self) -> float:
        return self._q_hat

    @property
    def scores(self) -> list[float]:
        return list(self._scores)

    @property
    def lattice(self) -> LinguisticLattice:
        if self._lattice is None:
            raise RuntimeError("Gọi fit() trước.")
        return self._lattice

    @property
    def ilts(self) -> ILTSModel:
        return self._ilts


# ----------------------------------------------------------------------
# Chọn mức k mà vẫn giữ bảo đảm độ phủ
# ----------------------------------------------------------------------
def _mean_width(model: HACPLTSModel, values: list[float]) -> float:
    span = model.lattice.ub - model.lattice.lb
    total, n = 0.0, 0
    for ctx, _ in model._pairs(values):
        b = model.predict_band(ctx, q=model.q_hat)
        total += (min(b.hi, model.lattice.ub) - max(b.lo, model.lattice.lb)) / span
        n += 1
    return total / n if n else math.inf


def select_level(
    params: HAParams,
    levels: list[int],
    train: list[float],
    cal: list[float],
    lb: float,
    ub: float,
    method: str = "split",
    **kwargs,
) -> tuple[HACPLTSModel, dict[int, float]]:
    """Chọn k cho HACP-LTS theo độ rộng trung bình, giữ bảo đảm độ phủ (theory.tex Prop. 4).

    - ``"split"``: chia ``cal`` (đã gồm ``order`` điểm ngữ cảnh ở đầu) làm hai nửa
      theo thời gian; chọn k có độ rộng nhỏ nhất khi hiệu chuẩn trên nửa đầu, rồi
      hiệu chuẩn lại k đã chọn **chỉ trên nửa sau** — lựa chọn độc lập với tập hiệu
      chuẩn cuối nên Định lý 1 giữ nguyên.
    - ``"bonferroni"``: hiệu chuẩn mọi k ở mức ``α/|K|`` trên toàn bộ ``cal``; theo
      bất đẳng thức hợp, mọi dải đồng thời phủ với xác suất ``≥ 1 − α`` nên chọn k
      tuỳ ý theo dữ liệu vẫn hợp lệ.

    Trả về (mô hình đã hiệu chuẩn với k được chọn, {k: độ rộng chuẩn hoá dùng để chọn}).
    """
    if method not in {"split", "bonferroni"}:
        raise ValueError("method phải là 'split' hoặc 'bonferroni'.")
    order = kwargs.get("order", 1)
    alpha = kwargs.pop("alpha", 0.1)
    widths: dict[int, float] = {}
    models: dict[int, HACPLTSModel] = {}

    if method == "split":
        mid = order + (len(cal) - order) // 2
        part_a, part_b = cal[:mid], cal[mid - order:]
        for k in levels:
            m = HACPLTSModel(params, level=k, alpha=alpha, **kwargs)
            m.fit(train, lb, ub)
            m.calibrate(part_a)
            widths[k] = _mean_width(m, part_a)
            models[k] = m
        best = min(levels, key=lambda k: widths[k])
        chosen = models[best]
        chosen.calibrate(part_b)
        return chosen, widths

    for k in levels:
        m = HACPLTSModel(params, level=k, alpha=alpha / len(levels), **kwargs)
        m.fit(train, lb, ub)
        m.calibrate(cal)
        widths[k] = _mean_width(m, cal)
        models[k] = m
    best = min(levels, key=lambda k: widths[k])
    return models[best], widths
