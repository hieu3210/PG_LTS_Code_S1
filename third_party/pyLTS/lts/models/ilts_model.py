"""I-LTS — Interval Linguistic Time Series dựa trên khoảng ngữ nghĩa của HA.

Mở rộng LTS (2020) theo trục *bất định khoảng*: thay vì ngữ nghĩa hoá bằng
**điểm ngữ nghĩa gần nhất** (argmin |s(w) − x|), I-LTS gán nhãn cho một điểm dữ
liệu theo **khoảng ngữ nghĩa** ``ℑ(w)`` của từ — điểm rơi vào khoảng nào thì mang
nhãn đó (interval-membership fuzzification). Nhờ mỗi nhãn là một *khoảng* (không
phải điểm), mô hình sinh ra **dự báo khoảng** ``[ŷ⁻, ŷ⁺]`` bên cạnh dự báo điểm.

Xem :mod:`lts.core.fuzziness_interval` cho cơ sở lý thuyết ℑ(x).

So với LTS gốc:
- Từ vựng = phân hoạch mức ``level`` (2^level khoảng phủ kín [0,1]).
- Ngữ nghĩa hoá = interval-membership (khác argmin điểm) → labeling khác → LLRG
  khác → dự báo khác.
- Bổ sung dự báo khoảng; đánh giá được bằng PICP/MPIW.
"""

from __future__ import annotations

from lts.core.fuzziness_interval import FuzzinessIntervals, Interval
from lts.core.hedge_algebras import HAParams, HedgeAlgebra
from lts.models.base import ForecastResult
from lts.models.lts_model import LTSModel


class ILTSModel(LTSModel):
    """Interval Linguistic Time Series.

    Parameters
    ----------
    params : HAParams
        Tham số fuzziness (theta, alpha).
    level : int
        Mức phân hoạch khoảng ngữ nghĩa. level=2 → 4 nhãn, level=3 → 8 nhãn,
        level=4 → 16 nhãn (2^level nhãn phủ kín [0,1]).
    order : int
        Bậc mô hình.
    use_repeat : bool
        Cho phép RHS trùng lặp trong cùng LLRG.
    band : {"mean", "union"}
        Cách gộp khoảng của tập RHS thành khoảng dự báo:
        - "mean"  : trung bình các cận (khoảng hẹp, hiệu chỉnh).
        - "union" : min cận dưới / max cận trên (khoảng bao, phủ rộng).
    """

    def __init__(
        self,
        params: HAParams,
        level: int = 3,
        order: int = 1,
        use_repeat: bool = True,
        band: str = "mean",
        encoding: str = "containment",
    ) -> None:
        if band not in {"mean", "union"}:
            raise ValueError("band phải là 'mean' hoặc 'union'.")
        if encoding not in {"containment", "nearest"}:
            raise ValueError("encoding phải là 'containment' hoặc 'nearest'.")
        self.params = params
        self.level = level
        self.order = order
        self.use_repeat = use_repeat
        self.band = band
        self.encoding = encoding

        self.ha = HedgeAlgebra(params)
        self._fi = FuzzinessIntervals(self.ha)
        # Từ vựng = phân hoạch mức `level`, sắp theo SQM tăng dần
        self._vocab: list[str] = self._fi.partition_words(level)
        self._words = list(self._vocab)

        # State điền bởi fit()
        self._data: list[float] = []
        self._lb: float = 0.0
        self._ub: float = 1.0
        self._semantic_points: dict[str, float] = {}
        self._norm_intervals: dict[str, Interval] = {}
        self._intervals: dict[str, Interval] = {}   # miền dữ liệu [lb, ub]
        self._labels: list[str] = []
        self._rules: dict[tuple[str, ...], list[str]] = {}
        self._forecasted: list[float] = []
        self._forecasted_intervals: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Bước 4 (ghi đè): ngữ nghĩa hoá theo KHOẢNG thay vì điểm gần nhất
    # ------------------------------------------------------------------
    def _fuzzify(self, value: float, semantic_points: dict[str, float]) -> str:
        """Gán nhãn cho ``value``.

        - encoding="containment" (mặc định): nhãn w có ℑ(w) chứa ``value``.
        - encoding="nearest": nhãn có điểm ngữ nghĩa gần nhất (dùng cho ablation,
          cô lập tác động của cơ chế ngữ nghĩa hoá trên CÙNG một phân hoạch).
        """
        if self.encoding == "nearest":
            return min(self._vocab, key=lambda w: abs(semantic_points[w] - value))
        for w in self._vocab:  # vocab tăng dần → khoảng liền kề tăng dần
            if self._intervals[w].contains(value):
                return w
        # ngoài biên phân hoạch → kẹp về nhãn cực trị
        if value < self._intervals[self._vocab[0]].lo:
            return self._vocab[0]
        return self._vocab[-1]

    # ------------------------------------------------------------------
    # Bước 6 (bổ sung): dự báo KHOẢNG cho một trạng thái LHS
    # ------------------------------------------------------------------
    def _forecast_interval_one(self, lhs: tuple[str, ...]) -> tuple[float, float]:
        if lhs in self._rules:
            rhs = self._rules[lhs]
            los = [self._intervals[w].lo for w in rhs]
            his = [self._intervals[w].hi for w in rhs]
            if self.band == "union":
                return min(los), max(his)
            return sum(los) / len(los), sum(his) / len(his)
        iv = self._intervals[lhs[-1]]
        return iv.lo, iv.hi

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------
    def fit(self, data: list[float], lb: float, ub: float) -> None:
        self._data = list(data)
        self._lb = lb
        self._ub = ub

        self._semantic_points = self._compute_semantic_points()
        self._norm_intervals = {w: self._fi.interval(w) for w in self._vocab}
        self._intervals = {
            w: iv.scaled(lb, ub) for w, iv in self._norm_intervals.items()
        }

        self._labels = self._label_data(self._data, self._semantic_points)
        self._rules = self._build_rules(self._labels, self.order, self.use_repeat)

        self._forecasted = [
            self._forecast_one(
                tuple(self._labels[i - self.order : i]),
                self._rules,
                self._semantic_points,
            )
            for i in range(self.order, len(self._data))
        ]
        self._forecasted_intervals = [
            self._forecast_interval_one(tuple(self._labels[i - self.order : i]))
            for i in range(self.order, len(self._data))
        ]

    def predict_intervals(self) -> list[tuple[float, float]]:
        """Trả về danh sách khoảng dự báo ``[(lo, hi), ...]`` (căn với predict())."""
        return list(self._forecasted_intervals)

    def forecast_next_interval(self) -> tuple[float, float]:
        """Khoảng dự báo 1 bước ngoài mẫu cho điểm kế tiếp."""
        return self._forecast_interval_one(tuple(self._labels[-self.order :]))

    def forecast_external_intervals(
        self, values: list[float]
    ) -> list[tuple[float, float]]:
        """Khoảng dự báo 1-bước cho chuỗi NGOÀI, dùng cơ sở luật đã fit (không refit).

        Song song với :meth:`LTSModel.forecast_external` (dự báo điểm) nhưng trả
        về khoảng; căn chỉnh với ``values[order:]``. Dùng cho đánh giá out-of-sample.
        """
        labels = self._label_data(list(values), self._semantic_points)
        return [
            self._forecast_interval_one(tuple(labels[i - self.order : i]))
            for i in range(self.order, len(values))
        ]

    def get_result(self) -> ForecastResult:
        return ForecastResult(
            data=list(self._data),
            labels=list(self._labels),
            semantic_points=dict(self._semantic_points),
            rules={lhs: list(rhs) for lhs, rhs in self._rules.items()},
            forecasted=list(self._forecasted),
            order=self.order,
            lb=self._lb,
            ub=self._ub,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def semantic_intervals(self) -> dict[str, Interval]:
        """Ánh xạ word → khoảng ngữ nghĩa ℑ(word) trong miền [lb, ub]."""
        return dict(self._intervals)

    @property
    def forecasted_intervals(self) -> list[tuple[float, float]]:
        return list(self._forecasted_intervals)
