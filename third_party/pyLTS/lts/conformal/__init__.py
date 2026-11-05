"""HACP — Hedge-Algebraic Conformal Prediction: hiệu chuẩn conformal trên lưới ngôn ngữ của ĐSGT."""

from lts.conformal.calibration import AdaptiveQuantile, conformal_quantile
from lts.conformal.design import plugin_alpha, quantile_cell_bound, resolution_level, select_alpha, universe
from lts.conformal.lattice import BinLattice, LinguisticBand, LinguisticLattice
from lts.conformal.wrapper import HACP

__all__ = [
    "AdaptiveQuantile",
    "conformal_quantile",
    "LinguisticBand",
    "LinguisticLattice",
    "BinLattice",
    "HACP",
    "resolution_level",
    "select_alpha",
    "plugin_alpha",
    "quantile_cell_bound",
    "universe",
]
