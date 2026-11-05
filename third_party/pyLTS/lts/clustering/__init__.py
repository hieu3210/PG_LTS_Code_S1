"""lts.clustering — Semantic-regime discovery and regime-conditioned forecasting.

Mở rộng lõi LTS/Hedge-Algebra (2020–2023) theo hướng phân cụm ngữ nghĩa
(semantic regime discovery). Pipeline:

    Data → Linguistic Transform (HA) → Representation → Clustering → Forecasting

Đóng góp chính — **CLTS (Clustered Linguistic Time Series)**:
    Mỗi cửa sổ trạng thái ngôn ngữ (LHS) được biểu diễn trong không gian
    ngữ nghĩa của Hedge Algebra, gom cụm thành K "regime", rồi dự báo có
    điều kiện theo regime (hard-assignment hoặc soft/mixture-of-experts).

Mô hình thống nhất các công trình trước như trường hợp đặc biệt:
    K=1, exact-match   → LTS (2020)
    K=1, kernel-weight → SW-LTS
    K>1                → CLTS (đề xuất mới)

Các thành phần (mỗi module độc lập, có thể thay thế):
    representation.LinguisticRepresentation — window → feature vector
    algorithms.*                            — KMeans / GMM / FuzzyCMeans
    clts_model.CLTSModel                    — regime-conditioned forecaster
    metrics.*                               — silhouette / purity / stability
"""

from lts.clustering.representation import LinguisticRepresentation
from lts.clustering.algorithms import (
    ClusterModel,
    KMeansCluster,
    GMMCluster,
    FuzzyCMeans,
    make_clusterer,
)
from lts.clustering.clts_model import CLTSModel
from lts.clustering.regime_switching import FuzzyRegimeSwitchingLTS
from lts.clustering.efrs_lts import EFRSLTSModel, EFRSConfig
from lts.clustering.online_efrs import OnlineEFRSLTS
from lts.clustering.metrics import (
    silhouette,
    regime_purity,
    cluster_stability,
    regime_transition_matrix,
    select_k,
)

__all__ = [
    "LinguisticRepresentation",
    "ClusterModel",
    "KMeansCluster",
    "GMMCluster",
    "FuzzyCMeans",
    "make_clusterer",
    "CLTSModel",
    "FuzzyRegimeSwitchingLTS",
    "EFRSLTSModel",
    "EFRSConfig",
    "OnlineEFRSLTS",
    "silhouette",
    "regime_purity",
    "cluster_stability",
    "regime_transition_matrix",
    "select_k",
]
