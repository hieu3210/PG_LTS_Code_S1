"""Độ đo đánh giá chất lượng phân cụm / regime cho CLTS.

Bổ sung cho các độ đo dự báo (MAE/RMSE/MAPE trong lts.metrics):

    silhouette(Z, labels)        — độ tách cụm trong không gian đặc trưng.
    regime_purity(...)           — mức độ thuần nhất ngữ nghĩa của mỗi regime
                                   (entropy của phân phối từ RHS, chuẩn hóa).
    cluster_stability(...)       — độ ổn định regime khi thay seed (ARI trung bình).
    regime_transition_matrix(...) — ma trận chuyển trạng thái giữa các regime
                                   theo thời gian (đo tính dính/persistence).
    select_k(...)                — chọn K theo silhouette (model selection).

Tất cả độc lập với mô hình; nhận đầu vào là nhãn/đặc trưng đã tính sẵn.
"""

from __future__ import annotations

import math

import numpy as np


def silhouette(Z: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score trung bình (cao hơn = cụm tách tốt hơn).

    Trả về 0.0 nếu chỉ có 1 cụm hoặc không đủ điểm.
    """
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2 or len(Z) < 3:
        return 0.0
    from sklearn.metrics import silhouette_score

    try:
        return float(silhouette_score(Z, labels))
    except ValueError:
        return 0.0


def regime_purity(
    window_regimes: np.ndarray,
    rhs_words: list[str],
    sqm: dict[str, float] | None = None,
) -> float:
    """Độ thuần nhất ngữ nghĩa trung bình của các regime ∈ [0, 1].

    Với mỗi regime, ta xét phân phối các từ RHS (hệ quả dự báo) của những cửa
    sổ thuộc regime đó. Regime "tốt" gom các trạng thái có cùng hành vi → phân
    phối RHS tập trung → entropy thấp → purity cao.

        purity_k = 1 - H(p_k) / log(V_k)

    trong đó H là entropy Shannon, V_k = số từ RHS riêng biệt xuất hiện. Trả về
    trung bình có trọng số theo kích thước regime.

    Parameters
    ----------
    window_regimes : np.ndarray
        Nhãn regime của từng cửa sổ train (n,).
    rhs_words : list[str]
        Từ RHS tương ứng từng cửa sổ (cùng độ dài).
    sqm : dict[str, float] | None
        (Không bắt buộc) — giữ chữ ký mở rộng; hiện dùng entropy thuần.
    """
    window_regimes = np.asarray(window_regimes)
    n = len(window_regimes)
    if n == 0:
        return 0.0
    total = 0.0
    for k in sorted(set(window_regimes.tolist())):
        idx = np.where(window_regimes == k)[0]
        words = [rhs_words[i] for i in idx]
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        V = len(counts)
        m = len(words)
        if V <= 1:
            purity = 1.0
        else:
            H = -sum((c / m) * math.log(c / m) for c in counts.values())
            purity = 1.0 - H / math.log(V)
        total += purity * (m / n)
    return total


def cluster_stability(
    Z: np.ndarray,
    method: str,
    n_clusters: int,
    seeds: list[int] | None = None,
) -> float:
    """Độ ổn định regime: ARI trung bình giữa các lần phân cụm khác seed.

    1.0 = phân cụm hoàn toàn ổn định; ~0 = ngẫu nhiên.
    """
    from itertools import combinations

    from sklearn.metrics import adjusted_rand_score

    from lts.clustering.algorithms import make_clusterer

    seeds = seeds or [0, 1, 2, 3, 4]
    if len(Z) < n_clusters + 1 or n_clusters < 2:
        return 1.0
    labelings = []
    for s in seeds:
        try:
            lab = make_clusterer(method, n_clusters=n_clusters, seed=s).fit(Z).predict(Z)
            labelings.append(lab)
        except Exception:
            continue
    if len(labelings) < 2:
        return 1.0
    scores = [
        adjusted_rand_score(a, b) for a, b in combinations(labelings, 2)
    ]
    return float(np.mean(scores)) if scores else 1.0


def regime_transition_matrix(window_regimes: np.ndarray, K: int) -> np.ndarray:
    """Ma trận chuyển regime theo thời gian (K×K), chuẩn hóa theo hàng.

    Phần tử [i, j] = P(regime tiếp theo = j | regime hiện tại = i). Đường chéo
    lớn → regime "dính" (persistent), phù hợp với giả thuyết regime-switching.
    """
    window_regimes = np.asarray(window_regimes)
    M = np.zeros((K, K), dtype=float)
    for a, b in zip(window_regimes[:-1], window_regimes[1:]):
        M[int(a), int(b)] += 1.0
    row = M.sum(axis=1, keepdims=True)
    return np.divide(M, row, out=np.zeros_like(M), where=row > 0)


def select_k(
    Z: np.ndarray,
    method: str,
    k_range: range,
    seed: int = 0,
) -> tuple[int, dict[int, float]]:
    """Chọn K tối ưu theo silhouette score.

    Returns
    -------
    best_k : int
    scores : dict[int, float]   — silhouette theo từng K (để vẽ/ablation).
    """
    from lts.clustering.algorithms import make_clusterer

    scores: dict[int, float] = {}
    for k in k_range:
        if k < 2 or len(Z) <= k:
            continue
        try:
            lab = make_clusterer(method, n_clusters=k, seed=seed).fit(Z).predict(Z)
            scores[k] = silhouette(Z, lab)
        except Exception:
            continue
    if not scores:
        return 1, {}
    best_k = max(scores, key=scores.get)
    return best_k, scores
