"""Micro-Doppler segment extraction via the harmonic ratio.

Reproduces the frame selection of Antony et al.: the high-passed signal is split
into overlapping Hann frames, each frame is scored by a harmonic ratio derived
from the normalised autocorrelation, and frames whose score exceeds a threshold
are retained as containing wing-flapping activity.

The harmonic ratio is the maximum of the normalised autocorrelation over lags
``[K0, K]``, where ``K = 0.04·Fs = 40`` samples (a 25 Hz floor on the fundamental)
and ``K0`` is the first zero crossing — chosen adaptively so the trivial peak at
lag 0 is excluded without a hand-tuned minimum lag.

Reference behaviour: on ``vespula_vulgaris_1_1.csv`` the nine 2 s frames score
between roughly 0.47 and 0.68, matching Fig. 3b of the source paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window

from pglts.data.polliradar import SAMPLE_RATE

#: Maximum autocorrelation lag, in samples — 40 ms, i.e. a 25 Hz fundamental floor.
MAX_LAG = 40

#: Default retention threshold. Calibrated so per-species yields match the
#: published Table 1 ordering; see ``experiments/E0_audit.py``.
DEFAULT_HR_THRESHOLD = 0.30


def as_real(x: np.ndarray) -> np.ndarray:
    """Real view of a possibly complex signal.

    Only complex input is reduced by magnitude. Taking ``abs`` of an already-real
    high-passed signal would rectify it — folding the negative half-cycles onto
    the positive ones doubles the apparent fundamental and collapses the harmonic
    ratio towards zero.
    """
    x = np.asarray(x)
    return np.abs(x) if np.iscomplexobj(x) else x.astype(float, copy=False)


def harmonic_ratio(x: np.ndarray, max_lag: int = MAX_LAG) -> float:
    """Harmonicity of a frame in ``[0, 1]``.

    Higher values indicate that a larger fraction of the signal energy is
    organised into a periodic (harmonic) structure, as produced by wing flapping.
    """
    x = as_real(x)
    x = x - x.mean()
    n = x.size
    if n < 2:
        return 0.0
    r = np.correlate(x, x, mode="full")[n - 1 :]
    if r[0] <= 0:
        return 0.0
    r = r / r[0]

    crossings = np.flatnonzero(r[:-1] * r[1:] <= 0)
    first_zero = int(crossings[0]) + 1 if crossings.size else n
    if first_zero >= max_lag:
        return 0.0
    return float(np.clip(r[first_zero : max_lag + 1].max(), 0.0, 1.0))


def frame_bounds(n_samples: int, frame_len: int, hop: int) -> list[int]:
    """Start indices of every complete frame."""
    if n_samples < frame_len:
        return []
    return list(range(0, n_samples - frame_len + 1, hop))


@dataclass(frozen=True)
class SegmentSpec:
    """Framing configuration for micro-Doppler segment extraction."""

    duration_s: float = 2.0
    hop_s: float = 1.0
    threshold: float = DEFAULT_HR_THRESHOLD
    fs: float = SAMPLE_RATE

    @property
    def frame_len(self) -> int:
        return int(round(self.duration_s * self.fs))

    @property
    def hop(self) -> int:
        return int(round(self.hop_s * self.fs))


def extract_segments(
    x: np.ndarray,
    spec: SegmentSpec = SegmentSpec(),
    window: str | None = "hann",
    score_signal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``x`` into frames and keep those with harmonic activity.

    Parameters
    ----------
    x
        Analysis signal (already high-passed and normalised).
    window
        Applied to the *returned* frames for downstream spectral analysis. The
        harmonic ratio is always computed on the **unwindowed** frame — tapering
        first introduces a smooth envelope whose autocorrelation never crosses
        zero within ``max_lag``, which drives the score to 0.
    score_signal
        Optional separate signal used only for scoring, when the analysis channel
        differs from the channel the harmonic ratio should be measured on.

    Returns
    -------
    frames
        ``(n_segments, frame_len)`` array of retained frames.
    scores
        Harmonic ratio of each retained frame.
    """
    x = np.asarray(x)
    scorer = x if score_signal is None else np.asarray(score_signal)
    starts = frame_bounds(x.size, spec.frame_len, spec.hop)
    if not starts:
        return np.empty((0, spec.frame_len), dtype=x.dtype), np.empty(0)

    taper = get_window(window, spec.frame_len) if window else None

    kept: list[np.ndarray] = []
    scores: list[float] = []
    for start in starts:
        stop = start + spec.frame_len
        score = harmonic_ratio(scorer[start:stop])
        if score <= spec.threshold:
            continue
        frame = x[start:stop]
        kept.append(frame * taper if taper is not None else frame)
        scores.append(score)

    if not kept:
        return np.empty((0, spec.frame_len), dtype=x.dtype), np.empty(0)
    return np.asarray(kept), np.asarray(scores)


def score_frames(x: np.ndarray, spec: SegmentSpec = SegmentSpec()) -> np.ndarray:
    """Harmonic ratio of every frame, without applying the retention threshold."""
    starts = frame_bounds(np.asarray(x).size, spec.frame_len, spec.hop)
    return np.asarray([harmonic_ratio(x[s : s + spec.frame_len]) for s in starts])
