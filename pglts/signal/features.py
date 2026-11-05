"""Window-level physically-grounded micro-Doppler features.

Every feature is a quantity with a signal-processing or biomechanical meaning —
wingbeat frequency, harmonic structure, energy distribution, spectral dynamics —
so that linguistic states derived from them ("F0 is *Rather Large*", "harmonicity
is *Very Large*") remain interpretable at the rule level.

A segment is split into short windows and each window yields one feature vector;
stacking them gives the multivariate trajectory that PG-LTS semantises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from librosa.filters import mel
from scipy.fftpack import dct
from scipy.signal import get_window

from pglts.data.polliradar import SAMPLE_RATE
from pglts.signal.segmentation import harmonic_ratio

#: Analysis band. The 50 Hz high-pass makes anything below ~60 Hz filter
#: transition rather than signal; 480 Hz stays clear of the 500 Hz Nyquist edge.
BAND_LO, BAND_HI = 60.0, 480.0

#: Energy bands, following the band-power features of the source paper.
BANDS: tuple[tuple[float, float], ...] = (
    (60, 100),
    (100, 150),
    (150, 200),
    (200, 250),
    (250, 300),
    (300, 350),
    (350, 480),
)

N_MFCC = 13
N_MEL = 20

#: Wingbeat search range for fundamental estimation, in Hz. Hymenopteran
#: pollinators flap between roughly 60 and 350 Hz.
F0_LO, F0_HI = 60.0, 350.0


def _band_name(lo: float, hi: float) -> str:
    return f"bp_{int(lo)}_{int(hi)}"


BAND_NAMES = tuple(_band_name(lo, hi) for lo, hi in BANDS)
MFCC_NAMES = tuple(f"mfcc{i}" for i in range(N_MFCC))

#: Named feature groups, used by the ablation in E6.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "f0": ("spec_f0", "ncf_f0", "cep_f0"),
    "harmonic": ("harmonic_ratio", "h2_h1", "h3_h1", "hbr"),
    "spectral": ("centroid", "entropy", "flatness", "rolloff95"),
    "band": BAND_NAMES,
    "mfcc": MFCC_NAMES,
}

ALL_FEATURES: tuple[str, ...] = tuple(n for g in FEATURE_GROUPS.values() for n in g)

#: The compact set proposed for PG-LTS — one representative per physical concept,
#: mirroring the feature vector of the research-direction draft. Keeping this
#: small is what makes the extracted rules readable.
PGLTS_CORE: tuple[str, ...] = (
    "spec_f0",
    "harmonic_ratio",
    "h2_h1",
    "h3_h1",
    "bp_100_150",
    "bp_250_300",
    "bp_300_350",
    "centroid",
    "entropy",
    "hbr",
)


@dataclass
class FeatureExtractor:
    """Computes the window-level feature matrix of one segment.

    Parameters
    ----------
    window_s, hop_s
        Short-window length and stride, in seconds. Controls the time resolution
        of the feature trajectory: too short degrades spectral stability, too long
        erases within-segment variation.
    names
        Which features to emit, in order. Defaults to :data:`PGLTS_CORE`.
    """

    window_s: float = 0.2
    hop_s: float = 0.1
    fs: float = SAMPLE_RATE
    names: tuple[str, ...] = PGLTS_CORE
    _mel_fb: np.ndarray | None = field(default=None, init=False, repr=False)
    _freqs: np.ndarray | None = field(default=None, init=False, repr=False)
    _taper: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        unknown = set(self.names) - set(ALL_FEATURES)
        if unknown:
            raise ValueError(f"unknown feature names: {sorted(unknown)}")

    # -- geometry -------------------------------------------------------
    @property
    def window_len(self) -> int:
        return int(round(self.window_s * self.fs))

    @property
    def hop(self) -> int:
        return int(round(self.hop_s * self.fs))

    @property
    def n_fft(self) -> int:
        """Next power of two at least as long as the window, min 64 bins."""
        return max(64, 1 << (self.window_len - 1).bit_length())

    def _ensure_cache(self) -> None:
        if self._freqs is not None:
            return
        self._freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.fs)
        self._taper = get_window("hann", self.window_len)
        self._mel_fb = mel(
            sr=self.fs, n_fft=self.n_fft, n_mels=N_MEL, fmin=BAND_LO, fmax=BAND_HI
        )

    # -- single window --------------------------------------------------
    def _window_features(self, x: np.ndarray) -> dict[str, float]:
        self._ensure_cache()
        assert self._freqs is not None and self._taper is not None and self._mel_fb is not None

        x = np.asarray(x, dtype=float)
        x = x - x.mean()
        spectrum = np.abs(np.fft.rfft(x * self._taper, self.n_fft)) ** 2

        freqs = self._freqs
        in_band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
        p_band = spectrum[in_band]
        f_band = freqs[in_band]
        total = float(p_band.sum()) + 1e-20
        p_norm = p_band / total

        out: dict[str, float] = {}

        # --- fundamental frequency, three independent estimators ---
        # (spectral peak, normalised autocorrelation, real cepstrum)
        searchable = (f_band >= F0_LO) & (f_band <= F0_HI)
        # Spectral-peak estimate: the strongest in-band bin. Deliberately *not*
        # called PEF: the source paper's PEF-F0 is a pitch-estimation-filter
        # output, a different estimator. Naming it pef_f0 would misattribute.
        spec_f0 = float(f_band[searchable][np.argmax(p_band[searchable])]) if searchable.any() else 0.0
        out["spec_f0"] = spec_f0
        out["ncf_f0"] = _ncf_f0(x, self.fs)
        out["cep_f0"] = _cepstral_f0(spectrum, self.fs, self.n_fft)

        # --- harmonic structure ---
        out["harmonic_ratio"] = harmonic_ratio(x)
        h1 = _peak_near(p_band, f_band, spec_f0)
        h2 = _peak_near(p_band, f_band, 2 * spec_f0)
        h3 = _peak_near(p_band, f_band, 3 * spec_f0)
        out["h2_h1"] = float(np.log10(h2 / (h1 + 1e-20) + 1e-6))
        out["h3_h1"] = float(np.log10(h3 / (h1 + 1e-20) + 1e-6))
        # Harmonic-to-background ratio: energy at the first three harmonics
        # against the median (background) level — a micro-Doppler clarity score.
        background = float(np.median(p_band)) + 1e-20
        out["hbr"] = float(np.log10((h1 + h2 + h3) / (3.0 * background) + 1e-6))

        # --- spectral shape ---
        out["centroid"] = float((f_band * p_norm).sum())
        out["entropy"] = float(-(p_norm * np.log(p_norm + 1e-20)).sum() / np.log(p_norm.size))
        out["flatness"] = float(
            np.exp(np.mean(np.log(p_band + 1e-20))) / (np.mean(p_band) + 1e-20)
        )
        cumulative = np.cumsum(p_norm)
        idx = int(np.searchsorted(cumulative, 0.95))
        out["rolloff95"] = float(f_band[min(idx, f_band.size - 1)])

        # --- band powers (fractions, so amplitude-invariant) ---
        for lo, hi in BANDS:
            sel = (f_band >= lo) & (f_band < hi)
            out[_band_name(lo, hi)] = float(p_band[sel].sum() / total)

        # --- cepstral coefficients ---
        if any(n.startswith("mfcc") for n in self.names):
            log_mel = np.log10(self._mel_fb @ spectrum + 1e-12)
            coeffs = dct(log_mel, type=2, norm="ortho")[:N_MFCC]
            for i, c in enumerate(coeffs):
                out[f"mfcc{i}"] = float(c)

        return out

    # -- segment --------------------------------------------------------
    def transform(self, segment: np.ndarray) -> np.ndarray:
        """Feature trajectory of one segment: ``(n_windows, n_features)``."""
        segment = np.asarray(segment)
        if np.iscomplexobj(segment):
            segment = np.abs(segment)
        n = segment.size
        starts = range(0, max(n - self.window_len + 1, 0), self.hop)
        rows = [
            [self._window_features(segment[s : s + self.window_len])[name] for name in self.names]
            for s in starts
        ]
        if not rows:
            return np.empty((0, len(self.names)), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)


def _peak_near(power: np.ndarray, freqs: np.ndarray, target: float) -> float:
    """Peak power in a +/-1-bin neighbourhood of ``target``, 0 if out of band."""
    if target <= 0 or target > freqs[-1]:
        return 0.0
    i = int(np.argmin(np.abs(freqs - target)))
    return float(power[max(0, i - 1) : i + 2].max())


def _ncf_f0(x: np.ndarray, fs: float) -> float:
    """Fundamental from the normalised autocorrelation peak."""
    x = x - x.mean()
    n = x.size
    r = np.correlate(x, x, mode="full")[n - 1 :]
    if r[0] <= 0:
        return 0.0
    lo = max(int(fs / F0_HI), 1)
    hi = min(int(fs / F0_LO), n - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(r[lo : hi + 1]))
    return float(fs / lag)


def _cepstral_f0(spectrum: np.ndarray, fs: float, n_fft: int) -> float:
    """Fundamental from the real cepstrum quefrency peak."""
    cepstrum = np.fft.irfft(np.log(spectrum + 1e-20), n=n_fft)
    lo = max(int(fs / F0_HI), 1)
    hi = min(int(fs / F0_LO), n_fft // 2 - 1)
    if hi <= lo:
        return 0.0
    q = lo + int(np.argmax(cepstrum[lo : hi + 1]))
    return float(fs / q)


def add_deltas(trajectory: np.ndarray, order: int = 2) -> np.ndarray:
    """Append first- (and optionally second-) order differences along time.

    Deltas capture how fast the spectral shape evolves. SHAP analysis in the
    source paper ranked delta-MFCCs as the single most influential feature group,
    so they are available to the numeric baselines even though PG-LTS prefers to
    model dynamics through linguistic transitions instead.
    """
    if trajectory.shape[0] < 2:
        return trajectory
    parts = [trajectory]
    current = trajectory
    for _ in range(order):
        d = np.diff(current, axis=0, prepend=current[:1])
        parts.append(d)
        current = d
    return np.concatenate(parts, axis=1)
