"""Signal front end: complex reconstruction, high-pass filtering, normalisation.

Follows the acquisition chain of Antony et al. (PNAS Nexus 2026): the recorded
``amplitude``/``phase`` pair is the quadrature-demodulated baseband signal, so the
complex form ``z[m] = a[m]·exp(j·φ[m])`` preserves the motion information needed
for micro-Doppler analysis. A second-order Chebyshev-I high-pass at 50 Hz removes
slow body movement and background scattering.

The ``channel`` argument selects which part of the signal reaches the feature
extractor. It exists for the confound analysis (H5): absolute reflected power is
a function of distance and body size rather than taxonomy, so a model that is
allowed to key on it can identify *individuals* instead of *species*.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import cheby1, filtfilt

from pglts.data.polliradar import SAMPLE_RATE

Channel = Literal["amplitude", "phase", "complex", "amplitude_raw"]

#: Filter design shared by every call — cheap to rebuild, but this keeps it exact.
_HPF_ORDER = 2
_HPF_RIPPLE_DB = 0.5
_HPF_CUTOFF_HZ = 50.0


def highpass_coefficients(
    cutoff_hz: float = _HPF_CUTOFF_HZ,
    fs: float = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Second-order Chebyshev-I high-pass coefficients ``(b, a)``."""
    return cheby1(_HPF_ORDER, _HPF_RIPPLE_DB, cutoff_hz / (fs / 2.0), btype="high")


def highpass(x: np.ndarray, cutoff_hz: float = _HPF_CUTOFF_HZ, fs: float = SAMPLE_RATE) -> np.ndarray:
    """Zero-phase high-pass filter. Handles real and complex input."""
    b, a = highpass_coefficients(cutoff_hz, fs)
    return filtfilt(b, a, x)


def normalise(x: np.ndarray, mode: str = "std") -> np.ndarray:
    """Make the signal invariant to absolute reflected power.

    ``"std"`` divides by the standard deviation (the default, and the only mode
    that fully removes the amplitude/RCS confound). ``"none"`` leaves the signal
    untouched and is used to *demonstrate* the confound in E8.
    """
    if mode == "none":
        return x
    if mode == "std":
        scale = float(np.std(np.abs(x)))
        return x if scale <= 0 else x / scale
    raise ValueError(f"unknown normalisation mode: {mode!r}")


def prepare(
    amplitude: np.ndarray,
    phase: np.ndarray,
    channel: Channel = "amplitude",
    normalisation: str = "std",
    cutoff_hz: float = _HPF_CUTOFF_HZ,
    fs: float = SAMPLE_RATE,
) -> np.ndarray:
    """Run the front end and return the analysis signal.

    Parameters
    ----------
    channel
        ``"amplitude"``   high-passed, normalised amplitude (default analysis path);
        ``"amplitude_raw"`` amplitude without normalisation (confound control);
        ``"phase"``       high-passed phase channel only;
        ``"complex"``     high-passed complex baseband, normalised.
    """
    if channel == "amplitude":
        return normalise(highpass(amplitude, cutoff_hz, fs), normalisation)
    if channel == "amplitude_raw":
        return highpass(amplitude, cutoff_hz, fs)
    if channel == "phase":
        return normalise(highpass(np.unwrap(phase), cutoff_hz, fs), normalisation)
    if channel == "complex":
        z = amplitude * np.exp(1j * phase)
        return normalise(highpass(z, cutoff_hz, fs), normalisation)
    raise ValueError(f"unknown channel: {channel!r}")


def instantaneous_frequency(
    amplitude: np.ndarray,
    phase: np.ndarray,
    cutoff_hz: float = _HPF_CUTOFF_HZ,
    fs: float = SAMPLE_RATE,
) -> np.ndarray:
    """Instantaneous Doppler frequency in Hz from the complex baseband phase.

    Returns an array of length ``len(amplitude) - 1``.
    """
    z = highpass(amplitude * np.exp(1j * phase), cutoff_hz, fs)
    return np.diff(np.unwrap(np.angle(z))) * fs / (2.0 * np.pi)


def perturb(
    x: np.ndarray,
    snr_db: float | None = None,
    attenuation: float | None = None,
    phase_jitter_rad: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply controlled synthetic degradation (E8 robustness sweep).

    ``snr_db`` is referenced to the power of ``x`` as supplied, so callers should
    pass the micro-Doppler component rather than the raw recording — the raw
    amplitude is DC-dominated and an SNR defined against it is not meaningful.

    ``phase_jitter_rad`` rotates a complex signal by a random phase. For a *real*
    signal there is no phase to rotate, so it degenerates to multiplication by
    ``cos(jitter)``, i.e. a small multiplicative amplitude perturbation. That is a
    weaker stressor than true phase noise and the results should be read as such.

    These perturbations characterise *synthetic sensitivity* only; they are not a
    substitute for field robustness testing.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    y = np.asarray(x).copy()

    if attenuation is not None:
        y = y * attenuation
    if phase_jitter_rad is not None:
        jitter = rng.normal(0.0, phase_jitter_rad, size=y.shape)
        y = y * np.exp(1j * jitter) if np.iscomplexobj(y) else y * np.cos(jitter)
    if snr_db is not None:
        power = float(np.mean(np.abs(y) ** 2))
        noise_power = power / (10.0 ** (snr_db / 10.0))
        if np.iscomplexobj(y):
            noise = rng.normal(0, np.sqrt(noise_power / 2), y.shape) + 1j * rng.normal(
                0, np.sqrt(noise_power / 2), y.shape
            )
        else:
            noise = rng.normal(0, np.sqrt(noise_power), y.shape)
        y = y + noise
    return y
