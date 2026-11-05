"""Wingbeat-cycle detection and phase-locked (cycle-synchronous) resampling.

Motivation
----------
Free-running short windows inside a 2 s segment observe a quasi-stationary
process: the insect flaps at a near-constant rate, so a first-order transition
matrix over linguistic states is determined by the marginal state distribution
and carries no ordering information. Probes confirmed this — permuting the window
order left classification performance unchanged.

Order information, if it exists at all, lives *within* one wingbeat cycle: the
downstroke and the upstroke are biomechanically asymmetric, and that asymmetry is
species-dependent. This module therefore aligns the analysis to wingbeat phase
rather than to wall-clock time. Each detected cycle is resampled onto a fixed
number of phase bins, so the resulting sequence is indexed by *stroke phase*.

Resolution caveat
-----------------
At ``Fs = 1 kHz`` and a 76-170 Hz wingbeat, one cycle spans only 6-13 samples.
Resampling uses band-limited interpolation, which is legitimate here because the
50 Hz high-pass and the 500 Hz Nyquist limit already band-limit the signal, but
the achievable within-cycle detail is inherently modest. ``n_phase = 8`` is the
default for that reason.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from pglts.data.polliradar import SAMPLE_RATE
from pglts.signal.features import F0_HI, F0_LO

#: Channels of a cycle-synchronous trajectory.
CYCLE_CHANNELS: tuple[str, ...] = ("waveform", "envelope", "inst_freq")

DEFAULT_N_PHASE = 8


def dominant_f0(x: np.ndarray, fs: float = SAMPLE_RATE) -> float:
    """Dominant wingbeat frequency of a whole segment, in Hz.

    Returns 0.0 when no in-band peak exists.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if x.size < 8:
        return 0.0
    n_fft = 1 << (x.size - 1).bit_length()
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size), n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    sel = (freqs >= F0_LO) & (freqs <= F0_HI)
    if not sel.any():
        return 0.0
    return float(freqs[sel][np.argmax(spectrum[sel])])


def cycle_boundaries(
    x: np.ndarray,
    f0: float,
    fs: float = SAMPLE_RATE,
    relative_bandwidth: float = 0.35,
) -> np.ndarray:
    """Sample indices at which a new wingbeat cycle starts.

    The signal is narrowband-filtered around ``f0`` to isolate the fundamental,
    and cycle starts are taken where the analytic phase of that component wraps
    through zero. Working on the fundamental rather than the broadband waveform
    keeps the boundaries stable in the presence of harmonics and noise.
    """
    x = np.asarray(x, dtype=float)
    nyquist = fs / 2.0
    lo = max(f0 * (1.0 - relative_bandwidth), 5.0)
    hi = min(f0 * (1.0 + relative_bandwidth), nyquist * 0.98)
    if f0 <= 0 or hi <= lo or x.size < 16:
        return np.empty(0, dtype=int)

    sos = butter(2, [lo / nyquist, hi / nyquist], btype="band", output="sos")
    fundamental = sosfiltfilt(sos, x)
    phase = np.unwrap(np.angle(hilbert(fundamental)))

    # A new cycle begins each time the unwrapped phase passes a multiple of 2*pi.
    turns = np.floor((phase - phase[0]) / (2.0 * np.pi))
    starts = np.flatnonzero(np.diff(turns) > 0) + 1
    return starts.astype(int)


def resample_cycle(x: np.ndarray, start: int, stop: int, n_phase: int) -> np.ndarray:
    """Interpolate ``x[start:stop]`` onto ``n_phase`` uniformly spaced phase bins."""
    span = stop - start
    if span < 2:
        return np.zeros(n_phase, dtype=float)
    source = np.arange(span, dtype=float)
    target = np.linspace(0.0, span - 1.0, n_phase)
    return np.interp(target, source, np.asarray(x[start:stop], dtype=float))


def cycle_trajectory(
    amplitude_hp: np.ndarray,
    complex_hp: np.ndarray,
    n_phase: int = DEFAULT_N_PHASE,
    fs: float = SAMPLE_RATE,
    max_cycles: int | None = None,
) -> np.ndarray:
    """Phase-locked trajectory of one segment.

    Parameters
    ----------
    amplitude_hp
        High-passed, normalised amplitude channel — used for cycle detection and
        as the ``waveform`` channel.
    complex_hp
        High-passed complex baseband, supplying envelope and instantaneous Doppler.
    n_phase
        Phase bins per cycle.
    max_cycles
        Optional cap, to bound cost on long segments.

    Returns
    -------
    ndarray
        ``(n_cycles, n_phase, len(CYCLE_CHANNELS))``. Empty when no cycle could be
        detected. Each cycle is standardised independently so that absolute
        reflected power cannot leak into the representation.
    """
    amplitude_hp = np.asarray(amplitude_hp, dtype=float)
    f0 = dominant_f0(amplitude_hp, fs)
    starts = cycle_boundaries(amplitude_hp, f0, fs)
    if starts.size < 2:
        return np.empty((0, n_phase, len(CYCLE_CHANNELS)), dtype=np.float32)

    envelope = np.abs(complex_hp)
    inst_freq = np.diff(np.unwrap(np.angle(complex_hp))) * fs / (2.0 * np.pi)
    inst_freq = np.append(inst_freq, inst_freq[-1])

    pairs = list(zip(starts[:-1], starts[1:]))
    if max_cycles is not None:
        pairs = pairs[:max_cycles]

    cycles = []
    for start, stop in pairs:
        block = np.stack(
            [
                resample_cycle(amplitude_hp, start, stop, n_phase),
                resample_cycle(envelope, start, stop, n_phase),
                resample_cycle(inst_freq, start, stop, n_phase),
            ],
            axis=1,
        )
        # Per-cycle standardisation: shape of the stroke, not its magnitude.
        scale = block.std(axis=0, keepdims=True)
        scale[scale <= 0] = 1.0
        cycles.append((block - block.mean(axis=0, keepdims=True)) / scale)

    return np.asarray(cycles, dtype=np.float32)


def cycle_statistics(amplitude_hp: np.ndarray, fs: float = SAMPLE_RATE) -> dict[str, float]:
    """Summary of cycle timing: rate, period jitter, number of cycles."""
    f0 = dominant_f0(amplitude_hp, fs)
    starts = cycle_boundaries(amplitude_hp, f0, fs)
    if starts.size < 3:
        return {"f0": f0, "n_cycles": float(max(starts.size - 1, 0)), "period_cv": 0.0}
    periods = np.diff(starts).astype(float)
    return {
        "f0": f0,
        "n_cycles": float(periods.size),
        "period_cv": float(periods.std() / (periods.mean() + 1e-20)),
    }
