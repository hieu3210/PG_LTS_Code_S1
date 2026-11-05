"""Corpus-level dataset construction with on-disk caching.

Extraction over the whole corpus takes on the order of tens of seconds, so every
experiment shares cached trajectories keyed by the configuration that produced
them. Delete ``cache/`` to force a rebuild.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pglts.data.polliradar import Recording, index
from pglts.signal.cycle import DEFAULT_N_PHASE, cycle_trajectory
from pglts.signal.features import PGLTS_CORE, FeatureExtractor
from pglts.signal.preprocess import highpass, normalise, prepare
from pglts.signal.segmentation import SegmentSpec, extract_segments, harmonic_ratio


def cache_dir() -> Path:
    path = Path(__file__).resolve().parents[2] / "cache"
    path.mkdir(exist_ok=True)
    return path


@dataclass
class Dataset:
    """Segment-level dataset with the metadata needed for grouped evaluation."""

    X: np.ndarray                      # (n_segments, n_windows, n_features)
    y: np.ndarray                      # species label per segment
    groups: np.ndarray                 # "species#subject" per segment
    recordings: np.ndarray             # source file stem per segment
    scores: np.ndarray                 # harmonic ratio of the source frame
    feature_names: tuple[str, ...] = field(default=())

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_subjects(self) -> int:
        return int(np.unique(self.groups).size)

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(
            X=self.X[mask],
            y=self.y[mask],
            groups=self.groups[mask],
            recordings=self.recordings[mask],
            scores=self.scores[mask],
            feature_names=self.feature_names,
        )


def _key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def build_window_dataset(
    spec: SegmentSpec = SegmentSpec(),
    window_s: float = 0.2,
    hop_s: float = 0.1,
    feature_names: tuple[str, ...] = PGLTS_CORE,
    channel: str = "amplitude",
    normalisation: str = "std",
    records: list[Recording] | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> Dataset:
    """Free-running window trajectories for every retained micro-Doppler segment."""
    records = records if records is not None else index()
    key = _key(
        {
            "kind": "window",
            "duration": spec.duration_s,
            "hop_s_frame": spec.hop_s,
            "threshold": spec.threshold,
            "window_s": window_s,
            "hop_s": hop_s,
            "features": list(feature_names),
            "channel": channel,
            "normalisation": normalisation,
            "n_records": len(records),
        }
    )
    path = cache_dir() / f"window_{key}.npz"
    if use_cache and path.exists():
        blob = np.load(path, allow_pickle=True)
        return Dataset(
            X=blob["X"],
            y=blob["y"],
            groups=blob["groups"],
            recordings=blob["recordings"],
            scores=blob["scores"],
            feature_names=tuple(blob["feature_names"].tolist()),
        )

    extractor = FeatureExtractor(
        window_s=window_s, hop_s=hop_s, fs=spec.fs, names=tuple(feature_names)
    )
    started = time.time()
    trajectories: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    origins: list[str] = []
    scores: list[float] = []
    expected_len: int | None = None

    for rec in records:
        amplitude, phase = rec.load()
        signal = prepare(amplitude, phase, channel=channel, normalisation=normalisation, fs=spec.fs)
        scorer = normalise(highpass(amplitude, fs=spec.fs), normalisation)
        frames, frame_scores = extract_segments(signal, spec, window=None, score_signal=scorer)
        for frame, score in zip(frames, frame_scores):
            traj = extractor.transform(frame)
            if traj.shape[0] == 0:
                continue
            if expected_len is None:
                expected_len = traj.shape[0]
            elif traj.shape[0] != expected_len:
                continue
            trajectories.append(traj)
            labels.append(rec.species)
            groups.append(rec.subject_key)
            origins.append(rec.path.stem)
            scores.append(float(score))
        if verbose:
            print(f"  {rec.path.name}: {len(trajectories)} segments so far", flush=True)

    dataset = Dataset(
        X=np.asarray(trajectories, dtype=np.float32),
        y=np.asarray(labels),
        groups=np.asarray(groups),
        recordings=np.asarray(origins),
        scores=np.asarray(scores),
        feature_names=tuple(feature_names),
    )
    if verbose:
        print(f"built {len(dataset)} segments in {time.time() - started:.0f}s", flush=True)
    if use_cache:
        np.savez_compressed(
            path,
            X=dataset.X,
            y=dataset.y,
            groups=dataset.groups,
            recordings=dataset.recordings,
            scores=dataset.scores,
            feature_names=np.asarray(dataset.feature_names),
        )
    return dataset


def build_cycle_dataset(
    spec: SegmentSpec = SegmentSpec(),
    n_phase: int = DEFAULT_N_PHASE,
    max_cycles: int | None = 200,
    records: list[Recording] | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Cycle-synchronous trajectories.

    Returns ``(cycles, y, groups, recordings)`` where ``cycles[i]`` has shape
    ``(n_cycles_i, n_phase, n_channels)``. The number of cycles varies per segment,
    so this cannot be a rectangular array.
    """
    records = records if records is not None else index()
    key = _key(
        {
            "kind": "cycle",
            "duration": spec.duration_s,
            "hop_s_frame": spec.hop_s,
            "threshold": spec.threshold,
            "n_phase": n_phase,
            "max_cycles": max_cycles,
            "n_records": len(records),
        }
    )
    path = cache_dir() / f"cycle_{key}.npz"
    if use_cache and path.exists():
        blob = np.load(path, allow_pickle=True)
        return (
            list(blob["cycles"]),
            blob["y"],
            blob["groups"],
            blob["recordings"],
        )

    cycles: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    origins: list[str] = []

    for rec in records:
        amplitude, phase = rec.load()
        amp_hp = normalise(highpass(amplitude, fs=spec.fs), "std")
        cpx_hp = normalise(highpass(amplitude * np.exp(1j * phase), fs=spec.fs), "std")
        frames, _ = extract_segments(amp_hp, spec, window=None)
        starts = _retained_starts(amp_hp, spec)
        for start in starts:
            stop = start + spec.frame_len
            block = cycle_trajectory(
                amp_hp[start:stop], cpx_hp[start:stop], n_phase=n_phase, fs=spec.fs,
                max_cycles=max_cycles,
            )
            if block.shape[0] < 2:
                continue
            cycles.append(block)
            labels.append(rec.species)
            groups.append(rec.subject_key)
            origins.append(rec.path.stem)
        if verbose:
            print(f"  {rec.path.name}: {len(cycles)} segments so far", flush=True)

    y = np.asarray(labels)
    g = np.asarray(groups)
    o = np.asarray(origins)
    if use_cache:
        np.savez_compressed(
            path,
            cycles=np.asarray(cycles, dtype=object),
            y=y,
            groups=g,
            recordings=o,
        )
    return cycles, y, g, o


def build_paired_dataset(
    spec: SegmentSpec = SegmentSpec(),
    window_s: float = 0.2,
    hop_s: float = 0.1,
    feature_names: tuple[str, ...] = PGLTS_CORE,
    n_phase: int = DEFAULT_N_PHASE,
    max_cycles: int | None = 200,
    records: list[Recording] | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> tuple[Dataset, list[np.ndarray]]:
    """Both views of the *same* retained segments, index-aligned.

    The fused model needs the spectral-window trajectory and the wingbeat-phase
    block of one segment to sit at the same index. Building them separately does
    not guarantee that: a frame can pass the harmonic-ratio threshold yet yield
    fewer than two detectable cycles, so the two builders would drop different
    segments. This routine keeps only frames for which both views exist.
    """
    records = records if records is not None else index()
    key = _key(
        {
            "kind": "paired",
            "duration": spec.duration_s,
            "hop_s_frame": spec.hop_s,
            "threshold": spec.threshold,
            "window_s": window_s,
            "hop_s": hop_s,
            "features": list(feature_names),
            "n_phase": n_phase,
            "max_cycles": max_cycles,
            "n_records": len(records),
        }
    )
    path = cache_dir() / f"paired_{key}.npz"
    if use_cache and path.exists():
        blob = np.load(path, allow_pickle=True)
        ds = Dataset(
            X=blob["X"],
            y=blob["y"],
            groups=blob["groups"],
            recordings=blob["recordings"],
            scores=blob["scores"],
            feature_names=tuple(blob["feature_names"].tolist()),
        )
        return ds, list(blob["cycles"])

    extractor = FeatureExtractor(
        window_s=window_s, hop_s=hop_s, fs=spec.fs, names=tuple(feature_names)
    )
    started = time.time()
    trajectories: list[np.ndarray] = []
    cycles: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    origins: list[str] = []
    scores: list[float] = []
    expected_len: int | None = None

    for rec in records:
        amplitude, phase = rec.load()
        amp_hp = normalise(highpass(amplitude, fs=spec.fs), "std")
        cpx_hp = normalise(highpass(amplitude * np.exp(1j * phase), fs=spec.fs), "std")
        for start in _retained_starts(amp_hp, spec):
            stop = start + spec.frame_len
            block = cycle_trajectory(
                amp_hp[start:stop], cpx_hp[start:stop], n_phase=n_phase,
                fs=spec.fs, max_cycles=max_cycles,
            )
            if block.shape[0] < 2:
                continue
            traj = extractor.transform(amp_hp[start:stop])
            if traj.shape[0] == 0:
                continue
            if expected_len is None:
                expected_len = traj.shape[0]
            elif traj.shape[0] != expected_len:
                continue
            trajectories.append(traj)
            cycles.append(block)
            labels.append(rec.species)
            groups.append(rec.subject_key)
            origins.append(rec.path.stem)
            scores.append(float(harmonic_ratio(amp_hp[start:stop])))
        if verbose:
            print(f"  {rec.path.name}: {len(trajectories)} segments so far", flush=True)

    ds = Dataset(
        X=np.asarray(trajectories, dtype=np.float32),
        y=np.asarray(labels),
        groups=np.asarray(groups),
        recordings=np.asarray(origins),
        scores=np.asarray(scores),
        feature_names=tuple(feature_names),
    )
    if verbose:
        print(f"built {len(ds)} paired segments in {time.time() - started:.0f}s", flush=True)
    if use_cache:
        np.savez_compressed(
            path,
            X=ds.X,
            y=ds.y,
            groups=ds.groups,
            recordings=ds.recordings,
            scores=ds.scores,
            feature_names=np.asarray(ds.feature_names),
            cycles=np.asarray(cycles, dtype=object),
        )
    return ds, cycles


def _retained_starts(signal: np.ndarray, spec: SegmentSpec) -> list[int]:
    """Frame start indices that pass the harmonic-ratio threshold."""
    from pglts.signal.segmentation import frame_bounds

    return [
        s
        for s in frame_bounds(signal.size, spec.frame_len, spec.hop)
        if harmonic_ratio(signal[s : s + spec.frame_len]) > spec.threshold
    ]
