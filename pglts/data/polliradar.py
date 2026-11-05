"""Polliradar corpus index and loading.

The corpus ships one CSV per recording with two columns, ``amplitude`` and
``phase``, sampled at 1 kHz. File names follow::

    {species}_{subject}_{recording}.csv

Subject numbering **restarts within each species**, so the independent
statistical unit is the composite key ``species#subject`` rather than the bare
integer. Grouping on the integer alone would merge unrelated individuals and
silently break subject-disjoint evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 1000.0

_FNAME = re.compile(r"^(?P<species>[a-z_]+)_(?P<subject>\d+)_(?P<recording>\d+)\.csv$")

#: Taxonomy of the five species, as used by the hierarchical baseline.
TAXONOMY: dict[str, dict[str, str]] = {
    "apis_mellifera": {"family": "Apidae", "genus": "Apis", "display": "A. mellifera"},
    "bombus_lapidaries": {"family": "Apidae", "genus": "Bombus", "display": "B. lapidarius"},
    "bombus_muscorum": {"family": "Apidae", "genus": "Bombus", "display": "B. muscorum"},
    "bombus_terrestris": {"family": "Apidae", "genus": "Bombus", "display": "B. terrestris"},
    "vespula_vulgaris": {"family": "Vespidae", "genus": "Vespula", "display": "V. vulgaris"},
}

SPECIES = tuple(sorted(TAXONOMY))


def default_root() -> Path:
    """Path to ``dataset/polliradar`` relative to the repository root."""
    return Path(__file__).resolve().parents[2] / "dataset" / "polliradar"


@dataclass(frozen=True)
class Recording:
    """One CSV file with its parsed provenance."""

    path: Path
    species: str
    subject: int
    recording: int

    @property
    def subject_key(self) -> str:
        """Composite grouping key — the unit that must not straddle folds."""
        return f"{self.species}#{self.subject}"

    @property
    def family(self) -> str:
        return TAXONOMY[self.species]["family"]

    @property
    def genus(self) -> str:
        return TAXONOMY[self.species]["genus"]

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(amplitude, phase)`` as float64 arrays."""
        return load_csv(self.path)


@lru_cache(maxsize=None)
def _load_cached(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path_str, delimiter=",", skiprows=1)
    amplitude = np.ascontiguousarray(data[:, 0])
    phase = np.ascontiguousarray(data[:, 1])
    amplitude.setflags(write=False)
    phase.setflags(write=False)
    return amplitude, phase


def load_csv(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load one recording. Results are cached; returned arrays are read-only."""
    return _load_cached(str(path))


def index(root: Path | str | None = None) -> list[Recording]:
    """Index every recording under ``root``, sorted by species/subject/recording.

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist or contains no parsable CSV.
    """
    root = Path(root) if root is not None else default_root()
    if not root.is_dir():
        raise FileNotFoundError(f"Polliradar root not found: {root}")

    records: list[Recording] = []
    for path in sorted(root.glob("*.csv")):
        m = _FNAME.match(path.name)
        if m is None:
            continue
        species = m.group("species")
        if species not in TAXONOMY:
            raise ValueError(f"Unknown species {species!r} in {path.name}")
        records.append(
            Recording(
                path=path,
                species=species,
                subject=int(m.group("subject")),
                recording=int(m.group("recording")),
            )
        )
    if not records:
        raise FileNotFoundError(f"No Polliradar CSV files under {root}")
    records.sort(key=lambda r: (r.species, r.subject, r.recording))
    return records


def summarise(records: list[Recording] | None = None) -> dict[str, dict[str, int]]:
    """Per-species counts of subjects and recordings (reproduces Table 1)."""
    records = records if records is not None else index()
    out: dict[str, dict[str, int]] = {}
    for species in SPECIES:
        subset = [r for r in records if r.species == species]
        out[species] = {
            "subjects": len({r.subject for r in subset}),
            "recordings": len(subset),
        }
    return out
