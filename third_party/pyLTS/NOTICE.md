# pyLTS (vendored)

`lts/` is the hedge-algebra Linguistic Time Series library pyLTS, version 0.2.0,
by Nguyen Duy Hieu, licensed GPL-3.0-or-later (see `pyproject.toml`). It is bundled
unmodified so that the archive is self-contained.

PG-LTS uses only three modules from it:

- `lts.core.hedge_algebras` (`HAParams`, `HedgeAlgebra`)
- `lts.core.fuzziness_interval` (`FuzzinessIntervals`)
- `lts.optimization.pso` (`PSO`, `PSOConfig`)

Install with `pip install -e third_party/pyLTS`.
