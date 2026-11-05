# PG-LTS — code for reproducing the experiments (S1)

Source code of PG-LTS (phase-grouped linguistic time series: hedge-algebra
semantization of radar phase cycles, sparse-group linear classifier) together
with every experiment script used in the reviewer-response analyses. It contains
code, configuration files, tiny reference inputs and checksums. It does **not**
contain the radar recordings, run outputs, or manuscript sources.

## Requirements

- Python 3.12, pinned packages in `requirements.txt` (numpy 2.3.5, scipy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3, aeon 1.5.0, librosa 0.11.0, psutil 7.2.2).
- Results were produced on macOS (Apple Silicon). Other platforms/library versions can change floating-point results; do not claim bitwise equality across environments.
- aeon 1.5.0 needs one library edit for the DrCIF/RDST baselines (see below).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/patch_aeon.py        # DrCIF/RDST baselines only; idempotent
export PY=python
scripts/run_pipeline.sh test        # unit tests (no data needed)
scripts/run_pipeline.sh smoke       # synthetic smoke run (NOT a scientific result)
```

## Data

The Polliradar recordings (167 CSV files, columns `amplitude`, `phase`, 1 kHz) are
public: DOI [10.21227/8397-ch84](https://doi.org/10.21227/8397-ch84). Put them
directly under `dataset/polliradar/` with the original file names.
`dataset/SHA256SUMS.txt` lists the expected hashes; full runs verify them before
fitting and stop on any mismatch. Do not edit the expected hashes.

## Layout

| Path | Content |
|---|---|
| `pglts/` | Core PG-LTS library (signal processing, cycle segmentation, linguistic descriptors, sparse-group classifier, evaluation protocol) |
| `revision/` | Experiment runners (see table below); atomic, checksummed checkpoints |
| `third_party/pyLTS/` | Vendored hedge-algebra / LTS library (see `THIRD_PARTY_NOTICES.md`) |
| `configs/` | JSON configuration of every experiment |
| `LEGACY_COMPATIBILITY.json` | Hash list of the legacy core files; used by the follow-up identity checks and tests |
| `reference/` | Small inputs required by the last analysis step: locked OOF predictions, selected configs, inner search, and the eight DrCIF/RDST outer checkpoints plus CSVs of the intermediate step |
| `inputs/` | Two small derived inputs needed by tests and step 4: `main/search_space.json` (identical to the frozen grid written by step 1) and `order_v120/cycle_manifest.csv` (output of step 3 on the primary non-overlapping cohort, block null) |
| `tools/` | Read-only audit exporter (`export_audit.py`, `verify_export.py`, optional `capture_hooks.py`) and `pack_repeat_inputs.py` |
| `checks/` | Algebraic verification scripts |
| `tests/`, `tools/tests/` | Unit tests |
| `scripts/` | `run_pipeline.sh` driver and `patch_aeon.py` |

## Experiment steps

Run from the repository root. `SRC` is the run directory created by step 1
(`runs/review_full-<hash>`); everything after step 1 only reads it.

| Step | Command | Module | Purpose |
|---|---|---|---|
| 0 | `scripts/run_pipeline.sh smoke` | `revision.cli` | Synthetic end-to-end check |
| 1 | `scripts/run_pipeline.sh source` | `revision.cli` (`configs/full.json`) | Preparation, nested 4-outer/3-inner subject-level CV, permutation, robustness, rule fidelity |
| 1b | `scripts/run_pipeline.sh multirocket` | `revision.cli` (`configs/multirocket.json`) | Matched-input MultiRocket arm |
| 2 | `scripts/run_pipeline.sh calibration` / `followup` / `raw` | `revision.followup`, `followup_calibration`, `raw_baselines` | Synthetic/empirical calibration, pooled rule metrics, ROCKET/MultiRocket on prepared I/Q or STFT |
| 3 | `SRC=... scripts/run_pipeline.sh order` | `revision.radar_order` | Radar-order randomization test (non-overlapping cohorts, block-shared null) |
| 4 | `SRC=... scripts/run_pipeline.sh validation` | `revision.recording_calibration`, `repeated_nested` | Recording-shared calibration; repeated nested CV over five seeds |
| 5 | `SRC=... scripts/run_pipeline.sh closure` | `revision.reviewer_closure` | Matched-input DrCIF/RDST vs PG-LTS, phase/boundary robustness, rule top-k fidelity |
| 6 | `SRC=... scripts/run_pipeline.sh v2` | `revision.closure_supervisor`, `independent_validation` | Baseline replay and HA-vs-quantile ablation with fixed grid, resource monitor |
| 7 | `SRC=... scripts/run_pipeline.sh v3` | `revision.final_three_checks` | Reconciliation with the locked run, full-grid nested HA-vs-quantile ablation, DrCIF/RDST audit (raw-file hashes, tensors, saved-model replay), PG-LTS runtime |
| – | `python tools/export_audit.py --help` | `inputs/` | Two small derived inputs needed by tests and step 4: `main/search_space.json` (identical to the frozen grid written by step 1) and `order_v120/cycle_manifest.csv` (output of step 3 on the primary non-overlapping cohort, block null) |
| `tools/` | Read-only export of an audit bundle with receipt/hash checks |

Steps 6–7 need the `SRC` run (167 prepared checkpoints, `splits.csv`,
`segment_manifest.csv`, `oof_predictions.csv`); the `reference/` folder supplies
the remaining small inputs of step 7.

## Reproducibility properties

- **Resume:** rerun the same command after an interruption. Each task is written atomically with a SHA-256 receipt; completed tasks are reused, broken ones are recomputed. Exit code 75 means an intentional stop requested with `--stop-after`.
- **Run namespace:** the run directory name contains a hash of config, source code, environment and inputs; changing any of them creates a new run instead of overwriting.
- **Seeds:** all random seeds are in the configs; per-task seeds are independent of the order in which tasks run.
- **No test-set selection:** hyperparameters and mappings are chosen within the training subjects of each outer fold.

## Documented deviations from library defaults

1. **aeon patch.** aeon 1.5.0 raises *"Input collection has too little variation"* for some fold tensors. `scripts/patch_aeon.py` changes the call in `aeon/base/_base_collection.py` to `check_collection_variance(X, raise_error=False)` (advisory). It must be applied to the installed library, because worker processes do not inherit runtime monkeypatches.
2. **float64 tensors.** The DrCIF/RDST input tensor is float64 (`revision/reviewer_closure.py`); with float32, RDST fails in numba type unification.
3. **Threads.** `PGLTS_AEON_JOBS` (default 1) sets DrCIF `n_jobs`. Predictions do not depend on it; use the default for timing measurements.

## Limits of what the code establishes

- Checksummed checkpoints and replay show that a run is internally consistent and re-loadable; they do not prove bitwise equality with a historical training run or across software environments.
- All analyses evaluate the same 167-recording cohort (33 insects, 5 species); none is a test on newly collected data.
- Runtime and sampled peak RSS are measurements on the executing machine; no energy measurement is made or inferred.
- `tools/capture_hooks.py` (optional) is needed to compare baseline tensors with the exact inputs the PG-LTS pipeline consumed; it is not run by `run_pipeline.sh`.

## License

MIT for the original code (`LICENSE`); vendored third-party code keeps its own terms (`THIRD_PARTY_NOTICES.md`).
