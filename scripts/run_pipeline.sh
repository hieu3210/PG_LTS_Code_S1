#!/usr/bin/env bash
# Reproduction driver. Every step is resumable: rerun the same command after an
# interruption (exit code 75 = intentional stop via --stop-after).
# Usage: scripts/run_pipeline.sh <step>
# Steps are listed in README.md. Set PY to the Python interpreter of your venv.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
SRC=${SRC:-}          # path to the source run produced by step "source" (runs/review_full-<hash>)
step=${1:-help}
need_src(){ [ -n "$SRC" ] || { echo "Set SRC=runs/review_full-<hash>"; exit 2; }; }
retry(){ for t in 1 2 3 4 5; do "$@" && return 0; c=$?; [ $c -eq 75 ] || [ $c -eq 0 ] || echo "retry $t (exit $c)"; sleep 5; done; return 1; }
case "$step" in
  test)        $PY -m pytest -q tests tools/tests ;;
  smoke)       $PY -m revision.cli --config configs/smoke.json --stage all ;;
  source)      retry $PY -m revision.cli --config configs/full.json --stage all ;;            # step 1: main nested experiment (needs dataset/polliradar)
  multirocket) retry $PY -m revision.cli --config configs/multirocket.json --stage nested ;;
  calibration) retry $PY -m revision.followup --stage calibration --config configs/empirical_representative.json ;;  # step 2 (v1.1.0)
  followup)    need_src; $PY -m revision.followup --stage summarize --source-results "$SRC" ;;
  raw)         need_src; retry $PY -m revision.raw_baselines --source-results "$SRC" --config configs/raw_baselines.json ;;
  order)       need_src; retry $PY -m revision.radar_order --stage test --source-results "$SRC" ;;                    # step 3 (v1.2.0)
  validation)  need_src
               retry $PY -m revision.recording_calibration --config configs/recording_calibration_full.json --workers 4
               retry $PY -m revision.repeated_nested --config configs/repeated_nested_full.json --source-results "$SRC" --stage run ;;  # step 4 (v1.3.0)
  closure)     need_src; PGLTS_AEON_JOBS=${PGLTS_AEON_JOBS:-1} retry $PY -m revision.reviewer_closure --source-results "$SRC" --stage all ;;  # step 5 (v1.4.0)
  v2)          need_src; retry $PY -m revision.closure_supervisor --source-results "$SRC" --reference-oof "$SRC/oof_predictions.csv" --stage all ;;  # step 6
  v3)          need_src
               R=reference
               for st in reconcile audit runtime ablation; do
                 extra=""; [ $st = audit ] && extra="--v2-checkpoints $R/v2_reference/checkpoints --raw-root dataset/polliradar"
                 [ $st = runtime ] && extra="--repeats 3"
                 retry $PY -m revision.final_three_checks --stage $st --source "$SRC" --locked $R/locked_reference --v2 $R/v2_reference $extra --out outputs/v3/$st || exit 1
               done ;;
  *) sed -n 2,6p "$0"; echo "steps: test smoke source multirocket calibration followup raw order validation closure v2 v3" ;;
esac
