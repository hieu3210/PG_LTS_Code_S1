# External datasets

Raw measurement data are not included.

## Polliradar (required for configs/full.json)

Place all 167 original CSV files directly under `dataset/polliradar/`.
DOI: https://doi.org/10.21227/8397-ch84

Columns: amplitude, phase. Sample rate: 1 kHz. Keep original filenames and spelling. The code uses `species#subject` because individual numbers restart within each species.
`SHA256SUMS.txt` contains the original supplied manifests; full runs verify recording names and content hashes automatically before fitting. Stop and resolve any mismatch instead of editing the expected hashes.

## UEA InsectWingbeat (not needed by the new radar runner)

The legacy loader under `pglts/data/ucr.py` remains available, but the new reviewer runner does not rerun UEA. Its 200-band to 20-band pooling and representation-only scope are already addressed in the revised manuscript. Do not treat UEA as validation of radar phase alignment.

The synthetic smoke run and calibration controls require neither dataset.
