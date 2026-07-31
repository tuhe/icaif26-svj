# Parallel MCMC for stochastic volatility with jumps on one GPU


**Under construction.**
Anonymized code release for an ICAIF 2026 submission.

## Layout
- `scripts/` — pipeline and evaluation scripts
- `configs/` — experiment configuration files
- `manifests/` — date lists and cohort manifests used in the paper
- `results/` — aggregate result files (JSON) behind the paper's tables

## Data
No raw market data is redistributed here. The paper uses
CRSP/S&P 500 daily returns; both are licensed sources. The README in each script
directory documents the exact pull and cleaning steps so that users
with access to the same sources can rebuild the datasets.

## Reproduction
Population of this repository is in progress; the final version
accompanies the submission.
