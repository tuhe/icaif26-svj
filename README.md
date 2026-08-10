# svjmcmc: parallel MCMC for stochastic volatility with jumps on one GPU

Anonymized code release for an ICAIF 2026 submission.

## Layout

- `src/svjmcmc/` — the package: stochastic-volatility model variants
  (SV, SV-t, SVJ, SVJ-t), the associative-scan volatility
  reconstruction with scan-structured reverse-mode gradients, batched
  independently-adapted No-U-Turn Samplers, forecasting, tail scores,
  and simulation-based calibration
- `results/`, `figures/` — created by the entry points (not committed)

## Install (about 5 minutes)

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
```

Python 3.11 or newer. A CUDA-capable GPU with the matching `jax[cuda]`
wheel reproduces the paper's timings; everything also runs on CPU at
reduced speed.

## No data needed to start

The two headline methodological claims run entirely on simulated data:

```bash
python -m svjmcmc.sbc19 --sims 8 --T 256 --warmup 100 --samples 100   # smoke, minutes
python -m svjmcmc.sbc19                                               # paper setting: 512 sims, T=1759
SVJMCMC_SCAN=assoc python -m svjmcmc.predvar19                        # scan benchmark arm
SVJMCMC_SCAN=seq   python -m svjmcmc.predvar19                        # sequential comparison arm
```

## Market data

No market data is redistributed. The empirical study uses daily
returns for point-in-time S&P 500 cohorts (503 names as of 2023-12-28
for the main dataset, 505 as of 2021-12-27 for the extended dataset),
pulled via WRDS with index membership resolved as of the last trading
day before each test period and members retained through subsequent
delisting.

The pull is scripted, so a licensed user can rebuild the inputs rather
than reconstruct the query from prose:

```bash
pip install wrds                      # only needed for this step
export WRDS_USERNAME=<your-wrds-user>
python scripts/pull_cohort.py         # writes data/returns/
```

It writes one file per cohort member,
`{TICKER}.{PERMNO}_2018-01-01_2024-12-31.parquet` with columns
`date, ret_pct` (percent daily log-returns), alongside the cohort
membership it resolved. Set `SVJMCMC_RETURNS` to write elsewhere.
PERMNO is the identifier used throughout and tickers are labels only, so
ticker changes do not break cohort membership.

## Status

This release accompanies the submission. The full-universe run driver,
the evaluation stage against the GARCH-t and filtered historical
simulation baselines, and the table/figure generators are being ported
from the research codebase as part of a clean-checkout validation pass;
the repository will be updated in place.
