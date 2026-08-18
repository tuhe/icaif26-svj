# Parallel MCMC for stochastic volatility with jumps on one GPU

Anonymized code for a full replication of an ICAIF 2026 submission.

## Content

- `src/svjmcmc/`: stochastic-volatility model variants
  (SV, SV-t, SVJ, SVJ-t), the associative-scan volatility
  reconstruction, batched
  independently-adapted NUTS, evaluation and calibration.
- `scripts/`: reproduction scripts: `pull_cohort.py` (rebuilds the
  CRSP point-in-time cohort returns from WRDS data, not
  redistributed), `eval_cohort19.py` + `eval_svt_rescore19.py` (the
  evaluate stage), and `make_tables.py` / `make_figures.py` (regenerate
  the manuscript's tables and figures from `results/`)
- `universe_sp500.txt`: the ticker list used by
  `data.load_universe` (the header documents the selection rule and
  its survivorship caveat)
- `results/`, `figures/`: created by running the above scripts and contains the results.

## Install

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
```

Python 3.11 or newer. A CUDA-capable GPU with the matching `jax[cuda]`
 reproduces the paper's timings; everything also runs on CPU at
reduced speed.

## No-data start

Simulation-based calibration runs entirely on simulated data:

```bash
python -m svjmcmc.sbc19 --sims 8 --T 256 --warmup 100 --samples 100 --out results/sbc_smoke.json  # smoke, minutes
python -m svjmcmc.sbc19 --out results/sbc19.json                       # paper setting: 512 sims, T=1759
```

The predictive-VaR / scan-consistency audit uses a 20-ticker demo
panel of daily returns; on first run it is downloaded automatically via `yfinance`
into `data/data_cache/` (so the exact per-ticker values depend on the
Yahoo data version, however, the audit's conclusions do not):

```bash
SVJMCMC_SCAN=assoc python -m svjmcmc.predvar19 --out results/predvar_assoc.json  # minutes on a GPU
SVJMCMC_SCAN=seq   python -m svjmcmc.predvar19 --out results/predvar_seq.json    # ~2.5 h on an RTX 4090; the gap is the point
```

`all_terminal_commands.sh` runs all of the above.

## Market data

No market data is redistributed due to licensing constraints, but it can be downloaded using a standard WRDS subscription. The empirical study uses daily
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

The `yfinance` demo panel used by `predvar19` lives separately under
`data/data_cache/` (override with `SVJMCMC_DATA_CACHE`; universe list
override: `SVJMCMC_UNIVERSE`).

## Reproducing the empirical study

With the cohort returns in `data/returns/` (see the above step), the study is
per-name forecasting, then a single evaluate pass, then the tables.
Per name (`TICKER.PERMNO` from the cohort manifest), on one GPU:

```bash
# NUTS ladder (the SVJ-t arm ran as a 19-name pilot; main arms below)
python -m svjmcmc.forecast_cohort19 --id AAPL.14593 --cohort 252 \
    --engine ladder --models sv_hn,svt_hn,svj_hn --out-dir results/cohort19/oos252
# GARCH-t + FHS baselines (CPU)
python -m svjmcmc.forecast_cohort19 --id AAPL.14593 --cohort 252 \
    --engine baselines --out-dir results/cohort19/oos252
# mean-field VI baseline
python -m svjmcmc.forecast_cohort19 --id AAPL.14593 --cohort 252 \
    --engine vi --out-dir results/cohort19/vi252
```

Loop the three commands over the cohort members. Measured on an
RTX 4090: about 20 minutes per NUTS model per ticker
plus about 20 minutes for the rolling VI, so a full serial 500-name
reproduction is on the order of weeks on one GPU. Names are
independent, so in a real run, it should be run in parallel across several GPUs; a small subset of tickers already exercises every stage. Then run:

```bash
python scripts/eval_cohort19.py --cohort 252 \
    --res-dir results/cohort19/oos252 --vi-dir results/cohort19/vi252 \
    --out results/eval_cohort252.json
python scripts/make_tables.py --eval results/eval_cohort252.json --out-dir figures
```

`eval_cohort252.json` contains the headline numbers (the abstract's
VI breach rate is `breach_1pct.vi_mf_hn.rate_pct`), and `make_tables.py`
writes the manuscript's calibration and skill tables from it. The
nightly deployment-timing pass is

`python -m svjmcmc.forecast_cohort19 --cohort 252 --engine deploy
--out-dir results/cohort19/deploy252`.

`python scripts/make_figures.py` regenerates the remaining figures and
appendix tables (SBC, scan-scaling, VI-family and deployment tables)
from whatever is present under `results/`, skipping any stage whose
inputs are absent, making it safe to run at any point.

