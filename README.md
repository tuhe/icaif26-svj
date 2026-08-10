# svjmcmc: parallel MCMC for stochastic volatility with jumps on one GPU

Anonymized code release for an ICAIF 2026 submission.

## Layout

- `src/svjmcmc/` — the package: stochastic-volatility model variants
  (SV, SV-t, SVJ, SVJ-t), the associative-scan volatility
  reconstruction with scan-structured reverse-mode gradients, batched
  independently-adapted No-U-Turn Samplers, forecasting, tail scores,
  and simulation-based calibration
- `scripts/pull_cohort.py` — rebuilds the CRSP point-in-time cohort
  returns from WRDS (licensed data, not redistributed)
- `universe_sp500.txt` — the sealed ticker list used by
  `data.load_universe` (its header documents the selection rule and
  its survivorship caveat)
- `results/`, `figures/` — created by the entry points (not committed)

## Install (about 5 minutes)

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
```

Python 3.11 or newer. A CUDA-capable GPU with the matching `jax[cuda]`
wheel reproduces the paper's timings; everything also runs on CPU at
reduced speed (the GPU guard prints a warning and continues; set
`SVJMCMC_REQUIRE_GPU=1` to make it abort instead, e.g. on batch nodes
where a silent CPU fallback would waste a GPU slot).

## No data needed to start

Simulation-based calibration runs entirely on simulated data:

```bash
python -m svjmcmc.sbc19 --sims 8 --T 256 --warmup 100 --samples 100 --out results/sbc_smoke.json  # smoke, minutes
python -m svjmcmc.sbc19 --out results/sbc19.json                       # paper setting: 512 sims, T=1759
```

The predictive-VaR / scan-consistency audit uses a 20-ticker demo
panel of daily returns; on first run it is downloaded via `yfinance`
into `data/data_cache/` (so exact per-ticker values depend on the
Yahoo data vintage; the audit's conclusions do not):

```bash
SVJMCMC_SCAN=assoc python -m svjmcmc.predvar19 --out results/predvar_assoc.json  # scan benchmark arm
SVJMCMC_SCAN=seq   python -m svjmcmc.predvar19 --out results/predvar_seq.json    # sequential comparison arm
```

`all_terminal_commands.sh` runs all of the above.

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

The `yfinance` demo panel used by `predvar19` lives separately under
`data/data_cache/` (override with `SVJMCMC_DATA_CACHE`; universe list
override: `SVJMCMC_UNIVERSE`).

## Status

This release accompanies the submission. The full-universe run driver,
the evaluation stage against the GARCH-t and filtered historical
simulation baselines, and the table/figure generators are being ported
from the research codebase as part of a clean-checkout validation pass;
the repository will be updated in place.
