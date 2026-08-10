set -e
export PYTHONPATH=src

# Simulation-based calibration, smoke setting (minutes on GPU; runs on CPU
# too — the GPU guard only warns unless SVJMCMC_REQUIRE_GPU=1).
python -m svjmcmc.sbc19 --sims 8 --T 256 --warmup 100 --samples 100 --out results/sbc_smoke.json
# Paper setting (long): python -m svjmcmc.sbc19 --out results/sbc19.json

# Predictive-VaR audit on the 20-ticker demo panel, associative vs
# sequential scan. Downloads the panel via yfinance on first run if
# data/data_cache/ is empty.
SVJMCMC_SCAN=assoc python -m svjmcmc.predvar19 --out results/predvar_assoc.json
SVJMCMC_SCAN=seq   python -m svjmcmc.predvar19 --out results/predvar_seq.json
