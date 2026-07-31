set -e
export PYTHONPATH=src
python -m svjmcmc.sbc19 --sims 8 --T 256 --warmup 100 --samples 100
SVJMCMC_SCAN=assoc python -m svjmcmc.predvar19
SVJMCMC_SCAN=seq python -m svjmcmc.predvar19
