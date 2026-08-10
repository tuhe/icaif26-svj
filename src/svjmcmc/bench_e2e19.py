from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import blackjax
MODE = os.environ.get('SVJMCMC_SCAN', 'assoc')
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'NVDA', 'PG', 'KO']
try:
    from blackjax.adaptation.base import get_filter_adapt_info_fn
    _ADAPT_INFO = get_filter_adapt_info_fn((), ('num_integration_steps', 'is_divergent'), ())
except Exception:
    _ADAPT_INFO = None

def run_instrumented(Rmat, M, L, *, seed, num_warmup, num_samples, n_chains, target_accept=0.8):
    from svjmcmc.batched_indep import unc_logdensity, init_z
    R = jnp.asarray(Rmat)
    Mm = jnp.asarray(M)
    B, Tmax = R.shape

    def run_one(rng, r, m, z0):
        ld = lambda z: unc_logdensity(z, r, m)
        warm = blackjax.window_adaptation(blackjax.nuts, ld, is_mass_matrix_diagonal=True, target_acceptance_rate=target_accept, progress_bar=False, adaptation_info_fn=_ADAPT_INFO)
        rk_w, rk_s = jax.random.split(rng)
        (state, params), winfo = warm.run(rk_w, z0, num_steps=num_warmup)
        w_nstep = winfo[1].num_integration_steps.sum()
        w_div = winfo[1].is_divergent.sum()
        step = blackjax.nuts(ld, **params).step

        def body(st, k):
            st, info = step(k, st)
            return (st, (info.is_divergent, info.num_integration_steps))
        keys = jax.random.split(rk_s, num_samples)
        _, (div, nstep) = jax.lax.scan(body, state, keys)
        return (div.sum(), nstep, w_nstep, w_div, params['step_size'])
    rng = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(rng, B * n_chains).reshape(B, n_chains, 2)
    z0 = jax.vmap(init_z)(R, Mm)
    z0 = jnp.broadcast_to(z0[:, None, :], (B, n_chains, z0.shape[1]))
    Rb = jnp.broadcast_to(R[:, None, :], (B, n_chains, Tmax))
    Mb = jnp.broadcast_to(Mm[:, None, :], (B, n_chains, Tmax))
    run_vv = jax.vmap(jax.vmap(run_one))
    t0 = time.perf_counter()
    div, nstep, w_nstep, w_div, ssz = run_vv(keys, Rb, Mb, z0)
    jax.block_until_ready(nstep)
    wall = time.perf_counter() - t0
    nstep = np.asarray(nstep)
    return {'wall_s': wall, 'sampling_leapfrogs_total': int(nstep.sum()), 'warmup_leapfrogs_total': int(np.asarray(w_nstep).sum()), 'grad_evals_total': int(nstep.sum() + np.asarray(w_nstep).sum()), 'divergences_sampling': int(np.asarray(div).sum()), 'divergences_warmup': int(np.asarray(w_div).sum()), 'depth_hits': int((nstep >= 1023).sum()), 'tree_size_quantiles': {q: float(np.quantile(nstep, q)) for q in (0.5, 0.9, 0.99, 1.0)}, 'step_size_per_lane': np.asarray(ssz, float).round(6).tolist(), 'mean_leapfrogs_per_draw': float(nstep.mean())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tickers', default=','.join(DEFAULT_TICKERS))
    ap.add_argument('--T', type=int, default=1759)
    ap.add_argument('--warmup', type=int, default=800)
    ap.add_argument('--samples', type=int, default=800)
    ap.add_argument('--n-chains', type=int, default=2)
    ap.add_argument('--warm-repeats', type=int, default=5)
    a = ap.parse_args()
    from svjmcmc.data import load_returns
    from svjmcmc.batched import pad_and_mask
    from svjmcmc.gpu_guard19 import require_gpu
    require_gpu()
    tickers = [t for t in a.tickers.split(',') if t]
    series = []
    for t in tickers:
        r = np.asarray(load_returns(t), float)
        series.append(r[-a.T:] if a.T > 0 and r.size > a.T else r)
    Rmat, M, L = pad_and_mask(series)
    t0 = time.perf_counter()
    first = run_instrumented(Rmat, M, L, seed=0, num_warmup=a.warmup, num_samples=a.samples, n_chains=a.n_chains)
    cold = time.perf_counter() - t0
    warms = []
    for k in range(a.warm_repeats):
        warms.append(run_instrumented(Rmat, M, L, seed=1 + k, num_warmup=a.warmup, num_samples=a.samples, n_chains=a.n_chains))
    ww = [w['wall_s'] for w in warms]
    out = {'mode': MODE, 'T': a.T, 'B': len(tickers), 'warmup': a.warmup, 'samples': a.samples, 'n_chains': a.n_chains, 'host': 'anon', 'wall_cold_s': round(cold, 2), 'wall_warm_median_s': round(float(np.median(ww)), 2), 'wall_warm_all_s': [round(w, 2) for w in ww], 'compile_s': round(cold - float(np.median(ww)), 2), 'first_run': first, 'warm_runs': warms}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.out_dir, f'bench_e2e19_{MODE}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('mode', 'wall_cold_s', 'wall_warm_median_s', 'compile_s')} | {'grad_evals_total': first['grad_evals_total'], 'mean_leapfrogs': first['mean_leapfrogs_per_draw']}), flush=True)
if __name__ == '__main__':
    main()
