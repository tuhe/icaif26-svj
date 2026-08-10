from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np

def _funcs_flat(samples, model='svj'):
    from .functionals import per_draw_functionals
    flat = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in samples.items()}
    return per_draw_functionals(flat, model)

def forensics(tickers, out_dir, x64=False, oracle_warmup=2000, oracle_samples=6000, indep_warmup=1500, indep_samples=2000, seed=123, model_name='svj'):
    if x64:
        import jax
        jax.config.update('jax_enable_x64', True)
    import numpyro
    from .data import load_returns
    from .batched import pad_and_mask
    from .batched_indep import run_independent, per_ticker_func_chains
    from .sampler import run_nuts
    from .diagnostics import ess_rhat_table
    from .functionals import functional_names
    names = functional_names('svj')
    if model_name != 'svj':
        from .batched_indep_ext import run_independent_g
        _indep = lambda R, M, L, **kw: run_independent_g(R, M, L, model=model_name, **kw)
    else:
        _indep = run_independent
    out = {'x64': x64, 'model': model_name, 'oracle': [oracle_warmup, oracle_samples], 'indep': [indep_warmup, indep_samples], 'seed': seed, 'tickers': {}}
    for t in tickers:
        r = load_returns(t)
        R, M, L = pad_and_mask([r])
        res = _indep(R, M, L, seed=seed, num_warmup=indep_warmup, num_samples=indep_samples, n_chains=2)
        fcb = per_ticker_func_chains(res, 0)
        tabb = ess_rhat_table(fcb)
        nu = run_nuts(model_name, r, seed=seed + 1, num_warmup=oracle_warmup, num_samples=oracle_samples, num_chains=4)
        fo = _funcs_flat(nu.samples)
        rec = {}
        allok = True
        for n in names:
            mi = float(np.asarray(fcb[n]).mean())
            mo = float(fo[n].mean())
            mcse = float(np.asarray(fcb[n]).std() / np.sqrt(max(tabb[n]['ess_bulk'], 1.0)) + fo[n].std() / np.sqrt(max(len(fo[n]), 1.0)))
            within = abs(mi - mo) <= 2.0 * mcse
            allok = allok and within
            rec[n] = {'indep': mi, 'oracle': mo, 'abs_diff': abs(mi - mo), 'two_mcse': 2 * mcse, 'n_sigma': abs(mi - mo) / max(mcse, 1e-12), 'within': bool(within)}
        rec['all_within_2mcse'] = bool(allok)
        out['tickers'][t] = rec
        worst = max(names, key=lambda n: rec[n]['n_sigma'])
        print(json.dumps({'ticker': t, 'all_within_2mcse': allok, 'worst': worst, 'n_sigma': round(rec[worst]['n_sigma'], 2)}), flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fn = 'ga2_forensics.json' if model_name == 'svj' else f'ga2_forensics_{model_name}.json'
    Path(out_dir, fn).write_text(json.dumps(out, indent=2, default=float))
    return out

def _gpd_fit(x):
    x = np.sort(x)
    n = len(x)
    if n < 5:
        return (np.nan, np.nan)
    m = 30 + int(np.sqrt(n))
    theta = 1.0 / x[-1] + (1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) / (3.0 * x[int(n / 4 + 0.5) - 1] + 1e-30)
    k_theta = np.array([np.mean(np.log1p(-th * x)) for th in theta])
    l_theta = n * (np.log(-theta / (k_theta + 1e-30)) - k_theta - 1)
    w = 1.0 / np.array([np.sum(np.exp(l_theta - lt)) for lt in l_theta])
    theta_hat = np.sum(theta * w)
    k = float(np.mean(np.log1p(-theta_hat * x)))
    sigma = float(-k / (theta_hat + 1e-30))
    return (k, sigma)

def _psis_khat(log_weights):
    lw = np.asarray(log_weights, dtype=float)
    lw = lw - lw.max()
    n = len(lw)
    M = min(n - 1, int(3 * np.sqrt(n)))
    w = np.exp(lw)
    cutoff = np.sort(w)[-(M + 1)]
    tail = w[w > cutoff] - cutoff
    if len(tail) < 5:
        return np.nan
    k, _ = _gpd_fit(tail)
    return k

def vi(tickers, out_dir, families=('meanfield', 'lowrank', 'fullrank'), svi_steps=4000, lr=0.002, n_vi=4000, oracle_warmup=1500, oracle_samples=3000, lowrank=20, seed=7, model_name='svj'):
    import jax
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO, Predictive
    from numpyro.infer.autoguide import AutoNormal, AutoLowRankMultivariateNormal, AutoMultivariateNormal
    import numpyro.optim as optim
    from numpyro.infer.util import log_density
    from .data import load_returns
    from .model import MODELS
    from .sampler import run_nuts
    from .functionals import functional_names
    svj_model = MODELS[model_name]
    names = functional_names('svj')
    GUIDES = {'meanfield': lambda: AutoNormal(svj_model), 'lowrank': lambda: AutoLowRankMultivariateNormal(svj_model, rank=lowrank), 'fullrank': lambda: AutoMultivariateNormal(svj_model)}
    out = {'svi_steps': svi_steps, 'n_vi': n_vi, 'families': list(families), 'model': model_name, 'tickers': {}}
    for t in tickers:
        r = load_returns(t)
        rj = jax.numpy.asarray(r)
        nu = run_nuts(model_name, r, seed=seed, num_warmup=oracle_warmup, num_samples=oracle_samples, num_chains=4)
        fo = {n: float(_funcs_flat(nu.samples)[n].mean()) for n in names}
        rec = {'oracle': fo, 'vi': {}}
        for fam in families:
            try:
                guide = GUIDES[fam]()
                svi = SVI(svj_model, guide, optim.Adam(lr), Trace_ELBO())
                res = svi.run(jax.random.PRNGKey(seed), svi_steps, rj, progress_bar=False)
                pred = Predictive(svj_model, guide=guide, params=res.params, num_samples=n_vi, return_sites=['mu', 'phi', 'sigma_eta', 'h', 'p_J', 'sigma_J'])
                s = pred(jax.random.PRNGKey(seed + 1), rj)
                fv = {n: float(np.asarray(_funcs_flat({k: np.asarray(v)[None] for k, v in s.items()})[n]).mean()) for n in names}
                khat = np.nan
                try:
                    from numpyro.infer.util import potential_energy
                    pdist = guide.get_posterior(res.params)
                    z = pdist.sample(jax.random.PRNGKey(seed + 2), (1000,))
                    lq = np.asarray(pdist.log_prob(z))
                    unpack = guide._unpack_latent
                    lp = np.asarray(jax.vmap(lambda zz: -potential_energy(svj_model, (rj,), {}, unpack(zz)))(z))
                    khat = float(_psis_khat(lp - lq))
                except Exception:
                    khat = float('nan')
                rec['vi'][fam] = {'funcs': fv, 'final_elbo': float(np.asarray(res.losses[-50:]).mean()), 'rel_err': {n: abs(fv[n] - fo[n]) / max(abs(fo[n]), 1e-09) for n in names}, 'abs_err': {n: abs(fv[n] - fo[n]) for n in names}, 'psis_khat': khat}
                print(json.dumps({'ticker': t, 'fam': fam, 'relerr_VaR99': round(rec['vi'][fam]['rel_err']['VaR99'], 3), 'relerr_pjump': round(rec['vi'][fam]['rel_err']['p_jump'], 3), 'khat': None if np.isnan(khat) else round(khat, 2)}), flush=True)
            except Exception as e:
                rec['vi'][fam] = {'error': f'{type(e).__name__}: {str(e)[:120]}'}
                print(json.dumps({'ticker': t, 'fam': fam, 'error': str(e)[:80]}), flush=True)
        out['tickers'][t] = rec
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fn = 'gb3_vi.json' if model_name == 'svj' else f'gb3_vi_{model_name}.json'
    Path(out_dir, fn).write_text(json.dumps(out, indent=2, default=float))
    return out

def gradbench(out_dir, B_list=(1, 8, 32, 128), T=1759, reps=20, seed=0):
    import os, time
    import jax, jax.numpy as jnp
    from .model import _SCAN_MODE
    from .batched_indep import unc_logdensity
    mode = _SCAN_MODE
    rng = np.random.default_rng(seed)
    rows = []
    for B in B_list:
        R = jnp.asarray(rng.standard_normal((B, T)) * 1.5)
        M = jnp.ones((B, T))
        Z = jnp.asarray(rng.standard_normal((B, T + 5)) * 0.1)

        def total(z):
            return jax.vmap(unc_logdensity)(z, R, M).sum()
        vg = jax.jit(jax.value_and_grad(total))
        v, g = vg(Z)
        jax.block_until_ready(g)
        t0 = time.perf_counter()
        for _ in range(reps):
            v, g = vg(Z)
            jax.block_until_ready(g)
        ms = (time.perf_counter() - t0) / reps * 1000.0
        row = {'B': B, 'T': T, 'scan': mode, 'grad_ms': ms}
        print(json.dumps(row), flush=True)
        rows.append(row)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, f'gb2_gradbench_{mode}.json').write_text(json.dumps(rows, indent=2, default=float))
    return rows

def gradbench_T(out_dir, T_list=(256, 512, 1024, 1759, 4096, 8192, 16384), B=32, reps=20, seed=0):
    import time
    import jax, jax.numpy as jnp
    from .model import _SCAN_MODE
    from .batched_indep import unc_logdensity
    mode = _SCAN_MODE
    rng = np.random.default_rng(seed)
    rows = []
    for T in T_list:
        R = jnp.asarray(rng.standard_normal((B, T)) * 1.5)
        M = jnp.ones((B, T))
        Z = jnp.asarray(rng.standard_normal((B, T + 5)) * 0.1)

        def total(z):
            return jax.vmap(unc_logdensity)(z, R, M).sum()
        vg = jax.jit(jax.value_and_grad(total))
        v, g = vg(Z)
        jax.block_until_ready(g)
        t0 = time.perf_counter()
        for _ in range(reps):
            v, g = vg(Z)
            jax.block_until_ready(g)
        ms = (time.perf_counter() - t0) / reps * 1000.0
        row = {'B': B, 'T': int(T), 'scan': mode, 'grad_ms': ms}
        print(json.dumps(row), flush=True)
        rows.append(row)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, f'gb2_gradbench_T_{mode}.json').write_text(json.dumps(rows, indent=2, default=float))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['forensics', 'vi', 'gradbench', 'gradbench-t'])
    ap.add_argument('--tickers', default='')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--x64', action='store_true')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--lowrank', type=int, default=20)
    ap.add_argument('--families', default='meanfield,lowrank,fullrank')
    ap.add_argument('--model', default='svj')
    a = ap.parse_args()
    if a.cmd == 'gradbench':
        gradbench(a.out_dir)
    elif a.cmd == 'gradbench-t':
        gradbench_T(a.out_dir)
    elif a.cmd == 'forensics':
        forensics(a.tickers.split(','), a.out_dir, x64=a.x64, model_name=a.model)
    else:
        vi(a.tickers.split(','), a.out_dir, families=tuple(a.families.split(',')), lowrank=a.lowrank, seed=a.seed, model_name=a.model)
if __name__ == '__main__':
    main()
