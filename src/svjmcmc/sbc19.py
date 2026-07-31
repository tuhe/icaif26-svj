from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .model import MU_LOC, MU_SCALE, PHI_A, PHI_B, PJ_A, PJ_B, SIGMA_ETA_HN_SCALE, SIGMA_J_SCALE
PARAMS = ('mu', 'phi', 'sigma_eta', 'p_J', 'sigma_J', 'h_T', 'pit_next')
_COL = {'mu': 0, 'phi': 1, 'sigma_eta': 2, 'p_J': 3, 'sigma_J': 4, 'h_T': 5}
SEED0 = 20260719

def simulate_svj_hn_numpy(T, rng):
    while True:
        mu = rng.normal(MU_LOC, MU_SCALE)
        phi = 2.0 * rng.beta(PHI_A, PHI_B) - 1.0
        sigma_eta = abs(rng.normal(0.0, SIGMA_ETA_HN_SCALE))
        p_J = rng.beta(PJ_A, PJ_B)
        sigma_J = abs(rng.normal(0.0, SIGMA_J_SCALE))
        h = np.empty(T + 1)
        h[0] = mu + sigma_eta / np.sqrt(max(1.0 - phi * phi, 1e-12)) * rng.standard_normal()
        for t in range(1, T + 1):
            h[t] = mu + phi * (h[t - 1] - mu) + sigma_eta * rng.standard_normal()
        jump = rng.random(T + 1) < p_J
        sd = np.sqrt(np.exp(h) + jump * sigma_J ** 2)
        r_all = sd * rng.standard_normal(T + 1)
        if np.all(np.isfinite(r_all)) and np.abs(r_all).max() < 1000000.0 and (h.max() < 40.0):
            theta = {'mu': mu, 'phi': phi, 'sigma_eta': sigma_eta, 'p_J': p_J, 'sigma_J': sigma_J, 'h_T': h[T - 1]}
            return (theta, r_all[:T], float(r_all[T]))

def _rank_u(draws, truth, rng):
    L = draws.size
    return (float((draws < truth).sum()) + rng.random()) / (L + 1.0)

def _thin(draws, ess):
    k = max(1, int(np.ceil(draws.size / max(ess, 1.0))))
    return draws[::k]

def _chi2_20bin(u):
    from scipy.stats import chi2 as chi2d
    counts, _ = np.histogram(np.asarray(u), bins=20, range=(0.0, 1.0))
    exp = len(u) / 20.0
    c = float(((counts - exp) ** 2 / exp).sum())
    return (c, float(chi2d.sf(c, 19)))

def _ecdf_max_dev(u):
    u = np.sort(np.asarray(u))
    n = u.size
    grid = np.arange(1, n + 1) / n
    return float(np.max(np.abs(grid - u)))

def _band99(R, n_rep=2000, seed=1):
    rng = np.random.default_rng(seed)
    devs = [_ecdf_max_dev(rng.random(R)) for _ in range(n_rep)]
    return float(np.quantile(devs, 0.99))

def run(sims, T, chunk, out, num_warmup=800, num_samples=800, n_chains=1, seed=SEED0):
    from .batched import pad_and_mask
    from .batched_indep_ext import run_independent_g
    from .ladder19 import predictive_samples_l19
    from .gpu_guard19 import require_gpu
    import numpyro.diagnostics as ndiag
    import jax
    require_gpu()
    t0 = time.time()
    rng_master = np.random.default_rng(seed)
    sims_data = [simulate_svj_hn_numpy(T, np.random.default_rng(seed + 1 + i)) for i in range(sims)]
    ranks = {p: {'raw': [], 'thin': []} for p in PARAMS}
    ess_log = []
    for i0 in range(0, sims, chunk):
        ck = sims_data[i0:i0 + chunk]
        R, M, L = pad_and_mask([c[1] for c in ck])
        res = run_independent_g(R, M, L, model='svj_hn', seed=seed + 90000 + i0, num_warmup=num_warmup, num_samples=num_samples, n_chains=n_chains)
        for b, (theta, _r, r_next) in enumerate(ck):
            d = res.draws[b].reshape(-1, 6)
            rng = np.random.default_rng(seed + 500000 + i0 + b)
            for p in ('mu', 'phi', 'sigma_eta', 'p_J', 'sigma_J', 'h_T'):
                col = d[:, _COL[p]]
                ranks[p]['raw'].append(_rank_u(col, theta[p], rng))
                ess = float(ndiag.effective_sample_size(col.reshape(n_chains, -1)))
                ranks[p]['thin'].append(_rank_u(_thin(col, ess), theta[p], rng))
                if p == 'h_T':
                    ess_log.append(ess)
            pred = predictive_samples_l19(d, 'svj_hn', 8, rng)
            pit = (float((pred < r_next).sum()) + rng.random()) / (pred.size + 1.0)
            ranks['pit_next']['raw'].append(pit)
            ranks['pit_next']['thin'].append(pit)
        del res
        jax.clear_caches()
        print(json.dumps({'sbc_done': min(i0 + chunk, sims), 'of': sims, 'elapsed_s': round(time.time() - t0, 1)}), flush=True)
    band = _band99(sims)
    tests = {}
    for p in PARAMS:
        c_raw, p_raw = _chi2_20bin(ranks[p]['raw'])
        c_th, p_th = _chi2_20bin(ranks[p]['thin'])
        tests[p] = {'chi2_raw': round(c_raw, 2), 'p_raw': round(p_raw, 4), 'chi2_thin': round(c_th, 2), 'p_thin': round(p_th, 4), 'ecdf_max_dev_raw': round(_ecdf_max_dev(ranks[p]['raw']), 4), 'ecdf_max_dev_thin': round(_ecdf_max_dev(ranks[p]['thin']), 4), 'within_band99': bool(_ecdf_max_dev(ranks[p]['thin']) <= band)}
    out_d = {'generator': 'independent sequential NumPy simulator (no JAX reconstruction import)', 'model': 'svj_hn', 'sims': sims, 'T': T, 'budget': {'num_warmup': num_warmup, 'num_samples': num_samples, 'n_chains': n_chains}, 'rank_protocol': 'randomized PIT (rank+V)/(L+1); thinned stride ceil(S/ESS_bulk) per quantity; chi2 20-bin df19; band simultaneous over rank positions WITHIN each panel (not across panels)', 'band99_per_panel': round(band, 4), 'tests': tests, 'median_hT_ess': round(float(np.median(ess_log)), 1), 'ranks': {p: {k: [round(float(x), 6) for x in v] for k, v in d.items()} for p, d in ranks.items()}, 'host': 'anon', 'seed': seed, 'wall_s': round(time.time() - t0, 1)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(out_d))
    print('SUMMARY ' + json.dumps({p: tests[p] for p in PARAMS}), flush=True)
    return out_d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=512)
    ap.add_argument('--T', type=int, default=1759)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--warmup', type=int, default=800)
    ap.add_argument('--samples', type=int, default=800)
    ap.add_argument('--n-chains', type=int, default=1)
    ap.add_argument('--seed', type=int, default=SEED0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    run(a.sims, a.T, a.chunk, a.out, num_warmup=a.warmup, num_samples=a.samples, n_chains=a.n_chains, seed=a.seed)
if __name__ == '__main__':
    main()
