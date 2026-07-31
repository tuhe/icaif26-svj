from __future__ import annotations
import numpy as np
from scipy.stats import norm
N_ETA_CRN = 512
_CRN_SEED = 20260719

def _eta():
    return np.random.default_rng(_CRN_SEED).standard_normal(N_ETA_CRN)

def per_draw_pred_cdf(reduced, q, eta=None):
    eta = _eta() if eta is None else eta
    mu, phi, se, pJ, sJ, hT = reduced.T
    h_next = (mu + phi * (hT - mu))[:, None] + se[:, None] * eta[None, :]
    sd0 = np.exp(h_next / 2.0)
    sd1 = np.sqrt(sd0 ** 2 + (sJ ** 2)[:, None])
    qq = np.asarray(q)[..., None] if np.ndim(q) else q
    F = (1.0 - pJ)[:, None] * norm.cdf(qq / sd0) + pJ[:, None] * norm.cdf(qq / sd1)
    return F.mean(axis=1)

def per_draw_pred_pdf(reduced, q, eta=None):
    eta = _eta() if eta is None else eta
    mu, phi, se, pJ, sJ, hT = reduced.T
    h_next = (mu + phi * (hT - mu))[:, None] + se[:, None] * eta[None, :]
    sd0 = np.exp(h_next / 2.0)
    sd1 = np.sqrt(sd0 ** 2 + (sJ ** 2)[:, None])
    qq = np.asarray(q)[..., None] if np.ndim(q) else q
    f = (1.0 - pJ)[:, None] * norm.pdf(qq / sd0) / sd0 + pJ[:, None] * norm.pdf(qq / sd1) / sd1
    return f.mean(axis=1)

def predictive_var(reduced, alpha=0.01, tol=1e-10, max_iter=200):
    lo, hi = (-200.0, 0.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if per_draw_pred_cdf(reduced, mid).mean() < alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

def predictive_var_mcse(reduced, n_chains, alpha=0.01):
    import numpyro.diagnostics as ndiag
    q = predictive_var(reduced, alpha)
    Fs = per_draw_pred_cdf(reduced, q)
    ess = float(ndiag.effective_sample_size(Fs.reshape(n_chains, -1)))
    mcse_F = float(Fs.std() / np.sqrt(max(ess, 1.0)))
    fbar = float(per_draw_pred_pdf(reduced, q).mean())
    return (q, mcse_F / max(fbar, 1e-12))

def audit(tickers=None, out=None, num_warmup=1000, num_samples=1000, n_chains=2, seed=0):
    import json
    from pathlib import Path
    from .batched import pad_and_mask
    from .batched_indep_ext import run_independent_g
    from .data import load_universe
    from .functionals import per_draw_functionals
    from .ladder19 import predictive_samples_l19
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'NVDA', 'PG', 'KO', 'TSLA', 'BRK-B', 'MAA', 'RF', 'KLAC', 'CIEN', 'EXPE', 'FE', 'EOG', 'SLB', 'CCL', 'BRO']
    from .data import load_returns
    panel = {}
    for t in tickers:
        r = load_returns(t)
        if r is not None and len(r) >= 1759:
            panel[t] = np.asarray(r, float)
    tickers = [t for t in tickers if t in panel]
    R, M, L = pad_and_mask([panel[t] for t in tickers])
    res = run_independent_g(R, M, L, model='svj_hn', seed=seed, num_warmup=num_warmup, num_samples=num_samples, n_chains=n_chains)
    rows = {}
    for b, t in enumerate(tickers):
        red = res.draws[b].reshape(-1, 6)
        f = per_draw_functionals({'mu': red[:, 0], 'phi': red[:, 1], 'sigma_eta': red[:, 2], 'p_J': red[:, 3], 'sigma_J': red[:, 4], 'h': red[:, 5][:, None]}, 'svj')
        var_cond_mean = float(f['VaR99'].mean())
        q, mcse = predictive_var_mcse(red, n_chains, alpha=0.01)
        var_pred = -q
        brute = []
        for sd in (11, 22, 33):
            rng = np.random.default_rng(sd)
            s = predictive_samples_l19(red, 'svj_hn', 32, rng)
            brute.append(-float(np.quantile(s, 0.01)))
        rows[t] = {'var99_cond_mean': round(var_cond_mean, 5), 'var99_pred_quantile': round(var_pred, 5), 'mcse_pred_quantile': round(mcse, 6), 'rel_gap_cond_vs_pred': round(var_cond_mean / var_pred - 1.0, 5), 'brute_seeds': [round(x, 5) for x in brute], 'brute_max_rel_dev_vs_pred': round(max((abs(x / var_pred - 1.0) for x in brute)), 6)}
    gaps = [abs(r['rel_gap_cond_vs_pred']) for r in rows.values()]
    summary = {'n': len(rows), 'median_abs_rel_gap': round(float(np.median(gaps)), 5), 'max_abs_rel_gap': round(float(np.max(gaps)), 5), 'median_mcse_rel': round(float(np.median([r['mcse_pred_quantile'] / r['var99_pred_quantile'] for r in rows.values()])), 6), 'max_brute_dev': round(float(np.max([r['brute_max_rel_dev_vs_pred'] for r in rows.values()])), 6)}
    out_d = {'summary': summary, 'per_ticker': rows, 'host': 'anon', 'budget': {'num_warmup': num_warmup, 'num_samples': num_samples, 'n_chains': n_chains}, 'n_eta_crn': N_ETA_CRN}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(out_d, indent=1))
    print('SUMMARY ' + json.dumps(summary), flush=True)
    return out_d
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--samples', type=int, default=1000)
    a = ap.parse_args()
    audit(out=a.out, num_warmup=a.warmup, num_samples=a.samples)
