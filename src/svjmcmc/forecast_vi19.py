from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .data import load_returns
from .ladder19 import svj_model_masked_hn, predictive_samples_l19
from .tail_scores19 import PROBS_L19, VAR_LEVELS_L19, crps_fair_per_day, crps_pwm_per_day, qgrid_from_samples, twcrps_from_quantiles, var_es_from_samples
N_GUIDE_DRAWS = 1200
N_INNER_PRED = 8
LR = 0.002

def _vi_day(svi, guide, r_pad, m_pad, steps, seed, rng):
    import jax
    import jax.numpy as jnp
    try:
        res = svi.run(jax.random.PRNGKey(seed), steps, jnp.asarray(r_pad), jnp.asarray(m_pad), progress_bar=False)
        losses = np.asarray(res.losses)
        final_elbo = float(-np.mean(losses[-50:]))
        if not np.isfinite(losses[-50:]).all():
            return (None, final_elbo, 'nan_elbo')
        post = guide.sample_posterior(jax.random.PRNGKey(seed + 1), res.params, sample_shape=(N_GUIDE_DRAWS,))
        mu = np.asarray(post['mu'])
        phi = 2.0 * np.asarray(post['phi_raw']) - 1.0
        se = np.asarray(post['sigma_eta'])
        pJ = np.asarray(post['p_J'])
        sJ = np.asarray(post['sigma_J'])
        h_std = np.asarray(post['h_std'])
        T_eff = int(m_pad.sum())
        hT = _h_last(h_std[:, :T_eff], mu, phi, se)
        red = np.stack([mu, phi, se, pJ, sJ, hT], axis=1)
        if not np.isfinite(red).all():
            return (None, final_elbo, 'nonfinite_draws')
        return (red, final_elbo, 'ok')
    except Exception as e:
        return (None, float('nan'), f'error:{type(e).__name__}')

def _h_last(h_std, mu, phi, se):
    M, T = h_std.shape
    g = se[:, None] / np.sqrt(1.0 - phi[:, None] ** 2) * h_std[:, :1]
    for t in range(1, T):
        g = phi[:, None] * g + se[:, None] * h_std[:, t:t + 1]
    return (mu[:, None] + g)[:, 0]

def run(ticker, out_dir, oos=252, steps=4000, seed=0, tag=''):
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro import optim
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    t0 = time.time()
    r = np.asarray(load_returns(ticker), float)
    N = r.size
    oos_start = N - oos
    days = list(range(oos_start, N))
    realized = np.array([r[t] for t in days])
    thr5 = float(np.quantile(r[:-oos], 0.05))
    thr1 = float(np.quantile(r[:-oos], 0.01))
    Tmax = days[-1]
    guide = AutoNormal(svj_model_masked_hn)
    svi = SVI(svj_model_masked_hn, guide, optim.Adam(LR), Trace_ELBO())
    rng = np.random.default_rng(777 + seed)
    rows, elbos, status = ([], [], [])
    for k, t in enumerate(days):
        r_pad = np.zeros(Tmax)
        r_pad[:t] = r[:t]
        m_pad = np.zeros(Tmax)
        m_pad[:t] = 1.0
        red, fe, st = _vi_day(svi, guide, r_pad, m_pad, steps, seed, rng)
        elbos.append(fe)
        status.append(st)
        if red is None:
            rows.append(np.full(N_GUIDE_DRAWS * N_INNER_PRED, np.nan))
        else:
            rows.append(predictive_samples_l19(red, 'svj_hn', N_INNER_PRED, rng))
        if (k + 1) % 20 == 0:
            jax.clear_caches()
            print(json.dumps({'ticker': ticker, 'done': k + 1, 'of': len(days), 'elapsed_s': round(time.time() - t0, 1)}), flush=True)
    samples = np.stack(rows)
    ok = np.array([s == 'ok' for s in status])
    qg = qgrid_from_samples(samples[ok]) if ok.any() else np.zeros((0, len(PROBS_L19)))
    var, es = var_es_from_samples(samples[ok]) if ok.any() else (np.zeros((0, 4)), np.zeros((0, 4)))
    scr = {'crps_fair': _masked(crps_fair_per_day, samples, realized, ok), 'crps_pwm': _masked(crps_pwm_per_day, samples, realized, ok), 'twcrps5': _expand(twcrps_from_quantiles(qg, PROBS_L19, realized[ok], thr5), ok), 'twcrps1': _expand(twcrps_from_quantiles(qg, PROBS_L19, realized[ok], thr1), ok), 'pit': _masked(lambda s, y: (s <= y[:, None]).mean(axis=1), samples, realized, ok), 'var': {str(a): _expand(var[:, j], ok) for j, a in enumerate(VAR_LEVELS_L19)}, 'es': {str(a): _expand(es[:, j], ok) for j, a in enumerate(VAR_LEVELS_L19)}}
    out = {'ticker': ticker, 'model': 'vi_mf_hn', 'oos': oos, 'host': 'anon', 'protocol': {'guide': 'AutoNormal', 'steps': steps, 'lr': LR, 'elbo_particles': 1, 'init': 'cold daily', 'seed': seed, 'n_pred_draws': N_GUIDE_DRAWS * N_INNER_PRED}, 'thr_train': {'q05': thr5, 'q01': thr1}, 'realized': realized.tolist(), 'final_elbo': [round(float(e), 3) if np.isfinite(e) else None for e in elbos], 'fit_status': status, 'n_fit_status': {'ok': int(ok.sum()), 'failed': int((~ok).sum())}, 'scores': scr, 'wall_s': round(time.time() - t0, 1)}
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    name = f'vi19_{ticker}{('_' + tag if tag else '')}.json'
    (outp / name).write_text(json.dumps(out, default=float))
    print('SUMMARY ' + json.dumps({'ticker': ticker, 'steps': steps, 'seed': seed, 'n_fit_status': out['n_fit_status'], 'wall_s': out['wall_s']}), flush=True)
    return out

def _masked(fn, samples, realized, ok):
    vals = np.full(len(realized), np.nan)
    if ok.any():
        vals[ok] = fn(samples[ok], realized[ok])
    return vals.tolist()

def _expand(vals, ok):
    out = np.full(ok.size, np.nan)
    out[ok] = np.asarray(vals)
    return out.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--oos', type=int, default=252)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    from .gpu_guard19 import require_gpu
    require_gpu()
    outp = Path(a.out_dir) / f'vi19_{a.ticker}{('_' + a.tag if a.tag else '')}.json'
    if outp.exists():
        print(f'skip {a.ticker} (exists)')
        return
    run(a.ticker, a.out_dir, oos=a.oos, steps=a.steps, seed=a.seed, tag=a.tag)
if __name__ == '__main__':
    main()
