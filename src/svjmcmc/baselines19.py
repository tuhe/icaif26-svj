from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .data import load_returns
from .tail_scores19 import PROBS_L19, VAR_LEVELS_L19, crps_fair_per_day, crps_pwm_per_day, qgrid_from_samples, twcrps_from_quantiles, var_es_from_samples
N_SIM = 9600

def _fit_status(res):
    ok = getattr(res, 'convergence_flag', 0) == 0 and np.all(np.isfinite(np.asarray(res.params, float)))
    return bool(ok)

def _daily_baseline(r, oos_start, kind, seed):
    from arch import arch_model
    rng = np.random.default_rng(seed)
    N = r.size
    days = list(range(oos_start, N))
    samples = np.zeros((len(days), N_SIM))
    status = []
    good = None
    good_resid = None
    for k, t in enumerate(days):
        dist = 't' if kind == 'garch_t' else 'normal'
        am = arch_model(r[:t], mean='Constant', vol='GARCH', p=1, q=1, dist=dist)
        res = am.fit(disp='off')
        st = 'ok'
        if not _fit_status(res):
            sv = np.asarray(good.params, float) if good is not None else None
            res2 = am.fit(disp='off', starting_values=sv, options={'maxiter': 1000})
            if _fit_status(res2):
                res, st = (res2, 'retried')
            else:
                st = 'carried'
                res = good
        if res is None:
            raise RuntimeError(f'{kind}: no converged fit by day index {k}')
        if st != 'carried':
            good = res
            if kind == 'fhs':
                sr = np.asarray(res.resid / res.conditional_volatility)
                good_resid = sr[np.isfinite(sr)]
        f = good.forecast(horizon=1, reindex=False)
        mu = float(good.params.get('mu', 0.0))
        var1 = float(f.variance.values[-1, 0])
        if kind == 'garch_t':
            nu = float(good.params.get('nu', 8.0))
            z = rng.standard_t(df=nu, size=N_SIM) / np.sqrt(nu / (nu - 2))
        else:
            z = rng.choice(good_resid, size=N_SIM, replace=True)
        samples[k] = mu + np.sqrt(var1) * z
        status.append(st)
    return (samples, status)

def run(ticker, out_dir, oos=252, seed=0):
    t0 = time.time()
    r = np.asarray(load_returns(ticker), float)
    N = r.size
    oos_start = N - oos
    realized = r[oos_start:]
    thr5 = float(np.quantile(r[:-oos], 0.05))
    thr1 = float(np.quantile(r[:-oos], 0.01))
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    for kind in ('garch_t', 'fhs'):
        fout = outp / f'ladder19_{kind}_{ticker}.json'
        if fout.exists():
            print(f'skip {kind}/{ticker} (exists)', flush=True)
            continue
        samples, status = _daily_baseline(r, oos_start, kind, seed + (7 if kind == 'garch_t' else 8))
        qg = qgrid_from_samples(samples)
        var, es = var_es_from_samples(samples)
        out = {'ticker': ticker, 'model': kind, 'oos': oos, 'host': 'anon', 'seed': seed, 'refit_every': 1, 'n_pred_draws': N_SIM, 'retry_policy': '1 retry (last-good starting_values, maxiter 1000); else carry last good', 'thr_train': {'q05': thr5, 'q01': thr1}, 'realized': realized.tolist(), 'fit_status': status, 'n_fit_status': {s: status.count(s) for s in ('ok', 'retried', 'carried')}, 'scores': {'crps_fair': crps_fair_per_day(samples, realized).tolist(), 'crps_pwm': crps_pwm_per_day(samples, realized).tolist(), 'twcrps5': twcrps_from_quantiles(qg, PROBS_L19, realized, thr5).tolist(), 'twcrps1': twcrps_from_quantiles(qg, PROBS_L19, realized, thr1).tolist(), 'pit': (samples <= realized[:, None]).mean(axis=1).tolist(), 'var': {str(a): var[:, j].tolist() for j, a in enumerate(VAR_LEVELS_L19)}, 'es': {str(a): es[:, j].tolist() for j, a in enumerate(VAR_LEVELS_L19)}}, 'wall_s': round(time.time() - t0, 1)}
        fout.write_text(json.dumps(out, default=float))
        np.savez_compressed(outp / f'ladder19_{kind}_{ticker}_qgrid.npz', probs=PROBS_L19, qgrid=qg, realized=realized)
        print('SUMMARY ' + json.dumps({'ticker': ticker, 'model': kind, 'n_fit_status': out['n_fit_status'], 'wall_s': out['wall_s']}), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--oos', type=int, default=252)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    run(a.ticker, a.out_dir, oos=a.oos, seed=a.seed)
if __name__ == '__main__':
    main()
