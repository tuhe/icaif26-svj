from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .batched import pad_and_mask
from .data import load_returns
from .diagnostics import ess_rhat_table
from .ladder19 import LADDER_MODELS, REDUCE_COLS_L19, per_draw_functionals_l19, predictive_samples_l19, run_independent_l19
from .tail_scores19 import PROBS_L19, VAR_LEVELS_L19, crps_fair_per_day, crps_pwm_per_day, qgrid_from_samples, twcrps_from_quantiles, var_es_from_samples
RHAT_MAX = 1.01
REL_MCSE_TARGET = 0.01
N_PRED_DRAWS = 1200
N_INNER_PRED = 8
INIT_BUDGET = dict(num_warmup=600, num_samples=600, n_chains=2, target_accept=0.8)
REMED_BUDGET = dict(num_warmup=3000, num_samples=6000, n_chains=4, target_accept=0.95)
POLICY = 'route if max split-Rhat>1.01 (monitored set) or VaR99 rel-MCSE>1% or any post-warmup divergence or any depth hit; remediate 4ch 3000+6000 @0.95; standard/warned(residual div)/abstained(Rhat|MCSE)'

def _day_diag(res, b, model):
    d = res.draws[b]
    nc, S, ncol = d.shape
    flat = d.reshape(nc * S, ncol)
    f = per_draw_functionals_l19(flat, model)
    chains = {k: v.reshape(nc, S) for k, v in f.items()}
    tab = ess_rhat_table(chains)
    v99 = f['VaR99']
    ess99 = max(tab['VaR99']['ess_bulk'], 1.0)
    mcse = float(v99.std() / np.sqrt(ess99))
    rel = mcse / max(abs(float(v99.mean())), 1e-09)
    divs = res.extra['divergences_per_lane'][b]
    dhits = res.extra['depth_hits_per_lane'][b]
    lfs = res.extra['leapfrogs_per_lane'][b]
    sszs = res.extra['step_size_per_lane'][b]
    return {'max_rhat': round(max((v['rhat'] for v in tab.values())), 5), 'rhat': {k: round(v['rhat'], 5) for k, v in tab.items()}, 'ess_bulk_min': round(min((v['ess_bulk'] for v in tab.values())), 1), 'var99_mean': round(float(v99.mean()), 5), 'var99_rel_mcse': round(rel, 6), 'divergences': [int(x) for x in divs], 'divergences_total': int(sum(divs)), 'depth_hits_total': int(sum(dhits)), 'total_leapfrogs': int(sum(lfs)), 'step_size': [round(float(x), 6) for x in sszs]}

def _flagged(rec):
    return rec['max_rhat'] > RHAT_MAX or rec['var99_rel_mcse'] > REL_MCSE_TARGET or rec['divergences_total'] > 0 or (rec['depth_hits_total'] > 0)

def _status(rec):
    if rec['max_rhat'] > RHAT_MAX or rec['var99_rel_mcse'] > REL_MCSE_TARGET:
        return 'abstained'
    if rec['divergences_total'] > 0:
        return 'warned'
    return 'standard'

def _pred_matrix(draws_by_day, model, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for red in draws_by_day:
        m = red.shape[0]
        if m > N_PRED_DRAWS:
            red = red[::m // N_PRED_DRAWS][:N_PRED_DRAWS]
        rows.append(predictive_samples_l19(red, model, N_INNER_PRED, rng))
    return np.stack(rows)

def _scores(samples, realized, thr5, thr1):
    qg = qgrid_from_samples(samples)
    var, es = var_es_from_samples(samples)
    return ({'crps_fair': crps_fair_per_day(samples, realized).tolist(), 'crps_pwm': crps_pwm_per_day(samples, realized).tolist(), 'twcrps5': twcrps_from_quantiles(qg, PROBS_L19, realized, thr5).tolist(), 'twcrps1': twcrps_from_quantiles(qg, PROBS_L19, realized, thr1).tolist(), 'pit': (samples <= realized[:, None]).mean(axis=1).tolist(), 'var': {str(a): var[:, j].tolist() for j, a in enumerate(VAR_LEVELS_L19)}, 'es': {str(a): es[:, j].tolist() for j, a in enumerate(VAR_LEVELS_L19)}}, qg)

def run_one_model(ticker, model, r, oos, out_dir, seed=0, chunk_days=64, mc_sens=False):
    N = r.size
    oos_start = N - oos
    days = list(range(oos_start, N))
    realized = np.array([r[t] for t in days])
    thr5 = float(np.quantile(r[:-oos], 0.05))
    thr1 = float(np.quantile(r[:-oos], 0.01))
    t_start = time.time()
    diag = [None] * len(days)
    draws_init = [None] * len(days)
    wall_init = 0.0
    for i in range(0, len(days), chunk_days):
        ck = days[i:i + chunk_days]
        R, M, L = pad_and_mask([r[:t] for t in ck])
        res = run_independent_l19(R, M, L, model=model, seed=seed, **INIT_BUDGET)
        wall_init += res.timing['wall_total']
        for b in range(len(ck)):
            diag[i + b] = {**_day_diag(res, b, model), 'stage': 'initial', 'seed': seed, **{k: INIT_BUDGET[k] for k in ('num_warmup', 'num_samples', 'n_chains', 'target_accept')}}
            draws_init[i + b] = res.draws[b].reshape(-1, res.draws[b].shape[-1])
        del res
        import jax
        jax.clear_caches()
        print(json.dumps({'ticker': ticker, 'model': model, 'stage': 'initial', 'chunk': [i, i + len(ck)], 'wall_s': round(wall_init, 1)}), flush=True)
    routed = [k for k in range(len(days)) if _flagged(diag[k])]
    draws_final = list(draws_init)
    diag_remed = {}
    wall_remed = 0.0
    remed_chunk = max(4, chunk_days // 4)
    for i in range(0, len(routed), remed_chunk):
        ck = routed[i:i + remed_chunk]
        R, M, L = pad_and_mask([r[:days[k]] for k in ck])
        res = run_independent_l19(R, M, L, model=model, seed=10000 + seed, **REMED_BUDGET)
        wall_remed += res.timing['wall_total']
        for b, k in enumerate(ck):
            diag_remed[k] = {**_day_diag(res, b, model), 'stage': 'remediation', 'seed': 10000 + seed, **{kk: REMED_BUDGET[kk] for kk in ('num_warmup', 'num_samples', 'n_chains', 'target_accept')}}
            draws_final[k] = res.draws[b].reshape(-1, res.draws[b].shape[-1])
        del res
        import jax
        jax.clear_caches()
        print(json.dumps({'ticker': ticker, 'model': model, 'stage': 'remediation', 'done': min(i + remed_chunk, len(routed)), 'of': len(routed), 'wall_s': round(wall_remed, 1)}), flush=True)
    final = [diag_remed.get(k, diag[k]) for k in range(len(days))]
    status = [_status(final[k]) if k in diag_remed else 'standard' if not _flagged(diag[k]) else _status(diag[k]) for k in range(len(days))]
    pred_final = _pred_matrix(draws_final, model, seed=777 + seed)
    scores_final, qg = _scores(pred_final, realized, thr5, thr1)
    scores_init_routed = {}
    if routed:
        pred_init = _pred_matrix([draws_init[k] for k in routed], model, seed=777 + seed)
        s_init, _ = _scores(pred_init, realized[routed], thr5, thr1)
        scores_init_routed = {'days': routed, **{k: v for k, v in s_init.items() if k in ('crps_fair', 'twcrps1', 'pit')}, 'var_0.01': s_init['var']['0.01'], 'var_0.005': s_init['var']['0.005']}
    mc = {}
    if mc_sens:
        for tag, sd, mult in (('s1', 1777, 1), ('s2', 2777, 1), ('s3', 3777, 1), ('big', 4777, 10)):
            rng = np.random.default_rng(sd + seed)
            rows = []
            for red in draws_final:
                m = red.shape[0]
                if m > N_PRED_DRAWS:
                    red = red[::m // N_PRED_DRAWS][:N_PRED_DRAWS]
                rows.append(predictive_samples_l19(red, model, N_INNER_PRED * mult, rng))
            pm = np.stack(rows)
            var, _ = var_es_from_samples(pm)
            mc[tag] = {'crps_fair': crps_fair_per_day(pm, realized).tolist(), 'var_0.01': var[:, 2].tolist(), 'var_0.005': var[:, 3].tolist(), 'n_draws': pm.shape[1]}
    out = {'ticker': ticker, 'model': model, 'oos': oos, 'host': 'anon', 'seed': seed, 'policy': POLICY, 'reduce_cols': REDUCE_COLS_L19[model], 'n_pred_draws': int(pred_final.shape[1]), 'thr_train': {'q05': thr5, 'q01': thr1}, 'days_abs_idx': days, 'realized': realized.tolist(), 'wall_initial_s': round(wall_init, 1), 'wall_remediation_s': round(wall_remed, 1), 'wall_total_s': round(time.time() - t_start, 1), 'n_routed': len(routed), 'routed_days': routed, 'status': status, 'n_status': {s: status.count(s) for s in ('standard', 'warned', 'abstained')}, 'diag_initial': diag, 'diag_remediation': diag_remed and {str(k): v for k, v in diag_remed.items()}, 'scores': scores_final, 'scores_initial_routed': scores_init_routed, 'mc_sensitivity': mc}
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / f'ladder19_{model}_{ticker}.json').write_text(json.dumps(out, default=float))
    np.savez_compressed(outp / f'ladder19_{model}_{ticker}_qgrid.npz', probs=PROBS_L19, qgrid=qg, realized=realized)
    print('SUMMARY ' + json.dumps({'ticker': ticker, 'model': model, 'n_status': out['n_status'], 'n_routed': len(routed), 'wall_total_s': out['wall_total_s']}), flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--models', default=','.join(LADDER_MODELS))
    ap.add_argument('--oos', type=int, default=252)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk-days', type=int, default=64)
    ap.add_argument('--mc-sens', action='store_true')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    from .gpu_guard19 import require_gpu
    require_gpu()
    r = np.asarray(load_returns(a.ticker), float)
    for model in [m.strip() for m in a.models.split(',') if m.strip()]:
        outp = Path(a.out_dir) / f'ladder19_{model}_{a.ticker}.json'
        if outp.exists():
            print(f'skip {model}/{a.ticker} (exists)', flush=True)
            continue
        run_one_model(a.ticker, model, r, a.oos, a.out_dir, seed=a.seed, chunk_days=a.chunk_days, mc_sens=a.mc_sens)
if __name__ == '__main__':
    main()
