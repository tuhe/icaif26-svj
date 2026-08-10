from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .batched import pad_and_mask
from .batched_indep_ext import run_independent_g
from .data import load_universe
from .fulluniverse_diag import _diag_one, RHAT_MAX, REL_MCSE_TARGET
MODEL = 'svj_hn'
REMED = dict(num_warmup=3000, num_samples=6000, n_chains=4, target_accept=0.95)

def _pass_stage(names, panel, batch, seed, keep_draws=None, **budget):
    out, wall = ({}, 0.0)
    for i in range(0, len(names), batch):
        sub = names[i:i + batch]
        R, M, L = pad_and_mask([panel[t] for t in sub])
        t0 = time.perf_counter()
        res = run_independent_g(R, M, L, model=MODEL, seed=seed, **budget)
        wall += time.perf_counter() - t0
        for b, t in enumerate(sub):
            out[t] = _diag_one(res, b, budget['num_samples'])
            if keep_draws is not None:
                keep_draws[t] = res.draws[b].reshape(-1, res.draws[b].shape[-1]).copy()
        print(json.dumps({'batch': [i, i + len(sub)], 'wall_s': round(wall, 1)}), flush=True)
    return (out, wall)

def _routed(rec):
    return rec['max_rhat'] > RHAT_MAX or rec['var99_rel_mcse'] > REL_MCSE_TARGET or rec['divergences_total'] > 0 or (sum(rec['depth_hits']) > 0)

def run(out_dir, batch, rep, seed=0, universe_json=None, report_var=False):
    from .gpu_guard19 import require_gpu
    require_gpu()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    if universe_json:
        from .data import load_returns
        keep = list(json.loads(Path(universe_json).read_text())['per_ticker'])
        panel = {}
        for t in keep:
            r = load_returns(t)
            if r is not None and len(r) >= 1759:
                panel[t] = np.asarray(r, float)
    else:
        panel = load_universe()
    names = sorted(panel)
    t_all = time.perf_counter()
    draws1 = {} if report_var else None
    s1, wall1 = _pass_stage(names, panel, batch, seed, keep_draws=draws1, num_warmup=1000, num_samples=1000, n_chains=2, target_accept=0.8)
    union = sorted((t for t in names if _routed(s1[t])))
    t_route = time.perf_counter()
    s2, wall2 = ({}, 0.0)
    draws2 = {} if report_var else None
    if union:
        s2, wall2 = _pass_stage(union, panel, 64, seed + 1, keep_draws=draws2, **REMED)
    status = {}
    for t in names:
        rec = s2.get(t, s1[t])
        if rec['max_rhat'] > RHAT_MAX or rec['var99_rel_mcse'] > REL_MCSE_TARGET:
            status[t] = 'abstained'
        elif rec['divergences_total'] > 0:
            status[t] = 'warned'
        else:
            status[t] = 'standard'
    var_report = {}
    if report_var:
        from .predvar19 import predictive_var_mcse
        t_var = time.perf_counter()
        for t in names:
            red = (draws2 or {}).get(t)
            nch = REMED['n_chains'] if red is not None else 2
            if red is None:
                red = draws1[t]
            q, mcse = predictive_var_mcse(red, nch, alpha=0.01)
            q95, _ = predictive_var_mcse(red, nch, alpha=0.05)
            var_report[t] = {'var99_pred': round(-q, 5), 'var99_mcse': round(mcse, 6), 'var95_pred': round(-q95, 5), 'stage': 'remediation' if t in (draws2 or {}) else 'initial'}
        print(json.dumps({'var_report_wall_s': round(time.perf_counter() - t_var, 1)}), flush=True)
    wall_total = time.perf_counter() - t_all
    out = {'model': MODEL, 'batch_policy': batch, 'rep': rep, 'seed': seed, 'host': 'anon', 'n_stocks': len(names), 'wall_initial_s': round(wall1, 1), 'wall_remediation_s': round(wall2, 1), 'wall_total_s': round(wall_total, 1), 'routing_overhead_s': round(t_route - t_all - wall1, 1), 'n_routed': len(union), 'routed': union, 'n_status': {s: sum((1 for v in status.values() if v == s)) for s in ('standard', 'warned', 'abstained')}, 'abstained': sorted((t for t, s in status.items() if s == 'abstained')), 'per_ticker_status': status, 'var_report': var_report}
    (outp / f'deploy19_B{batch}_rep{rep}.json').write_text(json.dumps(out, indent=1))
    print('SUMMARY ' + json.dumps({k: out[k] for k in ('batch_policy', 'rep', 'wall_initial_s', 'wall_remediation_s', 'wall_total_s', 'n_routed', 'n_status', 'host')}), flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, required=True)
    ap.add_argument('--rep', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--universe-json', default=None)
    ap.add_argument('--report-var', action='store_true', help='persist per-stock posterior-predictive VaR99/VaR95 (A3 definition)')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    run(a.out_dir, a.batch, a.rep, seed=a.seed, universe_json=a.universe_json, report_var=a.report_var)
if __name__ == '__main__':
    main()
