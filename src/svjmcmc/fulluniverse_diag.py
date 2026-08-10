from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .batched import pad_and_mask
from .batched_indep_ext import run_independent_g
from .batched_indep import per_ticker_func_chains
from .data import load_universe
from .diagnostics import ess_rhat_table
MODEL = 'svj_hn'
RHAT_MAX = 1.01
REL_MCSE_TARGET = 0.01

def _diag_one(res, b, num_samples):
    fc = per_ticker_func_chains(res, b)
    d = res.draws[b]
    fc['mu'] = d[:, :, 0]
    fc['phi'] = d[:, :, 1]
    tab = ess_rhat_table(fc)
    v99 = np.asarray(fc['VaR99']).reshape(-1)
    ess99 = max(tab['VaR99']['ess_bulk'], 1.0)
    mcse = float(v99.std() / np.sqrt(ess99))
    rel = mcse / max(abs(float(v99.mean())), 1e-09)
    divs = res.extra['divergences_per_lane'][b]
    dhits = res.extra['depth_hits_per_lane'][b]
    lfs = res.extra['leapfrogs_per_lane'][b]
    means = {k: round(float(np.asarray(v).mean()), 6) for k, v in fc.items()}
    mcses = {k: round(float(np.asarray(v).std() / np.sqrt(max(tab[k]['ess_bulk'], 1.0))), 8) for k, v in fc.items()}
    return {'means': means, 'mcse': mcses, 'rhat': {k: round(v['rhat'], 5) for k, v in tab.items()}, 'ess_bulk': {k: round(v['ess_bulk'], 1) for k, v in tab.items()}, 'max_rhat': round(max((v['rhat'] for v in tab.values())), 5), 'var99_mean': round(float(v99.mean()), 5), 'var99_rel_mcse': round(rel, 6), 'divergences': [int(x) for x in divs], 'divergences_total': int(sum(divs)), 'depth_hits': [int(x) for x in dhits], 'mean_leapfrogs': round(float(np.mean(lfs)), 2)}

def _run_panel(names, panel, num_warmup, num_samples, n_chains, target_accept, seed, batch):
    out, wall, ndiv = ({}, 0.0, 0)
    for i in range(0, len(names), batch):
        sub = names[i:i + batch]
        R, M, L = pad_and_mask([panel[t] for t in sub])
        res = run_independent_g(R, M, L, model=MODEL, seed=seed, num_warmup=num_warmup, num_samples=num_samples, n_chains=n_chains, target_accept=target_accept)
        wall += res.timing['wall_total']
        ndiv += res.extra['num_divergences']
        for b, t in enumerate(sub):
            out[t] = _diag_one(res, b, num_samples)
        print(json.dumps({'batch': [i, i + len(sub)], 'wall_s': round(res.timing['wall_total'], 1), 'divs': res.extra['num_divergences'], 'target_accept': target_accept}), flush=True)
    return (out, wall, ndiv)

def run(out_dir, universe_json=None, seed=0):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    panel = load_universe()
    if universe_json:
        keep = list(json.loads(Path(universe_json).read_text())['per_ticker'])
        missing = [t for t in keep if t not in panel]
        assert not missing, f'{len(missing)} roster names missing from load_universe() (sealed file / data cache not visible?): {missing[:8]}'
        panel = {t: panel[t] for t in keep}
    names = sorted(panel)
    print(f'panel: {len(names)} stocks', flush=True)
    assert names, 'empty panel: data cache not visible from this tree'
    t0 = time.perf_counter()
    s1, wall1, div1 = _run_panel(names, panel, 1000, 1000, 2, 0.8, seed, batch=len(names))
    fails1 = [t for t in names if s1[t]['max_rhat'] > RHAT_MAX]
    divstocks1 = [t for t in names if s1[t]['divergences_total'] > 0]
    union = sorted(set(fails1) | set(divstocks1))
    print(json.dumps({'stage1_wall_min': round(wall1 / 60, 1), 'divergences': div1, 'rhat_fail': len(fails1), 'div_stocks': len(divstocks1), 'union': len(union)}), flush=True)
    s2, wall2, div2 = ({}, 0.0, 0)
    if union:
        s2, wall2, div2 = _run_panel(union, panel, 3000, 6000, 4, 0.9, seed, batch=64)
    esc = sorted((t for t in union if s2[t]['divergences_total'] > 0 or s2[t]['max_rhat'] > RHAT_MAX))
    s3, wall3, div3 = ({}, 0.0, 0)
    if esc:
        s3, wall3, div3 = _run_panel(esc, panel, 3000, 6000, 4, 0.95, seed, batch=64)

    def final_rec(t):
        for stage, rec in (('s3', s3.get(t)), ('s2', s2.get(t)), ('s1', s1.get(t))):
            if rec is not None:
                return (stage, rec)
        raise KeyError(t)
    report, abstain = ({}, [])
    for t in names:
        stage, rec = final_rec(t)
        ok = rec['max_rhat'] <= RHAT_MAX and rec['var99_rel_mcse'] <= REL_MCSE_TARGET and (rec['divergences_total'] == 0)
        report[t] = {'stage': stage, 'pass': bool(ok), **rec}
        if not ok:
            abstain.append(t)
    result = {'model': MODEL, 'seed': seed, 'rule': 'max split-rhat<=1.01 over {VaR95,VaR99,sigma_eta,vol_T,p_J,sigma_J,mu,phi}; VaR99 rel-MCSE<=1%; zero post-warmup divergences', 'stage1': {'wall_min': round(wall1 / 60, 2), 'divergences': div1, 'rhat_fail_stocks': fails1, 'divergence_stocks': divstocks1, 'n_div_stocks': len(divstocks1)}, 'stage2': {'union_n': len(union), 'wall_min': round(wall2 / 60, 2), 'divergences': div2}, 'stage3': {'escalated_n': len(esc), 'escalated': esc, 'wall_min': round(wall3 / 60, 2), 'divergences': div3}, 'final': {'n_pass': sum((1 for r in report.values() if r['pass'])), 'abstain': abstain}, 'per_ticker_stage1': s1, 'per_ticker_stage2': s2, 'per_ticker_stage3': s3, 'per_ticker_final': report}
    (outp / 'universe_diag.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({'FINAL': result['final'], 'stage1': result['stage1']['divergences'], 'stage2': result['stage2'], 'stage3_div': div3, 'total_wall_min': round((wall1 + wall2 + wall3) / 60, 1)}), flush=True)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='results/paper_hn_diag')
    ap.add_argument('--universe-json', default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    run(a.out_dir, universe_json=a.universe_json, seed=a.seed)
if __name__ == '__main__':
    main()
