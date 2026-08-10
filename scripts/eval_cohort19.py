from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from eval_ladder19 import stationary_indices
MODELS_NUTS = ('sv_hn', 'svt_hn', 'svj_hn', 'svjt_hn')
MODELS_BASE = ('garch_t', 'fhs')
REF = 'svj_hn'

def load_panel(res_dir, n_sessions, vi_dir=None):
    data = {m: {} for m in MODELS_NUTS + MODELS_BASE + ('vi_mf_hn',)}
    for m in MODELS_NUTS + MODELS_BASE:
        for f in glob.glob(os.path.join(res_dir, f'ladder19_{m}_*.json')):
            d = json.loads(Path(f).read_text())
            cm = d.get('cohort_meta')
            if not cm:
                continue
            data[m][d['ticker']] = {'widx': cm['window_day_idx'], 'd': d}
    if vi_dir and os.path.isdir(vi_dir):
        for f in glob.glob(os.path.join(vi_dir, 'vi19_*.json')):
            d = json.loads(Path(f).read_text())
            cm = d.get('cohort_meta')
            if cm:
                data['vi_mf_hn'][d['ticker']] = {'widx': cm['window_day_idx'], 'd': d}
    return data

def matrix(data, model, ids, key, n_sessions, sub=None):
    M = np.full((len(ids), n_sessions), np.nan)
    for i, t in enumerate(ids):
        rec = data[model][t]
        v = rec['d']['scores'][key] if sub is None else rec['d']['scores'][key][sub]
        v = np.asarray(v, float)
        w = np.asarray(rec['widx'], int)
        assert v.size == w.size, (model, t, v.size, w.size)
        M[i, w] = v
    return M

def pair_stats(Ms, Ma, idx):
    W = np.isfinite(Ms) & np.isfinite(Ma)
    s, a = (np.where(W, Ms, 0.0), np.where(W, Ma, 0.0))
    skill = 1.0 - s.sum() / a.sum()
    sb = 1.0 - s[:, idx].sum(axis=(0, 2)) / a[:, idx].sum(axis=(0, 2))
    cnt = W.sum(axis=1).astype(float)
    ok = cnt > 0
    ms = np.where(ok, s.sum(axis=1) / np.maximum(cnt, 1), np.nan)
    ma = np.where(ok, a.sum(axis=1) / np.maximum(cnt, 1), np.nan)
    win = float(np.mean(ms[ok] < ma[ok]))
    cb = W[:, idx].sum(axis=2).astype(float)
    okb = cb > 0
    msb = s[:, idx].sum(axis=2) / np.maximum(cb, 1)
    mab = a[:, idx].sum(axis=2) / np.maximum(cb, 1)
    wb = np.where(okb, msb < mab, np.nan)
    win_b = np.nanmean(wb, axis=0)
    return {'n_names': int(ok.sum()), 'n_stock_days': int(W.sum()), 'pooled_skill_pct': round(100 * skill, 3), 'skill_ci95_pct': [round(100 * float(np.quantile(sb, q)), 3) for q in (0.025, 0.975)], 'win_rate': round(win, 3), 'win_ci95': [round(float(np.nanquantile(win_b, q)), 3) for q in (0.025, 0.975)]}

def breach(data, model, ids, n_sessions, lvl, idx):
    V = matrix(data, model, ids, 'var', n_sessions, sub=lvl)
    R = np.full((len(ids), n_sessions), np.nan)
    for i, t in enumerate(ids):
        rec = data[model][t]
        R[i, np.asarray(rec['widx'], int)] = np.asarray(rec['d']['realized'], float)
    W = np.isfinite(V) & np.isfinite(R)
    B = np.where(W, R < V, 0.0)
    rate = B.sum() / W.sum()
    rb = B[:, idx].sum(axis=(0, 2)) / np.maximum(W[:, idx].sum(axis=(0, 2)), 1)
    return {'rate_pct': round(100 * float(rate), 3), 'ci95_pct': [round(100 * float(np.quantile(rb, q)), 3) for q in (0.025, 0.975)], 'n_stock_days': int(W.sum())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True, choices=('252', '756'))
    ap.add_argument('--res-dir', required=True)
    ap.add_argument('--vi-dir', default=None)
    ap.add_argument('--models', default=None, help='comma list; default = all present')
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    n_sessions = {'252': 252, '756': 756}[a.cohort]
    data = load_panel(a.res_dir, n_sessions, vi_dir=a.vi_dir)
    counts = {m: len(v) for m, v in data.items()}
    print('artifact counts:', counts)
    models = [m.strip() for m in a.models.split(',')] if a.models else [m for m in MODELS_NUTS + MODELS_BASE if data[m]]
    assert REF in models and data[REF], 'reference svj_hn required'
    rng = np.random.default_rng(0)
    idx = stationary_indices(rng, n_sessions, 10, a.B)
    out = {'cohort': a.cohort, 'n_sessions': n_sessions, 'artifact_counts': counts, 'reference': REF, 'pairwise_vs_ref': {}, 'breach_1pct': {}, 'breach_0p5pct': {}, 'diagnostics': {}}
    for alt in [m for m in models if m != REF]:
        ids = sorted(set(data[REF]) & set(data[alt]))
        if not ids:
            continue
        res = {}
        for key in ('crps_fair', 'twcrps1'):
            Ms = matrix(data, REF, ids, key, n_sessions)
            Ma = matrix(data, alt, ids, key, n_sessions)
            res[key] = pair_stats(Ms, Ma, idx)
        out['pairwise_vs_ref'][alt] = res
    for m in models + (['vi_mf_hn'] if data['vi_mf_hn'] else []):
        if not data[m]:
            continue
        ids = sorted(data[m])
        out['breach_1pct'][m] = breach(data, m, ids, n_sessions, '0.01', idx)
        out['breach_0p5pct'][m] = breach(data, m, ids, n_sessions, '0.005', idx)
    for m in MODELS_NUTS:
        if not data[m]:
            continue
        n_days = sum((len(v['d']['status']) for v in data[m].values()))
        st = {s: sum((v['d']['n_status'][s] for v in data[m].values())) for s in ('standard', 'warned', 'abstained')}
        routed = sum((v['d']['n_routed'] for v in data[m].values()))
        out['diagnostics'][m] = {'stock_days': n_days, 'routed_rate': round(routed / max(n_days, 1), 4), 'status_rates': {k: round(v / max(n_days, 1), 5) for k, v in st.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('pairwise_vs_ref', 'breach_1pct')}, indent=1))
if __name__ == '__main__':
    main()
