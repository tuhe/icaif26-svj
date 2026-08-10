from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path
import numpy as np
MODELS_NUTS = ('sv_hn', 'svt_hn', 'svj_hn', 'svjt_hn')
MODELS_BASE = ('garch_t', 'fhs')
REF = 'svj_hn'

def stationary_indices(rng, T, L, n_rep):
    p = 1.0 / L
    idx = np.empty((n_rep, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, size=n_rep)
    newblock = rng.random((n_rep, T)) < p
    steps = rng.integers(0, T, size=(n_rep, T))
    for t in range(1, T):
        cont = (idx[:, t - 1] + 1) % T
        idx[:, t] = np.where(newblock[:, t], steps[:, t], cont)
    return idx

def load_panel(res_dir, oos, vi_dir=None):
    data = {m: {} for m in MODELS_NUTS + MODELS_BASE + ('vi_mf_hn',)}
    for m in MODELS_NUTS + MODELS_BASE:
        for f in glob.glob(os.path.join(res_dir, f'ladder19_{m}_*.json')):
            d = json.loads(Path(f).read_text())
            data[m][d['ticker']] = d
    if vi_dir:
        for f in glob.glob(os.path.join(vi_dir, 'vi19_*.json')):
            d = json.loads(Path(f).read_text())
            if '_st' not in os.path.basename(f):
                data['vi_mf_hn'][d['ticker']] = d
    have_all = set.intersection(*[set(data[m]) for m in MODELS_NUTS + MODELS_BASE]) if all((data[m] for m in MODELS_NUTS + MODELS_BASE)) else set()
    return (data, sorted(have_all))

def diag_aggregate(data, tickers):
    out = {}
    for m in MODELS_NUTS:
        n_days = 0
        n_routed = 0
        status = {'standard': 0, 'warned': 0, 'abstained': 0}
        causes = {'rhat': 0, 'mcse': 0, 'divergence': 0, 'depth': 0}
        divs = 0
        depth = 0
        lf_init = []
        ssz = []
        routed_day_idx = []
        for t in tickers:
            d = data[m][t]
            n_days += len(d['status'])
            n_routed += d['n_routed']
            routed_day_idx += d['routed_days']
            for s in ('standard', 'warned', 'abstained'):
                status[s] += d['n_status'][s]
            for rec in d['diag_initial']:
                if rec['max_rhat'] > 1.01:
                    causes['rhat'] += 1
                if rec['var99_rel_mcse'] > 0.01:
                    causes['mcse'] += 1
                if rec['divergences_total'] > 0:
                    causes['divergence'] += 1
                if rec['depth_hits_total'] > 0:
                    causes['depth'] += 1
                divs += rec['divergences_total']
                depth += rec['depth_hits_total']
                lf_init.append(rec['total_leapfrogs'])
                ssz += rec['step_size']
        hist = np.bincount(routed_day_idx, minlength=1) if routed_day_idx else np.array([0])
        out[m] = {'stock_days': n_days, 'routed': n_routed, 'routed_rate': round(n_routed / max(n_days, 1), 4), 'status_counts': status, 'status_rates': {k: round(v / max(n_days, 1), 5) for k, v in status.items()}, 'initial_screen_causes': causes, 'initial_divergences_total': divs, 'initial_depth_hits_total': depth, 'median_leapfrogs_per_fit': float(np.median(lf_init)) if lf_init else None, 'median_step_size': float(np.median(ssz)) if ssz else None, 'top5_routed_day_idx': np.argsort(hist)[::-1][:5].tolist()}
    return out

def _score_matrix(data, tickers, model, key):
    rows = [np.asarray(data[model][t]['scores'][key], float) for t in tickers]
    T = min((len(r) for r in rows))
    return np.stack([r[-T:] for r in rows])

def score_tables(data, tickers, L=10, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    ref5 = _score_matrix(data, tickers, REF, 'crps_fair')
    T = ref5.shape[1]
    idx = stationary_indices(rng, T, L, B)
    for key, label in (('crps_fair', 'overall fair CRPS'), ('twcrps1', '1% twCRPS')):
        Ms = _score_matrix(data, tickers, REF, key)
        blk = {}
        for alt in [m for m in MODELS_NUTS + MODELS_BASE if m != REF]:
            Ma = _score_matrix(data, tickers, alt, key)
            skill_pt = 1.0 - Ms.mean() / Ma.mean()
            sb = 1.0 - Ms[:, idx].mean(axis=(0, 2)) / Ma[:, idx].mean(axis=(0, 2))
            win_pt = float(np.mean(Ms.mean(1) < Ma.mean(1)))
            wb = (Ms[:, idx].mean(axis=2) < Ma[:, idx].mean(axis=2)).mean(axis=0)
            blk[alt] = {'pooled_skill_pct': round(100 * skill_pt, 3), 'skill_ci95_pct': [round(100 * float(np.quantile(sb, q)), 3) for q in (0.025, 0.975)], 'win_rate': round(win_pt, 3), 'win_ci95': [round(float(np.quantile(wb, q)), 3) for q in (0.025, 0.975)]}
        out[key] = {'label': label, 'vs': blk}
    return out

def fair_vs_pwm(data, tickers):
    out = {}
    for m in MODELS_NUTS + MODELS_BASE:
        f = _score_matrix(data, tickers, m, 'crps_fair').mean()
        p = _score_matrix(data, tickers, m, 'crps_pwm').mean()
        out[m] = {'fair': round(float(f), 6), 'pwm': round(float(p), 6), 'pwm_minus_fair_pct': round(100 * (p / f - 1.0), 4)}
    return out

def breach_rates(data, tickers):
    out = {}
    for m in MODELS_NUTS + MODELS_BASE:
        rates = {}
        for lvl in ('0.05', '0.025', '0.01', '0.005'):
            n_b = 0
            n_d = 0
            for t in tickers:
                d = data[m][t]
                v = np.asarray(d['scores']['var'][lvl], float)
                y = np.asarray(d['realized'], float)[-v.size:]
                n_b += int((y < v).sum())
                n_d += v.size
            rates[lvl] = round(n_b / max(n_d, 1), 5)
        out[m] = rates
    return out

def remediation_sensitivity(data, tickers):
    diffs = {}
    for m in MODELS_NUTS:
        d_fair = []
        for t in tickers:
            d = data[m][t]
            sr = d.get('scores_initial_routed') or {}
            if not sr or not sr.get('days'):
                continue
            days = sr['days']
            fin = np.asarray(d['scores']['crps_fair'], float)[days]
            ini = np.asarray(sr['crps_fair'], float)
            d_fair += (fin - ini).tolist()
        if d_fair:
            diffs[m] = {'n_routed_days': len(d_fair), 'median_final_minus_initial_crps': round(float(np.median(d_fair)), 6), 'p90_abs': round(float(np.quantile(np.abs(d_fair), 0.9)), 6)}
    return diffs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oos', type=int, default=252)
    ap.add_argument('--res-dir', default=None)
    ap.add_argument('--vi-dir', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res_dir = a.res_dir or f'results/ladder19/oos{a.oos}'
    vi_dir = a.vi_dir or f'results/vi19/oos{a.oos}'
    data, tickers = load_panel(res_dir, a.oos, vi_dir=vi_dir if os.path.isdir(vi_dir) else None)
    counts = {m: len(data[m]) for m in data}
    print('artifact counts:', counts, '| complete tickers:', len(tickers))
    out = {'oos': a.oos, 'n_complete_tickers': len(tickers), 'artifact_counts': counts, 'diagnostics': diag_aggregate(data, tickers) if tickers else {}, 'scores': score_tables(data, tickers) if tickers else {}, 'fair_vs_pwm': fair_vs_pwm(data, tickers) if tickers else {}, 'breach_rates': breach_rates(data, tickers) if tickers else {}, 'remediation_sensitivity': remediation_sensitivity(data, tickers) if tickers else {}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('n_complete_tickers',)}, indent=1))
if __name__ == '__main__':
    main()
