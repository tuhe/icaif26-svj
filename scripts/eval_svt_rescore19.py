from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path
import numpy as np
WINSOR = 100.0
RES = 'results/ladder19/oos252'

def _stationary_idx(rng, T, L, B):
    p = 1.0 / L
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, size=B)
    newb = rng.random((B, T)) < p
    steps = rng.integers(0, T, size=(B, T))
    for t in range(1, T):
        idx[:, t] = np.where(newb[:, t], steps[:, t], (idx[:, t - 1] + 1) % T)
    return idx

def _crps_from_qgrid(q, probs, y, thr=None):
    if thr is not None:
        q = np.minimum(q, thr[:, None] if np.ndim(thr) else thr)
        y = np.minimum(y, thr)
    y = y[:, None]
    p = probs[None, :]
    pin = np.where(y <= q, (1 - p) * (q - y), p * (y - q))
    return 2.0 * pin.mean(axis=1)

def _load_model(model):
    out = {}
    for f in glob.glob(os.path.join(RES, f'ladder19_{model}_*_qgrid.npz')):
        t = os.path.basename(f).split('_')[3]
        z = np.load(f)
        j = os.path.join(RES, f'ladder19_{model}_{t}.json')
        thr1 = json.loads(Path(j).read_text())['thr_train']['q01'] if os.path.exists(j) else -3.0
        q = np.clip(np.asarray(z['qgrid'], float), -WINSOR, WINSOR)
        out[t] = {'q': q, 'p': np.asarray(z['probs'], float), 'y': np.asarray(z['realized'], float), 'thr1': float(thr1), 'degenerate': bool(np.abs(z['qgrid']).max() > WINSOR)}
    return out

def _score(model_data, tickers, metric):
    rows = []
    for t in tickers:
        d = model_data[t]
        thr = None if metric == 'crps' else d['thr1']
        rows.append(_crps_from_qgrid(d['q'], d['p'], d['y'], thr))
    T = min((len(r) for r in rows))
    return np.stack([r[-T:] for r in rows])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/ladder19/svt_rescore.json')
    ap.add_argument('--res-dir', default=None, help='override result dir (e.g. results/cohort19/oos252)')
    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    if a.res_dir:
        global RES
        RES = a.res_dir
    svj = _load_model('svj_hn')
    svt = _load_model('svt_hn')
    degen = sorted((t for t, d in svt.items() if d['degenerate']))
    common = sorted(set(svj) & set(svt))
    print(f'common tickers: {len(common)} | SV-t winsorized (degenerate-nu) tickers: {len(degen)} {degen}')
    rng = np.random.default_rng(0)
    out = {'winsor_pct': WINSOR, 'n_common': len(common), 'n_svt_winsorized': len(degen), 'svt_winsorized_tickers': degen, 'results': {}}
    for panel_name, ticks in (('all474', common), ('drop_degenerate', [t for t in common if t not in degen])):
        blk = {}
        for metric, label in (('crps', 'robust qgrid CRPS'), ('twcrps1', '1% twCRPS')):
            Ms = _score(svj, ticks, metric)
            Ma = _score(svt, ticks, metric)
            T = min(Ms.shape[1], Ma.shape[1])
            Ms, Ma = (Ms[:, -T:], Ma[:, -T:])
            idx = _stationary_idx(rng, T, a.L, a.B)
            skill_pt = 1.0 - Ms.mean() / Ma.mean()
            sb = 1.0 - Ms[:, idx].mean(axis=(0, 2)) / Ma[:, idx].mean(axis=(0, 2))
            win_pt = float((Ms.mean(1) < Ma.mean(1)).mean())
            wb = (Ms[:, idx].mean(axis=2) < Ma[:, idx].mean(axis=2)).mean(axis=0)
            blk[metric] = {'label': label, 'svj_mean': round(float(Ms.mean()), 5), 'svt_mean': round(float(Ma.mean()), 5), 'pooled_skill_pct': round(100 * skill_pt, 3), 'skill_ci95_pct': [round(100 * float(np.quantile(sb, q)), 3) for q in (0.025, 0.975)], 'win_rate': round(win_pt, 3), 'win_ci95': [round(float(np.quantile(wb, q)), 3) for q in (0.025, 0.975)]}
        out['results'][panel_name] = blk
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for pn, blk in out['results'].items():
        print(f'\n== {pn} ==')
        for m, r in blk.items():
            print(f'  {r['label']:18}: SVJ {r['svj_mean']} vs SV-t {r['svt_mean']} | skill {r['pooled_skill_pct']:+.2f}% CI{r['skill_ci95_pct']} | win {r['win_rate']} CI{r['win_ci95']}')
if __name__ == '__main__':
    main()
