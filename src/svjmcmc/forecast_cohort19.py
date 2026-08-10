from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .forecast_ladder19 import run_one_model
COHORT_DIR = Path(os.environ.get('SVJMCMC_RETURNS', 'data/returns'))
START, END = ('2018-01-01', '2024-12-31')

def load_manifest(cohort: str, cohort_dir: Path=COHORT_DIR) -> dict:
    return json.loads((cohort_dir / f'cohort_{cohort}.json').read_text())

def member_record(manifest: dict, ident: str) -> dict:
    for m in manifest['members']:
        if m['id'] == ident:
            return m
    raise KeyError(f'{ident} not in cohort {manifest['tag']}')

def load_cohort_returns(ident: str, cohort_dir: Path=COHORT_DIR) -> np.ndarray:
    p = cohort_dir / f'{ident}_{START}_{END}.parquet'
    return pd.read_parquet(p)['ret_pct'].to_numpy(float)

def cohort_meta(manifest: dict, rec: dict) -> dict:
    return {'cohort': manifest['tag'], 'cohort_date': manifest['cohort_date'], 'window_start': manifest['window_start'], 'window_end': manifest['window_end'], 'n_window_sessions': manifest['n_sessions'], 'min_hist': manifest['min_hist'], 'permno': rec['permno'], 'ticker': rec['ticker'], 'delist': rec['delist'], 'floored_days': rec['floored_days'], 'n_oos': rec['n_oos'], 'window_day_idx': rec['window_day_idx'], 'data_source': 'CRSP point-in-time cohort (survivorship fix)'}

def amend_json(path: Path, meta: dict):
    d = json.loads(path.read_text())
    d['cohort_meta'] = meta
    path.write_text(json.dumps(d, default=float))

def run_cohort_ladder(ident: str, models: list[str], cohort: str, out_dir: str, seed: int=0, chunk_days: int=64, cohort_dir: Path=COHORT_DIR):
    man = load_manifest(cohort, cohort_dir)
    rec = member_record(man, ident)
    if rec.get('status') != 'scored' or rec['n_oos'] <= 0:
        print(f'skip {ident}: status={rec.get('status')}', flush=True)
        return
    r = load_cohort_returns(ident, cohort_dir)
    assert rec['n_oos'] < r.size, (ident, rec['n_oos'], r.size)
    meta = cohort_meta(man, rec)
    for model in models:
        outp = Path(out_dir) / f'ladder19_{model}_{ident}.json'
        if outp.exists():
            print(f'skip {model}/{ident} (exists)', flush=True)
            continue
        run_one_model(ident, model, r, rec['n_oos'], out_dir, seed=seed, chunk_days=chunk_days)
        amend_json(outp, meta)

def _cohort_loader(ident: str, cohort_dir: Path):

    def _load(ticker, *args, **kwargs):
        assert ticker == ident, (ticker, ident)
        return load_cohort_returns(ident, cohort_dir)
    return _load

def _cohort_loader_any(cohort_dir: Path):

    def _load(ticker, *args, **kwargs):
        return load_cohort_returns(ticker, cohort_dir)
    return _load

def run_cohort_deploy(cohort: str, out_dir: str, batch: int=128, rep: int=0, seed: int=0, report_var: bool=False, cohort_dir: Path=COHORT_DIR):
    from . import data as _data
    from . import deploy_timing19
    man = load_manifest(cohort, cohort_dir)
    alive = [m['id'] for m in man['members'] if m.get('status') == 'scored' and m['last_date'] == man['window_end']]
    uni = Path(out_dir) / f'universe_cohort{cohort}.json'
    uni.parent.mkdir(parents=True, exist_ok=True)
    uni.write_text(json.dumps({'per_ticker': alive, 'cohort': cohort, 'rule': 'cohort members alive at window end', 'n': len(alive)}))
    print(f'cohort deployment roster: {len(alive)} names -> {uni}', flush=True)
    orig = _data.load_returns
    _data.load_returns = _cohort_loader_any(cohort_dir)
    try:
        deploy_timing19.run(out_dir, batch, rep, seed=seed, universe_json=str(uni), report_var=report_var)
    finally:
        _data.load_returns = orig

def run_cohort_baselines(ident: str, cohort: str, out_dir: str, seed: int=0, cohort_dir: Path=COHORT_DIR):
    from . import baselines19
    man = load_manifest(cohort, cohort_dir)
    rec = member_record(man, ident)
    if rec.get('status') != 'scored' or rec['n_oos'] <= 0:
        print(f'skip {ident}: status={rec.get('status')}', flush=True)
        return
    meta = cohort_meta(man, rec)
    orig = baselines19.load_returns
    baselines19.load_returns = _cohort_loader(ident, cohort_dir)
    try:
        baselines19.run(ident, out_dir, oos=rec['n_oos'], seed=seed)
    finally:
        baselines19.load_returns = orig
    for kind in ('garch_t', 'fhs'):
        p = Path(out_dir) / f'ladder19_{kind}_{ident}.json'
        if p.exists():
            amend_json(p, meta)

def run_cohort_vi(ident: str, cohort: str, out_dir: str, seed: int=0, steps: int=4000, cohort_dir: Path=COHORT_DIR):
    from . import forecast_vi19
    man = load_manifest(cohort, cohort_dir)
    rec = member_record(man, ident)
    if rec.get('status') != 'scored' or rec['n_oos'] <= 0:
        print(f'skip {ident}: status={rec.get('status')}', flush=True)
        return
    outp = Path(out_dir) / f'vi19_{ident}.json'
    if outp.exists():
        print(f'skip vi/{ident} (exists)', flush=True)
        return
    meta = cohort_meta(man, rec)
    orig = forecast_vi19.load_returns
    forecast_vi19.load_returns = _cohort_loader(ident, cohort_dir)
    try:
        forecast_vi19.run(ident, out_dir, oos=rec['n_oos'], steps=steps, seed=seed)
    finally:
        forecast_vi19.load_returns = orig
    amend_json(outp, meta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', default=None, help='cohort member id TICKER.PERMNO (not for deploy)')
    ap.add_argument('--cohort', required=True, choices=('252', '756'))
    ap.add_argument('--engine', default='ladder', choices=('ladder', 'baselines', 'vi', 'deploy'))
    ap.add_argument('--models', default='', help='ladder engine only')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk-days', type=int, default=64)
    ap.add_argument('--vi-steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=128, help='deploy engine only')
    ap.add_argument('--rep', type=int, default=0, help='deploy engine only')
    ap.add_argument('--report-var', action='store_true', help='deploy engine only')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--cohort-dir', default=str(COHORT_DIR))
    a = ap.parse_args()
    cdir = Path(a.cohort_dir)
    if a.engine == 'baselines':
        run_cohort_baselines(a.id, a.cohort, a.out_dir, seed=a.seed, cohort_dir=cdir)
        return
    from .gpu_guard19 import require_gpu
    require_gpu()
    if a.engine == 'deploy':
        run_cohort_deploy(a.cohort, a.out_dir, batch=a.batch, rep=a.rep, seed=a.seed, report_var=a.report_var, cohort_dir=cdir)
    elif a.engine == 'vi':
        run_cohort_vi(a.id, a.cohort, a.out_dir, seed=a.seed, steps=a.vi_steps, cohort_dir=cdir)
    else:
        run_cohort_ladder(a.id, [m.strip() for m in a.models.split(',') if m.strip()], a.cohort, a.out_dir, seed=a.seed, chunk_days=a.chunk_days, cohort_dir=cdir)
if __name__ == '__main__':
    main()
