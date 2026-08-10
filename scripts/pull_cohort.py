from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import wrds
OUT = Path(os.environ.get('SVJMCMC_RETURNS', 'data/returns'))
START, END = ('2018-01-01', '2024-12-31')
MIN_HIST = 504
FLOOR_RTOTAL = -0.99
AAPL_PERMNO = 14593
COHORTS = {'252': {'cohort_date': '2023-12-28', 'win_start': '2023-12-29', 'win_end': '2024-12-30'}, '756': {'cohort_date': '2021-12-27', 'win_start': '2021-12-28', 'win_end': '2024-12-30'}}

def pull_cohort(db, tag, spec):
    cd, ws, we = (spec['cohort_date'], spec['win_start'], spec['win_end'])
    print(f'=== cohort {tag}: members at {cd}, window {ws}..{we} ===', flush=True)
    members = db.raw_sql(f"\n        select permno, start, ending from crsp.dsp500list\n        where start <= '{cd}' and ending >= '{cd}'")
    members['permno'] = members['permno'].astype(int)
    permnos = sorted(members['permno'].tolist())
    print(f'members: {len(permnos)}', flush=True)
    names = db.raw_sql(f"\n        select permno, ticker, comnam, shrcd, namedt, nameendt from crsp.dsenames\n        where permno in ({','.join(map(str, permnos))})\n          and namedt <= '{cd}' and nameendt >= '{cd}'")
    names['permno'] = names['permno'].astype(int)
    names = names.drop_duplicates('permno').set_index('permno')
    cal = db.raw_sql(f"\n        select date from crsp.dsf where permno = {AAPL_PERMNO}\n          and date between '{ws}' and '{we}' order by date")
    sessions = [str(d) for d in cal['date'].tolist()]
    print(f'window sessions: {len(sessions)} (expect {tag})', flush=True)
    dsf = db.raw_sql(f"\n        select permno, date, ret from crsp.dsf\n        where permno in ({','.join(map(str, permnos))})\n          and date between '{START}' and '{END}' order by permno, date")
    dsf['permno'] = dsf['permno'].astype(int)
    dl = db.raw_sql(f"\n        select permno, dlstdt, dlstcd, dlret from crsp.dsedelist\n        where permno in ({','.join(map(str, permnos))})\n          and dlstdt between '{START}' and '{END}'\n          and dlstcd >= 200")
    dl['permno'] = dl['permno'].astype(int)
    dl = dl.set_index('permno')
    session_idx = {d: i for i, d in enumerate(sessions)}
    manifest = {'tag': tag, 'cohort_date': cd, 'window_start': ws, 'window_end': we, 'n_sessions': len(sessions), 'sessions': sessions, 'min_hist': MIN_HIST, 'floor_rtotal': FLOOR_RTOTAL, 'source': 'crsp.dsp500list + dsf + dsenames + dsedelist (WRDS)', 'members': []}
    for pn in permnos:
        g = dsf[dsf.permno == pn].copy()
        nm = names.loc[pn] if pn in names.index else None
        ticker = str(nm['ticker']).strip() if nm is not None and pd.notna(nm['ticker']) else f'P{pn}'
        ident = f'{ticker}.{pn}'
        rec = {'permno': int(pn), 'ticker': ticker, 'id': ident, 'comnam': str(nm['comnam']).strip() if nm is not None else None, 'shrcd': int(nm['shrcd']) if nm is not None and pd.notna(nm['shrcd']) else None, 'membership_start': str(members[members.permno == pn]['start'].iloc[0]), 'membership_end': str(members[members.permno == pn]['ending'].iloc[0]), 'delist': None, 'n_nan_ret_dropped': int(g['ret'].isna().sum()), 'floored_days': []}
        g = g.dropna(subset=['ret'])
        g = g[g['date'].astype(str) <= we]
        if not len(g):
            rec['status'] = 'no_returns'
            manifest['members'].append(rec)
            continue
        dates = [str(d) for d in g['date'].tolist()]
        rtot = g['ret'].to_numpy(float)
        if pn in dl.index:
            drow = dl.loc[pn]
            if isinstance(drow, pd.DataFrame):
                drow = drow.iloc[0]
            dld, dlc = (str(drow['dlstdt']), int(drow['dlstcd']) if pd.notna(drow['dlstcd']) else None)
            dlr = float(drow['dlret']) if pd.notna(drow['dlret']) else None
            rec['delist'] = {'date': dld, 'code': dlc, 'dlret': dlr, 'dlret_available': dlr is not None}
            if dlr is not None:
                if dates and dates[-1] == dld:
                    rtot[-1] = (1.0 + rtot[-1]) * (1.0 + dlr) - 1.0
                elif dates and dates[-1] < dld <= we:
                    dates.append(dld)
                    rtot = np.append(rtot, dlr)
        fl = rtot < FLOOR_RTOTAL
        for j in np.where(fl)[0]:
            rec['floored_days'].append({'date': dates[j], 'rtotal': float(rtot[j])})
        rtot = np.maximum(rtot, FLOOR_RTOTAL)
        ret_pct = 100.0 * np.log1p(rtot)
        in_win = [i for i, d in enumerate(dates) if d in session_idx]
        if in_win:
            start_i = max(min(in_win), MIN_HIST)
            n_oos = len(dates) - start_i
        else:
            start_i, n_oos = (None, 0)
        rec.update({'first_date': dates[0], 'last_date': dates[-1], 'n_obs': len(dates), 'oos_start_idx': int(start_i) if n_oos > 0 else None, 'n_oos': int(max(n_oos, 0)), 'window_day_idx': [session_idx[d] for d in dates[start_i:]] if n_oos > 0 else [], 'status': 'scored' if n_oos > 0 else 'no_eligible_days'})
        manifest['members'].append(rec)
        if n_oos > 0:
            OUT.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'date': dates, 'ret_pct': ret_pct}).to_parquet(OUT / f'{ident}_{START}_{END}.parquet')
    scored = [m for m in manifest['members'] if m.get('status') == 'scored']
    delisted = [m for m in scored if m['delist'] is not None]
    floored = [m for m in scored if m['floored_days']]
    partial = [m for m in scored if m['n_oos'] < len(sessions)]
    print(json.dumps({'cohort': tag, 'members': len(permnos), 'scored': len(scored), 'with_delist_record': len(delisted), 'floored_names': len(floored), 'partial_window': len(partial)}), flush=True)
    (OUT / f'cohort_{tag}.json').write_text(json.dumps(manifest, indent=1))
    return manifest

def main():
    db = wrds.Connection(wrds_username=os.environ['WRDS_USERNAME'])
    for tag, spec in COHORTS.items():
        pull_cohort(db, tag, spec)
    db.close()
    print('cohort pull complete:', OUT, flush=True)
if __name__ == '__main__':
    main()
