from __future__ import annotations
import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

def r2(v) -> str:
    return str(Decimal(str(v)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
LABELS = {'svj_hn': 'NUTS-SVJ', 'svt_hn': 'SV-$t$', 'sv_hn': 'Gaussian-SV', 'garch_t': 'GARCH-$t$', 'fhs': 'FHS', 'vi_mf_hn': 'VI-SVJ'}
VAR99_ORDER = ['svj_hn', 'svt_hn', 'sv_hn', 'garch_t', 'fhs', 'vi_mf_hn']
DATEBLOCK_ROWS = [('sv_hn', 'crps_fair', 'CRPS'), ('sv_hn', 'twcrps1', '1\\% twCRPS'), ('garch_t', 'twcrps1', '1\\% twCRPS'), ('fhs', 'twcrps1', '1\\% twCRPS')]

def tab_var99_cal(ev: dict) -> str:
    lines = ['\\begin{tabular}{lrr}\\toprule', 'method & realized breach & 95\\% CI \\\\ \\midrule']
    rows = []
    for m in VAR99_ORDER:
        b = ev['breach_1pct'].get(m)
        if b is None:
            continue
        rate = f'{r2(b['rate_pct'])}\\%'
        if m == 'vi_mf_hn':
            rate = f'\\mathbf{{{rate}}}'
        lo, hi = b['ci95_pct']
        rows.append(f'{LABELS[m]} & ${rate}$ & $[{r2(lo)}, {r2(hi)}]$ \\\\')
    rows[-1] += ' \\bottomrule'
    return '\n'.join(lines + rows + ['\\end{tabular}']) + '\n'

def tab_dateblock_short(ev: dict) -> str:
    lines = ['\\begin{tabular}{llrr}\\toprule', 'comparison & score & win (95\\% CI) & skill (95\\% CI) \\\\ \\midrule']
    rows = []
    for m, key, score_label in DATEBLOCK_ROWS:
        r = ev['pairwise_vs_ref'][m][key]
        wl, wh = r['win_ci95']
        sl, sh = r['skill_ci95_pct']
        sk = r2(r['pooled_skill_pct'])
        if not sk.startswith('-'):
            sk = '+' + sk
        rows.append(f'vs {LABELS[m]} & {score_label} & {r2(r['win_rate'])} [{r2(wl)}, {r2(wh)}] & ${sk}\\%$ [{r2(sl)}, {r2(sh)}] \\\\')
    rows[-1] += ' \\bottomrule'
    return '\n'.join(lines + rows + ['\\end{tabular}']) + '\n'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', dest='eval_json', required=True, help='eval_cohort252.json from scripts/eval_cohort19.py')
    ap.add_argument('--out-dir', default='figures')
    a = ap.parse_args()
    ev = json.loads(Path(a.eval_json).read_text())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, render in (('tab_var99_cal', tab_var99_cal), ('tab_dateblock_short', tab_dateblock_short)):
        t = render(ev)
        (out / f'{name}.tex').write_text(t)
        print(f'== {name}.tex\n{t}')
if __name__ == '__main__':
    main()
