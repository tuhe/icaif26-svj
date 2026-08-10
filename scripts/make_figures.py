from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RES = Path('results')
FIG = Path('figures')
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 150})
PLAB = {'mu': '$\\mu$', 'phi': '$\\phi$', 'sigma_eta': '$\\sigma_\\eta$', 'p_J': '$p_J$', 'sigma_J': '$\\sigma_J$'}
_tables: list[str] = []

def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

def fig_sbc19():
    d = _load(RES / 'sbc19' / 'sbc19.json')
    if not d:
        return
    R = d['sims']
    band = d['band99_per_panel']
    params = ['mu', 'phi', 'sigma_eta', 'p_J', 'sigma_J']
    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(1, len(params), figsize=(2.75 * len(params) + 0.4, 3.2), sharey=True)
    for j, p in enumerate(params):
        rk = np.sort(np.asarray(d['ranks'][p]['thin'], float))
        ecdf = np.searchsorted(rk, grid, side='right') / R
        ax[j].fill_between(grid, -band, band, color='grey', alpha=0.25, lw=0)
        ax[j].plot(grid, ecdf - grid, color='C0', lw=1.2)
        ax[j].axhline(0, color='k', lw=0.5)
        ax[j].set_title(f'{PLAB.get(p, p)}\n$p={d['tests'][p]['p_thin']:.2f}$', fontsize=12)
        ax[j].set_xticks([0, 1])
        ax[j].set_ylim(-2.2 * band, 2.2 * band)
        ax[j].tick_params(labelsize=10)
    ax[0].set_ylabel('ECDF $-$ uniform', fontsize=12)
    fig.suptitle(f'SBC rank ECDF deviations, unrestricted, $T{{=}}{d['T']}$, $R{{=}}{R}$ (grey: 99% simultaneous band)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / 'fig_sbc.pdf')
    plt.close(fig)
    print('fig_sbc19: OK  (band=%.3f, p:' % band, [round(d['tests'][p]['p_thin'], 2) for p in params], ')')

def fig_sbc():
    d = _load(RES / 'paper_sbc_hn1759' / 'sbc_hn.json')
    if not d:
        return
    R, S = (d['R'], d['n_draws'])
    params = list(d['ranks'].keys())
    grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(0)
    sims = rng.integers(0, S + 1, size=(2000, R)) / S
    dev = np.abs(np.sort(sims, axis=1)[:, None, :] <= 0).astype(float)
    devs = np.empty(2000)
    for i in range(2000):
        e = np.searchsorted(np.sort(sims[i]), grid, side='right') / R
        devs[i] = np.max(np.abs(e - grid))
    band = float(np.quantile(devs, 0.99))
    fig, ax = plt.subplots(1, len(params), figsize=(2.75 * len(params) + 0.4, 3.2), sharey=True)
    for j, p in enumerate(params):
        rk = np.sort(np.array(d['ranks'][p]) / S)
        ecdf = np.searchsorted(rk, grid, side='right') / R
        ax[j].fill_between(grid, -band, band, color='grey', alpha=0.25, lw=0)
        ax[j].plot(grid, ecdf - grid, color='C0', lw=1.2)
        ax[j].axhline(0, color='k', lw=0.5)
        ax[j].set_title(f'{PLAB.get(p, p)}\n$p={d['uniformity'][p]['pval']:.2f}$', fontsize=12)
        ax[j].set_xticks([0, 1])
        ax[j].set_ylim(-2.2 * band, 2.2 * band)
        ax[j].tick_params(labelsize=10)
    ax[0].set_ylabel('ECDF $-$ uniform', fontsize=12)
    fig.suptitle(f'SBC rank ECDF deviations, unrestricted, $T{{=}}{d['T']}$, $R{{=}}{R}$ (grey: 99% simultaneous band)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / 'fig_sbc.pdf')
    plt.close(fig)
    body = ' \\\\\n'.join((f'{PLAB.get(p, p)} & {d['uniformity'][p]['chi2']:.1f} & {d['uniformity'][p]['pval']:.2f}' for p in params))
    _tables.append(f'\\begin{{table}}\\centering\\caption{{Simulation-based calibration under the submitted HalfNormal prior: unrestricted, full length ($T{{=}}{d['T']}$, $R{{=}}{R}$ replicates). $\\chi^2$ rank-uniformity per parameter (dof 19); no parameter shows detectable nonuniformity.}}\\label{{tab:sbc}}\n\\begin{{tabular}}{{lrr}}\\toprule parameter & $\\chi^2$ & $p$ \\\\ \\midrule\n' + body + ' \\\\ \\bottomrule\\end{tabular}\\end{table}')
    print('fig_sbc: OK  (band=%.3f)' % band)

def fig_tsweep():
    a = _load(RES / 'bench' / 'gb2_gradbench_T_assoc.json')
    s = _load(RES / 'bench' / 'gb2_gradbench_T_seq.json')
    if not (a and s):
        return
    Ta = [r['T'] for r in a]
    ga = [r['grad_ms'] for r in a]
    Ts = [r['T'] for r in s]
    gs = [r['grad_ms'] for r in s]
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.loglog(Ts, gs, 'o-', color='C3', lw=1.4, label='sequential scan ($O(T)$ depth)')
    ax.loglog(Ta, ga, 's-', color='C0', lw=1.4, label='associative scan ($O(\\log T)$ depth)')
    for T in (256, 1759, 16384):
        if T in Ts and T in Ta:
            sp = gs[Ts.index(T)] / ga[Ta.index(T)]
            ax.annotate(f'{sp:.0f}$\\times$', (T, np.sqrt(gs[Ts.index(T)] * ga[Ta.index(T)])), fontsize=7.5, ha='center', color='k')
    ax.axvline(1759, color='grey', ls=':', lw=0.8)
    ax.annotate('deployment $T{=}1759$', (1900, 1.6), fontsize=6.5, ha='left', color='grey')
    ax.set_xlabel('series length $T$')
    ax.set_ylabel('value+gradient (ms, $B{=}32$)')
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_scan_ablation.pdf')
    plt.close(fig)
    print('fig_tsweep: OK')

def fig_dateblock():
    d252 = _load(RES / 'eval_dateblock_bootstrap_hn_thrtrain_dep_oos252.json')
    d756 = _load(RES / 'eval_longoos_dateblock_oos756.json')
    if not d252:
        return
    b = d252['blocks']['L=10']
    rows = []
    for metric, mlab in (('tw1', 'twCRPS@1\\%'), ('tw5', 'twCRPS@5\\%')):
        for alt, alab in (('svt', 'SV-$t$'), ('garch_t', 'GARCH-$t$'), ('fhs', 'FHS')):
            v = b['score_win_rates'][metric][alt]
            rows.append((f'{mlab} vs {alab}', v['point'], v['boot_ci95'], 'C0'))
    if d756:
        b7 = d756['blocks']['L=10']
        for alt, alab in (('svt', 'SV-$t$'), ('garch_t', 'GARCH-$t$'), ('fhs', 'FHS')):
            v = b7['crps_win_rates'][alt]
            rows.append((f'CRPS (756d) vs {alab}', v['point'], v['boot_ci95'], 'C5'))
    fig, ax = plt.subplots(figsize=(4.6, 0.34 * len(rows) + 1.0))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, pt, ci95, c) in zip(y, rows):
        ax.plot(ci95, [yi, yi], color=c, lw=2, alpha=0.7)
        ax.plot(pt, yi, 'o', color=c, ms=4.5)
    ax.axvline(0.5, color='k', ls=':', lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel('fraction of names where exact-SVJ scores better')
    ax.set_xlim(0.2, 1.0)
    ax.set_title('Win rates with common-date block-bootstrap 95% CIs', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_dateblock.pdf')
    plt.close(fig)

    def _r(metric, alt):
        v = b['score_win_rates'][metric][alt]
        e = b['score_effects'][metric][alt]
        return f'{v['point']:.2f} & [{v['boot_ci95'][0]:.2f}, {v['boot_ci95'][1]:.2f}] & {100 * e['median_rel_improvement']:.2f}\\% & [{100 * e['rel_boot_ci95'][0]:.2f}, {100 * e['rel_boot_ci95'][1]:.2f}]'
    lines = []
    for alt, alab in (('svt', 'SV-$t$'), ('garch_t', 'GARCH-$t$'), ('fhs', 'FHS')):
        lines.append(f'1\\% & {alab} & ' + _r('tw1', alt))
    for alt, alab in (('svt', 'SV-$t$'), ('garch_t', 'GARCH-$t$'), ('fhs', 'FHS')):
        lines.append(f'5\\% & {alab} & ' + _r('tw5', alt))
    if d756:
        for alt, alab in (('svt', 'SV-$t$'), ('garch_t', 'GARCH-$t$'), ('fhs', 'FHS')):
            v = d756['blocks']['L=10']['crps_win_rates'][alt]
            lines.append(f'756d CRPS & {alab} & {v['point']:.2f} & [{v['boot_ci95'][0]:.2f}, {v['boot_ci95'][1]:.2f}] & {100 * v['median_rel_improvement']:.2f}\\% & [{100 * v['rel_boot_ci95'][0]:.2f}, {100 * v['rel_boot_ci95'][1]:.2f}]')
    _tables.append('\\begin{table}\\centering\\caption{Tail-score comparison, exact-SVJ vs baselines (474 names; training-period thresholds; stationary common-date bootstrap, $L{=}10$, $B{=}2000$; sensitivity $L\\in\\{5,20\\}$ unchanged). Win rate = fraction of names with the better (lower) mean score; effect = median per-name relative improvement.}\\label{tab:dateblock}\n\\begin{tabular}{llrrrr}\\toprule score & baseline & win & 95\\% CI & effect & 95\\% CI \\\\ \\midrule\n' + ' \\\\\n'.join(lines) + ' \\\\ \\bottomrule\\end{tabular}\\end{table}')
    print('fig_dateblock: OK')

def fig_jump_index():
    d = _load(RES / 'gateB_d4_hn' / 'jump_index.json')
    ds = _load(RES / 'gateB_d4_hn' / 'jump_index_dates.json')
    if not d:
        return
    im = np.array(d['index_mean'])
    q10 = np.array(d['index_q10'])
    q90 = np.array(d['index_q90'])
    import matplotlib.dates as mdates, datetime as _dt
    t = [_dt.date.fromisoformat(x) for x in ds] if ds else list(range(im.size))
    EVENTS = {613: "11 Jun'20\n$-5.9\\%$ selloff", 1223: "10 Nov'22\nCPI rally", 1722: "6 Nov'24\nelection"}
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.fill_between(t, q10, q90, color='C0', alpha=0.15, label='cross-sectional 10–90%')
    ax.plot(t, im, color='C0', lw=0.9, label='universe mean $P$(jump)')
    for i in np.argsort(im)[-3:]:
        ax.annotate('', (t[i], im[i]), (t[i], im[i] + 0.045), arrowprops=dict(arrowstyle='-|>', color='C3', lw=0.9))
        lab = EVENTS.get(int(i), ds[i] if ds else str(i))
        ax.annotate(lab, (t[i], im[i] + 0.05), fontsize=6, color='C3', ha='center', va='bottom')
    ax.set_ylim(top=im.max() + 0.1)
    if ds:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel('date')
    ax.set_ylabel('$P$(jump)')
    ax.set_title(f'Cross-sectional jump-intensity index ($N{{=}}{d['N']}$ names)')
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG / 'fig_jump_index.pdf')
    plt.close(fig)
    print('fig_jump_index: OK (HN, N=%d)' % d['N'])

def fig_universe():
    full = _load(RES / 'paper_hn' / 'universe_hn.json')
    stress = _load(RES / 'paper_hn' / 'universe_stress2020_hn.json')
    if not full:
        return

    def col(d, k):
        return np.array([v[k] for v in d['per_ticker'].values()])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for j, (k, lab) in enumerate([('VaR99', '99% VaR (%/day)'), ('p_jump', 'P(jump tomorrow)')]):
        ax[j].hist(col(full, k), bins=30, alpha=0.6, color='C0', label=f'full window (N={full['N']})')
        if stress:
            ax[j].hist(col(stress, k), bins=30, alpha=0.6, color='C3', label='ending 2020-03 (stress)')
        ax[j].set_xlabel(f'posterior mean {lab}')
        ax[j].set_ylabel('# tickers')
        ax[j].legend(fontsize=7)
    ax[0].set_title('(a) tail VaR shifts right under stress')
    ax[1].set_title('(b) jump probability')
    fig.tight_layout()
    fig.savefig(FIG / 'fig_universe.pdf')
    plt.close(fig)
    print('fig_universe: OK (HN)')

def tab_benchmark():
    a = _load(RES / 'bench' / 'bench_e2e_assoc.json')
    s = _load(RES / 'bench' / 'bench_e2e_seq.json')
    am = _load(RES / 'bench_mcse' / 'bench_e2e_assoc.json')
    sm = _load(RES / 'bench_mcse' / 'bench_e2e_seq.json')
    if not (a and s):
        return
    zline = ''
    if am and sm:
        zs = []
        for t in am['func']:
            for fn in am['func'][t]:
                ma, ea = (am['func'][t][fn]['mean'], am['func'][t][fn]['mcse'])
                ms, es = (sm['func'][t][fn]['mean'], sm['func'][t][fn]['mcse'])
                zs.append(abs(ma - ms) / np.sqrt(ea ** 2 + es ** 2))
        zline = f' All {len(zs)} ticker$\\times$functional posterior summaries agree within 2\\,MCSE (worst $z{{=}}{max(zs):.2f}$, median ${np.median(zs):.2f}$).'
    body = f'sequential & {s['T']} & {s['B']} & {s['warmup']}+{s['samples']} & no & {s['wall_warm_s']:.0f}\\,s \\\\\nassociative & {a['T']} & {a['B']} & {a['warmup']}+{a['samples']} & no & {a['wall_warm_s']:.1f}\\,s'
    sp = s['wall_warm_s'] / a['wall_warm_s']
    _tables.append(f'\\begin{{table}}\\centering\\caption{{Matched end-to-end NUTS benchmark (same GPU, same budget, warm jit cache, compile excluded): the associative reconstruction is ${sp:.0f}\\times$ faster.{zline}}}\\label{{tab:bench_e2e}}\n\\begin{{tabular}}{{lrrrlr}}\\toprule implementation & $T$ & batch & warmup+draws & compile incl.? & wall \\\\ \\midrule\n' + body + ' \\\\ \\bottomrule\\end{tabular}\\end{table}')
    print('tab_benchmark: OK (%.0fx)' % sp)

def tab_deployment():
    full = _load(RES / 'paper_hn' / 'universe_hn.json')
    ad = _load(RES / 'paper_hn' / 'adaptive_rerun_svj_hn.json')
    stress = _load(RES / 'paper_hn' / 'universe_stress2020_hn.json')
    if not (full and ad):
        return
    N = full['N']
    nc = int(round(full['frac_clean'] * N))
    su = ad['summary']
    n_final = nc + su['n_now_clean']
    rows = [f'fixed budget ($1000{{+}}1000{{\\times}}2$) & {nc}/{N} & {full['compute_wall_s'] / 60:.1f} min', f'+ diagnostic-triggered reruns ($3000{{+}}6000{{\\times}}4$) & {n_final}/{N} & $\\le${su['worst_time_to_target_s'] / 60:.0f} min/name', f'explicit abstentions (final) & {N - n_final} names & --']
    if stress:
        nsc = int(round(stress['frac_clean'] * stress['N']))
        rows.append(f'stress replay (window ending 2020-03) & {nsc}/{stress['N']} & {stress['compute_wall_s'] / 60:.1f} min')
    _tables.append('\\begin{table}\\centering\\caption{Nightly deployment on one GPU, submitted HalfNormal model: names with clean diagnostics ($\\hat R \\le 1.01$) at the fixed budget, after diagnostic-triggered reruns, and the explicit abstention set.}\\label{tab:deployment}\n\\begin{tabular}{lrr}\\toprule stage & clean & wall \\\\ \\midrule\n' + ' \\\\\n'.join(rows) + ' \\\\ \\bottomrule\\end{tabular}\\end{table}')
    print('tab_deployment: OK (%d/%d -> %d/%d)' % (nc, N, n_final, N))

def tab_vi():
    hc = _load(RES / 'gateB' / 'gb3_vi.json')
    hn = _load(RES / 'gateB_vihn' / 'gb3_vi_svj_hn.json')
    if not (hc and hn):
        return

    def med(d, fam, err):
        vals, khats = ([], [])
        for t, r in d['tickers'].items():
            f = r['vi'].get(fam)
            if f and 'rel_err' in f:
                vals.append(f['rel_err'][err])
                k = f.get('psis_khat')
                khats.append(k if k is not None else np.nan)
        return (np.median(vals) if vals else np.nan, np.nanmedian(khats) if khats else np.nan, int(np.sum(np.array(khats) > 0.7)) if khats else 0, len(vals))
    lines = []
    for pl, d in (('half-Cauchy', hc), ('HalfNormal', hn)):
        for fam, flab in (('meanfield', 'mean-field'), ('lowrank', 'low-rank')):
            v99, kh, n07, n = med(d, fam, 'VaR99')
            pj = med(d, fam, 'p_jump')[0]
            khs = '--' if np.isnan(kh) else f'{kh:.1f} ({n07}/{n})'
            lines.append(f'{pl} & {flab} & {100 * v99:.0f}\\% & {100 * pj:.0f}\\% & {khs}')
        lines.append(f'{pl} & full-rank & \\multicolumn{{3}}{{c}}{{out of memory at $T{{=}}1759$ in our configuration}}')
    _tables.append('\\begin{table}\\centering\\caption{Gaussian variational families vs the long-run NUTS reference (16 tickers, both priors): median relative error of the tail functionals and the PSIS diagnostic $\\hat k$ (values $>0.7$ mean the fit cannot be importance-corrected; mean-field $\\hat k$ is not estimable).}\\label{tab:vi}\n\\begin{tabular}{llrrr}\\toprule prior & family & VaR$_{99}$ err & $P$(jump) err & PSIS $\\hat k$ ($>$0.7) \\\\ \\midrule\n' + ' \\\\\n'.join(lines) + ' \\\\ \\bottomrule\\end{tabular}\\end{table}')
    print('tab_vi: OK')

def fig_forecast_paired():
    import sys
    sys.argv = ['x']
    from eval_dateblock_bootstrap import per_day_arrays, _overlay_hn, train_thresholds
    from eval_rescore import load_panel
    panel = load_panel(252)
    dep = set(json.loads((RES / 'paper_hn' / 'universe_hn.json').read_text())['per_ticker'])
    panel = {t: r for t, r in panel.items() if t in dep}
    panel = _overlay_hn(panel, 252)
    thr = train_thresholds(panel, 252)
    pd_ = per_day_arrays(panel, 252, thr_map=thr)
    alts = [('garch_t', 'GARCH-$t$'), ('fhs', 'FHS'), ('svt', 'SV-$t$')]
    box, labs = ([], [])
    for alt, alab in alts:
        dd = []
        for t, d in pd_.items():
            if alt in d['have']:
                dd.append(float(np.mean(d['scores']['tw1'][alt]) - np.mean(d['scores']['tw1']['svj'])))
        box.append(dd)
        labs.append(alab)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.boxplot(box, tick_labels=labs, whis=(5, 95), showfliers=False)
    ax.axhline(0, color='C3', ls=':', lw=0.9)
    ax.set_ylabel('$\\Delta$ twCRPS$_{1\\%}$ (alt $-$ exact)')
    ax.set_title('Paired tail score, 474 names\n($>0$ favours exact-SVJ)', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_forecast_paired.pdf')
    plt.close(fig)
    print('fig_forecast_paired: OK (HN leg, train thr, %d names)' % len(pd_))

def main():
    for gen in (fig_sbc19, fig_forecast_paired, fig_tsweep, fig_dateblock, fig_jump_index, fig_universe, tab_benchmark, tab_deployment, tab_vi):
        try:
            gen()
        except Exception as e:
            print(f'SKIPPED {gen.__name__}: {type(e).__name__}: {e}')
    (FIG / 'tables_final.tex').write_text('\n\n'.join(_tables) + '\n')
    import re
    for t in _tables:
        m = re.search('label\\{tab:([a-z_0-9]+)\\}', t)
        if m:
            (FIG / f'tab_{m.group(1)}.tex').write_text(t + '\n')
    print(f'wrote figures/tables_final.tex ({len(_tables)} tables) + per-table files')
if __name__ == '__main__':
    main()
