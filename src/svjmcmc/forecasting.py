from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import chi2 as _chi2

def kupiec_pof(exceed: np.ndarray, alpha: float) -> dict:
    exceed = np.asarray(exceed, bool)
    N = exceed.size
    x = int(exceed.sum())
    pi = x / max(N, 1)
    eps = 1e-12
    ll0 = x * np.log(alpha + eps) + (N - x) * np.log(1 - alpha + eps)
    ll1 = x * np.log(pi + eps) + (N - x) * np.log(1 - pi + eps)
    lr = 2 * (ll1 - ll0)
    return {'N': N, 'breaches': x, 'rate': pi, 'expected': alpha, 'LR_uc': float(lr), 'pval': float(_chi2.sf(lr, 1))}

def christoffersen(exceed: np.ndarray, alpha: float) -> dict:
    e = np.asarray(exceed, int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(e[:-1], e[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    eps = 1e-12
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_ind = n01 * np.log(pi01 + eps) + n00 * np.log(1 - pi01 + eps) + n11 * np.log(pi11 + eps) + n10 * np.log(1 - pi11 + eps)
    ll_pooled = (n01 + n11) * np.log(pi + eps) + (n00 + n10) * np.log(1 - pi + eps)
    lr_ind = 2 * (ll_ind - ll_pooled)
    uc = kupiec_pof(exceed, alpha)
    lr_cc = uc['LR_uc'] + lr_ind
    return {'LR_ind': float(lr_ind), 'pval_ind': float(_chi2.sf(lr_ind, 1)), 'LR_cc': float(lr_cc), 'pval_cc': float(_chi2.sf(lr_cc, 2))}

def pit_values(samples: np.ndarray, realized: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    realized = np.asarray(realized)
    return (samples <= realized[:, None]).mean(axis=1)

def pit_ks(u: np.ndarray) -> dict:
    u = np.sort(np.asarray(u))
    n = u.size
    cdf = np.arange(1, n + 1) / n
    d = float(np.max(np.abs(cdf - u))) if n else float('nan')
    return {'ks': d, 'n': n}

def crps_sample(samples: np.ndarray, realized: np.ndarray, tail_q: float | None=None) -> dict:
    samples = np.asarray(samples)
    realized = np.asarray(realized)
    Nd, Ns = samples.shape
    term1 = np.abs(samples - realized[:, None]).mean(axis=1)
    s = np.sort(samples, axis=1)
    idx = np.arange(Ns)
    term2 = 2 * (s * (2 * idx - Ns + 1)).sum(axis=1) / (Ns * Ns)
    crps = term1 - 0.5 * term2
    out = {'crps': float(crps.mean())}
    if tail_q is not None:
        thr = np.quantile(samples, tail_q, axis=1)
        mask = realized <= thr
        out[f'crps_tail_q{tail_q}'] = float(crps[mask].mean()) if mask.any() else float('nan')
        out['tail_n'] = int(mask.sum())
    return out

@dataclass
class Forecast:
    samples: np.ndarray
    realized: np.ndarray
    name: str = 'forecast'

    def var(self, alpha: float) -> np.ndarray:
        return -np.percentile(self.samples, 100 * alpha, axis=1)

    def per_day_crps(self, tail_q: float=0.05) -> dict:
        s = np.asarray(self.samples)
        r = np.asarray(self.realized)
        Nd, Ns = s.shape
        t1 = np.abs(s - r[:, None]).mean(axis=1)
        ss = np.sort(s, axis=1)
        idx = np.arange(Ns)
        t2 = 2 * (ss * (2 * idx - Ns + 1)).sum(axis=1) / (Ns * Ns)
        crps = t1 - 0.5 * t2
        thr = np.quantile(s, tail_q, axis=1)
        return {'crps': crps.tolist(), 'tail_mask': (r <= thr).tolist()}

    def breach_counts(self, levels=(0.05, 0.01, 0.005, 0.001)) -> dict:
        out = {}
        for a in levels:
            v = self.var(a)
            exc = int((-self.realized > v).sum())
            n = int(self.realized.size)
            out[str(a)] = {'breaches': exc, 'n': n, 'rate': exc / max(n, 1), 'nominal': a}
        return out

    def es_ratio(self, level: float=0.025) -> dict:
        v = self.var(level)
        thr = np.quantile(self.samples, level, axis=1)
        pred_es = np.array([-self.samples[d][self.samples[d] <= thr[d]].mean() if np.any(self.samples[d] <= thr[d]) else np.nan for d in range(self.realized.size)])
        breach = -self.realized > v
        if not breach.any():
            return {'level': level, 'n_breach': 0, 'ratio': float('nan')}
        real_es = float((-self.realized[breach]).mean())
        pe = float(np.nanmean(pred_es[breach]))
        return {'level': level, 'n_breach': int(breach.sum()), 'realized_es': real_es, 'predicted_es': pe, 'ratio': real_es / pe if pe else float('nan')}

    def score(self) -> dict:
        rep = {'name': self.name, 'n_days': int(self.realized.size)}
        for a, lab in [(0.05, 'VaR95'), (0.01, 'VaR99')]:
            v = self.var(a)
            exceed = -self.realized > v
            rep[lab] = {**kupiec_pof(exceed, a), **christoffersen(exceed, a)}
        u = pit_values(self.samples, self.realized)
        rep['pit'] = pit_ks(u)
        rep.update({f'{k}': v for k, v in crps_sample(self.samples, self.realized, tail_q=0.05).items()})
        return rep

def garch_t_forecast(returns: np.ndarray, oos_start: int, n_sim: int=4000, refit_every: int=25, seed: int=0) -> Forecast:
    from arch import arch_model
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, float)
    N = r.size
    samples = np.zeros((N - oos_start, n_sim))
    realized = r[oos_start:]
    res = None
    for k, t in enumerate(range(oos_start, N)):
        if res is None or (t - oos_start) % refit_every == 0:
            am = arch_model(r[:t], mean='Constant', vol='GARCH', p=1, q=1, dist='t')
            res = am.fit(disp='off')
        f = res.forecast(horizon=1, reindex=False)
        mu = float(res.params.get('mu', 0.0))
        var1 = float(f.variance.values[-1, 0])
        nu = float(res.params.get('nu', 8.0))
        z = rng.standard_t(df=nu, size=n_sim) / np.sqrt(nu / (nu - 2))
        samples[k] = mu + np.sqrt(var1) * z
    return Forecast(samples=samples, realized=realized, name='GARCH(1,1)-t')

def fhs_forecast(returns: np.ndarray, oos_start: int, n_sim: int=4000, refit_every: int=25, seed: int=0) -> Forecast:
    from arch import arch_model
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, float)
    N = r.size
    samples = np.zeros((N - oos_start, n_sim))
    realized = r[oos_start:]
    res = None
    std_resid = None
    for k, t in enumerate(range(oos_start, N)):
        if res is None or (t - oos_start) % refit_every == 0:
            am = arch_model(r[:t], mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
            res = am.fit(disp='off')
            std_resid = np.asarray(res.resid / res.conditional_volatility)
            std_resid = std_resid[np.isfinite(std_resid)]
        f = res.forecast(horizon=1, reindex=False)
        mu = float(res.params.get('mu', 0.0))
        var1 = float(f.variance.values[-1, 0])
        draws = rng.choice(std_resid, size=n_sim, replace=True)
        samples[k] = mu + np.sqrt(var1) * draws
    return Forecast(samples=samples, realized=realized, name='FHS-GARCH(1,1)')
