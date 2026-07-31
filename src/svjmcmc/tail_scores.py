from __future__ import annotations
import numpy as np

def twcrps_sample(samples: np.ndarray, realized: np.ndarray, thr: float | np.ndarray) -> np.ndarray:
    s = np.minimum(np.asarray(samples, float), np.asarray(thr)[..., None] if np.ndim(thr) else thr)
    y = np.minimum(np.asarray(realized, float), thr)
    t1 = np.abs(s - y[:, None]).mean(axis=1)
    ss = np.sort(s, axis=1)
    Ns = s.shape[1]
    idx = np.arange(Ns)
    t2 = 2 * (ss * (2 * idx - Ns + 1)).sum(axis=1) / (Ns * Ns)
    return t1 - 0.5 * t2

def twcrps_from_quantiles(qgrid: np.ndarray, probs: np.ndarray, realized: np.ndarray, thr: float) -> np.ndarray:
    q = np.minimum(np.asarray(qgrid, float), thr)
    y = np.minimum(np.asarray(realized, float), thr)[:, None]
    p = np.asarray(probs, float)[None, :]
    qs = np.where(y <= q, (1 - p) * (q - y), p * (y - q))
    return 2.0 * qs.mean(axis=1)

def fz0_loss(var_f: np.ndarray, es_f: np.ndarray, realized: np.ndarray, alpha: float) -> np.ndarray:
    v = np.asarray(var_f, float)
    e = np.minimum(np.asarray(es_f, float), -1e-08)
    y = np.asarray(realized, float)
    hit = (y <= v).astype(float)
    return -(hit * (v - y)) / (alpha * e) + v / e + np.log(-e) - 1.0

def newey_west_var(d: np.ndarray, lags: int | None=None) -> float:
    d = np.asarray(d, float)
    n = d.size
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2 / 9)))
    d = d - d.mean()
    g0 = float((d * d).mean())
    s = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float((d[k:] * d[:-k]).mean())
        s += 2 * (1 - k / (lags + 1)) * gk
    return s / n

def dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> dict:
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    v = newey_west_var(d)
    t = float(d.mean() / np.sqrt(v)) if v > 0 else np.nan
    from scipy.stats import norm
    return {'t': t, 'p': float(2 * norm.sf(abs(t))), 'mean_diff': float(d.mean()), 'n': int(d.size)}

def mcs_lite(losses: dict[str, np.ndarray], alpha: float=0.1, B: int=2000, block: int=10, seed: int=0) -> list[str]:
    rng = np.random.default_rng(seed)
    names = sorted(losses)
    L = np.stack([np.asarray(losses[k], float) for k in names])
    keep = list(range(len(names)))
    n = L.shape[1]
    starts_max = max(n - block, 1)
    while len(keep) > 1:
        Lk = L[keep]
        dbar = Lk.mean(axis=1)[:, None] - Lk.mean(axis=1)[None, :]
        tmax_obs, tmax_boot = (0.0, np.zeros(B))
        nb = int(np.ceil(n / block))
        bs_means = np.zeros((B, len(keep)))
        for b in range(B):
            idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, starts_max, nb)])[:n]
            bs_means[b] = Lk[:, idx].mean(axis=1)
        se = bs_means.std(axis=0, ddof=1)
        tij = dbar / np.sqrt(se[:, None] ** 2 + se[None, :] ** 2 + 1e-300)
        tmax_obs = float(np.abs(tij).max())
        centered = bs_means - Lk.mean(axis=1)[None, :]
        for b in range(B):
            db = centered[b][:, None] - centered[b][None, :]
            tb = db / np.sqrt(se[:, None] ** 2 + se[None, :] ** 2 + 1e-300)
            tmax_boot[b] = np.abs(tb).max()
        p = float((tmax_boot >= tmax_obs).mean())
        if p >= alpha:
            break
        worst = int(np.argmax(Lk.mean(axis=1)))
        keep.pop(worst)
    return [names[i] for i in keep]

def _selfcheck() -> None:
    rng = np.random.default_rng(1)
    y = rng.standard_normal(4000)
    good = rng.standard_normal((4000, 400))
    bad = 2.0 * rng.standard_normal((4000, 400))
    thr = -1.0
    assert twcrps_sample(good, y, thr).mean() < twcrps_sample(bad, y, thr).mean()
    probs = np.arange(1, 200) / 200.0
    qg = np.quantile(good, probs, axis=1).T
    a = twcrps_from_quantiles(qg, probs, y, thr).mean()
    b = twcrps_sample(good, y, thr).mean()
    assert abs(a - b) / b < 0.05, (a, b)
    alpha = 0.025
    from scipy.stats import norm as _n
    v_true = _n.ppf(alpha)
    e_true = -_n.pdf(_n.ppf(alpha)) / alpha
    l_true = fz0_loss(np.full_like(y, v_true), np.full_like(y, e_true), y, alpha).mean()
    for f in (0.7, 1.4):
        l_bad = fz0_loss(np.full_like(y, v_true * f), np.full_like(y, e_true * f), y, alpha).mean()
        assert l_true < l_bad, (f, l_true, l_bad)
    r = dm_test(twcrps_sample(bad, y, thr), twcrps_sample(good, y, thr))
    assert r['t'] > 2 and r['p'] < 0.05
    print('tail_scores selfcheck: OK')
if __name__ == '__main__':
    _selfcheck()
