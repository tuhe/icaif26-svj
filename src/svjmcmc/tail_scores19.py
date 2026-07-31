from __future__ import annotations
import numpy as np
_EXTRA_LO = np.array([0.001, 0.002, 0.003, 0.004])
PROBS_L19 = np.unique(np.concatenate([_EXTRA_LO, np.arange(1, 200) / 200.0, 1.0 - _EXTRA_LO]))
VAR_LEVELS_L19 = (0.05, 0.025, 0.01, 0.005)

def crps_fair_per_day(samples: np.ndarray, realized: np.ndarray) -> np.ndarray:
    s = np.asarray(samples, float)
    y = np.asarray(realized, float)
    Nd, m = s.shape
    t1 = np.abs(s - y[:, None]).mean(axis=1)
    ss = np.sort(s, axis=1)
    idx = np.arange(m)
    t2 = 2.0 * (ss * (2 * idx - m + 1)).sum(axis=1) / (m * m)
    return t1 - 0.5 * t2 * (m / (m - 1.0))

def crps_pwm_per_day(samples: np.ndarray, realized: np.ndarray) -> np.ndarray:
    s = np.asarray(samples, float)
    y = np.asarray(realized, float)
    Nd, m = s.shape
    t1 = np.abs(s - y[:, None]).mean(axis=1)
    ss = np.sort(s, axis=1)
    idx = np.arange(m)
    t2 = 2.0 * (ss * (2 * idx - m + 1)).sum(axis=1) / (m * m)
    return t1 - 0.5 * t2

def qgrid_from_samples(samples: np.ndarray, probs: np.ndarray=PROBS_L19) -> np.ndarray:
    return np.quantile(np.asarray(samples, float), probs, axis=1).T

def twcrps_from_quantiles(qgrid, probs, realized, thr) -> np.ndarray:
    q = np.minimum(np.asarray(qgrid, float), thr)
    y = np.minimum(np.asarray(realized, float), thr)[:, None]
    p = np.asarray(probs, float)[None, :]
    qs = np.where(y <= q, (1 - p) * (q - y), p * (y - q))
    return 2.0 * qs.mean(axis=1)

def var_es_from_samples(samples: np.ndarray, levels=VAR_LEVELS_L19):
    s = np.asarray(samples, float)
    var = np.stack([np.quantile(s, a, axis=1) for a in levels], axis=1)
    es = np.empty_like(var)
    for j in range(len(levels)):
        for d in range(s.shape[0]):
            tail = s[d][s[d] <= var[d, j]]
            es[d, j] = tail.mean() if tail.size else var[d, j]
    return (var, es)
