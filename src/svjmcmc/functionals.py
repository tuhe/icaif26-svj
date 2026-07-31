from __future__ import annotations
import numpy as np
N_INNER = 4000
_CRN_SEED = 20260611

def _crn_block():
    rng = np.random.default_rng(_CRN_SEED)
    return {'eta': rng.standard_normal(N_INNER), 'eps': rng.standard_normal(N_INNER), 'u_jump': rng.random(N_INNER), 'z_jump': rng.standard_normal(N_INNER)}
_CRN = _crn_block()

def per_draw_functionals(samples: dict, model: str) -> dict[str, np.ndarray]:
    mu = np.asarray(samples['mu'])
    phi = np.asarray(samples['phi'])
    sigma_eta = np.asarray(samples['sigma_eta'])
    h = np.asarray(samples['h'])
    h_T = h[:, -1]
    n = mu.shape[0]
    is_svj = model == 'svj'
    if is_svj:
        p_J = np.asarray(samples['p_J'])
        sigma_J = np.asarray(samples['sigma_J'])
    else:
        p_J = np.zeros(n)
        sigma_J = np.zeros(n)
    eta, eps, u, z = (_CRN['eta'], _CRN['eps'], _CRN['u_jump'], _CRN['z_jump'])
    sd0 = sigma_eta
    h_next = (mu + phi * (h_T - mu))[:, None] + sd0[:, None] * eta[None, :]
    r = np.exp(h_next / 2.0) * eps[None, :]
    if is_svj:
        jump = (u[None, :] < p_J[:, None]) * (sigma_J[:, None] * z[None, :])
        r = r + jump
    var95 = -np.percentile(r, 5.0, axis=1)
    var99 = -np.percentile(r, 1.0, axis=1)
    out = {'VaR95': var95, 'VaR99': var99, 'sigma_eta': sigma_eta, 'vol_T': np.exp(h_T / 2.0)}
    if is_svj:
        out['p_jump'] = p_J
        out['sigma_J'] = sigma_J
    return out

def functional_names(model: str) -> list[str]:
    base = ['VaR95', 'VaR99', 'sigma_eta', 'vol_T']
    return base + (['p_jump', 'sigma_J'] if model == 'svj' else [])
