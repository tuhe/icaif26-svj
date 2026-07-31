from __future__ import annotations
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, logsumexp
from jax.scipy.stats import beta as jbeta, gamma as jgamma, norm as jnorm
import blackjax
try:
    from blackjax.adaptation.base import get_filter_adapt_info_fn
except ImportError:
    get_filter_adapt_info_fn = None
from .model import MU_LOC, MU_SCALE, PHI_A, PHI_B, PJ_A, PJ_B, SIGMA_J_SCALE, SIGMA_ETA_HN_SCALE, _ar1_reconstruct, _gauss_logprob_logvar
from .batched_indep import IndepResult, _LOG2
NU_SHAPE_L19, NU_SCALE_L19 = (2.0, 10.0)
_LOGPI = float(np.log(np.pi))
LADDER_MODELS = ('sv_hn', 'svt_hn', 'svj_hn', 'svjt_hn')
REDUCE_COLS_L19 = {'sv_hn': ['mu', 'phi', 'sigma_eta', 'h_T'], 'svt_hn': ['mu', 'phi', 'sigma_eta', 'nu', 'h_T'], 'svj_hn': ['mu', 'phi', 'sigma_eta', 'p_J', 'sigma_J', 'h_T'], 'svjt_hn': ['mu', 'phi', 'sigma_eta', 'p_J', 'sigma_J', 'nu', 'h_T']}

def _studentt_logpdf(x, nu, scale):
    z = x / scale
    return gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * (jnp.log(nu) + _LOGPI) - jnp.log(scale) - 0.5 * (nu + 1.0) * jnp.log1p(z * z / nu)

def _lp_dynamics_hn(z_mu, phi_raw, z_se, h_std):
    mu = z_mu
    phi = 2.0 * phi_raw - 1.0
    sigma_eta = jnp.exp(z_se)
    lp = jnorm.logpdf(mu, MU_LOC, MU_SCALE)
    lp += jbeta.logpdf(phi_raw, PHI_A, PHI_B) + jnp.log(phi_raw) + jnp.log1p(-phi_raw)
    lp += jnorm.logpdf(sigma_eta, 0.0, SIGMA_ETA_HN_SCALE) + _LOG2 + z_se
    lp += jnorm.logpdf(h_std, 0.0, 1.0).sum()
    return (lp, mu, phi, sigma_eta)

def unc_ld_sv_hn(z, returns, mask):
    z_mu, z_phi, z_se, h_std = (z[0], z[1], z[2], z[3:])
    lp, mu, phi, sigma_eta = _lp_dynamics_hn(z_mu, jax.nn.sigmoid(z_phi), z_se, h_std)
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    lp += (_gauss_logprob_logvar(returns, h) * mask).sum()
    return lp

def init_z_sv_hn(returns, mask):
    T = returns.shape[0]
    v = jnp.maximum((returns ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0), 0.001)
    z = jnp.zeros(T + 3)
    return z.at[0].set(jnp.log(v)).at[1].set(2.5).at[2].set(jnp.log(0.2))

def _reduce_sv_hn(z, last, r):
    z_mu, z_phi, z_se, h_std = (z[0], z[1], z[2], z[3:])
    mu = z_mu
    phi = 2.0 * jax.nn.sigmoid(z_phi) - 1.0
    se = jnp.exp(z_se)
    h = _ar1_reconstruct(h_std, mu, phi, se)
    return jnp.array([mu, phi, se, h[last]])

def unc_ld_svt_hn(z, returns, mask):
    z_mu, z_phi, z_se, z_nu, h_std = (z[0], z[1], z[2], z[3], z[4:])
    lp, mu, phi, sigma_eta = _lp_dynamics_hn(z_mu, jax.nn.sigmoid(z_phi), z_se, h_std)
    nu = 2.0 + jnp.exp(z_nu)
    lp += jgamma.logpdf(nu - 2.0, NU_SHAPE_L19, scale=NU_SCALE_L19) + z_nu
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    scale = jnp.exp(h / 2.0) * jnp.sqrt((nu - 2.0) / nu)
    lp += (_studentt_logpdf(returns, nu, scale) * mask).sum()
    return lp

def init_z_svt_hn(returns, mask):
    T = returns.shape[0]
    v = jnp.maximum((returns ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0), 0.001)
    z = jnp.zeros(T + 4)
    return z.at[0].set(jnp.log(v)).at[1].set(2.5).at[2].set(jnp.log(0.2)).at[3].set(jnp.log(13.0))

def _reduce_svt_hn(z, last, r):
    z_mu, z_phi, z_se, z_nu, h_std = (z[0], z[1], z[2], z[3], z[4:])
    mu = z_mu
    phi = 2.0 * jax.nn.sigmoid(z_phi) - 1.0
    se = jnp.exp(z_se)
    h = _ar1_reconstruct(h_std, mu, phi, se)
    return jnp.array([mu, phi, se, 2.0 + jnp.exp(z_nu), h[last]])

def unc_ld_svj_hn(z, returns, mask):
    z_mu, z_phi, z_se, z_pj, z_sj, h_std = (z[0], z[1], z[2], z[3], z[4], z[5:])
    lp, mu, phi, sigma_eta = _lp_dynamics_hn(z_mu, jax.nn.sigmoid(z_phi), z_se, h_std)
    p_J = jax.nn.sigmoid(z_pj)
    sigma_J = jnp.exp(z_sj)
    lp += jbeta.logpdf(p_J, PJ_A, PJ_B) + jnp.log(p_J) + jnp.log1p(-p_J)
    lp += jnorm.logpdf(sigma_J, 0.0, SIGMA_J_SCALE) + _LOG2 + z_sj
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    logvar_jp = jnp.logaddexp(h, 2.0 * jnp.log(sigma_J))
    ll = logsumexp(jnp.stack([jnp.log1p(-p_J) + _gauss_logprob_logvar(returns, h), jnp.log(p_J) + _gauss_logprob_logvar(returns, logvar_jp)]), axis=0)
    lp += (ll * mask).sum()
    return lp

def init_z_svj_hn(returns, mask):
    T = returns.shape[0]
    v = jnp.maximum((returns ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0), 0.001)
    z = jnp.zeros(T + 5)
    return z.at[0].set(jnp.log(v)).at[1].set(2.5).at[2].set(jnp.log(0.2)).at[3].set(-2.9).at[4].set(jnp.log(3.0))

def _reduce_svj_hn(z, last, r):
    z_mu, z_phi, z_se, z_pj, z_sj, h_std = (z[0], z[1], z[2], z[3], z[4], z[5:])
    mu = z_mu
    phi = 2.0 * jax.nn.sigmoid(z_phi) - 1.0
    se = jnp.exp(z_se)
    h = _ar1_reconstruct(h_std, mu, phi, se)
    return jnp.array([mu, phi, se, jax.nn.sigmoid(z_pj), jnp.exp(z_sj), h[last]])

def unc_ld_svjt_hn(z, returns, mask):
    z_mu, z_phi, z_se, z_pj, z_sj, z_nu, h_std = (z[0], z[1], z[2], z[3], z[4], z[5], z[6:])
    lp, mu, phi, sigma_eta = _lp_dynamics_hn(z_mu, jax.nn.sigmoid(z_phi), z_se, h_std)
    p_J = jax.nn.sigmoid(z_pj)
    sigma_J = jnp.exp(z_sj)
    nu = 2.0 + jnp.exp(z_nu)
    lp += jbeta.logpdf(p_J, PJ_A, PJ_B) + jnp.log(p_J) + jnp.log1p(-p_J)
    lp += jnorm.logpdf(sigma_J, 0.0, SIGMA_J_SCALE) + _LOG2 + z_sj
    lp += jgamma.logpdf(nu - 2.0, NU_SHAPE_L19, scale=NU_SCALE_L19) + z_nu
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    fac = jnp.sqrt((nu - 2.0) / nu)
    scale_no = jnp.exp(0.5 * h) * fac
    scale_jp = jnp.sqrt(jnp.exp(h) + sigma_J ** 2) * fac
    ll = logsumexp(jnp.stack([jnp.log1p(-p_J) + _studentt_logpdf(returns, nu, scale_no), jnp.log(p_J) + _studentt_logpdf(returns, nu, scale_jp)]), axis=0)
    lp += (ll * mask).sum()
    return lp

def init_z_svjt_hn(returns, mask):
    T = returns.shape[0]
    v = jnp.maximum((returns ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0), 0.001)
    z = jnp.zeros(T + 6)
    return z.at[0].set(jnp.log(v)).at[1].set(2.5).at[2].set(jnp.log(0.2)).at[3].set(-2.9).at[4].set(jnp.log(3.0)).at[5].set(jnp.log(13.0))

def _reduce_svjt_hn(z, last, r):
    z_mu, z_phi, z_se, z_pj, z_sj, z_nu, h_std = (z[0], z[1], z[2], z[3], z[4], z[5], z[6:])
    mu = z_mu
    phi = 2.0 * jax.nn.sigmoid(z_phi) - 1.0
    se = jnp.exp(z_se)
    h = _ar1_reconstruct(h_std, mu, phi, se)
    return jnp.array([mu, phi, se, jax.nn.sigmoid(z_pj), jnp.exp(z_sj), 2.0 + jnp.exp(z_nu), h[last]])
NU_FLOOR_L19B = 4.0

def unc_ld_svt_hn_nu4(z, returns, mask):
    z_mu, z_phi, z_se, z_nu, h_std = (z[0], z[1], z[2], z[3], z[4:])
    lp, mu, phi, sigma_eta = _lp_dynamics_hn(z_mu, jax.nn.sigmoid(z_phi), z_se, h_std)
    nu = NU_FLOOR_L19B + jnp.exp(z_nu)
    lp += jgamma.logpdf(nu - NU_FLOOR_L19B, NU_SHAPE_L19, scale=NU_SCALE_L19) + z_nu
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    scale = jnp.exp(h / 2.0) * jnp.sqrt((nu - 2.0) / nu)
    lp += (_studentt_logpdf(returns, nu, scale) * mask).sum()
    return lp

def init_z_svt_hn_nu4(returns, mask):
    T = returns.shape[0]
    v = jnp.maximum((returns ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0), 0.001)
    z = jnp.zeros(T + 4)
    return z.at[0].set(jnp.log(v)).at[1].set(2.5).at[2].set(jnp.log(0.2)).at[3].set(jnp.log(11.0))

def _reduce_svt_hn_nu4(z, last, r):
    z_mu, z_phi, z_se, z_nu, h_std = (z[0], z[1], z[2], z[3], z[4:])
    mu = z_mu
    phi = 2.0 * jax.nn.sigmoid(z_phi) - 1.0
    se = jnp.exp(z_se)
    h = _ar1_reconstruct(h_std, mu, phi, se)
    return jnp.array([mu, phi, se, NU_FLOOR_L19B + jnp.exp(z_nu), h[last]])
REDUCE_COLS_L19['svt_hn_nu4'] = ['mu', 'phi', 'sigma_eta', 'nu', 'h_T']
_L19 = {'sv_hn': (unc_ld_sv_hn, init_z_sv_hn, _reduce_sv_hn, 4), 'svt_hn': (unc_ld_svt_hn, init_z_svt_hn, _reduce_svt_hn, 5), 'svj_hn': (unc_ld_svj_hn, init_z_svj_hn, _reduce_svj_hn, 6), 'svjt_hn': (unc_ld_svjt_hn, init_z_svjt_hn, _reduce_svjt_hn, 7), 'svt_hn_nu4': (unc_ld_svt_hn_nu4, init_z_svt_hn_nu4, _reduce_svt_hn_nu4, 5)}
DEPTH_HIT_LEAPFROGS = 1023

def run_independent_l19(returns, mask, lengths, *, model, seed=0, num_warmup=600, num_samples=600, n_chains=2, target_accept=0.8):
    ld_fn, init_fn, reduce_fn, ncol = _L19[model]
    R = jnp.asarray(returns)
    M = jnp.asarray(mask)
    B, Tmax = R.shape

    def run_one(rng, r, m, z0):
        ld = lambda z: ld_fn(z, r, m)
        last = jnp.clip(m.sum().astype(jnp.int32) - 1, 0, Tmax - 1)
        warm = blackjax.window_adaptation(blackjax.nuts, ld, is_mass_matrix_diagonal=True, target_acceptance_rate=target_accept, progress_bar=False, adaptation_info_fn=get_filter_adapt_info_fn())
        rk_w, rk_s = jax.random.split(rng)
        (state, params), _ = warm.run(rk_w, z0, num_steps=num_warmup)
        step = blackjax.nuts(ld, **params).step

        def body(st, k):
            st, info = step(k, st)
            return (st, (reduce_fn(st.position, last, r), info.is_divergent, info.num_integration_steps))
        keys = jax.random.split(rk_s, num_samples)
        _, (red, div, nstep) = jax.lax.scan(body, state, keys)
        return (red, div.sum(), nstep.sum(), (nstep >= DEPTH_HIT_LEAPFROGS).sum(), params['step_size'])
    rng = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(rng, B * n_chains).reshape(B, n_chains, 2)
    z0 = jax.vmap(init_fn)(R, M)
    z0 = jnp.broadcast_to(z0[:, None, :], (B, n_chains, z0.shape[1]))
    Rb = jnp.broadcast_to(R[:, None, :], (B, n_chains, Tmax))
    Mb = jnp.broadcast_to(M[:, None, :], (B, n_chains, Tmax))
    run_vv = jax.vmap(jax.vmap(run_one))
    t0 = time.perf_counter()
    draws, divs, nsteps, dhits, steps = run_vv(keys, Rb, Mb, z0)
    jax.block_until_ready(draws)
    wall = time.perf_counter() - t0
    return IndepResult(draws=np.asarray(draws), lengths=np.asarray(lengths), timing={'wall_total': wall}, extra={'host': 'anon', 'B': int(B), 'Tmax': int(Tmax), 'n_chains': int(n_chains), 'model': model, 'ncol': ncol, 'num_warmup': int(num_warmup), 'num_samples': int(num_samples), 'target_accept': float(target_accept), 'seed': int(seed), 'num_divergences': int(np.asarray(divs).sum()), 'mean_leapfrogs': float(np.asarray(nsteps).mean() / num_samples), 'divergences_per_lane': np.asarray(divs, int).tolist(), 'depth_hits_per_lane': np.asarray(dhits, int).tolist(), 'leapfrogs_per_lane': np.asarray(nsteps, int).tolist(), 'step_size_per_lane': np.asarray(steps, float).tolist()})
N_INNER_FUNC_L19 = 1024
_CRN_SEED_L19 = 20260719
_NU_GRID = np.exp(np.linspace(np.log(2.05), np.log(200.0), 96))

def _crn_l19():
    rng = np.random.default_rng(_CRN_SEED_L19)
    K = N_INNER_FUNC_L19
    from scipy.stats import t as _t
    u_shock = rng.random(K)
    return {'eta': rng.standard_normal(K), 'u_shock': u_shock, 'z_shock': rng.standard_normal(K), 'u_jump': rng.random(K), 'z_jump': rng.standard_normal(K), 't_table': _t.ppf(u_shock[None, :], _NU_GRID[:, None])}
_CRN19 = None

def _crn():
    global _CRN19
    if _CRN19 is None:
        _CRN19 = _crn_l19()
    return _CRN19

def _cols(model):
    return {c: i for i, c in enumerate(REDUCE_COLS_L19[model])}

def _t_shocks(nu, crn):
    lg = np.log(np.clip(nu, _NU_GRID[0], _NU_GRID[-1]))
    grid = np.log(_NU_GRID)
    idx = np.clip(np.searchsorted(grid, lg) - 1, 0, len(grid) - 2)
    w = (lg - grid[idx]) / (grid[idx + 1] - grid[idx])
    tt = crn['t_table']
    raw = (1.0 - w)[:, None] * tt[idx] + w[:, None] * tt[idx + 1]
    return raw * np.sqrt((nu - 2.0) / nu)[:, None]

def per_draw_functionals_l19(reduced, model):
    crn = _crn()
    c = _cols(model)
    mu = reduced[:, c['mu']]
    phi = reduced[:, c['phi']]
    se = reduced[:, c['sigma_eta']]
    hT = reduced[:, c['h_T']]
    M = mu.shape[0]
    h_next = (mu + phi * (hT - mu))[:, None] + se[:, None] * crn['eta'][None, :]
    if 'nu' in c:
        shock = _t_shocks(reduced[:, c['nu']], crn)
    else:
        shock = np.broadcast_to(crn['z_shock'][None, :], (M, N_INNER_FUNC_L19))
    if 'p_J' in c:
        pJ = reduced[:, c['p_J']]
        sJ = reduced[:, c['sigma_J']]
        jump = crn['u_jump'][None, :] < pJ[:, None]
        if 'nu' in c:
            sd = np.where(jump, np.sqrt(np.exp(h_next) + (sJ ** 2)[:, None]), np.exp(h_next / 2.0))
            r = sd * shock
        else:
            r = np.exp(h_next / 2.0) * shock + jump * (sJ[:, None] * crn['z_jump'][None, :])
    else:
        r = np.exp(h_next / 2.0) * shock
    out = {'VaR95': -np.percentile(r, 5.0, axis=1), 'VaR99': -np.percentile(r, 1.0, axis=1), 'sigma_eta': se, 'vol_T': np.exp(hT / 2.0), 'mu': mu, 'phi': phi}
    if 'p_J' in c:
        out['p_J'] = reduced[:, c['p_J']]
        out['sigma_J'] = reduced[:, c['sigma_J']]
    if 'nu' in c:
        out['nu'] = reduced[:, c['nu']]
    return out

def predictive_samples_l19(reduced, model, n_inner, rng):
    c = _cols(model)
    mu = reduced[:, c['mu']]
    phi = reduced[:, c['phi']]
    se = reduced[:, c['sigma_eta']]
    hT = reduced[:, c['h_T']]
    M = mu.shape[0]
    eta = rng.standard_normal((M, n_inner))
    h_next = (mu + phi * (hT - mu))[:, None] + se[:, None] * eta
    if 'nu' in c:
        nu = np.clip(reduced[:, c['nu']], 2.05, None)[:, None]
        shock = rng.standard_t(df=np.broadcast_to(nu, (M, n_inner)))
        shock = shock * np.sqrt((nu - 2.0) / nu)
    else:
        shock = rng.standard_normal((M, n_inner))
    if 'p_J' in c:
        pJ = reduced[:, c['p_J']][:, None]
        sJ = reduced[:, c['sigma_J']][:, None]
        jump = rng.random((M, n_inner)) < pJ
        if 'nu' in c:
            sd = np.where(jump, np.sqrt(np.exp(h_next) + sJ ** 2), np.exp(h_next / 2.0))
            r = sd * shock
        else:
            r = np.exp(h_next / 2.0) * shock + jump * (sJ * rng.standard_normal((M, n_inner)))
    else:
        r = np.exp(h_next / 2.0) * shock
    return r.reshape(-1)

def svj_model_masked_hn(returns, mask):
    import numpyro
    import numpyro.distributions as dist
    T = returns.shape[0]
    mu = numpyro.sample('mu', dist.Normal(MU_LOC, MU_SCALE))
    phi_raw = numpyro.sample('phi_raw', dist.Beta(PHI_A, PHI_B))
    phi = numpyro.deterministic('phi', 2.0 * phi_raw - 1.0)
    sigma_eta = numpyro.sample('sigma_eta', dist.HalfNormal(SIGMA_ETA_HN_SCALE))
    h_std = numpyro.sample('h_std', dist.Normal(0.0, 1.0).expand([T]).to_event(1))
    h = numpyro.deterministic('h', _ar1_reconstruct(h_std, mu, phi, sigma_eta))
    p_J = numpyro.sample('p_J', dist.Beta(PJ_A, PJ_B))
    sigma_J = numpyro.sample('sigma_J', dist.HalfNormal(SIGMA_J_SCALE))
    logvar_jp = jnp.logaddexp(h, 2.0 * jnp.log(sigma_J))
    lp_no = jnp.log1p(-p_J) + _gauss_logprob_logvar(returns, h)
    lp_jp = jnp.log(p_J) + _gauss_logprob_logvar(returns, logvar_jp)
    log_lik = logsumexp(jnp.stack([lp_no, lp_jp], axis=0), axis=0)
    import numpyro as _n
    _n.factor('obs', (log_lik * mask).sum())
