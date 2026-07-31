from __future__ import annotations
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, logsumexp
from jax.scipy.stats import beta as jbeta, gamma as jgamma, norm as jnorm, cauchy as jcauchy
import blackjax
try:
    from blackjax.adaptation.base import get_filter_adapt_info_fn
except ImportError:
    get_filter_adapt_info_fn = None
from .model import MU_LOC, MU_SCALE, PHI_A, PHI_B, SIGMA_ETA_SCALE, PJ_A, PJ_B, SIGMA_J_SCALE, _ar1_reconstruct, _gauss_logprob_logvar
from .batched_indep import IndepResult, _LOG2
SIGMA_ETA_HN_SCALE = 0.5
NU_GAMMA_A, NU_GAMMA_RATE = (2.0, 0.15)
_LOGPI = float(np.log(np.pi))

def _studentt_logpdf(x, nu, scale):
    z = x / scale
    return gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * (jnp.log(nu) + _LOGPI) - jnp.log(scale) - 0.5 * (nu + 1.0) * jnp.log1p(z * z / nu)

def unc_logdensity_svjt(z, returns, mask):
    z_mu, z_phi, z_se, z_pj, z_sj, z_nu = (z[0], z[1], z[2], z[3], z[4], z[5])
    h_std = z[6:]
    mu = z_mu
    phi_raw = jax.nn.sigmoid(z_phi)
    phi = 2.0 * phi_raw - 1.0
    sigma_eta = jnp.exp(z_se)
    p_J = jax.nn.sigmoid(z_pj)
    sigma_J = jnp.exp(z_sj)
    nu = 2.0 + jnp.exp(z_nu)
    lp = jnorm.logpdf(mu, MU_LOC, MU_SCALE)
    lp += jbeta.logpdf(phi_raw, PHI_A, PHI_B) + jnp.log(phi_raw) + jnp.log1p(-phi_raw)
    lp += jcauchy.logpdf(sigma_eta, 0.0, SIGMA_ETA_SCALE) + _LOG2 + z_se
    lp += jbeta.logpdf(p_J, PJ_A, PJ_B) + jnp.log(p_J) + jnp.log1p(-p_J)
    lp += jnorm.logpdf(sigma_J, 0.0, SIGMA_J_SCALE) + _LOG2 + z_sj
    lp += jgamma.logpdf(nu - 2.0, NU_GAMMA_A, scale=1.0 / NU_GAMMA_RATE) + z_nu
    lp += jnorm.logpdf(h_std, 0.0, 1.0).sum()
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    fac = jnp.sqrt((nu - 2.0) / nu)
    scale_no = jnp.exp(0.5 * h) * fac
    scale_jp = jnp.sqrt(jnp.exp(h) + sigma_J ** 2) * fac
    ll = logsumexp(jnp.stack([jnp.log1p(-p_J) + _studentt_logpdf(returns, nu, scale_no), jnp.log(p_J) + _studentt_logpdf(returns, nu, scale_jp)]), axis=0)
    lp += (ll * mask).sum()
    return lp

def init_z_svjt(returns, mask):
    T = returns.shape[0]
    v = jnp.maximum((returns ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0), 0.001)
    z = jnp.zeros(T + 6)
    z = z.at[0].set(jnp.log(v))
    z = z.at[1].set(2.5)
    z = z.at[2].set(jnp.log(0.2))
    z = z.at[3].set(-2.9)
    z = z.at[4].set(jnp.log(3.0))
    z = z.at[5].set(jnp.log(8.0))
    return z

def _reduce_svjt(z, last, r):
    z_mu, z_phi, z_se, z_pj, z_sj, z_nu = (z[0], z[1], z[2], z[3], z[4], z[5])
    h_std = z[6:]
    mu = z_mu
    phi = 2.0 * jax.nn.sigmoid(z_phi) - 1.0
    sigma_eta = jnp.exp(z_se)
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    return jnp.array([mu, phi, sigma_eta, jax.nn.sigmoid(z_pj), jnp.exp(z_sj), 2.0 + jnp.exp(z_nu), h[last]])

def unc_logdensity_svj_hn(z, returns, mask):
    z_mu, z_phi, z_se, z_pj, z_sj = (z[0], z[1], z[2], z[3], z[4])
    h_std = z[5:]
    mu = z_mu
    phi_raw = jax.nn.sigmoid(z_phi)
    phi = 2.0 * phi_raw - 1.0
    sigma_eta = jnp.exp(z_se)
    p_J = jax.nn.sigmoid(z_pj)
    sigma_J = jnp.exp(z_sj)
    lp = jnorm.logpdf(mu, MU_LOC, MU_SCALE)
    lp += jbeta.logpdf(phi_raw, PHI_A, PHI_B) + jnp.log(phi_raw) + jnp.log1p(-phi_raw)
    lp += jnorm.logpdf(sigma_eta, 0.0, SIGMA_ETA_HN_SCALE) + _LOG2 + z_se
    lp += jbeta.logpdf(p_J, PJ_A, PJ_B) + jnp.log(p_J) + jnp.log1p(-p_J)
    lp += jnorm.logpdf(sigma_J, 0.0, SIGMA_J_SCALE) + _LOG2 + z_sj
    lp += jnorm.logpdf(h_std, 0.0, 1.0).sum()
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    logvar_jp = jnp.logaddexp(h, 2.0 * jnp.log(sigma_J))
    ll = logsumexp(jnp.stack([jnp.log1p(-p_J) + _gauss_logprob_logvar(returns, h), jnp.log(p_J) + _gauss_logprob_logvar(returns, logvar_jp)]), axis=0)
    lp += (ll * mask).sum()
    return lp
FEEDBACK_C = 0.1
GAMMA_SCALE = 0.5

def _realized_logvar(returns):
    return jnp.log(returns ** 2 + FEEDBACK_C)

def _ar1_reconstruct_fb(h_std, mu, phi, sigma_eta, gamma, s):
    T = h_std.shape[0]
    b = sigma_eta * h_std
    b = b + jnp.concatenate([jnp.zeros((1,)), gamma * s[:-1]])
    b = b.at[0].set(sigma_eta / jnp.sqrt(1.0 - phi ** 2) * h_std[0])
    a = jnp.concatenate([jnp.zeros((1,)), jnp.full((T - 1,), phi)])

    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return (a_l * a_r, a_r * b_l + b_r)
    _, g = jax.lax.associative_scan(combine, (a, b))
    return mu + g

def unc_logdensity_svjgarch(z, returns, mask):
    z_mu, z_phi, z_se, z_pj, z_sj, gamma = (z[0], z[1], z[2], z[3], z[4], z[5])
    h_std = z[6:]
    mu = z_mu
    phi_raw = jax.nn.sigmoid(z_phi)
    phi = 2.0 * phi_raw - 1.0
    sigma_eta = jnp.exp(z_se)
    p_J = jax.nn.sigmoid(z_pj)
    sigma_J = jnp.exp(z_sj)
    lp = jnorm.logpdf(mu, MU_LOC, MU_SCALE)
    lp += jbeta.logpdf(phi_raw, PHI_A, PHI_B) + jnp.log(phi_raw) + jnp.log1p(-phi_raw)
    lp += jcauchy.logpdf(sigma_eta, 0.0, SIGMA_ETA_SCALE) + _LOG2 + z_se
    lp += jbeta.logpdf(p_J, PJ_A, PJ_B) + jnp.log(p_J) + jnp.log1p(-p_J)
    lp += jnorm.logpdf(sigma_J, 0.0, SIGMA_J_SCALE) + _LOG2 + z_sj
    lp += jnorm.logpdf(gamma, 0.0, GAMMA_SCALE)
    lp += jnorm.logpdf(h_std, 0.0, 1.0).sum()
    h = _ar1_reconstruct_fb(h_std, mu, phi, sigma_eta, gamma, _realized_logvar(returns))
    logvar_jp = jnp.logaddexp(h, 2.0 * jnp.log(sigma_J))
    ll = logsumexp(jnp.stack([jnp.log1p(-p_J) + _gauss_logprob_logvar(returns, h), jnp.log(p_J) + _gauss_logprob_logvar(returns, logvar_jp)]), axis=0)
    lp += (ll * mask).sum()
    return lp

def init_z_svjgarch(returns, mask):
    T = returns.shape[0]
    v = jnp.maximum((returns ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0), 0.001)
    z = jnp.zeros(T + 6)
    z = z.at[0].set(jnp.log(v))
    z = z.at[1].set(2.5)
    z = z.at[2].set(jnp.log(0.2))
    z = z.at[3].set(-2.9)
    z = z.at[4].set(jnp.log(3.0))
    z = z.at[5].set(0.0)
    return z

def _reduce_svjgarch(z, last, r):
    z_mu, z_phi, z_se, z_pj, z_sj, gamma = (z[0], z[1], z[2], z[3], z[4], z[5])
    h_std = z[6:]
    mu = z_mu
    phi = 2.0 * jax.nn.sigmoid(z_phi) - 1.0
    sigma_eta = jnp.exp(z_se)
    h = _ar1_reconstruct_fb(h_std, mu, phi, sigma_eta, gamma, _realized_logvar(r))
    return jnp.array([mu, phi, sigma_eta, jax.nn.sigmoid(z_pj), jnp.exp(z_sj), gamma, h[last]])
SJ1_SCALE, SJ2_INC_SCALE = (2.0, 5.0)
WLOGIT_LOC, WLOGIT_SCALE = (-2.0, 1.0)

def unc_logdensity_svj2j(z, returns, mask):
    z_mu, z_phi, z_se, z_w1, z_w2, z_sj1, z_sj2 = (z[0], z[1], z[2], z[3], z[4], z[5], z[6])
    h_std = z[7:]
    mu = z_mu
    phi_raw = jax.nn.sigmoid(z_phi)
    phi = 2.0 * phi_raw - 1.0
    sigma_eta = jnp.exp(z_se)
    logits = jnp.stack([jnp.zeros(()), z_w1, z_w2])
    logw = logits - logsumexp(logits)
    sj1 = jnp.exp(z_sj1)
    sj2 = sj1 + jnp.exp(z_sj2)
    lp = jnorm.logpdf(mu, MU_LOC, MU_SCALE)
    lp += jbeta.logpdf(phi_raw, PHI_A, PHI_B) + jnp.log(phi_raw) + jnp.log1p(-phi_raw)
    lp += jcauchy.logpdf(sigma_eta, 0.0, SIGMA_ETA_SCALE) + _LOG2 + z_se
    lp += jnorm.logpdf(z_w1, WLOGIT_LOC, WLOGIT_SCALE) + jnorm.logpdf(z_w2, WLOGIT_LOC, WLOGIT_SCALE)
    lp += jnorm.logpdf(sj1, 0.0, SJ1_SCALE) + _LOG2 + z_sj1
    lp += jnorm.logpdf(jnp.exp(z_sj2), 0.0, SJ2_INC_SCALE) + _LOG2 + z_sj2
    lp += jnorm.logpdf(h_std, 0.0, 1.0).sum()
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    lv1 = jnp.logaddexp(h, 2.0 * jnp.log(sj1))
    lv2 = jnp.logaddexp(h, 2.0 * jnp.log(sj2))
    ll = logsumexp(jnp.stack([logw[0] + _gauss_logprob_logvar(returns, h), logw[1] + _gauss_logprob_logvar(returns, lv1), logw[2] + _gauss_logprob_logvar(returns, lv2)]), axis=0)
    lp += (ll * mask).sum()
    return lp

def init_z_svj2j(returns, mask):
    T = returns.shape[0]
    v = jnp.maximum((returns ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0), 0.001)
    z = jnp.zeros(T + 7)
    z = z.at[0].set(jnp.log(v))
    z = z.at[1].set(2.5)
    z = z.at[2].set(jnp.log(0.2))
    z = z.at[3].set(-2.0)
    z = z.at[4].set(-2.5)
    z = z.at[5].set(jnp.log(1.5))
    z = z.at[6].set(jnp.log(3.0))
    return z

def _reduce_svj2j(z, last, r):
    z_mu, z_phi, z_se, z_w1, z_w2, z_sj1, z_sj2 = (z[0], z[1], z[2], z[3], z[4], z[5], z[6])
    h_std = z[7:]
    mu = z_mu
    phi = 2.0 * jax.nn.sigmoid(z_phi) - 1.0
    sigma_eta = jnp.exp(z_se)
    logits = jnp.stack([jnp.zeros(()), z_w1, z_w2])
    w = jax.nn.softmax(logits)
    sj1 = jnp.exp(z_sj1)
    sj2 = sj1 + jnp.exp(z_sj2)
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    return jnp.array([mu, phi, sigma_eta, w[1], w[2], sj1, sj2, h[last]])
CFG_PRIORS = {'pj_a': PJ_A, 'pj_b': PJ_B, 'sigma_J_scale': SIGMA_J_SCALE, 'phi_a': PHI_A, 'phi_b': PHI_B, 'sigma_eta_scale': SIGMA_ETA_SCALE}

def unc_logdensity_svj_cfg(z, returns, mask):
    z_mu, z_phi, z_se, z_pj, z_sj = (z[0], z[1], z[2], z[3], z[4])
    h_std = z[5:]
    pj_a = CFG_PRIORS['pj_a']
    pj_b = CFG_PRIORS['pj_b']
    sJ_sc = CFG_PRIORS['sigma_J_scale']
    phi_a = CFG_PRIORS['phi_a']
    phi_b = CFG_PRIORS['phi_b']
    se_sc = CFG_PRIORS['sigma_eta_scale']
    mu = z_mu
    phi_raw = jax.nn.sigmoid(z_phi)
    phi = 2.0 * phi_raw - 1.0
    sigma_eta = jnp.exp(z_se)
    p_J = jax.nn.sigmoid(z_pj)
    sigma_J = jnp.exp(z_sj)
    lp = jnorm.logpdf(mu, MU_LOC, MU_SCALE)
    lp += jbeta.logpdf(phi_raw, phi_a, phi_b) + jnp.log(phi_raw) + jnp.log1p(-phi_raw)
    lp += jcauchy.logpdf(sigma_eta, 0.0, se_sc) + _LOG2 + z_se
    if CFG_PRIORS.get('pj_logit_normal'):
        lp += jnorm.logpdf(z_pj, CFG_PRIORS['pj_eta'], CFG_PRIORS['pj_tau'])
    else:
        lp += jbeta.logpdf(p_J, pj_a, pj_b) + jnp.log(p_J) + jnp.log1p(-p_J)
    lp += jnorm.logpdf(sigma_J, 0.0, sJ_sc) + _LOG2 + z_sj
    lp += jnorm.logpdf(h_std, 0.0, 1.0).sum()
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    logvar_jp = jnp.logaddexp(h, 2.0 * jnp.log(sigma_J))
    ll = logsumexp(jnp.stack([jnp.log1p(-p_J) + _gauss_logprob_logvar(returns, h), jnp.log(p_J) + _gauss_logprob_logvar(returns, logvar_jp)]), axis=0)
    lp += (ll * mask).sum()
    return lp
_VARIANTS = {'svjt': (unc_logdensity_svjt, init_z_svjt, _reduce_svjt, 7), 'svj_hn': (unc_logdensity_svj_hn, None, None, 6), 'svj_garch': (unc_logdensity_svjgarch, init_z_svjgarch, _reduce_svjgarch, 7), 'svj_2j': (unc_logdensity_svj2j, init_z_svj2j, _reduce_svj2j, 8), 'svj_cfg': (unc_logdensity_svj_cfg, None, None, 6)}

def run_independent_g(returns, mask, lengths, *, model='svjt', seed=0, num_warmup=1000, num_samples=1000, n_chains=2, target_accept=0.8):
    from .batched_indep import init_z as _base_init, _reduce_one as _base_reduce
    ld_fn, init_fn, reduce_fn, ncol = _VARIANTS[model]
    if init_fn is None:
        init_fn = _base_init
        reduce_fn = lambda z, last, r: _base_reduce(z, last)
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
        return (red, div.sum(), nstep.sum(), (nstep >= 1023).sum())
    rng = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(rng, B * n_chains).reshape(B, n_chains, 2)
    z0 = jax.vmap(init_fn)(R, M)
    z0 = jnp.broadcast_to(z0[:, None, :], (B, n_chains, z0.shape[1]))
    Rb = jnp.broadcast_to(R[:, None, :], (B, n_chains, Tmax))
    Mb = jnp.broadcast_to(M[:, None, :], (B, n_chains, Tmax))
    run_vv = jax.vmap(jax.vmap(run_one))
    t0 = time.perf_counter()
    draws, divs, nsteps, dhits = run_vv(keys, Rb, Mb, z0)
    jax.block_until_ready(draws)
    wall = time.perf_counter() - t0
    return IndepResult(draws=np.asarray(draws), lengths=np.asarray(lengths), timing={'wall_total': wall}, extra={'host': 'anon', 'B': int(B), 'Tmax': int(Tmax), 'n_chains': int(n_chains), 'model': model, 'ncol': ncol, 'num_divergences': int(np.asarray(divs).sum()), 'mean_leapfrogs': float(np.asarray(nsteps).mean() / num_samples), 'divergences_per_lane': np.asarray(divs, int).tolist(), 'depth_hits_per_lane': np.asarray(dhits, int).tolist(), 'leapfrogs_per_lane': (np.asarray(nsteps, float) / num_samples).tolist()})
