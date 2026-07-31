from __future__ import annotations
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
MU_LOC, MU_SCALE = (0.0, 10.0)
PHI_A, PHI_B = (20.0, 1.5)
SIGMA_ETA_SCALE = 0.5
SIGMA_ETA_HN_SCALE = 0.5
PJ_A, PJ_B = (2.0, 38.0)
SIGMA_J_SCALE = 5.0
import os
_SCAN_MODE = os.environ.get('SVJMCMC_SCAN', 'assoc')

def _ar1_reconstruct_assoc(h_std, mu, phi, sigma_eta):
    T = h_std.shape[0]
    b = sigma_eta * h_std
    b = b.at[0].set(sigma_eta / jnp.sqrt(1.0 - phi ** 2) * h_std[0])
    a = jnp.concatenate([jnp.zeros((1,)), jnp.full((T - 1,), phi)])

    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return (a_l * a_r, a_r * b_l + b_r)
    _, g = jax.lax.associative_scan(combine, (a, b))
    return mu + g

def _ar1_reconstruct_seq(h_std, mu, phi, sigma_eta):
    h0 = mu + sigma_eta / jnp.sqrt(1.0 - phi ** 2) * h_std[0]

    def step(h_prev, eps):
        h = mu + phi * (h_prev - mu) + sigma_eta * eps
        return (h, h)
    _, h_rest = jax.lax.scan(step, h0, h_std[1:])
    return jnp.concatenate([h0[None], h_rest])

def _ar1_reconstruct(h_std, mu, phi, sigma_eta):
    if _SCAN_MODE == 'seq':
        return _ar1_reconstruct_seq(h_std, mu, phi, sigma_eta)
    return _ar1_reconstruct_assoc(h_std, mu, phi, sigma_eta)

def _sample_dynamics(T, sigma_eta_prior='cauchy'):
    mu = numpyro.sample('mu', dist.Normal(MU_LOC, MU_SCALE))
    phi_raw = numpyro.sample('phi_raw', dist.Beta(PHI_A, PHI_B))
    phi = numpyro.deterministic('phi', 2.0 * phi_raw - 1.0)
    if sigma_eta_prior == 'halfnormal':
        sigma_eta = numpyro.sample('sigma_eta', dist.HalfNormal(SIGMA_ETA_HN_SCALE))
    else:
        sigma_eta = numpyro.sample('sigma_eta', dist.HalfCauchy(SIGMA_ETA_SCALE))
    h_std = numpyro.sample('h_std', dist.Normal(0.0, 1.0).expand([T]).to_event(1))
    h = numpyro.deterministic('h', _ar1_reconstruct(h_std, mu, phi, sigma_eta))
    return h
_LOG2PI = 1.8378770664093453

def _gauss_logprob_logvar(x, logvar):
    inv_var = jnp.exp(-jnp.clip(logvar, -50.0, 50.0))
    return -0.5 * (x ** 2 * inv_var + logvar + _LOG2PI)

def sv_model(returns):
    T = returns.shape[0]
    h = _sample_dynamics(T)
    numpyro.factor('obs', _gauss_logprob_logvar(returns, h).sum())

def svj_model(returns):
    T = returns.shape[0]
    h = _sample_dynamics(T)
    p_J = numpyro.sample('p_J', dist.Beta(PJ_A, PJ_B))
    sigma_J = numpyro.sample('sigma_J', dist.HalfNormal(SIGMA_J_SCALE))
    logvar_jp = jnp.logaddexp(h, 2.0 * jnp.log(sigma_J))
    lp_no = jnp.log1p(-p_J) + _gauss_logprob_logvar(returns, h)
    lp_jp = jnp.log(p_J) + _gauss_logprob_logvar(returns, logvar_jp)
    log_lik = logsumexp(jnp.stack([lp_no, lp_jp], axis=0), axis=0)
    numpyro.factor('obs', log_lik.sum())

def svj_model_masked(returns, mask):
    T = returns.shape[0]
    h = _sample_dynamics(T)
    p_J = numpyro.sample('p_J', dist.Beta(PJ_A, PJ_B))
    sigma_J = numpyro.sample('sigma_J', dist.HalfNormal(SIGMA_J_SCALE))
    logvar_jp = jnp.logaddexp(h, 2.0 * jnp.log(sigma_J))
    lp_no = jnp.log1p(-p_J) + _gauss_logprob_logvar(returns, h)
    lp_jp = jnp.log(p_J) + _gauss_logprob_logvar(returns, logvar_jp)
    log_lik = logsumexp(jnp.stack([lp_no, lp_jp], axis=0), axis=0)
    numpyro.factor('obs', (log_lik * mask).sum())

def svj_model_hn(returns):
    T = returns.shape[0]
    h = _sample_dynamics(T, sigma_eta_prior='halfnormal')
    p_J = numpyro.sample('p_J', dist.Beta(PJ_A, PJ_B))
    sigma_J = numpyro.sample('sigma_J', dist.HalfNormal(SIGMA_J_SCALE))
    logvar_jp = jnp.logaddexp(h, 2.0 * jnp.log(sigma_J))
    lp_no = jnp.log1p(-p_J) + _gauss_logprob_logvar(returns, h)
    lp_jp = jnp.log(p_J) + _gauss_logprob_logvar(returns, logvar_jp)
    log_lik = logsumexp(jnp.stack([lp_no, lp_jp], axis=0), axis=0)
    numpyro.factor('obs', log_lik.sum())
MODELS = {'sv': sv_model, 'svj': svj_model, 'svj_hn': svj_model_hn}
