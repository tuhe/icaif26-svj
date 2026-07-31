from __future__ import annotations
import time
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import beta as jbeta, cauchy as jcauchy, norm as jnorm
import blackjax
from blackjax.adaptation.base import get_filter_adapt_info_fn
from .model import MU_LOC, MU_SCALE, PHI_A, PHI_B, SIGMA_ETA_SCALE, SIGMA_J_SCALE, _ar1_reconstruct, _gauss_logprob_logvar
_LOG2 = float(np.log(2.0))
MUL_LOC, MUL_SCALE = (-3.0, 1.5)
RHOL_A, RHOL_B = (4.0, 2.0)
ALPHA_SCALE = 0.5
MARK_SCALE = 25.0
MARK_CLIP = 4.0

def _split_z(z):
    return (z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7:])

def _intensity_logodds(mark, mu_lambda, rho_lambda, alpha):
    T = mark.shape[0]
    add = jnp.concatenate([jnp.zeros((1,)), alpha * mark[:-1]])
    a = jnp.concatenate([jnp.zeros((1,)), jnp.full((T - 1,), rho_lambda)])

    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return (a_l * a_r, a_r * b_l + b_r)
    _, g = jax.lax.associative_scan(combine, (a, add))
    return mu_lambda + g

def _surprise_mark(returns, mask):
    return jnp.clip(returns ** 2 / MARK_SCALE, 0.0, MARK_CLIP) * mask

def unc_logdensity_se(z, returns, mask):
    z_mu, z_phi, z_se, z_sj, z_mul, z_rhol, z_alpha, h_std = _split_z(z)
    mu = z_mu
    phi_raw = jax.nn.sigmoid(z_phi)
    phi = 2.0 * phi_raw - 1.0
    sigma_eta = jnp.exp(z_se)
    sigma_J = jnp.exp(z_sj)
    mu_lambda = z_mul
    rho_lambda = jax.nn.sigmoid(z_rhol)
    alpha = jnp.exp(z_alpha)
    lp = jnorm.logpdf(mu, MU_LOC, MU_SCALE)
    lp += jbeta.logpdf(phi_raw, PHI_A, PHI_B) + jnp.log(phi_raw) + jnp.log1p(-phi_raw)
    lp += jcauchy.logpdf(sigma_eta, 0.0, SIGMA_ETA_SCALE) + _LOG2 + z_se
    lp += jnorm.logpdf(sigma_J, 0.0, SIGMA_J_SCALE) + _LOG2 + z_sj
    lp += jnorm.logpdf(mu_lambda, MUL_LOC, MUL_SCALE)
    lp += jbeta.logpdf(rho_lambda, RHOL_A, RHOL_B) + jnp.log(rho_lambda) + jnp.log1p(-rho_lambda)
    lp += jnorm.logpdf(alpha, 0.0, ALPHA_SCALE) + _LOG2 + z_alpha
    lp += jnorm.logpdf(h_std, 0.0, 1.0).sum()
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    mark = _surprise_mark(returns, mask)
    logodds = _intensity_logodds(mark, mu_lambda, rho_lambda, alpha)
    log_pJ = jax.nn.log_sigmoid(logodds)
    log_1mpJ = jax.nn.log_sigmoid(-logodds)
    logvar_jp = jnp.logaddexp(h, 2.0 * jnp.log(sigma_J))
    ll = logsumexp(jnp.stack([log_1mpJ + _gauss_logprob_logvar(returns, h), log_pJ + _gauss_logprob_logvar(returns, logvar_jp)]), axis=0)
    lp += (ll * mask).sum()
    return lp

def init_z_se(returns, mask):
    T = returns.shape[0]
    v = jnp.maximum((returns ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0), 0.001)
    z = jnp.zeros(T + 7)
    z = z.at[0].set(jnp.log(v))
    z = z.at[1].set(2.5)
    z = z.at[2].set(jnp.log(0.2))
    z = z.at[3].set(jnp.log(3.0))
    z = z.at[4].set(MUL_LOC)
    z = z.at[5].set(0.0)
    z = z.at[6].set(jnp.log(0.1))
    return z

def _reduce_one_se(z, last):
    z_mu, z_phi, z_se, z_sj, z_mul, z_rhol, z_alpha, h_std = _split_z(z)
    mu = z_mu
    phi = 2.0 * jax.nn.sigmoid(z_phi) - 1.0
    sigma_eta = jnp.exp(z_se)
    h = _ar1_reconstruct(h_std, mu, phi, sigma_eta)
    return jnp.array([mu, phi, sigma_eta, jnp.exp(z_sj), z_mul, jax.nn.sigmoid(z_rhol), jnp.exp(z_alpha), h[last]])
REDUCE_COLS = ['mu', 'phi', 'sigma_eta', 'sigma_J', 'mu_lambda', 'rho_lambda', 'alpha', 'h_T']

@dataclass
class IndepResultSE:
    draws: np.ndarray
    lengths: np.ndarray
    timing: dict
    extra: dict = field(default_factory=dict)

def run_independent_se(returns, mask, lengths, *, seed=0, num_warmup=1000, num_samples=1000, n_chains=2, target_accept=0.8):
    R = jnp.asarray(returns)
    M = jnp.asarray(mask)
    B, Tmax = R.shape

    def run_one(rng, r, m, z0):
        ld = lambda z: unc_logdensity_se(z, r, m)
        last = jnp.clip(m.sum().astype(jnp.int32) - 1, 0, Tmax - 1)
        warm = blackjax.window_adaptation(blackjax.nuts, ld, is_mass_matrix_diagonal=True, target_acceptance_rate=target_accept, progress_bar=False, adaptation_info_fn=get_filter_adapt_info_fn())
        rk_w, rk_s = jax.random.split(rng)
        (state, params), _ = warm.run(rk_w, z0, num_steps=num_warmup)
        step = blackjax.nuts(ld, **params).step

        def body(st, k):
            st, info = step(k, st)
            return (st, (_reduce_one_se(st.position, last), info.is_divergent, info.num_integration_steps))
        keys = jax.random.split(rk_s, num_samples)
        _, (red, div, nstep) = jax.lax.scan(body, state, keys)
        return (red, div.sum(), nstep.sum())
    rng = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(rng, B * n_chains).reshape(B, n_chains, 2)
    z0 = jax.vmap(init_z_se)(R, M)
    z0 = jnp.broadcast_to(z0[:, None, :], (B, n_chains, z0.shape[1]))
    Rb = jnp.broadcast_to(R[:, None, :], (B, n_chains, Tmax))
    Mb = jnp.broadcast_to(M[:, None, :], (B, n_chains, Tmax))
    run_vv = jax.vmap(jax.vmap(run_one))
    t0 = time.perf_counter()
    draws, divs, nsteps = run_vv(keys, Rb, Mb, z0)
    jax.block_until_ready(draws)
    wall = time.perf_counter() - t0
    return IndepResultSE(draws=np.asarray(draws), lengths=np.asarray(lengths), timing={'wall_total': wall}, extra={'host': 'anon', 'B': int(B), 'Tmax': int(Tmax), 'n_chains': int(n_chains), 'num_divergences': int(np.asarray(divs).sum()), 'mean_leapfrogs': float(np.asarray(nsteps).mean() / num_samples)})
