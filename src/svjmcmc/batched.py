from __future__ import annotations
import time
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS
from .model import MU_LOC, MU_SCALE, PHI_A, PHI_B, SIGMA_ETA_SCALE, PJ_A, PJ_B, SIGMA_J_SCALE, _LOG2PI, _ar1_reconstruct, _gauss_logprob_logvar

def pad_and_mask(returns_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.array([len(r) for r in returns_list], dtype=np.int64)
    Tmax = int(lengths.max())
    B = len(returns_list)
    R = np.zeros((B, Tmax), dtype=np.float64)
    M = np.zeros((B, Tmax), dtype=np.float64)
    for i, r in enumerate(returns_list):
        R[i, :len(r)] = r
        M[i, :len(r)] = 1.0
    return (R, M, lengths)

def batched_svj_model(returns, mask):
    B, Tmax = returns.shape
    mu = numpyro.sample('mu', dist.Normal(MU_LOC, MU_SCALE).expand([B]).to_event(1))
    phi_raw = numpyro.sample('phi_raw', dist.Beta(PHI_A, PHI_B).expand([B]).to_event(1))
    phi = numpyro.deterministic('phi', 2.0 * phi_raw - 1.0)
    sigma_eta = numpyro.sample('sigma_eta', dist.HalfCauchy(SIGMA_ETA_SCALE).expand([B]).to_event(1))
    h_std = numpyro.sample('h_std', dist.Normal(0.0, 1.0).expand([B, Tmax]).to_event(2))
    p_J = numpyro.sample('p_J', dist.Beta(PJ_A, PJ_B).expand([B]).to_event(1))
    sigma_J = numpyro.sample('sigma_J', dist.HalfNormal(SIGMA_J_SCALE).expand([B]).to_event(1))
    h = jax.vmap(_ar1_reconstruct)(h_std, mu, phi, sigma_eta)
    last_idx = jnp.clip(mask.sum(axis=1).astype(jnp.int32) - 1, 0, Tmax - 1)
    numpyro.deterministic('h_T', h[jnp.arange(B), last_idx])
    logvar_jp = jnp.logaddexp(h, 2.0 * jnp.log(sigma_J)[:, None])
    lp_no = jnp.log1p(-p_J)[:, None] + _gauss_logprob_logvar(returns, h)
    lp_jp = jnp.log(p_J)[:, None] + _gauss_logprob_logvar(returns, logvar_jp)
    log_lik = logsumexp(jnp.stack([lp_no, lp_jp], axis=0), axis=0)
    numpyro.factor('obs', (log_lik * mask).sum())

@dataclass
class BatchedResult:
    samples: dict
    lengths: np.ndarray
    timing: dict
    extra: dict = field(default_factory=dict)

def run_batched_nuts(returns, mask, lengths, *, seed=0, num_warmup=1000, num_samples=1000, num_chains=4, target_accept=0.9, chain_method='vectorized') -> BatchedResult:
    R = jnp.asarray(returns)
    M = jnp.asarray(mask)
    kernel = NUTS(batched_svj_model, target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method=chain_method, progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(int(seed)), R, M, extra_fields=('diverging',))
    jax.block_until_ready(mcmc.get_samples(group_by_chain=False)['mu'])
    wall = time.perf_counter() - t0
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    try:
        n_div = int(np.asarray(mcmc.get_extra_fields()['diverging']).sum())
    except Exception:
        n_div = -1
    return BatchedResult(samples=samples, lengths=np.asarray(lengths), timing={'wall_total': wall}, extra={'host': 'anon', 'B': int(returns.shape[0]), 'Tmax': int(returns.shape[1]), 'num_divergences': n_div})

def per_ticker_chains(res: BatchedResult, b: int) -> dict:
    s = res.samples
    h_T = s['h_T'][..., b]
    out = {'mu': s['mu'][..., b], 'phi': s['phi'][..., b], 'sigma_eta': s['sigma_eta'][..., b], 'p_J': s['p_J'][..., b], 'sigma_J': s['sigma_J'][..., b], 'h': h_T[..., None]}
    return out
