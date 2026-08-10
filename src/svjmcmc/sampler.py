from __future__ import annotations
import time
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median, init_to_value
from numpyro.infer.autoguide import AutoIAFNormal
from numpyro.infer.reparam import NeuTraReparam
import numpyro.optim as optim
from .model import MODELS, _ar1_reconstruct

@dataclass
class SampleResult:
    samples: dict
    timing: dict
    extra: dict = field(default_factory=dict)

def _recover_latents(samples: dict) -> dict:
    out = dict(samples)
    if 'phi' not in out and 'phi_raw' in out:
        out['phi'] = 2.0 * np.asarray(out['phi_raw']) - 1.0
    if 'h' not in out and 'h_std' in out:
        orig_shape = np.asarray(out['mu']).shape
        mu = jnp.asarray(np.asarray(out['mu']).reshape(-1))
        phi = jnp.asarray(np.asarray(out['phi']).reshape(-1))
        se = jnp.asarray(np.asarray(out['sigma_eta']).reshape(-1))
        hstd = jnp.asarray(np.asarray(out['h_std']).reshape(-1, np.asarray(out['h_std']).shape[-1]))
        h = jax.vmap(_ar1_reconstruct)(hstd, mu, phi, se)
        out['h'] = np.asarray(h).reshape(orig_shape + (hstd.shape[-1],))
    return out

def _flatten_chains(samples: dict) -> dict:
    return {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in samples.items()}

def run_nuts(model_name: str, returns, *, seed=0, num_warmup=1000, num_samples=1000, num_chains=4, target_accept=0.9, chain_method='vectorized') -> SampleResult:
    model = MODELS[model_name]
    rng = jax.random.PRNGKey(int(seed))
    kernel = NUTS(model, target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method=chain_method, progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(rng, jnp.asarray(returns))
    jax.block_until_ready(mcmc.get_samples(group_by_chain=False)['mu'])
    wall = time.perf_counter() - t0
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    samples = _recover_latents(samples)
    return SampleResult(samples=samples, timing={'wall_total': wall, 'sample': wall}, extra={'num_divergences': int(np.asarray(mcmc.get_extra_fields().get('diverging', np.array([0])).sum()) if False else 0)})

def run_neutra(model_name: str, returns, *, seed=0, num_warmup=1000, num_samples=1000, num_chains=4, num_flows=2, hidden_dim_factor=1.0, svi_steps=3000, svi_lr=0.001, target_accept=0.9, chain_method='vectorized') -> SampleResult:
    model = MODELS[model_name]
    returns = jnp.asarray(returns)
    T = returns.shape[0]
    latent_dim = T + 5
    hidden = max(latent_dim + 4, int(hidden_dim_factor * (T + 8)))
    rng_svi, rng_mcmc = jax.random.split(jax.random.PRNGKey(int(seed)))
    guide = AutoIAFNormal(model, num_flows=num_flows, hidden_dims=[hidden, hidden])
    svi = SVI(model, guide, optim.Adam(svi_lr), Trace_ELBO())
    t0 = time.perf_counter()
    svi_res = svi.run(rng_svi, svi_steps, returns, progress_bar=False)
    jax.block_until_ready(svi_res.params)
    t_train = time.perf_counter() - t0
    neutra = NeuTraReparam(guide, svi_res.params)
    reparam_model = neutra.reparam(model)
    D = int(guide.latent_dim)
    init_strat = init_to_value(values={'auto_shared_latent': jnp.zeros(D)})
    kernel = NUTS(reparam_model, target_accept_prob=target_accept, init_strategy=init_strat)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method=chain_method, progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(rng_mcmc, returns)
    zs = mcmc.get_samples(group_by_chain=True)['auto_shared_latent']
    jax.block_until_ready(zs)
    t_sample = time.perf_counter() - t0
    C, S, D = zs.shape
    flat = zs.reshape(C * S, D)
    transformed = neutra.transform_sample(flat)
    samples = {k: np.asarray(v).reshape((C, S) + np.asarray(v).shape[1:]) for k, v in transformed.items()}
    samples = _recover_latents(samples)
    final_elbo = float(np.asarray(svi_res.losses[-50:]).mean())
    return SampleResult(samples=samples, timing={'wall_total': t_train + t_sample, 'train': t_train, 'sample': t_sample}, extra={'final_elbo': final_elbo, 'hidden': hidden, 'num_flows': num_flows, 'svi_steps': svi_steps, 'latent_dim': int(D)})
