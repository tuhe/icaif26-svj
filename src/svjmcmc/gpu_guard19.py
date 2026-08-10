from __future__ import annotations

def require_gpu():
    import os
    import jax
    plats = {d.platform for d in jax.devices()}
    if 'gpu' not in plats and 'cuda' not in plats:
        if os.environ.get('SVJMCMC_REQUIRE_GPU', '0') != '0':
            raise SystemExit(f'GPU_GUARD_ABORT: no GPU device visible to JAX (devices={jax.devices()}). Refusing to run on CPU — set SVJMCMC_REQUIRE_GPU=0 to allow CPU execution at reduced speed.')
        print(f'GPU_GUARD_SKIP: no GPU visible, continuing on {jax.devices()} (SVJMCMC_REQUIRE_GPU=0)', flush=True)
        return
    print(f'GPU_GUARD_OK: {jax.devices()}', flush=True)
