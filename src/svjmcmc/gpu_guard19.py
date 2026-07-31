from __future__ import annotations

def require_gpu():
    import jax
    plats = {d.platform for d in jax.devices()}
    if 'gpu' not in plats and 'cuda' not in plats:
        raise SystemExit(f"GPU_GUARD_ABORT: no GPU device visible to JAX (devices={jax.devices()}). Refusing to run on CPU — this node's CUDA state is likely broken (cuInit failure). Free the slot and resubmit elsewhere.")
    print(f'GPU_GUARD_OK: {jax.devices()}', flush=True)
