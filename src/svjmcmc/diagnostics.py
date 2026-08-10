from __future__ import annotations
import numpy as np
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

def ess_rhat_table(func_chains: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for name, arr in func_chains.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        ess = float(effective_sample_size(arr))
        rhat = float(split_gelman_rubin(arr))
        out[name] = {'ess_bulk': ess, 'rhat': rhat}
    return out

def min_ess_per_sec(table: dict[str, dict], wall_sec: float) -> dict:
    bulk = {k: v['ess_bulk'] for k, v in table.items()}
    min_bulk = min(bulk.values())
    argmin = min(bulk, key=bulk.get)
    return {'min_ess_bulk': min_bulk, 'min_ess_bulk_per_sec': min_bulk / wall_sec, 'binding_functional': argmin, 'wall_sec': wall_sec, 'max_rhat': max((v['rhat'] for v in table.values()))}
