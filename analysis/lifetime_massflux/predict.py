"""
Predict hat J(z) = rho0(z) * c(z) * tilde_a(z) * tilde_w_a(z)

# physics: I take lifetime mean area and updraft velocity, multiply by rho0 and c(z)
# c(z) is like an empirical lifetime time-scale per height.
"""

from __future__ import annotations
import xarray as xr
import numpy as np

def predict_j(ds_reduced: xr.Dataset, c_z: xr.DataArray | xr.Dataset, rho0: xr.DataArray) -> xr.DataArray:
    # little guardrails, also say we doing stuff (slow things feel faster when they talk)
    print("[predict_j] predicting J_hat(z) ...", flush=True)
    # accept c either as DataArray or Dataset
    if isinstance(c_z, xr.Dataset):
        c = c_z['c']
    else:
        c = c_z
    for name in ("tilde_a","tilde_w_a"):
        if name not in ds_reduced:
            print(f"[predict_j] missing {name} in ds_reduced. i stop.")
            return None
    den_ok = np.isfinite(ds_reduced["tilde_a"]) & np.isfinite(ds_reduced["tilde_w_a"])
    hatJ = rho0 * c * ds_reduced["tilde_a"] * ds_reduced["tilde_w_a"]
    hatJ = xr.where(den_ok, hatJ, np.nan)
    hatJ = hatJ.rename("J_hat")
    print("[predict_j] done", flush=True)
    return hatJ
