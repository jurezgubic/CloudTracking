"""
Basic per-level metrics: MAPE, RMSE, Pearson r.

# physics/intuition: 
# - MAPE: relative error per level in % (guard small denom)
# - RMSE: absolute error norm per level
# - corr: Pearson r per level across clouds (0..1 means alignment, even if biased)
"""

from __future__ import annotations
import numpy as np
import xarray as xr

def mape_per_z(J: xr.DataArray, J_hat: xr.DataArray, delta: float = 1e-9) -> xr.DataArray:
    # avoid division by tiny values (mass flux sometimes zero at high z)
    num = np.abs(J - J_hat)
    den = xr.where(np.abs(J) > delta, np.abs(J), delta)
    mape = (num / den) * 100.0
    return mape.rename("MAPE")

def rmse_per_z(J: xr.DataArray, J_hat: xr.DataArray) -> xr.DataArray:
    return np.sqrt(((J - J_hat) ** 2)).rename("RMSE")

def corr_per_z(J_stack: xr.DataArray, Jhat_stack: xr.DataArray) -> xr.DataArray:
    """
    Inputs stacked along 'cloud' with coord z: shape [cloud,z].
    Returns Pearson r per z.
    """
    if "cloud" not in J_stack.dims:
        print("[corr_per_z] expect dim 'cloud'. i stop.")
        z = J_stack['z'] if 'z' in J_stack.coords else xr.DataArray(np.array([]), dims=('z',), name='z')
        return xr.DataArray(np.array([]), coords=dict(z=z), dims=("z",), name="corr")
    def _r(j, h):
        jv = j.values; hv = h.values
        mask = np.isfinite(jv) & np.isfinite(hv)
        if mask.sum() < 3: 
            return np.nan
        jv = jv[mask]; hv = hv[mask]
        jv = (jv - jv.mean()) / (jv.std() + 1e-12)
        hv = (hv - hv.mean()) / (hv.std() + 1e-12)
        return float(np.mean(jv*hv))
    r = []
    for z in J_stack["z"].values:
        r.append(_r(J_stack.sel(z=z), Jhat_stack.sel(z=z)))
    return xr.DataArray(np.array(r), coords=dict(z=J_stack["z"]), dims=("z",), name="corr")

def coverage_per_z(S_a_stack: xr.DataArray) -> xr.DataArray:
    """Fraction of clouds occupying each level (S_a > 0) as a simple coverage measure."""
    return (S_a_stack > 0).mean(dim='cloud').rename('coverage')
