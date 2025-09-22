"""
Density sensitivity diagnostic for tracking output:

gamma(z) = J_rho(z) / [rho0(z) * S_aw(z)]
where J_rho sums instantaneous mass flux M+(z,t) dt and S_aw sums A(z,t) * w+(z,t) dt.

# physics read: if gamma ~ 1 then using a fixed ref density rho0 is fine.
# if gamma > 1 clouds (as sampled by their updraft) are denser than rho0 on average, < 1 lighter.
"""

from __future__ import annotations
import xarray as xr
import numpy as np

def gamma_ratio_track(ds: xr.Dataset, track_index: int, rho0: xr.DataArray, dt: float,
                      positive_only: bool = True, eps: float = 0.0) -> xr.DataArray:
    """
    Compute gamma(z) for one track from cloud_results.nc fields.
    Requires: area_per_level, w_per_level, mass_flux_per_level, height.
    """
    print(f"[gamma_ratio_track] track={track_index}", flush=True)
    z = xr.DataArray(ds['height'].values, dims=('z',), name='z') if 'height' in ds else None
    if z is None:
        raise ValueError("Missing 'height' variable for z coordinate.")
    A = ds['area_per_level'].isel(track=track_index)
    W = ds['w_per_level'].isel(track=track_index)
    M = ds['mass_flux_per_level'].isel(track=track_index)
    if 'level' in A.dims:
        A = A.rename({'level':'z'})
        W = W.rename({'level':'z'})
        M = M.rename({'level':'z'})

    if positive_only:
        Wp = xr.where(W > 0.0, W, 0.0)
        Mp = xr.where(M > 0.0, M, 0.0)
    else:
        Wp = xr.where(np.isfinite(W), W, 0.0)
        Mp = xr.where(np.isfinite(M), M, 0.0)

    S_aw = (xr.where(np.isfinite(A*Wp), A*Wp, 0.0) * dt).sum(dim='time')   # [z] m^3
    J_rho = (Mp * dt).sum(dim='time')                                      # [z] kg

    # Align rho0 to z
    if 'z' not in rho0.dims:
        rho0 = rho0.rename({rho0.dims[0]: 'z'})
    rho0_z = rho0.sel(z=z.values)
    denom = rho0_z * S_aw
    gamma = xr.where(np.isfinite(denom) & (denom > eps), J_rho / denom, np.nan)
    return gamma.rename('gamma')

def gamma_ratio_all_tracks(ds: xr.Dataset, rho0: xr.DataArray, dt: float,
                           only_valid: bool = True,
                           positive_only: bool = True) -> xr.DataArray:
    """Stack gamma(z) for all (optionally valid) tracks → DataArray[track,z]."""
    ntracks = ds.sizes.get('track', 0)
    print(f"[gamma_ratio_all_tracks] ntracks={ntracks}, only_valid={only_valid}", flush=True)
    gammas = []
    tids = []
    valid = ds.get('valid_track')
    for i in range(ntracks):
        if only_valid and valid is not None and int(valid[i].item()) != 1:
            continue
        g = gamma_ratio_track(ds, i, rho0=rho0, dt=dt, positive_only=positive_only)
        gammas.append(g.expand_dims(track=[i]))
        tids.append(i)
    if not gammas:
        return xr.DataArray(np.full((0, ds.sizes.get('level', 0)), np.nan),
                            dims=('track','z'), coords=dict(z=ds['height']))
    out = xr.concat(gammas, dim='track')
    print(f"[gamma_ratio_all_tracks] done, tracks_out={out.sizes.get('track',0)}", flush=True)
    return out
