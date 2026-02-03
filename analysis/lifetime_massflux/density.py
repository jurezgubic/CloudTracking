"""
Density tools used by the analysis:

- compute_rho0_from_raw: build a reference density profile rho0(z) from raw LES files.
- gamma_ratio_track/all_tracks: optional diagnostic of density influence on mass flux.

gamma(z) = J_rho(z) / (rho0(z) * S_aw(z))
where J_rho sums instantaneous mass flux M_plus(z,t) * dt and S_aw sums A(z,t) * w_plus(z,t) * dt.

# physics read: if gamma ~ 1 then using a fixed ref density rho0 is fine.
"""

from __future__ import annotations
import xarray as xr
import numpy as np

# ----------------------------
# Field-based reference density rho0(z)
# ----------------------------

def compute_rho0_from_raw(
    base_path: str,
    file_map: dict | None = None,
    time_indices: np.ndarray | None = None,
    reduce: str = 'median',
    sample_frac: float = 1.0,
    sample_seed: int | None = None,
    time_max: int | None = None,
    sample_mode: str = 'stride',  # 'stride' or 'random'
) -> xr.DataArray:
    """
    Compute a domain-mean reference density profile rho0(z) from raw LES fields
    using the thermodynamic formula described in this project.

    Inputs (files in base_path):
    - l: liquid water mixing ratio (g/kg)
    - q: total water mixing ratio (g/kg)
    - p: pressure (Pa)
    - t: liquid-water potential temperature theta_l (K)

    Returns xr.DataArray rho0[z] (kg/m^3). If cannot compute, returns None and prints a note.
    I can random-subsample the horizontal domain via sample_frac (e.g. 0.01 for one percent of points).
    I can also limit number of time steps via time_max.
    """
    import xarray as xr
    import numpy as np

    if file_map is None:
        file_map = {
            'l': 'rico.l.nc',
            'q': 'rico.q.nc',
            'p': 'rico.p.nc',
            't': 'rico.t.nc',
        }

    try:
        ds_l = xr.open_dataset(f"{base_path.rstrip('/')}/{file_map['l']}")
        ds_q = xr.open_dataset(f"{base_path.rstrip('/')}/{file_map['q']}")
        ds_p = xr.open_dataset(f"{base_path.rstrip('/')}/{file_map['p']}")
        ds_t = xr.open_dataset(f"{base_path.rstrip('/')}/{file_map['t']}")
    except Exception as e:
        print(f"[compute_rho0_from_raw] could not open raw files at {base_path}: {e}")
        return None

    def _var(ds, prefer: str, fallback: str | None = None):
        if prefer in ds:
            return ds[prefer]
        if fallback and fallback in ds:
            return ds[fallback]
        candidates = [k for k in ds.data_vars]
        if len(candidates) == 1:
            return ds[candidates[0]]
        print(f"[compute_rho0_from_raw] missing var '{prefer}' in dataset; vars={list(ds.data_vars.keys())}")
        return None

    l_gpkg = _var(ds_l, 'l')            # g/kg
    q_gpkg = _var(ds_q, 'q')            # g/kg
    p = _var(ds_p, 'p')                 # Pa
    theta_l = _var(ds_t, 't', 'theta_l')# K

    # Normalize dim names so later reduction uses 'time' and 'z'
    def _normalize_dims(da: xr.DataArray) -> xr.DataArray:
        dims = list(da.dims)
        # time dim
        if 'time' not in dims and len(dims) >= 1:
            da = da.rename({dims[0]: 'time'})
        # z dim (often 'zt' or similar)
        dims = list(da.dims)
        zcand = None
        for name in ('z','zt','level','height'):
            if name in dims:
                zcand = name; break
        if zcand is None and len(dims) >= 2:
            zcand = dims[1]
        if zcand != 'z':
            da = da.rename({zcand: 'z'})
        return da

    l_gpkg = _normalize_dims(l_gpkg)
    q_gpkg = _normalize_dims(q_gpkg)
    p = _normalize_dims(p)
    theta_l = _normalize_dims(theta_l)

    # Optionally limit time indices
    if time_indices is not None:
        l_gpkg = l_gpkg.isel(time=time_indices)
        q_gpkg = q_gpkg.isel(time=time_indices)
        p = p.isel(time=time_indices)
        theta_l = theta_l.isel(time=time_indices)
    elif time_max is not None and 'time' in l_gpkg.dims:
        l_gpkg = l_gpkg.isel(time=slice(0, int(time_max)))
        q_gpkg = q_gpkg.isel(time=slice(0, int(time_max)))
        p = p.isel(time=slice(0, int(time_max)))
        theta_l = theta_l.isel(time=slice(0, int(time_max)))

    # Optionally subsample horizontal dims to speed up
    if sample_frac < 1.0:
        # Build one set of indexers and apply to all fields to keep alignment
        ref = p
        spatial = [d for d in ref.dims if d not in ('time','z')]
        if spatial:
            per_dim_frac = sample_frac ** (1.0 / len(spatial))
            indexers = {}
            if sample_mode == 'random':
                rng = np.random.default_rng(sample_seed)
                for d in spatial:
                    n = ref.sizes[d]
                    k = max(1, int(round(per_dim_frac * n)))
                    idx = np.sort(rng.choice(n, size=k, replace=False))
                    indexers[d] = idx
            else:
                # stride decimation: contiguous strided reads (much faster than random gather)
                for d in spatial:
                    n = ref.sizes[d]
                    step = max(1, int(round(1.0 / per_dim_frac)))
                    indexers[d] = slice(0, n, step)
            try:
                l_gpkg = l_gpkg.isel(**{d: indexers[d] for d in l_gpkg.dims if d in indexers})
                q_gpkg = q_gpkg.isel(**{d: indexers[d] for d in q_gpkg.dims if d in indexers})
                p = p.isel(**{d: indexers[d] for d in p.dims if d in indexers})
                theta_l = theta_l.isel(**{d: indexers[d] for d in theta_l.dims if d in indexers})
            except Exception as e:
                print(f"[compute_rho0_from_raw] subsample apply failed: {e}")

    # Note: RICO data water species (l, q) are already in kg/kg despite metadata saying g/kg
    q_l = l_gpkg.astype('float64')
    q_t = q_gpkg.astype('float64')
    q_v = q_t - q_l

    # Constants
    R_d = 287.04
    R_v = 461.5
    c_pd = 1005.0
    p_0 = 100000.0
    epsilon = 0.622
    L_v = 2.5e6
    T_0 = 273.15
    e_s0 = 611.0

    # UCLA-LES saturation adjustment approach:
    # For unsaturated air (environment mean), T ≈ theta_l * Pi
    # For saturated air, we'd need Newton-Raphson iteration.
    # Since this is a domain-mean density calculation (mostly environment),
    # we use the simplified unsaturated formula which is appropriate for domain means.
    kappa = R_d / c_pd
    Pi = (p / p_0) ** kappa
    T = theta_l * Pi
    
    # Virtual temperature approach for density (includes vapor buoyancy and liquid loading)
    r_total = q_v + q_l
    T_v = T * (1.0 + q_v / epsilon) / (1.0 + r_total)
    rho = p / (R_d * T_v)

    axes = [d for d in rho.dims if d not in ('time','z')]
    rho_zy = rho
    if len(axes) > 0:
        rho_zy = rho.mean(dim=tuple(axes), skipna=True)
    if 'time' in rho_zy.dims:
        if reduce == 'median':
            rho0 = rho_zy.median(dim='time', skipna=True)
        elif reduce == 'mean':
            rho0 = rho_zy.mean(dim='time', skipna=True)
        else:
            print("[compute_rho0_from_raw] reduce must be 'median' or 'mean'")
            return None
    else:
        rho0 = rho_zy

    rho0 = rho0.rename('rho0').transpose('z')
    if not np.isfinite(rho0).any():
        print("[compute_rho0_from_raw] computed rho0 has no finite values. check raw files and units.")
        return None
    return rho0


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
