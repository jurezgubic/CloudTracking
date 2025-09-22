"""
Lifetime reductions from the tracking output (cloud_results.nc).

What we compute per cloud (by height z):
- S_a(z) = ∑_t A(z,t) dt [m^2 s]  (time-integrated in‑cloud area)
- S_aw(z) = ∑_t A(z,t) max(⟨w⟩(z,t), 0) dt [m^3]  (time‑integrated upward volume flux)
- T_c = N_valid · dt [s]  (cloud lifetime in seconds)
- tilde_a(z) = S_a/T_c [m^2]  (lifetime‑mean area)
- tilde_w_a(z) = S_aw/S_a [m s^-1]  (area‑weighted lifetime‑mean w+, i.e. mean of upward part)
- J_rho(z) = ∑_t max(M(z,t), 0) dt [kg]  (time‑integrated mass flux using instantaneous density)
- If a reference density profile ρ0(z) is provided, also J(z) = ρ0(z) · S_aw(z) [kg].

Inputs from cloud_results.nc (per track, time, level):
- area_per_level[track,time,level] = A(z,t) [m^2]
- w_per_level[track,time,level] = ⟨w⟩(z,t) over in‑cloud points [m s^-1]
- mass_flux_per_level[track,time,level] = M(z,t) = ∑ cells ρ w ΔxΔy [kg s^-1]
- height[level] = z [m]
- valid_track[track] ∈ {0,1}. We normally use only valid (complete lifetime) tracks.

Note on signs: we use only upward transport (w+ or M+). This matches the idea that
convective mass flux is an updraft quantity. Using M+ or ⟨w⟩+A gives nearly the same
result; the former additionally includes density variations.
"""

from __future__ import annotations
import numpy as np
import xarray as xr

def _z_coord(ds: xr.Dataset) -> xr.DataArray:
    """Return height coord as DataArray named 'z'."""
    if 'height' in ds:
        return xr.DataArray(ds['height'].values, dims=('z',), name='z')
    if 'level' in ds.dims:
        return xr.DataArray(np.arange(ds.sizes['level'], dtype=float), dims=('z',), name='z')
    raise ValueError("Expected 'height' variable or 'level' dimension in dataset.")

def _track_is_valid(ds: xr.Dataset, i: int) -> bool:
    v = ds.get('valid_track')
    if v is None:
        return True
    try:
        return bool(v[i].item() == 1)
    except Exception:
        return True

def reduce_track(ds: xr.Dataset, track_index: int, dt: float,
                 rho0: xr.DataArray | None = None,
                 positive_only: bool = True,
                 require_valid: bool = True,
                 eps: float = 0.0) -> xr.Dataset:
    """
    Reduce one tracked cloud (one row in 'track') to lifetime-mean profiles.

    Physics:
    - Area A(z,t) times w+(z,t) integrated over time gives an upward volume transport S_aw(z).
    - Multiplying S_aw by a reference density ρ0(z) gives a time‑integrated mass J(z) [kg].
    - Using instantaneous density inside the cloud and summing M+(z,t) dt gives J_rho(z) [kg].
    """
    if require_valid and not _track_is_valid(ds, track_index):
        raise ValueError("Track is flagged invalid (partial lifetime). Set require_valid=False to force.")

    z = _z_coord(ds)
    # say what we doing (simple progress)
    print(f"[reduce_track] reducing track {track_index} ...", flush=True)
    # Extract per‑level, per‑time arrays for this track
    A = ds['area_per_level'].isel(track=track_index)
    W = ds['w_per_level'].isel(track=track_index)
    M = ds['mass_flux_per_level'].isel(track=track_index)
    # Ensure vertical dim is named 'z' for clarity/consistency
    if 'level' in A.dims:
        A = A.rename({'level':'z'})
        W = W.rename({'level':'z'})
        M = M.rename({'level':'z'})

    # Time indices when the cloud exists (any level has finite area)
    live_t = np.isfinite(A).any(dim='z') & (xr.where(np.isfinite(A), A, 0.0).sum(dim='z') > 0)
    if live_t.any():
        A = A.where(live_t, other=0.0)
        W = W.where(live_t)
        M = M.where(live_t, other=0.0)
        nt = int(live_t.sum().item())
    else:
        nt = 0
    print(f"[reduce_track] active time steps = {nt}", flush=True)

    # Use upward part only if requested
    # physics: convective mass flux is about updrafts, so keep w+ and M+
    if positive_only:
        Wp = xr.where(W > 0.0, W, 0.0)
        Mp = xr.where(M > 0.0, M, 0.0)
    else:
        Wp = xr.where(np.isfinite(W), W, 0.0)
        Mp = xr.where(np.isfinite(M), M, 0.0)

    # Time‑integrated area and volume flux
    # physics: S_a = ∑ A dt  (area*time). S_aw = ∑ A*w+ dt (volume)
    S_a = (xr.where(np.isfinite(A), A, 0.0) * dt).sum(dim='time')          # [z] m^2 s
    S_aw = (xr.where(np.isfinite(A*Wp), A*Wp, 0.0) * dt).sum(dim='time')   # [z] m^3

    # Lifetime in seconds
    T_c = float(nt * dt)

    # Lifetime means
    # physics: divide time-integrals by lifetime
    tilde_a = xr.where(T_c > 0, S_a / T_c, np.nan)                         # [level] m^2
    tilde_w_a = xr.where(S_a > eps, S_aw / S_a, np.nan)                    # [level] m s^-1

    # Time‑integrated mass flux using instantaneous rho (from M per level)
    J_rho = (Mp * dt).sum(dim='time')                                      # [z] kg

    # If a reference density profile is given, also compute J = ρ0 S_aw
    data_vars = dict(S_a=S_a, S_aw=S_aw, tilde_a=tilde_a, tilde_w_a=tilde_w_a,
                     J_rho=J_rho)
    if rho0 is not None:
        # Align rho0 length and coordinate to current z-grid
        if 'z' not in rho0.dims:
            rho0 = rho0.rename({rho0.dims[0]: 'z'})
        if rho0.sizes.get('z', None) != z.sizes['z']:
            raise ValueError("rho0 length does not match number of levels")
        rho0_z = xr.DataArray(np.asarray(rho0.values, dtype=float), coords=dict(z=z.values), dims=('z',))
        data_vars['J'] = (rho0_z * S_aw).rename('J')                       # [level] kg

    out = xr.Dataset(
        data_vars=data_vars,
        coords=dict(z=z),
        attrs=dict(T_c=T_c, dt=float(dt), track_index=int(track_index))
    )
    # Effective radius from lifetime‑mean area (useful geometric proxy)
    out['R_eff_tilde'] = xr.where(out['tilde_a'] > 0, np.sqrt(out['tilde_a'] / np.pi), np.nan)
    print(f"[reduce_track] done track {track_index}", flush=True)
    return out

def reduce_all_tracks(ds: xr.Dataset, dt: float,
                      rho0: xr.DataArray | None = None,
                      only_valid: bool = True,
                      min_timesteps: int = 1,
                      positive_only: bool = True) -> list[xr.Dataset]:
    """Convenience: reduce all tracks in a cloud_results.nc Dataset.

    - Skips tracks with less than `min_timesteps` active entries.
    - If `only_valid`, uses only complete lifetime tracks (valid_track==1).
    """
    ntracks = ds.sizes.get('track', 0)
    out = []
    print(f"[reduce_all_tracks] start: ntracks={ntracks}, only_valid={only_valid}, min_timesteps={min_timesteps}", flush=True)
    step = max(1, ntracks // 20)  # ~5% progress steps
    for i in range(ntracks):
        if only_valid and not _track_is_valid(ds, i):
            continue
        # Quick check: any finite area at any level/time?
        A = ds['area_per_level'].isel(track=i)
        if 'level' in A.dims:
            A = A.rename({'level':'z'})
        live_t = np.isfinite(A).any(dim='z') & (xr.where(np.isfinite(A), A, 0.0).sum(dim='z') > 0)
        if int(live_t.sum().item()) < min_timesteps:
            continue
        out.append(reduce_track(ds, i, dt, rho0=rho0, positive_only=positive_only, require_valid=False))
        if (i + 1) % step == 0:
            print(f"[reduce_all_tracks] processed tracks: {i+1}/{ntracks}, reduced clouds={len(out)}", flush=True)
    print(f"[reduce_all_tracks] done: reduced_clouds={len(out)}", flush=True)
    return out
