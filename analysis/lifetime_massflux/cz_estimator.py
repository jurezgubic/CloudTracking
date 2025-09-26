"""
Estimate c(z) from per-cloud reductions using a median of ratios.

# physics idea:
# I want J(z) ~ rho0(z) * c(z) * tilde_a(z) * tilde_w_plus(z)
# So per cloud I form r(z) = J / (rho0 * tilde_a * tilde_w_plus). r has units of seconds.
# Then take median over clouds. Smooth in z to reduce noise.

# note: if J_rho is used (instantaneous density), c(z) also absorbs density bias vs rho0.
"""

from __future__ import annotations
import numpy as np
import xarray as xr
from scipy.interpolate import UnivariateSpline

def fit_cz(
    dsets: list[xr.Dataset],
    rho0: xr.DataArray,
    smooth: bool = True,
    s: float | None = None,
    min_samples_per_z: int = 5,
    use: str = "auto",  # 'auto' | 'J' | 'J_rho'
) -> xr.Dataset:
    """
    dsets: list of per-cloud reduced datasets from features.reduce_per_cloud (same z-grid).
    rho0: DataArray rho0[z]
    Returns c[z] with optional spline smoothing.
    """
    if len(dsets) == 0:
        print("[fit_cz] no clouds provided. i stop.")
        return xr.Dataset(data_vars=dict(c=xr.DataArray(np.array([]), dims=('z',), name='c'),
                                         c_raw=xr.DataArray(np.array([]), dims=('z',), name='c_raw'),
                                         valid_counts=xr.DataArray(np.array([]), dims=('z',), name='valid_counts')),
                          coords=dict(z=np.array([])), attrs=dict(target='J'))

    print(f"[fit_cz] clouds={len(dsets)}, smoothing={'on' if smooth else 'off'}, use={use}", flush=True)
    # Align rho0 to first dataset z
    z = dsets[0]["z"].values
    r0 = rho0
    if 'z' not in r0.dims:
        r0 = r0.rename({r0.dims[0]: 'z'})
    r0 = xr.DataArray(np.asarray(r0.sel(z=z).values, dtype=float), coords=dict(z=z), dims=('z',))
    # Stack ratios per z
    ratios = []
    # decide single target for all clouds
    use = use.lower()
    hasJ = all(('J' in ds) for ds in dsets)
    hasJR = all(('J_rho' in ds) for ds in dsets)
    if use == 'auto':
        target = 'J' if hasJ else ('J_rho' if hasJR else None)
    elif use in ('j','j_rho'):
        target = 'J' if use == 'j' else 'J_rho'
        if not all((target in ds) for ds in dsets):
            print("[fit_cz] chosen target missing in some clouds. i stop.")
            return xr.Dataset(data_vars=dict(c=xr.DataArray(np.array([]), dims=('z',), name='c'),
                                             c_raw=xr.DataArray(np.array([]), dims=('z',), name='c_raw'),
                                             valid_counts=xr.DataArray(np.array([]), dims=('z',), name='valid_counts')),
                              coords=dict(z=np.array([])), attrs=dict(target='J'))
    else:
        print("[fit_cz] bad use flag. i stop.")
        return xr.Dataset(data_vars=dict(c=xr.DataArray(np.array([]), dims=('z',), name='c'),
                                         c_raw=xr.DataArray(np.array([]), dims=('z',), name='c_raw'),
                                         valid_counts=xr.DataArray(np.array([]), dims=('z',), name='valid_counts')),
                          coords=dict(z=np.array([])), attrs=dict(target='J'))
    if target is None:
        print("[fit_cz] no common target found. i stop.")
        return xr.Dataset(data_vars=dict(c=xr.DataArray(np.array([]), dims=('z',), name='c'),
                                         c_raw=xr.DataArray(np.array([]), dims=('z',), name='c_raw'),
                                         valid_counts=xr.DataArray(np.array([]), dims=('z',), name='valid_counts')),
                          coords=dict(z=np.array([])), attrs=dict(target='J'))
    for ds in dsets:
        num = ds[target]
        den = r0 * ds["tilde_a"] * ds["tilde_w_a"]        # [kg/s]
        r = xr.where(np.isfinite(den) & (den != 0), num / den, np.nan)  # [s]
        ratios.append(r)
    print(f"[fit_cz] ratio uses: {target}", flush=True)

    R = xr.concat(ratios, dim="cloud")               # [cloud,z]
    # Median across clouds, require minimum samples at each z
    valid_counts = np.isfinite(R).sum(dim="cloud")
    c_raw = R.median(dim="cloud", skipna=True)
    c_raw = xr.where(valid_counts >= min_samples_per_z, c_raw, np.nan)
    c_raw = c_raw.rename("c_raw").assign_coords(z=z)
    print("[fit_cz] computed raw median c(z)", flush=True)

    if smooth:
        # Smooth only finite segments; preserve NaNs
        c_vals = c_raw.values
        mask = np.isfinite(c_vals)
        if mask.sum() >= 4:
            x = np.arange(len(z))[mask].astype(float)
            y = c_vals[mask].astype(float)
            if s is None:
                s = 0.5 * float(mask.sum())  # mild smoothing by default, scale with valid points
            spl = UnivariateSpline(x, y, s=s)
            y_s = c_vals.copy()
            y_s[mask] = spl(x)
            c_smooth = xr.DataArray(y_s, coords=dict(z=z), dims=("z",), name="c")
        else:
            c_smooth = c_raw.rename("c")
        print("[fit_cz] smoothing with SciPy spline", flush=True)
    else:
        c_smooth = c_raw.rename("c")

    return xr.Dataset(
        data_vars=dict(c=c_smooth, c_raw=c_raw, valid_counts=valid_counts.rename('valid_counts')),
        coords=dict(z=z),
        attrs=dict(target=target)
    )
