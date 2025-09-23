"""
Estimate c(z) from per‑cloud reductions using a median of ratios.

# physics idea:
# we want J(z) ≈ rho0(z) * c(z) * tilde_a(z) * tilde_w_plus(z)
# so per cloud we form r(z) = J / (rho0 * tilde_a * tilde_w_plus). r has units of seconds.
# then take median over clouds. smooth in z to reduce noise if you want.

# note: if J_rho used (instantaneous density), c(z) will also soak up density bias vs rho0.
"""

from __future__ import annotations
import numpy as np
import xarray as xr

try:
    from scipy.interpolate import UnivariateSpline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def fit_cz(
    dsets: list[xr.Dataset],
    rho0: xr.DataArray,
    smooth: bool = True,
    s: float | None = None,
    min_samples_per_z: int = 5,
    use: str = "auto",  # 'auto' | 'J' | 'J_rho'
) -> xr.DataArray:
    """
    dsets: list of per-cloud reduced datasets from features.reduce_per_cloud (same z-grid).
    rho0: DataArray rho0[z]
    Returns c[z] with optional spline smoothing.
    """
    if len(dsets) == 0:
        raise ValueError("No clouds provided")

    print(f"[fit_cz] clouds={len(dsets)}, smoothing={'on' if smooth else 'off'}, use={use}", flush=True)
    z = dsets[0]["z"].values
    # Stack ratios per z
    ratios = []
    use = use.lower()
    if use not in ("auto","j","j_rho"):
        raise ValueError("use must be one of: 'auto','J','J_rho'")
    used = None
    for ds in dsets:
        if use == "j":
            if "J" not in ds:
                raise ValueError("Requested use='J' but input dataset missing 'J'")
            num = ds["J"]
            used = 'J'
        elif use == "j_rho":
            if "J_rho" not in ds:
                raise ValueError("Requested use='J_rho' but input dataset missing 'J_rho'")
            num = ds["J_rho"]
            used = 'J_rho'
        else:
            num = ds["J"] if "J" in ds else ds["J_rho"]        # [kg]
            if used is None:
                used = 'J' if 'J' in ds else 'J_rho'
        den = rho0 * ds["tilde_a"] * ds["tilde_w_a"]        # [kg/s]
        r = xr.where(np.isfinite(den) & (den != 0), num / den, np.nan)  # [s]
        ratios.append(r)
    print(f"[fit_cz] ratio uses: {used}", flush=True)

    R = xr.concat(ratios, dim="cloud")               # [cloud,z]
    # Median across clouds, require minimum samples at each z
    valid_counts = np.isfinite(R).sum(dim="cloud")
    c_raw = R.median(dim="cloud", skipna=True)
    c_raw = xr.where(valid_counts >= min_samples_per_z, c_raw, np.nan)
    c_raw = c_raw.rename("c_raw").assign_coords(z=z)
    print("[fit_cz] computed raw median c(z)", flush=True)

    if smooth and _HAS_SCIPY:
        # Smooth only finite segments; preserve NaNs
        c_vals = c_raw.values
        mask = np.isfinite(c_vals)
        if mask.sum() >= 4:
            x = np.arange(len(z))[mask].astype(float)
            y = c_vals[mask].astype(float)
            if s is None:
                s = 0.5 * len(x)  # mild smoothing by default
            spl = UnivariateSpline(x, y, s=s)
            y_s = c_vals.copy()
            y_s[mask] = spl(x)
            c_smooth = xr.DataArray(y_s, coords=dict(z=z), dims=("z",), name="c")
        else:
            c_smooth = c_raw.rename("c")
        print("[fit_cz] smoothing with SciPy spline", flush=True)
    else:
        # Fallback: running median (window=5)
        w = 5
        vals = c_raw.values
        out = np.full_like(vals, np.nan, dtype=float)
        for i in range(len(vals)):
            lo = max(0, i - w//2); hi = min(len(vals), i + w//2 + 1)
            window = vals[lo:hi]
            ws = window[np.isfinite(window)]
            if ws.size:
                out[i] = np.median(ws)
        c_smooth = xr.DataArray(out, coords=dict(z=z), dims=("z",), name="c")
        print("[fit_cz] smoothing with running median (no SciPy)", flush=True)

    return c_smooth
