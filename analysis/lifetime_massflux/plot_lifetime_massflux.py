"""
Plot diagnostics for lifetime mass-flux analysis (simple, no CLI required).

Run:
  python plot_lifetime_massflux.py

Assumptions:
- Reads cloud_results from '../../cloud_results.nc' (relative to this script)
- Reads c(z) from 'cz_estimate.nc' (same folder as this script). If missing,
  computes an unsmoothed c_raw(z) via a simple median-of-ratio across clouds.

Dependencies: numpy, xarray, matplotlib. No other deps.
Saves PNGs next to this script: fig_profiles.png, fig_cz.png, fig_ratio_collapse.png, fig_coverage_mape.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from density import compute_rho0_from_raw


def load_reduced_from_results(ds_path: Path, max_clouds: int = 32) -> tuple[list[xr.Dataset], xr.DataArray]:
    ds_full = xr.open_dataset(ds_path)
    # infer rho0 from env density per level if available
    if 'env_rho_mean_per_level' in ds_full:
        da = ds_full['env_rho_mean_per_level']
        if da.ndim == 2 and 'time' in da.dims:
            rho0 = xr.DataArray(np.nanmedian(da.values, axis=0), dims=('z',), coords=dict(z=da['level'] if 'level' in da.dims else da['z']))
        else:
            rho0 = xr.DataArray(np.asarray(da.values, dtype=float), dims=('z',), coords=dict(z=da['level'] if 'level' in da.dims else da['z']))
    else:
        # fallback: compute rho0 from raw LES files (simple + fast settings)
        base_path = '/Users/jure/PhD/coding/RICO_1hr/'
        print(f"[plot] env_rho_mean_per_level missing; computing rho0 from raw at {base_path}")
        rho0 = compute_rho0_from_raw(base_path, time_max=2, sample_frac=0.01)
        if rho0 is None:
            print("[plot] could not compute rho0 from raw. i stop.")
            return [], None
    # align rho0 to dataset z grid
    z = ds_full['height'] if 'height' in ds_full else (ds_full['z'] if 'z' in ds_full else None)
    if z is None:
        print("[plot] cloud_results.nc missing vertical coordinate. i stop.")
        return [], None
    rho0 = xr.DataArray(np.asarray(rho0.sel(z=z.values if 'z' in rho0.coords else rho0.coords['z']).values, dtype=float), dims=('z',), coords=dict(z=z.values))
    # reduce a manageable number of clouds for quick plots
    try:
        from features import reduce_all_tracks
    except Exception as e:
        print(f"[plot] cannot import reduce_all_tracks: {e}")
        return [], rho0
    dt_eff = float(ds_full.attrs.get('timestep_duration_seconds', 60.0))
    reduced = reduce_all_tracks(ds_full, dt=dt_eff, rho0=rho0, only_valid=True, min_timesteps=1, positive_only=True)
    n = min(max_clouds, len(reduced))
    print(f"[plot] reduced {n} clouds on the fly from {ds_path}")
    return reduced[:n], rho0


def ensure_rho0(ds0: xr.Dataset) -> xr.DataArray:
    z = ds0['z']
    if 'rho0' in ds0 and ds0['rho0'].dims == ('z',):
        rho0 = ds0['rho0']
    elif 'env_rho_mean_per_level' in ds0:
        da = ds0['env_rho_mean_per_level']
        if da.ndim == 2 and 'time' in da.dims:
            vals = np.nanmedian(da.values, axis=0)
        else:
            vals = np.asarray(da.values, dtype=float)
        rho0 = xr.DataArray(vals, dims=('z',), coords=dict(z=da['z'] if 'z' in da.coords else z))
    else:
        raise ValueError("Cannot infer rho0: need 'rho0[z]' or 'env_rho_mean_per_level' in reduced files")
    # align to ds0 z grid
    try:
        rho0 = rho0.sel(z=z)
    except Exception:
        # last resort: rebuild with ds0 z
        if rho0.sizes.get('z', None) == z.sizes['z']:
            rho0 = xr.DataArray(np.asarray(rho0.values, dtype=float), dims=('z',), coords=dict(z=z.values))
        else:
            raise
    return rho0


def load_cz(path: Path, reduced: list[xr.Dataset], rho0: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray | None, xr.DataArray | None, str]:
    p = Path(path)
    if not p.exists():
        # simple fallback: compute c_raw via median-of-ratio from reduced clouds
        print(f"[plot] cz_file not found at {path}. Computing simple c_raw from reduced clouds.")
        # choose target based on availability
        hasJ = all(('J' in ds) for ds in reduced)
        hasJR = all(('J_rho' in ds) for ds in reduced)
        target = 'J' if hasJ else ('J_rho' if hasJR else None)
        if target is None:
            # build J via rho0*S_aw
            target = 'J'
            for i, ds in enumerate(reduced):
                reduced[i] = ds.assign(J=(('z',), (rho0 * ds['S_aw']).values))
        z = reduced[0]['z']
        ratios = []
        for ds in reduced:
            num = ds[target]
            den = rho0 * ds['tilde_a'] * ds['tilde_w_a']
            ratios.append(xr.where(np.isfinite(den) & (den != 0), num / den, np.nan))
        R = xr.concat(ratios, dim='cloud')
        c_raw = R.median(dim='cloud', skipna=True)
        c = c_raw.rename('c')
        return c, c_raw, None, target
    target = 'J'
    c_raw = None
    valid_counts = None
    if p.suffix.lower() == '.npz':
        data = np.load(p, allow_pickle=True)
        if 'c' not in data:
            print("[plot] NPZ missing 'c'; computing simple c_raw from reduced clouds")
            # fallback
            hasJ = all(('J' in ds) for ds in reduced)
            hasJR = all(('J_rho' in ds) for ds in reduced)
            target = 'J' if hasJ else ('J_rho' if hasJR else 'J')
            if target == 'J' and (not hasJ):
                for i, ds in enumerate(reduced):
                    reduced[i] = ds.assign(J=(('z',), (rho0 * ds['S_aw']).values))
            ratios = [xr.where(np.isfinite(rho0*ds['tilde_a']*ds['tilde_w_a']) & ((rho0*ds['tilde_a']*ds['tilde_w_a'])!=0), (ds[target] if target in ds else rho0*ds['S_aw'])/(rho0*ds['tilde_a']*ds['tilde_w_a']), np.nan) for ds in reduced]
            R = xr.concat(ratios, dim='cloud')
            c_raw = R.median(dim='cloud', skipna=True)
            c = c_raw.rename('c')
            return c, c_raw, None, target
        z = data['z'] if 'z' in data else np.arange(data['c'].shape[0])
        c = xr.DataArray(np.asarray(data['c'], dtype=float), dims=('z',), coords=dict(z=z), name='c')
        if 'c_raw' in data:
            c_raw = xr.DataArray(np.asarray(data['c_raw'], dtype=float), dims=('z',), coords=dict(z=z), name='c_raw')
        if 'valid_counts' in data:
            valid_counts = xr.DataArray(np.asarray(data['valid_counts'], dtype=float), dims=('z',), coords=dict(z=z), name='valid_counts')
        target = str(data['target']) if 'target' in data else 'J'
        return c, c_raw, valid_counts, target
    else:
        ds = xr.open_dataset(p)
        if 'c' in ds:
            c = ds['c']
            if 'c_raw' in ds:
                c_raw = ds['c_raw']
            if 'valid_counts' in ds:
                valid_counts = ds['valid_counts']
            if 'target' in ds.attrs:
                target = str(ds.attrs['target'])
            return c, c_raw, valid_counts, target
        # Support a file that is just a DataArray serialized as NetCDF
        if set(ds.data_vars) == set():
            print("[plot] cz_file missing 'c'; will compute simple c_raw from reduced clouds")
            # fallback same as above
            hasJ = all(('J' in dsx) for dsx in reduced)
            hasJR = all(('J_rho' in dsx) for dsx in reduced)
            target = 'J' if hasJ else ('J_rho' if hasJR else 'J')
            if target == 'J' and (not hasJ):
                for i, d in enumerate(reduced):
                    reduced[i] = d.assign(J=(('z',), (rho0 * d['S_aw']).values))
            ratios = [xr.where(np.isfinite(rho0*d['tilde_a']*d['tilde_w_a']) & ((rho0*d['tilde_a']*d['tilde_w_a'])!=0), (d[target] if target in d else rho0*d['S_aw'])/(rho0*d['tilde_a']*d['tilde_w_a']), np.nan) for d in reduced]
            R = xr.concat(ratios, dim='cloud')
            c_raw = R.median(dim='cloud', skipna=True)
            c = c_raw.rename('c')
            return c, c_raw, None, target
        print("[plot] cz_file missing 'c' variable; computing fallback")
        hasJ = all(('J' in dsx) for dsx in reduced)
        hasJR = all(('J_rho' in dsx) for dsx in reduced)
        target = 'J' if hasJ else ('J_rho' if hasJR else 'J')
        if target == 'J' and (not hasJ):
            for i, d in enumerate(reduced):
                reduced[i] = d.assign(J=(('z',), (rho0 * d['S_aw']).values))
        ratios = [xr.where(np.isfinite(rho0*d['tilde_a']*d['tilde_w_a']) & ((rho0*d['tilde_a']*d['tilde_w_a'])!=0), (d[target] if target in d else rho0*d['S_aw'])/(rho0*d['tilde_a']*d['tilde_w_a']), np.nan) for d in reduced]
        R = xr.concat(ratios, dim='cloud')
        c_raw = R.median(dim='cloud', skipna=True)
        c = c_raw.rename('c')
        return c, c_raw, None, target


def compute_truth_hat(ds: xr.Dataset, rho0: xr.DataArray, c: xr.DataArray, target: str):
    z = ds['z']
    present = (ds['S_a'] > 0)
    # J_true selection
    J_true = None
    if target == 'J' and 'J' in ds:
        J_true = ds['J']
    elif target == 'J_rho' and 'J_rho' in ds:
        J_true = ds['J_rho']
    elif 'S_aw' in ds:
        J_true = rho0 * ds['S_aw']
    else:
        print("[plot] cannot determine J_true. i stop this cloud.")
        return None, None, None, None, None
    # Prediction
    J_hat = rho0 * c * ds['tilde_a'] * ds['tilde_w_a']
    # Shape proxy
    denom = (rho0 * ds['tilde_a'] * ds['tilde_w_a'])
    ratio_r = xr.where(np.isfinite(denom) & (denom != 0), J_true / denom, np.nan)
    # MAPE masked to present; avoid trivial zeros when J_true==0
    abs_err = np.abs(J_true - J_hat)
    denom_mape = xr.where(np.abs(J_true) > 0, np.abs(J_true), np.nan)
    mape = (abs_err / denom_mape) * 100.0
    mape = mape.where(present)
    return J_true, J_hat, ratio_r, present, mape


def _panel_grid(n: int):
    # 1 row x n columns
    return 1, n


def fig_profiles(out_dir: Path, dsets: list[xr.Dataset], rho0: xr.DataArray, c: xr.DataArray, target: str, n_examples: int):
    n = min(n_examples, len(dsets))
    if n <= 0:
        return
    nrows, ncols = _panel_grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*n, 4), squeeze=False)
    for i in range(n):
        ax = axes[0, i]
        ds = dsets[i]
        J_true, J_hat, ratio_r, present, mape = compute_truth_hat(ds, rho0, c, target)
        z = ds['z'].values
        mask = np.isfinite(present.values) & (present.values > 0)
        ax.plot(J_true.values[mask], z[mask], label='J true', lw=2)
        ax.plot(J_hat.values[mask], z[mask], label='J hat', lw=2, ls='--')
        # proxy scaled by median(r)
        denom = (rho0 * ds['tilde_a'] * ds['tilde_w_a']).values
        proxy = denom * np.nanmedian(ratio_r.values)
        ax.plot(proxy[mask], z[mask], color='0.7', lw=1, label='shape proxy (scaled)')
        # annotations
        Tc_min = float(ds.attrs.get('T_c', np.nan)) / 60.0
        cloud_mape = float(np.nanmedian(mape.values))
        ax.set_title(f"cloud {i}  T_c={Tc_min:.1f} min\nmedian MAPE={cloud_mape:.1f}%")
        ax.set_xlabel('J [kg]')
        if i == 0:
            ax.set_ylabel('z')
        ax.invert_yaxis() if (z[0] > z[-1]) else None
        ax.grid(True, alpha=0.2)
        if i == n-1:
            ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_profiles.png', dpi=150)
    plt.close(fig)


def fig_cz(out_dir: Path, c: xr.DataArray, c_raw: xr.DataArray | None, valid_counts: xr.DataArray | None, dsets: list[xr.Dataset]):
    z = c['z'].values
    fig, ax1 = plt.subplots(figsize=(5, 5))
    if c_raw is not None:
        ax1.plot(c_raw.values, z, color='C0', alpha=0.25, lw=1, label='c_raw')
    ax1.plot(c.values, z, color='C0', lw=2.5, label='c')
    ax1.set_xlabel('c [s]')
    ax1.set_ylabel('z')
    ax1.grid(True, alpha=0.2)
    med_c = float(np.nanmedian(c.values))
    med_c_min = med_c / 60.0
    # training median T_c
    tc_list = [float(ds.attrs.get('T_c', np.nan)) for ds in dsets]
    med_tc = float(np.nanmedian(np.array(tc_list))) if len(tc_list) > 0 else np.nan
    med_tc_min = med_tc / 60.0 if np.isfinite(med_tc) else np.nan
    txt = f"median c = {med_c:.0f} s ({med_c_min:.1f} min)\ntrain median T_c = {med_tc:.0f} s ({med_tc_min:.1f} min)"
    ax1.text(0.02, 0.02, txt, transform=ax1.transAxes, fontsize=9, va='bottom', ha='left')
    if valid_counts is not None:
        ax2 = ax1.twiny()
        ax2.plot(valid_counts.values, z, color='C2', lw=2, alpha=0.6, label='valid_counts')
        ax2.set_xlabel('valid_counts')
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_cz.png', dpi=150)
    plt.close(fig)


def fig_ratio_collapse(out_dir: Path, dsets: list[xr.Dataset], rho0: xr.DataArray, c: xr.DataArray, target: str, n_examples: int):
    n = min(n_examples, len(dsets))
    if n <= 0:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    zref = dsets[0]['z'].values
    c_med = float(np.nanmedian(c.values))
    for i in range(n):
        ds = dsets[i]
        _, _, ratio_r, present, _ = compute_truth_hat(ds, rho0, c, target)
        z = ds['z'].values
        mask = (present.values > 0)
        ax.plot(ratio_r.values[mask], z[mask], lw=1.5, label=f'cloud {i}')
    ax.axvline(c_med, color='k', ls='--', lw=1, label='median c')
    ax.set_xlabel('r [s]')
    ax.set_ylabel('z')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_ratio_collapse.png', dpi=150)
    plt.close(fig)


def fig_coverage_mape(out_dir: Path, dsets: list[xr.Dataset], rho0: xr.DataArray, c: xr.DataArray, target: str):
    # stack coverage and mape per cloud
    J_list = []
    Jhat_list = []
    Sa_list = []
    for i, ds in enumerate(dsets):
        J_true, J_hat, ratio_r, present, mape = compute_truth_hat(ds, rho0, c, target)
        J_list.append(J_true.expand_dims(cloud=[i]))
        Jhat_list.append(J_hat.expand_dims(cloud=[i]))
        Sa_list.append(ds['S_a'].expand_dims(cloud=[i]))
    J_stack = xr.concat(J_list, dim='cloud')
    Jhat_stack = xr.concat(Jhat_list, dim='cloud')
    Sa_stack = xr.concat(Sa_list, dim='cloud')
    present = Sa_stack > 0
    mape = (np.abs(J_stack - Jhat_stack) / xr.where(np.abs(J_stack) > 0, np.abs(J_stack), np.nan)) * 100.0
    mape = mape.where(present)
    mape_z = mape.median(dim='cloud', skipna=True)
    cov = present.mean(dim='cloud')
    overall = float(mape_z.median(skipna=True).values)
    # plot
    fig, ax1 = plt.subplots(figsize=(5, 5))
    z = J_stack['z'].values
    ax1.plot(cov.values, z, color='C3', lw=2, label='coverage')
    ax1.set_xlabel('coverage [0..1]')
    ax1.set_ylabel('z')
    ax2 = ax1.twiny()
    ax2.plot(mape_z.values, z, color='C1', lw=2, label='median MAPE per z')
    ax2.set_xlabel('median MAPE per z [%]')
    ax1.grid(True, alpha=0.2)
    ax1.text(0.02, 0.02, f"overall median (per-z median) MAPE = {overall:.1f}%", transform=ax1.transAxes,
             fontsize=9, va='bottom', ha='left')
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_coverage_mape.png', dpi=150)
    plt.close(fig)


def main():
    # fixed default paths relative to this script
    here = Path(__file__).resolve().parent
    results_path = (here / '../../cloud_results.nc').resolve()
    cz_path = (here / 'cz_estimate.nc').resolve()

    dsets, rho0 = load_reduced_from_results(results_path, max_clouds=32)

    c, c_raw, valid_counts, target = load_cz(cz_path, dsets, rho0)
    # align c to z grid of reduced files if needed
    try:
        c = c.sel(z=dsets[0]['z'])
        if c_raw is not None:
            c_raw = c_raw.sel(z=dsets[0]['z'])
        if valid_counts is not None:
            valid_counts = valid_counts.sel(z=dsets[0]['z'])
    except Exception:
        pass

    out_dir = here

    # Figures
    fig_profiles(out_dir, dsets, rho0, c, target, n_examples=4)
    fig_cz(out_dir, c, c_raw, valid_counts, dsets)
    fig_ratio_collapse(out_dir, dsets, rho0, c, target, n_examples=4)
    fig_coverage_mape(out_dir, dsets, rho0, c, target)

    # Stdout summary
    tc_list = [float(ds.attrs.get('T_c', np.nan)) for ds in dsets]
    med_tc = float(np.nanmedian(np.array(tc_list))) if len(tc_list) > 0 else np.nan
    med_c = float(np.nanmedian(c.values))
    print("Train median T_c [s]:", med_tc)
    print("Median c(z) [s]:", med_c, "; [min]:", med_c/60.0)


if __name__ == '__main__':
    main()
